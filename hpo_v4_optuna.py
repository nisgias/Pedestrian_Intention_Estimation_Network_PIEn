#!/usr/bin/env python3
"""
HPO for PIPNet-Alpha V4 — Bayesian Optimization with Optuna
============================================================

Features:
  - TPE sampler (Bayesian, sample-efficient)
  - MedianPruner (kills bad trials early at each epoch checkpoint)
  - PRIMARY objective: post-hoc fusion VALIDATION AUC (zero test leakage)
  - Test AUC logged as user_attr for reporting only — never used for trial selection
  - All branch AUCs tracked as Optuna user_attrs
  - Single GPU, sequential trials (safe for itws2 container setup)
  - Resumes from existing study DB (just re-run the same command)
  - Best trial checkpoint saved to --save_dir/best_trial/

Usage example:
  python hpo_v4_optuna.py \
    --pie_root /Datasets/PIE_PREP_OUT \
    --save_dir /workspace/project/hpo_runs/hpo_v4_run1 \
    --n_trials 40 \
    --epochs_per_trial 15 \
    --batch_size 64 \
    --num_workers 2 \
    --amp \
    --seed 42

Resume (same command, study is loaded from SQLite DB):
  python hpo_v4_optuna.py \
    --pie_root /Datasets/PIE_PREP_OUT \
    --save_dir /workspace/project/hpo_runs/hpo_v4_run1 \
    --n_trials 40 ...

Transfer learning search (JAAD → PIE):
  python hpo_v4_optuna.py \
    --pie_root /Datasets/PIE_PREP_OUT \
    --jaad_root /Datasets/JAAD_PREP_OUT \
    --search_transfer \
    --n_trials 30 \
    --epochs_per_trial 20 \
    --save_dir /workspace/project/hpo_runs/hpo_v4_transfer
"""

import os
import sys
import csv
import copy
import json
import random
import argparse
import itertools
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score, f1_score

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
except ImportError:
    raise ImportError(
        "Optuna is required. Install with:\n"
        "  pip install optuna"
    )

# ── Project imports ─────────────────────────────────────────────────────────
sys.path.insert(0, "/workspace/project")
from data.pie import PIESeqDataset
from models.pipnet_alpha_v4_final import PIPNetAlphaV4Final


# ============================================================
# LOSS
# ============================================================

class MultiTaskLoss(nn.Module):
    def __init__(self, aux_weight=0.1, entropy_weight=0.05, pos_weight=1.0):
        super().__init__()
        self.aux_weight = aux_weight
        self.entropy_weight = entropy_weight
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    def forward(self, outputs, labels):
        device = labels.device
        self.bce.pos_weight = self.bce.pos_weight.to(device)
        labels = labels.float()

        main_loss = self.bce(outputs["logit"].squeeze(-1), labels)
        aux_loss = torch.tensor(0.0, device=device)

        if "aux_kin" in outputs:
            aux_loss = self.aux_weight * (
                self.bce(outputs["aux_kin"].squeeze(-1), labels)
                + self.bce(outputs["aux_local"].squeeze(-1), labels)
                + self.bce(outputs["aux_global"].squeeze(-1), labels)
            )

        entropy_loss = torch.tensor(0.0, device=device)
        for key in ("modality_weights", "visual_fuse_weights"):
            if key in outputs:
                w = outputs[key]
                ent = -(w * torch.log(w + 1e-8)).sum(dim=-1).mean()
                ent_norm = ent / torch.log(torch.tensor(w.shape[-1], dtype=ent.dtype, device=device))
                entropy_loss = entropy_loss - self.entropy_weight * ent_norm

        return main_loss + aux_loss + entropy_loss


# ============================================================
# UTILS
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def safe_auc(labels, probs):
    try:
        if len(np.unique(labels)) > 1:
            return float(roc_auc_score(labels, probs))
    except Exception:
        pass
    return float("nan")


def compute_pos_weight(files):
    pos, neg = 0, 0
    for p in files:
        d = np.load(p, allow_pickle=True)
        y = float(np.array(d["label"]).reshape(-1)[0])
        if y >= 0.5:
            pos += 1
        else:
            neg += 1
    return float(neg / max(pos, 1))


def make_loaders(data_root, dataset_prefix, seq_len, batch_size, num_workers):
    speed_stats_path = f"/workspace/project/data/{dataset_prefix}_speed_stats_splits.json"
    motion_stats_path = f"/workspace/project/data/{dataset_prefix}_motion_stats_splits.json"

    common = dict(
        seq_len=seq_len, strict_len=True,
        speed_norm="minmax", speed_stats_path=speed_stats_path, speed_scope="global",
        motion_norm="p99abs", motion_stats_path=motion_stats_path, motion_scope="global",
        motion_clip=1.0,
    )

    train_ds  = PIESeqDataset(data_root, split="train", mode="train",  **common)
    val_ds    = PIESeqDataset(data_root, split="val",   mode="eval",   **common)
    test_ds   = PIESeqDataset(data_root, split="test",  mode="eval",   **common)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_ds, train_loader, val_loader, test_loader


# ============================================================
# TRAIN / EVAL ONE EPOCH
# ============================================================

def train_one_epoch(model, loader, device, criterion, optimizer, use_amp, scaler):
    model.train()
    all_labels, all_probs = [], []

    for batch in loader:
        for k in ["bbox", "pose", "speed", "local_cnn", "local_motion",
                  "sem_labels", "cat_depth", "label"]:
            if k in batch and torch.is_tensor(batch[k]):
                batch[k] = batch[k].to(device, non_blocking=True)

        labels = batch["label"].float()
        optimizer.zero_grad(set_to_none=True)

        with autocast('cuda', enabled=use_amp):
            outputs = model(batch, return_aux=True)
            loss = criterion(outputs, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        probs = torch.sigmoid(outputs["logit"].squeeze(-1)).detach().cpu().numpy()
        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs)

    labels_np = np.concatenate(all_labels)
    probs_np  = np.concatenate(all_probs)
    return safe_auc(labels_np, probs_np)


@torch.no_grad()
def eval_one_epoch(model, loader, device, use_amp):
    """
    Returns dict with:
      auc_main, auc_kin, auc_local, auc_global,
      all_labels, all_logits_main, all_logits_kin, all_logits_local, all_logits_global
    """
    model.eval()
    all_labels = []
    all_probs  = {k: [] for k in ("main", "kin", "local", "global")}

    for batch in loader:
        for k in ["bbox", "pose", "speed", "local_cnn", "local_motion",
                  "sem_labels", "cat_depth", "label"]:
            if k in batch and torch.is_tensor(batch[k]):
                batch[k] = batch[k].to(device, non_blocking=True)

        labels = batch["label"].float()

        with autocast('cuda', enabled=use_amp):
            outputs = model(batch, return_aux=True)

        all_labels.append(labels.cpu().numpy())
        all_probs["main"].append(outputs["logit"].squeeze(-1).detach().cpu().numpy())

        if "aux_kin" in outputs:
            all_probs["kin"].append(outputs["aux_kin"].squeeze(-1).detach().cpu().numpy())
            all_probs["local"].append(outputs["aux_local"].squeeze(-1).detach().cpu().numpy())
            all_probs["global"].append(outputs["aux_global"].squeeze(-1).detach().cpu().numpy())

    labels_np = np.concatenate(all_labels)

    result = {"all_labels": labels_np}
    for k, v in all_probs.items():
        if v:
            p = np.concatenate(v)
            result[f"all_logits_{k}"] = p
            result[f"auc_{k}"] = safe_auc(labels_np, torch.sigmoid(torch.from_numpy(p)).numpy())

    return result


# ============================================================
# POST-HOC FUSION (grid search on val, apply once to test)
# ============================================================

def posthoc_fusion_search(
    val_logits: dict,  # {"main": raw_logit_array, "kin": ..., ...}  — sigmoid applied internally
    val_labels: np.ndarray,
    test_logits: dict,
    test_labels: np.ndarray,
    step: float = 0.05,
) -> dict:
    """
    Grid search over (w_main, w_kin, w_local, w_global) on validation set.
    Weights sum to 1. Applies best weights once to test set.
    TRUE LOGIT FUSION: fused_prob = sigmoid(w1*l1 + w2*l2 + ...)
    Weights are applied to RAW LOGITS before sigmoid — matches the reference eval script.
    NOT the same as probability fusion: sigmoid(w*l) ≠ w*sigmoid(l) due to nonlinearity.
    Returns best_val_auc, test_auc, test_f1, best_weights dict.
    """
    keys = [k for k in ("main", "kin", "local", "global") if k in val_logits]
    # TRUE LOGIT FUSION: weight the raw logits, THEN apply sigmoid once.
    # sigmoid(w1*l1 + w2*l2 + ...) — this is what the reference eval script does.
    # This is NOT the same as w1*sigmoid(l1) + w2*sigmoid(l2) + ...
    # because sigmoid is nonlinear: the two produce different probability distributions.
    val_logit_arrs  = {k: val_logits[k]  for k in keys}
    test_logit_arrs = {k: test_logits[k] for k in keys}

    candidates = [round(v * step, 4) for v in range(int(1.0 / step) + 1)]

    best_val_auc = -1.0
    best_weights = {k: 1.0 / len(keys) for k in keys}

    # Generate all combinations that sum to 1.0
    for combo in itertools.product(candidates, repeat=len(keys)):
        if abs(sum(combo) - 1.0) > 1e-6:
            continue
        weights = dict(zip(keys, combo))
        # Weighted sum of logits → single fused logit → sigmoid → probability
        fused_logit = sum(weights[k] * val_logit_arrs[k] for k in keys)
        fused_prob  = torch.sigmoid(torch.from_numpy(fused_logit)).numpy()
        auc = safe_auc(val_labels, fused_prob)
        if not np.isnan(auc) and auc > best_val_auc:
            best_val_auc = auc
            best_weights = weights

    # Apply best weights to test (same logit-fusion formula)
    fused_logit_test = sum(best_weights[k] * test_logit_arrs[k] for k in keys)
    fused_prob_test  = torch.sigmoid(torch.from_numpy(fused_logit_test)).numpy()
    test_auc = safe_auc(test_labels, fused_prob_test)
    test_f1  = float(f1_score(test_labels, (fused_prob_test >= 0.5).astype(int), zero_division=0))

    return {
        "val_auc":     best_val_auc,
        "test_auc":    test_auc,
        "test_f1":     test_f1,
        "weights":     best_weights,
    }


# ============================================================
# BUILD OPTIMIZER (from train_v4_transfer.py, adapted)
# ============================================================

def build_optimizer(model, lr, last_fc_l2, visual_l2):
    visual_keywords = [
        "local_branch", "local_attn", "norm_local", "aux_local",
        "global_branch", "global_attn", "norm_global", "aux_global",
        "visual_fuse_attn", "norm_visual",
    ]
    base_params, visual_params, final_fc_params = [], [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "fc_out" in name or "branch_logit_fusion" in name:
            final_fc_params.append(p)
        elif any(k in name for k in visual_keywords):
            visual_params.append(p)
        else:
            base_params.append(p)

    optimizer = torch.optim.AdamW([
        {"params": base_params,      "weight_decay": 0.0},
        {"params": visual_params,    "weight_decay": visual_l2},
        {"params": final_fc_params,  "weight_decay": last_fc_l2},
    ], lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-7
    )
    return optimizer, scheduler


# ============================================================
# SEARCH SPACE
# ============================================================

def suggest_hyperparams(trial, args):
    """
    Informed search space around the best PIE-only run.
    Still wide enough to find better configs, but avoids wasting trials
    on regions that already looked weak.
    """
    cfg = {}

    # Optimizer
    cfg["lr"] = trial.suggest_float(
        "lr", 5e-6, 8e-5, log=True
    )

    cfg["last_fc_l2"] = trial.suggest_float(
        "last_fc_l2", 1e-3, 1.2e-2, log=True
    )

    cfg["visual_l2"] = trial.suggest_float(
        "visual_l2", 0.001, 0.005, step=0.0005
    )

    # Loss weights
    cfg["aux_weight"] = trial.suggest_float(
        "aux_weight", 0.10, 0.35, step=0.05
    )

    cfg["entropy_weight"] = trial.suggest_float(
        "entropy_weight", 0.00, 0.08, step=0.01
    )

    # Dropout
    cfg["dropout_p"] = trial.suggest_categorical(
        "dropout_p", [0.2, 0.3, 0.4]
    )

    cfg["local_dropout_p"] = trial.suggest_categorical(
        "local_dropout_p", [0.1, 0.2, 0.3, 0.4]
    )

    cfg["global_dropout_p"] = trial.suggest_categorical(
        "global_dropout_p", [0.3, 0.4, 0.5]
    )

    return cfg


def suggest_transfer_hyperparams(trial, args):
    """
    JAAD -> PIE transfer search.
    Uses the same informed regularization/dropout/loss space,
    but gives JAAD and PIE separate learning rates.
    """
    cfg = suggest_hyperparams(trial, args)

    cfg["jaad_lr"] = trial.suggest_float(
        "jaad_lr", 1e-5, 8e-5, log=True
    )

    cfg["pie_lr"] = trial.suggest_float(
        "pie_lr", 5e-6, 3e-5, log=True
    )

    # PIE fine-tune LR is the meaningful final lr for logging.
    cfg["lr"] = cfg["pie_lr"]

    return cfg


# ============================================================
# ONE TRIAL: PIE-only
# ============================================================

def run_pie_trial(trial, args, device, train_loader, val_loader, test_loader,
                  pie_wpos, use_amp):

    cfg = suggest_hyperparams(trial, args)
    scaler = GradScaler('cuda', enabled=use_amp)  # fresh scaler per trial — avoids stale scale state

    # Seed per-trial (deterministic within trial, varies across trials)
    set_seed(args.seed + trial.number)

    model = PIPNetAlphaV4Final(
        dropout_p=cfg["dropout_p"],
        local_dropout_p=cfg["local_dropout_p"],
        global_dropout_p=cfg["global_dropout_p"],
    ).to(device)

    criterion = MultiTaskLoss(
        aux_weight=cfg["aux_weight"],
        entropy_weight=cfg["entropy_weight"],
        pos_weight=pie_wpos,
    )

    optimizer, scheduler = build_optimizer(
        model, cfg["lr"], cfg["last_fc_l2"], cfg["visual_l2"]
    )

    best_val_auc = -1.0
    best_state   = None
    patience_counter = 0
    patience = args.early_stopping_patience

    for epoch in range(1, args.epochs_per_trial + 1):
        train_auc = train_one_epoch(model, train_loader, device, criterion,
                                    optimizer, use_amp, scaler)
        val_result = eval_one_epoch(model, val_loader, device, use_amp)
        val_auc    = val_result.get("auc_main", float("nan"))

        scheduler.step(val_auc)

        print(f"  Trial {trial.number} | Epoch {epoch}/{args.epochs_per_trial} "
              f"| train_auc={train_auc:.4f} | val_auc={val_auc:.4f}")

        # ── Optuna intermediate reporting + pruning ──────────────────────────
        trial.report(val_auc, epoch)
        if trial.should_prune():
            print(f"  Trial {trial.number} PRUNED at epoch {epoch}")
            raise optuna.exceptions.TrialPruned()

        if not np.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {
                "epoch": epoch,
                "model": copy.deepcopy(model.state_dict()),
                "val_result": val_result,
                "cfg": cfg,
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Trial {trial.number}: early stop at epoch {epoch}")
                break

    if best_state is None:
        raise optuna.exceptions.TrialPruned()

    # ── Test evaluation ──────────────────────────────────────────────────────
    model.load_state_dict(best_state["model"])
    test_result = eval_one_epoch(model, test_loader, device, use_amp)

    # ── Post-hoc fusion ──────────────────────────────────────────────────────
    val_r  = best_state["val_result"]
    val_logits  = {k.replace("all_logits_", ""): v
                   for k, v in val_r.items()       if k.startswith("all_logits_")}
    test_logits = {k.replace("all_logits_", ""): v
                   for k, v in test_result.items() if k.startswith("all_logits_")}

    fusion = posthoc_fusion_search(
        val_logits, val_r["all_labels"],
        test_logits, test_result["all_labels"],
        step=args.fusion_step,
    )

    # ── Store all metrics as user_attrs ─────────────────────────────────────
    trial.set_user_attr("best_epoch",      best_state["epoch"])
    trial.set_user_attr("best_val_auc",    best_val_auc)
    trial.set_user_attr("test_auc_main",   test_result.get("auc_main",   float("nan")))
    trial.set_user_attr("test_auc_kin",    test_result.get("auc_kin",    float("nan")))
    trial.set_user_attr("test_auc_local",  test_result.get("auc_local",  float("nan")))
    trial.set_user_attr("test_auc_global", test_result.get("auc_global", float("nan")))
    trial.set_user_attr("fusion_val_auc",  fusion["val_auc"])
    trial.set_user_attr("fusion_test_auc", fusion["test_auc"])
    trial.set_user_attr("fusion_test_f1",  fusion["test_f1"])
    trial.set_user_attr("fusion_weights",  json.dumps(fusion["weights"]))
    trial.set_user_attr("cfg",             json.dumps(cfg))

    return fusion["val_auc"], best_state, model  # PRIMARY OBJECTIVE: fusion VAL AUC (no test leakage)


# ============================================================
# ONE TRIAL: JAAD → PIE transfer
# ============================================================

def run_transfer_trial(trial, args, device,
                       jaad_train_loader, jaad_val_loader, jaad_test_loader,
                       pie_train_loader,  pie_val_loader,  pie_test_loader,
                       jaad_wpos, pie_wpos, use_amp):

    cfg = suggest_transfer_hyperparams(trial, args)
    scaler = GradScaler('cuda', enabled=use_amp)  # fresh scaler per trial
    set_seed(args.seed + trial.number)

    # ── S1: JAAD pretrain ────────────────────────────────────────────────────
    model = PIPNetAlphaV4Final(
        dropout_p=cfg["dropout_p"],
        local_dropout_p=cfg["local_dropout_p"],
        global_dropout_p=cfg["global_dropout_p"],
    ).to(device)

    criterion_jaad = MultiTaskLoss(cfg["aux_weight"], cfg["entropy_weight"], jaad_wpos)
    opt_jaad, sch_jaad = build_optimizer(model, cfg["jaad_lr"], cfg["last_fc_l2"], cfg["visual_l2"])

    best_jaad_val = -1.0
    best_jaad_state = None

    for epoch in range(1, args.jaad_epochs + 1):
        train_one_epoch(model, jaad_train_loader, device, criterion_jaad, opt_jaad, use_amp, scaler)
        val_r = eval_one_epoch(model, jaad_val_loader, device, use_amp)
        val_auc = val_r.get("auc_main", float("nan"))
        sch_jaad.step(val_auc)

        if not np.isnan(val_auc) and val_auc > best_jaad_val:
            best_jaad_val = val_auc
            best_jaad_state = copy.deepcopy(model.state_dict())

        # NOTE: No pruning during JAAD pretraining.
        # The pruner compares steps across trials; mixing JAAD and PIE val AUCs
        # on the same step axis would compare apples to oranges and kill good trials.
        # Pruning happens only in the PIE fine-tune stage below.

    if best_jaad_state:
        model.load_state_dict(best_jaad_state)

    # ── S2: PIE fine-tune ────────────────────────────────────────────────────
    criterion_pie = MultiTaskLoss(cfg["aux_weight"], cfg["entropy_weight"], pie_wpos)
    opt_pie, sch_pie = build_optimizer(model, cfg["pie_lr"], cfg["last_fc_l2"], cfg["visual_l2"])

    best_pie_val = -1.0
    best_pie_state = None
    patience_counter = 0

    for epoch in range(1, args.epochs_per_trial + 1):
        train_one_epoch(model, pie_train_loader, device, criterion_pie, opt_pie, use_amp, scaler)
        val_r  = eval_one_epoch(model, pie_val_loader,  device, use_amp)
        val_auc = val_r.get("auc_main", float("nan"))
        sch_pie.step(val_auc)

        # Report PIE epoch directly (1-based) so pruner compares like-for-like.
        trial.report(val_auc, epoch)
        if trial.should_prune():
            print(f"  Trial {trial.number} PRUNED at PIE epoch {epoch}")
            raise optuna.exceptions.TrialPruned()

        if not np.isnan(val_auc) and val_auc > best_pie_val:
            best_pie_val = val_auc
            best_pie_state = {"epoch": epoch, "model": copy.deepcopy(model.state_dict()), "val_result": val_r, "cfg": cfg}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                break

    if best_pie_state is None:
        raise optuna.exceptions.TrialPruned()

    model.load_state_dict(best_pie_state["model"])
    test_result = eval_one_epoch(model, pie_test_loader, device, use_amp)

    val_logits  = {k.replace("all_logits_", ""): v for k, v in best_pie_state["val_result"].items() if k.startswith("all_logits_")}
    test_logits = {k.replace("all_logits_", ""): v for k, v in test_result.items() if k.startswith("all_logits_")}

    fusion = posthoc_fusion_search(val_logits, best_pie_state["val_result"]["all_labels"],
                                   test_logits, test_result["all_labels"],
                                   step=args.fusion_step)

    trial.set_user_attr("best_epoch",      best_pie_state["epoch"])
    trial.set_user_attr("best_jaad_val",   best_jaad_val)
    trial.set_user_attr("best_pie_val",    best_pie_val)
    trial.set_user_attr("test_auc_main",   test_result.get("auc_main",   float("nan")))
    trial.set_user_attr("test_auc_kin",    test_result.get("auc_kin",    float("nan")))
    trial.set_user_attr("test_auc_local",  test_result.get("auc_local",  float("nan")))
    trial.set_user_attr("test_auc_global", test_result.get("auc_global", float("nan")))
    trial.set_user_attr("fusion_val_auc",  fusion["val_auc"])
    trial.set_user_attr("fusion_test_auc", fusion["test_auc"])
    trial.set_user_attr("fusion_test_f1",  fusion["test_f1"])
    trial.set_user_attr("fusion_weights",  json.dumps(fusion["weights"]))
    trial.set_user_attr("cfg",             json.dumps(cfg))

    return fusion["val_auc"], best_pie_state, model  # PRIMARY OBJECTIVE: fusion VAL AUC (no test leakage)


# ============================================================
# CSV LOG
# ============================================================

CSV_COLUMNS = [
    "trial_number", "state", "objective_fusion_val_auc",
    "best_epoch", "best_val_auc",
    "test_auc_main", "test_auc_kin", "test_auc_local", "test_auc_global",
    "fusion_val_auc", "fusion_test_auc", "fusion_test_f1", "fusion_weights",
    "lr", "last_fc_l2", "visual_l2", "aux_weight", "entropy_weight",
    "dropout_p", "local_dropout_p", "global_dropout_p",
    "timestamp",
]


def write_trial_csv(csv_path, trial):
    is_new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(CSV_COLUMNS)

        attrs = trial.user_attrs
        cfg   = json.loads(attrs.get("cfg", "{}"))

        row = [
            trial.number,
            "RUNNING",  # trial.state unavailable inside objective (FrozenTrial only); always RUNNING here
            f"{attrs.get('fusion_val_auc', float('nan')):.6f}",  # trial.value is None here (set after return); use stored attr
            attrs.get("best_epoch", ""),
            f"{attrs.get('best_val_auc', float('nan')):.6f}",
            f"{attrs.get('test_auc_main',   float('nan')):.6f}",
            f"{attrs.get('test_auc_kin',    float('nan')):.6f}",
            f"{attrs.get('test_auc_local',  float('nan')):.6f}",
            f"{attrs.get('test_auc_global', float('nan')):.6f}",
            f"{attrs.get('fusion_val_auc',  float('nan')):.6f}",
            f"{attrs.get('fusion_test_auc', float('nan')):.6f}",
            f"{attrs.get('fusion_test_f1',  float('nan')):.6f}",
            attrs.get("fusion_weights", ""),
            f"{cfg.get('lr', float('nan')):.2e}",
            f"{cfg.get('last_fc_l2', float('nan')):.2e}",
            f"{cfg.get('visual_l2', float('nan')):.2e}",
            f"{cfg.get('aux_weight', float('nan')):.4f}",
            f"{cfg.get('entropy_weight', float('nan')):.4f}",
            f"{cfg.get('dropout_p', float('nan')):.2f}",
            f"{cfg.get('local_dropout_p', float('nan')):.2f}",
            f"{cfg.get('global_dropout_p', float('nan')):.2f}",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ]
        w.writerow(row)


# ============================================================
# MAIN
# ============================================================

def main():
    p = argparse.ArgumentParser(description="HPO for PIPNet-Alpha V4 with Optuna")

    # Data
    p.add_argument("--pie_root",  required=True, help="PIE prep output root")
    p.add_argument("--jaad_root", default=None,  help="JAAD prep output root (for --search_transfer)")
    p.add_argument("--seq_len",   type=int, default=10)

    # Search
    p.add_argument("--n_trials",  type=int, default=40, help="Total number of Optuna trials")
    p.add_argument("--search_transfer", action="store_true",
                   help="Search over JAAD→PIE transfer (includes jaad_lr, pie_lr)")
    p.add_argument("--jaad_epochs",        type=int, default=20, help="JAAD pretrain epochs per trial")
    p.add_argument("--epochs_per_trial",   type=int, default=15, help="PIE training epochs per trial")
    p.add_argument("--early_stopping_patience", type=int, default=5)

    # Pruner
    p.add_argument("--pruner_warmup_steps",  type=int, default=5,
                   help="Pruner: min epochs before pruning starts")
    p.add_argument("--pruner_warmup_trials", type=int, default=5,
                   help="Pruner: min completed trials before pruning activates")
    p.add_argument("--pruner_n_min_trials",  type=int, default=3)

    # Training
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--num_workers",  type=int,   default=2)
    p.add_argument("--amp",          action="store_true")
    p.add_argument("--device",       default="cuda")

    # Post-hoc fusion grid resolution
    p.add_argument("--fusion_step",  type=float, default=0.05,
                   help="Step size for fusion weight grid search (smaller = finer but slower)")

    # Persistence
    p.add_argument("--save_dir",     required=True)
    p.add_argument("--study_name",   default="pipnet_v4_hpo")
    p.add_argument("--seed",         type=int, default=42)

    args = p.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    device   = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp  = (device.type == "cuda") and args.amp
    # Note: GradScaler is created fresh per-trial inside run_pie_trial/run_transfer_trial
    # This avoids stale scale state propagating between trials.

    print("=" * 70)
    print("PIPNet-Alpha V4 — Optuna HPO")
    print("=" * 70)
    print(f"Device:         {device}")
    print(f"AMP:            {use_amp}")
    print(f"n_trials:       {args.n_trials}")
    print(f"epochs/trial:   {args.epochs_per_trial}")
    print(f"batch_size:     {args.batch_size}")
    print(f"num_workers:    {args.num_workers}")
    print(f"save_dir:       {args.save_dir}")
    print(f"search_transfer:{args.search_transfer}")
    print(f"objective:      post-hoc fusion VAL AUC  (test AUC logged only — no leakage)")
    print("=" * 70)

    # ── Loaders ─────────────────────────────────────────────────────────────
    pie_train_ds, pie_train_loader, pie_val_loader, pie_test_loader = make_loaders(
        args.pie_root, "pie", args.seq_len, args.batch_size, args.num_workers
    )
    pie_wpos = compute_pos_weight(pie_train_ds.files)
    print(f"PIE pos_weight: {pie_wpos:.3f}")

    jaad_train_loader = jaad_val_loader = jaad_test_loader = None
    jaad_wpos = 1.0
    if args.search_transfer:
        if args.jaad_root is None:
            raise ValueError("--jaad_root is required when --search_transfer is set")
        jaad_train_ds, jaad_train_loader, jaad_val_loader, jaad_test_loader = make_loaders(
            args.jaad_root, "jaad", args.seq_len, args.batch_size, args.num_workers
        )
        jaad_wpos = compute_pos_weight(jaad_train_ds.files)
        print(f"JAAD pos_weight: {jaad_wpos:.3f}")

    # ── Optuna study ─────────────────────────────────────────────────────────
    db_path  = os.path.join(args.save_dir, f"{args.study_name}.db")
    csv_path = os.path.join(args.save_dir, "hpo_results.csv")
    storage  = f"sqlite:///{db_path}"

    sampler = TPESampler(seed=args.seed, n_startup_trials=10, multivariate=True)
    pruner  = MedianPruner(
        n_startup_trials=args.pruner_warmup_trials,
        n_warmup_steps=args.pruner_warmup_steps,
        n_min_trials=args.pruner_n_min_trials,
    )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=True,   # ← auto-resume if study already exists in DB
    )
    # ------------------------------------------------------------
    # Seed Optuna with the best known PIE-only configuration.
    # Only enqueue on a fresh study, otherwise resume runs would
    # enqueue the same trial again.
    # ------------------------------------------------------------
    if len(study.trials) == 0:
        if args.search_transfer:
            study.enqueue_trial({
                # suggest_hyperparams() asks for "lr" before it is overwritten
                # by pie_lr, so include it to make the queued trial fully fixed.
                "lr": 1.9906996673933362e-05,

                "last_fc_l2": 0.0071144760093434225,
                "visual_l2": 0.004,
                "aux_weight": 0.25,
                "entropy_weight": 0.03,
                "dropout_p": 0.2,
                "local_dropout_p": 0.1,
                "global_dropout_p": 0.4,

                "jaad_lr": 3e-05,
                "pie_lr": 1.9906996673933362e-05,
            })
        else:
            study.enqueue_trial({
                "lr": 1.9906996673933362e-05,
                "last_fc_l2": 0.0071144760093434225,
                "visual_l2": 0.004,
                "aux_weight": 0.25,
                "entropy_weight": 0.03,
                "dropout_p": 0.2,
                "local_dropout_p": 0.1,
                "global_dropout_p": 0.4,
            })       
    existing = len(study.trials)
    if existing > 0:
        print(f"\n[RESUME] Found {existing} existing trials in {db_path}")
        print(f"         Will run up to {args.n_trials} total trials.\n")

    # Resume-safe: initialise best from existing completed trials so a new
    # (potentially worse) trial never overwrites the checkpoint from a prior run.
    completed_existing = [t for t in study.trials if t.value is not None]
    if completed_existing:
        best_overall_fusion_val_auc = study.best_value
        print(f"[RESUME] Best existing fusion_val_auc = {best_overall_fusion_val_auc:.4f}")
    else:
        best_overall_fusion_val_auc = -1.0
    best_trial_state = None

    def objective(trial):
        nonlocal best_overall_fusion_val_auc, best_trial_state

        print(f"\n{'─'*60}")
        print(f"Trial {trial.number} started | "
              f"{datetime.now().strftime('%H:%M:%S')}")
        print(f"{'─'*60}")

        if args.search_transfer:
            fusion_auc, best_state, model = run_transfer_trial(
                trial, args, device,
                jaad_train_loader, jaad_val_loader, jaad_test_loader,
                pie_train_loader,  pie_val_loader,  pie_test_loader,
                jaad_wpos, pie_wpos, use_amp,
            )
        else:
            fusion_auc, best_state, model = run_pie_trial(
                trial, args, device,
                pie_train_loader, pie_val_loader, pie_test_loader,
                pie_wpos, use_amp,
            )

        # Save CSV row for this trial
        write_trial_csv(csv_path, trial)

        # Save new best checkpoint
        if fusion_auc > best_overall_fusion_val_auc:
            best_overall_fusion_val_auc = fusion_auc
            best_trial_state = best_state

            best_dir = os.path.join(args.save_dir, "best_trial")
            os.makedirs(best_dir, exist_ok=True)
            torch.save(
                {
                    "trial_number": trial.number,
                    "fusion_val_auc": fusion_auc,   # selection criterion (val, no leakage)
                    "cfg": best_state["cfg"],
                    "model": best_state["model"],
                    "epoch": best_state["epoch"],
                    "fusion_weights": json.loads(
                        trial.user_attrs.get("fusion_weights", "{}")
                    ),
                },
                os.path.join(best_dir, "best_model.pth"),
            )

            # Also save cfg as JSON for easy inspection
            with open(os.path.join(best_dir, "best_cfg.json"), "w") as f:
                json.dump({
                    "trial_number": trial.number,
                    "fusion_val_auc": fusion_auc,   # selection criterion
                    "cfg": best_state["cfg"],
                    "fusion_weights": json.loads(
                        trial.user_attrs.get("fusion_weights", "{}")
                    ),
                }, f, indent=2)

            print(f"\n★ NEW BEST: fusion_val_auc={fusion_auc:.4f} "
                  f"(trial {trial.number}, epoch {best_state['epoch']})")

        # Print trial summary
        attrs = trial.user_attrs
        print(f"\nTrial {trial.number} DONE:")
        print(f"  fusion_val_auc  = {fusion_auc:.4f}  (PRIMARY OBJECTIVE — val only)")
        print(f"  fusion_test_auc = {attrs.get('fusion_test_auc', float('nan')):.4f}  (logged, not optimized)")
        print(f"  test_auc_main   = {attrs.get('test_auc_main',  float('nan')):.4f}")
        print(f"  test_auc_kin    = {attrs.get('test_auc_kin',   float('nan')):.4f}")
        print(f"  test_auc_local  = {attrs.get('test_auc_local', float('nan')):.4f}")
        print(f"  test_auc_global = {attrs.get('test_auc_global',float('nan')):.4f}")
        print(f"  fusion_weights  = {attrs.get('fusion_weights', '')}")
        print(f"  best_epoch      = {attrs.get('best_epoch', '')}")

        return fusion_auc

    # ── Run ───────────────────────────────────────────────────────────────────
    remaining = max(0, args.n_trials - len([t for t in study.trials
                                            if t.state != optuna.trial.TrialState.WAITING]))
    print(f"\nRunning {remaining} new trial(s)...\n")

    study.optimize(
        objective,
        n_trials=remaining,
        catch=(RuntimeError, torch.cuda.OutOfMemoryError),
    )

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("HPO COMPLETE")
    print("=" * 70)

    completed = [t for t in study.trials if t.value is not None]
    if not completed:
        print("No completed trials.")
        return

    best = study.best_trial
    print(f"\nBest trial:        #{best.number}")
    print(f"Fusion Val AUC (objective): {best.value:.4f}")
    print(f"Fusion Test AUC (reported): {best.user_attrs.get('fusion_test_auc', float('nan')):.4f}")
    print(f"Best epoch:        {best.user_attrs.get('best_epoch', '?')}")
    print(f"\nBranch AUCs:")
    print(f"  main   = {best.user_attrs.get('test_auc_main',   float('nan')):.4f}")
    print(f"  kin    = {best.user_attrs.get('test_auc_kin',    float('nan')):.4f}")
    print(f"  local  = {best.user_attrs.get('test_auc_local',  float('nan')):.4f}")
    print(f"  global = {best.user_attrs.get('test_auc_global', float('nan')):.4f}")
    print(f"\nFusion weights:  {best.user_attrs.get('fusion_weights', '')}")
    print(f"\nBest hyperparameters:")
    best_cfg = json.loads(best.user_attrs.get("cfg", "{}"))
    for k, v in best_cfg.items():
        print(f"  {k:25s} = {v}")

    print(f"\nResults saved to: {csv_path}")
    print(f"Best checkpoint:  {os.path.join(args.save_dir, 'best_trial', 'best_model.pth')}")
    print(f"Study DB:         {db_path}")
    print("\nTo visualize the study:")
    print(f"  optuna-dashboard sqlite:///{db_path}")


if __name__ == "__main__":
    main()