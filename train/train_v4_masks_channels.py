# train_v4_masks_channels.py
"""
Two-stage Transfer Learning: JAAD pretrain -> PIE fine-tune
Includes PIE-only baseline for direct comparison.

V4:
  - Uses PIPNetAlphaV4Final mask-channel semantic variant.
  - Global Branch Refactor: Conv3D towers + Flatten + FC + GRU.
  - visual_fuse_weights entropy operates over T=10 tokens.
  - CSV logs parameters, training arguments, command, and visual diagnostic AUC.
  - Targeted L2 regularization:
        base/kinematic params: no weight decay
        visual/global params: --visual_l2
        final fc_out params:  --last_fc_l2

Fixes vs original V4:
  Fix A -- local_dropout_p and global_dropout_p added to argparse + make_model()
           so the best Optuna trial (local=0.1, global=0.4) is reproducible.
  Fix B -- ReduceLROnPlateau patience 4 -> 3, matching Optuna build_optimizer().
  Fix C -- CSV columns and log_csv() include local_dropout_p, global_dropout_p.
"""

import os
import sys
import csv
import argparse
import copy
from datetime import datetime
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import random
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

from data.pie import PIESeqDataset
from experiments_archive.old_models.pipnet_alpha_v4_masks_channels_final import PIPNetAlphaV4Final
from torch.cuda.amp import autocast, GradScaler


# ============================================================
# MULTI-TASK LOSS
# ============================================================

class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        aux_weight: float = 0.1,
        entropy_weight: float = 0.05,
        pos_weight: float = 1.0,
    ):
        super().__init__()
        self.aux_weight = aux_weight
        self.entropy_weight = entropy_weight
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )

    def forward(self, outputs: dict, labels: torch.Tensor):
        device = labels.device
        self.bce.pos_weight = self.bce.pos_weight.to(device)
        labels = labels.float()

        losses = {}

        main_loss = self.bce(outputs["logit"].squeeze(-1), labels)
        losses["main"] = main_loss

        if "aux_kin" in outputs:
            aux_kin_loss = self.bce(outputs["aux_kin"].squeeze(-1), labels)
            aux_local_loss = self.bce(outputs["aux_local"].squeeze(-1), labels)
            aux_global_loss = self.bce(outputs["aux_global"].squeeze(-1), labels)

            aux_loss = self.aux_weight * (
                aux_kin_loss + aux_local_loss + aux_global_loss
            )

            losses["aux"] = aux_loss
            losses["aux_kin_loss"] = aux_kin_loss.item()
            losses["aux_local_loss"] = aux_local_loss.item()
            losses["aux_global_loss"] = aux_global_loss.item()
        else:
            aux_loss = torch.tensor(0.0, device=device)
            losses["aux"] = aux_loss

        entropy_loss = torch.tensor(0.0, device=device)

        if "modality_weights" in outputs:
            w = outputs["modality_weights"]
            ent = -(w * torch.log(w + 1e-8)).sum(dim=-1).mean()
            ent_norm = ent / torch.log(
                torch.tensor(w.shape[-1], device=device, dtype=ent.dtype)
            )
            term = -self.entropy_weight * ent_norm
            entropy_loss = entropy_loss + term

            losses["entropy_modality"] = term.item()
            losses["entropy_modality_raw"] = ent.item()
            losses["entropy_modality_norm"] = ent_norm.item()

        if "visual_fuse_weights" in outputs:
            w2 = outputs["visual_fuse_weights"]
            ent2 = -(w2 * torch.log(w2 + 1e-8)).sum(dim=-1).mean()
            ent2_norm = ent2 / torch.log(
                torch.tensor(w2.shape[-1], device=device, dtype=ent2.dtype)
            )
            term2 = -self.entropy_weight * ent2_norm
            entropy_loss = entropy_loss + term2

            losses["entropy_visual_fuse"] = term2.item()
            losses["entropy_visual_fuse_raw"] = ent2.item()
            losses["entropy_visual_fuse_norm"] = ent2_norm.item()

        losses["entropy"] = entropy_loss

        total = main_loss + aux_loss + entropy_loss
        losses["total"] = total

        return losses


# ============================================================
# UTILITIES
# ============================================================

def compute_pos_weight_from_npz(files):
    pos, neg = 0, 0

    for p in files:
        d = np.load(p, allow_pickle=True)
        y = float(np.array(d["label"]).reshape(-1)[0])

        if y >= 0.5:
            pos += 1
        else:
            neg += 1

    return float(neg / max(pos, 1)), int(pos), int(neg)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# Fix A: single factory so all three stages use identical dropout config
def make_model(args, device):
    return PIPNetAlphaV4Final(
        dropout_p=args.dropout_p,
        local_dropout_p=args.local_dropout_p,
        global_dropout_p=args.global_dropout_p,
    ).to(device)


def make_loaders(
    data_root,
    dataset_prefix,
    seq_len,
    batch_size,
    num_workers=4,
    strict_len=True,
):
    speed_stats_path = (
        f"/workspace/project/data/{dataset_prefix}_speed_stats_splits.json"
    )
    motion_stats_path = (
        f"/workspace/project/data/{dataset_prefix}_motion_stats_splits.json"
    )

    common_args = dict(
        seq_len=seq_len,
        strict_len=strict_len,
        speed_norm="minmax",
        speed_stats_path=speed_stats_path,
        speed_scope="global",
        motion_norm="p99abs",
        motion_stats_path=motion_stats_path,
        motion_scope="global",
        motion_clip=1.0,
    )

    train_ds = PIESeqDataset(data_root, split="train", mode="train", **common_args)
    val_ds   = PIESeqDataset(data_root, split="val",   mode="eval",  **common_args)
    test_ds  = PIESeqDataset(data_root, split="test",  mode="eval",  **common_args)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_ds, train_loader, val_loader, test_loader


def safe_auc(labels, probs):
    try:
        if len(np.unique(labels)) > 1:
            return roc_auc_score(labels, probs)
    except Exception:
        pass
    return float("nan")


@torch.no_grad()
def compute_metrics(labels_np, probs_np):
    metrics = {}
    metrics["auc"] = safe_auc(labels_np, probs_np)
    preds_bin = (probs_np >= 0.5).astype(np.int32)
    metrics["acc"]       = (preds_bin == labels_np).mean()
    metrics["f1"]        = f1_score(labels_np, preds_bin, zero_division=0)
    metrics["precision"] = precision_score(labels_np, preds_bin, zero_division=0)
    metrics["recall"]    = recall_score(labels_np, preds_bin, zero_division=0)
    return metrics


# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def run_train_epoch(model, loader, device, criterion, optimizer, use_amp=False, scaler=None):
    model.train()

    all_labels, all_probs = [], []
    all_probs_kin, all_probs_local, all_probs_global, all_probs_visual = [], [], [], []

    running_loss = running_aux_kin = running_aux_local = running_aux_global = 0.0
    total = 0
    modality_weights_sum = torch.zeros(2)
    weight_count = 0
    visual_fuse_n_tokens = 0

    pbar = tqdm(loader, desc="Train")

    for batch in pbar:
        for k in ["bbox", "pose", "speed", "local_cnn", "local_motion",
                  "sem_labels", "cat_depth", "label"]:
            if k in batch and torch.is_tensor(batch[k]):
                batch[k] = batch[k].to(device, non_blocking=True)

        labels = batch["label"].float()
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            outputs = model(batch, return_aux=True)
            losses  = criterion(outputs, labels)

        if use_amp and scaler is not None:
            scaler.scale(losses["total"]).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses["total"].backward()
            optimizer.step()

        bs = labels.size(0)
        running_loss += losses["total"].item() * bs
        total += bs

        if "aux_kin_loss" in losses:
            running_aux_kin    += losses["aux_kin_loss"]    * bs
            running_aux_local  += losses["aux_local_loss"]  * bs
            running_aux_global += losses["aux_global_loss"] * bs

        probs = torch.sigmoid(outputs["logit"].squeeze(-1)).detach()
        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

        if "aux_kin" in outputs:
            all_probs_kin.append(
                torch.sigmoid(outputs["aux_kin"].squeeze(-1)).detach().cpu().numpy())
            all_probs_local.append(
                torch.sigmoid(outputs["aux_local"].squeeze(-1)).detach().cpu().numpy())
            all_probs_global.append(
                torch.sigmoid(outputs["aux_global"].squeeze(-1)).detach().cpu().numpy())

            if "z_visual" in outputs and hasattr(model, "aux_local"):
                with torch.no_grad():
                    visual_logit = model.aux_local(outputs["z_visual"].detach()).squeeze(-1)
                    all_probs_visual.append(torch.sigmoid(visual_logit).detach().cpu().numpy())

        if "modality_weights" in outputs:
            modality_weights_sum += outputs["modality_weights"].detach().cpu().mean(0)
            weight_count += 1

        if "visual_fuse_weights" in outputs and visual_fuse_n_tokens == 0:
            visual_fuse_n_tokens = outputs["visual_fuse_weights"].shape[1]

        pbar.set_postfix({"loss": f"{losses['total'].item():.3f}"})

    labels_np = np.concatenate(all_labels)
    probs_np  = np.concatenate(all_probs)
    metrics   = compute_metrics(labels_np, probs_np)
    metrics["loss"] = running_loss / max(total, 1)

    if all_probs_kin:
        metrics["auc_kin"]    = safe_auc(labels_np, np.concatenate(all_probs_kin))
        metrics["auc_local"]  = safe_auc(labels_np, np.concatenate(all_probs_local))
        metrics["auc_global"] = safe_auc(labels_np, np.concatenate(all_probs_global))
        metrics["loss_kin"]    = running_aux_kin    / max(total, 1)
        metrics["loss_local"]  = running_aux_local  / max(total, 1)
        metrics["loss_global"] = running_aux_global / max(total, 1)

    if all_probs_visual:
        metrics["auc_visual"] = safe_auc(labels_np, np.concatenate(all_probs_visual))

    if weight_count > 0:
        avg = modality_weights_sum / weight_count
        metrics["w_visual"] = avg[0].item()
        metrics["w_kin"]    = avg[1].item()

    if visual_fuse_n_tokens > 0:
        metrics["visual_fuse_tokens"] = visual_fuse_n_tokens

    return metrics


@torch.no_grad()
def run_eval_epoch(model, loader, device, criterion, use_amp=False):
    model.eval()

    all_labels, all_probs = [], []
    all_probs_kin, all_probs_local, all_probs_global, all_probs_visual = [], [], [], []

    running_loss = running_aux_kin = running_aux_local = running_aux_global = 0.0
    total = 0
    modality_weights_sum = torch.zeros(2)
    weight_count = 0
    visual_fuse_n_tokens = 0

    for batch in tqdm(loader, desc="Eval"):
        for k in ["bbox", "pose", "speed", "local_cnn", "local_motion",
                  "sem_labels", "cat_depth", "label"]:
            if k in batch and torch.is_tensor(batch[k]):
                batch[k] = batch[k].to(device, non_blocking=True)

        labels = batch["label"].float()

        with autocast(enabled=use_amp):
            outputs = model(batch, return_aux=True)
            losses  = criterion(outputs, labels)

        bs = labels.size(0)
        running_loss += losses["total"].item() * bs
        total += bs

        if "aux_kin_loss" in losses:
            running_aux_kin    += losses["aux_kin_loss"]    * bs
            running_aux_local  += losses["aux_local_loss"]  * bs
            running_aux_global += losses["aux_global_loss"] * bs

        probs = torch.sigmoid(outputs["logit"].squeeze(-1))
        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

        if "aux_kin" in outputs:
            all_probs_kin.append(
                torch.sigmoid(outputs["aux_kin"].squeeze(-1)).cpu().numpy())
            all_probs_local.append(
                torch.sigmoid(outputs["aux_local"].squeeze(-1)).cpu().numpy())
            all_probs_global.append(
                torch.sigmoid(outputs["aux_global"].squeeze(-1)).cpu().numpy())

            if "z_visual" in outputs and hasattr(model, "aux_local"):
                visual_logit = model.aux_local(outputs["z_visual"]).squeeze(-1)
                all_probs_visual.append(torch.sigmoid(visual_logit).cpu().numpy())

        if "modality_weights" in outputs:
            modality_weights_sum += outputs["modality_weights"].cpu().mean(0)
            weight_count += 1

        if "visual_fuse_weights" in outputs and visual_fuse_n_tokens == 0:
            visual_fuse_n_tokens = outputs["visual_fuse_weights"].shape[1]

    labels_np = np.concatenate(all_labels)
    probs_np  = np.concatenate(all_probs)
    metrics   = compute_metrics(labels_np, probs_np)
    metrics["loss"] = running_loss / max(total, 1)

    if all_probs_kin:
        metrics["auc_kin"]    = safe_auc(labels_np, np.concatenate(all_probs_kin))
        metrics["auc_local"]  = safe_auc(labels_np, np.concatenate(all_probs_local))
        metrics["auc_global"] = safe_auc(labels_np, np.concatenate(all_probs_global))
        metrics["loss_kin"]    = running_aux_kin    / max(total, 1)
        metrics["loss_local"]  = running_aux_local  / max(total, 1)
        metrics["loss_global"] = running_aux_global / max(total, 1)

    if all_probs_visual:
        metrics["auc_visual"] = safe_auc(labels_np, np.concatenate(all_probs_visual))

    if weight_count > 0:
        avg = modality_weights_sum / weight_count
        metrics["w_visual"] = avg[0].item()
        metrics["w_kin"]    = avg[1].item()

    if visual_fuse_n_tokens > 0:
        metrics["visual_fuse_tokens"] = visual_fuse_n_tokens

    return metrics


def print_metrics(metrics, prefix=""):
    print(
        f"{prefix} | loss: {metrics['loss']:.4f} | "
        f"auc: {metrics['auc']:.3f} | "
        f"f1: {metrics['f1']:.3f} | "
        f"acc: {metrics['acc']:.3f}"
    )
    if "auc_kin" in metrics:
        print(
            f"{prefix} | Branch AUC:  "
            f"kin={metrics['auc_kin']:.3f}  "
            f"local={metrics['auc_local']:.3f}  "
            f"global(v4)={metrics['auc_global']:.3f}  "
            f"visual={metrics.get('auc_visual', float('nan')):.3f}"
        )
    if "w_visual" in metrics:
        print(
            f"{prefix} | Final Attn:  "
            f"visual={metrics['w_visual']:.3f}  "
            f"kin={metrics['w_kin']:.3f}"
        )
    if "visual_fuse_tokens" in metrics:
        print(f"{prefix} | Visual fuse tokens: {metrics['visual_fuse_tokens']}")


# ============================================================
# CSV LOGGER
# ============================================================

# Fix C: added local_dropout_p and global_dropout_p
CSV_COLUMNS = [
    "stage", "epoch", "split",
    "loss", "auc", "f1", "acc", "precision", "recall",
    "auc_kin", "auc_local", "auc_global", "auc_visual",
    "loss_kin", "loss_local", "loss_global",
    "w_visual", "w_kin", "visual_fuse_tokens",
    "total_params", "trainable_params",
    "jaad_root", "pie_root",
    "skip_transfer", "skip_baseline",
    "jaad_epochs", "jaad_lr", "jaad_batch_size",
    "pie_epochs", "pie_lr", "pie_batch_size",
    "baseline_epochs", "baseline_lr",
    "aux_weight", "entropy_weight", "last_fc_l2", "visual_l2",
    "local_dropout_p",   # Fix C
    "global_dropout_p",  # Fix C
    "early_stopping_patience",
    "amp", "seed", "save_dir", "command",
    "lr", "timestamp",
]


def init_csv(path):
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(CSV_COLUMNS)


def log_csv(path, stage, epoch, split, metrics, args, total_params, trainable_params, lr=None):
    row = [
        stage, epoch, split,
        f"{metrics.get('loss', 0):.6f}",
        f"{metrics.get('auc', 0):.6f}",
        f"{metrics.get('f1', 0):.6f}",
        f"{metrics.get('acc', 0):.6f}",
        f"{metrics.get('precision', 0):.6f}",
        f"{metrics.get('recall', 0):.6f}",
        f"{metrics.get('auc_kin', '')}"    if "auc_kin"    in metrics else "",
        f"{metrics.get('auc_local', '')}"  if "auc_local"  in metrics else "",
        f"{metrics.get('auc_global', '')}" if "auc_global" in metrics else "",
        f"{metrics.get('auc_visual', '')}" if "auc_visual" in metrics else "",
        f"{metrics.get('loss_kin', '')}"   if "loss_kin"   in metrics else "",
        f"{metrics.get('loss_local', '')}" if "loss_local" in metrics else "",
        f"{metrics.get('loss_global', '')}"if "loss_global"in metrics else "",
        f"{metrics.get('w_visual', '')}"   if "w_visual"   in metrics else "",
        f"{metrics.get('w_kin', '')}"      if "w_kin"      in metrics else "",
        f"{metrics.get('visual_fuse_tokens', '')}" if "visual_fuse_tokens" in metrics else "",
        total_params, trainable_params,
        args.jaad_root, args.pie_root,
        args.skip_transfer, args.skip_baseline,
        args.jaad_epochs, args.jaad_lr, args.jaad_batch_size,
        args.pie_epochs, args.pie_lr, args.pie_batch_size,
        args.baseline_epochs, args.baseline_lr,
        args.aux_weight, args.entropy_weight, args.last_fc_l2, args.visual_l2,
        args.local_dropout_p,   # Fix C
        args.global_dropout_p,  # Fix C
        args.early_stopping_patience,
        args.amp, args.seed, args.save_dir,
        getattr(args, "command", ""),
        f"{lr:.2e}" if lr is not None else "",
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    ]
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)


# ============================================================
# SINGLE TRAINING STAGE
# ============================================================

def run_training_stage(
    stage_name, model, device, criterion,
    train_loader, val_loader, test_loader,
    optimizer, scheduler, scaler,
    epochs, use_amp, save_dir, csv_path, args,
    early_stop_patience=10,
):
    os.makedirs(save_dir, exist_ok=True)
    total_params, trainable_params = count_parameters(model)

    print("\n" + "=" * 80)
    print(f"[{stage_name}] Model parameters")
    print("=" * 80)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print("=" * 80)

    best_val_auc = -1.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        print(f"\n{'=' * 80}")
        print(f"[{stage_name}] Epoch {epoch}/{epochs}")
        print(f"{'=' * 80}")

        current_lr = optimizer.param_groups[0]["lr"]

        train_m = run_train_epoch(model, train_loader, device, criterion,
                                  optimizer, use_amp, scaler)
        print_metrics(train_m, f"[{stage_name}] Train")
        log_csv(csv_path, stage_name, epoch, "train", train_m, args,
                total_params, trainable_params, lr=current_lr)

        val_m = run_eval_epoch(model, val_loader, device, criterion, use_amp)
        print_metrics(val_m, f"[{stage_name}] Val  ")
        log_csv(csv_path, stage_name, epoch, "val", val_m, args,
                total_params, trainable_params, lr=current_lr)

        if scheduler is not None:
            scheduler.step(val_m["auc"])

        if val_m["auc"] > best_val_auc:
            best_val_auc = val_m["auc"]
            best_state = {
                "epoch":       epoch,
                "stage":       stage_name,
                "model":       copy.deepcopy(model.state_dict()),
                "optimizer":   copy.deepcopy(optimizer.state_dict()),
                "val_metrics": val_m,
            }
            torch.save(best_state, os.path.join(save_dir, "best_model.pth"))
            print(f"  * New best model (Val AUC: {best_val_auc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  ! No improvement for {patience_counter} epoch(s).")

        if patience_counter >= early_stop_patience:
            print(f"\n Early stopping triggered after {early_stop_patience} epochs.")
            break

    test_m = {}
    if best_state:
        print(f"\n[{stage_name}] Loading best model (Epoch {best_state['epoch']})...")
        model.load_state_dict(best_state["model"])
        test_m = run_eval_epoch(model, test_loader, device, criterion, use_amp)
        print_metrics(test_m, f"[{stage_name}] Test ")
        log_csv(csv_path, stage_name, best_state["epoch"], "test", test_m, args,
                total_params, trainable_params)

    return test_m, best_state


# ============================================================
# OPTIMIZER + SCHEDULER
# ============================================================

def build_optimizer(model, lr, last_fc_l2, fixed_lr=False, visual_l2=0.0):
    """
    Targeted weight decay:
      base_params    — kinematic + kin_attn + final_attn: no decay
      visual_params  — local/global branches + fusion:   visual_l2
      final_fc_params — fc_out:                          last_fc_l2
    """
    base_params = []
    visual_params = []
    final_fc_params = []

    visual_keywords = [
        "local_branch", "local_attn", "norm_local", "aux_local",
        "global_branch", "global_attn", "norm_global", "aux_global",
        "visual_fuse_attn", "norm_visual",
    ]

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "fc_out" in name or "branch_logit_fusion" in name:
            final_fc_params.append(p)
        elif any(k in name for k in visual_keywords):
            visual_params.append(p)
        else:
            base_params.append(p)

    print("\nOptimizer parameter groups:")
    print(f"  base/no_decay params:       {sum(p.numel() for p in base_params):,}")
    print(f"  visual weight_decay params: {sum(p.numel() for p in visual_params):,} wd={visual_l2}")
    print(f"  final_fc params:            {sum(p.numel() for p in final_fc_params):,} wd={last_fc_l2}")

    optimizer = torch.optim.AdamW(
        [
            {"params": base_params,       "weight_decay": 0.0},
            {"params": visual_params,     "weight_decay": visual_l2},
            {"params": final_fc_params,   "weight_decay": last_fc_l2},
        ],
        lr=lr,
    )

    if fixed_lr:
        scheduler = None
    else:
        # Fix B: patience=3 matches Optuna's build_optimizer() (was 4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-7,
        )

    return optimizer, scheduler


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--jaad_root", type=str, required=True)
    parser.add_argument("--pie_root",  type=str, required=True)

    parser.add_argument("--jaad_ckpt",     type=str, default=None)
    parser.add_argument("--skip_transfer", action="store_true")
    parser.add_argument("--skip_baseline", action="store_true")

    parser.add_argument("--seq_len", type=int, default=10)

    # Fix A: all three dropout params exposed
    parser.add_argument("--dropout_p",        type=float, default=0.5)
    parser.add_argument("--local_dropout_p",  type=float, default=0.3)
    parser.add_argument("--global_dropout_p", type=float, default=0.2)

    parser.add_argument("--jaad_epochs",     type=int,   default=80)
    parser.add_argument("--jaad_lr",         type=float, default=5e-5)
    parser.add_argument("--jaad_batch_size", type=int,   default=10)

    parser.add_argument("--pie_epochs",    type=int,   default=50)
    parser.add_argument("--pie_lr",        type=float, default=1e-5)
    parser.add_argument("--pie_batch_size",type=int,   default=10)

    parser.add_argument("--baseline_epochs", type=int,   default=50)
    parser.add_argument("--baseline_lr",     type=float, default=5e-5)

    parser.add_argument("--last_fc_l2",    type=float, default=1e-4)
    parser.add_argument("--visual_l2",     type=float, default=0.0)

    parser.add_argument("--aux_weight",     type=float, default=0.1)
    parser.add_argument("--entropy_weight", type=float, default=0.05)

    parser.add_argument("--fixed_lr",               action="store_true")
    parser.add_argument("--early_stopping_patience",type=int, default=10)

    parser.add_argument("--device",      type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp",         action="store_true")
    parser.add_argument("--wpos",        type=float, default=-1.0)
    parser.add_argument("--save_dir",    type=str, default="checkpoints_transfer")
    parser.add_argument("--no_strict_len", action="store_true")

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    args.command = " ".join(sys.argv)

    set_seed(args.seed)

    device    = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp   = (device.type == "cuda") and args.amp
    scaler    = GradScaler(enabled=use_amp)
    strict_len = not args.no_strict_len

    os.makedirs(args.save_dir, exist_ok=True)
    csv_path = os.path.join(args.save_dir, "training_log.csv")
    init_csv(csv_path)

    transfer_best_state = None
    jaad_ckpt           = None

    # ------------------------------------------------------------------ S1/S2
    if not args.skip_transfer:
        stage1_dir = os.path.join(args.save_dir, "stage1_jaad")

        if args.jaad_ckpt:
            jaad_ckpt = torch.load(args.jaad_ckpt, map_location=device)
        else:
            jaad_train_ds, jaad_train_loader, jaad_val_loader, jaad_test_loader = \
                make_loaders(args.jaad_root, "jaad", args.seq_len,
                             args.jaad_batch_size, args.num_workers, strict_len)

            jaad_wpos = args.wpos if args.wpos > 0 else \
                compute_pos_weight_from_npz(jaad_train_ds.files)[0]

            # Fix A: make_model()
            model_s1     = make_model(args, device)
            criterion_s1 = MultiTaskLoss(args.aux_weight, args.entropy_weight, jaad_wpos)
            opt_s1, sch_s1 = build_optimizer(model_s1, args.jaad_lr, args.last_fc_l2,
                                              args.fixed_lr, visual_l2=args.visual_l2)

            _, jaad_ckpt = run_training_stage(
                "S1_JAAD", model_s1, device, criterion_s1,
                jaad_train_loader, jaad_val_loader, jaad_test_loader,
                opt_s1, sch_s1, scaler,
                args.jaad_epochs, use_amp, stage1_dir, csv_path, args,
                early_stop_patience=args.early_stopping_patience,
            )

        # Stage 2: PIE fine-tune
        stage2_dir = os.path.join(args.save_dir, "stage2_pie_transfer")
        pie_train_ds, pie_train_loader, pie_val_loader, pie_test_loader = \
            make_loaders(args.pie_root, "pie", args.seq_len,
                         args.pie_batch_size, args.num_workers, strict_len)

        pie_wpos = args.wpos if args.wpos > 0 else \
            compute_pos_weight_from_npz(pie_train_ds.files)[0]

        # Fix A: make_model()
        model_s2 = make_model(args, device)
        model_s2.load_state_dict(jaad_ckpt["model"])
        criterion_s2 = MultiTaskLoss(args.aux_weight, args.entropy_weight, pie_wpos)
        opt_s2, sch_s2 = build_optimizer(model_s2, args.pie_lr, args.last_fc_l2,
                                          args.fixed_lr, visual_l2=args.visual_l2)

        transfer_test_m, transfer_best_state = run_training_stage(
            "S2_PIE_transfer", model_s2, device, criterion_s2,
            pie_train_loader, pie_val_loader, pie_test_loader,
            opt_s2, sch_s2, scaler,
            args.pie_epochs, use_amp, stage2_dir, csv_path, args,
            early_stop_patience=args.early_stopping_patience,
        )

    else:
        pie_train_ds, pie_train_loader, pie_val_loader, pie_test_loader = \
            make_loaders(args.pie_root, "pie", args.seq_len,
                         args.pie_batch_size, args.num_workers, strict_len)
        pie_wpos = args.wpos if args.wpos > 0 else \
            compute_pos_weight_from_npz(pie_train_ds.files)[0]

    # ------------------------------------------------------------------ S3
    if not args.skip_baseline:
        stage3_dir = os.path.join(args.save_dir, "stage3_pie_baseline")

        # Fix A: make_model()
        model_s3     = make_model(args, device)
        criterion_s3 = MultiTaskLoss(args.aux_weight, args.entropy_weight, pie_wpos)
        opt_s3, sch_s3 = build_optimizer(model_s3, args.baseline_lr, args.last_fc_l2,
                                          args.fixed_lr, visual_l2=args.visual_l2)

        baseline_test_m, _ = run_training_stage(
            "S3_PIE_baseline", model_s3, device, criterion_s3,
            pie_train_loader, pie_val_loader, pie_test_loader,
            opt_s3, sch_s3, scaler,
            args.baseline_epochs, use_amp, stage3_dir, csv_path, args,
            early_stop_patience=args.early_stopping_patience,
        )


if __name__ == "__main__":
    main()