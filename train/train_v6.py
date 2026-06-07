# train_v6.py
"""
Two-stage Transfer Learning: JAAD pretrain -> PIE fine-tune
Includes PIE-only baseline for direct comparison.

V6 update (factorized space-time global branch):
  - Uses PIPNetAlphaV6Final:
        * Spatial Transformer per-frame + funnel pooling (64 -> 16 patches)
        * Enriched pedestrian token (pose center + bbox geometry)
        * GRU temporal encoder (replaces in-Transformer temporal attention)
        * Trajectory auxiliary decoder (predicts bbox displacement from frame 0)
  - MultiTaskLossV6 adds an L2/Smooth-L1 trajectory regression term
    (controlled by --traj_weight).
  - build_branch_lr_optimizer_v6 splits the global branch:
        * spatial Transformer + stem  -> transformer LR
        * temporal GRU + traj decoder  -> cnn_gru LR (more stable)
  - New CLI flags: --global_funnel_grid, --global_gru_dropout, --traj_weight.
  - --global_n_layers is removed (V6 uses a fixed 2 spatial blocks).

Checkpoint selection remains clean:
  - Best epoch is selected by validation AUC.
  - Test set is evaluated only after loading that best-validation checkpoint.
"""

import os
import sys
import csv
import argparse
import copy
import random
from contextlib import nullcontext
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, average_precision_score

from data.pie import PIESeqDataset
from models.pipnet_alpha_v6_final import PIPNetAlphaV6Final


# Backward-compatible fallback for older PyTorch versions.
try:
    from torch.cuda.amp import autocast as cuda_autocast
    from torch.cuda.amp import GradScaler as CudaGradScaler
except Exception:  # pragma: no cover
    cuda_autocast = None
    CudaGradScaler = None


# ============================================================
# AMP HELPERS
# ============================================================

def amp_autocast(device: torch.device, enabled: bool):
    """Modern torch.amp autocast with fallback to torch.cuda.amp."""
    if not enabled:
        return nullcontext()

    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=True)

    if device.type == "cuda" and cuda_autocast is not None:
        return cuda_autocast(enabled=True)

    return nullcontext()


def make_grad_scaler(use_amp: bool):
    """Modern torch.amp GradScaler with fallback to torch.cuda.amp."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=use_amp)
        except TypeError:
            return torch.amp.GradScaler(enabled=use_amp)

    if CudaGradScaler is not None:
        return CudaGradScaler(enabled=use_amp)

    return None


# ============================================================
# LOSSES
# ============================================================

class FocalLoss(nn.Module):
    """
    Binary Focal Loss for logits.

    alpha is applied as class balancing:
      alpha_t = alpha for positive labels, 1-alpha for negative labels.
    gamma controls how strongly easy examples are down-weighted.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = inputs.squeeze(-1)
        targets = targets.float().view_as(inputs)

        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        probs = torch.sigmoid(inputs)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)

        focal_loss = alpha_t * torch.pow((1.0 - pt).clamp(min=1e-8), self.gamma) * bce_loss

        if self.reduction == "sum":
            return focal_loss.sum()
        if self.reduction == "none":
            return focal_loss
        return focal_loss.mean()


class MultiTaskLossV6(nn.Module):
    """
    V6 multi-task loss.

    Adds a trajectory regression term (Smooth-L1) on `aux_trajectory`
    versus the bbox displacement from frame 0. Controlled by traj_weight;
    set to 0 to disable (recovers the V5 behaviour).
    """
    def __init__(
        self,
        aux_weight: float = 0.1,
        entropy_weight: float = 0.05,
        traj_weight: float = 0.05,
        pos_weight: float = 1.0,
        loss_type: str = "focal",
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.aux_weight = aux_weight
        self.entropy_weight = entropy_weight
        self.traj_weight = traj_weight
        self.loss_type = loss_type.lower()

        if self.loss_type == "focal":
            self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            self.bce = None
        elif self.loss_type == "bce":
            self.criterion = None
            self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        else:
            raise ValueError(f"Unknown loss_type={loss_type}. Use 'focal' or 'bce'.")

    def _cls_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.float()
        logits = logits.squeeze(-1)

        if self.loss_type == "focal":
            return self.criterion(logits, labels)

        device = labels.device
        self.bce.pos_weight = self.bce.pos_weight.to(device)
        return self.bce(logits, labels)

    def forward(self, outputs: dict, labels: torch.Tensor, traj_target: torch.Tensor = None):
        device = labels.device
        labels = labels.float()
        losses = {}

        # ── Main loss ─────────────────────────────────────────────────────
        main_loss = self._cls_loss(outputs["logit"], labels)
        losses["main"] = main_loss

        # ── Auxiliary branch losses ───────────────────────────────────────
        if "aux_kin" in outputs:
            aux_kin_loss = self._cls_loss(outputs["aux_kin"], labels)
            aux_local_loss = self._cls_loss(outputs["aux_local"], labels)
            aux_global_loss = self._cls_loss(outputs["aux_global"], labels)
            aux_loss = self.aux_weight * (aux_kin_loss + aux_local_loss + aux_global_loss)
            losses["aux"] = aux_loss
            losses["aux_kin_loss"] = aux_kin_loss.item()
            losses["aux_local_loss"] = aux_local_loss.item()
            losses["aux_global_loss"] = aux_global_loss.item()
        else:
            aux_loss = torch.tensor(0.0, device=device)
            losses["aux"] = aux_loss

        # ── NEW V6: Trajectory regression loss ────────────────────────────
        if (self.traj_weight > 0
                and "aux_trajectory" in outputs
                and traj_target is not None):
            traj_pred = outputs["aux_trajectory"]
            traj_loss_raw = F.smooth_l1_loss(traj_pred, traj_target)
            traj_loss = self.traj_weight * traj_loss_raw
            losses["traj"] = traj_loss
            losses["traj_raw"] = traj_loss_raw.item()
        else:
            traj_loss = torch.tensor(0.0, device=device)
            losses["traj"] = traj_loss

        # ── Entropy regularisation on both attention distributions ────────
        entropy_loss = torch.tensor(0.0, device=device)

        # (a) 2-way final attention: visual vs kin
        if "modality_weights" in outputs:
            w = outputs["modality_weights"]                       # (B, 2)
            ent = -(w * torch.log(w + 1e-8)).sum(dim=-1).mean()
            ent_norm = ent / torch.log(
                torch.tensor(w.shape[-1], device=device, dtype=ent.dtype)
            )
            term = -self.entropy_weight * ent_norm
            entropy_loss = entropy_loss + term
            losses["entropy_modality"] = term.item()
            losses["entropy_modality_raw"] = ent.item()
            losses["entropy_modality_norm"] = ent_norm.item()

        # (b) T-token visual fusion attention  (T=10 for V6)
        if "visual_fuse_weights" in outputs:
            w2 = outputs["visual_fuse_weights"]                   # (B, T)
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
        losses["total"] = main_loss + aux_loss + traj_loss + entropy_loss
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


def make_model(args, device):
    """Single factory — all stages use the same config."""
    return PIPNetAlphaV6Final(
        dropout_p=args.dropout_p,
        local_dropout_p=args.local_dropout_p,
        global_patch_grid=args.global_patch_grid,
        global_funnel_grid=args.global_funnel_grid,
        global_d_model=args.global_d_model,
        global_n_heads=args.global_n_heads,
        global_ff_dim=args.global_ff_dim,
        global_tf_dropout=args.global_tf_dropout,
        global_stem_dropout=args.global_stem_dropout,
        global_gru_dropout=args.global_gru_dropout,
    ).to(device)


def make_criterion(args, pos_weight: float) -> MultiTaskLossV6:
    return MultiTaskLossV6(
        aux_weight=args.aux_weight,
        entropy_weight=args.entropy_weight,
        traj_weight=args.traj_weight,
        pos_weight=pos_weight,
        loss_type=args.loss_type,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
    )


def make_loaders(data_root, dataset_prefix, seq_len, batch_size,
                 num_workers=4, strict_len=True):
    speed_stats_path = f"/workspace/project/data/{dataset_prefix}_speed_stats_splits.json"
    motion_stats_path = f"/workspace/project/data/{dataset_prefix}_motion_stats_splits.json"

    common_args = dict(
        seq_len=seq_len, strict_len=strict_len,
        speed_norm="minmax", speed_stats_path=speed_stats_path, speed_scope="global",
        motion_norm="p99abs", motion_stats_path=motion_stats_path, motion_scope="global",
        motion_clip=1.0,
    )

    train_ds = PIESeqDataset(data_root, split="train", mode="train", **common_args)
    val_ds = PIESeqDataset(data_root, split="val", mode="eval", **common_args)
    test_ds = PIESeqDataset(data_root, split="test", mode="eval", **common_args)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_ds, train_loader, val_loader, test_loader


def safe_auc(labels, probs):
    try:
        if len(np.unique(labels)) > 1:
            return roc_auc_score(labels, probs)
    except Exception:
        pass
    return float("nan")

def safe_pr_auc(labels, probs):
    try:
        if len(np.unique(labels)) > 1:
            return average_precision_score(labels, probs)
    except Exception:
        pass
    return float("nan")

@torch.no_grad()
def compute_metrics(labels_np, probs_np):
    preds_bin = (probs_np >= 0.5).astype(np.int32)
    return {
        "auc": safe_auc(labels_np, probs_np),
        "pr_auc": safe_pr_auc(labels_np, probs_np),  # <-- NEW
        "acc": float((preds_bin == labels_np).mean()),
        "f1": f1_score(labels_np, preds_bin, zero_division=0),
        "precision": precision_score(labels_np, preds_bin, zero_division=0),
        "recall": recall_score(labels_np, preds_bin, zero_division=0),
    }


# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def _move_batch(batch, device):
    for k in ["bbox", "pose", "speed", "local_cnn", "local_motion",
              "sem_labels", "cat_depth", "label"]:
        if k in batch and torch.is_tensor(batch[k]):
            batch[k] = batch[k].to(device, non_blocking=True)
    return batch


def _traj_target_from_batch(batch):
    """V6: trajectory target = bbox displacement from frame 0 -> (B, T, 4)."""
    bbox = batch["bbox"]
    return bbox - bbox[:, 0:1, :]


def run_train_epoch(model, loader, device, criterion, optimizer,
                    use_amp=False, scaler=None, grad_clip_norm=5.0):
    model.train()

    all_labels, all_probs = [], []
    all_probs_kin, all_probs_local, all_probs_global, all_probs_visual = [], [], [], []
    running_loss = running_aux_kin = running_aux_local = running_aux_global = 0.0
    running_traj = 0.0
    total = 0
    modality_weights_sum = torch.zeros(2)
    weight_count = 0
    visual_fuse_n_tokens = 0

    pbar = tqdm(loader, desc="Train")
    for batch in pbar:
        batch = _move_batch(batch, device)
        labels = batch["label"].float()
        traj_target = _traj_target_from_batch(batch)
        optimizer.zero_grad(set_to_none=True)

        with amp_autocast(device, enabled=use_amp):
            outputs = model(batch, return_aux=True)
            losses = criterion(outputs, labels, traj_target=traj_target)

        total_loss = losses["total"]

        if use_amp and scaler is not None:
            scaler.scale(total_loss).backward()

            if grad_clip_norm and grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()

            if grad_clip_norm and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

            optimizer.step()

        bs = labels.size(0)
        running_loss += total_loss.item() * bs
        total += bs

        if "aux_kin_loss" in losses:
            running_aux_kin += losses["aux_kin_loss"] * bs
            running_aux_local += losses["aux_local_loss"] * bs
            running_aux_global += losses["aux_global_loss"] * bs
        if "traj_raw" in losses:
            running_traj += losses["traj_raw"] * bs

        probs = torch.sigmoid(outputs["logit"].squeeze(-1)).detach()
        all_labels.append(labels.detach().cpu().numpy())
        all_probs.append(probs.cpu().numpy())

        if "aux_kin" in outputs:
            all_probs_kin.append(
                torch.sigmoid(outputs["aux_kin"].squeeze(-1)).detach().cpu().numpy())
            all_probs_local.append(
                torch.sigmoid(outputs["aux_local"].squeeze(-1)).detach().cpu().numpy())
            all_probs_global.append(
                torch.sigmoid(outputs["aux_global"].squeeze(-1)).detach().cpu().numpy())

            # Visual fusion quality diagnostic: run aux_local on z_visual
            if "z_visual" in outputs and hasattr(model, "aux_local"):
                with torch.no_grad():
                    vl = model.aux_local(outputs["z_visual"].detach()).squeeze(-1)
                    all_probs_visual.append(torch.sigmoid(vl).cpu().numpy())

        if "modality_weights" in outputs:
            modality_weights_sum += outputs["modality_weights"].detach().cpu().mean(0)
            weight_count += 1

        if "visual_fuse_weights" in outputs and visual_fuse_n_tokens == 0:
            visual_fuse_n_tokens = outputs["visual_fuse_weights"].shape[1]

        pbar.set_postfix({"loss": f"{total_loss.item():.3f}"})

    labels_np = np.concatenate(all_labels)
    probs_np = np.concatenate(all_probs)
    metrics = compute_metrics(labels_np, probs_np)
    metrics["loss"] = running_loss / max(total, 1)

    if all_probs_kin:
        metrics["auc_kin"] = safe_auc(labels_np, np.concatenate(all_probs_kin))
        metrics["auc_local"] = safe_auc(labels_np, np.concatenate(all_probs_local))
        metrics["auc_global"] = safe_auc(labels_np, np.concatenate(all_probs_global))
        metrics["loss_kin"] = running_aux_kin / max(total, 1)
        metrics["loss_local"] = running_aux_local / max(total, 1)
        metrics["loss_global"] = running_aux_global / max(total, 1)

    if running_traj > 0:
        metrics["traj_raw"] = running_traj / max(total, 1)

    if all_probs_visual:
        metrics["auc_visual"] = safe_auc(labels_np, np.concatenate(all_probs_visual))

    if weight_count > 0:
        avg = modality_weights_sum / weight_count
        metrics["w_visual"] = avg[0].item()
        metrics["w_kin"] = avg[1].item()

    if visual_fuse_n_tokens > 0:
        metrics["visual_fuse_tokens"] = visual_fuse_n_tokens

    return metrics


@torch.no_grad()
def run_eval_epoch(model, loader, device, criterion, use_amp=False):
    model.eval()

    all_labels, all_probs = [], []
    all_probs_kin, all_probs_local, all_probs_global, all_probs_visual = [], [], [], []
    running_loss = running_aux_kin = running_aux_local = running_aux_global = 0.0
    running_traj = 0.0
    total = 0
    modality_weights_sum = torch.zeros(2)
    weight_count = 0
    visual_fuse_n_tokens = 0

    for batch in tqdm(loader, desc="Eval "):
        batch = _move_batch(batch, device)
        labels = batch["label"].float()
        traj_target = _traj_target_from_batch(batch)

        with amp_autocast(device, enabled=use_amp):
            outputs = model(batch, return_aux=True)
            losses = criterion(outputs, labels, traj_target=traj_target)

        bs = labels.size(0)
        running_loss += losses["total"].item() * bs
        total += bs

        if "aux_kin_loss" in losses:
            running_aux_kin += losses["aux_kin_loss"] * bs
            running_aux_local += losses["aux_local_loss"] * bs
            running_aux_global += losses["aux_global_loss"] * bs
        if "traj_raw" in losses:
            running_traj += losses["traj_raw"] * bs

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
                vl = model.aux_local(outputs["z_visual"]).squeeze(-1)
                all_probs_visual.append(torch.sigmoid(vl).cpu().numpy())

        if "modality_weights" in outputs:
            modality_weights_sum += outputs["modality_weights"].cpu().mean(0)
            weight_count += 1

        if "visual_fuse_weights" in outputs and visual_fuse_n_tokens == 0:
            visual_fuse_n_tokens = outputs["visual_fuse_weights"].shape[1]

    labels_np = np.concatenate(all_labels)
    probs_np = np.concatenate(all_probs)
    metrics = compute_metrics(labels_np, probs_np)
    metrics["loss"] = running_loss / max(total, 1)

    if all_probs_kin:
        metrics["auc_kin"] = safe_auc(labels_np, np.concatenate(all_probs_kin))
        metrics["auc_local"] = safe_auc(labels_np, np.concatenate(all_probs_local))
        metrics["auc_global"] = safe_auc(labels_np, np.concatenate(all_probs_global))
        metrics["loss_kin"] = running_aux_kin / max(total, 1)
        metrics["loss_local"] = running_aux_local / max(total, 1)
        metrics["loss_global"] = running_aux_global / max(total, 1)

    if running_traj > 0:
        metrics["traj_raw"] = running_traj / max(total, 1)

    if all_probs_visual:
        metrics["auc_visual"] = safe_auc(labels_np, np.concatenate(all_probs_visual))

    if weight_count > 0:
        avg = modality_weights_sum / weight_count
        metrics["w_visual"] = avg[0].item()
        metrics["w_kin"] = avg[1].item()

    if visual_fuse_n_tokens > 0:
        metrics["visual_fuse_tokens"] = visual_fuse_n_tokens

    return metrics


def print_metrics(metrics, prefix=""):
    print(
        f"{prefix} | loss: {metrics['loss']:.4f} | "
        f"auc: {metrics['auc']:.3f} | "
        f"pr_auc: {metrics.get('pr_auc', float('nan')):.3f} | "
        f"f1: {metrics['f1']:.3f} | "
        f"acc: {metrics['acc']:.3f}"
    )
    if "auc_kin" in metrics:
        print(
            f"{prefix} | Branch AUC: "
            f"kin={metrics['auc_kin']:.3f}  "
            f"local={metrics['auc_local']:.3f}  "
            f"global(factorized)={metrics['auc_global']:.3f}  "
            f"visual={metrics.get('auc_visual', float('nan')):.3f}"
        )
    if "traj_raw" in metrics:
        print(f"{prefix} | Trajectory Smooth-L1: {metrics['traj_raw']:.4f}")
    if "w_visual" in metrics:
        print(
            f"{prefix} | Final Attn: "
            f"visual={metrics['w_visual']:.3f}  "
            f"kin={metrics['w_kin']:.3f}"
        )
    if "visual_fuse_tokens" in metrics:
        print(f"{prefix} | Visual fuse tokens (T): {metrics['visual_fuse_tokens']}")


# ============================================================
# CSV LOGGER
# ============================================================

CSV_COLUMNS = [
    "stage", "epoch", "split",
    "loss", "auc", "pr_auc", "f1", "acc", "precision", "recall",
    "auc_kin", "auc_local", "auc_global", "auc_visual",
    "loss_kin", "loss_local", "loss_global", "traj_raw",
    "w_visual", "w_kin", "visual_fuse_tokens",
    "total_params", "trainable_params",
    # dataset paths
    "jaad_root", "pie_root",
    # stage flags
    "skip_transfer", "skip_baseline",
    # JAAD stage
    "jaad_epochs", "jaad_lr", "jaad_batch_size",
    # PIE fine-tune stage
    "pie_epochs", "pie_lr", "pie_batch_size",
    # baseline stage
    "baseline_epochs", "baseline_lr",
    # loss / regularisation
    "loss_type", "focal_alpha", "focal_gamma",
    "aux_weight", "entropy_weight", "traj_weight", "last_fc_l2", "visual_l2",
    # optimizer dynamics
    "optimizer_mode", "cnn_gru_lr_mult", "transformer_lr_mult", "fusion_lr_mult",
    "transformer_wd_mult", "grad_clip_norm",
    # dropout
    "dropout_p", "local_dropout_p",
    # V6 global-branch hyper-parameters
    "global_patch_grid", "global_funnel_grid", "global_d_model", "global_n_heads",
    "global_ff_dim", "global_tf_dropout", "global_stem_dropout", "global_gru_dropout",
    # misc
    "early_stopping_patience", "select_metric", "amp", "seed", "save_dir", "command",
    "lr", "lr_group_0", "lr_group_1", "lr_group_2", "timestamp",
]


def init_csv(path):
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(CSV_COLUMNS)


def log_csv(path, stage, epoch, split, metrics, args,
            total_params, trainable_params, lr=None, group_lrs=None):

    def fmt(v):
        return f"{v:.6f}" if isinstance(v, float) else str(v)

    group_lrs = list(group_lrs or [])
    while len(group_lrs) < 3:
        group_lrs.append("")

    row = [
        stage, epoch, split,
        fmt(metrics.get("loss", 0)),
        fmt(metrics.get("auc", 0)),
        fmt(metrics.get("pr_auc", 0)),     # <-- NEW
        fmt(metrics.get("f1", 0)),
        fmt(metrics.get("acc", 0)),
        fmt(metrics.get("precision", 0)),
        fmt(metrics.get("recall", 0)),
        fmt(metrics["auc_kin"]) if "auc_kin" in metrics else "",
        fmt(metrics["auc_local"]) if "auc_local" in metrics else "",
        fmt(metrics["auc_global"]) if "auc_global" in metrics else "",
        fmt(metrics["auc_visual"]) if "auc_visual" in metrics else "",
        fmt(metrics["loss_kin"]) if "loss_kin" in metrics else "",
        fmt(metrics["loss_local"]) if "loss_local" in metrics else "",
        fmt(metrics["loss_global"]) if "loss_global" in metrics else "",
        fmt(metrics["traj_raw"]) if "traj_raw" in metrics else "",
        fmt(metrics["w_visual"]) if "w_visual" in metrics else "",
        fmt(metrics["w_kin"]) if "w_kin" in metrics else "",
        metrics.get("visual_fuse_tokens", ""),
        total_params, trainable_params,
        # dataset paths
        args.jaad_root, args.pie_root,
        # flags
        args.skip_transfer, args.skip_baseline,
        # stages
        args.jaad_epochs, args.jaad_lr, args.jaad_batch_size,
        args.pie_epochs, args.pie_lr, args.pie_batch_size,
        args.baseline_epochs, args.baseline_lr,
        # loss / reg
        args.loss_type, args.focal_alpha, args.focal_gamma,
        args.aux_weight, args.entropy_weight, args.traj_weight,
        args.last_fc_l2, args.visual_l2,
        # optimizer dynamics
        args.optimizer_mode, args.cnn_gru_lr_mult, args.transformer_lr_mult,
        args.fusion_lr_mult, args.transformer_wd_mult, args.grad_clip_norm,
        # dropout
        args.dropout_p, args.local_dropout_p,
        # V6 global branch
        args.global_patch_grid, args.global_funnel_grid, args.global_d_model,
        args.global_n_heads, args.global_ff_dim,
        args.global_tf_dropout, args.global_stem_dropout, args.global_gru_dropout,
        # misc
        args.early_stopping_patience, args.select_metric, args.amp, args.seed, args.save_dir,
        getattr(args, "command", ""),
        f"{lr:.2e}" if lr is not None else "",
        f"{group_lrs[0]:.2e}" if isinstance(group_lrs[0], float) else group_lrs[0],
        f"{group_lrs[1]:.2e}" if isinstance(group_lrs[1], float) else group_lrs[1],
        f"{group_lrs[2]:.2e}" if isinstance(group_lrs[2], float) else group_lrs[2],
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

    # Per-module breakdown
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        print(f"    {name:<30s}: {n:>10,}")
    print("=" * 80)

    # Which validation metric drives checkpoint selection + LR scheduling.
    # Both AUC and F1 are "higher is better", so ReduceLROnPlateau(mode="max")
    # is correct either way. NOTE: F1 depends on the 0.5 threshold, so it can
    # be noisier epoch-to-epoch than (threshold-free) AUC.
    select_metric = getattr(args, "select_metric", "f1")
    best_val_score = -1.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        print(f"\n{'=' * 80}")
        print(f"[{stage_name}] Epoch {epoch}/{epochs}")
        print(f"{'=' * 80}")

        group_lrs = [pg["lr"] for pg in optimizer.param_groups]
        current_lr = group_lrs[0] if group_lrs else None

        train_m = run_train_epoch(
            model, train_loader, device, criterion, optimizer,
            use_amp=use_amp, scaler=scaler, grad_clip_norm=args.grad_clip_norm,
        )
        print_metrics(train_m, f"[{stage_name}] Train")
        log_csv(csv_path, stage_name, epoch, "train", train_m, args,
                total_params, trainable_params, lr=current_lr, group_lrs=group_lrs)

        val_m = run_eval_epoch(model, val_loader, device, criterion, use_amp)
        print_metrics(val_m, f"[{stage_name}] Val  ")
        log_csv(csv_path, stage_name, epoch, "val", val_m, args,
                total_params, trainable_params, lr=current_lr, group_lrs=group_lrs)

        if scheduler is not None:
            scheduler.step(val_m[select_metric])

        if val_m[select_metric] > best_val_score:
            best_val_score = val_m[select_metric]
            best_state = {
                "epoch": epoch,
                "stage": stage_name,
                "select_metric": select_metric,
                "model": copy.deepcopy(model.state_dict()),
                "optimizer": copy.deepcopy(optimizer.state_dict()),
                "val_metrics": val_m,
            }
            torch.save(best_state, os.path.join(save_dir, "best_model.pth"))
            print(f"  * New best model (Val {select_metric.upper()}: "
                  f"{best_val_score:.4f} | Val AUC: {val_m['auc']:.4f} | "
                  f"Val F1: {val_m['f1']:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  ! No improvement for {patience_counter} epoch(s).")

        if patience_counter >= early_stop_patience:
            print(f"\n  Early stopping triggered after {early_stop_patience} stagnant epochs.")
            break

    test_m = {}
    if best_state:
        print(f"\n[{stage_name}] Loading best model (Epoch {best_state['epoch']})...")
        model.load_state_dict(best_state["model"])
        test_m = run_eval_epoch(model, test_loader, device, criterion, use_amp)
        print_metrics(test_m, f"[{stage_name}] Test ")
        group_lrs = [pg["lr"] for pg in optimizer.param_groups]
        current_lr = group_lrs[0] if group_lrs else None
        log_csv(csv_path, stage_name, best_state["epoch"], "test", test_m, args,
                total_params, trainable_params, lr=current_lr, group_lrs=group_lrs)

    return test_m, best_state


# ============================================================
# OPTIMIZER + SCHEDULER
# ============================================================

def _params_by_prefix(model: nn.Module, prefixes):
    """Collect trainable params whose names start with any prefix, no duplicates."""
    out, ids = [], set()
    prefixes = tuple(prefixes)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(name == pref or name.startswith(pref + ".") for pref in prefixes):
            if id(p) not in ids:
                out.append(p)
                ids.add(id(p))
    return out, ids


def build_branch_lr_optimizer(model, lr, last_fc_l2, visual_l2, args):
    """
    V6 branch-specific AdamW.

    Param groups:
      cnn_gru_params      — kinematic + local CNN/GRU branch
                            + global temporal GRU + global trajectory decoder
      transformer_params  — global spatial Transformer blocks + stem +
                            positional encodings + ped/pose/bbox projections
                            + sem/depth downsamplers
      fusion_params       — attention, norms, aux heads, final classifier
    """
    # GRU + trajectory decoder live inside global_branch but train like the
    # GRU/MLP family, so they go in the cnn_gru group.
    cnn_gru_params, cnn_ids = _params_by_prefix(model, [
        "kinematic_branch",
        "local_branch",
        "global_branch.temporal_gru",
        "global_branch.traj_decoder",
    ])

    # Everything else inside global_branch is the spatial Transformer stack.
    transformer_params, _ = _params_by_prefix(model, ["global_branch"])
    transformer_params = [p for p in transformer_params if id(p) not in cnn_ids]
    transformer_ids = {id(p) for p in transformer_params}

    fusion_prefixes = [
        "kin_attn", "local_attn", "global_attn", "visual_fuse_attn", "final_attn",
        "fc_out", "aux_kin", "aux_local", "aux_global",
        "norm_kin", "norm_local", "norm_global", "norm_visual",
    ]
    fusion_params, fusion_ids = _params_by_prefix(model, fusion_prefixes)

    assigned = cnn_ids | transformer_ids | fusion_ids
    leftovers = [p for _, p in model.named_parameters()
                 if p.requires_grad and id(p) not in assigned]
    fusion_params.extend(leftovers)

    cnn_lr = lr * args.cnn_gru_lr_mult
    tf_lr = lr * args.transformer_lr_mult
    fusion_lr = lr * args.fusion_lr_mult
    tf_wd = visual_l2 * args.transformer_wd_mult

    print("\nOptimizer parameter groups: V6 branch-specific AdamW")
    print(f"  cnn_gru (+ temporal_gru + traj_decoder) lr={cnn_lr:.2e}  wd={visual_l2}: "
          f"{sum(p.numel() for p in cnn_gru_params):>10,}")
    print(f"  transformer (spatial blocks + stem)     lr={tf_lr:.2e}  wd={tf_wd}: "
          f"{sum(p.numel() for p in transformer_params):>10,}")
    print(f"  fusion/heads                            lr={fusion_lr:.2e}  wd={last_fc_l2}: "
          f"{sum(p.numel() for p in fusion_params):>8,}")
    if leftovers:
        print(f"  note: {sum(p.numel() for p in leftovers):,} unclassified "
              f"parameter(s) were added to fusion/heads")

    optimizer = torch.optim.AdamW(
        [
            {"params": cnn_gru_params, "lr": cnn_lr, "weight_decay": visual_l2},
            {"params": transformer_params, "lr": tf_lr, "weight_decay": tf_wd},
            {"params": fusion_params, "lr": fusion_lr, "weight_decay": last_fc_l2},
        ]
    )
    return optimizer


def build_targeted_optimizer(model, lr, last_fc_l2, visual_l2):
    """
    Original targeted AdamW weight decay:
      base_params       — kinematic + kin_attn + final_attn: no decay
      visual_params     — local/global branches + visual fusion: visual_l2
      final_fc_params   — fc_out: last_fc_l2
    """
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

    print("\nOptimizer parameter groups: targeted AdamW")
    print(f"  base / no_decay:       {sum(p.numel() for p in base_params):>10,}")
    print(f"  visual / wd={visual_l2}: {sum(p.numel() for p in visual_params):>10,}")
    print(f"  final_fc / wd={last_fc_l2}: {sum(p.numel() for p in final_fc_params):>8,}")

    optimizer = torch.optim.AdamW(
        [
            {"params": base_params, "lr": lr, "weight_decay": 0.0},
            {"params": visual_params, "lr": lr, "weight_decay": visual_l2},
            {"params": final_fc_params, "lr": lr, "weight_decay": last_fc_l2},
        ]
    )
    return optimizer


def build_optimizer(model, lr, last_fc_l2, fixed_lr=False, visual_l2=0.0, args=None):
    if args is not None and args.optimizer_mode == "branch_lr":
        optimizer = build_branch_lr_optimizer(model, lr, last_fc_l2, visual_l2, args)
    else:
        optimizer = build_targeted_optimizer(model, lr, last_fc_l2, visual_l2)

    scheduler = None
    if not fixed_lr:
        # patience=3: LR halves after 3 stagnant val-AUC epochs.
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
    parser = argparse.ArgumentParser(
        description="PIPNet-Alpha V6 — Factorized Space-Time Global Branch "
                    "+ Focal Loss + Branch LR + Trajectory Aux"
    )

    # ── Data ────────────────────────────────────────────────────────────────
    parser.add_argument("--jaad_root", type=str, required=True,
                        help="Path to JAAD_PREP_OUT root")
    parser.add_argument("--pie_root", type=str, required=True,
                        help="Path to PIE_PREP_OUT root")

    # ── Stage control ────────────────────────────────────────────────────────
    parser.add_argument("--jaad_ckpt", type=str, default=None,
                        help="Skip S1 and use this JAAD checkpoint directly")
    parser.add_argument("--skip_transfer", action="store_true",
                        help="Skip S1+S2 entirely")
    parser.add_argument("--skip_baseline", action="store_true",
                        help="Skip S3 PIE-only baseline")

    # ── Sequence ─────────────────────────────────────────────────────────────
    parser.add_argument("--seq_len", type=int, default=10)
    parser.add_argument("--no_strict_len", action="store_true")

    # ── Model dropout ────────────────────────────────────────────────────────
    parser.add_argument("--dropout_p", type=float, default=0.5,
                        help="Output / attention / fc dropout")
    parser.add_argument("--local_dropout_p", type=float, default=0.3,
                        help="LocalVisualBranch inter-GRU dropout")

    # ── V6: global-branch hyper-parameters ───────────────────────────────────
    parser.add_argument("--global_patch_grid", type=int, default=8,
                        help="Patch grid side P (P×P tokens/frame before funnel). "
                             "target_size must be divisible by P.")
    parser.add_argument("--global_funnel_grid", type=int, default=4,
                        help="Patch grid side after funnel pooling "
                             "(4 → 16 tokens). Must divide patch_grid.")
    parser.add_argument("--global_d_model", type=int, default=128,
                        help="Transformer token dimension (must be even)")
    parser.add_argument("--global_n_heads", type=int, default=4,
                        help="Number of self-attention heads (d_model % n_heads == 0)")
    parser.add_argument("--global_ff_dim", type=int, default=256,
                        help="Feedforward inner dimension")
    parser.add_argument("--global_tf_dropout", type=float, default=0.1,
                        help="Transformer internal dropout (attention + ff)")
    parser.add_argument("--global_stem_dropout", type=float, default=0.1,
                        help="Dropout after CNN stem projection")
    parser.add_argument("--global_gru_dropout", type=float, default=0.1,
                        help="Dropout after the temporal GRU")

    # ── JAAD stage ───────────────────────────────────────────────────────────
    parser.add_argument("--jaad_epochs", type=int, default=80)
    parser.add_argument("--jaad_lr", type=float, default=5e-5)
    parser.add_argument("--jaad_batch_size", type=int, default=10)

    # ── PIE fine-tune stage ──────────────────────────────────────────────────
    parser.add_argument("--pie_epochs", type=int, default=50)
    parser.add_argument("--pie_lr", type=float, default=1e-5)
    parser.add_argument("--pie_batch_size", type=int, default=10)

    # ── PIE baseline stage ───────────────────────────────────────────────────
    parser.add_argument("--baseline_epochs", type=int, default=50)
    parser.add_argument("--baseline_lr", type=float, default=5e-5)

    # ── Loss / regularisation ────────────────────────────────────────────────
    parser.add_argument("--loss_type", choices=["bce", "focal"], default="focal",
                        help="Classification loss for main and aux heads.")
    parser.add_argument("--focal_alpha", type=float, default=0.25,
                        help="Positive-class alpha for binary focal loss.")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Focusing gamma for focal loss.")
    parser.add_argument("--aux_weight", type=float, default=0.1)
    parser.add_argument("--entropy_weight", type=float, default=0.05)
    parser.add_argument("--traj_weight", type=float, default=0.05,
                        help="Weight on the V6 trajectory Smooth-L1 aux loss. "
                             "Set to 0 to disable.")
    parser.add_argument("--last_fc_l2", type=float, default=1e-4,
                        help="AdamW weight decay on fc_out / fusion heads")
    parser.add_argument("--visual_l2", type=float, default=0.0,
                        help="AdamW weight decay on visual / global params")
    parser.add_argument("--wpos", type=float, default=-1.0,
                        help="Manual pos_weight for BCEWithLogitsLoss (-1 = auto). "
                             "Ignored by focal loss.")

    # ── Optimizer dynamics ───────────────────────────────────────────────────
    parser.add_argument("--optimizer_mode", choices=["branch_lr", "targeted"], default="branch_lr",
                        help="branch_lr = different LR per branch; targeted = old AdamW grouping.")
    parser.add_argument("--cnn_gru_lr_mult", type=float, default=1.0,
                        help="LR multiplier for kinematic + local CNN/GRU + "
                             "global temporal GRU + traj decoder.")
    parser.add_argument("--transformer_lr_mult", type=float, default=1.5,
                        help="LR multiplier for the global spatial Transformer. "
                             "V6 is more stable than V5, so 1.5 is a good start "
                             "(V5 used 2.5).")
    parser.add_argument("--fusion_lr_mult", type=float, default=1.0,
                        help="LR multiplier for attention/fusion/aux/final heads.")
    parser.add_argument("--transformer_wd_mult", type=float, default=3.0,
                        help="Multiplier over visual_l2 for Transformer weight decay "
                             "(V5 used 5.0).")
    parser.add_argument("--grad_clip_norm", type=float, default=5.0,
                        help="Max grad-norm clipping. Use 0 to disable.")

    # ── Scheduler ────────────────────────────────────────────────────────────
    parser.add_argument("--fixed_lr", action="store_true")
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--select_metric", choices=["auc", "pr_auc", "f1"], default="f1",
                        help="Validation metric used for best-checkpoint "
                             "selection, LR scheduling, and early stopping. "
                             "All are logged regardless. pr_auc (average "
                             "precision) is threshold-free and sensitive to "
                             "the minority class — best for imbalanced data. "
                             "f1 uses a 0.5 threshold so it can be noisier.")

    # ── Runtime ──────────────────────────────────────────────────────────────
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="checkpoints_v6")

    args = parser.parse_args()
    args.command = " ".join(sys.argv)

    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and args.amp
    scaler = make_grad_scaler(use_amp)
    strict_len = not args.no_strict_len

    os.makedirs(args.save_dir, exist_ok=True)
    csv_path = os.path.join(args.save_dir, "training_log.csv")
    init_csv(csv_path)

    print("=" * 80)
    print("PIPNet-Alpha V6 — Factorized Space-Time Global Branch")
    print("Focal Loss + Branch-Specific LR + Trajectory Auxiliary Head")
    print("=" * 80)
    print(f"  Device:          {device}")
    print(f"  AMP:             {use_amp}")
    print(f"  Seed:            {args.seed}")
    print(f"  Loss:            {args.loss_type}")
    if args.loss_type == "focal":
        print(f"  Focal alpha/gamma: {args.focal_alpha} / {args.focal_gamma}")
    print(f"  Traj weight:     {args.traj_weight}")
    print(f"  Select metric:   {args.select_metric}  (best checkpoint / early stop)")
    print(f"  Optimizer mode:  {args.optimizer_mode}")
    print(f"  LR multipliers:  cnn_gru={args.cnn_gru_lr_mult}, "
          f"transformer={args.transformer_lr_mult}, fusion={args.fusion_lr_mult}")
    print(f"  Grad clip norm:  {args.grad_clip_norm}")
    print(f"  Patch grid:      {args.global_patch_grid}×{args.global_patch_grid} "
          f"→ funnel {args.global_funnel_grid}×{args.global_funnel_grid}")
    print(f"  d_model:         {args.global_d_model}")
    print(f"  n_heads:         {args.global_n_heads}")
    print(f"  ff_dim:          {args.global_ff_dim}")
    print(f"  tf_dropout:      {args.global_tf_dropout}")
    print(f"  stem_dropout:    {args.global_stem_dropout}")
    print(f"  gru_dropout:     {args.global_gru_dropout}")
    print("=" * 80)

    jaad_ckpt = None

    # ─────────────────────────────────────────────────────────── S1 + S2
    if not args.skip_transfer:
        stage1_dir = os.path.join(args.save_dir, "stage1_jaad")

        if args.jaad_ckpt:
            print(f"\n[S1] Loading existing JAAD checkpoint: {args.jaad_ckpt}")
            jaad_ckpt = torch.load(args.jaad_ckpt, map_location=device)
        else:
            # Stage 1 — JAAD pretrain
            jaad_train_ds, jaad_train_loader, jaad_val_loader, jaad_test_loader = \
                make_loaders(args.jaad_root, "jaad", args.seq_len,
                             args.jaad_batch_size, args.num_workers, strict_len)

            jaad_wpos = (
                args.wpos if args.wpos > 0
                else compute_pos_weight_from_npz(jaad_train_ds.files)[0]
            )
            print(f"\n[S1] JAAD pos_weight = {jaad_wpos:.3f} "
                  f"({'ignored by focal loss' if args.loss_type == 'focal' else 'BCE'})")

            model_s1 = make_model(args, device)
            criterion_s1 = make_criterion(args, jaad_wpos)
            opt_s1, sch_s1 = build_optimizer(
                model_s1, args.jaad_lr, args.last_fc_l2,
                args.fixed_lr, visual_l2=args.visual_l2, args=args,
            )

            _, jaad_ckpt = run_training_stage(
                "S1_JAAD", model_s1, device, criterion_s1,
                jaad_train_loader, jaad_val_loader, jaad_test_loader,
                opt_s1, sch_s1, scaler,
                args.jaad_epochs, use_amp, stage1_dir, csv_path, args,
                early_stop_patience=args.early_stopping_patience,
            )

        # Stage 2 — PIE fine-tune
        stage2_dir = os.path.join(args.save_dir, "stage2_pie_transfer")
        pie_train_ds, pie_train_loader, pie_val_loader, pie_test_loader = \
            make_loaders(args.pie_root, "pie", args.seq_len,
                         args.pie_batch_size, args.num_workers, strict_len)

        pie_wpos = (
            args.wpos if args.wpos > 0
            else compute_pos_weight_from_npz(pie_train_ds.files)[0]
        )
        print(f"\n[S2] PIE pos_weight = {pie_wpos:.3f} "
              f"({'ignored by focal loss' if args.loss_type == 'focal' else 'BCE'})")

        model_s2 = make_model(args, device)
        model_s2.load_state_dict(jaad_ckpt["model"])
        criterion_s2 = make_criterion(args, pie_wpos)
        opt_s2, sch_s2 = build_optimizer(
            model_s2, args.pie_lr, args.last_fc_l2,
            args.fixed_lr, visual_l2=args.visual_l2, args=args,
        )

        transfer_test_m, _ = run_training_stage(
            "S2_PIE_transfer", model_s2, device, criterion_s2,
            pie_train_loader, pie_val_loader, pie_test_loader,
            opt_s2, sch_s2, scaler,
            args.pie_epochs, use_amp, stage2_dir, csv_path, args,
            early_stop_patience=args.early_stopping_patience,
        )

    else:
        # Still need PIE loaders + pos_weight for S3
        pie_train_ds, pie_train_loader, pie_val_loader, pie_test_loader = \
            make_loaders(args.pie_root, "pie", args.seq_len,
                         args.pie_batch_size, args.num_workers, strict_len)
        pie_wpos = (
            args.wpos if args.wpos > 0
            else compute_pos_weight_from_npz(pie_train_ds.files)[0]
        )

    # ─────────────────────────────────────────────────────────── S3
    if not args.skip_baseline:
        stage3_dir = os.path.join(args.save_dir, "stage3_pie_baseline")
        print(f"\n[S3] PIE-only baseline — pos_weight = {pie_wpos:.3f} "
              f"({'ignored by focal loss' if args.loss_type == 'focal' else 'BCE'})")

        model_s3 = make_model(args, device)
        criterion_s3 = make_criterion(args, pie_wpos)
        opt_s3, sch_s3 = build_optimizer(
            model_s3, args.baseline_lr, args.last_fc_l2,
            args.fixed_lr, visual_l2=args.visual_l2, args=args,
        )

        _, _ = run_training_stage(
            "S3_PIE_baseline", model_s3, device, criterion_s3,
            pie_train_loader, pie_val_loader, pie_test_loader,
            opt_s3, sch_s3, scaler,
            args.baseline_epochs, use_amp, stage3_dir, csv_path, args,
            early_stop_patience=args.early_stopping_patience,
        )

    print("\n" + "=" * 80)
    print("Training complete. Results in:", args.save_dir)
    print("=" * 80)


if __name__ == "__main__":
    main()