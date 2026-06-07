#!/usr/bin/env python3
"""
Train script for PIPNetKinFormerGRUMM
=====================================

Purpose
-------
Train the kinematic-GRU anchored multimodal KinFormer model:

    bbox + pose + speed -> old KinematicBranch GRUs -> kinematic anchor token
    + optional gated local-context token
    + optional gated optical-flow token
    + optional gated segmentation patch tokens
    + optional gated categorical-depth patch tokens
    -> per-frame cross-modal Transformer
    -> read only kinematic anchor
    -> temporal attention pooling -> classifier

Why this script exists
----------------------
The original train_v6.py was written for PIPNetAlphaV6Final and its module names
(global_branch, local_branch, aux heads, trajectory decoder). This script is
cleaner and uses optimizer groups that match PIPNetKinFormerGRUMM.

Recommended progressive runs
----------------------------
A) GRU-anchor sanity check:
   no --use_* flags
B) + segmentation:
   --use_seg
C) + segmentation + categorical depth:
   --use_seg --use_depth
D) full target/local/scene multimodal:
   --use_local_context --use_local_flow --use_seg --use_depth
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.pie import PIESeqDataset
from models.pipnet_kinformer_gru_mm import PIPNetKinFormerGRUMM

try:
    from torch.cuda.amp import autocast as cuda_autocast
    from torch.cuda.amp import GradScaler as CudaGradScaler
except Exception:  # pragma: no cover
    cuda_autocast = None
    CudaGradScaler = None


# =============================================================================
# AMP helpers
# =============================================================================


def amp_autocast(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=True)
    if device.type == "cuda" and cuda_autocast is not None:
        return cuda_autocast(enabled=True)
    return nullcontext()


def make_grad_scaler(use_amp: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=use_amp)
        except TypeError:
            return torch.amp.GradScaler(enabled=use_amp)
    if CudaGradScaler is not None:
        return CudaGradScaler(enabled=use_amp)
    return None


# =============================================================================
# Losses
# =============================================================================


class FocalLoss(nn.Module):
    """Binary focal loss for logits."""

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = inputs.squeeze(-1)
        targets = targets.float().view_as(inputs)

        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        probs = torch.sigmoid(inputs)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_t * torch.pow((1.0 - pt).clamp(min=1e-8), self.gamma) * bce

        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


class KinFormerLoss(nn.Module):
    """Main classification loss plus optional gate L1 penalty from model output."""

    def __init__(
        self,
        loss_type: str = "focal",
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        pos_weight: float = 1.0,
    ):
        super().__init__()
        self.loss_type = loss_type.lower()
        if self.loss_type == "focal":
            self.cls_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            self.bce = None
        elif self.loss_type == "bce":
            self.cls_loss = None
            self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        else:
            raise ValueError(f"Unknown loss_type={loss_type}; use 'focal' or 'bce'.")

    def _classification_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.float()
        logits = logits.squeeze(-1)
        if self.loss_type == "focal":
            return self.cls_loss(logits, labels)
        self.bce.pos_weight = self.bce.pos_weight.to(labels.device)
        return self.bce(logits, labels)

    def forward(self, outputs: Dict[str, torch.Tensor], labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        main = self._classification_loss(outputs["logit"], labels)
        gate_l1 = outputs.get("gate_l1", main.new_tensor(0.0))
        total = main + gate_l1
        return {
            "main": main,
            "gate_l1": gate_l1,
            "total": total,
        }


# =============================================================================
# Metrics and utilities
# =============================================================================


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def safe_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    try:
        if len(np.unique(labels)) > 1:
            return float(roc_auc_score(labels, probs))
    except Exception:
        pass
    return float("nan")


def safe_pr_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    try:
        if len(np.unique(labels)) > 1:
            return float(average_precision_score(labels, probs))
    except Exception:
        pass
    return float("nan")


def compute_metrics(labels_np: np.ndarray, probs_np: np.ndarray) -> Dict[str, float]:
    preds = (probs_np >= 0.5).astype(np.int32)
    return {
        "auc": safe_auc(labels_np, probs_np),
        "pr_auc": safe_pr_auc(labels_np, probs_np),
        "acc": float((preds == labels_np).mean()),
        "f1": float(f1_score(labels_np, preds, zero_division=0)),
        "precision": float(precision_score(labels_np, preds, zero_division=0)),
        "recall": float(recall_score(labels_np, preds, zero_division=0)),
    }


def compute_pos_weight_from_npz(files: Iterable[str]) -> Tuple[float, int, int]:
    pos, neg = 0, 0
    for p in files:
        with np.load(p, allow_pickle=True) as d:
            y = float(np.array(d["label"]).reshape(-1)[0])
        if y >= 0.5:
            pos += 1
        else:
            neg += 1
    return float(neg / max(pos, 1)), int(pos), int(neg)


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    keys = [
        "bbox", "pose", "speed", "local_cnn", "local_motion",
        "sem_labels", "cat_depth", "label", "sem_masks",  # <-- add sem_masks
    ]
    for k in keys:
        if k in batch and torch.is_tensor(batch[k]):
            batch[k] = batch[k].to(device, non_blocking=True)
    return batch


def get_gate_values(model: nn.Module) -> Dict[str, float]:
    if hasattr(model, "gate_values"):
        try:
            return model.gate_values(detach=True)
        except TypeError:
            return model.gate_values()
    return {}


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    print(
        f"{prefix} | loss: {metrics.get('loss', 0):.4f} | "
        f"auc: {metrics.get('auc', float('nan')):.3f} | "
        f"pr_auc: {metrics.get('pr_auc', float('nan')):.3f} | "
        f"f1: {metrics.get('f1', float('nan')):.3f} | "
        f"acc: {metrics.get('acc', float('nan')):.3f}"
    )
    if "gate_l1" in metrics:
        print(f"{prefix} | gate_l1: {metrics['gate_l1']:.6f}")
    gate_keys = [k for k in metrics.keys() if k.endswith("_gate")]
    if gate_keys:
        gates = "  ".join(f"{k}={metrics[k]:.3f}" for k in sorted(gate_keys))
        print(f"{prefix} | Gates: {gates}")


# =============================================================================
# Dataset / model / criterion factories
# =============================================================================


def make_loaders(
    data_root: str,
    dataset_prefix: str,
    seq_len: int,
    batch_size: int,
    num_workers: int = 4,
    strict_len: bool = True,
    masks_cache_dir: str = None,          # <-- add this
):
    speed_stats_path = f"/workspace/project/data/{dataset_prefix}_speed_stats_splits.json"
    motion_stats_path = f"/workspace/project/data/{dataset_prefix}_motion_stats_splits.json"

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
        masks_cache_dir=masks_cache_dir,   # <-- add this
    )

    train_ds = PIESeqDataset(data_root, split="train", mode="train", **common_args)
    val_ds   = PIESeqDataset(data_root, split="val",   mode="eval",  **common_args)
    test_ds  = PIESeqDataset(data_root, split="test",  mode="eval",  **common_args)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_ds, train_loader, val_loader, test_loader


def make_model(args, device: torch.device) -> nn.Module:
    model = PIPNetKinFormerGRUMM(
        dropout_p=args.dropout_p,
        use_local_context=args.use_local_context,
        use_local_flow=args.use_local_flow,
        use_seg=args.use_seg,
        use_depth=args.use_depth,
        sem_mode=args.sem_mode, 
        scene_patch_grid=args.scene_patch_grid,
        gate_init=args.gate_init,
        gate_l1_weight=args.gate_l1_weight,
        d_model=args.global_d_model,
        n_heads=args.global_n_heads,
        spatial_layers=args.spatial_layers,
        ff_dim=args.global_ff_dim,
        tf_dropout=args.global_tf_dropout,
        run_transformer_without_context=args.run_transformer_without_context,
    )
    return model.to(device)


def make_criterion(args, pos_weight: float) -> KinFormerLoss:
    return KinFormerLoss(
        loss_type=args.loss_type,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        pos_weight=pos_weight,
    )


# =============================================================================
# Optimizer
# =============================================================================


def _params_by_prefix(model: nn.Module, prefixes: Iterable[str]) -> Tuple[List[nn.Parameter], set]:
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


def build_kinformer_optimizer(model: nn.Module, lr: float, args):
    """AdamW groups matched to PIPNetKinFormerGRUMM module names."""
    kin_prefixes = [
        "kinematic_branch", "kin_proj", "kin_pos", "kin_attn", "norm_kin",
    ]
    context_prefixes = [
        "local_pool", "local_proj", "local_pos", "local_gate",
        "flow_stem", "flow_pos", "flow_gate",
        "sem_preprocessor", "seg_stem", "seg_pos", "seg_gate",  # <-- ΑΛΛΑΓΗ
        "depth_down", "depth_stem", "depth_pos", "depth_gate",
    ]
    transformer_prefixes = ["spatial_transformer"]
    head_prefixes = ["fc_out"]

    kin_params, kin_ids = _params_by_prefix(model, kin_prefixes)
    context_params, context_ids = _params_by_prefix(model, context_prefixes)
    transformer_params, transformer_ids = _params_by_prefix(model, transformer_prefixes)
    head_params, head_ids = _params_by_prefix(model, head_prefixes)

    assigned = kin_ids | context_ids | transformer_ids | head_ids
    leftovers = [p for _, p in model.named_parameters() if p.requires_grad and id(p) not in assigned]
    if leftovers:
        head_params.extend(leftovers)

    groups = []
    if kin_params:
        groups.append({
            "name": "kinematic_anchor",
            "params": kin_params,
            "lr": lr * args.kin_lr_mult,
            "weight_decay": args.kin_l2,
        })
    if context_params:
        groups.append({
            "name": "gated_context",
            "params": context_params,
            "lr": lr * args.context_lr_mult,
            "weight_decay": args.visual_l2,
        })
    if transformer_params:
        groups.append({
            "name": "crossmodal_transformer",
            "params": transformer_params,
            "lr": lr * args.transformer_lr_mult,
            "weight_decay": args.visual_l2 * args.transformer_wd_mult,
        })
    if head_params:
        groups.append({
            "name": "classifier_head",
            "params": head_params,
            "lr": lr * args.head_lr_mult,
            "weight_decay": args.last_fc_l2,
        })

    print("\nOptimizer parameter groups: KinFormer-GRU-MM AdamW")
    for g in groups:
        n = sum(p.numel() for p in g["params"])
        print(f"  {g['name']:<24s} lr={g['lr']:.2e} wd={g['weight_decay']:.2e} params={n:>10,}")
    if leftovers:
        print(f"  note: {sum(p.numel() for p in leftovers):,} leftover parameter(s) added to classifier_head")

    optimizer = torch.optim.AdamW(groups)
    scheduler = None
    if not args.fixed_lr:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=args.lr_patience,
            min_lr=args.min_lr,
        )
    return optimizer, scheduler


# =============================================================================
# Train / eval epochs
# =============================================================================


def run_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: KinFormerLoss,
    optimizer: torch.optim.Optimizer,
    use_amp: bool,
    scaler,
    grad_clip_norm: float,
) -> Dict[str, float]:
    model.train()
    all_labels, all_probs = [], []
    running_loss = 0.0
    running_gate_l1 = 0.0
    total = 0

    pbar = tqdm(loader, desc="Train")
    for batch in pbar:
        batch = move_batch(batch, device)
        labels = batch["label"].float()
        optimizer.zero_grad(set_to_none=True)

        with amp_autocast(device, enabled=use_amp):
            outputs = model(batch, return_aux=True)
            losses = criterion(outputs, labels)

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
        running_gate_l1 += float(losses.get("gate_l1", labels.new_tensor(0.0)).item()) * bs
        total += bs

        probs = torch.sigmoid(outputs["logit"].squeeze(-1)).detach()
        all_labels.append(labels.detach().cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        pbar.set_postfix({"loss": f"{total_loss.item():.3f}"})

    labels_np = np.concatenate(all_labels)
    probs_np = np.concatenate(all_probs)
    metrics = compute_metrics(labels_np, probs_np)
    metrics["loss"] = running_loss / max(total, 1)
    metrics["gate_l1"] = running_gate_l1 / max(total, 1)
    metrics.update(get_gate_values(model))
    return metrics


@torch.no_grad()
def run_eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: KinFormerLoss,
    use_amp: bool,
) -> Dict[str, float]:
    model.eval()
    all_labels, all_probs = [], []
    running_loss = 0.0
    running_gate_l1 = 0.0
    total = 0

    for batch in tqdm(loader, desc="Eval "):
        batch = move_batch(batch, device)
        labels = batch["label"].float()
        with amp_autocast(device, enabled=use_amp):
            outputs = model(batch, return_aux=True)
            losses = criterion(outputs, labels)

        bs = labels.size(0)
        running_loss += losses["total"].item() * bs
        running_gate_l1 += float(losses.get("gate_l1", labels.new_tensor(0.0)).item()) * bs
        total += bs

        probs = torch.sigmoid(outputs["logit"].squeeze(-1))
        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    labels_np = np.concatenate(all_labels)
    probs_np = np.concatenate(all_probs)
    metrics = compute_metrics(labels_np, probs_np)
    metrics["loss"] = running_loss / max(total, 1)
    metrics["gate_l1"] = running_gate_l1 / max(total, 1)
    metrics.update(get_gate_values(model))
    return metrics


# =============================================================================
# CSV logging / training stage
# =============================================================================


CSV_COLUMNS = [
    "stage", "epoch", "split",
    "loss", "gate_l1", "auc", "pr_auc", "f1", "acc", "precision", "recall",
    "local_gate", "flow_gate", "seg_gate", "depth_gate",
    "total_params", "trainable_params",
    "jaad_root", "pie_root", "skip_transfer", "skip_baseline",
    "jaad_epochs", "jaad_lr", "jaad_batch_size",
    "pie_epochs", "pie_lr", "pie_batch_size",
    "baseline_epochs", "baseline_lr",
    "loss_type", "focal_alpha", "focal_gamma",
    "dropout_p", "global_d_model", "global_n_heads", "global_ff_dim",
    "global_tf_dropout", "spatial_layers", "scene_patch_grid",
    "use_local_context", "use_local_flow", "use_seg", "use_depth",
    "gate_init", "gate_l1_weight", "run_transformer_without_context",
    "kin_lr_mult", "context_lr_mult", "transformer_lr_mult", "head_lr_mult",
    "kin_l2", "visual_l2", "transformer_wd_mult", "last_fc_l2",
    "grad_clip_norm", "early_stopping_patience", "select_metric",
    "amp", "seed", "save_dir", "command",
    "lr_group_0", "lr_group_1", "lr_group_2", "lr_group_3", "timestamp",
]


def init_csv(path: str):
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(CSV_COLUMNS)


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.6f}"
    return "" if v is None else str(v)


def log_csv(
    path: str,
    stage: str,
    epoch: int,
    split: str,
    metrics: Dict[str, float],
    args,
    total_params: int,
    trainable_params: int,
    group_lrs: List[float] | None = None,
):
    group_lrs = list(group_lrs or [])
    while len(group_lrs) < 4:
        group_lrs.append("")

    row = [
        stage, epoch, split,
        _fmt(metrics.get("loss", 0.0)),
        _fmt(metrics.get("gate_l1", 0.0)),
        _fmt(metrics.get("auc", 0.0)),
        _fmt(metrics.get("pr_auc", 0.0)),
        _fmt(metrics.get("f1", 0.0)),
        _fmt(metrics.get("acc", 0.0)),
        _fmt(metrics.get("precision", 0.0)),
        _fmt(metrics.get("recall", 0.0)),
        _fmt(metrics.get("local_gate", "")),
        _fmt(metrics.get("flow_gate", "")),
        _fmt(metrics.get("seg_gate", "")),
        _fmt(metrics.get("depth_gate", "")),
        total_params, trainable_params,
        args.jaad_root, args.pie_root, args.skip_transfer, args.skip_baseline,
        args.jaad_epochs, args.jaad_lr, args.jaad_batch_size,
        args.pie_epochs, args.pie_lr, args.pie_batch_size,
        args.baseline_epochs, args.baseline_lr,
        args.loss_type, args.focal_alpha, args.focal_gamma,
        args.dropout_p, args.global_d_model, args.global_n_heads, args.global_ff_dim,
        args.global_tf_dropout, args.spatial_layers, args.scene_patch_grid,
        args.use_local_context, args.use_local_flow, args.use_seg, args.use_depth,
        args.gate_init, args.gate_l1_weight, args.run_transformer_without_context,
        args.kin_lr_mult, args.context_lr_mult, args.transformer_lr_mult, args.head_lr_mult,
        args.kin_l2, args.visual_l2, args.transformer_wd_mult, args.last_fc_l2,
        args.grad_clip_norm, args.early_stopping_patience, args.select_metric,
        args.amp, args.seed, args.save_dir, getattr(args, "command", ""),
        f"{group_lrs[0]:.2e}" if isinstance(group_lrs[0], float) else group_lrs[0],
        f"{group_lrs[1]:.2e}" if isinstance(group_lrs[1], float) else group_lrs[1],
        f"{group_lrs[2]:.2e}" if isinstance(group_lrs[2], float) else group_lrs[2],
        f"{group_lrs[3]:.2e}" if isinstance(group_lrs[3], float) else group_lrs[3],
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    ]
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)


def save_json(path: str | Path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def run_training_stage(
    stage_name: str,
    model: nn.Module,
    device: torch.device,
    criterion: KinFormerLoss,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    epochs: int,
    use_amp: bool,
    save_dir: str,
    csv_path: str,
    args,
):
    os.makedirs(save_dir, exist_ok=True)
    total_params, trainable_params = count_parameters(model)

    print("\n" + "=" * 80)
    print(f"[{stage_name}] Model parameters")
    print("=" * 80)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        print(f"    {name:<30s}: {n:>10,}")
    print("=" * 80)

    best_val_score = -1.0
    best_state = None
    patience_counter = 0
    select_metric = args.select_metric

    for epoch in range(1, epochs + 1):
        print(f"\n{'=' * 80}")
        print(f"[{stage_name}] Epoch {epoch}/{epochs}")
        print(f"{'=' * 80}")
        group_lrs = [pg["lr"] for pg in optimizer.param_groups]

        train_m = run_train_epoch(
            model, train_loader, device, criterion, optimizer,
            use_amp=use_amp, scaler=scaler, grad_clip_norm=args.grad_clip_norm,
        )
        print_metrics(train_m, f"[{stage_name}] Train")
        log_csv(csv_path, stage_name, epoch, "train", train_m, args,
                total_params, trainable_params, group_lrs=group_lrs)

        val_m = run_eval_epoch(model, val_loader, device, criterion, use_amp)
        print_metrics(val_m, f"[{stage_name}] Val  ")
        log_csv(csv_path, stage_name, epoch, "val", val_m, args,
                total_params, trainable_params, group_lrs=group_lrs)

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
                "args": vars(args),
            }
            torch.save(best_state, os.path.join(save_dir, "best_model.pth"))
            save_json(os.path.join(save_dir, "best_val_metrics.json"), val_m)
            print(
                f"  * New best model (Val {select_metric.upper()}: {best_val_score:.4f} | "
                f"Val AUC: {val_m['auc']:.4f} | Val PR-AUC: {val_m['pr_auc']:.4f} | "
                f"Val F1: {val_m['f1']:.4f})"
            )
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  ! No improvement for {patience_counter} epoch(s).")

        if patience_counter >= args.early_stopping_patience:
            print(f"\n  Early stopping after {args.early_stopping_patience} stagnant epochs.")
            break

    test_m = {}
    if best_state is not None:
        print(f"\n[{stage_name}] Loading best model (Epoch {best_state['epoch']})...")
        model.load_state_dict(best_state["model"])
        test_m = run_eval_epoch(model, test_loader, device, criterion, use_amp)
        print_metrics(test_m, f"[{stage_name}] Test ")
        group_lrs = [pg["lr"] for pg in optimizer.param_groups]
        log_csv(csv_path, stage_name, best_state["epoch"], "test", test_m, args,
                total_params, trainable_params, group_lrs=group_lrs)
        save_json(os.path.join(save_dir, "test_metrics.json"), test_m)
        save_json(os.path.join(save_dir, "gate_values.json"), get_gate_values(model))

    return test_m, best_state


# =============================================================================
# Main
# =============================================================================


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Train KinFormer-GRU-MM")

    # Data
    p.add_argument("--jaad_root", type=str, required=True)
    p.add_argument("--pie_root", type=str, required=True)

    # Stage control
    p.add_argument("--skip_transfer", action="store_true", help="Skip JAAD pretrain + PIE fine-tune")
    p.add_argument("--skip_baseline", action="store_true", help="Skip PIE-only baseline stage")
    p.add_argument("--jaad_ckpt", type=str, default=None, help="Existing JAAD checkpoint for S2")

    # Sequence/data loading
    p.add_argument("--seq_len", type=int, default=10)
    p.add_argument("--no_strict_len", action="store_true")

    # Model modality flags
    p.add_argument("--use_local_context", action="store_true")
    p.add_argument("--use_local_flow", action="store_true")
    p.add_argument("--use_seg", action="store_true")
    p.add_argument("--use_depth", action="store_true")
    p.add_argument("--masks_cache_dir", type=str, default=None,
               help="Directory with precomputed sem_masks (output of precompute_masks.py). "
                    "If None, fall back to runtime mask generation.")
   
    # Model hyperparams
    p.add_argument("--dropout_p", type=float, default=0.4)
    p.add_argument("--global_d_model", type=int, default=128)
    p.add_argument("--global_n_heads", type=int, default=2)
    p.add_argument("--global_ff_dim", type=int, default=256)
    p.add_argument("--global_tf_dropout", type=float, default=0.1)
    p.add_argument("--spatial_layers", type=int, default=1)
    p.add_argument("--scene_patch_grid", type=int, default=4)
    p.add_argument("--gate_init", type=float, default=-2.0)
    p.add_argument("--sem_mode",
                    choices=["masks", "embed_reduced", "embed_full"],  # removed masks_inst
                    default="masks",
                    help="Segmentation preprocessing mode. "
                        "embed_full = legacy 20-class learned embedding (control); "
                        "embed_reduced = Solution 1 only; "
                        "masks = Solutions 1+2 (default)")
    p.add_argument("--gate_l1_weight", type=float, default=0.0)
    p.add_argument("--run_transformer_without_context", action="store_true",
                   help="For ablation only: run spatial Transformer even with no context tokens")

    # Stage training settings
    p.add_argument("--jaad_epochs", type=int, default=80)
    p.add_argument("--jaad_lr", type=float, default=5e-5)
    p.add_argument("--jaad_batch_size", type=int, default=10)
    p.add_argument("--pie_epochs", type=int, default=50)
    p.add_argument("--pie_lr", type=float, default=1e-5)
    p.add_argument("--pie_batch_size", type=int, default=64)
    p.add_argument("--baseline_epochs", type=int, default=15)
    p.add_argument("--baseline_lr", type=float, default=2e-5)

    # Loss
    p.add_argument("--loss_type", choices=["focal", "bce"], default="focal")
    p.add_argument("--focal_alpha", type=float, default=0.75)
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--wpos", type=float, default=-1.0,
                   help="Manual BCE pos_weight. Ignored by focal loss unless loss_type=bce")

    # Optimizer groups
    p.add_argument("--kin_lr_mult", type=float, default=1.0)
    p.add_argument("--context_lr_mult", type=float, default=1.0)
    p.add_argument("--transformer_lr_mult", type=float, default=1.0)
    p.add_argument("--head_lr_mult", type=float, default=1.0)
    p.add_argument("--kin_l2", type=float, default=0.0)
    p.add_argument("--visual_l2", type=float, default=0.005)
    p.add_argument("--transformer_wd_mult", type=float, default=1.0)
    p.add_argument("--last_fc_l2", type=float, default=0.01)
    p.add_argument("--grad_clip_norm", type=float, default=5.0)

    # Scheduler/selection
    p.add_argument("--fixed_lr", action="store_true")
    p.add_argument("--lr_patience", type=int, default=3)
    p.add_argument("--min_lr", type=float, default=1e-7)
    p.add_argument("--early_stopping_patience", type=int, default=10)
    p.add_argument("--select_metric", choices=["auc", "pr_auc", "f1"], default="pr_auc")

    # Runtime
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", type=str, default="checkpoints_kinformer_gru_mm")

    return p


def main():
    parser = build_argparser()
    args = parser.parse_args()
    args.command = " ".join(sys.argv)

    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = bool(device.type == "cuda" and args.amp)
    scaler = make_grad_scaler(use_amp)
    strict_len = not args.no_strict_len

    os.makedirs(args.save_dir, exist_ok=True)
    csv_path = os.path.join(args.save_dir, "training_log.csv")
    init_csv(csv_path)
    save_json(os.path.join(args.save_dir, "args.json"), vars(args))

    print("=" * 80)
    print("PIPNet — KinFormer-GRU-MM")
    print("Kinematic GRU anchor + gated local/flow/seg/depth context")
    print("=" * 80)
    print(f"Device:          {device}")
    print(f"AMP:             {use_amp}")
    print(f"Seed:            {args.seed}")
    print(f"Loss:            {args.loss_type}")
    print(f"Select metric:   {args.select_metric}")
    print(f"Modalities:      local={args.use_local_context} flow={args.use_local_flow} "
          f"seg={args.use_seg} depth={args.use_depth}")
    print(f"Gate init/L1:    {args.gate_init} / {args.gate_l1_weight}")
    print(f"d_model/heads:   {args.global_d_model} / {args.global_n_heads}")
    print(f"spatial layers:  {args.spatial_layers}")
    print(f"scene grid:      {args.scene_patch_grid}x{args.scene_patch_grid}")
    print("=" * 80)

    jaad_ckpt = None

    if not args.skip_transfer:
        if args.jaad_ckpt:
            print(f"\n[S1] Loading existing JAAD checkpoint: {args.jaad_ckpt}")
            jaad_ckpt = torch.load(args.jaad_ckpt, map_location=device)
        else:
            jaad_train_ds, jaad_train_loader, jaad_val_loader, jaad_test_loader = make_loaders(
                args.jaad_root, "jaad", args.seq_len, args.jaad_batch_size,
                args.num_workers, strict_len,
                masks_cache_dir=args.masks_cache_dir,   # <-- NEW
            )
            jaad_wpos = args.wpos if args.wpos > 0 else compute_pos_weight_from_npz(jaad_train_ds.files)[0]
            print(f"\n[S1] JAAD pos_weight={jaad_wpos:.3f} "
                  f"({'ignored by focal' if args.loss_type == 'focal' else 'used by BCE'})")
            model_s1 = make_model(args, device)
            criterion_s1 = make_criterion(args, jaad_wpos)
            opt_s1, sch_s1 = build_kinformer_optimizer(model_s1, args.jaad_lr, args)
            _, jaad_ckpt = run_training_stage(
                "S1_JAAD", model_s1, device, criterion_s1,
                jaad_train_loader, jaad_val_loader, jaad_test_loader,
                opt_s1, sch_s1, scaler, args.jaad_epochs, use_amp,
                os.path.join(args.save_dir, "stage1_jaad"), csv_path, args,
            )

        pie_train_ds, pie_train_loader, pie_val_loader, pie_test_loader = make_loaders(
            args.pie_root, "pie", args.seq_len, args.pie_batch_size,
            args.num_workers, strict_len,
            masks_cache_dir=args.masks_cache_dir,   # <-- NEW
        )
        pie_wpos = args.wpos if args.wpos > 0 else compute_pos_weight_from_npz(pie_train_ds.files)[0]
        print(f"\n[S2] PIE pos_weight={pie_wpos:.3f} "
              f"({'ignored by focal' if args.loss_type == 'focal' else 'used by BCE'})")
        model_s2 = make_model(args, device)
        model_s2.load_state_dict(jaad_ckpt["model"])
        criterion_s2 = make_criterion(args, pie_wpos)
        opt_s2, sch_s2 = build_kinformer_optimizer(model_s2, args.pie_lr, args)
        run_training_stage(
            "S2_PIE_transfer", model_s2, device, criterion_s2,
            pie_train_loader, pie_val_loader, pie_test_loader,
            opt_s2, sch_s2, scaler, args.pie_epochs, use_amp,
            os.path.join(args.save_dir, "stage2_pie_transfer"), csv_path, args,
        )
    else:
        pie_train_ds, pie_train_loader, pie_val_loader, pie_test_loader = make_loaders(
            args.pie_root, "pie", args.seq_len, args.pie_batch_size,
            args.num_workers, strict_len,
            masks_cache_dir=args.masks_cache_dir,   # <-- NEW
        )
        pie_wpos = args.wpos if args.wpos > 0 else compute_pos_weight_from_npz(pie_train_ds.files)[0]

    if not args.skip_baseline:
        print(f"\n[S3] PIE-only baseline pos_weight={pie_wpos:.3f} "
              f"({'ignored by focal' if args.loss_type == 'focal' else 'used by BCE'})")
        model_s3 = make_model(args, device)
        criterion_s3 = make_criterion(args, pie_wpos)
        opt_s3, sch_s3 = build_kinformer_optimizer(model_s3, args.baseline_lr, args)
        run_training_stage(
            "S3_PIE_baseline", model_s3, device, criterion_s3,
            pie_train_loader, pie_val_loader, pie_test_loader,
            opt_s3, sch_s3, scaler, args.baseline_epochs, use_amp,
            os.path.join(args.save_dir, "stage3_pie_baseline"), csv_path, args,
        )

    print("\n" + "=" * 80)
    print("Training complete. Results in:", args.save_dir)
    print("=" * 80)


if __name__ == "__main__":
    main()
