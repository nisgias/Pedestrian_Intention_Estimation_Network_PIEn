# train_v3_cross.py
"""
Training script for PIPNet-Alpha V3 with Cross-Attention
Matches Paper Figure 3 architecture more closely
"""

import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

from data.pie import PIESeqDataset
from models.pipnet_alpha_v3_cross import PIPNetAlphaV3Cross
from torch.cuda.amp import autocast, GradScaler


# ============================================================
# MULTI-TASK LOSS
# ============================================================

class MultiTaskLoss(nn.Module):
    def __init__(self, aux_weight: float = 0.1, entropy_weight: float = 0.01, pos_weight: float = 1.0):
        super().__init__()
        self.aux_weight = aux_weight
        self.entropy_weight = entropy_weight
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        
    def forward(self, outputs: dict, labels: torch.Tensor):
        device = labels.device
        self.bce.pos_weight = self.bce.pos_weight.to(device)
        labels = labels.float()
        
        losses = {}
        
        # Main loss
        main_loss = self.bce(outputs['logit'].squeeze(-1), labels)
        losses['main'] = main_loss
        
        # Auxiliary losses
        if 'aux_kin' in outputs:
            aux_kin_loss = self.bce(outputs['aux_kin'].squeeze(-1), labels)
            aux_local_loss = self.bce(outputs['aux_local'].squeeze(-1), labels)
            aux_global_loss = self.bce(outputs['aux_global'].squeeze(-1), labels)
            
            aux_loss = self.aux_weight * (aux_kin_loss + aux_local_loss + aux_global_loss)
            
            losses['aux'] = aux_loss
            losses['aux_kin_loss'] = aux_kin_loss.item()
            losses['aux_local_loss'] = aux_local_loss.item()
            losses['aux_global_loss'] = aux_global_loss.item()
        else:
            aux_loss = torch.tensor(0.0, device=device)
            losses['aux'] = aux_loss
        
        # Entropy regularization on modality weights (visual vs kin)
        if 'modality_weights' in outputs:
            weights = outputs['modality_weights']  # [B, 2]
            entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1).mean()
            entropy_loss = -self.entropy_weight * entropy
            losses['entropy'] = entropy_loss
        else:
            entropy_loss = torch.tensor(0.0, device=device)
            losses['entropy'] = entropy_loss
        
        total = main_loss + aux_loss + entropy_loss
        losses['total'] = total
        
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


def make_loaders(data_root, seq_len, batch_size, num_workers=4, strict_len=True):
    speed_stats_path = "/workspace/project/data/speed_stats_splits.json"
    motion_stats_path = "/workspace/project/data/motion_stats_splits.json"
    
    common_args = dict(
        seq_len=seq_len, strict_len=strict_len,
        speed_norm="minmax", speed_stats_path=speed_stats_path, speed_scope="global",
        motion_norm="p99abs", motion_stats_path=motion_stats_path, motion_scope="global", motion_clip=1.0,
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
    except:
        pass
    return float("nan")


@torch.no_grad()
def compute_metrics(labels_np: np.ndarray, probs_np: np.ndarray):
    metrics = {}
    metrics["auc"] = safe_auc(labels_np, probs_np)
    
    preds_bin = (probs_np >= 0.5).astype(np.int32)
    metrics["acc"] = (preds_bin == labels_np).mean()
    metrics["f1"] = f1_score(labels_np, preds_bin, zero_division=0)
    metrics["precision"] = precision_score(labels_np, preds_bin, zero_division=0)
    metrics["recall"] = recall_score(labels_np, preds_bin, zero_division=0)
    
    return metrics


# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def run_train_epoch(model, loader, device, criterion, optimizer, use_amp=False, scaler=None):
    model.train()
    
    all_labels, all_probs = [], []
    all_probs_kin, all_probs_local, all_probs_global = [], [], []
    
    running_loss = 0.0
    running_aux_kin, running_aux_local, running_aux_global = 0.0, 0.0, 0.0
    total = 0
    
    # Track weights: [visual, kin] and [local, cross]
    modality_weights_sum = torch.zeros(2)
    visual_weights_sum = torch.zeros(2)
    weight_count = 0
    
    pbar = tqdm(loader, desc="Train")
    
    for batch in pbar:
        for k in ["bbox", "pose", "speed", "local_cnn", "local_motion", "sem_labels", "cat_depth", "label"]:
            if k in batch and torch.is_tensor(batch[k]):
                batch[k] = batch[k].to(device, non_blocking=True)
        
        labels = batch["label"].float()
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast(enabled=use_amp):
            outputs = model(batch, return_aux=True)
            losses = criterion(outputs, labels)
        
        if use_amp and scaler is not None:
            scaler.scale(losses['total']).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses['total'].backward()
            optimizer.step()
        
        batch_size = labels.size(0)
        running_loss += losses['total'].item() * batch_size
        total += batch_size
        
        if 'aux_kin_loss' in losses:
            running_aux_kin += losses['aux_kin_loss'] * batch_size
            running_aux_local += losses['aux_local_loss'] * batch_size
            running_aux_global += losses['aux_global_loss'] * batch_size
        
        # Main predictions
        probs = torch.sigmoid(outputs['logit'].squeeze(-1)).detach()
        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        
        # Branch predictions
        if 'aux_kin' in outputs:
            all_probs_kin.append(torch.sigmoid(outputs['aux_kin'].squeeze(-1)).detach().cpu().numpy())
            all_probs_local.append(torch.sigmoid(outputs['aux_local'].squeeze(-1)).detach().cpu().numpy())
            all_probs_global.append(torch.sigmoid(outputs['aux_global'].squeeze(-1)).detach().cpu().numpy())
        
        # Track attention weights
        if 'modality_weights' in outputs:
            modality_weights_sum += outputs['modality_weights'].detach().cpu().mean(0)
            visual_weights_sum += outputs['visual_weights'].detach().cpu().mean(0)
            weight_count += 1
        
        pbar.set_postfix({"loss": f"{losses['total'].item():.3f}"})
    
    # Compute metrics
    labels_np = np.concatenate(all_labels)
    probs_np = np.concatenate(all_probs)
    metrics = compute_metrics(labels_np, probs_np)
    metrics["loss"] = running_loss / max(total, 1)
    
    # Per-branch AUC
    if all_probs_kin:
        probs_kin = np.concatenate(all_probs_kin)
        probs_local = np.concatenate(all_probs_local)
        probs_global = np.concatenate(all_probs_global)
        
        metrics["auc_kin"] = safe_auc(labels_np, probs_kin)
        metrics["auc_local"] = safe_auc(labels_np, probs_local)
        metrics["auc_global"] = safe_auc(labels_np, probs_global)
        
        metrics["loss_kin"] = running_aux_kin / max(total, 1)
        metrics["loss_local"] = running_aux_local / max(total, 1)
        metrics["loss_global"] = running_aux_global / max(total, 1)
    
    # Attention weights
    if weight_count > 0:
        avg_modality = modality_weights_sum / weight_count
        avg_visual = visual_weights_sum / weight_count
        metrics["w_visual"] = avg_modality[0].item()
        metrics["w_kin"] = avg_modality[1].item()
        metrics["w_local"] = avg_visual[0].item()
        metrics["w_cross"] = avg_visual[1].item()
    
    return metrics


@torch.no_grad()
def run_eval_epoch(model, loader, device, criterion, use_amp=False):
    model.eval()
    
    all_labels, all_probs = [], []
    all_probs_kin, all_probs_local, all_probs_global = [], [], []
    
    running_loss = 0.0
    running_aux_kin, running_aux_local, running_aux_global = 0.0, 0.0, 0.0
    total = 0
    
    modality_weights_sum = torch.zeros(2)
    visual_weights_sum = torch.zeros(2)
    weight_count = 0
    
    for batch in tqdm(loader, desc="Eval"):
        for k in ["bbox", "pose", "speed", "local_cnn", "local_motion", "sem_labels", "cat_depth", "label"]:
            if k in batch and torch.is_tensor(batch[k]):
                batch[k] = batch[k].to(device, non_blocking=True)
        
        labels = batch["label"].float()
        
        with autocast(enabled=use_amp):
            outputs = model(batch, return_aux=True)
            losses = criterion(outputs, labels)
        
        batch_size = labels.size(0)
        running_loss += losses['total'].item() * batch_size
        total += batch_size
        
        if 'aux_kin_loss' in losses:
            running_aux_kin += losses['aux_kin_loss'] * batch_size
            running_aux_local += losses['aux_local_loss'] * batch_size
            running_aux_global += losses['aux_global_loss'] * batch_size
        
        probs = torch.sigmoid(outputs['logit'].squeeze(-1))
        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        
        if 'aux_kin' in outputs:
            all_probs_kin.append(torch.sigmoid(outputs['aux_kin'].squeeze(-1)).cpu().numpy())
            all_probs_local.append(torch.sigmoid(outputs['aux_local'].squeeze(-1)).cpu().numpy())
            all_probs_global.append(torch.sigmoid(outputs['aux_global'].squeeze(-1)).cpu().numpy())
        
        if 'modality_weights' in outputs:
            modality_weights_sum += outputs['modality_weights'].cpu().mean(0)
            visual_weights_sum += outputs['visual_weights'].cpu().mean(0)
            weight_count += 1
    
    labels_np = np.concatenate(all_labels)
    probs_np = np.concatenate(all_probs)
    metrics = compute_metrics(labels_np, probs_np)
    metrics["loss"] = running_loss / max(total, 1)
    
    if all_probs_kin:
        probs_kin = np.concatenate(all_probs_kin)
        probs_local = np.concatenate(all_probs_local)
        probs_global = np.concatenate(all_probs_global)
        
        metrics["auc_kin"] = safe_auc(labels_np, probs_kin)
        metrics["auc_local"] = safe_auc(labels_np, probs_local)
        metrics["auc_global"] = safe_auc(labels_np, probs_global)
        
        metrics["loss_kin"] = running_aux_kin / max(total, 1)
        metrics["loss_local"] = running_aux_local / max(total, 1)
        metrics["loss_global"] = running_aux_global / max(total, 1)
    
    if weight_count > 0:
        avg_modality = modality_weights_sum / weight_count
        avg_visual = visual_weights_sum / weight_count
        metrics["w_visual"] = avg_modality[0].item()
        metrics["w_kin"] = avg_modality[1].item()
        metrics["w_local"] = avg_visual[0].item()
        metrics["w_cross"] = avg_visual[1].item()
    
    return metrics


def print_metrics(metrics, prefix=""):
    print(f"{prefix} | loss: {metrics['loss']:.4f} | auc: {metrics['auc']:.3f} | "
          f"f1: {metrics['f1']:.3f} | acc: {metrics['acc']:.3f}")
    
    if 'auc_kin' in metrics:
        print(f"{prefix} | Branch AUC:  kin={metrics['auc_kin']:.3f}  local={metrics['auc_local']:.3f}  global(cross)={metrics['auc_global']:.3f}")
    
    if 'w_visual' in metrics:
        print(f"{prefix} | Final Attn:  visual={metrics['w_visual']:.3f}  kin={metrics['w_kin']:.3f}")
        print(f"{prefix} | Visual Attn: local={metrics['w_local']:.3f}  cross={metrics['w_cross']:.3f}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train PIPNet-Alpha V3 Cross-Attention")
    
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--last_fc_l2", type=float, default=1e-4)
    parser.add_argument("--aux_weight", type=float, default=0.1)
    parser.add_argument("--entropy_weight", type=float, default=0.01)
    parser.add_argument("--dropout_p", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--wpos", type=float, default=-1.0)
    parser.add_argument("--save_dir", type=str, default="checkpoints_v3_cross")
    parser.add_argument("--no_strict_len", action="store_true")
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and args.amp
    scaler = GradScaler(enabled=use_amp)
    strict_len = not args.no_strict_len
    
    print("=" * 80)
    print("PIPNet-Alpha V3 Cross-Attention (Paper Figure 3 Architecture)")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Architecture: Local→Attn + Global→CrossAttn → VisualAttn → FinalAttn(+Kin)")
    print(f"Dropout: {args.dropout_p}")
    print(f"Aux weight: {args.aux_weight}")
    print("=" * 80)
    
    # Data
    print("\nLoading data...")
    train_ds, train_loader, val_loader, test_loader = make_loaders(
        args.data_root, args.seq_len, args.batch_size, args.num_workers, strict_len
    )
    print(f"Train: {len(train_ds)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")
    
    # Class weight
    if args.wpos > 0:
        w_pos = args.wpos
    else:
        w_pos, pos_n, neg_n = compute_pos_weight_from_npz(train_ds.files)
        print(f"Auto pos_weight: {w_pos:.3f} (Pos={pos_n}, Neg={neg_n})")
    
    # Model
    print("\nBuilding model...")
    model = PIPNetAlphaV3Cross(dropout_p=args.dropout_p).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
    
    # Loss
    criterion = MultiTaskLoss(
        aux_weight=args.aux_weight,
        entropy_weight=args.entropy_weight,
        pos_weight=w_pos,
    )
    
    # Optimizer
    fc_params = list(model.fc_out.parameters())
    fc_ids = {id(p) for p in fc_params}
    base_params = [p for p in model.parameters() if id(p) not in fc_ids]
    
    optimizer = torch.optim.RMSprop([
        {"params": base_params, "weight_decay": 0.0},
        {"params": fc_params, "weight_decay": args.last_fc_l2},
    ], lr=args.lr)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True, min_lr=1e-7
    )
    
    # Training
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_auc = -1.0
    best_state = None
    
    print("\n" + "=" * 80)
    print("Starting training...")
    print("Watch: global(cross) AUC - this uses cross-attention, should be better than before!")
    print("=" * 80)
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*80}")
        
        train_m = run_train_epoch(model, train_loader, device, criterion, optimizer, use_amp, scaler)
        print_metrics(train_m, "Train")
        
        val_m = run_eval_epoch(model, val_loader, device, criterion, use_amp)
        print_metrics(val_m, "Val  ")
        
        scheduler.step(val_m['auc'])
        
        if val_m['auc'] > best_val_auc:
            best_val_auc = val_m['auc']
            best_state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_metrics": val_m,
            }
            torch.save(best_state, os.path.join(args.save_dir, "best_model.pth"))
            print(f"  ★ New best model (AUC: {best_val_auc:.4f})")
    
    # Final test
    if best_state:
        print(f"\n{'='*80}")
        print(f"Loading best model (Epoch {best_state['epoch']})...")
        model.load_state_dict(best_state["model"])
        
        test_m = run_eval_epoch(model, test_loader, device, criterion, use_amp)
        
        print("\n" + "=" * 80)
        print("FINAL TEST RESULTS")
        print("=" * 80)
        print(f"Main Model:")
        print(f"  AUC:       {test_m['auc']:.4f}")
        print(f"  Accuracy:  {test_m['acc']:.4f}")
        print(f"  F1:        {test_m['f1']:.4f}")
        print(f"  Precision: {test_m['precision']:.4f}")
        print(f"  Recall:    {test_m['recall']:.4f}")
        
        print(f"\nBranch Contributions (Individual AUC):")
        if 'auc_kin' in test_m:
            print(f"  Kinematic:      AUC={test_m['auc_kin']:.4f}")
            print(f"  Local:          AUC={test_m['auc_local']:.4f}")
            print(f"  Global (Cross): AUC={test_m['auc_global']:.4f}  ← Cross-attention output!")
        
        print(f"\nAttention Weights:")
        if 'w_visual' in test_m:
            print(f"  Final: visual={test_m['w_visual']:.3f}  kin={test_m['w_kin']:.3f}")
            print(f"  Visual: local={test_m['w_local']:.3f}  cross={test_m['w_cross']:.3f}")
        print("=" * 80)


if __name__ == "__main__":
    main()
