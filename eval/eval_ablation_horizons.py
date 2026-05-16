"""
eval_ablation_horizons.py

Branch-zeroing ablation across TTE horizons.

For each (horizon, ablation) we report the FUSED prediction's AUC, F1, etc.
Ablations:
    Full          - no zeroing, baseline
    NoGlobal      - zero out h_global before visual fusion
    NoLocal       - zero out z_local before visual fusion
    NoKin         - zero out z_kin before final fusion
    KinOnly       - zero both local AND global (kin only)
    VisualOnly    - zero kin (visual = local + global only)

How: we patch model.forward() at runtime via a context manager. No checkpoint
change, no retraining.

NOTE: the existing model already exposes self._ablate_global. We add the other
flags symmetrically below (monkey-patch). The patched forward is a *minimal* copy
of the original forward but with two extra zeroing knobs.
"""

import argparse
import csv
import os
import contextlib
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    precision_recall_curve, accuracy_score,
)

from data.pie import PIESeqDataset
from models.pipnet_alpha_v4_final import PIPNetAlphaV4Final


# ============================================================================
# Patched forward (matches PIPNetAlphaV4Final.forward but honors three flags)
# ============================================================================

def _patched_forward(self, batch, return_aux=False):
    bbox         = batch["bbox"]
    pose         = batch["pose"]
    speed        = batch["speed"]
    local_cnn    = batch["local_cnn"]
    local_motion = batch["local_motion"]
    sem_labels   = batch["sem_labels"]
    cat_depth    = batch["cat_depth"]

    B = bbox.size(0)
    T = bbox.size(1)

    # ---- Kin branch
    h_kin = self.kinematic_branch(bbox, pose, speed)
    z_kin, _ = self.kin_attn(h_kin)
    z_kin = self.norm_kin(z_kin)

    # ---- Local branch
    h_local = self.local_branch(local_cnn, local_motion)
    z_local, _ = self.local_attn(h_local)
    z_local = self.norm_local(z_local)

    # ---- Global branch
    h_global = self.global_branch(sem_labels, cat_depth)

    # ---- Apply ablation flags BEFORE visual fusion
    if getattr(self, "_ablate_local", False):
        z_local_used = torch.zeros_like(z_local)
    else:
        z_local_used = z_local

    if getattr(self, "_ablate_global", False):
        h_global_used = torch.zeros_like(h_global)
    else:
        h_global_used = h_global

    # ---- Visual fusion
    z_local_expanded = z_local_used.unsqueeze(1).expand(-1, T, -1)
    h_fused = torch.cat([z_local_expanded, h_global_used], dim=-1)
    z_visual, alpha_visual = self.visual_fuse_attn(h_fused)
    z_visual = self.norm_visual(z_visual)

    # ---- Apply kin ablation BEFORE final fusion
    if getattr(self, "_ablate_kin", False):
        z_kin_used = torch.zeros_like(z_kin)
    else:
        z_kin_used = z_kin

    # ---- Final fusion
    z_visual_kin = torch.stack([z_visual, z_kin_used], dim=1)
    z_final, alpha_final = self.final_attn(z_visual_kin, use_mean_query=True)
    z_final = self.fc_drop(z_final)
    logit = self.fc_out(z_final)

    if return_aux:
        z_global_att, _ = self.global_attn(h_global, use_mean_query=True)
        z_global_att = self.norm_global(z_global_att)
        return {
            "logit":               logit,
            "aux_kin":             self.aux_kin(z_kin),
            "aux_local":           self.aux_local(z_local),
            "aux_global":          self.aux_global(z_global_att),
            "modality_weights":    alpha_final,
            "visual_fuse_weights": alpha_visual,
        }
    return logit


@contextlib.contextmanager
def ablation_mode(model, ablate_kin=False, ablate_local=False, ablate_global=False):
    """Temporarily swap forward() and set ablation flags. Restores on exit."""
    original_forward = model.forward
    model.forward = _patched_forward.__get__(model, type(model))
    model._ablate_kin    = ablate_kin
    model._ablate_local  = ablate_local
    model._ablate_global = ablate_global
    try:
        yield model
    finally:
        model.forward = original_forward
        model._ablate_kin    = False
        model._ablate_local  = False
        model._ablate_global = False


# ============================================================================
# Data + metrics (same as before)
# ============================================================================

def make_loader(root, split, dataset, batch_size, num_workers, seq_len):
    ds = PIESeqDataset(
        root, split=split, mode="eval", seq_len=seq_len,
        strict_len=True, return_meta=True,
        speed_norm="minmax",
        speed_stats_path=f"/workspace/project/data/{dataset}_speed_stats_splits.json",
        speed_scope="global",
        motion_norm="p99abs",
        motion_stats_path=f"/workspace/project/data/{dataset}_motion_stats_splits.json",
        motion_scope="global", motion_clip=1.0,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


@torch.no_grad()
def collect_fused_predictions(model, loader, device, amp=False):
    model.eval()
    labels_list, probs_list = [], []
    for batch in loader:
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            out = model(batch, return_aux=False)
        if isinstance(out, dict):
            logit = out["logit"]
        else:
            logit = out
        p = torch.sigmoid(logit.squeeze(-1).float()).cpu().numpy()
        probs_list.append(p)
        labels_list.append(batch["label"].cpu().numpy().astype(np.int32))
    return np.concatenate(labels_list), np.concatenate(probs_list)


def safe_auc(y, p):
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def best_f1_threshold(y, p):
    if len(y) == 0 or len(np.unique(y)) < 2:
        return 0.5, 0.0
    prec, rec, thrs = precision_recall_curve(y, p)
    if len(thrs) == 0:
        return 0.5, 0.0
    f1s = 2.0 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
    best_idx = int(np.argmax(f1s))
    return float(thrs[best_idx]), float(f1s[best_idx])


def calc_metrics(y, p, thr):
    if len(y) == 0:
        return 0.0, 0.0, 0.0, 0.0
    preds = (p >= thr).astype(np.int32)
    return (
        float(f1_score(y, preds, zero_division=0)),
        float(precision_score(y, preds, zero_division=0)),
        float(recall_score(y, preds, zero_division=0)),
        float(accuracy_score(y, preds)),
    )


# ============================================================================
# Main
# ============================================================================

ABLATIONS = [
    # name,         kin,    local,  global
    ("Full",        False,  False,  False),
    ("NoGlobal",    False,  False,  True),
    ("NoLocal",     False,  True,   False),
    ("NoKin",       True,   False,  False),
    ("KinOnly",     False,  True,   True),
    ("VisualOnly",  True,   False,  False),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_root", type=str, default="/Datasets/PIE_PREP_OUT/ETCs")
    ap.add_argument("--horizons", type=str, default="ETC0_5,ETC1,ETC2,ETC3,ETC4")
    ap.add_argument("--dataset", type=str, default="pie", choices=["pie", "jaad"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seq_len", type=int, default=10)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--static_thr", type=float, default=0.7585)
    ap.add_argument("--out_csv", type=str, default="ablation_horizons.csv")

    ap.add_argument("--dropout_p",        type=float, default=0.2)
    ap.add_argument("--local_dropout_p",  type=float, default=0.1)
    ap.add_argument("--global_dropout_p", type=float, default=0.4)
    args = ap.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and args.amp

    model = PIPNetAlphaV4Final(
        dropout_p=args.dropout_p,
        local_dropout_p=args.local_dropout_p,
        global_dropout_p=args.global_dropout_p,
    ).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)

    print(f"\n[INFO] Loaded Checkpoint: {args.ckpt}")
    print(f"[INFO] Static threshold: {args.static_thr:.4f}\n")

    horizons = args.horizons.split(",")
    rows = []

    for hz in horizons:
        hz_path = os.path.join(args.base_root, hz)
        if not os.path.exists(hz_path):
            print(f"[WARN] {hz_path} not found - skipping")
            continue

        print(f"Evaluating {hz} ...")
        loader = make_loader(hz_path, "test", args.dataset,
                             args.batch_size, args.num_workers, args.seq_len)

        for name, ak, al, ag in ABLATIONS:
            with ablation_mode(model, ablate_kin=ak, ablate_local=al, ablate_global=ag):
                labels, probs = collect_fused_predictions(model, loader, device, use_amp)

            n_total = len(labels)
            n_pos   = int(labels.sum())
            n_neg   = n_total - n_pos
            auc = safe_auc(labels, probs)
            s_f1, s_prec, s_rec, s_acc = calc_metrics(labels, probs, args.static_thr)
            t_thr, _ = best_f1_threshold(labels, probs)
            t_f1, t_prec, t_rec, t_acc = calc_metrics(labels, probs, t_thr)

            rows.append({
                "horizon": hz, "ablation": name,
                "ablate_kin": ak, "ablate_local": al, "ablate_global": ag,
                "n": n_total, "n_pos": n_pos, "n_neg": n_neg,
                "auc": auc,
                "static_thr": args.static_thr,
                "static_f1": s_f1, "static_prec": s_prec,
                "static_rec": s_rec, "static_acc": s_acc,
                "tuned_thr": t_thr,
                "tuned_f1": t_f1, "tuned_prec": t_prec,
                "tuned_rec": t_rec, "tuned_acc": t_acc,
            })
            print(f"   [{name:<11}] AUC={auc:.4f}  StaticF1={s_f1:.4f}  TunedF1={t_f1:.4f}")

    # Pretty print
    print("\n" + "=" * 120)
    print(f"{'BRANCH-ZEROING ABLATION ACROSS HORIZONS':^120}")
    print("=" * 120)
    print(f"{'Horizon':<8} | {'Ablation':<12} | {'N':<4} | {'AUC':<6} || "
          f"{'StaticF1':<8} | {'StaticAcc':<9} || {'TunedThr':<8} | {'TunedF1':<8} | {'TunedAcc':<8}")
    print("-" * 120)
    prev_hz = None
    for r in rows:
        if prev_hz is not None and r["horizon"] != prev_hz:
            print("-" * 120)
        prev_hz = r["horizon"]
        print(
            f"{r['horizon']:<8} | {r['ablation']:<12} | {r['n']:<4} | {r['auc']:.4f} || "
            f"{r['static_f1']:.4f}   | {r['static_acc']:.4f}    || "
            f"{r['tuned_thr']:.3f}    | {r['tuned_f1']:.4f}   | {r['tuned_acc']:.4f}"
        )
    print("=" * 120 + "\n")

    if rows:
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[OK] Wrote {args.out_csv}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()