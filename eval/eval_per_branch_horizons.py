"""
eval_per_branch_horizons.py

Extends eval_per_branch.py to extract per-branch predictions (kin / local / global / fused)
across multiple TTE horizons (ETC0.5, ETC1, ETC2, ETC3, ETC4).

For each horizon and each branch we report:
  - AUC
  - F1 / Precision / Recall / Accuracy at the STATIC validation threshold (default 0.7585)
  - F1 / Precision / Recall / Accuracy at the per-branch ORACLE threshold (max F1 on test)

Outputs:
  - Pretty console table
  - CSV file with all per-branch metrics

This script ASSUMES the model is `PIPNetAlphaV4Final` and that `forward(batch, return_aux=True)`
returns 'logit', 'aux_kin', 'aux_local', 'aux_global'. No retraining required.
"""

import argparse
import csv
import os
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
# Data loader (identical to your eval_per_branch.py)
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
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )


# ============================================================================
# Branch collection
# ============================================================================

@torch.no_grad()
def collect_branch_predictions(model, loader, device, amp=False):
    """
    Collect probabilities for all four outputs:
        p_fused  = sigmoid(model.logit)         -- main fused prediction
        p_kin    = sigmoid(model.aux_kin)       -- kinematic branch only
        p_local  = sigmoid(model.aux_local)     -- local visual branch only
        p_global = sigmoid(model.aux_global)    -- global context branch only
    Returns:
        labels:   (N,)   int32
        probs:    dict { 'fused': (N,), 'kin': (N,), 'local': (N,), 'global': (N,) }
    """
    model.eval()
    labels_list = []
    probs = {"fused": [], "kin": [], "local": [], "global": []}

    for batch in loader:
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            out = model(batch, return_aux=True)

        # Fused (main) prediction
        p_fused = torch.sigmoid(out["logit"].squeeze(-1).float()).cpu().numpy()
        # Per-branch auxiliary predictions
        p_kin    = torch.sigmoid(out["aux_kin"].squeeze(-1).float()).cpu().numpy()
        p_local  = torch.sigmoid(out["aux_local"].squeeze(-1).float()).cpu().numpy()
        p_global = torch.sigmoid(out["aux_global"].squeeze(-1).float()).cpu().numpy()

        probs["fused"].append(p_fused)
        probs["kin"].append(p_kin)
        probs["local"].append(p_local)
        probs["global"].append(p_global)
        labels_list.append(batch["label"].cpu().numpy().astype(np.int32))

    labels = np.concatenate(labels_list)
    probs = {k: np.concatenate(v) for k, v in probs.items()}
    return labels, probs


# ============================================================================
# Metric helpers
# ============================================================================

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
    ap.add_argument("--static_thr", type=float, default=0.7585,
                    help="Static threshold tuned on val@1s for the fused branch.")
    ap.add_argument("--out_csv", type=str, default="per_branch_horizons.csv")

    # Model params (must match training)
    ap.add_argument("--dropout_p",        type=float, default=0.2)
    ap.add_argument("--local_dropout_p",  type=float, default=0.1)
    ap.add_argument("--global_dropout_p", type=float, default=0.4)
    args = ap.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and args.amp

    # ------------------------------------------------------------- Load model
    model = PIPNetAlphaV4Final(
        dropout_p=args.dropout_p,
        local_dropout_p=args.local_dropout_p,
        global_dropout_p=args.global_dropout_p,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)

    print(f"\n[INFO] Loaded Checkpoint: {args.ckpt}")
    print(f"[INFO] Static threshold (val-tuned on fused @1s): {args.static_thr:.4f}\n")

    # ------------------------------------------------------------ Run horizons
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
        labels, probs = collect_branch_predictions(model, loader, device, use_amp)

        n_total = len(labels)
        n_pos   = int(labels.sum())
        n_neg   = n_total - n_pos
        if n_total == 0:
            print(f"   -> empty test set; skipping")
            continue

        for branch in ["fused", "kin", "local", "global"]:
            p = probs[branch]
            auc = safe_auc(labels, p)

            # Static threshold: only meaningful for fused (it was calibrated there).
            # For aux branches we still report it for completeness, BUT also report
            # the per-branch oracle threshold which is the fair number.
            s_f1, s_prec, s_rec, s_acc = calc_metrics(labels, p, args.static_thr)
            t_thr, _ = best_f1_threshold(labels, p)
            t_f1, t_prec, t_rec, t_acc = calc_metrics(labels, p, t_thr)

            rows.append({
                "horizon": hz, "branch": branch,
                "n": n_total, "n_pos": n_pos, "n_neg": n_neg,
                "auc": auc,
                "static_thr": args.static_thr,
                "static_f1": s_f1, "static_prec": s_prec,
                "static_rec": s_rec, "static_acc": s_acc,
                "tuned_thr": t_thr,
                "tuned_f1": t_f1, "tuned_prec": t_prec,
                "tuned_rec": t_rec, "tuned_acc": t_acc,
            })

    # ------------------------------------------------------------ Pretty print
    print("\n" + "=" * 130)
    print(f"{'PER-BRANCH PERFORMANCE ACROSS HORIZONS':^130}")
    print("=" * 130)
    print(f"{'Horizon':<8} | {'Branch':<7} | {'N':<4} | {'AUC':<6} || "
          f"{'STATIC (thr=' + f'{args.static_thr:.3f}' + ')':^32} || "
          f"{'TUNED (oracle F1)':^32}")
    print(f"{'':<8} | {'':<7} | {'':<4} | {'':<6} || "
          f"{'Acc':<7} | {'F1':<6} | {'Prec':<6} | {'Rec':<5} || "
          f"{'Thr':<5} | {'F1':<6} | {'Prec':<6} | {'Rec':<5}")
    print("-" * 130)

    prev_hz = None
    for r in rows:
        if prev_hz is not None and r["horizon"] != prev_hz:
            print("-" * 130)
        prev_hz = r["horizon"]
        print(
            f"{r['horizon']:<8} | {r['branch']:<7} | {r['n']:<4} | "
            f"{r['auc']:.4f} || "
            f"{r['static_acc']:.4f} | {r['static_f1']:.4f} | "
            f"{r['static_prec']:.4f} | {r['static_rec']:.4f} || "
            f"{r['tuned_thr']:.3f} | {r['tuned_f1']:.4f} | "
            f"{r['tuned_prec']:.4f} | {r['tuned_rec']:.4f}"
        )
    print("=" * 130 + "\n")

    # ------------------------------------------------------------ Write CSV
    if rows:
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[OK] Wrote {args.out_csv}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()