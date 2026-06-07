"""
eval_compare_strategies.py

Compare multiple training strategies on the same test horizons:
    1. PIE-only baseline (S3)
    2. Combine JAAD+PIE training
    3. JAAD->PIE transfer learning (S2)

For each (strategy, horizon) we report:
    AUC, Static F1, Static Acc, Static Prec, Static Rec, Tuned F1

Outputs:
    - Console table comparing all strategies side by side
    - CSV with full per-strategy per-horizon metrics
    - Optional: separate figure for the paper

USAGE:
    python eval_compare_strategies.py \
        --pie_only_ckpt   checkpoints_v4_best_seed46/stage3_pie_baseline/best_model.pth \
        --combine_ckpt     checkpoints_v4_combine_matched_trial12_seed42/.../best_model.pth \
        --transfer_ckpt   checkpoints_reproduce_transfer_trial12/stage2_pie_transfer/best_model.pth \
        --base_root /Datasets/ETCS \
        --horizons ETC0_5,ETC1,ETC2,ETC3,ETC4 \
        --amp
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
# Data loader
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


# ============================================================================
# Inference
# ============================================================================

@torch.no_grad()
def collect_predictions(model, loader, device, amp=False):
    """Returns labels (N,) and dict of probs per branch."""
    model.eval()
    labels_list = []
    probs = {"fused": [], "kin": [], "local": [], "global": []}

    for batch in loader:
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            out = model(batch, return_aux=True)

        probs["fused"].append(
            torch.sigmoid(out["logit"].squeeze(-1).float()).cpu().numpy())
        probs["kin"].append(
            torch.sigmoid(out["aux_kin"].squeeze(-1).float()).cpu().numpy())
        probs["local"].append(
            torch.sigmoid(out["aux_local"].squeeze(-1).float()).cpu().numpy())
        probs["global"].append(
            torch.sigmoid(out["aux_global"].squeeze(-1).float()).cpu().numpy())
        labels_list.append(batch["label"].cpu().numpy().astype(np.int32))

    labels = np.concatenate(labels_list)
    probs = {k: np.concatenate(v) for k, v in probs.items()}
    return labels, probs


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
    preds = (p >= thr).astype(np.int32)
    return dict(
        auc=safe_auc(y, p),
        f1=float(f1_score(y, preds, zero_division=0)),
        prec=float(precision_score(y, preds, zero_division=0)),
        rec=float(recall_score(y, preds, zero_division=0)),
        acc=float(accuracy_score(y, preds)),
    )


# ============================================================================
# Build/load model
# ============================================================================

def load_model(ckpt_path, args, device):
    model = PIPNetAlphaV4Final(
        dropout_p=args.dropout_p,
        local_dropout_p=args.local_dropout_p,
        global_dropout_p=args.global_dropout_p,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    return model


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()

    # The three strategies; pass empty string to skip one
    ap.add_argument("--pie_only_ckpt", type=str, default="",
                    help="PIE-only baseline (S3) checkpoint.")
    ap.add_argument("--combine_ckpt", type=str, default="",
                    help="Combine JAAD+PIE checkpoint.")
    ap.add_argument("--transfer_ckpt", type=str, default="",
                    help="JAAD->PIE transfer (S2) checkpoint.")

    ap.add_argument("--base_root", type=str, default="/Datasets/ETCS")
    ap.add_argument("--horizons", type=str, default="ETC0_5,ETC1,ETC2,ETC3,ETC4")
    ap.add_argument("--dataset", type=str, default="pie", choices=["pie", "jaad"])
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seq_len", type=int, default=10)
    ap.add_argument("--amp", action="store_true")

    ap.add_argument("--static_thr", type=float, default=0.7585)
    ap.add_argument("--out_csv", type=str, default="compare_strategies.csv")

    # Model dropouts (must match training)
    ap.add_argument("--dropout_p",        type=float, default=0.2)
    ap.add_argument("--local_dropout_p",  type=float, default=0.1)
    ap.add_argument("--global_dropout_p", type=float, default=0.4)
    args = ap.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and args.amp

    # Build the list of strategies to evaluate
    strategies = []
    if args.pie_only_ckpt:
        strategies.append(("PIE-only",  args.pie_only_ckpt))
    if args.transfer_ckpt:
        strategies.append(("Transfer",  args.transfer_ckpt))
    if args.combine_ckpt:
        strategies.append(("Combine",    args.combine_ckpt))

    if not strategies:
        raise ValueError("Provide at least one checkpoint flag.")

    print("\n" + "=" * 100)
    print("TRAINING-STRATEGY COMPARISON ACROSS TTE HORIZONS")
    print("=" * 100)
    for name, ckpt in strategies:
        print(f"  {name:<10}: {ckpt}")
    print(f"  Horizons:    {args.horizons}")
    print(f"  Base root:   {args.base_root}")
    print(f"  Static thr:  {args.static_thr}")
    print("=" * 100)

    horizons = args.horizons.split(",")
    rows = []  # one row per (strategy, horizon)

    # Iterate strategy first (so we load each model once)
    for strat_name, ckpt_path in strategies:
        print(f"\n>>> Loading {strat_name}  from  {ckpt_path}")
        model = load_model(ckpt_path, args, device)

        for hz in horizons:
            hz_path = os.path.join(args.base_root, hz)
            if not os.path.exists(hz_path):
                print(f"  [WARN] {hz_path} not found - skipping")
                continue

            print(f"  [{strat_name}] {hz} ...", end=" ", flush=True)
            loader = make_loader(hz_path, "test", args.dataset,
                                 args.batch_size, args.num_workers, args.seq_len)
            labels, probs = collect_predictions(model, loader, device, use_amp)

            # Metrics for fused branch only (the headline number)
            p = probs["fused"]
            m_static = calc_metrics(labels, p, args.static_thr)
            t_thr, _ = best_f1_threshold(labels, p)
            m_tuned  = calc_metrics(labels, p, t_thr)

            row = dict(
                strategy=strat_name, horizon=hz, n=len(labels),
                n_pos=int(labels.sum()),
                auc=m_static["auc"],
                static_thr=args.static_thr,
                static_f1=m_static["f1"], static_prec=m_static["prec"],
                static_rec=m_static["rec"], static_acc=m_static["acc"],
                tuned_thr=t_thr,
                tuned_f1=m_tuned["f1"], tuned_prec=m_tuned["prec"],
                tuned_rec=m_tuned["rec"], tuned_acc=m_tuned["acc"],
                # Per-branch AUCs for diagnostics
                auc_kin=safe_auc(labels, probs["kin"]),
                auc_local=safe_auc(labels, probs["local"]),
                auc_global=safe_auc(labels, probs["global"]),
            )
            rows.append(row)
            print(f"AUC={m_static['auc']:.4f}  staticF1={m_static['f1']:.4f}  "
                  f"tunedF1={m_tuned['f1']:.4f}")

        # Free GPU memory before next strategy
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------ Pretty table
    print("\n" + "=" * 110)
    print(f"{'FUSED-BRANCH PERFORMANCE ACROSS STRATEGIES AND HORIZONS':^110}")
    print("=" * 110)
    print(f"{'Strategy':<10} | {'Horizon':<8} | {'N':<4} | {'AUC':<7} | "
          f"{'StaticF1':<9} | {'StaticAcc':<10} | {'TunedF1':<8} | "
          f"{'AUC_kin':<8} | {'AUC_global':<10}")
    print("-" * 110)

    prev_strat = None
    for r in rows:
        if prev_strat is not None and r["strategy"] != prev_strat:
            print("-" * 110)
        prev_strat = r["strategy"]
        print(
            f"{r['strategy']:<10} | {r['horizon']:<8} | {r['n']:<4} | "
            f"{r['auc']:.4f}  | {r['static_f1']:.4f}    | "
            f"{r['static_acc']:.4f}     | {r['tuned_f1']:.4f}   | "
            f"{r['auc_kin']:.4f}   | {r['auc_global']:.4f}"
        )
    print("=" * 110)

    # ------------------------------------------------------- Side-by-side AUC
    if len(strategies) > 1:
        print("\n" + "=" * 90)
        print(f"{'SIDE-BY-SIDE COMPARISON (Fused AUC)':^90}")
        print("=" * 90)
        header = f"{'Horizon':<8} | " + " | ".join(
            f"{name:<14}" for name, _ in strategies
        )
        print(header)
        print("-" * 90)
        for hz in horizons:
            cells = [f"{hz:<8}"]
            for name, _ in strategies:
                hit = [r for r in rows if r["strategy"] == name and r["horizon"] == hz]
                if hit:
                    cells.append(f"{hit[0]['auc']:.4f}        ")
                else:
                    cells.append("---           ")
            print(" | ".join(cells))
        print("=" * 90)

        # Average performance across horizons
        print("\nAVERAGE FUSED AUC (across all horizons):")
        for name, _ in strategies:
            aucs = [r["auc"] for r in rows if r["strategy"] == name]
            if aucs:
                print(f"  {name:<10}: {np.mean(aucs):.4f}  (range "
                      f"{min(aucs):.4f}--{max(aucs):.4f})")

        print("\nAVERAGE STATIC F1 (across all horizons):")
        for name, _ in strategies:
            f1s = [r["static_f1"] for r in rows if r["strategy"] == name]
            if f1s:
                print(f"  {name:<10}: {np.mean(f1s):.4f}")

    # ------------------------------------------------------------ CSV output
    if rows:
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[OK] Wrote {args.out_csv}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()