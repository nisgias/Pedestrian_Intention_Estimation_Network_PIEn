"""
eval_compare_strategies_jaad.py

Cross-dataset evaluation: test all three training strategies on the JAAD
val and test splits.

Each model is evaluated as a PEDESTRIAN INTENT PREDICTOR on JAAD, with
JAAD-native statistics for speed/motion normalization. This is the standard
cross-dataset protocol used in the literature.

Strategies compared:
    1. PIE-only (S3)       - trained only on PIE
    2. Transfer (S2)       - JAAD pretrain -> PIE fine-tune
    3. Pooled (S0)         - JAAD + PIE mixed training

For each (strategy, split) we report:
    AUC, Static F1 / Acc / Prec / Rec, Tuned F1, per-branch AUCs
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
# Data loader — note that we explicitly pass dataset_prefix to pick correct stats
# ============================================================================

def make_loader(root, split, dataset_prefix, batch_size, num_workers, seq_len):
    """
    `dataset_prefix` selects which speed/motion stats JSON to use.
    For JAAD test set -> use "jaad" stats files.
    """
    ds = PIESeqDataset(
        root, split=split, mode="eval", seq_len=seq_len,
        strict_len=True, return_meta=True,
        speed_norm="minmax",
        speed_stats_path=f"/workspace/project/data/{dataset_prefix}_speed_stats_splits.json",
        speed_scope="global",
        motion_norm="p99abs",
        motion_stats_path=f"/workspace/project/data/{dataset_prefix}_motion_stats_splits.json",
        motion_scope="global", motion_clip=1.0,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


# ============================================================================
# Inference
# ============================================================================

@torch.no_grad()
def collect_predictions(model, loader, device, amp=False):
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

    ap.add_argument("--pie_only_ckpt", type=str, default="")
    ap.add_argument("--pooled_ckpt",   type=str, default="")
    ap.add_argument("--transfer_ckpt", type=str, default="")

    ap.add_argument("--jaad_root", type=str, required=True,
                    help="Root containing JAAD /val and /test splits.")
    ap.add_argument("--splits", type=str, default="val,test",
                    help="Comma-separated JAAD splits to evaluate.")
    ap.add_argument("--stats_prefix", type=str, default="jaad",
                    help="Stats prefix for speed/motion normalization (use 'jaad').")

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seq_len", type=int, default=10)
    ap.add_argument("--amp", action="store_true")

    ap.add_argument("--static_thr", type=float, default=0.7585,
                    help="Threshold used for static F1 (PIE-val calibrated; "
                         "may not be optimal for JAAD).")
    ap.add_argument("--out_csv", type=str, default="compare_strategies_jaad.csv")

    ap.add_argument("--dropout_p",        type=float, default=0.2)
    ap.add_argument("--local_dropout_p",  type=float, default=0.1)
    ap.add_argument("--global_dropout_p", type=float, default=0.4)
    args = ap.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and args.amp

    strategies = []
    if args.pie_only_ckpt:
        strategies.append(("PIE-only",  args.pie_only_ckpt))
    if args.transfer_ckpt:
        strategies.append(("Transfer",  args.transfer_ckpt))
    if args.pooled_ckpt:
        strategies.append(("Pooled",    args.pooled_ckpt))
    if not strategies:
        raise ValueError("Provide at least one checkpoint flag.")

    splits = args.splits.split(",")

    print("\n" + "=" * 100)
    print("CROSS-DATASET EVAL: STRATEGIES ON JAAD SPLITS")
    print("=" * 100)
    for name, ckpt in strategies:
        print(f"  {name:<10}: {ckpt}")
    print(f"  JAAD root:    {args.jaad_root}")
    print(f"  Splits:       {splits}")
    print(f"  Stats prefix: {args.stats_prefix}  (normalization uses {args.stats_prefix}_*_stats_splits.json)")
    print(f"  Static thr:   {args.static_thr}  (NOTE: tuned on PIE val; may be suboptimal for JAAD)")
    print("=" * 100)

    rows = []

    # Iterate strategy first to load each model only once
    for strat_name, ckpt_path in strategies:
        print(f"\n>>> Loading {strat_name}  from  {ckpt_path}")
        model = load_model(ckpt_path, args, device)

        for split in splits:
            split_path = os.path.join(args.jaad_root, split)
            # The PIESeqDataset class itself looks for the split inside the root;
            # we pass jaad_root and let it find /val or /test internally.
            print(f"  [{strat_name}] JAAD/{split} ...", end=" ", flush=True)

            try:
                loader = make_loader(args.jaad_root, split, args.stats_prefix,
                                     args.batch_size, args.num_workers, args.seq_len)
                labels, probs = collect_predictions(model, loader, device, use_amp)
            except FileNotFoundError as e:
                print(f"[ERROR] {e}")
                continue
            except Exception as e:
                print(f"[ERROR] {type(e).__name__}: {e}")
                continue

            if len(labels) == 0:
                print("empty split, skipped")
                continue

            p = probs["fused"]
            m_static = calc_metrics(labels, p, args.static_thr)
            t_thr, _ = best_f1_threshold(labels, p)
            m_tuned  = calc_metrics(labels, p, t_thr)

            row = dict(
                strategy=strat_name, split=f"JAAD/{split}",
                n=len(labels), n_pos=int(labels.sum()),
                n_neg=int((labels == 0).sum()),
                auc=m_static["auc"],
                static_thr=args.static_thr,
                static_f1=m_static["f1"], static_prec=m_static["prec"],
                static_rec=m_static["rec"], static_acc=m_static["acc"],
                tuned_thr=t_thr,
                tuned_f1=m_tuned["f1"], tuned_prec=m_tuned["prec"],
                tuned_rec=m_tuned["rec"], tuned_acc=m_tuned["acc"],
                auc_kin=safe_auc(labels, probs["kin"]),
                auc_local=safe_auc(labels, probs["local"]),
                auc_global=safe_auc(labels, probs["global"]),
            )
            rows.append(row)
            print(f"N={len(labels):4d} pos={int(labels.sum()):3d} "
                  f"AUC={m_static['auc']:.4f}  staticF1={m_static['f1']:.4f}  "
                  f"tunedF1={m_tuned['f1']:.4f}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------ Pretty table
    print("\n" + "=" * 115)
    print(f"{'FUSED-BRANCH PERFORMANCE ON JAAD SPLITS':^115}")
    print("=" * 115)
    print(f"{'Strategy':<10} | {'Split':<11} | {'N':<4} | {'Pos':<4} | "
          f"{'AUC':<7} | {'StaticF1':<9} | {'TunedF1':<8} | "
          f"{'AUC_kin':<8} | {'AUC_local':<10} | {'AUC_global':<10}")
    print("-" * 115)

    prev_strat = None
    for r in rows:
        if prev_strat is not None and r["strategy"] != prev_strat:
            print("-" * 115)
        prev_strat = r["strategy"]
        print(
            f"{r['strategy']:<10} | {r['split']:<11} | {r['n']:<4} | "
            f"{r['n_pos']:<4} | {r['auc']:.4f}  | "
            f"{r['static_f1']:.4f}    | {r['tuned_f1']:.4f}   | "
            f"{r['auc_kin']:.4f}   | {r['auc_local']:.4f}     | "
            f"{r['auc_global']:.4f}"
        )
    print("=" * 115)

    # ------------------------------------------------- Side-by-side per-split
    if len(strategies) > 1:
        print("\n" + "=" * 90)
        print(f"{'SIDE-BY-SIDE COMPARISON (JAAD splits)':^90}")
        print("=" * 90)
        header = f"{'Split':<11} | " + " | ".join(
            f"{name:<14}" for name, _ in strategies
        )
        print(header)
        print("-" * 90)
        for split in splits:
            full_split = f"JAAD/{split}"
            cells = [f"{full_split:<11}"]
            for name, _ in strategies:
                hit = [r for r in rows
                       if r["strategy"] == name and r["split"] == full_split]
                if hit:
                    cells.append(f"AUC={hit[0]['auc']:.4f}   ")
                else:
                    cells.append("---           ")
            print(" | ".join(cells))
        print("=" * 90)

    # ------------------------------------------------------------ CSV output
    if rows:
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[OK] Wrote {args.out_csv}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()