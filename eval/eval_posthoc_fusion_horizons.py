"""
eval_posthoc_fusion_horizons.py

Post-hoc probability-level fusion of branch outputs, calibrated on the FULL
validation set (1-2s TTE range) and frozen across all test horizons.

Two fusion variants are evaluated:

    A) Branch-only fusion (pure aux-head ensemble):
       p_posthoc_A = w_kin*p_kin + w_local*p_local + w_global*p_global
       with w_kin + w_local + w_global = 1

    B) Native + branches fusion (as in original spec):
       p_posthoc_B = w_fused*p_fused + w_kin*p_kin + w_local*p_local + w_global*p_global
       with all four summing to 1

Both grids are searched on the FULL validation set. Best weights and best
threshold are SELECTED ON VAL ONLY, then FROZEN and applied to every test
horizon. No test-set leakage anywhere.

The native learned fusion (p_fused with --native_thr) is also reported for
direct comparison.

CSV columns include results for variant A, variant B, and native fusion.
"""

import argparse
import csv
import os
import itertools
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    precision_recall_curve, accuracy_score, average_precision_score,
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
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )


# ============================================================================
# Inference: collect probabilities AND tte_sec_actual
# ============================================================================

@torch.no_grad()
def collect_all(model, loader, device, amp=False):
    """
    Returns:
        labels: (N,) int32
        probs:  dict { 'fused', 'kin', 'local', 'global' } each (N,)
        tte:    (N,) float32  (-1 if unavailable)
    """
    model.eval()
    labels_list, tte_list = [], []
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

        meta = batch.get("meta", {})
        if isinstance(meta, dict) and "tte_sec_actual" in meta:
            tte_arr = np.asarray(meta["tte_sec_actual"]).reshape(-1).astype(np.float32)
        else:
            tte_arr = np.full(batch["label"].shape[0], -1.0, dtype=np.float32)
        tte_list.append(tte_arr)

    labels = np.concatenate(labels_list)
    probs = {k: np.concatenate(v) for k, v in probs.items()}
    tte = np.concatenate(tte_list)
    return labels, probs, tte


# ============================================================================
# Metrics
# ============================================================================

def safe_auc(y, p):
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def safe_ap(y, p):
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


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
        return dict(f1=0.0, prec=0.0, rec=0.0, acc=0.0, auc=float("nan"), ap=float("nan"))
    preds = (p >= thr).astype(np.int32)
    return dict(
        f1   = float(f1_score(y, preds, zero_division=0)),
        prec = float(precision_score(y, preds, zero_division=0)),
        rec  = float(recall_score(y, preds, zero_division=0)),
        acc  = float(accuracy_score(y, preds)),
        auc  = safe_auc(y, p),
        ap   = safe_ap(y, p),
    )


# ============================================================================
# Grid search for fusion weights
# ============================================================================

def enumerate_simplex(n_weights, step):
    """
    Generate all weight tuples summing to 1.0 with given step.
    For n=3, step=0.05 -> 231 candidates.
    For n=4, step=0.05 -> 1771 candidates.
    """
    n_steps = int(round(1.0 / step))
    out = []
    if n_weights == 3:
        for i in range(n_steps + 1):
            for j in range(n_steps + 1 - i):
                k = n_steps - i - j
                if k < 0:
                    continue
                out.append((i * step, j * step, k * step))
    elif n_weights == 4:
        for i in range(n_steps + 1):
            for j in range(n_steps + 1 - i):
                for k in range(n_steps + 1 - i - j):
                    l = n_steps - i - j - k
                    if l < 0:
                        continue
                    out.append((i * step, j * step, k * step, l * step))
    else:
        raise ValueError(f"Unsupported n_weights={n_weights}")
    return out


def grid_search_fusion(probs_dict, labels, branches, step):
    """
    probs_dict: dict with at least the keys in `branches`
    branches:   list, e.g. ['kin','local','global'] or ['fused','kin','local','global']
    step:       float, grid resolution

    Returns dict with best_weights, best_threshold, val_metrics, n_evaluated
    """
    candidates = enumerate_simplex(len(branches), step)

    best = dict(weights=None, threshold=0.5, f1=-1.0)
    arrs = [probs_dict[b] for b in branches]

    for combo in candidates:
        p = np.zeros_like(arrs[0])
        for w, a in zip(combo, arrs):
            p = p + w * a
        thr, f1 = best_f1_threshold(labels, p)
        if f1 > best["f1"]:
            best = dict(weights=tuple(combo), threshold=thr, f1=f1)

    # Compute full metrics at the chosen weights+threshold
    p_best = np.zeros_like(arrs[0])
    for w, a in zip(best["weights"], arrs):
        p_best = p_best + w * a
    m = calc_metrics(labels, p_best, best["threshold"])
    return dict(
        weights=dict(zip(branches, best["weights"])),
        threshold=best["threshold"],
        metrics=m,
        n_candidates=len(candidates),
    )


def apply_fusion(probs_dict, weights_dict):
    """Apply frozen weights to a probs dict. Returns (N,) array."""
    branches = list(weights_dict.keys())
    arrs = [probs_dict[b] for b in branches]
    p = np.zeros_like(arrs[0])
    for b, a in zip(branches, arrs):
        p = p + weights_dict[b] * a
    return p


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val_root", required=True,
                    help="Root containing /val with TTE 1-2s samples.")
    ap.add_argument("--base_root", type=str, default="/Datasets/PIE_PREP_OUT/ETCs")
    ap.add_argument("--horizons", type=str, default="ETC0_5,ETC1,ETC2,ETC3,ETC4")
    ap.add_argument("--dataset", type=str, default="pie", choices=["pie", "jaad"])

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seq_len", type=int, default=10)
    ap.add_argument("--amp", action="store_true")

    ap.add_argument("--native_thr", type=float, default=0.7585,
                    help="Threshold for native learned fusion (val-tuned).")
    ap.add_argument("--weight_step", type=float, default=0.05,
                    help="Grid step for fusion weights.")

    ap.add_argument("--out_csv", type=str, default="posthoc_fusion_results.csv")

    # Model dropouts
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

    print("\n" + "=" * 80)
    print("POST-HOC PROBABILITY FUSION  -  FULL VAL CALIBRATION  +  FROZEN TEST EVAL")
    print("=" * 80)
    print(f"Checkpoint:        {args.ckpt}")
    print(f"Validation root:   {args.val_root}")
    print(f"Test base root:    {args.base_root}")
    print(f"Horizons:          {args.horizons}")
    print(f"Native threshold:  {args.native_thr:.4f}")
    print(f"Grid step:         {args.weight_step}")
    print("=" * 80)

    # ----------------------------------------------- Step 1: collect val preds
    print("\n[Step 1/3] Collecting predictions on validation set (1-2s TTE)...")
    val_loader = make_loader(args.val_root, "val", args.dataset,
                             args.batch_size, args.num_workers, args.seq_len)
    val_y, val_probs, val_tte = collect_all(model, val_loader, device, use_amp)

    n_val = len(val_y)
    n_pos = int(val_y.sum())
    valid_tte = val_tte[val_tte > 0]
    print(f"  Total val samples: {n_val}  (pos={n_pos}, neg={n_val - n_pos})")
    if len(valid_tte) > 0:
        print(f"  TTE range: [{valid_tte.min():.3f}, {valid_tte.max():.3f}] s"
              f"  mean={valid_tte.mean():.3f}s")

    # ----------------------------------------- Step 2: grid-search both variants
    print(f"\n[Step 2/3] Grid search on FULL validation set (step={args.weight_step})...")

    # Variant A: branch-only (3 weights)
    print("  Variant A: branch-only fusion (kin + local + global)")
    res_A = grid_search_fusion(
        val_probs, val_y,
        branches=["kin", "local", "global"],
        step=args.weight_step,
    )
    wA = res_A["weights"]
    print(f"    Candidates evaluated: {res_A['n_candidates']}")
    print(f"    Best weights:  kin={wA['kin']:.2f}  local={wA['local']:.2f}  global={wA['global']:.2f}")
    print(f"    Best threshold: {res_A['threshold']:.4f}")
    mA = res_A["metrics"]
    print(f"    Val: AUC={mA['auc']:.4f}  AP={mA['ap']:.4f}  "
          f"F1={mA['f1']:.4f}  Acc={mA['acc']:.4f}  "
          f"Prec={mA['prec']:.4f}  Rec={mA['rec']:.4f}")

    # Variant B: native + branches (4 weights)
    print("  Variant B: native + branches fusion (fused + kin + local + global)")
    res_B = grid_search_fusion(
        val_probs, val_y,
        branches=["fused", "kin", "local", "global"],
        step=args.weight_step,
    )
    wB = res_B["weights"]
    print(f"    Candidates evaluated: {res_B['n_candidates']}")
    print(f"    Best weights:  fused={wB['fused']:.2f}  kin={wB['kin']:.2f}  "
          f"local={wB['local']:.2f}  global={wB['global']:.2f}")
    print(f"    Best threshold: {res_B['threshold']:.4f}")
    mB = res_B["metrics"]
    print(f"    Val: AUC={mB['auc']:.4f}  AP={mB['ap']:.4f}  "
          f"F1={mB['f1']:.4f}  Acc={mB['acc']:.4f}  "
          f"Prec={mB['prec']:.4f}  Rec={mB['rec']:.4f}")

    # Native fusion val performance (for comparison)
    mN_val = calc_metrics(val_y, val_probs["fused"], args.native_thr)
    print(f"\n  Native fusion @ thr={args.native_thr:.4f} on FULL val:")
    print(f"    Val: AUC={mN_val['auc']:.4f}  AP={mN_val['ap']:.4f}  "
          f"F1={mN_val['f1']:.4f}  Acc={mN_val['acc']:.4f}  "
          f"Prec={mN_val['prec']:.4f}  Rec={mN_val['rec']:.4f}")

    # ------------------------------------------- Step 3: evaluate on test horizons
    print(f"\n[Step 3/3] Evaluating on test horizons with FROZEN weights+threshold...\n")

    horizons = args.horizons.split(",")
    rows = []

    for hz in horizons:
        hz_path = os.path.join(args.base_root, hz)
        if not os.path.exists(hz_path):
            print(f"  [WARN] {hz_path} not found - skipping")
            continue

        print(f"  Evaluating {hz} ...")
        loader = make_loader(hz_path, "test", args.dataset,
                             args.batch_size, args.num_workers, args.seq_len)
        labels, probs, _ = collect_all(model, loader, device, use_amp)

        if len(labels) == 0:
            print(f"    [WARN] empty test set; skipping")
            continue

        # Native fusion (frozen threshold)
        mN = calc_metrics(labels, probs["fused"], args.native_thr)
        # Variant A (frozen weights + frozen threshold)
        pA = apply_fusion(probs, wA)
        mA_test = calc_metrics(labels, pA, res_A["threshold"])
        # Variant B
        pB = apply_fusion(probs, wB)
        mB_test = calc_metrics(labels, pB, res_B["threshold"])

        rows.append(dict(
            horizon=hz, n=len(labels), n_pos=int(labels.sum()),
            n_neg=int((labels == 0).sum()),
            # Native
            native_thr=args.native_thr,
            native_auc=mN["auc"], native_ap=mN["ap"],
            native_acc=mN["acc"], native_f1=mN["f1"],
            native_prec=mN["prec"], native_rec=mN["rec"],
            # Variant A
            A_w_kin=wA["kin"], A_w_local=wA["local"], A_w_global=wA["global"],
            A_thr=res_A["threshold"],
            A_auc=mA_test["auc"], A_ap=mA_test["ap"],
            A_acc=mA_test["acc"], A_f1=mA_test["f1"],
            A_prec=mA_test["prec"], A_rec=mA_test["rec"],
            # Variant B
            B_w_fused=wB["fused"], B_w_kin=wB["kin"],
            B_w_local=wB["local"], B_w_global=wB["global"],
            B_thr=res_B["threshold"],
            B_auc=mB_test["auc"], B_ap=mB_test["ap"],
            B_acc=mB_test["acc"], B_f1=mB_test["f1"],
            B_prec=mB_test["prec"], B_rec=mB_test["rec"],
            # Val calibration metadata
            val_n=n_val, val_n_pos=n_pos,
        ))

        print(f"    Native:     AUC={mN['auc']:.4f}  F1={mN['f1']:.4f}  Acc={mN['acc']:.4f}")
        print(f"    Variant A:  AUC={mA_test['auc']:.4f}  F1={mA_test['f1']:.4f}  Acc={mA_test['acc']:.4f}")
        print(f"    Variant B:  AUC={mB_test['auc']:.4f}  F1={mB_test['f1']:.4f}  Acc={mB_test['acc']:.4f}")

    # ------------------------------------------------ Pretty summary table
    print("\n" + "=" * 130)
    print(f"{'FROZEN-FUSION TEST PERFORMANCE ACROSS HORIZONS':^130}")
    print("=" * 130)
    print(f"{'Horizon':<8} | {'N':<4} | "
          f"{'Native (fused only)':^32} | "
          f"{'Variant A (3-branch)':^32} | "
          f"{'Variant B (native+3)':^32}")
    print(f"{'':<8} | {'':<4} | "
          f"{'AUC':<6} {'F1':<6} {'Acc':<6} {'Rec':<6} | "
          f"{'AUC':<6} {'F1':<6} {'Acc':<6} {'Rec':<6} | "
          f"{'AUC':<6} {'F1':<6} {'Acc':<6} {'Rec':<6}")
    print("-" * 130)
    for r in rows:
        print(
            f"{r['horizon']:<8} | {r['n']:<4} | "
            f"{r['native_auc']:.4f} {r['native_f1']:.4f} {r['native_acc']:.4f} {r['native_rec']:.4f} | "
            f"{r['A_auc']:.4f} {r['A_f1']:.4f} {r['A_acc']:.4f} {r['A_rec']:.4f} | "
            f"{r['B_auc']:.4f} {r['B_f1']:.4f} {r['B_acc']:.4f} {r['B_rec']:.4f}"
        )
    print("=" * 130)

    # ------------------------------------------------ CSV output
    if rows:
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[OK] Wrote {args.out_csv}  ({len(rows)} rows)")

    # ------------------------------------------------ Interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print("Calibration: full validation set (TTE 1-2s) — frozen on all test horizons.")
    print()
    print(f"Variant A weights (kin + local + global = 1):")
    print(f"    kin    = {wA['kin']:.2f}")
    print(f"    local  = {wA['local']:.2f}")
    print(f"    global = {wA['global']:.2f}")
    if wA["local"] == 0.0:
        print("    -> Local branch received zero weight: redundant given kin+global.")
    if wA["global"] == 0.0:
        print("    -> Global branch received zero weight on validation.")
    if wA["kin"] == 0.0:
        print("    -> Kin branch received zero weight (unusual).")
    print()
    print(f"Variant B weights (fused + kin + local + global = 1):")
    print(f"    fused  = {wB['fused']:.2f}")
    print(f"    kin    = {wB['kin']:.2f}")
    print(f"    local  = {wB['local']:.2f}")
    print(f"    global = {wB['global']:.2f}")

    # Improvement summary
    print("\nPer-horizon improvement (frozen Variant A vs native, F1):")
    for r in rows:
        d_A = r["A_f1"] - r["native_f1"]
        d_B = r["B_f1"] - r["native_f1"]
        sign_A = "+" if d_A >= 0 else ""
        sign_B = "+" if d_B >= 0 else ""
        print(f"    {r['horizon']:<8}: A={sign_A}{d_A:+.4f}   B={sign_B}{d_B:+.4f}")

    print()
    print("Reminder: weights and thresholds were selected ONLY on the validation set.")
    print("Test horizons used the same frozen weights/threshold — no per-horizon tuning.")


if __name__ == "__main__":
    main()