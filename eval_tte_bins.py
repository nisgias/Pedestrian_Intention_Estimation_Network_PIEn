import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
)

from data.pie import PIESeqDataset
from models.pipnet_alpha_v4_wide_compat import PIPNetAlphaV4Final

# Non-overlapping TTE bins. 
TTE_BINS = [
    ("ETC0.5 [0.49-0.55]",       0.49, 0.55),
    ("0.55-1.0s [Preparation]",  0.55, 1.0),
    ("1.0-1.5s [Anticipation A]",1.0,  1.5),
    ("1.5-2.0s [Anticipation B]",1.5,  2.0),
    ("2.0-3.0s [Early A]",       2.0,  3.0),
    ("3.0-4.0s [Early B]",       3.0,  4.0),
    ("ETC4.0 [4.0+ Pre-intent]", 4.0, 999.0),
]

FUSION_WEIGHTS = [
    (1.0, 0.0, 0.0),
    (0.9, 0.05, 0.05),
    (0.8, 0.1, 0.1),
    (0.7, 0.2, 0.1),
    (0.7, 0.1, 0.2),
    (0.6, 0.2, 0.2),
    (0.5, 0.25, 0.25),
]

def make_loader(root, split, dataset, batch_size, num_workers, seq_len):
    ds = PIESeqDataset(
        root,
        split=split,
        mode="eval",
        seq_len=seq_len,
        strict_len=True,
        return_meta=True,
        speed_norm="minmax",
        speed_stats_path=f"/workspace/project/data/{dataset}_speed_stats_splits.json",
        speed_scope="global",
        motion_norm="p99abs",
        motion_stats_path=f"/workspace/project/data/{dataset}_motion_stats_splits.json",
        motion_scope="global",
        motion_clip=1.0,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

@torch.no_grad()
def collect_predictions(model, loader, device, amp=False):
    model.eval()
    labels_list, probs_list, logits_list, tte_list = [], [], [], []
    aux_prob_lists = {"kin": [], "local": [], "global": []}
    aux_logit_lists = {"kin": [], "local": [], "global": []}

    for batch in loader:
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            out = model(batch, return_aux=True)

        main_logit = out["logit"].squeeze(-1).float()
        logits_list.append(main_logit.cpu().numpy())
        probs_list.append(torch.sigmoid(main_logit).cpu().numpy())
        labels_list.append(batch["label"].cpu().numpy().astype(np.int32))

        meta = batch.get("meta", {})
        if isinstance(meta, dict) and "tte_sec_actual" in meta:
            tte_arr = np.array(meta["tte_sec_actual"]).reshape(-1).astype(np.float32)
        else:
            tte_arr = np.full(batch["label"].shape[0], -1.0, dtype=np.float32)
        tte_list.append(tte_arr)

        for out_key, short_key in (("aux_kin", "kin"), ("aux_local", "local"), ("aux_global", "global")):
            if out_key in out:
                aux_logit = out[out_key].squeeze(-1).float()
                aux_logit_lists[short_key].append(aux_logit.cpu().numpy())
                aux_prob_lists[short_key].append(torch.sigmoid(aux_logit).cpu().numpy())

    labels = np.concatenate(labels_list)
    probs = np.concatenate(probs_list)
    logits = np.concatenate(logits_list)
    tte = np.concatenate(tte_list)
    aux_probs = {k: np.concatenate(v) for k, v in aux_prob_lists.items() if v}
    aux_logits = {k: np.concatenate(v) for k, v in aux_logit_lists.items() if v}

    return labels, probs, logits, tte, aux_probs, aux_logits

# ---------------- Metrics ----------------

def safe_auc(y, p):
    y, p = np.asarray(y), np.asarray(p)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))

def best_f1_threshold(y, p):
    y, p = np.asarray(y).astype(np.int32), np.asarray(p).astype(np.float32)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return 0.5, 0.0

    prec, rec, thrs = precision_recall_curve(y, p)
    if len(thrs) == 0:
        return 0.5, 0.0

    f1s = 2.0 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
    best_idx = int(np.argmax(f1s))
    return float(thrs[best_idx]), float(f1s[best_idx])

def bin_metrics(y, p, thr):
    y, p = np.asarray(y).astype(np.int32), np.asarray(p).astype(np.float32)
    if len(y) == 0:
        return dict(n=0, pos=0, neg=0, auc=float("nan"), f1=float("nan"), prec=float("nan"), rec=float("nan"), threshold=float(thr))

    preds = (p >= thr).astype(np.int32)
    return dict(
        n=int(len(y)),
        pos=int(y.sum()),
        neg=int((1 - y).sum()),
        auc=safe_auc(y, p),
        f1=float(f1_score(y, preds, zero_division=0)),
        prec=float(precision_score(y, preds, zero_division=0)),
        rec=float(recall_score(y, preds, zero_division=0)),
        threshold=float(thr),
    )

# ---------------- Formatting ----------------

COL = 30

def header():
    cols = ["Bin/Metric", "N", "Pos%", "AUC", "F1", "Prec", "Rec", "Thr"]
    widths = [COL, 6, 7, 8, 8, 8, 8, 8]
    row = "".join(c.ljust(w) for c, w in zip(cols, widths))
    sep = "-" * len(row)
    return f"\n{sep}\n{row}\n{sep}"

def fmt_row(label, m):
    if m["n"] == 0:
        return f"{label:<{COL}}{0:>6}  {'-':>7}  {'-':>8}  {'-':>8}  {'-':>8}  {'-':>8}  {'-':>8}"
    pos_pct = 100.0 * m["pos"] / max(m["n"], 1)
    return "  ".join([
        f"{label:<{COL}}", f"{m['n']:>6}", f"{pos_pct:>7.1f}",
        f"{m['auc']:>8.4f}" if not np.isnan(m["auc"]) else f"{'nan':>8}",
        f"{m['f1']:>8.4f}", f"{m['prec']:>8.4f}", f"{m['rec']:>8.4f}", f"{m['threshold']:>8.3f}",
    ])

def print_metric_table(title, rows):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    print(header())
    for label, m in rows:
        print(fmt_row(label, m))

# ---------------- Late fusion ----------------

def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float32)))

def evaluate_late_fusion(test_labels, test_aux_logits, static_thr):
    static_rows, tuned_rows = [], []
    required = {"kin", "local", "global"}
    if not required.issubset(test_aux_logits.keys()):
        return static_rows, tuned_rows

    for weights in FUSION_WEIGHTS:
        label = f"w={weights[0]:.2f},{weights[1]:.2f},{weights[2]:.2f}"
        score = sigmoid_np(weights[0]*test_aux_logits["kin"] + weights[1]*test_aux_logits["local"] + weights[2]*test_aux_logits["global"])
        
        # Fixed
        static_rows.append((label, bin_metrics(test_labels, score, static_thr)))
        # Tuned
        tuned_thr, _ = best_f1_threshold(test_labels, score)
        tuned_rows.append((label, bin_metrics(test_labels, score, tuned_thr)))

    return static_rows, tuned_rows

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--dataset", required=True, choices=["pie", "jaad"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seq_len", type=int, default=10)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--dropout_p", type=float, default=0.5)
    ap.add_argument("--local_dropout_p", type=float, default=0.3)
    ap.add_argument("--global_dropout_p", type=float, default=0.2)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and args.amp

    model = PIPNetAlphaV4Final(
        dropout_p=args.dropout_p,
        local_dropout_p=args.local_dropout_p,
        global_dropout_p=args.global_dropout_p,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    
    # Extract threshold saved in checkpoint (Fallback to 0.5)
    ckpt_thr = 0.5
    if "best_thr" in ckpt:
        ckpt_thr = ckpt["best_thr"]
    elif "val_metrics" in ckpt and "best_thr" in ckpt["val_metrics"]:
        ckpt_thr = ckpt["val_metrics"]["best_thr"]

    print(f"Loaded  : {args.ckpt}")
    print(f"Dataset : {args.dataset.upper()}  root={args.data_root}")
    print(f"Extracted Checkpoint Thr: {ckpt_thr:.3f}")

    print("\nRunning inference purely on test split ...")
    test_loader = make_loader(args.data_root, "test", args.dataset, args.batch_size, args.num_workers, args.seq_len)
    (test_labels, test_probs, test_logits, test_tte, test_aux_probs, test_aux_logits) = collect_predictions(model, test_loader, device, use_amp)

    # 1. Global Metrics
    global_static_m = bin_metrics(test_labels, test_probs, ckpt_thr)
    best_tuned_thr, _ = best_f1_threshold(test_labels, test_probs)
    global_tuned_m = bin_metrics(test_labels, test_probs, best_tuned_thr)

    print_metric_table("1. GLOBAL METRICS", [
        (f"Fixed Thr ({ckpt_thr:.3f})", global_static_m),
        (f"Test-Tuned Thr ({best_tuned_thr:.3f})", global_tuned_m)
    ])

    # 2. Aux Metrics
    aux_static_rows, aux_tuned_rows = [], []
    for k, p in test_aux_probs.items():
        aux_static_rows.append((f"{k} (Fixed Thr)", bin_metrics(test_labels, p, ckpt_thr)))
        t_thr, _ = best_f1_threshold(test_labels, p)
        aux_tuned_rows.append((f"{k} (Tuned Thr)", bin_metrics(test_labels, p, t_thr)))

    print_metric_table("2A. AUX BRANCHES (STATIC CHECKPOINT THRESHOLD)", aux_static_rows)
    print_metric_table("2B. AUX BRANCHES (OPTIMALLY TUNED ON TEST)", aux_tuned_rows)

    # 3. Binned Metrics
    bin_static_rows, bin_tuned_rows = [], []
    for label, lo, hi in TTE_BINS:
        mask = (test_tte >= lo) & (test_tte < hi)
        ty, tp = test_labels[mask], test_probs[mask]
        
        bin_static_rows.append((label, bin_metrics(ty, tp, ckpt_thr)))
        t_thr, _ = best_f1_threshold(ty, tp)
        bin_tuned_rows.append((label, bin_metrics(ty, tp, t_thr)))

    print_metric_table(f"3A. HORIZON BINS (STATIC THR: {ckpt_thr:.3f})", bin_static_rows)
    print_metric_table("3B. HORIZON BINS (OPTIMALLY TUNED ON EACH BIN)", bin_tuned_rows)

    # 4. Late Fusion Grid
    fusion_static, fusion_tuned = evaluate_late_fusion(test_labels, test_aux_logits, ckpt_thr)
    if fusion_static:
        print_metric_table(f"4A. LATE LOGIT FUSION (STATIC THR: {ckpt_thr:.3f})", fusion_static)
        print_metric_table("4B. LATE LOGIT FUSION (OPTIMALLY TUNED ON TEST)", fusion_tuned)

if __name__ == "__main__":
    main()