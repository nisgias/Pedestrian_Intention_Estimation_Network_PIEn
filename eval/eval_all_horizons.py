import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, precision_recall_curve, accuracy_score

from data.pie import PIESeqDataset
from models.pipnet_alpha_v4_final import PIPNetAlphaV4Final

# The TTE ranges that correspond to your ETC folders (used if needed for validation)
HORIZON_BINS = {
    "ETC0_5": (0.49, 0.55),
    "ETC1":   (0.99, 1.05),
    "ETC2":   (1.99, 2.05),
    "ETC3":   (2.99, 3.05),
    "ETC4":   (3.99, 4.05)
}

def make_loader(root, split, dataset, batch_size, num_workers, seq_len):
    # Τα stats files διαβάζονται με ασφάλεια από τον φάκελο του κώδικα
    ds = PIESeqDataset(
        root, split=split, mode="eval", seq_len=seq_len, strict_len=True, return_meta=True,
        speed_norm="minmax", speed_stats_path=f"/workspace/project/data/{dataset}_speed_stats_splits.json",
        speed_scope="global", motion_norm="p99abs",
        motion_stats_path=f"/workspace/project/data/{dataset}_motion_stats_splits.json",
        motion_scope="global", motion_clip=1.0,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

@torch.no_grad()
def collect_predictions(model, loader, device, amp=False):
    model.eval()
    labels_list, probs_list, tte_list = [], [], []

    for batch in loader:
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            out = model(batch, return_aux=True)

        main_logit = out["logit"].squeeze(-1).float()
        probs_list.append(torch.sigmoid(main_logit).cpu().numpy())
        labels_list.append(batch["label"].cpu().numpy().astype(np.int32))

        meta = batch.get("meta", {})
        if isinstance(meta, dict) and "tte_sec_actual" in meta:
            tte_arr = np.array(meta["tte_sec_actual"]).reshape(-1).astype(np.float32)
        else:
            tte_arr = np.full(batch["label"].shape[0], -1.0, dtype=np.float32)
        tte_list.append(tte_arr)

    return np.concatenate(labels_list), np.concatenate(probs_list), np.concatenate(tte_list)

def safe_auc(y, p):
    if len(y) == 0 or len(np.unique(y)) < 2: return float("nan")
    return float(roc_auc_score(y, p))

def best_f1_threshold(y, p):
    if len(y) == 0 or len(np.unique(y)) < 2: return 0.5, 0.0
    prec, rec, thrs = precision_recall_curve(y, p)
    if len(thrs) == 0: return 0.5, 0.0
    f1s = 2.0 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
    best_idx = int(np.argmax(f1s))
    return float(thrs[best_idx]), float(f1s[best_idx])

def calc_metrics(y, p, thr):
    if len(y) == 0: return 0.0, 0.0, 0.0, 0.0
    preds = (p >= thr).astype(np.int32)
    return (
        float(f1_score(y, preds, zero_division=0)),
        float(precision_score(y, preds, zero_division=0)),
        float(recall_score(y, preds, zero_division=0)),
        float(accuracy_score(y, preds)) # <-- Προστέθηκε το Accuracy
    )

def main():
    ap = argparse.ArgumentParser()
    # base_root points directly to the ETCs folder
    ap.add_argument("--base_root", type=str, default="/Datasets/PIE_PREP_OUT/ETCs", help="Base path")
    ap.add_argument("--horizons", type=str, default="ETC0_5,ETC1,ETC2,ETC3,ETC4", help="Test folders")
    ap.add_argument("--dataset", type=str, default="pie", choices=["pie", "jaad"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch_size", type=int, default=32) # Αυξημένο για ταχύτερο testing
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seq_len", type=int, default=10)
    ap.add_argument("--amp", action="store_true")
    
    # Model params (matching your best checkpoint)
    ap.add_argument("--dropout_p", type=float, default=0.2)
    ap.add_argument("--local_dropout_p", type=float, default=0.1)
    ap.add_argument("--global_dropout_p", type=float, default=0.4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and args.amp

    # Load Model
    model = PIPNetAlphaV4Final(
        dropout_p=args.dropout_p, local_dropout_p=args.local_dropout_p, global_dropout_p=args.global_dropout_p
    ).to(device)

    # weights_only=False to bypass PyTorch 2.6 security restriction
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    
    # ---------------------------------------------------------
    # ΘΕΣΕ ΤΟ STATIC THRESHOLD ΠΟΥ ΒΡΗΚΑΜΕ ΑΠΟ ΤΟ VALIDATION
    # ---------------------------------------------------------
    ckpt_thr = 0.7585 
    # ---------------------------------------------------------

    print(f"\n[INFO] Loaded Checkpoint: {args.ckpt}")
    print(f"[INFO] Using Static Threshold: {ckpt_thr:.4f}\n")

    # Evaluate Test Sets directly
    horizons = args.horizons.split(",")
    summary_results = []

    for hz in horizons:
        hz_path = os.path.join(args.base_root, hz)
        if not os.path.exists(hz_path):
            print(f"[WARNING] Path {hz_path} not found. Skipping...")
            continue
            
        print(f"Evaluating {hz} ...")
        loader = make_loader(hz_path, "test", args.dataset, args.batch_size, args.num_workers, args.seq_len)
        test_y, test_p, _ = collect_predictions(model, loader, device, use_amp)

        if len(test_y) == 0:
            print(f"   -> No test samples found in {hz_path}/test. Skipping.")
            continue

        auc = safe_auc(test_y, test_p)
        
        # 1. Static Threshold Evaluation (Πραγματική απόδοση)
        static_f1, static_prec, static_rec, static_acc = calc_metrics(test_y, test_p, ckpt_thr)

        # 2. Tuned (Oracle) Threshold Evaluation (Μαθηματικό ταβάνι)
        tuned_thr, tuned_f1 = best_f1_threshold(test_y, test_p)
        _, tuned_prec, tuned_rec, tuned_acc = calc_metrics(test_y, test_p, tuned_thr)

        summary_results.append({
            "horizon": hz, "n": len(test_y), "auc": auc, 
            "s_f1": static_f1, "s_prec": static_prec, "s_rec": static_rec, "s_acc": static_acc, "s_thr": ckpt_thr,
            "t_f1": tuned_f1,  "t_prec": tuned_prec,  "t_rec": tuned_rec,  "t_acc": tuned_acc,  "t_thr": tuned_thr
        })

    # --- Print Beautiful Summary Table ---
    print("\n" + "="*145)
    print(f"{'FULL PERFORMANCE REPORT (STATIC vs TUNED)':^145}")
    print("="*145)
    print(f"{'Horizon':<8} | {'N':<4} | {'AUC':<6} || {'STATIC (Thr=0.758)':^38} || {'TUNED (Max Capability)':^40}")
    print(f"{'':<8} | {'':<4} | {'':<6} || {'Acc':<7} | {'F1':<7} | {'Prec':<7} | {'Rec':<7} || {'Thr':<6} | {'Acc':<7} | {'F1':<7} | {'Prec':<7} | {'Rec':<7}")
    print("-" * 145)
    
    for res in summary_results:
        print(f"{res['horizon']:<8} | {res['n']:<4} | {res['auc']:.4f} || "
              f"{res['s_acc']:.4f} | {res['s_f1']:.4f} | {res['s_prec']:.4f} | {res['s_rec']:.4f} || "
              f"{res['t_thr']:.3f}  | {res['t_acc']:.4f} | {res['t_f1']:.4f} | {res['t_prec']:.4f} | {res['t_rec']:.4f}")
    print("="*145 + "\n")

if __name__ == "__main__":
    main()