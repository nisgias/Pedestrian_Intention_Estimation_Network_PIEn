import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score

from data.pie import PIESeqDataset
from models.pipnet_alpha_v4_final import PIPNetAlphaV4Final

def make_loader(root, split, dataset, batch_size, num_workers, seq_len):
    # Τα stats διαβάζονται με ασφάλεια από τον κώδικα
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

def best_f1_threshold(y, p):
    if len(y) == 0 or len(np.unique(y)) < 2: return 0.5, 0.0
    prec, rec, thrs = precision_recall_curve(y, p)
    if len(thrs) == 0: return 0.5, 0.0
    f1s = 2.0 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
    best_idx = int(np.argmax(f1s))
    return float(thrs[best_idx]), float(f1s[best_idx])

def main():
    ap = argparse.ArgumentParser()
    # Το base root δείχνει στον κεντρικό φάκελο, ώστε να βρει τον υποφάκελο 'val'
    ap.add_argument("--base_root", type=str, default="/Datasets/PIE_PREP_OUT", help="Base path containing the 'val' folder")
    ap.add_argument("--dataset", type=str, default="pie", choices=["pie", "jaad"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch_size", type=int, default=64) # Αυξημένο για ταχύτητα
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seq_len", type=int, default=10)
    ap.add_argument("--amp", action="store_true")
    
    # Model params
    ap.add_argument("--dropout_p", type=float, default=0.2)
    ap.add_argument("--local_dropout_p", type=float, default=0.1)
    ap.add_argument("--global_dropout_p", type=float, default=0.4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and args.amp

    # Φόρτωση Μοντέλου
    model = PIPNetAlphaV4Final(
        dropout_p=args.dropout_p, local_dropout_p=args.local_dropout_p, global_dropout_p=args.global_dropout_p
    ).to(device)

    # weights_only=False για το PyTorch 2.6
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    
    print(f"\n[INFO] Model loaded successfully.")
    print(f"[INFO] Running inference on the entire Validation Set...\n")

    # 1. Inference στο Validation Set
    val_loader = make_loader(args.base_root, "val", args.dataset, args.batch_size, args.num_workers, args.seq_len)
    val_y, val_p, val_tte = collect_predictions(model, val_loader, device, use_amp)

    # 2. Φιλτράρισμα: Κρατάμε ΜΟΝΟ τα δείγματα που είναι κοντά στο 1 δευτερόλεπτο
    # (0.95 έως 1.05 πιάνει τα 30fps frames γύρω από το 1 δευτερόλεπτο)
    mask = (val_tte >= 0.95) & (val_tte <= 1.05)
    
    val_y_1s = val_y[mask]
    val_p_1s = val_p[mask]
    
    if len(val_y_1s) == 0:
        print("[ERROR] Δεν βρέθηκαν δείγματα με TTE γύρω στο 1s στο Validation set!")
        return

    # 3. Εύρεση Βέλτιστου Κατωφλίου
    best_thr, best_f1 = best_f1_threshold(val_y_1s, val_p_1s)
    
    # Υπολογισμός Precision/Recall για επιβεβαίωση
    preds = (val_p_1s >= best_thr).astype(np.int32)
    prec = float(precision_score(val_y_1s, preds, zero_division=0))
    rec = float(recall_score(val_y_1s, preds, zero_division=0))

    # Εκτύπωση Αποτελεσμάτων
    print("="*60)
    print(f"{'CALIBRATION FOR 1-SECOND HORIZON (Validation Set)':^60}")
    print("="*60)
    print(f"Total Validation Samples    : {len(val_y)}")
    print(f"Samples at 1s (0.95-1.05)   : {len(val_y_1s)}")
    print(f"Positives in 1s split       : {val_y_1s.sum()}")
    print("-" * 60)
    print(f"OPTIMAL THRESHOLD FOUND     : {best_thr:.4f}")
    print(f"Expected F1 at this Thr     : {best_f1:.4f}")
    print(f"Expected Precision          : {prec:.4f}")
    print(f"Expected Recall             : {rec:.4f}")
    print("="*60 + "\n")
    print(f"[ACTION ITEM] Use Threshold {best_thr:.4f} to evaluate your ETC1 Test set for your paper!")

if __name__ == "__main__":
    main()