import torch
from models.pipnet_alpha_v3_final import PIPNetAlphaV3Final
from train_v3_transfer import make_loaders, run_eval_epoch, MultiTaskLoss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("============================================================")
    print(" CROSS-DATASET GENERALIZATION TEST (EVAL ON JAAD)")
    print("============================================================")

    # --- ΤΑ ΣΩΣΤΑ ΜΟΝΟΠΑΤΙΑ ΑΠΟ ΤΗΝ ΕΙΚΟΝΑ ΣΟΥ ---
    jaad_root = "/data/JAAD_PREP_OUT"
    pie_only_ckpt = "checkpoints_transfer/stage3_pie_baseline/best_model.pth"
    transfer_ckpt = "checkpoints_transfer/stage2_pie_transfer/best_model.pth"
    # ---------------------------------------------

    # Φόρτωση του JAAD Test Set
    print("\nLoading JAAD Test Dataset...")
    try:
        _, _, _, jaad_test_loader = make_loaders(
            jaad_root, "jaad", seq_len=10, batch_size=8, num_workers=4, strict_len=True
        )
        print(f"JAAD Test Sequences: {len(jaad_test_loader.dataset)}")
    except Exception as e:
        print(f"Σφάλμα κατά τη φόρτωση του JAAD dataset: {e}")
        return

    # Loss (To pos_weight δεν επηρεάζει το AUC στο evaluation, βάζουμε 1.0)
    criterion = MultiTaskLoss(pos_weight=1.0)

    # ==========================================================
    # 1. PIE-Only Baseline Model
    # ==========================================================
    print("\n[1/2] Evaluating PIE-Only Model on JAAD Test Set...")
    model_baseline = PIPNetAlphaV3Final(dropout_p=0.5).to(device)
    try:
        model_baseline.load_state_dict(torch.load(pie_only_ckpt, map_location=device)['model'])
        baseline_metrics = run_eval_epoch(model_baseline, jaad_test_loader, device, criterion, use_amp=True)
    except Exception as e:
        print(f"Error loading PIE-Only checkpoint: {e}")
        baseline_metrics = {}

    # ==========================================================
    # 2. Transfer Model (JAAD -> PIE)
    # ==========================================================
    print("\n[2/2] Evaluating Transfer Model (JAAD->PIE) on JAAD Test Set...")
    model_transfer = PIPNetAlphaV3Final(dropout_p=0.5).to(device)
    try:
        model_transfer.load_state_dict(torch.load(transfer_ckpt, map_location=device)['model'])
        transfer_metrics = run_eval_epoch(model_transfer, jaad_test_loader, device, criterion, use_amp=True)
    except Exception as e:
        print(f"Error loading Transfer checkpoint: {e}")
        transfer_metrics = {}

    # ==========================================================
    # FINAL COMPARISON TABLE
    # ==========================================================
    print("\n" + "=" * 70)
    print(" RESULTS: CROSS-DATASET GENERALIZATION (Tested on JAAD Test Set)")
    print("=" * 70)
    print(f"{'Metric':<18} | {'PIE-Only Model':<20} | {'Transfer (JAAD->PIE)':<20}")
    print("-" * 70)
    
    metrics_to_print = [
        ('auc', 'AUC (Main)'), ('acc', 'Accuracy'), ('f1', 'F1 Score'),
        ('auc_kin', 'Branch: Kinematic'), ('auc_local', 'Branch: Local'), 
        ('auc_global', 'Branch: Global')
    ]
    
    for key, label in metrics_to_print:
        if key in baseline_metrics and key in transfer_metrics:
            b_val = baseline_metrics.get(key, float('nan'))
            t_val = transfer_metrics.get(key, float('nan'))
            if isinstance(b_val, float) and isinstance(t_val, float):
                diff = t_val - b_val
                diff_str = f"({diff:+.4f})"
                print(f"{label:<18} | {b_val:<20.4f} | {t_val:.4f} {diff_str}")
            
    print("=" * 70)
    print("Interpretation:")
    print(" - Αν το Transfer (Δεξιά) έχει μεγαλύτερο AUC, το μοντέλο κράτησε")
    print("   χρήσιμα χαρακτηριστικά από το JAAD (Robustness/Knowledge Retention).")
    print(" - Αν το PIE-Only (Αριστερά) είναι καλύτερο, το Transfer έπαθε")
    print("   'Catastrophic Forgetting'.")
    print("============================================================")

if __name__ == "__main__":
    main()
