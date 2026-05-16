import torch
from models.pipnet_alpha_v4_final import PIPNetAlphaV4Final
from train.train_v4_transfer import make_loaders, run_eval_epoch, MultiTaskLoss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("============================================================")
    print(" CROSS-DATASET GENERALIZATION TEST (EVAL ON JAAD)")
    print("============================================================")

    jaad_root = "/Datasets/JAAD_PREP_OUT"

    # PIE-only baseline — seed46 (best single model)
    pie_only_ckpt = "checkpoints_v4_best_seed46/stage3_pie_baseline/best_model.pth"

    # Transfer model — trial12 (JAAD pretrain -> PIE fine-tune)
    transfer_ckpt = "checkpoints_reproduce_transfer_trial12/stage2_pie_transfer/best_model.pth"

    print("\nLoading JAAD Test Dataset...")
    try:
        _, _, _, jaad_test_loader = make_loaders(
            jaad_root, "jaad", seq_len=10, batch_size=32,
            num_workers=2, strict_len=True,
        )
        print(f"JAAD Test Sequences: {len(jaad_test_loader.dataset)}")
    except Exception as e:
        print(f"Error loading JAAD dataset: {e}")
        return

    criterion = MultiTaskLoss(pos_weight=1.0)

    # ------------------------------------------------------------------
    # 1. PIE-Only Baseline (seed46 params)
    # ------------------------------------------------------------------
    print("\n[1/2] Evaluating PIE-Only baseline (seed46) on JAAD Test Set...")
    model_baseline = PIPNetAlphaV4Final(
        dropout_p=0.2,
        local_dropout_p=0.1,
        global_dropout_p=0.4,
    ).to(device)
    baseline_metrics = {}
    try:
        ckpt = torch.load(pie_only_ckpt, map_location=device, weights_only=False)
        model_baseline.load_state_dict(ckpt["model"])
        print(f"  Loaded epoch {ckpt.get('epoch','?')} | "
              f"PIE val AUC at save: {ckpt.get('val_metrics',{}).get('auc','?')}")
        baseline_metrics = run_eval_epoch(
            model_baseline, jaad_test_loader, device, criterion, use_amp=True,
        )
    except Exception as e:
        print(f"  Error: {e}")

    # ------------------------------------------------------------------
    # 2. Transfer Model — trial12 (JAAD → PIE)
    #    dropout_p=0.2, local_dropout_p=0.4, global_dropout_p=0.5
    #    jaad_lr=1.47e-05, pie_lr=2.08e-05
    #    aux_weight=0.2, entropy_weight=0.08
    # ------------------------------------------------------------------
    print("\n[2/2] Evaluating Transfer model (trial12: JAAD->PIE) on JAAD Test Set...")
    model_transfer = PIPNetAlphaV4Final(
        dropout_p=0.2,
        local_dropout_p=0.4,
        global_dropout_p=0.5,
    ).to(device)
    transfer_metrics = {}
    try:
        ckpt = torch.load(transfer_ckpt, map_location=device, weights_only=False)
        model_transfer.load_state_dict(ckpt["model"])
        print(f"  Loaded epoch {ckpt.get('epoch','?')} | "
              f"PIE val AUC at save: {ckpt.get('val_metrics',{}).get('auc','?')}")
        transfer_metrics = run_eval_epoch(
            model_transfer, jaad_test_loader, device, criterion, use_amp=True,
        )
    except Exception as e:
        print(f"  Error: {e}")

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(" RESULTS: CROSS-DATASET GENERALIZATION (tested on JAAD test set)")
    print("=" * 70)
    print(f"{'Metric':<22} | {'PIE-Only (s46)':>14} | {'Transfer (t12)':>14} | {'Delta':>10}")
    print("-" * 70)

    metrics_to_print = [
        ("auc",        "AUC (main)"),
        ("acc",        "Accuracy"),
        ("f1",         "F1 Score"),
        ("precision",  "Precision"),
        ("recall",     "Recall"),
        ("auc_kin",    "Branch: kinematic"),
        ("auc_local",  "Branch: local"),
        ("auc_global", "Branch: global"),
    ]

    for key, label in metrics_to_print:
        b = baseline_metrics.get(key)
        t = transfer_metrics.get(key)
        if b is None or t is None:
            continue
        delta = t - b
        arrow = "+" if delta >= 0 else ""
        print(f"{label:<22} | {b:>14.4f} | {t:>14.4f} | {arrow}{delta:>+.4f}")

    print("=" * 70)
    print("\nΣημείωση Transfer trial12:")
    print("  fusion_val_auc = 0.8949")
    print("  fusion weights: main=0.85, kin=0.15, local=0.00, global=0.00")
    print("\nΕρμηνεία:")
    print("  Transfer > PIE-Only -> JAAD pretraining βοήθησε τη γενίκευση")
    print("  PIE-Only > Transfer -> Catastrophic forgetting κατά PIE fine-tuning")
    print("=" * 70)


if __name__ == "__main__":
    main()