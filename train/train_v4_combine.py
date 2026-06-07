#!/usr/bin/env python3
"""
Combine JAAD+PIE training strategy for PIPNetAlphaV4Final.

Purpose:
  Train one V4 model on JAAD train + PIE train together, then early-stop and
  test on PIE only. This is the combine counterpart of the V4 JAAD->PIE
  transfer experiment. Named "combine" in the report.

Important:
  JAAD has no real ego-speed. In the current preprocessing, JAAD speed is stored
  as zeros. Therefore this evaluates combine training with zero-filled
  missing speed, not every possible multi-dataset training strategy.
"""

import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.cuda.amp import GradScaler

from data.pie import PIESeqDataset

# Works whether this script is run from train/ or project root
try:
    from train_v4_transfer import (
        MultiTaskLoss,
        make_model,
        build_optimizer,
        run_training_stage,
        init_csv,
        set_seed,
    )
except ImportError:
    from train.train_v4_transfer import (
        MultiTaskLoss,
        make_model,
        build_optimizer,
        run_training_stage,
        init_csv,
        set_seed,
    )


def make_single_dataset(data_root, dataset_prefix, split, seq_len, strict_len=True):
    speed_stats_path = f"/workspace/project/data/{dataset_prefix}_speed_stats_splits.json"
    motion_stats_path = f"/workspace/project/data/{dataset_prefix}_motion_stats_splits.json"

    return PIESeqDataset(
        data_root,
        split=split,
        mode="train" if split == "train" else "eval",
        seq_len=seq_len,
        strict_len=strict_len,
        speed_norm="minmax",
        speed_stats_path=speed_stats_path,
        speed_scope="global",
        motion_norm="p99abs",
        motion_stats_path=motion_stats_path,
        motion_scope="global",
        motion_clip=1.0,
    )


def compute_pos_weight_from_datasets(datasets):
    pos, neg = 0, 0

    for ds in datasets:
        for p in ds.files:
            d = np.load(p, allow_pickle=True)
            y = float(np.array(d["label"]).reshape(-1)[0])
            if y >= 0.5:
                pos += 1
            else:
                neg += 1

    return float(neg / max(pos, 1)), int(pos), int(neg)


def make_combine_loaders(args, strict_len=True):
    jaad_train = make_single_dataset(
        args.jaad_root, "jaad", "train", args.seq_len, strict_len=strict_len
    )

    pie_train = make_single_dataset(
        args.pie_root, "pie", "train", args.seq_len, strict_len=strict_len
    )

    pie_val = make_single_dataset(
        args.pie_root, "pie", "val", args.seq_len, strict_len=strict_len
    )

    pie_test = make_single_dataset(
        args.pie_root, "pie", "test", args.seq_len, strict_len=strict_len
    )

    pooled_train = ConcatDataset([jaad_train, pie_train])

    train_loader = DataLoader(
        pooled_train,
        batch_size=args.pie_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        pie_val,
        batch_size=args.pie_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        pie_test,
        batch_size=args.pie_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return jaad_train, pie_train, pooled_train, train_loader, val_loader, test_loader


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--jaad_root", type=str, required=True)
    parser.add_argument("--pie_root", type=str, required=True)

    parser.add_argument("--seq_len", type=int, default=10)

    # V4 model params
    parser.add_argument("--dropout_p", type=float, default=0.2)
    parser.add_argument("--local_dropout_p", type=float, default=0.3)
    parser.add_argument("--global_dropout_p", type=float, default=0.2)

    # Keep same names as train_v4_transfer because log_csv expects them
    parser.add_argument("--jaad_epochs", type=int, default=0)
    parser.add_argument("--jaad_lr", type=float, default=0.0)
    parser.add_argument("--jaad_batch_size", type=int, default=10)

    parser.add_argument("--pie_epochs", type=int, default=30)
    parser.add_argument("--pie_lr", type=float, default=2e-5)
    parser.add_argument("--pie_batch_size", type=int, default=64)

    parser.add_argument("--baseline_epochs", type=int, default=0)
    parser.add_argument("--baseline_lr", type=float, default=0.0)

    parser.add_argument("--aux_weight", type=float, default=0.1)
    parser.add_argument("--entropy_weight", type=float, default=0.05)
    parser.add_argument("--last_fc_l2", type=float, default=1e-4)
    parser.add_argument("--visual_l2", type=float, default=0.0)

    parser.add_argument("--fixed_lr", action="store_true")
    parser.add_argument("--early_stopping_patience", type=int, default=10)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--wpos", type=float, default=-1.0)

    parser.add_argument("--save_dir", type=str, default="checkpoints_v4_combine")
    parser.add_argument("--no_strict_len", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    args.command = " ".join(sys.argv)

    # These are only for CSV compatibility with train_v4_transfer.log_csv
    args.skip_transfer = False
    args.skip_baseline = True

    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and args.amp
    scaler = GradScaler(enabled=use_amp)
    strict_len = not args.no_strict_len

    os.makedirs(args.save_dir, exist_ok=True)
    csv_path = os.path.join(args.save_dir, "training_log.csv")
    init_csv(csv_path)

    print("=" * 90)
    print("V4 COMBINE TRAINING: JAAD train + PIE train -> PIE val/test")
    print("=" * 90)
    print(f"JAAD root: {args.jaad_root}")
    print(f"PIE root:  {args.pie_root}")
    print(f"Save dir:  {args.save_dir}")
    print(f"Seed:      {args.seed}")
    print("=" * 90)

    jaad_train, pie_train, pooled_train, train_loader, val_loader, test_loader = make_combine_loaders(
        args, strict_len=strict_len
    )

    print(f"[INFO] JAAD train samples: {len(jaad_train)}")
    print(f"[INFO] PIE train samples:  {len(pie_train)}")
    print(f"[INFO] Combine samples:    {len(pooled_train)}")
    print("[INFO] Early stopping and final test are PIE-only.")

    if args.wpos > 0:
        combine_wpos = args.wpos
        print(f"[INFO] Using manual pos_weight={combine_wpos:.4f}")
    else:
        combine_wpos, combine_pos, combine_neg = compute_pos_weight_from_datasets(
            [jaad_train, pie_train]
        )
        print(
            f"[INFO] Combine train class counts: "
            f"pos={combine_pos}, neg={combine_neg}, pos_weight={combine_wpos:.4f}"
        )

    model = make_model(args, device)

    criterion = MultiTaskLoss(
        aux_weight=args.aux_weight,
        entropy_weight=args.entropy_weight,
        pos_weight=combine_wpos,
    )

    optimizer, scheduler = build_optimizer(
        model,
        args.pie_lr,
        args.last_fc_l2,
        fixed_lr=args.fixed_lr,
        visual_l2=args.visual_l2,
    )

    combine_dir = os.path.join(args.save_dir, "stage0_combine_jaad_pie")

    test_m, best_state = run_training_stage(
        "S0_COMBINE_JAAD_PIE",
        model,
        device,
        criterion,
        train_loader,
        val_loader,
        test_loader,
        optimizer,
        scheduler,
        scaler,
        args.pie_epochs,
        use_amp,
        combine_dir,
        csv_path,
        args,
        early_stop_patience=args.early_stopping_patience,
    )

    print("\n" + "=" * 90)
    print("COMBINE FINAL PIE TEST RESULT")
    print("=" * 90)
    for k, v in test_m.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")
    print("=" * 90)


if __name__ == "__main__":
    main()