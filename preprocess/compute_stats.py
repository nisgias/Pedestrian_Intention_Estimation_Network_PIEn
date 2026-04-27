#!/usr/bin/env python3
"""
Compute normalization statistics for speed and motion features.
Saves {dataset}_speed_stats_splits.json and {dataset}_motion_stats_splits.json.

Usage (PIE):
    python scripts/compute_stats.py --npz_root /workspace/PIE_PREP_OUT --dataset pie

Usage (JAAD):
    python scripts/compute_stats.py --npz_root /data/JAAD_PREP_OUT --dataset jaad

Note on JAAD:
  - JAAD has no ego-vehicle speed in annotations (only vehicle action labels).
    The prep pipeline stores speed as 0.0 for all frames.
    Speed stats will reflect this (min=0, max=0), and minmax normalization
    will gracefully produce all-zeros (which is correct — speed is not a
    useful feature for JAAD, matching the original PIP-Net paper).
  - Motion (RAFT optical flow) is still meaningful for JAAD.
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm


def compute_speed_stats(npz_files):
    """Compute min/max for speed normalization."""
    all_speeds = []
    for f in tqdm(npz_files, desc="Speed stats"):
        try:
            d = np.load(f, allow_pickle=True)
            if "speed" in d.files:
                s = d["speed"].astype(np.float32).reshape(-1)
                all_speeds.append(s)
        except Exception as e:
            print(f"[WARN] {f}: {e}")

    if not all_speeds:
        return {"min": 0.0, "max": 1.0, "mean": 0.0, "std": 0.0}

    all_speeds = np.concatenate(all_speeds)
    return {
        "min": float(np.min(all_speeds)),
        "max": float(np.max(all_speeds)),
        "mean": float(np.mean(all_speeds)),
        "std": float(np.std(all_speeds)),
    }


def compute_motion_stats(npz_files):
    """Compute p99 absolute value for motion/flow normalization."""
    all_vals = []
    for f in tqdm(npz_files, desc="Motion stats"):
        try:
            d = np.load(f, allow_pickle=True)
            if "local_motion" in d.files:
                m = d["local_motion"].astype(np.float32).reshape(-1)
                # Skip placeholder arrays (all zeros = not yet preprocessed)
                if np.all(m == 0):
                    continue
                # Sample to avoid memory issues
                if len(m) > 10000:
                    idx = np.random.choice(len(m), 10000, replace=False)
                    m = m[idx]
                all_vals.append(np.abs(m))
        except Exception as e:
            print(f"[WARN] {f}: {e}")

    if not all_vals:
        print("[WARN] No non-zero motion data found! Using fallback stats.")
        return {"p99_abs": 1.0, "mean_abs": 0.0, "p50_abs": 0.0, "p95_abs": 1.0}

    all_vals = np.concatenate(all_vals)
    return {
        "p99_abs": float(np.percentile(all_vals, 99)),
        "mean_abs": float(np.mean(all_vals)),
        "p50_abs": float(np.percentile(all_vals, 50)),
        "p95_abs": float(np.percentile(all_vals, 95)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute speed/motion normalization stats for PIE or JAAD"
    )
    parser.add_argument("--npz_root", required=True,
                        help="Path to preprocessed NPZ root (e.g. /data/JAAD_PREP_OUT)")
    parser.add_argument("--dataset", type=str, required=True, choices=["pie", "jaad"],
                        help="Dataset name prefix for output JSON files")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--out_dir", default="/workspace/project/data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    splits = [s.strip() for s in args.splits.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)

    npz_root = Path(args.npz_root)

    print("=" * 70)
    print(f"Computing normalization stats for: {args.dataset.upper()}")
    print(f"NPZ root: {npz_root}")
    print(f"Output dir: {args.out_dir}")
    print("=" * 70)

    # Collect files per split
    split_files = {}
    for split in splits:
        split_dir = npz_root / split
        if not split_dir.exists():
            print(f"[WARN] {split_dir} not found, skipping")
            continue
        files = sorted(split_dir.glob("seq_*.npz"))
        split_files[split] = [str(f) for f in files]
        print(f"[INFO] {split}: {len(files)} files")

    # Use only train split for computing global stats (standard practice)
    if "train" not in split_files:
        print("[ERROR] Train split not found!")
        return

    train_files = split_files["train"]
    print(f"\nComputing stats from {len(train_files)} train files...")

    # ---- Speed stats ----
    print("\n--- Speed ---")
    speed_global = compute_speed_stats(train_files)
    print(f"  min={speed_global['min']:.4f}  max={speed_global['max']:.4f}  "
          f"mean={speed_global['mean']:.4f}  std={speed_global['std']:.4f}")

    if speed_global['min'] == speed_global['max']:
        print(f"  [NOTE] Speed is constant ({speed_global['min']:.4f}) across all train samples.")
        if args.dataset == "jaad":
            print(f"  [NOTE] This is expected for JAAD — no ego-vehicle speed in annotations.")
            print(f"         The model's kinematic branch will learn to ignore speed.")

    speed_stats = {"global": speed_global}
    for split, files in split_files.items():
        speed_stats[split] = compute_speed_stats(files)

    speed_out = os.path.join(args.out_dir, f"{args.dataset}_speed_stats_splits.json")
    with open(speed_out, "w") as f:
        json.dump(speed_stats, f, indent=2)
    print(f"  Saved: {speed_out}")

    # ---- Motion stats ----
    print("\n--- Motion (optical flow) ---")
    motion_global = compute_motion_stats(train_files)
    print(f"  p99_abs={motion_global['p99_abs']:.4f}  mean_abs={motion_global['mean_abs']:.4f}")

    motion_stats = {"global": motion_global}
    for split, files in split_files.items():
        motion_stats[split] = compute_motion_stats(files)

    motion_out = os.path.join(args.out_dir, f"{args.dataset}_motion_stats_splits.json")
    with open(motion_out, "w") as f:
        json.dump(motion_stats, f, indent=2)
    print(f"  Saved: {motion_out}")

    print("\n" + "=" * 70)
    print("[DONE] Stats computed successfully!")
    print(f"  {speed_out}")
    print(f"  {motion_out}")
    print()
    print(f"Training script usage:")
    print(f"  python train_v3_final.py --data_root {args.npz_root} --dataset {args.dataset} --amp")
    print("=" * 70)


if __name__ == "__main__":
    main()