#!/usr/bin/env python3
"""
Offline mask precomputation – cache version.
Creates traffic-relevant semantic mask cache from sem_labels in .npz files.
"""

from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


PIP_TO_REDUCED = np.array([
    0,  # 0 bg          -> bg/noise
    1,  # 1 person      -> person
    2,  # 2 rider       -> rider
    3,  # 3 car         -> vehicle
    3,  # 4 truck       -> vehicle
    3,  # 5 bus         -> vehicle
    3,  # 6 train       -> vehicle
    3,  # 7 motorcycle  -> vehicle
    3,  # 8 bicycle     -> vehicle
    4,  # 9 traffic_lt  -> traffic
    4,  # 10 traffic_sn -> traffic
    5,  # 11 road       -> road
    6,  # 12 sidewalk   -> sidewalk
    0,  # 13 building   -> bg/noise
    0,  # 14 wall       -> bg/noise
    0,  # 15 fence      -> bg/noise
    7,  # 16 pole       -> pole
    0,  # 17 vegetation -> bg/noise
    0,  # 18 terrain    -> bg/noise
    0,  # 19 sky        -> bg/noise
], dtype=np.int64)

MASK_GROUPS = [
    (1, 2),    # 0: pedestrians
    (3,),      # 1: vehicles
    (4,),      # 2: traffic signals/signs
    (5,),      # 3: road
    (6,),      # 4: sidewalk
    (7,),      # 5: pole
]

N_MASK_CHANNELS = len(MASK_GROUPS)


def compute_masks(sem_labels: np.ndarray, target_size: int) -> np.ndarray:
    sem_labels = sem_labels.astype(np.int64, copy=False)
    sem_clamped = np.clip(sem_labels, 0, 19)
    sem_reduced = PIP_TO_REDUCED[sem_clamped]

    T, H, W = sem_reduced.shape

    masks_full = np.zeros((T, N_MASK_CHANNELS, H, W), dtype=np.float32)

    for ci, group in enumerate(MASK_GROUPS):
        m = np.zeros((T, H, W), dtype=bool)
        for g in group:
            m |= (sem_reduced == g)
        masks_full[:, ci] = m.astype(np.float32)

    masks_t = torch.from_numpy(masks_full)
    masks_small = F.adaptive_avg_pool2d(masks_t, (target_size, target_size))
    masks_uint8 = (masks_small * 255.0).clamp(0, 255).byte().numpy()

    return masks_uint8


def process_file(npz_path: str, root: str, cache_dir: str, target_size: int, overwrite: bool) -> str:
    try:
        rel_path = os.path.relpath(npz_path, start=root)
        split = rel_path.split(os.sep)[0]

        basename = os.path.basename(npz_path).replace(".npz", ".mask.npz")
        cache_path = os.path.join(cache_dir, split, basename)

        if not overwrite and os.path.exists(cache_path):
            return "SKIPPED"

        with np.load(npz_path, allow_pickle=True) as data:
            if "sem_labels" not in data:
                return "NO_SEM"
            sem_labels = data["sem_labels"]

        masks = compute_masks(sem_labels, target_size=target_size)

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, sem_masks=masks)

        return "OK"

    except Exception as e:
        print(f"\nERROR on {npz_path}: {e}", flush=True)
        return "ERROR"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=str, help="Dataset root, e.g. /Datasets/PIE_PREP_OUT or /Datasets/PIE_PREP_OUT/ETCs/ETC1")
    p.add_argument("--cache_dir", type=str, default="/workspace/project/masks_cache")
    p.add_argument("--target_size", type=int, default=64)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    print(f"Precomputing masks to cache: {args.cache_dir}")
    print(f"  root        = {args.root}")
    print(f"  target_size = {args.target_size}")
    print(f"  splits      = {args.splits}")
    print()

    grand_total = {"OK": 0, "SKIPPED": 0, "NO_SEM": 0, "ERROR": 0}
    t_start = time.time()

    for split in args.splits:
        split_dir = os.path.join(args.root, split)

        if not os.path.isdir(split_dir):
            print(f"[{split}] directory not found: {split_dir}, skipping")
            continue

        files = sorted(glob.glob(os.path.join(split_dir, "*.npz")))

        if args.limit:
            files = files[:args.limit]

        if not files:
            print(f"[{split}] no .npz files found")
            continue

        print(f"[{split}] {len(files)} files")

        counts = {"OK": 0, "SKIPPED": 0, "NO_SEM": 0, "ERROR": 0}

        for path in tqdm(files, desc=f"  {split}"):
            status = process_file(path, args.root, args.cache_dir, args.target_size, args.overwrite)
            counts[status] += 1
            grand_total[status] += 1

        print(
            f"  -> OK: {counts['OK']}, "
            f"SKIPPED: {counts['SKIPPED']}, "
            f"NO_SEM: {counts['NO_SEM']}, "
            f"ERROR: {counts['ERROR']}\n"
        )

    print("=" * 60)
    print(f"Total OK:       {grand_total['OK']}")
    print(f"Total SKIPPED:  {grand_total['SKIPPED']}")
    print(f"Total NO_SEM:   {grand_total['NO_SEM']}")
    print(f"Total ERROR:    {grand_total['ERROR']}")
    print(f"Elapsed:        {time.time() - t_start:.1f} sec")
    print("=" * 60)


if __name__ == "__main__":
    main()
