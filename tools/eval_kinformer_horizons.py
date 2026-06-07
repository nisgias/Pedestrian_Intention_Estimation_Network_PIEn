#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    precision_recall_curve,
)

from data.pie import PIESeqDataset
from models.pipnet_kinformer_gru_mm import PIPNetKinFormerGRUMM


HORIZONS = ["ETC0_5", "ETC1", "ETC2", "ETC3", "ETC4"]


class MaskCacheWrapper(Dataset):
    """
    Wraps PIESeqDataset and adds sem_masks from precomputed cache.

    Expected cache:
      cache_dir/test/<seq_name>.mask.npz

    The underlying sample already contains sample["path"].
    """
    def __init__(self, base_ds: PIESeqDataset, root: str, cache_dir: str | None):
        self.base_ds = base_ds
        self.root = root
        self.cache_dir = cache_dir

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        item = self.base_ds[idx]

        if self.cache_dir is not None:
            npz_path = item["path"]
            rel_path = os.path.relpath(npz_path, start=self.root)
            split = rel_path.split(os.sep)[0]
            basename = os.path.basename(npz_path).replace(".npz", ".mask.npz")
            mask_path = os.path.join(self.cache_dir, split, basename)

            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Missing mask cache: {mask_path}")

            with np.load(mask_path, allow_pickle=False) as d:
                sem_masks = d["sem_masks"]  # uint8, (T,6,64,64)

            sem_masks = torch.from_numpy(sem_masks).float() / 255.0
            item["sem_masks"] = sem_masks

        return item


def make_loader(root, split, dataset, batch_size, num_workers, seq_len, masks_cache_dir=None):
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

    ds = MaskCacheWrapper(ds, root=root, cache_dir=masks_cache_dir)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def safe_auc(y, p):
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def safe_pr_auc(y, p):
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def calc_metrics(y, p, thr):
    if len(y) == 0:
        return {
            "acc": float("nan"),
            "f1": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
        }

    pred = (p >= thr).astype(np.int32)

    return {
        "acc": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
    }


def best_f1_threshold(y, p):
    if len(y) == 0 or len(np.unique(y)) < 2:
        return 0.5, 0.0

    prec, rec, thrs = precision_recall_curve(y, p)
    if len(thrs) == 0:
        return 0.5, 0.0

    f1s = 2.0 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
    idx = int(np.argmax(f1s))
    return float(thrs[idx]), float(f1s[idx])


@torch.no_grad()
def collect_predictions(model, loader, device, amp=False):
    model.eval()

    ys, ps = [], []

    for batch in loader:
        for k, v in list(batch.items()):
            if torch.is_tensor(v):
                batch[k] = v.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(amp and device.type == "cuda")):
            out = model(batch, return_aux=True)

        logit = out["logit"].squeeze(-1).float()
        prob = torch.sigmoid(logit)

        ps.append(prob.detach().cpu().numpy())
        ys.append(batch["label"].detach().cpu().numpy().astype(np.int32))

    return np.concatenate(ys), np.concatenate(ps)


def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict):
        if "model" in ckpt:
            state = ckpt["model"]
        elif "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
    else:
        state = ckpt

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[CKPT] Loaded: {ckpt_path}")
    print(f"[CKPT] Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    if len(missing) > 0:
        print("       first missing:", missing[:5])
    if len(unexpected) > 0:
        print("       first unexpected:", unexpected[:5])


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--base_root", type=str, default="/Datasets/ETCS")
    ap.add_argument("--horizons", type=str, default="ETC0_5,ETC1,ETC2,ETC3,ETC4")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--dataset", type=str, default="pie", choices=["pie", "jaad"])

    ap.add_argument("--ckpt", type=str, required=True)

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seq_len", type=int, default=10)
    ap.add_argument("--amp", action="store_true")

    ap.add_argument("--use_local_context", action="store_true")
    ap.add_argument("--use_local_flow", action="store_true")
    ap.add_argument("--use_seg", action="store_true")
    ap.add_argument("--use_depth", action="store_true")
    ap.add_argument("--sem_mode", type=str, default="masks")

    ap.add_argument("--masks_cache_base", type=str, default="/workspace/project/masks_cache_etc")

    ap.add_argument("--threshold", type=float, default=0.5)

    # model params
    ap.add_argument("--dropout_p", type=float, default=0.4)
    ap.add_argument("--global_d_model", type=int, default=128)
    ap.add_argument("--global_n_heads", type=int, default=2)
    ap.add_argument("--global_ff_dim", type=int, default=256)
    ap.add_argument("--global_tf_dropout", type=float, default=0.1)
    ap.add_argument("--spatial_layers", type=int, default=1)
    ap.add_argument("--scene_patch_grid", type=int, default=4)
    ap.add_argument("--gate_init", type=float, default=-2.0)

    ap.add_argument("--out_csv", type=str, default="horizon_eval_results.csv")

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PIPNetKinFormerGRUMM(
        dropout_p=args.dropout_p,
        use_local_context=args.use_local_context,
        use_local_flow=args.use_local_flow,
        use_seg=args.use_seg,
        use_depth=args.use_depth,
        sem_mode=args.sem_mode,
        scene_patch_grid=args.scene_patch_grid,
        gate_init=args.gate_init,
        d_model=args.global_d_model,
        n_heads=args.global_n_heads,
        spatial_layers=args.spatial_layers,
        ff_dim=args.global_ff_dim,
        tf_dropout=args.global_tf_dropout,
    ).to(device)

    load_checkpoint(model, args.ckpt, device)

    print()
    print("=" * 90)
    print("KinFormer-GRU-MM Horizon Evaluation")
    print("=" * 90)
    print(f"Base root:   {args.base_root}")
    print(f"Checkpoint:  {args.ckpt}")
    print(f"Modalities:  local={args.use_local_context} flow={args.use_local_flow} seg={args.use_seg} depth={args.use_depth}")
    print(f"sem_mode:    {args.sem_mode}")
    print(f"Threshold:   {args.threshold}")
    print("=" * 90)
    print()

    horizons = [h.strip() for h in args.horizons.split(",") if h.strip()]
    rows = []

    for hz in horizons:
        root = os.path.join(args.base_root, hz)
        if not os.path.isdir(root):
            print(f"[WARN] Missing horizon root: {root}")
            continue

        masks_cache_dir = None
        if args.use_seg and args.sem_mode == "masks":
            masks_cache_dir = os.path.join(args.masks_cache_base, hz)

        print(f"[EVAL] {hz}")
        print(f"       root:  {root}")
        print(f"       masks: {masks_cache_dir}")

        loader = make_loader(
            root=root,
            split=args.split,
            dataset=args.dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seq_len=args.seq_len,
            masks_cache_dir=masks_cache_dir,
        )

        y, p = collect_predictions(model, loader, device, amp=args.amp)

        auc = safe_auc(y, p)
        pr_auc = safe_pr_auc(y, p)

        static = calc_metrics(y, p, args.threshold)
        tuned_thr, tuned_f1 = best_f1_threshold(y, p)
        tuned = calc_metrics(y, p, tuned_thr)

        row = {
            "horizon": hz,
            "n": int(len(y)),
            "pos": int(y.sum()),
            "neg": int(len(y) - y.sum()),
            "auc": auc,
            "pr_auc": pr_auc,
            "thr": args.threshold,
            "acc": static["acc"],
            "f1": static["f1"],
            "precision": static["precision"],
            "recall": static["recall"],
            "oracle_thr": tuned_thr,
            "oracle_acc": tuned["acc"],
            "oracle_f1": tuned["f1"],
            "oracle_precision": tuned["precision"],
            "oracle_recall": tuned["recall"],
        }
        rows.append(row)

    print()
    print("=" * 150)
    print(f"{'HORIZON PERFORMANCE':^150}")
    print("=" * 150)
    print(
        f"{'Horizon':<8} | {'N':>4} | {'Pos':>4} | {'AUC':>7} | {'PR-AUC':>7} || "
        f"{'STATIC @thr':^43} || {'ORACLE F1 ON TEST':^43}"
    )
    print(
        f"{'':<8} | {'':>4} | {'':>4} | {'':>7} | {'':>7} || "
        f"{'Acc':>7} | {'F1':>7} | {'Prec':>7} | {'Rec':>7} || "
        f"{'Thr':>7} | {'Acc':>7} | {'F1':>7} | {'Prec':>7} | {'Rec':>7}"
    )
    print("-" * 150)

    for r in rows:
        print(
            f"{r['horizon']:<8} | {r['n']:>4d} | {r['pos']:>4d} | {r['auc']:>7.4f} | {r['pr_auc']:>7.4f} || "
            f"{r['acc']:>7.4f} | {r['f1']:>7.4f} | {r['precision']:>7.4f} | {r['recall']:>7.4f} || "
            f"{r['oracle_thr']:>7.3f} | {r['oracle_acc']:>7.4f} | {r['oracle_f1']:>7.4f} | "
            f"{r['oracle_precision']:>7.4f} | {r['oracle_recall']:>7.4f}"
        )

    print("=" * 150)

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nSaved CSV: {args.out_csv}\n")


if __name__ == "__main__":
    main()
