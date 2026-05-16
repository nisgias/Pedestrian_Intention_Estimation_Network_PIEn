#!/usr/bin/env python3
"""
eval/eval_multiseed_v4.py  —  PIPNet-Alpha V4 comprehensive multi-seed evaluation.

Per-seed pipeline
─────────────────
  1.  Parameter count (total / trainable)
  2.  Inference speed  (FPS + latency ms/sample)
  3.  Individual branch ROC-AUC + PR-AUC  (val & test)
  4.  Val-tuned F1 threshold → applied once to test
      per branch: precision / recall / F1 / accuracy / MCC
  5.  Expected Calibration Error (ECE, 15 bins) per branch on test
  6.  Post-hoc logit fusion
        a) val-tuned grid search → weights → evaluate on test
        b) oracle  (grid search on test directly) → upper-bound AUC
  7.  TTE bucket analysis  (ROC-AUC & PR-AUC at 0-1s, 1-2s, 2-3s, >3s)
  8.  Temporal prediction jitter  (per-track σ and mean |ΔP| across windows)
  9.  Global branch ablation  (h_global zeroed before visual fusion)
 10.  Pearson error-correlation matrix across branches

After all seeds
───────────────
 11.  Aggregate mean ± std for every numeric metric
 12.  Full CSV export (per-seed + aggregate) + JSON

Usage
─────
  PYTHONPATH=/workspace/project python eval_multiseed_v4.py \\
    --pie_root /Datasets/PIE_PREP_OUT \\
    --seeds 42 43 44 45 46 \\
    --ckpt_template "checkpoints_v4_best_seed{seed}/stage3_pie_baseline/best_model.pth" \\
    --batch_size 64 --amp --out_dir eval_multiseed_results

Notes on jitter
───────────────
  Jitter is measured by grouping all NPZ windows that share the same
  (set_id, video_name, pid) key — i.e. they belong to the same pedestrian
  track but correspond to different temporal windows.  For each multi-window
  track we sort by t_end_full (last raw-frame index) and compute:
    σ(P)      → std of P(crossing) across windows (intra-track spread)
    mean|ΔP|  → mean absolute frame-to-frame change
  A Conv3D model with a rigid sliding window will show high jitter because
  there is no state carried between windows.  A causal Transformer that
  processes the full stream should show much lower values.
"""

import argparse
import csv
import itertools
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# ── optional scipy ───────────────────────────────────────────────────────────
try:
    from scipy.stats import pearsonr as _pearsonr
    _SCIPY = True
except ImportError:
    _SCIPY = False

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# Project root discovery
# ============================================================

def find_project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent] + list(here.parents):
        if (p / "models").exists() and (p / "data").exists():
            return p
    return Path.cwd().resolve()


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.pie import PIESeqDataset  # noqa: E402

try:
    from models.pipnet_alpha_v4_final import PIPNetAlphaV4Final
except ImportError:
    raise ImportError(
        "Cannot import PIPNetAlphaV4Final. "
        "Make sure PYTHONPATH points to your project root."
    )


# ============================================================
# Data loaders
# ============================================================

def make_loader(
    root: str,
    split: str,
    batch_size: int,
    num_workers: int,
    return_meta: bool = False,
) -> DataLoader:
    speed_stats  = PROJECT_ROOT / "data" / "pie_speed_stats_splits.json"
    motion_stats = PROJECT_ROOT / "data" / "pie_motion_stats_splits.json"
    for p in [speed_stats, motion_stats]:
        if not p.exists():
            raise FileNotFoundError(f"Missing stats file: {p}")

    ds = PIESeqDataset(
        root, split=split, mode="eval", seq_len=10, strict_len=True,
        speed_norm="minmax",  speed_stats_path=str(speed_stats),  speed_scope="global",
        motion_norm="p99abs", motion_stats_path=str(motion_stats), motion_scope="global",
        motion_clip=1.0,
        return_meta=return_meta,
    )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )


# ============================================================
# Inference
# ============================================================

@torch.no_grad()
def collect(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool = False,
    ablate_global: bool = False,
    collect_meta: bool = False,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Optional[List[dict]]]:
    """
    Run full inference pass.

    Returns
    -------
    y           : (N,) int32 ground-truth labels
    logits      : dict[branch -> (N,) float32 logits]
    metas       : list of per-sample meta dicts, or None
    """
    model.eval()
    model._ablate_global = ablate_global

    ys     = []
    logits = {k: [] for k in ["main", "kin", "local", "global"]}
    metas: Optional[List[dict]] = [] if collect_meta else None
    use_amp = amp and device.type == "cuda"

    for batch in loader:
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            out = model(batch, return_aux=True)

        ys.append(batch["label"].detach().cpu().numpy().astype(np.int32))
        logits["main"].append(out["logit"].squeeze(-1).detach().float().cpu().numpy())
        logits["kin"].append(out["aux_kin"].squeeze(-1).detach().float().cpu().numpy())
        logits["local"].append(out["aux_local"].squeeze(-1).detach().float().cpu().numpy())
        logits["global"].append(out["aux_global"].squeeze(-1).detach().float().cpu().numpy())

        if collect_meta and "meta" in batch:
            batch_meta = batch["meta"]
            bs = int(batch["label"].shape[0])
            for i in range(bs):
                entry: dict = {}
                for mk, mv in batch_meta.items():
                    if isinstance(mv, (list, tuple)):
                        entry[mk] = mv[i] if i < len(mv) else ""
                    elif torch.is_tensor(mv):
                        entry[mk] = mv[i].cpu().numpy()
                    elif isinstance(mv, np.ndarray):
                        entry[mk] = mv[i]
                    else:
                        entry[mk] = mv
                metas.append(entry)  # type: ignore[union-attr]

    model._ablate_global = False
    return (
        np.concatenate(ys),
        {k: np.concatenate(v) for k, v in logits.items()},
        metas,
    )


# ============================================================
# Metric helpers
# ============================================================

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def safe_roc_auc(y: np.ndarray, logit: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, sigmoid(logit)))


# backwards-compat alias
safe_auc = safe_roc_auc


def safe_pr_auc(y: np.ndarray, logit: np.ndarray) -> float:
    """
    PR-AUC (Average Precision).
    More informative than ROC-AUC on imbalanced datasets — directly measures
    how well the model ranks the positive (crossing) class.
    """
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, sigmoid(logit)))


def safe_mcc(y: np.ndarray, pred: np.ndarray) -> float:
    """
    Matthews Correlation Coefficient.
    A single balanced metric that accounts for all four confusion-matrix cells.
    Ranges from -1 (worst) to +1 (perfect).  Robust to class imbalance.
    """
    try:
        return float(matthews_corrcoef(y, pred))
    except Exception:
        return float("nan")


def metrics_at_threshold(y: np.ndarray, probs: np.ndarray, thr: float) -> dict:
    pred = (probs >= thr).astype(np.int32)
    return {
        "thr":       float(thr),
        "f1":        float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall":    float(recall_score(y, pred, zero_division=0)),
        "acc":       float(accuracy_score(y, pred)),
        "mcc":       safe_mcc(y, pred),
    }


def best_f1_threshold(
    y: np.ndarray,
    logit: np.ndarray,
    step: float = 0.01,
) -> Tuple[float, dict]:
    """Select threshold maximising F1 on the provided split."""
    probs = sigmoid(logit)
    thresholds = np.unique(np.concatenate([np.arange(0.01, 1.0, step), [0.5]]))
    best_m: Optional[dict] = None
    best_thr = 0.5
    for thr in thresholds:
        m = metrics_at_threshold(y, probs, thr)
        if best_m is None or m["f1"] > best_m["f1"] or (
            abs(m["f1"] - best_m["f1"]) < 1e-12 and m["acc"] > best_m["acc"]
        ):
            best_m, best_thr = m, thr
    return best_thr, best_m  # type: ignore[return-value]


def safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    if not _SCIPY or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    r, _ = _pearsonr(a, b)
    return float(r)


# ============================================================
# Expected Calibration Error  (ECE)
# ============================================================

def expected_calibration_error(
    y: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    ECE measures how aligned model confidence is with empirical accuracy.
    ECE = 0  → perfectly calibrated.
    ECE > 0  → confidence does not reflect true probability (over/under-confident).

    Formula: ECE = Σ_b (|B_b| / N) * |acc(B_b) - conf(B_b)|
    """
    bins  = np.linspace(0.0, 1.0, n_bins + 1)
    ece   = 0.0
    n     = len(y)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        mean_conf = float(probs[mask].mean())
        mean_acc  = float(y[mask].mean())
        ece += (mask.sum() / n) * abs(mean_conf - mean_acc)
    return float(ece)


# ============================================================
# TTE bucket analysis
# ============================================================

# (label, lo_sec_inclusive, hi_sec_exclusive)
TTE_BUCKETS = [
    ("0_1s",  0.0,  1.0),
    ("1_2s",  1.0,  2.0),
    ("2_3s",  2.0,  3.0),
    ("gt3s",  3.0,  9999.0),
]


def tte_bucket_metrics(
    y: np.ndarray,
    logit: np.ndarray,
    tte: np.ndarray,
) -> dict:
    """
    Compute ROC-AUC and PR-AUC broken down by time-to-event bucket.

    Parameters
    ----------
    y     : ground-truth labels (N,)
    logit : main branch logits (N,)
    tte   : tte_sec_actual per sample (N,)  — values <= 0 are skipped.

    Why it matters
    ──────────────
    A model might look good overall but fail at distant horizons (>3 s)
    which is exactly where early-warning is most needed.  Conv3D models
    tend to drop sharply beyond 2 s because the crossing signal is not
    yet visible in the local crop.  A Transformer that leverages long-range
    context should maintain higher AUC at distant horizons.
    """
    out: dict = {}
    valid = tte > 0.0
    y_v, logit_v, tte_v = y[valid], logit[valid], tte[valid]

    for label, lo, hi in TTE_BUCKETS:
        mask = (tte_v >= lo) & (tte_v < hi)
        n = int(mask.sum())
        out[f"tte_{label}_n"]       = n
        if n < 10 or len(np.unique(y_v[mask])) < 2:
            out[f"tte_{label}_roc_auc"] = float("nan")
            out[f"tte_{label}_pr_auc"]  = float("nan")
        else:
            out[f"tte_{label}_roc_auc"] = safe_roc_auc(y_v[mask], logit_v[mask])
            out[f"tte_{label}_pr_auc"]  = safe_pr_auc(y_v[mask], logit_v[mask])
    return out


# ============================================================
# Temporal prediction jitter
# ============================================================

def compute_temporal_jitter(
    metas: List[dict],
    probs: np.ndarray,
) -> dict:
    """
    Group NPZ windows by (set_id, video_name, pid) to reconstruct per-track
    prediction sequences.  Sort by t_end_full (last raw frame index) to
    respect temporal order.

    Metrics
    ───────
    jitter_mean_std   — mean of per-track std(P(crossing))
                        measures how much the confidence varies within a track
    jitter_mean_delta — mean of per-track mean|P[t] - P[t-1]|
                        measures frame-to-frame instability (phantom braking signal)
    jitter_n_tracks   — number of tracks with ≥ 2 windows (needed for measurement)

    Interpretation
    ──────────────
    Conv3D with a rigid sliding window: each window is processed independently,
    so consecutive windows may give very different predictions → high jitter.
    A causal Transformer that carries state across time should show lower values.
    """
    if not metas:
        return {
            "jitter_mean_std":   float("nan"),
            "jitter_mean_delta": float("nan"),
            "jitter_n_tracks":   0,
        }

    tracks: Dict[str, List[Tuple[int, float]]] = defaultdict(list)

    for i, meta in enumerate(metas):
        sid   = str(meta.get("set_id",    ""))
        vid   = str(meta.get("video_name",""))
        pid   = str(meta.get("pid",       ""))
        t_end = int(np.array(meta.get("t_end_full", i)).reshape(-1)[0])
        key   = f"{sid}/{vid}/{pid}"
        tracks[key].append((t_end, float(probs[i])))

    stds:   List[float] = []
    deltas: List[float] = []

    for entries in tracks.values():
        if len(entries) < 2:
            continue
        entries.sort(key=lambda x: x[0])
        ps = np.array([e[1] for e in entries])
        stds.append(float(np.std(ps)))
        deltas.append(float(np.mean(np.abs(np.diff(ps)))))

    if not stds:
        return {
            "jitter_mean_std":   float("nan"),
            "jitter_mean_delta": float("nan"),
            "jitter_n_tracks":   0,
        }

    return {
        "jitter_mean_std":   float(np.mean(stds)),
        "jitter_mean_delta": float(np.mean(deltas)),
        "jitter_n_tracks":   len(stds),
    }


# ============================================================
# Inference speed measurement
# ============================================================

def measure_fps(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool = True,
    warmup: int = 10,
    runs:   int = 100,
) -> Tuple[float, float]:
    """
    Measure strictly forward-pass FPS (no DataLoader overhead).

    Returns
    -------
    fps             : frames processed per second
    latency_ms      : milliseconds per single sample

    Implementation notes
    ────────────────────
    - Uses CUDA events when available (sub-ms accuracy).
    - Falls back to perf_counter on CPU.
    - A warmup phase ensures CUDA kernels are fully compiled/cached.
    """
    model.eval()
    use_amp = amp and device.type == "cuda"

    # grab one batch and keep it resident on device
    batch = next(iter(loader))
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device)
    batch_size = int(batch["label"].shape[0])

    # warmup
    with torch.no_grad():
        for _ in range(warmup):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                _ = model(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # timed
    if device.type == "cuda":
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        with torch.no_grad():
            for _ in range(runs):
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    _ = model(batch)
        t1.record()
        torch.cuda.synchronize()
        total_ms = float(t0.elapsed_time(t1))
    else:
        t_start = time.perf_counter()
        with torch.no_grad():
            for _ in range(runs):
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    _ = model(batch)
        total_ms = (time.perf_counter() - t_start) * 1000.0

    ms_per_batch  = total_ms / runs
    ms_per_sample = ms_per_batch / batch_size
    fps           = 1000.0 / ms_per_sample
    return float(fps), float(ms_per_sample)


# ============================================================
# Post-hoc fusion
# ============================================================

def grid_search_fusion(
    y: np.ndarray,
    z: Dict[str, np.ndarray],
    step: float = 0.05,
) -> Tuple[float, Dict[str, float]]:
    """
    Exhaustive grid search over convex combinations of branch logits.
    Optimises ROC-AUC on the provided (y, z) split.
    Returns (best_auc, best_weight_dict).
    """
    names = ["main", "kin", "local", "global"]
    grid  = np.arange(0.0, 1.0 + 1e-9, step)
    best_auc, best_w = -1.0, None

    for ws in itertools.product(grid, repeat=4):
        if abs(sum(ws) - 1.0) > 1e-6:
            continue
        fused = sum(w * z[n] for w, n in zip(ws, names))
        score = safe_roc_auc(y, fused)
        if score > best_auc:
            best_auc, best_w = score, dict(zip(names, [float(w) for w in ws]))

    return best_auc, best_w  # type: ignore[return-value]


def apply_fusion(z: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
    return sum(weights[k] * z[k] for k in weights)


# ============================================================
# Per-seed evaluation
# ============================================================

def evaluate_seed(
    seed: int,
    ckpt_path: str,
    val_loader: DataLoader,
    test_loader: DataLoader,
    test_loader_meta: DataLoader,
    device: torch.device,
    amp: bool,
    grid_step: float,
    thr_step: float,
    fps_runs: int,
) -> dict:

    print(f"\n{'=' * 72}")
    print(f"  SEED {seed}  |  {ckpt_path}")
    print(f"{'=' * 72}")

    if not os.path.isfile(ckpt_path):
        print(f"  [SKIP] Checkpoint not found: {ckpt_path}")
        return {}

    # ── load model ──────────────────────────────────────────────────────────
    model = PIPNetAlphaV4Final(dropout_p=0.5).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"  Loaded  epoch={ckpt.get('epoch','?')}  "
          f"val_auc_at_save={ckpt.get('val_metrics',{}).get('auc','?')}")

    results: dict = {"seed": seed, "ckpt": ckpt_path}
    branches = ["main", "kin", "local", "global"]

    # ── 1. Parameter count ───────────────────────────────────────────────────
    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    results["total_parameters"]     = total_p
    results["trainable_parameters"] = trainable_p
    print(f"\n  [1] Parameters:  total={total_p:,}  trainable={trainable_p:,}")

    # ── 2. Inference speed ───────────────────────────────────────────────────
    print(f"\n  [2] Measuring inference speed ({fps_runs} forward passes)...")
    try:
        fps, lat_ms = measure_fps(model, test_loader, device, amp, runs=fps_runs)
        results["fps"]        = fps
        results["latency_ms"] = lat_ms
        print(f"      FPS={fps:.1f}   latency={lat_ms:.3f} ms/sample")
    except Exception as e:
        print(f"      [WARN] FPS measurement failed: {e}")
        results["fps"]        = float("nan")
        results["latency_ms"] = float("nan")

    # ── 3. Inference: collect logits ─────────────────────────────────────────
    print("\n  Running val inference...")
    y_val,  z_val,  _           = collect(model, val_loader,       device, amp)
    print("  Running test inference (with meta for jitter + TTE)...")
    y_test, z_test, test_metas  = collect(model, test_loader_meta, device, amp,
                                          collect_meta=True)

    # ── 3. Branch ROC-AUC + PR-AUC ───────────────────────────────────────────
    print(f"\n  [3] Branch ROC-AUC / PR-AUC")
    print(f"  {'branch':>8s}  {'val_roc':>8s}  {'test_roc':>9s}  "
          f"{'val_pr':>7s}  {'test_pr':>8s}")
    for br in branches:
        v_roc = safe_roc_auc(y_val,  z_val[br])
        t_roc = safe_roc_auc(y_test, z_test[br])
        v_pr  = safe_pr_auc(y_val,   z_val[br])
        t_pr  = safe_pr_auc(y_test,  z_test[br])
        results[f"val_roc_auc_{br}"]  = v_roc
        results[f"test_roc_auc_{br}"] = t_roc
        results[f"val_pr_auc_{br}"]   = v_pr
        results[f"test_pr_auc_{br}"]  = t_pr
        print(f"  {br:>8s}  {v_roc:>8.4f}  {t_roc:>9.4f}  "
              f"{v_pr:>7.4f}  {t_pr:>8.4f}")

    # ── 4. Val-tuned threshold → test metrics (F1, Prec, Rec, Acc, MCC) ─────
    print(f"\n  [4] Val-tuned F1 threshold (step={thr_step})")
    print(f"  {'branch':>8s}  {'thr':>5s}  "
          f"{'f1':>7s}  {'prec':>7s}  {'rec':>7s}  {'acc':>7s}  {'mcc':>7s}")
    for br in branches:
        best_thr, val_m   = best_f1_threshold(y_val,  z_val[br], thr_step)
        test_tuned_m      = metrics_at_threshold(y_test, sigmoid(z_test[br]), best_thr)
        test_default_m    = metrics_at_threshold(y_test, sigmoid(z_test[br]), 0.5)
        for k, v in val_m.items():
            results[f"val_best_{k}_{br}"] = v
        for k, v in test_tuned_m.items():
            results[f"test_tuned_{k}_{br}"] = v
        for k, v in test_default_m.items():
            results[f"test_default_{k}_{br}"] = v
        print(f"  {br:>8s}  {best_thr:>5.2f}  "
              f"{test_tuned_m['f1']:>7.4f}  {test_tuned_m['precision']:>7.4f}  "
              f"{test_tuned_m['recall']:>7.4f}  {test_tuned_m['acc']:>7.4f}  "
              f"{test_tuned_m['mcc']:>7.4f}")

    # ── 5. ECE ───────────────────────────────────────────────────────────────
    print(f"\n  [5] Expected Calibration Error (ECE, 15 bins, test)")
    for br in branches:
        ece = expected_calibration_error(y_test, sigmoid(z_test[br]))
        results[f"test_ece_{br}"] = ece
        print(f"      {br:>8s}: ECE={ece:.4f}")

    # ── 6. Post-hoc fusion ───────────────────────────────────────────────────
    print(f"\n  [6] Post-hoc logit fusion (grid_step={grid_step})")

    # 6a: val-tuned
    print("      Grid search on VAL...")
    _, val_w       = grid_search_fusion(y_val, z_val, grid_step)
    fused_val      = apply_fusion(z_val,  val_w)
    fused_test     = apply_fusion(z_test, val_w)

    fus_val_roc    = safe_roc_auc(y_val,  fused_val)
    fus_test_roc   = safe_roc_auc(y_test, fused_test)
    fus_val_pr     = safe_pr_auc(y_val,   fused_val)
    fus_test_pr    = safe_pr_auc(y_test,  fused_test)
    fus_test_ece   = expected_calibration_error(y_test, sigmoid(fused_test))

    best_fus_thr, val_fus_m = best_f1_threshold(y_val, fused_val, thr_step)
    test_fus_tuned   = metrics_at_threshold(y_test, sigmoid(fused_test), best_fus_thr)
    test_fus_default = metrics_at_threshold(y_test, sigmoid(fused_test), 0.5)

    results["fusion_val_weights"]          = val_w
    results["fusion_val_roc_auc"]          = fus_val_roc
    results["fusion_test_roc_auc"]         = fus_test_roc
    results["fusion_val_pr_auc"]           = fus_val_pr
    results["fusion_test_pr_auc"]          = fus_test_pr
    results["fusion_test_ece"]             = fus_test_ece
    results["fusion_best_thr"]             = best_fus_thr
    for k, v in test_fus_tuned.items():
        results[f"fusion_test_tuned_{k}"]  = v
    for k, v in test_fus_default.items():
        results[f"fusion_test_default_{k}"] = v

    print(f"      Val-tuned weights: {val_w}")
    print(f"      val   ROC={fus_val_roc:.4f}  PR={fus_val_pr:.4f}")
    print(f"      test  ROC={fus_test_roc:.4f}  PR={fus_test_pr:.4f}  ECE={fus_test_ece:.4f}")
    print(f"      test (thr=0.50)          f1={test_fus_default['f1']:.4f}  "
          f"mcc={test_fus_default['mcc']:.4f}")
    print(f"      test (thr={best_fus_thr:.2f} tuned)    f1={test_fus_tuned['f1']:.4f}  "
          f"mcc={test_fus_tuned['mcc']:.4f}")

    # 6b: oracle (upper bound — search directly on test)
    print("      Oracle grid search on TEST (upper-bound, not for reporting)...")
    oracle_roc, oracle_w = grid_search_fusion(y_test, z_test, grid_step)
    oracle_fused         = apply_fusion(z_test, oracle_w)
    oracle_pr            = safe_pr_auc(y_test, oracle_fused)
    results["oracle_test_roc_auc"] = oracle_roc
    results["oracle_test_pr_auc"]  = oracle_pr
    results["oracle_test_weights"] = oracle_w
    print(f"      Oracle weights: {oracle_w}")
    print(f"      Oracle test  ROC={oracle_roc:.4f}  PR={oracle_pr:.4f}")

    # ── 7. TTE bucket analysis ───────────────────────────────────────────────
    print(f"\n  [7] TTE bucket analysis (main branch)")
    tte_vals: Optional[np.ndarray] = None
    if test_metas:
        raw = []
        for m in test_metas:
            v = m.get("tte_sec_actual", None)
            try:
                raw.append(float(np.array(v).reshape(-1)[0]) if v is not None else float("nan"))
            except Exception:
                raw.append(float("nan"))
        tte_arr = np.array(raw, dtype=np.float32)
        if not np.all(np.isnan(tte_arr)):
            tte_vals = tte_arr

    if tte_vals is not None and len(tte_vals) == len(y_test):
        tte_res = tte_bucket_metrics(y_test, z_test["main"], tte_vals)
        results.update(tte_res)
        print(f"  {'bucket':>8s}  {'n':>6s}  {'roc_auc':>8s}  {'pr_auc':>8s}")
        for lbl, *_ in TTE_BUCKETS:
            roc = results.get(f"tte_{lbl}_roc_auc", float("nan"))
            pr  = results.get(f"tte_{lbl}_pr_auc",  float("nan"))
            n   = results.get(f"tte_{lbl}_n", 0)
            print(f"  {lbl:>8s}  {n:>6d}  {roc:>8.4f}  {pr:>8.4f}")
    else:
        print("      [SKIP] tte_sec_actual not available in meta.")
        for lbl, *_ in TTE_BUCKETS:
            results[f"tte_{lbl}_roc_auc"] = float("nan")
            results[f"tte_{lbl}_pr_auc"]  = float("nan")
            results[f"tte_{lbl}_n"]       = 0

    # ── 8. Temporal jitter ───────────────────────────────────────────────────
    print(f"\n  [8] Temporal prediction jitter (main branch, test)")
    if test_metas:
        jitter = compute_temporal_jitter(test_metas, sigmoid(z_test["main"]))
    else:
        jitter = {"jitter_mean_std": float("nan"),
                  "jitter_mean_delta": float("nan"),
                  "jitter_n_tracks": 0}
    results.update(jitter)
    print(f"      Tracks with ≥2 windows  : {jitter['jitter_n_tracks']}")
    print(f"      Mean intra-track σ(P)   : {jitter['jitter_mean_std']:.4f}")
    print(f"      Mean |ΔP| per window gap: {jitter['jitter_mean_delta']:.4f}")

 # ── 9. Global branch ablation ────────────────────────────────────────────
    print(f"\n  [9] Global branch ablation (zeroing h_global)")
    
    # Run full inference passes with the global branch suppressed
    y_val_abl,  z_val_abl,  _ = collect(model, val_loader, device, amp, ablate_global=True)
    y_test_abl, z_test_abl, _ = collect(model, test_loader_meta, device, amp, ablate_global=True)

    abl_val_roc  = safe_roc_auc(y_val_abl,  z_val_abl["main"])
    abl_test_roc = safe_roc_auc(y_test_abl, z_test_abl["main"])
    abl_delta    = results["test_roc_auc_main"] - abl_test_roc

    results["ablation_val_roc_auc_main"]  = abl_val_roc
    results["ablation_test_roc_auc_main"] = abl_test_roc
    results["ablation_delta_test_roc"]    = abl_delta

    verdict = ("HELPS" if abl_delta > 0.005 else "HURTS" if abl_delta < -0.005 else "NEUTRAL")
    print(f"      With global : val={results['val_roc_auc_main']:.4f}  test={results['test_roc_auc_main']:.4f}")
    print(f"      Without     : val={abl_val_roc:.4f}  test={abl_test_roc:.4f}")
    print(f"      Δtest={abl_delta:+.4f}  → global branch {verdict}")

    # ── 10. Pearson error-correlation matrix ─────────────────────────────────
    print("\n  [10] Pearson error correlation (test, thr=0.5)")
    y_bin = (y_test >= 0.5).astype(np.int32)
    
    # Binary error arrays per branch
    errors = {
        br: np.abs((sigmoid(z_test[br]) >= 0.5).astype(np.int32) - y_bin) 
        for br in branches
    }
    
    pairs = [
        ("main","kin"), ("main","local"), ("main","global"),
        ("kin","local"), ("kin","global"), ("local","global")
    ]
    
    for a, b in pairs:
        r = safe_pearson(errors[a], errors[b])
        results[f"pearson_{a}_{b}"] = r
        print(f"      {a:>5s} vs {b:<6s} : r = {r:.4f}")

    return results


# ============================================================
# CSV Export & Aggregation
# ============================================================

def flatten_for_csv(r: dict) -> dict:
    """Flatten nested dicts/lists so every value is a scalar string."""
    out = {}
    for k, v in r.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                out[f"{k}__{kk}"] = f"{vv:.6f}" if isinstance(vv, float) else str(vv)
        elif isinstance(v, (list, np.ndarray)):
            out[k] = str(v)
        elif isinstance(v, float):
            out[k] = f"{v:.6f}"
        else:
            out[k] = str(v)
    return out


def write_csv(path: str, rows: List[dict]):
    """Safely write a list of dictionaries to a CSV file."""
    if not rows:
        return
    flat_rows = [flatten_for_csv(r) for r in rows]
    
    # Gather all unique keys across all rows
    all_keys = []
    seen = set()
    for r in flat_rows:
        for k in r:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
                
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        for r in flat_rows:
            w.writerow(r)
    print(f"\nCSV written → {path}")


def aggregate(all_results: List[dict]) -> dict:
    """Calculates Mean ± Std for every numeric metric across all seeds."""
    numeric_keys = {}
    for r in all_results:
        for k, v in r.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and k != "seed":
                numeric_keys.setdefault(k, []).append(float(v))

    agg = {}
    for k, vals in numeric_keys.items():
        arr = np.array([v for v in vals if not np.isnan(v)])
        if arr.size:
            agg[k] = {"mean": float(arr.mean()), "std": float(arr.std()), "n": int(arr.size)}
    return agg


# ============================================================
# Main Loop
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="Multi-seed evaluation for PIPNet-Alpha V4")
    ap.add_argument("--pie_root", required=True, help="Path to PIE_PREP_OUT")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--ckpt_template", required=True, help="Path template with {seed}")
    ap.add_argument("--batch_size",  type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--amp",         action="store_true")
    ap.add_argument("--grid_step",   type=float, default=0.05)
    ap.add_argument("--thr_step",    type=float, default=0.01)
    ap.add_argument("--fps_runs",    type=int, default=100)
    ap.add_argument("--out_dir",     default="eval_multiseed_results")
    ap.add_argument("--device",      default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 72)
    print("  PIPNet Alpha V4 — Comprehensive Multi-Seed Evaluation")
    print("=" * 72)
    print(f"  Project root : {PROJECT_ROOT}")
    print(f"  PIE root     : {args.pie_root}")
    print(f"  Seeds        : {args.seeds}")
    print(f"  Device       : {device}  AMP={args.amp and device.type=='cuda'}")
    print(f"  Output dir   : {out_dir}")
    print("=" * 72)

    print("\nBuilding data loaders...")
    val_loader       = make_loader(args.pie_root, "val",  args.batch_size, args.num_workers)
    test_loader      = make_loader(args.pie_root, "test", args.batch_size, args.num_workers)
    test_loader_meta = make_loader(args.pie_root, "test", args.batch_size, args.num_workers, return_meta=True)
    
    print(f"  Val  : {len(val_loader.dataset)} samples")
    print(f"  Test : {len(test_loader.dataset)} samples")

    all_results: List[dict] = []

    # Run pipeline for each seed
    for seed in args.seeds:
        ckpt_path = args.ckpt_template.format(seed=seed)
        r = evaluate_seed(
            seed=seed,
            ckpt_path=ckpt_path,
            val_loader=val_loader,
            test_loader=test_loader,
            test_loader_meta=test_loader_meta,
            device=device,
            amp=args.amp,
            grid_step=args.grid_step,
            thr_step=args.thr_step,
            fps_runs=args.fps_runs,
        )
        if r:
            all_results.append(r)

    if not all_results:
        print("\n[ERROR] No seeds evaluated successfully. Exiting.")
        return

    # Aggregate & Export
    agg = aggregate(all_results)
    
    per_seed_csv  = out_dir / f"per_seed_{timestamp}.csv"
    aggregate_csv = out_dir / f"aggregate_{timestamp}.csv"
    aggregate_json = out_dir / f"aggregate_{timestamp}.json"

    write_csv(str(per_seed_csv), all_results)
    
    agg_rows = [{"metric": k, "mean": v["mean"], "std": v["std"], "n": v["n"]} 
                for k, v in sorted(agg.items())]
    write_csv(str(aggregate_csv), agg_rows)

    with open(aggregate_json, "w") as f:
        json.dump(agg, f, indent=2)

    # ── Final Summary Table ──────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("  FINAL AGGREGATE SUMMARY (MEAN ± STD)")
    print(f"{'=' * 72}")
    
    def p_metric(name: str, key: str):
        if key in agg:
            print(f"  {name:<30s} : {agg[key]['mean']:.4f} ± {agg[key]['std']:.4f}")

    print("\n[ Main Branch (Test) ]")
    p_metric("ROC-AUC", "test_roc_auc_main")
    p_metric("PR-AUC", "test_pr_auc_main")
    p_metric("F1 (Tuned)", "test_tuned_f1_main")
    p_metric("MCC (Tuned)", "test_tuned_mcc_main")
    p_metric("ECE", "test_ece_main")

    print("\n[ Anticipation Time (ROC-AUC) ]")
    p_metric("0 - 1s", "tte_0_1s_roc_auc")
    p_metric("1 - 2s", "tte_1_2s_roc_auc")
    p_metric("2 - 3s", "tte_2_3s_roc_auc")

    print("\n[ Stability & Speed ]")
    p_metric("Jitter (Track σ)", "jitter_mean_std")
    p_metric("FPS", "fps")

    print("\n[ Fusion (Test) ]")
    p_metric("Val-Tuned Fusion ROC-AUC", "fusion_test_roc_auc")
    p_metric("Oracle Fusion ROC-AUC", "oracle_test_roc_auc")

    print(f"\nResults saved to: {out_dir}")
    print(f"{'=' * 72}")
    print("  Evaluation Complete.")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main() 