#!/usr/bin/env python3
"""
eval_branch_fusion_v4.py

Evaluation script for PIPNet-Alpha V4.

Does:
  1. Individual branch AUCs:
       main, kin, local, global

  2. Validation-based F1 threshold tuning:
       selects best F1 threshold on VAL, applies it once to TEST

  3. Val-tuned post-hoc logit fusion:
       grid search over main/kin/local/global logits

  4. F1 threshold tuning for post-hoc fusion:
       selects best threshold on VAL fused logits, applies it to TEST fused logits

  5. Global ablation:
       re-run model with global branch zeroed before visual fusion

  6. Pearson error correlation:
       checks if branches make similar or different mistakes

Usage:

    PYTHONPATH=. python eval/eval_branch_fusion_v4.py \
      --pie_root /data/PIE_PREP_OUT \
      --ckpt checkpoints_v4_balanced_attn_seed42/stage3_pie_baseline/best_model.pth \
      --batch_size 10 \
      --amp

Faster version without grid search:

    PYTHONPATH=. python eval/eval_branch_fusion_v4.py \
      --pie_root /data/PIE_PREP_OUT \
      --ckpt checkpoints_v4_balanced_attn_seed42/stage3_pie_baseline/best_model.pth \
      --batch_size 10 \
      --amp \
      --skip_grid
"""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
from scipy.stats import pearsonr


# ============================================================
# Project root setup
# ============================================================

def find_project_root() -> Path:
    """
    Find project root whether this script is in:
      project/eval/eval_branch_fusion_v4.py
    or:
      project/eval_branch_fusion_v4.py
    """
    here = Path(__file__).resolve()

    for p in [here.parent] + list(here.parents):
        if (p / "models").exists() and (p / "data").exists():
            return p

    return Path.cwd().resolve()


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from data.pie import PIESeqDataset

try:
    from models.pipnet_alpha_v4_final import PIPNetAlphaV4Final
except ImportError:
    from models.pipnet_alpha_v4_final import PIPNetAlphaV3Final as PIPNetAlphaV4Final


# ============================================================
# Data loader
# ============================================================

def make_loader(root, split, batch_size, num_workers):
    speed_stats_path = PROJECT_ROOT / "data" / "pie_speed_stats_splits.json"
    motion_stats_path = PROJECT_ROOT / "data" / "pie_motion_stats_splits.json"

    if not speed_stats_path.exists():
        raise FileNotFoundError(f"Missing speed stats file: {speed_stats_path}")

    if not motion_stats_path.exists():
        raise FileNotFoundError(f"Missing motion stats file: {motion_stats_path}")

    ds = PIESeqDataset(
        root,
        split=split,
        mode="eval",
        seq_len=10,
        strict_len=True,

        speed_norm="minmax",
        speed_stats_path=str(speed_stats_path),
        speed_scope="global",

        motion_norm="p99abs",
        motion_stats_path=str(motion_stats_path),
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


# ============================================================
# Inference
# ============================================================

@torch.no_grad()
def collect(model, loader, device, amp=False, ablate_global=False):
    """
    Run inference and collect logits for all four heads.

    ablate_global=True sets model._ablate_global = True.
    In V4 this zeroes h_global before visual fusion.
    """
    model.eval()
    model._ablate_global = ablate_global

    ys = []
    logits = {
        "main": [],
        "kin": [],
        "local": [],
        "global": [],
    }

    use_amp = bool(amp and device.type == "cuda")

    for batch in loader:
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            out = model(batch, return_aux=True)

        ys.append(batch["label"].detach().cpu().numpy().astype(np.int32))

        logits["main"].append(
            out["logit"].squeeze(-1).detach().float().cpu().numpy()
        )
        logits["kin"].append(
            out["aux_kin"].squeeze(-1).detach().float().cpu().numpy()
        )
        logits["local"].append(
            out["aux_local"].squeeze(-1).detach().float().cpu().numpy()
        )
        logits["global"].append(
            out["aux_global"].squeeze(-1).detach().float().cpu().numpy()
        )

    model._ablate_global = False

    y = np.concatenate(ys)
    z = {k: np.concatenate(v) for k, v in logits.items()}

    return y, z


# ============================================================
# Helpers
# ============================================================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def auc(y, logit):
    if len(np.unique(y)) < 2:
        return float("nan")
    return roc_auc_score(y, sigmoid(logit))


def acc(y, logit, thr=0.5):
    pred = (sigmoid(logit) >= thr).astype(np.int32)
    return float((pred == y).mean())


def binary_metrics_from_probs(y, probs, thr):
    pred = (probs >= thr).astype(np.int32)

    return {
        "thr": float(thr),
        "f1": f1_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "acc": accuracy_score(y, pred),
    }


def threshold_sweep(y_val, z_val, y_test, z_test, name="main", step=0.01):
    """
    Select best F1 threshold on validation set.
    Apply that threshold once to test set.

    This is fair:
      VAL chooses threshold.
      TEST only evaluates the selected threshold.
    """
    val_probs = sigmoid(z_val[name])
    test_probs = sigmoid(z_test[name])

    thresholds = np.arange(0.01, 0.99 + 1e-9, step)
    thresholds = np.unique(np.concatenate([thresholds, np.array([0.50])]))

    best_val = None

    for thr in thresholds:
        m = binary_metrics_from_probs(y_val, val_probs, thr)

        if best_val is None:
            best_val = m
        else:
            if (m["f1"] > best_val["f1"]) or (
                abs(m["f1"] - best_val["f1"]) < 1e-12
                and m["acc"] > best_val["acc"]
            ):
                best_val = m

    default_val = binary_metrics_from_probs(y_val, val_probs, 0.50)
    default_test = binary_metrics_from_probs(y_test, test_probs, 0.50)

    tuned_test = binary_metrics_from_probs(
        y_test,
        test_probs,
        best_val["thr"],
    )

    print("\n" + "=" * 60)
    print(f"F1 threshold tuning — {name}")
    print("=" * 60)

    print("Default threshold = 0.50")
    print(
        f"  VAL  | f1={default_val['f1']:.4f}  "
        f"prec={default_val['precision']:.4f}  "
        f"rec={default_val['recall']:.4f}  "
        f"acc={default_val['acc']:.4f}"
    )
    print(
        f"  TEST | f1={default_test['f1']:.4f}  "
        f"prec={default_test['precision']:.4f}  "
        f"rec={default_test['recall']:.4f}  "
        f"acc={default_test['acc']:.4f}"
    )

    print(f"\nBest threshold selected on VAL = {best_val['thr']:.2f}")
    print(
        f"  VAL  | f1={best_val['f1']:.4f}  "
        f"prec={best_val['precision']:.4f}  "
        f"rec={best_val['recall']:.4f}  "
        f"acc={best_val['acc']:.4f}"
    )
    print(
        f"  TEST | f1={tuned_test['f1']:.4f}  "
        f"prec={tuned_test['precision']:.4f}  "
        f"rec={tuned_test['recall']:.4f}  "
        f"acc={tuned_test['acc']:.4f}"
    )

    return best_val["thr"], best_val, tuned_test


def threshold_sweep_logits(
    y_val,
    logits_val,
    y_test,
    logits_test,
    name="post_hoc_fusion",
    step=0.01,
):
    """
    Same as threshold_sweep(), but accepts raw logits directly.
    Useful for post-hoc fused logits.
    """
    val_probs = sigmoid(logits_val)
    test_probs = sigmoid(logits_test)

    thresholds = np.arange(0.01, 0.99 + 1e-9, step)
    thresholds = np.unique(np.concatenate([thresholds, np.array([0.50])]))

    best_val = None

    for thr in thresholds:
        m = binary_metrics_from_probs(y_val, val_probs, thr)

        if best_val is None:
            best_val = m
        else:
            if (m["f1"] > best_val["f1"]) or (
                abs(m["f1"] - best_val["f1"]) < 1e-12
                and m["acc"] > best_val["acc"]
            ):
                best_val = m

    default_val = binary_metrics_from_probs(y_val, val_probs, 0.50)
    default_test = binary_metrics_from_probs(y_test, test_probs, 0.50)

    tuned_test = binary_metrics_from_probs(
        y_test,
        test_probs,
        best_val["thr"],
    )

    print("\n" + "=" * 60)
    print(f"F1 threshold tuning — {name}")
    print("=" * 60)

    print("Default threshold = 0.50")
    print(
        f"  VAL  | f1={default_val['f1']:.4f}  "
        f"prec={default_val['precision']:.4f}  "
        f"rec={default_val['recall']:.4f}  "
        f"acc={default_val['acc']:.4f}"
    )
    print(
        f"  TEST | f1={default_test['f1']:.4f}  "
        f"prec={default_test['precision']:.4f}  "
        f"rec={default_test['recall']:.4f}  "
        f"acc={default_test['acc']:.4f}"
    )

    print(f"\nBest threshold selected on VAL = {best_val['thr']:.2f}")
    print(
        f"  VAL  | f1={best_val['f1']:.4f}  "
        f"prec={best_val['precision']:.4f}  "
        f"rec={best_val['recall']:.4f}  "
        f"acc={best_val['acc']:.4f}"
    )
    print(
        f"  TEST | f1={tuned_test['f1']:.4f}  "
        f"prec={tuned_test['precision']:.4f}  "
        f"rec={tuned_test['recall']:.4f}  "
        f"acc={tuned_test['acc']:.4f}"
    )

    return best_val["thr"], best_val, tuned_test


def safe_pearson(a, b):
    """
    Pearson correlation can be undefined if one vector is constant.
    Return NaN safely instead of crashing.
    """
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    r, _ = pearsonr(a, b)
    return float(r)


# ============================================================
# Method 2 — Grid search post-hoc fusion
# ============================================================

def grid_search(y_val, z_val, step=0.05):
    names = ["main", "kin", "local", "global"]
    grid = np.arange(0.0, 1.0 + 1e-9, step)

    best_auc = -1.0
    best_weights = None

    for ws in itertools.product(grid, repeat=4):
        if abs(sum(ws) - 1.0) > 1e-6:
            continue

        fused = sum(w * z_val[n] for w, n in zip(ws, names))
        score = auc(y_val, fused)

        if score > best_auc:
            best_auc = score
            best_weights = dict(zip(names, ws))

    return best_auc, best_weights


def apply_weights(z, weights):
    return sum(weights[k] * z[k] for k in weights)


# ============================================================
# Method 3 — Pearson error correlation matrix
# ============================================================

def error_correlation(y, z, threshold=0.5):
    """
    For each pair of branches compute Pearson r between binary error vectors:

      1 = wrong prediction
      0 = correct prediction
    """
    names = ["main", "kin", "local", "global"]
    y_bin = (y >= 0.5).astype(np.int32)

    errors = {}

    for n in names:
        pred = (sigmoid(z[n]) >= threshold).astype(np.int32)
        errors[n] = np.abs(pred - y_bin)

    print("\n" + "=" * 60)
    print("Method 3 — Branch error correlation")
    print("=" * 60)
    print("Interpretation:")
    print("  r < 0.30     → complementary")
    print("  r 0.30–0.60  → partial overlap")
    print("  r > 0.60     → redundant")
    print("=" * 60)

    print("\nIndividual error rates:")
    for n in names:
        err_rate = errors[n].mean()
        wrong = int(errors[n].sum())
        print(f"  {n:>6s}: {err_rate:.3f}  ({wrong} wrong / {len(y)} total)")

    print("\nPairwise error correlation:")
    header = f"{'':>8s}" + "".join(f"{n:>10s}" for n in names)
    print(header)
    print("-" * (8 + 10 * len(names)))

    matrix = {}

    for n1 in names:
        row = f"{n1:>8s}"

        for n2 in names:
            if n1 == n2:
                r = 1.0
                matrix[(n1, n2)] = r
                row += f"{'1.000':>10s}"
            else:
                r = safe_pearson(errors[n1], errors[n2])
                matrix[(n1, n2)] = r

                marker = ""
                if not np.isnan(r):
                    if r < 0.30:
                        marker = "✓"
                    elif r > 0.60:
                        marker = "✗"

                if np.isnan(r):
                    row += f"{'nan':>10s}"
                else:
                    row += f"{r:>8.3f}{marker:>2s}"

        print(row)

    print("\nGlobal branch complementarity summary:")
    for n in ["main", "kin", "local"]:
        r = matrix[("global", n)]

        if np.isnan(r):
            verdict = "UNDEFINED — constant error vector"
        elif r < 0.30:
            verdict = "COMPLEMENTARY — global adds unique signal vs " + n
        elif r < 0.60:
            verdict = "PARTIAL — some unique signal vs " + n
        else:
            verdict = "REDUNDANT — global overlaps heavily with " + n

        print(f"  global vs {n:>5s}: r={r:.3f}  → {verdict}")

    return matrix


# ============================================================
# Method 1 — Global ablation
# ============================================================

def run_ablation(model, val_loader, test_loader, device, amp, y_val, z_val, y_test, z_test):
    print("\n" + "=" * 60)
    print("Method 1 — Global branch ablation")
    print("(V4: h_global zeroed before visual fusion)")
    print("=" * 60)

    y_val_abl, z_val_abl = collect(
        model,
        val_loader,
        device,
        amp,
        ablate_global=True,
    )

    y_test_abl, z_test_abl = collect(
        model,
        test_loader,
        device,
        amp,
        ablate_global=True,
    )

    val_with = auc(y_val, z_val["main"])
    test_with = auc(y_test, z_test["main"])

    val_without = auc(y_val_abl, z_val_abl["main"])
    test_without = auc(y_test_abl, z_test_abl["main"])

    print("\nWith global branch:")
    print(f"  main  val={val_with:.4f}  test={test_with:.4f}")

    print("\nWithout global branch:")
    print(f"  main  val={val_without:.4f}  test={test_without:.4f}")

    delta_val = val_with - val_without
    delta_test = test_with - test_without

    print("\nDelta: with global - without global")
    print(f"  val:  {delta_val:+.4f}")
    print(f"  test: {delta_test:+.4f}")

    if delta_test > 0.005:
        print(
            "\n  ✓ Global branch HELPS — removing it hurts test AUC by "
            f"{delta_test:.4f}"
        )
    elif delta_test < -0.005:
        print(
            "\n  ✗ Global branch HURTS — removing it improves test AUC by "
            f"{abs(delta_test):.4f}"
        )
    else:
        print(
            "\n  ~ Global branch is NEUTRAL on test AUC "
            f"(delta={delta_test:.4f})"
        )

    return z_val_abl, z_test_abl


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--pie_root", default="/data/PIE_PREP_OUT")
    ap.add_argument("--ckpt", required=True)

    ap.add_argument("--batch_size", type=int, default=10)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--amp", action="store_true")

    ap.add_argument(
        "--no_global",
        action="store_true",
        help="Run primary inference with global branch zeroed",
    )

    ap.add_argument(
        "--skip_grid",
        action="store_true",
        help="Skip post-hoc grid search",
    )

    ap.add_argument(
        "--grid_step",
        type=float,
        default=0.05,
        help="Grid search step size. Smaller is finer but slower.",
    )

    ap.add_argument(
        "--threshold_step",
        type=float,
        default=0.01,
        help="Threshold sweep step for F1 tuning.",
    )

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("PIPNet Alpha V4 evaluation")
    print("=" * 60)
    print(f"Project root:    {PROJECT_ROOT}")
    print(f"Device:          {device}")
    print(f"AMP:             {args.amp and device.type == 'cuda'}")
    print(f"PIE root:        {args.pie_root}")
    print(f"Checkpoint:      {args.ckpt}")
    print(f"Threshold step:  {args.threshold_step}")
    print("=" * 60)

    # ── Load model ─────────────────────────────────────────────
    model = PIPNetAlphaV4Final(dropout_p=0.5).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])

    print(f"\nLoaded checkpoint: {args.ckpt}")
    print(
        f"  Best epoch: {ckpt.get('epoch', '?')}  |  "
        f"Val AUC at save: {ckpt.get('val_metrics', {}).get('auc', '?')}"
    )

    # ── Loaders ────────────────────────────────────────────────
    val_loader = make_loader(
        args.pie_root,
        "val",
        args.batch_size,
        args.num_workers,
    )

    test_loader = make_loader(
        args.pie_root,
        "test",
        args.batch_size,
        args.num_workers,
    )

    # ── Normal or ablated inference ────────────────────────────
    ablate = args.no_global

    print(f"\nRunning inference: ablate_global={ablate}")

    y_val, z_val = collect(
        model,
        val_loader,
        device,
        args.amp,
        ablate_global=ablate,
    )

    y_test, z_test = collect(
        model,
        test_loader,
        device,
        args.amp,
        ablate_global=ablate,
    )

    # ── Individual AUCs ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Individual branch AUCs")
    print("=" * 60)

    for name in ["main", "kin", "local", "global"]:
        v = auc(y_val, z_val[name])
        t = auc(y_test, z_test[name])
        print(f"  {name:>6s} | val={v:.4f} | test={t:.4f}")

    # ── F1 threshold tuning ────────────────────────────────────
    print("\nRunning validation-based F1 threshold tuning...")

    for name in ["main", "kin", "local", "global"]:
        threshold_sweep(
            y_val,
            z_val,
            y_test,
            z_test,
            name=name,
            step=args.threshold_step,
        )

    # ── Grid search ────────────────────────────────────────────
    if not args.skip_grid:
        print(f"\nRunning grid search with step={args.grid_step}...")

        best_val_auc, weights = grid_search(
            y_val,
            z_val,
            step=args.grid_step,
        )

        val_fused = apply_weights(z_val, weights)
        test_fused = apply_weights(z_test, weights)
        test_auc = auc(y_test, test_fused)

        print("\n" + "=" * 60)
        print("Best val-tuned post-hoc logit fusion")
        print("=" * 60)
        print(f"  weights  = {weights}")
        print(f"  val AUC  = {best_val_auc:.4f}")
        print(f"  test AUC = {test_auc:.4f}")

        main_only_test = auc(y_test, z_test["main"])
        delta = test_auc - main_only_test

        print(f"\n  Main-only test AUC: {main_only_test:.4f}")
        print(f"  Fusion test AUC:    {test_auc:.4f}")

        threshold_sweep_logits(
            y_val,
            val_fused,
            y_test,
            test_fused,
            name="post_hoc_fusion",
            step=args.threshold_step,
        )

        if delta > 0.002:
            print(f"  ✓ Post-hoc fusion adds {delta:+.4f} over main alone")
        elif delta < -0.002:
            print(f"  ✗ Post-hoc fusion loses {delta:+.4f} vs main alone")
        else:
            print(f"  ~ Post-hoc fusion neutral ({delta:+.4f})")

    # ── Error correlation ──────────────────────────────────────
    print("\nRunning error correlation analysis on test set...")
    error_correlation(y_test, z_test)

    # ── Global ablation ────────────────────────────────────────
    if not args.no_global:
        run_ablation(
            model,
            val_loader,
            test_loader,
            device,
            args.amp,
            y_val,
            z_val,
            y_test,
            z_test,
        )
    else:
        print("\n[--no_global was set, so primary run was already ablated.]")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()