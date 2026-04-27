"""
eval_branch_fusion.py

Three things in one script:
  1. Individual branch AUCs (main, kin, local, global)
  2. Val-tuned post-hoc logit fusion (grid search)
  3. Method 1 — Global ablation: re-run with global zeroed out
  4. Method 3 — Pearson error correlation matrix between branches

Usage — normal eval:
    python eval_branch_fusion.py \
        --pie_root /data/PIE_PREP_OUT \
        --ckpt checkpoints_fixes_40ep_seed42/stage3_pie_baseline/best_model.pth \
        --amp

Usage — with global ablation:
    python eval_branch_fusion.py \
        --pie_root /data/PIE_PREP_OUT \
        --ckpt checkpoints_fixes_40ep_seed42/stage3_pie_baseline/best_model.pth \
        --amp \
        --no_global

Usage — correlation only (skip grid search, faster):
    python eval_branch_fusion.py \
        --pie_root /data/PIE_PREP_OUT \
        --ckpt checkpoints_fixes_40ep_seed42/stage3_pie_baseline/best_model.pth \
        --amp \
        --skip_grid
"""

import argparse
import itertools
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

from data.pie import PIESeqDataset
from models.pipnet_alpha_v3_final import PIPNetAlphaV3Final


# ============================================================
# Data loader
# ============================================================

def make_loader(root, split, batch_size, num_workers):
    ds = PIESeqDataset(
        root,
        split=split,
        mode="eval",
        seq_len=10,
        strict_len=True,
        speed_norm="minmax",
        speed_stats_path="/workspace/project/data/pie_speed_stats_splits.json",
        speed_scope="global",
        motion_norm="p99abs",
        motion_stats_path="/workspace/project/data/pie_motion_stats_splits.json",
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

    ablate_global=True zeros out h_global_flat inside the model
    before visual fusion, so the global branch contributes nothing.
    This is set via model._ablate_global before calling.
    """
    model.eval()
    model._ablate_global = ablate_global

    ys = []
    logits = {"main": [], "kin": [], "local": [], "global": []}

    for batch in loader:
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device, non_blocking=True)

        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=amp
        ):
            out = model(batch, return_aux=True)

        ys.append(batch["label"].detach().cpu().numpy().astype(np.int32))
        logits["main"].append(out["logit"].squeeze(-1).detach().float().cpu().numpy())
        logits["kin"].append(out["aux_kin"].squeeze(-1).detach().float().cpu().numpy())
        logits["local"].append(out["aux_local"].squeeze(-1).detach().float().cpu().numpy())
        logits["global"].append(out["aux_global"].squeeze(-1).detach().float().cpu().numpy())

    model._ablate_global = False  # always reset

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
    return float(((sigmoid(logit) >= thr).astype(int) == y).mean())


# ============================================================
# Method 2 — Grid search post-hoc fusion
# ============================================================

def grid_search(y_val, z_val, step=0.05):
    names = ["main", "kin", "local", "global"]
    grid  = np.arange(0.0, 1.0 + 1e-9, step)
    best  = (-1.0, None)

    for ws in itertools.product(grid, repeat=4):
        if abs(sum(ws) - 1.0) > 1e-6:
            continue
        fused = sum(w * z_val[n] for w, n in zip(ws, names))
        score = auc(y_val, fused)
        if score > best[0]:
            best = (score, dict(zip(names, ws)))

    return best


def apply_weights(z, weights):
    return sum(weights[k] * z[k] for k in weights)


# ============================================================
# Method 3 — Pearson error correlation matrix
# ============================================================

def error_correlation(y, z, threshold=0.5):
    """
    For each pair of branches compute Pearson r between their
    binary error vectors (1 = wrong prediction, 0 = correct).

    Low r  → branches make DIFFERENT mistakes → complementary → fusion helps
    High r → branches make SAME mistakes     → redundant     → fusion doesn't help
    """
    names  = ["main", "kin", "local", "global"]
    y_bin  = (y >= 0.5).astype(int)

    errors = {}
    for n in names:
        pred      = (sigmoid(z[n]) >= threshold).astype(int)
        errors[n] = np.abs(pred - y_bin)          # 1 where wrong

    print("\n" + "=" * 60)
    print("Method 3 — Branch error correlation (Pearson r)")
    print("Interpretation:")
    print("  r < 0.30  → complementary  → fusion captures extra signal")
    print("  r 0.30–0.60 → partial overlap")
    print("  r > 0.60  → redundant      → fusion adds little")
    print("=" * 60)

    # Print error rates first
    print("\nIndividual error rates:")
    for n in names:
        err_rate = errors[n].mean()
        print(f"  {n:>6s}: {err_rate:.3f}  ({int(errors[n].sum())} wrong / {len(y)} total)")

    # Correlation matrix
    print("\nPairwise error correlation:")
    header = f"{'':>8s}" + "".join(f"{n:>10s}" for n in names)
    print(header)
    print("-" * (8 + 10 * len(names)))

    matrix = {}
    for n1 in names:
        row = f"{n1:>8s}"
        for n2 in names:
            if n1 == n2:
                row += f"{'1.000':>10s}"
                matrix[(n1, n2)] = 1.0
            else:
                r, p = pearsonr(errors[n1], errors[n2])
                matrix[(n1, n2)] = r
                marker = ""
                if r < 0.30:
                    marker = "✓"    # complementary
                elif r > 0.60:
                    marker = "✗"    # redundant
                row += f"{r:>8.3f}{marker:>2s}"
        print(row)

    # Summary: how complementary is global vs each other branch?
    print("\nGlobal branch complementarity summary:")
    for n in ["main", "kin", "local"]:
        r = matrix[("global", n)]
        if r < 0.30:
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
    print("(h_global_flat zeroed before visual fusion)")
    print("=" * 60)

    y_val_abl,  z_val_abl  = collect(model, val_loader,  device, amp, ablate_global=True)
    y_test_abl, z_test_abl = collect(model, test_loader, device, amp, ablate_global=True)

    print("\nWith global branch:")
    print(f"  main  val={auc(y_val,  z_val['main']):.4f}  test={auc(y_test,  z_test['main']):.4f}")

    print("\nWithout global branch (ablated):")
    print(f"  main  val={auc(y_val_abl,  z_val_abl['main']):.4f}  test={auc(y_test_abl,  z_test_abl['main']):.4f}")

    delta_val  = auc(y_val,  z_val['main'])  - auc(y_val_abl,  z_val_abl['main'])
    delta_test = auc(y_test, z_test['main']) - auc(y_test_abl, z_test_abl['main'])

    print(f"\nDelta (with - without):")
    print(f"  val:  {delta_val:+.4f}")
    print(f"  test: {delta_test:+.4f}")

    if delta_test > 0.005:
        print("\n  ✓ Global branch HELPS — removing it hurts test AUC by "
              f"{delta_test:.4f}")
    elif delta_test < -0.005:
        print("\n  ✗ Global branch HURTS — removing it improves test AUC by "
              f"{abs(delta_test):.4f}")
    else:
        print("\n  ~ Global branch is NEUTRAL on test AUC "
              f"(delta={delta_test:.4f})")

    return z_val_abl, z_test_abl


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pie_root",    default="/data/PIE_PREP_OUT")
    ap.add_argument("--ckpt",        required=True)
    ap.add_argument("--batch_size",  type=int,  default=10)
    ap.add_argument("--num_workers", type=int,  default=4)
    ap.add_argument("--amp",         action="store_true")
    ap.add_argument("--no_global",   action="store_true",
                    help="Run inference with global branch zeroed (ablation only mode)")
    ap.add_argument("--skip_grid",   action="store_true",
                    help="Skip the grid search (faster, just prints AUCs + correlation)")
    ap.add_argument("--grid_step",   type=float, default=0.05,
                    help="Grid search step size (smaller = finer but slower)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = PIPNetAlphaV3Final(dropout_p=0.5).to(device)
    ckpt  = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint: {args.ckpt}")
    print(f"  Best epoch: {ckpt.get('epoch', '?')}  |  "
          f"Val AUC at save: {ckpt.get('val_metrics', {}).get('auc', '?')}")

    # Loaders
    val_loader  = make_loader(args.pie_root, "val",  args.batch_size, args.num_workers)
    test_loader = make_loader(args.pie_root, "test", args.batch_size, args.num_workers)

    # Normal inference
    ablate = args.no_global
    print(f"\nRunning inference (ablate_global={ablate})...")
    y_val,  z_val  = collect(model, val_loader,  device, args.amp, ablate_global=ablate)
    y_test, z_test = collect(model, test_loader, device, args.amp, ablate_global=ablate)

    # ── Individual AUCs ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("Individual branch AUCs")
    print("=" * 60)
    for name in ["main", "kin", "local", "global"]:
        v = auc(y_val,  z_val[name])
        t = auc(y_test, z_test[name])
        print(f"  {name:>6s} | val={v:.4f} | test={t:.4f}")

    # ── Grid search ──────────────────────────────────────────
    if not args.skip_grid:
        print(f"\nRunning grid search (step={args.grid_step})...")
        best_val_auc, weights = grid_search(y_val, z_val, step=args.grid_step)
        test_fused = apply_weights(z_test, weights)
        test_auc   = auc(y_test, test_fused)

        print("\n" + "=" * 60)
        print("Best val-tuned logit fusion")
        print("=" * 60)
        print(f"  weights  = {weights}")
        print(f"  val AUC  = {best_val_auc:.4f}")
        print(f"  test AUC = {test_auc:.4f}")

        # How much does each branch contribute above main-only?
        main_only_test = auc(y_test, z_test["main"])
        print(f"\n  Main-only test AUC:  {main_only_test:.4f}")
        print(f"  Fusion test AUC:     {test_auc:.4f}")
        delta = test_auc - main_only_test
        if delta > 0.002:
            print(f"  ✓ Post-hoc fusion adds {delta:+.4f} over main alone")
        elif delta < -0.002:
            print(f"  ✗ Post-hoc fusion loses {delta:+.4f} vs main alone")
        else:
            print(f"  ~ Post-hoc fusion neutral ({delta:+.4f})")

    # ── Method 3: correlation ────────────────────────────────
    print("\nRunning error correlation analysis on test set...")
    matrix = error_correlation(y_test, z_test)

    # ── Method 1: ablation (only if not already ablated) ────
    if not args.no_global:
        run_ablation(
            model, val_loader, test_loader,
            device, args.amp,
            y_val, z_val, y_test, z_test
        )
    else:
        print("\n[--no_global was set: ablation was the primary run, skipping re-run]")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()