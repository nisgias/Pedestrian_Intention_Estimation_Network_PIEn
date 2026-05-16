"""
eval_roc_horizons.py

Run inference across all ETC test horizons and save raw (labels, probs) per
horizon as .npz files. Then optionally generate a ROC curve figure.

Two-step design:
    1) Inference pass per horizon  -> saves rocs/{horizon}.npz
    2) Plot ROC curves              -> writes fig_roc_horizons.pdf/.png

You can skip step 1 by passing --skip_inference (uses cached .npz files).
You can skip step 2 by passing --skip_plot.

The .npz files contain ONLY the fused-branch probabilities for clean ROC plots.
If you later need per-branch ROC curves, extend the saved keys.
"""

import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, roc_auc_score

from data.pie import PIESeqDataset
from models.pipnet_alpha_v4_final import PIPNetAlphaV4Final


# ============================================================================
# Data loader
# ============================================================================

def make_loader(root, split, dataset, batch_size, num_workers, seq_len):
    ds = PIESeqDataset(
        root, split=split, mode="eval", seq_len=seq_len,
        strict_len=True, return_meta=True,
        speed_norm="minmax",
        speed_stats_path=f"/workspace/project/data/{dataset}_speed_stats_splits.json",
        speed_scope="global",
        motion_norm="p99abs",
        motion_stats_path=f"/workspace/project/data/{dataset}_motion_stats_splits.json",
        motion_scope="global", motion_clip=1.0,
    )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )


# ============================================================================
# Inference: collect fused probabilities and labels
# ============================================================================

@torch.no_grad()
def collect_fused(model, loader, device, amp=False):
    model.eval()
    labels_list, probs_list = [], []

    for batch in loader:
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            out = model(batch, return_aux=False)
        if isinstance(out, dict):
            logit = out["logit"]
        else:
            logit = out

        p = torch.sigmoid(logit.squeeze(-1).float()).cpu().numpy()
        labels_list.append(batch["label"].cpu().numpy().astype(np.int32))
        probs_list.append(p)

    return np.concatenate(labels_list), np.concatenate(probs_list)


# ============================================================================
# Plotting
# ============================================================================

def plot_roc_curves(roc_data, out_pdf, out_png):
    """
    roc_data: list of dicts with keys 'horizon', 'labels', 'probs', 'n'
    Produces a square ROC plot with one curve per horizon.
    """
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # Sequential colormap: shorter horizon = darker, longer = lighter
    # This works because we want the eye to follow horizon progression
    cmap = plt.get_cmap("viridis")
    n = len(roc_data)
    colors = [cmap(0.15 + 0.70 * i / max(n - 1, 1)) for i in range(n)]

    fig, ax = plt.subplots(figsize=(4.6, 4.6))  # square

    # Diagonal reference line first (so curves draw over it)
    ax.plot([0, 1], [0, 1], color="#999999", linestyle="--",
            linewidth=0.8, zorder=1, label="Random")

    for i, d in enumerate(roc_data):
        fpr, tpr, _ = roc_curve(d["labels"], d["probs"])
        auc = roc_auc_score(d["labels"], d["probs"])
        ax.plot(
            fpr, tpr,
            color=colors[i], linewidth=1.6,
            label=f"ETC {d['horizon_label']}  (AUC = {auc:.3f})",
            zorder=3 + i,
        )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.005)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)
    ax.legend(loc="lower right", frameon=False, handlelength=2.0)

    # Tick marks at sensible places
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close()
    print(f"[OK] Saved {out_pdf}")
    print(f"[OK] Saved {out_png}")


# ============================================================================
# Main
# ============================================================================

# Map folder names to nice display labels
HORIZON_LABELS = {
    "ETC0_5": "0.5\u202fs",
    "ETC1":   "1.0\u202fs",
    "ETC2":   "2.0\u202fs",
    "ETC3":   "3.0\u202fs",
    "ETC4":   "4.0\u202fs",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base_root", type=str, default="/Datasets/PIE_PREP_OUT/ETCs")
    ap.add_argument("--horizons", type=str, default="ETC0_5,ETC1,ETC2,ETC3,ETC4")
    ap.add_argument("--dataset", type=str, default="pie", choices=["pie", "jaad"])
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seq_len", type=int, default=10)
    ap.add_argument("--amp", action="store_true")

    ap.add_argument("--out_dir", type=str, default="rocs",
                    help="Directory to cache (labels, probs) .npz per horizon.")
    ap.add_argument("--out_pdf", type=str, default="fig_roc_horizons.pdf")
    ap.add_argument("--out_png", type=str, default="fig_roc_horizons.png")

    ap.add_argument("--skip_inference", action="store_true",
                    help="Use already-cached .npz files in --out_dir.")
    ap.add_argument("--skip_plot", action="store_true",
                    help="Run inference and cache results, but do not plot.")

    # Model dropouts
    ap.add_argument("--dropout_p",        type=float, default=0.2)
    ap.add_argument("--local_dropout_p",  type=float, default=0.1)
    ap.add_argument("--global_dropout_p", type=float, default=0.4)
    args = ap.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and args.amp

    os.makedirs(args.out_dir, exist_ok=True)
    horizons = args.horizons.split(",")

    # ------------------------------------------- Step 1: inference per horizon
    if not args.skip_inference:
        print(f"\n[Step 1] Loading model from {args.ckpt}")
        model = PIPNetAlphaV4Final(
            dropout_p=args.dropout_p,
            local_dropout_p=args.local_dropout_p,
            global_dropout_p=args.global_dropout_p,
        ).to(device)
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)

        for hz in horizons:
            hz_path = os.path.join(args.base_root, hz)
            if not os.path.exists(hz_path):
                print(f"[WARN] {hz_path} not found - skipping")
                continue

            print(f"  Inferring {hz} ...")
            loader = make_loader(hz_path, "test", args.dataset,
                                 args.batch_size, args.num_workers, args.seq_len)
            labels, probs = collect_fused(model, loader, device, use_amp)
            auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else float("nan")
            print(f"    N={len(labels)}  pos={int(labels.sum())}  AUC={auc:.4f}")

            np.savez_compressed(
                os.path.join(args.out_dir, f"{hz}.npz"),
                labels=labels, probs=probs,
            )

    # ------------------------------------------- Step 2: plot
    if args.skip_plot:
        print("\n[Step 2] Skipped plotting (--skip_plot)")
        return

    print(f"\n[Step 2] Loading cached results and plotting ROC curves...")
    roc_data = []
    for hz in horizons:
        npz_path = os.path.join(args.out_dir, f"{hz}.npz")
        if not os.path.exists(npz_path):
            print(f"  [WARN] missing {npz_path} - skipping")
            continue
        d = np.load(npz_path)
        roc_data.append(dict(
            horizon=hz,
            horizon_label=HORIZON_LABELS.get(hz, hz),
            labels=d["labels"],
            probs=d["probs"],
            n=int(len(d["labels"])),
        ))
        auc = roc_auc_score(d["labels"], d["probs"])
        print(f"  {hz}: N={len(d['labels'])} AUC={auc:.4f}")

    if not roc_data:
        print("[ERROR] no horizons to plot")
        return

    plot_roc_curves(roc_data, args.out_pdf, args.out_png)
    print(f"\nCache directory: {args.out_dir}/")
    print("To regenerate just the plot without re-running inference:")
    print(f"  python {os.path.basename(__file__)} --ckpt {args.ckpt} --skip_inference")


if __name__ == "__main__":
    main()