import os
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # safe για server/docker χωρίς GUI
import matplotlib.pyplot as plt


# ============================================================
# Output directory
# ============================================================

OUT_DIR = Path("figures_horizon_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Common x-axis
# ============================================================

HORIZON_ORDER = ["ETC0_5", "ETC1", "ETC2", "ETC3", "ETC4"]
HORIZON_TO_SEC = {
    "ETC0_5": 0.5,
    "ETC1": 1.0,
    "ETC2": 2.0,
    "ETC3": 3.0,
    "ETC4": 4.0,
}
HORIZON_LABELS = ["0.5s", "1s", "2s", "3s", "4s"]


# ============================================================
# 1) PER-BRANCH AUX RESULTS
# These are from:
# fused  = native learned fusion output
# kin    = aux_kin
# local  = aux_local
# global = aux_global
# ============================================================

branch_rows = [
    # horizon, branch, n, auc, static_acc, static_f1, static_prec, static_rec, tuned_thr, tuned_f1, tuned_prec, tuned_rec
    ("ETC0_5", "Fused",  684, 0.8826, 0.8567, 0.7366, 0.7135, 0.7611, 0.863, 0.7596, 0.8153, 0.7111),
    ("ETC0_5", "Kin",    684, 0.8796, 0.8450, 0.6513, 0.7984, 0.5500, 0.608, 0.7448, 0.7010, 0.7944),
    ("ETC0_5", "Local",  684, 0.7672, 0.7617, 0.5356, 0.5497, 0.5222, 0.684, 0.5514, 0.5023, 0.6111),
    ("ETC0_5", "Global", 684, 0.7411, 0.7471, 0.4927, 0.5217, 0.4667, 0.378, 0.5326, 0.4006, 0.7944),

    ("ETC1", "Fused",  664, 0.8984, 0.8584, 0.7360, 0.7198, 0.7529, 0.897, 0.7582, 0.8788, 0.6667),
    ("ETC1", "Kin",    664, 0.8767, 0.8343, 0.6333, 0.7540, 0.5460, 0.591, 0.7306, 0.6651, 0.8103),
    ("ETC1", "Local",  664, 0.8232, 0.7967, 0.6087, 0.6140, 0.6034, 0.593, 0.6279, 0.5273, 0.7759),
    ("ETC1", "Global", 664, 0.7650, 0.7726, 0.5325, 0.5772, 0.4943, 0.641, 0.5538, 0.5202, 0.5920),

    ("ETC2", "Fused",  603, 0.9082, 0.8673, 0.7576, 0.7310, 0.7862, 0.796, 0.7702, 0.7607, 0.7799),
    ("ETC2", "Kin",    603, 0.8553, 0.8076, 0.5672, 0.6972, 0.4780, 0.536, 0.6967, 0.5792, 0.8742),
    ("ETC2", "Local",  603, 0.8028, 0.7678, 0.5513, 0.5621, 0.5409, 0.458, 0.6071, 0.4850, 0.8113),
    ("ETC2", "Global", 603, 0.8188, 0.8209, 0.6115, 0.7143, 0.5346, 0.705, 0.6507, 0.7143, 0.5975),

    ("ETC3", "Fused",  539, 0.8980, 0.8553, 0.7329, 0.6993, 0.7698, 0.623, 0.7429, 0.6648, 0.8417),
    ("ETC3", "Kin",    539, 0.8317, 0.8033, 0.5620, 0.6602, 0.4892, 0.596, 0.6513, 0.5433, 0.8129),
    ("ETC3", "Local",  539, 0.8209, 0.7718, 0.5393, 0.5625, 0.5180, 0.529, 0.6431, 0.5450, 0.7842),
    ("ETC3", "Global", 539, 0.8004, 0.8145, 0.6032, 0.6726, 0.5468, 0.679, 0.6187, 0.6187, 0.6187),

    ("ETC4", "Fused",  487, 0.8713, 0.8234, 0.6791, 0.6319, 0.7339, 0.684, 0.7071, 0.6346, 0.7984),
    ("ETC4", "Kin",    487, 0.7975, 0.7680, 0.4744, 0.5604, 0.4113, 0.576, 0.6145, 0.4904, 0.8226),
    ("ETC4", "Local",  487, 0.7943, 0.7700, 0.5214, 0.5545, 0.4919, 0.374, 0.5869, 0.4537, 0.8306),
    ("ETC4", "Global", 487, 0.7908, 0.8419, 0.6516, 0.7423, 0.5806, 0.823, 0.6574, 0.7717, 0.5726),
]

branch_df = pd.DataFrame(
    branch_rows,
    columns=[
        "horizon", "branch", "n", "auc",
        "static_acc", "static_f1", "static_precision", "static_recall",
        "tuned_thr", "tuned_f1", "tuned_precision", "tuned_recall",
    ],
)
branch_df["tte_sec"] = branch_df["horizon"].map(HORIZON_TO_SEC)


# ============================================================
# 2) BRANCH-ZEROING ABLATION RESULTS
# These are from eval_ablation_horizons.py
# Tuned metrics are diagnostic/oracle only.
# ============================================================

ablation_rows = [
    # horizon, ablation, n, auc, static_f1, static_acc, tuned_thr, tuned_f1, tuned_acc
    ("ETC0_5", "Full",       684, 0.8826, 0.7366, 0.8567, 0.863, 0.7596, 0.8816),
    ("ETC0_5", "NoGlobal",   684, 0.8624, 0.6512, 0.7807, 0.883, 0.7123, 0.8523),
    ("ETC0_5", "NoLocal",    684, 0.8034, 0.5034, 0.7865, 0.159, 0.6128, 0.7266),
    ("ETC0_5", "NoKin",      684, 0.8156, 0.2441, 0.7646, 0.589, 0.6232, 0.8056),
    ("ETC0_5", "KinOnly",    684, 0.8776, 0.7475, 0.8509, 0.782, 0.7525, 0.8567),
    ("ETC0_5", "VisualOnly", 684, 0.8156, 0.2441, 0.7646, 0.589, 0.6232, 0.8056),

    ("ETC1", "Full",       664, 0.8984, 0.7360, 0.8584, 0.897, 0.7582, 0.8886),
    ("ETC1", "NoGlobal",   664, 0.8806, 0.6875, 0.8042, 0.872, 0.7198, 0.8464),
    ("ETC1", "NoLocal",    664, 0.8222, 0.5474, 0.8057, 0.181, 0.6105, 0.7425),
    ("ETC1", "NoKin",      664, 0.8617, 0.3019, 0.7771, 0.556, 0.7052, 0.8464),
    ("ETC1", "KinOnly",    664, 0.8700, 0.7154, 0.8298, 0.789, 0.7287, 0.8419),
    ("ETC1", "VisualOnly", 664, 0.8617, 0.3019, 0.7771, 0.556, 0.7052, 0.8464),

    ("ETC2", "Full",       603, 0.9082, 0.7576, 0.8673, 0.796, 0.7702, 0.8773),
    ("ETC2", "NoGlobal",   603, 0.8595, 0.6802, 0.7910, 0.753, 0.6818, 0.7910),
    ("ETC2", "NoLocal",    603, 0.8751, 0.6275, 0.8425, 0.378, 0.6983, 0.8524),
    ("ETC2", "NoKin",      603, 0.8748, 0.3216, 0.7761, 0.555, 0.6968, 0.8441),
    ("ETC2", "KinOnly",    603, 0.8422, 0.6835, 0.7927, 0.792, 0.6997, 0.8093),
    ("ETC2", "VisualOnly", 603, 0.8748, 0.3216, 0.7761, 0.555, 0.6968, 0.8441),

    ("ETC3", "Full",       539, 0.8980, 0.7329, 0.8553, 0.623, 0.7429, 0.8497),
    ("ETC3", "NoGlobal",   539, 0.8336, 0.6556, 0.7699, 0.647, 0.6649, 0.7625),
    ("ETC3", "NoLocal",    539, 0.8540, 0.6261, 0.8404, 0.303, 0.6761, 0.8293),
    ("ETC3", "NoKin",      539, 0.8788, 0.3596, 0.7885, 0.584, 0.6996, 0.8534),
    ("ETC3", "KinOnly",    539, 0.8065, 0.6356, 0.7532, 0.772, 0.6480, 0.7662),
    ("ETC3", "VisualOnly", 539, 0.8788, 0.3596, 0.7885, 0.584, 0.6996, 0.8534),

    ("ETC4", "Full",       487, 0.8713, 0.6791, 0.8234, 0.684, 0.7071, 0.8316),
    ("ETC4", "NoGlobal",   487, 0.7998, 0.6000, 0.7372, 0.563, 0.6243, 0.7331),
    ("ETC4", "NoLocal",    487, 0.8434, 0.6538, 0.8522, 0.727, 0.6698, 0.8542),
    ("ETC4", "NoKin",      487, 0.8575, 0.2968, 0.7762, 0.439, 0.6859, 0.8214),
    ("ETC4", "KinOnly",    487, 0.7718, 0.5988, 0.7166, 0.792, 0.6061, 0.7331),
    ("ETC4", "VisualOnly", 487, 0.8575, 0.2968, 0.7762, 0.439, 0.6859, 0.8214),
]

ablation_df = pd.DataFrame(
    ablation_rows,
    columns=[
        "horizon", "ablation", "n", "auc",
        "static_f1", "static_acc",
        "tuned_thr", "tuned_f1", "tuned_acc",
    ],
)
ablation_df["tte_sec"] = ablation_df["horizon"].map(HORIZON_TO_SEC)


# ============================================================
# Style
# ============================================================

branch_style = {
    "Fused":  dict(color="red",    marker="o", label="Native fusion"),
    "Kin":    dict(color="blue",   marker="s", label="Kinematic aux"),
    "Local":  dict(color="orange", marker="^", label="Local aux"),
    "Global": dict(color="green",  marker="D", label="Global aux"),
}

ablation_style = {
    "Full":       dict(color="red",     marker="o", label="Full"),
    "NoGlobal":   dict(color="green",   marker="s", label="No global"),
    "NoLocal":    dict(color="orange",  marker="^", label="No local"),
    "NoKin":      dict(color="blue",    marker="D", label="No kin"),
    "KinOnly":    dict(color="purple",  marker="P", label="Kin only"),
    "VisualOnly": dict(color="brown",   marker="X", label="Visual only"),
}


def setup_axis(ax, ylabel, title, y_min=None, y_max=None, y_step=None):
    ax.grid(True, which="both", linestyle="-", linewidth=0.5, color="gray", alpha=0.55)
    ax.set_xticks([0.5, 1, 2, 3, 4])
    ax.set_xticklabels(HORIZON_LABELS, fontsize=11)
    ax.set_xlabel("Estimated Time to Cross (ETC)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)

    if y_min is not None and y_max is not None:
        ax.set_ylim(y_min, y_max)
        if y_step is not None:
            ax.set_yticks(np.arange(y_min, y_max + 1e-9, y_step))


def savefig(fig, name):
    path_png = OUT_DIR / f"{name}.png"
    path_pdf = OUT_DIR / f"{name}.pdf"
    fig.tight_layout()
    fig.savefig(path_png, dpi=300, bbox_inches="tight")
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {path_png}")
    print(f"[OK] Saved {path_pdf}")


# ============================================================
# Figure 1: Aux branch AUC
# Main figure for branch-level discrimination
# ============================================================

fig, ax = plt.subplots(figsize=(9, 6))

for branch, st in branch_style.items():
    d = branch_df[branch_df["branch"] == branch].sort_values("tte_sec")
    ax.plot(
        d["tte_sec"], d["auc"],
        color=st["color"], marker=st["marker"], linestyle="--",
        linewidth=2.5, markersize=8, label=st["label"],
    )

setup_axis(
    ax,
    ylabel="AUC",
    title="Auxiliary Branch AUC Across Prediction Horizons",
    y_min=0.70,
    y_max=0.93,
    y_step=0.025,
)
ax.legend(loc="best", fontsize=10, framealpha=0.9)
savefig(fig, "01_aux_branch_auc")


# ============================================================
# Figure 2: Aux branch static F1
# Fixed threshold behavior, useful but secondary to AUC
# ============================================================

fig, ax = plt.subplots(figsize=(9, 6))

for branch, st in branch_style.items():
    d = branch_df[branch_df["branch"] == branch].sort_values("tte_sec")
    ax.plot(
        d["tte_sec"], d["static_f1"],
        color=st["color"], marker=st["marker"], linestyle="--",
        linewidth=2.5, markersize=8, label=st["label"],
    )

setup_axis(
    ax,
    ylabel="Static F1",
    title="Auxiliary Branch Static F1 Across Prediction Horizons",
    y_min=0.45,
    y_max=0.80,
    y_step=0.05,
)
ax.legend(loc="best", fontsize=10, framealpha=0.9)
savefig(fig, "02_aux_branch_static_f1")


# ============================================================
# Figure 3: Branch-zeroing AUC, all ablations
# Shows actual fused-output degradation when branches are removed
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6.5))

for ablation, st in ablation_style.items():
    d = ablation_df[ablation_df["ablation"] == ablation].sort_values("tte_sec")
    ax.plot(
        d["tte_sec"], d["auc"],
        color=st["color"], marker=st["marker"], linestyle="--",
        linewidth=2.3, markersize=8, label=st["label"],
    )

setup_axis(
    ax,
    ylabel="AUC",
    title="Branch-Zeroing Ablation AUC Across Prediction Horizons",
    y_min=0.75,
    y_max=0.92,
    y_step=0.025,
)
ax.legend(loc="best", fontsize=10, framealpha=0.9, ncol=2)
savefig(fig, "03_ablation_auc_all")


# ============================================================
# Figure 4: ΔAUC loss relative to Full
# This is the cleanest ablation contribution plot.
# Positive value means removing that branch hurts performance.
# ============================================================

full_auc = (
    ablation_df[ablation_df["ablation"] == "Full"]
    .set_index("horizon")["auc"]
)

delta_rows = []
for remove_name in ["NoGlobal", "NoLocal", "NoKin"]:
    d = ablation_df[ablation_df["ablation"] == remove_name].copy()
    for _, row in d.iterrows():
        horizon = row["horizon"]
        delta_rows.append({
            "horizon": horizon,
            "tte_sec": HORIZON_TO_SEC[horizon],
            "removed_branch": remove_name.replace("No", "Remove "),
            "delta_auc": full_auc[horizon] - row["auc"],
        })

delta_auc_df = pd.DataFrame(delta_rows)

delta_style = {
    "Remove Global": dict(color="green",  marker="s", label="Remove global"),
    "Remove Local":  dict(color="orange", marker="^", label="Remove local"),
    "Remove Kin":    dict(color="blue",   marker="D", label="Remove kin"),
}

fig, ax = plt.subplots(figsize=(9, 6))

for name, st in delta_style.items():
    d = delta_auc_df[delta_auc_df["removed_branch"] == name].sort_values("tte_sec")
    ax.plot(
        d["tte_sec"], d["delta_auc"],
        color=st["color"], marker=st["marker"], linestyle="-",
        linewidth=2.8, markersize=8, label=st["label"],
    )

ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)

setup_axis(
    ax,
    ylabel="AUC drop vs Full",
    title="AUC Drop Caused by Removing Each Branch",
    y_min=0.00,
    y_max=0.09,
    y_step=0.01,
)
ax.legend(loc="best", fontsize=10, framealpha=0.9)
savefig(fig, "04_delta_auc_removed_branches")


# ============================================================
# Figure 5: ΔStatic F1 loss relative to Full
# Static threshold = 0.7585. This is a deployment-style stress test.
# ============================================================

full_f1 = (
    ablation_df[ablation_df["ablation"] == "Full"]
    .set_index("horizon")["static_f1"]
)

delta_f1_rows = []
for remove_name in ["NoGlobal", "NoLocal", "NoKin"]:
    d = ablation_df[ablation_df["ablation"] == remove_name].copy()
    for _, row in d.iterrows():
        horizon = row["horizon"]
        delta_f1_rows.append({
            "horizon": horizon,
            "tte_sec": HORIZON_TO_SEC[horizon],
            "removed_branch": remove_name.replace("No", "Remove "),
            "delta_static_f1": full_f1[horizon] - row["static_f1"],
        })

delta_f1_df = pd.DataFrame(delta_f1_rows)

fig, ax = plt.subplots(figsize=(9, 6))

for name, st in delta_style.items():
    d = delta_f1_df[delta_f1_df["removed_branch"] == name].sort_values("tte_sec")
    ax.plot(
        d["tte_sec"], d["delta_static_f1"],
        color=st["color"], marker=st["marker"], linestyle="-",
        linewidth=2.8, markersize=8, label=st["label"],
    )

ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)

setup_axis(
    ax,
    ylabel="Static F1 drop vs Full",
    title="Static F1 Drop Caused by Removing Each Branch",
    y_min=0.00,
    y_max=0.50,
    y_step=0.05,
)
ax.legend(loc="best", fontsize=10, framealpha=0.9)
savefig(fig, "05_delta_static_f1_removed_branches")


# ============================================================
# Figure 6: Static F1 for all ablations
# Good secondary plot, but can be visually crowded.
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6.5))

for ablation, st in ablation_style.items():
    d = ablation_df[ablation_df["ablation"] == ablation].sort_values("tte_sec")
    ax.plot(
        d["tte_sec"], d["static_f1"],
        color=st["color"], marker=st["marker"], linestyle="--",
        linewidth=2.3, markersize=8, label=st["label"],
    )

setup_axis(
    ax,
    ylabel="Static F1",
    title="Branch-Zeroing Ablation Static F1 Across Prediction Horizons",
    y_min=0.20,
    y_max=0.80,
    y_step=0.05,
)
ax.legend(loc="best", fontsize=10, framealpha=0.9, ncol=2)
savefig(fig, "06_ablation_static_f1_all")


# ============================================================
# Figure 7: Tuned F1 diagnostic only
# WARNING: tuned threshold is selected on each test horizon.
# Do not use this as reported performance.
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6.5))

for ablation, st in ablation_style.items():
    d = ablation_df[ablation_df["ablation"] == ablation].sort_values("tte_sec")
    ax.plot(
        d["tte_sec"], d["tuned_f1"],
        color=st["color"], marker=st["marker"], linestyle="--",
        linewidth=2.3, markersize=8, label=st["label"],
    )

setup_axis(
    ax,
    ylabel="Tuned F1",
    title="Diagnostic Tuned F1 Across Horizons (Test Oracle)",
    y_min=0.55,
    y_max=0.80,
    y_step=0.025,
)
ax.text(
    0.02, 0.02,
    "Diagnostic only: threshold tuned on each test horizon.",
    transform=ax.transAxes,
    fontsize=9,
    bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray"),
)
ax.legend(loc="best", fontsize=10, framealpha=0.9, ncol=2)
savefig(fig, "07_diagnostic_tuned_f1_oracle")


# ============================================================
# Figure 8: Compact paper-style 2-panel figure
# Panel A = aux AUC
# Panel B = delta AUC from ablation
# This is probably the best single figure for paper/report.
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))

# Panel A: Aux branch AUC
ax = axes[0]
for branch, st in branch_style.items():
    d = branch_df[branch_df["branch"] == branch].sort_values("tte_sec")
    ax.plot(
        d["tte_sec"], d["auc"],
        color=st["color"], marker=st["marker"], linestyle="--",
        linewidth=2.4, markersize=8, label=st["label"],
    )
setup_axis(
    ax,
    ylabel="AUC",
    title="A) Standalone Branch Discrimination",
    y_min=0.70,
    y_max=0.93,
    y_step=0.05,
)
ax.legend(loc="best", fontsize=9, framealpha=0.9)

# Panel B: ΔAUC
ax = axes[1]
for name, st in delta_style.items():
    d = delta_auc_df[delta_auc_df["removed_branch"] == name].sort_values("tte_sec")
    ax.plot(
        d["tte_sec"], d["delta_auc"],
        color=st["color"], marker=st["marker"], linestyle="-",
        linewidth=2.6, markersize=8, label=st["label"],
    )
ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)
setup_axis(
    ax,
    ylabel="AUC drop vs Full",
    title="B) Contribution to Learned Fusion",
    y_min=0.00,
    y_max=0.09,
    y_step=0.02,
)
ax.legend(loc="best", fontsize=9, framealpha=0.9)

savefig(fig, "08_paper_style_branch_analysis")


# ============================================================
# Save tables
# ============================================================

branch_df.to_csv(OUT_DIR / "aux_branch_results.csv", index=False)
ablation_df.to_csv(OUT_DIR / "ablation_results.csv", index=False)
delta_auc_df.to_csv(OUT_DIR / "delta_auc_removed_branches.csv", index=False)
delta_f1_df.to_csv(OUT_DIR / "delta_static_f1_removed_branches.csv", index=False)

print(f"[OK] Wrote {OUT_DIR / 'aux_branch_results.csv'}")
print(f"[OK] Wrote {OUT_DIR / 'ablation_results.csv'}")
print(f"[OK] Wrote {OUT_DIR / 'delta_auc_removed_branches.csv'}")
print(f"[OK] Wrote {OUT_DIR / 'delta_static_f1_removed_branches.csv'}")


# ============================================================
# Print concise numeric summary for paper reasoning
# ============================================================

print("\n" + "=" * 100)
print("KEY ABLATION SUMMARY")
print("=" * 100)

for horizon in HORIZON_ORDER:
    full = ablation_df[(ablation_df["horizon"] == horizon) & (ablation_df["ablation"] == "Full")].iloc[0]
    nog = ablation_df[(ablation_df["horizon"] == horizon) & (ablation_df["ablation"] == "NoGlobal")].iloc[0]
    nol = ablation_df[(ablation_df["horizon"] == horizon) & (ablation_df["ablation"] == "NoLocal")].iloc[0]
    nok = ablation_df[(ablation_df["horizon"] == horizon) & (ablation_df["ablation"] == "NoKin")].iloc[0]

    print(
        f"{horizon:<7} | "
        f"Full AUC={full['auc']:.4f} | "
        f"ΔNoGlobal={full['auc'] - nog['auc']:.4f} | "
        f"ΔNoLocal={full['auc'] - nol['auc']:.4f} | "
        f"ΔNoKin={full['auc'] - nok['auc']:.4f}"
    )

print("=" * 100)

print("\nInterpretation reminder:")
print("- Use AUC and ΔAUC as the main ablation evidence.")
print("- Static F1 uses the fixed threshold 0.7585 and is useful as deployment-style behavior.")
print("- Tuned F1 is test-oracle diagnostic only; do not report it as main performance.")
print("- The strongest paper figure is: 08_paper_style_branch_analysis.png")