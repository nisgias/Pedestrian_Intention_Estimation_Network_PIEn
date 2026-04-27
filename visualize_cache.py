import numpy as np
import matplotlib
matplotlib.use("Agg")   # 🔴 FIX for Docker / no GUI

import matplotlib.pyplot as plt
from pathlib import Path
import random

# ============================================================
# CONFIG
# ============================================================

NPZ_ROOT = "/workspace/PIE_PREP_OUT/train"
OUT_DIR = "/workspace/project/visualizations"   # 🔴 output folder
NUM_SAMPLES = 3
FRAME_ID = None  # None = random

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# ============================================================
# COLOR MAPS
# ============================================================

def get_semantic_colormap():
    return plt.get_cmap("tab20", 20)

def get_depth_colormap():
    return plt.get_cmap("inferno")


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_sample(npz_path):
    data = np.load(npz_path, allow_pickle=True)

    # ⚠️ handle empty placeholders (VERY IMPORTANT)
    sem = data["sem_labels"]
    depth = data["cat_depth"]

    if sem.shape[-1] == 1:  # 아직 scene not generated
        print(f"[SKIP] {npz_path.name} (no scene data yet)")
        return

    T = sem.shape[0]
    t = FRAME_ID if FRAME_ID is not None else random.randint(0, T - 1)

    sem_t = sem[t]
    ped_depth = depth[t, 0]
    veh_depth = depth[t, 1]

    # combine depth
    depth_combined = np.maximum(ped_depth, veh_depth)

    # normalize
    if depth_combined.max() > 0:
        depth_norm = depth_combined / depth_combined.max()
    else:
        depth_norm = depth_combined

    # ========================================================
    # PLOT
    # ========================================================

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    # --- Semantic ---
    axs[0].imshow(sem_t, cmap=get_semantic_colormap(), vmin=0, vmax=19)
    axs[0].set_title("Semantic Map")
    axs[0].axis("off")

    # --- Depth ---
    axs[1].imshow(depth_norm, cmap=get_depth_colormap())
    axs[1].set_title("Categorical Depth Map")
    axs[1].axis("off")

    plt.suptitle(f"{npz_path.name} | frame {t}")
    plt.tight_layout()

    # 🔴 SAVE instead of show
    out_path = Path(OUT_DIR) / f"{npz_path.stem}_frame{t}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"[SAVED] {out_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    npz_files = list(Path(NPZ_ROOT).glob("*.npz"))

    if len(npz_files) == 0:
        print("No NPZ files found!")
        return

    samples = random.sample(npz_files, min(NUM_SAMPLES, len(npz_files)))

    for f in samples:
        print(f"Visualizing: {f}")
        visualize_sample(f)


if __name__ == "__main__":
    main()