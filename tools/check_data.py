import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# 1. Grab a random training file
files = glob.glob('/data/JAAD_PREP_OUT/train/*.npz')
if not files:
    print("No NPZ files found!")
    exit()
    
file_path = files[0] # Grab the first one
data = np.load(file_path, allow_pickle=True)

print(f"=== DEEP INTEGRITY CHECK: {os.path.basename(file_path)} ===")
print(f"Video/Track: {data['track_id'][0]} | Crossing Label: {data['label']}")

# 2. Mathematical Integrity Checks
tensors_to_check = ['bbox', 'pose', 'local_cnn', 'local_motion', 'sem_labels', 'cat_depth']

print("\n--- Tensor Statistics ---")
for name in tensors_to_check:
    arr = data[name]
    has_nan = np.isnan(arr).any()
    has_inf = np.isinf(arr).any()
    
    # We use np.min/np.max with nan_to_num just in case there are NaNs so it doesn't crash
    val_min = np.nanmin(arr)
    val_max = np.nanmax(arr)
    
    status = "❌ FAILED (NaN/Inf)" if has_nan or has_inf else "✅ PASS"
    print(f"[{name.ljust(12)}] {status} | Shape: {str(arr.shape).ljust(17)} | Min: {val_min:7.2f} | Max: {val_max:7.2f}")

# 3. Generate Visualizations (Frame 0 of the sequence)
print("\n=== GENERATING VISUALIZATION ===")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f"Neural Network Input Data: {os.path.basename(file_path)} (Frame 0)", fontsize=16)

# A. Semantic Mask (Mask2Former)
sem = data['sem_labels'][0]  # Shape: (384, 672)
ax1 = axes[0, 0]
c1 = ax1.imshow(sem, cmap='tab20')
ax1.set_title("Mask2Former Semantic Classes")
fig.colorbar(c1, ax=ax1)

# B. RAFT Optical Flow Magnitude
# Flow is (2, 224, 224) representing X and Y movement. We calculate the hypotenuse for overall speed.
motion = data['local_motion'][0] 
magnitude = np.sqrt(motion[0]**2 + motion[1]**2)
ax2 = axes[0, 1]
c2 = ax2.imshow(magnitude, cmap='inferno')
ax2.set_title("RAFT Flow Magnitude (Local Crop)")
fig.colorbar(c2, ax=ax2)

# C. ManyDepth: Pedestrian Channel
depth_ped = data['cat_depth'][0, 0] # Channel 0 is Pedestrians
ax3 = axes[1, 0]
c3 = ax3.imshow(depth_ped, cmap='magma')
ax3.set_title("ManyDepth: Pedestrian Channel")
fig.colorbar(c3, ax=ax3)

# D. ManyDepth: Vehicle Channel
depth_veh = data['cat_depth'][0, 1] # Channel 1 is Vehicles
ax4 = axes[1, 1]
c4 = ax4.imshow(depth_veh, cmap='viridis')
ax4.set_title("ManyDepth: Vehicle Channel")
fig.colorbar(c4, ax=ax4)

plt.tight_layout()
out_img = "data_viz_check.png"
plt.savefig(out_img, dpi=150)
print(f"[SUCCESS] Visualization saved to your current folder as '{out_img}'")