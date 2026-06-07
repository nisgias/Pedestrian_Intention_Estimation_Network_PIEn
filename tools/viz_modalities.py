#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
viz_modalities.py
=================
Παράγει εικόνες (Σχήματα 1-5) από ένα PIE_PREP_OUT .npz, για τη διπλωματική PIPN.
"""

import os
import sys
import argparse
import numpy as np
import cv2

# headless matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

# ---------------------------------------------------------------------------
# Project import path
# ---------------------------------------------------------------------------
sys.path.insert(0, "/workspace/project")
sys.path.insert(0, "/workspace/project/preprocess/pie")
sys.path.insert(0, "/data/pie_interface")

# ---------------------------------------------------------------------------
# matplotlib style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),            # κεφάλι
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),   # χέρια / ώμοι
    (5, 11), (6, 12), (11, 12),                # κορμός
    (11, 13), (13, 15), (12, 14), (14, 16),    # πόδια
]

# ===========================================================================
# Helpers
# ===========================================================================
def load_npz(path):
    d = np.load(path, allow_pickle=True)
    return d

def get_scalar(d, key, default=None):
    if key not in d.files:
        return default
    v = d[key]
    try:
        return v.item()
    except Exception:
        try:
            return v[0]
        except Exception:
            return v

def read_frame_bgr(pie_root, set_id, video, frame_idx, off):
    path = os.path.join(pie_root, set_id, f"{video}.mp4")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Δεν ανοίγει το video: {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx) + int(off))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Read failed: frame {frame_idx} (off={off}) στο {path}")
    return frame  # BGR

def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def colorize_labels(label_map, seed=42):
    rng = np.random.RandomState(seed)
    ids = np.unique(label_map)
    out = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    palette = {}
    for i in ids:
        if i == 0:
            palette[i] = np.array([0, 0, 0], dtype=np.uint8)
        else:
            palette[i] = rng.randint(40, 255, size=3).astype(np.uint8)
    for i in ids:
        out[label_map == i] = palette[i]
    return out

def flow_to_hsv(flow):
    fx, fy = flow[0], flow[1]
    mag, ang = cv2.cartToPolar(fx.astype(np.float32), fy.astype(np.float32))
    hsv = np.zeros((flow.shape[1], flow.shape[2], 3), dtype=np.uint8)
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def denorm_bbox(bbox_norm, W, H):
    x1, y1, x2, y2 = bbox_norm
    return int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)

# ===========================================================================
# ΣΧΗΜΑ 1 — Ego suppression
# ===========================================================================
def fig_ego_suppression(d, args, frame_bgr, out_h, out_w):
    from prep_scene import (
        init_cityscapes_meta_from_m2f, get_m2f, suppress_ego_car_dashboard,
    )
    init_cityscapes_meta_from_m2f(args.m2f_model, device=args.device)
    m2f = get_m2f(args.m2f_model, device=args.device)

    sem, inst, inst_lbl = m2f.run_single(frame_bgr, out_size=(out_h, out_w))

    sem_before = sem.copy()[None, ...]
    inst_c = inst.copy()[None, ...]
    inst_lbl_c = inst_lbl.copy()[None, ...]

    sem_after = sem.copy()[None, ...]
    ego_mask = suppress_ego_car_dashboard(
        sem_after, inst_c, inst_lbl_c,
        bottom_frac=args.bottom_frac,
    )
    sem_after = sem_after[0]
    sem_before = sem_before[0]

    rgb = bgr_to_rgb(cv2.resize(frame_bgr, (out_w, out_h)))
    seg_b_rgb = colorize_labels(sem_before)
    seg_a_rgb = colorize_labels(sem_after)

    overlay = rgb.copy()
    y0 = int(out_h * args.bottom_frac)
    overlay[y0:, :] = (0.5 * overlay[y0:, :] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    for ax in axes:
        ax.axis("off")
    axes[0].imshow(rgb);        axes[0].set_title("(a) RGB frame")
    axes[1].imshow(seg_b_rgb);  axes[1].set_title("(b) Segmentation (πριν)")
    axes[2].imshow(seg_a_rgb);  axes[2].set_title("(c) Segmentation (μετά ego-suppression)")
    axes[3].imshow(overlay);    axes[3].set_title(f"(d) Περιοχή καπό (y ≥ {args.bottom_frac:.2f}·H)")
    plt.tight_layout()
    out = os.path.join(args.out_dir, "fig1_ego_suppression.png")
    plt.savefig(out); plt.close()
    print(f"[OK] {out}")

# ===========================================================================
# ΣΧΗΜΑ 2 — Categorical depth (raw -> per-instance)
# ===========================================================================
def fig_categorical_depth(d, args, frames_bgr_seq, out_h, out_w):
    from prep_scene import estimate_depth

    # 1. Υπολογισμός Raw Depth (για να έχουμε το global heatmap)
    frame_idxs = [int(x) for x in list(d["frame_idx"])]
    depth_seq = estimate_depth(
        frames=frames_bgr_seq,
        frame_indices=frame_idxs,
        out_size=(out_h, out_w),
        weights_path=args.depth_weights,
    )

    # 2. Χρήση του ΗΔΗ ΥΠΟΛΟΓΙΣΜΕΝΟΥ categorical depth από το .npz
    # (Δεν το ξαναϋπολογίζουμε διότι τα sem_labels στο npz είναι σε PIP space, 
    # ενώ η compute_cat_depth περιμένει Cityscapes space)
    cat = d["cat_depth"].astype(np.float32)

    fi = args.frame
    rgb = bgr_to_rgb(cv2.resize(frames_bgr_seq[fi], (out_w, out_h)))
    raw = depth_seq[fi]
    ped = cat[fi, 0]
    car = cat[fi, 1]

    # Δυναμικό vmax για καλύτερη αντίθεση στην εικόνα της διπλωματικής
    vmax = max(float(ped.max()), float(car.max()), 1e-6)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    for ax in axes:
        ax.axis("off")
        
    axes[0].imshow(rgb);                  
    axes[0].set_title("(a) RGB frame")
    
    # Το raw depth παραμένει [0..1] για να δείχνει την απόλυτη κλίμακα
    im1 = axes[1].imshow(raw, cmap="magma", vmin=0, vmax=1)
    axes[1].set_title("(b) Raw ManyDepth (global)")
    
    # Τα categorical channels χρησιμοποιούν δυναμικό contrast (vmax)
    im2 = axes[2].imshow(ped, cmap="magma", vmin=0, vmax=vmax)
    axes[2].set_title("(c) Pedestrian depth (stored)")
    
    im3 = axes[3].imshow(car, cmap="magma", vmin=0, vmax=vmax)
    axes[3].set_title("(d) Vehicle depth (stored)")
    
    fig.colorbar(im3, ax=axes, fraction=0.025, pad=0.01, label="relative proximity")
    
    out = os.path.join(args.out_dir, "fig2_categorical_depth.png")
    plt.savefig(out); plt.close()
    print(f"[OK] {out}")

# ===========================================================================
# ΣΧΗΜΑ 3 — Optical flow
# ===========================================================================
def fig_optical_flow(d, args, frame_bgr, out_h, out_w):
    flow_seq = d["local_motion"]
    fi = args.frame
    flow = flow_seq[fi]
    flow_rgb = flow_to_hsv(flow)

    H = frame_bgr.shape[0]; W = frame_bgr.shape[1]
    bb = d["bbox"][fi]
    x1, y1, x2, y2 = denorm_bbox(bb, W, H)
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(W, x2); y2 = min(H, y2)
    crop = frame_bgr[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else frame_bgr
    crop_rgb = bgr_to_rgb(cv2.resize(crop, (224, 224)))

    fig, axes = plt.subplots(1, 2, figsize=(8, 4.2))
    for ax in axes:
        ax.axis("off")
    axes[0].imshow(crop_rgb);  axes[0].set_title("(a) Pedestrian crop")
    axes[1].imshow(flow_rgb);  axes[1].set_title("(b) RAFT optical flow (HSV)")
    plt.tight_layout()
    out = os.path.join(args.out_dir, "fig3_optical_flow.png")
    plt.savefig(out); plt.close()
    print(f"[OK] {out}")

# ===========================================================================
# ΣΧΗΜΑ 4 — Local context
# ===========================================================================
def fig_local_context(d, args, frame_bgr):
    H, W = frame_bgr.shape[:2]
    fi = args.frame
    bb = d["bbox"][fi]
    x1, y1, x2, y2 = denorm_bbox(bb, W, H)

    full = bgr_to_rgb(frame_bgr).copy()
    cv2.rectangle(full, (x1, y1), (x2, y2), (255, 215, 0), 3)

    xx1 = max(0, x1); yy1 = max(0, y1); xx2 = min(W, x2); yy2 = min(H, y2)
    crop = frame_bgr[yy1:yy2, xx1:xx2] if (xx2 > xx1 and yy2 > yy1) else frame_bgr
    crop_rgb = bgr_to_rgb(cv2.resize(crop, (224, 224)))

    fig = plt.figure(figsize=(11, 4.6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.2, 1])
    ax0 = fig.add_subplot(gs[0]); ax0.axis("off")
    ax0.imshow(full); ax0.set_title("(a) Full frame + bounding box")
    ax1 = fig.add_subplot(gs[1]); ax1.axis("off")
    ax1.imshow(crop_rgb); ax1.set_title("(b) Local context crop (224×224)")
    plt.tight_layout()
    out = os.path.join(args.out_dir, "fig4_local_context.png")
    plt.savefig(out); plt.close()
    print(f"[OK] {out}")

# ===========================================================================
# ΣΧΗΜΑ 5 — Bbox + pose
# ===========================================================================
def fig_bbox_pose(d, args, frame_bgr):
    H, W = frame_bgr.shape[:2]
    fi = args.frame
    bb = d["bbox"][fi]
    x1, y1, x2, y2 = denorm_bbox(bb, W, H)
    pose = d["pose"][fi].reshape(17, 2)

    img = bgr_to_rgb(frame_bgr).copy()
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 215, 0), 3)

    pts = []
    for (px, py) in pose:
        if px <= 0 and py <= 0:
            pts.append(None)
        else:
            pts.append((int(px * W), int(py * H)))

    for a, b in COCO_SKELETON:
        if pts[a] is not None and pts[b] is not None:
            cv2.line(img, pts[a], pts[b], (0, 200, 255), 2)
    for p in pts:
        if p is not None:
            cv2.circle(img, p, 4, (255, 0, 0), -1)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.axis("off")
    ax.imshow(img)
    ax.set_title("Bounding box + pose (17 COCO keypoints)")
    plt.tight_layout()
    out = os.path.join(args.out_dir, "fig5_bbox_pose.png")
    plt.savefig(out); plt.close()
    print(f"[OK] {out}")

# ===========================================================================
# Main
# ===========================================================================
def main():
    ap = argparse.ArgumentParser("Visualize PIPN modalities (figs 1-5)")
    ap.add_argument("--npz", required=True)
    ap.add_argument("--pie_root", default="/data/PIE")
    ap.add_argument("--m2f_model", default="/workspace/project/m2f_cache")
    ap.add_argument("--depth_weights", default="/workspace/models/manydepth/KITTI_HR")
    ap.add_argument("--out_dir", default="/workspace/project/thesis_figs")
    ap.add_argument("--frame", type=int, default=0, help="ποιο frame (0..T-1)")
    ap.add_argument("--bottom_frac", type=float, default=0.88, help="ego band (PIE=0.88)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--figs", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    d = load_npz(args.npz)

    set_id = str(get_scalar(d, "set_id"))
    video = str(get_scalar(d, "video_name"))
    off = int(get_scalar(d, "mp4_png_offset", 0))
    frame_idxs = list(d["frame_idx"])
    out_h, out_w = (int(d["_scene_size"][0]), int(d["_scene_size"][1])) if "_scene_size" in d.files else (384, 672)

    print(f"[INFO] seq={os.path.basename(args.npz)} set={set_id} video={video} "
          f"off={off} frame_idx={frame_idxs[args.frame]} out=({out_h},{out_w})")

    frame_bgr = read_frame_bgr(args.pie_root, set_id, video, frame_idxs[args.frame], off)
    frame_bgr_r = cv2.resize(frame_bgr, (out_w, out_h))

    if 1 in args.figs:
        fig_ego_suppression(d, args, frame_bgr_r, out_h, out_w)
    if 2 in args.figs:
        frames_seq = [read_frame_bgr(args.pie_root, set_id, video, fx, off) for fx in frame_idxs]
        fig_categorical_depth(d, args, frames_seq, out_h, out_w)
    if 3 in args.figs:
        fig_optical_flow(d, args, frame_bgr, out_h, out_w)
    if 4 in args.figs:
        fig_local_context(d, args, frame_bgr)
    if 5 in args.figs:
        fig_bbox_pose(d, args, frame_bgr)

if __name__ == "__main__":
    main()