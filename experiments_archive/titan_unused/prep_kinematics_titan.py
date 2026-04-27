#!/usr/bin/env python3
"""
Step 1: TITAN Kinematics → PIPNet NPZ Contract.

TITAN dataset structure (honda-titan-dataset):
  titan_root/
    clip_XXXXX/                  ← one folder per clip
      images_anonymized/
        000001.png  000002.png  ...
      annotations/
        000001.csv  000002.csv  ...   ← one CSV per FRAME

Each per-frame CSV has at minimum:
  obj_track_id, left, top, width, height,
  attributes.pedestrian_cross_road  (or attributes.crossing_road)
  attributes.pedestrian_motion      (or attributes.person_motion)

Key design decisions vs PIE/JAAD
─────────────────────────────────
1. data_fstride = 1  (stride across SAVED frames is always 1 — no downsampling).
   TITAN is 10 fps; stride=1 means every frame is used.
   This guarantees the crossing event is NEVER skipped between two
   consecutive window frames (the bug you had on PIE with stride=2).

2. Horizontal mirroring (--mirror flag, train-split only by default).
   When enabled, every accepted window is written TWICE:
     seq_XXXXXX.npz  → original
     seq_YYYYYY.npz  → mirrored (is_mirrored=1)
   Mirror operations applied consistently:
     • image:        cv2.flip(img, 1)
     • bbox pixel:   x1_m = (W-1) - x2_orig,  x2_m = (W-1) - x1_orig
     • bbox norm:    x1n_m = 1 - x2n_orig,     x2n_m = 1 - x1n_orig
     • pose (norm):  x_m = 1 - x_orig,  then swap COCO left/right pairs
   Mirrored images are written to:
     <clip_dir>/images_anonymized/mirrored/<stem>_mirror<ext>
   so that prep_local (VGG/RAFT) can find them as normal PNG paths.
   val/test splits are NEVER mirrored (no data leakage).

3. Hard negatives without intention scores.
   TITAN has no human intention rating (unlike PIE's 0–1 prob).
   Instead we use the per-frame motion annotation as a difficulty proxy:
     "hard"  mode → pseudo_event placed at the end of the longest
                    consecutive walking/running run AFTER the window.
                    Rationale: the model sees "pedestrian is actively
                    moving" right before the window ends — the hardest
                    possible negative (walking but not crossing).
     "last"  mode → pseudo_event = last annotated frame (simple baseline).
     "random"mode → random frame in [end_frame + tte_min_f, last_frame].

4. Splits are deterministic per-CLIP (not per-track, not per CSV).

Usage
─────
# No mirroring, no pose (fast debug):
python prep_kinematics_titan.py \\
    --titan_root /data/TITAN/titan_data/dataset \\
    --out_root   /data/TITAN_PREP_OUT \\
    --splits     train,val,test \\
    --seq_len    10 --overlap 0.5 \\
    --tte_min_sec 1.0 --tte_max_sec 2.0 \\
    --no_pose

# With mirroring on train only:
python prep_kinematics_titan.py \\
    --titan_root /data/TITAN/titan_data/dataset \\
    --out_root   /data/TITAN_PREP_OUT \\
    --mirror --mirror_splits train \\
    --seq_len 10 --no_pose
"""

import argparse
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ============================================================
# Constants
# ============================================================

RAW_FPS = 10.0  # TITAN default

CROSS_COLS = [
    "attributes.Atomic Actions",
    "attributes.Simple Context",
    "attributes.Complex Contextual",
    "attributes.Communicative",
]

MOTION_COLS = [
    "attributes.Atomic Actions",
    "attributes.Simple Context",
    "attributes.Motion Status",
]

CROSS_POSITIVE = {"crossing", "cross", "yes", "1", "true"}
CROSS_NEGATIVE = {
    "not crossing", "not_crossing", "not cross",
    "no", "0", "false", "standing", "waiting",
}

MOTION_ACTIVE = {"walking", "running", "moving", "jogging"}
MOTION_STATIC = {"standing", "waiting", "stopped", "stationary"}

# COCO 17-keypoint left↔right swap pairs (0-indexed)
COCO_LR_PAIRS = [
    (1, 2),    # left_eye ↔ right_eye
    (3, 4),    # left_ear ↔ right_ear
    (5, 6),    # left_shoulder ↔ right_shoulder
    (7, 8),    # left_elbow ↔ right_elbow
    (9, 10),   # left_wrist ↔ right_wrist
    (11, 12),  # left_hip ↔ right_hip
    (13, 14),  # left_knee ↔ right_knee
    (15, 16),  # left_ankle ↔ right_ankle
]


# ============================================================
# TITAN Directory Helpers
# ============================================================

def find_clips(titan_root: Path) -> List[Path]:
    """
    Actual TITAN layout:
      dataset/clip_90/
      dataset/clip_90.csv
      dataset/images_anonymized/clip_90/images/
    """
    clips = []
    for p in sorted(titan_root.iterdir()):
        if p.is_dir() and p.name.startswith("clip_"):
            csv_path = titan_root / f"{p.name}.csv"
            img_dir = titan_root / "images_anonymized" / p.name / "images"
            if csv_path.exists() and img_dir.exists():
                clips.append(p)
    return clips


def list_frame_csvs(clip_dir: Path) -> List[Path]:
    """Return per-frame annotation CSVs sorted by frame number."""
    ann_dir = clip_dir / "annotations"
    if not ann_dir.exists():
        return []
    return sorted(ann_dir.glob("*.csv"), key=lambda p: _frame_num_from_path(p))


def _frame_num_from_path(p: Path) -> int:
    nums = re.findall(r"\d+", p.stem)
    return int(nums[-1]) if nums else 0


def frame_image_path(clip_dir: Path, frame_num: int) -> Optional[Path]:
    """
    Actual image layout:
      dataset/images_anonymized/clip_90/images/000006.png
    """
    titan_root = clip_dir.parent
    clip_name = clip_dir.name
    img_dir = titan_root / "images_anonymized" / clip_name / "images"

    if not img_dir.exists():
        return None

    stems = [
        f"{frame_num:06d}",
        f"{frame_num:05d}",
        f"{frame_num:04d}",
        str(frame_num),
    ]

    for stem in stems:
        for ext in (".png", ".jpg", ".jpeg"):
            p = img_dir / f"{stem}{ext}"
            if p.exists():
                return p

    for stem in stems:
        for p in img_dir.glob(f"{stem}*"):
            if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
                return p

    return None

def frame_num_from_row(row: pd.Series) -> Optional[int]:
    """
    TITAN annotation CSV uses a `frames` column like 000006.png.
    """
    if "frames" not in row.index or pd.isna(row["frames"]):
        return None

    val = str(row["frames"])
    nums = re.findall(r"\d+", val)
    if not nums:
        return None

    return int(nums[-1])
# ============================================================
# Annotation Parsing
# ============================================================

def _col_value(row: pd.Series, candidates: List[str], default=None):
    for col in candidates:
        if col in row.index and pd.notna(row[col]):
            return str(row[col]).strip().lower()
    return default


def infer_cross_label_from_row(row: pd.Series) -> int:
    POSITIVE = {
        "crossing a street at pedestrian crossing",
        "jaywalking (illegally crossing not at pedestrian crossing)",
    }

    texts = []
    for col in [
        "attributes.Atomic Actions",
        "attributes.Simple Context",
        "attributes.Complex Contextual",
        "attributes.Communicative",
    ]:
        if col in row.index and pd.notna(row[col]):
            texts.append(str(row[col]).lower().replace("_", " ").strip())

    # Pass 1: any column positive → positive
    for text in texts:
        if text in POSITIVE:
            return 1

    # Pass 2: no positive found → negative
    return 0


def infer_motion_active(row: pd.Series) -> int:
    """
    Active motion proxy for hard negatives.
    """
    texts = []
    for col in [
        "attributes.Atomic Actions",
        "attributes.Simple Context",
        "attributes.Motion Status",
    ]:
        if col in row.index and pd.notna(row[col]):
            texts.append(str(row[col]).lower().replace("_", " "))

    text = " ".join(texts)

    if any(k in text for k in ["walking", "running", "moving", "jogging"]):
        return 1
    return 0


def is_pedestrian_row(row: pd.Series) -> bool:
    """Return True if this CSV row is a pedestrian annotation."""
    label_col = _col_value(row, ["label", "obj_class", "category"])
    if label_col is None:
        return True
    return label_col == "person"



def parse_bbox(row: pd.Series, img_w: int, img_h: int) -> Optional[np.ndarray]:
    """
    Parse TITAN bbox (left/top/width/height) → [x1,y1,x2,y2] clamped to image.
    """
    try:
        x1 = float(row["left"])
        y1 = float(row["top"])
        bw = float(row["width"])
        bh = float(row["height"])
    except (KeyError, ValueError, TypeError):
        return None
    x2, y2 = x1 + bw, y1 + bh
    x1 = float(np.clip(x1, 0, max(img_w - 1, 1)))
    x2 = float(np.clip(x2, 0, max(img_w - 1, 1)))
    y1 = float(np.clip(y1, 0, max(img_h - 1, 1)))
    y2 = float(np.clip(y2, 0, max(img_h - 1, 1)))
    if x2 <= x1:
        x2 = min(float(img_w - 1), x1 + 1.0)
    if y2 <= y1:
        y2 = min(float(img_h - 1), y1 + 1.0)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


# ============================================================
# Build Per-Pedestrian Tracks
# ============================================================

def build_tracks(clip_dir: Path) -> Dict[str, List[dict]]:
    """
    Read one annotation CSV per clip:
      dataset/clip_90.csv

    Group rows by obj_track_id.
    Also assign `sample_idx`, the 10Hz annotation index, because frame
    filenames jump by 6 and should not be used directly for timing.
    """
    titan_root = clip_dir.parent
    clip_csv = titan_root / f"{clip_dir.name}.csv"

    if not clip_csv.exists():
        return {}

    try:
        df = pd.read_csv(clip_csv)
    except Exception as e:
        print(f"[WARN] Could not read {clip_csv}: {e}")
        return {}

    if df.empty:
        return {}

    # Keep only pedestrian rows.
    df = df[df.apply(is_pedestrian_row, axis=1)].copy()
    if df.empty:
        return {}

    # Parse frame numbers.
    df["frame_num_parsed"] = df.apply(frame_num_from_row, axis=1)
    df = df[df["frame_num_parsed"].notna()].copy()
    if df.empty:
        return {}

    df["frame_num_parsed"] = df["frame_num_parsed"].astype(int)

    # Build frame_num -> 10Hz sample index.
    # Example: 000006, 000012, 000018 become sample_idx 0,1,2.
    unique_frames = sorted(df["frame_num_parsed"].unique().tolist())
    frame_to_sample_idx = {fn: i for i, fn in enumerate(unique_frames)}

    # Group by track first, then build per-track sample_idx
    tracks: Dict[str, List[dict]] = defaultdict(list)
    img_size_cache: Dict[int, Tuple[int, int]] = {}

    # First pass: collect all rows per track
    track_rows: Dict[str, List] = defaultdict(list)
    for _, row in df.iterrows():
        tid = str(row.get("obj_track_id", row.get("track_id", ""))).strip()
        if not tid or tid == "nan":
            continue
        try:
            tid = str(int(float(tid)))
        except Exception:
            pass
        track_rows[tid].append(row)

    # Second pass: build per-track sample_idx (0,1,2,... per track)
    for tid, rows in track_rows.items():
        # Sort rows by frame number
        parsed = []
        for row in rows:
            fn = frame_num_from_row(row)
            if fn is None:
                continue
            parsed.append((fn, row))
        parsed.sort(key=lambda x: x[0])

        # Per-track unique frames → sample_idx starts at 0 for each track
        track_unique_frames = sorted(set(fn for fn, _ in parsed))
        track_frame_to_sample = {fn: i for i, fn in enumerate(track_unique_frames)}

        for frame_num, row in parsed:
            sample_idx = track_frame_to_sample[frame_num]

            img_path = frame_image_path(clip_dir, frame_num)
            if img_path is None:
                continue

            if frame_num not in img_size_cache:
                probe = cv2.imread(str(img_path))
                if probe is None:
                    continue
                img_size_cache[frame_num] = probe.shape[:2]

            img_h, img_w = img_size_cache[frame_num]
            bbox = parse_bbox(row, img_w, img_h)
            if bbox is None:
                continue

            tracks[tid].append({
                "frame_num":         frame_num,
                "sample_idx":        sample_idx,
                "img_path":          str(img_path),
                "bbox_xyxy":         bbox,
                "cross_label_frame": infer_cross_label_from_row(row),
                "motion_active":     infer_motion_active(row),
                "img_w":             img_w,
                "img_h":             img_h,
                "clip_id":           clip_dir.name,
                "track_id":          tid,
            })

    # Sort and deduplicate per track by sample_idx.
    for tid in list(tracks.keys()):
        seen: Set[int] = set()
        deduped = []
        for rec in sorted(tracks[tid], key=lambda r: r["sample_idx"]):
            if rec["sample_idx"] not in seen:
                seen.add(rec["sample_idx"])
                deduped.append(rec)

        if deduped:
            tracks[tid] = deduped
        else:
            del tracks[tid]

    return tracks


def derive_track_label(
    track: List[dict],
    tte_min_f: int = 10,
) -> Tuple[int, int, int]:
    """
    Returns track_label, first_cross_frame_num, first_cross_sample_idx.
    
    If crossing starts at sample_idx=0 there is no pre-event window
    possible, so we require at least tte_min_f non-crossing frames
    before the first crossing frame.
    """
    for i, rec in enumerate(track):
        if rec["cross_label_frame"] == 1:
            # Need at least tte_min_f frames before this point
            if rec["sample_idx"] < tte_min_f:
                # crossing starts too early — no valid observation window
                return 0, -1, -1
            return 1, rec["frame_num"], rec["sample_idx"]
    return 0, -1, -1


# ============================================================
# Hard-Negative Pseudo Event Selection
# ============================================================

def choose_pseudo_event_frame_titan(
    track: List[dict],
    end_idx_in_track: int,
    tte_min_f: int,
    mode: str = "hard",
) -> Tuple[int, str]:
    """
    Choose a pseudo-event frame for negative tracks.

    TITAN has NO human intention score unlike PIE.
    We use motion_active labels as a difficulty proxy.

    "hard"   → end of the longest consecutive walking/running run after
               the window. This creates the hardest negative: a pedestrian
               who was actively moving but ultimately did not cross.
    "last"   → last frame of the track.
    "random" → uniformly random frame in the feasible range.
    """
    last_frame_num   = track[-1]["frame_num"]
    end_frame_num    = track[end_idx_in_track]["frame_num"]
    feasible_min_num = end_frame_num + tte_min_f

    if mode == "last":
        return last_frame_num, "last"

    post = [r for r in track[end_idx_in_track + 1:]
            if r["frame_num"] >= feasible_min_num]

    if not post:
        return last_frame_num, "fallback_last"

    if mode == "random":
        return random.choice(post)["frame_num"], "random"

    # "hard": find longest consecutive walk/run run in feasible post-window
    best_end_frame = None
    best_run_len   = 0
    cur_run_len    = 0
    cur_run_end    = None

    for rec in post:
        if rec["motion_active"] == 1:
            cur_run_len += 1
            cur_run_end  = rec["frame_num"]
            if cur_run_len > best_run_len:
                best_run_len   = cur_run_len
                best_end_frame = cur_run_end
        else:
            cur_run_len = 0

    if best_end_frame is not None:
        return best_end_frame, f"hard_walk_run{best_run_len}"

    # Fallback: last active frame, or plain last frame
    active = [r for r in post if r["motion_active"] == 1]
    if active:
        return active[-1]["frame_num"], "hard_last_active"
    return last_frame_num, "hard_fallback_last"


# ============================================================
# Pose Extraction (YOLO)
# ============================================================

_POSE_MODEL = None
_POSE_DEVICE = None


def init_pose(model_path: str):
    global _POSE_MODEL, _POSE_DEVICE
    if _POSE_MODEL is not None:
        return
    try:
        from ultralytics import YOLO
        _POSE_MODEL = YOLO(model_path)
        _POSE_DEVICE = 0 if torch.cuda.is_available() else "cpu"
        print(f"[INFO] YOLO-pose on device={_POSE_DEVICE}")
    except Exception as e:
        print(f"[WARN] YOLO init failed: {e}")


def _expand_box(x1, y1, x2, y2, w, h, scale=1.2):
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw = (x2 - x1) * scale
    bh = (y2 - y1) * scale
    nx1 = max(0, int(cx - bw / 2))
    ny1 = max(0, int(cy - bh / 2))
    nx2 = min(w - 1, int(cx + bw / 2))
    ny2 = min(h - 1, int(cy + bh / 2))
    if nx2 <= nx1 or ny2 <= ny1:
        return 0, 0, w - 1, h - 1
    return nx1, ny1, nx2, ny2


def extract_pose(frames, boxes, model_path, scale=1.2, imgsz=512):
    """Returns (T, 34) array of normalized keypoints."""
    init_pose(model_path)
    if _POSE_MODEL is None:
        return np.zeros((len(frames), 34), dtype=np.float32)

    all_kps = []
    for img, (x1, y1, x2, y2) in zip(frames, boxes):
        h, w = img.shape[:2]
        ex1, ey1, ex2, ey2 = _expand_box(
            float(x1), float(y1), float(x2), float(y2), w, h, scale
        )
        crop = img[ey1:ey2, ex1:ex2]
        if crop.size == 0:
            crop = img
        res = _POSE_MODEL.predict(
            crop, imgsz=imgsz, verbose=False, device=_POSE_DEVICE
        )
        if (res
                and res[0].keypoints is not None
                and res[0].keypoints.xy is not None
                and res[0].keypoints.xy.shape[0] > 0):
            xy = res[0].keypoints.xy[0].cpu().numpy().astype(np.float32)
            xy[:, 0] = (ex1 + xy[:, 0]) / max(w, 1)
            xy[:, 1] = (ey1 + xy[:, 1]) / max(h, 1)
        else:
            xy = np.zeros((17, 2), dtype=np.float32)
        all_kps.append(xy)

    return np.stack(all_kps).reshape(len(frames), -1)


# ============================================================
# Mirroring Helpers
# ============================================================

def mirror_bbox_pixel(x1, y1, x2, y2, img_w: int):
    """
    Flip bbox horizontally in pixel coordinates.
      new_x1 = (W-1) - x2_old
      new_x2 = (W-1) - x1_old
    y coords unchanged.
    """
    W = float(img_w - 1)
    mx1 = W - x2
    mx2 = W - x1
    # Guard: ensure x1 <= x2 after potential float rounding
    if mx1 > mx2:
        mx1, mx2 = mx2, mx1
    mx1 = float(np.clip(mx1, 0, W))
    mx2 = float(np.clip(mx2, 0, W))
    return mx1, y1, mx2, y2


def mirror_bbox_normalized(x1n, y1n, x2n, y2n):
    """
    Flip normalized bbox.
      new_x1n = 1 - x2n_old
      new_x2n = 1 - x1n_old
    """
    mx1n = 1.0 - x2n
    mx2n = 1.0 - x1n
    if mx1n > mx2n:
        mx1n, mx2n = mx2n, mx1n
    return float(np.clip(mx1n, 0, 1)), y1n, float(np.clip(mx2n, 0, 1)), y2n


def mirror_pose_array(pose: np.ndarray) -> np.ndarray:
    """
    Mirror a (T, 34) pose array.
    Steps:
      1. Flip x coords: x_new = 1 - x_old  (pose is already normalized [0,1])
      2. Swap COCO left/right keypoint pairs so body semantics are preserved.
    """
    T = pose.shape[0]
    kps = pose.reshape(T, 17, 2).copy()
    kps[:, :, 0] = 1.0 - kps[:, :, 0]  # flip x
    for l_idx, r_idx in COCO_LR_PAIRS:
        kps[:, [l_idx, r_idx], :] = kps[:, [r_idx, l_idx], :]
    return kps.reshape(T, 34)


def mirrored_image_path(original_path: str) -> str:
    """
    Build the on-disk path for a mirrored frame.
    Written to: <parent>/mirrored/<stem>_mirror<ext>
    This path is a real, loadable PNG — prep_local (VGG/RAFT) can use it
    as a normal image path without any special handling.
    """
    p = Path(original_path)
    return str(p.parent / "mirrored" / f"{p.stem}_mirror{p.suffix}")


def ensure_mirrored_image_on_disk(original_path: str) -> str:
    """
    Write the flipped image to disk if not already present.
    Returns the path to the mirrored file.
    On I/O failure falls back to returning original_path (graceful).
    """
    m_path = mirrored_image_path(original_path)
    if Path(m_path).exists():
        return m_path
    img = cv2.imread(original_path)
    if img is None:
        return original_path
    mp = Path(m_path)
    mp.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(m_path, cv2.flip(img, 1))
    return m_path


# ============================================================
# NPZ Save — exactly matches PIE/JAAD contract
# ============================================================

def save_npz(path: Path, feats: dict, meta: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    seq_len = int(meta["seq_len"])

    save_dict = {
        # ── Features ──────────────────────────────────────────
        "bbox":         feats["bbox"].astype(np.float32),
        "bbox_xyxy":    feats["bbox_xyxy"].astype(np.float32),
        "pose":         feats["pose"].astype(np.float32),
        "speed":        feats["speed"].astype(np.float32),

        # ── Placeholders (filled by later pipeline stages) ─────
        "local_cnn":    np.zeros((seq_len, 512, 7, 7),   dtype=np.float32),
        "local_motion": np.zeros((seq_len, 2, 224, 224), dtype=np.float32),
        "sem_labels":   np.zeros((seq_len, 1, 1),         dtype=np.uint16),
        "cat_depth":    np.zeros((seq_len, 2, 1, 1),      dtype=np.float32),

        # ── Labels ────────────────────────────────────────────
        "label":                 np.int32(meta["label"]),
        "intention_prob_seq":    np.asarray(meta["intention_prob_seq"],  dtype=np.float32),
        "intent_label_track":    np.int32(meta["intent_label_track"]),
        "intent_prob_track_max": np.float32(meta["intent_prob_track_max"]),
        "cross_label_track":     np.int32(meta["cross_label_track"]),

        # ── Identifiers ───────────────────────────────────────
        "pid":        np.array([meta["pid"]],        dtype=object),
        "set_id":     np.array([meta["set_id"]],     dtype=object),
        "video_name": np.array([meta["video_name"]], dtype=object),
        "track_id":   np.array([meta["track_id"]],   dtype=object),
        "dataset":    np.array(["titan"],            dtype=object),

        # ── Frame metadata ─────────────────────────────────────
        "img_paths": np.array(meta["img_paths"], dtype=object),
        "frame_idx": np.asarray(meta["frame_idx"], dtype=np.int32),

        # ── Window metadata ────────────────────────────────────
        "t_start_ds":   np.int32(meta["t_start_ds"]),
        "t_end_ds":     np.int32(meta["t_end_ds"]),
        "t_start_full": np.int32(meta["t_start_full"]),
        "t_end_full":   np.int32(meta["t_end_full"]),

        "event_frame":           np.int32(meta["event_frame"]),
        "pseudo_event_frame":    np.int32(meta["pseudo_event_frame"]),
        "pseudo_event_strategy": np.array(
            [meta.get("pseudo_event_strategy", "na")], dtype=object
        ),
        "first_cross_full": np.int32(meta["first_cross_full"]),

        # ── Timing ────────────────────────────────────────────
        "tte_sec_target": np.float32(meta["tte_sec_target"]),
        "tte_sec_actual": np.float32(meta["tte_sec_actual"]),

        # ── Params ────────────────────────────────────────────
        # data_fstride is ALWAYS 1 (see module docstring — no cross-event skip)
        "data_fstride": np.int32(1),
        "out_stride":   np.int32(1),          # stride=1 always for TITAN
        "raw_fps":      np.float32(meta["raw_fps"]),
        "eff_fps":      np.float32(meta["raw_fps"]),  # = raw_fps since stride=1
        "seq_len":      np.int32(seq_len),
        "obs_sec":      np.float32(meta["obs_sec"]),
        "split":        np.array([meta["split"]], dtype=object),
        "mp4_png_offset": np.int32(0),

        "intention_prob_full": np.asarray(
            meta["intention_prob_full"], dtype=np.float32
        ),

        # ── Augmentation flag ──────────────────────────────────
        # 0 = original orientation, 1 = horizontally mirrored
        "is_mirrored": np.int32(meta.get("is_mirrored", 0)),

        # ── Processing flags ───────────────────────────────────
        "_stage_kinematics": np.int32(1),
        "_stage_local":      np.int32(0),
        "_stage_scene":      np.int32(0),
    }

    np.savez_compressed(path, **save_dict)


# ============================================================
# Deterministic Split Assignment
# ============================================================

def make_split_map(
    clip_names: List[str],
    seed: int,
    train_frac: float,
    val_frac: float,
) -> Dict[str, str]:
    """clip_name → 'train' | 'val' | 'test'. Fully deterministic."""
    clips = sorted(clip_names)
    rng   = random.Random(seed)
    rng.shuffle(clips)
    n       = len(clips)
    n_train = int(round(n * train_frac))
    n_val   = int(round(n * val_frac))
    result  = {}
    for i, clip in enumerate(clips):
        if i < n_train:
            result[clip] = "train"
        elif i < n_train + n_val:
            result[clip] = "val"
        else:
            result[clip] = "test"
    return result


# ============================================================
# Core: Sliding-Window Extraction for One Track
# ============================================================

def process_track(
    track: List[dict],
    track_label: int,
    first_cross_frame: int,
    first_cross_sample_idx: int,
    args,
    split: str,
    do_mirror: bool,
    out_dir: Path,
    start_seq_index: int,
) -> Tuple[int, int, int]:
    """
    Extract all valid sliding windows from a single pedestrian track.

    stride is ALWAYS 1 — see module docstring for why.
    With stride=1 on 10fps TITAN data, a seq_len=10 window = 1 second.

    When do_mirror=True, every accepted window produces TWO NPZ files:
      1. Original orientation
      2. Horizontally mirrored

    Returns: (n_written, n_pos, n_neg)
      n_written counts both original and mirror NPZs.
    """
    T_full = len(track)
    if T_full < args.seq_len:
        return 0, 0, 0

    obs_sec   = args.seq_len / args.raw_fps   # stride=1 so eff_fps=raw_fps
    tte_min_f = int(round(args.tte_min_sec * args.raw_fps))
    tte_max_f = int(round(args.tte_max_sec * args.raw_fps))

    frame_nums = np.array([r["frame_num"] for r in track], dtype=np.int32)
    sample_idxs = np.array([r["sample_idx"] for r in track], dtype=np.int32)
    step       = max(1, int(round(args.seq_len * (1.0 - args.overlap))))

    clip_id  = track[0]["clip_id"]
    track_id = track[0]["track_id"]

    written, n_pos, n_neg = 0, 0, 0
    neg_windows_written = 0
    for w_start in range(0, T_full - args.seq_len + 1, step):
        w_end      = w_start + args.seq_len - 1       # inclusive
        win_idx    = np.arange(w_start, w_end + 1)
        win_frames = frame_nums[win_idx]       # actual image filenames, e.g. 6,12,18
        win_samples = sample_idxs[win_idx]     # 10Hz sample indices, e.g. 0,1,2
        end_sample = int(win_samples[-1])

        # ── TTE filter ─────────────────────────────────────────────────
        if track_label == 1:
            tte_steps = first_cross_sample_idx - end_sample
            if tte_steps <= 0 or tte_steps < tte_min_f or tte_steps > tte_max_f:
                continue

            tte_actual = float(tte_steps) / args.raw_fps
            pseudo_event_frame = -1
            pseudo_strat = "na"
            # Cap negatives per track to limit imbalance
            max_neg = getattr(args, "max_neg_per_track", 2)
            if max_neg > 0 and neg_windows_written >= max_neg:
                continue

        else:
            # Compute pseudo-event FIRST, then filter on its TTE (mirrors PIE logic)
            pseudo_event_frame, pseudo_strat = choose_pseudo_event_frame_titan(
                track=track,
                end_idx_in_track=int(win_idx[-1]),
                tte_min_f=tte_min_f,
                mode=args.neg_mode,
            )

            # Convert pseudo_event_frame (image frame number) -> sample index
            pseudo_sample_idx = next(
                (r["sample_idx"] for r in track if r["frame_num"] == pseudo_event_frame),
                None,
            )
            if pseudo_sample_idx is None:
                continue

            tte_steps = pseudo_sample_idx - end_sample
            if tte_steps < tte_min_f or tte_steps > tte_max_f:
                continue

            tte_actual = float(tte_steps) / args.raw_fps

        # ── Monotonicity (paranoia guard with stride=1) ─────────────────
        # Require consecutive 10Hz annotated frames.
        if not np.all(np.diff(win_samples) == 1):
            continue

        # ── Load images ─────────────────────────────────────────────────
        records = [track[i] for i in win_idx]
        frames  = []
        ok      = True
        for rec in records:
            img = cv2.imread(rec["img_path"])
            if img is None:
                ok = False
                break
            frames.append(img)
        if not ok:
            continue

        # ── Bbox arrays ─────────────────────────────────────────────────
        bbox_norm_list  = []
        boxes_xyxy_list = []
        for img, rec in zip(frames, records):
            img_h, img_w = img.shape[:2]
            x1, y1, x2, y2 = [float(v) for v in rec["bbox_xyxy"]]
            x1 = float(np.clip(x1, 0, max(img_w - 1, 1)))
            x2 = float(np.clip(x2, 0, max(img_w - 1, 1)))
            y1 = float(np.clip(y1, 0, max(img_h - 1, 1)))
            y2 = float(np.clip(y2, 0, max(img_h - 1, 1)))
            boxes_xyxy_list.append([x1, y1, x2, y2])
            bbox_norm_list.append([x1 / img_w, y1 / img_h,
                                   x2 / img_w, y2 / img_h])

        bbox_norm  = np.array(bbox_norm_list,  dtype=np.float32)
        boxes_xyxy = np.array(boxes_xyxy_list, dtype=np.float32)

        # ── Pose ────────────────────────────────────────────────────────
        if not args.no_pose:
            pose = extract_pose(
                frames, boxes_xyxy, args.pose_model,
                scale=args.scale_crop, imgsz=args.pose_imgsz,
            )
        else:
            pose = np.zeros((args.seq_len, 34), dtype=np.float32)

        # ── Labels / intention seq ───────────────────────────────────────
        cross_seq           = np.array([r["cross_label_frame"] for r in records],
                                        dtype=np.float32)
        intention_prob_full = np.array([r["cross_label_frame"] for r in track],
                                        dtype=np.float32)

        # ── Shared meta ─────────────────────────────────────────────────
        base_meta = dict(
            label=track_label,
            intention_prob_seq=cross_seq.tolist(),
            intent_label_track=track_label,
            intent_prob_track_max=float(track_label),
            cross_label_track=track_label,
            pid=track_id,
            set_id="titan",
            video_name=clip_id,
            track_id=f"titan/{clip_id}/{track_id}",
            img_paths=[r["img_path"] for r in records],
            frame_idx=win_frames.tolist(),
            t_start_ds=int(w_start),
            t_end_ds=int(w_end),
            t_start_full=int(win_idx[0]),
            t_end_full=int(win_idx[-1]),
            event_frame=first_cross_frame if track_label == 1 else -1,
            pseudo_event_frame=pseudo_event_frame,
            pseudo_event_strategy=pseudo_strat,
            first_cross_full=first_cross_frame if track_label == 1 else -1,
            tte_sec_target=float(args.tte_min_sec) if track_label == 1 else -1.0,
            tte_sec_actual=tte_actual,
            raw_fps=args.raw_fps,
            obs_sec=obs_sec,
            split=split,
            intention_prob_full=intention_prob_full.tolist(),
            seq_len=args.seq_len,
            is_mirrored=0,
        )

        base_feats = dict(
            bbox=bbox_norm,
            bbox_xyxy=boxes_xyxy,
            pose=pose,
            speed=np.zeros((args.seq_len, 1), dtype=np.float32),
        )

        # ── Write ORIGINAL window ────────────────────────────────────────
        out_path = out_dir / f"seq_{start_seq_index + written:06d}.npz"
        save_npz(out_path, base_feats, base_meta)
        written += 1
        if track_label == 1:
            n_pos += 1
        else:
            n_neg += 1
            if track_label == 0:
                    neg_windows_written += 1
        # ── Write MIRRORED window (if requested) ─────────────────────────
        if do_mirror:
            # -- Mirror images to disk --
            mirror_img_paths = [
                ensure_mirrored_image_on_disk(rec["img_path"]) for rec in records
            ]

            # -- Mirror bboxes --
            m_boxes_xyxy_list = []
            m_bbox_norm_list  = []
            for img, rec in zip(frames, records):
                img_h, img_w = img.shape[:2]
                x1, y1, x2, y2 = [float(v) for v in rec["bbox_xyxy"]]
                x1 = float(np.clip(x1, 0, max(img_w - 1, 1)))
                x2 = float(np.clip(x2, 0, max(img_w - 1, 1)))

                # Pixel mirror
                mx1, my1, mx2, my2 = mirror_bbox_pixel(x1, y1, x2, y2, img_w)
                m_boxes_xyxy_list.append([mx1, my1, mx2, my2])

                # Normalized mirror
                nx1n = x1 / img_w
                nx2n = x2 / img_w
                ny1n = y1 / img_h
                ny2n = y2 / img_h
                mnx1n, mny1n, mnx2n, mny2n = mirror_bbox_normalized(
                    nx1n, ny1n, nx2n, ny2n
                )
                m_bbox_norm_list.append([mnx1n, mny1n, mnx2n, mny2n])

            m_boxes_xyxy = np.array(m_boxes_xyxy_list, dtype=np.float32)
            m_bbox_norm  = np.array(m_bbox_norm_list,  dtype=np.float32)

            # -- Mirror pose --
            m_pose = mirror_pose_array(pose)

            # -- Mirror meta --
            m_meta = dict(base_meta)
            m_meta["img_paths"]            = mirror_img_paths
            m_meta["track_id"]             = f"titan/{clip_id}/{track_id}_mirror"
            m_meta["is_mirrored"]          = 1
            m_meta["pseudo_event_strategy"] = (
                pseudo_strat + "_mirror" if pseudo_strat != "na" else "na"
            )

            m_feats = dict(
                bbox=m_bbox_norm,
                bbox_xyxy=m_boxes_xyxy,
                pose=m_pose,
                speed=np.zeros((args.seq_len, 1), dtype=np.float32),
            )

            out_path_m = out_dir / f"seq_{start_seq_index + written:06d}.npz"
            save_npz(out_path_m, m_feats, m_meta)
            written += 1
            if track_label == 1:
                n_pos += 1
            else:
                n_neg += 1
                if track_label == 0:
                    neg_windows_written += 1

    return written, n_pos, n_neg


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="TITAN Step 1: Kinematics — bbox, pose, speed=0, sliding window"
    )
    ap.add_argument("--titan_root", required=True,
                    help="Root containing clip_XXXXX/ dirs "
                         "(e.g. /data/TITAN/titan_data/dataset)")
    ap.add_argument("--out_root", required=True,
                    help="Output root (e.g. /data/TITAN_PREP_OUT)")

    # Window / timing
    ap.add_argument("--splits",      default="train,val,test")
    ap.add_argument("--seq_len",     type=int,   default=10)
    ap.add_argument("--overlap",     type=float, default=0.5)
    ap.add_argument("--tte_min_sec", type=float, default=0.5)
    ap.add_argument("--tte_max_sec", type=float, default=2.0)
    ap.add_argument("--max_neg_per_track", type=int, default=2,
                help="Max negative windows per track (0=unlimited)")
    ap.add_argument("--raw_fps",     type=float, default=10.0)
    # NOTE: no --stride arg. stride is always 1. See module docstring.

    # Mirroring
    ap.add_argument("--mirror", action="store_true",
                    help="Write horizontally-mirrored NPZs alongside originals")
    ap.add_argument("--mirror_splits", default="train",
                    help="Which splits to mirror (default: train only). "
                         "val/test should never be mirrored.")

    # Negative hard-mining
    ap.add_argument("--neg_mode", default="hard",
                    choices=["hard", "last", "random"],
                    help="Pseudo-event strategy for negative tracks "
                         "(hard=walking-run-based, last=last-frame, random)")

    # Pose
    ap.add_argument("--no_pose",    action="store_true")
    ap.add_argument("--pose_model", default="yolov8n-pose.pt")
    ap.add_argument("--pose_imgsz", type=int,   default=512)
    ap.add_argument("--scale_crop", type=float, default=1.2)

    # Split
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--val_frac",   type=float, default=0.1)
    ap.add_argument("--seed",       type=int,   default=42)

    # Misc
    ap.add_argument("--min_track_len", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0,
                    help="Process at most N clips (0=all)")
    ap.add_argument("--debug", type=int, default=0)

    args = ap.parse_args()

    mirror_splits    = {s.strip() for s in args.mirror_splits.split(",") if s.strip()}
    requested_splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    # Safety: never allow mirroring on val/test
    mirror_splits -= {"val", "test"}

    obs_sec = args.seq_len / args.raw_fps   # stride=1 always

    print("=" * 70)
    print("TITAN Step 1: KINEMATICS  (stride=1, data_fstride=1)")
    print("=" * 70)
    print(f"  titan_root:    {args.titan_root}")
    print(f"  out_root:      {args.out_root}")
    print(f"  splits:        {requested_splits}")
    print(f"  seq_len:       {args.seq_len}  "
          f"obs={obs_sec:.2f}s @ {args.raw_fps}fps")
    print(f"  overlap:       {args.overlap}")
    print(f"  TTE:           [{args.tte_min_sec}, {args.tte_max_sec}] s")
    print(f"  neg_mode:      {args.neg_mode}")
    print(f"  mirror:        {args.mirror}"
          + (f"  (splits: {mirror_splits})" if args.mirror else ""))
    print(f"  pose:          {'DISABLED' if args.no_pose else args.pose_model}")
    print(f"  speed:         zeros (TITAN has no OBD speed)")
    print("=" * 70)

    titan_root = Path(args.titan_root)
    clips      = find_clips(titan_root)

    if not clips:
        print(f"[ERROR] No clip directories found under {titan_root}")
        print("  Expected: <titan_root>/clip_XXXXX/images_anonymized/")
        return

    print(f"\n[INFO] Found {len(clips)} clips")

    split_map = make_split_map(
        [c.name for c in clips], args.seed, args.train_frac, args.val_frac
    )
    cnt = {"train": 0, "val": 0, "test": 0}
    for v in split_map.values():
        cnt[v] = cnt.get(v, 0) + 1
    print(f"[INFO] Clip split: train={cnt['train']}  val={cnt['val']}  test={cnt['test']}")

    out_dirs     = {}
    seq_counters = {}
    stats        = {}
    for sp in requested_splits:
        d = Path(args.out_root) / sp
        d.mkdir(parents=True, exist_ok=True)
        out_dirs[sp]     = d
        seq_counters[sp] = 0
        stats[sp]        = {"clips": 0, "tracks": 0,
                            "orig": 0, "mirror": 0, "pos": 0, "neg": 0}

    clips_to_run = clips if args.limit == 0 else clips[:args.limit]

    for clip_dir in tqdm(clips_to_run, desc="Clips"):
        clip_name  = clip_dir.name
        split_name = split_map.get(clip_name)
        if split_name not in requested_splits:
            continue

        do_mirror = args.mirror and (split_name in mirror_splits)

        tracks = build_tracks(clip_dir)
        if not tracks:
            if args.debug > 0:
                print(f"  [SKIP] {clip_name}: no valid annotations/images")
            continue

        stats[split_name]["clips"] += 1

        for tid, track in tracks.items():
            if args.min_track_len > 0 and len(track) < args.min_track_len:
                continue

            tte_min_f = int(round(args.tte_min_sec * args.raw_fps))
            track_label, first_cross_frame, first_cross_sample_idx = derive_track_label(
                track, tte_min_f=tte_min_f
            )
            start_idx = seq_counters[split_name]
            n_written, n_pos, n_neg = process_track(
                track, track_label, first_cross_frame, first_cross_sample_idx,
                args, split_name, do_mirror,
                out_dirs[split_name], start_idx,
            )

            seq_counters[split_name] += n_written
            stats[split_name]["tracks"] += 1
            stats[split_name]["pos"]    += n_pos
            stats[split_name]["neg"]    += n_neg

            if do_mirror:
                # Originals and mirrors are interleaved; each pair = 2 writes
                stats[split_name]["orig"]   += n_written // 2 + n_written % 2
                stats[split_name]["mirror"] += n_written // 2
            else:
                stats[split_name]["orig"] += n_written

            if args.debug > 0 and n_written > 0:
                print(f"  {clip_name}/{tid}  label={track_label}  "
                      f"T={len(track)}  wrote={n_written}"
                      f"{'  [+mirror]' if do_mirror else ''}")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DONE — Summary")
    print("=" * 70)
    for sp in requested_splits:
        st = stats[sp]
        total = st["orig"] + st["mirror"]
        m_note = (f"  orig={st['orig']} mirror={st['mirror']}"
                  if st["mirror"] > 0 else "")
        print(f"  [{sp:5s}]  clips={st['clips']:4d}  tracks={st['tracks']:5d}  "
              f"total={total:6d}{m_note}  "
              f"pos={st['pos']}  neg={st['neg']}")

    print()
    print("Next steps:")
    print("  Step 2 — Local features (VGG19 + RAFT):")
    print("    Reuse prep_local_jaad.py — it reads NPZ img_paths directly.")
    print("    Mirrored NPZs point to the _mirror.png files already on disk,")
    print("    so VGG/RAFT will process them as normal images automatically.")
    print()
    print("  Step 3 — Scene (Mask2Former + ManyDepth):")
    print("    Use prep_scene_jaad.py with --jaad_root pointing to titan_root.")
    print("    NOTE: mirrored NPZs share scene features with their originals")
    print("    (same frame, same semantic/depth content). You can skip re-running")
    print("    scene prep for mirror NPZs by checking is_mirrored==1 if desired.")
    print()
    print("  Step 4 — Normalization stats:")
    print("    python compute_stats.py \\")
    print("        --npz_root /data/TITAN_PREP_OUT --dataset titan")
    print("=" * 70)


if __name__ == "__main__":
    main()