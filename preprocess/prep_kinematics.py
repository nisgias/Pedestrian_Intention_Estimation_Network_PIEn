#!/usr/bin/env python3
"""
Step 1: Kinematics Features - bbox, pose, speed
Creates initial NPZ files with track metadata and kinematic features.

This is the FIRST script to run - it creates the NPZ structure.

Now uses an overlapping sliding window approach to generate multiple
sequences per pedestrian, following the benchmark protocol:
  - seq_len=10, stride=2 -> 10 frames * 2 stride = 20 raw frames @ 30fps = 0.5s observation
  - overlap=0.5 (60% used in PIE benchmark)
  - TTE between 30-60 raw frames (1-2s)

Usage:
    python prep_kinematics.py \
        --pie_root /data/PIE \
        --iface_dir /data/pie_interface \
        --out_root /data/output \
        --splits train,val,test
"""

import sys
import os
import re
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import cv2
import numpy as np
import torch

# ============================================================================
# Constants
# ============================================================================

PIP_CLASSES = [
    "bg",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "handbag",
    "road", "sidewalk", "sky", "building", "vegetation",
]
PIP_ID = {n: i for i, n in enumerate(PIP_CLASSES)}

IMG_RE = re.compile(r"(?:.*/)?images/(set\d{2})/(video_\d{4})/(\d{5})\.png$")

# ============================================================================
# Helpers
# ============================================================================

def parse_img_path(path: str) -> Tuple[str, str, int]:
    path = path.replace("\\", "/")
    m = IMG_RE.match(path)
    if not m:
        raise ValueError(f"Cannot parse img path: {path}")
    set_id, video_name, frame_str = m.groups()
    return set_id, video_name, int(frame_str)


def expand_box(x1, y1, x2, y2, w, h, scale=1.2):
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw, bh = (x2 - x1) * scale, (y2 - y1) * scale
    nx1 = max(0, int(cx - bw / 2))
    ny1 = max(0, int(cy - bh / 2))
    nx2 = min(w - 1, int(cx + bw / 2))
    ny2 = min(h - 1, int(cy + bh / 2))
    if nx2 <= nx1 or ny2 <= ny1:
        return 0, 0, w - 1, h - 1
    return nx1, ny1, nx2, ny2


def _downsample_gray(img_bgr, side=64):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if side > 0:
        g = cv2.resize(g, (side, side), interpolation=cv2.INTER_AREA)
    return g.astype(np.float32)


def _mad(a, b):
    return float(np.mean(np.abs(a - b)))


# ============================================================================
# Crossing / Intention Helpers
# ============================================================================

def _cross_masks(cross_seq, crossing_values=(1,), irrelevant_values=(2, -1)):
    cross_seq = np.asarray(cross_seq)
    if cross_seq.dtype.kind in ("U", "S", "O"):
        s = np.array([str(x).lower() for x in cross_seq], dtype=object)
        irrelevant_mask = np.array([("irrelevant" in x) for x in s], dtype=bool)
        crossing_mask = np.array(
            [("cross" in x) and ("not" not in x) and ("irrelevant" not in x) for x in s],
            dtype=bool
        )
        return crossing_mask, irrelevant_mask
    crossing_mask = np.isin(cross_seq, list(crossing_values))
    irrelevant_mask = np.isin(cross_seq, list(irrelevant_values))
    return crossing_mask, irrelevant_mask


def find_first_crossing_idx(cross_seq) -> Optional[int]:
    crossing_mask, _ = _cross_masks(cross_seq)
    idx = np.where(crossing_mask)[0]
    return int(idx[0]) if len(idx) else None


def to_intent_array(x, T: int) -> np.ndarray:
    if x is None:
        return np.zeros((T,), dtype=np.float32)
    if isinstance(x, (int, float, np.number)):
        return np.full((T,), float(x), dtype=np.float32)
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.size == 1:
        return np.full((T,), float(arr[0]), dtype=np.float32)
    if arr.size < T:
        out = np.full((T,), float(arr[-1]), dtype=np.float32)
        out[:arr.size] = arr
        return out
    return arr[:T]


def _unwrap_pid(raw_pid) -> str:
    """Unwrap PIE pedestrian ID from nested list format.
    
    PIE returns pid per track as a list-of-lists, e.g.
    [['1_1_1'], ['1_1_1'], ...] (one entry per frame).
    We extract the unique scalar string.
    """
    if raw_pid is None:
        return "unk"
    # If it's already a plain string
    if isinstance(raw_pid, str):
        return raw_pid
    # numpy scalar
    if isinstance(raw_pid, (np.str_, np.bytes_)):
        return str(raw_pid)
    # List or array: dig into it
    item = raw_pid
    while isinstance(item, (list, np.ndarray)):
        if len(item) == 0:
            return "unk"
        item = item[0]
    return str(item)


def intent_label(intent_seq: np.ndarray, thr: float = 0.5) -> int:
    if intent_seq.size == 0:
        return 0
    return 1 if float(np.nanmax(intent_seq)) >= thr else 0


# ============================================================================
# Negative Pseudo-Event Selection
# ============================================================================

def choose_pseudo_event_frame(
    frame_idx_full: np.ndarray,
    intent_full: np.ndarray,
    keep_idx: np.ndarray,
    seq_len: int,
    raw_offset: int,
    raw_fps: float,
    mode: str = "hybrid",
    intent_hi_thr: float = 0.5,
    topk: int = 5,
    last_sec: float = 3.0,
) -> Tuple[Optional[int], str]:
    """Choose pseudo-event frame for negative samples."""
    frame_idx_full = np.asarray(frame_idx_full, dtype=np.int32).reshape(-1)
    intent_full = np.asarray(intent_full, dtype=np.float32).reshape(-1)
    keep_idx = np.asarray(keep_idx, dtype=np.int32).reshape(-1)

    if frame_idx_full.size == 0 or keep_idx.size < seq_len:
        return None, "no_data"

    earliest_end = int(frame_idx_full[int(keep_idx[seq_len - 1])])
    min_pseudo = earliest_end + raw_offset

    feas = np.where(frame_idx_full >= min_pseudo)[0]
    if feas.size == 0:
        return None, "no_feasible"

    intent_feas = np.nan_to_num(intent_full[feas], nan=-1.0)
    max_int = float(intent_feas.max()) if intent_feas.size else -1.0

    def pick_last():
        return int(frame_idx_full[feas[-1]]), "last"

    def pick_topk():
        k = max(1, min(topk, feas.size))
        top = feas[np.argsort(intent_feas)[-k:]]
        return int(frame_idx_full[np.random.choice(top)]), f"topk({k})"

    def pick_late():
        late_min = int(frame_idx_full[-1] - round(last_sec * raw_fps))
        cand = feas[frame_idx_full[feas] >= late_min]
        if cand.size == 0:
            return pick_last()
        return int(frame_idx_full[np.random.choice(cand)]), f"late({last_sec}s)"

    if mode == "last":
        return pick_last()
    if mode == "topk_intent":
        return pick_topk()
    if mode == "random_late":
        return pick_late()
    # hybrid
    if max_int >= intent_hi_thr:
        f, s = pick_topk()
        return f, f"hybrid->{s}"
    f, s = pick_late()
    return f, f"hybrid->{s}"


# ============================================================================
# Video Cache
# ============================================================================

class VideoCache:
    def __init__(self, clips_root: Path):
        self.clips_root = Path(clips_root)
        self.cache = {}
        self.offset_cache = {}

    def _key(self, set_id, video_name):
        return f"{set_id}/{video_name}"

    def get_cap(self, set_id, video_name):
        key = self._key(set_id, video_name)
        if key not in self.cache:
            mp4 = self.clips_root / set_id / f"{video_name}.mp4"
            if not mp4.exists():
                raise FileNotFoundError(f"Video not found: {mp4}")
            cap = cv2.VideoCapture(str(mp4))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open: {mp4}")
            self.cache[key] = cap
        return self.cache[key]

    def infer_offset(self, set_id, video_name, img_paths, frame_idx, max_off=2, n_samples=3, verbose=False):
        key = self._key(set_id, video_name)
        if key in self.offset_cache:
            return self.offset_cache[key]

        cap = self.get_cap(set_id, video_name)
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        T = min(len(img_paths), len(frame_idx))

        if T <= 0 or fc <= 0:
            self.offset_cache[key] = 0
            return 0

        samples = []
        for si in np.linspace(0, T - 1, min(n_samples, T), dtype=int):
            png_path = Path(img_paths[si])
            # Try absolute as-is
            cand = png_path
            if not cand.exists():
                # Try relative to PIE root (clips_root is PIE_root/PIE_clips, so parent = PIE_root)
                cand = self.clips_root.parent / png_path

            if not cand.exists():
                if verbose:
                    print(f"[SYNC] missing png: {png_path}")
                continue

            png = cv2.imread(str(cand))
            if png is None:
                if verbose:
                    print(f"[SYNC] failed imread: {cand}")
                continue

            samples.append((int(frame_idx[si]), _downsample_gray(png)))

        if not samples:
            self.offset_cache[key] = 0
            return 0

        best_off, best_score = 0, float('inf')
        for off in range(-max_off, max_off + 1):
            diffs = []
            for fi, png_ds in samples:
                idx = fi + off
                if idx < 0 or idx >= fc:
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, fr = cap.read()
                if ok and fr is not None:
                    fr_ds = _downsample_gray(fr)
                    if fr_ds.shape != png_ds.shape:
                        fr_ds = cv2.resize(fr_ds, (png_ds.shape[1], png_ds.shape[0]))
                    diffs.append(_mad(png_ds, fr_ds))
            if diffs and np.mean(diffs) < best_score:
                best_score = np.mean(diffs)
                best_off = off

        if verbose:
            print(f"[SYNC] {key}: offset={best_off}, score={best_score:.2f}")
        self.offset_cache[key] = best_off
        return best_off

    def read_frame(self, set_id, video_name, frame_idx):
        cap = self.get_cap(set_id, video_name)
        key = self._key(set_id, video_name)
        off = self.offset_cache.get(key, 0)
        idx = frame_idx + off
        if idx < 0:
            raise RuntimeError(f"Invalid idx {idx}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read frame {idx} from {key}")
        return frame

    def release_all(self):
        for cap in self.cache.values():
            try:
                cap.release()
            except:
                pass
        self.cache.clear()
        self.offset_cache.clear()


# ============================================================================
# YOLO Pose
# ============================================================================

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
        print(f"[INFO] YOLO-pose on {_POSE_DEVICE}")
    except Exception as e:
        print(f"[WARN] YOLO init failed: {e}")


def extract_pose(frames, boxes, model_path, scale=1.2, imgsz=512):
    init_pose(model_path)
    if _POSE_MODEL is None:
        return np.zeros((len(frames), 34), dtype=np.float32)

    all_kps = []
    for img, (x1, y1, x2, y2) in zip(frames, boxes):
        h, w = img.shape[:2]
        x1, y1, x2, y2 = expand_box(x1, y1, x2, y2, w, h, scale)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            crop = img

        res = _POSE_MODEL.predict(crop, imgsz=imgsz, verbose=False, device=_POSE_DEVICE)
        if len(res) and res[0].keypoints is not None and res[0].keypoints.xy is not None and res[0].keypoints.xy.shape[0] > 0:
            xy = res[0].keypoints.xy[0].cpu().numpy().astype(np.float32)
            ih, iw = img.shape[:2]
            # convert crop coords -> full image coords, then normalize to [0,1]
            xy[:, 0] = (x1 + xy[:, 0]) / max(iw, 1)
            xy[:, 1] = (y1 + xy[:, 1]) / max(ih, 1)

        else:
            xy = np.zeros((17, 2), dtype=np.float32)
        all_kps.append(xy)

    return np.stack(all_kps).reshape(len(frames), -1)


# ============================================================================
# Save
# ============================================================================

def save_npz(path: Path, feats: dict, meta: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    
    save_dict = {
        # Features
        "bbox": feats["bbox"].astype(np.float32),
        "bbox_xyxy": feats["bbox_xyxy"].astype(np.float32),
        "pose": feats["pose"].astype(np.float32),
        "speed": feats["speed"].astype(np.float32),
        
        # Placeholders for later stages
        "local_cnn": np.zeros((meta["seq_len"], 512, 7, 7), dtype=np.float32),
        "local_motion": np.zeros((meta["seq_len"], 2, 224, 224), dtype=np.float32),
        "sem_labels": np.zeros((meta["seq_len"], 1, 1), dtype=np.uint16),
        "cat_depth": np.zeros((meta["seq_len"], 2, 1, 1), dtype=np.float32),
        
        # Metadata - labels
        "label": np.int32(meta["label"]),
        "intention_prob_seq": np.asarray(meta["intention_prob_seq"], dtype=np.float32),
        "intent_label_track": np.int32(meta["intent_label_track"]),
        "intent_prob_track_max": np.float32(meta["intent_prob_track_max"]),
        "cross_label_track": np.int32(meta["cross_label_track"]),
        
        # Metadata - identifiers
        "pid": np.array([meta["pid"]], dtype=object),
        "set_id": np.array([meta["set_id"]], dtype=object),
        "video_name": np.array([meta["video_name"]], dtype=object),
        "track_id": np.array([meta["track_id"]], dtype=object),
        
        # Metadata - frames
        "img_paths": np.array(meta["img_paths"], dtype=object),
        "frame_idx": np.asarray(meta["frame_idx"], dtype=np.int32),
        
        # Metadata - window
        "t_start_ds": np.int32(meta["t_start_ds"]),
        "t_end_ds": np.int32(meta["t_end_ds"]),
        "t_start_full": np.int32(meta["t_start_full"]),
        "t_end_full": np.int32(meta["t_end_full"]),
        "event_frame": np.int32(meta["event_frame"]),
        "pseudo_event_frame": np.int32(meta["pseudo_event_frame"]),
        "pseudo_event_strategy": np.array([meta.get("pseudo_event_strategy", "na")], dtype=object),
        "first_cross_full": np.int32(meta["first_cross_full"]),
        
        # Metadata - timing
        "tte_sec_target": np.float32(meta["tte_sec_target"]),
        "tte_sec_actual": np.float32(meta["tte_sec_actual"]),
        
        # Metadata - params
        "data_fstride": np.int32(meta["data_fstride"]),
        "out_stride": np.int32(meta["out_stride"]),
        "raw_fps": np.float32(meta["raw_fps"]),
        "eff_fps": np.float32(meta["eff_fps"]),
        "seq_len": np.int32(meta["seq_len"]),
        "obs_sec": np.float32(meta["obs_sec"]),
        "split": np.array([meta["split"]], dtype=object),
        "mp4_png_offset": np.int32(meta.get("mp4_png_offset", 0)),
        "intention_prob_full": np.asarray(meta["intention_prob_full"], dtype=np.float32),

        
        # Processing flags
        "_stage_kinematics": np.int32(1),
        "_stage_local": np.int32(0),
        "_stage_scene": np.int32(0),
    }
    
    np.savez_compressed(path, **save_dict)


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser("Step 1: Kinematics (bbox, pose, speed) - Sliding Window")
    ap.add_argument("--pie_root", required=True)
    ap.add_argument("--iface_dir", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--splits", default="train,val,test")
    
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--seq_len", type=int, default=10)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--tte_min", type=int, default=30)
    ap.add_argument("--tte_max", type=int, default=60)
    ap.add_argument("--raw_fps", type=float, default=30.0)
    
    ap.add_argument("--pose_model", default="yolov8n-pose.pt")
    ap.add_argument("--pose_imgsz", type=int, default=512)
    ap.add_argument("--scale_crop", type=float, default=1.2)
    
    ap.add_argument("--no_pose", action="store_true")
    ap.add_argument("--no_speed", action="store_true")
    ap.add_argument("--keep_irrelevant", action="store_true")
    ap.add_argument("--intent_thresh", type=float, default=0.5)
    
    ap.add_argument("--neg_mode", default="hybrid")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--debug", type=int, default=0)
    
    args = ap.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    out_stride = max(1, args.stride)
    eff_fps = args.raw_fps / out_stride
    obs_sec = args.seq_len / eff_fps
    
    # Sliding window step size (in downsampled frame indices)
    step = max(1, int(args.seq_len * (1.0 - args.overlap)))
    
    print("=" * 70)
    print("Step 1: KINEMATICS (bbox, pose, speed) - SLIDING WINDOW")
    print("=" * 70)
    print(f"Splits: {splits}, stride={out_stride}, seq_len={args.seq_len}")
    print(f"eff_fps={eff_fps:.2f}, obs={obs_sec:.2f}s")
    print(f"overlap={args.overlap}, window step={step} (downsampled frames)")
    print(f"TTE range: [{args.tte_min}, {args.tte_max}] raw frames")
    print("=" * 70)
    
    # PIE interface
    if args.iface_dir not in sys.path:
        sys.path.insert(0, args.iface_dir)
    from pie_data import PIE
    imdb = PIE(data_path=args.pie_root)
    
    data_opts = {
        "fstride": 1,
        "data_split_type": "default",
        "seq_type": "crossing",
        "height_rng": [0, float("inf")],
        "squarify_ratio": 0,
        "min_track_size": 1,
        "random_params": {"ratios": None, "val_data": True, "regen_data": False},
        "kfold_params": {"num_folds": 5, "fold": 1},
    }
    
    clips_root = Path(args.pie_root) / "PIE_clips"
    vc = VideoCache(clips_root)
    
    try:
        for split in splits:
            print(f"\n[INFO] Loading {split}...")
            data = imdb.generate_data_trajectory_sequence(image_set=split, **data_opts)
            if data is None or "image" not in data:
                print(f"[WARN] No data for {split}")
                continue
            
            out_dir = Path(args.out_root) / split
            out_dir.mkdir(parents=True, exist_ok=True)
            
            images_all = data["image"]
            boxes_all = data["bbox"]
            cross_all = data["cross"]
            pid_all = data.get("pid")
            intent_all = data.get("intention_prob")
            speed_all = data.get("obd_speed")
            
            n = len(images_all)
            if args.limit > 0:
                n = min(n, args.limit)
            
            written, pos, neg = 0, 0, 0
            tracks_used = 0
            
            for idx in range(n):
                img_paths = images_all[idx]
                boxes = np.asarray(boxes_all[idx], dtype=np.float32) if boxes_all[idx] is not None else None
                cross = np.asarray(cross_all[idx]) if cross_all[idx] is not None else None
                
                if img_paths is None or boxes is None or len(img_paths) == 0:
                    continue
                
                T0 = min(len(img_paths), len(boxes), len(cross) if cross is not None else len(img_paths))
                pid = _unwrap_pid(pid_all[idx]) if pid_all and idx < len(pid_all) else f"unk_{idx}"
                
                intent = to_intent_array(intent_all[idx] if intent_all and idx < len(intent_all) else None, T0)
                
                speed_raw = None
                if not args.no_speed and speed_all and idx < len(speed_all) and speed_all[idx] is not None:
                    s = np.asarray(speed_all[idx], dtype=np.float32).reshape(-1)
                    if s.size > 0:
                        speed_raw = s
                
                T_full = T0
                if speed_raw is not None:
                    T_full = min(T_full, speed_raw.size)
                
                if T_full < args.seq_len:
                    continue
                
                # Trim
                img_paths = img_paths[:T_full]
                boxes = boxes[:T_full]
                cross = cross[:T_full] if cross is not None else np.zeros(T_full)
                intent = intent[:T_full]
                speed = speed_raw[:T_full] if speed_raw is not None else None
                
                # Labels
                first_cross = find_first_crossing_idx(cross)
                cross_label = 1 if first_cross is not None else 0
                int_label = intent_label(intent, args.intent_thresh)
                
                # Parse paths
                try:
                    set_id, video_name, _ = parse_img_path(img_paths[0])
                    frame_idx = np.array([parse_img_path(p)[2] for p in img_paths], dtype=np.int32)
                except Exception as e:
                    if args.debug > 1:
                        print(f"[WARN] parse_img_path failed pid={pid}: {e} | first_path={img_paths[0]}")
                    continue
                
                # Sync
                vc.infer_offset(set_id, video_name, img_paths, frame_idx, verbose=(args.debug > 0 and tracks_used < 3))
                mp4_off = vc.offset_cache.get(f"{set_id}/{video_name}", 0)
                
                # Stride (downsample)
                keep_idx = np.arange(0, T_full, out_stride, dtype=np.int32)
                if keep_idx.size < args.seq_len:
                    continue
                
                frame_idx_ds = frame_idx[keep_idx]
                
                # Determine reference (event) frame
                if cross_label == 1:
                    event_frame = int(frame_idx[first_cross])
                    pseudo_frame = -1
                    pseudo_strat = "na"
                    ref_frame = event_frame
                else:
                    event_frame = -1
                    pseudo_frame, pseudo_strat = choose_pseudo_event_frame(
                        frame_idx_full=frame_idx,
                        intent_full=intent,
                        keep_idx=keep_idx,
                        seq_len=args.seq_len,
                        raw_offset=args.tte_max,  # use tte_max as the offset for feasibility
                        raw_fps=args.raw_fps,
                        mode=args.neg_mode,
                    )
                    if pseudo_frame is None:
                        continue
                    ref_frame = pseudo_frame
                
                tracks_used += 1
                track_windows = 0
                
                # ============================================================
                # Sliding window loop over downsampled frames
                # ============================================================
                n_ds = keep_idx.size
                
                for w_start in range(0, n_ds - args.seq_len + 1, step):
                    w_end = w_start + args.seq_len - 1  # inclusive end in ds indices
                    
                    # Map back to full-track indices
                    t_start_full = int(keep_idx[w_start])
                    t_end_full = int(keep_idx[w_end])
                    
                    # Frame indices for this window
                    frame_idx_win = frame_idx_ds[w_start:w_end + 1]
                    
                    # Compute TTE in raw frames
                    tte = ref_frame - int(frame_idx_win[-1])
                    
                    # Filter by TTE range
                    if tte < args.tte_min or tte > args.tte_max:
                        continue
                    
                    # For positive samples, TTE must be > 0 (observation ends before crossing)
                    if cross_label == 1 and tte <= 0:
                        continue
                    
                    # Monotonicity check
                    if np.any(np.diff(frame_idx_win) <= 0):
                        continue
                    
                    # Irrelevant check
                    if not args.keep_irrelevant:
                        _, irr = _cross_masks(cross)
                        if irr[keep_idx[w_start:w_end + 1]].any():
                            continue
                    
                    # Slice boxes, intent, img_paths for this window
                    boxes_win = boxes[keep_idx][w_start:w_end + 1]
                    intent_win = intent[keep_idx][w_start:w_end + 1]
                    img_paths_win = [img_paths[k] for k in keep_idx[w_start:w_end + 1]]
                    
                    # TTE in seconds
                    tte_actual = float(tte) / args.raw_fps
                    tte_target = tte_actual  # actual TTE for this window
                    
                    # Read frames from video
                    try:
                        frames = [vc.read_frame(set_id, video_name, fi) for fi in frame_idx_win]
                        # One-time bbox-vs-frame sanity check (first few written samples)
                        if args.debug > 0 and written < 3:
                            h, w_img = frames[0].shape[:2]
                            mx = float(np.max(boxes_win[:, [0, 2]]))
                            my = float(np.max(boxes_win[:, [1, 3]]))
                            mnx = float(np.min(boxes_win[:, [0, 2]]))
                            mny = float(np.min(boxes_win[:, [1, 3]]))
                            print(f"[CHECK] frame WxH={w_img}x{h} | box x in [{mnx:.1f},{mx:.1f}] | box y in [{mny:.1f},{my:.1f}]")
                            if mx > w_img + 2 or my > h + 2 or mnx < -2 or mny < -2:
                                print("[WARN] Boxes look out-of-range for mp4 frame size -> PNG/MP4 scale mismatch likely.")
                    except (FileNotFoundError, RuntimeError, ValueError) as e:
                        if args.debug > 0:
                            print(f"[WARN] Frame read failed for {set_id}/{video_name} pid={pid} frames={frame_idx_win[:3]}... : {e}")
                        continue
                    
                    # Paper-aligned bbox: [x1,y1,x2,y2] normalized to image size (m x 4)
                    bbox_norm = []
                    for fr, (x1, y1, x2, y2) in zip(frames, boxes_win):
                        h, w_img = fr.shape[:2]

                        # ensure correct order
                        if x2 < x1:
                            x1, x2 = x2, x1
                        if y2 < y1:
                            y1, y2 = y2, y1

                        # clamp to image bounds
                        x1 = float(np.clip(x1, 0, w_img - 1))
                        x2 = float(np.clip(x2, 0, w_img - 1))
                        y1 = float(np.clip(y1, 0, h - 1))
                        y2 = float(np.clip(y2, 0, h - 1))

                        bbox_norm.append([x1 / w_img, y1 / h, x2 / w_img, y2 / h])

                    bbox_norm = np.asarray(bbox_norm, dtype=np.float32)
                    
                    # Pose
                    if not args.no_pose:
                        pose = extract_pose(frames, boxes_win, args.pose_model, args.scale_crop, args.pose_imgsz)
                    else:
                        pose = np.zeros((args.seq_len, 34), dtype=np.float32)
                    
                    # Speed
                    if speed is not None:
                        speed_win = speed[keep_idx][w_start:w_end + 1].reshape(-1, 1)
                    else:
                        speed_win = np.zeros((args.seq_len, 1), dtype=np.float32)
                    
                    feats = {
                        "bbox": bbox_norm,
                        "bbox_xyxy": boxes_win,
                        "pose": pose,
                        "speed": speed_win,
                    }
                    
                    meta = {
                        "label": cross_label,
                        "intention_prob_seq": intent_win.tolist(),
                        "intent_label_track": int_label,
                        "intent_prob_track_max": float(np.nanmax(intent)),
                        "cross_label_track": cross_label,
                        "pid": pid,
                        "set_id": set_id,
                        "video_name": video_name,
                        "track_id": f"{set_id}/{video_name}/{pid}",
                        "img_paths": img_paths_win,
                        "frame_idx": frame_idx_win.tolist(),
                        "t_start_ds": w_start,
                        "t_end_ds": w_end,
                        "t_start_full": t_start_full,
                        "t_end_full": t_end_full,
                        "event_frame": event_frame,
                        "pseudo_event_frame": pseudo_frame,
                        "pseudo_event_strategy": pseudo_strat,
                        "first_cross_full": first_cross if first_cross is not None else -1,
                        "tte_sec_target": tte_target,
                        "tte_sec_actual": tte_actual,
                        "data_fstride": 1,
                        "out_stride": out_stride,
                        "raw_fps": args.raw_fps,
                        "eff_fps": eff_fps,
                        "seq_len": args.seq_len,
                        "obs_sec": obs_sec,
                        "split": split,
                        "mp4_png_offset": mp4_off,
                        "intention_prob_full": intent.tolist(),
                    }
                    
                    out_path = out_dir / f"seq_{written:06d}.npz"
                    save_npz(out_path, feats, meta)
                    
                    written += 1
                    track_windows += 1
                    if cross_label == 1:
                        pos += 1
                    else:
                        neg += 1
                    
                    if args.debug > 0 and written <= args.debug:
                        print(f"  [{written}] {pid} label={cross_label} tte={tte_actual:.2f}s "
                              f"win=[{w_start}:{w_end}] frames=[{frame_idx_win[0]}..{frame_idx_win[-1]}]")
                
                if args.debug > 1 and track_windows > 0:
                    print(f"  track {pid}: {track_windows} windows generated")
            
            print(f"[OK] {split}: {written} samples (pos={pos}, neg={neg}) from {tracks_used} tracks")
    
    finally:
        vc.release_all()
    
    print("\n[DONE] Step 1 complete!")


if __name__ == "__main__":
    main()