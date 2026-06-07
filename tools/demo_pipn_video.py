#!/usr/bin/env python3
"""
demo_pipn_video.py – PIPN thesis demo (FINAL)
  * Detection + tracking : YOLOv8-pose on the FULL frame (classes=[0]) -> stable boxes
  * Model bbox feature    : the YOLO box (robustness-verified equivalent to mask boxes)
  * Pose feature          : training-matched 1.2x crop @ imgsz=512 (kept consistent)
  * Scene features        : Mask2Former -> sem_labels + cat_depth @ out_h x out_w (unchanged)
  * Tracker               : per-track Kalman filter (ego-motion compensation)
  * Labels                : anticipatory intent  -> "WILL CROSS" / "NO CROSS INTENT"
  * Optional latch        : hold a positive prediction through the crossing (--latch_frames)
  * Display               : model inputs stay at 672x384, output can be larger & prettier
"""

import argparse, os, sys, json, re
from pathlib import Path
from collections import deque
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
from torchvision import models, transforms
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

# ----------------------------------------------------------------------
# 1. Paths
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "preprocess" / "pie"))

try:
    import prep_scene as prep
except ImportError:
    import prep_scene_mask2former_cityscapes as prep

from models.pipnet_alpha_v4_wide_compat import PIPNetAlphaV4Final

# ----------------------------------------------------------------------
# 2. Palette & COCO edges
# ----------------------------------------------------------------------
FIXED_COLORS = {
    "bg": (0, 0, 0), "person": (0, 255, 0), "rider": (0, 220, 120),
    "car": (0, 0, 255), "truck": (0, 0, 200), "bus": (0, 80, 220),
    "train": (0, 100, 180), "motorcycle": (255, 80, 0), "bicycle": (255, 120, 0),
    "traffic light": (200, 200, 0), "traffic sign": (200, 200, 100),
    "road": (60, 60, 60), "sidewalk": (120, 80, 80), "building": (100, 100, 140),
    "wall": (80, 80, 100), "fence": (60, 60, 100), "pole": (100, 100, 120),
    "vegetation": (60, 160, 60), "terrain": (80, 140, 60), "sky": (255, 180, 80),
}

COCO_EDGES = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 1), (0, 2), (1, 3), (2, 4),
]

# Nice UI colors (BGR)
C_CROSS      = (60, 60, 235)     # red  -> will cross
C_NOCROSS    = (90, 210, 90)     # green-> no intent
C_UNCERTAIN  = (0, 190, 255)     # amber-> uncertain
C_COLLECT    = (170, 170, 170)   # gray -> collecting
C_WHITE      = (255, 255, 255)
C_PANEL      = (24, 24, 28)

# ----------------------------------------------------------------------
# 3. Feature extractors (VGG19, YOLO-pose, RAFT)
# ----------------------------------------------------------------------
_vgg = None; _vgg_transform = None; _vgg_device = None

def init_vgg():
    global _vgg, _vgg_transform, _vgg_device
    if _vgg is not None: return
    _vgg_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    _vgg = vgg.features.to(_vgg_device).eval()
    _vgg_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

@torch.inference_mode()
def extract_vgg_single(crop_rgb):
    init_vgg()
    t = _vgg_transform(crop_rgb).unsqueeze(0).to(_vgg_device)
    return _vgg(t).squeeze(0).cpu().numpy().astype(np.float32)

_pose_model = None; _pose_device = None; _pose_model_path = None

def init_pose(model_path="yolov8n-pose.pt"):
    global _pose_model, _pose_device, _pose_model_path
    if _pose_model is not None and _pose_model_path == model_path: return
    from ultralytics import YOLO
    _pose_model = YOLO(model_path)
    _pose_model_path = model_path
    _pose_device = 0 if torch.cuda.is_available() else "cpu"

@torch.inference_mode()
def yolo_detect_persons(frame_bgr, imgsz=960, conf=0.25, min_area=2000, model_path="yolov8n-pose.pt"):
    """Full-frame YOLO person detection. Returns list of (x1,y1,x2,y2)."""
    init_pose(model_path)
    res = _pose_model.predict(frame_bgr, imgsz=imgsz, classes=[0], conf=conf,
                              verbose=False, device=_pose_device)
    out = []
    if len(res) and res[0].boxes is not None and res[0].boxes.xyxy is not None:
        for b in res[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = [float(v) for v in b]
            if (x2 - x1) * (y2 - y1) < min_area:
                continue
            out.append((x1, y1, x2, y2))
    return out

def extract_pose_crop(crop_bgr, x1_e, y1_e, full_w, full_h, imgsz=512, model_path="yolov8n-pose.pt"):
    """Training-matched pose: YOLO on the 1.2x crop, keypoints offset to full frame & normalized."""
    init_pose(model_path)
    res = _pose_model.predict(crop_bgr, imgsz=imgsz, verbose=False, device=_pose_device)
    if len(res) and res[0].keypoints is not None and res[0].keypoints.xy is not None \
            and res[0].keypoints.xy.shape[0] > 0:
        xy = res[0].keypoints.xy[0].cpu().numpy().astype(np.float32)
        xy[:, 0] = (xy[:, 0] + x1_e) / max(full_w, 1)
        xy[:, 1] = (xy[:, 1] + y1_e) / max(full_h, 1)
        return xy.reshape(-1)
    return np.zeros(34, dtype=np.float32)

_raft = None; _raft_device = None

def init_raft():
    global _raft, _raft_device
    if _raft is not None: return
    _raft_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _raft = raft_large(weights=Raft_Large_Weights.C_T_SKHT_K_V2).to(_raft_device).eval()

def preprocess_for_raft(img_bgr, size=(224, 224)):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return t * 2.0 - 1.0

@torch.inference_mode()
def extract_flow_pair(crop1_bgr, crop2_bgr):
    init_raft()
    t1 = preprocess_for_raft(crop1_bgr).unsqueeze(0).to(_raft_device)
    t2 = preprocess_for_raft(crop2_bgr).unsqueeze(0).to(_raft_device)
    flows = _raft(t1, t2)
    return flows[-1].squeeze(0).cpu().numpy().astype(np.float32)

# ----------------------------------------------------------------------
# 4. Kalman Tracker (per-track filter)
# ----------------------------------------------------------------------
class KalmanTracker:
    def __init__(self, iou_threshold=0.2, max_lost=60, max_area_ratio=2.0):
        self.iou_thr = float(iou_threshold)
        self.max_lost = int(max_lost)
        self.max_area_ratio = float(max_area_ratio)  # reject matches whose box areas differ more than this
        self.next_id = 1
        self.tracks = {}
        self.measurement_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.transition_matrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1],
                                           [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.process_noise = np.eye(4, dtype=np.float32) * 0.03

    def _new_kalman(self, cx, cy):
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = self.measurement_matrix.copy()
        kf.transitionMatrix = self.transition_matrix.copy()
        kf.processNoiseCov = self.process_noise.copy()
        kf.statePost = np.array([[cx], [cy], [0], [0]], np.float32)
        return kf

    @staticmethod
    def _center(b):
        return np.array([[(b[0] + b[2]) / 2.0], [(b[1] + b[3]) / 2.0]], dtype=np.float32)

    @staticmethod
    def _center_to_bbox(c, w, h):
        cx, cy = c[0, 0], c[1, 0]
        return [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]

    def update(self, det_bboxes):
        for tid, trk in self.tracks.items():
            pred = trk['kalman'].predict()
            x1, y1, x2, y2 = trk['bbox']
            trk['pred_bbox'] = self._center_to_bbox(pred, x2 - x1, y2 - y1)
            trk['lost'] += 1

        matched_t, matched_d, assigns = set(), set(), []
        for tid, trk in self.tracks.items():
            if trk['lost'] > self.max_lost: continue
            trk_area = max(1.0, (trk['bbox'][2] - trk['bbox'][0]) * (trk['bbox'][3] - trk['bbox'][1]))
            best_iou, best_d = 0.0, -1
            for d, db in enumerate(det_bboxes):
                if d in matched_d: continue
                # size gate: areas must be within max_area_ratio of each other
                det_area = max(1.0, (db[2] - db[0]) * (db[3] - db[1]))
                ratio = max(trk_area / det_area, det_area / trk_area)
                if ratio > self.max_area_ratio: continue
                i = self._box_iou(trk['pred_bbox'], db)
                if i > best_iou: best_iou, best_d = i, d
            if best_iou >= self.iou_thr and best_d != -1:
                assigns.append((tid, best_d, best_iou))

        assigns.sort(key=lambda x: -x[2])
        result = []
        for tid, d, _ in assigns:
            if tid in matched_t or d in matched_d: continue
            matched_t.add(tid); matched_d.add(d)
            db = det_bboxes[d]
            self.tracks[tid]['kalman'].correct(self._center(db))
            self.tracks[tid]['bbox'] = db
            self.tracks[tid]['lost'] = 0
            result.append((tid, db))

        for d, db in enumerate(det_bboxes):
            if d in matched_d: continue
            tid = self.next_id; self.next_id += 1
            c = self._center(db)
            self.tracks[tid] = {'kalman': self._new_kalman(c[0, 0], c[1, 0]),
                                'bbox': db, 'lost': 0, 'pred_bbox': db}
            result.append((tid, db))

        for tid in list(self.tracks.keys()):
            if self.tracks[tid]['lost'] > self.max_lost:
                del self.tracks[tid]
        return result

    @staticmethod
    def _box_iou(a, b):
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        ua = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
        ub = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
        u = ua + ub - inter
        return inter / u if u > 0 else 0.0

# ----------------------------------------------------------------------
# 5. Smoothed box (visualization only)
# ----------------------------------------------------------------------
class SmoothedBox:
    def __init__(self, alpha=0.7):
        self.alpha = alpha; self.bbox = None
    def update(self, bbox):
        b = np.array(bbox, dtype=np.float32)
        self.bbox = b if self.bbox is None else self.alpha * self.bbox + (1 - self.alpha) * b
        return self.bbox.copy()

# ----------------------------------------------------------------------
# 6. Pedestrian buffer
# ----------------------------------------------------------------------
class PedestrianBuffer:
    def __init__(self, pid, seq_len=10, stride=1):
        self.pid = pid; self.seq_len = seq_len; self.stride = stride
        self.window_len = (seq_len - 1) * stride + 1
        self.frames = deque(maxlen=self.window_len)
        self.ready = False
    def add(self, frame_idx, bbox_xyxy, crop_bgr, pose, vgg_feat, flow, sem_labels, cat_depth):
        self.frames.append({
            "frame_idx": frame_idx,
            "bbox_xyxy": np.array(bbox_xyxy, dtype=np.float32),
            "crop_bgr": crop_bgr,
            "pose": pose.astype(np.float32),
            "vgg_feat": vgg_feat.astype(np.float32),
            "flow": flow if flow is not None else np.zeros((2, 224, 224), dtype=np.float32),
            "sem_labels": sem_labels.astype(np.int64),
            "cat_depth": cat_depth.astype(np.float32),
        })
        self.ready = len(self.frames) == self.window_len
    def get_sequence(self):
        if not self.ready: raise RuntimeError("Buffer not full")
        idx = [i * self.stride for i in range(self.seq_len)]
        s = [self.frames[i] for i in idx]
        return {
            "bbox": np.stack([e["bbox_xyxy"] for e in s]),
            "pose": np.stack([e["pose"] for e in s]),
            "vgg": np.stack([e["vgg_feat"] for e in s]),
            "flow": np.stack([e["flow"] for e in s]),
            "sem_labels": np.stack([e["sem_labels"] for e in s]),
            "cat_depth": np.stack([e["cat_depth"] for e in s]),
            "frame_idx": [e["frame_idx"] for e in s],
        }
    def pop_oldest(self):
        self.frames.popleft()
        self.ready = len(self.frames) == self.window_len

# ----------------------------------------------------------------------
# 7. Crop expansion
# ----------------------------------------------------------------------
def expand_box_xyxy(x1, y1, x2, y2, w, h, scale=1.2):
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw, bh = (x2 - x1) * scale, (y2 - y1) * scale
    nx1 = max(0, int(cx - bw / 2)); ny1 = max(0, int(cy - bh / 2))
    nx2 = min(w - 1, int(cx + bw / 2)); ny2 = min(h - 1, int(cy + bh / 2))
    if nx2 <= nx1 or ny2 <= ny1:
        return 0, 0, w - 1, h - 1
    return nx1, ny1, nx2, ny2

# ----------------------------------------------------------------------
# 8. Rendering helpers (prettier)
# ----------------------------------------------------------------------
def classify_intent(prob, threshold=0.5, margin=0.05):
    if prob is None: return "COLLECTING", C_COLLECT
    p = float(np.clip(prob, 0.0, 1.0))
    margin = max(0.0, float(margin))
    if abs(p - threshold) <= margin: return "UNCERTAIN", C_UNCERTAIN
    if p >= threshold: return "WILL CROSS", C_CROSS
    return "NO CROSS INTENT", C_NOCROSS

def scale_bbox_for_vis(bbox, ow, oh, dw, dh):
    x1, y1, x2, y2 = bbox
    sx, sy = dw / ow, dh / oh
    return (x1 * sx, y1 * sy, x2 * sx, y2 * sy)

def _alpha_rect(img, p1, p2, color, alpha):
    x1, y1 = p1; x2, y2 = p2
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(img.shape[1], x2); y2 = min(img.shape[0], y2)
    if x2 <= x1 or y2 <= y1: return
    roi = img[y1:y2, x1:x2].astype(np.float32)
    ov = np.empty_like(roi); ov[:] = color
    img[y1:y2, x1:x2] = (roi * (1 - alpha) + ov * alpha).astype(np.uint8)

def draw_prediction(img, bbox, prob, ped_id, threshold=0.5, margin=0.05,
                    seq_len=10, warming=None, latched=False):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    label, color = classify_intent(prob, threshold, margin)

    # box (rounded-ish: just a clean 2px rect + subtle inner)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    # label chip above the box
    if prob is None:
        n = warming if warming else 0
        txt = f"#{ped_id}  {label} {n}/{seq_len}"
        pct = None
    else:
        pct = float(np.clip(prob, 0.0, 1.0))
        tag = "  [LATCHED]" if latched else ""
        txt = f"#{ped_id}  {label}  {pct*100:.0f}%{tag}"

    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    chip_h = th + 12
    cy1 = max(0, y1 - chip_h - 6)
    cx2 = min(img.shape[1], x1 + tw + 16)
    _alpha_rect(img, (x1, cy1), (cx2, cy1 + chip_h), C_PANEL, 0.65)
    cv2.rectangle(img, (x1, cy1), (cx2, cy1 + chip_h), color, 1, cv2.LINE_AA)
    cv2.putText(img, txt, (x1 + 8, cy1 + chip_h - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_WHITE, 1, cv2.LINE_AA)

    # probability bar under the chip
    if pct is not None:
        bar_y = cy1 + chip_h + 3
        bar_w = max(60, x2 - x1)
        bx2 = min(img.shape[1], x1 + bar_w)
        _alpha_rect(img, (x1, bar_y), (bx2, bar_y + 6), (60, 60, 60), 0.8)
        fill = int((bx2 - x1) * pct)
        cv2.rectangle(img, (x1, bar_y), (x1 + fill, bar_y + 6), color, -1, cv2.LINE_AA)

def draw_pose(img, pose_flat, dw, dh, min_valid=1e-4):
    if pose_flat is None: return
    arr = np.asarray(pose_flat, dtype=np.float32).reshape(-1)
    if arr.size != 34: return
    pts = arr.reshape(17, 2)
    h, w = img.shape[:2]
    P, V = [], []
    for x, y in pts:
        ok = bool(x > min_valid and y > min_valid and np.isfinite(x) and np.isfinite(y))
        px = max(0, min(w - 1, int(round(x * dw))))
        py = max(0, min(h - 1, int(round(y * dh))))
        P.append((px, py)); V.append(ok)
    for a, b in COCO_EDGES:
        if V[a] and V[b]:
            cv2.line(img, P[a], P[b], (255, 255, 255), 2, cv2.LINE_AA)
    for i, p in enumerate(P):
        if V[i]:
            cv2.circle(img, p, 4, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img, p, 3, (0, 230, 255), -1, cv2.LINE_AA)

def draw_header(img, frame_idx, total, seq_len, max_peds, use_speed):
    h, w = img.shape[:2]
    s = max(0.6, min(1.1, h / 540.0))
    hh = int(34 * s)
    _alpha_rect(img, (0, 0), (w, hh), C_PANEL, 0.7)
    cv2.putText(img, "PIP-Net  -  Pedestrian Crossing-Intention Prediction",
                (12, int(23 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, C_WHITE, 1, cv2.LINE_AA)
    meta = f"frame {frame_idx}/{total}  |  obs={seq_len}  |  max ped={max_peds}  |  speed={'live' if use_speed else 'const'}"
    (mw, _), _ = cv2.getTextSize(meta, cv2.FONT_HERSHEY_SIMPLEX, 0.45 * s, 1)
    cv2.putText(img, meta, (w - mw - 12, int(23 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45 * s, (200, 200, 200), 1, cv2.LINE_AA)

def draw_legend(img, threshold, margin):
    h, w = img.shape[:2]
    s = max(0.6, min(1.1, h / 540.0))
    lines = [
        (f"Red  = WILL CROSS  (intent >{threshold+margin:.2f}, next ~1-2s)", C_CROSS),
        (f"Green= NO CROSS INTENT  (<{threshold-margin:.2f})", C_NOCROSS),
        ("Amber= uncertain", C_UNCERTAIN),
        ("Gray = collecting frames", C_COLLECT),
    ]
    lh = int(20 * s)
    box_h = lh * len(lines) + 14
    box_w = int(430 * s)
    y0 = h - box_h - 6
    _alpha_rect(img, (6, y0), (6 + box_w, y0 + box_h), C_PANEL, 0.7)
    y = y0 + lh
    for txt, col in lines:
        cv2.circle(img, (18, y - int(4 * s)), int(5 * s), col, -1, cv2.LINE_AA)
        cv2.putText(img, txt, (32, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42 * s, C_WHITE, 1, cv2.LINE_AA)
        y += lh

def draw_depth_overlay(img, cat_depth, alpha=0.5):
    if cat_depth is None or cat_depth.ndim != 3 or cat_depth.shape[0] < 2: return img
    combined = np.maximum(cat_depth[0], cat_depth[1]).astype(np.float32)
    if combined.max() <= 0: return img
    h, w = img.shape[:2]
    if combined.shape != (h, w): combined = cv2.resize(combined, (w, h))
    scaled = np.clip(combined * 1.5, 0, 1)
    heat = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_JET)
    mask = cv2.GaussianBlur((combined > 0).astype(np.float32), (5, 5), 0)[..., None]
    a = float(np.clip(alpha, 0, 1))
    return np.clip(img.astype(np.float32) * (1 - a * mask) + heat.astype(np.float32) * (a * mask), 0, 255).astype(np.uint8)

# ----------------------------------------------------------------------
# 9. Data helpers
# ----------------------------------------------------------------------
def cityscapes_to_pip(sem_city):
    s = sem_city.astype(np.int32).copy()
    s[(s < 0) | (s >= len(prep.CLASS_MAP))] = 0
    return prep.CLASS_MAP[s].astype(np.int64)

def compute_live_cat_depth(frame_original, frame_count, sem_city, inst, inst_lbl, prev_state, out_size, args):
    pf, pi = prev_state.get("frame"), prev_state.get("idx")
    ps, pin, pil = prev_state.get("sem_city"), prev_state.get("inst"), prev_state.get("inst_lbl")
    if pf is not None:
        depth_frames, depth_idxs = [pf, frame_original], [pi, frame_count]
    else:
        depth_frames, depth_idxs = [frame_original, frame_original], [frame_count, frame_count]
    depth_seq = prep.estimate_depth(depth_frames, depth_idxs, out_size, args.depth_weights, max_gap=args.depth_max_gap)
    if pin is not None and ps is not None and pil is not None:
        inst_seq = np.stack([pin, inst], 0); inst_lbl_seq = np.stack([pil, inst_lbl], 0)
        sem_city_seq = np.stack([ps, sem_city], 0); depth_for_cat = depth_seq
    else:
        inst_seq = inst[np.newaxis]; inst_lbl_seq = inst_lbl[np.newaxis]
        sem_city_seq = sem_city[np.newaxis]; depth_for_cat = depth_seq[-1:]
    trk = prep.track_instances_iou_window(inst_seq, inst_lbl_seq, iou_thr=args.track_iou_thr,
                                          min_area=args.track_min_area, match_same_group=True)
    cd = prep.compute_cat_depth(depth_for_cat, sem_city_seq, trk, inst_lbl_seq,
                                min_area=args.cat_depth_min_area, morph_kernel_size=args.cat_depth_morph_kernel,
                                prefer_instances_for_ped=True)
    return cd[-1].astype(np.float32)

PIE_IMG_RE = re.compile(r"(?:.*/)?images/(set\d{2})/(video_\d{4})/(\d{5})\.png$")
def parse_pie_img_path(path):
    m = PIE_IMG_RE.match(str(path).replace("\\", "/"))
    if not m: raise ValueError(f"Cannot parse {path}")
    return m.group(1), m.group(2), int(m.group(3))

def parse_video_id(video_path):
    p = Path(video_path); return p.parent.name, p.stem

def _unwrap_scalar(v):
    if v is None: return 0.0
    if isinstance(v, (int, float, np.number)): return float(v)
    while isinstance(v, (list, tuple, np.ndarray)):
        if len(v) == 0: return 0.0
        v = v[0]
        if isinstance(v, (int, float, np.number)): return float(v)
    try: return float(v)
    except (ValueError, TypeError): return 0.0

def load_pie_frame_speed(pie_root, iface_dir, video_path):
    if iface_dir not in sys.path: sys.path.insert(0, iface_dir)
    from pie_data import PIE
    target_set, target_video = parse_video_id(video_path)
    imdb = PIE(data_path=pie_root)
    data_opts = {"fstride": 1, "data_split_type": "default", "seq_type": "crossing",
                 "height_rng": [0, float("inf")], "squarify_ratio": 0, "min_track_size": 1,
                 "random_params": {"ratios": None, "val_data": True, "regen_data": False},
                 "kfold_params": {"num_folds": 5, "fold": 1}}
    speed_by_frame = {}
    for split in ["train", "val", "test"]:
        if speed_by_frame: break
        data = imdb.generate_data_trajectory_sequence(image_set=split, **data_opts)
        if not data or "image" not in data: continue
        images_all = data["image"]; speed_all = data.get("obd_speed")
        if speed_all is None: continue
        for ti in range(len(images_all)):
            paths = images_all[ti]; sp = speed_all[ti]
            if not paths or sp is None or len(paths) == 0: continue
            try: sid, vname, _ = parse_pie_img_path(paths[0])
            except: continue
            if sid != target_set or vname != target_video: continue
            T = min(len(paths), len(sp))
            for t in range(T):
                try: _, _, fi = parse_pie_img_path(paths[t])
                except: continue
                if int(fi) not in speed_by_frame:
                    speed_by_frame[int(fi)] = _unwrap_scalar(sp[t])
    return speed_by_frame

# ----------------------------------------------------------------------
# 10. Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--m2f_model", default="facebook/mask2former-swin-large-cityscapes-panoptic")
    ap.add_argument("--max_frames", type=int, default=-1)
    ap.add_argument("--start_frame", type=int, default=0)
    ap.add_argument("--max_pedestrians", type=int, default=6)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--action_margin", type=float, default=0.05)
    ap.add_argument("--debug_action", action="store_true")
    ap.add_argument("--seq_len", type=int, default=10)
    ap.add_argument("--process_stride", type=int, default=2)
    ap.add_argument("--seq_sample_stride", type=int, default=1)
    # model input resolution
    ap.add_argument("--out_w", type=int, default=672)
    ap.add_argument("--out_h", type=int, default=384)
    # display resolution
    ap.add_argument("--display_w", type=int, default=1344)
    ap.add_argument("--display_h", type=int, default=768)
    # YOLO detection
    ap.add_argument("--yolo_imgsz", type=int, default=960)
    ap.add_argument("--yolo_conf", type=float, default=0.25)
    ap.add_argument("--min_box_area", type=int, default=2000)
    ap.add_argument("--no_depth", action="store_true")
    ap.add_argument("--show_depth", action="store_true")
    ap.add_argument("--depth_alpha", type=float, default=0.5)
    ap.add_argument("--seg_alpha", type=float, default=0.6)
    ap.add_argument("--pose_model", default="yolov8n-pose.pt")
    ap.add_argument("--pose_imgsz", type=int, default=512)
    ap.add_argument("--show_pose", action="store_true")
    ap.add_argument("--skip_pipn", action="store_true")
    ap.add_argument("--depth_weights", default="/workspace/models/manydepth/KITTI_HR")
    ap.add_argument("--depth_max_gap", type=int, default=3)
    ap.add_argument("--track_iou_thr", type=float, default=0.30)
    ap.add_argument("--track_min_area", type=int, default=64)
    ap.add_argument("--cat_depth_min_area", type=int, default=50)
    ap.add_argument("--cat_depth_morph_kernel", type=int, default=3)
    ap.add_argument("--tracker_iou", type=float, default=0.2)
    ap.add_argument("--tracker_max_lost", type=int, default=60)
    ap.add_argument("--tracker_area_ratio", type=float, default=2.0,
                    help="Reject a track<->detection match if box areas differ by more than this factor")
    ap.add_argument("--smooth_alpha", type=float, default=0.5)
    ap.add_argument("--latch_frames", type=int, default=0,
                    help="Hold a positive prediction red for N processed frames after it fires (0=off)")
    ap.add_argument("--speed_stats", default="/workspace/project/data/pie_speed_stats_splits.json")
    ap.add_argument("--motion_stats", default="/workspace/project/data/pie_motion_stats_splits.json")
    ap.add_argument("--no_speed", action="store_true")
    ap.add_argument("--pie_root", default="/data/PIE")
    ap.add_argument("--iface_dir", default="/data/pie_interface")
    ap.add_argument("--suppress_ego", action="store_true")
    ap.add_argument("--ego_bottom_frac", type=float, default=0.88)
    ap.add_argument("--debug_buffer", action="store_true")
    args = ap.parse_args()

    eff = args.process_stride * args.seq_sample_stride
    if eff != 2:
        print(f"[WARN] effective temporal stride = {eff} (proc {args.process_stride} x sample {args.seq_sample_stride}); "
              f"model trained with out_stride=2.")
    if args.seq_sample_stride > 1:
        print("[WARN] seq_sample_stride>1: flow computed between adjacent stored frames only.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    out_w, out_h = int(args.out_w), int(args.out_h)
    cv_size, m2f_size = (out_w, out_h), (out_h, out_w)
    disp_w, disp_h = int(args.display_w), int(args.display_h)
    disp_size = (disp_w, disp_h)

    out_dir = os.path.dirname(args.out)
    if out_dir: os.makedirs(out_dir, exist_ok=True)

    print("[INFO] Loading Mask2Former (scene features only)...")
    m2f_device = "cuda" if torch.cuda.is_available() else "cpu"
    prep.init_cityscapes_meta_from_m2f(args.m2f_model, device=m2f_device)
    m2f = prep.get_m2f(args.m2f_model, device=m2f_device)
    pip_classes = prep.PIP_CLASSES
    palette = np.zeros((len(pip_classes), 3), dtype=np.uint8)
    for i, name in enumerate(pip_classes): palette[i] = FIXED_COLORS.get(name, (128, 128, 128))
    def sem_to_rgb(sem): return palette[sem.clip(0, len(pip_classes) - 1)]

    model = None
    if not args.skip_pipn:
        print("[INFO] Loading PIPN model...")
        model = PIPNetAlphaV4Final(dropout_p=0.2, local_dropout_p=0.1, global_dropout_p=0.4).to(device)
        ckpt = torch.load(args.ckpt, map_location=device)
        state = ckpt.get("model", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state, strict=True)
        model.eval()

    print("[INFO] Initializing YOLO detector...")
    init_pose(args.pose_model)

    speed_stats = None
    if os.path.exists(args.speed_stats):
        with open(args.speed_stats) as f: speed_stats = json.load(f)
    else:
        print("[WARN] speed_stats not found - speed will be constant/zero!")
    motion_stats = None
    if os.path.exists(args.motion_stats):
        with open(args.motion_stats) as f: motion_stats = json.load(f)
    else:
        print("[WARN] motion_stats not found - flow normalization falls back to /20.0!")

    frame_speed = None
    if not args.no_speed:
        print("[INFO] Loading live ego-vehicle speed...")
        frame_speed = load_pie_frame_speed(args.pie_root, args.iface_dir, args.video)
        print(f"[INFO] Speed loaded for {len(frame_speed)} frames")
    use_live_speed = frame_speed is not None

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened(): raise RuntimeError(f"Cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    video_orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Original video resolution: {video_orig_w}x{video_orig_h}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.start_frame))
        print(f"[INFO] Starting from frame {args.start_frame}")
    if args.max_frames > 0 and total_frames > 0:
        total_disp = min(total_frames, int(args.start_frame) + int(args.max_frames) * args.process_stride)
    else:
        total_disp = total_frames

    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps / args.process_stride, disp_size)
    if not writer.isOpened(): raise RuntimeError(f"Cannot open output {args.out}")

    tracker = KalmanTracker(iou_threshold=args.tracker_iou, max_lost=args.tracker_max_lost,
                            max_area_ratio=args.tracker_area_ratio)
    ped_buffers: Dict[int, PedestrianBuffer] = {}
    last_pred: Dict[int, float] = {}
    latch: Dict[int, int] = {}
    smoothed: Dict[int, SmoothedBox] = {}
    prev_state = {"frame": None, "idx": None, "sem_city": None, "inst": None, "inst_lbl": None}
    ego_fill = prep.PIP_ID.get("road", 0)

    frame_count = int(args.start_frame)
    written = 0
    import time as _time
    _t0 = _time.time()

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret: break
            if frame_count % args.process_stride != 0:
                frame_count += 1; continue
            if args.max_frames > 0 and written >= args.max_frames: break

            if written > 0 and written % 50 == 0:
                el = _time.time() - _t0
                rate = el / written
                remain = (args.max_frames - written) if args.max_frames > 0 else 0
                eta = rate * remain
                print(f"[PROGRESS] {written} frames done | {rate:.2f}s/frame | "
                      f"elapsed {el/60:.1f}min | ETA {eta/60:.1f}min   ")
            print(f"Frame {frame_count}/{total_disp}", end="\r")
            orig_h, orig_w = frame_bgr.shape[:2]

            # ---- Scene features from Mask2Former (sem_labels + cat_depth ONLY) ----
            sem_city, inst, inst_lbl = m2f.run_single(frame_bgr, m2f_size)
            sem_pip = cityscapes_to_pip(sem_city)
            frame_vis = cv2.resize(frame_bgr, cv_size)

            if args.no_depth:
                cat_depth = np.zeros((2, out_h, out_w), dtype=np.float32)
            else:
                cat_depth = compute_live_cat_depth(frame_bgr, frame_count, sem_city, inst, inst_lbl,
                                                   prev_state, m2f_size, args)

            if args.suppress_ego:
                sem_pip = np.expand_dims(sem_pip, 0)
                inst_e = np.expand_dims(inst, 0); inst_lbl_e = np.expand_dims(inst_lbl, 0)
                cd_e = None if args.no_depth else np.expand_dims(cat_depth, 0)
                prep.suppress_ego_car_dashboard(sem_pip=sem_pip, inst_vip=inst_e, inst_lbl=inst_lbl_e,
                                                depth_norm=cd_e, bottom_frac=args.ego_bottom_frac,
                                                fill_label=ego_fill, zero_depth=True)
                sem_pip = sem_pip[0]; inst = inst_e[0]; inst_lbl = inst_lbl_e[0]
                if cd_e is not None: cat_depth = cd_e[0]

            prev_state = {"frame": frame_bgr, "idx": frame_count, "sem_city": sem_city,
                          "inst": inst, "inst_lbl": inst_lbl}

            # ---- Detection: YOLO on full frame (stable boxes, original coords) ----
            detections = yolo_detect_persons(frame_bgr, imgsz=args.yolo_imgsz, conf=args.yolo_conf,
                                             min_area=args.min_box_area, model_path=args.pose_model)
            if len(detections) > args.max_pedestrians:
                detections.sort(key=lambda d: (d[2] - d[0]) * (d[3] - d[1]), reverse=True)
                detections = detections[:args.max_pedestrians]

            tracked = tracker.update(detections)

            # ---- Memory hygiene: drop state for tracks the tracker has deleted ----
            alive = set(tracker.tracks.keys())
            for d in (ped_buffers, last_pred, latch, smoothed):
                for dead in [k for k in d.keys() if k not in alive]:
                    del d[dead]

            # ---- Update buffers (YOLO box for model bbox; training-matched crop pose) ----
            for tid, (x1, y1, x2, y2) in tracked:
                if tid not in smoothed: smoothed[tid] = SmoothedBox(alpha=0.7)
                smoothed[tid].update((x1, y1, x2, y2))

                x1_e, y1_e, x2_e, y2_e = expand_box_xyxy(x1, y1, x2, y2, orig_w, orig_h, scale=1.2)
                crop_bgr = frame_bgr[y1_e:y2_e + 1, x1_e:x2_e + 1]
                if crop_bgr.size == 0: continue

                vgg_feat = extract_vgg_single(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
                pose = extract_pose_crop(crop_bgr, x1_e, y1_e, orig_w, orig_h,
                                         imgsz=args.pose_imgsz, model_path=args.pose_model)

                flow = None
                if tid in ped_buffers and len(ped_buffers[tid].frames) > 0:
                    flow = extract_flow_pair(ped_buffers[tid].frames[-1]["crop_bgr"], crop_bgr)

                if tid not in ped_buffers:
                    ped_buffers[tid] = PedestrianBuffer(tid, args.seq_len, stride=args.seq_sample_stride)
                ped_buffers[tid].add(frame_count, (x1, y1, x2, y2), crop_bgr, pose, vgg_feat,
                                     flow, sem_pip, cat_depth)

            # ---- PIPN prediction ----
            if model is not None:
                for pid, buf in list(ped_buffers.items()):
                    if not buf.ready: continue
                    seq = buf.get_sequence()
                    bbox_norm = seq["bbox"].copy()
                    bbox_norm[:, 0] /= video_orig_w; bbox_norm[:, 1] /= video_orig_h
                    bbox_norm[:, 2] /= video_orig_w; bbox_norm[:, 3] /= video_orig_h

                    if use_live_speed:
                        sp = np.array([frame_speed.get(fi, 0.0) for fi in seq["frame_idx"]],
                                      dtype=np.float32).reshape(-1, 1)
                    else:
                        m = np.float32(speed_stats["global"]["mean"]) if speed_stats else 0.0
                        sp = np.full((args.seq_len, 1), m, dtype=np.float32)
                    if speed_stats is not None:
                        lo = np.float32(speed_stats["global"]["min"]); hi = np.float32(speed_stats["global"]["max"])
                        sp = (np.clip(sp, lo, hi) - lo) / ((hi - lo) + 1e-6)

                    flow = seq["flow"]
                    if motion_stats is not None:
                        sc = np.float32(motion_stats["global"]["p99_abs"]) + 1e-6
                        flow = np.clip(flow / sc, -1, 1).astype(np.float32)
                    else:
                        flow = np.clip(flow / 20.0, -1, 1).astype(np.float32)

                    def tt(a, dt="float"):
                        if dt == "long": return torch.from_numpy(a.astype(np.int64)).unsqueeze(0).to(device)
                        return torch.from_numpy(a.astype(np.float32)).unsqueeze(0).to(device)

                    inputs = {"bbox": tt(bbox_norm), "pose": tt(seq["pose"]), "speed": tt(sp),
                              "local_cnn": tt(seq["vgg"]), "local_motion": tt(flow),
                              "sem_labels": tt(seq["sem_labels"], "long"), "cat_depth": tt(seq["cat_depth"])}
                    with torch.no_grad():
                        out = model(inputs, return_aux=True)
                        logit = out["logit"].squeeze() if isinstance(out, dict) else out.squeeze()
                        prob_raw = torch.sigmoid(logit).item()

                    old = last_pred.get(pid, prob_raw)
                    prob_smooth = args.smooth_alpha * old + (1 - args.smooth_alpha) * prob_raw

                    if args.latch_frames > 0:
                        if prob_raw >= args.threshold:
                            latch[pid] = args.latch_frames
                        elif latch.get(pid, 0) > 0:
                            latch[pid] -= 1
                            prob_smooth = max(prob_smooth, args.threshold + args.action_margin + 0.01)
                    last_pred[pid] = prob_smooth

                    if args.debug_action:
                        lbl, _ = classify_intent(prob_smooth, args.threshold, args.action_margin)
                        print(f"\n[ACTION] frame={frame_count} id={pid} raw={prob_raw:.3f} "
                              f"shown={prob_smooth:.3f} {lbl}")
                    buf.pop_oldest()

            # ---- Render ----
            # Composite at DISPLAY resolution so the RGB video stays sharp.
            # Only the low-res (out_h x out_w) segmentation is upscaled, not the video.
            disp_rgb = cv2.resize(frame_bgr, disp_size, interpolation=cv2.INTER_AREA)
            seg_bgr = sem_to_rgb(sem_pip)                                   # (out_h, out_w, 3)
            seg_up = cv2.resize(seg_bgr, disp_size, interpolation=cv2.INTER_NEAREST)
            blended = cv2.addWeighted(seg_up, args.seg_alpha, disp_rgb, 1 - args.seg_alpha, 0)
            if args.show_depth and not args.no_depth:
                blended = draw_depth_overlay(blended, cat_depth, args.depth_alpha)

            draw_header(blended, frame_count, total_disp, args.seq_len, args.max_pedestrians, use_live_speed)

            for tid, bbox_orig in tracked:
                prob = last_pred.get(tid)
                warming = len(ped_buffers[tid].frames) if tid in ped_buffers else 0
                box_raw = smoothed[tid].bbox if tid in smoothed else bbox_orig
                bbox_vis = scale_bbox_for_vis(box_raw, video_orig_w, video_orig_h, disp_w, disp_h)
                is_latched = args.latch_frames > 0 and latch.get(tid, 0) > 0
                draw_prediction(blended, bbox_vis, prob, tid, args.threshold, args.action_margin,
                                args.seq_len, warming, is_latched)
                if args.show_pose and tid in ped_buffers and len(ped_buffers[tid].frames) > 0:
                    draw_pose(blended, ped_buffers[tid].frames[-1]["pose"], disp_w, disp_h)

            if args.debug_buffer:
                y = int(disp_h * 0.12)
                for tid, buf in ped_buffers.items():
                    cv2.putText(blended, f"T{tid}: {len(buf.frames)}/{buf.window_len}",
                                (disp_w - 150, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_WHITE, 1, cv2.LINE_AA)
                    y += 20

            draw_legend(blended, args.threshold, args.action_margin)
            writer.write(blended)
            written += 1
            frame_count += 1

    finally:
        cap.release(); writer.release(); cv2.destroyAllWindows()

    print(f"\n[DONE] Wrote {written} frames to {args.out}")

if __name__ == "__main__":
    main()