#!/usr/bin/env python3
"""
Step 3: Scene Features using Mask2Former (Cityscapes Panoptic) + ManyDepth


Includes:
- Mask2Former panoptic segmentation (per-frame) using HF checkpoint:
    facebook/mask2former-swin-large-cityscapes-panoptic
- Cityscapes-style inference resize (shortest_edge=1024, longest_edge=2048)
- Cityscapes -> PIP mapping (NAME-BASED, robust) from model id2label
- ManyDepth (temporal depth estimation for driving scenes)
- In-window IoU tracking for instance id consistency (LABEL-ID based)
- Categorical depth (ped/car channels) - prefers instances, fallback to CC for pedestrians
- Ego dashboard/hood suppression (stable "car" mask at bottom region)

IMPORTANT CHANGE:
- We keep semantic context for ALL classes (sem_labels in PIP space),
  but we keep INSTANCE masks for: person/rider + all vehicles (car/truck/bus/train/motorcycle/bicycle).
- We store inst_label (Cityscapes label_id) for each instance pixel so we do NOT
  need sem_pip voting to decide instance type.

Cache format:
cache_root/setXX/video_XXXX/{semantic,instance,inst_label}/00000.png

NPZ keys kept compatible:
  sem_labels, cat_depth, dvis_inst_raw, dvis_inst

Run examples:
python scripts/prep_scene_mask2former_cityscapes.py --mode generate \
  --pie_root /data/PIE \
  --cache_root /workspace/project/m2f_cache_672_384 \
  --npz_root /workspace/PIE_PREP_OUT \
  --splits train,val,test \
  --out_h 384 --out_w 672
  --force if you want to regenerate existing files

2) preprocess (fills sem_labels, cat_depth, dvis_inst_raw, dvis_inst):
python scripts/prep_scene_mask2former_cityscapes.py --mode preprocess \
  --pie_root /data/PIE \
  --cache_root /workspace/project/m2f_cache \
  --npz_root /workspace/PIE_PREP_OUT \
  --depth_weights /workspace/models/manydepth/KITTI_HR \
  --splits train,val,test \
  --suppress_ego_car
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Set
from collections import defaultdict, Counter
import inspect

import cv2
import numpy as np
import torch
from tqdm import tqdm


def safe_torch_load(path: Path, map_location):
    """torch.load compatible with older torch versions (no weights_only)."""
    try:
        sig = inspect.signature(torch.load)
        if "weights_only" in sig.parameters:
            return torch.load(path, map_location=map_location, weights_only=False)
        return torch.load(path, map_location=map_location)
    except TypeError:
        return torch.load(path, map_location=map_location)

def unwrap_state_dict(obj):
    """Handle checkpoints saved as {'state_dict':...} / {'model':...} etc."""
    if isinstance(obj, dict):
        for k in ("state_dict", "model", "net", "weights"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return obj


# Silence HF transformers spam safely (do NOT hard-depend at import time)
try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    hf_logging = None


# ============================================================
# PIP-Net Class Space (20 classes)
# ============================================================

PIP_CLASSES = [
    "bg",
    # things
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
    # stuff
    "traffic light", "traffic sign",
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "vegetation", "terrain", "sky",
]
PIP_ID = {name: idx for idx, name in enumerate(PIP_CLASSES)}
PIP_PED = {PIP_ID["person"], PIP_ID["rider"]}

# Critical semantic classes that MUST be mapped correctly for scene context
CRITICAL_SEMANTIC_CLASSES = {"road", "sidewalk", "building", "vegetation", "sky"}


# ============================================================
# Cityscapes -> PIP mapping (NAME-BASED, robust)
# ============================================================

CLASS_MAP: Optional[np.ndarray] = None     # cityscapes label_id -> PIP id (uint16)
VIP_IS_THING: Optional[np.ndarray] = None  # cityscapes label_id -> isthing bool (uint8)

CITY_LABEL2ID: Optional[Dict[str, int]] = None
KEEP_INST_LABEL_IDS: Optional[Set[int]] = None  # label_ids we keep as instances (person/rider/car)

PED_LABEL_IDS: Optional[Set[int]] = None        # cityscapes label_ids for person/rider
CAR_LABEL_IDS: Optional[Set[int]] = None        # cityscapes label_ids for car


def _norm_name(s: str) -> str:
    return s.lower().replace("_", " ").replace("-", " ").strip()


def init_cityscapes_meta_from_m2f(model_id_or_path: str, device: str = "cuda"):
    """
    Initializes metadata from HF config only (no model weights/GPU):
      - CLASS_MAP: Cityscapes label_id -> PIP id (uint16)
      - VIP_IS_THING: Cityscapes label_id -> isthing (uint8)
      - CITY_LABEL2ID: normalized label name -> label_id
      - KEEP_INST_LABEL_IDS: pedestrians + ALL vehicles (instances kept/tracked)
      - PED_LABEL_IDS, CAR_LABEL_IDS (CAR_LABEL_IDS now contains all vehicles)
    """
    global CLASS_MAP, VIP_IS_THING, CITY_LABEL2ID, KEEP_INST_LABEL_IDS, PED_LABEL_IDS, CAR_LABEL_IDS

    # If already initialized, do nothing
    if (
        CLASS_MAP is not None and VIP_IS_THING is not None and
        CITY_LABEL2ID is not None and KEEP_INST_LABEL_IDS is not None and
        PED_LABEL_IDS is not None and CAR_LABEL_IDS is not None
    ):
        return

    try:
        from transformers import AutoConfig
    except Exception as e:
        raise RuntimeError(
            "transformers is required.\n"
            "Install: pip install transformers accelerate\n"
            f"Original import error: {e}"
        )

    # Prefer local cache; fallback to download if needed
    try:
        cfg = AutoConfig.from_pretrained(model_id_or_path, local_files_only=True)
    except Exception:
        cfg = AutoConfig.from_pretrained(model_id_or_path)

    id2label = getattr(cfg, "id2label", None)
    if not id2label:
        raise RuntimeError("Checkpoint config missing id2label; cannot build mapping.")

    # Make sure keys are ints
    id2label = {int(k): str(v) for k, v in id2label.items()}

    # name -> label_id
    CITY_LABEL2ID = {_norm_name(v): int(k) for k, v in id2label.items()}

    # ------------------------------
    # Group definitions
    # ------------------------------
    ped_names = ["person", "rider"]

    # Channel-1 (vehicle) should include all these
    veh_names = ["car", "truck", "bus", "train", "motorcycle", "bicycle"]

    # Optional aliases (sometimes configs differ)
    aliases = {
        "motorbike": "motorcycle",
        "bike": "bicycle",
    }
    # Add alias keys if present in config
    for a, canonical in aliases.items():
        if a in CITY_LABEL2ID and canonical not in CITY_LABEL2ID:
            CITY_LABEL2ID[canonical] = CITY_LABEL2ID[a]

    keep_names = ped_names + veh_names

    missing = [n for n in keep_names if n not in CITY_LABEL2ID]
    if missing:
        print(f"[WARN] Missing labels in model id2label (will skip them): {missing}")

    KEEP_INST_LABEL_IDS = {CITY_LABEL2ID[n] for n in keep_names if n in CITY_LABEL2ID}
    PED_LABEL_IDS = {CITY_LABEL2ID[n] for n in ped_names if n in CITY_LABEL2ID}
    CAR_LABEL_IDS = {CITY_LABEL2ID[n] for n in veh_names if n in CITY_LABEL2ID}

    print(f"[INFO] KEEP_INST_LABEL_IDS: {sorted(KEEP_INST_LABEL_IDS)} (count={len(KEEP_INST_LABEL_IDS)})")
    print(f"[INFO] PED_LABEL_IDS: {sorted(PED_LABEL_IDS)}")
    print(f"[INFO] CAR_LABEL_IDS (ALL vehicles): {sorted(CAR_LABEL_IDS)}")

    # ------------------------------
    # Build CLASS_MAP + VIP_IS_THING
    # ------------------------------
    max_id = max(id2label.keys())
    class_map = np.zeros(max_id + 1, dtype=np.uint16)
    is_thing = np.zeros(max_id + 1, dtype=np.uint8)

    # Cityscapes "things" we care about (ped + vehicles)
    thing_names = set(ped_names + veh_names)

    name_to_pip = {
        "person": PIP_ID["person"],
        "rider": PIP_ID["rider"],
        "car": PIP_ID["car"],
        "truck": PIP_ID["truck"],
        "bus": PIP_ID["bus"],
        "train": PIP_ID["train"],
        "motorcycle": PIP_ID["motorcycle"],
        "bicycle": PIP_ID["bicycle"],
        "traffic light": PIP_ID["traffic light"],
        "traffic sign": PIP_ID["traffic sign"],
        "road": PIP_ID["road"],
        "sidewalk": PIP_ID["sidewalk"],
        "building": PIP_ID["building"],
        "wall": PIP_ID["wall"],
        "fence": PIP_ID["fence"],
        "pole": PIP_ID["pole"],
        "vegetation": PIP_ID["vegetation"],
        "terrain": PIP_ID["terrain"],
        "sky": PIP_ID["sky"],
    }

    for cid, name in id2label.items():
        nm = _norm_name(name)

        if nm in thing_names and 0 <= cid < len(is_thing):
            is_thing[cid] = 1

        if nm in name_to_pip and 0 <= cid < len(class_map):
            class_map[cid] = np.uint16(name_to_pip[nm])

    CLASS_MAP = class_map
    VIP_IS_THING = is_thing

    print(f"[INFO] Meta loaded from config: {model_id_or_path}")
    print(f"[INFO] CLASS_MAP size={len(CLASS_MAP)} | nonzero_mapped={int((CLASS_MAP > 0).sum())}")

    # ------------------------------
    # Optional mapping validation (only if you have the constant)
    # ------------------------------
    if "CRITICAL_SEMANTIC_CLASSES" in globals():
        expected_unmapped = {
            "unlabeled", "ego vehicle", "rectification border", "out of roi",
            "static", "dynamic", "ground", "parking", "rail track",
            "guard rail", "bridge", "tunnel", "polegroup", "cargroup",
            "license plate", "trailer"
        }

        missing = []
        critical_missing = []
        for cid, name in sorted(id2label.items(), key=lambda x: int(x[0])):
            nm = _norm_name(name)
            pip = int(CLASS_MAP[cid]) if cid < len(CLASS_MAP) else -1

            if pip == 0 and nm not in expected_unmapped:
                missing.append((cid, nm))
                if nm in CRITICAL_SEMANTIC_CLASSES:
                    critical_missing.append((cid, nm))

        print(f"[CHECK] Labels mapped to bg (0): {missing}")

        if critical_missing:
            raise RuntimeError(
                f"[FATAL] Critical semantic classes unmapped: {critical_missing}\n"
                f"Fix name_to_pip mapping!"
            )


# ============================================================
# Collect Required Frames from NPZ files (+ per-video mp4 offset)
# NOW HONORS limit_npz PARAMETER
# ============================================================

def collect_required_frames(
    npz_root: Path, 
    splits: List[str], 
    limit_npz: int = 0
) -> Tuple[Dict[str, Set[int]], Dict[str, int]]:
    """
    Collect required frames from NPZ files.
    
    Args:
        npz_root: Root directory containing split subdirectories
        splits: List of split names to process
        limit_npz: If > 0, only process this many NPZ files per split
    
    Returns:
        required: Dict mapping "set_id/video" -> set of frame indices
        offsets: Dict mapping "set_id/video" -> mp4_png_offset
    """
    required = defaultdict(set)
    off_counts: Dict[str, Counter] = defaultdict(Counter)

    for split in splits:
        split_dir = npz_root / split
        if not split_dir.exists():
            continue

        files = sorted(split_dir.glob("seq_*.npz"))
        
        # HONOR LIMIT
        if limit_npz and limit_npz > 0:
            files = files[:limit_npz]

        for npz_path in files:
            try:
                # Use context manager to prevent file handle leaks
                with np.load(npz_path, allow_pickle=True) as data:
                    set_id = str(data["set_id"][0]) if getattr(data["set_id"], "shape", None) else str(data["set_id"])
                    video = str(data["video_name"][0]) if getattr(data["video_name"], "shape", None) else str(data["video_name"])
                    frames = data["frame_idx"].tolist()
                    key = f"{set_id}/{video}"
                    required[key].update(frames)

                    off = int(data.get("mp4_png_offset", 0))
                    off_counts[key][off] += 1
            except Exception as e:
                print(f"[WARN] Cannot read {npz_path}: {e}")

    offsets: Dict[str, int] = {}
    for k, c in off_counts.items():
        if len(c) == 0:
            offsets[k] = 0
        else:
            best_off, _ = c.most_common(1)[0]
            if len(c) > 1:
                print(f"[WARN] Inconsistent mp4_png_offset for {k}: {dict(c)} -> using {best_off}")
            offsets[k] = int(best_off)

    return required, offsets


# ============================================================
# Mask2Former (HF) Panoptic Segmentation Wrapper
# ============================================================

class Mask2FormerHF:
    """Mask2Former HF panoptic inference wrapper - processes one frame at a time."""

    def __init__(self, model_id_or_path: str, device: str = "cuda"):
        try:
            from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
        except Exception as e:
            raise RuntimeError(
                "transformers is required for Mask2Former HF inference.\n"
                "Install: pip install transformers accelerate\n"
                f"Original import error: {e}"
            )

        self.device = torch.device(device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu")

        # Cityscapes-style inference resize (keep aspect ratio)
        self.processor = Mask2FormerImageProcessor.from_pretrained(
            model_id_or_path,
            do_resize=True,
            size={"shortest_edge": 1024, "longest_edge": 2048},
            size_divisor=32,
        )

        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model_id_or_path).to(self.device).eval()

        print(f"[INFO] Mask2Former loaded on {self.device}")
        print(f"[INFO] Model: {model_id_or_path}")
        print(f"[INFO] Resize: shortest_edge=1024, longest_edge=2048 (Cityscapes-style)")

    @torch.no_grad()
    def run_single(self, frame_bgr: np.ndarray, out_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            sem:      (H,W) uint16 Cityscapes label_id per pixel (ALL classes)
            inst:     (H,W) uint16 segment-id per pixel (ONLY kept instances; else 0)
            inst_lbl: (H,W) uint16 label_id per pixel for kept instances; else 0
        """
        if CLASS_MAP is None or KEEP_INST_LABEL_IDS is None:
            raise RuntimeError("Meta not initialized. Call init_cityscapes_meta_from_m2f() first.")

        assert frame_bgr.dtype == np.uint8, f"Expected uint8 input, got {frame_bgr.dtype}"

        out_h, out_w = out_size
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        pps = self.processor.post_process_panoptic_segmentation
        try:
            sig = inspect.signature(pps)
            if "label_ids_to_fuse" in sig.parameters:
                pan = pps(outputs, target_sizes=[(out_h, out_w)], label_ids_to_fuse=[])[0]
            else:
                pan = pps(outputs, target_sizes=[(out_h, out_w)])[0]
        except TypeError:
            pan = pps(outputs, target_sizes=[(out_h, out_w)])[0]
        except Exception:
            pan = pps(outputs, target_sizes=[(out_h, out_w)])[0]


        seg = pan["segmentation"].detach().cpu().numpy().astype(np.int64)

        sem = np.zeros_like(seg, dtype=np.uint16)
        inst = np.zeros_like(seg, dtype=np.uint16)
        inst_lbl = np.zeros_like(seg, dtype=np.uint16)
        next_inst_id = 1
        sid_remap = {}

        for s in pan["segments_info"]:
            sid = int(s["id"])
            cid = int(s["label_id"])
            m = (seg == sid)
            if not m.any():
                continue

            # Full semantic context always
            sem[m] = np.uint16(cid)

            # Instances for person/rider + all vehicles
            if cid in KEEP_INST_LABEL_IDS:
                if sid not in sid_remap:
                    sid_remap[sid] = next_inst_id
                    next_inst_id += 1
                inst[m] = np.uint16(sid_remap[sid])
                inst_lbl[m] = np.uint16(cid)


        return sem, inst, inst_lbl


_M2F: Optional[Mask2FormerHF] = None


def get_m2f(model_id_or_path: str, device: str) -> Mask2FormerHF:
    global _M2F
    if _M2F is not None:
        return _M2F
    _M2F = Mask2FormerHF(model_id_or_path=model_id_or_path, device=device)
    return _M2F


# ============================================================
# ManyDepth - Temporal Depth Estimation for Driving Scenes
# ============================================================

_MANYDEPTH_ENCODER = None
_MANYDEPTH_DEPTH = None
_MANYDEPTH_DEVICE = None
_MANYDEPTH_FEED_WIDTH = 640
_MANYDEPTH_FEED_HEIGHT = 192
_MANYDEPTH_MIN_BIN = 0.1
_MANYDEPTH_MAX_BIN = 20.0


def init_manydepth(weights_path: str = "/workspace/models/manydepth/KITTI_HR"):
    global _MANYDEPTH_ENCODER, _MANYDEPTH_DEPTH, _MANYDEPTH_DEVICE
    global _MANYDEPTH_FEED_WIDTH, _MANYDEPTH_FEED_HEIGHT
    global _MANYDEPTH_MIN_BIN, _MANYDEPTH_MAX_BIN

    if _MANYDEPTH_ENCODER is not None:
        return

    manydepth_code_root = "/workspace/project/manydepth"
    if manydepth_code_root not in sys.path:
        sys.path.insert(0, manydepth_code_root)

    from manydepth import networks

    _MANYDEPTH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights_path = Path(weights_path)
    encoder_path = weights_path / "encoder.pth"
    depth_path = weights_path / "depth.pth"

    if not encoder_path.exists():
        raise FileNotFoundError(f"ManyDepth encoder not found: {encoder_path}")
    if not depth_path.exists():
        raise FileNotFoundError(f"ManyDepth depth decoder not found: {depth_path}")

    encoder_raw = safe_torch_load(encoder_path, map_location=_MANYDEPTH_DEVICE)
    encoder_dict = unwrap_state_dict(encoder_raw)
    _MANYDEPTH_FEED_HEIGHT = int(encoder_raw.get("height", 192)) if isinstance(encoder_raw, dict) else 192
    _MANYDEPTH_FEED_WIDTH  = int(encoder_raw.get("width", 640))  if isinstance(encoder_raw, dict) else 640


    _MANYDEPTH_MIN_BIN = 0.1
    _MANYDEPTH_MAX_BIN = 20.0

    _MANYDEPTH_ENCODER = networks.ResnetEncoderMatching(
        18, False,
        input_width=_MANYDEPTH_FEED_WIDTH,
        input_height=_MANYDEPTH_FEED_HEIGHT,
        adaptive_bins=True,
        min_depth_bin=_MANYDEPTH_MIN_BIN,
        max_depth_bin=_MANYDEPTH_MAX_BIN,
        depth_binning="linear",
        num_depth_bins=96
    )

    filtered_dict = {k: v for k, v in encoder_dict.items() if k in _MANYDEPTH_ENCODER.state_dict()}
    _MANYDEPTH_ENCODER.load_state_dict(filtered_dict, strict=False)
    _MANYDEPTH_ENCODER.to(_MANYDEPTH_DEVICE)
    _MANYDEPTH_ENCODER.eval()

    _MANYDEPTH_DEPTH = networks.DepthDecoder(
        num_ch_enc=_MANYDEPTH_ENCODER.num_ch_enc,
        scales=range(4)
    )
    depth_raw = safe_torch_load(depth_path, map_location=_MANYDEPTH_DEVICE)
    depth_dict = unwrap_state_dict(depth_raw)
    _MANYDEPTH_DEPTH.load_state_dict(depth_dict, strict=True)
    _MANYDEPTH_DEPTH.to(_MANYDEPTH_DEVICE)
    _MANYDEPTH_DEPTH.eval()

    print(f"[INFO] ManyDepth loaded on {_MANYDEPTH_DEVICE}")
    print(f"[INFO] Feed size: {_MANYDEPTH_FEED_WIDTH}x{_MANYDEPTH_FEED_HEIGHT}")
    
    # Warn about aspect ratio mismatch
    feed_ratio = _MANYDEPTH_FEED_WIDTH / _MANYDEPTH_FEED_HEIGHT
    pie_ratio = 1920 / 1080
    if abs(feed_ratio - pie_ratio) > 0.5:
        print(f"[WARN] ManyDepth aspect ratio ({feed_ratio:.2f}) differs significantly from PIE ({pie_ratio:.2f})")
        print(f"[WARN] This may cause depth distortion. Consider letterboxing if cat_depth looks wrong.")


def preprocess_frame_manydepth(frame_bgr: np.ndarray) -> torch.Tensor:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (_MANYDEPTH_FEED_WIDTH, _MANYDEPTH_FEED_HEIGHT))
    x = torch.from_numpy(frame_resized.astype(np.float32) / 255.0)
    x = x.permute(2, 0, 1).unsqueeze(0)
    return x


def _make_default_K_invK(batch: int, width: int, height: int, device: torch.device):
    K = torch.zeros((batch, 4, 4), device=device, dtype=torch.float32)
    K[:, 0, 0] = 0.58 * width
    K[:, 1, 1] = 1.92 * height
    K[:, 0, 2] = 0.5 * width
    K[:, 1, 2] = 0.5 * height
    K[:, 2, 2] = 1.0
    K[:, 3, 3] = 1.0
    inv_K = torch.inverse(K)
    return K, inv_K


def _make_identity_poses(batch: int, num_frames: int, device: torch.device):
    eye = torch.eye(4, device=device, dtype=torch.float32).view(1, 1, 4, 4)
    return eye.repeat(batch, num_frames, 1, 1)


def _manydepth_encode(curr_img_4d: torch.Tensor, lookup_img_4d: torch.Tensor, min_d: float, max_d: float):
    enc = _MANYDEPTH_ENCODER
    sig = inspect.signature(enc.forward)
    pnames = list(sig.parameters.keys())

    curr = curr_img_4d
    if curr.ndim != 4:
        raise ValueError(f"curr tensor must be 4D (B,C,H,W), got {curr.shape}")

    lookup = lookup_img_4d
    if lookup.ndim == 4:
        lookup = lookup.unsqueeze(1)  # (B,1,C,H,W)
    if lookup.ndim != 5:
        raise ValueError(f"lookup_images must be 5D (B,N,C,H,W), got {lookup.shape}")

    B, N, C, H, W = lookup.shape
    device = curr.device

    poses = _make_identity_poses(B, N, device)
    K, inv_K = _make_default_K_invK(B, W, H, device)

    args = [curr, lookup]

    for name in pnames[2:]:
        param = sig.parameters[name]
        if param.default is not inspect._empty:
            break
        lname = name.lower()
        if "pose" in lname:
            args.append(poses)
        elif name in ("K", "k") or "intr" in lname:
            args.append(K)
        elif "inv_k" in lname or "kinv" in lname or "invk" in lname:
            args.append(inv_K)
        else:
            args.append(None)

    kwargs = {}
    if "min_depth_bin" in sig.parameters:
        kwargs["min_depth_bin"] = float(min_d)
    if "max_depth_bin" in sig.parameters:
        kwargs["max_depth_bin"] = float(max_d)

    try:
        return enc(*args, **kwargs)
    except TypeError:
        if "min_depth_bin" in sig.parameters and "max_depth_bin" in sig.parameters:
            return enc(*args, float(min_d), float(max_d))
        raise


def _unwrap_manydepth_features(enc_out):
    if isinstance(enc_out, tuple):
        for item in enc_out:
            if isinstance(item, (list, tuple)) and len(item) > 0 and torch.is_tensor(item[0]):
                return list(item)
        enc_out = enc_out[0]

    if isinstance(enc_out, dict):
        for k in ("features", "feats", "x"):
            if k in enc_out and isinstance(enc_out[k], (list, tuple)) and len(enc_out[k]) > 0:
                return list(enc_out[k])
        vals = list(enc_out.values())
        if vals and all(torch.is_tensor(v) for v in vals):
            return vals

    if isinstance(enc_out, (list, tuple)) and len(enc_out) > 0 and torch.is_tensor(enc_out[0]):
        return list(enc_out)

    raise TypeError(f"Cannot unwrap ManyDepth encoder output into feature list. Got type={type(enc_out)}")

_DEPTH_DEBUG_CALLS_LEFT = 0   # <-- print for first 5 estimate_depth() calls
_EGO_DEBUG_CALLS_LEFT = 0     # <-- print for first 5 suppress_ego_car_dashboard() calls

@torch.no_grad()
def estimate_depth(
    frames: List[np.ndarray],
    frame_indices: List[int],
    out_size: Tuple[int, int],
    weights_path: str,
    max_gap: int = 3,
) -> np.ndarray:
    init_manydepth(weights_path)

    out_h, out_w = out_size
    raw_depths = []
    prev_tensor = None
    prev_idx = None

    for i, frame in enumerate(frames):
        curr_tensor = preprocess_frame_manydepth(frame).to(_MANYDEPTH_DEVICE)
        curr_idx = frame_indices[i] if frame_indices else i

        if prev_tensor is None or (prev_idx is not None and curr_idx - prev_idx > max_gap):
            prev_tensor = curr_tensor

        enc_out = _manydepth_encode(curr_tensor, prev_tensor, _MANYDEPTH_MIN_BIN, _MANYDEPTH_MAX_BIN)
        features = _unwrap_manydepth_features(enc_out)

        outputs = _MANYDEPTH_DEPTH(features)
        disp = outputs[("disp", 0)]
        disp_np = disp.squeeze().detach().cpu().numpy()

        disp_resized = cv2.resize(disp_np, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        raw_depths.append(disp_resized)

        prev_tensor = curr_tensor
        prev_idx = curr_idx

    raw_depths = np.stack(raw_depths, axis=0).astype(np.float32)
    #keep absolute disp scale (close -> brighter), no per-window minmax
    depth_norm = np.clip(raw_depths, 0.0, 1.0)
    #debug flags
    global _DEPTH_DEBUG_CALLS_LEFT
    if _DEPTH_DEBUG_CALLS_LEFT > 0:
        print(f"[DBG] disp raw range: {raw_depths.min():.4f}..{raw_depths.max():.4f} | mean={raw_depths.mean():.4f}")
        print(f"[DBG] depth_norm range: {depth_norm.min():.4f}..{depth_norm.max():.4f} | mean={depth_norm.mean():.4f}")
        _DEPTH_DEBUG_CALLS_LEFT -= 1

    return depth_norm



# ============================================================
# Tracking helpers (label_id-based, no sem_pip voting)
# ============================================================

def _majority_label_id(lbl_frame: np.ndarray, mask: np.ndarray) -> int:
    vals = lbl_frame[mask].astype(np.int64)
    if vals.size == 0:
        return 0
    bc = np.bincount(vals)
    return int(bc.argmax())


def _label_group(label_id: int) -> int:
    # 0 = ped, 1 = car, 2 = other
    if PED_LABEL_IDS is not None and label_id in PED_LABEL_IDS:
        return 0
    if CAR_LABEL_IDS is not None and label_id in CAR_LABEL_IDS:
        return 1
    return 2


def track_instances_iou_window(
    inst_seq: np.ndarray,       # (T,H,W) uint16 segment ids (ONLY person/rider/car)
    inst_lbl_seq: np.ndarray,   # (T,H,W) uint16 cityscapes label_id for those instance pixels (else 0)
    iou_thr: float = 0.30,
    min_area: int = 64,
    match_same_group: bool = True,
) -> np.ndarray:
    """
    IoU tracking inside a window. Grouping uses inst_lbl_seq (label_id), not sem_pip.
    """
    assert inst_seq.ndim == 3 and inst_lbl_seq.ndim == 3
    assert inst_lbl_seq.shape == inst_seq.shape

    T, H, W = inst_seq.shape
    trk_seq = np.zeros_like(inst_seq, dtype=np.uint16)

    track_group = np.zeros(4096, dtype=np.uint8)
    next_tid = 1
    eps = 1e-8

    def _ensure_group_capacity(tid: int):
        nonlocal track_group
        if tid < track_group.shape[0]:
            return
        new_size = max(track_group.shape[0] * 2, tid + 1024)
        ng = np.zeros(new_size, dtype=np.uint8)
        ng[: track_group.shape[0]] = track_group
        track_group = ng

    for t in range(T):
        curr = inst_seq[t]
        curr_max = int(curr.max())
        if curr_max == 0:
            continue

        area_curr = np.bincount(curr.ravel().astype(np.int64), minlength=curr_max + 1)
        curr_ids = np.nonzero(area_curr >= min_area)[0]
        curr_ids = curr_ids[curr_ids > 0]

        curr_group: Dict[int, int] = {}
        for sid in curr_ids.tolist():
            mask = (curr == sid)
            label_id = _majority_label_id(inst_lbl_seq[t], mask)
            curr_group[sid] = _label_group(label_id)

        if t == 0 or int(trk_seq[t - 1].max()) == 0:
            mapping = np.zeros(curr_max + 1, dtype=np.uint16)
            for sid, g in curr_group.items():
                _ensure_group_capacity(next_tid)
                mapping[sid] = np.uint16(next_tid)
                track_group[next_tid] = np.uint8(g)
                next_tid += 1
            trk_seq[t] = mapping[curr]
            continue

        prev = trk_seq[t - 1]
        prev_max = int(prev.max())
        if prev_max == 0:
            mapping = np.zeros(curr_max + 1, dtype=np.uint16)
            for sid, g in curr_group.items():
                _ensure_group_capacity(next_tid)
                mapping[sid] = np.uint16(next_tid)
                track_group[next_tid] = np.uint8(g)
                next_tid += 1
            trk_seq[t] = mapping[curr]
            continue

        area_prev = np.bincount(prev.ravel().astype(np.int64), minlength=prev_max + 1)

        factor = curr_max + 1
        m = (prev > 0) & (curr > 0)
        if m.any():
            pairs = (prev.astype(np.int64) * factor + curr.astype(np.int64))[m]
            inter_flat = np.bincount(pairs, minlength=(prev_max + 1) * factor)
            inter = inter_flat.reshape(prev_max + 1, factor)
        else:
            inter = np.zeros((prev_max + 1, factor), dtype=np.int64)

        candidates = []
        for sid, g in curr_group.items():
            tids = np.nonzero(inter[:, sid])[0]
            for tid in tids.tolist():
                if tid == 0:
                    continue
                if match_same_group and int(track_group[tid]) != int(g):
                    continue
                inter_cnt = float(inter[tid, sid])
                union = float(area_prev[tid] + area_curr[sid]) - inter_cnt
                iou = inter_cnt / (union + eps)
                if iou >= iou_thr:
                    candidates.append((iou, sid, tid))

        candidates.sort(reverse=True, key=lambda x: x[0])

        used_s = set()
        used_t = set()
        mapping = np.zeros(curr_max + 1, dtype=np.uint16)

        for iou, sid, tid in candidates:
            if sid in used_s or tid in used_t:
                continue
            mapping[sid] = np.uint16(tid)
            used_s.add(sid)
            used_t.add(tid)

        for sid, g in curr_group.items():
            if mapping[sid] != 0:
                continue
            _ensure_group_capacity(next_tid)
            mapping[sid] = np.uint16(next_tid)
            track_group[next_tid] = np.uint8(g)
            next_tid += 1

        trk_seq[t] = mapping[curr]

    return trk_seq


# ============================================================
# Categorical Depth (ped / car only)
# ============================================================

def compute_cat_depth(
    depth: np.ndarray,         # (T,H,W) float32 [0..1]  (your "global proximity heatmap")
    sem_city: np.ndarray,      # (T,H,W) uint16 cityscapes label_id (FULL context)
    inst_trk: np.ndarray,      # (T,H,W) uint16 tracked ids
    inst_lbl: np.ndarray,      # (T,H,W) uint16 cityscapes label_id for instance pixels
    min_area: int = 50,
    morph_kernel_size: int = 3,
    prefer_instances_for_ped: bool = True,
) -> np.ndarray:
    """
    Paper-style:
      - Start from global heatmap (depth/proximity)
      - Crop by instance mask
      - Replace pixels in each instance by the instance mean (one scalar)

    Output:
      cat_depth: (T, 2, H, W)
        channel 0: pedestrians (person/rider)
        channel 1: vehicles (car, truck, bus, train, motorcycle, bicycle)
    """
    assert sem_city.shape == depth.shape == inst_trk.shape == inst_lbl.shape

    T, H, W = depth.shape
    ped = np.zeros((T, H, W), dtype=np.float32)
    car = np.zeros((T, H, W), dtype=np.float32)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))

    for t in range(T):
        # ---------------------------------------------------
        # 1) Use tracked instances if available (ped + car)
        # ---------------------------------------------------
        if prefer_instances_for_ped:
            ids = np.unique(inst_trk[t])
            ids = ids[ids > 0]
            for tid in ids.tolist():
                mask = (inst_trk[t] == tid)
                if int(mask.sum()) < min_area:
                    continue

                label_id = _majority_label_id(inst_lbl[t], mask)
                mean_val = float(depth[t][mask].mean())  # <-- keep global proximity

                if PED_LABEL_IDS is not None and label_id in PED_LABEL_IDS:
                    ped[t][mask] = mean_val
                elif CAR_LABEL_IDS is not None and label_id in CAR_LABEL_IDS:
                    car[t][mask] = mean_val

        else:
            # still fill cars from instances even if prefer_instances_for_ped=False
            ids = np.unique(inst_trk[t])
            ids = ids[ids > 0]
            for tid in ids.tolist():
                mask = (inst_trk[t] == tid)
                if int(mask.sum()) < min_area:
                    continue
                label_id = _majority_label_id(inst_lbl[t], mask)
                if CAR_LABEL_IDS is not None and label_id in CAR_LABEL_IDS:
                    car[t][mask] = float(depth[t][mask].mean())

        # ---------------------------------------------------
        # 2) Pedestrian fallback: CC from semantic if no ped instance pixels
        # ---------------------------------------------------
        if (ped[t] > 0).sum() == 0:
            if PED_LABEL_IDS is not None and len(PED_LABEL_IDS) > 0:
                ped_sem_mask = np.isin(sem_city[t], list(PED_LABEL_IDS)).astype(np.uint8)
            else:
                ped_sem_mask = np.zeros((H, W), dtype=np.uint8)

            if ped_sem_mask.any():
                cleaned = cv2.morphologyEx(ped_sem_mask, cv2.MORPH_OPEN, kernel)
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
                for i in range(1, num_labels):
                    area = int(stats[i, cv2.CC_STAT_AREA])
                    if area < min_area:
                        continue
                    m = (labels == i)
                    ped[t][m] = float(depth[t][m].mean())

    # depth is already [0..1], so outputs stay [0..1]
    return np.stack([ped, car], axis=1).astype(np.float32)



def suppress_ego_car_dashboard(
    sem_pip: np.ndarray,                     # (T,H,W) uint16 PIP ids
    inst_vip: np.ndarray,                    # (T,H,W) uint16 segment ids
    inst_lbl: Optional[np.ndarray] = None,   # (T,H,W) uint16 label ids
    depth_norm: Optional[np.ndarray] = None, # (T,H,W) float32 [0..1]
    bottom_frac: float = 0.88,
    fill_label: int = PIP_ID["road"],        # label bottom band as road
    zero_depth: bool = True,                 # remove depth glow in band
) -> np.ndarray:
    """
    Always suppress the bottom band y >= bottom_frac*H in ALL frames.
    Returns: ego_mask (H,W) bool
    """
    assert sem_pip.ndim == 3 and inst_vip.ndim == 3
    T, H, W = sem_pip.shape

    y0 = int(H * bottom_frac)
    ego_mask = np.zeros((H, W), dtype=bool)
    ego_mask[y0:, :] = True

    # Apply suppression consistently
    sem_pip[:, ego_mask] = np.uint16(fill_label)   # or PIP_ID["bg"] if you prefer
    inst_vip[:, ego_mask] = np.uint16(0)
    if inst_lbl is not None:
        inst_lbl[:, ego_mask] = np.uint16(0)
    if zero_depth and depth_norm is not None:
        depth_norm[:, ego_mask] = 0.0

    return ego_mask




# ============================================================
# Video Cache
# ============================================================

class VideoCache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.caps = {}

    def get(self, set_id: str, video: str) -> cv2.VideoCapture:
        key = f"{set_id}/{video}"
        if key not in self.caps:
            path = self.root / set_id / f"{video}.mp4"
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open: {path}")
            self.caps[key] = cap
        return self.caps[key]

    def read(self, set_id: str, video: str, idx: int, off: int = 0) -> np.ndarray:
        cap = self.get(set_id, video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx + off)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Read failed: {idx} (off={off})")
        return frame

    def reads(self, set_id: str, video: str, idxs: List[int], off: int = 0) -> List[np.ndarray]:
        return [self.read(set_id, video, i, off) for i in idxs]

    def release(self):
        for c in self.caps.values():
            c.release()
        self.caps.clear()


# ============================================================
# Cache I/O
# ============================================================

def _build_inst_lbl_from_sem_inst(sem: np.ndarray, inst: np.ndarray) -> np.ndarray:
    """
    Reconstruct inst_label if missing:
      for each instance segment-id sid, set inst_lbl[mask] = majority(sem[mask])
    sem is cityscapes label_id per pixel.
    inst is segment-id per pixel (only kept instances).
    """
    inst_lbl = np.zeros_like(inst, dtype=np.uint16)
    ids = np.unique(inst)
    ids = ids[ids > 0]
    for sid in ids.tolist():
        mask = (inst == sid)
        if not mask.any():
            continue
        label_id = _majority_label_id(sem, mask)
        inst_lbl[mask] = np.uint16(label_id)
    return inst_lbl


def save_frame_cache(path: Path, sem: np.ndarray, inst: np.ndarray, inst_lbl: np.ndarray, idx: int):
    (path / "semantic").mkdir(parents=True, exist_ok=True)
    (path / "instance").mkdir(parents=True, exist_ok=True)
    (path / "inst_label").mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(path / "semantic" / f"{idx:05d}.png"), sem.astype(np.uint16))
    cv2.imwrite(str(path / "instance" / f"{idx:05d}.png"), inst.astype(np.uint16))
    cv2.imwrite(str(path / "inst_label" / f"{idx:05d}.png"), inst_lbl.astype(np.uint16))


def load_frame_cache(
    path: Path, idx: int, expected_size: Tuple[int, int]
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    sp = path / "semantic" / f"{idx:05d}.png"
    ip = path / "instance" / f"{idx:05d}.png"
    lp = path / "inst_label" / f"{idx:05d}.png"

    if not sp.exists():
        return None, None, None

    s = cv2.imread(str(sp), cv2.IMREAD_UNCHANGED)
    if s is None:
        return None, None, None

    expected_h, expected_w = expected_size
    if s.shape[:2] != (expected_h, expected_w):
        print(f"[WARN] Cache size mismatch for {sp}: got {s.shape[:2]}, expected {expected_size}. Will regenerate.")
        return None, None, None

    i = cv2.imread(str(ip), cv2.IMREAD_UNCHANGED) if ip.exists() else None
    if i is None:
        i = np.zeros_like(s, dtype=np.uint16)
    elif i.shape[:2] != (expected_h, expected_w):
        print(f"[WARN] Instance cache size mismatch for {ip}. Will regenerate.")
        return None, None, None

    l = cv2.imread(str(lp), cv2.IMREAD_UNCHANGED) if lp.exists() else None
    if l is None:
        # Old cache: reconstruct inst_label from sem+inst (safe + accurate)
        l = _build_inst_lbl_from_sem_inst(s.astype(np.uint16), i.astype(np.uint16))
        # Write back so it stops being silently missing
        (path / "inst_label").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(lp), l.astype(np.uint16))
    elif l.shape[:2] != (expected_h, expected_w):
        print(f"[WARN] inst_label cache size mismatch for {lp}. Will regenerate.")
        return None, None, None

    return s.astype(np.uint16), i.astype(np.uint16), l.astype(np.uint16)


def load_cache(
    path: Path, idxs: List[int], expected_size: Tuple[int, int]
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    sems, insts, lbls = [], [], []
    for idx in idxs:
        s, i, l = load_frame_cache(path, idx, expected_size)
        if s is None:
            return None, None, None
        sems.append(s)
        insts.append(i)
        lbls.append(l)
    return np.stack(sems), np.stack(insts), np.stack(lbls)


# ============================================================
# Sanity Checks
# ============================================================

def validate_cache_arrays(
    sem_city: np.ndarray,
    inst: np.ndarray,
    inst_lbl: np.ndarray,
    expected_size: Tuple[int, int],
    verbose: bool = False,
) -> bool:
    """Validate per-window shapes/dtypes/consistency. Returns True if valid."""
    out_h, out_w = expected_size
    
    # Shape/dtype checks
    assert sem_city.dtype == np.uint16, f"sem_city dtype={sem_city.dtype}"
    assert inst.dtype == np.uint16, f"inst dtype={inst.dtype}"
    assert inst_lbl.dtype == np.uint16, f"inst_lbl dtype={inst_lbl.dtype}"
    assert sem_city.shape == inst.shape == inst_lbl.shape, \
        f"Shape mismatch: sem={sem_city.shape}, inst={inst.shape}, lbl={inst_lbl.shape}"
    assert sem_city.shape[1:] == (out_h, out_w), \
        f"Spatial size mismatch: got {sem_city.shape[1:]}, expected ({out_h}, {out_w})"
    
    # Instance-label consistency check A: inst>0 -> inst_lbl>0
    m_inst = inst > 0
    if m_inst.any():
        if (inst_lbl[m_inst] == 0).any():
            bad_ratio = float((inst_lbl[m_inst] == 0).sum()) / float(m_inst.sum())
            print(f"[WARN] inst>0 but inst_lbl==0 on {bad_ratio*100:.2f}% of instance pixels (bad cache?)")
            return False
        
        # inst_lbl should match sem_city on those pixels
        mismatch = (inst_lbl[m_inst] != sem_city[m_inst]).mean()
        if mismatch > 0.01:
            print(f"[WARN] inst_lbl != sem_city on {mismatch*100:.2f}% of instance pixels (unexpected)")
    
    # Instance-label consistency check B: inst_lbl>0 -> inst>0
    m_lbl = inst_lbl > 0
    if m_lbl.any() and (inst[m_lbl] == 0).any():
        bad_ratio = float((inst[m_lbl] == 0).sum()) / float(m_lbl.sum())
        print(f"[WARN] inst_lbl>0 but inst==0 on {bad_ratio*100:.2f}% of labeled pixels (bad cache)")
        return False
    
    # Instance-label consistency check C: inst_lbl values must be subset of allowed labels
    if m_lbl.any() and KEEP_INST_LABEL_IDS is not None:
        allowed = set(KEEP_INST_LABEL_IDS)
        vals = np.unique(inst_lbl[m_lbl])
        bad_vals = [int(v) for v in vals if int(v) not in allowed]
        if bad_vals:
            print(f"[WARN] inst_lbl contains unexpected label_ids: {bad_vals} (allowed: {sorted(allowed)})")
            return False
    
    if verbose:
        print(f"[OK] Cache validated: T={sem_city.shape[0]}, size={out_h}x{out_w}")
    
    return True


def validate_sem_pip(sem_pip: np.ndarray, verbose: bool = False) -> bool:
    """Validate mapped semantic labels."""
    assert sem_pip.dtype == np.uint16, f"sem_pip dtype={sem_pip.dtype}"
    assert sem_pip.min() >= 0 and sem_pip.max() < len(PIP_CLASSES), \
        f"sem_pip range invalid: [{sem_pip.min()}, {sem_pip.max()}]"
    
    if verbose:
        u, c = np.unique(sem_pip, return_counts=True)
        top = sorted(zip(c.tolist(), u.tolist()), reverse=True)[:10]
        top_names = [(PIP_CLASSES[i], cnt) for cnt, i in top]
        print(f"[DBG] Top PIP classes: {top_names}")
    
    return True


def validate_cat_depth(cat_depth: np.ndarray, T: int, out_h: int, out_w: int) -> bool:
    """Validate categorical depth output."""
    assert cat_depth.shape == (T, 2, out_h, out_w), \
        f"cat_depth shape mismatch: got {cat_depth.shape}, expected ({T}, 2, {out_h}, {out_w})"
    assert np.isfinite(cat_depth).all(), "cat_depth contains NaN/Inf"
    assert cat_depth.min() >= -1e-6 and cat_depth.max() <= 1.0 + 1e-6, \
        f"cat_depth range invalid: [{cat_depth.min()}, {cat_depth.max()}]"
    return True


def print_instance_stats(inst: np.ndarray, inst_trk: np.ndarray, verbose: bool = False):
    """Print instance count stats for debugging."""
    if not verbose:
        return
    inst_count = int((np.unique(inst[0]) > 0).sum())
    trk_count = int((np.unique(inst_trk[0]) > 0).sum())
    print(f"[DBG] Frame 0: raw_inst_segs={inst_count} | tracked_ids={trk_count}")


# ============================================================
# Debug Visualization
# ============================================================

def save_debug_frames(
    debug_dir: Path,
    npz_name: str,
    frame_bgr: np.ndarray,
    sem_pip: np.ndarray,
    inst: np.ndarray,
    cat_depth: np.ndarray,
    frame_idx: int = 0,
    ego_mask: Optional[np.ndarray] = None,
):
    debug_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{npz_name}_f{frame_idx:03d}"

    # pick the frame
    sem_frame = sem_pip[frame_idx] if sem_pip.ndim == 3 else sem_pip
    inst_frame = inst[frame_idx] if inst.ndim == 3 else inst
    cd_frame = cat_depth[frame_idx] if cat_depth.ndim == 4 else cat_depth  # (2,H,W)

    H, W = sem_frame.shape

    # --- IMPORTANT: resize raw frame to match HxW (avoids your old IndexError) ---
    if frame_bgr.shape[:2] != (H, W):
        frame_bgr = cv2.resize(frame_bgr, (W, H), interpolation=cv2.INTER_AREA)

    # 1) Save RGB (for viewing)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    cv2.imwrite(str(debug_dir / f"{prefix}_rgb.png"), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))  # save correctly

    # 1b) Ego mask + overlay (optional)
    if ego_mask is not None:
        if ego_mask.shape != (H, W):
            raise ValueError(f"ego_mask shape {ego_mask.shape} != {(H,W)}")

        ego_u8 = (ego_mask.astype(np.uint8) * 255)
        cv2.imwrite(str(debug_dir / f"{prefix}_ego_mask.png"), ego_u8)

        overlay = frame_bgr.copy()  # BGR
        alpha = 0.65
        color = np.array([255, 0, 0], dtype=np.float32)  # BLUE in BGR
        m = ego_mask
        overlay[m] = ((1 - alpha) * overlay[m].astype(np.float32) + alpha * color).astype(np.uint8)
        cv2.imwrite(str(debug_dir / f"{prefix}_ego_overlay.png"), overlay)

    # 2) sem_pip pseudo-color
    sem_color_hsv = np.zeros((H, W, 3), dtype=np.uint8)
    for cls_id in range(1, len(PIP_CLASSES)):
        mask = (sem_frame == cls_id)
        if mask.any():
            hue = int((cls_id * 180 / len(PIP_CLASSES)) % 180)
            sem_color_hsv[mask] = [hue, 255, 200]
    sem_color_bgr = cv2.cvtColor(sem_color_hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(str(debug_dir / f"{prefix}_sem.png"), sem_color_bgr)

    # 3) instance boundaries visualization
    inst_vis = frame_bgr.copy()
    ids = np.unique(inst_frame)
    for iid in ids.tolist():
        if iid == 0:
            continue
        mask = (inst_frame == iid).astype(np.uint8)
        if mask.sum() < 10:
            continue
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # deterministic-ish color from id
        color = (
            int((37 * iid) % 256),
            int((17 * iid) % 256),
            int((97 * iid) % 256),
        )
        cv2.drawContours(inst_vis, contours, -1, color, 2)
    cv2.imwrite(str(debug_dir / f"{prefix}_inst.png"), inst_vis)

    # 4) cat_depth heatmaps (ped/car)
    ped = cd_frame[0]
    car = cd_frame[1]

    ped_u8 = np.clip(ped * 255.0, 0, 255).astype(np.uint8)
    car_u8 = np.clip(car * 255.0, 0, 255).astype(np.uint8)

    ped_cm = cv2.applyColorMap(ped_u8, cv2.COLORMAP_JET)
    car_cm = cv2.applyColorMap(car_u8, cv2.COLORMAP_JET)

    cv2.imwrite(str(debug_dir / f"{prefix}_catdepth_ped.png"), ped_cm)
    cv2.imwrite(str(debug_dir / f"{prefix}_catdepth_car.png"), car_cm)

    # 4b) combined heatmap: max(ped, car)
    combined = np.maximum(ped, car)
    combined_u8 = np.clip(combined * 255.0, 0, 255).astype(np.uint8)
    combined_cm = cv2.applyColorMap(combined_u8, cv2.COLORMAP_JET)
    cv2.imwrite(str(debug_dir / f"{prefix}_catdepth_combined.png"), combined_cm)

    # 4c) combined overlay on RGB: intensity from cat_depth
    overlay2 = frame_bgr.astype(np.float32)

    alpha = 0.55  # or whatever you want

    car_w = np.clip(car, 0.0, 1.0)[..., None]  # (H,W,1)
    ped_w = np.clip(ped, 0.0, 1.0)[..., None]

    # car -> red (BGR = 0,0,255)
    overlay2 = overlay2 * (1.0 - alpha * car_w) + (alpha * car_w) * np.array([0, 0, 255], dtype=np.float32)

    # ped -> green (BGR = 0,255,0)
    overlay2 = overlay2 * (1.0 - alpha * ped_w) + (alpha * ped_w) * np.array([0, 255, 0], dtype=np.float32)

    overlay2 = np.clip(overlay2, 0, 255).astype(np.uint8)
    cv2.imwrite(str(debug_dir / f"{prefix}_catdepth_overlay.png"), overlay2)


# ============================================================
# Generate Cache (only needed frames)
# ============================================================

def generate(args):
    init_cityscapes_meta_from_m2f(args.m2f_model_id, device=args.m2f_device)

    pie = Path(args.pie_root)
    cache = Path(args.cache_root)
    npz_root = Path(args.npz_root)
    clips = pie / "PIE_clips"

    splits = [s.strip() for s in args.splits.split(",")]
    out_size = (args.out_h, args.out_w)

    print("[INFO] Scanning NPZ files for required frames...")
    # NOW HONORS --limit
    required, offsets = collect_required_frames(npz_root, splits, limit_npz=args.limit)

    total_frames = sum(len(v) for v in required.values())
    print(f"[INFO] Found {len(required)} videos, {total_frames} unique frames needed")
    print(f"[INFO] Output size: {args.out_w}x{args.out_h} (WxH)")

    if total_frames == 0:
        print("[WARN] No frames to generate!")
        return

    m2f = get_m2f(args.m2f_model_id, args.m2f_device)
    vc = VideoCache(clips)

    processed_total = 0
    try:
        for video_key in tqdm(sorted(required.keys()), desc="Videos"):
            set_id, video = video_key.split("/")
            cdir = cache / set_id / video

            frame_idxs = sorted(required[video_key])
            off = int(offsets.get(video_key, 0))

            if not args.force:
                missing = []
                for idx in frame_idxs:
                    s, _, _ = load_frame_cache(cdir, idx, out_size)
                    if s is None:
                        missing.append(idx)
                frame_idxs = missing

            if not frame_idxs:
                continue

            for idx in tqdm(frame_idxs, desc=f"{set_id}/{video}", leave=False):
                try:
                    frame = vc.read(set_id, video, idx, off=off)
                    sem, inst, inst_lbl = m2f.run_single(frame, out_size)
                    save_frame_cache(cdir, sem, inst, inst_lbl, idx)
                    processed_total += 1
                except Exception as e:
                    print(f"[ERR] {set_id}/{video} frame {idx} (off={off}): {e}")
    finally:
        vc.release()

    print(f"[DONE] Generated {processed_total} frames")


# ============================================================
# Preprocess
# ============================================================

def preprocess(args):
    init_cityscapes_meta_from_m2f(args.m2f_model_id, device=args.m2f_device)

    pie = Path(args.pie_root)
    npz_root = Path(args.npz_root)
    cache = Path(args.cache_root)
    clips = pie / "PIE_clips"

    splits = [s.strip() for s in args.splits.split(",")]
    out_size = (args.out_h, args.out_w)
    vc = VideoCache(clips)

    debug_dir = Path(args.debug_dump_dir) if args.debug_dump_dir else None
    debug_count = 0

    print("[INFO] Semantic model: Mask2Former Swin-L (Cityscapes panoptic)")
    print(f"[INFO] HF model id/path: {args.m2f_model_id}")
    print("[INFO] Depth model: ManyDepth (KITTI)")
    print(f"[INFO] Depth weights: {args.depth_weights}")
    print(f"[INFO] Output size: {args.out_w}x{args.out_h} (WxH)")
    print(f"[INFO] Depth max gap: {args.depth_max_gap}")
    print("[INFO] cat_depth channels: [0]=pedestrians (person/rider), [1]=vehicles (car/truck/bus/train/cycle)")
    try:
        for split in splits:
            sdir = npz_root / split
            if not sdir.exists():
                continue

            files = sorted(sdir.glob("seq_*.npz"))
            if args.limit > 0:
                files = files[:args.limit]

            print(f"[{split}] {len(files)} files")
            updated = 0
            skipped = 0
            cache_miss = 0

            for npz in tqdm(files, desc=split):
                try:
                    with np.load(npz, allow_pickle=True) as z:
                        data = dict(z)
                except Exception as e:
                    print(f"[WARN] Corrupt NPZ, skipping: {npz} ({e})")
                    continue

                if int(data.get("_stage_scene", 0)) == 1 and not args.force:
                    skipped += 1
                    continue

                set_id = str(data["set_id"][0]) if getattr(data["set_id"], "shape", None) else str(data["set_id"])
                video = str(data["video_name"][0]) if getattr(data["video_name"], "shape", None) else str(data["video_name"])
                fidx = list(data["frame_idx"])
                off = int(data.get("mp4_png_offset", 0))

                # Load cache
                sem_city, inst, inst_lbl = load_cache(cache / set_id / video, fidx, out_size)
                if sem_city is None:
                    cache_miss += 1
                    continue

                # ========================================
                # SANITY CHECK A: Validate cache arrays
                # ========================================
                try:
                    if not validate_cache_arrays(sem_city, inst, inst_lbl, out_size, verbose=args.verbose):
                        print(f"[ERR] Cache validation failed for {npz}")
                        continue
                except AssertionError as e:
                    print(f"[ERR] Cache validation failed for {npz}: {e}")
                    continue

                try:
                    frames = vc.reads(set_id, video, fidx, off)
                except Exception:
                    continue

                # 1) depth
                depth = estimate_depth(
                    frames=frames,
                    frame_indices=fidx,
                    out_size=out_size,
                    weights_path=args.depth_weights,
                    max_gap=args.depth_max_gap,
                )

                # 2) map Cityscapes label_id -> PIP id
                sem_safe = sem_city.astype(np.int32)
                max_idx = len(CLASS_MAP) - 1
                bad = (sem_safe < 0) | (sem_safe > max_idx)
                sem_safe[bad] = 0
                sem_pip = CLASS_MAP[sem_safe]  # (T,H,W) uint16
                if args.verbose and updated < 3:
                    u, c = np.unique(sem_pip, return_counts=True)
                    top = sorted(zip(c.tolist(), u.tolist()), reverse=True)[:10]
                    print("[DBG] Top sem_pip:", [(PIP_CLASSES[i], cnt) for cnt, i in top])

                # ========================================
                # SANITY CHECK B: Validate sem_pip
                # ========================================
                try:
                    validate_sem_pip(sem_pip, verbose=(args.verbose and updated < 3))
                except AssertionError as e:
                    print(f"[ERR] sem_pip validation failed for {npz}: {e}")
                    continue

                # 3) ego suppression on PIP car + instance ids + inst_lbl
                inst_clean = inst.copy()
                inst_lbl_clean = inst_lbl.copy()
                ego_mask = None
                if args.suppress_ego_car:
                    ego_mask = suppress_ego_car_dashboard(
                        sem_pip=sem_pip,
                        inst_vip=inst_clean,
                        inst_lbl=inst_lbl_clean,
                        depth_norm=depth,  # <-- ALWAYS pass depth so it gets zeroed
                        bottom_frac=args.ego_bottom_frac,
                        # freq_thr/depth_thr can stay in signature but won't matter
                    )


                # 4) tracking using inst_lbl (label_id-based)
                trk_inst = track_instances_iou_window(
                    inst_seq=inst_clean,
                    inst_lbl_seq=inst_lbl_clean,
                    iou_thr=args.track_iou_thr,
                    min_area=args.track_min_area,
                    match_same_group=(not args.track_cross_group),
                )

                # ========================================
                # SANITY CHECK D: Instance count stats
                # ========================================
                print_instance_stats(inst, trk_inst, verbose=(args.verbose and updated < 3))

                # 5) cat depth (ped + car only) using inst_lbl and sem_city fallback
                cat_depth = compute_cat_depth(
                    depth=depth,
                    sem_city=sem_city,
                    inst_trk=trk_inst,
                    inst_lbl=inst_lbl_clean,
                    min_area=args.cat_depth_min_area,
                    morph_kernel_size=args.cat_depth_morph_kernel,
                    prefer_instances_for_ped=args.prefer_instances_for_ped,
                )
                
                # Apply ego mask to cat_depth (zero out hood region in car channel)
                if args.suppress_ego_car and ego_mask is not None and ego_mask.any():
                    cat_depth[:, :, ego_mask] = 0.0

                if args.verbose and updated < 3:
                    for ch, name in [(0,"ped"), (1,"car")]:
                        vals = cat_depth[0, ch]
                        u = np.unique(vals[vals > 0])
                        print(name, "unique mean values (frame0):", np.sort(u)[:10], "count:", len(u))

                # ========================================
                # SANITY CHECK C: Validate cat_depth
                # ========================================
                try:
                    validate_cat_depth(cat_depth, len(fidx), args.out_h, args.out_w)
                except AssertionError as e:
                    print(f"[ERR] cat_depth validation failed for {npz}: {e}")
                    continue

                # ========================================
                # SANITY CHECK E: Debug dump (optional)
                # ========================================
                if debug_dir is not None and debug_count < args.debug_first_n:
                    save_debug_frames(
                        debug_dir=debug_dir,
                        npz_name=npz.stem,
                        frame_bgr=frames[0],
                        sem_pip=sem_pip,
                        inst=inst,
                        cat_depth=cat_depth,
                        frame_idx=0,
                        ego_mask=ego_mask,   # <--- NEW

                    )
                    debug_count += 1

                dtype = np.float16 if args.fp16 else np.float32
                data["sem_labels"] = sem_pip.astype(np.uint16)
                data["cat_depth"] = cat_depth.astype(dtype)

                # Keep key names for downstream compatibility:
                data["dvis_inst_raw"] = inst.astype(np.uint16)    # raw (before suppression)
                data["dvis_inst"] = trk_inst.astype(np.uint16)    # tracked ids (after suppression)
                data["_inst_tracking"] = np.array(["iou_window_labelid"], dtype=object)

                data["_stage_scene"] = np.int32(1)
                data["_sem_model"] = np.array(["mask2former_swinl_cityscapes"], dtype=object)
                data["_depth_model"] = np.array(["manydepth"], dtype=object)
                data["_scene_size"] = np.array([args.out_h, args.out_w], dtype=np.int32)

                tmp = npz.with_name(npz.stem + "._tmp.npz")
                np.savez_compressed(tmp, **data)
                os.replace(tmp, npz)

                updated += 1

            print(f"  -> {updated} updated, {skipped} skipped, {cache_miss} cache misses")
    finally:
        vc.release()

    print("[DONE]")


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["generate", "preprocess"], required=True)
    p.add_argument("--pie_root", required=True)
    p.add_argument("--cache_root", required=True)
    p.add_argument("--npz_root", required=True, help="NPZ root to scan for required frames")
    p.add_argument("--splits", default="train,val,test")

    # Output resolution - RECTANGULAR (PIE is 1920x1080, default to half: 960x540 to keep 16:9)
    p.add_argument("--out_h", type=int, default=540, help="Output height (default: 540)")
    p.add_argument("--out_w", type=int, default=960, help="Output width (default: 960)")

    p.add_argument("--fp16", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="Limit number of NPZ files to process (0 = no limit)")
    p.add_argument("--verbose", action="store_true", help="Print debug info for first few NPZs")

    # Mask2Former (HF) Cityscapes panoptic
    p.add_argument(
        "--m2f_model_id",
        default="facebook/mask2former-swin-large-cityscapes-panoptic",
        help="HF model id or local path for Mask2Former Cityscapes panoptic"
    )
    p.add_argument("--m2f_device", default="cuda", help="cuda or cpu")

    # Depth
    p.add_argument("--depth_weights", default="/workspace/models/manydepth/KITTI_HR")
    p.add_argument("--depth_max_gap", type=int, default=3,
                   help="Max frame gap before resetting ManyDepth temporal reference")

    # Tracking
    p.add_argument("--track_iou_thr", type=float, default=0.30)
    p.add_argument("--track_min_area", type=int, default=64)
    p.add_argument("--track_cross_group", action="store_true",
                   help="If set, allow matching across groups (ped/car/other). Default: same-group only.")

    # Categorical depth
    p.add_argument("--cat_depth_min_area", type=int, default=50,
                   help="Minimum area (pixels) for valid instance in cat_depth")
    p.add_argument("--cat_depth_morph_kernel", type=int, default=3,
                   help="Kernel size for morphological cleanup in cat_depth (OPEN only for ped)")
    p.add_argument("--prefer_instances_for_ped", action="store_true", default=True,
                   help="Prefer tracked instances for pedestrians (fallback to CC if none found)")
    p.add_argument("--no_prefer_instances_for_ped", action="store_false", dest="prefer_instances_for_ped",
                   help="Always use CC for pedestrians")

    # Ego dashboard suppression
    p.add_argument("--suppress_ego_car", action="store_true",
                   help="Suppress ego dashboard/hood pixels mislabeled as car (stable mask in bottom region).")
    p.add_argument("--ego_bottom_frac", type=float, default=0.88,
                   help="Bottom region start (0..1). 0.82 means bottom 18%%.")
    
    # Debug
    p.add_argument("--debug_dump_dir", type=str, default=None,
                   help="Directory to save debug visualizations (RGB, sem, inst, cat_depth)")
    p.add_argument("--debug_first_n", type=int, default=5,
                   help="Number of NPZs to dump debug frames for")

    args = p.parse_args()

    if args.mode == "generate":
        generate(args)
    else:
        preprocess(args)


if __name__ == "__main__":
    main()
