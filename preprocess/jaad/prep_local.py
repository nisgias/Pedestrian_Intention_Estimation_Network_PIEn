#!/usr/bin/env python3
"""
Step 2: Local Features - VGG19 CNN + RAFT Optical Flow  (JAAD version)
Updates existing NPZ files created by prep_kinematics.py

Example:
  python scripts/prep_local_jaad.py \
    --npz_root /workspace/JAAD_PREP_OUT/ETC0_5 \
    --jaad_root /data/JAAD \
    --splits train,val,test \
    --raft_variant large \
    --raft_weights C_T_SKHT_K_V2 \
    --flow_size 224 \
    --flow_batch 4 \
    --vgg_batch 16 \
    --amp
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torchvision import models, transforms
from torchvision.models.optical_flow import (
    raft_small, raft_large,
    Raft_Small_Weights, Raft_Large_Weights
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Video Cache  (single-open, OOM-safe)
# -----------------------------------------------------------------------------

class VideoCache:
    """
    JAAD-specific video loader.
    - Flat directory structure (no set_id nesting).
    - Only ONE VideoCapture is kept open at a time to prevent OOM.
    - Searches multiple candidate directories under jaad_root.
    """

    CANDIDATE_SUBDIRS = ["videos", "JAAD_clips", ".", "clips"]

    def __init__(self, jaad_root: Path):
        self.jaad_root = Path(jaad_root)
        self._current_name: str | None = None
        self._current_cap: cv2.VideoCapture | None = None

    # ---- internal helpers ---------------------------------------------------

    def _find_video(self, video_name: str) -> Path:
        """Search candidate directories for <video_name>.mp4"""
        for subdir in self.CANDIDATE_SUBDIRS:
            candidate = self.jaad_root / subdir / f"{video_name}.mp4"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Video '{video_name}.mp4' not found in any of: "
            + ", ".join(str(self.jaad_root / s) for s in self.CANDIDATE_SUBDIRS)
        )

    def _release_current(self):
        if self._current_cap is not None:
            try:
                self._current_cap.release()
            except Exception:
                pass
            self._current_cap = None
            self._current_name = None

    # ---- public API ---------------------------------------------------------

    def get_cap(self, video_name: str) -> cv2.VideoCapture:
        if self._current_name == video_name and self._current_cap is not None:
            return self._current_cap

        # Release the previous video before opening a new one
        self._release_current()

        mp4 = self._find_video(video_name)
        cap = cv2.VideoCapture(str(mp4))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open: {mp4}")

        self._current_cap = cap
        self._current_name = video_name
        return cap

    def read_frame(self, video_name: str, frame_idx: int, offset: int = 0) -> np.ndarray:
        cap = self.get_cap(video_name)
        idx = int(frame_idx) + int(offset)
        if idx < 0:
            raise RuntimeError(
                f"Invalid frame idx={idx} (frame_idx={frame_idx}, offset={offset})"
            )

        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read frame {idx} from {video_name}")
        return frame

    def release_all(self):
        self._release_current()


# -----------------------------------------------------------------------------
# VGG19 Local CNN (ImageNet-1K)
# -----------------------------------------------------------------------------

_VGG = None
_VGG_TRANSFORM = None
_VGG_DEVICE = None

def init_vgg():
    global _VGG, _VGG_TRANSFORM, _VGG_DEVICE
    if _VGG is not None:
        return

    _VGG_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Explicit ImageNet weights
    vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    _VGG = vgg.features.to(_VGG_DEVICE).eval()

    _VGG_TRANSFORM = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    print(f"[INFO] VGG19 ImageNet1K on {_VGG_DEVICE}")

@torch.inference_mode()
def extract_vgg(frames: List[np.ndarray], boxes: np.ndarray, scale=1.2, batch_size=8, amp=False):
    init_vgg()
    crops = []

    for img, (x1, y1, x2, y2) in zip(frames, boxes):
        h, w = img.shape[:2]
        x1, y1, x2, y2 = expand_box(x1, y1, x2, y2, w, h, scale)

        crop = img[y1:y2+1, x1:x2+1]
        if crop.size == 0:
            crop = img

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crops.append(_VGG_TRANSFORM(crop_rgb))

    feats = []
    use_amp = bool(amp and _VGG_DEVICE.type == "cuda")
    for i in range(0, len(crops), batch_size):
        batch = torch.stack(crops[i:i + batch_size]).to(_VGG_DEVICE)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                feat = _VGG(batch)
        else:
            feat = _VGG(batch)
        feats.append(feat.float().cpu())

    return torch.cat(feats, dim=0).numpy().astype(np.float32)


# -----------------------------------------------------------------------------
# RAFT Optical Flow (small/large + selectable weights)
# -----------------------------------------------------------------------------

_RAFT = None
_RAFT_DEVICE = None
_RAFT_VARIANT = None
_RAFT_WEIGHTS_NAME = None

def _resolve_raft_weights(variant: str, weights_name: str):
    """
    Torchvision RAFT-large includes KITTI-finetuned weights (best for driving):
      Raft_Large_Weights.C_T_SKHT_K_V2
    If your installed torchvision doesn't have that enum, we fall back to DEFAULT.
    """
    if variant == "large":
        W = Raft_Large_Weights
        default_name = "C_T_SKHT_K_V2"
    else:
        W = Raft_Small_Weights
        default_name = "C_T_V2"

    name = weights_name or default_name
    if hasattr(W, name):
        return getattr(W, name), name

    print(f"[WARN] RAFT weights '{name}' not found for variant='{variant}'. Falling back to DEFAULT.")
    return W.DEFAULT, "DEFAULT"

def init_raft(variant: str, weights_name: str):
    global _RAFT, _RAFT_DEVICE, _RAFT_VARIANT, _RAFT_WEIGHTS_NAME
    if _RAFT is not None and _RAFT_VARIANT == variant and _RAFT_WEIGHTS_NAME == weights_name:
        return

    _RAFT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights, resolved = _resolve_raft_weights(variant, weights_name)

    if variant == "large":
        _RAFT = raft_large(weights=weights).to(_RAFT_DEVICE).eval()
    else:
        _RAFT = raft_small(weights=weights).to(_RAFT_DEVICE).eval()

    _RAFT_VARIANT = variant
    _RAFT_WEIGHTS_NAME = resolved
    print(f"[INFO] RAFT-{variant} weights={resolved} on {_RAFT_DEVICE}")

def _to_raft_tensor(img_bgr: np.ndarray, out_hw: Tuple[int, int]) -> torch.Tensor:
    H, W = out_hw
    img = cv2.resize(img_bgr, (W, H), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img).permute(2, 0, 1).contiguous().float()  # 0..255
    t = (t / 255.0) * 2.0 - 1.0  # [-1,1] (matches torchvision RAFT transforms behaviour)
    return t

@torch.inference_mode()
def extract_flow(
    frames: List[np.ndarray],
    boxes: np.ndarray,
    scale=1.2,
    flow_size=224,
    batch_size=8,
    raft_variant="large",
    raft_weights="C_T_SKHT_K_V2",
    amp=False,
):
    init_raft(raft_variant, raft_weights)

    T = len(frames)
    Hf = Wf = int(flow_size)
    if T == 0:
        return np.zeros((0, 2, Hf, Wf), dtype=np.float32)
    if T == 1:
        return np.zeros((1, 2, Hf, Wf), dtype=np.float32)

    crops = []
    for img, (x1, y1, x2, y2) in zip(frames, boxes):
        h, w = img.shape[:2]
        x1, y1, x2, y2 = expand_box(x1, y1, x2, y2, w, h, scale)
        crop = img[y1:y2+1, x1:x2+1]
        if crop.size == 0:
            crop = img
        crops.append(crop)

    t_all = [_to_raft_tensor(c, (Hf, Wf)) for c in crops]
    t_prev = t_all[:-1]
    t_curr = t_all[1:]

    flows_out = [np.zeros((2, Hf, Wf), dtype=np.float32)]  # flow at t=0

    use_amp = bool(amp and _RAFT_DEVICE.type == "cuda")

    for i in range(0, len(t_prev), batch_size):
        b1 = torch.stack(t_prev[i:i+batch_size], dim=0).to(_RAFT_DEVICE)
        b2 = torch.stack(t_curr[i:i+batch_size], dim=0).to(_RAFT_DEVICE)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                flows = _RAFT(b1, b2)
        else:
            flows = _RAFT(b1, b2)

        flow_b = flows[-1]  # (B,2,H,W)

        flows_out.extend([f.float().cpu().numpy().astype(np.float32) for f in flow_b])

    return np.stack(flows_out, axis=0).astype(np.float32)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser("Step 2: Local Features – JAAD (VGG + RAFT Flow)")
    ap.add_argument("--npz_root", required=True)
    ap.add_argument("--jaad_root", required=True)
    ap.add_argument("--splits", default="train,val,test")

    ap.add_argument("--scale_crop", type=float, default=1.2)
    ap.add_argument("--vgg_batch", type=int, default=8)

    ap.add_argument("--no_flow", action="store_true")
    ap.add_argument("--flow_size", type=int, default=224)
    ap.add_argument("--flow_batch", type=int, default=8)

    ap.add_argument("--raft_variant", choices=["small", "large"], default="large")
    ap.add_argument("--raft_weights", type=str, default="C_T_SKHT_K_V2")

    ap.add_argument("--amp", action="store_true", help="Use autocast on CUDA (faster, lower VRAM).")

    ap.add_argument("--store_fp16", action="store_true",
                    help="Store local_cnn/local_motion as fp16 on disk (loader can upcast later).")

    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--debug", type=int, default=0)
    ap.add_argument("--force", action="store_true")

    args = ap.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    jaad_root = Path(args.jaad_root)

    print("=" * 70)
    print("Step 2: LOCAL FEATURES – JAAD (VGG19 + RAFT)")
    print("=" * 70)
    print(f"NPZ root : {args.npz_root}")
    print(f"JAAD root: {jaad_root}")
    print(f"Splits   : {splits}")
    print(f"Flow: {'disabled' if args.no_flow else 'enabled'} | flow_size={args.flow_size} | flow_batch={args.flow_batch}")
    print(f"RAFT: {args.raft_variant} | weights={args.raft_weights} | amp={args.amp}")
    print("=" * 70)

    vc = VideoCache(jaad_root)

    try:
        for split in splits:
            split_dir = Path(args.npz_root) / split
            if not split_dir.exists():
                print(f"[WARN] {split_dir} not found")
                continue

            npz_files = sorted(split_dir.glob("seq_*.npz"))
            if args.limit > 0:
                npz_files = npz_files[:args.limit]

            print(f"\n[INFO] Processing {split}: {len(npz_files)} files")

            updated = 0
            for npz_path in npz_files:
                data = dict(np.load(npz_path, allow_pickle=True))

                if (not args.force) and data.get("_stage_local", np.int32(0)) == 1:
                    if args.debug > 0:
                        print(f"  [SKIP] {npz_path.name} already processed")
                    continue

                video_name = str(data["video_name"][0]) if getattr(data["video_name"], "ndim", 1) != 0 else str(data["video_name"])

                frame_idx = data["frame_idx"]
                boxes = data["bbox_xyxy"]
                mp4_off = int(data.get("mp4_png_offset", 0))

                try:
                    frames = [vc.read_frame(video_name, int(fi), mp4_off) for fi in frame_idx]
                except Exception as e:
                    print(f"  [ERR] {npz_path.name}: {e}")
                    continue

                local_cnn = extract_vgg(frames, boxes, args.scale_crop, args.vgg_batch, amp=args.amp)

                if not args.no_flow:
                    local_motion = extract_flow(
                        frames, boxes,
                        scale=args.scale_crop,
                        flow_size=args.flow_size,
                        batch_size=args.flow_batch,
                        raft_variant=args.raft_variant,
                        raft_weights=args.raft_weights,
                        amp=args.amp,
                    )
                else:
                    T = len(frames)
                    local_motion = np.zeros((T, 2, args.flow_size, args.flow_size), dtype=np.float32)

                dtype = np.float16 if args.store_fp16 else np.float32
                data["local_cnn"] = local_cnn.astype(dtype, copy=False)
                data["local_motion"] = local_motion.astype(dtype, copy=False)
                data["_stage_local"] = np.int32(1)

                data["_vgg_weights"] = np.array(["IMAGENET1K_V1"], dtype=object)
                data["_raft_variant"] = np.array([args.raft_variant], dtype=object)
                data["_raft_weights"] = np.array([args.raft_weights], dtype=object)
                data["_flow_size"] = np.int32(args.flow_size)

                np.savez_compressed(npz_path, **data)
                updated += 1

                if args.debug > 0 and updated <= args.debug:
                    print(f"  [{updated}] {npz_path.name}")

            print(f"[OK] {split}: updated {updated} files")

    finally:
        vc.release_all()

    print("\n[DONE] Step 2 complete!")


if __name__ == "__main__":
    main()