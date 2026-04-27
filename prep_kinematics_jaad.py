#!/usr/bin/env python3
"""
Step 1: JAAD Kinematics Preprocessing
Reads directly from MP4 clips (no frame extraction needed).
Creates NPZ files compatible with the PIE-trained model.

Usage (recommended for seq_len=10 like your PIE run):
  python prep_kinematics_jaad.py \
      --jaad_root /data/JAAD \
      --iface_dir /data/jaad_interface \
      --out_root /data/JAAD_PREP_OUT \
      --seq_len 10 \
      --out_stride 2 \
      --tte_min 5 \
      --tte_max 200 \
      --overlap 0.8 \
      --no_pose \
      --splits train,val,test
"""

import sys
import os
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

RAW_FPS = 30


class JAADVideoCache:
    def __init__(self, jaad_root: str):
        self.jaad_root = Path(jaad_root)
        self.current_video = None
        self.cap = None

    def _find_video(self, video_name: str) -> Path:
        candidates = [
            self.jaad_root / "videos" / f"{video_name}.mp4",
            self.jaad_root / "JAAD_clips" / f"{video_name}.mp4",
            self.jaad_root / f"{video_name}.mp4",
            self.jaad_root / "clips" / f"{video_name}.mp4",
        ]
        for c in candidates:
            if c.exists():
                return c
        raise FileNotFoundError(f"Cannot find {video_name}.mp4.")

    def get_cap(self, video_name: str):
        if self.current_video != video_name:
            if self.cap is not None:
                self.cap.release()
            path = self._find_video(video_name)
            self.cap = cv2.VideoCapture(str(path))
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open: {path}")
            self.current_video = video_name
        return self.cap

    def read_frame(self, video_name: str, frame_idx: int):
        cap = self.get_cap(video_name)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed frame {frame_idx} from {video_name}")
        return frame

    def release_all(self):
        if self.cap is not None:
            self.cap.release()


_POSE_MODEL = None
_POSE_DEVICE = None

def init_pose(model_path):
    global _POSE_MODEL, _POSE_DEVICE
    if _POSE_MODEL is not None: return
    try:
        from ultralytics import YOLO
        _POSE_MODEL = YOLO(model_path)
        _POSE_DEVICE = 0 if torch.cuda.is_available() else "cpu"
        print(f"[INFO] YOLO-pose on {_POSE_DEVICE}")
    except Exception as e:
        print(f"[WARN] YOLO init failed: {e}")

def expand_box(x1, y1, x2, y2, w, h, scale=1.2):
    cx, cy = (x1+x2)/2, (y1+y2)/2
    bw, bh = (x2-x1)*scale, (y2-y1)*scale
    return max(0,int(cx-bw/2)), max(0,int(cy-bh/2)), min(w-1,int(cx+bw/2)), min(h-1,int(cy+bh/2))

def extract_pose(frames, boxes, model_path, scale=1.2, imgsz=512):
    init_pose(model_path)
    if _POSE_MODEL is None:
        return np.zeros((len(frames), 34), dtype=np.float32)
    all_kps = []
    for img, (x1,y1,x2,y2) in zip(frames, boxes):
        h, w = img.shape[:2]
        ex1,ey1,ex2,ey2 = expand_box(int(x1),int(y1),int(x2),int(y2), w, h, scale)
        crop = img[ey1:ey2, ex1:ex2]
        if crop.size == 0: crop = img
        res = _POSE_MODEL.predict(crop, imgsz=imgsz, verbose=False, device=_POSE_DEVICE)
        if len(res) and res[0].keypoints is not None and res[0].keypoints.xy is not None and res[0].keypoints.xy.shape[0]>0:
            xy = res[0].keypoints.xy[0].cpu().numpy().astype(np.float32)
            xy[:,0] = (ex1 + xy[:,0]) / max(w,1)
            xy[:,1] = (ey1 + xy[:,1]) / max(h,1)
        else:
            xy = np.zeros((17,2), dtype=np.float32)
        all_kps.append(xy)
    return np.stack(all_kps).reshape(len(frames), -1)


def save_npz(path, feats, meta):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    seq_len = meta["seq_len"]
    d = {
        "bbox": feats["bbox"].astype(np.float32),
        "bbox_xyxy": feats["bbox_xyxy"].astype(np.float32),
        "pose": feats["pose"].astype(np.float32),
        "speed": feats["speed"].astype(np.float32),
        "local_cnn": np.zeros((seq_len,512,7,7), dtype=np.float32),
        "local_motion": np.zeros((seq_len,2,224,224), dtype=np.float32),
        "sem_labels": np.zeros((seq_len,1,1), dtype=np.uint16),
        "cat_depth": np.zeros((seq_len,2,1,1), dtype=np.float32),
        "label": np.int32(meta["label"]),
        "intention_prob_seq": np.asarray(meta["intention_prob_seq"], dtype=np.float32),
        "intent_label_track": np.int32(meta["intent_label_track"]),
        "intent_prob_track_max": np.float32(meta["intent_prob_track_max"]),
        "cross_label_track": np.int32(meta["cross_label_track"]),
        "pid": np.array([meta["pid"]], dtype=object),
        "set_id": np.array(["jaad"], dtype=object),
        "video_name": np.array([meta["video_name"]], dtype=object),
        "track_id": np.array([meta["track_id"]], dtype=object),
        "img_paths": np.array(meta["img_paths"], dtype=object),
        "frame_idx": np.asarray(meta["frame_idx"], dtype=np.int32),
        "t_start_ds": np.int32(0), "t_end_ds": np.int32(seq_len-1),
        "t_start_full": np.int32(0), "t_end_full": np.int32(seq_len-1),
        "event_frame": np.int32(meta.get("event_frame",-1)),
        "pseudo_event_frame": np.int32(meta.get("pseudo_event_frame",-1)),
        "first_cross_full": np.int32(meta.get("first_cross_full",-1)),
        "tte_sec_target": np.float32(meta.get("tte_sec_target",1.0)),
        "tte_sec_actual": np.float32(meta.get("tte_sec_actual",1.0)),
        "data_fstride": np.int32(1),
        "out_stride": np.int32(meta.get("out_stride",3)),
        "raw_fps": np.float32(RAW_FPS),
        "eff_fps": np.float32(RAW_FPS / meta.get("out_stride",3)),
        "seq_len": np.int32(seq_len),
        "obs_sec": np.float32(seq_len / (RAW_FPS / meta.get("out_stride",3))),
        "split": np.array([meta["split"]], dtype=object),
        "dataset": np.array(["jaad"], dtype=object),
    }
    np.savez_compressed(str(path), **d)


def load_jaad(jaad_interface_path, jaad_root):
    sys.path.insert(0, jaad_interface_path)
    from jaad_data import JAAD
    jaad = JAAD(data_path=jaad_root)
    db = jaad.generate_database()
    return jaad, db


def load_splits(jaad_interface_path):
    split_dir = os.path.join(jaad_interface_path, "split_ids")
    splits = {}
    for name in ["train", "val", "test"]:
        for fname in [f"all_videos_{name}.txt", f"{name}.txt", f"{name}_ids.txt"]:
            fpath = os.path.join(split_dir, fname)
            if os.path.exists(fpath):
                with open(fpath) as f:
                    splits[name] = [l.strip() for l in f if l.strip()]
                print(f"  {name}: {len(splits[name])} videos (from {fname})")
                break
        if name not in splits:
            print(f"  [WARN] No split file for '{name}' in {split_dir}")
            if os.path.exists(split_dir):
                print(f"         Files: {os.listdir(split_dir)}")
            splits[name] = []
    return splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jaad_root", required=True)
    ap.add_argument("--iface_dir", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--seq_len", type=int, default=10)
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--overlap", type=float, default=0.8)
    ap.add_argument("--tte_min", type=int, default=30)
    ap.add_argument("--tte_max", type=int, default=60)
    ap.add_argument("--out_stride", type=int, default=3)
    ap.add_argument("--pose_model", default="yolov8n-pose.pt")
    ap.add_argument("--no_pose", action="store_true")
    ap.add_argument("--debug", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    print("="*70)
    print("JAAD Kinematics (reads MP4 clips directly)")
    print("="*70)
    print(f"Root:      {args.jaad_root}")
    print(f"Interface: {args.iface_dir}")
    print(f"Output:    {args.out_root}")
    print(f"Seq len:   {args.seq_len}, stride: {args.out_stride}")
    print(f"Speed:     ZEROS (no OBD in JAAD)")
    print("="*70)

    print("\nLoading JAAD database...")
    jaad, db = load_jaad(args.iface_dir, args.jaad_root)
    print(f"  Videos: {len(db)}")

    print("\nLoading splits...")
    splits = load_splits(args.iface_dir)

    vc = JAADVideoCache(args.jaad_root)

    try:
        for split_name in [s.strip() for s in args.splits.split(",")]:
            if not splits.get(split_name):
                print(f"\n[SKIP] {split_name}")
                continue

            print(f"\n{'='*60}")
            print(f"{split_name}: {len(splits[split_name])} videos")
            print(f"{'='*60}")

            out_dir = Path(args.out_root) / split_name
            out_dir.mkdir(parents=True, exist_ok=True)
            written, pos, neg, skip = 0, 0, 0, 0

            for video_name in splits[split_name]:
                if video_name not in db:
                    skip += 1; continue
                try:
                    vc.get_cap(video_name)
                except FileNotFoundError:
                    skip += 1; continue

                ped_anns = db[video_name].get('ped_annotations', {})

                for pid, pd in ped_anns.items():
                    frames = np.asarray(pd.get('frames',[]), dtype=np.int32)
                    bboxes = np.asarray(pd.get('bbox',[]), dtype=np.float32)

                    # === FIXED: JAAD stores 'cross' inside 'behavior' dictionary ===
                    behavior = pd.get('behavior', {})
                    actions = behavior.get('cross', behavior.get('action', []))

                    if len(frames)==0 or len(bboxes)==0 or len(actions)==0:
                        continue

                    T0 = min(len(frames), len(bboxes), len(actions))
                    frames = frames[:T0]
                    bboxes = bboxes[:T0]
                    actions = np.asarray(actions)[:T0]

                    if actions.dtype.kind in ('U','S','O'):
                        cmask = np.array([str(a).lower() in ('crossing','cross') for a in actions])
                    else:
                        cmask = (actions == 1)

                    cross_label = 1 if cmask.any() else 0
                    cidx = np.where(cmask)[0]
                    event_frame = int(frames[cidx[0]]) if len(cidx) else int(frames[-1])
                    first_cross = int(cidx[0]) if len(cidx) else -1

                    # Downsample
                    keep = np.arange(0, T0, args.out_stride, dtype=np.int32)
                    if len(keep) < args.seq_len:
                        continue
                    f_ds = frames[keep]
                    b_ds = bboxes[keep]

                    # Sliding window (correct version)
                    step = max(1, int(args.seq_len * (1.0 - args.overlap)))
                    for start in range(0, len(f_ds) - args.seq_len + 1, step):
                        end = start + args.seq_len
                        wf = f_ds[start:end]
                        wb = b_ds[start:end]

                        tte = event_frame - wf[-1]
                        if tte < args.tte_min or tte > args.tte_max:
                            continue

                        try:
                            imgs = [vc.read_frame(video_name, int(fi)) for fi in wf]
                        except Exception as e:
                            continue

                        # Normalize bbox
                        bn = []
                        for fr, (x1,y1,x2,y2) in zip(imgs, wb):
                            h, w = fr.shape[:2]
                            if x2 < x1: x1, x2 = x2, x1
                            if y2 < y1: y1, y2 = y2, y1
                            bn.append([np.clip(x1,0,w-1)/w, np.clip(y1,0,h-1)/h,
                                       np.clip(x2,0,w-1)/w, np.clip(y2,0,h-1)/h])
                        bn = np.asarray(bn, dtype=np.float32)

                        if not args.no_pose:
                            pose = extract_pose(imgs, wb, args.pose_model)
                        else:
                            pose = np.zeros((args.seq_len,34), dtype=np.float32)

                        ip = [f"images/{video_name}/{int(fi):05d}.png" for fi in wf]

                        save_npz(out_dir / f"seq_{written:06d}.npz",
                            {"bbox": bn, "bbox_xyxy": wb, "pose": pose,
                             "speed": np.zeros((args.seq_len,1),dtype=np.float32)},
                            {"label": cross_label,
                             "intention_prob_seq": np.full(args.seq_len, float(cross_label)),
                             "intent_label_track": cross_label,
                             "intent_prob_track_max": float(cross_label),
                             "cross_label_track": cross_label,
                             "pid": pid,
                             "video_name": video_name,
                             "track_id": f"jaad/{video_name}/{pid}",
                             "img_paths": ip,
                             "frame_idx": wf.tolist(),
                             "event_frame": event_frame,
                             "pseudo_event_frame": event_frame if cross_label == 0 else -1,
                             "first_cross_full": first_cross,
                             "tte_sec_target": args.tte_min / RAW_FPS,
                             "tte_sec_actual": tte / RAW_FPS,
                             "out_stride": args.out_stride,
                             "seq_len": args.seq_len,
                             "split": split_name})

                        written += 1
                        if cross_label == 1: pos += 1
                        else: neg += 1
                        if written % 200 == 0:
                            print(f"    {written} (pos={pos}, neg={neg})")
                        if args.limit > 0 and written >= args.limit: break
                    if args.limit > 0 and written >= args.limit: break
                if args.limit > 0 and written >= args.limit: break

            print(f"  [OK] {split_name}: {written} (pos={pos}, neg={neg}, skip={skip})")
    finally:
        vc.release_all()

    print(f"\n[DONE] Output: {args.out_root}")
    print(f"Next: prep_local.py → prep_scene.py → eval_jaad.py")

if __name__ == "__main__":
    main()