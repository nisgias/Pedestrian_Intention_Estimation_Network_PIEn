# data/pie.py
import os
import glob
import json
import numpy as np
import torch
from torch.utils.data import Dataset


class PIESeqDataset(Dataset):
    """
    Loader for precomputed PIE sequences (.npz).
    Compatible with prep_pie_direct.py output.
    """

    def __init__(
        self,
        root_dir,
        split="train",
        seq_len=16,
        mode="train",
        strict_len=True,
        return_meta=False,
        allow_pickle=True,

        # speed
        speed_norm="none",           # "none" | "minmax"
        speed_stats_path=None,       # "/workspace/project/data/speed_stats_splits.json"
        speed_scope="global",        # "global" recommended

        # motion
        motion_norm="none",          # "none" | "p99abs"
        motion_stats_path=None,      # "/workspace/project/data/motion_stats_splits.json"
        motion_scope="global",       # "global" recommended
        motion_clip=1.0,             # clip after scaling ([-1,1] default)
    ):
        self.seq_len = int(seq_len)
        self.mode = mode
        self.strict_len = bool(strict_len)
        self.return_meta = bool(return_meta)
        self.allow_pickle = bool(allow_pickle)

        # --- speed setup ---
        self.speed_norm = speed_norm
        self.speed_scope = speed_scope
        self.speed_stats = None
        if self.speed_norm != "none":
            if speed_stats_path is None:
                raise ValueError("speed_stats_path is required when speed_norm != 'none'")
            with open(speed_stats_path, "r") as f:
                self.speed_stats = json.load(f)

        # --- motion setup ---
        self.motion_norm = motion_norm
        self.motion_scope = motion_scope
        self.motion_clip = float(motion_clip)
        self.motion_stats = None
        if self.motion_norm != "none":
            if motion_stats_path is None:
                raise ValueError("motion_stats_path is required when motion_norm != 'none'")
            with open(motion_stats_path, "r") as f:
                self.motion_stats = json.load(f)

        # files
        self.split_dir = os.path.join(root_dir, split)
        self.files = sorted(glob.glob(os.path.join(self.split_dir, "*.npz")))
        assert len(self.files) > 0, f"No .npz files found in {self.split_dir}"

            
    def __len__(self):
        return len(self.files)

    def _safe_get_scalar_str(self, data, key: str, default: str = "") -> str:
        if key not in data.files:
            return default
        v = data[key]
        try:
            return str(np.array(v).reshape(-1)[0])
        except Exception:
            return default

    def _safe_get_str_list(self, data, key: str):
        if key not in data.files:
            return []
        arr = data[key]
        try:
            return [str(x) for x in arr.tolist()]
        except Exception:
            try:
                return [str(x) for x in np.array(arr).reshape(-1).tolist()]
            except Exception:
                return []

    def _load_npz(self, path: str):
        data = np.load(path, allow_pickle=self.allow_pickle)

        # ---- required tensors ----
        bbox = data["bbox"].astype(np.float32, copy=False)      # [T,4]
        pose = data["pose"].astype(np.float32, copy=False)      # [T,34]
        speed = data["speed"].astype(np.float32, copy=False)    # [T,1] or (T,)

        # IMPORTANT: keep local features as float32 ALWAYS
        local_cnn = np.asarray(data["local_cnn"])
        if local_cnn.dtype != np.float32:
            local_cnn = local_cnn.astype(np.float32, copy=False)

        local_motion = np.asarray(data["local_motion"])
        if local_motion.dtype != np.float32:
            local_motion = local_motion.astype(np.float32, copy=False)

        if speed.ndim == 1:
            speed = speed[:, None]  # -> [T,1]

        # ---- min-max normalize speed to [0,1] ----
        if self.speed_norm == "minmax":
            if isinstance(self.speed_stats, dict) and "global" in self.speed_stats:
                st = self.speed_stats[self.speed_scope]  # usually "global"
            else:
                st = self.speed_stats

            lo = np.float32(st["min"])
            hi = np.float32(st["max"])
            denom = (hi - lo) + np.float32(1e-6)

            speed = np.clip(speed, lo, hi)
            speed = (speed - lo) / denom

        # ---- local_motion normalization (p99abs), stays float32 ----
        if self.motion_norm == "p99abs":
            st = self.motion_stats[self.motion_scope]  # usually "global"
            scale = np.float32(st["p99_abs"]) + np.float32(1e-6)
            local_motion = np.clip(local_motion / scale, -self.motion_clip, self.motion_clip).astype(np.float32, copy=False)

        # semantic labels (int)
        if "sem_labels" in data.files:
            sem_labels = data["sem_labels"].astype(np.int64, copy=False)  # [T,H,W]
        else:
            sem_labels = np.zeros((bbox.shape[0], 1, 1), dtype=np.int64)

        # cat depth: KEEP STORED DTYPE (fp16 stays fp16)
        if "cat_depth" in data.files:
            cat_depth = np.asarray(data["cat_depth"])
            if cat_depth.dtype not in (np.float16, np.float32):
                cat_depth = cat_depth.astype(np.float32, copy=False)
            # NOTE: if it’s float32 in some files, we do NOT force fp16 here.
            # You said: "only cat_depth I want fp16" -> that is handled by preprocess --fp16.
        else:
            cat_depth = np.zeros((bbox.shape[0], 2, 1, 1), dtype=np.float32)

        # intention prob sequence (check both keys)
        if "intention_prob_seq" in data.files:
            label_prob_seq = data["intention_prob_seq"].astype(np.float32, copy=False).reshape(-1)
        elif "label_prob_seq" in data.files:
            label_prob_seq = data["label_prob_seq"].astype(np.float32, copy=False).reshape(-1)
        else:
            label_prob_seq = np.zeros((bbox.shape[0],), dtype=np.float32)

        # label (binary)
        label_val = float(np.array(data["label"]).reshape(-1)[0])
        label_cls = int(label_val) if label_val in [0, 1] else (1 if label_val >= 0.5 else 0)

        # ---- metadata (optional) ----
        meta = None
        if self.return_meta:
            meta = {
                "pid": self._safe_get_scalar_str(data, "pid", ""),
                "set_id": self._safe_get_scalar_str(data, "set_id", ""),
                "video_name": self._safe_get_scalar_str(data, "video_name", ""),
                "track_id": self._safe_get_scalar_str(data, "track_id", ""),
                "img_paths": self._safe_get_str_list(data, "img_paths"),
                "split": self._safe_get_scalar_str(data, "split", ""),
            }

            for k in [
                "frame_idx", "t_start_ds", "t_end_ds", "t_start_full", "t_end_full",
                "event_frame", "pseudo_event_frame", "first_cross_full",
                "tte_sec_target", "tte_sec_actual", "data_fstride", "out_stride",
                "raw_fps", "eff_fps", "seq_len", "obs_sec",
                "intent_label_track", "cross_label_track", "intent_prob_track_max",
            ]:
                if k in data.files:
                    meta[k] = np.array(data[k]).copy()

            if "bbox_xyxy" in data.files:
                meta["bbox_xyxy"] = data["bbox_xyxy"].astype(np.float32, copy=False)

        return (
            bbox, pose, speed,
            local_cnn, local_motion,
            sem_labels, cat_depth,
            label_cls, label_val, label_prob_seq,
            meta
        )


    def __getitem__(self, idx):
        path = self.files[idx]
        (
            bbox, pose, speed,
            local_cnn, local_motion,
            sem_labels, cat_depth,
            label_cls, label_prob, label_prob_seq,
            meta
        ) = self._load_npz(path)

        T = int(bbox.shape[0])
        if self.strict_len and (T != self.seq_len):
            raise ValueError(
                f"[STRICT] {os.path.basename(path)} has T={T}, expected seq_len={self.seq_len}. Fix in prep."
            )

        # convert to torch tensors
        bbox = torch.from_numpy(bbox)
        pose = torch.from_numpy(pose)
        speed = torch.from_numpy(speed)
        local_cnn = torch.from_numpy(local_cnn)
        local_motion = torch.from_numpy(local_motion)
        sem_labels = torch.from_numpy(sem_labels).long()
        cat_depth = torch.from_numpy(cat_depth)
        label_prob_seq = torch.from_numpy(label_prob_seq)

        label = torch.tensor(label_cls, dtype=torch.long)
        label_prob = torch.tensor(label_prob, dtype=torch.float32)

        out = {
            "bbox": bbox,
            "pose": pose,
            "speed": speed,
            "local_cnn": local_cnn,
            "local_motion": local_motion,
            "sem_labels": sem_labels,
            "cat_depth": cat_depth,
            "label": label,
            "label_prob": label_prob,
            "label_prob_seq": label_prob_seq,
            "path": path,
        }

        if self.return_meta:
            out["meta"] = meta

        return out
