#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import argparse
import numpy as np
import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


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
        raise RuntimeError(f"Cannot open video: {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx) + int(off))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_idx} from {path}")
    return frame


def denorm_bbox(bb, W, H):
    x1, y1, x2, y2 = bb
    return int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)


def load_predictions(pred_csv):
    """
    Returns dict: basename(npz) -> probability.
    Accepted columns:
      npz/file/seq/seq_name/path and prob/p/pred_prob/main_prob
    """
    if pred_csv is None:
        return {}

    preds = {}
    with open(pred_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []

        key_col = None
        for c in ["npz", "file", "seq", "seq_name", "path"]:
            if c in cols:
                key_col = c
                break

        prob_col = None
        for c in ["prob", "p", "pred_prob", "main_prob", "prediction"]:
            if c in cols:
                prob_col = c
                break

        if key_col is None or prob_col is None:
            raise ValueError(
                f"CSV must contain an npz/path column and probability column. "
                f"Found columns: {cols}"
            )

        for row in reader:
            key = os.path.basename(row[key_col])
            preds[key] = float(row[prob_col])

    return preds


def compute_crop_window_from_bboxes(bbox_seq, W, H, margin=4.0, min_w=420, aspect=16/9):
    """
    Fixed crop window for the whole row, based on all observation bboxes.
    This makes the pedestrian motion easier to see across frames.
    """
    xs1 = bbox_seq[:, 0] * W
    ys1 = bbox_seq[:, 1] * H
    xs2 = bbox_seq[:, 2] * W
    ys2 = bbox_seq[:, 3] * H

    x1, y1 = xs1.min(), ys1.min()
    x2, y2 = xs2.max(), ys2.max()

    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    crop_w = max(min_w, bw * margin)
    crop_h = max(crop_w / aspect, bh * margin)

    # enforce aspect ratio
    crop_w = max(crop_w, crop_h * aspect)
    crop_h = crop_w / aspect

    x0 = int(round(cx - crop_w / 2))
    y0 = int(round(cy - crop_h / 2))
    x1c = int(round(cx + crop_w / 2))
    y1c = int(round(cy + crop_h / 2))

    # clamp to image bounds while preserving size as much as possible
    if x0 < 0:
        x1c -= x0
        x0 = 0
    if y0 < 0:
        y1c -= y0
        y0 = 0
    if x1c > W:
        shift = x1c - W
        x0 -= shift
        x1c = W
    if y1c > H:
        shift = y1c - H
        y0 -= shift
        y1c = H

    x0 = max(0, x0)
    y0 = max(0, y0)
    x1c = min(W, x1c)
    y1c = min(H, y1c)

    return x0, y0, x1c, y1c


def compute_crop_window_from_bboxes(bbox_seq, W, H, margin=2.6, min_w=280, aspect=16/9):
    """
    Fixed target-centered crop for the whole observation window.
    Uses the union of all target bboxes, expanded with margin.
    """
    xs1 = bbox_seq[:, 0] * W
    ys1 = bbox_seq[:, 1] * H
    xs2 = bbox_seq[:, 2] * W
    ys2 = bbox_seq[:, 3] * H

    x1, y1 = xs1.min(), ys1.min()
    x2, y2 = xs2.max(), ys2.max()

    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    crop_w = max(min_w, bw * margin)
    crop_h = max(crop_w / aspect, bh * margin)

    crop_w = max(crop_w, crop_h * aspect)
    crop_h = crop_w / aspect

    x0 = int(round(cx - crop_w / 2))
    y0 = int(round(cy - crop_h / 2))
    x1c = int(round(cx + crop_w / 2))
    y1c = int(round(cy + crop_h / 2))

    if x0 < 0:
        x1c -= x0
        x0 = 0
    if y0 < 0:
        y1c -= y0
        y0 = 0
    if x1c > W:
        shift = x1c - W
        x0 -= shift
        x1c = W
    if y1c > H:
        shift = y1c - H
        y0 -= shift
        y1c = H

    x0 = max(0, x0)
    y0 = max(0, y0)
    x1c = min(W, x1c)
    y1c = min(H, y1c)

    return x0, y0, x1c, y1c


def draw_case_frames(npz_path, pie_root, pred_prob, threshold, n_frames):
    d = np.load(npz_path, allow_pickle=True)

    set_id = str(get_scalar(d, "set_id"))
    video = str(get_scalar(d, "video_name"))
    off = int(get_scalar(d, "mp4_png_offset", 0))
    label = int(get_scalar(d, "label", -1))

    frame_idxs = np.array(d["frame_idx"]).astype(int)
    bbox = d["bbox"].astype(np.float32)

    picks = np.linspace(0, len(frame_idxs) - 1, n_frames).astype(int)

    pred_label = int(pred_prob >= threshold)
    correct = pred_label == label

    if label == 1 and pred_label == 1:
        case_type = "TP"
    elif label == 0 and pred_label == 0:
        case_type = "TN"
    elif label == 0 and pred_label == 1:
        case_type = "FP"
    else:
        case_type = "FN"

    if pred_label == 1:
        color = (0, 220, 0)
        pred_text = "Pred: CROSS"
    else:
        color = (220, 40, 40)
        pred_text = "Pred: NO-CROSS"

    gt_text = "GT: CROSS" if label == 1 else "GT: NO-CROSS"
    status = "OK" if correct else "ERR"

    first_frame = read_frame_bgr(pie_root, set_id, video, int(frame_idxs[0]), off)
    H, W = first_frame.shape[:2]

    cx0, cy0, cx1, cy1 = compute_crop_window_from_bboxes(
        bbox_seq=bbox,
        W=W,
        H=H,
        margin=1.5,
        min_w=220,
        aspect=16/9,
    )

    centers = []
    for bb in bbox:
        x1, y1, x2, y2 = denorm_bbox(bb, W, H)
        centers.append((int(0.5 * (x1 + x2)) - cx0, int(0.5 * (y1 + y2)) - cy0))

    images = []
    titles = []

    for idx_pos, j in enumerate(picks):
        fidx = int(frame_idxs[j])
        frame = read_frame_bgr(pie_root, set_id, video, fidx, off)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).copy()

        crop = rgb[cy0:cy1, cx0:cx1].copy()

        x1, y1, x2, y2 = denorm_bbox(bbox[j], W, H)
        bx1 = max(0, min(crop.shape[1] - 1, x1 - cx0))
        bx2 = max(0, min(crop.shape[1] - 1, x2 - cx0))
        by1 = max(0, min(crop.shape[0] - 1, y1 - cy0))
        by2 = max(0, min(crop.shape[0] - 1, y2 - cy0))

        # trajectory line of target center
        for a, b in zip(centers[:-1], centers[1:]):
            if 0 <= a[0] < crop.shape[1] and 0 <= a[1] < crop.shape[0] and \
               0 <= b[0] < crop.shape[1] and 0 <= b[1] < crop.shape[0]:
                cv2.line(crop, a, b, (255, 215, 0), 2)

        cur = centers[j]
        if 0 <= cur[0] < crop.shape[1] and 0 <= cur[1] < crop.shape[0]:
            cv2.circle(crop, cur, 5, (255, 255, 255), -1)

        cv2.rectangle(crop, (bx1, by1), (bx2, by2), color, 3)

        crop = cv2.resize(crop, (360, 203), interpolation=cv2.INTER_AREA)
        images.append(crop)

        if idx_pos == 0:
            titles.append("start")
        elif idx_pos == len(picks) - 1:
            titles.append("end")
        else:
            titles.append("mid")

    meta = {
        "npz": os.path.basename(npz_path),
        "set_video": f"{set_id}/{video}",
        "label": label,
        "pred_prob": pred_prob,
        "pred_label": pred_label,
        "correct": correct,
        "gt_text": gt_text,
        "pred_text": pred_text,
        "status": status,
        "case_type": case_type,
    }

    d.close()
    return images, titles, meta


def main():
    ap = argparse.ArgumentParser("Qualitative prediction grid for PIPN thesis")
    ap.add_argument("--cases", nargs="+", required=True, help="List of NPZ files")
    ap.add_argument("--pie_root", default="/data/PIE/PIE_clips")
    ap.add_argument("--pred_csv", default=None, help="CSV with model predictions")
    ap.add_argument("--threshold", type=float, default=0.7585)
    ap.add_argument("--out_dir", default="/workspace/project/thesis_figs")
    ap.add_argument("--out_name", default="fig_qualitative_predictions.png")
    ap.add_argument("--n_frames", type=int, default=4)
    ap.add_argument(
        "--use_gt_as_pred",
        action="store_true",
        help="Only for layout testing. Do NOT use for final thesis results."
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    preds = load_predictions(args.pred_csv)

    rows = len(args.cases)
    cols = args.n_frames + 1  # last column for text/status

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(3.0 * args.n_frames + 2.3, 2.4 * rows),
        squeeze=False
    )

    for r, npz_path in enumerate(args.cases):
        base = os.path.basename(npz_path)

        # Load GT if needed for layout test
        d_tmp = np.load(npz_path, allow_pickle=True)
        gt_label = int(get_scalar(d_tmp, "label", -1))
        d_tmp.close()

        if base in preds:
            prob = preds[base]
        elif args.use_gt_as_pred:
            prob = 0.90 if gt_label == 1 else 0.10
        else:
            raise ValueError(
                f"No prediction found for {base}. Provide --pred_csv or use "
                f"--use_gt_as_pred only for layout testing."
            )

        images, titles, meta = draw_case_frames(
            npz_path=npz_path,
            pie_root=args.pie_root,
            pred_prob=prob,
            threshold=args.threshold,
            n_frames=args.n_frames,
        )

        for c in range(args.n_frames):
            ax = axes[r, c]
            ax.imshow(images[c])
            ax.axis("off")
            if r == 0:
                ax.set_title(titles[c], fontsize=10)

        # Text/status panel
        ax = axes[r, args.n_frames]
        ax.axis("off")

        color = "green" if meta["correct"] else "red"

        txt = (
            f"{meta['case_type']}\n"
            f"{meta['set_video']}\n"
            f"{meta['gt_text']}\n"
            f"{meta['pred_text']}\n"
            f"p={meta['pred_prob']:.3f}\n"
            f"{meta['status']}"
        )

        ax.text(
            0.05,
            0.5,
            txt,
            va="center",
            ha="left",
            fontsize=10,
            color=color,
            fontweight="bold",
            transform=ax.transAxes,
        )

    fig.suptitle(
        "Qualitative PIPN predictions at ETC≈0.5s",
        fontsize=13,
        fontweight="bold",
        y=1.02
    )

    plt.tight_layout()
    out = os.path.join(args.out_dir, args.out_name)
    plt.savefig(out)
    plt.close()
    print(f"[OK] saved: {out}")


if __name__ == "__main__":
    main()