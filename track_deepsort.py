"""
deep_sort_tracker.py
--------------------

Run DeepSORT tracking on YOLO-detected frames for occupancy measurement.

This script implements the DeepSORT tracking pipeline described in:

"Experimental Study on Surveillance Video-Based Indoor Occupancy Measurement 
for Occupant-Centric Control" (citation will be available after publication).

Outputs per video:
- tracks_deepsort.csv : frame_idx, timestamp_sec, track_id, x1, y1, x2, y2, score
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

# Try OpenCV first
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Fallback to PIL
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from deep_sort_realtime.deepsort_tracker import DeepSort

# ----------------------- Defaults -----------------------
OUT_NAME = "tracks_deepsort.csv"
EMBEDDER = "mobilenet"
MAX_AGE = 30
N_INIT = 3
MIN_DET_CONF = 0.0

# ----------------------- Helpers -----------------------
def read_image(path: str):
    if HAS_CV2:
        return cv2.imread(path)
    if HAS_PIL:
        img = Image.open(path).convert("RGB")
        return np.array(img)[:, :, ::-1].copy()  # convert RGB->BGR
    raise RuntimeError("Neither cv2 nor PIL is available.")

def safe_json_loads(x):
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return x
    try:
        return json.loads(str(x))
    except Exception:
        return []

def find_video_dirs(root: Path, date_filter: str | None = None):
    vids = []
    if not root.exists():
        return vids
    for date_dir in sorted(root.iterdir()):
        if not date_dir.is_dir():
            continue
        if date_filter and date_dir.name != date_filter:
            continue
        for vid_dir in sorted(date_dir.iterdir()):
            if not vid_dir.is_dir():
                continue
            frames_dir = vid_dir / "frames"
            det_csv = vid_dir / "yolo_detections.csv"
            if frames_dir.exists() and det_csv.exists():
                vids.append(vid_dir)
    return vids

# ----------------------- Tracking -----------------------
def run_deepsort_on_video(video_dir: Path, tracker: DeepSort, verbose: bool = False):
    det_csv = video_dir / "yolo_detections.csv"
    out_csv = video_dir / OUT_NAME

    df = pd.read_csv(det_csv)
    if df.empty:
        pd.DataFrame(columns=[
            "frame_idx", "timestamp_sec", "track_id", "x1", "y1", "x2", "y2", "score"
        ]).to_csv(out_csv, index=False)
        return True

    df = df.sort_values("frame_idx").reset_index(drop=True)
    rows_out = []

    # Reset tracker per video
    tracker.tracker.tracks = []
    tracker.tracker._next_id = 1

    for _, r in df.iterrows():
        frame_path = str(r["frame_path"])
        frame_idx = int(r["frame_idx"])
        timestamp_sec = float(r.get("timestamp_sec", np.nan)) if "timestamp_sec" in r else None
        boxes = safe_json_loads(r["json_boxes"])

        detections = []
        for b in boxes:
            try:
                x1 = float(b["x1"]); y1 = float(b["y1"])
                x2 = float(b["x2"]); y2 = float(b["y2"])
                conf = float(b.get("conf", 0.0))
                if conf < MIN_DET_CONF:
                    continue
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                if w <= 1 or h <= 1:
                    continue
                detections.append(([x1, y1, w, h], conf, "person"))
            except Exception:
                continue

        try:
            frame_img = read_image(frame_path)
        except Exception:
            if verbose:
                print(f"[WARN] Cannot read frame: {frame_path}")
            continue

        tracks = tracker.update_tracks(detections, frame=frame_img)
        for t in tracks:
            if not t.is_confirmed():
                continue
            l, ttop, rgt, btm = t.to_ltrb()
            score = getattr(t, "det_conf", None)
            rows_out.append({
                "frame_idx": frame_idx,
                "timestamp_sec": timestamp_sec,
                "track_id": int(t.track_id),
                "x1": float(l), "y1": float(ttop),
                "x2": float(rgt), "y2": float(btm),
                "score": float(score) if score is not None else None
            })

    pd.DataFrame(rows_out, columns=[
        "frame_idx", "timestamp_sec", "track_id", "x1", "y1", "x2", "y2", "score"
    ]).to_csv(out_csv, index=False)
    return True

# ----------------------- Main -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Processed root folder")
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--limit_videos", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    tracker = DeepSort(max_age=MAX_AGE, n_init=N_INIT, embedder=EMBEDDER)

    video_dirs = find_video_dirs(root, args.date)
    if args.limit_videos:
        video_dirs = video_dirs[:args.limit_videos]

    for vid_dir in tqdm(video_dirs, desc="DeepSORT", unit="video"):
        try:
            run_deepsort_on_video(vid_dir, tracker, verbose=args.verbose)
        except Exception as e:
            if args.verbose:
                print(f"[ERROR] {vid_dir} | {e}")

    print(f"[DONE] DeepSORT complete. Output per video: <video_dir>/{OUT_NAME}")


if __name__ == "__main__":
    main()
