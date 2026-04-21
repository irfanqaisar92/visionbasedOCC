"""
track_deepsort.py
-----------------

Run DeepSORT tracking on YOLO-detected frames for occupancy measurement.

This script implements the DeepSORT tracking pipeline described in:

"Experimental Study on Surveillance Video-Based Indoor Occupancy Measurement 
for Occupant-Centric Control" (citation will be available after publication).

Outputs per-video CSVs containing:
- video_folder
- date_folder
- frame_idx
- track_id
- bounding box coordinates (x1, y1, x2, y2)
"""

import os
import json
import argparse
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm
from deep_sort_realtime.deepsort_tracker import DeepSort

# --------------------------
# Configurable parameters
# --------------------------
EMBEDDER = "mobilenet"
MAX_AGE = 30
N_INIT = 3
NN_BUDGET = None


def find_video_folders(root: Path, date_filter: str | None = None):
    """
    Yield all video folders that contain:
      - a 'frames' subfolder
      - a 'yolo_detections.csv' file
    """
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
            if frames_dir.is_dir() and det_csv.exists():
                yield vid_dir, frames_dir, det_csv


def run_deepsort_on_video(vid_dir: Path, frames_dir: Path, det_csv: Path, tracker: DeepSort):
    """
    Run DeepSORT tracking for a single video folder.
    Outputs `tracks_deepsort.csv` in vid_dir.
    """
    out_csv = vid_dir / "tracks_deepsort.csv"
    if out_csv.exists():
        print(f"[INFO] {vid_dir.name}: tracks_deepsort.csv already exists, skipping.")
        return

    df = pd.read_csv(det_csv)
    if df.empty:
        print(f"[WARN] {vid_dir.name}: yolo_detections.csv is empty, skipping.")
        return

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Tracking {vid_dir.name}", unit="frame"):
        frame_idx = int(row["frame_idx"])
        frame_path = str(row["frame_path"])
        if not os.path.isabs(frame_path):
            frame_path = str(frames_dir / frame_path)

        img = cv2.imread(frame_path)
        if img is None:
            print(f"[WARN] Could not read frame: {frame_path}")
            continue

        # Parse YOLO detections
        try:
            boxes = json.loads(row["json_boxes"])
        except json.JSONDecodeError:
            boxes = []

        detections = [([float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])],
                       float(b["conf"]), "person") for b in boxes]

        tracks = tracker.update_tracks(detections, frame=img)
        for trk in tracks:
            if not trk.is_confirmed():
                continue
            l, t, r, b = trk.to_ltrb()
            results.append({
                "video_folder": vid_dir.name,
                "date_folder": vid_dir.parent.name,
                "frame_idx": frame_idx,
                "track_id": int(trk.track_id),
                "x1": float(l),
                "y1": float(t),
                "x2": float(r),
                "y2": float(b),
            })

    if results:
        out_df = pd.DataFrame(results)
        out_df.sort_values(["frame_idx", "track_id"], inplace=True)
        out_df.to_csv(out_csv, index=False)
        print(f"[DONE] Saved tracks to {out_csv}")
    else:
        print(f"[WARN] {vid_dir.name}: no tracks produced.")


def main():
    parser = argparse.ArgumentParser(description="Run DeepSORT tracking on YOLO-detected frames.")
    parser.add_argument("--root", type=str, required=True, help="Root folder containing date folders.")
    parser.add_argument("--date", type=str, default=None, help="Optional: only process this date folder.")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root path does not exist: {root}")

    print(f"[INFO] Root: {root}")
    print(f"[INFO] Initializing DeepSORT with embedder='{EMBEDDER}'")

    tracker = DeepSort(max_age=MAX_AGE, n_init=N_INIT, nn_budget=NN_BUDGET,
                       embedder=EMBEDDER, half=True, bgr=True)

    video_folders = list(find_video_folders(root, args.date))
    print(f"[INFO] Found {len(video_folders)} video folders with frames + yolo_detections.csv")

    for vid_dir, frames_dir, det_csv in video_folders:
        print(f"\n[INFO] Processing: {vid_dir}")
        run_deepsort_on_video(vid_dir, frames_dir, det_csv, tracker)


if __name__ == "__main__":
    main()
