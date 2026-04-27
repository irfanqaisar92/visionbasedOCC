"""
detect_yolo.py
------------------

YOLOv8-only occupancy counting from precomputed YOLO detections.

This script implements the YOLOv8-only detection pipeline described in:

"Experimental Study on Surveillance Video-Based Indoor Occupancy Measurement
for Occupant-Centric Control"

Method:
- Reads per-frame YOLO detections from yolo_detections.csv
- Counts the number of retained person detections in each frame
- Computes mean confidence per frame
- Outputs yolo_only_counts.csv per video

Output columns:
- video_folder
- date_folder
- frame_idx
- timestamp_sec
- yolo_count
- mean_conf
- status
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


# --------------------------
# Configurable parameters
# --------------------------
OUT_NAME = "yolo_only_counts.csv"
MIN_CONF = 0.0   # Set to 0.5 if you want additional filtering here


def safe_json_loads(x):
    """
    Safely parse the json_boxes column.
    Returns an empty list if parsing fails.
    """
    if pd.isna(x):
        return []

    if isinstance(x, list):
        return x

    try:
        return json.loads(str(x))
    except Exception:
        return []


def find_video_folders(root: Path, date_filter: str | None = None):
    """
    Find all video folders that contain yolo_detections.csv.

    Expected folder structure:
        root/
            date_folder/
                video_folder/
                    frames/
                    yolo_detections.csv
    """
    for date_dir in sorted(root.iterdir()):
        if not date_dir.is_dir():
            continue

        if date_filter and date_dir.name != date_filter:
            continue

        for vid_dir in sorted(date_dir.iterdir()):
            if not vid_dir.is_dir():
                continue

            det_csv = vid_dir / "yolo_detections.csv"

            if det_csv.exists():
                yield vid_dir, det_csv


def run_yolo_only_on_video(vid_dir: Path, det_csv: Path, overwrite: bool = False):
    """
    Run YOLO-only occupancy counting for one video folder.
    """
    out_csv = vid_dir / OUT_NAME

    if out_csv.exists() and not overwrite:
        print(f"[INFO] {vid_dir.name}: {OUT_NAME} already exists, skipping.")
        return out_csv

    df = pd.read_csv(det_csv)

    output_rows = []

    if df.empty:
        print(f"[WARN] {vid_dir.name}: yolo_detections.csv is empty.")
        pd.DataFrame(columns=[
            "video_folder",
            "date_folder",
            "frame_idx",
            "timestamp_sec",
            "yolo_count",
            "mean_conf",
            "status"
        ]).to_csv(out_csv, index=False)
        return out_csv

    df = df.sort_values("frame_idx").reset_index(drop=True)

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"YOLO-only {vid_dir.name}", unit="frame"):
        frame_idx = int(row["frame_idx"])

        if "timestamp_sec" in row and not pd.isna(row["timestamp_sec"]):
            timestamp_sec = float(row["timestamp_sec"])
        else:
            timestamp_sec = np.nan

        boxes = safe_json_loads(row.get("json_boxes", "[]"))

        retained_boxes = []

        for b in boxes:
            try:
                conf = float(b.get("conf", 1.0))

                if conf < MIN_CONF:
                    continue

                # Keep valid person detections.
                # Assumption: yolo_detections.csv already contains only person-class boxes.
                x1 = float(b["x1"])
                y1 = float(b["y1"])
                x2 = float(b["x2"])
                y2 = float(b["y2"])

                if x2 <= x1 or y2 <= y1:
                    continue

                retained_boxes.append({
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "conf": conf
                })

            except Exception:
                continue

        yolo_count = len(retained_boxes)

        if yolo_count > 0:
            mean_conf = float(np.mean([b["conf"] for b in retained_boxes]))
        else:
            mean_conf = 0.0

        status = "occupied" if yolo_count >= 1 else "unoccupied"

        output_rows.append({
            "video_folder": vid_dir.name,
            "date_folder": vid_dir.parent.name,
            "frame_idx": frame_idx,
            "timestamp_sec": timestamp_sec,
            "yolo_count": yolo_count,
            "mean_conf": mean_conf,
            "status": status
        })

    out_df = pd.DataFrame(output_rows)
    out_df.to_csv(out_csv, index=False)

    print(f"[DONE] Saved YOLO-only counts to {out_csv}")
    return out_csv


def main():
    parser = argparse.ArgumentParser(
        description="Run YOLOv8-only occupancy counting from yolo_detections.csv files."
    )

    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root folder containing date folders."
    )

    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Optional: only process this date folder."
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing yolo_only_counts.csv files."
    )

    args = parser.parse_args()

    root = Path(args.root)

    if not root.exists():
        raise FileNotFoundError(f"Root path does not exist: {root}")

    print(f"[INFO] Root: {root}")
    print(f"[INFO] YOLO-only counting script")
    print(f"[INFO] Minimum confidence threshold: {MIN_CONF}")

    video_folders = list(find_video_folders(root, args.date))

    print(f"[INFO] Found {len(video_folders)} video folders with yolo_detections.csv")

    done = 0

    for vid_dir, det_csv in video_folders:
        try:
            run_yolo_only_on_video(
                vid_dir=vid_dir,
                det_csv=det_csv,
                overwrite=args.overwrite
            )
            done += 1
        except Exception as e:
            print(f"[ERROR] Failed processing {vid_dir}: {e}")

    print(f"[DONE] YOLO-only counting finished for {done} videos.")


if __name__ == "__main__":
    main()
