import os
import json
import argparse
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm
from deep_sort_realtime.deepsort_tracker import DeepSort

# --------------------------
# Default config
# --------------------------
DEFAULT_ROOT = r"D:\PythonCode\FIT622_processed"   # folder with 20181030, 20181102, ...
EMBEDDER     = "mobilenet"
MAX_AGE      = 30
N_INIT       = 3
NN_BUDGET    = None


def find_video_folders(root: Path, date_filter: str | None = None):
    """
    Yield all video folders that contain:
      - a 'frames' subfolder
      - a 'yolo_detections.csv' file

    Structure expected:
      root/
        20181030/
          C2100.../
            frames/
            yolo_detections.csv
        20181102/
          C2100.../
            frames/
            yolo_detections.csv
        ...
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


def run_deepsort_on_video(
    vid_dir: Path,
    frames_dir: Path,
    det_csv: Path,
    tracker: DeepSort
):
    """
    Run DeepSORT for a single video folder.

    Input:
      - frames_dir: contains frame images
      - det_csv: yolo_detections.csv with columns:
          frame_path, frame_idx, timestamp_sec,
          count, mean_conf, box_confs, json_boxes
        where 'json_boxes' is a JSON list of dicts:
          [{"x1":..., "y1":..., "x2":..., "y2":..., "conf":...}, ...]
    Output:
      - tracks_deepsort.csv in vid_dir
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

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=f"Tracking {vid_dir.name}",
        unit="frame"
    ):
        frame_idx = int(row["frame_idx"])
        frame_path = row["frame_path"]

        # If frame_path in CSV is absolute, use it; otherwise build from frames_dir
        frame_path = str(frame_path)
        if not os.path.isabs(frame_path):
            frame_path = str(frames_dir / frame_path)

        img = cv2.imread(frame_path)
        if img is None:
            print(f"[WARN] Could not read frame: {frame_path}")
            continue

        # Parse YOLO detections from json_boxes
        detections = []
        try:
            boxes = json.loads(row["json_boxes"])
        except json.JSONDecodeError:
            boxes = []

        for b in boxes:
            x1 = float(b["x1"])
            y1 = float(b["y1"])
            x2 = float(b["x2"])
            y2 = float(b["y2"])
            conf = float(b["conf"])

            # DeepSORT realtime expects: ( [x1,y1,x2,y2], confidence, class_name )
            detections.append(([x1, y1, x2, y2], conf, "person"))

        tracks = tracker.update_tracks(detections, frame=img)

        for trk in tracks:
            if not trk.is_confirmed():
                continue

            track_id = trk.track_id
            l, t, r, b = trk.to_ltrb()  # left, top, right, bottom

            results.append(
                {
                    "video_folder": vid_dir.name,
                    "date_folder": vid_dir.parent.name,
                    "frame_idx": frame_idx,
                    "track_id": int(track_id),
                    "x1": float(l),
                    "y1": float(t),
                    "x2": float(r),
                    "y2": float(b),
                }
            )

    if results:
        out_df = pd.DataFrame(results)
        out_df.sort_values(["frame_idx", "track_id"], inplace=True)
        out_df.to_csv(out_csv, index=False)
        print(f"[DONE] Saved tracks to {out_csv}")
    else:
        print(f"[WARN] {vid_dir.name}: no tracks produced.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default=DEFAULT_ROOT,
        help="Root folder containing date folders (e.g., 20181030).",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Optional: only process this date folder (e.g., 20181030).",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root path does not exist: {root}")

    print(f"[INFO] Root: {root}")
    print("[INFO] Initializing DeepSORT with embedder='mobilenet'")

    tracker = DeepSort(
        max_age=MAX_AGE,
        n_init=N_INIT,
        nn_budget=NN_BUDGET,
        embedder=EMBEDDER,
        half=True,
        bgr=True,  # OpenCV uses BGR
    )

    video_folders = list(find_video_folders(root, args.date))
    print(f"[INFO] Found {len(video_folders)} video folders with frames + yolo_detections.csv")

    if not video_folders:
        print("[WARN] No valid video folders found. Check folder structure and file names.")
        return

    for vid_dir, frames_dir, det_csv in video_folders:
        print(f"\n[INFO] Processing: {vid_dir}")
        run_deepsort_on_video(vid_dir, frames_dir, det_csv, tracker)


if __name__ == "__main__":
    main()
