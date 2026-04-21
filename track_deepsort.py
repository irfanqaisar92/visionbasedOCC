import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Try OpenCV first (recommended)
try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

# Fallback to PIL if cv2 not available
try:
    from PIL import Image
    HAS_PIL = True
except Exception:
    HAS_PIL = False

from deep_sort_realtime.deepsort_tracker import DeepSort


# ----------------------------- Defaults -----------------------------
DEFAULT_ROOT = r"D:\PythonCode\FIT622_processed"
OUT_NAME = "tracks_deepsort.csv"

# DeepSORT params (reasonable defaults)
EMBEDDER = "mobilenet"
MAX_AGE = 30
N_INIT = 3

# If your yolo_detections.csv boxes are weak, you can ignore low conf boxes:
MIN_DET_CONF = 0.0  # set e.g. 0.25 if you want to filter
# -------------------------------------------------------------------


def read_image(path: str):
    """Read image from disk into numpy array (BGR for cv2, RGB for PIL fallback)."""
    if HAS_CV2:
        img = cv2.imread(path)
        return img  # BGR
    if HAS_PIL:
        img = Image.open(path).convert("RGB")
        return np.array(img)[:, :, ::-1].copy()  # convert RGB->BGR-like for consistency
    raise RuntimeError("Neither cv2 nor PIL is available. Install opencv-python or pillow.")


def safe_json_loads(x):
    """Parse JSON from a cell; return [] on failure."""
    if pd.isna(x):
        return []
    if isinstance(x, (list, dict)):
        return x if isinstance(x, list) else [x]
    s = str(x).strip()
    if not s:
        return []
    try:
        return json.loads(s)
    except Exception:
        return []


def find_video_dirs(root: Path, date_filter: str | None = None):
    """
    Find all video folders that contain:
      - frames/ directory
      - yolo_detections.csv
    Under structure: root/date_folder/video_id/
    """
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


def run_deepsort_on_video(video_dir: Path, tracker: DeepSort, verbose: bool = False):
    """
    Input:  video_dir/yolo_detections.csv with columns:
              - frame_path
              - frame_idx
              - timestamp_sec (optional)
              - json_boxes (list of {x1,y1,x2,y2,conf})
    Output: video_dir/tracks_deepsort.csv (always overwritten)
    """
    det_csv = video_dir / "yolo_detections.csv"
    out_csv = video_dir / OUT_NAME

    df = pd.read_csv(det_csv)
    if df.empty:
        if verbose:
            print(f"[WARN] Empty detections: {det_csv}")
        # still write empty file to mark processed
        pd.DataFrame(columns=[
            "frame_idx", "timestamp_sec", "track_id", "x1", "y1", "x2", "y2", "score"
        ]).to_csv(out_csv, index=False)
        return True

    # Ensure required columns
    for col in ["frame_path", "frame_idx"]:
        if col not in df.columns:
            raise RuntimeError(f"Missing column '{col}' in {det_csv}")

    if "json_boxes" not in df.columns:
        raise RuntimeError(f"Missing column 'json_boxes' in {det_csv} (expected YOLO boxes).")

    # Sort by frame index to ensure temporal order
    df = df.sort_values("frame_idx").reset_index(drop=True)

    rows_out = []

    # Reset tracker state per video (important!)
    tracker.tracker.tracks = []
    tracker.tracker._next_id = 1

    for _, r in df.iterrows():
        frame_path = str(r["frame_path"])
        frame_idx = int(r["frame_idx"])
        timestamp_sec = None
        if "timestamp_sec" in df.columns and not pd.isna(r.get("timestamp_sec", np.nan)):
            try:
                timestamp_sec = float(r["timestamp_sec"])
            except Exception:
                timestamp_sec = None

        # Load detections list for this frame
        boxes = safe_json_loads(r["json_boxes"])

        detections = []
        for b in boxes:
            # expected keys: x1,y1,x2,y2,conf
            try:
                x1 = float(b["x1"]); y1 = float(b["y1"])
                x2 = float(b["x2"]); y2 = float(b["y2"])
                conf = float(b.get("conf", 0.0))
            except Exception:
                continue

            if conf < MIN_DET_CONF:
                continue

            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            if w <= 1 or h <= 1:
                continue

            # deep_sort_realtime format: ([x, y, w, h], confidence, class_name)
            detections.append(([x1, y1, w, h], conf, "person"))

        # Read image (needed for embedder)
        try:
            frame_img = read_image(frame_path)
        except Exception as e:
            if verbose:
                print(f"[ERROR] Cannot read frame: {frame_path} | {e}")
            continue

        # Update tracker
        tracks = tracker.update_tracks(detections, frame=frame_img)

        # Save active tracks
        for t in tracks:
            if not t.is_confirmed():
                continue

            track_id = int(t.track_id)

            # deep_sort_realtime: to_ltrb() -> left, top, right, bottom
            l, ttop, rgt, btm = t.to_ltrb()

            # Score: best-effort
            score = None
            if hasattr(t, "det_conf") and t.det_conf is not None:
                try:
                    score = float(t.det_conf)
                except Exception:
                    score = None

            rows_out.append({
                "frame_idx": frame_idx,
                "timestamp_sec": timestamp_sec,
                "track_id": track_id,
                "x1": float(l),
                "y1": float(ttop),
                "x2": float(rgt),
                "y2": float(btm),
                "score": score
            })

    out_df = pd.DataFrame(rows_out, columns=[
        "frame_idx", "timestamp_sec", "track_id", "x1", "y1", "x2", "y2", "score"
    ])
    out_df.to_csv(out_csv, index=False)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=DEFAULT_ROOT, help="Processed root, contains date folders")
    ap.add_argument("--date", type=str, default=None, help="Optional: run only one date folder (e.g., 20181030)")
    ap.add_argument("--limit_videos", type=int, default=None, help="Debug: process only first N videos")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = ap.parse_args()

    root = Path(args.root)
    print(f"[INFO] Root: {root}")

    # Initialize DeepSORT once (reused; we reset state per video)
    print(f"[INFO] Initializing DeepSORT with embedder='{EMBEDDER}'")
    tracker = DeepSort(
        max_age=MAX_AGE,
        n_init=N_INIT,
        embedder=EMBEDDER
    )

    video_dirs = find_video_dirs(root, args.date)
    print(f"[INFO] Found {len(video_dirs)} video folders with frames + yolo_detections.csv.")

    done, failed = 0, 0
    iterable = video_dirs[:args.limit_videos] if args.limit_videos else video_dirs

    for vid_dir in tqdm(iterable, desc="DeepSORT", unit="video"):
        try:
            # ALWAYS overwrite: we simply write tracks_deepsort.csv every time
            ok = run_deepsort_on_video(vid_dir, tracker, verbose=args.verbose)
            done += 1 if ok else 0
        except Exception as e:
            failed += 1
            if args.verbose:
                print(f"[FAIL] {vid_dir} | {e}")

    print(f"[DONE] DeepSORT complete. done={done}, failed={failed}")
    print(f"[INFO] Output per video: <video_dir>\\{OUT_NAME}")


if __name__ == "__main__":
    main()
