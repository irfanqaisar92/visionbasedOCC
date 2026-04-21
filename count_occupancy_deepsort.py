"""
count_occupancy_deepsort.py
----------------------------

Compute per-frame occupancy counts from DeepSORT tracking outputs.

This script implements the DeepSORT tracking pipeline described in:

"Experimental Study on Surveillance Video-Based Indoor Occupancy Measurement 
for Occupant-Centric Control" (citation will be available after publication).

Outputs a CSV per video containing:
- frame_idx
- timestamp_sec (if available)
- occupancy count
- status ("occupied"/"unoccupied")
"""

import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Default output filename and threshold
OUT_NAME = "final_deepsort.csv"
OCCUPIED_THRESH = 1  # occupied if count >= 1


def find_video_dirs(root: Path):
    """Find video directories that contain DeepSORT tracks."""
    video_dirs = []
    for date_dir in sorted(root.iterdir()):
        if not date_dir.is_dir():
            continue
        for vid_dir in sorted(date_dir.iterdir()):
            if (vid_dir / "tracks_deepsort.csv").exists():
                video_dirs.append(vid_dir)
    return video_dirs


def count_occupancy(video_dir: Path, force: bool = False):
    """Build per-frame occupancy counts from DeepSORT tracks."""
    tracks_csv = video_dir / "tracks_deepsort.csv"
    out_csv = video_dir / OUT_NAME

    if out_csv.exists() and not force:
        return "skipped"

    df = pd.read_csv(tracks_csv)
    if df.empty:
        return "failed"

    # Group by frame and count unique track IDs
    occ = (
        df.groupby(["frame_idx", "timestamp_sec"])["track_id"]
        .nunique()
        .reset_index()
        .rename(columns={"track_id": "count"})
    )

    occ["status"] = occ["count"].apply(
        lambda x: "occupied" if x >= OCCUPIED_THRESH else "unoccupied"
    )

    occ = occ.sort_values("frame_idx")
    occ.to_csv(out_csv, index=False)
    return "done"


def main():
    parser = argparse.ArgumentParser(description="Compute per-frame occupancy counts from DeepSORT tracks.")
    parser.add_argument("--root", type=str, required=True,
                        help="Root folder containing date/video directories.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing final_deepsort.csv")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")

    video_dirs = find_video_dirs(root)
    print(f"[INFO] Found {len(video_dirs)} videos with DeepSORT tracks.")

    done = skipped = failed = 0
    for vid_dir in tqdm(video_dirs, desc="DeepSORT occupancy", unit="video"):
        try:
            res = count_occupancy(vid_dir, force=args.force)
            if res == "done":
                done += 1
            elif res == "skipped":
                skipped += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            print(f"[ERROR] {vid_dir.name}: {e}")

    print(f"[DONE] DeepSORT occupancy complete. done={done}, skipped={skipped}, failed={failed}")
    print(f"[INFO] Output per video: <video_dir>/{OUT_NAME}")


if __name__ == "__main__":
    main()
