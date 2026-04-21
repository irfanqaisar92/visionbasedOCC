"""
count_occupancy_bytetrack.py
----------------------------

Compute per-frame occupancy counts from ByteTrack outputs aligned with frame manifests.

This script implements the ByteTrack tracking pipeline described in:

"Experimental Study on Surveillance Video-Based Indoor Occupancy Measurement 
for Occupant-Centric Control" (citation will be available after publication).

Outputs a CSV per video containing:
- frame_idx
- frame_path
- timestamp_sec (if available)
- occupancy count
- status ("occupied"/"unoccupied")
- source ("bytetrack")
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

# Default output filename
OUT_NAME = "final_bytetrack.csv"


def find_videos_with_bytetrack(root: Path, date_filter: str | None = None):
    """
    Yield video directories that have both:
    - frames_manifest.csv
    - bytetrack_tracks.csv
    """
    for date_dir in sorted(root.iterdir()):
        if not date_dir.is_dir():
            continue
        if date_filter and date_dir.name != date_filter:
            continue

        for vid_dir in sorted(date_dir.iterdir()):
            if not vid_dir.is_dir():
                continue
            if (vid_dir / "frames_manifest.csv").exists() and (vid_dir / "bytetrack_tracks.csv").exists():
                yield vid_dir


def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def count_occupancy_for_video(video_dir: Path, force: bool = False) -> Path | None:
    """
    Build per-frame occupancy counts from ByteTrack outputs.
    """
    out_csv = video_dir / OUT_NAME
    if out_csv.exists() and not force:
        return out_csv

    df_mf = _safe_read_csv(video_dir / "frames_manifest.csv")
    df_bt = _safe_read_csv(video_dir / "bytetrack_tracks.csv")

    if df_mf.empty or "frame_idx" not in df_mf.columns or "frame_path" not in df_mf.columns:
        return None

    # Prepare manifest
    df_mf = df_mf.copy()
    df_mf["frame_idx"] = pd.to_numeric(df_mf["frame_idx"], errors="coerce").astype("Int64")
    df_mf = df_mf.dropna(subset=["frame_idx"])
    df_mf["frame_idx"] = df_mf["frame_idx"].astype(int)
    df_mf = df_mf.sort_values("frame_idx").reset_index(drop=True)

    if df_bt.empty:
        # All frames zero occupancy
        out = df_mf[["frame_idx", "frame_path"]].copy()
        out["timestamp_sec"] = df_mf.get("timestamp_sec", np.nan)
        out["count"] = 0
        out["status"] = "unoccupied"
        out["source"] = "bytetrack"
        out.to_csv(out_csv, index=False)
        return out_csv

    # Clean ByteTrack tracks
    df_bt = df_bt.copy()
    for col in ["frame_idx", "track_id"]:
        if col not in df_bt.columns:
            return None
        df_bt[col] = pd.to_numeric(df_bt[col], errors="coerce").astype("Int64")
    df_bt = df_bt.dropna(subset=["frame_idx", "track_id"])
    df_bt["frame_idx"] = df_bt["frame_idx"].astype(int)
    df_bt["track_id"] = df_bt["track_id"].astype(int)

    # Count unique track_ids per frame
    df_counts = df_bt.groupby("frame_idx")["track_id"].nunique().reset_index().rename(columns={"track_id": "count"})

    # Merge with manifest
    out = df_mf.merge(df_counts, on="frame_idx", how="left")
    out["count"] = out["count"].fillna(0).astype(int)

    # Timestamp: prefer manifest timestamp if available
    if "timestamp_sec" not in out.columns:
        if "timestamp_sec" in df_bt.columns:
            ts = df_bt.groupby("frame_idx")["timestamp_sec"].first().reset_index()
            out = out.merge(ts, on="frame_idx", how="left")
        else:
            out["timestamp_sec"] = np.nan

    out["status"] = out["count"].apply(lambda x: "occupied" if int(x) >= 1 else "unoccupied")
    out["source"] = "bytetrack"

    cols = ["frame_idx", "frame_path", "timestamp_sec", "count", "status", "source"]
    cols = [c for c in cols if c in out.columns]
    out = out[cols].copy()

    out.to_csv(out_csv, index=False)
    return out_csv


def main():
    parser = argparse.ArgumentParser(description="Compute per-frame occupancy counts from ByteTrack outputs.")
    parser.add_argument("--root", type=str, required=True, help="Processed root folder containing date/video directories.")
    parser.add_argument("--date", type=str, default=None, help="Optional: process only this date folder (e.g., 20231012).")
    parser.add_argument("--force", action="store_true", help="Overwrite existing final_bytetrack.csv.")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")

    videos = list(find_videos_with_bytetrack(root, args.date))
    print(f"[INFO] Found {len(videos)} videos with ByteTrack tracks.")

    done, failed = 0, 0
    for vid_dir in tqdm(videos, desc="ByteTrack occupancy", unit="video"):
        try:
            out = count_occupancy_for_video(vid_dir, force=args.force)
            if out is None:
                failed += 1
            else:
                done += 1
        except Exception as e:
            print(f"[ERROR] Failed for {vid_dir}: {e}")
            failed += 1

    print(f"[DONE] ByteTrack occupancy complete. done={done}, failed={failed}")
    print(f"[INFO] Output per video: <video_dir>/{OUT_NAME}")


if __name__ == "__main__":
    main()
