import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# --------------------------
# Default config
# --------------------------
DEFAULT_ROOT = r"D:\PythonCode\FIT622_processed"
OUT_NAME = "final_bytetrack.csv"


def find_videos_with_bytetrack(root: Path, date_filter: str | None = None):
    """
    Yield video directories that have:
      - frames_manifest.csv
      - bytetrack_tracks.csv
    under processed root: root/<date>/<video>/
    """
    for date_dir in sorted(root.iterdir()):
        if not date_dir.is_dir():
            continue
        if date_filter and date_dir.name != date_filter:
            continue

        for vid_dir in sorted(date_dir.iterdir()):
            if not vid_dir.is_dir():
                continue

            mf = vid_dir / "frames_manifest.csv"
            bt = vid_dir / "bytetrack_tracks.csv"
            if mf.exists() and bt.exists():
                yield vid_dir


def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def count_occupancy_for_video(video_dir: Path, force: bool = False) -> Path | None:
    """
    Build per-frame occupancy counts from bytetrack_tracks.csv, aligned to frames_manifest.csv.

    Inputs:
      - video_dir/frames_manifest.csv   must contain at least: frame_idx, frame_path
        optionally: timestamp_sec
      - video_dir/bytetrack_tracks.csv  contains: frame_idx, track_id, score, (bbox cols), timestamp_sec (optional)

    Output:
      - video_dir/final_bytetrack.csv
    """
    out_csv = video_dir / OUT_NAME
    if out_csv.exists() and not force:
        return out_csv

    mf_path = video_dir / "frames_manifest.csv"
    bt_path = video_dir / "bytetrack_tracks.csv"

    df_mf = _safe_read_csv(mf_path)
    if df_mf.empty or "frame_idx" not in df_mf.columns or "frame_path" not in df_mf.columns:
        return None

    df_bt = _safe_read_csv(bt_path)

    # Prepare manifest (all frames)
    df_mf = df_mf.copy()
    df_mf["frame_idx"] = pd.to_numeric(df_mf["frame_idx"], errors="coerce").astype("Int64")
    df_mf = df_mf.dropna(subset=["frame_idx"]).copy()
    df_mf["frame_idx"] = df_mf["frame_idx"].astype(int)
    df_mf = df_mf.sort_values("frame_idx").reset_index(drop=True)

    # If ByteTrack tracks empty => counts all zero (valid outcome)
    if df_bt.empty:
        out = df_mf[["frame_idx", "frame_path"]].copy()
        if "timestamp_sec" in df_mf.columns:
            out["timestamp_sec"] = df_mf["timestamp_sec"]
        else:
            out["timestamp_sec"] = np.nan

        out["count"] = 0
        out["status"] = "unoccupied"
        out["source"] = "bytetrack"
        out.to_csv(out_csv, index=False)
        return out_csv

    # Clean ByteTrack tracks
    df_bt = df_bt.copy()
    if "frame_idx" not in df_bt.columns or "track_id" not in df_bt.columns:
        return None

    df_bt["frame_idx"] = pd.to_numeric(df_bt["frame_idx"], errors="coerce").astype("Int64")
    df_bt["track_id"] = pd.to_numeric(df_bt["track_id"], errors="coerce").astype("Int64")
    df_bt = df_bt.dropna(subset=["frame_idx", "track_id"]).copy()
    df_bt["frame_idx"] = df_bt["frame_idx"].astype(int)
    df_bt["track_id"] = df_bt["track_id"].astype(int)

    # Optional: filter very low scores if desired (commented out by default)
    # if "score" in df_bt.columns:
    #     df_bt = df_bt[pd.to_numeric(df_bt["score"], errors="coerce").fillna(0) >= 0.0].copy()

    # Count unique track_ids per frame
    df_counts = (
        df_bt.groupby("frame_idx")["track_id"]
        .nunique()
        .reset_index()
        .rename(columns={"track_id": "count"})
    )

    # Merge counts into manifest so missing frames become 0
    out = df_mf.merge(df_counts, on="frame_idx", how="left")
    out["count"] = out["count"].fillna(0).astype(int)

    # Timestamp: prefer manifest timestamp_sec; if missing, try ByteTrack timestamp_sec
    if "timestamp_sec" in out.columns:
        # keep manifest timestamp if present
        pass
    else:
        if "timestamp_sec" in df_bt.columns:
            ts = df_bt.groupby("frame_idx")["timestamp_sec"].first().reset_index()
            out = out.merge(ts, on="frame_idx", how="left")
        else:
            out["timestamp_sec"] = np.nan

    out["status"] = out["count"].apply(lambda x: "occupied" if int(x) >= 1 else "unoccupied")
    out["source"] = "bytetrack"

    # Keep a clean column order
    cols = ["frame_idx", "frame_path", "timestamp_sec", "count", "status", "source"]
    cols = [c for c in cols if c in out.columns]
    out = out[cols].copy()

    out.to_csv(out_csv, index=False)
    return out_csv


def main():
    ap = argparse.ArgumentParser(description="Compute per-frame occupancy counts from ByteTrack outputs.")
    ap.add_argument("--root", type=str, default=DEFAULT_ROOT, help="Processed root (contains date folders).")
    ap.add_argument("--date", type=str, default=None, help="Optional: only process this date folder (e.g., 20181030).")
    ap.add_argument("--force", action="store_true", help="Overwrite existing final_bytetrack.csv.")
    args = ap.parse_args()

    root = Path(args.root)
    assert root.exists(), f"Root not found: {root}"

    videos = list(find_videos_with_bytetrack(root, args.date))
    print(f"[INFO] Found {len(videos)} videos with ByteTrack tracks.")

    done, skipped, failed = 0, 0, 0
    for vid_dir in tqdm(videos, desc="ByteTrack occupancy", unit="video"):
        try:
            out = count_occupancy_for_video(vid_dir, force=args.force)
            if out is None:
                failed += 1
            else:
                done += 1
        except Exception:
            failed += 1

    print(f"[DONE] ByteTrack occupancy complete. done={done}, skipped={skipped}, failed={failed}")
    print(f"[INFO] Output per video: <video_dir>\\{OUT_NAME}")


if __name__ == "__main__":
    main()
