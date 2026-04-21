"""
select_for_llm.py
-----------------

Select frames that require LLM-based occupancy refinement.

This script implements the selection step of the reasoning-enhanced
occupancy measurement pipeline described in:

"Experimental Study on Surveillance Video-Based Indoor Occupancy Measurement 
for Occupant-Centric Control" (citation will be available after publication).

- Reads per-video final.csv outputs from detector/tracker.
- Applies configurable rules to identify ambiguous or uncertain frames.
- Writes:
    - per-video to_review.csv
    - global to_review_master.csv
"""

import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import argparse

# ----------------------- Default parameters -----------------------
MIN_MEAN_CONF = 0.60
EDGE_COUNTS = {0, 1}
MAX_COUNT_REV = 8
MAX_FRAMES_PER_VIDEO = 60
FRAMES_PER_CALL = 4

# ----------------------- Selection helpers -----------------------
def select_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    need_cols = {"frame_path", "frame_idx", "timestamp_sec", "count", "mean_conf"}
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in final.csv: {missing}")

    rule_uncertain = (df["mean_conf"] < MIN_MEAN_CONF) & (df["mean_conf"] > 0) & (df["count"] > 0)
    rule_edge = df["count"].isin(EDGE_COUNTS)
    rule_crowd = df["count"] > MAX_COUNT_REV

    chosen = df[rule_uncertain | rule_edge | rule_crowd].copy()
    chosen = chosen.sort_values(by=["mean_conf", "count"], ascending=[True, True])

    if MAX_FRAMES_PER_VIDEO and len(chosen) > MAX_FRAMES_PER_VIDEO:
        chosen = chosen.head(MAX_FRAMES_PER_VIDEO)

    chosen["reason"] = chosen.apply(lambda r: _reason(r), axis=1)
    chosen["batch_group"] = (chosen.reset_index().index // FRAMES_PER_CALL) + 1
    return chosen

def _reason(r):
    reasons = []
    if (r["mean_conf"] < MIN_MEAN_CONF) and (r["mean_conf"] > 0) and (r["count"] > 0):
        reasons.append("low_conf")
    if r["count"] in EDGE_COUNTS:
        reasons.append("edge_count")
    if r["count"] > MAX_COUNT_REV:
        reasons.append("crowd")
    return "|".join(reasons) if reasons else "other"

# ----------------------- Main -----------------------
def main():
    parser = argparse.ArgumentParser(description="Select frames for LLM review.")
    parser.add_argument("--root", type=str, required=True, help="Root folder containing per-video final.csv")
    args = parser.parse_args()

    root = Path(args.root)
    all_video_dirs = [vid_dir for date_dir in sorted(root.iterdir()) if date_dir.is_dir()
                      for vid_dir in sorted(date_dir.iterdir()) if vid_dir.is_dir()]
    print(f"[INFO] Found {len(all_video_dirs)} video folders under {root}")

    masters = []
    for vid_dir in tqdm(all_video_dirs, desc="Selecting frames", unit="video"):
        final_csv = vid_dir / "final.csv"
        if not final_csv.exists():
            continue
        try:
            df = pd.read_csv(final_csv)
            chosen = select_rows(df)
            if not chosen.empty:
                out_csv = vid_dir / "to_review.csv"
                chosen.to_csv(out_csv, index=False)
                # Add context columns for master file
                chosen2 = chosen.copy()
                chosen2["video_id"] = vid_dir.name
                chosen2["date_folder"] = vid_dir.parent.name
                masters.append(chosen2)
        except Exception as e:
            print(f"[WARN] {final_csv}: {e}")

    if masters:
        master_df = pd.concat(masters, ignore_index=True)
        out_master = root / "to_review_master.csv"
        master_df.to_csv(out_master, index=False)
        print(f"[DONE] Wrote {out_master} with {len(master_df)} frames queued for LLM review.")
    else:
        print("[DONE] No frames matched the review rules.")

if __name__ == "__main__":
    main()
