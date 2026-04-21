import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# -----------------------
# Default paths & knobs
# -----------------------
ROOT = Path(r"D:\PythonCode\FIT622_processed")

# Escalation rules (you can tune these)
MIN_MEAN_CONF = 0.60          # send to LLM if detector mean_conf below this (and > 0)
EDGE_COUNTS   = {0, 1}        # ambiguous counts to review
MAX_COUNT_REV = 8             # review very high counts (possible double-counts)
MAX_FRAMES_PER_VIDEO = 60     # cap per video to keep LLM cost/latency bounded (optional)
FRAMES_PER_CALL = 4           # later used when batching prompts (info only here)

def select_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return subset of rows that should be reviewed by LLM."""
    df = df.copy()
    # Defensive: ensure required columns
    need_cols = {"frame_path", "frame_idx", "timestamp_sec", "count", "mean_conf"}
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in final.csv: {missing}")

    # Rules:
    # 1) mean_conf < MIN_MEAN_CONF and count > 0  (uncertain detections)
    rule_uncertain = (df["mean_conf"] < MIN_MEAN_CONF) & (df["mean_conf"] > 0) & (df["count"] > 0)
    # 2) count in EDGE_COUNTS (0 or 1)
    rule_edge = df["count"].isin(EDGE_COUNTS)
    # 3) count > MAX_COUNT_REV (crowded / double-count risk)
    rule_crowd = df["count"] > MAX_COUNT_REV

    chosen = df[rule_uncertain | rule_edge | rule_crowd].copy()

    # Optional prioritization: sort by lowest confidence first, then by count
    chosen = chosen.sort_values(by=["mean_conf", "count"], ascending=[True, True])

    # Optional cap per video
    if MAX_FRAMES_PER_VIDEO is not None and len(chosen) > MAX_FRAMES_PER_VIDEO:
        chosen = chosen.head(MAX_FRAMES_PER_VIDEO)

    # Add fields that will be useful downstream
    chosen["reason"] = chosen.apply(lambda r: _reason(r, MIN_MEAN_CONF, EDGE_COUNTS, MAX_COUNT_REV), axis=1)
    chosen["batch_group"] = (chosen.reset_index().index // FRAMES_PER_CALL) + 1
    return chosen

def _reason(r, min_conf, edge_counts, max_count):
    reasons = []
    if (r["mean_conf"] < min_conf) and (r["mean_conf"] > 0) and (r["count"] > 0):
        reasons.append("low_conf")
    if r["count"] in edge_counts:
        reasons.append("edge_count")
    if r["count"] > max_count:
        reasons.append("crowd")
    return "|".join(reasons) if reasons else "other"

def main():
    # Find all video dirs
    all_video_dirs = []
    for date_dir in sorted(ROOT.iterdir()):
        if not date_dir.is_dir():
            continue
        for vid_dir in sorted(date_dir.iterdir()):
            if vid_dir.is_dir():
                all_video_dirs.append(vid_dir)

    print(f"[INFO] Found {len(all_video_dirs)} video folders under {ROOT}")

    masters = []
    for vid_dir in tqdm(all_video_dirs, desc="Selecting", unit="video"):
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

    # Write master file
    if masters:
        master_df = pd.concat(masters, ignore_index=True)
        out_master = ROOT / "to_review_master.csv"
        master_df.to_csv(out_master, index=False)
        print(f"[DONE] Wrote {out_master} with {len(master_df)} frames queued for LLM review.")
    else:
        print("[DONE] No frames matched the review rules. You can relax rules if needed.")

if __name__ == "__main__":
    main()
