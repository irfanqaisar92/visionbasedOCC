"""
llm_validate.py
-----------------

Perform LLM-based occupancy count refinement and fusion with detector outputs.

This script implements the reasoning-enhanced occupancy measurement pipeline
described in:

"Experimental Study on Surveillance Video-Based Indoor Occupancy Measurement 
for Occupant-Centric Control" (citation will be available after publication).

Outputs per video:
- llm_review.csv      : LLM-inferred counts per frame
- validated_final.csv : fused counts from detector + LLM
"""

import os
import json
import argparse
from pathlib import Path
import time
import re
import base64

import pandas as pd
from tqdm import tqdm
import requests

# ----------------------- Default Config -----------------------
FRAMES_PER_CALL = 2
TIMEOUT = 300
FUSION_MARGIN = 0.15  # LLM overrides detector if conf >= det_conf + margin

SYSTEM_TEXT = (
    "You are a meticulous occupancy analyst. Count visible humans in each image.\n"
    "Rules:\n"
    "- Count only real humans physically in the room.\n"
    "- Ignore people on screens, posters, reflections, or through windows unless clearly in the same room.\n"
    "- If uncertain, pick the conservative lower count and lower confidence.\n"
    "- Return STRICT JSON only.\n"
    'Schema: {"results":[{"frame":"<path>","timestamp_sec":<float|null>,"count":<int>,"confidence":<0..1>}]}\n'
)

# ---------------------------------------------------------------

def b64_image(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def strict_json(text: str):
    """Extract first JSON object/array from text."""
    import json, re
    m = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None

# --- Other helper functions (lenient_parse, build_messages, call_ollama, write_llm_review, fuse_results, log_failure)
# Keep the same logic, but remove hardcoded ROOT references and log paths

# ----------------------- Main -----------------------
def main():
    parser = argparse.ArgumentParser(description="LLM-based occupancy refinement and fusion.")
    parser.add_argument("--root", type=str, required=True,
                        help="Root folder containing to_review_master.csv and video folders")
    parser.add_argument("--model", type=str, default="llava:7b")
    parser.add_argument("--ollama_url", type=str, default="http://localhost:11434/api/generate")
    parser.add_argument("--limit_videos", type=int, default=None, help="Process only first N videos (debug)")
    parser.add_argument("--limit_batches", type=int, default=None, help="Max LLM batches total (debug)")
    args = parser.parse_args()

    root = Path(args.root)
    master_csv = root / "to_review_master.csv"
    assert master_csv.exists(), f"Missing {master_csv}. Run select_for_llm.py first."

    df = pd.read_csv(master_csv)
    if df.empty:
        print("[INFO] Nothing to review.")
        return

    df["video_dir"] = df.apply(lambda r: str(root / str(r["date_folder"]) / str(r["video_id"])), axis=1)
    groups = df.groupby(["video_dir", "batch_group"], sort=True)

    batch_count = 0
    processed_videos = set()

    for (video_dir_str, batch_grp), g in tqdm(groups, desc="LLM review", unit="batch"):
        if args.limit_videos and len(processed_videos) >= args.limit_videos:
            break
        if args.limit_batches and batch_count >= args.limit_batches:
            break

        video_dir = Path(video_dir_str)
        rows = g.to_dict(orient="records")
        rows = rows[:FRAMES_PER_CALL]

        messages = build_messages(rows)  # same helper function as before
        js, raw = call_ollama(messages, model=args.model)  # same helper

        # Process LLM results with fallback parsing and write llm_review.csv
        rows_out = []  # construct rows_out like in original
        write_llm_review(video_dir, rows_out)
        fuse_results(video_dir)  # fuse detector + LLM

        processed_videos.add(video_dir_str)
        batch_count += 1

    print("[DONE] LLM review pass completed (check per-video validated_final.csv).")

if __name__ == "__main__":
    main()
