"""
llm_validate_deepseek.py
------------------------

Perform DeepSeek LLM-based occupancy count refinement and fusion with detector outputs.

This script implements the text-only reasoning-enhanced occupancy measurement pipeline
described in:

"Experimental Study on Surveillance Video-Based Indoor Occupancy Measurement 
for Occupant-Centric Control" (citation will be available after publication).

Outputs per video:
- llm_review_deepseek.csv       : LLM-inferred counts per frame
- validated_final_deepseek.csv  : fused counts from detector + DeepSeek
"""

import os
import re
import json
import argparse
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# ----------------------- Default Config -----------------------
FRAMES_PER_CALL = 10
TIMEOUT = 120
FUSION_MARGIN = 0.15

SYSTEM_TEXT = (
    "You are a meticulous occupancy analyst.\n"
    "You receive noisy person counts from an automatic detector (YOLO) "
    "for several camera frames in a video.\n"
    "Your task is to refine these counts using temporal and contextual reasoning.\n\n"
    "For each frame, you will be given:\n"
    "- frame_path\n"
    "- timestamp_sec (may be null)\n"
    "- det_count\n"
    "- det_conf\n"
    "- prev_count (optional)\n"
    "- next_count (optional)\n"
    "- reason (optional)\n\n"
    "Rules:\n"
    "- Output an integer corrected_count >= 0 for each frame.\n"
    "- Only adjust the detector count when strong evidence exists.\n"
    "- Return STRICT JSON only.\n"
    "Schema: {\"results\": [{\"frame_path\": \"<path>\", \"corrected_count\": <int>, \"confidence\": <0..1>}]}.\n"
)

# --------------------------------------------------------------

def strict_json(text: str):
    m = re.search(r'(\{.*\}|\[.*\])', str(text), re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None

def lenient_parse(text: str, rows):
    # Fallback parser
    out = []
    blocks = re.findall(r'\{[^{}]*?frame_path[^{}]*?\}', text, flags=re.DOTALL)
    if blocks:
        for b in blocks:
            frame = re.search(r'"frame_path"\s*:\s*"([^"]+)"', b)
            cnt = re.search(r'"corrected_count"\s*:\s*([0-9]+)', b)
            conf = re.search(r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)', b)
            out.append({
                "frame_path": frame.group(1) if frame else None,
                "corrected_count": int(cnt.group(1)) if cnt else None,
                "confidence": float(conf.group(1)) if conf else None
            })
        return out
    # Last resort: map integers by order
    ints = [int(x) for x in re.findall(r'\b([0-9]+)\b', text)]
    if ints:
        res = []
        for i, r in enumerate(rows):
            c = ints[i] if i < len(ints) else None
            res.append({"frame_path": r["frame_path"], "corrected_count": c, "confidence": None})
        return res
    return []

def build_prompt(batch_rows):
    lines = ["Refine the person counts for the following frames."]
    for r in batch_rows:
        lines.append(
            f"\nFrame:\n- frame_path: {r['frame_path']}\n- timestamp_sec: {r.get('timestamp_sec')}\n"
            f"- det_count: {r.get('count')}\n- det_conf: {r.get('mean_conf')}\n"
            f"- prev_count: {r.get('prev_count')}\n- next_count: {r.get('next_count')}\n"
            f"- reason: {r.get('reason','')}"
        )
    return "\n".join(lines)

def call_deepseek(prompt: str, api_key: str, model: str, api_url: str, retries: int = 2):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM_TEXT},
                     {"role": "user", "content": prompt}],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    for attempt in range(retries + 1):
        try:
            resp = requests.post(api_url, headers=headers, json=data, timeout=TIMEOUT)
            resp.raise_for_status()
            js = resp.json()
            content = js["choices"][0]["message"]["content"]
            parsed = strict_json(content)
            return parsed, content
        except Exception:
            if attempt == retries:
                raise
            time.sleep(1.5 * (attempt + 1))
    return None, ""

# Functions write_llm_review, fuse_results, log_failure remain same, but remove ROOT hardcoding

def main():
    parser = argparse.ArgumentParser(description="DeepSeek LLM occupancy refinement")
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--api_url", type=str, default=DEEPSEEK_API_URL)
    parser.add_argument("--model", type=str, default=DEEPSEEK_MODEL)
    parser.add_argument("--limit_videos", type=int, default=None)
    parser.add_argument("--limit_batches", type=int, default=None)
    args = parser.parse_args()

    root = Path(args.root)
    master_csv = root / "to_review_master.csv"
    assert master_csv.exists(), f"Missing {master_csv}. Run select_for_llm.py first."

    # ... same main logic as before ...
    # Read CSV, group by video_dir and batch_group, call_deepseek, write llm_review_deepseek.csv, fuse_results

if __name__ == "__main__":
    main()
