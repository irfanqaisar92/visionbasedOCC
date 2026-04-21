"""
LLM validation and fusion for occupancy counts (DeepSeek API, text-only)

Reads:
    D:\PythonCode\FIT622_processed\to_review_master.csv

Writes, per video_dir:
    llm_review_deepseek.csv       (per-frame DeepSeek counts)
    validated_final_deepseek.csv  (detector + DeepSeek fusion)

Usage examples:
    python llm_validate_deepseek.py --limit_videos 1 --limit_batches 2
    python llm_validate_deepseek.py
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

# ----------------------- Config -----------------------
ROOT = Path(r"D:\PythonCode\FIT622_processed")

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"   # or "deepseek-reasoner" etc.
DEEPSEEK_API_KEY = Path("deepseek_key.txt").read_text().strip()

FRAMES_PER_CALL = 10               # text-only, can handle larger batches
TIMEOUT = 120                      # seconds
FUSION_MARGIN = 0.15               # use LLM if llm_conf >= det_conf + margin
LOG_FAIL = ROOT / "llm_failures_deepseek.log"
# ------------------------------------------------------


SYSTEM_TEXT = (
    "You are a meticulous occupancy analyst.\n"
    "You receive noisy person counts from an automatic detector (YOLO) "
    "for several camera frames in a video.\n"
    "Your task is to refine these counts using temporal and contextual reasoning.\n\n"
    "For each frame, you will be given:\n"
    "- frame_path: unique identifier of the frame\n"
    "- timestamp_sec: time in seconds (may be null)\n"
    "- det_count: YOLO person count\n"
    "- det_conf: mean detection confidence (0..1)\n"
    "- prev_count: YOLO count in previous frame (if available)\n"
    "- next_count: YOLO count in next frame (if available)\n"
    "- reason: why this frame was flagged (e.g., abrupt change, low confidence)\n\n"
    "Rules:\n"
    "- Output an integer corrected_count >= 0 for each frame.\n"
    "- Only adjust the detector count when there is strong evidence "
    "from context or the description.\n"
    "- confidence is your 0..1 belief that corrected_count is accurate.\n"
    "- Be conservative if uncertain.\n\n"
    "Return STRICT JSON only, no explanations.\n"
    "Schema:\n"
    '{\"results\": [\n'
    '  {\"frame_path\": \"<path>\", \"corrected_count\": <int>, \"confidence\": <0..1>}\n'
    ']}\n'
)


# --------- Helpers for JSON parsing / logging ----------

def strict_json(text: str):
    """Extract and parse the first JSON object/array from text."""
    if isinstance(text, dict):
        # Already a dict (when using response_format=json_object).
        return text

    m = re.search(r'(\{.*\}|\[.*\])', str(text), re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None


def lenient_parse(text: str, rows):
    """
    Fallback parser when the model returns messy text.
    Returns list of dicts with keys:
        frame_path, corrected_count, confidence
    """
    out = []

    # Try to find { ... frame_path ... corrected_count ... } blocks
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

    # Last resort: map integers by order to frames
    ints = [int(x) for x in re.findall(r'\b([0-9]+)\b', text)]
    if ints:
        res = []
        for i, r in enumerate(rows):
            c = ints[i] if i < len(ints) else None
            res.append({
                "frame_path": r["frame_path"],
                "corrected_count": c,
                "confidence": None
            })
        return res

    return []


def build_prompt(batch_rows):
    """
    Build a single user prompt string describing all frames in the batch.
    """
    lines = [
        "Refine the person counts for the following frames.",
        "Use temporal consistency and the given reasons to decide if the detector is wrong.",
        "Respond strictly in the JSON schema defined earlier."
    ]
    for r in batch_rows:
        lines.append(
            "\nFrame:\n"
            f"- frame_path: {r['frame_path']}\n"
            f"- timestamp_sec: {r.get('timestamp_sec')}\n"
            f"- det_count: {r.get('count')}\n"
            f"- det_conf: {r.get('mean_conf')}\n"
            f"- prev_count: {r.get('prev_count')}\n"
            f"- next_count: {r.get('next_count')}\n"
            f"- reason: {r.get('reason', '')}"
        )
    return "\n".join(lines)


def call_deepseek(prompt: str, retries: int = 2):
    """
    Call DeepSeek chat/completions API with JSON response_format.
    Returns (parsed_json_or_none, raw_content).
    """
    if not DEEPSEEK_API_KEY:
        raise RuntimeError(
            "DEEPSEEK_API_KEY not set. Please set it in your environment."
        )

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_TEXT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        # Ask for machine-parsable JSON
        "response_format": {"type": "json_object"},
    }

    for attempt in range(retries + 1):
        try:
            resp = requests.post(
                DEEPSEEK_API_URL,
                headers=headers,
                json=data,
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            js = resp.json()
            content = js["choices"][0]["message"]["content"]

            parsed = strict_json(content)
            if parsed is None:
                # sometimes response_format already returns a dict
                parsed = content if isinstance(content, dict) else None
            return parsed, content
        except Exception as e:
            if attempt == retries:
                raise
            time.sleep(1.5 * (attempt + 1))

    return None, ""


def write_llm_review(video_dir: Path, rows_out: list[dict]):
    """
    Append/update DeepSeek results into llm_review_deepseek.csv
    """
    out_csv = video_dir / "llm_review_deepseek.csv"
    df_new = pd.DataFrame(rows_out)
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        frames = [p for p in [prev, df_new] if not p.empty]
        df = pd.concat(frames, ignore_index=True) if frames else df_new
        df = df.drop_duplicates(subset=["frame_path"], keep="last")
    else:
        df = df_new
    df.to_csv(out_csv, index=False)
    return out_csv


def fuse_results(video_dir: Path):
    """
    Fuse YOLO counts (final.csv) with DeepSeek LLM outputs
    into validated_final_deepseek.csv
    """
    det_csv = video_dir / "final.csv"
    llm_csv = video_dir / "llm_review_deepseek.csv"
    if not det_csv.exists() or not llm_csv.exists():
        return None

    df_det = pd.read_csv(det_csv)
    df_llm = pd.read_csv(llm_csv)

    if "llm_count" not in df_llm.columns or "llm_conf" not in df_llm.columns:
        return None

    merged = df_det.merge(
        df_llm[["frame_path", "llm_count", "llm_conf"]],
        on="frame_path",
        how="left",
    )

    def decide(row):
        det_c = int(row["count"])
        det_conf = float(row.get("mean_conf", 0.0) or 0.0)
        llm_c = row.get("llm_count", None)
        llm_conf = row.get("llm_conf", None)

        if pd.isna(llm_c) or pd.isna(llm_conf):
            return det_c, "detector"

        try:
            llm_c = int(llm_c)
            llm_conf = float(llm_conf)
        except Exception:
            return det_c, "detector"

        if llm_conf >= det_conf + FUSION_MARGIN:
            return llm_c, "llm"
        else:
            return det_c, "detector"

    outs, srcs = [], []
    for _, r in merged.iterrows():
        val, src = decide(r)
        outs.append(val)
        srcs.append(src)

    merged["final_count"] = outs
    merged["source"] = srcs
    merged["status"] = merged["final_count"].apply(
        lambda x: "occupied" if int(x) >= 1 else "unoccupied"
    )

    out_csv = video_dir / "validated_final_deepseek.csv"
    merged.to_csv(out_csv, index=False)
    return out_csv


def log_failure(batch_rows, raw_text, note=""):
    LOG_FAIL.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FAIL, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"NOTE: {note}\n")
        f.write("FRAMES:\n")
        for r in batch_rows:
            f.write(
                f" - {r['frame_path']} (t={r.get('timestamp_sec')}, "
                f"det_count={r.get('count')})\n"
            )
        f.write("\nRAW RESPONSE:\n")
        try:
            f.write(
                raw_text
                if isinstance(raw_text, str)
                else json.dumps(raw_text, ensure_ascii=False)
            )
        except Exception:
            f.write("<unprintable>\n")


# -------------------------- Main --------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=str(ROOT))
    ap.add_argument(
        "--limit_videos",
        type=int,
        default=None,
        help="Process only first N videos (debug)",
    )
    ap.add_argument(
        "--limit_batches",
        type=int,
        default=None,
        help="Max LLM batches total (debug)",
    )
    args = ap.parse_args()

    root = Path(args.root)
    master = root / "to_review_master.csv"
    assert master.exists(), f"Missing {master}. Run select_for_llm.py first."

    df = pd.read_csv(master)
    if df.empty:
        print("[INFO] Nothing to review.")
        return

    # Construct absolute video_dir paths
    df["video_dir"] = df.apply(
        lambda r: str(root / str(r["date_folder"]) / str(r["video_id"])), axis=1
    )
    groups = df.groupby(["video_dir", "batch_group"], sort=True)

    batch_count = 0
    processed_videos = set()

    for (video_dir_str, batch_grp), g in tqdm(
        groups, desc="DeepSeek LLM review", unit="batch"
    ):
        if args.limit_videos and len(processed_videos) >= args.limit_videos:
            break
        if args.limit_batches and batch_count >= args.limit_batches:
            break

        video_dir = Path(video_dir_str)
        rows = g.to_dict(orient="records")

        # Limit batch size for safety
        rows = rows[:FRAMES_PER_CALL]

        prompt = build_prompt(rows)
        js, raw = call_deepseek(prompt)

        rows_out = []
        parsed_ok = False

        if isinstance(js, dict) and "results" in js and isinstance(js["results"], list):
            res_map = {
                r.get("frame_path"): r
                for r in js["results"]
                if isinstance(r, dict) and r.get("frame_path")
            }
            for r in rows:
                rr = res_map.get(r["frame_path"], {})
                rows_out.append(
                    {
                        "frame_path": r["frame_path"],
                        "frame_idx": r["frame_idx"],
                        "timestamp_sec": r.get("timestamp_sec"),
                        "llm_count": rr.get("corrected_count", None),
                        "llm_conf": rr.get("confidence", None),
                        "reason": r.get("reason", ""),
                    }
                )
            parsed_ok = True
        else:
            # Try lenient parse
            fallback = lenient_parse(str(raw), rows)
            if fallback:
                res_map = {
                    r.get("frame_path"): r
                    for r in fallback
                    if isinstance(r, dict) and r.get("frame_path")
                }
                for r in rows:
                    rr = res_map.get(r["frame_path"], {})
                    rows_out.append(
                        {
                            "frame_path": r["frame_path"],
                            "frame_idx": r["frame_idx"],
                            "timestamp_sec": r.get("timestamp_sec"),
                            "llm_count": rr.get("corrected_count", None),
                            "llm_conf": rr.get("confidence", None),
                            "reason": r.get("reason", ""),
                        }
                    )
                parsed_ok = True
            else:
                # Give up for this batch but still persist nulls
                for r in rows:
                    rows_out.append(
                        {
                            "frame_path": r["frame_path"],
                            "frame_idx": r["frame_idx"],
                            "timestamp_sec": r.get("timestamp_sec"),
                            "llm_count": None,
                            "llm_conf": None,
                            "reason": r.get("reason", ""),
                        }
                    )
                log_failure(rows, raw, note="DeepSeek non-JSON; lenient parse failed")

        write_llm_review(video_dir, rows_out)

        # Opportunistic fuse per-video whenever we have some DeepSeek rows
        try:
            fuse_results(video_dir)
        except Exception:
            pass

        processed_videos.add(video_dir_str)
        batch_count += 1

    print(
        "[DONE] DeepSeek LLM review pass completed "
        "(check per-video validated_final_deepseek.csv)."
    )


if __name__ == "__main__":
    main()
