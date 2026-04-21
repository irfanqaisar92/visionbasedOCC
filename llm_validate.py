# LLM validation and fusion for occupancy counts (Ollama + LLaVA 7B)
# Reads:   D:\PythonCode\FIT622_processed\to_review_master.csv
# Writes:  <video_dir>\llm_review.csv         (per-frame LLM counts)
#          <video_dir>\validated_final.csv    (detector + LLM fusion)
# Usage:
#   python llm_validate.py --limit_videos 1 --limit_batches 2
#   python llm_validate.py

import os
import re
import json
import base64
import argparse
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# ----------------------- Config -----------------------
ROOT = Path(r"D:\PythonCode\FIT622_processed")
MODEL = "llava:7b"                                  # `ollama pull llava:7b`
OLLAMA_URL = "http://localhost:11434/api/generate"  # vision endpoint
FRAMES_PER_CALL = 2                                 # reduce to speed up
TIMEOUT = 300                                       # seconds
FUSION_MARGIN = 0.15                                # use LLM if llm_conf >= det_conf + margin
LOG_FAIL = ROOT / "llm_failures.log"                # where we log bad batches
# ------------------------------------------------------

SYSTEM_TEXT = (
    "You are a meticulous occupancy analyst. Count visible humans in each image.\n"
    "Rules:\n"
    "- Count only real humans physically in the room.\n"
    "- Ignore people on screens, posters, reflections, or through windows unless clearly the same room.\n"
    "- If uncertain, pick the conservative lower count and lower confidence.\n"
    "- Return STRICT JSON only. No explanations, no extra text.\n"
    'Schema: {"results":[{"frame":"<path>","timestamp_sec":<float|null>,"count":<int>,"confidence":<0..1>}]}\n'
)

def b64_image(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def strict_json(text: str):
    """Extract first JSON object/array from text."""
    m = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None

def lenient_parse(text: str, rows):
    """
    Fallback parser when the model returns messy text.
    Strategy:
      1) Try strict_json first (callers do this already).
      2) If that fails, attempt to parse per-frame blocks with regex.
      3) As a last resort, map any integers we find to frames by order.
    Returns list of dicts with keys: frame, count, confidence, timestamp_sec (opt)
    """
    # Try to find per-result blocks like {"frame": "...", "count": 2, "confidence": 0.8}
    results = []
    blocks = re.findall(r'\{[^{}]*?frame[^{}]*?\}', text, flags=re.DOTALL)
    if blocks:
        for b in blocks:
            frame = re.search(r'"frame"\s*:\s*"([^"]+)"', b)
            cnt   = re.search(r'"count"\s*:\s*([0-9]+)', b)
            conf  = re.search(r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)', b)
            results.append({
                "frame": frame.group(1) if frame else None,
                "count": int(cnt.group(1)) if cnt else None,
                "confidence": float(conf.group(1)) if conf else None,
            })
        return results

    # Last resort: map integers by order to the frames we sent
    ints = [int(x) for x in re.findall(r'\b([0-9]+)\b', text)]
    if ints:
        out = []
        for i, r in enumerate(rows):
            c = ints[i] if i < len(ints) else None
            out.append({"frame": r["frame_path"], "count": c, "confidence": None})
        return out

    return []

def build_messages(batch_rows):
    lines = ["Images:"]
    for r in batch_rows:
        lines.append(f'- {r["frame_path"]} (t={r["timestamp_sec"]}s)')
    user_text = "\n".join(lines) + (
        "\n\nOutput JSON schema:\n"
        '{ "results": [ '
        '{"frame":"<path>","timestamp_sec":<float|null>,"count":<int>,"confidence":<0..1>} ] }'
        "\nReturn ONLY valid JSON."
    )
    contents = [{"type": "text", "text": SYSTEM_TEXT + "\n" + user_text}]
    for r in batch_rows:
        p = Path(r["frame_path"])
        if p.exists():
            try:
                contents.append({"type": "image", "image": b64_image(p)})
            except Exception:
                pass
    return [{"role": "user", "content": contents}]

def _read_ndjson_to_text(resp):
    txt = ""
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        try:
            part = json.loads(line)
            txt += part.get("response", "")
            if part.get("done"):
                break
        except Exception:
            pass
    return txt

def call_ollama(messages, model=MODEL, retries=2):
    """Call Ollama generate; request JSON, handle both stream and non-stream."""
    user_msg = messages[-1]["content"]
    text_parts = [c["text"] for c in user_msg if c.get("type") == "text"]
    image_b64s = [c["image"] for c in user_msg if c.get("type") == "image"]
    if not image_b64s:
        raise RuntimeError("No images attached to the Ollama request.")

    payload = {
        "model": model,
        "prompt": "\n".join(text_parts),
        "images": image_b64s,
        "format": "json",       # <-- force JSON
        "options": {"temperature": 0},
        "stream": False,
        "keep_alive": "30m"
    }

    for attempt in range(retries + 1):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
            ctype = resp.headers.get("Content-Type", "")
            if "application/x-ndjson" in ctype or "text/event-stream" in ctype:
                txt = _read_ndjson_to_text(resp)
            else:
                # If server returned a JSON object with {response: "..."}
                try:
                    data = resp.json()
                    txt = data.get("response", "")
                except Exception:
                    # rare: some builds return raw text; fall back
                    txt = resp.text

            js = strict_json(txt)
            if js:
                return js, txt  # return both parsed and raw for logging
            else:
                return None, txt
        except Exception as e:
            if attempt == retries:
                raise
            time.sleep(1.5 * (attempt + 1))
    return None, ""

def write_llm_review(video_dir: Path, rows_out: list[dict]):
    out_csv = video_dir / "llm_review.csv"
    df_new = pd.DataFrame(rows_out)
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        # Drop all-NA columns (fix FutureWarning) and concat only if non-empty
        frames = [p for p in [prev, df_new] if not p.empty]
        df = pd.concat(frames, ignore_index=True) if frames else df_new
        df = df.drop_duplicates(subset=["frame_path"], keep="last")
    else:
        df = df_new
    df.to_csv(out_csv, index=False)
    return out_csv

def fuse_results(video_dir: Path):
    det_csv = video_dir / "final.csv"
    llm_csv = video_dir / "llm_review.csv"
    if not det_csv.exists() or not llm_csv.exists():
        return None
    df_det = pd.read_csv(det_csv)
    df_llm = pd.read_csv(llm_csv)

    if "llm_count" not in df_llm.columns or "llm_conf" not in df_llm.columns:
        return None

    merged = df_det.merge(
        df_llm[["frame_path", "llm_count", "llm_conf"]],
        on="frame_path",
        how="left"
    )

    def decide(row):
        det_c = int(row["count"])
        det_conf = float(row.get("mean_conf", 0.0) or 0.0)
        llm_c = row.get("llm_count", None)
        llm_conf = row.get("llm_conf", None)
        if pd.isna(llm_c) or pd.isna(llm_conf):
            return det_c, "detector"
        try:
            llm_c = int(llm_c); llm_conf = float(llm_conf)
        except Exception:
            return det_c, "detector"
        if llm_conf >= det_conf + FUSION_MARGIN:
            return llm_c, "llm"
        else:
            return det_c, "detector"

    outs, srcs = [], []
    for _, r in merged.iterrows():
        val, src = decide(r)
        outs.append(val); srcs.append(src)

    merged["final_count"] = outs
    merged["source"] = srcs
    merged["status"] = merged["final_count"].apply(lambda x: "occupied" if int(x) >= 1 else "unoccupied")

    out_csv = video_dir / "validated_final.csv"
    merged.to_csv(out_csv, index=False)
    return out_csv

def log_failure(batch_rows, raw_text, note=""):
    LOG_FAIL.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FAIL, "a", encoding="utf-8") as f:
        f.write("\n" + "="*80 + "\n")
        f.write(f"NOTE: {note}\n")
        f.write("FRAMES:\n")
        for r in batch_rows:
            f.write(f" - {r['frame_path']} (t={r['timestamp_sec']})\n")
        f.write("\nRAW RESPONSE:\n")
        try:
            f.write(raw_text if isinstance(raw_text, str) else json.dumps(raw_text, ensure_ascii=False))
        except Exception:
            f.write("<unprintable>\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=str(ROOT))
    ap.add_argument("--model", type=str, default=MODEL)
    ap.add_argument("--limit_videos", type=int, default=None, help="Process only first N videos (debug)")
    ap.add_argument("--limit_batches", type=int, default=None, help="Max LLM batches total (debug)")
    args = ap.parse_args()

    root = Path(args.root)
    master = root / "to_review_master.csv"
    assert master.exists(), f"Missing {master}. Run select_for_llm.py first."

    df = pd.read_csv(master)
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

        messages = build_messages(rows)

        js, raw = call_ollama(messages, model=args.model)

        # Build rows_out with either parsed JSON or lenient fallback
        rows_out = []
        parsed_ok = False

        if isinstance(js, dict) and "results" in js and isinstance(js["results"], list):
            res_map = {r.get("frame"): r for r in js["results"] if isinstance(r, dict) and "frame" in r}
            for r in rows:
                rr = res_map.get(r["frame_path"], {})
                rows_out.append({
                    "frame_path": r["frame_path"],
                    "frame_idx": r["frame_idx"],
                    "timestamp_sec": r["timestamp_sec"],
                    "llm_count": rr.get("count", None),
                    "llm_conf": rr.get("confidence", None),
                    "reason": r.get("reason", "")
                })
            parsed_ok = True
        else:
            # Try lenient parse
            fallback = lenient_parse(raw, rows)
            if fallback:
                # map back to the rows, by frame if possible
                res_map = {r.get("frame"): r for r in fallback if isinstance(r, dict) and r.get("frame")}
                for r in rows:
                    rr = res_map.get(r["frame_path"], {})
                    rows_out.append({
                        "frame_path": r["frame_path"],
                        "frame_idx": r["frame_idx"],
                        "timestamp_sec": r["timestamp_sec"],
                        "llm_count": rr.get("count", None),
                        "llm_conf": rr.get("confidence", None),
                        "reason": r.get("reason", "")
                    })
                parsed_ok = True
            else:
                # Give up for this batch, but persist nulls so we can rerun later if needed
                for r in rows:
                    rows_out.append({
                        "frame_path": r["frame_path"],
                        "frame_idx": r["frame_idx"],
                        "timestamp_sec": r["timestamp_sec"],
                        "llm_count": None,
                        "llm_conf": None,
                        "reason": r.get("reason", "")
                    })
                log_failure(rows, raw, note="Non-JSON and lenient parse failed")

        write_llm_review(video_dir, rows_out)

        # Opportunistic fuse per-video as soon as we have some LLM rows
        try:
            fuse_results(video_dir)
        except Exception:
            pass

        processed_videos.add(video_dir_str)
        batch_count += 1

    print("[DONE] LLM review pass completed (check per-video validated_final.csv).")

if __name__ == "__main__":
    main()
