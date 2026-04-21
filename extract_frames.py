import os, sys, math, argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm

def estimate_fps(cap, fallback=25.0, sample_frames=120):
    f = cap.get(cv2.CAP_PROP_FPS)
    if f and np.isfinite(f) and f > 0:
        return float(f)
    times = []
    pos = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while pos < sample_frames:
        ret, _ = cap.read()
        if not ret:
            break
        ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        times.append(ms)
        pos += 1
    deltas = [t2 - t1 for t1, t2 in zip(times[:-1], times[1:]) if (t2 - t1) > 0]
    if len(deltas) >= 5:
        avg_ms = float(np.median(deltas))
        if avg_ms > 0:
            return 1000.0 / avg_ms
    return fallback

def safe_stem(p: Path):
    # video_id without extension, keep directories for uniqueness
    return p.stem

def process_video(mp4_path: Path, out_root: Path, fps_target: float = 1.0, min_frames: int = 10, max_frames: int = 900):
    rel_parent = mp4_path.parent.name                             # e.g., 20181030
    video_id   = safe_stem(mp4_path)                              # e.g., C2100...Z
    out_dir    = out_root / rel_parent / video_id
    frames_dir = out_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    # If manifest already exists, skip
    manifest_csv = out_dir / "frames_manifest.csv"
    if manifest_csv.exists():
        return

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        print(f"[WARN] Cannot open: {mp4_path}")
        return

    # metadata
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_est = estimate_fps(cap)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # sampling step (frames per extracted frame)
    step = max(1, int(round(fps_est / fps_target))) if fps_est > 0 else 25

    # iterate and sample
    rows = []
    frame_idx = 0
    saved = 0

    # rough cap on frames to avoid huge dumps
    hard_cap = max_frames

    with tqdm(total=hard_cap, desc=f"[{rel_parent}] {video_id}", unit="frm", leave=False) as pbar:
        while saved < hard_cap:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                ts = frame_idx / fps_est if fps_est > 0 else None
                fname = f"{video_id}_f{frame_idx:06d}" + (f"_t{ts:.2f}s.jpg" if ts is not None else ".jpg")
                out_path = frames_dir / fname
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                Image.fromarray(rgb).save(out_path, quality=90)

                rows.append({
                    "video_id": video_id,
                    "date_folder": rel_parent,
                    "frame_path": str(out_path),
                    "frame_idx": frame_idx,
                    "timestamp_sec": None if ts is None else float(f"{ts:.2f}"),
                    "width": width,
                    "height": height,
                    "estimated_video_fps": round(float(fps_est), 3)
                })
                saved += 1
                if saved >= hard_cap:
                    break
            frame_idx += 1
            pbar.update(1)

    cap.release()

    # enforce minimum frames (optional)
    if saved < min_frames and saved > 0:
        # simple upsample: copy last frame to reach min (or just accept as-is)
        pass

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(manifest_csv, index=False)
        print(f"[OK] {mp4_path.name}: saved {saved} frames → {manifest_csv}")
    else:
        print(f"[SKIP] No frames saved for {mp4_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root",  type=str, required=True, help="Root folder with date subfolders of mp4s")
    ap.add_argument("--out_root", type=str, required=True, help="Where to write processed outputs")
    ap.add_argument("--fps", type=float, default=1.0, help="Sampling rate in frames per second")
    ap.add_argument("--min_frames", type=int, default=10)
    ap.add_argument("--max_frames", type=int, default=900)
    args = ap.parse_args()

    in_root  = Path(args.in_root)
    out_root = Path(args.out_root)
    assert in_root.exists(), f"Input root not found: {in_root}"
    out_root.mkdir(parents=True, exist_ok=True)

    mp4_list = []
    for date_dir in sorted(in_root.iterdir()):
        if not date_dir.is_dir():
            continue
        mp4_list.extend(sorted(date_dir.glob("*.mp4")))

    print(f"Found {len(mp4_list)} videos under {in_root}")
    for mp4 in mp4_list:
        try:
            process_video(mp4, out_root, fps_target=args.fps, min_frames=args.min_frames, max_frames=args.max_frames)
        except Exception as e:
            print(f"[ERROR] {mp4}: {e}")

if __name__ == "__main__":
    # === DEFAULT CONFIGURATION ===
    DEFAULT_IN_ROOT  = r"D:\PythonCode\FIT622"
    DEFAULT_OUT_ROOT = r"D:\PythonCode\FIT622_processed"
    DEFAULT_FPS      = 1.0
    DEFAULT_MIN_FRAMES = 10
    DEFAULT_MAX_FRAMES = 900

    # If you prefer to override these from command line, you still can
    # Example: python extract_frames.py --in_root X --out_root Y

    import sys
    if len(sys.argv) > 1:
        # user supplied arguments → use argparse
        main()
    else:
        # run with defaults
        from argparse import Namespace
        args = Namespace(
            in_root=DEFAULT_IN_ROOT,
            out_root=DEFAULT_OUT_ROOT,
            fps=DEFAULT_FPS,
            min_frames=DEFAULT_MIN_FRAMES,
            max_frames=DEFAULT_MAX_FRAMES
        )
        print(f"\n[INFO] Running extract_frames.py with defaults:")
        print(f"       in_root = {args.in_root}")
        print(f"       out_root = {args.out_root}")
        print(f"       fps = {args.fps}\n")

        # replicate main() logic manually
        in_root  = Path(args.in_root)
        out_root = Path(args.out_root)
        out_root.mkdir(parents=True, exist_ok=True)
        assert in_root.exists(), f"Input root not found: {in_root}"

        mp4_list = []
        for date_dir in sorted(in_root.iterdir()):
            if not date_dir.is_dir():
                continue
            mp4_list.extend(sorted(date_dir.glob("*.mp4")))

        print(f"Found {len(mp4_list)} videos under {in_root}")
        for mp4 in mp4_list:
            try:
                process_video(mp4, out_root,
                              fps_target=args.fps,
                              min_frames=args.min_frames,
                              max_frames=args.max_frames)
            except Exception as e:
                print(f"[ERROR] {mp4}: {e}")

