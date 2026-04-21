"""
bytetrack_simple.py
-------------------

Simple ByteTrack-style tracking over YOLO detections.

This script implements a simplified ByteTrack tracking pipeline as part of:

"Experimental Study on Surveillance Video-Based Indoor Occupancy Measurement
for Occupant-Centric Control" (citation will be available after publication).

- Reads per-frame YOLO detections (yolo_detections.csv).
- Tracks objects across frames using IoU-based association.
- Outputs `bytetrack_tracks.csv` per video:
  frame_idx, timestamp_sec, track_id, x1, y1, x2, y2, score
"""

import os
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from tqdm import tqdm

# --------------------------
# Default tracker parameters
# --------------------------
TRACK_THRESH = 0.5
MATCH_THRESH = 0.7
INACTIVE_TTL = 30

@dataclass
class Track:
    track_id: int
    bbox: np.ndarray
    score: float
    last_frame: int
    inactive: int = 0
    history: list = field(default_factory=list)

class ByteTrackerSimple:
    def __init__(self, track_thresh=TRACK_THRESH, match_thresh=MATCH_THRESH, inactive_ttl=INACTIVE_TTL):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.inactive_ttl = inactive_ttl
        self.tracks: list[Track] = []
        self.next_id = 1

    def update(self, frame_idx: int, bboxes: np.ndarray, scores: np.ndarray):
        # mark all tracks inactive
        for t in self.tracks:
            t.inactive += 1

        if bboxes is None or len(bboxes) == 0:
            self._cleanup()
            return self.tracks

        N = len(bboxes)
        M = len(self.tracks)

        if M == 0:
            for i in range(N):
                if scores[i] >= self.track_thresh:
                    self._start_new_track(frame_idx, bboxes[i], scores[i])
            self._cleanup()
            return self.tracks

        # IoU cost matrix
        iou_mat = np.zeros((M, N), dtype=np.float32)
        for ti, trk in enumerate(self.tracks):
            for di in range(N):
                iou_mat[ti, di] = self._iou(trk.bbox, bboxes[di])

        unmatched_tracks = set(range(M))
        unmatched_dets = set(range(N))
        matches = []

        while unmatched_tracks and unmatched_dets:
            best_iou = self.match_thresh
            best_pair = None
            for ti in unmatched_tracks:
                for di in unmatched_dets:
                    val = iou_mat[ti, di]
                    if val >= best_iou:
                        best_iou = val
                        best_pair = (ti, di)
            if best_pair is None:
                break
            ti, di = best_pair
            matches.append((ti, di))
            unmatched_tracks.remove(ti)
            unmatched_dets.remove(di)

        for ti, di in matches:
            trk = self.tracks[ti]
            trk.bbox = bboxes[di]
            trk.score = scores[di]
            trk.last_frame = frame_idx
            trk.inactive = 0

        for di in unmatched_dets:
            if scores[di] >= self.track_thresh:
                self._start_new_track(frame_idx, bboxes[di], scores[di])

        self._cleanup()
        return self.tracks

    def _start_new_track(self, frame_idx, bbox, score):
        trk = Track(track_id=self.next_id, bbox=np.array(bbox, dtype=np.float32),
                    score=float(score), last_frame=frame_idx)
        self.next_id += 1
        self.tracks.append(trk)

    def _cleanup(self):
        self.tracks = [t for t in self.tracks if t.inactive <= self.inactive_ttl]

    @staticmethod
    def _iou(box1, box2):
        x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
        inter_w = max(0.0, x2 - x1); inter_h = max(0.0, y2 - y1)
        inter = inter_w * inter_h
        if inter <= 0: return 0.0
        area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
        area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
        union = area1 + area2 - inter + 1e-6
        return inter / union

# --------------------------
# I/O helpers
# --------------------------
def find_yolo_detections(root: Path, date_filter: str | None = None):
    for date_dir in sorted(root.iterdir()):
        if not date_dir.is_dir(): continue
        if date_filter and date_dir.name != date_filter: continue
        for vid_dir in sorted(date_dir.iterdir()):
            if not vid_dir.is_dir(): continue
            csv_path = vid_dir / "yolo_detections.csv"
            if csv_path.exists():
                yield csv_path

def run_bytetrack_on_video(yolo_csv: Path):
    df = pd.read_csv(yolo_csv)
    if df.empty: return None
    out_csv = yolo_csv.parent / "bytetrack_tracks.csv"
    if out_csv.exists(): return out_csv
    tracker = ByteTrackerSimple()
    all_rows = []
    df = df.sort_values("frame_idx").reset_index(drop=True)
    for _, row in df.iterrows():
        frame_idx = int(row["frame_idx"])
        timestamp = float(row["timestamp_sec"]) if not pd.isna(row["timestamp_sec"]) else None
        try: boxes = json.loads(row.get("json_boxes", "[]"))
        except: boxes = []
        if len(boxes) > 0:
            bboxes = np.array([[b["x1"], b["y1"], b["x2"], b["y2"]] for b in boxes], dtype=np.float32)
            scores = np.array([b.get("conf", 1.0) for b in boxes], dtype=np.float32)
        else:
            bboxes = np.zeros((0, 4), dtype=np.float32)
            scores = np.zeros((0,), dtype=np.float32)
        tracks = tracker.update(frame_idx, bboxes, scores)
        for t in tracks:
            if t.last_frame != frame_idx: continue
            x1, y1, x2, y2 = t.bbox.tolist()
            all_rows.append({
                "frame_idx": frame_idx,
                "timestamp_sec": timestamp,
                "track_id": t.track_id,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "score": t.score
            })
    pd.DataFrame(all_rows).to_csv(out_csv, index=False)
    return out_csv

# --------------------------
# Main
# --------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run ByteTrack-simple on YOLO outputs")
    parser.add_argument("--root", type=str, required=True, help="Processed root folder")
    parser.add_argument("--date", type=str, default=None)
    args = parser.parse_args()

    root = Path(args.root)
    assert root.exists(), f"Root not found: {root}"
    yolo_files = list(find_yolo_detections(root, args.date))
    print(f"[INFO] Found {len(yolo_files)} YOLO CSVs under {root}")

    done, skipped = 0, 0
    for yolo_csv in tqdm(yolo_files, desc="ByteTrack", unit="video"):
        out_csv = run_bytetrack_on_video(yolo_csv)
        if out_csv is None: skipped += 1
        else: done += 1
    print(f"[DONE] ByteTrack finished for {done} videos; skipped {skipped}")

if __name__ == "__main__":
    main()
