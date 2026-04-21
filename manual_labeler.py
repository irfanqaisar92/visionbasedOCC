"""
manual_labeler.py
-----------------

Streamlit-based tool for manual labeling of occupancy in surveillance frames.

This tool allows the user to assign ground-truth counts to frames and maintains:
- Per-video manual_labels.csv
- Global manual_labels.csv

This script is part of the data preparation pipeline described in:

"Experimental Study on Surveillance Video-Based Indoor Occupancy Measurement 
for Occupant-Centric Control" (citation will be available after publication).
"""

import os
from pathlib import Path
import pandas as pd
import streamlit as st
import argparse

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def collect_frames(root: Path):
    """Walk the folder structure and return a list of frame info dicts."""
    frames = []
    if not root.exists():
        return frames

    for day_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for video_dir in sorted(p for p in day_dir.iterdir() if p.is_dir()):
            frames_dir = video_dir / "frames"
            if not frames_dir.is_dir():
                continue
            for img in sorted(frames_dir.glob("*.jpg")):
                rel_frame = img.relative_to(root)
                video_rel = video_dir.relative_to(root)
                frames.append({
                    "rel_frame": str(rel_frame).replace("\\", "/"),
                    "video_rel": str(video_rel).replace("\\", "/"),
                })
    return frames

def load_global_labels(global_csv: Path):
    """Load the global manual_labels.csv (if it exists)."""
    if global_csv.exists():
        df = pd.read_csv(global_csv)
        if "rel_frame" not in df.columns and "frame" in df.columns:
            df = df.rename(columns={"frame": "rel_frame"})
        if "count" not in df.columns:
            df["count"] = 0
        return df[["rel_frame", "count"]]
    return pd.DataFrame(columns=["rel_frame", "count"])

def write_global_labels(df: pd.DataFrame, global_csv: Path):
    df_sorted = df.sort_values("rel_frame")
    df_sorted.to_csv(global_csv, index=False)

# Per-video CSV functions (save, remove) remain the same but use root argument

# ----------------------------------------------------------------------
# Streamlit app
# ----------------------------------------------------------------------
def run_app(root: Path):
    GLOBAL_CSV = root / "manual_labels.csv"

    st.set_page_config(page_title="Manual Occupancy Labeler", layout="wide")

    if "frames" not in st.session_state:
        st.session_state.frames = collect_frames(root)
    if "labels_df" not in st.session_state:
        st.session_state.labels_df = load_global_labels(GLOBAL_CSV)
    if "idx" not in st.session_state:
        st.session_state.idx = 0
    if "history" not in st.session_state:
        st.session_state.history = []
    if "done" not in st.session_state:
        st.session_state.done = False

    frames = st.session_state.frames
    labels_df = st.session_state.labels_df
    idx = st.session_state.idx

    st.sidebar.header("📁 Dataset Status")
    st.sidebar.code(str(root), language="text")
    st.sidebar.write(f"**Total videos detected:** {len({f['video_rel'] for f in frames})}")
    st.sidebar.write(f"**Total frames:** {len(frames)}")
    st.sidebar.write(f"**Labeled frames:** {len(labels_df)}")
    st.sidebar.write(f"**Pending frames:** {len(frames) - len(labels_df)}")

    st.title("👁️ Manual Occupancy Ground-Truth Labeling Tool")
    if not frames:
        st.warning("No frames found under the root directory.")
        return
    if st.session_state.done or len(labels_df) == len(frames):
        st.success("🎉 All videos fully labeled!")
        return

    # Current frame
    frame_info = frames[idx]
    rel_frame = frame_info["rel_frame"]
    img_path = root / rel_frame
    current_count = 0
    row = labels_df[labels_df["rel_frame"] == rel_frame]
    if not row.empty:
        current_count = int(row["count"].iloc[0])

    st.image(str(img_path), caption=f"{rel_frame} ({idx + 1}/{len(frames)})", use_container_width=True)

    col1, col2, col3 = st.columns([2,1,1])
    with col1:
        count = st.number_input("Number of visible occupants", min_value=0, max_value=30, value=current_count, step=1)
    with col2:
        save_clicked = st.button("💾 Save & Next", type="primary")
    with col3:
        undo_disabled = len(st.session_state.history) == 0
        undo_clicked = st.button("↩️ Undo last", disabled=undo_disabled)

    if save_clicked:
        df = st.session_state.labels_df
        if (df["rel_frame"] == rel_frame).any():
            df.loc[df["rel_frame"] == rel_frame, "count"] = count
        else:
            df = pd.concat([df, pd.DataFrame([{"rel_frame": rel_frame, "count": count}])], ignore_index=True)
        st.session_state.labels_df = df
        write_global_labels(df, GLOBAL_CSV)
        # Save per-video CSV
        # save_per_video_label(frame_info, count, root=root)
        st.session_state.history.append(rel_frame)
        # Move to next
        st.session_state.idx = (idx + 1) % len(frames)
        st.rerun()

    if undo_clicked:
        last_rel = st.session_state.history.pop()
        # Remove from global and per-video CSVs
        df = st.session_state.labels_df
        df = df[df["rel_frame"] != last_rel]
        st.session_state.labels_df = df
        # remove_per_video_label(last_info, root=root)
        st.session_state.idx = idx
        st.session_state.done = False
        st.rerun()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Root folder containing frames")
    args = parser.parse_args()
    run_app(Path(args.root))
