# manual_labeler.py
import os
from pathlib import Path

import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
ROOT = Path(r"D:\PythonCode\FIT622_processed")  # change if needed
GLOBAL_CSV = ROOT / "manual_labels.csv"

# ----------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------
def collect_frames(root: Path):
    """
    Walk the FIT622_processed structure and return a list of frame infos.

    Each item is a dict:
      {
        'rel_frame': '20181030/C2100.../frames/xxx.jpg',
        'video_rel': '20181030/C2100...'
      }
    """
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
                frames.append(
                    {
                        "rel_frame": str(rel_frame).replace("\\", "/"),
                        "video_rel": str(video_rel).replace("\\", "/"),
                    }
                )
    return frames


def load_global_labels():
    """
    Load the global manual_labels.csv (if it exists) and normalise columns.
    """
    if GLOBAL_CSV.exists():
        df = pd.read_csv(GLOBAL_CSV)

        # Old version might have used "frame" column instead of "rel_frame"
        if "rel_frame" not in df.columns and "frame" in df.columns:
            df = df.rename(columns={"frame": "rel_frame"})

        if "count" not in df.columns:
            df["count"] = 0

        return df[["rel_frame", "count"]]

    return pd.DataFrame(columns=["rel_frame", "count"])


def write_global_labels(df: pd.DataFrame):
    df_sorted = df.sort_values("rel_frame")
    df_sorted.to_csv(GLOBAL_CSV, index=False)


def save_per_video_label(frame_info: dict, count: int):
    """
    Save/update per-video manual_labels.csv for the given frame.
    """
    video_rel = frame_info["video_rel"]
    rel_frame = frame_info["rel_frame"]
    video_dir = ROOT / video_rel
    video_csv = video_dir / "manual_labels.csv"

    # Frame name only for per-video CSV
    frame_name = Path(rel_frame).name

    if video_csv.exists():
        vdf = pd.read_csv(video_csv)
        if "frame" not in vdf.columns:
            vdf.rename(columns={vdf.columns[0]: "frame"}, inplace=True)
        if "count" not in vdf.columns:
            vdf["count"] = 0
    else:
        vdf = pd.DataFrame(columns=["frame", "count"])

    if (vdf["frame"] == frame_name).any():
        vdf.loc[vdf["frame"] == frame_name, "count"] = count
    else:
        vdf = pd.concat(
            [vdf, pd.DataFrame([{"frame": frame_name, "count": count}])],
            ignore_index=True,
        )

    vdf.sort_values("frame").to_csv(video_csv, index=False)


def remove_per_video_label(frame_info: dict):
    """
    Remove a label from the per-video CSV (used by Undo).
    """
    video_rel = frame_info["video_rel"]
    rel_frame = frame_info["rel_frame"]
    video_dir = ROOT / video_rel
    video_csv = video_dir / "manual_labels.csv"
    frame_name = Path(rel_frame).name

    if not video_csv.exists():
        return

    vdf = pd.read_csv(video_csv)
    if "frame" not in vdf.columns:
        vdf.rename(columns={vdf.columns[0]: "frame"}, inplace=True)

    vdf = vdf[vdf["frame"] != frame_name]
    vdf.to_csv(video_csv, index=False)


def get_initial_index(frames, labels_df):
    """
    First unlabeled frame index.
    """
    labeled = set(labels_df["rel_frame"])
    for i, f in enumerate(frames):
        if f["rel_frame"] not in labeled:
            return i
    return 0


def go_to_next_unlabeled(current_idx, frames, labels_df):
    labeled = set(labels_df["rel_frame"])

    # Search forwards
    for j in range(current_idx + 1, len(frames)):
        if frames[j]["rel_frame"] not in labeled:
            st.session_state.idx = j
            return

    # Wrap around from the start
    for j in range(0, len(frames)):
        if frames[j]["rel_frame"] not in labeled:
            st.session_state.idx = j
            return

    # Everything labeled
    st.session_state.done = True


# ----------------------------------------------------------------------
# STREAMLIT APP
# ----------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Manual Occupancy Labeler", layout="wide")

    # ----- Initialise session state -----
    if "frames" not in st.session_state:
        st.session_state.frames = collect_frames(ROOT)

    if "labels_df" not in st.session_state:
        st.session_state.labels_df = load_global_labels()

    if "idx" not in st.session_state:
        st.session_state.idx = get_initial_index(
            st.session_state.frames, st.session_state.labels_df
        )

    if "history" not in st.session_state:
        st.session_state.history = []

    if "done" not in st.session_state:
        st.session_state.done = False

    frames = st.session_state.frames
    labels_df = st.session_state.labels_df
    idx = st.session_state.idx

    # ----- Sidebar -----
    with st.sidebar:
        st.header("📁 Dataset Status")
        st.code(str(ROOT), language="text")
        total_videos = len({f["video_rel"] for f in frames})
        total_frames = len(frames)
        labeled_frames = len(labels_df)

        st.write(f"**Total videos detected:** {total_videos}")
        st.write(f"**Total frames:** {total_frames}")
        st.write(f"**Labeled frames:** {labeled_frames}")
        st.write(f"**Pending frames:** {total_frames - labeled_frames}")

    st.title("👁️ Manual Occupancy Ground-Truth Labeling Tool")

    if not frames:
        st.warning("No frames found under the root directory.")
        return

    if st.session_state.done or labeled_frames == len(frames):
        st.success("🎉 All videos fully labeled!")
        return

    # Current frame info
    frame_info = frames[idx]
    rel_frame = frame_info["rel_frame"]
    img_path = ROOT / rel_frame

    # If already labeled, use that as default
    current_count = 0
    row = labels_df[labels_df["rel_frame"] == rel_frame]
    if not row.empty:
        current_count = int(row["count"].iloc[0])

    # ----- Show image -----
    st.image(
        str(img_path),
        caption=f"{rel_frame} ({idx + 1} / {len(frames)} frames)",
        use_container_width=True,
    )

    # ----- Controls -----
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        count = st.number_input(
            "Number of visible occupants",
            min_value=0,
            max_value=30,
            value=current_count,
            step=1,
        )

    with col2:
        save_clicked = st.button("💾 Save & Next", type="primary")

    with col3:
        undo_disabled = len(st.session_state.history) == 0
        undo_clicked = st.button("↩️ Undo last", disabled=undo_disabled)

    # ----- Button logic -----
    if save_clicked:
        # Update global labels
        df = st.session_state.labels_df
        if (df["rel_frame"] == rel_frame).any():
            df.loc[df["rel_frame"] == rel_frame, "count"] = count
        else:
            df = pd.concat(
                [df, pd.DataFrame([{"rel_frame": rel_frame, "count": count}])],
                ignore_index=True,
            )
        st.session_state.labels_df = df
        write_global_labels(df)

        # Update per-video CSV
        save_per_video_label(frame_info, count)

        # Push to history (for Undo)
        st.session_state.history.append(rel_frame)

        # Move to next frame
        go_to_next_unlabeled(idx, frames, df)
        st.rerun()

    if undo_clicked:
        # Pop last frame from history
        last_rel = st.session_state.history.pop()

        # Find its frame_info
        last_info = None
        last_idx = 0
        for j, f in enumerate(frames):
            if f["rel_frame"] == last_rel:
                last_info = f
                last_idx = j
                break

        if last_info is not None:
            # Remove from global labels
            df = st.session_state.labels_df
            df = df[df["rel_frame"] != last_rel]
            st.session_state.labels_df = df
            write_global_labels(df)

            # Remove from per-video CSV
            remove_per_video_label(last_info)

            # Go back to that frame
            st.session_state.idx = last_idx
            st.session_state.done = False
            st.rerun()


if __name__ == "__main__":
    main()
