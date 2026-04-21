# Vision-Based Occupancy Measurement for Occupant-Centric Control

This repository contains the **Python code used in the paper**:

**"Experimental Study on Surveillance Video-Based Indoor Occupancy Measurement for Occupant-Centric Control"**  
Irfan Qaisar, Kailai Sun, Qingshan Jia, Qianchuan Zhao

---

## 📖 Overview

Accurate occupancy information is essential for occupant-centric control (OCC) in smart buildings. This project implements **detection-only, tracking-based, and reasoning-enhanced occupancy measurement pipelines** using surveillance video from a real research laboratory environment.

- **Detection-only:** YOLOv8 person detection.  
- **Tracking-based:** DeepSORT or ByteTrack for identity-consistent multi-object tracking.  
- **LLM-based refinement:** Optional reasoning step using DeepSeek or generic LLM to correct uncertain detector outputs.

The framework supports **frame-level occupancy counting**, optional reasoning-based refinement, and output suitable for downstream OCC or HVAC analysis.

---

## ⚙️ Installation

```bash
# Create virtual environment (recommended)
conda create -n vision_occ python=3.10
conda activate vision_occ

# Install dependencies
pip install -r requirements.txt
```
---

## 🗂 Folder Structure

```bash
  extract_frames.py
  manual_labeler.py
  detect_yolo.py
  count_occupancy_deepsort.py
  count_occupancy_bytetrack.py
  track_deepsort.py
  track_bytetrack.py
  select_for_llm.py
  llm_validate.py
  llm_validate_deepseek.py
```
---

## 🚀 Usage Examples
1. Preprocess videos
```bash
python src/extract_frames.py --input_dir data/raw_videos --output_dir data/frames
```

2. Run YOLOv8 detection
```bash
python src/detect_yolo.py --frames_dir data/frames --output_dir outputs/yolo
```

3. Tracking pipelines
```bash
python src/count_occupancy_deepsort.py --input_dir outputs/yolo --output_dir outputs/deepsort
python src/count_occupancy_bytetrack.py --input_dir outputs/yolo --output_dir outputs/bytetrack
```

4. LLM-based refinement
```bash
python src/select_for_llm.py --input_dir outputs/deepsort --output_dir outputs/llm_selected
python src/llm_validate_deepseek.py --input_dir outputs/llm_selected --output_dir outputs/llm_refined
```
---
## 🔬 References

- Citation will be available after publication.
