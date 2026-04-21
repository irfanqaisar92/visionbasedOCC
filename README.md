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

> “Among the evaluated pipelines, YOLOv8 + DeepSeek achieves the best overall performance, with an accuracy of 0.8824 and an F1-score of 0.9320. This demonstrates that selective reasoning-based refinement can improve temporal stability and reduce false unoccupied predictions, which is particularly important for control-oriented applications”:contentReference[oaicite:0]{index=0}.

---

## ⚙️ Installation

```bash
# Create virtual environment (recommended)
conda create -n vision_occ python=3.10
conda activate vision_occ

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage Examples
1. Preprocess videos
python src/extract_frames.py --input_dir data/raw_videos --output_dir data/frames

2. Run YOLOv8 detection

python src/detect_yolo.py --frames_dir data/frames --output_dir outputs/yolo

3. Tracking pipelines

python src/count_occupancy_deepsort.py --input_dir outputs/yolo --output_dir outputs/deepsort
python src/count_occupancy_bytetrack.py --input_dir outputs/yolo --output_dir outputs/bytetrack

4. LLM-based refinement

python src/select_for_llm.py --input_dir outputs/deepsort --output_dir outputs/llm_selected
python src/llm_validate_deepseek.py --input_dir outputs/llm_selected --output_dir outputs/llm_refined

