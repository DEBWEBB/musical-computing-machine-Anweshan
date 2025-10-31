# DancePose 3D — Intelligent Body & Gaze Tracking

**Developer:** Debjit Saha  
**Email:** sahadebjit357@gmail.com  
**GitHub:** https://github.com/DEBWEBB

## Overview
DancePose 3D detects body pose and face landmarks from video or webcam using MediaPipe,
computes a simple gaze vector, exports per-frame landmarks, and creates an interactive 3D avatar.
It includes a multi-person keypoint template using torchvision's Keypoint R-CNN and analysis scripts
to generate statistics and visualizations.

## Contents
- `pose_face_gaze_3d.py` — main single-person app (pose + face + gaze + 3D avatar)
- `multi_person_keypoints.py` — multi-person processing (Keypoint R-CNN template)
- `analyze_and_plot.py` — produces knee angle histograms, foot heatmap, step detection, gaze hist
- `requirements.txt` — dependencies
- `DEBJIT_RESUME.pdf` — your resume (included)
- `outputs/` — default output folder for generated JSON/CSV/HTML/plots

## Quick start (local)
1. Create and activate a Python environment (recommended Python 3.8+)
2. Install dependencies:
```bash
pip install -r app/requirements.txt
```
3. Run the main app (webcam):
```bash
python app/pose_face_gaze_3d.py --source 0
```
Or run with a video file:
```bash
python app/pose_face_gaze_3d.py --source path/to/video.mp4
```
4. After the run, use the JSON output with:
```bash
python app/analyze_and_plot.py --json app/outputs/landmarks_face_gaze_<ts>.json
```

## Deployment suggestions
- For a simple web UI, wrap `pose_face_gaze_3d.py` with **Streamlit** (WebRTC) or **Flask** + client-side video streaming.
- Use GPU-enabled environment for real-time multi-person inference (PyTorch + CUDA).
- Containerize with Docker for reproducible deployments.

## Notes & credits
- Pose & face detection use MediaPipe; multi-person template uses torchvision Keypoint R-CNN.
- Gaze vector is a simple geometric proxy (eye center toward nose). For production-grade gaze estimation, consider specialized gaze models.
- Author: Debjit Saha — B.Tech (Computer Science & Engineering), Narula Institute of Technology.