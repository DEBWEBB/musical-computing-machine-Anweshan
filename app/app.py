# app.py — DancePose Live Studio Pro v5.5 (Smooth Real-Time Playback + Replay)
import streamlit as st
import cv2
import numpy as np
import tempfile
import mediapipe as mp
import plotly.graph_objects as go
import time
from pose_utils import (
    compute_knee_angles_from_pose,
    compute_hand_distance,
    compute_head_height,
    compute_motion_energy,
    compute_symmetry_score,
)

st.set_page_config(page_title="DancePose Live Studio Pro v5.5", layout="wide", page_icon="💃")

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Header
st.title("💃 DancePose Live Studio Pro v5.5 — Smooth Live Edition")
st.caption("Track your dance motion with real-time skeleton visualization, smooth replay, and performance insights.")

# Sidebar
st.sidebar.header("🎛️ Controls")
video_file = st.sidebar.file_uploader("🎥 Upload Dance Video", type=["mp4", "mov", "avi", "mkv"])
show_skeleton = st.sidebar.checkbox("Show Skeleton Overlay", True)
trace_mode = st.sidebar.checkbox("Enable Motion Trails", True)
frame_skip = st.sidebar.slider("Frame Skip (for speed)", 1, 10, 2)
show_live_chart = st.sidebar.checkbox("Show Live Knee Chart", True)
max_frames = st.sidebar.number_input("Max Frames (0 = All)", 0, 1500, 0)

st.sidebar.markdown("---")
start_analysis = st.sidebar.button("🚀 Start Analysis")

if video_file and start_analysis:
    # Temporary save
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    FRAME_UPDATE_INTERVAL = max(int(fps // 2), 1)  # update twice per second

    stframe_original = st.empty()
    stframe_annot = st.empty()
    chart_placeholder = st.empty()
    progress = st.progress(0)

    st.info("⏳ Processing video frames... please wait.")

    # Data buffers
    frame_idx = 0
    head_trail, left_angles, right_angles = [], [], []
    hand_dists, head_heights = [], []
    annotated_frames = []  # store for replay

    with mp_pose.Pose(model_complexity=1) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if max_frames and frame_idx > max_frames:
                break
            if frame_idx % frame_skip != 0:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            annotated = rgb.copy()

            if results.pose_landmarks:
                pose_dict = {i: [lm.x, lm.y, lm.z, lm.visibility] for i, lm in enumerate(results.pose_landmarks.landmark)}
                l_angle, r_angle = compute_knee_angles_from_pose(pose_dict)
                left_angles.append(l_angle)
                right_angles.append(r_angle)
                hand_dists.append(compute_hand_distance(pose_dict))
                head_heights.append(compute_head_height(pose_dict))

                if show_skeleton:
                    mp_drawing.draw_landmarks(
                        annotated,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                    )

                h, w = frame.shape[:2]
                hx, hy = int(pose_dict[0][0] * w), int(pose_dict[0][1] * h)
                head_trail.append((hx, hy))

                if trace_mode and len(head_trail) > 5:
                    cv2.polylines(annotated, [np.int32(head_trail[-50:])], False, (255, 255, 0), 2)

            annotated_frames.append(annotated)

            # Stream every few frames for smoother experience
            if frame_idx % FRAME_UPDATE_INTERVAL == 0:
                stframe_original.image(rgb, use_container_width=True, caption=f"Original (Frame {frame_idx})")
                stframe_annot.image(annotated, use_container_width=True, caption="Pose Analysis")

                if show_live_chart and len(left_angles) > 10:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=left_angles, mode="lines", name="Left Knee"))
                    fig.add_trace(go.Scatter(y=right_angles, mode="lines", name="Right Knee"))
                    fig.update_layout(height=200, margin=dict(l=20, r=20, t=30, b=20))
                    chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"chart_{frame_idx}")

            if total_frames:
                progress.progress(min(frame_idx / total_frames, 1.0))

        cap.release()

    st.success("✅ Analysis Complete!")

    # Summary Stats
    energy = compute_motion_energy(np.array(head_trail))
    symmetry = compute_symmetry_score(left_angles, right_angles)
    avg_knee = np.nanmean([np.nanmean(left_angles), np.nanmean(right_angles)])

    col1, col2, col3 = st.columns(3)
    col1.metric("💫 Motion Energy", f"{energy:.2f}")
    col2.metric("🪞 Symmetry Index", f"{symmetry*100:.1f}%")
    col3.metric("🦵 Avg Knee Angle", f"{avg_knee:.1f}°")

    st.markdown("### 🎬 Replay Analyzed Video")
    if st.button("▶️ Replay"):
        for idx, frame in enumerate(annotated_frames[::frame_skip]):
            stframe_annot.image(frame, use_container_width=True, caption=f"Replay Frame {idx}")
            time.sleep(1 / fps)

else:
    st.info("📂 Upload a dance video and click **Start Analysis** to begin.")
