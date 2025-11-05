# DancePose Live Studio Pro v5.4 Elite Edition — Paste final app.py code here
# app.py — DancePose Live Studio Pro v5.4 (Elite Edition with Report Thumbnails)
import streamlit as st
import cv2
import numpy as np
import time
import tempfile
import mediapipe as mp
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from pose_utils import (
    compute_knee_angles_from_pose,
    compute_hand_distance,
    compute_head_height,
    compute_motion_energy,
    compute_symmetry_score,
)

st.set_page_config(page_title="DancePose Live Studio Pro v5.4", layout="wide", page_icon="💃")

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# App Header
st.title("💃 DancePose Live Studio Pro v5.4 — Elite Edition")
st.caption("Real-time AI dance analytics, visual skeleton tracking, performance metrics, and exportable professional reports.")

# Sidebar Controls
st.sidebar.header("🎛️ Controls")
upload = st.sidebar.file_uploader("📤 Upload Dance Video", type=["mp4", "mov", "avi", "mkv"])
show_skeleton = st.sidebar.checkbox("Show Skeleton Overlay", True)
trace_mode = st.sidebar.checkbox("Enable Trace Mode", True)
frame_skip = st.sidebar.slider("Frame Skip (for speed)", 1, 10, 2)
speed = st.sidebar.slider("Playback Speed", 0.3, 2.0, 1.0)
max_frames = st.sidebar.number_input("Max Frames to Process (0 = All)", 0, 2000, 0)
show_live_chart = st.sidebar.checkbox("Show Live Angle Graph", True)

st.sidebar.divider()
st.sidebar.markdown("👨‍💻 Developed by **Debjit Saha**")

if upload:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(upload.read())
    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    col1, col2 = st.columns(2)
    with col1:
        placeholder_orig = st.empty()
    with col2:
        placeholder_annot = st.empty()

    chart_placeholder = st.empty()
    progress = st.progress(0)

    frame_idx = 0
    hand_trail_L, hand_trail_R, head_trail = [], [], []
    left_angles, right_angles, hand_dists, head_heights = [], [], [], []
    thumbnail_img = None  # To capture one tracked frame for report

    with mp_pose.Pose(model_complexity=1) as pose:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
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

            orig_view = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)

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
                try:
                    lx, ly = int(pose_dict[15][0]*w), int(pose_dict[15][1]*h)
                    rx, ry = int(pose_dict[16][0]*w), int(pose_dict[16][1]*h)
                    hx, hy = int(pose_dict[0][0]*w), int(pose_dict[0][1]*h)
                    hand_trail_L.append((lx, ly))
                    hand_trail_R.append((rx, ry))
                    head_trail.append((hx, hy))
                    if trace_mode:
                        cv2.polylines(annotated, [np.int32(hand_trail_L[-30:])], False, (0, 255, 255), 2)
                        cv2.polylines(annotated, [np.int32(hand_trail_R[-30:])], False, (255, 255, 0), 2)
                        cv2.polylines(annotated, [np.int32(head_trail[-30:])], False, (255, 100, 100), 2)
                except:
                    pass

                if thumbnail_img is None:
                    thumbnail_img = annotated.copy()

            annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            placeholder_orig.image(orig_view, use_container_width=True, caption="Original")
            placeholder_annot.image(annotated_bgr, channels="BGR", use_container_width=True, caption="Analysis")
            progress.progress(frame_idx / total_frames)

            if show_live_chart and len(left_angles) > 5 and frame_idx % 5 == 0:
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=left_angles, mode="lines", name="Left Knee"))
                fig.add_trace(go.Scatter(y=right_angles, mode="lines", name="Right Knee"))
                fig.update_layout(title="Knee Angles Over Time", height=250, margin=dict(l=20, r=20, t=40, b=20))
                chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"knee_chart_{frame_idx}")

            time.sleep(1.0 / (fps * speed))

    cap.release()

    energy = compute_motion_energy(np.array(head_trail))
    symmetry = compute_symmetry_score(left_angles, right_angles)
    avg_knee = np.nanmean([np.nanmean(left_angles), np.nanmean(right_angles)])

    st.subheader("📊 Performance Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("💫 Motion Energy", f"{energy:.2f}")
    col2.metric("🪞 Symmetry Index", f"{symmetry*100:.1f}%")
    col3.metric("🦵 Avg Knee Angle", f"{avg_knee:.1f}°")

    st.subheader("🎙 AI Coach Feedback")
    feedback = []
    if symmetry < 0.8:
        feedback.append("⚠️ Improve symmetry between left and right movements.")
    if energy < 10:
        feedback.append("💡 Increase upper-body dynamics for better stage presence.")
    if avg_knee < 120:
        feedback.append("✅ Excellent knee control — great for stylistic moves.")
    if not feedback:
        feedback.append("🔥 Fantastic balance and energy — keep it up!")

    for tip in feedback:
        st.write(tip)

    st.subheader("🧾 Export Your Performance Report")
    if st.button("📄 Download PDF Report"):
        buffer = BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=letter)

        # Header
        pdf.setFillColor(colors.HexColor("#FF5C8A"))
        pdf.rect(0, 740, 612, 60, fill=True, stroke=False)
        pdf.setFillColor(colors.white)
        pdf.setFont("Helvetica-Bold", 22)
        pdf.drawString(40, 765, "DancePose Live Studio Pro v5.4")

        # Metrics
        pdf.setFillColor(colors.black)
        pdf.setFont("Helvetica", 12)
        pdf.drawString(40, 720, f"💫 Motion Energy: {energy:.2f}")
        pdf.drawString(40, 705, f"🪞 Symmetry Index: {symmetry*100:.1f}%")
        pdf.drawString(40, 690, f"🦵 Average Knee Angle: {avg_knee:.1f}°")

        pdf.drawString(40, 665, "🎙 AI Feedback:")
        y = 650
        for tip in feedback:
            pdf.drawString(60, y, f"- {tip}")
            y -= 15

        # Knee Angle Chart
        plt.figure(figsize=(4, 2))
        plt.plot(left_angles, label="Left Knee", color="blue")
        plt.plot(right_angles, label="Right Knee", color="red")
        plt.title("Knee Angles Over Time")
        plt.legend()
        plt.tight_layout()
        img_buf = BytesIO()
        plt.savefig(img_buf, format="png")
        img_buf.seek(0)
        chart_img = ImageReader(img_buf)
        pdf.drawImage(chart_img, 40, 400, width=400, height=150)

        # Skeleton thumbnail
        if thumbnail_img is not None:
            thumb_buf = BytesIO()
            cv2.imwrite("thumb_temp.jpg", cv2.cvtColor(thumbnail_img, cv2.COLOR_RGB2BGR))
            thumb_img = ImageReader("thumb_temp.jpg")
            pdf.drawImage(thumb_img, 460, 400, width=120, height=150)

        # Footer
        pdf.setFont("Helvetica-Oblique", 9)
        pdf.setFillColor(colors.grey)
        pdf.drawString(40, 30, "Generated by DancePose Live Studio Pro v5.4 — Debjit Saha © 2025")

        pdf.save()
        buffer.seek(0)

        st.download_button(
            label="⬇️ Download PDF Report",
            data=buffer,
            file_name="DancePose_Report_v5.4.pdf",
            mime="application/pdf",
        )

    st.success("✅ Analysis complete! You can replay or re-upload another dance clip.")
else:
    st.info("🎥 Upload a video to start analyzing your dance performance.")
