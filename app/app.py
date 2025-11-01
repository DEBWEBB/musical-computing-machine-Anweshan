# app.py — DancePose 3D Streamlit Web App (Stable Version)
# Author: Debjit Saha (DEBWEBB)
# Works locally and on Streamlit Cloud

import streamlit as st
import tempfile, os, cv2, time, json
import numpy as np
import mediapipe as mp
import plotly.graph_objs as go
import pandas as pd

st.set_page_config(page_title="DancePose 3D", layout="wide", page_icon="💃")
st.title("💃 DancePose 3D — Intelligent Dance Motion & Gaze Tracking")

st.markdown(
"""
Developed by **Debjit Saha (DEBWEBB)**  
🎓 B.Tech in Computer Science & Engineering (AI & ML)

This app analyzes dance movements — detecting poses, gaze direction, and generating a 3D avatar visualization.
"""
)

mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, smooth_landmarks=True)
face = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True)

st.sidebar.header("⚙️ Settings")
mode = st.sidebar.radio("Select Input Mode", ["Upload Video", "Use Webcam"])
generate_3d = st.sidebar.checkbox("Generate 3D Avatar", True)
save_json = st.sidebar.checkbox("Save Landmarks", True)
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

def process_frame(frame, frame_idx):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pr = pose.process(rgb)
    fr = face.process(rgb)
    result = {"frame": frame_idx, "pose": {}, "gaze": {}}

    if pr.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, pr.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255,255,0), thickness=2)
        )
        for i, lm in enumerate(pr.pose_landmarks.landmark):
            result["pose"][str(i)] = [lm.x, lm.y, lm.z]

    if fr.multi_face_landmarks and pr.pose_landmarks:
        fl = fr.multi_face_landmarks[0].landmark
        iris_l, iris_r = [468,469,470,471], [472,473,474,475]
        def avg_points(ids):
            pts = [np.array([fl[i].x, fl[i].y, fl[i].z]) for i in ids if i < len(fl)]
            return np.mean(pts, axis=0) if pts else None
        left_eye = avg_points(iris_l) or avg_points([33,133])
        right_eye = avg_points(iris_r) or avg_points([362,263])
        nose = np.array([pr.pose_landmarks.landmark[0].x, pr.pose_landmarks.landmark[0].y, pr.pose_landmarks.landmark[0].z])
        if left_eye is not None:
            gaze = nose - left_eye
            result["gaze"]["left"] = (gaze / (np.linalg.norm(gaze)+1e-8)).tolist()
        if right_eye is not None:
            gaze = nose - right_eye
            result["gaze"]["right"] = (gaze / (np.linalg.norm(gaze)+1e-8)).tolist()
    return frame, result

if mode == "Upload Video":
    video_file = st.file_uploader("🎥 Upload a dance video", type=["mp4", "mov", "avi", "mkv"])
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        st.info(f"Processing {total} frames...")
        json_data = []
        frame_window = st.empty()
        progress = st.progress(0)
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            frame, data = process_frame(frame, frame_idx)
            json_data.append(data)
            frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            progress.progress(frame_idx/total)
            time.sleep(0.01)  # smoother refresh
        cap.release()

        ts = int(time.time())
        if save_json:
            out_json = os.path.join(output_dir, f"landmarks_{ts}.json")
            with open(out_json, "w") as f:
                json.dump(json_data, f)
            st.success(f"✅ Saved landmarks to {out_json}")

        if generate_3d and json_data:
            last_pose = next((f["pose"] for f in reversed(json_data) if f["pose"]), None)
            if last_pose:
                xs, ys, zs = [], [], []
                for v in last_pose.values():
                    xs.append(v[0])
                    ys.append(-v[1])
                    zs.append(-v[2])
                fig = go.Figure(data=[go.Scatter3d(x=xs, y=ys, z=zs, mode='markers', marker=dict(size=4, color='red'))])
                fig.update_layout(scene=dict(aspectmode="data"), margin=dict(l=0, r=0, b=0, t=30))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ No pose data detected for 3D avatar.")

elif mode == "Use Webcam":
    st.warning("⚠️ Webcam mode only works locally (not on Streamlit Cloud).")
    st.code("python -m streamlit run app/app.py")

st.markdown("---")
st.header("📊 Motion Analytics Dashboard")
json_file = st.file_uploader("Upload a saved landmarks JSON file", type=["json"])
if json_file:
    data = json.load(json_file)
    st.success("✅ Data loaded successfully.")
    hips = []
    for f in data:
        try:
            h = np.array(f["pose"]["23"])
            k = np.array(f["pose"]["25"])
            a = np.array(f["pose"]["27"])
            ang = np.degrees(np.arccos(np.clip(np.dot(h-k, a-k)/(np.linalg.norm(h-k)*np.linalg.norm(a-k)+1e-8), -1, 1)))
            hips.append(ang)
        except Exception:
            continue
    if hips:
        st.subheader("🦵 Knee Angle Distribution")
        st.bar_chart(pd.Series(hips))
    gaze_hist = []
    for f in data:
        if f.get("gaze") and "left" in f["gaze"]:
            g = np.array(f["gaze"]["left"])
            gaze_hist.append(np.degrees(np.arctan2(g[1], g[0])))
    if gaze_hist:
        st.subheader("👁️ Gaze Direction Histogram")
        st.bar_chart(pd.Series(gaze_hist))

st.caption("© 2025 Debjit Saha — DancePose 3D")
