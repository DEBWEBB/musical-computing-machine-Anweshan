# app.py — DancePose 3D Streamlit Web App
# Author: Debjit Saha (DEBWEBB)
# Works on Streamlit Cloud and local machine (Windows/Linux/Mac)

import streamlit as st
import tempfile, os, cv2, time, json
import numpy as np
import mediapipe as mp
import plotly.graph_objs as go
import pandas as pd

# ==============================
# Streamlit Page Config
# ==============================
st.set_page_config(page_title="DancePose 3D", layout="wide", page_icon="💃")

st.title("💃 DancePose 3D — Intelligent Dance Motion & Gaze Tracking")
st.markdown(
"""
Developed by **Debjit Saha (DEBWEBB)**  
🎓 B.Tech in Computer Science & Engineering (AI & ML)  
📧 [sahadebjit357@gmail.com](mailto:sahadebjit357@gmail.com)

This app detects body pose, facial gaze, and movement analytics from dance videos or webcam streams.
"""
)

# ==============================
# Initialize MediaPipe
# ==============================
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(model_complexity=1, smooth_landmarks=True)
face = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True)

# ==============================
# Sidebar Settings
# ==============================
st.sidebar.header("⚙️ Settings")
mode = st.sidebar.radio("Input Mode", ["Upload Video", "Use Webcam"])
generate_3d = st.sidebar.checkbox("Generate 3D Avatar", True)
save_json = st.sidebar.checkbox("Save Landmarks (JSON)", True)
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# ==============================
# Helper Functions
# ==============================
def process_frame(frame, frame_idx):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pr = pose.process(rgb)
    fr = face.process(rgb)
    result = {"frame": frame_idx, "pose": {}, "face": {}, "gaze": {}}

    if pr.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, pr.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255,255,0), thickness=2)
        )
        for i, lm in enumerate(pr.pose_landmarks.landmark):
            result["pose"][str(i)] = [lm.x, lm.y, lm.z]

    if fr.multi_face_landmarks:
        fl = fr.multi_face_landmarks[0].landmark
        iris_l, iris_r = [468,469,470,471], [472,473,474,475]

        def avg_points(ids):
            pts = [np.array([fl[i].x, fl[i].y, fl[i].z]) for i in ids if i < len(fl)]
            return np.mean(pts, axis=0) if pts else None

        left_eye = avg_points(iris_l) or avg_points([33,133])
        right_eye = avg_points(iris_r) or avg_points([362,263])
        nose = None
        if pr.pose_landmarks:
            n = pr.pose_landmarks.landmark[0]
            nose = np.array([n.x, n.y, n.z])

        if left_eye is not None and nose is not None:
            gaze_vec = nose - left_eye
            gaze_vec /= np.linalg.norm(gaze_vec) + 1e-8
            result["gaze"]["left"] = gaze_vec.tolist()
        if right_eye is not None and nose is not None:
            gaze_vec = nose - right_eye
            gaze_vec /= np.linalg.norm(gaze_vec) + 1e-8
            result["gaze"]["right"] = gaze_vec.tolist()
    return frame, result

# ==============================
# Main Processing
# ==============================
if mode == "Upload Video":
    video_file = st.file_uploader("🎥 Upload a Dance Video", type=["mp4", "mov", "avi", "mkv"])
    if video_file:
        temp = tempfile.NamedTemporaryFile(delete=False)
        temp.write(video_file.read())
        cap = cv2.VideoCapture(temp.name)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        st.info(f"Processing {total} frames... Please wait.")
        json_data = []
        frame_box = st.empty()
        progress = st.progress(0)

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            frame, data = process_frame(frame, frame_idx)
            json_data.append(data)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_box.image(frame_rgb, channels="RGB", use_container_width=True)
            progress.progress(frame_idx / total)
        cap.release()

        # Save JSON
        ts = int(time.time())
        if save_json:
            json_path = os.path.join(output_dir, f"landmarks_{ts}.json")
            with open(json_path, "w") as f:
                json.dump(json_data, f)
            st.success(f"✅ Landmarks saved at `{json_path}`")

        # Generate 3D Avatar
        if generate_3d and json_data:
            last_pose = next((f["pose"] for f in reversed(json_data) if f["pose"]), None)
            if last_pose:
                xs, ys, zs = [], [], []
                for k, v in last_pose.items():
                    xs.append(v[0])
                    ys.append(-v[1])
                    zs.append(-v[2])
                fig = go.Figure(data=[
                    go.Scatter3d(x=xs, y=ys, z=zs, mode='markers', marker=dict(size=4, color='red'))
                ])
                fig.update_layout(title="3D Avatar", scene=dict(aspectmode="data"))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No pose data detected for 3D rendering.")

elif mode == "Use Webcam":
    st.warning("⚠️ Webcam mode only works locally (not on Streamlit Cloud).")
    st.info("Run locally using:\n\n```bash\npython -m streamlit run app/app.py\n```")

# ==============================
# Analysis Section
# ==============================
st.markdown("---")
st.header("📊 Motion Analytics Dashboard")

json_file = st.file_uploader("Upload Landmarks JSON for Analysis", type=["json"], key="analytics")
if json_file:
    data = json.load(json_file)
    st.success("✅ Landmarks loaded successfully!")

    # Knee Angle Distribution
    def extract_joint(id, frame_data):
        if "pose" in frame_data and str(id) in frame_data["pose"]:
            return np.array(frame_data["pose"][str(id)])
        return None

    hips, knees, ankles = [], [], []
    for frame in data:
        h, k, a = extract_joint(23, frame), extract_joint(25, frame), extract_joint(27, frame)
        if h is not None and k is not None and a is not None:
            v1 = h - k
            v2 = a - k
            ang = np.degrees(np.arccos(np.clip(np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-8), -1, 1)))
            hips.append(ang)
    if hips:
        st.subheader("🦵 Knee Angle Distribution")
        st.bar_chart(pd.Series(hips))

    # Gaze direction histogram
    gaze_angles = []
    for frame in data:
        if "gaze" in frame and "left" in frame["gaze"]:
            g = np.array(frame["gaze"]["left"])
            gaze_angles.append(np.degrees(np.arctan2(g[1], g[0])))
    if gaze_angles:
        st.subheader("👁️ Gaze Direction Histogram")
        st.bar_chart(pd.Series(gaze_angles))

st.caption("© 2025 Debjit Saha — DancePose 3D")
