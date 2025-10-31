# app.py — DancePose 3D Streamlit Web App
import streamlit as st
import tempfile, os, cv2, time, json
import numpy as np
import mediapipe as mp
import plotly.graph_objs as go
import pandas as pd
from analyze_and_plot import extract_landmark_series, angle_between

st.set_page_config(page_title="DancePose 3D", layout="wide", page_icon="💃")

st.title("💃 DancePose 3D — Intelligent Dance Motion & Gaze Tracking")
st.markdown(
"""
**Developer:** [Debjit Saha](mailto:sahadebjit357@gmail.com)  
Built with ❤️ using Python, OpenCV, MediaPipe & Streamlit.
"""
)

mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
pose = mp_pose.Pose(model_complexity=1, smooth_landmarks=True)
face = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True)

# ==============================
# Sidebar Options
# ==============================
st.sidebar.header("⚙️ Settings")
mode = st.sidebar.radio("Select Input Mode", ["Upload Video", "Use Webcam"])
show_3d = st.sidebar.checkbox("Generate 3D Avatar", True)
save_results = st.sidebar.checkbox("Save Landmarks", True)
out_dir = "outputs"
os.makedirs(out_dir, exist_ok=True)

# ==============================
# Helper: Draw landmarks
# ==============================
mp_drawing = mp.solutions.drawing_utils

def process_frame(frame, frame_idx):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pr = pose.process(rgb)
    fr = face.process(rgb)
    frame_json = {"frame": frame_idx, "pose": {}, "face": {}, "gaze": {}}

    if pr.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, pr.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255,255,0), thickness=2)
        )
        for i, lm in enumerate(pr.pose_landmarks.landmark):
            frame_json["pose"][str(i)] = [lm.x, lm.y, lm.z]

    if fr.multi_face_landmarks:
        face_lms = fr.multi_face_landmarks[0].landmark
        iris_l, iris_r = [468,469,470,471], [472,473,474,475]

        def avg_pts(indices):
            pts = []
            for idx in indices:
                if idx < len(face_lms):
                    lm = face_lms[idx]
                    pts.append(np.array([lm.x, lm.y, lm.z]))
            if not pts: return None
            return np.mean(pts, axis=0)

        left_eye = avg_pts(iris_l) or avg_pts([33,133])
        right_eye = avg_pts(iris_r) or avg_pts([362,263])
        nose_pt = None
        if pr.pose_landmarks:
            n = pr.pose_landmarks.landmark[0]
            nose_pt = np.array([n.x, n.y, n.z])

        if left_eye is not None and nose_pt is not None:
            gv = nose_pt - left_eye
            gv = gv / (np.linalg.norm(gv)+1e-8)
            frame_json["gaze"]["left"] = gv.tolist()
        if right_eye is not None and nose_pt is not None:
            gv = nose_pt - right_eye
            gv = gv / (np.linalg.norm(gv)+1e-8)
            frame_json["gaze"]["right"] = gv.tolist()

    return frame, frame_json

# ==============================
# Main App Logic
# ==============================
if mode == "Upload Video":
    video_file = st.file_uploader("Upload a dance video", type=["mp4", "mov", "avi"])
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)
        json_frames = []
        frame_placeholder = st.empty()
        frame_idx = 0
        progress = st.progress(0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1
            frame, fjson = process_frame(frame, frame_idx)
            json_frames.append(fjson)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")
            progress.progress(min(frame_idx/total,1.0))
        cap.release()

        ts = int(time.time())
        if save_results:
            json_path = os.path.join(out_dir, f"landmarks_face_gaze_{ts}.json")
            with open(json_path,"w") as fh: json.dump(json_frames, fh)
            st.success(f"Saved landmarks → {json_path}")

        # 3D Avatar
        if show_3d and json_frames:
            last_pose = None
            for f in reversed(json_frames):
                if f["pose"]:
                    last_pose = f["pose"]; break
            if last_pose:
                xs,ys,zs,labels=[],[],[],[]
                for k in sorted(last_pose.keys(), key=lambda x:int(x)):
                    v = last_pose[k]; xs.append(v[0]); ys.append(-v[1]); zs.append(-v[2]); labels.append(k)
                connections=[(0,11),(11,13),(13,15),(0,12),(12,14),(14,16),(11,12),(11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28)]
                data=[go.Scatter3d(x=xs,y=ys,z=zs,mode='markers',marker=dict(size=4,color='red'))]
                for a,b in connections:
                    if a<len(xs) and b<len(xs):
                        data.append(go.Scatter3d(x=[xs[a],xs[b]],y=[ys[a],ys[b]],z=[zs[a],zs[b]],mode='lines',line=dict(width=4)))
                layout=go.Layout(title="3D Pose Avatar",scene=dict(xaxis=dict(title='X'),yaxis=dict(title='Y'),zaxis=dict(title='Z')))
                fig=go.Figure(data=data,layout=layout)
                st.plotly_chart(fig,use_container_width=True)

elif mode == "Use Webcam":
    st.warning("⚠️ Webcam mode requires local Streamlit run (not supported in Streamlit Cloud).")

# ==============================
# Analysis Section
# ==============================
st.markdown("---")
st.header("📊 Motion Analytics Dashboard")
st.markdown("Upload a previously saved JSON file to view analytics:")
json_file = st.file_uploader("Upload Landmarks JSON", type=["json"], key="analysis")
if json_file:
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(json_file.read())
    df = json.load(open(tmp.name))
    st.success("File loaded successfully!")

    # Compute simple analytics (knee angles & gaze distribution)
    left_hip = [f["pose"].get("23") for f in df if "23" in f["pose"]]
    left_knee = [f["pose"].get("25") for f in df if "25" in f["pose"]]
    left_ankle = [f["pose"].get("27") for f in df if "27" in f["pose"]]
    if left_knee:
        # Compute knee angle if enough frames
        pts = min(len(left_hip), len(left_knee), len(left_ankle))
        angles = []
        for i in range(pts):
            hip = np.array(left_hip[i][:2]); knee = np.array(left_knee[i][:2]); ankle = np.array(left_ankle[i][:2])
            v1 = hip - knee; v2 = ankle - knee
            cos = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-8)
            angles.append(np.degrees(np.arccos(np.clip(cos,-1,1))))
        st.subheader("Knee Angle Distribution")
        st.bar_chart(pd.Series(angles))

    # Gaze histogram
    gaze_dirs = []
    for f in df:
        g = f.get("gaze")
        if g and g.get("left"):
            v = np.array(g["left"][:2]); ang = np.arctan2(v[1], v[0])
            gaze_dirs.append(ang)
    if gaze_dirs:
        st.subheader("Gaze Direction Polar Histogram")
        fig = go.Figure(data=[go.Histogrampolar(theta=gaze_dirs, nbins=36)])
        fig.update_layout(polar=dict(radialaxis=dict(showticklabels=False)), showlegend=False)
        st.plotly_chart(fig)
