# pose_face_gaze_3d.py
# Main app: MediaPipe Pose + FaceMesh, computes simple gaze vector, saves JSON/CSV, creates interactive 3D avatar (Plotly).
import cv2, os, time, json, argparse
import mediapipe as mp
import numpy as np
import pandas as pd
import plotly.graph_objs as go

parser = argparse.ArgumentParser()
parser.add_argument("--source", default="0", help="0 for webcam or path to video")
parser.add_argument("--out", default="app/outputs", help="output folder inside package")
parser.add_argument("--max_frames", type=int, default=0, help="0 for all frames")
args = parser.parse_args()

SRC = 0 if args.source == "0" else args.source
OUT = args.out
os.makedirs(OUT, exist_ok=True)

mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                    enable_segmentation=False, smooth_landmarks=True,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5)

face = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1,
                        refine_landmarks=True, min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

cap = cv2.VideoCapture(SRC)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open source {SRC}")
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

json_frames = []
frame_idx = 0
print("Press 'q' to quit")

def landmark_to_xy(lm, W, H):
    return (lm.x * W, lm.y * H, lm.z)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    if args.max_frames and frame_idx > args.max_frames:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pr = pose.process(rgb)
    fr = face.process(rgb)

    frame_json = {"frame": frame_idx, "pose": {}, "face": {}, "gaze": {}}

    if pr.pose_landmarks:
        for i, lm in enumerate(pr.pose_landmarks.landmark):
            x, y, z = landmark_to_xy(lm, w, h)
            frame_json["pose"][str(i)] = [float(x), float(y), float(z)]
        mp_drawing.draw_landmarks(frame, pr.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(255,255,0), thickness=2))

    if fr.multi_face_landmarks:
        face_lms = fr.multi_face_landmarks[0].landmark
        # iris indices (if refine_landmarks=True)
        iris_l = [468,469,470,471]
        iris_r = [472,473,474,475]
        def avg_points(indices):
            pts = []
            for idx in indices:
                if idx < len(face_lms):
                    lm = face_lms[idx]
                    pts.append([lm.x * w, lm.y * h, lm.z])
            if not pts:
                return None
            return np.mean(np.array(pts), axis=0)
        left_eye = avg_points(iris_l) or avg_points([33,133])
        right_eye = avg_points(iris_r) or avg_points([362,263])

        nose_pt = None
        if pr.pose_landmarks:
            n = pr.pose_landmarks.landmark[0]
            nose_pt = np.array([n.x * w, n.y * h, n.z])
        else:
            if len(face_lms) > 1:
                n = face_lms[1]
                nose_pt = np.array([n.x * w, n.y * h, n.z])

        if left_eye is not None and nose_pt is not None:
            gv_l = nose_pt - left_eye
            gv_l = gv_l / (np.linalg.norm(gv_l) + 1e-8)
            frame_json["gaze"]["left"] = gv_l.tolist()
            cv2.circle(frame, (int(left_eye[0]), int(left_eye[1])), 3, (0,0,255), -1)
            cv2.arrowedLine(frame, (int(left_eye[0]), int(left_eye[1])),
                            (int(left_eye[0] + gv_l[0]*60), int(left_eye[1] + gv_l[1]*60)),
                            (0,0,255), 2)
        if right_eye is not None and nose_pt is not None:
            gv_r = nose_pt - right_eye
            gv_r = gv_r / (np.linalg.norm(gv_r) + 1e-8)
            frame_json["gaze"]["right"] = gv_r.tolist()
            cv2.circle(frame, (int(right_eye[0]), int(right_eye[1])), 3, (0,0,255), -1)
            cv2.arrowedLine(frame, (int(right_eye[0]), int(right_eye[1])),
                            (int(right_eye[0] + gv_r[0]*60), int(right_eye[1] + gv_r[1]*60)),
                            (0,0,255), 2)

        mp_drawing.draw_landmarks(frame, fr.multi_face_landmarks[0], mp_face.FACEMESH_TESSELATION,
                                  mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(80,256,121), thickness=1))

        for i, lm in enumerate(face_lms):
            frame_json["face"][str(i)] = [float(lm.x * w), float(lm.y * h), float(lm.z)]

    json_frames.append(frame_json)
    cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.imshow("Pose+Face+Gaze", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save outputs
ts = int(time.time())
out_json = os.path.join(OUT, f"landmarks_face_gaze_{ts}.json")
out_csv = os.path.join(OUT, f"landmarks_face_gaze_{ts}.csv")
pd_rows = []
for f in json_frames:
    row = {"frame": f["frame"]}
    for k,v in f.get("pose", {}).items():
        row[f"pose_{k}_x"] = v[0]; row[f"pose_{k}_y"] = v[1]; row[f"pose_{k}_z"] = v[2]
    if f.get("gaze", {}).get("left") is not None:
        row["gaze_left_x"] = f["gaze"]["left"][0]; row["gaze_left_y"] = f["gaze"]["left"][1]; row["gaze_left_z"] = f["gaze"]["left"][2]
    if f.get("gaze", {}).get("right") is not None:
        row["gaze_right_x"] = f["gaze"]["right"][0]; row["gaze_right_y"] = f["gaze"]["right"][1]; row["gaze_right_z"] = f["gaze"]["right"][2]
    pd_rows.append(row)
pd.DataFrame(pd_rows).to_csv(out_csv, index=False)
with open(out_json, "w") as fh:
    json.dump(json_frames, fh)

print("Saved:", out_json, out_csv)
cap.release()
cv2.destroyAllWindows()

# Build 3D avatar (Plotly) using the last available pose
if json_frames:
    last_pose = None
    for f in reversed(json_frames):
        if f.get("pose"):
            last_pose = f["pose"]; break
    if not last_pose:
        print("No pose data for avatar.")
    else:
        xs=[]; ys=[]; zs=[]; labels=[]
        for k in sorted(last_pose.keys(), key=lambda x:int(x)):
            v = last_pose[k]
            xs.append(v[0]); ys.append(-v[1]); zs.append(-v[2]); labels.append(k)
        connections = [
            (0,11),(11,13),(13,15),(0,12),(12,14),(14,16),
            (11,12),(11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28)
        ]
        data = []
        data.append(go.Scatter3d(x=xs,y=ys,z=zs,mode='markers+text',marker=dict(size=4),text=labels,textposition='top center',name='joints'))
        for a,b in connections:
            if a < len(xs) and b < len(xs):
                data.append(go.Scatter3d(x=[xs[a],xs[b]], y=[ys[a],ys[b]], z=[zs[a],zs[b]], mode='lines', line=dict(width=6)))
        layout = go.Layout(title="3D Pose Avatar", scene=dict(xaxis=dict(title='X'),yaxis=dict(title='Y'),zaxis=dict(title='Z')), margin=dict(l=0,r=0,b=0,t=30))
        fig = go.Figure(data=data, layout=layout)
        html_path = os.path.join(OUT, f"avatar_{ts}.html")
        fig.write_html(html_path)
        print("Saved interactive 3D avatar:", html_path)