# analyze_and_plot.py
# Analysis & plotting utilities for outputs (knee angle hist, foot heatmap, step detection, gaze histogram).
import json, os, argparse
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

parser = argparse.ArgumentParser()
parser.add_argument("--json", required=True, help="path to landmarks_face_gaze_<ts>.json")
parser.add_argument("--out", default="app/analysis", help="output folder")
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)

with open(args.json, "r") as fh:
    frames = json.load(fh)

def extract_landmark_series(idx):
    pts = []
    for f in frames:
        p = f.get("pose", {}).get(str(idx))
        if p:
            pts.append(p[:2])
        else:
            pts.append([np.nan, np.nan])
    return np.array(pts)

left_hip = extract_landmark_series(23)
left_knee = extract_landmark_series(25)
left_ankle = extract_landmark_series(27)

def angle_between(a,b):
    dot = (a*b).sum(axis=1)
    na = np.linalg.norm(a,axis=1); nb = np.linalg.norm(b,axis=1)
    cos = dot/(na*nb + 1e-8)
    cos = np.clip(cos, -1, 1)
    return np.degrees(np.arccos(cos))

v1 = left_hip - left_knee
v2 = left_ankle - left_knee
valid = ~np.isnan(v1[:,0]) & ~np.isnan(v2[:,0])
knee_angles = np.full(len(v1), np.nan)
knee_angles[valid] = angle_between(v1[valid], v2[valid])

plt.figure(figsize=(8,4))
plt.hist(knee_angles[~np.isnan(knee_angles)], bins=30)
plt.title("Left knee angle distribution")
plt.xlabel("Angle (degrees)")
plt.tight_layout()
plt.savefig(os.path.join(args.out, "knee_angle_hist.png"))
print("Saved knee angle histogram")

left_ankle_xy = left_ankle[~np.isnan(left_ankle).any(axis=1)]
if len(left_ankle_xy) > 10:
    x = left_ankle_xy[:,0]; y = left_ankle_xy[:,1]
    xy = np.vstack([x,y])
    kde = gaussian_kde(xy)
    xi, yi = np.mgrid[x.min():x.max():200j, y.min():y.max():200j]
    zi = kde(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
    plt.figure(figsize=(6,6))
    plt.imshow(np.rot90(zi), extent=[x.min(), x.max(), y.min(), y.max()])
    plt.scatter(x,y,s=3)
    plt.title("Left ankle trajectory density")
    plt.savefig(os.path.join(args.out, "left_ankle_heatmap.png"))
    print("Saved left ankle heatmap")

ys = left_ankle[:,1]
valid_idx = ~np.isnan(ys)
if valid_idx.sum() > 5:
    ysv = np.gradient(ys[valid_idx])
    peaks, _ = find_peaks(-ysv, distance=10, prominence=0.5)
    plt.figure(figsize=(8,3))
    plt.plot(np.where(valid_idx)[0], ys[valid_idx], label="left ankle y")
    plt.plot(np.where(valid_idx)[0][peaks], ys[valid_idx][peaks], "rx", label="steps")
    plt.legend(); plt.title("Simple step detection (left ankle)")
    plt.savefig(os.path.join(args.out, "step_detection_left.png"))
    print("Saved simple step detection plot")

gaze_dirs = []
for f in frames:
    g = f.get("gaze")
    if g and g.get("left"):
        v = np.array(g["left"][:2]); ang = np.arctan2(v[1], v[0])
        gaze_dirs.append(ang)
if gaze_dirs:
    plt.figure(figsize=(6,6))
    ax = plt.subplot(111, projection='polar')
    ax.hist(gaze_dirs, bins=36)
    ax.set_title("Gaze directions (left eye) polar histogram")
    plt.savefig(os.path.join(args.out, "gaze_polar_hist.png"))
    print("Saved gaze polar histogram")

print("Analysis outputs in", args.out)