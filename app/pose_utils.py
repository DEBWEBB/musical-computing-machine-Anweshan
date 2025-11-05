# pose_utils.py — Utility functions (already provided in previous step)
# pose_utils.py — Utility module for DancePose Live Studio Pro v5.4
import numpy as np

# ---------------------------------------------------------
# 1️⃣ Compute Knee Angles (Left & Right)
# ---------------------------------------------------------
def compute_knee_angles_from_pose(pose_dict):
    """
    Compute knee joint angles from pose landmarks.
    pose_dict: {index: [x, y, z, visibility]}
    returns: (left_knee_angle, right_knee_angle)
    """
    def angle(a, b, c):
        a, b, c = np.array(a[:2]), np.array(b[:2]), np.array(c[:2])
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle_val = np.abs(radians * 180.0 / np.pi)
        if angle_val > 180:
            angle_val = 360 - angle_val
        return angle_val

    try:
        left_angle = angle(pose_dict[23], pose_dict[25], pose_dict[27])  # Left hip, knee, ankle
        right_angle = angle(pose_dict[24], pose_dict[26], pose_dict[28])  # Right hip, knee, ankle
    except Exception:
        left_angle, right_angle = np.nan, np.nan

    return left_angle, right_angle


# ---------------------------------------------------------
# 2️⃣ Compute Hand Distance
# ---------------------------------------------------------
def compute_hand_distance(pose_dict):
    """Compute distance between left and right wrists."""
    try:
        lw, rw = np.array(pose_dict[15][:2]), np.array(pose_dict[16][:2])
        return np.linalg.norm(lw - rw)
    except Exception:
        return np.nan


# ---------------------------------------------------------
# 3️⃣ Compute Head Height
# ---------------------------------------------------------
def compute_head_height(pose_dict):
    """Estimate dancer’s head height relative to shoulders."""
    try:
        head_y = pose_dict[0][1]
        l_sh, r_sh = pose_dict[11][1], pose_dict[12][1]
        return (l_sh + r_sh) / 2 - head_y
    except Exception:
        return np.nan


# ---------------------------------------------------------
# 4️⃣ Compute Motion Energy
# ---------------------------------------------------------
def compute_motion_energy(head_trail):
    """
    Compute motion energy based on head trajectory movement.
    Higher = more dynamic performance.
    """
    if len(head_trail) < 2:
        return 0.0
    diffs = np.diff(head_trail, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    energy = np.sum(distances)
    return energy


# ---------------------------------------------------------
# 5️⃣ Compute Symmetry Score
# ---------------------------------------------------------
def compute_symmetry_score(left_angles, right_angles):
    """
    Compute a normalized symmetry index between 0–1.
    1 = perfectly symmetrical.
    """
    if len(left_angles) == 0 or len(right_angles) == 0:
        return 0.0
    min_len = min(len(left_angles), len(right_angles))
    diff = np.abs(np.array(left_angles[:min_len]) - np.array(right_angles[:min_len]))
    score = 1 - (np.mean(diff) / 180)
    return max(0.0, min(score, 1.0))
