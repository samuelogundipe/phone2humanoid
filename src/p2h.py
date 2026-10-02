"""Phone video -> 3D pose (MediaPipe) -> joint angles -> reps -> DTW.

The same functions are in notebooks/Phone2Humanoid_Week1.ipynb.
"""
import json
import os

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# MediaPipe pose landmark indices (33-point model)
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_FOOT, R_FOOT = 31, 32

SKELETON = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
            (23, 24), (23, 25), (25, 27), (27, 31), (24, 26), (26, 28), (28, 32)]


def extract_pose(video_path, model_path, max_height=960, overlay_path=None):
    """Run MediaPipe Pose Landmarker (VIDEO mode) on every frame.

    Returns world landmarks (T, 33, 3) in metres, hip-centred, plus fps.
    Frames with no detection are filled with NaN.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    world, image_pts = [], []
    writer = None
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            if h > max_height:
                scale = max_height / h
                frame = cv2.resize(frame, (int(w * scale), max_height))
                h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(image, int(i * 1000 / fps))
            if result.pose_world_landmarks:
                wl = result.pose_world_landmarks[0]
                il = result.pose_landmarks[0]
                world.append([[p.x, p.y, p.z] for p in wl])
                image_pts.append([[p.x * w, p.y * h] for p in il])
            else:
                world.append(np.full((33, 3), np.nan).tolist())
                image_pts.append(np.full((33, 2), np.nan).tolist())
            if overlay_path:
                if writer is None:
                    writer = cv2.VideoWriter(overlay_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                pts = np.array(image_pts[-1])
                for a, b in SKELETON:
                    if not np.isnan(pts[[a, b]]).any():
                        cv2.line(frame, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)), (0, 220, 255), 3)
                for j in (23, 24, 25, 26, 27, 28):
                    if not np.isnan(pts[j]).any():
                        cv2.circle(frame, tuple(pts[j].astype(int)), 6, (255, 80, 0), -1)
                writer.write(frame)
            i += 1
    cap.release()
    if writer is not None:
        writer.release()
    return np.array(world), float(fps)


def gaussian_smooth(x, sigma=2.0):
    """Gaussian smoothing with edge padding (plain NumPy)."""
    radius = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(x, radius, mode="edge"), k, mode="valid")


def find_peaks_simple(x, min_rise, min_gap):
    """Local maxima that rise at least `min_rise` above the lowest point on both
    sides (up to the next higher peak), at least `min_gap` samples apart."""
    cand = [i for i in range(1, len(x) - 1) if x[i] >= x[i - 1] and x[i] > x[i + 1]]
    keep = []
    for i in cand:
        left = i
        while left > 0 and x[left - 1] <= x[i]:
            left -= 1
        right = i
        while right < len(x) - 1 and x[right + 1] <= x[i]:
            right += 1
        rise = x[i] - max(x[left:i + 1].min(), x[i:right + 1].min())
        if rise >= min_rise:
            keep.append(i)
    keep.sort(key=lambda i: -x[i])
    chosen = []
    for i in keep:
        if all(abs(i - j) >= min_gap for j in chosen):
            chosen.append(i)
    return sorted(chosen)


def joint_angle(a, b, c):
    """Angle at b (degrees) between segments b->a and b->c, per frame."""
    v1, v2 = a - b, c - b
    cos = np.sum(v1 * v2, axis=-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def squat_angles(world, sigma=2.0):
    """Knee, hip and ankle flexion (degrees, 0 = straight), averaged left/right,
    gap-filled and Gaussian-smoothed like the final-year thesis."""
    W = world
    knee = 180 - (joint_angle(W[:, L_HIP], W[:, L_KNEE], W[:, L_ANKLE]) +
                  joint_angle(W[:, R_HIP], W[:, R_KNEE], W[:, R_ANKLE])) / 2
    hip = 180 - (joint_angle(W[:, L_SHOULDER], W[:, L_HIP], W[:, L_KNEE]) +
                 joint_angle(W[:, R_SHOULDER], W[:, R_HIP], W[:, R_KNEE])) / 2
    ankle = (joint_angle(W[:, L_KNEE], W[:, L_ANKLE], W[:, L_FOOT]) +
             joint_angle(W[:, R_KNEE], W[:, R_ANKLE], W[:, R_FOOT])) / 2
    out = {}
    for name, series in (("knee", knee), ("hip", hip), ("ankle", ankle)):
        s = series.copy()
        bad = np.isnan(s)
        if bad.any() and (~bad).any():
            idx = np.arange(len(s))
            s[bad] = np.interp(idx[bad], idx[~bad], s[~bad])
        out[name] = gaussian_smooth(s, sigma)
    return out


def find_reps(knee, fps, min_depth=40.0):
    """Each rep = standing -> bottom -> standing. Bottoms are knee-flexion peaks;
    rep edges are the standing minima either side of each peak."""
    peaks = find_peaks_simple(knee, min_rise=min_depth, min_gap=int(0.8 * fps))
    reps = []
    for p in peaks:
        left = p - 1
        while left > 0 and knee[left - 1] <= knee[left]:
            left -= 1
        right = p + 1
        while right < len(knee) - 1 and knee[right + 1] <= knee[right]:
            right += 1
        # drop reps cut off by the start or end of the clip
        if left == 0 or right == len(knee) - 1:
            continue
        reps.append((int(left), int(p), int(right)))
    return reps


def dtw(a, b):
    """Classic DTW on multivariate sequences (T, D). Returns total cost and path."""
    n, m = len(a), len(b)
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            acc[i, j] = cost[i - 1, j - 1] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    i, j, path = n, m, []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = np.argmin([acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1]])
        if step == 0:
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    path.reverse()
    return acc[n, m], path


def mean_gap_degrees(a, b):
    """DTW-aligned mean absolute angle difference in degrees: 'after lining the
    two reps up in time, how many degrees apart are they on average?'"""
    _, path = dtw(a, b)
    ia, ib = zip(*path)
    return float(np.mean(np.abs(a[list(ia)] - b[list(ib)])))


def analyse(video_path, model_path, label, out_dir, overlay=False):
    """overlay=True also writes an UNBLURRED skeleton video; privacy_blur.py makes the shareable one."""
    os.makedirs(out_dir, exist_ok=True)
    world, fps = extract_pose(video_path, model_path,
                              overlay_path=os.path.join(out_dir, f"{label}_overlay.mp4") if overlay else None)
    detected = float(np.mean(~np.isnan(world[:, 0, 0])))
    ang = squat_angles(world)
    reps = find_reps(ang["knee"], fps)
    seqs = [np.stack([ang["knee"][l:r + 1], ang["hip"][l:r + 1]], axis=1) for l, _, r in reps]
    n = len(seqs)
    gaps = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            gaps[i, j] = gaps[j, i] = mean_gap_degrees(seqs[i], seqs[j])
    summary = {
        "label": label,
        "fps": fps,
        "frames": int(len(world)),
        "detected_share": round(detected, 3),
        "reps": [{"start_s": round(l / fps, 2), "bottom_s": round(p / fps, 2), "end_s": round(r / fps, 2),
                  "max_knee_flexion": round(float(ang["knee"][p]), 1),
                  "max_hip_flexion": round(float(ang["hip"][p]), 1),
                  "duration_s": round((r - l) / fps, 2)} for l, p, r in reps],
        "rep_gap_matrix_deg": np.round(gaps, 1).tolist(),
    }
    if n > 1:
        summary["mean_rep_to_rep_gap_deg"] = round(float(gaps[np.triu_indices(n, 1)].mean()), 1)
    np.savez(os.path.join(out_dir, f"{label}_pose.npz"), world=world, fps=fps,
             knee=ang["knee"], hip=ang["hip"], ankle=ang["ankle"])
    with open(os.path.join(out_dir, f"{label}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return world, fps, ang, reps, summary
