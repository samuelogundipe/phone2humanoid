"""Keep the athlete sharp and blur everyone and everything else.

The pose landmarker returns a segmentation mask for the tracked person. With
num_poses=1 that is the athlete only, so bystanders end up in the blurred
background. Also saves 2D image landmarks for drawing skeleton overlays.
"""
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from paths import CLIPS, MODEL, OUT

SKELETON = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
            (23, 24), (23, 25), (25, 27), (27, 31), (24, 26), (26, 28), (28, 32)]


def privacy_video(label, height=960, blur=61):
    """Writes outputs/<label>_privacy.mp4 and outputs/<label>_image_lm.npy."""
    cap = cv2.VideoCapture(str(CLIPS[label]))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.VIDEO, num_poses=1, output_segmentation_masks=True)
    writer, image_lm, i = None, [], 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    with vision.PoseLandmarker.create_from_options(opts) as lm:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (int(frame.shape[1] * height / frame.shape[0]), height))
            h, w = frame.shape[:2]
            res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                               data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), int(i * 1000 / fps))
            blurred = cv2.GaussianBlur(frame, (blur, blur), 0)
            if res.segmentation_masks:
                mask = res.segmentation_masks[0].numpy_view().squeeze()
                mask = cv2.dilate((cv2.resize(mask, (w, h)) > 0.5).astype(np.uint8), kernel)
                alpha = cv2.GaussianBlur(mask.astype(np.float32), (31, 31), 0)[..., None]
                out = (frame * alpha + blurred * (1 - alpha)).astype(np.uint8)
                image_lm.append([[p.x * w, p.y * h] for p in res.pose_landmarks[0]])
            else:
                out = blurred  # nobody detected: blur the whole frame
                image_lm.append(np.full((33, 2), np.nan).tolist())
            if writer is None:
                writer = cv2.VideoWriter(str(OUT / f"{label}_privacy.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(out)
            i += 1
    writer.release()
    np.save(OUT / f"{label}_image_lm.npy", np.array(image_lm))
    return OUT / f"{label}_privacy.mp4"


def skeleton_overlay(label, fps=30.0):
    """Draws the detected skeleton on the blurred video: outputs/<label>_skeleton.mp4."""
    pts_all = np.load(OUT / f"{label}_image_lm.npy")
    cap = cv2.VideoCapture(str(OUT / f"{label}_privacy.mp4"))
    writer, i = None, 0
    while True:
        ok, frame = cap.read()
        if not ok or i >= len(pts_all):
            break
        P = pts_all[i]
        for a, b in SKELETON:
            if not np.isnan(P[[a, b]]).any():
                cv2.line(frame, tuple(P[a].astype(int)), tuple(P[b].astype(int)), (0, 220, 255), 3)
        for j in (23, 24, 25, 26, 27, 28):
            if not np.isnan(P[j]).any():
                cv2.circle(frame, tuple(P[j].astype(int)), 6, (255, 80, 0), -1)
        if writer is None:
            writer = cv2.VideoWriter(str(OUT / f"{label}_skeleton.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                     (frame.shape[1], frame.shape[0]))
        writer.write(frame)
        i += 1
    writer.release()
    return OUT / f"{label}_skeleton.mp4"
