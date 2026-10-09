"""One pass per drill clip: MediaPipe pose (world + image landmarks + visibility)
and the privacy video (athlete's body sharp; bystanders and the athlete's face
blurred, as in privacy_blur.py).

    python src/drill_pose.py              # all clips in data/drills/
    python src/drill_pose.py lunge_A      # one clip
"""
import sys

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from paths import DATA, MODEL, OUT
from privacy_blur import FACE, blur_head

DRILL_DATA = DATA / "drills"
DRILL_OUT = OUT / "drills"
DRILL_OUT.mkdir(parents=True, exist_ok=True)


def process(label, height=960, blur=61):
    cap = cv2.VideoCapture(str(DRILL_DATA / f"{label}.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.VIDEO, num_poses=1, output_segmentation_masks=True,
        min_pose_detection_confidence=0.5, min_tracking_confidence=0.5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    world, image, vis, writer, i = [], [], [], None, 0
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
            if res.pose_world_landmarks:
                world.append([[p.x, p.y, p.z] for p in res.pose_world_landmarks[0]])
                image.append([[p.x * w, p.y * h] for p in res.pose_landmarks[0]])
                vis.append([p.visibility for p in res.pose_landmarks[0]])
                mask = res.segmentation_masks[0].numpy_view().squeeze()
                mask = cv2.dilate((cv2.resize(mask, (w, h)) > 0.5).astype(np.uint8), kernel)
                alpha = cv2.GaussianBlur(mask.astype(np.float32), (31, 31), 0)[..., None]
                out = (frame * alpha + blurred * (1 - alpha)).astype(np.uint8)
                out = blur_head(out, np.array(image[-1])[FACE])
            else:
                world.append(np.full((33, 3), np.nan).tolist())
                image.append(np.full((33, 2), np.nan).tolist())
                vis.append(np.zeros(33).tolist())
                out = blurred
            if writer is None:
                writer = cv2.VideoWriter(str(DRILL_OUT / f"{label}_privacy.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(out)
            i += 1
    cap.release()
    writer.release()
    np.savez(DRILL_OUT / f"{label}_pose.npz", world=np.array(world), image=np.array(image),
             visibility=np.array(vis), fps=fps, size=np.array([w, h]))
    det = float(np.mean(~np.isnan(np.array(world)[:, 0, 0])))
    print(f"{label}: {i} frames at {fps:.1f} fps, person found in {det:.1%}", flush=True)


if __name__ == "__main__":
    labels = sys.argv[1:] or sorted(p.stem for p in DRILL_DATA.glob("*.mp4"))
    for lab in labels:
        lock = DRILL_OUT / f"{lab}.lock"
        if (DRILL_OUT / f"{lab}_pose.npz").exists():
            continue
        try:  # several workers can run at once; each clip is claimed by one
            lock.open("x").close()
        except FileExistsError:
            continue
        process(lab)
        lock.unlink()
