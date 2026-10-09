"""Shareable copies of the drill videos with the athlete's face blurred too.

Bystanders are blurred in every video already. This adds the same head blur
as the squat videos (privacy_blur.blur_head, from the face landmarks) to the
athlete panel of each three-panel drill video, then re-encodes to H.264.

    python src/face_blur.py              # every outputs/drills/*_g1_compare.mp4
"""
import json
import subprocess
import sys

import cv2
import numpy as np

from drills import CLIPS, DRILL_OUT
from privacy_blur import FACE, blur_head


def public_video(label):
    z = np.load(DRILL_OUT / f"{label}_pose.npz")
    img, panel_w = z["image"], int(z["size"][0])
    rep = json.load(open(DRILL_OUT / f"{label}_report.json"))
    s0 = int(round(rep["robot_replay_starts_s"] * rep["fps"]))
    cap = cv2.VideoCapture(str(DRILL_OUT / f"{label}_g1_compare.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tmp = DRILL_OUT / f"{label}_public_tmp.mp4"
    writer, i = None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        pts = img[min(i + s0, len(img) - 1), FACE]
        if not np.isnan(pts).any():
            frame[:, :panel_w] = blur_head(frame[:, :panel_w].copy(), pts)
        if writer is None:
            writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame.shape[1], frame.shape[0]))
        writer.write(frame)
        i += 1
    cap.release()
    writer.release()
    out = DRILL_OUT / f"{label}_G1_compare_public.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(tmp), "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "26", "-movflags", "+faststart", str(out)], check=True)
    tmp.unlink()
    return out


if __name__ == "__main__":
    labels = sys.argv[1:] or [lab for lab in CLIPS if lab != "squat_S"]
    for lab in labels:
        if (DRILL_OUT / f"{lab}_g1_compare.mp4").exists():
            print(lab, "->", public_video(lab).name)
