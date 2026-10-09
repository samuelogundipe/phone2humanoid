"""Still frames for the write-up, taken from the public (face-blurred)
three-panel videos (athlete | copy only | + ankle balance): the deepest or
highest point of the first rep, and the moment the copy-only robot falls.

    python src/drills_figures.py
"""
import json

import cv2

from drills import CLIPS, DRILL_OUT
from paths import ROOT

FIG = ROOT / "results" / "drills" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def grab(video, t, fps, s0):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(int(round(t * fps)) - s0, 0))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def main():
    for lab in [c for c in CLIPS if c != "squat_S"]:
        rp, vid = DRILL_OUT / f"{lab}_report.json", DRILL_OUT / f"{lab}_G1_compare_public.mp4"
        if not (rp.exists() and vid.exists()):
            continue
        r = json.load(open(rp))
        fps, s0 = r["fps"], int(round(r["robot_replay_starts_s"] * r["fps"]))
        reps = r["human"]["reps"]
        key = {"squat": "bottom_s", "lunge": "bottom_s", "arm_raise": "peak_s", "kick": "peak_s"}[r["drill"]]
        moments = {}
        if reps:
            moments["rep1"] = reps[0][key]
        fell = r["physics"]["A_copy_only"]["fell_at_s"]
        if fell is not None:
            moments["fall"] = fell + 0.3
        for name, t in moments.items():
            f = grab(vid, t, fps, s0)
            if f is not None:
                f = cv2.resize(f, (f.shape[1] * 2 // 3, f.shape[0] * 2 // 3))
                cv2.imwrite(str(FIG / f"{lab}_{name}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(lab, moments)


if __name__ == "__main__":
    main()
