"""Run the whole pipeline for one clip.

    python run_all.py oblique45

Steps: pose + angles + reps + DTW  ->  privacy blur + skeleton overlay  ->
G1 kinematic replay  ->  G1 physics experiments  ->  three-panel physics video.
Everything lands in outputs/.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from paths import CLIPS, MODEL, OUT  # noqa: E402
import p2h  # noqa: E402
import privacy_blur  # noqa: E402
import g1_retarget  # noqa: E402
import g1_physics  # noqa: E402


def h264(path):
    """Re-encode to H.264 so the video plays in browsers and on phones (needs ffmpeg)."""
    dst = path.with_name(path.stem + "_h264.mp4")
    try:
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(path), "-vcodec", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "24", "-movflags", "+faststart", str(dst)], check=True)
        return dst
    except (FileNotFoundError, subprocess.CalledProcessError):
        return path


def main(label):
    t0 = time.time()
    print(f"[1/5] pose, angles, reps, DTW for {CLIPS[label].name}")
    *_, summary = p2h.analyse(str(CLIPS[label]), str(MODEL), label, str(OUT))
    print(f"      {len(summary['reps'])} full reps, rep-to-rep gap {summary.get('mean_rep_to_rep_gap_deg')} deg")
    print("[2/5] privacy blur + skeleton overlay")
    privacy_blur.privacy_video(label)
    skel = h264(privacy_blur.skeleton_overlay(label))
    print("[3/5] G1 kinematic replay")
    kin = g1_retarget.run(label)
    print(f"      ankle needed up to {kin['max_ankle_needed_deg']} deg; frames hitting a limit: {kin['share_frames_limited']}")
    print("[4/5] G1 physics experiments")
    phys = g1_physics.experiments(label)
    for k, v in phys.items():
        print(f"      {k}: {v}")
    print("[5/5] three-panel physics video")
    g1_physics.compare_video(label)
    videos = [skel, h264(OUT / f"{label}_g1_pose_replay.mp4"), h264(OUT / f"{label}_g1_physics_compare.mp4")]
    report = {"label": label, "week1": summary, "kinematic": kin, "physics": phys,
              "videos": [v.name for v in videos], "runtime_s": round(time.time() - t0)}
    json.dump(report, open(OUT / f"{label}_report.json", "w"), indent=1)
    print(f"done in {report['runtime_s']} s -> {OUT}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "oblique45")
