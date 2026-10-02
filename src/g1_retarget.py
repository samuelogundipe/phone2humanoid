"""Retarget a filmed movement onto the Unitree G1 (kinematic replay, no physics).

For every video frame:
  1. human angles from MediaPipe world landmarks, zeroed at standing
  2. G1 hip pitch / knee / elbow / shoulder pitch from those angles
     (G1 conventions measured with g1_probe.py: hip flexion = negative pitch,
      knee flexion = positive, toes-up = negative ankle pitch, elbow 0 = 90 deg bent,
      shoulder forward = negative pitch, base lean forward = +rotation about +y)
  3. base tilted by the trunk lean; ankle angle solved so the feet stay flat,
     then clipped to the G1's real ankle range
  4. base shifted so both feet stay planted where they stood
Records what the robot could not copy, and renders the G1 next to the video.
"""
import json

import cv2
import mujoco
import numpy as np

from p2h import joint_angle, gaussian_smooth, find_reps
from paths import AZIMUTH, G1_XML, OUT

L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 11, 12, 13, 14, 15, 16
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 23, 24, 25, 26, 27, 28


def human_angles(world, fps, sigma=2.0):
    """Knee, hip, trunk lean, elbow and shoulder flexion (deg), smoothed and
    zeroed at the athlete's standing posture."""
    W = world
    up = np.array([0.0, -1.0, 0.0])  # MediaPipe world: +y points down

    def smooth(x):
        x = x.copy()
        bad = np.isnan(x)
        if bad.any() and (~bad).any():
            i = np.arange(len(x))
            x[bad] = np.interp(i[bad], i[~bad], x[~bad])
        return gaussian_smooth(x, sigma)

    knee = smooth(180 - (joint_angle(W[:, L_HIP], W[:, L_KNEE], W[:, L_ANK]) +
                         joint_angle(W[:, R_HIP], W[:, R_KNEE], W[:, R_ANK])) / 2)
    hip = smooth(180 - (joint_angle(W[:, L_SH], W[:, L_HIP], W[:, L_KNEE]) +
                        joint_angle(W[:, R_SH], W[:, R_HIP], W[:, R_KNEE])) / 2)
    trunk = (W[:, L_SH] + W[:, R_SH]) / 2 - (W[:, L_HIP] + W[:, R_HIP]) / 2
    lean = smooth(np.degrees(np.arccos(np.clip(trunk @ up / np.linalg.norm(trunk, axis=1), -1, 1))))
    # Trunk frame: up (hips -> shoulders), left (right shoulder -> left shoulder),
    # forward (ears -> nose), each made orthogonal to the others.
    NOSE, L_EAR, R_EAR = 0, 7, 8
    unit = lambda v: v / np.linalg.norm(v, axis=1, keepdims=True)
    t_up = unit(trunk)
    t_left = W[:, L_SH] - W[:, R_SH]
    t_left = unit(t_left - (t_left * t_up).sum(1, keepdims=True) * t_up)
    t_fwd = W[:, NOSE] - (W[:, L_EAR] + W[:, R_EAR]) / 2
    t_fwd = t_fwd - (t_fwd * t_up).sum(1, keepdims=True) * t_up
    t_fwd = unit(t_fwd - (t_fwd * t_left).sum(1, keepdims=True) * t_left)

    arms = {}
    for side, sh, el, wr, out_sign in (("l", L_SH, L_EL, L_WR, 1), ("r", R_SH, R_EL, R_WR, -1)):
        ua = unit(W[:, el] - W[:, sh])
        down = -(ua * t_up).sum(1)
        arms[f"sh_pitch_{side}"] = smooth(np.degrees(np.arctan2((ua * t_fwd).sum(1), down)))      # forward raise
        arms[f"sh_roll_{side}"] = smooth(np.degrees(np.arctan2(out_sign * (ua * t_left).sum(1), down)))  # outward raise
        arms[f"elbow_{side}"] = smooth(180 - joint_angle(W[:, sh], W[:, el], W[:, wr]))

    standing = knee <= np.percentile(knee, 20)
    raw = {"knee": knee, "hip": hip, "lean": lean}
    zeroed = {k: np.clip(v - np.median(v[standing]), 0, None) for k, v in raw.items()}
    zeroed.update(arms)
    baseline = {k: float(np.median(v[standing])) for k, v in raw.items()}
    return zeroed, baseline


class G1:
    def __init__(self, xml):
        self.m = mujoco.MjModel.from_xml_path(str(xml))
        self.m.vis.global_.offwidth = max(self.m.vis.global_.offwidth, 1280)
        self.m.vis.global_.offheight = max(self.m.vis.global_.offheight, 1280)
        self.d = mujoco.MjData(self.m)
        self.key = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_KEY, "stand")
        mujoco.mj_resetDataKeyframe(self.m, self.d, self.key)
        mujoco.mj_kinematics(self.m, self.d)
        self.stand_q = self.d.qpos.copy()
        self.feet = [self.body("left_ankle_roll_link"), self.body("right_ankle_roll_link")]
        self.feet0 = np.mean([self.d.xpos[f] for f in self.feet], axis=0).copy()
        foot_geoms = [g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] == self.feet[0]]
        rel_x = [self.d.geom_xpos[g][0] - self.d.xpos[self.feet[0]][0] for g in foot_geoms]
        self.heel, self.toe = min(rel_x), max(rel_x)

    def body(self, name):
        return mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, name)

    def qadr(self, joint):
        return self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, joint)]

    def limits(self, joint):
        return self.m.jnt_range[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, joint)]

    def pose(self, knee, hip, lean, arms):
        """Set the G1 from human angles (deg). `arms` holds per-side shoulder
        pitch/roll and elbow flexion. Returns what had to be clipped."""
        m, d = self.m, self.d
        d.qpos[:] = self.stand_q
        report = {}

        def put(joint, value_rad, tag):
            lo, hi = self.limits(joint)
            clipped = float(np.clip(value_rad, lo, hi))
            report[tag] = report.get(tag, False) or bool(abs(clipped - value_rad) > 1e-6)
            d.qpos[self.qadr(joint)] = clipped
            return clipped

        need_ankle = knee - hip + lean  # dorsiflexion (deg) that keeps the foot flat
        report["ankle_needed_deg"] = float(need_ankle)
        for side, s, roll_sign in (("left", "l", 1), ("right", "r", -1)):
            put(f"{side}_hip_pitch_joint", -np.radians(hip), "hip_limited")
            put(f"{side}_knee_joint", np.radians(knee), "knee_limited")
            put(f"{side}_ankle_pitch_joint", -np.radians(need_ankle), "ankle_limited")
            put(f"{side}_shoulder_pitch_joint", -np.radians(arms[f"sh_pitch_{s}"]), "shoulder_limited")
            put(f"{side}_shoulder_roll_joint", roll_sign * np.radians(arms[f"sh_roll_{s}"]), "shoulder_limited")
            put(f"{side}_elbow_joint", np.radians(90 - arms[f"elbow_{s}"]), "elbow_limited")
        q = np.zeros(4)
        mujoco.mju_axisAngle2Quat(q, np.array([0.0, 1.0, 0.0]), np.radians(lean))
        d.qpos[3:7] = q
        mujoco.mj_kinematics(m, d)
        feet_now = np.mean([d.xpos[f] for f in self.feet], axis=0)
        d.qpos[0] += self.feet0[0] - feet_now[0]
        d.qpos[2] += self.feet0[2] - feet_now[2]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        # static balance: whole-robot centre of mass over the feet (x only)
        com_x = d.subtree_com[self.body("pelvis")][0]
        foot_x = self.feet0[0]
        report["balance_margin_m"] = float(min(com_x - (foot_x + self.heel), (foot_x + self.toe) - com_x))
        R = d.xmat[self.feet[0]].reshape(3, 3)
        report["foot_tilt_deg"] = float(np.degrees(np.arcsin(np.clip(R[2, 0], -1, 1))))
        return report


def replay(npz_path, video_path, xml, out_mp4, out_json, azimuth=225, label="oblique45"):
    data = np.load(npz_path)
    world, fps = data["world"], float(data["fps"])
    ang, base = human_angles(world, fps)
    robot = G1(xml)
    reports = []
    cap = cv2.VideoCapture(str(video_path))
    writer = None
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 2.3, azimuth, -8
    with mujoco.Renderer(robot.m, 960, 540) as r:
        for i in range(len(world)):
            ok, frame = cap.read()
            if not ok:
                break
            arms = {k: ang[k][i] for k in ang if k.startswith(("sh_", "elbow_"))}
            rep = robot.pose(ang["knee"][i], ang["hip"][i], ang["lean"][i], arms)
            reports.append(rep)
            cam.lookat[:] = [robot.feet0[0], 0.0, 0.55]
            r.update_scene(robot.d, camera=cam)
            img = cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR)
            h = 960
            frame = cv2.resize(frame, (int(frame.shape[1] * h / frame.shape[0]), h))
            panel = np.hstack([frame, img])
            txt = (f"knee {ang['knee'][i]:5.0f} deg   hip {ang['hip'][i]:5.0f} deg   "
                   f"ankle needed {rep['ankle_needed_deg']:4.0f} deg"
                   + ("  (OVER G1 LIMIT)" if rep["ankle_limited"] else ""))
            cv2.rectangle(panel, (0, h - 44), (panel.shape[1], h), (20, 20, 20), -1)
            cv2.putText(panel, txt, (14, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (80, 80, 255) if rep["ankle_limited"] else (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(panel, "you (phone video)", (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(panel, "Unitree G1 (simulation)", (frame.shape[1] + 14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (255, 255, 255), 2, cv2.LINE_AA)
            if writer is None:
                writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (panel.shape[1], panel.shape[0]))
            writer.write(panel)
    cap.release()
    writer.release()

    n = len(reports)
    reps = find_reps(ang["knee"], fps, min_depth=40)
    summary = {
        "label": label,
        "frames": n,
        "standing_baseline_deg": {k: round(v, 1) for k, v in base.items()},
        "deepest_knee_deg": round(float(ang["knee"].max()), 1),
        "deepest_hip_deg": round(float(ang["hip"].max()), 1),
        "max_trunk_lean_deg": round(float(ang["lean"].max()), 1),
        "max_ankle_needed_deg": round(max(r["ankle_needed_deg"] for r in reports), 1),
        "g1_ankle_toes_up_limit_deg": round(float(-np.degrees(robot.limits("left_ankle_pitch_joint")[0])), 1),
        "share_frames_limited": {k: round(float(np.mean([r[k] for r in reports])), 3)
                                 for k in ("hip_limited", "knee_limited", "ankle_limited", "shoulder_limited", "elbow_limited")},
        "min_balance_margin_m": round(min(r["balance_margin_m"] for r in reports), 3),
        "share_frames_com_outside_feet": round(float(np.mean([r["balance_margin_m"] < 0 for r in reports])), 3),
        "max_foot_tilt_deg": round(max(abs(r["foot_tilt_deg"]) for r in reports), 1),
        "reps": len(reps),
    }
    json.dump({"summary": summary,
               "per_frame": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in reports]},
              open(out_json, "w"), indent=1)
    return summary


def run(label):
    """Kinematic replay for one clip, using the blurred video as the left panel."""
    return replay(OUT / f"{label}_pose.npz", OUT / f"{label}_privacy.mp4", G1_XML,
                  OUT / f"{label}_g1_pose_replay.mp4", OUT / f"{label}_g1_pose_replay.json",
                  azimuth=AZIMUTH[label], label=label)


if __name__ == "__main__":
    import sys
    print(json.dumps(run(sys.argv[1] if len(sys.argv) > 1 else "oblique45"), indent=1))
