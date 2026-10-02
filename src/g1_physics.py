"""Physics replay on the Unitree G1: does the robot stay upright?

The G1's 29 position actuators track the joint targets from the kinematic
retarget while MuJoCo simulates gravity and foot contact. Conditions:
  A. athlete's arms, joint tracking only
  B. robot's default arms, joint tracking only
  C. athlete's arms + ankle balance feedback: both ankle-pitch targets are
     nudged in proportion to how far the centre of mass has drifted from
     mid-foot (a PD "ankle strategy")
  D. balance-aware targets: each pose tilted about the ankles so the centre of
     mass sits over mid-foot (hip and knee angles unchanged), with and without C
Every condition reports when the robot fell and how far its feet slid.
"""
import json

import cv2
import mujoco
import numpy as np

from g1_retarget import G1, human_angles
from paths import AZIMUTH, G1_XML, OUT

STAND_ARMS = {"sh_pitch_l": -11.46, "sh_pitch_r": -11.46, "sh_roll_l": 11.46, "sh_roll_r": 11.46,
              "elbow_l": 16.66, "elbow_r": 16.66}  # the G1 'stand' keyframe arms, as human-style angles
KP, KD, MAX_CORR = 4.0, 0.4, 0.3  # ankle feedback: rad per metre of CoM error, damping, clip (rad)


def _com_error(kin, knee, hip, lean, arms):
    kin.pose(knee, hip, lean, arms)
    mujoco.mj_comPos(kin.m, kin.d)
    return kin.d.subtree_com[kin.body("pelvis")][0] - (kin.feet0[0] + (kin.heel + kin.toe) / 2)


def _balanced_lean(kin, knee, hip, lean, arms, tol=1e-4):
    """lean' = lean + delta with the CoM over mid-foot (secant search on delta)."""
    a, b = lean, lean - 5.0
    fa, fb = _com_error(kin, knee, hip, a, arms), _com_error(kin, knee, hip, b, arms)
    for _ in range(30):
        if abs(fb) < tol or fb == fa:
            break
        a, b, fa = b, b - fb * (b - a) / (fb - fa), fb
        fb = _com_error(kin, knee, hip, b, arms)
    return b


def build_targets(label, default_arms=False, balance_aware=False):
    d0 = np.load(OUT / f"{label}_pose.npz")
    ang, _ = human_angles(d0["world"], float(d0["fps"]))
    kin = G1(G1_XML)
    T, deltas = [], []
    for i in range(len(ang["knee"])):
        arms = STAND_ARMS if default_arms else {k: ang[k][i] for k in ang if k.startswith(("sh_", "elbow_"))}
        lean = ang["lean"][i]
        if balance_aware:
            lean = _balanced_lean(kin, ang["knee"][i], ang["hip"][i], lean, arms)
        kin.pose(ang["knee"][i], ang["hip"][i], lean, arms)
        deltas.append(lean - ang["lean"][i])
        T.append(kin.d.qpos.copy())
    return kin, np.array(T), float(d0["fps"]), np.array(deltas)


class Sim:
    """One physics run that can be stepped frame by frame."""

    def __init__(self, kin, T, fps, feedback):
        self.kin, self.T, self.fps, self.feedback = kin, T, fps, feedback
        m = kin.m
        self.m = m
        self.act = np.array([m.jnt_qposadr[m.actuator_trnid[a, 0]] for a in range(m.nu)])
        self.ankle = [a for a in range(m.nu) if "ankle_pitch" in m.actuator(a).name]
        self.pel, self.feet = kin.body("pelvis"), kin.feet
        self.mid = (kin.heel + kin.toe) / 2
        self.d = mujoco.MjData(m)
        self.d.qpos[:] = T[0]
        self.d.qpos[2] += 0.002
        mujoco.mj_forward(m, self.d)
        self.f0 = np.array([self.d.xpos[f].copy() for f in self.feet])
        self.steps = int(round((1 / fps) / m.opt.timestep))
        self.prev, self.max_corr, self.fell_at, self.max_tilt = 0.0, 0.0, None, 0.0

    def step_frame(self, i):
        m, d = self.m, self.d
        for _ in range(self.steps):
            c = self.T[i][self.act].copy()
            if self.feedback:
                mujoco.mj_comPos(m, d)
                e = d.subtree_com[self.pel][0] - (np.mean([d.xpos[f][0] for f in self.feet]) + self.mid)
                corr = float(np.clip(KP * e + KD * (e - self.prev) / m.opt.timestep, -MAX_CORR, MAX_CORR))
                self.prev = e
                self.max_corr = max(self.max_corr, abs(corr))
                for a in self.ankle:
                    c[a] = np.clip(c[a] + corr, *m.actuator_ctrlrange[a])
            d.ctrl[:] = c
            mujoco.mj_step(m, d)
        R = d.xmat[self.pel].reshape(3, 3)
        tilt = float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))
        self.max_tilt = max(self.max_tilt, tilt) if self.fell_at is None else self.max_tilt
        if self.fell_at is None and (tilt > 60 or d.xpos[self.pel][2] < 0.35):
            self.fell_at = i / self.fps
        return tilt

    def result(self):
        slide = np.linalg.norm((np.array([self.d.xpos[f] for f in self.feet]) - self.f0)[:, :2], axis=1).max()
        out = {"fell_at_s": None if self.fell_at is None else round(self.fell_at, 2),
               "max_pelvis_tilt_before_fall_deg": round(self.max_tilt, 1),
               "max_foot_slide_cm": round(float(slide * 100), 1)}
        if self.feedback:
            out["max_ankle_correction_deg"] = round(float(np.degrees(self.max_corr)), 1)
        return out


def simulate(kin, T, fps, feedback, stop_on_fall=True):
    sim = Sim(kin, T, fps, feedback)
    for i in range(len(T)):
        sim.step_frame(i)
        if stop_on_fall and sim.fell_at is not None:
            break
    return sim.result()


def experiments(label):
    kin, T, fps, _ = build_targets(label)
    kinB, TB, _, _ = build_targets(label, default_arms=True)
    kinD, TD, _, deltas = build_targets(label, balance_aware=True)
    res = {
        "A_your_arms_tracking_only": simulate(kin, T, fps, feedback=False),
        "B_default_arms_tracking_only": simulate(kinB, TB, fps, feedback=False),
        "C_your_arms_plus_ankle_balance": simulate(kin, T, fps, feedback=True),
        "D_balance_aware_targets_tracking_only": simulate(kinD, TD, fps, feedback=False),
        "D_balance_aware_targets_plus_ankle_balance": simulate(kinD, TD, fps, feedback=True),
        "D_extra_trunk_lean_deg": {"mean": round(float(deltas.mean()), 1), "min": round(float(deltas.min()), 1),
                                   "max": round(float(deltas.max()), 1)},
        "clip_duration_s": round(len(T) / fps, 2),
        "controller": {"kp_rad_per_m": KP, "kd": KD, "max_correction_rad": MAX_CORR},
    }
    json.dump(res, open(OUT / f"{label}_g1_physics.json", "w"), indent=1)
    return res


def compare_video(label):
    """Three panels: athlete (blurred video) | G1 tracking only | G1 + ankle balance."""
    kin, T, fps, _ = build_targets(label)
    sims = {"plain": Sim(kin, T, fps, feedback=False), "balance": Sim(kin, T, fps, feedback=True)}
    cap = cv2.VideoCapture(str(OUT / f"{label}_privacy.mp4"))
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 2.4, AZIMUTH[label], -8
    H, W = 960, 540
    font = cv2.FONT_HERSHEY_SIMPLEX
    writer = None
    with mujoco.Renderer(kin.m, H, W) as r:
        for i in range(len(T)):
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (int(frame.shape[1] * H / frame.shape[0]), H))
            panels = [frame]
            for sim in sims.values():
                sim.step_frame(i)
                cam.lookat[:] = [kin.feet0[0], 0.0, 0.5]
                r.update_scene(sim.d, camera=cam)
                img = cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR)
                fell = sim.fell_at is not None
                cv2.putText(img, "FELL" if fell else "upright", (14, H - 20), font, 0.9,
                            (80, 80, 255) if fell else (140, 230, 140), 2, cv2.LINE_AA)
                panels.append(img)
            panel = np.hstack(panels)
            x = 0
            for t, p in zip(["you (phone video)", "G1: copy angles only", "G1: + ankle balance"], panels):
                cv2.rectangle(panel, (x, 0), (x + p.shape[1], 46), (20, 20, 20), -1)
                cv2.putText(panel, t, (x + 12, 32), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                x += p.shape[1]
            cv2.putText(panel, f"t = {i / fps:4.1f} s", (14, H - 20), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            if writer is None:
                writer = cv2.VideoWriter(str(OUT / f"{label}_g1_physics_compare.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps, (panel.shape[1], panel.shape[0]))
            writer.write(panel)
    writer.release()
    cap.release()
    return {k: s.result() for k, s in sims.items()}


if __name__ == "__main__":
    import sys
    print(json.dumps(experiments(sys.argv[1] if len(sys.argv) > 1 else "oblique45"), indent=1))
