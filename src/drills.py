"""Arm raise, forward lunge and kick: from phone video to the Unitree G1.

Builds on the squat pipeline but drops two squat shortcuts:
  * limbs are matched as 3D directions, not single angles, so arms can go
    overhead and the two legs can do different things (lunge, kick)
  * each foot is on or off the ground according to the video, and only the
    feet on the ground are kept flat and used for balance

Per clip it reports
  human:      reps, depth / height of each rep, rep-to-rep DTW gap
  kinematic:  how closely the G1 can copy each limb inside its joint limits,
              whether both planted feet can touch the floor at once, and where
              the centre of mass sits over the feet
  physics:    A  copy the joint angles only
              C  copy + ankle balance (pitch and roll, stance foot only)
              H  held in the air (pelvis fixed, no floor): can the motors follow
                 the motion at all, and do any joints hit their torque limit?

    python src/drills.py              # all clips
    python src/drills.py lunge_A      # one clip
"""
import json
import sys

import cv2
import mujoco
import numpy as np

from p2h import dtw, find_peaks_simple, gaussian_smooth, joint_angle
from paths import G1_XML, OUT

DRILL_OUT = OUT / "drills"

# kick_A1 was filmed almost front-on (13 deg) instead of about 45 deg; from the front the
# phone reads forward lean as depth and it swung by 20 deg while the athlete stood still,
# so it is processed but left out of the results table.
EXCLUDED = {"kick_A1": "filmed almost front-on (13 deg), outside the 45 deg protocol; one kick only"}
CLIPS = {  # label -> (drill, athlete); squat_S is the published squat, rerun as a control
    "squat_S": ("squat", "S"),
    "armraise_front_A": ("arm_raise", "A"), "armraise_side_A": ("arm_raise", "A"), "armraise_B": ("arm_raise", "B"),
    "lunge_A": ("lunge", "A"), "lunge_B": ("lunge", "B"),
    "kick_A1": ("kick", "A"), "kick_A2": ("kick", "A"), "kick_B": ("kick", "B"),
}

# MediaPipe landmarks
NOSE, L_EAR, R_EAR = 0, 7, 8
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 11, 12, 13, 14, 15, 16
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 23, 24, 25, 26, 27, 28
L_HEEL, R_HEEL, L_TOE, R_TOE = 29, 30, 31, 32
SIDES = {"l": dict(sh=L_SH, el=L_EL, wr=L_WR, hip=L_HIP, knee=L_KNEE, ank=L_ANK, heel=L_HEEL, toe=L_TOE),
         "r": dict(sh=R_SH, el=R_EL, wr=R_WR, hip=R_HIP, knee=R_KNEE, ank=R_ANK, heel=R_HEEL, toe=R_TOE)}
FULL = {"l": "left", "r": "right"}

KP, KD, MAX_PITCH, MAX_ROLL = 4.0, 0.4, 0.3, 0.26  # ankle balance gains (rad per m), clips (rad)
# symmetric drills copy one averaged leg pair (left/right differences are measurement noise there)
DRILL_OPTS = {"squat": dict(symmetric_legs=True), "arm_raise": dict(symmetric_legs=True), "lunge": {}, "kick": {}}


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def angle_between(a, b):
    return np.degrees(np.arccos(np.clip(np.sum(unit(a) * unit(b), axis=-1), -1, 1)))


def rotation_between(a, b):
    """Smallest rotation matrix taking unit vector a onto unit vector b."""
    a, b = unit(a), unit(b)
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


# ---------------------------------------------------------------- human side

def fill_smooth(X, sigma):
    """Interpolate NaN frames and Gaussian-smooth every coordinate over time."""
    X = X.astype(float).copy()
    idx = np.arange(len(X))
    good = ~np.isnan(X[:, 0, 0])
    for j in range(X.shape[1]):
        for k in range(X.shape[2]):
            s = X[:, j, k]
            s[~good] = np.interp(idx[~good], idx[good], s[good])
            X[:, j, k] = gaussian_smooth(s, sigma)
    return X, float(good.mean())


def load_world(label, sigma=1.5):
    """Hip-centred world landmarks (T, 33, 3) and image landmarks (T, 33, 2), smoothed."""
    if label == "squat_S":  # outputs of run_all.py oblique45
        z = np.load(OUT / "oblique45_pose.npz")
        W, detected = fill_smooth(z["world"], sigma)
        I, _ = fill_smooth(np.load(OUT / "oblique45_image_lm.npy")[:len(W)], 1.0)
        return W, I, float(z["fps"]), detected
    z = np.load(DRILL_OUT / f"{label}_pose.npz")
    W, detected = fill_smooth(z["world"], sigma)
    I, _ = fill_smooth(z["image"], 1.0)
    return W, I, float(z["fps"]), detected


def privacy_path(label):
    return OUT / "oblique45_privacy.mp4" if label == "squat_S" else DRILL_OUT / f"{label}_privacy.mp4"


def body_frame(W):
    """Per-frame body axes in MediaPipe world coordinates: fwd, left, up.
    up = hips -> shoulders; left = average of hip and shoulder lines."""
    up = unit((W[:, L_SH] + W[:, R_SH]) / 2 - (W[:, L_HIP] + W[:, R_HIP]) / 2)
    left = unit(W[:, L_HIP] - W[:, R_HIP]) + unit(W[:, L_SH] - W[:, R_SH])
    left = unit(left - np.sum(left * up, 1, keepdims=True) * up)
    fwd = np.cross(left, up)
    nose_fwd = W[:, NOSE] - (W[:, L_EAR] + W[:, R_EAR]) / 2
    if np.median(np.sum(fwd * nose_fwd, 1)) < 0:  # guard against handedness flips
        fwd = -fwd
    return fwd, left, up


def to_body(v, F):
    fwd, left, up = F
    return np.stack([np.sum(v * fwd, 1), np.sum(v * left, 1), np.sum(v * up, 1)], axis=1)


def foot_contact(I, W, still, g_up, thr=0.008):
    """Which feet are planted, from the phone image: a planted foot stays put in
    the picture (heel or toe still), a swinging foot moves. 3D foot heights from
    one phone are too noisy for this (up to 15 cm off in a squat). At least one
    foot is always down in these drills, so if both seem to move (camera shake)
    both count as planted."""
    body_px = np.median(np.abs(I[:, NOSE, 1] - (I[:, L_ANK, 1] + I[:, R_ANK, 1]) / 2))
    spd = {}
    for s, J in SIDES.items():
        v = np.minimum(np.linalg.norm(np.gradient(I[:, J["heel"]], axis=0), axis=1),
                       np.linalg.norm(np.gradient(I[:, J["toe"]], axis=0), axis=1)) / body_px
        spd[s] = gaussian_smooth(v, 1.5)
    low = {s: np.min(W[:, [J["ank"], J["heel"], J["toe"]]] @ g_up, axis=1) for s, J in SIDES.items()}
    diff = (low["l"] - low["r"]) - np.median((low["l"] - low["r"])[still])
    up_l = (spd["l"] > thr) & (spd["l"] > 1.5 * spd["r"]) & (diff > -0.02)
    up_r = (spd["r"] > thr) & (spd["r"] > 1.5 * spd["l"]) & (diff < 0.02)
    contact = {}
    for s, up in (("l", up_l), ("r", up_r)):
        k = 2  # median filter, 5 frames
        contact[s] = np.array([np.median(~up[max(0, i - k):i + k + 1]) >= 0.5 for i in range(len(up))])
    return contact, spd, diff


def human_features(W, fps, I=None):
    F = body_frame(W)
    # Vertical: a tilted phone makes MediaPipe's -y a poor "up", so take the
    # ankles -> shoulders direction while the athlete stands still.
    mid_sh, mid_ank = (W[:, L_SH] + W[:, R_SH]) / 2, (W[:, L_ANK] + W[:, R_ANK]) / 2
    pre = {"knee_l": 180 - joint_angle(W[:, L_HIP], W[:, L_KNEE], W[:, L_ANK]),
           "knee_r": 180 - joint_angle(W[:, R_HIP], W[:, R_KNEE], W[:, R_ANK])}
    still = (pre["knee_l"] + pre["knee_r"]) / 2 + np.abs(pre["knee_l"] - pre["knee_r"]) \
        + np.degrees(np.arccos(np.clip(np.sum(unit(W[:, L_KNEE] - W[:, L_HIP]) * unit(W[:, R_KNEE] - W[:, R_HIP]), 1), -1, 1)))
    still = still <= np.percentile(still, 15)
    g_up = unit(np.median(unit(mid_sh - mid_ank)[still], axis=0))
    g = to_body(np.tile(g_up, (len(W), 1)), F)  # world up in body frame
    pitch = np.degrees(np.arcsin(np.clip(-g[:, 0], -1, 1)))  # + = leaning forward
    roll = np.degrees(np.arctan2(g[:, 1], g[:, 2]))
    to_cam = to_body(np.tile([0.0, 0.0, -1.0], (len(W), 1)), F)
    view_yaw = float(np.degrees(np.arctan2(np.median(to_cam[:, 1]), np.median(to_cam[:, 0]))))
    seg = {}
    for s, J in SIDES.items():
        seg[f"uarm_{s}"] = to_body(W[:, J["el"]] - W[:, J["sh"]], F)
        seg[f"farm_{s}"] = to_body(W[:, J["wr"]] - W[:, J["el"]], F)
        seg[f"thigh_{s}"] = to_body(W[:, J["knee"]] - W[:, J["hip"]], F)
        seg[f"shin_{s}"] = to_body(W[:, J["ank"]] - W[:, J["knee"]], F)
        seg[f"foot_{s}"] = to_body(W[:, J["toe"]] - W[:, J["heel"]], F)
    ang = {}
    for s, J in SIDES.items():
        ang[f"knee_{s}"] = 180 - joint_angle(W[:, J["hip"]], W[:, J["knee"]], W[:, J["ank"]])
        ang[f"elbow_{s}"] = 180 - joint_angle(W[:, J["sh"]], W[:, J["el"]], W[:, J["wr"]])
        ang[f"ankle_{s}"] = joint_angle(W[:, J["knee"]], W[:, J["ank"]], W[:, J["toe"]])
        ua, th = unit(seg[f"uarm_{s}"]), unit(seg[f"thigh_{s}"])
        ang[f"arm_elev_{s}"] = np.degrees(np.arccos(np.clip(-ua[:, 2], -1, 1)))  # 0 hanging, 180 overhead
        ang[f"arm_plane_{s}"] = np.degrees(np.arctan2(np.abs(ua[:, 1]), ua[:, 0]))  # 0 front, 90 side
        ang[f"hip_flex_{s}"] = np.degrees(np.arctan2(th[:, 0], -th[:, 2]))  # + thigh forward, - behind
    if I is not None:
        contact, foot_speed, diff = foot_contact(I, W, still, g_up)
    else:
        contact = {s: np.ones(len(W), bool) for s in SIDES}
        foot_speed, diff = None, None
    base = {"pitch": pitch, "roll": roll}
    return dict(F=F, seg=seg, ang=ang, base=base, contact=contact, view_yaw=view_yaw, g_up=g_up,
                foot_speed=foot_speed, foot_height_diff=diff)


def standing_frames(H):
    """Frames that look most like quiet standing: knees straightest and arms lowest."""
    a = H["ang"]
    score = (a["knee_l"] + a["knee_r"]) / 2 + (a["arm_elev_l"] + a["arm_elev_r"]) / 4 \
        + np.abs(a["hip_flex_l"] - a["hip_flex_r"])
    both = H["contact"]["l"] & H["contact"]["r"]
    score = np.where(both, score, np.inf)
    return score <= np.percentile(score[np.isfinite(score)], 15)


def edges(x, p):
    left = p - 1
    while left > 0 and x[left - 1] <= x[left]:
        left -= 1
    right = p + 1
    while right < len(x) - 1 and x[right + 1] <= x[right]:
        right += 1
    return left, right


def reps_of(signal, fps, min_rise):
    out = []
    for p in find_peaks_simple(signal, min_rise=min_rise, min_gap=int(0.6 * fps)):
        l, r = edges(signal, p)
        if l == 0 or r == len(signal) - 1:
            continue  # cut off by the start or end of the clip
        out.append((int(l), int(p), int(r)))
    return out


def gap_matrix(seqs):
    n = len(seqs)
    G = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            _, path = dtw(seqs[i], seqs[j])
            ia, ib = zip(*path)
            G[i, j] = G[j, i] = float(np.mean(np.abs(seqs[i][list(ia)] - seqs[j][list(ib)])))
    return G


def human_reps(drill, H, W, fps):
    """Rep detection and per-rep numbers for each drill. Returns (reps, sequences)."""
    a, F = H["ang"], H["F"]
    reps, seqs = [], []
    if drill == "squat":
        sig = (a["knee_l"] + a["knee_r"]) / 2
        for l, p, r in reps_of(sig, fps, 40):
            reps.append({"start_s": l / fps, "bottom_s": p / fps, "end_s": r / fps, "knee_deg": float(sig[p]),
                         "hip_flex_deg": float((a["hip_flex_l"][p] + a["hip_flex_r"][p]) / 2)})
            seqs.append(np.stack([a[k][l:r + 1] for k in ("knee_l", "knee_r", "hip_flex_l", "hip_flex_r")], 1))
    elif drill == "arm_raise":
        sig = (a["arm_elev_l"] + a["arm_elev_r"]) / 2
        for l, p, r in reps_of(sig, fps, 60):
            reps.append({"start_s": l / fps, "peak_s": p / fps, "end_s": r / fps,
                         "peak_arm_elevation_deg": float(sig[p]),
                         "elevation_left_right_deg": [float(a["arm_elev_l"][p]), float(a["arm_elev_r"][p])],
                         "arm_plane_deg_0front_90side": float((a["arm_plane_l"][p] + a["arm_plane_r"][p]) / 2),
                         "elbow_bend_deg": float((a["elbow_l"][p] + a["elbow_r"][p]) / 2)})
            seqs.append(np.stack([a[k][l:r + 1] for k in ("arm_elev_l", "arm_elev_r", "elbow_l", "elbow_r")], 1))
    elif drill == "lunge":
        sig = (a["knee_l"] + a["knee_r"]) / 2
        fwd_pos = {s: np.sum((W[:, J["ank"]] - (W[:, L_HIP] + W[:, R_HIP]) / 2) * F[0], 1) for s, J in SIDES.items()}
        leg = np.linalg.norm(W[:, L_HIP] - W[:, L_KNEE], axis=1) + np.linalg.norm(W[:, L_KNEE] - W[:, L_ANK], axis=1)
        for l, p, r in reps_of(sig, fps, 35):
            f, b = ("l", "r") if fwd_pos["l"][p] > fwd_pos["r"][p] else ("r", "l")
            reps.append({"start_s": l / fps, "bottom_s": p / fps, "end_s": r / fps, "front_leg": FULL[f],
                         "front_knee_deg": float(a[f"knee_{f}"][p]), "rear_knee_deg": float(a[f"knee_{b}"][p]),
                         "front_hip_flex_deg": float(a[f"hip_flex_{f}"][p]), "rear_hip_flex_deg": float(a[f"hip_flex_{b}"][p]),
                         "stride_leg_lengths": float(abs(fwd_pos["l"][p] - fwd_pos["r"][p]) / np.median(leg)),
                         "rear_heel_up": bool(not H["contact"][b][p] or a[f"ankle_{b}"][p] > np.median(a[f"ankle_{b}"]) + 15)})
            seqs.append(np.stack([a[f"knee_{f}"][l:r + 1], a[f"knee_{b}"][l:r + 1],
                                  a[f"hip_flex_{f}"][l:r + 1], a[f"hip_flex_{b}"][l:r + 1]], 1))
    elif drill == "kick":
        hf = np.maximum(a["hip_flex_l"], a["hip_flex_r"])
        for l, p, r in reps_of(hf, fps, 30):
            k = "l" if a["hip_flex_l"][p] >= a["hip_flex_r"][p] else "r"
            st = "r" if k == "l" else "l"
            if np.sum(~H["contact"][k][l:r + 1]) / fps < 0.2:
                continue  # the "kicking" foot never left the ground: a weight shift, not a kick
            knee_v = np.gradient(a[f"knee_{k}"][l:r + 1]) * fps
            hip_v = np.gradient(a[f"hip_flex_{k}"][l:r + 1]) * fps
            foot_v = np.linalg.norm(np.gradient(W[l:r + 1, SIDES[k]["toe"]], axis=0), axis=1) * fps
            reps.append({"start_s": l / fps, "peak_s": p / fps, "end_s": r / fps, "kicking_leg": FULL[k],
                         "peak_hip_flexion_deg": float(a[f"hip_flex_{k}"][p]),
                         "knee_bend_at_peak_deg": float(a[f"knee_{k}"][p]),
                         "peak_hip_speed_deg_s": float(np.max(np.abs(hip_v))),
                         "peak_knee_speed_deg_s": float(np.max(np.abs(knee_v))),
                         "peak_foot_speed_m_s": float(np.max(foot_v)),
                         "single_leg_s": float(np.sum(~H["contact"][k][l:r + 1]) / fps),
                         "stance_knee_bend_deg": float(a[f"knee_{st}"][p])})
            seqs.append(np.stack([a[f"hip_flex_{k}"][l:r + 1], a[f"knee_{k}"][l:r + 1],
                                  a[f"hip_flex_{st}"][l:r + 1], a[f"knee_{st}"][l:r + 1]], 1))
    return reps, seqs


# ---------------------------------------------------------------- robot side

class G1Map:
    """Copies human limb directions onto the G1 within its joint limits."""

    LIMBS = {  # name -> (pitch, roll, yaw, bend joint, parent body, proximal body, mid body, distal body)
        f"arm_{s}": (f"{FULL[s]}_shoulder_pitch_joint", f"{FULL[s]}_shoulder_roll_joint", f"{FULL[s]}_shoulder_yaw_joint",
                     f"{FULL[s]}_elbow_joint", "torso_link", f"{FULL[s]}_shoulder_pitch_link", f"{FULL[s]}_elbow_link",
                     f"{FULL[s]}_wrist_roll_link") for s in SIDES} | {
        f"leg_{s}": (f"{FULL[s]}_hip_pitch_joint", f"{FULL[s]}_hip_roll_joint", f"{FULL[s]}_hip_yaw_joint",
                     f"{FULL[s]}_knee_joint", "pelvis", f"{FULL[s]}_hip_pitch_link", f"{FULL[s]}_knee_link",
                     f"{FULL[s]}_ankle_pitch_link") for s in SIDES}

    def __init__(self, xml=G1_XML, step_deg=1.5):
        self.m = mujoco.MjModel.from_xml_path(str(xml))
        self.m.vis.global_.offwidth, self.m.vis.global_.offheight = 1280, 1280
        self.d = mujoco.MjData(self.m)
        mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
        self.stand = self.d.qpos.copy()
        self.zero = self.stand.copy()
        self.zero[7:] = 0.0
        self.foot = {s: self.m.body(f"{FULL[s]}_ankle_roll_link").id for s in SIDES}
        self.spheres = {s: [g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] == self.foot[s]
                            and self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE] for s in SIDES}
        self.radius = float(self.m.geom_size[self.spheres["l"][0], 0])
        self.pelvis = self.m.body("pelvis").id
        self.grid = {name: self._grid(name, step_deg) for name in self.LIMBS}

    def jid(self, name):
        return self.m.joint(name).id

    def qadr(self, name):
        return self.m.jnt_qposadr[self.jid(name)]

    def lim(self, name):
        return self.m.jnt_range[self.jid(name)]

    def _segment(self, name, which):
        """Unit direction of a limb segment in its parent body frame (current d)."""
        _, _, _, _, parent, prox, mid, dist = self.LIMBS[name]
        a, b = (prox, mid) if which == "upper" else (mid, dist)
        v = self.d.xpos[self.m.body(b).id] - self.d.xpos[self.m.body(a).id]
        R = self.d.xmat[self.m.body(parent).id].reshape(3, 3)
        return unit(R.T @ v)

    def _grid(self, name, step):
        p_j, r_j, *_ = self.LIMBS[name]
        P = np.radians(np.arange(*np.degrees(self.lim(p_j)) + [0, step], step))
        Rr = np.radians(np.arange(*np.degrees(self.lim(r_j)) + [0, step], step))
        P, Rr = np.clip(P, *self.lim(p_j)), np.clip(Rr, *self.lim(r_j))
        pr, dirs = [], []
        self.d.qpos[:] = self.zero
        for p in P:
            for r in Rr:
                self.d.qpos[self.qadr(p_j)], self.d.qpos[self.qadr(r_j)] = p, r
                mujoco.mj_kinematics(self.m, self.d)
                pr.append((p, r))
                dirs.append(self._segment(name, "upper"))
        self.d.qpos[:] = self.zero
        mujoco.mj_kinematics(self.m, self.d)
        return np.array(pr), np.array(dirs)

    def rest_direction(self, name, which="upper"):
        self.d.qpos[:] = self.zero
        _, _, _, bend, *_ = self.LIMBS[name]
        if name.startswith("arm"):
            self.d.qpos[self.qadr(bend)] = np.pi / 2  # straight elbow
        mujoco.mj_kinematics(self.m, self.d)
        return self._segment(name, which)

    def track(self, name, uppers, w=0.15, K=10, spacing_deg=20):
        """Pitch/roll for a whole clip at once (dynamic programming).
        Each frame offers up to K well-separated near-best grid solutions; the
        chosen path minimises (direction error) + w * (joint travel between
        frames), so the robot never swaps shoulder or hip configuration unless
        staying put would cost more. Returns (pr per frame, error, jump)."""
        pr, dirs = self.grid[name]
        cands, errs = [], []
        for u in unit(uppers):
            err = np.degrees(np.arccos(np.clip(dirs @ u, -1, 1)))
            top = np.argpartition(err, 400)[:400]
            top = top[np.argsort(err[top])]
            chosen = []
            for k in top:
                if all(np.degrees(np.linalg.norm(pr[k] - pr[c])) > spacing_deg for c in chosen):
                    chosen.append(k)
                    if len(chosen) == K:
                        break
            cands.append(np.array(chosen))
            errs.append(err[chosen])
        n = len(cands)
        cost = [errs[0] + w * np.degrees(np.linalg.norm(pr[cands[0]], axis=1)) * 0.2]  # mild pull towards rest at the start
        back = [None]
        for i in range(1, n):
            D = np.degrees(np.linalg.norm(pr[cands[i]][:, None, :] - pr[cands[i - 1]][None, :, :], axis=2))
            tot = cost[-1][None, :] + w * D
            j = np.argmin(tot, axis=1)
            back.append(j)
            cost.append(errs[i] + tot[np.arange(len(j)), j])
        path = [int(np.argmin(cost[-1]))]
        for i in range(n - 1, 0, -1):
            path.append(int(back[i][path[-1]]))
        path = path[::-1]
        sel = np.array([cands[i][path[i]] for i in range(n)])
        P = pr[sel]
        E = np.array([errs[i][path[i]] for i in range(n)])
        J = np.concatenate([[0.0], np.degrees(np.linalg.norm(np.diff(P, axis=0), axis=1))])
        return P, E, J

    def solve_limb(self, name, upper, lower, bend_deg, prev, fixed=None, match_yaw=True):
        """Joint values (pitch, roll, yaw, bend) whose segments best match the
        target directions `upper` and `lower` (parent frame). `fixed` =
        (pitch, roll, error, jump) from track(); otherwise solved greedily with
        a continuity penalty against the previous frame."""
        p_j, r_j, y_j, b_j, *_ = self.LIMBS[name]
        pr, dirs = self.grid[name]
        if fixed is not None:
            p, r, best, jump = fixed
        else:
            err = np.degrees(np.arccos(np.clip(dirs @ unit(upper), -1, 1)))
            if prev is None:
                k = int(np.argmin(err))
                jump = 0.0
            else:  # a real joint moves continuously: trade a little accuracy for no sudden swaps
                dist = np.degrees(np.linalg.norm(pr - prev[:2], axis=1))
                k = int(np.argmin(err + 0.15 * dist))
                jump = float(dist[k])
            best = float(err[k])
            p, r = pr[k]
        bend = np.radians(bend_deg) if name.startswith("leg") else np.radians(90 - bend_deg)
        bend_c = float(np.clip(bend, *self.lim(b_j)))
        y_prev = 0.0 if prev is None else prev[2]
        y = y_prev
        if not match_yaw:
            y = 0.0
        elif bend_deg > 15:  # yaw only matters when the joint is bent
            lo, hi = self.lim(y_j)
            ys = np.unique(np.clip(np.concatenate([np.radians(np.arange(-150, 151, 6)), [y_prev]]), lo, hi))
            best_cost = np.inf
            for yy in ys:
                self.d.qpos[[self.qadr(p_j), self.qadr(r_j), self.qadr(y_j), self.qadr(b_j)]] = p, r, yy, bend_c
                mujoco.mj_kinematics(self.m, self.d)
                cost = angle_between(self._segment(name, "lower"), lower) + 0.05 * np.degrees(abs(yy - y_prev))
                if cost < best_cost:
                    best_cost, y = cost, yy
            for yy in np.clip(y + np.radians(np.arange(-5, 6, 1)), lo, hi):
                self.d.qpos[[self.qadr(p_j), self.qadr(r_j), self.qadr(y_j), self.qadr(b_j)]] = p, r, yy, bend_c
                mujoco.mj_kinematics(self.m, self.d)
                cost = angle_between(self._segment(name, "lower"), lower) + 0.05 * np.degrees(abs(yy - y_prev))
                if cost < best_cost:
                    best_cost, y = cost, yy
        self.d.qpos[[self.qadr(p_j), self.qadr(r_j), self.qadr(y_j), self.qadr(b_j)]] = p, r, y, bend_c
        mujoco.mj_kinematics(self.m, self.d)
        lower_err = float(angle_between(self._segment(name, "lower"), lower)) if (bend_deg > 15 and match_yaw) else 0.0
        return np.array([p, r, y]), dict(upper_err=best, lower_err=lower_err, jump_deg=jump,
                                         bend_clipped_deg=float(np.degrees(abs(bend - bend_c))))

    def plant_second_foot(self, s, sideways=False):
        """Both feet are planted but the copied leg angles disagree about where
        the floor is (each leg is measured separately and the far leg is the
        noisier one). Nudge leg `s` (hip pitch and knee, plus hip roll if
        sideways) by the least amount that puts the lowest point of its foot on
        the floor (a heel-up rear foot rests on its toes), re-levelling the foot
        in between. Returns the largest joint change in degrees."""
        from scipy.optimize import least_squares
        js = [f"{FULL[s]}_hip_pitch_joint", f"{FULL[s]}_knee_joint"] + ([f"{FULL[s]}_hip_roll_joint"] if sideways else [])
        adr = [self.qadr(j) for j in js]
        q_start = self.d.qpos[adr].copy()
        for _ in range(2):
            q0 = self.d.qpos[adr].copy()
            lo = np.array([self.lim(j)[0] for j in js]) - q0
            hi = np.array([self.lim(j)[1] for j in js]) - q0

            def resid(x):
                self.d.qpos[adr] = q0 + x
                mujoco.mj_kinematics(self.m, self.d)
                return np.concatenate([[self.sole_points(s)[:, 2].min() / 0.005], x / np.radians(10)])
            sol = least_squares(resid, np.zeros(len(js)), bounds=(np.minimum(lo, -1e-9), np.maximum(hi, 1e-9)))
            self.d.qpos[adr] = q0 + sol.x
            self.level_foot(s)
        return float(np.degrees(np.abs(self.d.qpos[adr] - q_start).max()))

    def sole_points(self, s):
        return np.array([self.d.geom_xpos[g] for g in self.spheres[s]]) - [0, 0, self.radius]

    def level_foot(self, s):
        """Ankle pitch/roll that make the sole level, clipped to the G1 range.
        Returns the remaining tilt in degrees."""
        pj, rj = f"{FULL[s]}_ankle_pitch_joint", f"{FULL[s]}_ankle_roll_joint"
        for _ in range(3):
            mujoco.mj_kinematics(self.m, self.d)
            Rf = self.d.xmat[self.foot[s]].reshape(3, 3)
            up_f = Rf.T @ np.array([0.0, 0, 1])  # world up seen from the foot
            dp = np.arctan2(-up_f[0], up_f[2])
            dr = np.arctan2(up_f[1], up_f[2])
            self.d.qpos[self.qadr(pj)] = np.clip(self.d.qpos[self.qadr(pj)] - dp, *self.lim(pj))
            self.d.qpos[self.qadr(rj)] = np.clip(self.d.qpos[self.qadr(rj)] - dr, *self.lim(rj))
        mujoco.mj_kinematics(self.m, self.d)
        Rf = self.d.xmat[self.foot[s]].reshape(3, 3)
        return float(np.degrees(np.arccos(np.clip(Rf[2, 2], -1, 1))))


def support_margin(points_xy, p):
    """Signed distance (m) from p to the convex hull of points (+ inside)."""
    pts = sorted(map(tuple, points_xy))
    if len(pts) < 3:
        return -np.inf

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def half(seq):
        h = []
        for q in seq:
            while len(h) >= 2 and cross(h[-2], h[-1], q) <= 0:
                h.pop()
            h.append(q)
        return h
    hull = half(pts)[:-1] + half(pts[::-1])[:-1]
    d = []
    for i in range(len(hull)):
        a, b = np.array(hull[i]), np.array(hull[(i + 1) % len(hull)])
        e = b - a
        n = np.array([-e[1], e[0]]) / np.linalg.norm(e)  # inward normal (counter-clockwise hull)
        d.append(float((p - a) @ n))
    return min(d)


def retarget(label, H, G, yaw_match=True, symmetric_legs=False, stand_arms=False, copy_roll=False, leg_twist=False,
             leg_lateral=False):
    """Joint targets for every frame + what the G1 could not copy.
    Options (used for the sensitivity checks): yaw_match turns the hip and
    shoulder yaw to match the shin and forearm; symmetric_legs averages the two
    legs (the published squat method); stand_arms keeps the G1's own arm pose;
    copy_roll copies sideways trunk tilt; leg_twist turns the hip yaw to match
    the shin; leg_lateral copies the sideways spread of each thigh.

    Defaults come from the squat check (see SENSITIVITY in the README): from one
    phone at 45 deg the sideways tilt of a squat reads up to 15 deg although a
    squat barely tilts, so it is noise, and copying it topples the robot even
    with ankle balance. The same goes for how far apart the feet are sideways:
    in athlete A's lunge and kick the phone put her feet 39-48 cm apart, which
    stood the robot in a diagonal stance with its weight off its feet while she
    was simply standing. So the legs are copied in the side view only (thigh
    swing and knee), at the G1's own stance width; sideways tilt, sideways leg
    spread and leg twist are not copied."""
    T = len(H["ang"]["knee_l"])
    stand = standing_frames(H)
    a, seg = H["ang"], H["seg"]
    # Rest offsets: rotate the athlete's resting limb directions onto the G1's.
    # Legs rest when standing; arms rest when that arm hangs lowest (an athlete
    # can stand still with the arms half raised, so standing alone is not enough).
    arms_low = {s: a[f"arm_elev_{s}"] <= np.percentile(a[f"arm_elev_{s}"], 8) for s in SIDES}
    off = {}
    for s in SIDES:
        for limb, key, frames in (("arm", "uarm", arms_low[s]), ("leg", "thigh", stand)):
            h_rest = unit(np.median(unit(seg[f"{key}_{s}"][frames]), axis=0))
            off[f"{limb}_{s}"] = rotation_between(h_rest, G.rest_direction(f"{limb}_{s}"))
    zero = {k: float(np.median(a[k][stand])) for k in ("knee_l", "knee_r", "ankle_l", "ankle_r")}
    # trunk lean is zeroed on the athlete's standing frames
    pitch0, roll0 = float(np.median(H["base"]["pitch"][stand])), float(np.median(H["base"]["roll"][stand]))
    # The replay begins at the first frame where the athlete stands on both feet
    # with straight knees (frame 0 when the clip starts that way).
    knees = (a["knee_l"] + a["knee_r"]) / 2 - (zero["knee_l"] + zero["knee_r"]) / 2
    ready = (knees < 15) & H["contact"]["l"] & H["contact"]["r"]
    s0 = int(np.argmax(ready)) if ready.any() else 0
    def side_view(u, lat):
        """Keep the forward/up swing of a thigh direction, set its sideways part to `lat`."""
        sag = u[..., [0, 2]] / np.linalg.norm(u[..., [0, 2]], axis=-1, keepdims=True) * np.sqrt(1 - lat ** 2)
        return np.stack([sag[..., 0], np.full(sag.shape[:-1], lat), sag[..., 1]], axis=-1)
    leg_lat = {s: float(G.rest_direction(f"leg_{s}")[1]) for s in SIDES}
    thigh_t = {s: (off[f"leg_{s}"] @ seg[f"thigh_{s}"].T).T for s in SIDES}
    if not leg_lateral:
        thigh_t = {s: side_view(thigh_t[s], leg_lat[s]) for s in SIDES}
    tracks = {}
    for s in SIDES:
        tracks[f"arm_{s}"] = G.track(f"arm_{s}", (off[f"arm_{s}"] @ seg[f"uarm_{s}"][s0:].T).T)
        tracks[f"leg_{s}"] = G.track(f"leg_{s}", thigh_t[s][s0:])
    targets, rows, prev = [], [], {k: None for k in G.LIMBS}
    anchor, anchor_xy, base_xyz = "l", None, G.zero[:3].copy()
    for i in range(s0, T):
        fix = {k: (*tracks[k][0][i - s0], tracks[k][1][i - s0], tracks[k][2][i - s0]) for k in tracks}
        G.d.qpos[:] = G.zero
        G.d.qpos[:3] = base_xyz  # start from last frame's placement so anchor switches stay continuous
        p, r = np.radians(H["base"]["pitch"][i] - pitch0), np.radians(H["base"]["roll"][i] - roll0)
        if not copy_roll:
            r = 0.0
        Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
        Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, (Ry @ Rx).flatten())
        G.d.qpos[3:7] = q
        row = {}
        for s in SIDES:
            knee = max(a[f"knee_{s}"][i] - zero[f"knee_{s}"], 0.0)
            sol, info = G.solve_limb(f"leg_{s}", thigh_t[s][i],
                                     off[f"leg_{s}"] @ seg[f"shin_{s}"][i], knee, prev[f"leg_{s}"], fix[f"leg_{s}"],
                                     match_yaw=leg_twist)
            prev[f"leg_{s}"] = sol
            row[f"leg_{s}"] = info
            sol, info = G.solve_limb(f"arm_{s}", off[f"arm_{s}"] @ seg[f"uarm_{s}"][i],
                                     off[f"arm_{s}"] @ seg[f"farm_{s}"][i], a[f"elbow_{s}"][i], prev[f"arm_{s}"],
                                     fix[f"arm_{s}"], match_yaw=yaw_match)
            prev[f"arm_{s}"] = sol
            row[f"arm_{s}"] = info
        if symmetric_legs:
            q = G.d.qpos
            for j in ("hip_pitch", "knee"):
                q[G.qadr(f"left_{j}_joint")] = q[G.qadr(f"right_{j}_joint")] =                     (q[G.qadr(f"left_{j}_joint")] + q[G.qadr(f"right_{j}_joint")]) / 2
            roll = (q[G.qadr("left_hip_roll_joint")] - q[G.qadr("right_hip_roll_joint")]) / 2
            q[G.qadr("left_hip_roll_joint")], q[G.qadr("right_hip_roll_joint")] = roll, -roll
            q[G.qadr("left_hip_yaw_joint")] = q[G.qadr("right_hip_yaw_joint")] = 0.0
        if stand_arms:
            for side in ("left", "right"):
                for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow"):
                    G.d.qpos[G.qadr(f"{side}_{j}_joint")] = G.stand[G.qadr(f"{side}_{j}_joint")]
        mujoco.mj_kinematics(G.m, G.d)
        contact = {s: bool(H["contact"][s][i]) for s in SIDES}
        if not any(contact.values()):
            contact = {"l": True, "r": True}
        for s in SIDES:
            if contact[s]:
                row[f"foot_tilt_{s}"] = G.level_foot(s)
            else:  # swing foot: copy the athlete's ankle bend
                pj = f"{FULL[s]}_ankle_pitch_joint"
                dorsi = zero[f"ankle_{s}"] - a[f"ankle_{s}"][i]
                G.d.qpos[G.qadr(pj)] = np.clip(-np.radians(dorsi), *G.lim(pj))
        mujoco.mj_kinematics(G.m, G.d)
        # anchor: keep one planted foot fixed on the floor
        if not contact[anchor]:
            anchor = "r" if anchor == "l" else "l"
            anchor_xy = None
        pts = G.sole_points(anchor)
        if anchor_xy is None:
            anchor_xy = pts[:, :2].mean(0)
        G.d.qpos[0:2] += anchor_xy - pts[:, :2].mean(0)
        G.d.qpos[2] -= pts[:, 2].min()
        mujoco.mj_kinematics(G.m, G.d)
        base_xyz = G.d.qpos[:3].copy()
        other = "r" if anchor == "l" else "l"
        row["contact"] = contact
        row["second_foot_height_cm"] = float(G.sole_points(other)[:, 2].min() * 100) if contact[other] else None
        row["second_foot_fix_deg"] = G.plant_second_foot(other, sideways=leg_lateral) if contact[other] else 0.0
        mujoco.mj_kinematics(G.m, G.d)
        mujoco.mj_comPos(G.m, G.d)
        support = np.vstack([G.sole_points(s)[:, :2] for s in SIDES if contact[s]])
        row["com_margin_cm"] = 100 * support_margin(support, G.d.subtree_com[G.pelvis][:2])
        targets.append(G.d.qpos.copy())
        rows.append(row)
    return np.array(targets), rows, s0


def kinematic_summary(rows, G):
    out = {}
    for limb in G.LIMBS:
        up = np.array([r[limb]["upper_err"] for r in rows])
        lo = np.array([r[limb]["lower_err"] for r in rows])
        bc = np.array([r[limb]["bend_clipped_deg"] for r in rows])
        jp = np.array([r[limb]["jump_deg"] for r in rows])
        out[limb] = {"max_dir_error_deg": round(float(up.max()), 1), "share_frames_over_10deg": round(float(np.mean(up > 10)), 3),
                     "max_lower_segment_error_deg": round(float(lo.max()), 1), "max_bend_clipped_deg": round(float(bc.max()), 1),
                     "max_joint_jump_per_frame_deg": round(float(jp.max()), 1), "jumps_over_30deg": int(np.sum(jp > 30))}
    tilts = [r[f"foot_tilt_{s}"] for r in rows for s in SIDES if r["contact"][s] and f"foot_tilt_{s}" in r]
    sec = [r["second_foot_height_cm"] for r in rows if r["second_foot_height_cm"] is not None]
    marg = np.array([r["com_margin_cm"] for r in rows])
    out["planted_foot_max_tilt_deg"] = round(float(max(tilts)), 1) if tilts else None
    out["second_planted_foot_height_cm"] = ({"min": round(float(min(sec)), 1), "max": round(float(max(sec)), 1)} if sec else None)
    fx = [r["second_foot_fix_deg"] for r in rows]
    out["second_foot_fix_deg"] = {"median": round(float(np.median(fx)), 1), "max": round(float(max(fx)), 1)}
    out["com_margin_cm"] = {"min": round(float(marg.min()), 1), "median": round(float(np.median(marg)), 1),
                            "share_frames_outside_feet": round(float(np.mean(marg < 0)), 3)}
    out["single_leg_share"] = round(float(np.mean([sum(r["contact"].values()) == 1 for r in rows])), 3)
    return out


# ---------------------------------------------------------------- physics

class DrillSim:
    """mode: 'copy' (A), 'balance' (C) or 'held' (H)."""

    def __init__(self, G, targets, contact, fps, mode):
        self.m = mujoco.MjModel.from_xml_path(str(G1_XML))
        self.m.vis.global_.offwidth, self.m.vis.global_.offheight = 1280, 1280
        if mode == "held":
            self.m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.G, self.T, self.contact, self.fps, self.mode = G, targets, contact, fps, mode
        self.act = np.array([m.jnt_qposadr[m.actuator_trnid[a, 0]] for a in range(m.nu)])
        self.dof = np.array([m.jnt_dofadr[m.actuator_trnid[a, 0]] for a in range(m.nu)])
        self.lim = np.array([m.jnt_actfrcrange[m.actuator_trnid[a, 0], 1] for a in range(m.nu)])
        self.ank = {s: (m.actuator(f"{FULL[s]}_ankle_pitch_joint").id, m.actuator(f"{FULL[s]}_ankle_roll_joint").id)
                    for s in SIDES}
        self.d.qpos[:] = targets[0]
        self.d.qpos[2] += 0.002
        mujoco.mj_forward(m, self.d)
        self.base0 = self.d.qpos[:7].copy()
        self.steps = int(round((1 / fps) / m.opt.timestep))
        self.prev_e = np.zeros(2)
        self.fell_at, self.max_tilt, self.slide = None, 0.0, {s: 0.0 for s in SIDES}
        self.last_foot = {s: self.d.xpos[G.foot[s]][:2].copy() for s in SIDES}
        self.sat = np.zeros(m.nu)
        self.n = 0
        self.track_err = []
        self.max_corr = 0.0

    def step_frame(self, i):
        m, d, G = self.m, self.d, self.G
        prev_t = self.T[max(i - 1, 0)][self.act]
        cur_t = self.T[i][self.act]
        stance = [s for s in SIDES if self.contact[s][i]] or list(SIDES)
        for k in range(self.steps):
            c = prev_t + (cur_t - prev_t) * (k + 1) / self.steps
            if self.mode == "balance":
                mujoco.mj_comPos(m, d)
                ref = np.mean([d.xpos[G.foot[s]][:2] + d.xmat[G.foot[s]].reshape(3, 3)[:2, 0] * 0.035 for s in stance], 0)
                yaw = np.arctan2(d.xmat[G.pelvis].reshape(3, 3)[1, 0], d.xmat[G.pelvis].reshape(3, 3)[0, 0])
                e_w = d.subtree_com[G.pelvis][:2] - ref
                e = np.array([np.cos(yaw) * e_w[0] + np.sin(yaw) * e_w[1], -np.sin(yaw) * e_w[0] + np.cos(yaw) * e_w[1]])
                de = (e - self.prev_e) / m.opt.timestep
                self.prev_e = e
                cp = float(np.clip(KP * e[0] + KD * de[0], -MAX_PITCH, MAX_PITCH))
                cr = float(np.clip(-(KP * e[1] + KD * de[1]), -MAX_ROLL, MAX_ROLL))
                self.max_corr = max(self.max_corr, abs(cp), abs(cr))
                for s in stance:
                    ap, ar = self.ank[s]
                    c[ap] = np.clip(c[ap] + cp, *m.actuator_ctrlrange[ap])
                    c[ar] = np.clip(c[ar] + cr, *m.actuator_ctrlrange[ar])
            d.ctrl[:] = c
            mujoco.mj_step(m, d)
            if self.mode == "held":
                d.qpos[:7] = self.base0
                d.qvel[:6] = 0
            self.sat += np.abs(d.qfrc_actuator[self.dof]) >= self.lim - 1e-6
            self.n += 1
        self.track_err.append(np.degrees(np.abs(d.qpos[self.act] - cur_t)))
        R = d.xmat[G.pelvis].reshape(3, 3)
        tilt = float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))
        if self.fell_at is None:
            self.max_tilt = max(self.max_tilt, tilt)
            if self.mode != "held" and (tilt > 60 or d.xpos[G.pelvis][2] < 0.35):
                self.fell_at = i / self.fps
        for s in SIDES:
            p = d.xpos[G.foot[s]][:2].copy()
            if self.fell_at is None and i > 0 and self.contact[s][i] and self.contact[s][i - 1]:
                self.slide[s] += float(np.linalg.norm(p - self.last_foot[s]))
            self.last_foot[s] = p
        return tilt

    def result(self):
        m = self.m
        names = [m.actuator(a).name.replace("_joint", "") for a in range(m.nu)]
        sat = self.sat / max(self.n, 1)
        err = np.array(self.track_err)
        out = {"fell_at_s": None if self.fell_at is None else round(self.fell_at, 2),
               "max_pelvis_tilt_deg": round(self.max_tilt, 1),
               "torque_limit_hit": {names[a]: round(float(sat[a]), 3) for a in np.argsort(-sat) if sat[a] > 0.01},
               "worst_tracking_joints_deg": {names[a]: round(float(err[:, a].max()), 1)
                                             for a in np.argsort(-err.max(0))[:4]},
               "median_tracking_error_deg": round(float(np.median(err)), 2)}
        if self.mode != "held":
            out["planted_foot_slide_cm"] = {FULL[s]: round(v * 100, 1) for s, v in self.slide.items()}
        if self.mode == "balance":
            out["max_ankle_correction_deg"] = round(float(np.degrees(self.max_corr)), 1)
        return out


def simulate(G, targets, contact, fps, mode):
    sim = DrillSim(G, targets, contact, fps, mode)
    for i in range(len(targets)):
        sim.step_frame(i)
        if sim.fell_at is not None:
            break
    return sim.result()


def compare_video(label, G, targets, contact, fps, azimuth, s0=0):
    sims = [DrillSim(G, targets, contact, fps, "copy"), DrillSim(G, targets, contact, fps, "balance")]
    cap = cv2.VideoCapture(str(privacy_path(label)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, s0)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 2.6, azimuth, -8
    H, W = 960, 540
    font = cv2.FONT_HERSHEY_SIMPLEX
    writer = None
    with mujoco.Renderer(sims[0].m, H, W) as r:
        for i in range(len(targets)):
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (int(frame.shape[1] * H / frame.shape[0]), H))
            panels = [frame]
            for sim in sims:
                sim.step_frame(i)
                cam.lookat[:] = [sim.d.xpos[G.pelvis][0], sim.d.xpos[G.pelvis][1], 0.55]
                r.update_scene(sim.d, camera=cam)
                img = cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR)
                fell = sim.fell_at is not None
                cv2.putText(img, "FELL" if fell else "upright", (14, H - 20), font, 0.9,
                            (80, 80, 255) if fell else (140, 230, 140), 2, cv2.LINE_AA)
                panels.append(img)
            panel = np.hstack(panels)
            x = 0
            for t, p in zip(["athlete (phone video)", "G1: copy angles only", "G1: + ankle balance"], panels):
                cv2.rectangle(panel, (x, 0), (x + p.shape[1], 46), (20, 20, 20), -1)
                cv2.putText(panel, t, (x + 12, 32), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                x += p.shape[1]
            cv2.putText(panel, f"t = {(i + s0) / fps:4.1f} s", (14, H - 20), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            if writer is None:
                writer = cv2.VideoWriter(str(DRILL_OUT / f"{label}_g1_compare.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps, (panel.shape[1], panel.shape[0]))
            writer.write(panel)
    writer.release()
    cap.release()


def r1(x):
    if isinstance(x, dict):
        return {k: r1(v) for k, v in x.items()}
    if isinstance(x, list):
        return [r1(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return round(float(x), 1)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def run(label, G=None, video=True):
    drill, athlete = CLIPS[label]
    W, I, fps, detected = load_world(label)
    H = human_features(W, fps, I)
    if drill in ("arm_raise", "squat"):  # feet never leave the floor in these
        H["contact"] = {s: np.ones(len(W), bool) for s in SIDES}
    reps, seqs = human_reps(drill, H, W, fps)
    gaps = gap_matrix(seqs)
    G = G or G1Map()
    targets, rows, s0 = retarget(label, H, G, **DRILL_OPTS[drill])
    contact = {s: H["contact"][s][s0:] for s in SIDES}
    kin = kinematic_summary(rows, G)
    phys = {mode: simulate(G, targets, contact, fps, mode) for mode in ("copy", "balance", "held")}
    for p in phys.values():  # report times on the clip's own clock
        if p["fell_at_s"] is not None:
            p["fell_at_s"] = round(p["fell_at_s"] + s0 / fps, 2)
    azimuth = 180.0 + H["view_yaw"]
    report = {"label": label, "drill": drill, "athlete": athlete, "fps": fps, "frames": len(W),
              "duration_s": round(len(W) / fps, 1), "person_found_share": round(detected, 3),
              "camera_view_deg_from_front": round(H["view_yaw"], 0), "robot_replay_starts_s": round(s0 / fps, 2),
              "human": {"reps": r1(reps),
                        "mean_rep_to_rep_gap_deg": round(float(gaps[np.triu_indices(len(seqs), 1)].mean()), 1) if len(seqs) > 1 else None},
              "kinematic": kin, "physics": {"A_copy_only": phys["copy"], "C_copy_plus_ankle_balance": phys["balance"],
                                            "H_held_in_air": phys["held"]}}
    json.dump(report, open(DRILL_OUT / f"{label}_report.json", "w"), indent=1)
    np.savez(DRILL_OUT / f"{label}_targets.npz", targets=targets, contact_l=contact["l"], contact_r=contact["r"],
             fps=fps, azimuth=azimuth, s0=s0)
    seq_arr = np.empty(len(seqs), dtype=object)
    seq_arr[:] = seqs
    np.save(DRILL_OUT / f"{label}_rep_seqs.npy", seq_arr, allow_pickle=True)
    if video:
        compare_video(label, G, targets, contact, fps, azimuth, s0)
    return report


if __name__ == "__main__":
    labels = [a for a in sys.argv[1:] if not a.startswith("--")] or list(CLIPS)
    G = G1Map()
    for lab in labels:
        rep = run(lab, G, video="--novideo" not in sys.argv)
        print(json.dumps({k: rep[k] for k in ("label", "duration_s", "person_found_share", "camera_view_deg_from_front")}))
        print("  reps:", len(rep["human"]["reps"]), "gap:", rep["human"]["mean_rep_to_rep_gap_deg"])
        print("  kin:", json.dumps(rep["kinematic"]))
        for k, v in rep["physics"].items():
            print(f"  {k}: {json.dumps(v)}")
