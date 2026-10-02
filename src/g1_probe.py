"""Find the G1's forward axis and the sign of each joint we drive."""
import numpy as np
import mujoco

from paths import G1_XML

m = mujoco.MjModel.from_xml_path(str(G1_XML))
d = mujoco.MjData(m)
key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand")


def body(name):
    return d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)].copy()


def qadr(joint):
    return m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint)]


def reset():
    mujoco.mj_resetDataKeyframe(m, d, key)
    mujoco.mj_kinematics(m, d)


print("bodies:", [m.body(i).name for i in range(m.nbody)])
reset()
print("stand qpos base:", np.round(d.qpos[:7], 3))
for b in ["pelvis", "torso_link", "left_hip_pitch_link", "left_knee_link", "left_ankle_roll_link", "left_elbow_link",
          "left_wrist_yaw_link"]:
    try:
        print(f"  {b:24s}", np.round(body(b), 3))
    except Exception as e:
        print("  missing", b)

# Forward axis: the foot's geometry extends further toward the toes.
foot_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
gids = [g for g in range(m.ngeom) if m.geom_bodyid[g] == foot_id]
pts = np.array([d.geom_xpos[g] for g in gids])
print("foot geom x positions rel. ankle:", np.round(pts[:, 0] - body("left_ankle_roll_link")[0], 3))


def test(joint, delta, watch, ref):
    reset()
    before = body(watch) - body(ref)
    d.qpos[qadr(joint)] += delta
    mujoco.mj_kinematics(m, d)
    after = body(watch) - body(ref)
    print(f"{joint:28s} +{delta:.2f} rad -> {watch} moves {np.round(after - before, 3)} (rel {ref})")


test("left_hip_pitch_joint", 0.4, "left_knee_link", "left_hip_pitch_link")
test("left_knee_joint", 0.6, "left_ankle_roll_link", "left_knee_link")
test("left_shoulder_pitch_joint", 0.5, "left_elbow_link", "left_shoulder_pitch_link")
test("left_elbow_joint", 0.5, "left_wrist_yaw_link", "left_elbow_link")

# Ankle: with the shank fixed, which sign lifts the toes?
for sgn in (+1, -1):
    reset()
    d.qpos[qadr("left_ankle_pitch_joint")] = 0.3 * sgn
    mujoco.mj_kinematics(m, d)
    R = d.xmat[foot_id].reshape(3, 3)
    print(f"ankle pitch {0.3*sgn:+.1f}: foot x-axis in world = {np.round(R[:, 0], 3)}")

# Base pitch: rotate about +y by +0.3 rad and see where the torso goes.
reset()
t0 = body("torso_link")
q = np.zeros(4)
mujoco.mju_axisAngle2Quat(q, np.array([0.0, 1.0, 0.0]), 0.3)
d.qpos[3:7] = q
mujoco.mj_kinematics(m, d)
print("base +0.3 about +y: torso moves", np.round(body("torso_link") - t0, 3))
