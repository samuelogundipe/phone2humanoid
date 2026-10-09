# Phone to Humanoid

Can a humanoid robot copy an athlete from ordinary phone video? This project films a drill on a phone, turns it into 3D joint angles, replays it on a simulated Unitree G1, and measures where the copy breaks.

![You, the G1 copying joint angles only, and the G1 with ankle balance, 3 seconds into a squat](results/figures/hero_physics_compare.jpg)

*Bodyweight squat, 3 s in. Middle: the G1 driven only by my joint angles has already fallen. Right: the same targets plus a simple ankle balance rule, still squatting. Bystanders and the athlete's face are blurred.*

## First result: the squat

Filmed on a phone at about 45°, 8 full reps in 17 s.

| Stage | Result |
|---|---|
| Human consistency | Reps differ by about 2° after dynamic time warping (DTW) alignment |
| Pose replay, no physics | Every frame fits the G1's joint limits. The ankle needs up to 38.7° of shin-forward tilt (limit 50°), and the centre of mass stays over the feet, with a 2 cm margin at the bottom. |
| Physics, joint angles only | **Falls forward at 1.2 s**, at the bottom of the first squat (2.7 s with the robot's default arms) |
| Physics + ankle balance feedback | **Upright through all 8 squats**, with up to 17° of ankle correction, but the feet shuffle back about 65 cm. Stricter contact settings don't change this. |
| Balance-aware retargeting | Tilting each pose so the centre of mass sits over mid-foot (about 2.5° of extra lean) still falls without feedback |

**In short:** the squat is kinematically feasible but dynamically unstable for the G1 when it only copies joint angles. A simple ankle strategy prevents the fall but not the foot shuffle. Standing still and balanced needs a whole-body balance controller. The full write-up is in [results/FINDINGS.md](results/FINDINGS.md), with a one-page PDF version in [results/Phone2Humanoid_findings.pdf](results/Phone2Humanoid_findings.pdf).

Videos: [physics comparison](media/squat_G1_physics_compare.mp4) · [pose replay](media/squat_G1_pose_replay.mp4) · [skeleton tracking](media/squat_45deg_skeleton.mp4)

## Arm raise, lunge and kick

Two more athletes, one gym session, the same phone at about 45°. The table shows what happens when the G1 copies each movement.

| Drill | What the G1 cannot copy | Copy angles only | + ankle balance |
|---|---|---|---|
| Squat (the clip above, rerun as a control) | nothing | falls at 1.1 s | stays up |
| Arm raise, arms in front of the shoulder line | arm off by up to 14° near the top | falls at 1.8 s | stays up |
| Arm raise, arms beside or behind the shoulder line | the shoulder has to swap configuration, up to 111° (athlete A) and 330° (athlete B) in one frame | falls at 1.5 to 3.1 s | A stays up, B falls at 3.3 s |
| Forward lunge (both athletes) | the rear foot cannot stay flat (no toes) | falls at 2.1 to 3.1 s | falls at 2.6 to 3.9 s |
| Kick (both athletes) | the weight never moves over the standing foot | falls at the first kick | falls at the first kick |

Held in the air, the G1's motors follow every drill to within a tenth of a degree (median), so the falls are about balance rather than motor strength or speed. One phone at 45° also turned out to measure sideways movement badly (it read up to 15° of sideways tilt in a squat), so the drills copy the legs in the side view only. That is why the kicks fail: standing on one leg needs a sideways weight shift. The full write-up is in [results/drills/FINDINGS_drills.md](results/drills/FINDINGS_drills.md).

Videos (faces blurred): [front raise](media/drills/armraise_front_A_G1_compare.mp4) · [side raise](media/drills/armraise_side_A_G1_compare.mp4) · [arm raise B](media/drills/armraise_B_G1_compare.mp4) · [lunge A](media/drills/lunge_A_G1_compare.mp4) · [lunge B](media/drills/lunge_B_G1_compare.mp4) · [kick A](media/drills/kick_A2_G1_compare.mp4) · [kick B](media/drills/kick_B_G1_compare.mp4)

## How it works

1. **Pose.** MediaPipe Pose Landmarker (heavy model) estimates 33 3D landmarks in every frame.
2. **Angles and reps.** Knee, hip and trunk lean are computed, smoothed and split into reps. Reps are compared with DTW, the method from my B.Sc. thesis on remote physiotherapy monitoring.
3. **Retargeting.** The angles drive a Unitree G1 (MuJoCo Menagerie model, 29 joints). Each joint's sign convention was measured on the model (`src/g1_probe.py`). The ankle is solved so both feet stay flat, and the body is shifted so the feet stay planted.
4. **Physics.** The G1's position actuators track those targets under gravity and contact, with and without an ankle balance rule.
5. **Privacy.** In every shared video a person segmentation mask blurs everyone except the athlete, and the athlete's face is blurred as well.

## Run it

Python 3.10 to 3.12 (MediaPipe has no 3.13 build yet) and ffmpeg for the browser-friendly video copies.

```bash
pip install -r requirements.txt
python setup_assets.py            # pose model + the Unitree G1 folder from MuJoCo Menagerie
# put your clip at data/oblique45.mp4 (film at about 45 degrees, whole body in frame)
python run_all.py oblique45       # about 10 minutes on a laptop CPU
```

Results land in `outputs/`. To run the first half (pose, angles, reps, DTW, and a G1 joint-limit check) in Google Colab with no setup, open `notebooks/Phone2Humanoid_Week1.ipynb`.

For the drills, put clips in `data/drills/` named like the labels in `src/drills.py` (for example `lunge_A.mp4`), then:

```bash
python src/drill_pose.py          # pose + privacy video for every clip (slow on a CPU)
python src/drills.py              # retarget, physics, three-panel videos
python src/drills_checks.py       # leg drills again with the G1's own arm pose
python src/face_blur.py           # shareable copies with faces blurred
python src/drills_summary.py      # results/drills/summary.md and summary.json
python src/drills_figures.py      # stills in results/drills/figures
```

## Layout

```
run_all.py              one command, whole pipeline
setup_assets.py         downloads the pose model and the G1 model
src/p2h.py              pose -> angles -> reps -> DTW
src/privacy_blur.py     blur bystanders and the athlete's face; skeleton overlay
src/g1_probe.py         measures the G1's joint sign conventions
src/g1_retarget.py      human angles -> G1 joints; kinematic replay video
src/g1_physics.py       physics experiments and the three-panel video
src/drill_pose.py       pose + privacy video for the drill clips
src/drills.py           drills: 3D limb retargeting, foot contact, physics, videos
src/drills_checks.py    leg drills with the G1's own arm pose
src/drills_summary.py   results table and athlete-vs-athlete DTW gap
src/drills_figures.py   stills for the drill write-up
src/face_blur.py        face-blurred copies of the drill videos
notebooks/              Colab notebook for the first half
results/                findings, figures, numbers
media/                  blurred result videos
```

## Limitations

- The squat is one athlete; the drills add two athletes from one session, all filmed with one phone. A front-on view reads the squat about 14° deeper and reads straight standing knees as bent, so camera placement matters.
- One phone at 45° measures sideways movement badly, so the drills copy the legs in the side view only and leave out sideways tilt, sideways leg spread and leg twist. A second camera would recover them.
- Angles are zeroed at the athlete's standing posture, because one camera reads a straight knee as about 18° bent.
- Arms copy shoulder and elbow angles, not hand positions. The G1's long arms relative to its torso put its hands higher than the athlete's.
- The Menagerie G1 enforces Unitree's joint torque limits (139 N·m at the knee, 50 N·m at the ankle), but its PD gains are untuned and the ankle balance rule reads the true centre of mass, which a real robot would have to estimate. A real G1 would find this harder.

## Next

A learned whole-body balance policy for the movements that fall (the lunges and kicks), compared on the same numbers: time to fall and foot slide. And a second camera, to give the robot the sideways weight shift a one-leg movement needs.

## Built on

[MediaPipe](https://github.com/google-ai-edge/mediapipe) · [MuJoCo](https://github.com/google-deepmind/mujoco) · [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (Unitree G1 model, BSD-3-Clause). Related work that does this at scale with learning: [VideoMimic](https://github.com/hongsukchoi/VideoMimic) (CoRL 2025) and [ASAP](https://www.ri.cmu.edu/robots-with-moves-like-ronaldo-lebron-and-kobe/) (RSS 2025).

Samuel Ogundipe · samuel.t.ogundipe@gmail.com · MIT License
