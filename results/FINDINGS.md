# Phone to Humanoid: squat results (weeks 1 and 2)

Samuel Ogundipe · 29 September 2026

**Question:** when a humanoid copies an athlete from ordinary phone video, which movements survive, and where do they break?

**First movement tested:** a bodyweight squat, filmed on a phone at about 45° to the athlete (8 full reps, 17 s).

## Pipeline
1. MediaPipe Pose Landmarker (heavy) estimates 3D joint positions from each video frame. The person was detected in 100% of frames.
2. Knee, hip and trunk-lean angles are computed, smoothed and split into reps. Reps are compared with DTW, the method from my final-year thesis.
3. The angles are mapped onto a simulated Unitree G1 (MuJoCo Menagerie model, 29 joints). I measured the G1's joint sign conventions directly. Hip and knee angles are copied. The ankle angle is solved so both feet stay flat, and the body is shifted so the feet stay planted.
4. Physics replay: the G1's position actuators track those joint targets under gravity and foot contact.

## Results

| Stage | Result |
|---|---|
| Human consistency | Reps differ by about 2° after DTW alignment |
| Pose replay (no physics) | Every frame fits inside the G1's joint limits. The ankle needs up to 38.7° of shin-forward tilt (limit 50°). The centre of mass stays over the feet, with a 2 cm margin at the bottom. |
| Physics, copying joint angles only | **Falls forward at 1.2 s**, at the bottom of the first squat. With the robot's default arms instead of mine, it falls at 2.7 s. |
| Physics + ankle balance feedback | **Stays upright through all 8 squats.** The feedback shifts the ankle target in proportion to how far the centre of mass has drifted from mid-foot. It needed up to 17° of correction. |
| Caveat on that success | The feet shuffle backward about 65 cm over 17 s (roughly 8 cm per squat). This isn't friction creep: stricter contact settings don't change it. The simple controller stays upright by stepping backward, not by standing still. |
| Balance-aware retargeting | Tilting each target pose so the centre of mass sits over mid-foot needs only about 2.5° of extra trunk lean (6° at most). It lowers the ankle demand to 34° and the needed feedback to 11°. It still falls without feedback, and the feet still shuffle with it. |

## What this means
The squat is **kinematically possible but dynamically unstable** for the G1 when it simply copies joint angles. Every pose is reachable, but moving through the poses without active balance tips it over. A one-line ankle strategy prevents the fall but not the shuffling. Keeping both feet planted and staying upright needs a whole-body balance controller. That's the gap current work on learning humanoid skills from human video addresses with reinforcement learning, and it's what I want to work on at MBZUAI.

## Limits of this study (honest list)
- One athlete (me), one movement and one camera angle so far. The front-view clip reads squat depth about 14° deeper than the 45° clip, and standing knees look bent from the front, so camera placement matters.
- Angles are zeroed at my standing posture, because the single camera reads a straight standing knee as about 18° bent.
- Arms copy shoulder and elbow angles, not hand positions. The G1's long arms relative to its torso put its hands higher than mine.
- The Menagerie G1 enforces Unitree's joint torque limits (139 N·m at the knee, 50 N·m at the ankle; in the balance run the waist and right shoulder joints reach theirs). Its PD gains are untuned and the ankle balance rule reads the true centre of mass, which a real G1 would have to estimate. A real G1 would find this harder, not easier.
- The two clips were separate sets, so the front-vs-45° comparison mixes camera effect with set-to-set variation.

## Next
- Film the other three drills (arm raise, forward lunge, instep kick) with 2–3 adult volunteers.
- Run the same pipeline on each, and fill in one results table per drill.
- Optional stretch: train a small balance policy for the squat in MuJoCo Playground (free Colab GPU) and compare it with the ankle feedback, measuring both falls and foot slide.
