# Arm raise, lunge and kick: which movements survive the copy?

The squat showed that a Unitree G1 can reach every pose of a phone-filmed squat but falls when it only copies the joint angles. This round asks the same question of three more drills, filmed with two more athletes, to see which movements survive the transfer and where each one breaks.

## Results

Two adult athletes, A and B, who agreed to be filmed and to have the results published. One gym session on 3 October 2026, one phone at 33° to 47° from the front. Faces and bystanders are blurred in every video.

| Drill | Clip | What the athlete did | What the G1 cannot copy | Copy angles only | + ankle balance |
|---|---|---|---|---|---|
| Squat (control) | the published squat | 8 full squats | nothing: fits every joint limit | falls at 1.1 s | stays up |
| Front raise | athlete A | 9 raises to 156°, arms in front of the shoulder line (78° from straight ahead) | arm direction off by up to 14° near the top | falls at 1.8 s | stays up |
| Side raise | athlete A | 9 raises to 131°, arms slightly behind the shoulder line (101°) | shoulder has to swap configuration 35 times, up to 111° in one frame | falls at 3.1 s | stays up |
| Arm raise | athlete B | 13 raises to 154°, the arm plane varying from 72° to 118° | shoulder swaps configuration 44 times, up to 330° in one frame | falls at 1.5 s | falls at 3.3 s |
| Forward lunge | athlete A | 16 lunges, alternating legs; front knee 99°, rear knee 86° | rear foot cannot stay flat (tilts up to 31°); leg angles nudged up to 34° to keep both feet down | falls at 3.1 s | falls at 3.9 s |
| Forward lunge | athlete B | 9 lunges, alternating legs; front knee 102°, rear knee 116° | rear foot cannot stay flat (tilts up to 48°); leg angles nudged up to 16° | falls at 2.1 s | falls at 2.6 s |
| Kick | athlete A | 15 kicks with both legs; thigh up to 72° on average, foot at 3.5 m/s | weight never moves over the standing foot: centre of mass outside the feet in 59% of frames | falls at 1.5 s | falls at 1.8 s |
| Kick | athlete B | 9 right-leg kicks; thigh up to 96°, foot at 3.6 m/s | same: centre of mass outside the feet in 67% of frames | falls at 1.1 s | falls at 1.6 s |

Held in the air with no floor, the G1's motors follow every one of these movements to within 0.03° to 0.08° (median). They reach a torque limit less than 3% of the time, except during the shoulder swaps in athlete B's arm raise (up to 10%). So the falls are about balance, not motor strength or speed.

## What each drill shows

**Squat (control).** The published squat clip, run through this new pipeline, gives the same answer as before: it falls in the first squat when the G1 only copies angles, and a simple ankle balance rule keeps it up through every rep. The feet slide about 24 cm here against about 65 cm in the first pipeline; the two pipelines build the robot's targets differently, so slide distance is not comparable between them.

**Arm raises.** Whether the G1 can follow depends on which side of the shoulder line the arms travel. Raised in front of it, the G1's shoulder (pitch, then roll) follows to the top of the raise. Raised beside or slightly behind it, the shoulder's sideways range runs out at 129°, so above about 110° of elevation the only way up is to swing the arm round through the front. Copied at the athlete's speed, that swap is a jump of up to 330° in one frame. Both of athlete A's clips stay up with ankle balance; athlete B's raises swap 44 times and the whipping arm topples the robot even with balance.

**Lunges.** The G1 has no toes, so the rear foot of a lunge cannot stay flat the way a heel-up human foot rests on the ball of the foot; it tilts by up to 31° to 48°. Because each leg is measured separately, the copied angles also disagree about where the floor is, and the legs need nudging (up to 16° to 34°) to keep both feet down. Both lunges fall during the first descent, with or without ankle balance.

**Kicks.** Both kick clips fall at the first kick. To stand on one leg, an athlete moves the hips sideways over the standing foot. From one phone at 45°, sideways movement is what the phone measures worst (see below), so this pipeline does not copy it. The G1 lifts a foot with its weight still between its feet and falls. One-leg movements need either a second camera or a balance controller that shifts the weight itself.

**Athlete against athlete.** Lining up each athlete's most typical rep with dynamic time warping, the lunge differs by 13.5° between A and B, against 6.5° and 7.4° from rep to rep within each athlete: B bends the rear knee about 30° more. The kicks differ by 8.0° between athletes, against 7.3° and 6.2° within. This is the coaching score from my final-year project, applied across athletes.

## What one phone can and cannot tell the robot

Three checks decided what gets copied:

- Rerun through this pipeline, the squat only reproduced the published result once sideways trunk tilt was left out. From one phone at 45°, the squat's sideways tilt read up to 15° although a squat barely tilts sideways, and copying it toppled the robot even with ankle balance.
- In athlete A's lunge and kick, the phone put her feet 39 to 48 cm apart sideways. Copying that stood the robot in a diagonal stance with its weight off its feet while she was simply standing.
- A third kick clip was filmed almost front-on (13° from the front). From the front, forward lean is a depth reading, and it swung by about 20° while she stood still, so that clip is left out of the table.

So the pipeline copies forward lean, hips and knees in the side view, and shoulders and elbows in 3D. It does not copy sideways trunk tilt, sideways leg spread or leg twist. For the two symmetric drills (squat, arm raise) it copies one averaged pair of legs.

## How it was run

1. **Pose.** MediaPipe Pose Landmarker (heavy) on every frame. The athlete was found in 100% of frames in all clips.
2. **Feet.** A foot counts as planted while it stays still in the picture. (3D foot heights from one phone were off by up to 15 cm in the squat.)
3. **Limbs.** Upper arms are copied as 3D directions and thighs in the side view, relative to the trunk, after lining up each athlete's resting pose with the G1's. The shoulder and hip angles are chosen for the whole clip at once, so the robot only swaps configuration when staying put would cost more.
4. **Contact.** The planted foot is kept flat and fixed. With both feet planted, the second leg is nudged by the least amount that puts its lowest point on the floor.
5. **Physics.** MuJoCo and the Menagerie G1 with its joint torque limits. Three conditions: copy the angles only; copy plus ankle balance (pitch and roll, on the standing foot); and held in the air with no floor. The replay starts at the athlete's first standing frame.

## Limits

- Two athletes, one session, one phone. A second camera would recover the sideways information this pipeline leaves out.
- Arms copy directions, not hand positions. Hands on hips or clasped at the chest push the G1's longer arms into its body. Rerunning the lunges and kicks with the G1's own arm pose did not change which ones fall; the squat with the G1's own arms lasted 12.8 s with ankle balance instead of the whole clip.
- The only balance rule is a simple ankle strategy. There is no stepping, no hip strategy and no learned controller.
- The G1's position gains are untuned, and the balance rule reads the true centre of mass, which a real robot would have to estimate.

## Videos

Three panels each: the athlete (faces blurred), the G1 copying angles only, the G1 with ankle balance.

- [Front raise, athlete A](../../media/drills/armraise_front_A_G1_compare.mp4)
- [Side raise, athlete A](../../media/drills/armraise_side_A_G1_compare.mp4)
- [Arm raise, athlete B](../../media/drills/armraise_B_G1_compare.mp4)
- [Lunge, athlete A](../../media/drills/lunge_A_G1_compare.mp4)
- [Lunge, athlete B](../../media/drills/lunge_B_G1_compare.mp4)
- [Kick, athlete A](../../media/drills/kick_A2_G1_compare.mp4)
- [Kick, athlete B](../../media/drills/kick_B_G1_compare.mp4)

All numbers are in [summary.json](summary.json); stills are in [figures/](figures/).
