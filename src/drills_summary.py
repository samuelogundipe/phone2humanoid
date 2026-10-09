"""Collect every drill report into one table (results/drills/summary.json and
summary.md), plus the athlete-vs-athlete DTW gap for drills both athletes did.

    python src/drills_summary.py
"""
import json

import numpy as np

from drills import CLIPS, DRILL_OUT, EXCLUDED, gap_matrix
from paths import ROOT

RES = ROOT / "results" / "drills"
RES.mkdir(parents=True, exist_ok=True)
NAMES = {"squat": "Squat", "arm_raise": "Arm raise", "lunge": "Forward lunge", "kick": "Kick"}


def fell(p):
    return "stays up" if p["fell_at_s"] is None else f"falls at {p['fell_at_s']:.1f} s"


def cannot_copy(k):
    """What the G1 could not copy, in words, from the kinematic summary."""
    notes = []
    arm_err = max(k["arm_l"]["max_dir_error_deg"], k["arm_r"]["max_dir_error_deg"])
    swaps = k["arm_l"]["jumps_over_30deg"] + k["arm_r"]["jumps_over_30deg"]
    if swaps:
        jump = max(k["arm_l"]["max_joint_jump_per_frame_deg"], k["arm_r"]["max_joint_jump_per_frame_deg"])
        notes.append(f"shoulder has to swap configuration {swaps} times (up to {jump:.0f} deg in one frame)")
    if arm_err > 10:
        notes.append(f"arm direction off by up to {arm_err:.0f} deg")
    if k["planted_foot_max_tilt_deg"] and k["planted_foot_max_tilt_deg"] > 10:
        notes.append(f"a planted foot cannot stay flat (tilts up to {k['planted_foot_max_tilt_deg']:.0f} deg)")
    if k["second_foot_fix_deg"]["max"] > 10:
        notes.append(f"leg angles nudged up to {k['second_foot_fix_deg']['max']:.0f} deg to keep both feet down")
    if k["single_leg_share"] > 0.05:
        notes.append(f"on one foot {k['single_leg_share']:.0%} of the time")
    out = k["com_margin_cm"]["share_frames_outside_feet"]
    if out > 0.02:
        notes.append(f"centre of mass outside the feet in {out:.0%} of frames")
    return "; ".join(notes) or "nothing: fits all joint limits"


def medoid(seqs):
    if len(seqs) == 1:
        return seqs[0]
    return seqs[int(np.argmin(gap_matrix(seqs).sum(1)))]


def main():
    reports = {lab: json.load(open(DRILL_OUT / f"{lab}_report.json")) for lab in CLIPS
               if (DRILL_OUT / f"{lab}_report.json").exists() and lab not in EXCLUDED}
    rows = []
    for lab, r in reports.items():
        k, ph, hu = r["kinematic"], r["physics"], r["human"]
        held = ph["H_held_in_air"]
        rows.append({
            "clip": lab, "drill": NAMES[r["drill"]], "athlete": r["athlete"], "seconds": r["duration_s"],
            "camera_deg_from_front": abs(r["camera_view_deg_from_front"]), "reps": len(hu["reps"]),
            "rep_gap_deg": hu["mean_rep_to_rep_gap_deg"], "cannot_copy": cannot_copy(k),
            "copy_only": fell(ph["A_copy_only"]), "with_ankle_balance": fell(ph["C_copy_plus_ankle_balance"]),
            "robot_arms_with_balance": fell(ph["B_robot_arms_plus_ankle_balance"]) if "B_robot_arms_plus_ankle_balance" in ph else None,
            "held_in_air_median_tracking_deg": held["median_tracking_error_deg"],
            "held_in_air_torque_limit": {j: v for j, v in held["torque_limit_hit"].items() if v >= 0.02},
            "kinematic": k,
        })
    cross = {}
    for drill in ("lunge", "kick"):
        per = {}
        for lab, (d, ath) in CLIPS.items():
            f = DRILL_OUT / f"{lab}_rep_seqs.npy"
            if d == drill and lab not in EXCLUDED and f.exists():
                per.setdefault(ath, []).extend(list(np.load(f, allow_pickle=True)))
        if len(per) == 2 and all(per.values()):
            cross[drill] = round(float(gap_matrix([medoid(per["A"]), medoid(per["B"])])[0, 1]), 1)
    json.dump({"clips": rows, "athlete_A_vs_B_gap_deg": cross, "excluded": EXCLUDED},
              open(RES / "summary.json", "w"), indent=1)

    lines = ["| Drill | Clip | Reps (rep-to-rep gap) | What the G1 cannot copy | Copy angles only | + ankle balance |",
             "|---|---|---|---|---|---|"]
    for x in rows:
        lines.append(f"| {x['drill']} | {x['clip']} | {x['reps']} ({x['rep_gap_deg']} deg) | {x['cannot_copy']} "
                     f"| {x['copy_only']} | {x['with_ankle_balance']} |")
    lines += ["", f"Athlete A vs B, DTW gap between each athlete's most typical rep: {cross}",
              f"Left out: {EXCLUDED}"]
    open(RES / "summary.md", "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
