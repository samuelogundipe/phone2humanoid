"""Extra physics condition for the leg drills: the same leg targets, but the
G1 keeps its own arm pose. Separates "the copied arms got in the way" (for
example hands clasped at the chest make the G1's longer arms press into each
other) from "the legs and balance are the problem". Adds two entries to each
report: B_robot_arms_copy_only and B_robot_arms_plus_ankle_balance.

    python src/drills_checks.py
"""
import json
import sys

import numpy as np

from drills import CLIPS, DRILL_OUT, G1Map, simulate

ARM_JOINTS = [f"{side}_{j}_joint" for side in ("left", "right")
              for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")]


def main(labels):
    G = G1Map()
    for lab in labels:
        drill, _ = CLIPS[lab]
        rp = DRILL_OUT / f"{lab}_report.json"
        if drill == "arm_raise" or not rp.exists():
            continue
        z = np.load(DRILL_OUT / f"{lab}_targets.npz")
        T = z["targets"].copy()
        for j in ARM_JOINTS:
            T[:, G.qadr(j)] = G.stand[G.qadr(j)]
        contact = {"l": z["contact_l"], "r": z["contact_r"]}
        fps, s0 = float(z["fps"]), int(z["s0"])
        r = json.load(open(rp))
        for key, mode in (("B_robot_arms_copy_only", "copy"), ("B_robot_arms_plus_ankle_balance", "balance")):
            res = simulate(G, T, contact, fps, mode)
            if res["fell_at_s"] is not None:
                res["fell_at_s"] = round(res["fell_at_s"] + s0 / fps, 2)
            r["physics"][key] = res
        json.dump(r, open(rp, "w"), indent=1)
        p = r["physics"]
        print(f"{lab}: athlete arms copy {p['A_copy_only']['fell_at_s']} / +bal {p['C_copy_plus_ankle_balance']['fell_at_s']}"
              f" | robot arms copy {p['B_robot_arms_copy_only']['fell_at_s']} / +bal {p['B_robot_arms_plus_ankle_balance']['fell_at_s']}",
              flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(CLIPS))
