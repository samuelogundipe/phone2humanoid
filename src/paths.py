"""Project paths. Put your clips in data/, run setup_assets.py once, then run_all.py."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
MODEL = ROOT / "models" / "pose_landmarker_heavy.task"
G1_XML = ROOT / "third_party" / "mujoco_menagerie" / "unitree_g1" / "scene.xml"

# label -> video file in data/
CLIPS = {"oblique45": DATA / "oblique45.mp4", "front": DATA / "front.mp4"}
# MuJoCo camera azimuth that matches each filming angle (G1 faces +x)
AZIMUTH = {"oblique45": 225.0, "front": 180.0}

OUT.mkdir(exist_ok=True)
