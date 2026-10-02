"""One-time setup: download the MediaPipe pose model and fetch only the Unitree G1
folder from MuJoCo Menagerie (sparse checkout, about 40 MB)."""
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task")


def main():
    model = ROOT / "models" / "pose_landmarker_heavy.task"
    model.parent.mkdir(exist_ok=True)
    if not model.exists():
        print("downloading pose model ...")
        urllib.request.urlretrieve(MODEL_URL, model)
    print("pose model:", model, f"{model.stat().st_size / 1e6:.1f} MB")

    menagerie = ROOT / "third_party" / "mujoco_menagerie"
    if not (menagerie / "unitree_g1" / "scene.xml").exists():
        menagerie.parent.mkdir(exist_ok=True)
        print("fetching Unitree G1 from MuJoCo Menagerie ...")
        subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--sparse",
                        "https://github.com/google-deepmind/mujoco_menagerie.git", str(menagerie)], check=True)
        subprocess.run(["git", "sparse-checkout", "set", "unitree_g1"], cwd=menagerie, check=True)
    print("G1 model:", menagerie / "unitree_g1" / "scene.xml")


if __name__ == "__main__":
    sys.exit(main())
