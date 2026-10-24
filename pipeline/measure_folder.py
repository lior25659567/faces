"""Measure every image in all_people_reference_images, resumably.

Resumable because the run is long and has been killed more than once: each image is written to
the results file as soon as it is measured, so re-running continues where it stopped rather than
starting over.

These are AI-generated images, as their own README says, so they cannot settle what real bodies
measure — ANSUR II is for that. What they are good for is the harder thing to get: 97 bodies in
the pose this app actually asks for, fitted clothing and a plain background, across ages, skin
tones and builds. That makes them a real test of whether the measurement works for everybody or
quietly fails for some.

    pipeline/.venv/bin/python pipeline/measure_folder.py [how_many]
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from PIL import Image, ImageOps

import apple_vision
import body_measure

ROOT = Path(__file__).parent.parent / "all_people_reference_images"
OUT = Path(__file__).parent.parent / "reference/user-photos/measured.json"


def main():
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    done = json.loads(OUT.read_text()) if OUT.exists() else {}
    shots = sorted(p for p in ROOT.rglob("*")
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
                   and "Group_previews" not in str(p))
    todo = [p for p in shots if str(p.relative_to(ROOT)) not in done]
    print(f"{len(shots)} images, {len(done)} already measured, {len(todo)} left")

    for path in todo[:budget]:
        key = str(path.relative_to(ROOT))
        try:
            photo = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        except Exception as error:
            done[key] = {"refused": f"unreadable ({type(error).__name__})"}
            continue
        faces = apple_vision.faces(photo)
        if not faces:
            done[key] = {"refused": "no face"}
        else:
            mask, joints = apple_vision.body(photo)
            build = body_measure.measure(mask, joints, faces[0])
            if build is None:
                if "leftShoulder" not in joints or "rightShoulder" not in joints:
                    why = "no shoulders"
                elif not any(j in joints for j in ("leftHip", "rightHip", "root")):
                    why = "no hip line, clothing over the waist"
                else:
                    why = "arms too close to the torso"
                done[key] = {"refused": why}
            else:
                done[key] = {k: v for k, v in {
                    "waist_to_shoulder": build.waist_to_shoulder,
                    "hip_to_shoulder": build.hip_to_shoulder,
                    "waist_to_hip": build.waist_to_hip,
                    "shoulder_to_head": build.shoulder_to_head,
                }.items() if v is not None}
        state = done[key].get("refused") or f"w/s {done[key]['waist_to_shoulder']:.3f}"
        print(f"  {len(done):3}/{len(shots)}  {state:34} {key[:46]}")
        OUT.write_text(json.dumps(done, indent=2) + "\n")   # after every one, so nothing is lost

    left = len(shots) - len(done)
    print(f"\n{len(done)} measured or refused, {left} left" + ("" if left else " — complete"))


if __name__ == "__main__":
    main()
