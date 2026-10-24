"""Faces found by Apple's Vision framework, which comes with macOS and iOS.

The looking is done by apple-vision.swift, a small Swift program built the first time it is needed,
with the compiler from Xcode or its command line tools.
"""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "apple-vision.swift"
PROGRAM = HERE / "vendor" / "apple-vision"

# Vision's face box is a square from about the eyebrows to the chin. The framing in
# generate-frames.py and the close-up in restyle.py were tuned on boxes from the hairline to the
# chin, which, measured on 20 photos and 26 avatars, are 0.87 Vision widths wide and 1.175 tall,
# with their middle 0.088 widths higher. Photos and cartoon faces came out the same.
WIDE, TALL, HIGHER = 0.87, 1.175, 0.088


def build():
    """Compile the Swift program if it is missing or older than its source."""
    if PROGRAM.exists() and PROGRAM.stat().st_mtime >= SOURCE.stat().st_mtime:
        return
    PROGRAM.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["xcrun", "swiftc", "-O", "-o", str(PROGRAM), str(SOURCE)], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as problem:
        details = getattr(problem, "stderr", None) or problem
        raise SystemExit(f"Could not build {SOURCE.name}; install Xcode's command line tools with xcode-select --install.\n{details}")


def _look(image, body_mask=False):
    build()
    with tempfile.TemporaryDirectory() as folder:
        picture = Path(folder) / "picture.png"
        image.save(picture)
        args = [str(PROGRAM), str(picture)]
        mask_path = Path(folder) / "mask.png"
        if body_mask:
            args += ["--body-mask", str(mask_path)]
        found = json.loads(subprocess.run(args, check=True, capture_output=True, text=True).stdout)
        mask = np.asarray(Image.open(mask_path).convert("L")) > 128 if body_mask else None
    return found, mask


def faces(image):
    """The faces in a PIL image, largest first, as x0, y0, x1, y1 in pixels: hairline to chin, cheek to cheek."""
    found, _ = _look(image)
    boxes = []
    for face in found["faces"]:
        x0, y0, x1, y1 = face["box"]
        side = x1 - x0
        middle_x, middle_y = (x0 + x1) / 2, (y0 + y1) / 2 - HIGHER * side
        boxes.append((middle_x - WIDE * side / 2, middle_y - TALL * side / 2, middle_x + WIDE * side / 2, middle_y + TALL * side / 2))
    return boxes


def body(image):
    """Vision's read of the person's body, for body_measure.py: the segmentation mask (True where
    Vision sees a person, the size of the image) and the joints it found with reasonable
    confidence, by name (neck, root, leftShoulder, rightShoulder, leftHip, rightHip), each an
    (x, y) pixel position. A joint Vision could not find confidently is left out."""
    found, mask = _look(image, body_mask=True)
    joints = {name: tuple(point) for name, point in found.get("joints", {}).items()}
    return mask, joints
