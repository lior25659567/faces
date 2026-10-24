#!/usr/bin/env python
"""Animate the house character by cutting it at its joints and turning the pieces.

The 3D route — mesh, rig, Blender — is the right long-term answer and is written up in
pipeline/blender/build_animation.py. It needs a modelled character, an auto-rigger behind a login
and Blender installed, none of which exist yet. This is the same motion data driving something
that can ship today.

The trick that makes it work is a property of the drawing rather than of the code: the neutral
figure was drawn with its arms hanging CLEAR of the ribs, so everything below the elbow sits over
transparent backdrop and not over the body. A forearm can therefore be lifted out and rotated
with nothing left to patch behind it. That is also why `muscle_figure.py` asks for the arms held
away from the sides — it was written for the muscle map, and it happens to be exactly what a
cut-out puppet needs.

What this is honest about: it turns flat pieces in the picture's own plane. It cannot show
anything that needs depth — a squat seen from the front barely changes, and a bench press is not
in the picture at all. It is right for movements where the motion is a rotation in the plane the
character was drawn in, which is the arms, and the same shortlist the still drawings succeeded on.

    pipeline/.venv/bin/python pipeline/puppet.py --exercise "Dumbbell Bicep Curl (standing, both arms)"
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

import muscle_regions

ROOT = Path(__file__).resolve().parent.parent
FIGURE = ROOT / "reference" / "muscle-figure"
OUT = ROOT / "reference" / "exercise-art" / "animated"
APP = ROOT.parent / "faces-ios" / "App" / "Resources"

FRAMES = 16


def slug(name):
    return muscle_regions.__dict__.get("slug", lambda s: s)(name) if False else (
        "".join(c if c.isalnum() else "-" for c in name.lower()).strip("-").replace("--", "-"))


def landmarks(mask):
    """Where the elbows are and how wide the arms are, measured off the drawing.

    Reuses the same measurement the muscle map runs on, for the same reason: a number typed
    while looking at one picture stops being true the moment the picture is redrawn.
    """
    m = muscle_regions.landmarks(mask)
    figure = muscle_regions.Figure(mask)
    figure._rows = {y: muscle_regions.runs(mask[y]) for y in range(mask.shape[0])}
    top, span = m["top"], m["span"]
    waist = (m["waist"] - top) / span if m["waist"] else 0.47
    arm_rows = sorted(m["arms"])
    hand = (arm_rows[-1] - top) / span if arm_rows else 0.68
    elbow_y = top + span * waist

    out = {"elbow_y": elbow_y, "hand_y": top + span * hand, "top": top, "span": span,
           "arm_rows": m["arms"], "arm_top": min(m["arms"]) if m["arms"] else top + span * 0.41}
    if m["arms"]:
        at_elbow = m["arms"][min(m["arms"], key=lambda y: abs(y - elbow_y))]
        at_top = m["arms"][min(m["arms"])]
        for index, side in ((0, "left"), (1, "right")):
            out[f"elbow_x.{side}"] = sum(at_elbow[index]) / 2
            # The hinge is the CUT, not the anatomical shoulder. The shoulder sits about a
            # tenth of the figure's height above the row where the arm first comes clear of the
            # ribs, and everything above that row stays with the torso — so turning the arm
            # about the true shoulder swung the piece away from its own stump and left the arms
            # floating beside the body. Pivoting on the cut keeps them attached. The cost is
            # that the deltoid does not move, which on a lateral raise is barely visible and is
            # the right trade for an arm that stays on.
            out[f"shoulder_x.{side}"] = sum(at_top[index]) / 2
            out[f"shoulder_y.{side}"] = min(m["arms"])
    return out


# Where the shoulder joint sits, as a fraction of the figure's height. Measured off this same
# drawing by muscle_regions, which finds it from the narrowest row where the head meets the body.
SHOULDER = 0.30


def cut_arm(image, marks, side, joint, pad=6):
    """The forearm and hand as their own layer, plus the body with that piece removed.

    Taken from the ARM's own pixels row by row, not from everything below the elbow. The first
    attempt used a horizontal line and a centre split, which lifted half the torso, the shorts
    and a leg along with the arm and turned them all about the elbow — the figure came apart.
    The per-row extents are already measured for the muscle map, and an arm is whatever sits
    outside the body on its own side of it, so that is what gets cut.
    """
    width, height = image.size
    pixels = np.array(image)
    keep = np.zeros((height, width), dtype=bool)
    index = 0 if side == "left" else 1
    # The whole arm when the shoulder is the hinge, everything below the elbow when it is the
    # elbow. Cutting at the shoulder has no seam to hide: the arm meets the torso at a rounded
    # joint and the rotation sweeps around it.
    cut_from = (marks["arm_top"] if joint == "shoulder" else marks["elbow_y"]) - pad
    for y, pieces in marks["arm_rows"].items():
        if y < cut_from:
            continue
        left, right = pieces[index]
        keep[y, max(left - pad, 0):min(right + pad + 1, width)] = True

    limb = pixels.copy()
    limb[~keep] = 0
    rest = pixels.copy()
    rest[keep] = 0
    return Image.fromarray(limb, "RGBA"), Image.fromarray(rest, "RGBA")


def rest_angle(marks, side):
    """How far the drawn arm already hangs away from straight down, in degrees.

    The neutral figure was drawn with its arms held clear of the ribs — deliberately, so the
    muscle map could show the lats — which means they do not start at zero. Rotating by a full
    90 degrees therefore finished at about 99 and put the hands ABOVE shoulder height, which the
    form notes name as a fault in so many words. Every swing is measured from here instead.
    """
    rows = marks["arm_rows"]
    index = 0 if side == "left" else 1
    top_y, bottom_y = min(rows), max(rows)
    top = sum(rows[top_y][index]) / 2
    bottom = sum(rows[bottom_y][index]) / 2
    return math.degrees(math.atan2(abs(bottom - top), bottom_y - top_y))


def attach_dumbbells(base, marks):
    """Put a dumbbell in each hand, taken from the curl drawing that was checked and shipped.

    A lateral raise without dumbbells is a person shrugging. The neutral figure was drawn
    empty-handed because it was made for the muscle map, and rather than spend a Workers AI call
    redrawing it — with the odds the rest of this project has measured — the hand is borrowed
    from `dumbbell-bicep-curl-standing-both-arms-start.png`, which is the same character from the
    same seed at the same scale and has already passed its form check.

    The sprite carries its own hand, so it replaces the figure's rather than sitting on top of
    one: a sliver of the original showing past a dumbbell would read as two hands.
    """
    meta_path = FIGURE / "dumbbell.json"
    if not meta_path.exists():
        return base
    meta = json.loads(meta_path.read_text())
    out = base.copy()
    pixels = np.array(out)
    rows = marks["arm_rows"]
    for index, side in ((0, "left"), (1, "right")):
        sprite = Image.open(FIGURE / f"dumbbell-{side}.png").convert("RGBA")
        wrist_x, wrist_y = meta[side]["wrist"]
        # The figure's own wrist: three quarters of the way from the elbow to the end of the arm.
        cut = marks["elbow_y"] + (max(rows) - marks["elbow_y"]) * 0.72
        near = min(rows, key=lambda y: abs(y - cut))
        piece = rows[near][index]
        here = (sum(piece) / 2, near)
        for y, pieces in rows.items():
            if y >= near:
                left, right = pieces[index]
                pixels[y, max(left - 4, 0):right + 5] = 0
        out = Image.fromarray(pixels, "RGBA")
        out.alpha_composite(sprite, (int(here[0] - wrist_x), int(here[1] - wrist_y)))
        pixels = np.array(out)
    return out


def shift(marks, dx, dy):
    """Move the measured landmarks with the picture they were measured on."""
    moved = dict(marks)
    for key, value in marks.items():
        if key.endswith("_y") or key.startswith("elbow_y"):
            moved[key] = value + dy
        elif "_y." in key:
            moved[key] = value + dy
        elif "_x." in key:
            moved[key] = value + dx
    moved["arm_top"] = marks["arm_top"] + dy
    moved["arm_rows"] = {y + dy: ((a[0] + dx, a[1] + dx), (b[0] + dx, b[1] + dx))
                         for y, (a, b) in marks["arm_rows"].items()}
    return moved


def trim(frames, margin=10):
    """Crop every frame to the union of what they all use, so the figure never moves between
    them and nothing is cut off at the extremes."""
    box = None
    for frame in frames:
        found = frame.getbbox()
        if not found:
            continue
        box = found if box is None else (min(box[0], found[0]), min(box[1], found[1]),
                                         max(box[2], found[2]), max(box[3], found[3]))
    if not box:
        return frames
    width, height = frames[0].size
    box = (max(box[0] - margin, 0), max(box[1] - margin, 0),
           min(box[2] + margin, width), min(box[3] + margin, height))
    return [frame.crop(box) for frame in frames]


def turn(layer, pivot, degrees):
    """Rotate a layer about a point in the image, keeping the canvas the same size."""
    x, y = pivot
    # PIL rotates about the centre, so the layer is shifted so the pivot IS the centre, turned,
    # and shifted back. Doing it in one resample keeps the arm's edges clean.
    width, height = layer.size
    shifted = layer.transform(
        (width, height), Image.AFFINE, (1, 0, x - width / 2, 0, 1, y - height / 2),
        resample=Image.BICUBIC)
    rotated = shifted.rotate(degrees, resample=Image.BICUBIC, center=(width / 2, height / 2))
    return rotated.transform(
        (width, height), Image.AFFINE, (1, 0, width / 2 - x, 0, 1, height / 2 - y),
        resample=Image.BICUBIC)


def ease(t):
    """Slow at both ends. A lift decelerates into the top and accelerates out of the bottom, and
    a linear sweep reads as a machine rather than as somebody lifting something."""
    return 0.5 - 0.5 * math.cos(math.pi * t)


def build(exercise, swing_from, swing_to, joint="elbow", frames=FRAMES, equipment=False):
    drawn = Image.open(FIGURE / "front.png").convert("RGBA")
    mask = np.array(drawn)[..., 3] > 128
    marks = landmarks(mask)

    # Room for the arms to swing into. A raised arm reaches roughly its own length out from the
    # shoulder, and the figure was cut out tight to its own silhouette, so without this the
    # hands leave the canvas and a lateral raise ends in two stumps. Padded generously and
    # trimmed back at the end to whatever the frames actually used.
    reach = int(marks["hand_y"] - marks["shoulder_y.left"]) + 24
    if equipment:
        drawn = attach_dumbbells(drawn, marks)
        mask = np.array(drawn)[..., 3] > 128
        marks = landmarks(mask)

    base = Image.new("RGBA", (drawn.width + 2 * reach, drawn.height + reach), (0, 0, 0, 0))
    base.alpha_composite(drawn, (reach, reach))
    marks = shift(marks, reach, reach)
    mask = np.array(base)[..., 3] > 128

    layers = {}
    body = base
    for side in ("left", "right"):
        limb, body = cut_arm(body, marks, side, joint)
        layers[side] = limb

    out = []
    for index in range(frames):
        t = ease(index / (frames - 1))
        reached = swing_from + (swing_to - swing_from) * t
        frame = body.copy()
        for side in ("left", "right"):
            pivot = ((marks[f"shoulder_x.{side}"], marks[f"shoulder_y.{side}"])
                     if joint == "shoulder"
                     else (marks.get(f"elbow_x.{side}", base.width / 2), marks["elbow_y"]))
            # Mirrored, and the sign was wrong the first time: PIL turns the picture one way,
            # which moves the content the other, so a lateral raise swept both arms IN across
            # the chest instead of out to the sides. `side` here is the side of the PICTURE the
            # arm sits on, not the character's own left and right.
            sign = -1.0 if side == "left" else 1.0
            # Measured from where the arm already is, not from straight down.
            angle = reached - (rest_angle(marks, side) if joint == "shoulder" else 0.0)
            frame.alpha_composite(turn(layers[side], pivot, angle * sign))
        out.append(frame)
    return trim(out), marks


def write_sheet(frames, path, scale=0.5):
    """All frames in one tall PNG. One file, one decode, and the app crops to a row — far cheaper
    than sixteen images decoded separately on a screen that also carries a 3D backdrop."""
    width = int(frames[0].width * scale)
    height = int(frames[0].height * scale)
    sheet = Image.new("RGBA", (width, height * len(frames)), (0, 0, 0, 0))
    for index, frame in enumerate(frames):
        sheet.alpha_composite(frame.resize((width, height), Image.LANCZOS), (0, index * height))
    sheet.save(path)
    return width, height, len(frames)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exercise", default="Dumbbell Bicep Curl (standing, both arms)")
    parser.add_argument("--joint", default="elbow", choices=("elbow", "shoulder"),
                        help="which hinge the limb turns about")
    parser.add_argument("--from-degrees", type=float, default=0.0)
    parser.add_argument("--to-degrees", type=float, default=140.0)
    parser.add_argument("--frames", type=int, default=FRAMES)
    parser.add_argument("--dumbbells", action="store_true",
                        help="put a dumbbell in each hand, borrowed from the curl drawing")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    frames, marks = build(args.exercise, args.from_degrees, args.to_degrees,
                          args.joint, args.frames, equipment=args.dumbbells)
    name = f"{slug(args.exercise)}-anim.png"
    path = Path(args.out) if args.out else OUT / name
    width, height, count = write_sheet(frames, path)
    print(f"{path}  {count} frames of {width}x{height}")
    print(f"elbow pivot at y={marks['elbow_y']:.0f}, "
          f"x={marks.get('elbow_x.left', 0):.0f}/{marks.get('elbow_x.right', 0):.0f}")


if __name__ == "__main__":
    main()
