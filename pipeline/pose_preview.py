#!/usr/bin/env python
"""Draw what `pose_model.solve` produced, so a wrong angle is seen rather than shipped.

This exists because of how the rest of this project has gone. Generated exercise pictures were
judged by eye and ten of thirteen were wrong; the muscle shapes looked fine as numbers and came
out as rectangles; the first arm geometry ran a forearm highlight down a thigh. Every one of
those was caught by rendering it and looking. Joint angles are worse than all of them, because a
sign error produces a figure that is perfectly plausible and backwards.

So: before Blender, before a mesh, before a rig, the angles get drawn as a stick figure from the
side and the front. A knee that bends the wrong way is obvious in a 200-pixel sketch and
invisible in a JSON file.

    pipeline/.venv/bin/python pipeline/pose_preview.py --exercise "Barbell Back Squat"
    pipeline/.venv/bin/python pipeline/pose_preview.py --all --out /tmp/poses.png
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

import pose_model

# Which pairs of joints to draw as a limb, and how heavily. The spine is drawn thickest because
# its angle is the one that decides whether a lift reads as a squat or a deadlift.
LIMBS = [
    ("pelvis", "chest", 7), ("chest", "neck", 5), ("neck", "head", 4),
    ("chest", "shoulder.left", 4), ("chest", "shoulder.right", 4),
    ("shoulder.left", "elbow.left", 5), ("elbow.left", "wrist.left", 4),
    ("wrist.left", "hand.left", 3),
    ("shoulder.right", "elbow.right", 5), ("elbow.right", "wrist.right", 4),
    ("wrist.right", "hand.right", 3),
    ("pelvis", "hip.left", 4), ("pelvis", "hip.right", 4),
    ("hip.left", "knee.left", 6), ("knee.left", "ankle.left", 5), ("ankle.left", "foot.left", 3),
    ("hip.right", "knee.right", 6), ("knee.right", "ankle.right", 5), ("ankle.right", "foot.right", 3),
]

# The right side is drawn darker so a figure seen from the side reads as a body and not as a
# tangle: without it, two legs in profile are one thick line.
NEAR = (30, 34, 40)
FAR = (150, 158, 170)
GROUND = (205, 210, 218)


def project(point, view):
    """Flatten to the drawing plane. `side` looks along -X at the character's right, so +Y
    (forward, the way the character faces) goes to the right of the picture."""
    x, y, z = point
    return (y, z) if view == "side" else (x, z)


def draw(skeleton, view, size=(300, 380), title=""):
    width, height = size
    image = Image.new("RGB", size, (255, 255, 255))
    pen = ImageDraw.Draw(image)

    points = {name: project(joint.position, view) for name, joint in skeleton.joints.items()}
    xs = [p[0] for p in points.values()]
    zs = [p[1] for p in points.values()]
    # Always include the floor and a full standing height in the extent, so poses can be compared
    # between phases — a squat that autoscaled to fill the frame would look the same as standing.
    lo_x, hi_x = min(xs + [-0.5]), max(xs + [0.5])
    lo_z, hi_z = min(zs + [0.0]), max(zs + [1.05])
    pad = 26
    span_x, span_z = hi_x - lo_x, hi_z - lo_z
    scale = min((width - 2 * pad) / span_x, (height - 2 * pad) / span_z)

    def place(point):
        x, z = point
        return (pad + (x - lo_x) * scale, height - pad - (z - lo_z) * scale)

    floor = height - pad - (0 - lo_z) * scale
    pen.line([(0, floor), (width, floor)], fill=GROUND, width=2)

    for parent, child, weight in LIMBS:
        if parent not in points or child not in points:
            continue
        far = view == "side" and parent.endswith(".left") or child.endswith(".left")
        pen.line([place(points[parent]), place(points[child])],
                 fill=FAR if far else NEAR, width=max(1, weight - (2 if far else 0)))

    for name in ("head",):
        if name in points:
            x, y = place(points[name])
            radius = 0.07 * scale
            pen.ellipse([x - radius, y - radius, x + radius, y + radius], fill=NEAR)

    if title:
        pen.text((6, 5), title, fill=(60, 64, 72))
    return image


def sheet(entries, out, size=(300, 380)):
    """A grid: one row per exercise, start and finish side by side, each from two views."""
    cols = 4
    tile_w, tile_h = size
    image = Image.new("RGB", (cols * tile_w, len(entries) * (tile_h + 18)), (255, 255, 255))
    pen = ImageDraw.Draw(image)
    for row, (name, phases) in enumerate(entries):
        y = row * (tile_h + 18)
        pen.text((6, y + 4), name, fill=(0, 0, 0))
        for column, (phase, view) in enumerate([("start", "side"), ("start", "front"),
                                                ("finish", "side"), ("finish", "front")]):
            if phase not in phases:
                continue
            try:
                skeleton = pose_model.solve(phases[phase])
            except ValueError as problem:
                pen.text((column * tile_w + 8, y + 24), f"{phase}: {problem}"[:46],
                         fill=(190, 40, 40))
                continue
            image.paste(draw(skeleton, view, size, f"{phase} · {view}"), (column * tile_w, y + 16))
    image.save(out)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exercise", action="append")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--out", default="/tmp/poses.png")
    args = parser.parse_args()

    data = pose_model.load()
    names = [e["name"] for e in data["exercises"]]
    wanted = names if args.all or not args.exercise else args.exercise
    entries = [(name, pose_model.phases_of(data, name)) for name in wanted]
    sheet(entries, args.out)
    print(f"{args.out}  {len(entries)} exercise(s)")


if __name__ == "__main__":
    main()
