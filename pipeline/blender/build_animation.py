#!/usr/bin/env python
"""Pose a rigged character from the form notes and keyframe the rep. Runs inside Blender.

    blender character.blend --background --python pipeline/blender/build_animation.py -- \
        --exercise "Barbell Back Squat" --render /tmp/squat.mp4

    # see what it would do to the rig, touching nothing:
    blender character.blend --background --python pipeline/blender/build_animation.py -- \
        --exercise "Barbell Back Squat" --dry-run

This is the last of three layers and the only one that knows what a bone is:

    reference/exercises/poses.json   what the source says, in anatomical terms
    pipeline/pose_model.py          anatomy -> a skeleton of joint directions, no Blender
    pipeline/blender/this file      those directions -> bone rotations, keyframes, a render

Why the middle layer exists at all: Blender cannot be driven to a screenshot in the environment
this was written in, so anything only checkable by rendering is unverifiable here. `pose_model`
is plain Python and `pose_preview` draws its output as a stick figure, which is how the hip,
knee, elbow and shoulder rotations were found to be inverted before any of this ran. Whatever is
wrong below is a mapping bug, not a biomechanics bug, because the biomechanics were checked
somewhere a picture could be looked at.

## What this does NOT do

It does not invent motion. Two positions come from the source and the frames between them are
interpolation — a lift is a reversal along one path, so easing between documented ends is a fair
account of it where inventing a mid-rep shape would not be. Where the source names more
keyframes than two, as the lat pulldown and the walking lunge do, they are taken: see
`KEYFRAMES_IN_SOURCE`.

It does not know your rig's proportions. `pose_model.PROPORTIONS` places joints for the preview;
here only the DIRECTIONS matter, and they are applied to whatever bone lengths the rig has.
"""

import argparse
import json
import math
import sys
from pathlib import Path

try:
    import bpy
    from mathutils import Vector, Quaternion
    INSIDE_BLENDER = True
except ImportError:  # importable outside Blender so the mapping can be unit-tested
    bpy = None
    INSIDE_BLENDER = False

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import pose_model  # noqa: E402


# --- the rig ------------------------------------------------------------------------------------
#
# One table, because the rig is the one thing here that varies between people. Mixamo is the
# default: it is the free auto-rigger most likely to be used on a character like ours, and its
# bone names are fixed. Rigify is the other common answer and is given below it.
#
# The left/right suffixes in `pose_model` are the CHARACTER's left and right, which is also what
# Mixamo means by Left and Right, so no mirroring is needed here. Getting that wrong swaps every
# asymmetric movement, so it is asserted in `check_rig` rather than assumed.

MIXAMO = {
    "spine": "mixamorig:Spine",
    "chest": "mixamorig:Spine2",
    "neck": "mixamorig:Neck",
    "head": "mixamorig:Head",
    "upper_arm.left": "mixamorig:LeftArm",
    "forearm.left": "mixamorig:LeftForeArm",
    "hand.left": "mixamorig:LeftHand",
    "upper_arm.right": "mixamorig:RightArm",
    "forearm.right": "mixamorig:RightForeArm",
    "hand.right": "mixamorig:RightHand",
    "thigh.left": "mixamorig:LeftUpLeg",
    "shin.left": "mixamorig:LeftLeg",
    "foot.left": "mixamorig:LeftFoot",
    "thigh.right": "mixamorig:RightUpLeg",
    "shin.right": "mixamorig:RightLeg",
    "foot.right": "mixamorig:RightFoot",
}

RIGIFY = {
    "spine": "spine_fk.001", "chest": "spine_fk.002", "neck": "neck", "head": "head",
    "upper_arm.left": "upper_arm_fk.L", "forearm.left": "forearm_fk.L", "hand.left": "hand_fk.L",
    "upper_arm.right": "upper_arm_fk.R", "forearm.right": "forearm_fk.R", "hand.right": "hand_fk.R",
    "thigh.left": "thigh_fk.L", "shin.left": "shin_fk.L", "foot.left": "foot_fk.L",
    "thigh.right": "thigh_fk.R", "shin.right": "shin_fk.R", "foot.right": "foot_fk.R",
}

RIGS = {"mixamo": MIXAMO, "rigify": RIGIFY}

# Which skeleton bone each rig bone's direction comes from. `pose_model` names joints by their
# upper end; a rig names bones by the segment. This is the join between the two vocabularies.
SEGMENTS = {
    "spine": ("pelvis", "chest"),
    "chest": ("chest", "neck"),
    "neck": ("neck", "head"),
    "upper_arm.left": ("shoulder.left", "elbow.left"),
    "forearm.left": ("elbow.left", "wrist.left"),
    "hand.left": ("wrist.left", "hand.left"),
    "upper_arm.right": ("shoulder.right", "elbow.right"),
    "forearm.right": ("elbow.right", "wrist.right"),
    "hand.right": ("wrist.right", "hand.right"),
    "thigh.left": ("hip.left", "knee.left"),
    "shin.left": ("knee.left", "ankle.left"),
    "foot.left": ("ankle.left", "foot.left"),
    "thigh.right": ("hip.right", "knee.right"),
    "shin.right": ("knee.right", "ankle.right"),
    "foot.right": ("ankle.right", "foot.right"),
}

# Movements whose own notes name more than two positions. Taking them is not inventing motion —
# it is reading what is written. Everything else gets start and finish and an eased interpolation
# between, which for a lift that reverses along one path is a fair account of the rep.
KEYFRAMES_IN_SOURCE = {
    "Lat Pulldown (seated, to the front)":
        "arms overhead with scapulae set, scapular depression initiating the pull, elbows "
        "driving down to the chest-level finish, then the controlled return",
    "Walking Lunge (dumbbells optional)":
        "tall stance, step-and-touchdown, 90/90 bottom, and drive-up with the rear leg swinging "
        "through, plus a visible alternation of lead leg between reps",
    "Front Plank (forearm plank)":
        "the moving frames are the lift-off and the controlled lower; the hold itself animates "
        "only through breathing and small postural corrections",
}

FPS = 30
# A rep that takes about two and a half seconds each way: slower than a working set on purpose,
# because this is a demonstration and the viewer is reading it, not keeping time with it.
HALF_REP_FRAMES = 36
HOLD_FRAMES = 8


# --- mapping ---------------------------------------------------------------------------------

def rotation_between(rest, target):
    """The shortest rotation taking one direction to another, as (axis, degrees).

    Kept free of mathutils so it can be reasoned about and tested outside Blender. Returns
    (None, 0) for directions already in line, and for the antiparallel case picks any
    perpendicular axis — a bone folded exactly back on itself has no unique shortest rotation,
    and this at least makes the choice visible rather than producing a NaN.
    """
    def norm(v):
        length = math.sqrt(sum(c * c for c in v))
        return (v[0] / length, v[1] / length, v[2] / length) if length else v

    a, b = norm(rest), norm(target)
    dot = max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b))))
    angle = math.degrees(math.acos(dot))
    if angle < 1e-6:
        return None, 0.0
    axis = (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])
    if math.sqrt(sum(c * c for c in axis)) < 1e-6:
        perpendicular = (1.0, 0.0, 0.0) if abs(a[0]) < 0.9 else (0.0, 1.0, 0.0)
        axis = (a[1] * perpendicular[2] - a[2] * perpendicular[1],
                a[2] * perpendicular[0] - a[0] * perpendicular[2],
                a[0] * perpendicular[1] - a[1] * perpendicular[0])
    return norm(axis), angle


def directions(skeleton):
    """Each rig-facing segment as a unit direction in world space."""
    out = {}
    for segment, (start, end) in SEGMENTS.items():
        if start not in skeleton.joints or end not in skeleton.joints:
            continue
        a, b = skeleton.at(start), skeleton.at(end)
        vector = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
        length = math.sqrt(sum(c * c for c in vector))
        if length < 1e-9:
            continue
        out[segment] = (vector[0] / length, vector[1] / length, vector[2] / length)
    return out


# --- Blender ----------------------------------------------------------------------------------

def check_rig(armature, mapping):
    """Refuse to pose a rig that does not have the bones this expects.

    Loudly, and before touching anything. A half-applied pose on somebody's character file is
    worse than no pose, and a silently skipped bone is a limb left standing while the rest of the
    body squats.
    """
    missing = [name for key, name in mapping.items() if name not in armature.pose.bones]
    if missing:
        raise SystemExit(
            "This rig is missing bones the mapping needs:\n  " + "\n  ".join(missing)
            + "\n\nPick another with --rig (" + ", ".join(RIGS) + "), or edit the table at the "
              "top of this file — it is one dict and it is the only rig-specific thing here.")


def apply_pose(armature, skeleton, mapping, frame=None):
    """Rotate every mapped bone to match the solved skeleton, and optionally key it."""
    for segment, direction in directions(skeleton).items():
        bone_name = mapping.get(segment)
        if not bone_name:
            continue
        bone = armature.pose.bones[bone_name]
        # The bone's rest direction in world space, which is what the target is measured against.
        rest = (armature.matrix_world @ bone.bone.matrix_local).to_3x3() @ Vector((0.0, 1.0, 0.0))
        axis, angle = rotation_between(tuple(rest.normalized()), direction)
        bone.rotation_mode = "QUATERNION"
        if axis is None:
            bone.rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))
        else:
            world = Quaternion(Vector(axis), math.radians(angle))
            # Into the bone's own space, so a parent that has already moved is not counted twice.
            parent = bone.parent.matrix.to_quaternion() if bone.parent else Quaternion()
            bone.rotation_quaternion = parent.inverted() @ world @ parent
        if frame is not None:
            bone.keyframe_insert("rotation_quaternion", frame=frame)


def build(exercise, data, armature, mapping, dry_run=False):
    phases = pose_model.phases_of(data, exercise)
    order = [p for p in ("start", "finish") if p in phases]
    if len(order) < 2:
        raise SystemExit(f"{exercise!r} has only {len(order)} documented position(s); there is "
                         "nothing to animate between.")
    if exercise in KEYFRAMES_IN_SOURCE:
        print(f"NOTE: the source names more keyframes for this movement than the two being used "
              f"here —\n      {KEYFRAMES_IN_SOURCE[exercise]}\n      The extra positions are not "
              f"in poses.json yet, so this is start-to-finish only.")

    # start -> finish -> start, so the clip loops as a rep rather than snapping back.
    timeline = [(0, "start"), (HALF_REP_FRAMES, "finish"),
                (HALF_REP_FRAMES + HOLD_FRAMES, "finish"),
                (2 * HALF_REP_FRAMES + HOLD_FRAMES, "start")]

    for frame, phase in timeline:
        skeleton = pose_model.solve(phases[phase])
        if dry_run:
            found = directions(skeleton)
            print(f"\nframe {frame:3} · {phase} · {phases[phase]['orientation']} · "
                  f"{len(phases[phase]['angles'])} angles from the source")
            for segment in sorted(found):
                x, y, z = found[segment]
                print(f"    {mapping.get(segment, segment):26} -> ({x:+.3f}, {y:+.3f}, {z:+.3f})")
            continue
        apply_pose(armature, skeleton, mapping, frame=frame)

    if not dry_run:
        scene = bpy.context.scene
        scene.frame_start, scene.frame_end = 0, timeline[-1][0]
        scene.render.fps = FPS
        for curve in armature.animation_data.action.fcurves:
            for point in curve.keyframe_points:
                # Eased, not linear: a lift accelerates out of the bottom and decelerates into
                # the top, and linear interpolation reads as a machine moving a mannequin.
                point.interpolation = "BEZIER"
                point.easing = "EASE_IN_OUT"
    return timeline[-1][0]


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exercise", required=True)
    parser.add_argument("--rig", default="mixamo", choices=sorted(RIGS))
    parser.add_argument("--armature", default=None, help="object name; the first armature by default")
    parser.add_argument("--poses", default=str(pose_model.POSES))
    parser.add_argument("--dry-run", action="store_true",
                        help="print the bone directions and change nothing; works outside Blender")
    parser.add_argument("--render", default=None, metavar="PATH")
    args = parser.parse_args(argv)

    data = pose_model.load(args.poses)
    mapping = RIGS[args.rig]

    if args.dry_run and not INSIDE_BLENDER:
        build(args.exercise, data, None, mapping, dry_run=True)
        return
    if not INSIDE_BLENDER:
        raise SystemExit("Run this inside Blender:\n  blender character.blend --background "
                         "--python " + __file__ + " -- --exercise \"...\"\n"
                         "Or pass --dry-run to see the bone directions without it.")

    armatures = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    if args.armature:
        armatures = [o for o in armatures if o.name == args.armature]
    if not armatures:
        raise SystemExit("No armature in this file.")
    armature = armatures[0]
    check_rig(armature, mapping)

    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="POSE")
    last = build(args.exercise, data, armature, mapping, dry_run=args.dry_run)
    print(f"{args.exercise}: keyframed 0..{last} at {FPS}fps")

    if args.render:
        scene = bpy.context.scene
        scene.render.filepath = args.render
        scene.render.image_settings.file_format = "FFMPEG"
        scene.render.ffmpeg.format = "MPEG4"
        bpy.ops.render.render(animation=True)
        print("rendered to", args.render)


if __name__ == "__main__":
    main()
