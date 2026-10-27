#!/usr/bin/env python
"""Drive the house rig from captured human motion — a Mixamo FBX. Runs inside Blender.

    blender --background --python pipeline/blender/animate.py -- \
        --exercise "Barbell Back Squat" --mixamo 3d-character/mixamo/squat.fbx \
        --wrap-skin "3d-character/3d cartoon boy.glb" --fast --out /tmp/squat --gif /tmp/squat.gif

`animate.py --mixamo` is the entry point; this module is what it calls. It replaces the
two-keyframe `pose_model` timeline (start, finish and a Bezier ease between them — the motion the
user called stiff) with a real performance, one keyframe per captured frame.

## The method: copy directions, keep our bones

Nothing here rotates a bone by a Mixamo bone's rotation. Each frame the source skeleton's joints
are read as WORLD positions, every one of our bones is given the direction its source segment
points in, and `animate.aim` swings it there — the same shortest-arc swing `animate.pose` uses, so
roll is carried over from the previous frame rather than invented, and every bone keeps its own
length. The source's proportions therefore never reach our character: a Mixamo Y Bot is 1.8 m of
long shins, our boy is a cartoon with a thigh 1.76 times his shin, and only angles cross over.

Three things are not plain direction copies, and each is here because a plain copy was wrong:

  * **Facing is measured, not assumed.** Whatever axes the importer leaves the source in, the
    character's own left (its `Left*` bones) crossed with its own up gives its forward, and its
    toes are asked whether they agree. If they point the other way the skeleton is MIRRORED — a
    left side that is physically on the right — and the clip is refused: a mirrored retarget looks
    perfectly plausible in a symmetric squat and swaps every asymmetric movement.
  * **The feet are copied as a whole rotation, relative to their rest pose.** A Mixamo ankle sits
    10 cm above the floor with the ball of the foot on it, so the ankle-to-toe line slopes 40
    degrees down; ours slopes 22. Copied as an absolute direction, a flat foot arrives tipped 20
    degrees onto its toes and the grounding stands the figure on tiptoe; copied as a direction at
    all, the foot inherits a ROLL from the shin above it, which on the real squat slid the heels
    16 mm while every joint stood still. So the source foot's change of rotation from its rest is
    applied to our foot's own neutral: flat stays flat, and a planted foot does not turn.
  * **The root's roll.** One bone (`spine`) carries both the pelvis and the hip joints, and a swing
    never turns a bone about its own length — so a figure that turns to its left would keep its
    hips square to the camera while its legs went elsewhere. The spine is turned about its own axis
    until the hip line matches the source's.

## Where the figure stands

Every frame is grounded with `animate.ground_rig` — the same rule set `pose_model.ground` uses for
each orientation (`RESTS_ON`, `PINNED_BY`, the hanging bar, the seat height `pelvis_at`, and the
mesh clamp that stands the SOLE on the floor rather than the ankle joint). Only its vertical answer
is kept: horizontally the pelvis follows the source's own path, scaled by our leg length over
theirs, because `ground` recentres a standing figure over its feet in every frame and would erase a
lunge's travel.

Then the planted feet are LOCKED. Copying directions onto legs of different proportions moves the
feet: our ankle is wherever our thigh and shin put it, not where Mixamo's was, and as the knee bends
the difference changes — the feet skate (74 mm on the real squat with the lock off; task 9 measured
the two-keyframe pipeline sinking the feet 55 mm and sliding them 9.4 cm). So while a foot is
planted its toe is pinned where it touched down, in two steps. First the whole figure is
translated so the planted toes are back on their marks — no joint angle changes, the pelvis simply
ends up further back than the source's. Then whatever one translation cannot fix (two feet that
disagree, as knees-out on a long thigh spreads them; the vertical) is taken up by re-solving thigh
and shin with an analytic two-bone IK, the knee kept in the plane the source's knee chose. The
pelvis moves; the foot does not. Which feet are planted is per exercise (`EXERCISES`): always,
never, or detected from the source's foot height and speed for the one exercise that steps.

## What is checked, every run

`contact_metrics` replays the keyed action frame by frame and measures, on the skin the viewer
sees, how far each planted sole travels and how far anything goes below the floor. `form_check`
finds the frames closest to the sourced start and finish poses and puts the knee, hip, elbow,
shoulder and torso angles beside what `pose_model` solves from `poses.json`, flagging anything more
than `FORM_TOLERANCE` off. It reports; it corrects nothing.
"""

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import bpy
from mathutils import Matrix, Quaternion, Vector

import pose_model
import make_character
import exercise_stage
import animate


# --- the source skeleton ---------------------------------------------------------------------------
#
# Our bones, and the two source points each one is aimed along. Points are Mixamo bone HEADS, named
# by suffix so `mixamorig:`, `mixamorig1:`, `mixamorig2:` and no prefix all match (see `_suffix`).
# "hips" is the midpoint of the two thigh heads: our `pelvis` joint is the hip-joint midpoint
# (`pose_model` puts it at hip height and `hip.*` beside it), where Mixamo's `Hips` bone sits 6 cm
# higher, so this is the point that matches the head of our `spine` bone.
#
# Written once for the left side and mirrored by name below. The names on both sides are the
# CHARACTER's own left and right — Mixamo's `LeftArm` is the arm on the character's left, and so is
# our `upper_arm.left` — so no side is swapped here. The asymmetric test proves it rather than this
# comment (`retarget_check.py`).
_LEFT = {
    "clavicle.left": ("LeftShoulder", "LeftArm"),
    "upper_arm.left": ("LeftArm", "LeftForeArm"),
    "forearm.left": ("LeftForeArm", "LeftHand"),
    "hand.left": ("LeftHand", "LeftHandMiddle1"),
    "thigh.left": ("LeftUpLeg", "LeftLeg"),
    "shin.left": ("LeftLeg", "LeftFoot"),
    "foot.left": ("LeftFoot", "LeftToeBase"),
}
FROM_SOURCE = {
    "spine": ("hips", "Neck"),
    "neck": ("Neck", "Head"),
    "head": ("Head", "HeadTop_End"),
    **_LEFT,
    **{bone.replace("left", "right"): (a.replace("Left", "Right"), b.replace("Left", "Right"))
       for bone, (a, b) in _LEFT.items()},
}

# Copied as the change from the source's rest pose rather than as an absolute direction — see the
# module docstring. Everything else is absolute, which is what the existing pipeline does with the
# solved skeleton too, so a retargeted torso leans exactly as far from vertical as the source's.
# The feet go further than a direction: their whole rotation is carried over (see `TURNED`).
RELATIVE = {"foot.left", "foot.right"}

# Points a clip must have. The three optional ones fall back to a bone's TAIL, which on an imported
# skeleton is wherever the importer chose to point it — good enough for a head or a hand, and the
# fallback is printed.
REQUIRED = sorted({p for pair in FROM_SOURCE.values() for p in pair}
                  - {"hips", "HeadTop_End", "LeftHandMiddle1", "RightHandMiddle1"}
                  | {"LeftUpLeg", "RightUpLeg"})
FALLBACK_TO_TAIL = {"HeadTop_End": "Head", "LeftHandMiddle1": "LeftHand",
                    "RightHandMiddle1": "RightHand"}


def _suffix(name):
    """The bone's own name with any rig prefix taken off: `mixamorig1:LeftArm` -> `LeftArm`.

    Mixamo writes `mixamorig:`, and `mixamorig1:`, `mixamorig2:` when one file holds several
    characters; some tools rewrite the colon as an underscore or drop it. Everything up to the last
    colon goes, and a leading `mixamorig<n>` with an optional `_` goes too.
    """
    name = name.rsplit(":", 1)[-1]
    return re.sub(r"^mixamorig\d*_?", "", name)


# --- what each exercise asks of the retarget ------------------------------------------------------
#
# `feet`:  "planted" — both feet locked for the whole clip (every exercise whose feet do not move);
#          "auto"    — locked only while the SOURCE foot is down and still (the walking lunge steps);
#          "free"    — never locked (a hanging pull-up; the leg press, whose feet ride the sled).
#          An exercise not in this table is "planted"; a clip run with no exercise is "auto".
# `hold`:  body regions taken from the SOURCED pose instead of the capture, as
#          {region: (phase, frame)}. Mixamo's gym moves are equipment-free — an air squat holds its
#          arms out in front — so the back squat keeps the legs, hips and spine from the capture and
#          holds its arms where `pose_model` puts them for the "start" phase: hands on the bar behind
#          the neck. `frame` says what the held pose is relative to:
#            "torso" — expressed in the chest's own frame and carried with it, so a bar on the back
#                      stays on the back as the torso hinges forward;
#            "world" — the solved world direction, kept as is: a deadlift's arms hang vertically
#                      from the shoulders at every height of the lift.
#          Decided per exercise once its clip has been SEEN, not in advance. The brief expected the
#          back squat to need its arms held, and Mixamo's "Back Squat" turned out to hold its own
#          grip: hands up beside the shoulders, elbows down, which puts the hands at the bar. The
#          sourced "start" arms, held in the torso frame, put them in front of the chest, short of
#          the bar (`pose_model` cannot reach a bar-width grip — its own note says so). So the
#          squat keeps the capture's arms; `--no-holds` / `holds=` override either way, and an
#          air-squat clip (arms out in front) would want ("start", "torso") back.
EXERCISES = {
    "Barbell Back Squat": {"feet": "planted"},
    "Conventional Barbell Deadlift": {"feet": "planted", "hold": {"arms": ("start", "world")}},
    "Walking Lunge (dumbbells optional)": {"feet": "auto"},
    "Pull-Up": {"feet": "free"},
    "Leg Press (seated machine)": {"feet": "free"},
}

REGIONS = {
    "arms": ["clavicle.left", "upper_arm.left", "forearm.left", "hand.left",
             "clavicle.right", "upper_arm.right", "forearm.right", "hand.right"],
    "arm.left": ["clavicle.left", "upper_arm.left", "forearm.left", "hand.left"],
    "arm.right": ["clavicle.right", "upper_arm.right", "forearm.right", "hand.right"],
    "legs": ["thigh.left", "shin.left", "foot.left", "thigh.right", "shin.right", "foot.right"],
    "head": ["neck", "head"],
}

# Contact detection for "auto", in units of OUR leg length (thigh + shin) and seconds. A planted
# ball of the foot does not rise or travel; one in swing does both within a couple of frames.
CONTACT_HEIGHT = 0.10        # ball of the foot within this of the clip's own floor
CONTACT_SPEED = 0.60         # and moving slower than this, horizontally
CONTACT_MIN_RUN = 3          # frames: shorter runs of contact or of swing are noise, not steps
RELEASE_FRAMES = 4           # a lock fades out over this many frames after lift-off, not snaps

FORM_TOLERANCE = 10.0        # degrees off the sourced angle before `form_check` flags it


# --- reading a clip --------------------------------------------------------------------------------

@dataclass
class Clip:
    """A captured performance as plain numbers: no Blender objects survive `load_clip`."""
    path: str
    fps: float
    first: int                                  # the source frame number of frames[0]
    rest: dict                                  # point -> Vector, the armature's rest pose, world
    frames: list                                # [{point: Vector}], world, one per source frame
    bones: dict = field(default_factory=dict)   # point -> the full bone name it was read from
    notes: list = field(default_factory=list)
    rest_turns: dict = field(default_factory=dict)   # bone -> world rotation at rest (Quaternion)
    turns: list = field(default_factory=list)        # [{bone: world rotation}], one per frame


# Source bones whose FULL rotation is read, not just where their ends are. A direction says
# nothing about a bone's roll about its own length, and for a foot the roll is the sole: the first
# real squat kept every toe and ankle joint within 0.1 mm and still slid the heels 16 mm, because
# the shin's pitch at the bottom was handed down to the foot as a twist that a shortest-arc swing
# never takes back. Read whole, a planted source foot hands over "no rotation at all".
TURNED = ("LeftFoot", "RightFoot")


def _pick_armature(objects):
    armatures = [o for o in objects if o.type == "ARMATURE"]
    if not armatures:
        raise SystemExit("no armature in the imported file")
    return max(armatures, key=lambda o: sum(_suffix(b.name) in REQUIRED for b in o.data.bones))


def sample(armature, label="", turned=TURNED):
    """Every point `FROM_SOURCE` needs, in world space, at the rest pose and at every frame.

    World space via `matrix_world @ pose_bone.head` makes the importer's axis conversion and its
    unit scale (Mixamo FBX arrives at 0.01 or 100x depending on the settings) irrelevant here:
    whatever the file did, these are where the joints are. Facing and scale are measured from
    these numbers afterwards, never assumed.
    """
    found = {}
    for bone in armature.data.bones:
        key = _suffix(bone.name)
        if key in FALLBACK_TO_TAIL or key in REQUIRED:
            found.setdefault(key, bone.name)
    missing = [p for p in REQUIRED if p not in found]
    if missing:
        raise SystemExit(f"{label}: the skeleton has no bone for {missing}; names seen: "
                         f"{sorted(_suffix(b.name) for b in armature.data.bones)[:60]}")
    notes = []
    tails = {}
    for point, owner in FALLBACK_TO_TAIL.items():
        if point not in found:
            tails[point] = found[owner]
            notes.append(f"no {point} bone: using the tail of {found[owner]}")

    world = armature.matrix_world
    rest = {p: world @ armature.data.bones[name].head_local for p, name in found.items()}
    for point, owner in tails.items():
        rest[point] = world @ armature.data.bones[owner].tail_local
    turn = lambda matrix: matrix.to_3x3().normalized().to_quaternion()      # noqa: E731
    # `turned="all"`: every bone's whole rotation, by suffix — what a bone-for-bone retarget onto
    # a rig with Mixamo's own bones needs (`BoneRetarget`).
    owners = ({_suffix(b.name): b.name for b in armature.data.bones} if turned == "all"
              else {p: found[p] for p in turned})
    rest_turns = {p: turn(world @ armature.data.bones[n].matrix_local) for p, n in owners.items()}
    turns = []

    action = armature.animation_data.action if armature.animation_data else None
    scene = bpy.context.scene
    if action is None:
        first, last = scene.frame_current, scene.frame_current
        notes.append("no action on the armature: one frame, its current pose")
    else:
        start, end = action.frame_range
        first, last = int(round(start)), int(round(end))
    frames = []
    for frame in range(first, last + 1):
        scene.frame_set(frame)
        world = armature.matrix_world
        pose = armature.pose.bones
        points = {p: world @ pose[name].head for p, name in found.items()}
        for point, owner in tails.items():
            points[point] = world @ pose[owner].tail
        frames.append({p: v.copy() for p, v in points.items()})
        turns.append({p: turn(world @ pose[n].matrix) for p, n in owners.items()})
    fps = scene.render.fps / scene.render.fps_base
    names = dict(found, **{p: f"{o} (tail)" for p, o in tails.items()})
    return Clip(path=label, fps=fps, first=first, rest=rest, frames=frames, bones=names,
                notes=notes, rest_turns=rest_turns, turns=turns)


def load_clip(path, turned=TURNED):
    """Import a Mixamo FBX (or a glTF with a Mixamo skeleton), sample it, and remove it again.

    Call this BEFORE building the character: `character.build` empties the scene, and nothing this
    returns refers to a Blender object.
    """
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"{path} does not exist")
    before, actions = set(bpy.data.objects), set(bpy.data.actions)
    if path.suffix.lower() == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path), use_anim=True)
    elif path.suffix.lower() in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=str(path))
    else:
        raise SystemExit(f"{path.name}: expected an .fbx (or a .glb/.gltf with a Mixamo skeleton)")
    new = [o for o in bpy.data.objects if o not in before]
    clip = sample(_pick_armature(new), label=str(path), turned=turned)
    facing(clip.rest)        # refuses a mirrored or lying-down skeleton now, before any building
    for obj in new:
        bpy.data.objects.remove(obj, do_unlink=True)
    for action in set(bpy.data.actions) - actions:
        bpy.data.actions.remove(action)
    bpy.context.scene.frame_set(1)
    print(f"retarget: read {path.name}: {len(clip.frames)} frames at {clip.fps:g} fps "
          f"(source frames {clip.first}..{clip.first + len(clip.frames) - 1})"
          + "".join(f"\n  note: {n}" for n in clip.notes))
    return clip


def _point(points, name):
    if name == "hips":
        return (points["LeftUpLeg"] + points["RightUpLeg"]) / 2
    return points[name]


def _flat(vector, up):
    return vector - up * vector.dot(up)


_AXES = [Vector(v) for v in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))]


def facing(rest):
    """The rotation taking the source's world into ours, measured off its rest pose. Returns
    (matrix, findings).

    Ours: the figure faces +Y, up is +Z, and so its own left is -X (`pose_model`'s convention,
    `exercise_stage.body_axes` derives forward the same way). The source's is read from its bones:
      up       the world axis nearest hips -> neck. Snapped to an axis on purpose, so a spine
               that leans two degrees in the rest pose does not tilt our floor by two degrees;
      left     its `Left*` thigh and arm heads minus its `Right*` ones, square to up;
      forward  left x up — which is +Y for ours, so this is the same construction on both sides;
    and then the toes, which know nothing about names, are asked which way is forward. They must
    agree with `left x up` to within a right angle, or the skeleton is mirrored and is refused.

    A proper rotation by construction (its rows are right, forward, up with right = forward x up),
    so it can turn the figure round but cannot mirror it.
    """
    hips = (rest["LeftUpLeg"] + rest["RightUpLeg"]) / 2
    rise = rest["Neck"] - hips
    up = max(_AXES, key=lambda axis: axis.dot(rise))
    tilt = math.degrees(rise.angle(up))
    if tilt > 35.0:
        raise SystemExit(f"the source's rest pose is not standing: hips -> neck is {tilt:.0f} degrees "
                         f"off the nearest world axis, so 'up' cannot be read from it")
    left = _flat((rest["LeftUpLeg"] - rest["RightUpLeg"]) + (rest["LeftArm"] - rest["RightArm"]),
                 up)
    if left.length < 1e-9:
        raise SystemExit("the source's left and right thigh/arm heads coincide")
    forward = left.normalized().cross(up).normalized()
    toes = _flat(sum((rest[f"{s}ToeBase"] - rest[f"{s}Foot"] for s in ("Left", "Right")),
                     Vector()), up)
    if toes.length < 1e-9:
        raise SystemExit("the source's feet have no forward extent to check facing against")
    disagree = math.degrees(forward.angle(toes))
    if disagree > 90.0:
        raise SystemExit(
            f"MIRRORED SKELETON: the bones named Left are on the character's right. 'left x up' "
            f"says forward is {tuple(round(c, 3) for c in forward)} but the toes point "
            f"{tuple(round(c, 3) for c in toes.normalized())} ({disagree:.0f} degrees apart). "
            f"Retargeting it would swap every asymmetric movement, so it is refused.")
    right = forward.cross(up)
    rotation = Matrix((right, forward, up))
    findings = {"source_up": tuple(up), "source_forward": tuple(round(c, 4) for c in forward),
                "source_left": tuple(round(c, 4) for c in -right),
                "rest_spine_tilt_deg": round(tilt, 2), "toes_vs_forward_deg": round(disagree, 2)}
    return rotation, findings


def _leg_length(points):
    return sum(((points[f"{s}Leg"] - points[f"{s}UpLeg"]).length +
                (points[f"{s}Foot"] - points[f"{s}Leg"]).length) for s in ("Left", "Right")) / 2


# --- the retarget ------------------------------------------------------------------------------------

def _vectors(skeleton):
    return {name: Vector(joint.position) for name, joint in skeleton.joints.items()}


def _bone_directions(joints):
    out = {}
    for bone, (head, tail) in make_character.BONES.items():
        if head in joints and tail in joints:
            d = joints[tail] - joints[head]
            if d.length > 1e-9:
                out[bone] = d.normalized()
    return out


def skeleton_from(directions, lengths, pelvis=None):
    """A `pose_model.Skeleton` whose bones point along `directions` with OUR lengths.

    It exists to be handed to `animate.aim_bones`, which aims each bone at its tail joint minus its
    head joint — so only the directions matter, and the hips (which no bone of ours runs to) are put
    on the pelvis because nothing reads their position from here.
    """
    pos = {"pelvis": Vector(pelvis) if pelvis is not None else Vector()}
    for bone, (head, tail) in make_character.BONES.items():
        if head not in pos:
            pos[head] = pos["pelvis"].copy()
        pos[tail] = pos[head] + directions[bone] * lengths[bone]
    skeleton = pose_model.Skeleton()
    for name, p in pos.items():
        skeleton.add(name, tuple(p))
    return skeleton


def _axes_matrix(joints):
    lateral, forward, up = exercise_stage.body_axes(joints)
    return Matrix((lateral, forward, up)).transposed()       # columns are the body's own axes


def _signed_angle(a, b, axis):
    return math.atan2(axis.dot(a.cross(b)), a.dot(b))


def _key_rotations(rig, frame, previous):
    """Key every bone's quaternion, flipped onto the same hemisphere as the frame before.

    `aim` sets a bone's matrix and Blender picks the sign of the quaternion it derives from it; q
    and -q are the same rotation but interpolate the long way round between each other, which a
    USDZ player sampling between frames would show as a bone spinning a full turn.
    """
    for name in make_character.BONES:
        if name not in rig.pose.bones:
            continue
        bone = rig.pose.bones[name]
        q = bone.rotation_quaternion.copy()
        if name in previous and q.dot(previous[name]) < 0.0:
            q.negate()
            bone.rotation_quaternion = q
        previous[name] = q.copy()
        bone.keyframe_insert("rotation_quaternion", frame=frame)


class Retarget:
    """One clip onto one rig. `run()` keys every frame; everything it measured is in `.result`."""

    def __init__(self, rig, body, clip, exercise=None, orientation="standing", rest_at=0.0,
                 pelvis_at=None, feet=None, foot_lock=True, holds=None, first_frame=1):
        if rig.parent is not None or rig.matrix_world.to_3x3() != Matrix.Identity(3):
            raise SystemExit("retarget expects an unrotated, unscaled, unparented rig "
                             "(directions are aimed in world space)")
        self.rig, self.body, self.clip = rig, body, clip
        self.exercise, self.orientation = exercise, orientation
        self.rest_at, self.pelvis_at = rest_at, pelvis_at
        self.first_frame = first_frame
        entry = EXERCISES.get(exercise, {}) if exercise else {}
        self.feet = feet or entry.get("feet", "planted" if exercise else "auto")
        self.foot_lock = foot_lock
        self.holds = entry.get("hold", {}) if holds is None else holds

        # Our figure as it stands before the first frame: the neutral the relative bones are
        # measured from, and the lengths the synthetic skeleton is built with. Put back first:
        # `exercise_stage.pelvis_anchor` poses the rig into a solved phase to measure a seated
        # exercise's hip height and leaves it there, and that pose is not this character's neutral.
        # A worn skin's neutral is the pose it was bound in; a bare rig's is its rest.
        if "skin_fit" in rig:
            import skin_wrap                     # lazy, as in `animate.wear_skin`
            skin_wrap.to_skin_pose(rig)
        else:
            for bone in rig.pose.bones:
                bone.rotation_mode = "QUATERNION"
                bone.rotation_quaternion = Quaternion()
        bpy.context.view_layer.update()
        neutral = exercise_stage.joints_of(rig)
        self.lengths = {bone: (neutral[tail] - neutral[head]).length
                        for bone, (head, tail) in make_character.BONES.items()}
        self.our_rest = _bone_directions(neutral)
        self.our_leg = (self.lengths["thigh.left"] + self.lengths["shin.left"] +
                        self.lengths["thigh.right"] + self.lengths["shin.right"]) / 2

        self.rotation, self.facing = facing(clip.rest)
        self.our_rest_turn = {s: (rig.matrix_world @ rig.pose.bones[f"foot.{s}"].matrix)
                              .to_3x3().normalized().to_quaternion() for s in ("left", "right")}
        self.scale = self.our_leg / _leg_length(clip.rest)
        self.src_rest = {bone: (self.rotation @ (_point(clip.rest, b) - _point(clip.rest, a)))
                         .normalized() for bone, (a, b) in FROM_SOURCE.items()}

        # The sourced poses the holds are taken from, solved once.
        self.held = {}
        if self.holds:
            phases = pose_model.phases_of(pose_model.load(), exercise)
            for region, (phase, frame) in self.holds.items():
                solved = pose_model.solve(phases[phase])
                axes = _axes_matrix(_vectors(solved))    # before `with_clavicles` moves shoulders
                directions = _bone_directions(_vectors(animate.with_clavicles(solved)))
                self.held[region] = (frame, axes, directions)

        self.held_bones = {bone for region in self.held for bone in REGIONS[region]}
        self.plan = self._plan_contacts()
        self.result = {}

    # -- per-frame pieces --

    def foot_turns(self, i):
        """Our feet's world rotations for frame `i`: the source foot's change from its rest pose,
        turned into our world, applied to our foot as it stood before the first frame."""
        if not self.clip.turns:
            return {}
        q = self.rotation.to_quaternion()
        out = {}
        for side in ("left", "right"):
            src = f"{side.title()}Foot"
            change = self.clip.turns[i][src] @ self.clip.rest_turns[src].inverted()
            out[side] = (q @ change @ q.inverted() @ self.our_rest_turn[side]).normalized()
        return out

    def turn_foot(self, side, rotation):
        """Set the foot bone's whole orientation (direction AND roll), keeping its head where it is."""
        bone = self.rig.pose.bones[f"foot.{side}"]
        matrix = bone.matrix.copy()
        turned = rotation.to_matrix().to_4x4()
        turned.translation = matrix.translation
        bone.matrix = turned
        bpy.context.view_layer.update()

    def directions(self, points):
        """Every bone's target direction in our world, from one source frame."""
        out = {}
        for bone, (a, b) in FROM_SOURCE.items():
            d = self.rotation @ (_point(points, b) - _point(points, a))
            if d.length < 1e-9:
                continue
            d.normalize()
            if bone in RELATIVE:
                d = self.src_rest[bone].rotation_difference(d) @ self.our_rest[bone]
            out[bone] = d
        return out

    def hold(self, directions):
        """Replace the held regions' directions with the sourced pose's. See `EXERCISES`."""
        if not self.held:
            return directions
        now = _axes_matrix(_vectors(skeleton_from(directions, self.lengths)))
        out = dict(directions)
        for region, (frame, axes, solved) in self.held.items():
            carry = now @ axes.transposed() if frame == "torso" else Matrix.Identity(3)
            for bone in REGIONS[region]:
                out[bone] = (carry @ solved[bone]).normalized()
        return out

    def roll_root(self, right):
        """Turn the spine about its own length until the hip line points along `right`."""
        bpy.context.view_layer.update()
        joints = exercise_stage.joints_of(self.rig)
        axis = (joints["chest"] - joints["pelvis"]).normalized()
        now = _flat(joints["hip.right"] - joints["hip.left"], axis)
        want = _flat(right, axis)
        if now.length < 1e-9 or want.length < 1e-9:
            return 0.0
        angle = _signed_angle(now.normalized(), want.normalized(), axis)
        bone = self.rig.pose.bones["spine"]
        matrix = bone.matrix.copy()
        turned = (Quaternion(axis, angle).to_matrix() @ matrix.to_3x3()).to_4x4()
        turned.translation = matrix.translation
        bone.matrix = turned
        bpy.context.view_layer.update()
        return math.degrees(angle)

    def solve_leg(self, side, ankle_target, pole, foot_turn):
        """Analytic two-bone IK: thigh and shin re-aimed so the ankle lands on `ankle_target`.

        The knee stays in the plane through the hip, the target and `pole` — the knee the direction
        copy produced — so a knees-out squat stays knees-out. Returns how far short the leg fell
        (zero unless the target is beyond a straight leg).
        """
        joints = exercise_stage.joints_of(self.rig)
        hip = joints[f"hip.{side}"]
        l1, l2 = self.lengths[f"thigh.{side}"], self.lengths[f"shin.{side}"]
        to = ankle_target - hip
        distance = to.length
        u = to.normalized()
        v = _flat(pole, u)
        if v.length < 1e-4 * l1:
            # A straight leg has no bend to keep; any knee-forward direction serves, and at a
            # straight leg it does not move the knee anyway.
            v = _flat(exercise_stage.body_axes(joints)[1], u)
        v.normalize()
        short = 0.0
        if distance >= l1 + l2:
            short = distance - (l1 + l2)
            knee = hip + u * l1
        else:
            distance = max(distance, abs(l1 - l2) + 1e-6)
            cos_a = (l1 * l1 + distance * distance - l2 * l2) / (2 * l1 * distance)
            a = math.acos(max(-1.0, min(1.0, cos_a)))
            knee = hip + (u * math.cos(a) + v * math.sin(a)) * l1
        pose = self.rig.pose.bones
        animate.aim(pose[f"thigh.{side}"], knee - hip)
        bpy.context.view_layer.update()
        animate.aim(pose[f"shin.{side}"], ankle_target - knee)
        bpy.context.view_layer.update()
        if foot_turn is not None:
            self.turn_foot(side, foot_turn)
        return short

    # -- contacts --

    def _plan_contacts(self):
        """Per frame and side: the lock's weight, and whether the foot touches down this frame."""
        n = len(self.clip.frames)
        sides = ("left", "right")
        if self.feet == "planted" or self.feet == "free":
            on = self.feet == "planted"
            down = {s: [on] * n for s in sides}
        elif self.feet == "auto":
            down = {}
            balls = {s: [self.scale * (self.rotation @ f[f"{s.title()}ToeBase"])
                         for f in self.clip.frames] for s in sides}
            floor = min(p.z for s in sides for p in balls[s])
            for s in sides:
                flags = []
                for i, p in enumerate(balls[s]):
                    a, b = balls[s][max(i - 1, 0)], balls[s][min(i + 1, n - 1)]
                    span = max(min(i + 1, n - 1) - max(i - 1, 0), 1)
                    speed = (Vector((b.x - a.x, b.y - a.y))).length * self.clip.fps / span
                    flags.append(p.z - floor < CONTACT_HEIGHT * self.our_leg and
                                 speed < CONTACT_SPEED * self.our_leg)
                down[s] = _debounce(flags, CONTACT_MIN_RUN)
        else:
            raise SystemExit(f"feet must be planted, auto or free, not {self.feet!r}")
        plan = []
        for i in range(n):
            frame = {}
            for s in sides:
                if down[s][i]:
                    weight = 1.0
                else:
                    since = next((k for k in range(1, RELEASE_FRAMES + 1)
                                  if i - k >= 0 and down[s][i - k]), None)
                    weight = 0.0 if since is None else 1.0 - since / (RELEASE_FRAMES + 1)
                touchdown = down[s][i] and (i == 0 or not down[s][i - 1])
                frame[s] = {"down": down[s][i], "weight": weight, "touchdown": touchdown}
            plan.append(frame)
        return plan

    # -- the loop --

    def run(self):
        rig, clip = self.rig, self.clip
        anchors = {"left": None, "right": None}
        previous = {}
        frames_out = []
        root_offset = None
        roll_max = 0.0
        shortfall = residual = 0.0
        for i, points in enumerate(clip.frames):
            frame = self.first_frame + i
            directions = self.hold(self.directions(points))
            turns = {side: rotation for side, rotation in self.foot_turns(i).items()
                     if f"foot.{side}" not in self.held_bones}
            for side, rotation in turns.items():
                directions[f"foot.{side}"] = rotation @ Vector((0.0, 1.0, 0.0))

            # The root: its direction, then its roll, then every other bone from it.
            animate.aim(rig.pose.bones["spine"], directions["spine"])
            right = self.rotation @ (points["RightUpLeg"] - points["LeftUpLeg"])
            roll_max = max(roll_max, abs(self.roll_root(right)))
            animate.aim_bones(rig, skeleton_from(directions, self.lengths))
            for side, rotation in turns.items():
                self.turn_foot(side, rotation)

            # Horizontally, the source's own pelvis path, scaled; anchored on frame one so that
            # our feet are centred on the origin, where the stage and camera expect the figure.
            joints = exercise_stage.joints_of(rig)
            path = self.scale * (self.rotation @ _point(points, "hips"))
            if root_offset is None:
                feet = sum((joints[f"{j}.{s}"] for j in ("ankle", "foot") for s in ("left", "right")),
                           Vector()) / 4
                root_offset = (joints["pelvis"] - feet) - path
                root_offset.z = 0.0
            want = path + root_offset
            shift = Vector((want.x - joints["pelvis"].x, want.y - joints["pelvis"].y, 0.0))

            # The feet, first by MOVING THE BODY. Copied directions on legs proportioned unlike
            # the source's put the feet somewhere else as the knees bend — 66 mm forward at the
            # bottom of the real Mixamo squat, on this boy's long thigh and short shin. Solving the
            # legs to pull them back cannot work there: the hips are already within a shin's
            # length of the ankles, and the first version of this lock that tried it folded the
            # knee into the floor (25 mm through it). So the planted toes are held by translating
            # the whole figure, which changes no joint angle at all — the pelvis goes further back
            # than the source's, the captured knee, hip and torso angles arrive intact — and the
            # two-bone solve below only takes up what one translation cannot: two feet that
            # disagree, and the vertical. The offset persists, so a foot leaving the ground does
            # not snap the body back onto the source's path.
            for side in ("left", "right"):
                state = self.plan[i][side]
                if state["touchdown"] or state["weight"] <= 0.0:
                    anchors[side] = None
            if self.foot_lock:
                pull, total = Vector(), 0.0
                for side in ("left", "right"):
                    weight = self.plan[i][side]["weight"]
                    if anchors[side] is not None and weight > 0.0:
                        pull += (anchors[side] - (joints[f"foot.{side}"] + shift)) * weight
                        total += weight
                if total > 0.0:
                    correction = Vector((pull.x / total, pull.y / total, 0.0))
                    shift += correction
                    root_offset += correction
            rig.location.x += shift.x
            rig.location.y += shift.y
            bpy.context.view_layer.update()

            # Vertically, `ground_rig` — every frame, on the orientation's own contact rules — and
            # nothing else from it: its horizontal answer recentres a standing figure on its feet.
            held = rig.location.copy()
            animate.ground_rig(rig, self.body, self.orientation, rest_at=self.rest_at,
                               pelvis_at=self.pelvis_at)
            rig.location.x, rig.location.y = held.x, held.y
            bpy.context.view_layer.update()

            # Then each planted toe exactly where it touched down: a two-bone solve of thigh and
            # shin for the residual the translation left (normally a millimetre or two).
            joints = exercise_stage.joints_of(rig)
            free = {k: joints[k].copy() for k in ("ankle.left", "ankle.right",
                                                   "foot.left", "foot.right")}
            for side in ("left", "right"):
                state = self.plan[i][side]
                if not self.foot_lock or state["weight"] <= 0.0:
                    continue
                toe, ankle = joints[f"foot.{side}"], joints[f"ankle.{side}"]
                if anchors[side] is None:
                    anchors[side] = toe.copy()          # touchdown: where it is, grounded
                target = anchors[side] - (toe - ankle)
                target = ankle.lerp(target, state["weight"])
                residual = max(residual, (target - ankle).length)
                pole = joints[f"knee.{side}"] - joints[f"hip.{side}"]
                shortfall = max(shortfall, self.solve_leg(side, target, pole,
                                                          turns.get(side)))

            _key_rotations(rig, frame, previous)
            rig.keyframe_insert("location", frame=frame)
            joints = exercise_stage.joints_of(rig)
            frames_out.append({
                "frame": frame,
                "joints": {k: [round(c, 6) for c in v] for k, v in joints.items()},
                "free": {k: [round(c, 6) for c in v] for k, v in free.items()},
                "lock": {s: round(self.plan[i][s]["weight"], 3) for s in ("left", "right")},
            })

        scene = bpy.context.scene
        scene.frame_start, scene.frame_end = self.first_frame, self.first_frame + len(clip.frames) - 1
        scene.render.fps = int(round(clip.fps))
        scene.render.fps_base = 1.0
        self.result = {
            "clip": clip.path, "fps": clip.fps, "frames": len(clip.frames),
            "source_frames": [clip.first, clip.first + len(clip.frames) - 1],
            "bones": clip.bones, "notes": clip.notes,
            "facing": self.facing, "scale": round(self.scale, 5),
            "our_leg": round(self.our_leg, 5), "source_leg": round(_leg_length(clip.rest), 5),
            "feet": self.feet, "foot_lock": self.foot_lock,
            "holds": {r: list(v) for r, v in self.holds.items()},
            "orientation": self.orientation,
            "max_root_roll_deg": round(roll_max, 2),
            "max_leg_shortfall": round(shortfall, 6),
            "max_ik_residual_mm": round(residual * 1000, 2),
            "per_frame": frames_out,
        }
        print(f"retarget: keyed {len(clip.frames)} frames ({scene.frame_start}..{scene.frame_end}) "
              f"at {scene.render.fps} fps | source forward {self.facing['source_forward']}, "
              f"up {self.facing['source_up']}, scale {self.scale:.4f} | feet {self.feet}"
              f"{'' if self.foot_lock else ' (lock OFF)'} | holds {self.holds or 'none'}")
        return self.result


def _debounce(flags, shortest):
    """Remove runs (of either value) shorter than `shortest` frames, interior ones only."""
    flags = list(flags)
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(flags):
            j = i
            while j < len(flags) and flags[j] == flags[i]:
                j += 1
            if 0 < i and j < len(flags) and j - i < shortest:
                for k in range(i, j):
                    flags[k] = not flags[k]
                changed = True
            i = j
    return flags


# --- measuring what was keyed ------------------------------------------------------------------------

def _skin_points(obj):
    import numpy as np
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    co = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", co)
    evaluated.to_mesh_clear()
    world = np.array(obj.matrix_world, dtype=np.float64)
    return co.reshape(-1, 3) @ world[:3, :3].T + world[:3, 3]


def contact_metrics(rig, body, plan, rest_at=0.0, first_frame=1, sole_band=0.012):
    """Replay the keyed action and measure the contact on the SKIN, not on the rig.

    Played back with `frame_set`, so what is measured is the animation a render or an export will
    get, not the state the loop left behind. Per side, for every run of frames in which the foot is
    planted (lock weight 1):
      slide     the largest horizontal distance any sole vertex travels from where it was on the
                run's first frame — the sole being the skin within `sole_band` of the floor on that
                side at that frame. Also the same for the ankle and toe joints;
      lift      how far those sole vertices rise or sink while planted;
    and, over every frame, the deepest any part of the skin goes below the plane it rests on.
    Returns the numbers and the union of the skin's box over the clip, for framing a camera.
    """
    import numpy as np
    scene = bpy.context.scene
    n = len(plan)
    worst = {"penetration": 0.0, "penetration_frame": None}
    per_side = {s: {"slide": 0.0, "slide_frame": None, "lift": 0.0, "ankle_slide": 0.0,
                    "toe_slide": 0.0, "planted_frames": 0} for s in ("left", "right")}
    runs = {s: None for s in ("left", "right")}
    low, high = Vector((1e9, 1e9, 1e9)), Vector((-1e9, -1e9, -1e9))
    replay = []
    for i in range(n):
        frame = first_frame + i
        scene.frame_set(frame)
        joints = exercise_stage.joints_of(rig)
        skin = _skin_points(body)
        low = Vector(np.minimum(low, skin.min(axis=0)))
        high = Vector(np.maximum(high, skin.max(axis=0)))
        depth = rest_at - float(skin[:, 2].min())
        if depth > worst["penetration"]:
            worst["penetration"], worst["penetration_frame"] = depth, frame
        replay.append({k: [round(c, 6) for c in v] for k, v in joints.items()})
        for s, sign in (("left", -1.0), ("right", 1.0)):
            if plan[i][s]["weight"] < 1.0:
                runs[s] = None
                continue
            per_side[s]["planted_frames"] += 1
            if runs[s] is None or plan[i][s]["touchdown"]:
                side = (skin[:, 0] - joints["pelvis"].x) * sign > 0.0
                sole = side & (skin[:, 2] < rest_at + sole_band)
                runs[s] = {"index": np.nonzero(sole)[0], "at": skin[sole].copy(),
                           "ankle": joints[f"ankle.{s}"].copy(), "toe": joints[f"foot.{s}"].copy()}
                continue
            run = runs[s]
            if len(run["index"]) == 0:
                continue
            now = skin[run["index"]]
            moved = np.linalg.norm(now[:, :2] - run["at"][:, :2], axis=1)
            rise = np.abs(now[:, 2] - run["at"][:, 2])
            m = per_side[s]
            if moved.max() > m["slide"]:
                m["slide"], m["slide_frame"] = float(moved.max()), frame
            m["lift"] = max(m["lift"], float(rise.max()))
            for joint, key in (("ankle", "ankle_slide"), ("foot", "toe_slide")):
                d = joints[f"{joint}.{s}"] - run["ankle" if joint == "ankle" else "toe"]
                m[key] = max(m[key], Vector((d.x, d.y)).length)
    scene.frame_set(first_frame)
    out = {"penetration_mm": round(worst["penetration"] * 1000, 2),
           "penetration_frame": worst["penetration_frame"],
           "sides": {s: {"sole_slide_mm": round(m["slide"] * 1000, 2), "slide_frame": m["slide_frame"],
                         "sole_lift_mm": round(m["lift"] * 1000, 2),
                         "ankle_slide_mm": round(m["ankle_slide"] * 1000, 2),
                         "toe_slide_mm": round(m["toe_slide"] * 1000, 2),
                         "planted_frames": m["planted_frames"]} for s, m in per_side.items()},
           "box": [list(low), list(high)]}
    return out, replay


# --- the form check ----------------------------------------------------------------------------------

def _included(a, b, c):
    return math.degrees((a - b).angle(c - b))


def angles_of(joints):
    """The angles the form notes talk about, measured identically on any skeleton with our joints.

      knee      included angle hip-knee-ankle (180 straight)
      hip       included angle between the torso (pelvis -> chest) and the thigh (180 standing)
      elbow     included angle shoulder-elbow-wrist (180 straight)
      shoulder  how far the upper arm is raised from hanging along the torso (0 hanging, 90 level
                with the shoulder, 180 overhead), whichever plane it was raised in
      torso     the torso's lean from vertical
    """
    up = joints["chest"] - joints["pelvis"]
    out = {}
    for s in ("left", "right"):
        out[f"knee.{s}"] = _included(joints[f"hip.{s}"], joints[f"knee.{s}"], joints[f"ankle.{s}"])
        out[f"hip.{s}"] = math.degrees(up.angle(joints[f"knee.{s}"] - joints[f"hip.{s}"]))
        out[f"elbow.{s}"] = _included(joints[f"shoulder.{s}"], joints[f"elbow.{s}"],
                                      joints[f"wrist.{s}"])
        out[f"shoulder.{s}"] = math.degrees((joints[f"elbow.{s}"] - joints[f"shoulder.{s}"])
                                            .angle(-up))
    out["torso"] = math.degrees(up.angle(Vector((0.0, 0.0, 1.0))))
    return out


_HELD_ANGLES = {"arms": ("elbow", "shoulder"), "arm.left": ("elbow.left", "shoulder.left"),
                "arm.right": ("elbow.right", "shoulder.right"), "legs": ("knee", "hip"),
                "head": ()}


def form_check(exercise, replay, holds=None, first_frame=1, tolerance=FORM_TOLERANCE):
    """For each sourced phase: the closest retargeted frame, and its angles beside the sourced ones.

    "Closest" is the smallest mean absolute difference over the angles the capture actually drives
    — a held region matches its own sourced pose by construction, so it is left out of the choice
    and marked in the table. Returns {phase: {"frame", "rows": [...], "flagged": [...]}}.
    """
    phases = pose_model.phases_of(pose_model.load(), exercise)
    held = set()
    for region in (holds or {}):
        for prefix in _HELD_ANGLES[region]:
            held |= {name for name in angles_of(_vectors(pose_model.solve(phases["start"])))
                     if name.startswith(prefix)}
    measured = [angles_of({k: Vector(v) for k, v in joints.items()}) for joints in replay]
    out = {}
    picks = []
    for phase in ("start", "finish"):
        if phase not in phases:
            continue
        sourced = angles_of(_vectors(pose_model.solve(phases[phase])))
        judged = [k for k in sourced if k not in held]
        best = min(range(len(measured)), key=lambda i: sum(abs(measured[i][k] - sourced[k])
                                                           for k in judged) / len(judged))
        picks.append((phase, sourced, judged, best))
    # And the capture's own far end of the rep — the frame furthest from where it starts, which
    # for a squat is the bottom. "Closest to the sourced finish" can be a frame half-way down when
    # the capture goes deeper than the source; this is the one that shows how much deeper.
    if len(picks) == 2:
        _, sourced, judged, _ = picks[1]
        home = measured[picks[0][3]]
        far = max(range(len(measured)), key=lambda i: sum(abs(measured[i][k] - home[k])
                                                          for k in judged))
        picks.append(("finish vs the capture's far end", sourced, judged, far))
    for phase, sourced, judged, best in picks:
        rows, flagged = [], []
        for k in sourced:
            diff = measured[best][k] - sourced[k]
            flag = k not in held and abs(diff) > tolerance
            rows.append({"angle": k, "sourced": round(sourced[k], 1),
                         "retargeted": round(measured[best][k], 1), "diff": round(diff, 1),
                         "held": k in held, "flag": flag})
            if flag:
                flagged.append(k)
        out[phase] = {"frame": first_frame + best, "rows": rows, "flagged": flagged,
                      "mean_abs_diff": round(sum(abs(measured[best][k] - sourced[k])
                                                 for k in judged) / len(judged), 1)}
    return out


# Our joint names on the source skeleton, so `angles_of` can measure the capture itself.
_SOURCE_JOINTS = {"pelvis": "hips", "chest": "Neck",
                  **{f"{ours}.{side}": f"{side.title()}{theirs}" for side in ("left", "right")
                     for ours, theirs in (("shoulder", "Arm"), ("elbow", "ForeArm"),
                                          ("wrist", "Hand"), ("hip", "UpLeg"), ("knee", "Leg"),
                                          ("ankle", "Foot"))}}


def fidelity(clip, rotation, replay):
    """How far the retargeted figure's angles drift from the CAPTURE's own, frame by frame.

    The form check compares against the sourced poses, and so mixes two questions: is the capture
    good form, and did the retarget keep it. This answers the second alone. Direction copying
    keeps every angle exactly; what can move them is the foot lock's leg solve (and a held region,
    which replaces them on purpose). Returns {angle: (max |difference| in degrees, at frame)}.
    """
    worst = {}
    for i, (points, joints) in enumerate(zip(clip.frames, replay)):
        source = angles_of({ours: rotation @ _point(points, theirs)
                            for ours, theirs in _SOURCE_JOINTS.items()})
        ours = angles_of({k: Vector(v) for k, v in joints.items()})
        for k in source:
            d = abs(ours[k] - source[k])
            if d > worst.get(k, (-1.0, None))[0]:
                worst[k] = (round(d, 2), i)
    return worst


def source_angles(clip, rotation, i):
    return angles_of({ours: rotation @ _point(clip.frames[i], theirs)
                      for ours, theirs in _SOURCE_JOINTS.items()})


def print_form(check, exercise):
    print(f"\nform check — {exercise} (flag beyond {FORM_TOLERANCE:g} degrees; held regions are "
          f"the sourced pose by construction)")
    for phase, found in check.items():
        print(f"  {phase}: frame {found['frame']}, mean |diff| {found['mean_abs_diff']} deg")
        for row in found["rows"]:
            mark = " HELD" if row["held"] else (" <-- OFF" if row["flag"] else "")
            capture = f"  (capture {row['capture']:6.1f})" if "capture" in row else ""
            print(f"    {row['angle']:15s} sourced {row['sourced']:7.1f}  retargeted "
                  f"{row['retargeted']:7.1f}  diff {row['diff']:+7.1f}{capture}{mark}")


# --- looking at it -----------------------------------------------------------------------------------

def frame_clip(body, box, azimuth=30.0, elevation=8.0):
    """A camera holding the skin's box over the whole clip, at the house distance.

    For a clip run with no exercise (and so no `exercise_stage` entry): the staged cameras are
    framed on the two sourced phases, which a test clip does not have. Same lens, same distance
    rule (`exercise_stage.camera_distance`), centred on the clip's own extent.
    """
    distance, _, _ = exercise_stage.camera_distance(body)
    low, high = Vector(box[0]), Vector(box[1])
    centre = (low + high) / 2
    camera = bpy.data.objects.get("camera")
    if camera is None:
        bpy.ops.object.camera_add()
        camera = bpy.context.object
        camera.name = "camera"
    exercise_stage.place_camera(camera, centre, azimuth, elevation, distance)
    return camera, centre, distance


SIDE_COLOURS = {"left": (0.05, 0.25, 0.95, 1.0), "right": (0.95, 0.12, 0.08, 1.0)}


def mark_sides(rig, radius=0.022):
    """A blue ball on the character's LEFT hand and foot, a red one on its right. Debug only.

    For telling left from right in a preview at a glance, and for comparing against the source's
    stick figure, which uses the same colours by the same names.
    """
    marks = []
    joints = exercise_stage.joints_of(rig)
    for side, colour in SIDE_COLOURS.items():
        mat = bpy.data.materials.new(f"mark.{side}")
        mat.diffuse_color = colour
        mat.use_nodes = True
        mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = colour
        for bone, joint in ((f"hand.{side}", f"hand.{side}"), (f"foot.{side}", f"ankle.{side}")):
            bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, segments=16, ring_count=8)
            ball = bpy.context.object
            ball.name = f"mark.{bone}"
            ball.data.materials.append(mat)
            exercise_stage.hold_at(ball, rig, bone, joints[joint], Matrix.Identity(3))
            marks.append(ball)
    return marks


def use_workbench(scene, width=360, height=480):
    """The fast preview: Workbench, studio light, the skin's own texture, no denoise, ~480 px.

    The user is judging motion from these, not shading. Workbench draws a frame in a fraction of a
    second where Cycles at the house settings takes about fifteen.
    """
    scene.render.engine = "BLENDER_WORKBENCH"
    shading = scene.display.shading
    shading.light = "STUDIO"
    shading.color_type = "TEXTURE"
    shading.show_shadows = True
    shading.shadow_intensity = 0.35
    shading.show_cavity = False
    scene.display.render_aa = "8"
    scene.render.resolution_x, scene.render.resolution_y = width, height
    scene.render.resolution_percentage = 100
    return scene


def frames_to_gif(directory, out, fps, pattern="frame_%04d.png", beside=None, loop=True):
    """Stitch rendered frames into a GIF with ffmpeg (palette pass, so the skin does not band).

    `beside` is a second directory of frames with the same numbering, put to the right of the
    first in every frame — the side view next to the staged one.
    """
    import shutil
    import subprocess
    ffmpeg = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    start = min(int(p.stem.split("_")[-1]) for p in Path(directory).glob("frame_*.png"))
    inputs = ["-framerate", str(fps), "-start_number", str(start), "-i",
              str(Path(directory) / pattern)]
    palette = "split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer"
    if beside is not None:
        inputs += ["-framerate", str(fps), "-start_number", str(start), "-i",
                   str(Path(beside) / pattern)]
        graph = ["-filter_complex", "[0][1]hstack=inputs=2," + palette]
    else:
        graph = ["-vf", palette]
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", *inputs, *graph, "-loop", "0" if loop else "-1",
                    str(out)], check=True)
    print(f"gif -> {out} ({out.stat().st_size / 1024:.0f} KB)")
    return out


def write_report(path, result):
    Path(path).write_text(json.dumps(result, indent=1, default=lambda v: list(v)))
    print(f"retarget report -> {path}")


# --- bone for bone: a rig that has Mixamo's own bones (MPFB's `rig.mixamo`) ---------------------
#
# The direction copy above exists because the house rig has 17 bones of its own and Mixamo has 52
# others. A rig built with Mixamo's skeleton (`mpfb_cage`) needs no translation of bones: each of
# its bones takes the source bone's ROTATION, as a change from the source's rest pose, applied to
# its own rest pose — target_world = facing . (source_world . source_rest^-1) . facing^-1 .
# target_rest. Rolls, twists, the spine's three joints, the collarbones, the toes and the fingers
# all arrive, which no direction copy can carry. Both rests are T-poses (Mixamo's FBX rest and the
# skin's, which `mpfb_cage` made the rig's rest), so equal rotations mean equal poses.
#
# The world rotations are turned into each bone's own basis analytically (pose_world = parent_world
# . parent_rest^-1 . rest . basis), so a frame is posed without re-evaluating the scene bone by
# bone. The feet are handled as `Retarget` handles them: first the whole figure is translated so
# the planted ankles are back on their marks (no joint angle changes), then a two-bone solve of
# thigh and shin takes up what one translation cannot, the foot keeping the rotation it was given.

# The neck and head take this share of the capture's turn relative to the chest. Measured on the
# cartoon boy (task 12): Mixamo's back squat extends the head 23 degrees on the chest standing and
# 53 at the bottom (the lifter looks forward while the torso leans); on a head a quarter of the
# figure's height carried by a 5 cm neck, every frame crumpled the throat into facets, and the cage
# itself folded there. At 0.4 the throat is clean in every frame, and the head stays nearer in line
# with the torso — which is also the coaching cue for a squat ("neutral neck"). 1.0 copies it as is.
NECK_SHARE = {"Neck": 0.4, "Head": 0.4}
T_POSE_TOLERANCE = 25.0     # degrees off level a source's resting upper arm may be for BoneRetarget


def _mixamo_bones(rig):
    """{suffix: bone name} for the rig's bones that carry Mixamo names."""
    return {_suffix(b.name): b.name for b in rig.data.bones}


def _parents_first(rig):
    out = []

    def walk(bone):
        out.append(bone.name)
        for child in bone.children:
            walk(child)
    for bone in rig.data.bones:
        if bone.parent is None:
            walk(bone)
    return out


class BoneRetarget:
    """One clip onto a Mixamo-boned rig. `run()` keys every frame; `.result` has what it measured.

    `skin` is what the viewer sees (used once, to stand the first frame's soles on `floor`). The
    rig must be unrotated and unscaled (its armature space is the house frame), as for `Retarget`.
    """

    def __init__(self, rig, skin, clip, exercise=None, feet=None, foot_lock=True, first_frame=1,
                 floor=0.0, share=None):
        if rig.parent is not None or rig.matrix_world.to_3x3() != Matrix.Identity(3):
            raise SystemExit("BoneRetarget expects an unrotated, unscaled, unparented rig")
        if not clip.turns or "Hips" not in clip.rest_turns:
            raise SystemExit("BoneRetarget needs every bone's rotation: load the clip with "
                             "load_clip(path, turned='all')")
        self.rig, self.skin, self.clip = rig, skin, clip
        entry = EXERCISES.get(exercise, {}) if exercise else {}
        self.feet = feet or entry.get("feet", "planted" if exercise else "planted")
        if self.feet not in ("planted", "free"):
            raise SystemExit(f"BoneRetarget does feet 'planted' or 'free', not {self.feet!r}")
        self.foot_lock, self.first_frame, self.floor = foot_lock, first_frame, floor
        self.share = {self._name(n): k for n, k in (NECK_SHARE if share is None else share).items()
                      if self._name(n)}
        self.names = _mixamo_bones(rig)
        self.order = [n for n in _parents_first(rig) if _suffix(n) in clip.rest_turns]
        self.rotation, self.facing = facing(clip.rest)
        self.q = self.rotation.to_quaternion()
        # Equal rotations from rest are equal poses only if both rests are T-poses: the rig's is
        # (the skin's), Mixamo's FBX rest is. A source resting with its arms down — task 11's
        # synthetic clips do — would put every arm 90 degrees too high, and the asymmetric test
        # (task 12) shows exactly that: legs, spine, twist, lean and turn right, arms wrong.
        for side in ("Left", "Right"):
            arm = self.rotation @ (clip.rest[f"{side}ForeArm"] - clip.rest[f"{side}Arm"])
            droop = math.degrees(math.asin(min(1.0, abs(arm.z) / max(arm.length, 1e-9))))
            if droop > T_POSE_TOLERANCE:
                raise SystemExit(f"BoneRetarget needs a T-pose source rest: the source's {side} upper "
                                 f"arm rests {droop:.0f} degrees off level (Mixamo FBX rests in a "
                                 f"T-pose; this clip does not). Use `Retarget` (directions) for it.")
        bones = rig.data.bones
        self.rest = {n: bones[n].matrix_local.to_quaternion() for n in bones.keys()}
        self.rest_head = {n: bones[n].head_local.copy() for n in bones.keys()}
        b = self.names
        self.leg = {s: (bones[b[f"{s.title()}UpLeg"]].length, bones[b[f"{s.title()}Leg"]].length)
                    for s in ("left", "right")}
        ours = sum(self.leg[s][0] + self.leg[s][1] for s in self.leg) / 2
        self.scale = ours / _leg_length(clip.rest)
        self.result = {}

    def _name(self, suffix):
        return next((b.name for b in self.rig.data.bones if _suffix(b.name) == suffix), None)

    def _hip_mid(self, world):
        b = self.names
        return (world[b["LeftUpLeg"]].translation + world[b["RightUpLeg"]].translation) / 2

    def _world(self):
        bpy.context.view_layer.update()
        return {pb.name: pb.matrix.copy() for pb in self.rig.pose.bones}

    def _set_basis(self, name, want, world_rot):
        """Pose `name` so its armature-space rotation is `want`, given its parent's (world_rot)."""
        bone = self.rig.data.bones[name]
        pb = self.rig.pose.bones[name]
        if bone.parent is None:
            basis = self.rest[name].inverted() @ want
        else:
            p = bone.parent.name
            basis = self.rest[name].inverted() @ self.rest[p] @ world_rot[p].inverted() @ want
        if name in self.share:
            # a share of the turn on its parent (`NECK_SHARE`); the world rotation follows
            basis = Quaternion().slerp(basis.normalized(), self.share[name])
            want = ((world_rot[bone.parent.name] @ self.rest[bone.parent.name].inverted()
                     @ self.rest[name] @ basis) if bone.parent else self.rest[name] @ basis)
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = basis.normalized()
        world_rot[name] = want

    def _move_root(self, delta):
        """Translate the whole figure by `delta` (armature space) through the root's location."""
        root = self.names["Hips"]
        pb = self.rig.pose.bones[root]
        pb.location = pb.location + self.rest[root].inverted() @ Vector(delta)

    def _two_bone(self, side, world, world_rot, ankle_target):
        """Thigh and shin re-aimed so the ankle lands on `ankle_target`, the knee kept in the plane
        it was in; the foot keeps its world rotation. Returns how far short a straight leg fell."""
        b = self.names
        up, low, foot = (b[f"{side.title()}{n}"] for n in ("UpLeg", "Leg", "Foot"))
        l1, l2 = self.leg[side]
        hip = world[up].translation
        knee_now = world[low].translation
        to = ankle_target - hip
        dist = to.length
        u = to.normalized()
        pole = _flat(knee_now - hip, u)
        if pole.length < 1e-6:
            pole = _flat(Vector((0.0, 1.0, 0.0)), u)
        pole.normalize()
        short = 0.0
        if dist >= l1 + l2:
            short = dist - (l1 + l2)
            knee = hip + u * l1
        else:
            dist = max(dist, abs(l1 - l2) + 1e-6)
            cos_a = (l1 * l1 + dist * dist - l2 * l2) / (2 * l1 * dist)
            a = math.acos(max(-1.0, min(1.0, cos_a)))
            knee = hip + (u * math.cos(a) + pole * math.sin(a)) * l1
        foot_rot = world_rot[foot]
        for name, start, end in ((up, hip, knee), (low, knee, ankle_target)):
            now = world_rot[name] @ Vector((0.0, 1.0, 0.0))
            want = now.rotation_difference((end - start).normalized()) @ world_rot[name]
            self._set_basis(name, want, world_rot)
        self._set_basis(foot, foot_rot, world_rot)
        return short

    def pose_frame(self, i):
        """Every driven bone from source frame i, and the root on the source's (scaled) path."""
        clip, b = self.clip, self.names
        world_rot = {}
        for name in self.order:
            s = _suffix(name)
            change = clip.turns[i][s] @ clip.rest_turns[s].inverted()
            want = self.q @ change @ self.q.inverted() @ self.rest[name]
            self._set_basis(name, want, world_rot)
        for name in self.rest:                   # bones the clip lacks keep their rest pose
            if name not in world_rot:
                bone = self.rig.data.bones[name]
                parent = world_rot.get(bone.parent.name) if bone.parent else None
                world_rot[name] = ((parent @ self.rest[bone.parent.name].inverted() @ self.rest[name])
                                   if parent is not None else self.rest[name])
        root = b["Hips"]
        self.rig.pose.bones[root].location = (0.0, 0.0, 0.0)
        world = self._world()
        path = self.scale * (self.rotation @ (_point(clip.frames[i], "hips")
                                              - _point(clip.rest, "hips")))
        want = self._rest_hip_mid + path
        self._move_root(want - self._hip_mid(world))
        return world_rot

    def run(self):
        rig, clip = self.rig, self.clip
        b = self.names
        scene = bpy.context.scene
        for pb in rig.pose.bones:
            pb.rotation_mode = "QUATERNION"
            pb.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
            pb.location = (0.0, 0.0, 0.0)
        rest_world = self._world()
        self._rest_hip_mid = self._hip_mid(rest_world)
        # Posing never needs the skin; its modifiers are switched off while bones are set (the
        # bind survives) and back on for the one evaluation that grounds the first frame.
        heavy = [m for m in self.skin.modifiers if m.show_viewport] if self.skin else []
        anchors, previous, frames_out = {}, {}, []
        shortfall = residual = 0.0
        initial_floor_shift = Vector((0.0, 0.0, 0.0))
        try:
            for m in heavy:
                m.show_viewport = False
            for i in range(len(clip.frames)):
                frame = self.first_frame + i
                world_rot = self.pose_frame(i)
                world = self._world()
                ankles = {s: world[b[f"{s.title()}Foot"]].translation.copy() for s in ("left", "right")}
                if i == 0:
                    # Stand the first frame's soles on the floor, measured on the skin.
                    for m in heavy:
                        m.show_viewport = True
                    lift = self.floor - float(_skin_points(self.skin)[:, 2].min()) if self.skin else 0.0
                    initial_floor_shift.z = lift
                    for m in heavy:
                        m.show_viewport = False
                    self._move_root(Vector((0.0, 0.0, lift)))
                    world = self._world()
                    anchors = {s: world[b[f"{s.title()}Foot"]].translation.copy()
                               for s in ("left", "right")}
                elif self.feet == "planted" and self.foot_lock:
                    # the whole figure first: the mean planted ankle back on its mark
                    shift = sum((anchors[s] - ankles[s] for s in anchors), Vector()) / 2
                    self._move_root(shift)
                    world = self._world()
                    for s in ("left", "right"):
                        residual = max(residual, (anchors[s] - world[b[f"{s.title()}Foot"]].translation).length)
                        shortfall = max(shortfall, self._two_bone(s, world, world_rot, anchors[s]))
                    world = self._world()
                elif self.feet == "free" and i > 0:
                    # pose_frame resets the root. Preserve the initial floor offset;
                    # otherwise a free-foot hold jumps between frames one and two.
                    self._move_root(initial_floor_shift)
                    world = self._world()
                # key: rotations on one hemisphere frame to frame, and the root's location
                for pb in rig.pose.bones:
                    q = pb.rotation_quaternion.copy()
                    if pb.name in previous and q.dot(previous[pb.name]) < 0.0:
                        q.negate()
                        pb.rotation_quaternion = q
                    previous[pb.name] = q.copy()
                    pb.keyframe_insert("rotation_quaternion", frame=frame)
                rig.pose.bones[b["Hips"]].keyframe_insert("location", frame=frame)
                frames_out.append({"frame": frame, "ankles": {s: list(world[b[f"{s.title()}Foot"]].translation)
                                                              for s in ("left", "right")},
                                   "hips": list(self._hip_mid(world))})
        finally:
            for m in heavy:
                m.show_viewport = True
            bpy.context.view_layer.update()
        scene.frame_start, scene.frame_end = self.first_frame, self.first_frame + len(clip.frames) - 1
        scene.render.fps = int(round(clip.fps))
        scene.render.fps_base = 1.0
        self.result = {"clip": clip.path, "fps": clip.fps, "frames": len(clip.frames),
                       "method": "bone for bone (rotation, from rest)",
                       "bones_driven": len(self.order), "bones": len(rig.data.bones),
                       "facing": self.facing, "scale": round(self.scale, 5), "feet": self.feet,
                       "foot_lock": self.foot_lock, "turn_share": self.share,
                       "max_ik_residual_mm": round(residual * 1000, 2),
                       "max_leg_shortfall_mm": round(shortfall * 1000, 2), "per_frame": frames_out}
        print(f"retarget (bone for bone): keyed {len(clip.frames)} frames, {len(self.order)} of "
              f"{len(rig.data.bones)} bones driven | source forward {self.facing['source_forward']} | "
              f"scale {self.scale:.4f} | feet {self.feet}{'' if self.foot_lock else ' (lock OFF)'} | "
              f"IK residual {residual * 1000:.1f} mm, shortfall {shortfall * 1000:.1f} mm")
        return self.result


def skin_contact(rig, skin, first_frame, last_frame, floor=0.0, sole_band=0.012):
    """Replay the keyed action and measure contact on the SKIN: per side, how far the sole vertices
    (within `sole_band` of the floor at the first frame) travel horizontally and vertically over
    the clip, and how deep anything goes below the floor. Sides by the character's own left (-X
    in the house frame) relative to the hip midpoint."""
    import numpy as np
    scene = bpy.context.scene
    names = _mixamo_bones(rig)
    out = {"penetration_mm": 0.0, "penetration_frame": None}
    sole = {}
    start = {}
    initial_gaps = {}
    worst = {s: {"slide": 0.0, "lift": 0.0, "frame": None} for s in ("left", "right")}
    low, high = None, None
    for frame in range(first_frame, last_frame + 1):
        scene.frame_set(frame)
        pts = _skin_points(skin)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        low = lo if low is None else np.minimum(low, lo)
        high = hi if high is None else np.maximum(high, hi)
        depth = floor - float(pts[:, 2].min())
        if depth > out["penetration_mm"] / 1000:
            out["penetration_mm"], out["penetration_frame"] = depth * 1000, frame
        if frame == first_frame:
            pose = rig.pose.bones
            mid = (rig.matrix_world @ pose[names["LeftUpLeg"]].head
                   + rig.matrix_world @ pose[names["RightUpLeg"]].head) / 2
            for s, sign in (("left", -1.0), ("right", 1.0)):
                pick = ((pts[:, 0] - mid.x) * sign > 0) & (pts[:, 2] < floor + sole_band)
                if not pick.any():
                    # A raised/hanging foot has no vertices near the floor. Track
                    # its own sole and report the floor gap instead of crashing or
                    # treating an empty set as zero movement.
                    title = s.title()
                    groups = {g.index for g in skin.vertex_groups if title + 'Foot' in g.name or title + 'ToeBase' in g.name}
                    foot = np.array([sum(g.weight for g in v.groups if g.group in groups) > .5 for v in skin.data.vertices])
                    if foot.any():
                        pick = foot & (pts[:, 2] < float(pts[foot, 2].min()) + sole_band)
                sole[s] = np.flatnonzero(pick)
                start[s] = pts[pick].copy()
                initial_gaps[s] = max(0., float(pts[pick, 2].min()) - floor) * 1000 if pick.any() else None
            continue
        for s in ("left", "right"):
            now = pts[sole[s]]
            if not len(now):
                continue
            slide = float(np.linalg.norm(now[:, :2] - start[s][:, :2], axis=1).max())
            lift = float(np.abs(now[:, 2] - start[s][:, 2]).max())
            if slide > worst[s]["slide"]:
                worst[s]["slide"], worst[s]["frame"] = slide, frame
            worst[s]["lift"] = max(worst[s]["lift"], lift)
    scene.frame_set(first_frame)
    out["penetration_mm"] = round(out["penetration_mm"], 2)
    out["sides"] = {s: {"sole_vertices": int(len(sole[s])), "sole_slide_mm": round(w["slide"] * 1000, 2) if len(sole[s]) else None,
                        "slide_frame": w["frame"], "sole_lift_mm": round(w["lift"] * 1000, 2),
                        "initial_floor_gap_mm": initial_gaps[s]}
                    for s, w in worst.items()}
    out["box"] = [list(map(float, low)), list(map(float, high))]
    return out
