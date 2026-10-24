"""Our character's measurements and comparison stills, from a finished run's saved scene. In Blender.

    blender --background RUN_DIR/scene.blend --python-exit-code 1 \
        --python studio/blender/inspect_run.py -- --run RUN_DIR

Reads RUN_DIR/targets.json, the run's snapshot RUN_DIR/references.json and (when present)
RUN_DIR/measurements.json. Writes RUN_DIR/ours.json (format: studio/measure/checks.py and
studio/tests/fixtures/ours-squat.json) and RUN_DIR/stills/<image-id>.png, 560x560 EEVEE, one for
every key image (not context-only, not deleted) at the frame of that image's position.

Our joints are measured with studio/measure/metrics.py, the same code that measures the coach.
"flat" values are projected on a camera's image plane like the coach's photo: the angles, the lean
and hip-vs-knee on the side camera's, the widths on the behind camera's; "3d" values use the joints
as they are. Nothing is guessed: a reading that cannot be made is written as null.

Our torso line (hip joints -> shoulder joints, as the coach's MediaPipe landmarks) is counted from the
rig's own upright bind pose (task 18): it is turned, about the side camera's axis, by the bind pose's
own lean on that camera (metrics.rest_turn) before every angle that uses it - torso_lean, hip_included,
upperarm_vs_torso and neck_vs_torso - so standing tall reads 0 as a person does. The cartoon's
shoulder joints sit 76 mm behind its hips (its bind pose reads 15 degrees back); matched joint for
joint, the coach's upright start cost a visible 22-degree bow. ours.json `torso_rest_lean` says how far
the bind pose leans (+ forward, - back) on the side camera. Our legs are read the way they look (task
20): each thigh line (hip -> knee joints) is turned, about that camera's axis, onto the thigh's own skin
outline as the bind pose shows it (thigh_outlines; metrics.limb_outline, rest_leg_turns), the shin with it,
before knee_included, hip_included and hip_below_knee, so a thigh level as its skin looks reads depth 0. The
cartoon's hip joint sits toward the front of the thigh's top: its thigh joint line slopes 10.6 degrees back
standing, its skin outline 4.5 (MediaPipe on its bind render reads about 4: bind_reading cross-checks it).
Read on the joints a squat at the coach's depth looked deeper than his; counted as straight down from the
bind pose, shallower. So the bind pose reads its hip about 175.5, its knee as its joints (straight).
ours.json `legs_at_rest` keeps the joint lines' and the outlines' slopes (+ forward of straight down), both
readings of the bind pose and `warning` (metrics.bind_legs_warning: they disagree, or a thigh has no outline),
which checks.py says on the leg rows.

The depth is judged the way a judge sees it, the same way on both pictures (task 22): how far the seat's lowest
point sits below the top of whichever knee shows, in thigh lengths, on the silhouette (posture.seat_depth) - the
coach's on MediaPipe's person mask, ours on our side still's mask render at the bottom position (seat_picture). Its
cross-check is the same two points read off the 3D skin at that frame (seat_skin). ours.json `seat_below_knee`
{"end": {"frame", "picture", "seat", "knee_top", "hip", "knee", "why", "skin", "skin_why"}}, for an exercise whose
kit row says so (kit.has_depth: the back squat).

Frames where the pose pops (metrics.jump_frames: the driving angle, the hips/knees against their 7-frame median, or a
displaced block between two snaps) are never picked as a position and are listed in ours.json `jumps`
with how far the hips and knees sit from that median; every snap between consecutive frames
(metrics.jump_steps) is listed in `steps`. The checks fail on either.

Ours read from the picture (see "our picture" below; task 19): at each position's frame our side view (the
side still, and four framings moved by 1.5% of the frame) is read by the coach's own reader, MediaPipe
(studio/measure/read_pose.py, run in the Studio's Python), and ours.json `picture` keeps, per position,
studio/measure/picture.py's summary: the median of each angle over the framings whose shoulders and hips
it sees clearly. checks.py judges the torso-based rows (lean, hip angle) on it, the rig's reading their
cross-check; on this cartoon the rig's shoulder joints sit at the back of the chest and the detector's
shoulders slide back toward the bar as he squats, so the two part by up to 12 degrees at the bottom. A
rig whose bind pose does not stand upright is warned about in the log (bind_pose_warning).

Posture (see "posture" below), every frame on the side camera's plane: the neck against the torso and
where the face points, measured like the coach's, and the back as the spine's curl from the rest pose
(read off the rig, not the silhouette: the cartoon's build and clothes never enter it).

Skin turned over (see "folded skin" below) is measured on every frame, after the stills (their
cameras are among those a fold is seen from), and listed in `folds`: one record per patch over the
check's limit (checks.fold_limit, scaled by the character's standing height `height_m`) with its
frame, bone, kind (crease, defect, hidden), area and visible area in mm² and centre; `folds_worst` is
{bone: largest area}. The keys are left out when the skin cannot be compared with its bind pose, and
the check then says "not measured".

Cameras are placed relative to the figure, not the world: the saved run scene faces +Y (the house
frame, `mpfb_cage`), a fresh MPFB human faces -Y. The figure's front is read from its hips at the
first frame, and each view's azimuth is counted from there the way `mpfb_cage.camera_on` counts it
from the house figure's front (+90 = the figure's left). The side camera looks at the side of the
body the reference photo shows (its `camera_side`), so a side still compares like with like.
"""
import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector
from mathutils.bvhtree import BVHTree

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "studio" / "measure"))
sys.path.insert(0, str(REPO / "pipeline" / "blender"))
from metrics import (bind_pose_warning, jump_frames, jump_sizes, jump_steps, measure, phase_driver,  # noqa: E402
                     pick_phase_frames, rest_lean as _rest_lean, rest_leg_slopes, rest_leg_turns, rest_turn,
                     limb_outline, limb_slope, bind_legs_warning, picture_slope)
from checks import fold_limit  # noqa: E402
from kit import has_depth  # noqa: E402
from creases import CREASE_SLACK, crease_of, hinges_from  # noqa: E402
from picture import FRAMINGS, PICTURE_METRICS, summarize  # noqa: E402
from posture import (KNEE_TORSO, KNEE_WINDOW, SEAT_BEHIND, back_profile, head_pitch, neck_vs_torso,  # noqa: E402
                     seat_depth, spine_bend)
import mpfb_cage  # noqa: E402

P = "mixamorig:"
JOINT_BONES = {"shoulder": "Arm", "elbow": "ForeArm", "wrist": "Hand", "hip": "UpLeg", "knee": "Leg", "ankle": "Foot"}
UP = np.array([0.0, 0.0, 1.0])
SIDE_PLANE = ("knee_included", "hip_included", "torso_lean", "hip_below_knee", "thigh_level")
BEHIND_PLANE = ("grip_ratio", "elbow_width_ratio")
# view -> (azimuth from the figure's front, + toward its left; elevation). "side" turns to the
# reference's camera side; "frontal" (front or behind, not told apart) uses the behind camera.
VIEWS = {"front": (0.0, 5.0), "behind": (180.0, 5.0), "three-quarter": (35.0, 10.0)}
SIDE_ELEVATION = 5.0
SIDE_TURN = {"left": 90.0, "right": -90.0}
FRAME_MARGIN = 1.05
FACING_MIN = 0.005      # image widths: a nose closer than this to the head's line gives no facing
KNOWN_VIEWS = ("side", "front", "behind", "three-quarter", "frontal")


class RunError(Exception):
    """A plain-words reason the stills step cannot finish."""


def log(msg):
    print(f"inspect: {msg}", flush=True)


# --- the character --------------------------------------------------------------------------------

def _skins(rig):
    """[(hide_render, -vertices, name, mesh)] of the meshes the rig's Armature modifier drives."""
    return [(obj.hide_render, -len(obj.data.vertices), obj.name, obj) for obj in bpy.data.objects
            if obj.type == "MESH" and any(m.type == "ARMATURE" and m.object == rig for m in obj.modifiers)]


def skin_of(rig):
    """The mesh measured for this rig (find_rig_and_skin's choice): the rendered one, then the most vertices."""
    skins = sorted(_skins(rig), key=lambda p: p[:3])
    if not skins:
        raise RunError("The rig drives no mesh: no skin to read the legs' outline from.")
    return skins[0][3]


def find_rig_and_skin():
    """(armature, mesh): the armature with a mixamorig:Hips bone and the mesh its Armature modifier
    drives. With several meshes (the run scene keeps the hidden MPFB cage) the rendered one wins,
    then the one with the most vertices."""
    pairs = []
    for rig in bpy.data.objects:
        if rig.type != "ARMATURE" or P + "Hips" not in rig.data.bones:
            continue
        for hide, verts, name, obj in _skins(rig):
            pairs.append((hide, verts, rig.name, name, rig, obj))
    if not pairs:
        raise RunError("The saved scene has no character: no armature with a mixamorig:Hips bone "
                       "driving a mesh.")
    pairs.sort(key=lambda p: p[:4])
    return pairs[0][4], pairs[0][5]


def joints_at(rig, frame):
    """World positions of the joints metrics.py reads ("shoulder.left" ... "ankle.right"): the heads
    of mixamorig:{Left,Right}{Arm,ForeArm,Hand,UpLeg,Leg,Foot} at that frame."""
    bpy.context.scene.frame_set(frame)
    world, pose = rig.matrix_world, rig.pose.bones
    return {f"{joint}.{side.lower()}": np.array(world @ pose[P + side + bone].head)
            for joint, bone in JOINT_BONES.items() for side in ("Left", "Right")}


def figure_yaw(joints):
    """The camera_on azimuth of a camera straight in front of the figure, from its hip line."""
    left = joints["hip.left"] - joints["hip.right"]
    left[2] = 0.0
    if np.linalg.norm(left) < 1e-9:
        raise RunError("Cannot tell which way the character faces: its hips are on top of each other.")
    forward = np.cross(left / np.linalg.norm(left), UP)
    return math.degrees(math.atan2(-forward[0], forward[1]))


def view_direction(azimuth, elevation):
    """Unit vector from the figure toward a camera_on camera (its view axis, reversed)."""
    a, e = math.radians(azimuth), math.radians(elevation)
    return np.array([-math.sin(a) * math.cos(e), math.cos(a) * math.cos(e), math.sin(e)])


def camera_for(view, side, yaw):
    """(world azimuth, elevation, camera view name) for a reference image's view."""
    if view == "side":
        return yaw + SIDE_TURN[side], SIDE_ELEVATION, "side"
    if view == "frontal":
        view = "behind"
    turn, elevation = VIEWS[view]
    return yaw + turn, elevation, view


def _fingertip(rig, side):
    """Rest position of the hand's tip: the end of the middle finger's chain (on the cartoon, the
    mitten's measured tip), else the hand bone's tail."""
    bones = rig.data.bones
    chain = sorted((n for n in bones.keys() if n.startswith(f"{P}{side}HandMiddle") and n[-1].isdigit()),
                   key=lambda n: int(n[len(f"{P}{side}HandMiddle"):]))
    return (bones[chain[-1]] if chain else bones[P + side + "Hand"]).tail_local


def arm_lengths(rig):
    """Rest-pose arm proportions (metres) for feasibility.closest_arm_pose, both sides averaged:
    half the shoulder-joint width, the upper arm (shoulder to elbow joint) and the forearm from the
    elbow to the middle of the palm, where a bar sits (mpfb_cage.PALM_ALONG of the way from the
    wrist to the fingertip): the point the feasibility search holds on the bar line, as the squat
    report measured the cartoon (scratchpad/t12/armfeas.py, 0.124)."""
    bones, world = rig.data.bones, rig.matrix_world

    def at(name):
        return np.array(world @ bones[P + name].head_local)

    def palm(side):
        wrist = bones[P + side + "Hand"].head_local
        return np.array(world @ (wrist + mpfb_cage.PALM_ALONG * (_fingertip(rig, side) - wrist)))

    half = float(np.linalg.norm(at("LeftArm") - at("RightArm"))) / 2
    upper = np.mean([np.linalg.norm(at(s + "ForeArm") - at(s + "Arm")) for s in ("Left", "Right")])
    fore = np.mean([np.linalg.norm(palm(s) - at(s + "ForeArm")) for s in ("Left", "Right")])
    return {"shoulder_half_width": round(half, 4), "upper_arm": round(float(upper), 4), "forearm": round(float(fore), 4)}


# --- measuring ------------------------------------------------------------------------------------

def _clean(values):
    return {k: (round(float(v), 3) if v is not None and math.isfinite(v) else None) for k, v in values.items()}


def rest_joints(rig):
    """The joints metrics.py reads, in the rig's rest (bind) pose, world space."""
    bones, world = rig.data.bones, rig.matrix_world
    return {f"{joint}.{side.lower()}": np.array(world @ bones[P + side + bone].head_local)
            for joint, bone in JOINT_BONES.items() for side in ("Left", "Right")}


def rest_lean(rig, side_normal):
    """Degrees the rig's bind pose leans (+ forward, - back) hip joints -> shoulder joints, seen on the side
    camera's plane: what standing tall reads before the turn (torso_turn)."""
    joints = rest_joints(rig)
    return _rest_lean(joints, UP, side_normal, figure_front(joints))


def torso_turn(rig, side_normal):
    """The turn about the side camera's axis that stands the bind pose's torso line upright (metrics.rest_turn):
    every torso-based angle of ours is counted from it (see the module's docstring)."""
    return rest_turn(rest_joints(rig), UP, side_normal)


_OUTLINES = {}       # (rig, skin and their data, the skin's groups, plane, rest joints) -> _thigh_rest: read once each


def rest_skin(rig, skin):
    """(world vertex positions, edges (M, 2), each vertex's bone with the largest deforming weight or None) of
    the skin in the rig's rest (bind) pose."""
    held = rig.data.pose_position
    rig.data.pose_position = "REST"
    bpy.context.view_layer.update()
    try:
        co, mesh, evaluated = _evaluated_mesh(skin)
        edges = np.empty(len(mesh.edges) * 2, dtype=np.int64)
        mesh.edges.foreach_get("vertices", edges)
        deform = {b.name for b in rig.data.bones if b.use_deform}
        names = {g.index: g.name for g in evaluated.vertex_groups}
        top = []
        for v in mesh.vertices:
            best = max(((g.weight, names[g.group]) for g in v.groups if names.get(g.group) in deform),
                       default=(0.0, None))
            top.append(best[1] if best[0] > 0 else None)
        evaluated.to_mesh_clear()
    finally:
        rig.data.pose_position = held
        bpy.context.view_layer.update()
    return co, edges.reshape(-1, 2), np.array(top, dtype=object)


def _thigh_rest(rig, skin, normal):
    """{"outline": {"left", "right"}, "bones": [...]}: each thigh's skin in the bind pose seen on the plane square
    to `normal` (metrics.limb_outline on the skin whose largest weight is on that UpLeg bone, between the hip and
    knee joints): its outline's direction, None where too little skin is found; and each vertex's bone with the
    largest weight (rest_skin), which seat_skin reads the seat and the knees by. Read once per rig, skin, plane and
    rest pose: the key holds the rig's and skin's data and the skin's vertex groups (the weights live in them), so
    another character or another weighting is read anew."""
    joints = rest_joints(rig)
    key = (rig.as_pointer(), rig.data.as_pointer(), skin.as_pointer(), skin.data.as_pointer(),
           tuple(g.name for g in skin.vertex_groups), len(skin.data.vertices),
           tuple(np.round(np.asarray(normal, float), 6)), tuple(np.round(np.concatenate([joints[k] for k in sorted(joints)]), 6)))
    if key not in _OUTLINES:
        co, edges, top = rest_skin(rig, skin)
        out = {"outline": {}, "bones": top}
        for side in ("left", "right"):
            mine = top == P + side.capitalize() + "UpLeg"
            own = edges[mine[edges[:, 0]] & mine[edges[:, 1]]]
            out["outline"][side] = limb_outline(co, own, joints[f"hip.{side}"], joints[f"knee.{side}"], normal)
        _OUTLINES[key] = out
    return _OUTLINES[key]


def thigh_outlines(rig, skin, normal):
    """{"left", "right"}: each thigh's skin outline in the bind pose, a unit direction on the plane square to
    `normal`; None where too little skin is found (_thigh_rest)."""
    return _thigh_rest(rig, skin, normal)["outline"]


def leg_turns(rig, side_normal, skin=None):
    """The turns about the side camera's axis that lay each thigh line (and its shin with it) onto the thigh's
    skin outline in the bind pose (metrics.rest_leg_turns): every leg-based measure of ours is counted from them
    (see the module's docstring)."""
    skin = skin_of(rig) if skin is None else skin
    return rest_leg_turns(rest_joints(rig), side_normal, thigh_outlines(rig, skin, side_normal))


def leg_slopes(rig, side_normal, skin=None):
    """{"joints": {"thigh.left", ...}, "outline": {"thigh.left", "thigh.right"}}: degrees each bind-pose joint line
    and each thigh's skin outline points forward (+) or back (-) of straight down on the side camera's plane."""
    skin = skin_of(rig) if skin is None else skin
    joints = rest_joints(rig)
    front = figure_front(joints)
    outline = thigh_outlines(rig, skin, side_normal)
    return {"joints": rest_leg_slopes(joints, UP, side_normal, front),
            "outline": {f"thigh.{s}": (None if d is None else limb_slope(d, UP, side_normal, front))
                        for s, d in outline.items()}}


def measure_frame(joints, side, side_normal, behind_normal, torso_turn=None, leg_turns=None):
    """{"flat": ..., "3d": ...} for one frame, by the measurement rules (plan, "Measurement rules"); the torso
    line turned by `torso_turn` (the rig's, see torso_turn) and the thigh and shin lines by `leg_turns` (see
    leg_turns) on the side camera's plane and in 3D. The behind camera gives only widths, which no torso or
    leg line enters."""
    on_side = measure(joints, side, UP, side_normal, torso_turn=torso_turn, leg_turns=leg_turns)
    behind = measure(joints, side, UP, behind_normal)
    flat = {k: on_side[k] for k in SIDE_PLANE}
    flat.update({k: behind[k] for k in BEHIND_PLANE})
    return {"flat": _clean(flat), "3d": _clean(measure(joints, side, UP, None, torso_turn=torso_turn,
                                                       leg_turns=leg_turns))}


# --- posture -------------------------------------------------------------------------------------
# The same definitions as the coach's (studio/measure/posture.py, read by read_pose.py), seen through
# the side camera:
#   neck_vs_torso  hip joints' midpoint -> shoulder joints' midpoint -> the ears' midpoint, the coach's own
#                  definition, absolute, the torso line counted from the bind pose (torso_turn, as every
#                  torso-based angle of ours), every frame: on the side camera's plane (flat), and on the
#                  figure's own side plane, square to its hip line (3d). The ears are found on the skin
#                  (ear_vertices, written as ours.json `ears`): of the vertices the Head bone alone
#                  carries, those at ear height - from the Head bone's head up to the middle of the head
#                  - within EAR_DEPTH of each side's widest point (the ears stand out from the head
#                  there). They move with the Head bone (ears_at);
#   head_pitch     the Head bone's forward axis (head_forward: the rest pose's front, so the rest pose
#                  reads 0) against level on the side camera's plane, every frame;
#   spine_curl     the back, judged from the rig: the side-seen bends of the chain Hips -> Spine ->
#                  Spine1 -> Spine2 -> Neck (their heads; posture.spine_bend) summed, less the same at rest
#                  (rest_spine, written as ours.json `spine_rest`): + rounded, - arched, every frame.
#
# The silhouette tools below (render_mask, arm_vertices, to_pixels, back_at) are not run by inspect():
# the Blender test uses them to check posture.back_profile, the coach's back measure, on rendered bodies
# and to reproduce posture's calibration (back-calibration.json). back_at renders a mask of the character
# alone (the rendered meshes its rig deforms: no bar, floor or backdrop) with a transparent film through
# the side still's camera at POSTURE_MASK_SCALE x the stills' size, and a second mask of its arms alone
# (the skin whose largest weight is on an arm, forearm, hand or finger bone, cut out with a Mask
# modifier), whose outline is left out of the back.

POSTURE_MASK_SCALE = 2
ARM_BONES = ("Arm", "ForeArm", "Hand")     # after Left/Right; "Hand" also takes the fingers (HandIndex1 ...)
ARM_GROUP = "posture-arms"
SPINE_CHAIN = ("Hips", "Spine", "Spine1", "Spine2", "Neck")
EAR_DEPTH = 0.10        # of each side's widest point at ear height: how far in the ear's cluster reaches
EAR_CARRIED = 0.99      # the share of a vertex's bone weight on the Head bone: it moves with the head alone


def figure_front(joints):
    """The figure's front, level, from its hip line (as figure_yaw reads it)."""
    left = joints["hip.left"] - joints["hip.right"]
    left = np.array([left[0], left[1], 0.0])
    if np.linalg.norm(left) < 1e-9:
        return None
    return np.cross(left / np.linalg.norm(left), UP)


def head_at(rig, local_forward):
    """(the Head bone's head, the face's direction) in world space at the current frame."""
    world, head = rig.matrix_world, rig.pose.bones[P + "Head"]
    face = (world.to_3x3() @ head.matrix.to_3x3() @ local_forward).normalized()
    return np.array(world @ head.head), np.array(face)


def posture_frame(joints, ear, face, side_normal, torso_turn=None):
    """{"flat": {neck_vs_torso, head_pitch}, "3d": {neck_vs_torso}} for one frame (see "posture"); `ear` the
    ears' midpoint (ears_at), None when the ears were not found; the torso line turned by `torso_turn`."""
    front = figure_front(joints)
    if front is None:
        return {"flat": {"neck_vs_torso": None, "head_pitch": None}, "3d": {"neck_vs_torso": None}}
    hip = (joints["hip.left"] + joints["hip.right"]) / 2
    shoulder = (joints["shoulder.left"] + joints["shoulder.right"]) / 2
    hip_line = joints["hip.left"] - joints["hip.right"]
    def neck(normal):
        return None if ear is None else neck_vs_torso(hip, shoulder, ear, front, normal=normal, up=UP,
                                                      torso_turn=torso_turn)
    flat = {"neck_vs_torso": neck(side_normal), "head_pitch": head_pitch(face, UP, front, normal=side_normal)}
    side_plane = {"neck_vs_torso": neck(hip_line)}
    return {"flat": _clean(flat), "3d": _clean(side_plane)}


def ear_vertices(rig, skin):
    """{"left", "right"}: the skin's vertex indices of each ear (see "posture"), from the bind pose."""
    bones, world = rig.data.bones, rig.matrix_world
    deform = {b.name for b in bones if b.use_deform}
    names = {g.index: g.name for g in skin.vertex_groups}
    co = np.array([skin.matrix_world @ v.co for v in skin.data.vertices])
    head = []
    for v in skin.data.vertices:
        weights = {names[g.group]: g.weight for g in v.groups if names.get(g.group) in deform}
        if weights and weights.get(P + "Head", 0.0) >= EAR_CARRIED * sum(weights.values()):
            head.append(v.index)
    if not head:
        raise RunError("Cannot find the ears: no skin moves with the Head bone alone.")
    head = np.array(head)
    joint = np.array(world @ bones[P + "Head"].head_local)
    left = np.array(world @ bones[P + "LeftUpLeg"].head_local) - np.array(world @ bones[P + "RightUpLeg"].head_local)
    left = np.array([left[0], left[1], 0.0]) / max(float(np.hypot(left[0], left[1])), 1e-9)
    height = co[head] @ UP
    band = head[(height >= joint @ UP) & (height <= (height.min() + height.max()) / 2)]
    if not len(band):
        raise RunError("Cannot find the ears: the head has no skin at ear height.")
    out = {}
    for side, sign in (("left", 1.0), ("right", -1.0)):
        out_by = sign * ((co[band] - joint) @ left)
        out[side] = [int(i) for i in band[out_by >= out_by.max() * (1 - EAR_DEPTH)]]
    return out


def ear_points(rig, skin):
    """{"left", "right"}: the middle of each ear (world, bind pose)."""
    co = np.array([skin.matrix_world @ v.co for v in skin.data.vertices])
    return {side: co[verts].mean(axis=0) for side, verts in ear_vertices(rig, skin).items()}


def ears_at(rig, ears):
    """The ears' midpoint (world) at the current frame: the bind-pose ear points carried by the Head bone."""
    world, pb = rig.matrix_world, rig.pose.bones[P + "Head"]
    carry = np.array(world @ pb.matrix @ rig.data.bones[P + "Head"].matrix_local.inverted() @ world.inverted())
    mid = (np.asarray(ears["left"]) + np.asarray(ears["right"])) / 2
    return carry[:3, :3] @ mid + carry[:3, 3]


def _spine(rig, joints, side_normal, rest):
    """posture.spine_bend of SPINE_CHAIN's heads (posed now, or at rest) on the side plane."""
    world = rig.matrix_world
    if rest:
        points = [np.array(world @ rig.data.bones[P + n].head_local) for n in SPINE_CHAIN]
    else:
        points = [np.array(world @ rig.pose.bones[P + n].head) for n in SPINE_CHAIN]
    front = figure_front(joints)
    return None if front is None else spine_bend(points, front, UP, normal=side_normal)


def rest_spine(rig, side_normal):
    """The spine's side-seen bend in the rig's rest pose: the character's own straight back."""
    bones, world = rig.data.bones, rig.matrix_world
    joints = {f"{joint}.{side.lower()}": np.array(world @ bones[P + side + bone].head_local)
              for joint, bone in JOINT_BONES.items() for side in ("Left", "Right")}
    return _spine(rig, joints, side_normal, rest=True)


def spine_curl(rig, side_normal, rest):
    """Degrees the spine curls (+, rounded) or arches (-) at the current frame, from its rest bend."""
    world, pose = rig.matrix_world, rig.pose.bones
    joints = {f"{joint}.{side.lower()}": np.array(world @ pose[P + side + bone].head)
              for joint, bone in JOINT_BONES.items() for side in ("Left", "Right")}
    now = _spine(rig, joints, side_normal, rest=False)
    return None if now is None or rest is None else now - rest


def arm_vertices(rig, obj):
    """Indices of the mesh's vertices whose largest bone weight is on an arm, forearm, hand or finger."""
    deform = {b.name for b in rig.data.bones if b.use_deform}
    names = {g.index: g.name for g in obj.vertex_groups}
    out = []
    for v in obj.data.vertices:
        weights = [(g.weight, names[g.group]) for g in v.groups if names.get(g.group) in deform]
        if not weights:
            continue
        part = max(weights)[1][len(P):]
        if any(part.startswith(side) and part[len(side):].startswith(ARM_BONES) for side in ("Left", "Right")):
            out.append(v.index)
    return out


def character_meshes(rig):
    """The rendered meshes the rig deforms (the skin, clothes): not the bar, floor or backdrop."""
    return [o for o in bpy.context.scene.objects if o.type == "MESH" and not o.hide_render
            and any(m.type == "ARMATURE" and m.object == rig for m in o.modifiers)]


def render_mask(rig, scale=POSTURE_MASK_SCALE, arms_only=False):
    """The character's coverage (0-1) through the scene camera, rows from the top: rendered with a
    transparent film and every other object hidden, at `scale` x the scene's size; with `arms_only`, its
    arms alone (arm_vertices, kept by a Mask modifier after its deformation). The scene's settings,
    objects and meshes are put back as they were."""
    scene = bpy.context.scene
    render = scene.render
    keep = set(character_meshes(rig))
    cut = []                                    # (mesh, its added modifier, its added group)
    if arms_only:
        for obj in list(keep):
            verts = arm_vertices(rig, obj)
            if not verts:
                keep.discard(obj)
                continue
            group = obj.vertex_groups.new(name=ARM_GROUP)
            group.add(verts, 1.0, "REPLACE")
            modifier = obj.modifiers.new(ARM_GROUP, "MASK")
            modifier.vertex_group = group.name
            cut.append((obj, modifier, group))
    hidden = [o for o in scene.objects if o not in keep and not o.hide_render]
    held = (render.film_transparent, render.resolution_x, render.resolution_y, render.resolution_percentage,
            render.filepath, render.image_settings.color_mode)
    folder = Path(tempfile.mkdtemp(prefix="posture-mask-"))
    try:
        for o in hidden:
            o.hide_render = True
        render.film_transparent = True
        render.resolution_x, render.resolution_y = held[1] * scale, held[2] * scale
        render.resolution_percentage = 100
        render.image_settings.color_mode = "RGBA"
        render.filepath = str(folder / "mask.png")
        bpy.ops.render.render(write_still=True)
        image = bpy.data.images.load(str(folder / "mask.png"))
        try:
            width, height = image.size
            px = np.empty(width * height * 4, dtype=np.float32)
            image.pixels.foreach_get(px)
        finally:
            bpy.data.images.remove(image)
        return px.reshape(height, width, 4)[::-1, :, 3].astype(float)
    finally:
        for obj, modifier, group in cut:
            obj.modifiers.remove(modifier)
            obj.vertex_groups.remove(group)
        for o in hidden:
            o.hide_render = False
        (render.film_transparent, render.resolution_x, render.resolution_y, render.resolution_percentage,
         render.filepath, render.image_settings.color_mode) = held
        shutil.rmtree(folder, ignore_errors=True)


def to_pixels(camera, point, shape):
    """A world point's (x, y) on a picture of `shape` (rows, columns) through the camera: y down, pixel
    centres on whole numbers (as posture.py reads a mask)."""
    v = world_to_camera_view(bpy.context.scene, camera, Vector(point))
    return (v.x * shape[1] - 0.5, (1.0 - v.y) * shape[0] - 0.5)


def back_at(rig, skin, frame, azimuth, elevation, local_forward, span=None):
    """{"profile": posture.back_profile (None when the face's side cannot be told), "facing", "mask",
    "arm_mask"} at that frame through the side still's camera (see "posture"), on `span` when given."""
    camera = aim_camera(skin, frame, azimuth, elevation)
    facing = facing_in(rig, camera, local_forward)
    mask, arm_mask = render_mask(rig), render_mask(rig, arms_only=True)
    joints = joints_at(rig, frame)
    shoulder = to_pixels(camera, (joints["shoulder.left"] + joints["shoulder.right"]) / 2, mask.shape)
    hip = to_pixels(camera, (joints["hip.left"] + joints["hip.right"]) / 2, mask.shape)
    profile = back_profile(mask, shoulder, hip, facing, arm_mask=arm_mask, span=span)
    return {"profile": profile, "facing": facing, "mask": mask, "arm_mask": arm_mask}


# --- folded skin ----------------------------------------------------------------------------------
# A skin face is turned over when its posed normal points against the normal its own bones give it:
# the bind-pose normal carried by the face's blended bone rotation (sum over its corners of
# sum_b w_b R_b, R_b the bone's rest->pose rotation, in the figure's own frame). A face that turns,
# bends or twists with its bones keeps a positive dot however far it turns, so a limb raised or bent
# is not flagged for having turned; only skin that has passed through itself is. Faces that move
# rigidly with one bone can never turn over, so every fold sits where weights blend, beside a joint.
# Turned-over faces sharing a corner (the mesh welded across its UV seams) form one patch.
#
# Only patches over the check's limit (checks.fold_limit: 100 mm² for a 1 m character, scaled by
# height²) are written, each of one kind:
#   crease  skin pinched inside a bent hinge (studio/measure/creases.py: near a knee, elbow, hip or
#           ankle bent past 60 degrees - the ankle from its rest angle, the others from straight - and
#           in the wedge between its two segments, or 4 mm past it on a 1 m character): normal in
#           deep bends;
#   defect  anything else that a camera the user looks at (the GIF's two, and each still's at its
#           frame) sees at least the limit's area of: the faces whose centre a ray from the camera
#           reaches with nothing more than SEEN_DEPTH in front of it (the skin and every other
#           rendered mesh, the bar too, can be in the way). The allowance is the fold's own top
#           layer: on the squat's belly the turned-over faces lie 1-2 mm under the skin folded back
#           over them, and that visible notch reads as 0 mm² seen if nothing may cover them;
#   hidden  anything else.

SEEN_DEPTH = 0.002      # x the character's height: how far in front of a face its own fold may lie
GIF_VIEWS = ((35.0, 10.0), (90.0, 5.0))     # animate.py --mpfb's GIF: 3/4 and the figure's left side
WELD_M = 1e-5           # vertices this close at rest are one point of the skin (the mesh is split at UV seams)


def _evaluated_mesh(skin):
    """(world vertex positions, the evaluated mesh, the evaluated object) as rendered now; call
    evaluated.to_mesh_clear() when done with the mesh."""
    evaluated = skin.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    co = np.empty(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", co)
    m = np.array(evaluated.matrix_world)
    return co.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3], mesh, evaluated


def _weld(points, tol):
    """One label per point; points within tol of each other (transitively) share it."""
    from mathutils.kdtree import KDTree
    parent = np.arange(len(points))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    kd = KDTree(len(points))
    for i, p in enumerate(points):
        kd.insert(p, i)
    kd.balance()
    for i, p in enumerate(points):
        for _, j, _ in kd.find_range(p, tol):
            a, b = root(i), root(j)
            if a != b:
                parent[max(a, b)] = min(a, b)
    return np.array([root(i) for i in range(len(points))])


class SkinAtRest:
    """The rendered skin in the bind pose (the rig's rest position): triangles, rest normals, each
    vertex's weights on the bones that deform it, and the welded vertex labels."""

    def __init__(self, rig, skin):
        held = rig.data.pose_position
        rig.data.pose_position = "REST"
        bpy.context.view_layer.update()
        try:
            rest, mesh, evaluated = _evaluated_mesh(skin)
            mesh.calc_loop_triangles()
            tris = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int64)
            mesh.loop_triangles.foreach_get("vertices", tris)
            deform = {b.name for b in rig.data.bones if b.use_deform}
            group_bone = {g.index: g.name for g in evaluated.vertex_groups if g.name in deform}
            self.bones = sorted(set(group_bone.values()))
            column = {name: i for i, name in enumerate(self.bones)}
            weights = np.zeros((len(mesh.vertices), len(self.bones)))
            for v in mesh.vertices:
                for g in v.groups:
                    if g.group in group_bone:
                        weights[v.index, column[group_bone[g.group]]] = g.weight
            evaluated.to_mesh_clear()
        finally:
            rig.data.pose_position = held
            bpy.context.view_layer.update()
        self.rig, self.skin, self.rest = rig, skin, rest
        self.tris = tris.reshape(-1, 3)
        total = weights.sum(axis=1, keepdims=True)
        self.weights = np.divide(weights, total, out=np.zeros_like(weights), where=total > 0)
        self.unweighted = total[:, 0] <= 0          # the Armature modifier leaves these where they are
        normal, area = _tri_normals(rest, self.tris)
        self.rest_normal, self.usable = normal, area > 1e-12
        self.label = _weld(rest, WELD_M)
        self.height = float(np.ptp(rest[:, 2]))     # standing in the bind pose

    def turned_over(self, co):
        """(mask of turned-over faces, posed area in m², posed world normals, expected normals) for
        the skin's posed vertices `co` (world)."""
        rig, pose, bones = self.rig, self.rig.pose.bones, self.rig.data.bones
        world = np.array(rig.matrix_world.to_3x3())
        unworld = np.linalg.inv(world)
        rot = np.stack([world @ np.array((pose[b].matrix @ bones[b].matrix_local.inverted()).to_3x3()) @ unworld
                        for b in self.bones]) if self.bones else np.zeros((0, 3, 3))
        per_vertex = (self.weights @ rot.reshape(len(rot), 9)).reshape(-1, 3, 3)
        per_vertex[self.unweighted] = np.eye(3)
        expect = np.einsum("fij,fj->fi", per_vertex[self.tris].sum(axis=1), self.rest_normal)
        normal, area = _tri_normals(co, self.tris)
        flipped = self.usable & (area > 1e-12) & (np.einsum("fi,fi->f", normal, expect) < 0)
        return flipped, area, normal, expect


def _tri_normals(co, tris):
    cross = np.cross(co[tris[:, 1]] - co[tris[:, 0]], co[tris[:, 2]] - co[tris[:, 0]])
    length = np.linalg.norm(cross, axis=1)
    return cross / np.maximum(length, 1e-30)[:, None], length / 2


def fold_patches(faces, tris, label):
    """Turned-over faces grouped into patches: faces sharing a (welded) corner are one patch."""
    parent = {int(f): int(f) for f in faces}

    def root(f):
        while parent[f] != f:
            parent[f] = parent[parent[f]]
            f = parent[f]
        return f

    corner = {}
    for f in faces:
        for v in label[tris[f]]:
            other = corner.setdefault(int(v), int(f))
            a, b = root(other), root(int(f))
            if a != b:
                parent[max(a, b)] = min(a, b)
    groups = {}
    for f in faces:
        groups.setdefault(root(int(f)), []).append(int(f))
    return list(groups.values())


CARRY_SHARE = 0.10      # a bone carries a patch when it has at least this share of the patch's skin weight


def nearest_joint(rig, names, point):
    """The bone (of `names`) whose head, the joint where it meets its parent, is nearest `point`."""
    world, pose = rig.matrix_world, rig.pose.bones
    return min(names, key=lambda n: (Vector(point) - world @ pose[n].head).length)


def fold_bone(rig, rest, faces, centre):
    """The bone a patch is named by: of the bones carrying it (CARRY_SHARE of its weight), the one whose
    joint is nearest its centre. Only carrying bones: in a back squat the hands sit on the shoulders,
    and the nearest joint of all to the deltoid is a knuckle."""
    share = rest.weights[np.unique(rest.tris[faces])].mean(axis=0)
    carrying = [rest.bones[i] for i in np.flatnonzero(share >= CARRY_SHARE)] or [rest.bones[int(np.argmax(share))]]
    return nearest_joint(rig, carrying, centre)


def hinges(rig):
    """creases.Hinge for each knee, elbow, ankle and hip, posed now, in world space (creases.hinges_from)."""
    world, pose, bones = rig.matrix_world, rig.pose.bones, rig.data.bones
    return hinges_from({pb.name: np.array(world @ pb.head) for pb in pose},
                       {b.name: np.array(b.head_local) for b in bones})


def gif_box(run, skin, first, last):
    """The box animate.py framed the GIF on: the run's frames/retarget.json contact box (the skin over
    the clip), or, without a readable one, that box measured again frame by frame."""
    try:
        report = json.loads((Path(run) / "frames" / "retarget.json").read_text())
        box = report.get("preview_box", report["contact"]["box"])
        box = np.array(box, dtype=float)
        if box.shape == (2, 3) and np.isfinite(box).all():
            return box
    except (TypeError, OSError, ValueError, KeyError):
        pass
    low, high = np.full(3, np.inf), np.full(3, -np.inf)
    for frame in range(first, last + 1):
        pts = figure_points(skin, frame)
        low, high = np.minimum(low, pts.min(axis=0)), np.maximum(high, pts.max(axis=0))
    return np.array([low, high])


def gif_cameras(box, yaw):
    """World positions of the GIF's cameras, as animate.py --mpfb places them around `box`: centred on
    it, at distance_for its height, at GIF_VIEWS. animate gives those azimuths in the house frame,
    where the figure faces +Y; here they are added to the figure's own yaw, so a figure turned any
    way is seen from the same sides."""
    low, high = np.asarray(box, dtype=float)
    centre = (low + high) / 2
    distance = mpfb_cage.distance_for(max(float(high[2] - low[2]), 0.5) * 1.05)
    return [centre + view_direction(yaw + azimuth, elevation) * distance for azimuth, elevation in GIF_VIEWS]


def rendered_meshes(skin):
    """The meshes the scene's view layer renders, the skin aside: the object not hidden from renders,
    in a collection that is, as are all its parents, not hidden from renders nor left out of the
    layer. A hidden helper (the MPFB cage, a guide) never hides a fold."""
    renders = set()

    def walk(layer, ok):
        ok = ok and not layer.exclude and not layer.collection.hide_render
        if ok:
            renders.add(layer.collection.name)
        for child in layer.children:
            walk(child, ok)

    walk(bpy.context.view_layer.layer_collection, True)
    return [o for o in bpy.context.scene.objects if o.type == "MESH" and o != skin and not o.hide_render
            and any(c.name in renders for c in o.users_collection)]


class Sight:
    """First hits along a line in world space, over the skin and every other rendered mesh: one BVH
    per object, built from its evaluated mesh in its own space."""

    def __init__(self, objects):
        depsgraph = bpy.context.evaluated_depsgraph_get()
        self.trees = []
        for obj in objects:
            m = obj.evaluated_get(depsgraph).matrix_world.copy()
            self.trees.append((BVHTree.FromObject(obj, depsgraph), m, m.inverted()))

    def first_hit(self, origin, target):
        """World distance from `origin` to the first surface on the way to `target` (a little past it
        too), or None."""
        best = None
        for tree, m, inverse in self.trees:
            a, b = inverse @ origin, inverse @ target
            d = b - a
            hit = tree.ray_cast(a, d.normalized(), d.length * 1.001 + 1e-6)
            if hit[0] is not None:
                distance = (m @ hit[0] - origin).length
                best = distance if best is None else min(best, distance)
        return best


def visible_area(sight, co, tris, area, faces, cameras, depth):
    """Area (m²) of the patch's faces some camera sees: the first thing on the line from it to the
    face's centre is no more than `depth` in front of that centre."""
    seen = set()
    for camera in cameras:
        origin = Vector(camera)
        for f in faces:
            if f in seen:
                continue
            target = Vector(co[tris[f]].mean(axis=0))
            hit = sight.first_hit(origin, target)
            if hit is None or hit >= (target - origin).length - depth:
                seen.add(f)
    return float(area[list(seen)].sum()) if seen else 0.0


def skin_folds(rig, skin, first, last, cameras):
    """{"height_m", "folds"}: every patch of turned-over skin over the limit, every frame, as
    {frame, bone (fold_bone), kind, area_mm2 (posed), visible_mm2, centre (world, m, to the mm)};
    `cameras(frame)` gives the positions the user sees the figure from at that frame. None when the
    skin cannot be compared with its bind pose (its rendered mesh changes topology)."""
    rest = SkinAtRest(rig, skin)
    limit = fold_limit(rest.height) / 1e6
    folds = []
    for frame in range(first, last + 1):
        bpy.context.scene.frame_set(frame)
        co, _, evaluated = _evaluated_mesh(skin)
        evaluated.to_mesh_clear()
        if co.shape != rest.rest.shape:
            log(f"folds: the skin has {len(co)} vertices at frame {frame}, {len(rest.rest)} at rest; not measured")
            return None
        flipped, area, _, _ = rest.turned_over(co)
        big = [f for f in fold_patches(np.flatnonzero(flipped), rest.tris, rest.label) if area[f].sum() > limit]
        if not big:
            continue
        sight = Sight([skin] + rendered_meshes(skin))
        hinge_list, views = hinges(rig), cameras(frame)
        for faces in big:
            a = float(area[faces].sum())
            centre = (co[rest.tris[faces]].mean(axis=1) * area[faces, None]).sum(axis=0) / a
            bone = fold_bone(rig, rest, faces, centre)
            seen = visible_area(sight, co, rest.tris, area, faces, views, SEEN_DEPTH * rest.height)
            crease = crease_of(hinge_list, bone, centre, CREASE_SLACK * rest.height)
            kind = "crease" if crease else "defect" if seen >= limit else "hidden"
            folds.append({"frame": frame, "bone": bone, "kind": kind, "area_mm2": round(a * 1e6, 1),
                          "visible_mm2": round(seen * 1e6, 1), "centre": [round(float(c), 3) for c in centre]})
    return {"height_m": round(rest.height, 4), "folds": folds}


def worst_folds(folds):
    """ours.json `folds_worst`: {bone: the largest patch's area, mm²}."""
    worst = {}
    for f in folds:
        worst[f["bone"]] = max(worst.get(f["bone"], 0.0), f["area_mm2"])
    return dict(sorted(worst.items(), key=lambda kv: -kv[1]))


# --- stills ---------------------------------------------------------------------------------------

def figure_points(skin, frame):
    """The skin's deformed vertices in world space at that frame."""
    bpy.context.scene.frame_set(frame)
    evaluated = skin.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    co = np.empty(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", co)
    evaluated.to_mesh_clear()
    m = np.array(evaluated.matrix_world)
    return co.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3]


def head_forward(rig):
    """The nose's direction in the Head bone's own frame, from the rest pose (where the figure's
    front is its hip line turned about up)."""
    bones = rig.data.bones
    left = bones[P + "LeftUpLeg"].head_local - bones[P + "RightUpLeg"].head_local
    left.z = 0.0
    front = left.normalized().cross(Vector((0.0, 0.0, 1.0)))
    return bones[P + "Head"].matrix_local.to_3x3().inverted() @ front


def facing_in(rig, camera, local_forward):
    """Which image side our nose is on ("left"/"right"), the Head bone projected through the camera;
    None when the face points too nearly at or away from the camera to tell."""
    scene, world, head = bpy.context.scene, rig.matrix_world, rig.pose.bones[P + "Head"]
    centre = world @ ((head.head + head.tail) / 2)
    forward = (world.to_3x3() @ head.matrix.to_3x3() @ local_forward).normalized()
    nose = centre + forward * max(head.length, 1e-3)
    dx = world_to_camera_view(scene, camera, nose).x - world_to_camera_view(scene, camera, centre).x
    if abs(dx) < FACING_MIN:
        return None
    return "right" if dx > 0 else "left"


def aim_camera(skin, frame, azimuth, elevation, framing=(0.0, 0.0, 1.0)):
    """The still camera, framing the figure's box at that frame from (azimuth, elevation). `framing`
    (across, up, scale; picture.FRAMINGS) moves the frame's centre by those fractions of the framed extent
    and scales its size: the still itself is (0, 0, 1)."""
    pts = figure_points(skin, frame)
    low, high = pts.min(axis=0), pts.max(axis=0)
    a = math.radians(azimuth)
    across_axis = np.array([math.cos(a), math.sin(a), 0.0])     # square to the view, level
    across = float(np.ptp(pts @ across_axis))   # width seen by this camera
    extent = max(float(high[2] - low[2]), across) * FRAME_MARGIN
    dx, dz, scale = framing
    centre = (low + high) / 2 + across_axis * (dx * extent) + UP * (dz * extent)
    camera = mpfb_cage.camera_on(Vector(centre), azimuth, elevation, mpfb_cage.distance_for(extent * scale))
    bpy.context.view_layer.update()     # the camera's matrix_world, read by facing_in, is current
    return camera


def render_still(skin, frame, azimuth, elevation, path, framing=(0.0, 0.0, 1.0)):
    """Frame the figure's box at that frame from (azimuth, elevation) and render one PNG."""
    camera = aim_camera(skin, frame, azimuth, elevation, framing)
    scene = bpy.context.scene
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    if not Path(path).is_file():
        raise RunError(f"Blender did not write the still {Path(path).name}.")
    return camera


# --- our picture, read as the coach's is (task 19; studio/measure/picture.py) --------------------------
# Our side view at each position's frame, in each of picture.FRAMINGS (the still itself first), read by the
# coach's own reader (read_pose.py, MediaPipe) in the Studio's Python: Blender's cannot import MediaPipe.
# The solver's picture loop (pipeline/blender/mpfb_cage.py `match_picture`) calls the same functions, so it
# aims at exactly what this step reads.

STUDIO_PYTHON = REPO / "studio" / ".venv" / "bin" / "python"
READ_POSE = REPO / "studio" / "measure" / "read_pose.py"
READ_TIMEOUT = 600      # seconds for every picture of a run (about 0.2 s each, and 3 s to load MediaPipe)


def read_pictures(files):
    """read_pose's result for each file, in order ({"image", "error"} where it found no body). Raises
    RunError (plain words) when the reader cannot run or does not answer for every file."""
    if not STUDIO_PYTHON.is_file():
        raise RunError(f"The Studio's Python is missing ({STUDIO_PYTHON}); run npm run setup.")
    try:
        done = subprocess.run([str(STUDIO_PYTHON), str(READ_POSE), *(str(f) for f in files)], cwd=str(REPO),
                              capture_output=True, text=True, timeout=READ_TIMEOUT)
    except subprocess.TimeoutExpired:
        raise RunError(f"MediaPipe took longer than {READ_TIMEOUT} s to read our pictures.") from None
    lines = [json.loads(line) for line in done.stdout.splitlines() if line.startswith("{")]
    if done.returncode != 0 or len(lines) != len(files):
        tail = (done.stderr.strip().splitlines() or ["no message"])[-1]
        raise RunError(f"MediaPipe could not read our pictures ({tail}).")
    return lines


def picture_at(skin, phase_frames, azimuth, elevation, folder, stills=None, log=None):
    """{phase: picture.summarize(...) with its "frame"}: our side view at each position's frame (the camera
    at azimuth, elevation, framed as the stills are) in each of picture.FRAMINGS, read with read_pose.
    `stills` {phase: path} reuses a still already rendered from that camera at that frame for the first
    framing; the others are rendered into `folder`."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    files = {}
    for phase, frame in phase_frames.items():
        files[phase] = []
        for n, framing in enumerate(FRAMINGS):
            if n == 0 and stills and stills.get(phase):
                files[phase].append(Path(stills[phase]))
                continue
            path = folder / f"{phase}-{n}.png"
            render_still(skin, frame, azimuth, elevation, path, framing)
            files[phase].append(path)
    order = [(phase, f) for phase, fs in files.items() for f in fs]
    results = read_pictures([f for _, f in order])
    out = {}
    for phase, frame in phase_frames.items():
        mine = [r for (p, _), r in zip(order, results) if p == phase]
        out[phase] = {"frame": frame, **summarize(mine)}
        if log:
            s = out[phase]
            log(f"picture: {phase} frame {frame}: " + (
                ", ".join(f"{m} {s['flat'][m]}" for m in PICTURE_METRICS) + f" ({s['seen']} of {s['reads']} framings seen)"
                if s["ok"] else f"no reading: {s['why']}"))
    return out


def picture_or_none(skin, phase_frames, azimuth, elevation, folder, stills=None, log=log):
    """picture_at, or every position "not read" with the reason when reading fails: the picture is one
    reading among many, and its rows then fall back to the rig, saying why (checks.py)."""
    try:
        return picture_at(skin, phase_frames, azimuth, elevation, folder, stills, log)
    except Exception as e:      # noqa: BLE001 - logged, and the rows say why
        why = str(e) if isinstance(e, RunError) else f"MediaPipe could not read our picture ({type(e).__name__}: {e})"
        log(f"picture not read: {why}")
        return {phase: {"frame": frame, **summarize([]), "why": why} for phase, frame in phase_frames.items()}


BIND_KEYS = ("hip_included", "knee_included", "torso_lean")
SEAT_MASK_SCALE = 2     # x the stills' size: the silhouette the seat and the knee's top are read on (render_mask)
SEAT_BONES = ("Hips", "LeftUpLeg", "RightUpLeg")                    # the skin of the seat (its largest weight's bone)
KNEE_BONES = ("LeftUpLeg", "RightUpLeg", "LeftLeg", "RightLeg")     # the skin a knee's top can be on, either leg's


def seat_picture(rig, skin, frame, side, side_camera, keep=False):
    """Our seat against the top of the knee at `frame` as the coach's is read (task 22; posture.seat_depth): the
    character's silhouette through the side camera framed as the still is (render_mask), our camera side's hip and
    knee joints projected on it, facing where the figure's front projects. {"picture": thigh lengths or None, "seat",
    "knee_top": [x, y] on the mask or None, "hip", "knee": the joints there (the mask is SEAT_MASK_SCALE x the still,
    so these over it fall on the still), "why"}; `keep`: also the "mask" and "facing" read."""
    camera = aim_camera(skin, frame, *side_camera)
    mask = render_mask(rig, SEAT_MASK_SCALE)
    j = joints_at(rig, frame)
    hip, knee = (to_pixels(camera, j[f"{k}.{side}"], mask.shape) for k in ("hip", "knee"))
    ahead = to_pixels(camera, j[f"hip.{side}"] + 0.1 * figure_front(j), mask.shape)
    facing = "right" if ahead[0] > hip[0] else "left"
    read = seat_depth(mask, hip, knee, facing)
    out = {"picture": None if read["depth"] is None else round(read["depth"], 3),
           "seat": read["seat"] and [int(v) for v in read["seat"]],
           "knee_top": read["knee_top"] and [int(v) for v in read["knee_top"]],
           "hip": [round(float(v), 1) for v in hip], "knee": [round(float(v), 1) for v in knee], "why": read["why"]}
    if keep:
        out.update(mask=mask, facing=facing)
    return out


def seat_skin(rig, skin, frame, side, side_normal):
    """The same depth read off the 3D skin at `frame`, independent of the picture (task 22, its cross-check): on the
    side camera's plane, the lowest point of the skin the hips and thighs carry (SEAT_BONES) from the hip joint back
    SEAT_BEHIND thigh lengths, against the highest point of the skin either leg's thigh or shin carries (KNEE_BONES)
    within KNEE_WINDOW of the knee joint, over the thigh's length on that plane, + = the seat lower; as on the picture,
    leg skin reaching KNEE_TORSO above the knee there is a thigh standing up, not a knee's top. Each vertex goes by
    its bone with the largest weight in the bind pose (_thigh_rest). {"skin": thigh lengths or None, "skin_why"}."""
    bones = _thigh_rest(rig, skin, side_normal)["bones"]
    j = joints_at(rig, frame)
    co, mesh, evaluated = _evaluated_mesh(skin)
    evaluated.to_mesh_clear()
    if len(co) != len(bones):
        return {"skin": None, "skin_why": "the skin's vertices are not the bind pose's"}
    n = np.asarray(side_normal, dtype=float) / np.linalg.norm(side_normal)
    up = UP - (UP @ n) * n
    up /= np.linalg.norm(up)
    front = figure_front(j)
    front = front - (front @ n) * n - (front @ up) * up
    front /= np.linalg.norm(front)
    hip, knee = j[f"hip.{side}"], j[f"knee.{side}"]
    thigh = knee - hip
    length = float(np.linalg.norm(thigh - (thigh @ n) * n))
    along, height = (co - hip) @ front, (co - hip) @ up
    names = np.array([b or "" for b in bones])
    seat = np.isin(names, [P + b for b in SEAT_BONES]) & (along <= 0.0) & (along >= -SEAT_BEHIND * length)
    knees = np.isin(names, [P + b for b in KNEE_BONES]) & (np.abs(along - thigh @ front) <= KNEE_WINDOW * length)
    if length < 1e-6 or not seat.any():
        return {"skin": None, "skin_why": "no skin of the seat behind the hip"}
    if not knees.any():
        return {"skin": None, "skin_why": "no skin of the legs at the knee"}
    top = float(height[knees].max())
    if top - float(thigh @ up) >= KNEE_TORSO * length:
        return {"skin": None, "skin_why": "no knee top: the leg's skin over the knee reaches a thigh length up"}
    return {"skin": round(float((top - height[seat].min()) / length), 3), "skin_why": None}


def seat_reading(rig, skin, frame, side, side_camera, side_normal, log=log):
    """{"frame", ...seat_picture, ...seat_skin}: the depth at `frame` read both ways, each on its own - whatever goes
    wrong reading one leaves it out with the reason, and the other stands (the Studio's row then says so)."""
    try:
        picture = seat_picture(rig, skin, frame, side, side_camera)
    except Exception as e:      # noqa: BLE001 - logged, and the row says why
        picture = {"picture": None, "seat": None, "knee_top": None, "hip": None, "knee": None,
                   "why": f"our silhouette could not be read ({type(e).__name__}: {e})"}
    try:
        on_skin = seat_skin(rig, skin, frame, side, side_normal)
    except Exception as e:      # noqa: BLE001 - logged, and the row says why
        on_skin = {"skin": None, "skin_why": f"the skin could not be read ({type(e).__name__}: {e})"}
    log(f"the seat below the top of the knee at frame {frame}: " + (
        f"{picture['picture']} thigh lengths on our picture (seat {picture['seat']}, knee top {picture['knee_top']})"
        if picture["picture"] is not None else f"not read on our picture ({picture['why']})") + "; " + (
        f"{on_skin['skin']} on the 3D skin" if on_skin["skin"] is not None
        else f"not read on the skin ({on_skin['skin_why']})"))
    return {"frame": frame, **picture, **on_skin}


def bind_reading(rig, skin, side_camera, side, side_normal, behind_normal, turn, folder, log=log):
    """The bind pose read two ways (task 20): ours (measure_frame on its joints, the torso and legs turned as on
    every frame) and MediaPipe's, on a render of it from the side camera framed as the stills are - the
    cross-check of the thigh's skin outline, which the legs are counted from (a person standing as the cartoon's
    bind pose stands reads about the same in both). {"ours": {...}, "mediapipe": {..., "knee_seen"} or {"why"}}."""
    ours = measure_frame(rest_joints(rig), side, side_normal, behind_normal, turn, leg_turns(rig, side_normal, skin))
    out = {"ours": {k: ours["flat"][k] for k in BIND_KEYS}, "warning": None}
    held = rig.data.pose_position
    try:
        rig.data.pose_position = "REST"
        bpy.context.view_layer.update()
        path = Path(folder) / "bind.png"
        render_still(skin, bpy.context.scene.frame_start, *side_camera, path)
        read = read_pictures([path])[0]
        if read.get("error"):
            out["mediapipe"] = {"why": read["error"]}
        else:
            pixels = read.get("pixels") or {}
            slope = picture_slope(pixels[f"{side}_hip"], pixels[f"{side}_knee"], read.get("facing")) \
                if f"{side}_hip" in pixels and f"{side}_knee" in pixels else None
            out["mediapipe"] = {**{k: read["flat"].get(k) for k in BIND_KEYS},
                                "thigh_slope": None if slope is None else round(slope, 3),
                                "knee_seen": (read.get("visibility") or {}).get(f"{side}_knee")}
    except Exception as e:      # noqa: BLE001 - a cross-check only: logged, the run stands
        out["mediapipe"] = {"why": str(e) if isinstance(e, RunError) else f"{type(e).__name__}: {e}"}
    finally:
        rig.data.pose_position = held
        bpy.context.view_layer.update()
    mp = out["mediapipe"]
    out["warning"] = bind_legs_warning(leg_slopes(rig, side_normal, skin)["outline"], mp, side)
    if out["warning"]:
        log("WARNING: " + out["warning"])
    log("bind pose read as ours: " + ", ".join(f"{k} {out['ours'][k]}" for k in BIND_KEYS) + "; by MediaPipe on its "
        "render: " + (", ".join(f"{k} {mp[k]}" for k in BIND_KEYS) + f", thigh {mp['thigh_slope']} degrees from "
                      f"straight down (knee seen at {mp['knee_seen']})"
                      if "why" not in mp else f"not read ({mp['why']})"))
    return out


# --- the run --------------------------------------------------------------------------------------

def _read_json(path, what):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        raise RunError(f"The run has no {what} ({Path(path).name}).") from None
    except ValueError:
        raise RunError(f"The run's {what} ({Path(path).name}) is not readable.") from None


def effective_view(image, measured):
    """The user's view if they set one, else the detector's, else side."""
    view = image.get("view") if image.get("viewSetBy") == "user" else None
    if not view:
        view = (measured.get(image["id"]) or {}).get("view")
    if view not in KNOWN_VIEWS:
        if view:
            log(f"{image['id']}: unknown view {view!r}, using the side camera")
        view = "side"
    return view


def studio_view(rig, side):
    """(yaw, side camera (azimuth, elevation), side camera's axis, behind camera's axis, torso turn): how the
    Studio sees the figure, from its hips at the scene's first frame (see the module's docstring)."""
    yaw = figure_yaw(joints_at(rig, bpy.context.scene.frame_start))
    azimuth, elevation, _ = camera_for("side", side, yaw)
    side_normal = view_direction(azimuth, elevation)
    behind_normal = view_direction(*camera_for("behind", side, yaw)[:2])
    return yaw, (azimuth, elevation), side_normal, behind_normal, torso_turn(rig, side_normal)


def measure_frames(rig, first, last, side, side_normal, behind_normal, turn):
    """([{"frame", "flat", "3d"}] of measure_frame, {frame: joints}) for every frame: the joints' readings
    the positions are picked by (posture is added by inspect); the torso turned by `turn`, the legs by the
    rig's own leg_turns."""
    per_frame, joints_by_frame = [], {}
    legs = leg_turns(rig, side_normal, skin_of(rig))
    for frame in range(first, last + 1):
        joints_by_frame[frame] = joints_at(rig, frame)
        per_frame.append({"frame": frame, **measure_frame(joints_by_frame[frame], side, side_normal, behind_normal,
                                                          turn, legs)})
    return per_frame, joints_by_frame


def pick_positions(per_frame, targets, joints_by_frame):
    """(phase frames, driver, jumps, steps): the coach's positions on our clip, as metrics picks them, never
    on a frame that jumps. RunError (plain words) when none can be picked."""
    try:
        jump_set = jump_frames(per_frame, targets, joints_by_frame)
        phase_frames = pick_phase_frames(per_frame, targets, skip=jump_set)
        driver = phase_driver(per_frame, targets)
        return phase_frames, driver, jump_sizes(joints_by_frame, jump_set), jump_steps(joints_by_frame)
    except ValueError as e:     # metrics says why in plain words (no position measured; every frame jumps)
        raise RunError(str(e)) from None


def chest_tilt(rig, side_normal):
    """Degrees the bind pose's chest - the Spine bone's head to the Neck bone's - tips forward (+) or back
    (-) from upright, seen on the side camera's plane; None when the rig lacks those bones."""
    bones, world = rig.data.bones, rig.matrix_world
    if P + "Spine" not in bones or P + "Neck" not in bones:
        return None
    joints = rest_joints(rig)
    front = figure_front(joints)
    if front is None:
        return None
    low, high = (np.array(world @ bones[P + n].head_local) for n in ("Spine", "Neck"))
    return _rest_lean({"hip.left": low, "hip.right": low, "shoulder.left": high, "shoulder.right": high},
                      UP, side_normal, front)


def bind_pose(rig, side):
    """(the bind pose's rest lean, its chest's rest tilt, metrics.bind_pose_warning's words or None), seen on the
    Studio's side camera for that side: the one rule, said by this step and by the solver."""
    side_normal = studio_view(rig, side)[2]
    lean, chest = rest_lean(rig, side_normal), chest_tilt(rig, side_normal)
    return lean, chest, bind_pose_warning(lean, chest)


def inspect(run):
    scene_path = run / "scene.blend"
    if not bpy.data.filepath or Path(bpy.data.filepath).resolve() != scene_path.resolve():
        if not scene_path.is_file():
            raise RunError("The run has no saved scene (scene.blend): the animate step did not finish.")
        bpy.ops.wm.open_mainfile(filepath=str(scene_path))
    targets = _read_json(run / "targets.json", "targets")
    snapshot = _read_json(run / "references.json", "reference list")
    measured = _read_json(run / "measurements.json", "measurements") if (run / "measurements.json").is_file() else {}
    side = (targets.get("camera") or {}).get("side")
    if side not in SIDE_TURN:
        raise RunError("The targets do not say which side the reference was photographed from.")

    scene = bpy.context.scene
    first, last = scene.frame_start, scene.frame_end
    rig, skin = find_rig_and_skin()
    log(f"rig {rig.name}, skin {skin.name}, frames {first}-{last}, measured side {side}")

    yaw, side_camera, side_normal, behind_normal, turn = studio_view(rig, side)
    local_forward = head_forward(rig)
    lean = rest_lean(rig, side_normal)
    log(f"torso line at rest (the bind pose, counted as upright): leans {lean:+.1f} degrees seen from the side")
    chest = chest_tilt(rig, side_normal)
    log(f"chest at rest (Spine to Neck): tips {chest:+.1f} degrees" if chest is not None else "chest at rest: no Spine or Neck bone")
    warning = bind_pose_warning(lean, chest)
    if warning:
        log(warning)
    slopes = leg_slopes(rig, side_normal, skin)
    log("legs at rest (the bind pose), degrees forward of straight down seen from the side: joint lines " + ", ".join(
        f"{k} {v:+.1f}" for k, v in slopes["joints"].items()) + "; the thighs' skin outlines (counted from) " + ", ".join(
        f"{k} " + ("not found: read on its joints" if v is None else f"{v:+.1f}") for k, v in slopes["outline"].items()))
    try:
        ear_verts = ear_vertices(rig, skin)
        co = np.array([skin.matrix_world @ v.co for v in skin.data.vertices])
        ears = {side: co[verts].mean(axis=0) for side, verts in ear_verts.items()}
        log("ears found on the skin: " + ", ".join(f"{side} {len(v)} vertices about "
                                                     f"{[round(float(c), 3) for c in ears[side]]}"
                                                     for side, v in ear_verts.items()))
    except RunError as e:       # the neck then reads "not measured"
        log(f"neck not measured: {e}")
        ear_verts, ears = None, None
    spine_rest = rest_spine(rig, side_normal)
    log(f"spine at rest (counted as 0): bent {spine_rest} degrees seen from the side")
    per_frame, joints_by_frame = measure_frames(rig, first, last, side, side_normal, behind_normal, turn)
    for entry in per_frame:
        frame = entry["frame"]
        bpy.context.scene.frame_set(frame)
        face = head_at(rig, local_forward)[1]
        posture = posture_frame(joints_by_frame[frame], None if ears is None else ears_at(rig, ears), face, side_normal,
                                turn)
        for method in ("flat", "3d"):
            entry[method].update(posture[method])
        entry["flat"].update(_clean({"spine_curl": spine_curl(rig, side_normal, spine_rest)}))
    phase_frames, driver, jumps, steps = pick_positions(per_frame, targets, joints_by_frame)
    log(f"figure faces azimuth {yaw:.1f}; phase frames {phase_frames} (timed by {driver})")
    if steps:
        log("the pose snaps on frames " + ", ".join(f"{s['frame']} ({s['mm']} mm)" for s in steps))
    if jumps:
        log("the pose jumps on frames " + ", ".join(f"{j['frame']} (hips {j['hips_mm']} mm, knees {j['knees_mm']} mm)"
                                                    for j in jumps) + "; not picked as positions")

    mpfb_cage.preview_set(560, 560)
    stills_dir = run / "stills"
    stills_dir.mkdir(parents=True, exist_ok=True)
    keys = [i for i in snapshot.get("images") or [] if not i.get("contextOnly") and not i.get("deleted")]
    stills, still_cameras, side_stills = [], {}, {}
    for n, image in enumerate(keys, 1):
        phase = image.get("phase")
        if phase not in phase_frames:
            raise RunError(f"Image {image.get('original') or image['id']} has no position (start, middle or end).")
        view = effective_view(image, measured)
        image_side = (measured.get(image["id"]) or {}).get("camera_side")
        image_side = image_side if image_side in SIDE_TURN else side
        azimuth, elevation, cam_view = camera_for(view, image_side, yaw)
        frame, rel = phase_frames[phase], f"stills/{image['id']}.png"
        log(f"still {n}/{len(keys)}: {image['id']} {phase} frame {frame}, {cam_view} camera "
            f"(azimuth {azimuth:.0f}, elevation {elevation:.0f})")
        camera = render_still(skin, frame, azimuth, elevation, run / rel)
        still_cameras.setdefault(frame, []).append(np.array(camera.location))
        facing = facing_in(rig, camera, local_forward) if cam_view == "side" else None
        stills.append({"image": image["id"], "phase": phase, "view": cam_view, "frame": frame,
                       "file": rel, "facing": facing})
        if cam_view == "side" and (azimuth, elevation) == side_camera:
            side_stills.setdefault(phase, run / rel)

    # ours read from the picture as the coach is, at every position, from the side camera of the coach's
    # photos (the stills reused as the first framing); the torso rows are judged on it (checks.py)
    with tempfile.TemporaryDirectory(prefix="picture-") as folder:
        picture = picture_or_none(skin, phase_frames, *side_camera, folder, side_stills)
        bind = bind_reading(rig, skin, side_camera, side, side_normal, behind_normal, turn, folder)
    # the depth as a judge sees it (task 22): the seat against the top of the knee at the bottom, read off our
    # silhouette as the coach's is and off the 3D skin - for an exercise whose kit row says so
    seat = {"end": seat_reading(rig, skin, phase_frames["end"], side, side_camera, side_normal)} \
        if has_depth(targets.get("exercise"), targets.get("equipment")) else None

    # the folds, judged by what the user sees: the GIF's cameras on every frame, each still's on its own.
    # They are one check among many: whatever goes wrong measuring them, the stills and the rest stand.
    try:
        gif = gif_cameras(gif_box(run, skin, first, last), yaw)
        measured_folds = skin_folds(rig, skin, first, last, lambda frame: gif + still_cameras.get(frame, []))
    except Exception as e:      # noqa: BLE001 - logged, and the check then says "not measured"
        log(f"folds not measured: {type(e).__name__}: {e}")
        measured_folds = None
    ours = {"first": first, "last": last, "camera_side": side,
            "ears": None if ears is None else {**{s: [round(float(c), 4) for c in p] for s, p in ears.items()},
                                               "count": [len(ear_verts["left"]), len(ear_verts["right"])]},
            "spine_rest": None if spine_rest is None else round(spine_rest, 3),
            "torso_rest_lean": round(lean, 3),
            "legs_at_rest": {"joints": {k: round(v, 3) for k, v in slopes["joints"].items()},
                             "outline": {k: (None if v is None else round(v, 3)) for k, v in slopes["outline"].items()},
                             "read": bind, "warning": bind.get("warning")},
            **({} if seat is None else {"seat_below_knee": seat}),
            "per_frame": per_frame,
            "picture": {**picture, "framings": [list(f) for f in FRAMINGS],
                        "camera": {"azimuth": round(side_camera[0], 2), "elevation": round(side_camera[1], 2)}},
            "phase_frames": phase_frames, "phase_driver": driver, "jumps": jumps, "steps": steps, "arm": arm_lengths(rig), "stills": stills}
    if measured_folds is not None:      # not measured: no keys, and the check says so
        folds = measured_folds["folds"]
        ours.update(height_m=measured_folds["height_m"], folds=folds, folds_worst=worst_folds(folds))
        kinds = {k: [f for f in folds if f["kind"] == k] for k in ("defect", "hidden", "crease")}
        log(f"skin turned over (over {fold_limit(measured_folds['height_m']):.0f} mm²): "
            + "; ".join(f"{k} " + ", ".join(f"{b} up to {a} mm²" for b, a in worst_folds(v).items())
                        for k, v in kinds.items() if v) if folds else "no skin turned over beyond the limit")
    text = ours_text(ours)
    tmp = run / "ours.json.tmp"
    tmp.write_text(text)
    tmp.replace(run / "ours.json")
    log(f"wrote ours.json ({len(per_frame)} frames) and {len(stills)} stills")


def ours_text(ours):
    """ours.json's text: indented, but one line per fold record (a squat has over a thousand)."""
    text = json.dumps({**ours, "folds": "__folds__"} if ours.get("folds") else ours, indent=1, allow_nan=False)
    if not ours.get("folds"):
        return text
    lines = ",\n  ".join(json.dumps(f, allow_nan=False) for f in ours["folds"])
    return text.replace('"__folds__"', "[\n  " + lines + "\n ]", 1)


def main(argv=None):
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="our measurements and comparison stills for a run")
    parser.add_argument("--run", required=True, help="the run folder (holds scene.blend, targets.json, ...)")
    args = parser.parse_args(argv)
    try:
        inspect(Path(args.run).resolve())
    except RunError as e:
        print(str(e), file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
