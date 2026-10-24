#!/usr/bin/env python
"""Wear an image-to-3D mesh as a SKIN over the clean character cage, in Blender.

    blender --background --factory-startup --python pipeline/blender/skin_wrap.py -- \
        --model "3d-character/3d cartoon boy.glb" --out /tmp/wrap/boy

Why this file exists. `character.py` builds a body that bends: all quads, loops at every joint,
weights assigned as each ring is made. It looks like a mannequin. The Tripo models look like the
character and cannot bend: triangle soup, three in four vertices duplicated along UV seams, and
either no skeleton or one whose weights are degenerate. A night of weighting work on the Tripo mesh
itself ended in cracks between duplicated vertices that no edge-based check could see.

So the Tripo mesh is not rigged at all here. It is bound to the cage with Surface Deform and the
cage is what the rig moves. Surface Deform places every skin vertex from the cage's local surface
frame at that vertex's own position, so two skin vertices at the same point get the same answer
whether or not an edge joins them: the crack cannot happen by construction. It is measured anyway.

Three steps, one function each:

    fit_cage_to(skin_path) -> (cage, rig, proportions)
        Measure the skin with `fit_model` and build a cage of the same proportions, posed to the
        skin's own rest pose. Does not touch the skin: it measures a copy it loads itself.

    region_fit(cage, skin) -> cage
        Pull the cage onto the skin's real surface, one body region at a time, without letting a
        finger fall onto a fingerless mitten or a thigh onto the other leg. Hands and face stay
        exactly as built.

    wrap(cage, rig, skin) -> modifier
        Triangulate the cage with a modifier (the shipped cage stays quads), bind, and RAISE if
        the bind did not take.

And one more, for anything that leaves Blender:

    bake_to_skin(cage, rig, skin) -> skin
        Turn the wrapped skin into an ORDINARY skinned mesh: bone weights fitted to what Surface
        Deform does over a spread of poses, garments then given the weights of the body under
        them, an Armature modifier on the rig, parented to it, Surface Deform removed. Surface
        Deform does not survive USD export as skeletal animation (task 7: the file carries baked
        per-frame points and a skeleton that drives nothing, and SceneKit plays it as a still
        figure). An Armature-skinned mesh does. Measured against the Surface Deform it replaces
        and RAISES if it strays; see the function.

`fit_cage_to` has to be called first: `character.build` clears the whole scene before it builds.
"""

import argparse
import dataclasses
import json
import math
import sys
import tempfile
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import character  # noqa: E402  — Proportions, build(), and the house camera/montage helpers
import fit_model  # noqa: E402  — face_forward, decimate, measure, joints_from
import make_character  # noqa: E402  — BONES, use_cycles


# --- loading -------------------------------------------------------------------------------------

def _world_box(obj):
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    return (Vector([min(c[i] for c in corners) for i in range(3)]),
            Vector([max(c[i] for c in corners) for i in range(3)]))


def load_skin(path, name="skin"):
    """Import a model as ONE plain mesh in world space, without disturbing the scene.

    This is `fit_model.load` with two differences, and both are forced:

      * it does not delete everything first. `fit_model.load` starts with select-all + delete,
        which would take the cage with it.
      * it does not join every mesh in the file. "new man model.glb" carries a stray 42-vertex
        `Icosphere`, two metres across and centred on the origin, beside the character.
        `fit_model.load` would join it in, the silhouette would be a disc and every landmark
        would be measured off it. A mesh is kept only if it sits inside the largest mesh's box.

    The Mixamo export's skeleton and skin weights are thrown away. Its `JOINTS_0`/`WEIGHTS_0`
    import as 55 vertex groups that bind every vertex to one bone at weight 1.0; nothing here may
    read them, so nothing here gets the chance — the groups are deleted, the armature with them.
    Only the rest-pose geometry is used. The model is then turned to face +Y by
    `fit_model.face_forward`, so it sits in exactly the frame `fit_model` measures in.
    """
    before = {o.name for o in bpy.data.objects}
    suffix = Path(path).suffix.lower()
    if suffix in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path))
    elif suffix == ".obj":
        bpy.ops.wm.obj_import(filepath=str(path))
    else:
        raise SystemExit(f"don't know how to import {suffix!r}")
    new = [o for o in bpy.data.objects if o.name not in before]
    meshes = [o for o in new if o.type == "MESH" and len(o.data.vertices)]
    if not meshes:
        raise SystemExit(f"no mesh in {path}")
    main = max(meshes, key=lambda o: len(o.data.vertices))
    lo, hi = _world_box(main)
    pad = (hi - lo).length * 0.05
    keep = [main]
    for other in meshes:
        if other is main:
            continue
        a, b = _world_box(other)
        if all(a[i] >= lo[i] - pad and b[i] <= hi[i] + pad for i in range(3)):
            keep.append(other)
    dropped = sorted(o.name for o in meshes if o not in keep)
    for obj in keep:
        if obj.data.users > 1:
            obj.data = obj.data.copy()
        world = obj.matrix_world.copy()
        obj.parent = None
        obj.data.transform(world)
        obj.matrix_world = Matrix.Identity(4)
        for mod in list(obj.modifiers):
            obj.modifiers.remove(mod)
        obj.vertex_groups.clear()
        if obj.data.shape_keys:
            obj.shape_key_clear()
    for gone in [o.name for o in new if o not in keep]:
        bpy.data.objects.remove(bpy.data.objects[gone], do_unlink=True)
    if len(keep) > 1:
        with bpy.context.temp_override(active_object=main, object=main,
                                       selected_objects=keep, selected_editable_objects=keep):
            bpy.ops.object.join()
    main.name = name
    main.data.update()
    fit_model.face_forward(main)
    main["dropped_on_load"] = ", ".join(dropped)
    if dropped:
        print(f"load_skin: left out {dropped} — not inside the character's own bounds")
    return main


def _co(mesh):
    out = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", out)
    return out.reshape(-1, 3)


def _span(values, lo=2, hi=98):
    a, b = np.percentile(values, [lo, hi])
    return float(a), float(b)


# --- measuring the skin --------------------------------------------------------------------------
#
# The landmarks are `fit_model`'s: floor, crown, crotch, neck and the T-pose arm band come off its
# silhouette, and its `joints_from` places hip, knee, ankle, elbow, wrist and fingertip. What the
# front silhouette CANNOT give — depth, limb thickness, how far the arm droops, how far apart the
# feet stand — is read off the vertices in thin slices, because the cage needs every one of those
# numbers and a default in their place is a number nobody measured.

def _section(co, z, cx, reach, height):
    """Half width, half depth and centre of the body in a thin slice at height `z`."""
    s = co[(np.abs(co[:, 2] - z) <= 0.006 * height) & (np.abs(co[:, 0] - cx) <= reach)]
    if len(s) < 12:
        return None
    x0, x1 = _span(s[:, 0])
    y0, y1 = _span(s[:, 1])
    return {"half": (x1 - x0) / 2, "depth": (y1 - y0) / 2, "x": (x0 + x1) / 2,
            "y": (y0 + y1) / 2, "n": int(len(s))}


def _bins(key, a, b, lo, hi, bins=10):
    """Slices across a limb: (slice centre, middle of a's extent, middle of b's, mean half-extent).

    The middle of each slice's EXTENT, not the mean of its vertices: a Tripo mesh puts far more
    vertices on a detailed face of a limb than on a plain one, and a mean drifts toward detail.
    """
    rows = []
    edges = np.linspace(lo, hi, bins + 1)
    for e0, e1 in zip(edges, edges[1:]):
        k = (key >= e0) & (key < e1)
        if k.sum() < 12:
            continue
        a0, a1 = _span(a[k])
        b0, b1 = _span(b[k])
        rows.append(((e0 + e1) / 2, (a0 + a1) / 2, (b0 + b1) / 2, ((a1 - a0) + (b1 - b0)) / 4))
    if len(rows) < 3:
        raise SystemExit(f"too few slices across a limb ({len(rows)}) — is this a T-pose model?")
    return np.array(rows)


def _arm(co, sign, cx, s_lo, s_wrist, z_guess, height):
    """A T-pose arm's centre line, z(s) and y(s), and its radius along s (outward distance).

    Two passes. The LINE comes from the middle of the arm only, 45-95% of the way to the wrist,
    where nothing else is at that height: the first version sliced from the chest outward inside
    a +-15% band and the slices near the shoulder took in the ears and the side of the vest, which
    tilted the arm 11 degrees forward — its hand ended up in front of the chin. The THICKNESS is
    then read in a tube around that line, all the way in to the shoulder.
    """
    s = sign * (co[:, 0] - cx)
    mid = (s >= 0.45 * s_wrist) & (s <= 0.95 * s_wrist) & (np.abs(co[:, 2] - z_guess) <= 0.10 * height)
    rows = _bins(s[mid], co[mid, 2], co[mid, 1], 0.45 * s_wrist, 0.95 * s_wrist)
    fz = np.polyfit(rows[:, 0], rows[:, 1], 1)
    fy = np.polyfit(rows[:, 0], rows[:, 2], 1)
    dz = co[:, 2] - np.polyval(fz, s)
    dy = co[:, 1] - np.polyval(fy, s)
    tube = (s >= s_lo) & (s <= s_wrist) & (np.hypot(dz, dy) <= 0.07 * height)
    thick = _bins(s[tube], dz[tube], dy[tube], s_lo, s_wrist, bins=12)
    return {"rows": thick, "fit1": fz, "fit2": fy,
            "radius": lambda t, r=thick: float(np.interp(t, r[:, 0], r[:, 3]))}


def _leg(co, sign, cx, z_lo, z_hi, height):
    """A leg's centre line, s(z) and y(z), and its radius along z, between ankle and crotch."""
    s = sign * (co[:, 0] - cx)
    pick = (s > 0.004 * height) & (co[:, 2] >= z_lo) & (co[:, 2] <= z_hi)
    rows = _bins(co[pick, 2], s[pick], co[pick, 1], z_lo, z_hi)
    return {"rows": rows, "fit1": np.polyfit(rows[:, 0], rows[:, 1], 1),
            "fit2": np.polyfit(rows[:, 0], rows[:, 2], 1),
            "radius": lambda t, r=rows: float(np.interp(t, r[:, 0], r[:, 3]))}


def measure_skin(path, scratch):
    """Load a measuring copy of the skin and read everything the cage's proportions need."""
    body = load_skin(path, name="measure")
    fit_model.decimate(body)
    shot = Path(scratch) / f"{Path(path).stem.replace(' ', '-')}-silhouette.png"
    # `fit_model.measure` renders whatever is in the scene; `fit_model.load` could count on an
    # empty one and this cannot. The first run here measured Blender's default cube: a solid
    # grey frame, read without complaint as a figure with a head 6% of its height.
    # (By name: `fit_model.silhouette` deletes every camera and lamp in the scene on its way in.)
    others = [o.name for o in bpy.data.objects if o is not body and not o.hide_render]
    for name in others:
        bpy.data.objects[name].hide_render = True
    try:
        m = fit_model.measure(body, shot)
    finally:
        for name in others:
            if name in bpy.data.objects:
                bpy.data.objects[name].hide_render = False
    co = _co(body.data)
    span = float(co[:, 2].max() - co[:, 2].min())
    if abs(m["height"] - span) > 0.02 * span:
        raise SystemExit(f"the silhouette is {m['height']:.3f} tall and the mesh {span:.3f}: "
                         f"something other than the model was in the measuring render")
    try:
        joints = fit_model.joints_from(m)
    except NameError as err:
        # fit_model.joints_from's arms-down branch reads `arm_x`, which nothing defines. Only a
        # T-pose model gets through it. Said plainly rather than let a NameError escape.
        raise SystemExit("fit_model.joints_from cannot measure a model that is not in a T-pose "
                         f"(its arms-down branch fails: {err}). Only T-pose skins are supported.")
    out = {"m": m, "joints": joints, "co": co, "silhouette": str(shot),
           "dropped": body.get("dropped_on_load", "")}
    bpy.data.objects.remove(body, do_unlink=True)
    return out


# The shoulder joint keeps the cage's own shoulder-to-chest ratio (read from `Proportions`'
# defaults in `proportions_from`, not copied here), because that is where the cage's TOPOLOGY
# depends on it: the arm socket is cut into the side of the chest, so a shoulder joint inside the
# chest would grow the arm's first rings inside the torso and cross the socket's bridge over
# itself. `fit_model` places a T-pose shoulder at 0.92 of the WAIST half-width, which on both
# test models is inside the chest.
NECK_MIN_F = 0.025                          # a neck shorter than this has no rings to bend with
SPLAY_RANGE = (0.0, 12.0)


def proportions_from(meas, rest="skin"):
    """A `character.Proportions` sized to the skin, plus where to stand it and how to pose it."""
    m, j, co = meas["m"], meas["joints"], meas["co"]
    height, low = m["height"], m["low"]
    d = character.Proportions()
    shoulder_per_chest = d.shoulder_half_f / d.chest_half_f
    notes = []

    hip_z = j["hip.right"][2]
    neck_z = j["neck"][2]
    knee_z = j["knee.right"][2]
    ankle_z = j["ankle.right"][2]
    shoulder_guess = j["shoulder.right"][2]

    # where the body's centre line is: the waist, which nothing else crosses in a T-pose
    waist = _section(co, hip_z + 0.33 * (shoulder_guess - hip_z), 0.0, 0.25 * height, height)
    cx = waist["x"]
    waist = _section(co, hip_z + 0.33 * (shoulder_guess - hip_z), cx, 0.25 * height, height)
    cy = waist["y"]

    # the arms: a line through slices of each, from just outside the chest to the wrist
    chest_probe = _section(co, hip_z + 0.70 * (shoulder_guess - hip_z), cx, 0.25 * height, height)
    s_wrist = abs(j["wrist.right"][0])
    s_elbow = abs(j["elbow.right"][0])
    s_tip = abs(j["hand.right"][0])
    arms = {side: _arm(co, sign, cx, chest_probe["half"] * 1.15, s_wrist, shoulder_guess, height)
            for side, sign in (("left", -1.0), ("right", 1.0))}
    z_slope = float(np.mean([arms[s]["fit1"][0] for s in arms]))
    y_slope = float(np.mean([arms[s]["fit2"][0] for s in arms]))
    stretch = math.sqrt(1.0 + z_slope ** 2 + y_slope ** 2)     # length along the arm per unit s

    def arm_z(s):
        return float(np.mean([np.polyval(arms[side]["fit1"], s) for side in arms]))

    def arm_r(s):
        return float(np.mean([arms[side]["radius"](s) for side in arms]))

    # The chest is measured below the armpit, never through it: a T-pose row through the
    # shoulders is one run from fingertip to fingertip.
    armpit_z = arm_z(chest_probe["half"] * 1.25) - arm_r(chest_probe["half"] * 1.25)
    shoulder_drop = d.shoulder_drop_f * height

    def torso_section(t, torso_len):
        z = min(hip_z + t * torso_len, armpit_z - 0.01 * height)
        out = _section(co, z, cx, 0.25 * height, height)
        out["z"] = z
        return out

    # the shoulder joint: the measured chest width times the cage's own ratio, or fit_model's
    # number if that is wider. Iterated once because the chest height depends on the torso length.
    torso_len = shoulder_guess + shoulder_drop - hip_z
    for _ in range(2):
        chest = torso_section(0.75, torso_len)
        s_shoulder = max(abs(j["shoulder.right"][0]), chest["half"] * shoulder_per_chest)
        shoulder_z = arm_z(s_shoulder)
        torso_len = shoulder_z + shoulder_drop - hip_z
    if s_shoulder > abs(j["shoulder.right"][0]) + 1e-9:
        notes.append(f"shoulder joint moved out from fit_model's {abs(j['shoulder.right'][0]):.3f} "
                     f"to {s_shoulder:.3f} (chest half-width {chest['half']:.3f} x "
                     f"{shoulder_per_chest:.3f}) so the arm grows out of the chest, not inside it")
    pelvis = torso_section(0.0, torso_len)
    waist = torso_section(0.33, torso_len)
    seat = torso_section(0.24, torso_len)
    front = torso_section(0.80, torso_len)

    # How far BEHIND the waist the shoulder joints are. On both test models the arm's root sits
    # 3.5-4.5cm behind the middle of the waist — a cartoon posture, belly forward and shoulders
    # back — and every joint the cage has sits on one plane. Built on the waist's plane, the
    # cage's shoulder was 4cm in front of the skin's, the cage's arm stuck out of the front of the
    # skin's deltoid, and the skin's shoulder top, 4cm behind any cage surface, was flung up into
    # a fin when the arm came down. So the cage stands on the waist and its spine is POSED back
    # until the shoulder joints are where the skin's are (`fit_cage_to`).
    shoulder_y = []
    for side in arms:
        rows = arms[side]["rows"]
        near = (rows[:, 0] >= s_shoulder) & (rows[:, 0] <= s_shoulder + 0.06 * height)
        shoulder_y += list(np.polyval(arms[side]["fit2"], rows[near, 0]) + rows[near, 2])
    shoulder_y = float(np.mean(shoulder_y))

    def lean_line(z):
        return cy + (shoulder_y - cy) * (z - hip_z) / (shoulder_z - hip_z)

    chest_z = shoulder_z + shoulder_drop
    neck_f = (neck_z - chest_z) / height
    if neck_f < NECK_MIN_F:
        notes.append(f"neck measured {neck_f:+.3f} of height (fit_model's T-pose neck row sits on "
                     f"the shoulder line); held at {NECK_MIN_F} and taken out of the head")
        neck_f = NECK_MIN_F
    leg_f = (hip_z - low) / height
    torso_f = (chest_z - hip_z) / height
    head_f = 1.0 - leg_f - torso_f - neck_f
    if head_f < 0.15:
        raise SystemExit(f"the head came out at {head_f:.3f} of height — the landmarks are wrong")

    upper = (s_elbow - s_shoulder) * stretch
    fore = (s_wrist - s_elbow) * stretch
    hand = (s_tip - s_wrist) * stretch
    hand_scale = (hand / height) / d.hand_f

    # the legs: centre lines through slices between the ankle and the crotch
    crotch_z = m["z_of"](m["crotch"]) if m["crotch"] is not None else hip_z - 0.07 * height
    legs = {side: _leg(co, sign, cx, ankle_z + 0.04 * height, crotch_z - 0.03 * height, height)
            for side, sign in (("left", -1.0), ("right", 1.0))}

    def leg_s(z):
        return float(np.mean([np.polyval(legs[side]["fit1"], z) for side in legs]))

    def leg_r(z):
        return float(np.mean([legs[side]["radius"](z) for side in legs]))

    hip_s, ankle_s = leg_s(hip_z), leg_s(ankle_z)
    splay = math.degrees(math.atan2(ankle_s - hip_s, hip_z - ankle_z))
    splay = min(max(splay, SPLAY_RANGE[0]), SPLAY_RANGE[1])
    hip_half = min(max(hip_s, 0.5 * pelvis["half"]), 0.8 * pelvis["half"])
    if abs(hip_half - hip_s) > 1e-9:
        notes.append(f"hip joint {hip_s:.3f} from the centre line held to {hip_half:.3f} "
                     f"(0.5-0.8 of the pelvis half-width)")

    # the feet, below the ankle on each side
    feet = {}
    for side, sign in (("left", -1.0), ("right", 1.0)):
        s = sign * (co[:, 0] - cx)
        foot = co[(co[:, 2] < ankle_z) & (s > 0.004 * height)]
        ankle_y = float(np.polyval(legs[side]["fit2"], ankle_z))
        y0, y1 = _span(foot[:, 1], 1, 99)
        x0, x1 = _span(foot[:, 0], 1, 99)
        feet[side] = (y1 - ankle_y, ankle_y - y0, (x1 - x0) / 2)
    foot_len = float(np.mean([f[0] for f in feet.values()]))
    heel = float(np.mean([f[1] for f in feet.values()]))
    foot_half = float(np.mean([f[2] for f in feet.values()]))
    foot_scale = (foot_len / height) / d.foot_f

    neck = _section(co, neck_z, cx, 0.25 * height, height)
    head = _section(co, neck_z + head_f * height / 2, cx, 0.25 * height, height)

    thigh_len = hip_z - knee_z
    shin_len = knee_z - ankle_z
    f = {
        "leg_f": leg_f, "torso_f": torso_f, "neck_f": neck_f, "head_f": head_f,
        "head_r_f": head_f / 2,
        "thigh_f": thigh_len / height,
        "ankle_f": (ankle_z - low) / height,
        "upper_arm_f": upper / height, "forearm_f": fore / height, "hand_f": hand / height,
        "shoulder_half_f": s_shoulder / height, "hip_half_f": hip_half / height,
        "chest_half_f": chest["half"] / height, "chest_depth_f": chest["depth"] / height,
        "waist_half_f": waist["half"] / height, "waist_depth_f": waist["depth"] / height,
        "pelvis_half_f": pelvis["half"] / height, "pelvis_depth_f": pelvis["depth"] / height,
        # measured from the LEANED centre line, because the cage's chest is posed back along it
        "seat_f": min(max(lean_line(seat["z"]) - seat["y"], -0.01 * height), 0.06 * height) / height,
        "chest_set_f": min(max(front["y"] - lean_line(front["z"]), -0.01 * height),
                           0.06 * height) / height,
        "neck_r_f": neck["half"] / height,
        "head_depth_f": head["depth"] / height if head else d.head_depth_f,
        "bicep_r_f": arm_r(s_shoulder + 0.35 * (s_elbow - s_shoulder)) / height,
        "elbow_r_f": arm_r(s_elbow) / height,
        "forearm_r_f": arm_r(s_elbow + 0.22 * (s_wrist - s_elbow)) / height,
        "wrist_r_f": arm_r(s_wrist - 0.04 * (s_wrist - s_elbow)) / height,
        "thigh_r_f": leg_r(hip_z - 0.35 * thigh_len) / height,
        "knee_r_f": leg_r(knee_z) / height,
        "calf_r_f": leg_r(knee_z - 0.35 * shin_len) / height,
        "ankle_r_f": leg_r(ankle_z + 0.02 * height) / height,
        "foot_f": foot_len / height, "heel_f": heel / height, "foot_half_f": foot_half / height,
        "leg_splay_deg": splay,
    }
    # The arm is built at the skin's own angle (rest="skin") or at the cage's 45-degree A-pose and
    # posed up to it (rest="A"); see `fit_cage_to` for which and why.
    droop = math.degrees(math.atan2(shoulder_z - arm_z(s_wrist), s_wrist - s_shoulder))
    if rest == "skin":
        f["arm_drop_deg"] = droop
    elif rest != "A":
        f["arm_drop_deg"] = float(rest)
    f["shin_f"] = f["leg_f"] - f["thigh_f"] - f["ankle_f"]
    # The radii where a limb meets the trunk keep the cage's own ratio to the next ring along:
    # a slice at the shoulder or the hip cuts through the trunk too and reads as a fat arm.
    f["shoulder_r_f"] = f["bicep_r_f"] * d.shoulder_r_f / d.bicep_r_f
    f["hip_r_f"] = f["thigh_r_f"] * d.hip_r_f / d.thigh_r_f
    # The hand and foot detail scales with the measured length. The skin's hand is a mitten and is
    # not measured finger by finger; the cage's fingers only have to sit inside it.
    for name in ("palm_len_f", "palm_knuckle_half_f", "palm_knuckle_depth_f", "finger_web_f",
                 "finger_r_f", "thumb_len_f", "thumb_r_f"):
        f[name] = getattr(d, name) * hand_scale
    f["finger_len_f"] = tuple(v * hand_scale for v in d.finger_len_f)
    for name in ("instep_f", "toe_half_f", "toe_depth_f", "sole_z_f", "arch_f", "toe_lift_f"):
        f[name] = getattr(d, name) * foot_scale

    # `character.py` is being reshaped by another task; only pass the fields it still has.
    known = {field.name for field in dataclasses.fields(character.Proportions)}
    missing = sorted(set(f) - known)
    if missing:
        notes.append(f"Proportions no longer has {missing}; those measurements were not used")
    props = dataclasses.replace(d, height=height, **{k: v for k, v in f.items() if k in known})

    # Where each limb has to point, as a world point for its first bone to aim at: the skin's
    # wrist on the arm's line, the skin's ankle on the leg's. Aimed from the CAGE's own joint, so
    # the hand and foot land where the skin's are even where the skin's shoulder or hip sits a
    # little off the cage's centre-line plane — the Tripo legs stand 3cm behind the waist.
    aim = {}
    for side, sign in (("left", -1.0), ("right", 1.0)):
        a, g = arms[side], legs[side]
        aim[f"upper_arm.{side}"] = (cx + sign * s_wrist, float(np.polyval(a["fit2"], s_wrist)),
                                    float(np.polyval(a["fit1"], s_wrist)))
        aim[f"thigh.{side}"] = (cx + sign * float(np.polyval(g["fit1"], ankle_z)),
                                float(np.polyval(g["fit2"], ankle_z)), ankle_z)
    place = {"origin": (cx, cy, low), "aim": aim, "shoulder_y": shoulder_y,
             "head_y": head["y"] if head else cy, "arm_droop_deg": droop,
             "notes": notes,
             "measured": {"height": height, "floor": low, "centre": (cx, cy),
                          "hip_z": hip_z, "shoulder_z": shoulder_z, "neck_z": neck_z,
                          "knee_z": knee_z, "ankle_z": ankle_z, "s_shoulder": s_shoulder,
                          "s_elbow": s_elbow, "s_wrist": s_wrist, "s_tip": s_tip,
                          "hip_s": hip_s, "ankle_s": ankle_s, "splay_deg": splay}}
    # The fitted lines themselves, for a cage that is not `character.py`'s (`mpfb_cage` places
    # every joint of its own skeleton on them). Read-only extras: nothing above depends on them.
    place["lines"] = {
        "arm": {s: (arms[s]["fit1"].tolist(), arms[s]["fit2"].tolist()) for s in arms},
        "arm_radius": {s: arms[s]["rows"][:, [0, 3]].tolist() for s in arms},
        "leg": {s: (legs[s]["fit1"].tolist(), legs[s]["fit2"].tolist()) for s in legs},
        "leg_radius": {s: legs[s]["rows"][:, [0, 3]].tolist() for s in legs},
        "crotch_z": float(crotch_z), "chest_z": float(chest_z), "hip_half": float(hip_half),
        "neck": neck, "head": head, "waist": waist, "pelvis": pelvis, "chest": chest,
        "foot": {"length": foot_len, "heel": heel, "half": foot_half},
        "lean": {"cy": cy, "shoulder_y": shoulder_y},
    }
    return props, place


# --- the cage, posed as the skin stands ----------------------------------------------------------

def _turn(rig, name, world):
    """Rotate a pose bone by `world` (a rotation in armature space) on top of how it stands now.

    A pose bone's own rotation is applied in its own posed frame B, so turning it by Q in the
    armature's frame means a local rotation of B^-1 Q B on top of the one it has. That holds
    whatever its parents are doing — which matters, because the spine is posed first here and
    the arms, neck and legs all hang off it.
    """
    pbone = rig.pose.bones[_bn(rig, name)]
    frame = pbone.matrix.to_3x3()
    turn = (frame.inverted() @ world.to_matrix() @ frame).to_quaternion()
    pbone.rotation_quaternion = pbone.rotation_quaternion @ turn
    bpy.context.view_layer.update()


def _aim(rig, name, want):
    """Turn a bone so it points along `want` (armature space) by the smallest rotation."""
    pbone = rig.pose.bones[_bn(rig, name)]
    now = (pbone.tail - pbone.head).normalized()
    _turn(rig, name, now.rotation_difference(Vector(want).normalized()))
    got = (pbone.tail - pbone.head).normalized()
    return math.degrees(got.angle(Vector(want).normalized()))


def to_skin_pose(rig):
    """Put the rig back in the skin's rest pose, as `fit_cage_to` recorded it."""
    fit = json.loads(rig["skin_fit"])
    for pbone in rig.pose.bones:
        pbone.rotation_mode = "QUATERNION"
        pbone.rotation_quaternion = Quaternion()
        pbone.location = (0.0, 0.0, 0.0)
    for name, q in fit["pose"].items():
        rig.pose.bones[name].rotation_quaternion = Quaternion(q)
    bpy.context.view_layer.update()


MAX_LEAN = 0.45            # sin of the steepest spine or neck lean `fit_cage_to` will pose


def fit_cage_to(skin_path, levels=2, scratch=None, rest="skin"):
    """Build the clean cage at the skin's proportions and pose it the way the skin stands.

    `rest` says what the cage is built in. "skin" (the default) builds the arms at the skin's
    own angle, so the cage the skin is bound to is the cage exactly as `character.py` made it.
    "A" builds the house 45-degree A-pose and poses the arms up to the skin's T-pose, which is
    what the brief's standalone test did — and which leaves the top of the shoulder crushed into
    a fold at bind time: lifting an arm 50 degrees by linear blend skinning collapses the quads
    on top of the socket, their normals turn inward, and Surface Deform binds the skin's deltoid
    to that fold. Kept selectable so the two can be compared; the report does.

    Everything else is posed, in this order, each from where the last left it:
      spine   leaned back until the shoulder joints are as far behind the waist as the skin's
      neck    leaned so the head's centre is over the skin's
      arms    aimed at the skin's wrists, from the cage's own shoulders
      thighs  aimed at the skin's ankles, from the cage's own hips
    """
    if rest not in ("skin", "A"):
        try:
            rest = float(rest)
        except ValueError:
            raise SystemExit(f"rest must be 'skin', 'A' or an arm angle in degrees, not {rest!r}")
    scratch = Path(scratch or tempfile.mkdtemp(prefix="skin_wrap_"))
    scratch.mkdir(parents=True, exist_ok=True)
    meas = measure_skin(skin_path, scratch)
    props, place = proportions_from(meas, rest=rest)
    cage, rig, props, at = character.build(props, levels=levels)
    rig.location = place["origin"]
    for pbone in rig.pose.bones:
        pbone.rotation_mode = "QUATERNION"
        pbone.rotation_quaternion = Quaternion()
    bpy.context.view_layer.update()
    world = rig.matrix_world
    cy = place["origin"][1]
    misses = {}

    h = (at["shoulder.right"] - at["pelvis"]).z
    lean = max(-MAX_LEAN, min(MAX_LEAN, (place["shoulder_y"] - cy) / h))
    misses["spine"] = _aim(rig, "spine", (0.0, lean, math.sqrt(1.0 - lean ** 2)))

    chest = world @ rig.pose.bones["neck"].head
    centre = world @ rig.pose.bones["head"].tail
    reach = (centre - chest).length
    tilt = max(-MAX_LEAN, min(MAX_LEAN, (place["head_y"] - chest.y) / reach))
    misses["neck"] = _aim(rig, "neck", (0.0, tilt, math.sqrt(1.0 - tilt ** 2)))

    for name, target in place["aim"].items():
        joint = world @ rig.pose.bones[name].head
        misses[name] = _aim(rig, name, Vector(target) - joint)

    pose = {pb.name: tuple(pb.rotation_quaternion) for pb in rig.pose.bones
            if pb.rotation_quaternion.angle > 1e-6}
    direction, angles = {}, {}
    for pbone in rig.pose.bones:
        bone = pbone.bone
        posed = (pbone.tail - pbone.head).normalized()
        direction[pbone.name] = tuple(posed)
        angles[pbone.name] = round(math.degrees(posed.angle(
            (bone.tail_local - bone.head_local).normalized())), 2)
    shoulder = {side: tuple(world @ rig.pose.bones[f"upper_arm.{side}"].head)
                for side in ("left", "right")}
    rig["skin_fit"] = json.dumps({
        "skin": str(skin_path), "rest": rest, "pose": pose, "origin": place["origin"],
        "direction": direction, "pose_angle_from_rest_deg": angles,
        "arm_droop_deg": place["arm_droop_deg"], "spine_lean_deg": math.degrees(math.asin(lean)),
        "neck_lean_deg": math.degrees(math.asin(tilt)), "shoulder_joint": shoulder,
        "notes": place["notes"], "measured": place["measured"],
        "dropped_on_load": meas["dropped"], "silhouette": meas["silhouette"],
    })
    # The pose has to have landed: an upper arm still pointing 45 degrees down while the skin's
    # points sideways would bind every sleeve to the cage's ribs. Re-applied from what was stored,
    # so what is checked is what every later step will get from `to_skin_pose`.
    to_skin_pose(rig)
    for name, miss in misses.items():
        want = Vector(direction[name])
        got = (rig.pose.bones[name].tail - rig.pose.bones[name].head).normalized()
        if miss > 0.5 or math.degrees(got.angle(want)) > 0.5:
            raise SystemExit(f"{name} is {max(miss, math.degrees(got.angle(want))):.1f} degrees "
                             f"off where it was aimed after posing")
    for side in ("left", "right"):
        y = (world @ rig.pose.bones[f"upper_arm.{side}"].head).y
        if abs(lean) < MAX_LEAN and abs(y - place["shoulder_y"]) > 0.003 * props.height:
            raise SystemExit(f"the {side} shoulder joint is at y={y:.4f}, the skin's at "
                             f"{place['shoulder_y']:.4f}: the spine lean did not land")
    return cage, rig, props


# --- the cage as it stands -----------------------------------------------------------------------

@contextmanager
def base_cage(cage):
    """Evaluate the cage WITHOUT its subsurf and triangulate modifiers: the control cage, posed."""
    hidden = [m for m in cage.modifiers if m.type in ("SUBSURF", "TRIANGULATE") and m.show_viewport]
    for mod in hidden:
        mod.show_viewport = False
    try:
        bpy.context.view_layer.update()
        yield
    finally:
        for mod in hidden:
            mod.show_viewport = True
        bpy.context.view_layer.update()


def evaluated(obj):
    """World-space vertex positions and vertex normals of an object as evaluated now."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(depsgraph)
    mesh = ev.to_mesh()
    co = _co(mesh)
    normals = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
    mesh.vertex_normals.foreach_get("vector", normals)
    ev.to_mesh_clear()
    world = np.array(obj.matrix_world, dtype=np.float64)
    co = co @ world[:3, :3].T + world[:3, 3]
    normal_matrix = np.linalg.inv(world[:3, :3]).T
    normals = normals.reshape(-1, 3) @ normal_matrix.T
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    return co, normals


def posed_cage(cage):
    with base_cage(cage):
        return evaluated(cage)


def _quads(mesh):
    loops = np.empty(len(mesh.polygons) * 4, dtype=np.int64)
    if any(len(p.vertices) != 4 for p in mesh.polygons):
        raise SystemExit("the cage is supposed to be all quads")
    mesh.polygons.foreach_get("vertices", loops)
    return loops.reshape(-1, 4)


def _edges(mesh):
    out = np.empty(len(mesh.edges) * 2, dtype=np.int64)
    mesh.edges.foreach_get("vertices", out)
    return out.reshape(-1, 2)


def _weights(cage):
    """Vertex x bone weights from the cage's own groups, normalised as the Armature modifier does.
    The bones are the rig profile's (`bake_bones`), left-out bones merged into the named one."""
    rig = next((m.object for m in cage.modifiers if m.type == "ARMATURE"), None)
    bones, merge = bake_bones(rig)
    index = {cage.vertex_groups[b].index: i for i, b in enumerate(bones) if b in cage.vertex_groups}
    for src, dst in merge.items():
        if src in cage.vertex_groups and dst in bones:
            index[cage.vertex_groups[src].index] = bones.index(dst)
    w = np.zeros((len(cage.data.vertices), len(bones)))
    for v in cage.data.vertices:
        for g in v.groups:
            if g.group in index:
                w[v.index, index[g.group]] += g.weight
    total = w.sum(axis=1, keepdims=True)
    if (total <= 1e-4).any():
        raise SystemExit("a cage vertex has no bone weight — the inverse skinning would divide by 0")
    return w / total, bones


def _face_normals_areas(points, quads):
    p = points[quads]
    n = (np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
         + np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]))
    area = np.linalg.norm(n, axis=1) / 2
    return n / np.maximum(area[:, None] * 2, 1e-15), area


def _damage_sets(co, quads, height, reference=None):
    """`damage`, as the sets themselves, so two states can be compared item by item."""
    tris = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    owner = np.concatenate([np.arange(len(quads)), np.arange(len(quads))])
    tree = BVHTree.FromPolygons([Vector(v) for v in co], tris.tolist(), all_triangles=True)
    crossing = set()
    for a, b in tree.overlap(tree):
        qa, qb = int(owner[a]), int(owner[b])
        if qa != qb:
            crossing.add((min(qa, qb), max(qa, qb)))
    kd = KDTree(len(co))
    for i, v in enumerate(co):
        kd.insert(Vector(v), i)
    kd.balance()
    near = 0.0005 * height
    overlapping = set()
    for i, v in enumerate(co):
        for _, j, _ in kd.find_range(Vector(v), near):
            if j != i:
                overlapping.add((min(i, j), max(i, j)))
    normals, area = _face_normals_areas(co, quads)
    sides = np.sort(np.stack([quads, np.roll(quads, -1, axis=1)], axis=2).reshape(-1, 2), axis=1)
    face_of = np.repeat(np.arange(len(quads)), 4)
    order = np.lexsort((sides[:, 1], sides[:, 0]))
    sides, face_of = sides[order], face_of[order]
    pair = np.flatnonzero((sides[1:] == sides[:-1]).all(axis=1))
    folded = pair[(normals[face_of[pair]] * normals[face_of[pair + 1]]).sum(axis=1) < 0]
    flipped, collapsed = set(), set()
    if reference is not None:
        ref_normals, ref_area = _face_normals_areas(reference, quads)
        collapsed = set(np.flatnonzero(area < 0.05 * np.maximum(ref_area, 1e-15)).tolist())
        flipped = set(np.flatnonzero((normals * ref_normals).sum(axis=1) < 0).tolist())
    return {"crossing": crossing, "overlapping": overlapping,
            "folds": {tuple(sides[i]) for i in folded}, "fold_index": folded,
            "fold_faces": face_of, "fold_pairs": {tuple(sides[i]): (int(face_of[i]),
                                                                    int(face_of[i + 1]))
                                                  for i in folded},
            "flipped": flipped, "collapsed": collapsed}


def damage(co, quads, height, reference=None, labels=None):
    """What a fit can do to a surface that Surface Deform will refuse, or render as a crease.

      crossing    pairs of faces whose triangles pass through each other (a real tri-tri test,
                  `BVHTree.overlap`; faces sharing an edge are not counted, faces sharing one
                  corner only if they cross away from it)
      overlapping pairs of vertices closer than 0.05% of height — what "Target contains
                  overlapping vertices" means, and what a finger collapsed onto a mitten makes
      folds       edges where the two faces either side meet at less than 90 degrees — the surface
                  doubled back on itself, whether or not it has yet passed through itself
      collapsed   faces with less than 5% of their reference area — zero-volume geometry
      flipped     faces whose normal now opposes their reference normal — inside-out geometry
    """
    found = _damage_sets(co, quads, height, reference)
    crossing, overlapping = found["crossing"], found["overlapping"]
    folded, face_of = found["fold_index"], found["fold_faces"]
    flipped, collapsed = sorted(found["flipped"]), sorted(found["collapsed"])
    out = {"crossing": len(crossing), "overlapping": len(overlapping), "folds": len(folded)}
    if reference is not None:
        out["collapsed"] = int(len(collapsed))
        out["flipped"] = int(len(flipped))
    if labels is not None:
        # where each kind of damage is, by body region — so a count that did not change can be
        # told apart from damage that moved, and the cage's own defects from the fit's
        face_label = [max(set(labels[f]), key=list(labels[f]).count) for f in quads]
        where = {}

        def tally(kind, names):
            for name in names:
                where.setdefault(name, {}).setdefault(kind, 0)
                where[name][kind] += 1

        tally("crossing", ("/".join(sorted({face_label[a], face_label[b]})) for a, b in crossing))
        tally("overlapping", ("/".join(sorted({labels[a], labels[b]})) for a, b in overlapping))
        tally("folds", (face_label[face_of[i]] for i in folded))
        tally("flipped", (face_label[i] for i in flipped))
        tally("collapsed", (face_label[i] for i in collapsed))
        out["by_region"] = where
    return out


# --- region fit ----------------------------------------------------------------------------------

REGION = {
    "spine": "torso", "clavicle.left": "torso", "clavicle.right": "torso",
    "neck": "head", "head": "head",
    "upper_arm.left": "arm.left", "forearm.left": "arm.left", "hand.left": "hand.left",
    "upper_arm.right": "arm.right", "forearm.right": "arm.right", "hand.right": "hand.right",
    "thigh.left": "leg.left", "shin.left": "leg.left", "foot.left": "leg.left",
    "thigh.right": "leg.right", "shin.right": "leg.right", "foot.right": "leg.right",
}
# Never moved. The cage's hands have five real fingers and the face is the house face's place;
# the skin's hands are mittens and its face is the least reliable thing image-to-3D produces.
FROZEN = ("hand.left", "hand.right", "face")
FACE_CONE_DEG = 55.0        # head vertices within this angle of straight ahead are the face
SHARE = 0.15                # a bone with at least this weight lets a vertex follow its region
REACH = 1.0                 # a match further than REACH x the vertex's own limb radius is refused
FACING = 0.0                # ...as is one whose surface faces away from the cage's (cosine)
INWARD = 0.6                # ...or that would pull a vertex in past 60% of its radius
FEATHER = 3                 # rings over which the pull fades to nothing next to a frozen part
SMOOTH_STEPS = 6            # Laplacian passes over the pull, so neighbours move together
SMOOTH_FACTOR = 0.5
BACKOFF_STEPS = 12          # rounds of halving (then dropping) the pull where it would do damage


def _region(rig, bone):
    """The body region a bone belongs to (`REGION`), for the house rig or a Mixamo-boned one."""
    if rig_profile(rig) == "house":
        return REGION[bone]
    own = bone[len(MIXAMO_PREFIX):]
    if own in ("Hips", "Spine", "Spine1", "Spine2", "LeftShoulder", "RightShoulder"):
        return "torso"
    if own in ("Neck", "Head"):
        return "head"
    side = "left" if own.startswith("Left") else "right"
    if "Hand" in own:
        return f"hand.{side}"
    return f"arm.{side}" if own[len(side.title()):] in ("Arm", "ForeArm") else f"leg.{side}"


def cage_regions(cage, rig, posed):
    """Each cage vertex's body region, from the bone that owns most of it — plus the face.

    The face is the front of the head: head-owned vertices within FACE_CONE_DEG of the head's own
    forward direction, as the head is posed now.
    """
    weights, bones = _weights(cage)
    dominant = np.array([bones[i] for i in weights.argmax(axis=1)])
    region = np.array([_region(rig, b) for b in dominant], dtype=object)
    head = rig.pose.bones[_bn(rig, "head")]
    centre = np.array(rig.matrix_world @ head.tail)
    turned = head.matrix.to_3x3() @ head.bone.matrix_local.to_3x3().inverted()
    forward = np.array(turned @ Vector((0.0, 1.0, 0.0)))
    out = posed - centre
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
    region[(dominant == head.name) & (out @ forward > math.cos(math.radians(FACE_CONE_DEG)))] = "face"
    return region, weights, bones, dominant


def _segment_distance(points, a, b):
    ab = b - a
    t = np.clip(((points - a) @ ab) / max(ab @ ab, 1e-15), 0.0, 1.0)
    return np.linalg.norm(points - (a + t[:, None] * ab), axis=1)


def _rings_from(frozen, edges, n, limit):
    """Graph distance, in edges, from the nearest frozen vertex (capped at `limit`)."""
    nbrs = [[] for _ in range(n)]
    for a, b in edges:
        nbrs[a].append(b)
        nbrs[b].append(a)
    dist = np.full(n, limit, dtype=np.int64)
    queue = deque()
    for i in np.flatnonzero(frozen):
        dist[i] = 0
        queue.append(i)
    while queue:
        i = queue.popleft()
        if dist[i] + 1 >= limit:
            continue
        for j in nbrs[i]:
            if dist[j] > dist[i] + 1:
                dist[j] = dist[i] + 1
                queue.append(j)
    return dist


def _smooth(field, known, pinned, edges, steps, factor):
    """Laplacian passes over a per-vertex vector field.

    Known values relax toward their neighbours by `factor`; unknown ones (refused matches) take
    their neighbours' mean outright, which fills a hole in the field from its rim; pinned ones
    stay zero.
    """
    n = len(field)
    degree = np.bincount(edges.ravel(), minlength=n).astype(np.float64)[:, None]
    for _ in range(steps):
        total = np.zeros_like(field)
        np.add.at(total, edges[:, 0], field[edges[:, 1]])
        np.add.at(total, edges[:, 1], field[edges[:, 0]])
        mean = total / np.maximum(degree, 1.0)
        field = np.where(known[:, None], (1 - factor) * field + factor * mean, mean)
        field[pinned] = 0.0
    return field


def _skin_matrices(cage, rig, weights, bones):
    """Each cage vertex's Armature-modifier matrix, in the cage's local space.

    Exactly what the modifier computes with envelopes and preserve-volume off (the case here):
    co' = post @ sum_i w_i (pose_i @ rest_i^-1) @ pre @ co, with w normalised to sum to one.
    """
    pre = rig.matrix_world.inverted() @ cage.matrix_world
    post = pre.inverted()
    per_bone = np.array([np.array(rig.pose.bones[b].matrix @ rig.data.bones[b].matrix_local.inverted())
                         for b in bones])
    blended = np.einsum("vb,bij->vij", weights, per_bone)
    return np.array(post) @ blended @ np.array(pre)


def region_fit(cage, skin, reach=REACH, facing=FACING, inward=INWARD, feather=FEATHER,
               smooth_steps=SMOOTH_STEPS, smooth_factor=SMOOTH_FACTOR):
    """Pull the cage onto the skin's surface, a body region at a time. Returns the cage.

    Blender's Shrinkwrap modifier has no way to limit which part of the target a vertex may land
    on, and that is the whole problem: a cage finger's nearest point on a Tripo mitten is the
    mitten, so a whole-mesh shrinkwrap collapses five fingers onto one blob and Surface Deform
    then refuses the cage for overlapping vertices. So the nearest-surface-point search is done
    here, with the same BVH Blender's shrinkwrap uses, against a target limited per vertex:

      1. Regions. Every cage vertex already knows which bones it belongs to — `character.py`
         assigns weights at birth. Each bone maps to a region (torso, head, an arm, a hand, a
         leg). The hands and the face are frozen.
      2. Per-group targets. Each skin vertex is labelled with the region of its nearest cage
         vertex. A cage vertex may land only on skin of the regions its own bones (weight >=
         SHARE) belong to, and never on a frozen region: an elbow cannot land on a thigh, a
         thigh cannot land on the other leg, a wrist cannot land on the mitten.
      3. Proximity. A match is refused if it is further than `reach` x the vertex's own limb
         radius (its distance from its bone), if the skin there faces away from the cage, or if
         it would pull the vertex in past `inward` of its radius — any of which means the nearest
         point is on some other body part, however the labels came out.
      4. The pull is smoothed as a field rather than applied vertex by vertex, so neighbours
         move together and a crease in the skin cannot fold the cage; refused vertices take
         their neighbours' pull; the pull fades to zero over `feather` rings next to a frozen
         part, so a wrist does not step.

    The fit is computed with the cage POSED as the skin stands and written back into the cage's
    REST shape by inverting each vertex's own skinning matrix — so the cage keeps its A-pose rest,
    its weights and its armature, and lands exactly on the fitted surface when posed. That landing
    is checked, not assumed.
    """
    rig = next(m.object for m in cage.modifiers if m.type == "ARMATURE")
    fit = json.loads(rig["skin_fit"])
    height = fit["measured"]["height"]
    to_skin_pose(rig)
    mesh = cage.data
    quads = _quads(mesh)
    edges = _edges(mesh)
    n = len(mesh.vertices)
    posed, normals = posed_cage(cage)
    region, weights, bones, dominant = cage_regions(cage, rig, posed)
    before = damage(posed, quads, height, labels=region)
    frozen = np.isin(region, FROZEN)

    # the limb radius at each vertex: its distance from its own bone, posed
    radius = np.zeros(n)
    for bone in bones:
        pick = dominant == bone
        if pick.any():
            pb = rig.pose.bones[bone]
            a = np.array(rig.matrix_world @ pb.head)
            b = np.array(rig.matrix_world @ pb.tail)
            radius[pick] = _segment_distance(posed[pick], a, b)

    # the skin, triangulated, labelled by its nearest cage vertex
    skin.data.calc_loop_triangles()
    skin_co = _co(skin.data) @ np.array(skin.matrix_world)[:3, :3].T + np.array(skin.matrix_world)[:3, 3]
    tris = np.empty(len(skin.data.loop_triangles) * 3, dtype=np.int64)
    skin.data.loop_triangles.foreach_get("vertices", tris)
    tris = tris.reshape(-1, 3)
    kd = KDTree(n)
    for i, v in enumerate(posed):
        kd.insert(Vector(v), i)
    kd.balance()
    skin_region = np.array([region[kd.find(Vector(v))[1]] for v in skin_co], dtype=object)
    skin_region[skin_region == "face"] = "head"

    # which regions each cage vertex may land on
    region_of_bone = np.array([_region(rig, b) for b in bones], dtype=object)
    allowed = []
    for i in range(n):
        names = set(region_of_bone[weights[i] >= SHARE]) | {region[i]}
        allowed.append(frozenset(r for r in names if r not in FROZEN and r != "face"))
    trees = {}
    for key in set(allowed):
        if not key:
            continue
        keep = np.isin(skin_region[tris], list(key)).any(axis=1)
        if keep.any():
            trees[key] = BVHTree.FromPolygons([Vector(v) for v in skin_co],
                                              tris[keep].tolist(), all_triangles=True)

    pull = np.zeros((n, 3))
    known = np.zeros(n, dtype=bool)
    refused = {"too far": 0, "faces away": 0, "too deep": 0, "no target": 0}
    gap_before = []
    for i in range(n):
        if frozen[i]:
            continue
        tree = trees.get(allowed[i])
        hit = tree.find_nearest(Vector(posed[i])) if tree else (None, None, None, None)
        if hit[0] is None:
            refused["no target"] += 1
            continue
        point, normal, _, dist = hit
        gap_before.append(dist)
        move = np.array(point) - posed[i]
        if dist > reach * radius[i]:
            refused["too far"] += 1
            continue
        if np.dot(np.array(normal), normals[i]) <= facing:
            refused["faces away"] += 1
            continue
        if np.dot(move, normals[i]) < -inward * radius[i]:
            refused["too deep"] += 1
            continue
        pull[i] = move
        known[i] = True

    raw = pull.copy()
    pull = _smooth(pull, known | frozen, frozen, edges, smooth_steps, smooth_factor)
    rings = _rings_from(frozen, edges, n, feather)
    pull *= np.minimum(rings / float(feather), 1.0)[:, None]

    # Never leave the cage worse than it was found. Projection onto a nearest surface point
    # bunches vertices wherever the skin is thin or concave — on the boy's bare toes it crossed
    # the cage's sole through itself in 18 places — and smoothing reduces that but cannot promise
    # it. So every crossing, fold, inverted or collapsed face and overlapping pair the pull would
    # ADD is found, and the pull is halved on its vertices and their neighbours until there are
    # none; four halvings that do not clear it and the pull there is dropped. Whatever damage the
    # cage had before the fit is left alone and counted, never hidden.
    base = _damage_sets(posed, quads, height, reference=posed)
    nbrs = [[] for _ in range(n)]
    for a, b in edges:
        nbrs[a].append(b)
        nbrs[b].append(a)
    scale = np.ones(n)
    backed_off = set()
    for step in range(BACKOFF_STEPS):
        found = _damage_sets(posed + pull * scale[:, None], quads, height, reference=posed)
        bad_faces = set(found["flipped"]) | set(found["collapsed"])
        for a, b in found["crossing"] - base["crossing"]:
            bad_faces |= {a, b}
        for edge in found["folds"] - base["folds"]:
            bad_faces |= set(found["fold_pairs"][edge])
        bad = set(quads[sorted(bad_faces)].ravel().tolist()) if bad_faces else set()
        for a, b in found["overlapping"] - base["overlapping"]:
            bad |= {a, b}
        bad = {v for v in bad if scale[v] > 0}
        if not bad:
            break
        ring = set(bad)
        for v in bad:
            ring.update(nbrs[v])
        ring = np.array(sorted(ring))
        scale[ring] = 0.0 if step >= 4 else scale[ring] * 0.5
        backed_off |= set(ring.tolist())
    else:
        raise SystemExit(f"region_fit could not back off far enough to leave the cage undamaged "
                         f"after {BACKOFF_STEPS} rounds")
    pull *= scale[:, None]
    target = posed + pull

    # back into the rest shape through each vertex's own skinning matrix
    world_inv = np.array(cage.matrix_world.inverted())
    local_target = target @ world_inv[:3, :3].T + world_inv[:3, 3]
    matrices = _skin_matrices(cage, rig, weights, bones)
    homogeneous = np.concatenate([local_target, np.ones((n, 1))], axis=1)
    rest = np.einsum("vij,vj->vi", np.linalg.inv(matrices), homogeneous)[:, :3]
    moved = np.linalg.norm(pull, axis=1) > 0.0
    co = _co(mesh)
    co[moved] = rest[moved]
    mesh.vertices.foreach_set("co", co.ravel())
    mesh.update()

    landed, _ = posed_cage(cage)
    miss = float(np.linalg.norm(landed - target, axis=1).max())
    if miss > 1e-5 * height:
        raise SystemExit(f"the fitted cage does not land where the fit put it: off by "
                         f"{miss * 1000:.3f}mm. The inverse skinning is wrong; nothing built on "
                         f"this cage can be trusted.")
    after = damage(landed, quads, height, reference=posed, labels=region)
    before.update({"collapsed": 0, "flipped": 0})

    gaps_after = []
    for i in np.flatnonzero(~frozen):
        tree = trees.get(allowed[i])
        if tree:
            hit = tree.find_nearest(Vector(landed[i]))
            if hit[0] is not None:
                gaps_after.append(hit[3])
    stats = {
        "vertices": n, "frozen": int(frozen.sum()),
        "frozen_by_region": {r: int((region == r).sum()) for r in FROZEN},
        "pulled": int(known.sum()), "refused": refused,
        "moved": int(moved.sum()),
        "backed_off": int(len(backed_off)), "backoff_rounds": int(step),
        "pull_dropped": int(((scale == 0) & ~frozen).sum()),
        "pull_mm": {"mean": float(np.linalg.norm(pull[moved], axis=1).mean() * 1000),
                    "max": float(np.linalg.norm(pull, axis=1).max() * 1000)},
        "raw_pull_max_mm": float(np.linalg.norm(raw, axis=1).max() * 1000),
        "gap_to_skin_mm": {"before_mean": float(np.mean(gap_before) * 1000),
                           "before_p95": float(np.percentile(gap_before, 95) * 1000),
                           "after_mean": float(np.mean(gaps_after) * 1000),
                           "after_p95": float(np.percentile(gaps_after, 95) * 1000)},
        "landing_error_mm": miss * 1000,
        "damage_before": before, "damage_after": after,
        "settings": {"reach": reach, "facing": facing, "inward": inward, "feather": feather,
                     "smooth_steps": smooth_steps, "smooth_factor": smooth_factor},
    }
    cage["region_fit"] = json.dumps(stats)
    return cage


# --- the wrap ------------------------------------------------------------------------------------

def wrap(cage, rig, skin, name="wrap", falloff=None):
    """Bind `skin` to the cage with Surface Deform. Returns the modifier, guaranteed bound.

    The cage is triangulated by a MODIFIER, last in its stack, so what Surface Deform binds to is
    the subdivided, triangulated surface while the cage's own mesh stays all quads. Surface
    Deform refuses concave polygons outright, and a bent quad of a Catmull-Clark cage often is.

    The bind is made in the skin's own rest pose — the rig is put back there first — so the skin
    does not move at all when bound, and moves only when the rig does.

    It does NOT parent the skin to the rig. Surface Deform follows the cage's mesh but ignores the
    cage's OBJECT transform after the bind, so a caller that moves the rig object (grounding does)
    must parent the skin to the rig itself (task 7); `animate.wear_skin` does, and `bake_to_skin`
    does for the skinned mesh it leaves behind.
    """
    to_skin_pose(rig)
    # Surface Deform stores the target's face count at bind time and refuses a target whose count
    # has changed. A render evaluates subsurf at `render_levels`; if that differed from `levels`
    # the skin would bind in the viewport and fall off the cage in every render.
    for sub in (m for m in cage.modifiers if m.type == "SUBSURF"):
        if sub.levels != sub.render_levels:
            raise SystemExit(f"the cage's subsurf has levels {sub.levels} but render_levels "
                             f"{sub.render_levels}: the wrap would not survive a render")
    tri = next((m for m in cage.modifiers if m.type == "TRIANGULATE"), None)
    if tri is None:
        tri = cage.modifiers.new("triangulate", "TRIANGULATE")
    tri.quad_method = "BEAUTY"
    tri.min_vertices = 4
    cage.modifiers.move(list(cage.modifiers).index(tri), len(cage.modifiers) - 1)
    for old in [m for m in skin.modifiers if m.type == "SURFACE_DEFORM"]:
        skin.modifiers.remove(old)
    mod = skin.modifiers.new(name, "SURFACE_DEFORM")
    mod.target = cage
    if falloff is not None:
        mod.falloff = falloff         # read at bind time only: setting it afterwards does nothing
    bpy.context.view_layer.update()
    with bpy.context.temp_override(object=skin, active_object=skin, selected_objects=[skin],
                                   selected_editable_objects=[skin]):
        bpy.ops.object.surfacedeform_bind(modifier=mod.name)
    bpy.context.view_layer.update()
    skin.evaluated_get(bpy.context.evaluated_depsgraph_get())
    # A Surface Deform that failed to bind does NOTHING, silently: the skin renders exactly as it
    # was imported, every deformation number built on it comes back zero, and zero is what a
    # perfect result looks like. That false zero has been hit three times in this pipeline
    # already. So a failed bind is an exception here, never a value for the caller to forget to
    # check. Blender keeps the reason ("Target contains overlapping vertices", "...concave
    # polygons") in the modifier panel only; `damage()` on the cage measures the same things.
    if not mod.is_bound:
        raise SystemExit(f"Surface Deform did not bind {skin.name!r} to {cage.name!r}. Run "
                         f"damage() on the posed cage: overlapping vertices and degenerate faces "
                         f"are what make it refuse.")
    return mod


# --- rig profiles: which bones, by which names -----------------------------------------------------
#
# Everything above and below was written for `character.py`'s rig (17 bones named for
# `pose_model`'s joints: spine, upper_arm.left, ...). A cage can now also be an MPFB human with
# Mixamo's 52-bone skeleton (`mpfb_cage.py`), which marks its rig `rig["rig_profile"] =
# "mpfb-mixamo"`. The training poses, the arm swing and the bake keep speaking house names; `_bn`
# turns each into the rig's own, and a rig with no profile is the house rig, unchanged.
#
#   names    upper_arm.left -> mixamorig:LeftArm, clavicle -> Shoulder, thigh -> UpLeg, shin -> Leg,
#            forearm -> ForeArm, hand/foot/neck/head as named.
#   spine    the house rig bends at one bone; the Mixamo one shares a bend between Hips, Spine,
#            Spine1 and Spine2 a quarter each (the thighs are re-aimed afterwards, as before).
#   bones    the bake fits every bone but four of the five fingers. The cartoon's hands are
#            mittens: the index, ring, pinky and thumb weights go to the hand (`merge`), and the
#            middle finger's three bones carry the mitten's front half, so it can close (a grip).
#   rigid    above the neck the skin is one rigid head; past the wrist a mitten that bends only
#            where a fist does (`_rigid_parts`).

MIXAMO_PREFIX = "mixamorig:"
_MIXAMO_NAMES = {"neck": "Neck", "head": "Head", "clavicle": "Shoulder", "upper_arm": "Arm",
                 "forearm": "ForeArm", "hand": "Hand", "thigh": "UpLeg", "shin": "Leg",
                 "foot": "Foot"}
_SPINE_CHAIN = {"house": (("spine", 1.0),),
                "mpfb-mixamo": tuple((MIXAMO_PREFIX + b, 0.25)
                                     for b in ("Hips", "Spine", "Spine1", "Spine2"))}
MITTEN_JOINTS = ((0.40, 0.10), (0.62, 0.08), (0.80, 0.08))   # (start, width) of each finger joint's blend
THUMB_FRONT = 0.30          # x hand length forward of the wrist's line: the thumb, which does not curl
NECK_FADE = 0.35            # x the neck's length: the designed neck gradient fades in over its bottom part
HAND_BAND = 0.012           # x height: either side of the wrist, the mitten's rigid share fades in


def rig_profile(rig):
    return rig.get("rig_profile", "house") if rig is not None else "house"


def _bn(rig, name):
    """The rig's own name for a house bone name (`upper_arm.left`), or `name` if it is already one."""
    if rig_profile(rig) == "house" or name.startswith(MIXAMO_PREFIX):
        return name
    base, _, side = name.partition(".")
    if base not in _MIXAMO_NAMES:
        raise SystemExit(f"no {rig_profile(rig)} bone for the house bone {name!r}")
    return MIXAMO_PREFIX + side.title() + _MIXAMO_NAMES[base]


def _spine_chain(rig):
    return _SPINE_CHAIN[rig_profile(rig)]


def bake_bones(rig):
    """(the bones the bake weights, {bone: the bone that takes its weight ON THE CAGE}).

    With a measured hand (`hand_rig`, task 13: the cartoon's fingers are separate, each with its
    own fitted chain), every finger bone is a bake bone, but the cage's own finger weights go to the
    hand: MPFB's fingers sit 5-34 mm off the cartoon's, so what they would teach the fit is wrong;
    the hand's weights are designed instead (`_rigid_parts`). Without one (task 12's mitten), the
    middle finger's chain carries the fused fingers and the rest merge into the hand."""
    if rig_profile(rig) == "house":
        return list(make_character.BONES), {}
    measured = _hand_models(rig) is not None
    bones, merge = [], {}
    for b in rig.data.bones:
        own = b.name[len(MIXAMO_PREFIX):]
        for side in ("Left", "Right"):
            if own.startswith(side + "Hand") and own != side + "Hand":
                if measured:
                    bones.append(b.name)
                    merge[b.name] = MIXAMO_PREFIX + side + "Hand"
                elif not own.startswith(side + "HandMiddle"):
                    merge[b.name] = MIXAMO_PREFIX + side + "Hand"
                else:
                    bones.append(b.name)
                break
        else:
            bones.append(b.name)
    return bones, merge


def _hand_models(rig):
    """The hands measured finger by finger at the cage fit (`hand_rig.measure`), or None."""
    if rig is None or "skin_fit" not in rig:
        return None
    hands = json.loads(rig["skin_fit"]).get("hands", {})
    models = {s: hands[s]["model"] for s in ("left", "right") if "model" in hands.get(s, {})}
    return models if len(models) == 2 else None


def _smooth01(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _rigid_parts(rig, points, w, bones, height, hand_labels=None, neck=None):
    """Head and mitten hands rigid, on a Mixamo-boned cage; the house rig is left as fitted.

    The cartoon's head is 25% of its height on a human-sized neck, and its hands are fingerless
    mittens: there is nothing in either that should bend. So every point above the head joint
    takes the Head bone outright, and the neck column below it a designed Neck -> Head gradient
    (see the comment in the body); every point past the wrist takes the Hand bone,
    fading in over HAND_BAND either side. Only points near the neck's axis (within 0.2 x height)
    or the forearm's line (within 0.08 x height) are touched, so a T-posed hand at shoulder height
    is never taken for the head. Works per coincident group, as every pass of the bake does.

    With `neck` (`_joint_blends` designed the whole neck, task 15) the column design here is not
    made, and the rigid head is `neck["head"]`: above the joint but for the nape's band, which the
    design carries from the neck into the head.
    """
    if rig_profile(rig) == "house":
        return w, None
    out = w.copy()
    world = rig.matrix_world
    pose = rig.pose.bones
    stats = {}
    hb = bones.index(_bn(rig, "head"))
    joint = np.array(world @ pose[_bn(rig, "head")].head)
    neck_r = json.loads(rig["skin_fit"]).get("neck_radius", 0.035 * height)
    # Height above the head joint, in world z (the bind pose stands upright; the Head bone itself
    # leans, and measuring along it put the top of a shoulder 2 cm below the joint "above" it).
    # Everything above the joint is head.
    #
    # Below it, the NECK COLUMN (within ~2 neck radii of the joint's vertical, fading out between
    # 1.5 and 2 so no hard edge is cut across the shoulders) is given a designed gradient — Neck
    # at the neck bone's root, Head at the head joint, linear in height — faded in over the
    # bottom third of the neck. The fitted weights there were noisy (Surface Deform on a cage neck
    # half the skin's girth), and the capture turns the head 11-30 degrees on the neck: the throat
    # crumpled into facets in every frame. A smooth gradient carries the turn up the neck instead.
    # The shoulder tops sit in the same height band 11 cm out to the side and are untouched (a
    # first version without the lateral limit gave them to the head: 131 mm off Surface Deform at
    # the arm swing; the bake's guard caught it).
    nk = bones.index(_bn(rig, "neck"))
    base = np.array(world @ pose[_bn(rig, "neck")].head)
    span = max(joint[2] - base[2], 1e-6)
    up = points[:, 2] - joint[2]
    across = np.linalg.norm(points[:, :2] - joint[:2], axis=1)
    t = np.clip((points[:, 2] - base[2]) / span, 0.0, 1.0)
    column = _smooth01((2.0 * neck_r - across) / (0.5 * neck_r))
    mix = _smooth01((points[:, 2] - base[2]) / (NECK_FADE * span)) * column * (up < 0.0)
    if neck is not None:
        mix = np.zeros_like(mix)
    designed = np.zeros_like(out)
    designed[:, hb] = t
    designed[:, nk] = 1.0 - t
    out = (1.0 - mix[:, None]) * out + mix[:, None] * designed
    stats["neck"] = {"designed": int((mix >= 1.0).sum()), "blended": int(((mix > 0) & (mix < 1)).sum())}
    share = (up >= 0.0).astype(float) if neck is None else neck["head"].astype(float)
    out = (1.0 - share[:, None]) * out
    out[:, hb] += share
    stats["head"] = {"rigid": int((share >= 1.0).sum()), "blended": int(((share > 0) & (share < 1)).sum())}
    for side in ("left", "right"):
        hand = _bn(rig, f"hand.{side}")
        k = bones.index(hand)
        wrist = np.array(world @ pose[hand].head)
        elbow = np.array(world @ pose[_bn(rig, f"forearm.{side}")].head)
        axis = (wrist - elbow) / np.linalg.norm(wrist - elbow)
        rel = points - wrist
        along = rel @ axis
        across = np.linalg.norm(rel - along[:, None] * axis, axis=1)
        share = (_smooth01((along + HAND_BAND * height) / (2 * HAND_BAND * height))
                 * (across < 0.08 * height))
        models = _hand_models(rig)
        if models is not None and hand_labels is not None:
            # Task 13: the cartoon's hand as it is — four separate fingers and a thumb, each on
            # its own chain fitted inside it (`hand_rig`). Weights designed per finger.
            import hand_rig
            designed = np.zeros_like(out)
            on = share > 0
            cols = [bones.index(b) for b in hand_rig.hand_bones(side.title())]
            designed[np.ix_(np.flatnonzero(on), cols)] = hand_rig.design_weights(
                points[on], hand_labels[side][on], models[side])
            out = (1.0 - share[:, None]) * out + share[:, None] * designed
            lab = hand_labels[side][share >= 1.0]
            stats[f"hand.{side}"] = {"rigid": int((share >= 1.0).sum()),
                                     "blended": int(((share > 0) & (share < 1)).sum()),
                                     "finger_vertices": {hand_rig.FINGERS[f]: int((lab == f).sum()) for f in range(5)}}
            continue
        # The mitten: the hand bone to the knuckles, then the middle finger's three bones in turn
        # toward the tip (MITTEN_JOINTS, as fractions of wrist-to-tip), so curling that finger
        # closes the whole front of the mitten like a cartoon fist.
        length = json.loads(rig["skin_fit"]).get("hands", {}).get(side, {}).get("length")
        designed = np.zeros_like(out)
        if length:
            frac = along / length
            m = [_smooth01((frac - a) / b) for a, b in MITTEN_JOINTS]
            # ...but not the thumb. The boy's hand is four fused fingers and a separate thumb,
            # pointing forward in the T-pose (palms down); curling it with the fingers crumpled
            # it into the palm. Points forward of THUMB_FRONT x the hand's length (from the
            # wrist's line) stay on the hand bone: a thumb laid along the bar.
            front = (points - wrist) @ np.array((0.0, 1.0, 0.0))
            fingers_only = 1.0 - _smooth01((front - THUMB_FRONT * length) / (0.1 * length))
            m = [mm * fingers_only for mm in m]
            fingers = [bones.index(_bn(rig, f"hand.{side}")[:-4] + f"HandMiddle{n}") for n in (1, 2, 3)]
            designed[:, k] = 1.0 - m[0]
            designed[:, fingers[0]] = m[0] - m[1]
            designed[:, fingers[1]] = m[1] - m[2]
            designed[:, fingers[2]] = m[2]
        else:
            designed[:, k] = 1.0
        out = (1.0 - share[:, None]) * out + share[:, None] * designed
        stats[f"hand.{side}"] = {"rigid": int((share >= 1.0).sum()),
                                 "blended": int(((share > 0) & (share < 1)).sum())}
    return out, stats


# --- the bake: from Surface Deform to an ordinary skinned mesh -----------------------------------

MAX_INFLUENCES = 4          # bones per vertex: the common real-time skinning limit
WEIGHT_FLOOR = 1e-3         # a bone share below this is dropped before the cap
COINCIDENT = 1e-5           # x height: vertices this close share ONE weight set (deformation()'s grid)
BAKE_TOLERANCE = 0.03       # x height: the most the bake may stray from Surface Deform at the check
NEAR = 1e-3                 # x height: "nearly coincident" for the second crack check
NEAR_SLACK = 3e-3           # x height: how much MORE than Surface Deform such a pair may open


def _cage_surface(cage):
    """The surface Surface Deform binds to — the cage evaluated through its armature, subsurf and
    triangulate — in world space, with each of its vertices' bone weights (vertex x BONES).

    Subsurf interpolates vertex groups the way it interpolates positions, so the weights of a
    subdivided vertex are the cage's own weights carried to where that vertex is.

    Which bones: the rig's profile says (`bake_bones`); a bone the bake leaves out hands its
    weight to the bone `merge` names (an MPFB cage's finger weights go to the hand).
    """
    rig = next(m.object for m in cage.modifiers if m.type == "ARMATURE")
    bones, merge = bake_bones(rig)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    ev = cage.evaluated_get(depsgraph)
    mesh = ev.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
    try:
        co = _co(mesh)
        world = np.array(cage.matrix_world)
        co = co @ world[:3, :3].T + world[:3, 3]
        mesh.calc_loop_triangles()
        tris = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int64)
        mesh.loop_triangles.foreach_get("vertices", tris)
        index = {cage.vertex_groups[b].index: i for i, b in enumerate(bones) if b in cage.vertex_groups}
        for src, dst in merge.items():
            if src in cage.vertex_groups and dst in bones:
                index[cage.vertex_groups[src].index] = bones.index(dst)
        weights = np.zeros((len(mesh.vertices), len(bones)))
        for vertex in mesh.vertices:
            for g in vertex.groups:
                if g.group in index:
                    weights[vertex.index, index[g.group]] += g.weight
    finally:
        ev.to_mesh_clear()
    total = weights.sum(axis=1, keepdims=True)
    if (total <= 1e-4).any():
        raise SystemExit(f"{int((total <= 1e-4).sum())} vertices of the evaluated cage carry no bone "
                         f"weight; there is nothing to read the skin's weights from there")
    return co, tris.reshape(-1, 3), weights / total, bones


def _coincident_groups(points, tol):
    """Label every point with a group; points within `tol` of each other (transitively) share one."""
    n = len(points)
    parent = np.arange(n)

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    kd = KDTree(n)
    for i, v in enumerate(points):
        kd.insert(Vector(v), i)
    kd.balance()
    for i, v in enumerate(points):
        for _, j, _ in kd.find_range(Vector(v), tol):
            a, b = root(i), root(j)
            if a != b:
                parent[max(a, b)] = min(a, b)
    return np.array([root(i) for i in range(n)])


def _near_pairs(points, mesh, reach, exact):
    """Pairs of vertices closer than `reach` but further apart than `exact`, not joined by an edge."""
    connected = {(min(a, b), max(a, b)) for a, b in _edges(mesh).tolist()}
    kd = KDTree(len(points))
    for i, v in enumerate(points):
        kd.insert(Vector(v), i)
    kd.balance()
    out = [(i, j) for i, v in enumerate(points) for _, j, d in kd.find_range(Vector(v), reach)
           if j > i and d > exact and (i, j) not in connected]
    return np.array(out, dtype=np.int64).reshape(-1, 2)


def _nearest_weights(points, co, tris, weights):
    """Bone weights at each point's nearest point on the triangulated surface, interpolated
    barycentrically across that triangle; plus the distance to it."""
    tree = BVHTree.FromPolygons([Vector(v) for v in co], tris.tolist(), all_triangles=True)
    where = np.empty((len(points), 3))
    face = np.empty(len(points), dtype=np.int64)
    dist = np.empty(len(points))
    for i, p in enumerate(points):
        loc, _, f, d = tree.find_nearest(Vector(p))
        if loc is None:
            raise SystemExit("a skin vertex found no nearest point on the cage")
        where[i], face[i], dist[i] = loc, f, d
    a, b, c = (co[tris[face, k]] for k in range(3))
    v0, v1, v2 = b - a, c - a, where - a
    d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
    d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
    den = np.maximum(d00 * d11 - d01 * d01, 1e-30)
    bv = (d11 * d20 - d01 * d21) / den
    bw = (d00 * d21 - d01 * d20) / den
    bary = np.clip(np.stack([1.0 - bv - bw, bv, bw], axis=1), 0.0, 1.0)
    bary /= np.maximum(bary.sum(axis=1, keepdims=True), 1e-12)
    out = np.einsum("vk,vkb->vb", bary, weights[tris[face]])
    return out, dist


def _limit(weights, floor, most):
    """At most `most` bones per vertex and none under `floor`, renormalised to sum to one — and
    CONTINUOUS in the weights, which a plain cut is not.

    The plain cut (zero everything under the floor, keep the `most` largest) was the first version
    and it opened cracks: at the back of the neck the cage gives five bones, `head` and
    `upper_arm` both about 0.09, and which of the two is fifth flips between skin vertices 0.7 mm
    apart. One kept `head`, its neighbour kept `upper_arm`, and the arm swing pulled the two
    8.5 mm apart where Surface Deform moved them 0.4 mm. Measured, not supposed (task 8 report).
    So instead every share is lowered by the (`most`+1)-th largest share at that vertex (and by
    `floor`) and clamped at zero: a bone fades out as it approaches the cut instead of dropping
    out at it, and two vertices with nearly the same weights keep nearly the same weights.
    """
    w = weights.copy()
    if most is not None and w.shape[1] > most:
        w = w - np.sort(w, axis=1)[:, -(most + 1)][:, None]
    w = np.maximum(w - floor, 0.0)
    total = w.sum(axis=1, keepdims=True)
    if (total <= 0).any():
        raise SystemExit("a skin vertex lost every bone weight to the floor/cap")
    return w / total


def _lbs_matrices(obj, rig, weights, bones):
    """Each vertex's Armature-modifier matrix in `obj`'s local space (see `_skin_matrices`)."""
    pre = np.array(rig.matrix_world.inverted() @ obj.matrix_world)
    per_bone = np.array([np.array(rig.pose.bones[b].matrix @ rig.data.bones[b].matrix_local.inverted())
                         for b in bones])
    return np.linalg.inv(pre) @ np.einsum("vb,bij->vij", weights, per_bone) @ pre


# --- the bake's weights: fitted to Surface Deform, then layered ----------------------------------
#
# Nearest-point sampling (`_nearest_weights`) reads each skin vertex's weights off ONE point of
# the cage. Two places on the skin that should move together can land on different points: a vest
# strap 5-8 mm over the shoulder projects onto the arm socket's ring ({upper_arm 0.55, clavicle
# 0.45}) while the skin under it projects beside that, onto the trunk's top ring (upper_arm
# 0.26-0.38), and the arm swing then slides the two 8-18 mm apart (task 5c). The same divergence
# swaps two overlapping vest panels in the overhead press (task 8). So the weights are made in
# two more steps, both measured in task 10:
#
#   1. FITTED to Surface Deform. Surface Deform's answer is computed at a spread of training
#      poses and each coincident group's weights are the convex mix of bones that best reproduces
#      it (least squares, non-negative, summing to one), starting from and gently held to the
#      nearest-point weights. This learns the bake from Surface Deform's own behaviour instead of
#      guessing it from one projection.
#   2. LAYERED. The garments in these meshes are separate welded shells lying over the body shell
#      (the boy's vest: 3,919 welded vertices, 78% within 8 mm of the body; no edge joins them), and
#      Surface Deform does NOT keep them together either: it slides the boy's strap 8.9 / 6.7 mm
#      (p95, left / right) over the skin beside it. So each vertex of a shell lying on another
#      takes the weights of the layer directly under it, and garment shells get a slightly wider
#      blend of their own. Details at `_layered_weights`.

FIT_RINGS = 4               # edge rings of the evaluated cage whose bones a skin vertex may use
FIT_PULL = 1e-4             # x height^2: how hard a fitted weight is held to the nearest-point one
FIT_BIND = 4.0              # the bind pose's share in the fit, against 1 for each training pose
FIT_ROUNDS = 2              # rest positions re-solved this many times (they depend on the weights)
FIT_STEPS = 300             # projected-gradient steps per round
FIT_CHUNK = 2048            # coincident groups solved at a time (memory, not results)
LAYER_NEAR = 0.010          # x height: a layer this close under a vertex hands it its weights fully...
LAYER_FAR = 0.016           # ...fading to nothing by this far
LAYER_GATE = (0.15, 0.30)   # ...and fading out as the two weight sets differ by this much (L1 / 2)
GARMENT_BLEND = 0.004       # x height: radius of the garment shells' own, slightly wider blend


def _pose_arms(rig, direction, forearm=None):
    """Both upper arms aimed along `direction` (x mirrored for the left), collarbones following
    as `arms_down` and the animation have them; optionally the forearms too."""
    for side, s in (("left", -1.0), ("right", 1.0)):
        final = Vector((direction[0] * s, direction[1], direction[2])).normalized()
        clavicle = rig.pose.bones[_bn(rig, f"clavicle.{side}")]
        along = (clavicle.tail - clavicle.head).normalized()
        _turn(rig, clavicle.name, Quaternion().slerp(along.rotation_difference(final), CLAVICLE_FOLLOW))
        _aim(rig, f"upper_arm.{side}", final)
        if forearm is not None:
            _aim(rig, f"forearm.{side}", (forearm[0] * s, forearm[1], forearm[2]))


def _pose_legs(rig, thigh, shin=None, left_thigh=None, left_shin=None):
    for side, s in (("left", -1.0), ("right", 1.0)):
        t = left_thigh if (side == "left" and left_thigh is not None) else thigh
        h = left_shin if (side == "left" and left_shin is not None) else shin
        if t is not None:
            _aim(rig, f"thigh.{side}", (t[0] * s, t[1], t[2]))
        if h is not None:
            _aim(rig, f"shin.{side}", (h[0] * s, h[1], h[2]))


def _pose_spine(rig, forward=0.0, side=0.0, twist=0.0):
    """Bend the spine by degrees. The thighs hang off it, so they are re-aimed where they pointed:
    otherwise the whole body turns as one and there is no bend at the waist to learn from."""
    thighs = {s: rig.pose.bones[_bn(rig, f"thigh.{s}")] for s in ("left", "right")}
    legs = {s: (b.tail - b.head).normalized() for s, b in thighs.items()}
    for axis, degrees in (((1, 0, 0), -forward), ((0, 1, 0), side), ((0, 0, 1), twist)):
        if degrees:
            # the house rig bends at its one `spine` bone; a Mixamo-boned rig shares the bend
            # between its pelvis and three spine joints (see "rig profiles" above)
            for bone, share in _spine_chain(rig):
                _turn(rig, bone, Quaternion(axis, math.radians(degrees * share)))
    for s, d in legs.items():
        _aim(rig, f"thigh.{s}", d)


def _turn_pair(rig, prefix, axis, degrees):
    for side in ("left", "right"):
        _turn(rig, f"{prefix}.{side}", Quaternion(axis, math.radians(degrees)))


def _training_poses():
    """The poses the bake's weights are fitted over, each applied on top of the skin pose: every
    joint on its own and the common combinations (arms down, out, overhead, forward, curled,
    racked; bent over, twisted; nodding; seated, squatting, lunging). Generic on purpose — no
    exercise data — so all 28 exercise poses stay a held-out test of the fit (task 10 report).
    Directions are armature space: +y is the character's front, +x its right."""
    down = (0.05, 0.0, -1.0)
    return [
        ("arms down", lambda r: _pose_arms(r, down)),
        ("arms down 45", lambda r: _pose_arms(r, (0.7, 0.0, -0.7))),
        ("arms up 60", lambda r: _pose_arms(r, (0.5, 0.0, 0.87))),
        ("arms overhead", lambda r: _pose_arms(r, (0.15, 0.1, 1.0))),
        ("arms forward", lambda r: _pose_arms(r, (0.1, 1.0, 0.0))),
        ("arms forward-down", lambda r: _pose_arms(r, (0.15, 0.7, -0.7))),
        ("arms back-down", lambda r: _pose_arms(r, (0.2, -0.5, -0.85))),
        ("arms across", lambda r: _pose_arms(r, (-0.35, 1.0, 0.1))),
        ("curl 90", lambda r: _pose_arms(r, (0.05, 0.05, -1.0), (0.0, 1.0, 0.0))),
        ("curl top", lambda r: _pose_arms(r, (0.05, 0.1, -1.0), (-0.1, 0.4, 1.0))),
        ("goalpost", lambda r: _pose_arms(r, (1.0, 0.0, -0.1), (0.0, 0.0, 1.0))),
        ("arms out, forearms forward", lambda r: _pose_arms(r, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
        ("front rack", lambda r: _pose_arms(r, (0.3, 0.6, -0.75), (-0.1, 0.2, 1.0))),
        ("press top", lambda r: _pose_arms(r, (0.25, 0.05, 1.0), (0.05, 0.0, 1.0))),
        ("pulldown", lambda r: _pose_arms(r, (0.6, 0.1, 0.8), (-0.1, 0.0, 1.0))),
        ("pushdown", lambda r: _pose_arms(r, (0.1, -0.05, -1.0), (0.05, 0.6, -0.8))),
        ("wrists flexed", lambda r: (_pose_arms(r, down), _turn_pair(r, "hand", (1, 0, 0), 45))),
        ("wrists turned", lambda r: (_pose_arms(r, (1.0, 0.0, 0.0)), _turn_pair(r, "hand", (0, 1, 0), 40))),
        ("bend 30", lambda r: _pose_spine(r, forward=30)),
        ("bend 70, arms hanging", lambda r: (_pose_spine(r, forward=70), _pose_arms(r, down))),
        ("lean back 20", lambda r: _pose_spine(r, forward=-20)),
        ("side bend 20", lambda r: _pose_spine(r, side=20)),
        ("twist 30", lambda r: (_pose_spine(r, twist=30), _pose_arms(r, (0.7, 0.0, -0.7)))),
        ("nod", lambda r: _turn(r, "neck", Quaternion((1, 0, 0), -math.radians(30)))),
        ("look up", lambda r: _turn(r, "head", Quaternion((1, 0, 0), math.radians(25)))),
        ("head turn", lambda r: _turn(r, "head", Quaternion((0, 0, 1), math.radians(40)))),
        ("head tilt", lambda r: _turn(r, "neck", Quaternion((0, 1, 0), math.radians(20)))),
        ("seated", lambda r: _pose_legs(r, (0.05, 1.0, -0.05), (0.0, 0.1, -1.0))),
        ("squat", lambda r: (_pose_spine(r, forward=35), _pose_arms(r, (0.1, 1.0, 0.0)),
                             _pose_legs(r, (0.15, 1.0, -0.15), (0.05, 0.45, -0.9)))),
        ("half squat", lambda r: _pose_legs(r, (0.1, 0.7, -0.7), (0.0, 0.35, -0.95))),
        ("lunge", lambda r: _pose_legs(r, (0.05, -0.45, -0.9), (0.0, -0.9, -0.45),
                                       left_thigh=(0.05, 0.85, -0.5), left_shin=(0.0, -0.1, -1.0))),
        ("legs apart", lambda r: _pose_legs(r, (0.5, 0.0, -0.87))),
        ("legs back", lambda r: _pose_legs(r, (0.05, -0.35, -0.94))),
        ("knees bent back", lambda r: _pose_legs(r, (0.05, 0.0, -1.0), (0.0, -0.9, -0.45))),
        ("ankles", lambda r: _turn_pair(r, "foot", (1, 0, 0), 25)),
        ("deadlift", lambda r: (_pose_spine(r, forward=50), _pose_arms(r, down),
                                _pose_legs(r, (0.05, 0.5, -0.87), (0.0, -0.1, -1.0)))),
    ]


def _bone_matrices(obj, rig, bones):
    """Each bone's skinning matrix in `obj`'s local space: the Armature modifier moves a vertex
    by sum_b w_b K_b (the per-vertex blend `_lbs_matrices` builds, kept per bone)."""
    pre = np.array(rig.matrix_world.inverted() @ obj.matrix_world)
    per_bone = np.array([np.array(rig.pose.bones[b].matrix @ rig.data.bones[b].matrix_local.inverted())
                         for b in bones])
    return np.linalg.inv(pre) @ per_bone @ pre


def _simplex(v, allowed):
    """Euclidean projection of each row onto {w >= 0, sum w = 1, w = 0 where not allowed}."""
    u = np.where(allowed, v, -1e9)
    s = -np.sort(-u, axis=1)
    css = np.cumsum(s, axis=1) - 1.0
    rho = (s - css / np.arange(1, v.shape[1] + 1) > 0).sum(axis=1)
    theta = css[np.arange(len(v)), rho - 1] / rho
    return np.maximum(u - theta[:, None], 0.0)


def _allowed_bones(points, co, tris, weights, rings):
    """The bones a skin point may be fitted with: those carrying weight on the evaluated cage
    within `rings` edge rings of its nearest cage triangle. Topological, so a thigh vertex never
    sees the other thigh however close they stand (task 8's 3-D blur failed exactly there)."""
    tree = BVHTree.FromPolygons([Vector(v) for v in co], tris.tolist(), all_triangles=True)
    face = np.array([tree.find_nearest(Vector(p))[2] for p in points])
    present = weights > 1e-3
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    for _ in range(rings):
        grown = present.copy()
        np.logical_or.at(grown, e[:, 0], present[e[:, 1]])
        np.logical_or.at(grown, e[:, 1], present[e[:, 0]])
        present = grown
    return present[tris[face, 0]] | present[tris[face, 1]] | present[tris[face, 2]]


def _fitted_weights(skin, rig, heads, w0, allowed, height):
    """Per coincident group (`heads`, indices into the skin), the convex bone mix whose linear
    blend skinning best reproduces Surface Deform's position over `_training_poses()`.

    The rest position a weight set implies is itself a function of the weights (the bake inverts
    the blended matrix at the bind), so the fit alternates: rest positions from the current
    weights, then weights by projected (accelerated) gradient on the simplex, minimising
        sum over poses ||sum_b w_b K_b r - s||^2  +  FIT_BIND x the same at the bind
        + FIT_PULL x height^2 x ||w - w0||^2.
    `skin` must still be bound by Surface Deform; the rig is left in the skin pose.
    Returns (weights, stats).
    """
    t0 = time.process_time()
    bones = bake_bones(rig)[0]
    to_local = np.linalg.inv(np.array(skin.matrix_world))
    x = _co(skin.data)[heads]
    xh = np.c_[x, np.ones(len(x))]
    to_skin_pose(rig)
    k_bind = _bone_matrices(skin, rig, bones)
    targets, mats = [], []
    # Posing turns one bone at a time and updates the scene after each turn; with Surface Deform
    # and the cage's subsurf live, every one of those updates re-evaluates the whole skin. They
    # are switched off while the bones are set and back on for the one evaluation that is read
    # (the bind is kept: visibility does not unbind). Same answer, about half the time.
    deform = next(m for m in skin.modifiers if m.type == "SURFACE_DEFORM")
    heavy = [deform] + [m for m in deform.target.modifiers if m.type in ("SUBSURF", "TRIANGULATE")]
    try:
        for _, pose in _training_poses():
            for m in heavy:
                m.show_viewport = False
            to_skin_pose(rig)
            pose(rig)
            for m in heavy:
                m.show_viewport = True
            bpy.context.view_layer.update()
            s = evaluated(skin)[0][heads]
            targets.append(s @ to_local[:3, :3].T + to_local[:3, 3])
            mats.append(_bone_matrices(skin, rig, bones))
    finally:
        for m in heavy:
            m.show_viewport = True
    to_skin_pose(rig)
    if not deform.is_bound:
        raise SystemExit("Surface Deform came unbound while the training poses were set")
    t_posed = time.process_time()

    lam = FIT_PULL * height * height
    n, nb = w0.shape
    w = w0.copy()
    before, after = np.empty((len(mats), n)), np.empty((len(mats), n))
    # Group by group the problems are independent, so they are solved FIT_CHUNK at a time: all
    # at once is a few hundred MB of temporaries, and on a machine already deep in swap that
    # turned a 20-second solve into five minutes of page-ins (task 10, measured).
    for lo in range(0, n, FIT_CHUNK):
        sl = slice(lo, min(lo + FIT_CHUNK, n))
        xs, xhs, w0s, ok = x[sl], xh[sl], w0[sl], allowed[sl]
        goals = [s[sl] for s in targets]

        def rest_of(ww):
            return np.einsum("vij,vj->vi", np.linalg.inv(np.einsum("vb,bij->vij", ww, k_bind)), xhs)

        def error(ww):
            r = rest_of(ww)
            return np.array([np.linalg.norm(np.einsum("vb,bij,vj->vi", ww, k, r)[:, :3] - s, axis=1)
                             for k, s in zip(mats, goals)])

        ws = w0s.copy()
        for _ in range(FIT_ROUNDS):
            r = rest_of(ws)
            gram = np.zeros((len(xs), nb, nb))
            for k, s in zip(mats, goals):
                d = np.einsum("bij,vj->vib", k[:, :3, :], r) - s[:, :, None]
                gram += np.einsum("vib,vic->vbc", d, d)
            d = np.einsum("bij,vj->vib", k_bind[:, :3, :], r) - xs[:, :, None]
            gram += FIT_BIND * np.einsum("vib,vic->vbc", d, d)
            q = gram + lam * np.eye(nb)[None]
            step = (0.5 / np.linalg.eigvalsh(q)[:, -1])[:, None]
            pull = 2.0 * lam * w0s
            y, prev, t = ws.copy(), ws.copy(), 1.0
            for _ in range(FIT_STEPS):
                nxt = _simplex(y - step * (2.0 * np.einsum("vbc,vc->vb", q, y) - pull), ok)
                t_next = (1 + math.sqrt(1 + 4 * t * t)) / 2
                y = nxt + ((t - 1) / t_next) * (nxt - prev)
                prev, t = nxt, t_next
            ws = prev
        w[sl] = ws
        before[:, sl], after[:, sl] = error(w0s), error(ws)
    stats = {"poses": len(mats), "cpu_seconds_posing_solving": [t_posed - t0, time.process_time() - t_posed],
             "vs_surface_deform_mm_mean_p99": {"nearest": [float(before.mean() * 1000), float(np.percentile(before, 99) * 1000)],
                                               "fitted": [float(after.mean() * 1000), float(np.percentile(after, 99) * 1000)]}}
    return w, stats


def _shells(skin, back, n):
    """Welded shells: each coincident group's connected component once coincident vertices are
    merged, and the welded edges (as pairs of group indices)."""
    edges = back[_edges(skin.data)]
    edges = np.unique(np.sort(edges[edges[:, 0] != edges[:, 1]], axis=1), axis=0)
    parent = np.arange(n)

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in edges:
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return np.array([root(i) for i in range(n)]), edges


def _layered_weights(points, w, co, tris, skin, back, height):
    """Garment shells take the weights of the layer under them, then a slightly wider blend.

    `points` are the coincident groups' bind positions (world), `w` their weights, `back` maps
    skin vertices to groups. Four rules, each there because the version without it failed a
    measurement (task 10 report):

      * Which of two shells is UNDER the other is decided once per pair of shells: the median,
        where they come within LAYER_NEAR of each other, of how much further from the cage one
        is than the other. A per-vertex test split a thin two-sheet strap into sheets with
        different weights. A shell never counts as under one less than half its size, so the
        body never takes its weights from an eyeball.
      * Every group on a shell that lies over another takes the weights of the nearest point on
        a shell under it (barycentric over that triangle): fully within LAYER_NEAR, fading out by
        LAYER_FAR.
      * ...and fading out as the two weight sets differ by LAYER_GATE. The layer under a garment
        corrects its weights; it never hands it to another limb. Without this the shorts at the
        bottom of a squat took the OTHER thigh's weights (62 mm off Surface Deform on the boy,
        89 mm on the man).
      * Then each garment shell is blended over GARMENT_BLEND of its own surface, by POSITION
        (so a seam's two sides a millimetre apart, which no short path of edges joins, get the
        same answer), never across shells and never between weight sets more than LAYER_GATE[1]
        apart. Without it the transfer split such seams and the near-pair guard caught a 5 mm
        crack at the right strap; it is also the garment's "slightly wider blend than skin".
    Two passes, so a layer over a layer reads the finished layer under it.
    """
    t0 = time.process_time()
    near, far = LAYER_NEAR * height, LAYER_FAR * height
    shell, welded = _shells(skin, back, len(points))
    cage_tree = BVHTree.FromPolygons([Vector(v) for v in co], tris.tolist(), all_triangles=True)
    depth = np.empty(len(points))
    for i, p in enumerate(points):
        loc, normal, _, _ = cage_tree.find_nearest(Vector(p))
        depth[i] = (np.array(p) - np.array(loc)) @ np.array(normal)
    skin.data.calc_loop_triangles()
    st = np.empty(len(skin.data.loop_triangles) * 3, dtype=np.int64)
    skin.data.loop_triangles.foreach_get("vertices", st)
    st = back[st.reshape(-1, 3)]
    ids, sizes = np.unique(shell, return_counts=True)
    size = dict(zip(ids.tolist(), sizes.tolist()))
    trees = {}
    for c in ids.tolist():
        own = st[shell[st[:, 0]] == c]
        if len(own):
            trees[c] = (own, BVHTree.FromPolygons([Vector(v) for v in points], own.tolist(), all_triangles=True))
    hits = {}
    for i, p in enumerate(points):
        for c, (own, tree) in trees.items():
            if c == shell[i]:
                continue
            loc, _, f, d = tree.find_nearest(Vector(p), far)
            if loc is None:
                continue
            tri = own[f]
            a, b, cc = points[tri]
            v0, v1, v2 = b - a, cc - a, np.array(loc) - a
            d00, d01, d11, d20, d21 = v0 @ v0, v0 @ v1, v1 @ v1, v2 @ v0, v2 @ v1
            den = max(d00 * d11 - d01 * d01, 1e-30)
            bv, bw = (d11 * d20 - d01 * d21) / den, (d00 * d21 - d01 * d20) / den
            bary = np.clip([1.0 - bv - bw, bv, bw], 0.0, 1.0)
            hits[(i, c)] = (d, tri, bary / bary.sum())
    diffs = {}
    for (i, c), (d, tri, bary) in hits.items():
        if d <= near:
            diffs.setdefault((int(shell[i]), c), []).append(depth[i] - bary @ depth[tri])
    under = {}
    for (a, c), dd in diffs.items():
        score = np.median(dd) - (np.median(diffs[(c, a)]) if (c, a) in diffs else 0.0)
        if len(dd) >= 10 and score > 0 and size[c] >= 0.5 * size[a]:
            under.setdefault(a, set()).add(c)
    src_tri = np.zeros((len(points), 3), dtype=np.int64)
    src_bary = np.zeros((len(points), 3))
    src_dist = np.full(len(points), np.inf)
    for (i, c), (d, tri, bary) in hits.items():
        if c in under.get(int(shell[i]), ()) and d < src_dist[i]:
            src_tri[i], src_bary[i], src_dist[i] = tri, bary, d
    has = np.isfinite(src_dist)
    t = np.clip((src_dist[has] - near) / (far - near), 0.0, 1.0)
    reach = 1.0 - t * t * (3.0 - 2.0 * t)
    out = w.copy()
    for _ in range(2):
        below = np.einsum("vk,vkb->vb", src_bary[has], out[src_tri[has]])
        g = np.clip((np.abs(below - w[has]).sum(axis=1) / 2 - LAYER_GATE[0])
                    / (LAYER_GATE[1] - LAYER_GATE[0]), 0.0, 1.0)
        share = reach * (1.0 - g * g * (3.0 - 2.0 * g))
        out = w.copy()
        out[has] = (1.0 - share[:, None]) * w[has] + share[:, None] * below
    garment = np.isin(shell, list(under))
    radius = GARMENT_BLEND * height
    idx = np.flatnonzero(garment)
    kd = KDTree(len(idx))
    for k, i in enumerate(idx):
        kd.insert(Vector(points[i]), k)
    kd.balance()
    blended = out.copy()
    for i in idx:
        found = [(idx[j], d) for _, j, d in kd.find_range(Vector(points[i]), radius)
                 if shell[idx[j]] == shell[i] and np.abs(out[idx[j]] - out[i]).sum() / 2 < LAYER_GATE[1]]
        js = np.array([j for j, _ in found])
        kernel = (1.0 - (np.array([d for _, d in found]) / radius) ** 2) ** 2
        blended[i] = (kernel[:, None] * out[js]).sum(axis=0) / kernel.sum()
    stats = {"cpu_seconds": time.process_time() - t0, "shells": len(ids), "garment_shells": {int(a): sorted(int(c) for c in s) for a, s in under.items()},
             "garment_groups": int(garment.sum()), "taking_from_under": int((has & garment).sum()),
             "weight_change_l1_p95_max": [float(np.percentile(np.abs(blended - w).sum(1)[garment] / 2, 95)),
                                          float((np.abs(blended - w).sum(1) / 2).max())] if garment.any() else None}
    return blended, stats


# --- the joints: the neck designed, the trunk's blends relaxed (task 15) --------------------------
#
# Linear blend skinning turns a face over where its corners are carried by bones that turn far
# apart and the weights change too fast across it. The Studio's folded-skin check found five such
# places on the MPFB squat, every one a near-step in the weights or a wide turn blended too tightly
# (task 15 report, measured on the live run's weights and motion):
#   * the neck: down the middle of the throat the skin went from Spine2 0.79 straight to Head 1.00
#     within 3 mm, and at the nape from Spine2 0.84 to Head 1.00 within 9 mm; the head tilts back
#     21-34 degrees on the chest through the rep, so Head-carried skin swung 2-4 cm past the
#     chest-carried skin beside it (3,281 mm² folded at the nape, 1,161 at the throat). The throat's
#     step was the cage sampling (the skin there is 30-47 mm from the cage, and neighbouring skin
#     vertices read the cage's jaw and its chest); the nape's was `_rigid_parts`' hard head edge,
#     its designed neck spanning only the Neck bone's 21 mm and only round the head joint's vertical;
#   * the belly (tank top): the fit made Spine2 0.87 against Spine1 0.59 within 8 mm (sampled: a
#     0.31 jump; fitted: 0.92), the three spine bones being nearly interchangeable to the fit;
#   * the shorts' front seam: the cage's left and right thighs meet on the midline, 0.8 apart;
#   * the shoulders and collarbone: the upper arm twists ~100 degrees about itself to hold the bar,
#     and the deltoid's Arm / Shoulder / Spine2 blend shears through itself (the cage folds there too).
# So, on the Mixamo-boned profile, after the layers: the neck is DESIGNED over its whole surface
# (`_design_neck`), and each trunk joint's blend is RELAXED to the smoothest field its own bones
# allow (`_relax_joints`). Measured off-line on the live run's motion (linear blend skinning
# recomputed to 0.0002 mm of Blender's): every one of the five places goes to no patch over the
# check's limit in any frame, and no new one appears.

NECK_REACH = 0.10           # x height: the neck pass covers the skin this far from the neck's centre
# x height: behind the neck, the head's bottom this far above its joint is neck. 0.03-0.06 all
# leave no fold; 0.05 took enough of the nape's hair below Head 0.5 that `mpfb_cage._head_clearance`
# (which reads the head as Head > 0.5) let the bar ride a step higher, and the arms followed it.
NAPE_BAND = 0.03
NECK_COVER = 0.008          # x height: skin this close under a garment is held to the trunk with it
NECK_ARMS = 0.6             # x NECK_REACH: within this the arms carry none of the neck
NECK_PEAK = 0.5             # the Neck bone's share at the middle of the neck
RELAX_REACH = 0.04          # x height: a joint's blend zone is widened this far along the skin
RELAX_EPS = 0.02            # a bone with at least this share takes part in a blend
SPINE_STEP = 0.3            # an edge across which this much of the spine's weight changes hands is a step
SPINE_REACH = 0.02          # x height: ...and the spine is relaxed this far round each step
HARMONIC_STEPS = 3000       # conjugate-gradient steps at most (they converge in a few hundred)
SOLE_KEEP = 0.5             # x the ankle joints' height above the skin's lowest point: below it, not relaxed
# (joint, where its zone starts, how far it is widened, the bones its blend is relaxed among).
# The shoulders and hips start where their two sides' bones blend. The spine starts only at steps:
# its three bones blend over the whole torso, and relaxing all of that redistributed the fitted
# torso wholesale (5,251 points changed by over 0.2; the pelvis's weight, given a part, climbed to
# the shoulder blades on 2,512), where the belly's fault was one step the fit made.
RELAX_JOINTS = (
    ("spine", ("step", SPINE_STEP), SPINE_REACH, ("Spine", "Spine1", "Spine2")),
    ("left shoulder", ("blend", ("LeftArm",), ("LeftShoulder", "Spine2", "Spine1")), RELAX_REACH,
     ("LeftArm", "LeftShoulder", "Spine2", "Spine1")),
    ("right shoulder", ("blend", ("RightArm",), ("RightShoulder", "Spine2", "Spine1")), RELAX_REACH,
     ("RightArm", "RightShoulder", "Spine2", "Spine1")),
    # the collarbones against the chest, where the vest's back crosses the upper back: the shoulders'
    # relaxation leaves a 0.45 step there (Spine2 0.98 beside Spine2 0.52 / LeftShoulder 0.46, 15 mm
    # apart) that turned over 130-200 mm² of the vest's back at standing (full run 3, hidden)
    ("collarbones", ("step", SPINE_STEP), SPINE_REACH, ("LeftShoulder", "RightShoulder", "Spine2")),
    ("hips", ("blend", ("LeftUpLeg",), ("RightUpLeg",)), RELAX_REACH, ("LeftUpLeg", "RightUpLeg", "Hips")),
    # the ankles: with the back straight and the coach's lean met at the bottom (task 16) the shins come
    # 2 degrees further over the feet, and the front of the right ankle (Foot 0.64 / Leg 0.36) turned
    # over 168 mm2 at the deepest frame (66 before, under the limit)
    ("left ankle", ("blend", ("LeftFoot",), ("LeftLeg",)), RELAX_REACH, ("LeftLeg", "LeftFoot")),
    ("right ankle", ("blend", ("RightFoot",), ("RightLeg",)), RELAX_REACH, ("RightLeg", "RightFoot")),
)


class _Surface:
    """The skin's welded surface as a graph over the coincident groups: harmonic interpolation
    (the graph Laplace equation, each edge conducting 1 / its length) and growing a set along it.
    Points closer than NEAR x height that no edge joins (a seam's two sides, a garment lying on
    the skin) are one node of the graph, so whatever is solved here gives them the same value:
    solved apart, the shorts' crotch seam opened 7 mm through the squat and the vest parted from
    the skin at the back of the shoulder (the bake's near-pair guard refused it: 3.3 mm against
    3.0). Pure numpy: Blender's Python has no scipy."""

    def __init__(self, points, edges, height):
        edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        self.point_edges = edges
        n = len(points)
        # one node per cluster of points within NEAR x height of each other
        node = _coincident_groups(points, NEAR * height)
        ids, node = np.unique(node, return_inverse=True)
        self.node = node.ravel()
        self.m = len(ids)
        self.count = np.bincount(self.node, minlength=self.m).astype(float)
        centre = np.zeros((self.m, 3))
        np.add.at(centre, self.node, points)
        centre /= self.count[:, None]
        e = self.node[edges]
        e = np.unique(np.sort(e[e[:, 0] != e[:, 1]], axis=1), axis=0)
        self.e = e
        length = np.linalg.norm(centre[e[:, 0]] - centre[e[:, 1]], axis=1)
        self.length = np.maximum(length, 1e-9)
        self.k = 1.0 / np.maximum(length, 1e-4)
        self.deg = np.bincount(e[:, 0], self.k, self.m) + np.bincount(e[:, 1], self.k, self.m)
        self.n = n

    def _adjacent(self, x):
        a, b, k = self.e[:, 0], self.e[:, 1], self.k[:, None]
        out = np.zeros_like(x)
        np.add.at(out, a, k * x[b])
        np.add.at(out, b, k * x[a])
        return out

    def _any(self, mask):
        return np.bincount(self.node, mask.astype(float), self.m) > 0

    def grow(self, mask, reach):
        """The points within `reach` of `mask`, measured along the edges."""
        dist = np.where(self._any(mask), 0.0, np.inf)
        a, b, ln = self.e[:, 0], self.e[:, 1], self.length
        while True:
            step = np.full(self.m, np.inf)
            np.minimum.at(step, a, dist[b] + ln)
            np.minimum.at(step, b, dist[a] + ln)
            step[step > reach] = np.inf
            new = np.minimum(dist, step)
            if np.array_equal(new, dist):
                return (dist <= reach)[self.node]
            dist = new

    def harmonic(self, free, values):
        """`values` (n x k) with the rows in `free` replaced by the harmonic interpolation of the
        others: each free node the conductance-weighted mean of its neighbours. A node is free when
        all its points are; a free point of a node that is not takes the node's mean. A free
        component with no fixed neighbour keeps its own values (nothing to interpolate from)."""
        values = np.array(values, dtype=float)
        sums = np.zeros((self.m, values.shape[1]))
        np.add.at(sums, self.node, values)
        x = sums / self.count[:, None]
        free_m = np.bincount(self.node, free.astype(float), self.m) >= self.count - 0.5
        a, b = self.e[:, 0], self.e[:, 1]
        reach = np.zeros(self.m, bool)
        reach[a[free_m[a] & ~free_m[b]]] = True
        reach[b[free_m[b] & ~free_m[a]]] = True
        both = free_m[a] & free_m[b]
        while True:
            more = reach.copy()
            more[b[both & reach[a]]] = True
            more[a[both & reach[b]]] = True
            if np.array_equal(more, reach):
                break
            reach = more
        solve = free_m & reach
        if solve.any():
            fixed = x.copy()
            fixed[solve] = 0.0
            rhs = self._adjacent(fixed)[solve]

            def apply(y):
                full = np.zeros_like(x)
                full[solve] = y
                return self.deg[solve][:, None] * y - self._adjacent(full)[solve]

            y = x[solve].copy()
            r = rhs - apply(y)
            p = r.copy()
            rs = (r * r).sum(0)
            tol = 1e-10 * np.maximum((rhs * rhs).sum(0), 1e-30)
            for _ in range(HARMONIC_STEPS):
                if np.all(rs <= tol):
                    break
                ap = apply(p)
                alpha = rs / np.maximum((p * ap).sum(0), 1e-300)
                y += alpha * p
                r -= alpha * ap
                new = (r * r).sum(0)
                p = r + (new / np.maximum(rs, 1e-300)) * p
                rs = new
            x[solve] = y
        out = values.copy()
        out[free] = x[self.node][free]
        return out


def _neck_centre_y(points, body, joint, height, cx):
    """The y of the neck's middle: the middle of the body's front-to-back extent in thin slabs 2-5%
    of the height under the head joint, near the centre line. (The joint itself is placed in the
    slab of the jaw, which reaches forward to the chin: on the boy it sits on the throat's surface,
    3 cm in front of the neck's middle.)"""
    found = []
    for k in (0.02, 0.03, 0.04, 0.05):
        s = points[body & (np.abs(points[:, 2] - (joint[2] - k * height)) <= 0.006 * height)
                   & (np.abs(points[:, 0] - cx) <= 0.012 * height)]
        if len(s) >= 6:
            y0, y1 = np.percentile(s[:, 1], [2, 98])
            found.append((y0 + y1) / 2)
    return float(np.median(found)) if found else float(joint[1])


def _design_neck(surface, points, w, col, shell, body, joint, height, cx, bones):
    """The neck, chest to head, over the whole neck: Spine2 (with whatever trunk bones it had) at
    the bottom, the Neck bone in the middle, Head at the top.

    Where: the skin within NECK_REACH of the neck's middle that is not head. The head is everything
    above the head joint except, behind the neck's middle, a NAPE_BAND above it: the nape needs the
    room. The head joint sits at the front of the boy's neck, so a 34 degree tilt moves the nape's
    Head-carried skin ~4 cm against the chest's; a blend only the neck's 3-4 cm tall there cannot
    take that without passing through itself (measured: 1,264-1,783 mm² folded with the blend
    stopped at the joint).
    How: a neck coordinate t, 0 on the trunk round the zone and 1 on the head, harmonic over the
    skin's surface, so it climbs evenly however long the neck is at the front or the back. The
    weights are then (1 - h - n) x the trunk's own + n x Neck + h x Head, h = smoothstep(t) and
    n = NECK_PEAK x 4t(1-t) x (1-h).
    Two rules keep the vest on the skin:
      * a garment shell in the zone (the vest's collar) has no head on it, so its t is 0: it is
        carried by the trunk, as a vest hangs on the shoulders. Its collar had read up to 0.44
        Head/Neck off the cage; taking the skin's t instead folded the collar's back (309 mm²);
      * the skin under a garment (within NECK_COVER of it) is held at t = 0 too. Left free, the
        throat slid up to 55 mm against the collar lying on it through the squat (p95 38 mm; the
        live run: 42 / 6.5), and showed through the collar's edge in the GIF.
    And the arms are faded out of the zone's trunk weights (gone within NECK_ARMS of its reach,
    whole again at its rim): the cage leaks 0.07-0.16 of an upper arm onto the throat, and an arm
    twisted 100 degrees about itself folded the throat above the collar (224-255 mm²) whatever the
    neck's own blend. Settings 0.5-0.7 x reach, cover 0.006-0.010 and nape band 0.04-0.06 all gave
    no fold at any of the five places and nothing new (off-line, the live motion).
    Returns (weights, designed mask, rigid-head mask, stats)."""
    hk, nk, s2 = col["Head"], col["Neck"], col["Spine2"]
    on_body = shell == body
    yc = _neck_centre_y(points, on_body, joint, height, cx)
    centre = np.array([cx, yc, joint[2] - 0.02 * height])
    reach = np.linalg.norm(points - centre, axis=1) / (NECK_REACH * height)
    near = reach < 1.0
    nape = near & (points[:, 1] < yc) & (points[:, 2] < joint[2] + NAPE_BAND * height)
    head = (points[:, 2] >= joint[2]) & ~nape
    zone = near & ~head
    covered = np.zeros(len(points), bool)
    garment = np.flatnonzero(~on_body & zone)
    if len(garment):
        kd = KDTree(len(garment))
        for k, i in enumerate(garment):
            kd.insert(Vector(points[i]), k)
        kd.balance()
        for i in np.flatnonzero(on_body & zone):
            covered[i] = kd.find(Vector(points[i]))[2] < NECK_COVER * height
    t = surface.harmonic(zone & ~covered, np.where(head, 1.0, 0.0)[:, None])[:, 0]
    t = np.where(covered, 0.0, np.clip(t, 0.0, 1.0))
    h = _smooth01(t)
    n = NECK_PEAK * 4.0 * t * (1.0 - t) * (1.0 - h)
    trunk = w.copy()
    trunk[:, [hk, nk]] = 0.0
    arms = [i for i, b in enumerate(bones) if any(part in b for part in ("Arm", "Hand"))]
    trunk[:, arms] *= _smooth01((reach - NECK_ARMS) / (1.0 - NECK_ARMS))[:, None]
    trunk[trunk.sum(1) <= 1e-9, s2] = 1.0
    trunk /= trunk.sum(1, keepdims=True)
    design = (1.0 - h - n)[:, None] * trunk
    design[:, nk] += n
    design[:, hk] += h
    out = w.copy()
    out[zone] = design[zone]
    return out, zone, head, {"neck_centre_y": round(yc, 4), "designed": int(zone.sum()),
                             "skin_under_garment": int(covered.sum()),
                             "nape_band_above_joint": int((nape & (points[:, 2] >= joint[2])).sum())}


def _relax_joints(surface, w, col, protect, height):
    """Each trunk joint's blend relaxed to the smoothest field its own bones allow.

    For each of RELAX_JOINTS: the zone starts at every point where the two sides' bones both have
    RELAX_EPS or more ("blend"), or at both ends of every edge across which the joint's bones'
    weights change by the step or more ("step"); it is widened along the skin and `protect` is
    taken out. Inside it the joint's bones' weights are replaced by their harmonic interpolation
    from the zone's edge, the other bones' left as they were. Among the joint's own bones only:
    relaxing all 52 spread small shares of many bones, the 4-influence limit then cut them
    unevenly, and that folded the upper back (2,014 mm², off-line)."""
    out = w.copy()
    stats = {}
    for name, start, reach, among in RELAX_JOINTS:
        k = [col[x] for x in among]
        if start[0] == "blend":
            a = [col[x] for x in start[1]]
            b = [col[x] for x in start[2]]
            seed = (out[:, a].sum(1) > RELAX_EPS) & (out[:, b].sum(1) > RELAX_EPS)
        else:
            i, j = surface.point_edges[:, 0], surface.point_edges[:, 1]
            jump = np.abs(out[i][:, k] - out[j][:, k]).sum(1) / 2
            seed = np.zeros(len(out), bool)
            seed[i[jump >= start[1]]] = True
            seed[j[jump >= start[1]]] = True
        zone = surface.grow(seed, reach * height) & ~protect
        relaxed = np.clip(surface.harmonic(zone, out[:, k]), 0.0, None)
        ok = zone & (relaxed.sum(1) > 1e-6)
        new = out.copy()
        new[np.ix_(ok, k)] = relaxed[ok]
        new /= new.sum(1, keepdims=True)
        stats[name] = {"points": int(ok.sum()),
                       "weight_change_l1_max": round(float(np.abs(new - out).sum(1).max() / 2), 3)}
        out = new
    return out, stats


def _joint_blends(rig, points, w, bones, skin, back, height):
    """The neck designed and the trunk's joint blends relaxed, on the Mixamo-boned profile (the
    section above says why); the house rig is left as fitted. Returns (weights, stats, neck) where
    `neck` tells `_rigid_parts` which points the design made (it no longer makes its own neck
    there) and which head points stay rigid."""
    if rig_profile(rig) == "house":
        return w, None, None
    t0 = time.process_time()
    shell, welded = _shells(skin, back, len(points))
    ids, sizes = np.unique(shell, return_counts=True)
    body = ids[np.argmax(sizes)]
    surface = _Surface(points, welded, height)
    col = {b[len(MIXAMO_PREFIX):]: i for i, b in enumerate(bones) if b.startswith(MIXAMO_PREFIX)}
    joint = np.array(rig.matrix_world @ rig.pose.bones[_bn(rig, "head")].head)
    cx = json.loads(rig["skin_fit"])["origin"][0]
    out, designed, head, neck_stats = _design_neck(surface, points, w, col, shell, body, joint, height, cx, bones)
    hands = out[:, [i for i, b in enumerate(bones) if "Hand" in b]].sum(1) > RELAX_EPS
    # the soles keep their own weights: relaxed with the ankles, the shin came to carry the heel and the
    # planted feet slid (1.1 / 1.4 -> 3.0 / 4.4 mm, task 16 full run 4)
    ankle = min(float((rig.matrix_world @ rig.pose.bones[MIXAMO_PREFIX + s + "Foot"].head).z) for s in ("Left", "Right"))
    lowest = float(points[:, 2].min())
    soles = points[:, 2] < lowest + SOLE_KEEP * (ankle - lowest)
    out, relax_stats = _relax_joints(surface, out, col, designed | head | hands | soles, height)
    stats = {"neck": neck_stats, "relaxed": relax_stats, "cpu_seconds": round(time.process_time() - t0, 1)}
    return out, stats, {"designed": designed, "head": head}


def bake_to_skin(cage, rig, skin, max_influences=MAX_INFLUENCES, tolerance=BAKE_TOLERANCE):
    """Replace the skin's Surface Deform with real bone weights and an Armature modifier. Returns
    the skin, now an ordinary skinned mesh parented to `rig`.

    Why. Surface Deform is the right way to MAKE the skin follow the cage, and it cannot leave
    Blender: the USD writer bakes it to per-frame points and binds nothing to the skeleton, and
    SceneKit plays that file as a still figure (task 7, measured three ways). The writer exports a
    mesh as skeletal only through an Armature modifier. So once the skin is bound and checked,
    the binding is converted, once, into the form every exporter and player understands.

    How — four steps, all at the BIND pose, where Surface Deform measured its own binding:
      1. Weights, in three passes (the section above this function says why):
           a. sampled: each skin vertex takes the bone weights of its nearest point on the cage
              surface Surface Deform binds to (armature + subsurf + triangulate, evaluated),
              barycentric across that triangle — the starting point, and what (b) is held to;
           b. fitted (`_fitted_weights`): the convex bone mix that best reproduces Surface
              Deform's own answer over `_training_poses()`;
           c. layered (`_layered_weights`): a garment shell lying on another takes the weights
              of the layer under it, then a slightly wider blend of its own.
         Every pass works on coincident GROUPS, never on single vertices (step 2).
      2. Cracks. Vertices within COINCIDENT x height of each other are welded into one group FIRST
         and the group is sampled ONCE, so two coincident vertices get bit-identical weights and
         bit-identical rest positions, and so the same position in every pose. This is the
         property Surface Deform gave by construction and the reason this architecture exists;
         a per-vertex nearest-point lookup only gives it by luck. It is then measured anyway.
      3. Limits. At most `max_influences` bones per vertex (4: every real-time skinning path takes
         that) and no share under WEIGHT_FLOOR, renormalised to sum to one — by fading the
         smallest bones out continuously, never by cutting them (`_limit` says what the cut did).
      4. Rest shape. The mesh is rewritten in the RIG's rest pose, as an Armature-skinned mesh
         must be (UsdSkel's bind transforms are the bones' rest): each vertex is taken back from
         where it stands at the bind through the inverse of its own skinning matrix — the same
         inversion `region_fit` uses on the cage. So at the bind pose the skinned mesh lands on
         the imported geometry exactly, the identity `check_wrap` requires of Surface Deform.

    Then it is checked, and RAISES rather than return a skin that does not follow:
      * at the bind pose the skin stands where it was imported (not just "it evaluates");
      * through the shipped arm swing (`arms_down` with the collarbone following) it lands within
        `tolerance` x height of where Surface Deform put the same vertex, and it moved — a skin
        weighted to nothing would sit perfectly still and match nothing;
      * the crack width on coincident-and-unconnected vertex pairs is zero;
      * NEARLY coincident unconnected pairs (within NEAR x height) open no more than Surface Deform
        opens the same pair, plus NEAR_SLACK x height — the crack the exact-pair check is blind to.
    The numbers are kept on the skin as `skin["skin_bake"]` (JSON).

    What changes. Linear blend skinning is not Surface Deform, and the layering departs from it
    on purpose (Surface Deform slides garments over the body; the bake keeps them on it). Measured
    on both test skins through the lateral raise (task 10 report), baked against Surface Deform:
    mean 0.33 / p99 5.5 / worst 10.7 mm (boy), 0.30 / 5.2 / 9.3 mm (man) — task 8's nearest-point
    weights were 0.52 / 9.1 / 13.7 and 0.47 / 8.5 / 12.5. Further from the training poses it strays
    further — 43-45 mm at the hand in the overhead press's front rack, as before — and should be
    looked at per exercise. The bake now takes ~20 s instead of ~2 s, nearly all of it the fit
    (CPU on the boy: training poses ~5 s, solve ~14 s, layering ~2 s).

    Order. Call it after `wrap` (the skin must be bound to THIS cage) and before any animation is
    keyed or the rig is moved; the rig's pose and action are restored on the way out either way.
    """
    sd = [m for m in skin.modifiers if m.type == "SURFACE_DEFORM"]
    if len(sd) != 1 or sd[0].target is not cage or not sd[0].is_bound:
        raise SystemExit(f"bake_to_skin needs {skin.name!r} bound to {cage.name!r} by exactly one "
                         f"Surface Deform; it has {[(m.type, m.is_bound) for m in sd]}")
    others = [m.type for m in skin.modifiers if m.type != "SURFACE_DEFORM"]
    if others:
        # the USD writer turns a mesh with any modifier but Subdivision into per-frame points
        raise SystemExit(f"{skin.name!r} carries {others} beside Surface Deform; the baked skin "
                         f"must end with an Armature modifier and nothing else")
    if not any(m.type == "ARMATURE" and m.object is rig for m in cage.modifiers):
        raise SystemExit(f"{cage.name!r} is not driven by {rig.name!r}; there are no weights to read")
    height = json.loads(rig["skin_fit"])["measured"]["height"]
    t0 = time.time()
    anim = rig.animation_data
    action = anim.action if anim else None
    held = {pb.name: (pb.rotation_mode, pb.rotation_quaternion.copy(), pb.rotation_euler.copy(),
                      pb.location.copy(), pb.scale.copy()) for pb in rig.pose.bones}
    if anim:
        anim.action = None
    try:
        # Surface Deform's answers, before it goes: the bind and the shipped swing
        to_skin_pose(rig)
        imported = _co(skin.data)
        world = np.array(skin.matrix_world)
        imported_world = imported @ world[:3, :3].T + world[:3, 3]
        drift = float(np.linalg.norm(evaluated(skin)[0] - imported_world, axis=1).max())
        if drift > 1e-4 * height:
            raise SystemExit(f"at the bind pose the Surface Deform skin is {drift * 1000:.2f} mm off "
                             f"its own geometry; the rig is not at the pose it was bound in")
        arms_down(rig, clavicle_follow=CLAVICLE_FOLLOW)
        reference = evaluated(skin)[0]
        to_skin_pose(rig)

        # 1-3. weights, one set per coincident group: sampled, fitted, layered, limited
        co, tris, cage_w, bones = _cage_surface(cage)
        group = _coincident_groups(imported_world, COINCIDENT * height)
        heads, back = np.unique(group, return_inverse=True)
        back = back.ravel()
        sampled, gap = _nearest_weights(imported_world[heads], co, tris, cage_w)
        allowed = _allowed_bones(imported_world[heads], co, tris, cage_w, FIT_RINGS) | (sampled > 0)
        fitted, fit_stats = _fitted_weights(skin, rig, heads, sampled, allowed, height)
        layered, layer_stats = _layered_weights(imported_world[heads], fitted, co, tris, skin, back, height)
        # the joints after the layers, on the weights the garments will keep (run before them, the
        # layers put back the vest's 0.45 collarbone step on the upper back: 200 mm² turned over)
        layered, joint_stats, neck = _joint_blends(rig, imported_world[heads], layered, bones, skin, back, height)
        hand_labels = None
        if _hand_models(rig) is not None:
            # which skin vertex is which finger (the same measure the bones were fitted from)
            import hand_rig
            joints = json.loads(rig["skin_fit"])["joints"]
            mesh_edges = _edges(skin.data)
            hand_labels = {}
            for side in ("left", "right"):
                _, lab = hand_rig.measure(imported_world, mesh_edges, group,
                                          joints[f"wrist.{side}"], joints[f"tip.{side}"])
                hand_labels[side] = lab[heads]
        layered, rigid_stats = _rigid_parts(rig, imported_world[heads], layered, bones, height,
                                            hand_labels, neck=neck)
        weights = _limit(layered, WEIGHT_FLOOR, max_influences)[back]
        t_weights = time.time() - t0

        # 4. rest shape: back through each vertex's own skinning matrix
        mats = _lbs_matrices(skin, rig, weights, bones)
        det = np.linalg.det(mats[:, :3, :3])
        if det.min() < 0.2:
            raise SystemExit(f"a skin vertex's blended bone matrix is near-singular at the bind "
                             f"(det {det.min():.3f}); its rest position cannot be recovered")
        homogeneous = np.concatenate([imported, np.ones((len(imported), 1))], axis=1)
        rest = np.einsum("vij,vj->vi", np.linalg.inv(mats), homogeneous)[:, :3]
        # every member of a coincident group takes its group's one rest position, so members
        # cannot come apart in any pose
        rest = rest[heads][back.ravel()]

        # the swap
        skin.modifiers.remove(sd[0])
        skin.vertex_groups.clear()
        groups = [skin.vertex_groups.new(name=b) for b in bones]
        for b, vg in enumerate(groups):
            for v in np.flatnonzero(weights[:, b] > 0.0):
                vg.add([int(v)], float(weights[v, b]), "REPLACE")
        skin.data.vertices.foreach_set("co", rest.ravel())
        skin.data.update()
        # `exercise_stage.rest_height` sizes every prop and the camera off the UNDEFORMED mesh,
        # which until now was the skin as imported and is now the skin in the rig's rest pose —
        # spine straight, head level — 1.4% taller on the boy (1.0135 against 0.9995) and 0.7% on
        # the man. The character has not grown; only the bake's bookkeeping changed. So its cache
        # is filled with the height it measured before the rewrite, and nothing downstream moves.
        skin["rest_height"] = float(np.ptp(imported_world[:, 2]))
        mod = skin.modifiers.new("armature", "ARMATURE")
        mod.object = rig
        mod.use_vertex_groups = True
        mod.use_bone_envelopes = False
        # Linear blending, as the cage itself deforms and as UsdSkel and SceneKit play it.
        # Dual quaternions here would render one thing in Blender and ship another.
        mod.use_deform_preserve_volume = False
        if skin.parent is not rig:
            keep = skin.matrix_world.copy()
            skin.parent = rig
            skin.matrix_parent_inverse = rig.matrix_world.inverted()
            skin.matrix_world = keep
        bpy.context.view_layer.update()

        # the checks
        to_skin_pose(rig)
        bind = evaluated(skin)[0]
        landed = float(np.linalg.norm(bind - imported_world, axis=1).max())
        if landed > 1e-4 * height:
            raise SystemExit(f"the baked skin stands {landed * 1000:.3f} mm off the imported "
                             f"geometry at the bind pose; the rest shape is wrong")
        arms_down(rig, clavicle_follow=CLAVICLE_FOLLOW)
        posed = evaluated(skin)[0]
        off = np.linalg.norm(posed - reference, axis=1)
        moved = float(np.linalg.norm(posed - bind, axis=1).max())
        crack = deformation(skin, bind, posed, height)
        if moved < 0.1 * height:
            raise SystemExit(f"through the arm swing the baked skin moved at most "
                             f"{moved * 1000:.1f} mm; it is not following the rig")
        if off.max() > tolerance * height:
            raise SystemExit(f"through the arm swing the baked skin strays {off.max() * 1000:.1f} mm "
                             f"from Surface Deform (allowed {tolerance * height * 1000:.0f} mm), at "
                             f"{np.round(reference[int(off.argmax())], 3).tolist()}")
        if crack["crack_max_mm"] > 1e-4 * height * 1000:
            raise SystemExit(f"the bake opened a crack: coincident vertices "
                             f"{crack['crack_max_mm']:.4f} mm apart after the swing")
        # ...and the pairs that check cannot see: separate vertices NEARLY on top of each other,
        # not joined by an edge. They are sampled separately, so nothing above forces them to
        # agree, and the first version of `_limit` pulled such pairs 8.5 mm apart while the
        # exact-pair crack read 0.0000. Judged against Surface Deform on the same pair, because
        # some of these pairs are on two body parts that SHOULD part (Surface Deform opens a
        # few of them too).
        near = _near_pairs(imported_world, skin.data, NEAR * height, COINCIDENT * height)
        if len(near):
            start = np.linalg.norm(imported_world[near[:, 0]] - imported_world[near[:, 1]], axis=1)
            grew = np.linalg.norm(posed[near[:, 0]] - posed[near[:, 1]], axis=1) - start
            grew_sd = np.linalg.norm(reference[near[:, 0]] - reference[near[:, 1]], axis=1) - start
            excess = grew - grew_sd
        else:
            grew = grew_sd = excess = np.zeros(1)
        if excess.max() > NEAR_SLACK * height:
            k = int(excess.argmax())
            raise SystemExit(f"the bake opened a crack between two nearly coincident vertices "
                             f"({near[k].tolist()}, at {np.round(imported_world[near[k, 0]], 3).tolist()}): "
                             f"{grew[k] * 1000:.2f} mm wider after the swing where Surface Deform "
                             f"widened it {grew_sd[k] * 1000:.2f} mm")
        influences = (weights > 0).sum(axis=1)
        stats = {
            "vertices": int(len(weights)), "weight_samples": int(len(heads)),
            "coincident_groups": int((np.bincount(back.ravel()) > 1).sum()),
            "influences": {str(k): int(n) for k, n in enumerate(np.bincount(influences)) if n},
            "skin_to_cage_mm": {"mean": float(gap.mean() * 1000), "max": float(gap.max() * 1000)},
            "bind_landing_mm": landed * 1000,
            "vs_surface_deform_mm": {"pose": "arms_down, collarbone following",
                                     "mean": float(off.mean() * 1000),
                                     "p99": float(np.percentile(off, 99) * 1000),
                                     "max": float(off.max() * 1000),
                                     "worst_at": np.round(reference[int(off.argmax())], 4).tolist()},
            "skin_max_move_mm": moved * 1000,
            "crack_max_mm": crack["crack_max_mm"],
            "crack_pairs": crack["coincident_unconnected_pairs"],
            "crack_pairs_that_moved_over_1pct": crack["crack_pairs_that_moved_over_1pct"],
            "near_pairs": int(len(near)),
            "near_gap_growth_mm": {"baked": float(grew.max() * 1000),
                                   "surface_deform": float(grew_sd.max() * 1000),
                                   "worst_excess": float(excess.max() * 1000)},
            "fit": fit_stats,
            "layers": layer_stats,
            "joints": joint_stats,
            "rigid_parts": rigid_stats,
            "seconds_weights": t_weights,
            "seconds": time.time() - t0,
        }
        skin["skin_bake"] = json.dumps(stats)
    finally:
        for pb in rig.pose.bones:
            mode, quat, euler, loc, scale = held[pb.name]
            pb.rotation_mode = mode
            pb.rotation_quaternion, pb.rotation_euler = quat, euler
            pb.location, pb.scale = loc, scale
        if anim:
            anim.action = action
        bpy.context.view_layer.update()
    print(f"bake_to_skin: {skin.name} -> {stats['vertices']} verts, "
          f"{len(bones)} bone groups, influences {stats['influences']}; vs Surface Deform "
          f"mean {stats['vs_surface_deform_mm']['mean']:.2f} / max "
          f"{stats['vs_surface_deform_mm']['max']:.1f} mm; crack {stats['crack_max_mm']:.4f} mm "
          f"over {stats['crack_pairs']} pairs ({stats['seconds']:.1f}s)")
    return skin


# --- verification --------------------------------------------------------------------------------

# `animate.CLAVICLE_FOLLOW`: the share of an arm's swing the house rig gives the collarbone in
# every animated pose. Repeated rather than imported because importing `animate` drags in the
# whole exercise stage.
CLAVICLE_FOLLOW = 0.34


def arms_down(rig, degrees=90.0, clavicle_follow=0.0):
    """Swing both upper arms `degrees` downward from wherever the skin pose holds them.

    The motion that tore the old pipeline: arms out to the side, then down by the side. The axis
    is horizontal and perpendicular to the arm, so the arm sweeps through the frontal plane.

    `clavicle_follow` = 0 is the brief's test: the upper arm alone turns. CLAVICLE_FOLLOW is what
    the shipped animation actually does (`animate.with_clavicles`): the collarbone turns that
    share of the way toward the arm, so the shoulder drops a little as the arm comes down.
    """
    to_skin_pose(rig)
    for side in ("left", "right"):
        pbone = rig.pose.bones[_bn(rig, f"upper_arm.{side}")]
        now = (pbone.tail - pbone.head).normalized()
        axis = now.cross(Vector((0.0, 0.0, -1.0))).normalized()
        final = Quaternion(axis, math.radians(degrees)) @ now
        if clavicle_follow:
            clavicle = rig.pose.bones[_bn(rig, f"clavicle.{side}")]
            along = (clavicle.tail - clavicle.head).normalized()
            _turn(rig, clavicle.name,
                  Quaternion().slerp(along.rotation_difference(final), clavicle_follow))
            miss = _aim(rig, pbone.name, final)
        else:
            _turn(rig, pbone.name, Quaternion(axis, math.radians(degrees)))
            miss = math.degrees((pbone.tail - pbone.head).normalized().angle(final))
        if miss > 0.5:
            raise SystemExit(f"{pbone.name} ended {miss:.1f} degrees off a {degrees}-degree swing")


def deformation(skin, rest, posed, height, labels=None):
    """Edge growth and crack width on the skin, between two evaluated states of it.

    `labels` (one body region per skin vertex) breaks the grown edges down by where they are.
    """
    edges = _edges(skin.data)
    rest_len = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=1)
    posed_len = np.linalg.norm(posed[edges[:, 0]] - posed[edges[:, 1]], axis=1)
    growth = posed_len - rest_len
    worst = int(np.argmax(growth))

    # Coincident, unconnected vertices: the pairs that cracked open in the old pipeline. Grouped
    # by exact rest position (to 1e-5 of height), then every pair in a group not joined by an edge.
    keys = np.round(rest / (1e-5 * height)).astype(np.int64)
    _, group, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    group = group.ravel()
    connected = {(min(a, b), max(a, b)) for a, b in edges.tolist()}
    members = {}
    for v in np.flatnonzero(counts[group] > 1):
        members.setdefault(int(group[v]), []).append(int(v))
    pairs = [(a, b) for vs in members.values() for k, a in enumerate(vs) for b in vs[k + 1:]
             if (a, b) not in connected]
    moved = np.linalg.norm(posed - rest, axis=1)
    if pairs:
        p = np.array(pairs)
        crack = np.linalg.norm(posed[p[:, 0]] - posed[p[:, 1]], axis=1)
        crack_max = float(crack.max())
        # A zero crack only means something on pairs that MOVED: pairs on a still torso would
        # read zero under any binding at all.
        travelling = int((moved[p[:, 0]] > 0.01 * height).sum())
    else:
        crack_max, travelling = 0.0, 0
    return {
        "edges": int(len(edges)),
        "max_growth_mm": float(growth.max() * 1000),
        "max_growth_pct": float(growth.max() / height * 100),
        "edges_over_1pct": int((growth > 0.01 * height).sum()),
        "edges_over_0_5pct": int((growth > 0.005 * height).sum()),
        "worst_edge_at": [round(float(c), 4) for c in (rest[edges[worst, 0]] + rest[edges[worst, 1]]) / 2],
        "worst_edge_posed_at": [round(float(c), 4)
                                for c in (posed[edges[worst, 0]] + posed[edges[worst, 1]]) / 2],
        "worst_edge_rest_mm": float(rest_len[worst] * 1000),
        "max_shrink_mm": float(-growth.min() * 1000),
        "coincident_groups": int(len(members)),
        "coincident_unconnected_pairs": int(len(pairs)),
        "crack_max_mm": crack_max * 1000,
        "crack_pairs_that_moved_over_1pct": travelling,
        "skin_max_move_mm": float(moved.max() * 1000),
        "skin_vertices_moved_over_1mm": int((moved > 0.001).sum()),
        "over_0_5pct_by_region": ({} if labels is None else
                                  {str(k): int(v) for k, v in zip(*np.unique(
                                      labels[edges[growth > 0.005 * height, 0]].astype(str),
                                      return_counts=True))}),
        "worst_region": None if labels is None else str(labels[edges[worst, 0]]),
    }


def skin_state(skin):
    co, _ = evaluated(skin)
    return co


def check_wrap(skin, rig, mod, height, label):
    """Every number a wrapped skin is judged on, and the checks that make the numbers mean something.

    In this order, because each check protects the next:
      1. the modifier is bound (`wrap` already raised if not; asserted again here);
      2. at the bind pose the skin is where it was imported — the bind did not distort it;
      3. after the swing, something actually moved, by about an arm's length at the hand;
      4. only then edge growth and crack width.
    """
    if not mod.is_bound:
        raise SystemExit(f"{label}: not bound")
    original = _co(skin.data) @ np.array(skin.matrix_world)[:3, :3].T + np.array(skin.matrix_world)[:3, 3]
    to_skin_pose(rig)
    rest = skin_state(skin)
    drift = float(np.linalg.norm(rest - original, axis=1).max())
    if drift > 1e-4 * height:
        raise SystemExit(f"{label}: at the bind pose the skin is already {drift * 1000:.2f}mm "
                         f"off its own geometry — the bind is not an identity")
    # each skin vertex's body region: that of the nearest cage vertex, as bound
    cage_now, _ = posed_cage(mod.target)
    region = cage_regions(mod.target, rig, cage_now)[0]
    kd = KDTree(len(cage_now))
    for i, v in enumerate(cage_now):
        kd.insert(Vector(v), i)
    kd.balance()
    labels = np.array([region[kd.find(Vector(v))[1]] for v in rest], dtype=object)
    arms_down(rig)
    posed = skin_state(skin)
    result = deformation(skin, rest, posed, height, labels=labels)
    fit = json.loads(rig["skin_fit"])
    reach = fit["measured"]["s_tip"] - fit["measured"]["s_shoulder"]
    if result["skin_max_move_mm"] < 0.5 * reach * 1000:
        raise SystemExit(f"{label}: after a 90-degree swing the skin moved at most "
                         f"{result['skin_max_move_mm']:.1f}mm; the arm is {reach * 1000:.0f}mm "
                         f"long. The wrap is not following the cage.")
    result["bound"] = bool(mod.is_bound)
    result["bind_pose_drift_mm"] = drift * 1000
    result["arm_length_mm"] = reach * 1000
    # the same swing as the shipped animation poses it, collarbone following
    arms_down(rig, clavicle_follow=CLAVICLE_FOLLOW)
    follow = deformation(skin, rest, skin_state(skin), height, labels=labels)
    result["with_clavicle_follow"] = {k: follow[k] for k in (
        "max_growth_mm", "max_growth_pct", "edges_over_1pct", "edges_over_0_5pct",
        "worst_edge_at", "worst_region", "crack_max_mm", "skin_max_move_mm")}
    arms_down(rig)
    return result


# --- looking at it -------------------------------------------------------------------------------

def _stage(origin, height):
    """`character.stage`'s backdrop, floor and three lamps, stood at the skin's own floor."""
    before = {o.name for o in bpy.data.objects}
    character.stage(character.Proportions(height=height))
    for obj in bpy.data.objects:
        if obj.name not in before:
            obj.location = obj.location + Vector(origin)


def _wire(cage, rig):
    """The control cage as dark wire over the skin, following the rig."""
    mesh = cage.data.copy()
    wire = bpy.data.objects.new("cage_wire", mesh)
    bpy.context.collection.objects.link(wire)
    wire.parent = rig
    wire.matrix_parent_inverse = Matrix.Identity(4)
    wire.matrix_basis = Matrix.Identity(4)
    arm = wire.modifiers.new("armature", "ARMATURE")
    arm.object = rig
    mod = wire.modifiers.new("wire", "WIREFRAME")
    mod.thickness = 0.0018
    mod.use_replace = True
    mat = make_character.material("cage_wire", (0.012, 0.014, 0.020, 1.0), roughness=0.5)
    mesh.materials.clear()
    mesh.materials.append(mat)
    return wire


def render(path, show, eye, target, extent, samples=32, resolution=(620, 900)):
    scene = bpy.context.scene
    for obj in bpy.data.objects:
        if obj.type == "MESH" and obj.name not in ("backdrop", "floor"):
            obj.hide_render = obj.name not in show
    camera = character.frame_camera("cam", eye, target, extent)
    scene.camera = camera
    scene.render.film_transparent = False
    scene.render.resolution_x, scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.filepath = str(path)
    scene.render.image_settings.file_format = "PNG"
    make_character.use_cycles(scene, samples=samples)
    bpy.ops.render.render(write_still=True)
    bpy.data.objects.remove(camera, do_unlink=True)
    return str(path)


def views(fit):
    """Camera (eye, target, extent) for: whole body front, three-quarter, and both shoulders."""
    cx, cy, low = fit["origin"]
    m = fit["measured"]
    h = m["height"]
    mid = Vector((cx, cy, low + h * 0.5))
    sh_z = m["shoulder_z"]
    right = Vector(fit["shoulder_joint"]["right"])
    left = Vector(fit["shoulder_joint"]["left"])
    return {
        "front": (mid + Vector((0, 3, 0)), mid, h * 1.12),
        "three_quarter": (mid + Vector((2.2, 2.2, 0.5)), mid, h * 1.12),
        "shoulder_right_front": (right + Vector((0.9, 1.6, 0.25)), right, h * 0.36),
        "shoulder_left_back": (left + Vector((-0.9, -1.6, 0.35)), left, h * 0.36),
    }


# --- main ----------------------------------------------------------------------------------------

def _copy_skin(skin, name):
    copy = skin.copy()
    copy.data = skin.data.copy()
    copy.name = name
    for mod in list(copy.modifiers):
        copy.modifiers.remove(mod)
    bpy.context.collection.objects.link(copy)
    return copy


def naive_control(cage, rig, skin, height):
    """What a plain whole-mesh shrinkwrap does to this cage: the failure region_fit exists for.

    Kept as a live control rather than a story: if the damage counter cannot see THIS collapse,
    its zero on the region fit means nothing.
    """
    to_skin_pose(rig)
    copy = cage.copy()
    copy.data = cage.data.copy()
    copy.name = "naive_cage"
    bpy.context.collection.objects.link(copy)
    shrink = copy.modifiers.new("shrink", "SHRINKWRAP")
    shrink.target = skin
    shrink.wrap_method = "NEAREST_SURFACEPOINT"
    arm_index = next(i for i, m in enumerate(copy.modifiers) if m.type == "ARMATURE")
    copy.modifiers.move(len(copy.modifiers) - 1, arm_index + 1)
    reference, _ = posed_cage(cage)
    shrunk, _ = posed_cage(copy)
    labels = cage_regions(cage, rig, reference)[0]
    result = damage(shrunk, _quads(copy.data), height, reference=reference, labels=labels)
    probe = _copy_skin(skin, "naive_skin")
    try:
        mod = wrap(copy, rig, probe, name="naive_wrap")
        result["surface_deform_bound"] = True
    except SystemExit as err:
        result["surface_deform_bound"] = False
        result["bind_error"] = str(err)
    else:
        # it bound — so what does a collapsed-finger cage do to the skin it carries?
        follow = check_wrap(probe, rig, mod, height, "naive")
        result["skin"] = {k: follow[k] for k in ("max_growth_mm", "max_growth_pct",
                                                  "edges_over_1pct", "worst_region",
                                                  "over_0_5pct_by_region", "crack_max_mm")}
    bpy.data.objects.remove(probe, do_unlink=True)
    bpy.data.objects.remove(copy, do_unlink=True)
    return result


def run(model, out, levels=2, samples=32, render_it=True, rest="skin"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cage, rig, props = fit_cage_to(model, levels=levels, scratch=out, rest=rest)
    fit = json.loads(rig["skin_fit"])
    height = props.height
    skin = load_skin(model)
    report = {"model": str(model), "fit": fit, "proportions": dataclasses.asdict(props),
              "cage_vertices": len(cage.data.vertices), "cage_faces": len(cage.data.polygons),
              "skin_vertices": len(skin.data.vertices), "skin_faces": len(skin.data.polygons)}
    print(f"[{time.time() - t0:6.1f}s] cage built: {report['cage_vertices']} verts; "
          f"skin {report['skin_vertices']} verts")

    report["naive_shrinkwrap"] = naive_control(cage, rig, skin, height)
    print(f"[{time.time() - t0:6.1f}s] naive shrinkwrap: {report['naive_shrinkwrap']}")

    # BEFORE: the same cage, unfitted, wrapped onto a copy of the skin
    unfitted = _copy_skin(skin, "skin_unfitted")
    mod = wrap(cage, rig, unfitted, name="wrap_unfitted")
    report["unfitted"] = check_wrap(unfitted, rig, mod, height, "unfitted")
    print(f"[{time.time() - t0:6.1f}s] unfitted wrap: {report['unfitted']}")
    cams = views(fit)
    panels = []
    if render_it:
        _stage(fit["origin"], height)
        arms_down(rig)
        panels.append(("before_shoulder", render(out / "before-shoulder-right.png", {"skin_unfitted"},
                                                 *cams["shoulder_right_front"], samples=samples)))
        panels.append(("before_front", render(out / "before-front.png", {"skin_unfitted"},
                                              *cams["front"], samples=samples)))
    bpy.data.objects.remove(unfitted, do_unlink=True)

    region_fit(cage, skin)
    report["region_fit"] = json.loads(cage["region_fit"])
    print(f"[{time.time() - t0:6.1f}s] region fit: {report['region_fit']}")

    mod = wrap(cage, rig, skin)
    report["fitted"] = check_wrap(skin, rig, mod, height, "fitted")
    print(f"[{time.time() - t0:6.1f}s] fitted wrap: {report['fitted']}")
    # the fitted cage, posed, checked again for the pose the skin is actually shown in — and its
    # own edge growth through the same swing, so the skin's numbers can be read against what the
    # cage itself was told to do
    to_skin_pose(rig)
    posed_bind, _ = posed_cage(cage)
    labels = cage_regions(cage, rig, posed_bind)[0]
    arms_down(rig)
    posed_down, _ = posed_cage(cage)
    report["cage_damage_arms_down"] = damage(posed_down, _quads(cage.data), height,
                                             reference=posed_bind, labels=labels)
    cage_edges = _edges(cage.data)
    growth = (np.linalg.norm(posed_down[cage_edges[:, 0]] - posed_down[cage_edges[:, 1]], axis=1)
              - np.linalg.norm(posed_bind[cage_edges[:, 0]] - posed_bind[cage_edges[:, 1]], axis=1))
    report["cage_growth_arms_down"] = {"max_mm": float(growth.max() * 1000),
                                       "max_pct": float(growth.max() / height * 100),
                                       "edges_over_1pct": int((growth > 0.01 * height).sum())}

    if render_it:
        to_skin_pose(rig)
        wire = _wire(cage, rig)
        panels.append(("rest_with_cage", render(out / "rest-with-cage.png", {"skin", "cage_wire"},
                                                *cams["front"], samples=samples)))
        arms_down(rig)
        panels.append(("after_front", render(out / "after-front.png", {"skin"}, *cams["front"],
                                             samples=samples)))
        panels.append(("after_three_quarter", render(out / "after-three-quarter.png", {"skin"},
                                                     *cams["three_quarter"], samples=samples)))
        panels.append(("after_shoulder", render(out / "after-shoulder-right.png", {"skin"},
                                                *cams["shoulder_right_front"], samples=samples)))
        panels.append(("after_shoulder_back", render(out / "after-shoulder-left-back.png", {"skin"},
                                                     *cams["shoulder_left_back"], samples=samples)))
        panels.append(("after_with_cage", render(out / "after-with-cage.png", {"skin", "cage_wire"},
                                                 *cams["shoulder_right_front"], samples=samples)))
        # where the single worst-grown skin edge ended up, marked
        spot = Vector(report["fitted"]["worst_edge_posed_at"])
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.008 * height, location=spot)
        marker = bpy.context.object
        marker.name = "worst_edge"
        marker.data.materials.append(make_character.material("marker", (0.9, 0.05, 0.05, 1.0)))
        side = 1.0 if spot.x >= fit["origin"][0] else -1.0
        eye = spot + Vector((0.45 * side, 1.0, 0.8)).normalized() * 1.5    # front, above, outside
        panels.append(("worst_edge", render(out / "worst-edge.png", {"skin", "worst_edge"},
                                            eye, spot, height * 0.3, samples=samples)))
        bpy.data.objects.remove(marker, do_unlink=True)
        bpy.data.objects.remove(wire, do_unlink=True)
        order = ["rest_with_cage", "after_front", "after_three_quarter",
                 "before_shoulder", "after_shoulder", "after_shoulder_back",
                 "before_front", "after_with_cage", "worst_edge"]
        paths = dict(panels)
        report["sheet"] = character.montage([paths[k] for k in order if k in paths], 3,
                                            str(out / "sheet.png"))
        report["panels"] = paths
    report["seconds"] = time.time() - t0
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"[{time.time() - t0:6.1f}s] wrote {out / 'report.json'}")
    return report


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="wrap a Tripo skin over the character cage")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True, help="directory for renders and report.json")
    parser.add_argument("--levels", type=int, default=2)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--rest", default="skin",
                        help="build the cage's arms at the skin's angle ('skin'), the A-pose ('A'), "
                             "or this many degrees below horizontal")
    parser.add_argument("--save", default=None, help="write the .blend here")
    args = parser.parse_args(argv)
    run(args.model, args.out, levels=args.levels, samples=args.samples,
        render_it=not args.no_render, rest=args.rest)
    if args.save:
        bpy.ops.wm.save_as_mainfile(filepath=args.save)


if __name__ == "__main__":
    main()
