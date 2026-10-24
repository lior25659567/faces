#!/usr/bin/env python
"""Rig a supplied character mesh by measuring it, and save a .blend ready to animate.

    blender --background --python pipeline/blender/fit_model.py -- \
        --model "3d cartoon boy model.glb" --save /tmp/rigged.blend --still /tmp/rigged.png

The mesh comes from image-to-3D of the drawn mascot, so it is the right character and it has no
skeleton at all. This finds one.

Nothing here is a number typed while looking at the model. The joints are found by slicing the
mesh horizontally and watching where it comes apart: two clusters below the hips are the legs, so
the lowest slice that splits is the crotch; three or more higher up are arms either side of a
torso, so the lowest slice that splits that way is where the arms come clear of the ribs. Same
approach the muscle map uses on the drawing, for the same reason — regenerate the model and every
landmark moves with it.

What it does NOT do is pretend to measure what the mesh cannot show. A smooth arm has no elbow in
its silhouette, so the elbow is placed proportionally between two landmarks that were measured,
and `LANDMARKS` says so out loud.
"""

import argparse
import math
import sys
from pathlib import Path

import bmesh
import bpy
from mathutils import Matrix, Vector

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import make_character  # noqa: E402

# A million triangles is what image-to-3D hands back, and far more than a figure this size needs.
# Posing and rendering it is slow for no visible gain, and weighting is O(vertices x bones).
TARGET_FACES = 90_000

# The model is measured from an orthographic SILHOUETTE, not from its vertices.
#
# Two vertex-based attempts failed the same way. Clustering raw positions in thin z-bands found
# gaps between neighbouring vertices and read them as the body splitting — the count flickered
# between two and six and every landmark landed at the ankles. Binning and gap-closing helped and
# still gave runs of nine and thirteen across a solid thigh, because a decimated mesh simply does
# not have enough vertices per band to fill one.
#
# A rendered silhouette has no such problem: a solid body is solid pixels. It is also exactly how
# `pipeline/muscle_regions.py` measures the drawn mascot, so the model and the drawing are now
# measured the same way, which is the point.
SILHOUETTE = (360, 900)
LANDMARKS = {
    "measured": ["floor", "crown", "crotch", "arms_clear", "hand_low", "neck"],
    "interpolated": ["shoulder", "chest", "pelvis", "hip", "elbow", "wrist", "knee", "ankle"],
}


def load(path):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    suffix = Path(path).suffix.lower()
    if suffix in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path))
    elif suffix == ".obj":
        bpy.ops.wm.obj_import(filepath=str(path))
    else:
        raise SystemExit(f"don't know how to import {suffix!r}")
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    if not meshes:
        raise SystemExit("no mesh in that file")
    if len(meshes) > 1:
        bpy.ops.object.select_all(action="DESELECT")
        for m in meshes:
            m.select_set(True)
        bpy.context.view_layer.objects.active = meshes[0]
        bpy.ops.object.join()
    # The joined result if there were several, otherwise the one mesh — and never whatever
    # Blender left active. A glTF usually imports an empty "RootNode" alongside the mesh, and on
    # one of these that empty was the active object, so `body.data` was None and the whole run
    # died on an AttributeError that said nothing about the real cause.
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    body = meshes[0]
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = body
    body.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    face_forward(body)
    return body


def face_forward(body):
    """Turn the model to face +Y, which is where `pose_model` believes forward is.

    Image-to-3D exported this one facing -Y, so the first render was the back of its head — and
    worse, every pose would have been applied mirrored front-to-back: a hip flexed forward would
    have bent the figure backwards.

    Decided on the feet, measured against the LEG, not against the model's origin.

    Two simpler tests failed on this character. The head was tried first, on the reasoning that a
    nose points forward: its swept hair reaches further behind the skull than the nose does in
    front, so the test said it already faced the camera while the render showed its back. Then
    the foot's raw Y extent: that came out 0.084 forward against 0.089 back — all but tied, and
    it picked wrong.

    The signal that is not ambiguous is where the foot sits relative to the ankle above it. A
    foot is long in front of the leg and blunt behind it. So the leg's own centre line is the
    reference, and the side with the longer reach is the front.
    """
    verts = [v.co for v in body.data.vertices]
    high = max(v.z for v in verts)
    low = min(v.z for v in verts)
    span = high - low
    feet = [v for v in verts if v.z < low + span * 0.05]
    shins = [v for v in verts if low + span * 0.13 < v.z < low + span * 0.22]
    if not feet or not shins:
        return body
    axis = sum(v.y for v in shins) / len(shins)
    forward = max(v.y for v in feet) - axis
    backward = axis - min(v.y for v in feet)
    print(f"facing check: foot reaches {forward:+.3f} ahead of the leg, {backward:+.3f} behind")
    if forward >= backward:
        return body
    # The mesh data is rotated directly rather than by setting an object rotation and calling
    # transform_apply. The operator route silently did nothing here — the verification line below
    # printed the same reach before and after — because the operator depends on selection and
    # context that do not always hold in a background run. Transforming the mesh needs neither.
    body.data.transform(Matrix.Rotation(math.pi, 4, "Z"))
    body.data.update()
    after = [v.co for v in body.data.vertices if v.co.z < low + span * 0.05]
    shins_after = [v.co for v in body.data.vertices
                   if low + span * 0.13 < v.co.z < low + span * 0.22]
    axis_after = sum(v.y for v in shins_after) / len(shins_after)
    reach = max(v.y for v in after) - axis_after
    print(f"turned the model to face +Y; foot now reaches {reach:+.3f} ahead")
    if reach < backward * 0.8:
        raise SystemExit("the turn did not take: the foot still points the other way. The mesh "
                         "was not transformed, so every pose would come out mirrored.")
    return body


def decimate(body, target=TARGET_FACES):
    if len(body.data.polygons) <= target:
        return body
    modifier = body.modifiers.new("decimate", "DECIMATE")
    modifier.ratio = target / len(body.data.polygons)
    bpy.context.view_layer.objects.active = body
    bpy.ops.object.modifier_apply(modifier="decimate")
    return body


def non_manifold(mesh):
    return sum(1 for edge in mesh.edges if not edge.is_manifold)


def clean(body):
    """Weld and repair the mesh. NOT called by default — see below.

    Written to make Blender's bone-heat solver work, which needs a connected surface. It got the
    non-manifold edges from 35,393 down to 5,190 and the solver still refused, so heat weighting
    was abandoned for the radius-scaled binding in `weight`.

    Keeping it switched off, because it is not free: welding took this mesh from 62,853 vertices
    to 46,626 and left a visible web of inverted faces under each arm, present in the REST pose
    before anything was posed. That is what gave the game away — a deformation artifact cannot
    appear in a figure that has not deformed. With heat weighting gone there is nothing left that
    needs the repair, so the mesh is used exactly as supplied.

    Image-to-3D output is not watertight. This one arrived with 35,393 non-manifold edges out of
    90,000 faces — mostly duplicated vertices along seams — and Blender's heat weighting needs to
    diffuse across a connected surface, so it gave up and bound nothing at all. It says so, in a
    warning easily lost among the import chatter:

        Bone Heat Weighting: failed to find solution for one or more bones

    One BMesh, held in a local and freed explicitly. The first version called a helper that
    returned a BMesh and iterated it inline — `bmesh_of(body).edges` — which keeps no reference,
    so CPython was free to collect it part-way through the loop. Blender segfaulted outright.
    """
    mesh = bmesh.new()
    mesh.from_mesh(body.data)
    before = non_manifold(mesh)
    bmesh.ops.remove_doubles(mesh, verts=mesh.verts, dist=1e-4)
    bmesh.ops.recalc_face_normals(mesh, faces=mesh.faces)
    loose = [vertex for vertex in mesh.verts if not vertex.link_edges]
    if loose:
        bmesh.ops.delete(mesh, geom=loose, context="VERTS")
    after = non_manifold(mesh)
    mesh.to_mesh(body.data)
    mesh.free()
    body.data.update()
    print(f"cleaned: non-manifold edges {before} -> {after}, "
          f"{len(body.data.vertices)} vertices remain")
    return body


def bounds(body):
    xs = [v.co.x for v in body.data.vertices]
    zs = [v.co.z for v in body.data.vertices]
    return min(xs), max(xs), min(zs), max(zs)


def silhouette(body, path):
    """Render the model flat-on as an alpha mask, and return how to map pixels back to the model.

    Orthographic on purpose: a perspective camera makes a near hand wider than a far one, and
    every measurement taken off the picture would inherit that. Orthographic keeps the mapping a
    single scale factor, which is what makes `to_world` below one line.
    """
    x_lo, x_hi, z_lo, z_hi = bounds(body)
    height = z_hi - z_lo
    width = x_hi - x_lo
    pad = height * 0.04
    span_z = height + pad * 2
    span_x_needed = width + pad * 2
    middle_z = (z_lo + z_hi) / 2

    for stray in [o for o in bpy.data.objects if o.type in {"CAMERA", "LIGHT"}]:
        bpy.data.objects.remove(stray, do_unlink=True)
    bpy.ops.object.camera_add(location=(0, max(width, height) * 3, middle_z),
                              rotation=(math.pi / 2, 0, math.pi))
    camera = bpy.context.object
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = span_z
    camera.data.sensor_fit = "VERTICAL"
    bpy.context.scene.camera = camera

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in [
        i.identifier for i in
        scene.bl_rna.properties["render"].fixed_type.bl_rna.properties["engine"].enum_items
    ] else "BLENDER_EEVEE"
    # The frame has to hold the model's WIDTH as well as its height. Framing on height alone
    # gave a horizontal span of 0.432 on a T-pose model that is 0.778 across, so both arms were
    # cropped off the picture every landmark is measured from — the pose came back looking like
    # hanging arms and the rig was built for the wrong shape. An arms-down model fits inside the
    # narrower span, which is why this stayed hidden until a T-pose arrived.
    tall = SILHOUETTE[1]
    wide = max(SILHOUETTE[0], int(round(tall * span_x_needed / span_z)))
    scene.render.resolution_x, scene.render.resolution_y = wide, tall
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)

    return {"width": wide, "height": tall, "span_z": span_z, "middle_z": middle_z,
            "span_x": span_z * wide / tall, "z_lo": z_lo, "z_hi": z_hi, "model_height": height}


def read_mask(path, meta, threshold=0.35):
    """The silhouette as rows of True/False, top row first."""
    image = bpy.data.images.load(str(path))
    wide, tall = image.size
    pixels = list(image.pixels)
    rows = []
    for y in range(tall):
        base = y * wide * 4
        rows.append([pixels[base + x * 4 + 3] > threshold for x in range(wide)])
    bpy.data.images.remove(image)
    # Blender's framebuffer starts at the BOTTOM row, so flip to make row 0 the crown.
    rows.reverse()
    return rows


def runs_in(row, min_width=3):
    """Solid stretches across one row of the silhouette, as (first, last) pixel columns."""
    out, start = [], None
    for x, on in enumerate(row):
        if on and start is None:
            start = x
        elif not on and start is not None:
            if x - start >= min_width:
                out.append((start, x - 1))
            start = None
    if start is not None and len(row) - start >= min_width:
        out.append((start, len(row) - 1))
    return out


# How far out a run's centre has to sit, as a share of the figure's half-width, before it is an
# arm rather than part of the body. Legs straddle the centre line; a hanging arm does not.
ARM_OUT = 0.42


def sort_runs(runs, centre, half):
    """Split one row's runs into (left arm, central, right arm).

    Counting runs alone is what broke the first silhouette pass: at the height where the arms
    peel away from the ribs, one side separates a row or two before the other, so there is a row
    with exactly two runs — an arm and everything else. Read as a count that is indistinguishable
    from two legs, and the crotch was found at the armpit. Position tells them apart.
    """
    left, middle, right = [], [], []
    for run in runs:
        where = (run[0] + run[1]) / 2 - centre
        if where < -ARM_OUT * half:
            left.append(run)
        elif where > ARM_OUT * half:
            right.append(run)
        else:
            middle.append(run)
    return left, middle, right


def measure(body, path):
    meta = silhouette(body, path)
    rows = read_mask(path, meta)
    tall, wide = len(rows), len(rows[0])
    profile = {y: runs_in(rows[y]) for y in range(tall)}
    filled = [y for y in range(tall) if profile[y]]
    if not filled:
        raise SystemExit("the silhouette came out empty")
    crown, floor = min(filled), max(filled)
    span = floor - crown

    def frac(y):
        return (floor - y) / span          # 0 at the feet, 1 at the crown

    def to_world_z(y):
        return meta["middle_z"] + (0.5 - y / tall) * meta["span_z"]

    def to_world_x(x):
        return (x / wide - 0.5) * meta["span_x"]

    widths = [(profile[y][0][0], profile[y][-1][1]) for y in filled]
    centre = (min(w[0] for w in widths) + max(w[1] for w in widths)) / 2
    half = (max(w[1] for w in widths) - min(w[0] for w in widths)) / 2
    sorted_runs = {y: sort_runs(profile[y], centre, half) for y in filled}

    # The crotch is the TOP of the two-legged stretch: scanning up from the floor the body is two
    # legs and then becomes one torso. Judged on the CENTRAL runs only, so an arm coming away
    # from the ribs cannot be mistaken for a second leg.
    legs = [y for y in filled
            if len(sorted_runs[y][1]) == 2 and 0.25 < frac(y) < 0.62]
    crotch = min(legs) if legs else None

    # Arms are a run out on each side at the same height.
    arms = [y for y in filled
            if sorted_runs[y][0] and sorted_runs[y][2] and 0.25 < frac(y) < 0.86]
    arms_clear = min(arms) if arms else None
    hand_low = max(arms) if arms else None

    upper = [y for y in filled if 0.74 < frac(y) < 0.94 and len(profile[y]) == 1]
    neck = (min(upper, key=lambda y: profile[y][0][1] - profile[y][0][0]) if upper else None)

    # The shoulder JOINT is not a feature of the outline. Looking for the widest row between the
    # armpit and the neck returned the armpit itself, because on a figure with its arms down the
    # silhouette is at its widest exactly where they leave the ribs — and a joint placed there
    # hinges the arm from under the armpit. It sits above that, a third of the way up to the
    # neck, and it is listed as interpolated rather than measured for exactly this reason.
    shoulder = None
    if arms_clear is not None and neck is not None:
        shoulder = int(round(arms_clear + (neck - arms_clear) * 0.34))

    return {"profile": profile, "runs": sorted_runs, "rows": rows, "meta": meta, "crown": crown, "floor": floor,
            "frac": frac, "z_of": to_world_z, "x_of": to_world_x,
            "height": to_world_z(crown) - to_world_z(floor),
            "low": to_world_z(floor), "high": to_world_z(crown),
            "crotch": crotch, "arms_clear": arms_clear, "hand_low": hand_low,
            "neck": neck, "shoulder": shoulder}


def joints_from(m):
    """The skeleton, in the model's own coordinates.

    Measured landmarks come straight off the silhouette. The rest are placed proportionally
    BETWEEN measured ones and `LANDMARKS` says which is which — an elbow leaves no mark on a
    smooth arm's outline, so claiming to have measured one would be a lie in a file whose whole
    point is that nothing is guessed silently.
    """
    z, frac, x_of, profile = m["z_of"], m["frac"], m["x_of"], m["profile"]
    low, high, height = m["low"], m["high"], m["height"]

    shoulder_z = z(m["shoulder"]) if m["shoulder"] is not None else low + height * 0.78
    neck_z = z(m["neck"]) if m["neck"] is not None else low + height * 0.84
    crotch_z = z(m["crotch"]) if m["crotch"] is not None else low + height * 0.47
    hand_z = z(m["hand_low"]) if m["hand_low"] is not None else low + height * 0.42

    runs = m["runs"]

    # Is this a T-pose? It matters because everything below assumed hanging arms, and on a
    # T-pose model that produced an elbow and a wrist stacked vertically BELOW the shoulder
    # while the actual arm ran horizontally out to the side. A T-pose is wide: this model spans
    # 0.78 of its own height across, where the arms-down one spans 0.41.
    width = max(abs(x_of(r[0][0])) + abs(x_of(r[-1][1])) for r in profile.values() if r)
    t_pose = width > height * 0.62

    # Where the arm begins and ends.
    #
    # A T-pose cannot be read the way hanging arms are. With the arms down there is air either
    # side of the ribs, so a row across the chest gives three runs — arm, torso, arm — and the
    # outer two ARE the arms. Held out horizontally there is no gap beside the torso at all: the
    # row is one unbroken band from fingertip to fingertip, 0.765 wide on this model. The
    # three-run test then matched only a couple of stray slivers at the top edge of each hand,
    # out at 0.37, and the shoulder was placed there — giving an arm 0.045 long that barely
    # moved when posed.
    #
    # So the arm band is found by WIDTH instead: the rows where the figure is far wider than its
    # own torso are the outstretched arms, the shoulder sits at the edge of the torso under them,
    # and the reach is simply how far the figure extends.
    at_arms = runs.get(m["arms_clear"])
    have_arms = bool(at_arms and at_arms[0] and at_arms[2])
    if t_pose:
        torso_rows = [y for y in profile if 0.48 < frac(y) < 0.62 and profile[y]]
        torso_half = (min(abs(x_of(profile[y][0][0])) for y in torso_rows)
                      if torso_rows else height * 0.11)
        span_of = {y: abs(x_of(profile[y][0][0])) + abs(x_of(profile[y][-1][1]))
                   for y in profile if profile[y]}
        band = [y for y, w in span_of.items() if w > torso_half * 2 * 1.7 and frac(y) > 0.5]
        shoulder_x = torso_half * 0.92
        reach = max(span_of.values()) / 2
        if band:
            shoulder_z = sum(z(y) for y in band) / len(band)
    elif have_arms:
        left_arm = sum(at_arms[0][0]) / 2
        right_arm = sum(at_arms[2][-1]) / 2
        shoulder_x = (abs(x_of(left_arm)) + abs(x_of(right_arm))) / 2 * 0.88
        reach = shoulder_x * 1.05
    else:
        wide = profile.get(m["arms_clear"]) or [(0, 0)]
        shoulder_x = (abs(x_of(wide[0][0])) + abs(x_of(wide[-1][1]))) / 2 * 0.62
        reach = shoulder_x * 1.05

    # Half a thigh in from the centre, measured on the two central runs at the crotch.
    legs = runs.get(m["crotch"]) if m["crotch"] is not None else None
    if legs and len(legs[1]) == 2:
        hip_x = abs(x_of(sum(legs[1][0]) / 2) + x_of(sum(legs[1][1]) / 2)) / 2
        hip_x = max(abs(x_of(sum(legs[1][1]) / 2)), 1e-3) * 0.85
    else:
        hip_x = height * 0.055

    pelvis_z = crotch_z + height * 0.085
    chest_z = shoulder_z - height * 0.015
    knee_z = crotch_z - (crotch_z - low) * 0.47
    ankle_z = low + height * 0.05

    # Where the elbow, wrist and hand sit along the arm. On a T-pose the arm runs out along X at
    # roughly shoulder height; hanging, it runs down Z. Same fractions along the limb either way,
    # measured from the shoulder toward the hand.
    if t_pose:
        arm_z = shoulder_z
        elbow_z = wrist_z = hand_z = arm_z
        elbow_x = shoulder_x + (reach - shoulder_x) * 0.46
        wrist_x = shoulder_x + (reach - shoulder_x) * 0.84
        hand_x = reach
    else:
        elbow_z = shoulder_z - (shoulder_z - hand_z) * 0.46
        wrist_z = shoulder_z - (shoulder_z - hand_z) * 0.86
        elbow_x = arm_x * 0.97
        wrist_x = hand_x = arm_x

    j = {"pelvis": (0.0, 0.0, pelvis_z),
         "chest": (0.0, 0.0, chest_z),
         "neck": (0.0, 0.0, neck_z),
         "head": (0.0, 0.0, high - height * 0.07)}
    for side, sign in (("left", -1.0), ("right", 1.0)):
        j[f"shoulder.{side}"] = (sign * shoulder_x, 0.0, shoulder_z)
        j[f"elbow.{side}"] = (sign * elbow_x, 0.0, elbow_z)
        j[f"wrist.{side}"] = (sign * wrist_x, 0.0, wrist_z)
        j[f"hand.{side}"] = (sign * hand_x, 0.0, hand_z)
        j[f"hip.{side}"] = (sign * hip_x, 0.0, pelvis_z - height * 0.015)
        j[f"knee.{side}"] = (sign * hip_x, 0.0, knee_z)
        j[f"ankle.{side}"] = (sign * hip_x, 0.0, ankle_z)
        j[f"foot.{side}"] = (sign * hip_x, height * 0.06, low + height * 0.015)
    return j


def build_rig(joints):
    bpy.ops.object.armature_add(location=(0, 0, 0))
    rig = bpy.context.object
    rig.name = "rig"
    bpy.ops.object.mode_set(mode="EDIT")
    bones = rig.data.edit_bones
    for bone in list(bones):
        bones.remove(bone)
    for name, (head, tail) in make_character.BONES.items():
        bone = bones.new(name)
        bone.head = Vector(joints[head])
        bone.tail = Vector(joints[tail])
        if (bone.tail - bone.head).length < 1e-5:
            raise SystemExit(f"bone {name!r} came out with no length")
    for child, parent in make_character.PARENTS.items():
        bones[child].parent = bones[parent]
        bones[child].use_connect = False
    bpy.ops.object.mode_set(mode="OBJECT")
    return rig


# How thick each bone is, as a share of body height. This is the fix for the ambiguity that
# caused the tearing, and it is why plain nearest-segment was wrong rather than merely unlucky.
#
# A torso is a volume and a bone is a line. Measured raw, a chest vertex on this model sat 0.101
# from the shoulder joint and 0.102 from the spine — a coin flip that handed the arm roughly half
# the chest, so raising it dragged the vest out into a sheet. Collarbones halved that and could
# not finish, because no arrangement of centre-line bones makes a wide chest unambiguous.
#
# Dividing each distance by what that bone is meant to cover settles it. The spine stands for the
# whole trunk and is fat; an arm bone is thin. The same chest vertex now scores 0.78 against the
# spine and 2.0 against the upper arm, which is not close.
RADIUS = {
    "spine": 0.130, "neck": 0.070, "head": 0.145,
    "clavicle.left": 0.060, "clavicle.right": 0.060,
    "upper_arm.left": 0.050, "upper_arm.right": 0.050,
    "forearm.left": 0.042, "forearm.right": 0.042,
    "hand.left": 0.046, "hand.right": 0.046,
    "thigh.left": 0.078, "thigh.right": 0.078,
    "shin.left": 0.058, "shin.right": 0.058,
    "foot.left": 0.052, "foot.right": 0.052,
}

# How far a neighbouring bone's influence reaches past a joint, as a share of body height — the
# same unit RADIUS is in, turned into a real distance in `weight()` by multiplying by `height`
# exactly as RADIUS is. The old value, 0.26, was in SCALED distance (distance divided by a
# bone's own radius) and became meaningless the moment the metric changed to a real distance:
# "0.26 scaled units" past a 0.05-radius upper arm is 2.5x the arm's own thickness, while past a
# 0.145-radius head it is 2.5x the head's — two wildly different real distances masquerading as
# the same number.
#
# Started from 0.025 as instructed and swept upward while watching `hard_seams`'s breakdown by
# bone pair on `3d cartoon boy.glb`. At 0.025 there were still 19+15 hard-seam edges between
# `head` and `upper_arm` — the exact pair the capsule-surface metric was supposed to fix — because
# a handful of vertices right at the top of the shoulder (|x| about 0.12-0.14, at the very edge
# of the `head` capsule's hemisphere cap at the neck joint) still won for `head` by only 1-2mm,
# too fine a margin for BLEND to paper over since `head` and `upper_arm` do not share a joint and
# so are never in each other's `near[]` — BLEND only widens sharing between bones that already
# share a joint, it cannot create a new one. Those specific edges turned out to resolve on their
# own as BLEND grew, because the vertices that were losing to `head` are close enough to `neck`
# (which DOES share `head`'s joint) that widening the shared zone pulls `neck` in and dilutes
# `head`'s own weight on the LOSING side's immediate neighbours, breaking the >=0.85/>=0.85 hard
# -seam condition on that edge even though `head` is still nominally the single closest capsule.
# Swept in 0.005 steps from 0.050 to 0.070: every hard seam naming `head` (both the `head`/
# `upper_arm` pairs and a smaller, nearer-the-neck `head`/`spine` cluster) is gone by BLEND=0.060
# and none reappear above it, while 0.055 still leaves 91 `head`/`spine` edges. 0.060 is the
# smallest value that clears all of them, so that is what shipped rather than a rounder-looking
# but unjustified 0.05 or 0.1.
#
# What BLEND could NOT fix, at any width, was a `spine`/`thigh` hard seam at the hip (350 edges,
# present under the OLD metric too — nothing before this task could see it). `pelvis` and
# `hip.left`/`hip.right` are different joint names, and BLEND only widens sharing between bones
# `near[]` already considers neighbours — it cannot invent a new entry in that graph. That one
# was not a tuning problem: `make_character.PARENTS` had no entry at all for `thigh.left`/
# `thigh.right`, so `neighbours_of()`'s family test could never see them as anything but
# strangers to `spine`. Fixed by adding those two lines to PARENTS (see make_character.py and
# TASK-1's fix-round report), which closes the hip gap through the existing family test rather
# than needing a change here.
BLEND = 0.060

# Change B's percentile and minimum-sample floor for turning RADIUS from a guess into a
# measurement — see the long comment inside `weight()` for why a percentile and not the max.
RADIUS_PERCENTILE = 95
RADIUS_MIN_SAMPLE = 20

# Change B's taper: how many `t`-bins each bone's winners are split into along its own length,
# and the minimum vertex count an END bin needs before its percentile is trusted directly rather
# than falling back to the bone's one plain (untapered) percentile — see the long comment on
# `measure_taper` inside `weight()` for why a fallback that tries to be cleverer than that (a
# line fit across all the bins) was tried and measurably made things worse.
TAPER_BINS = 5
TAPER_MIN_BIN = 20

# Change A's smoothing strength — see the long comment inside `weight()` for why this exists at
# all, and TASK-1's fix-round report for the factor/repeat sweep that justified this value.
#
# Re-tuned DOWN from 90 once UNIFY (below) started handling the coincident-vertex crack that 90
# was partly fighting blind. With UNIFY doing that job structurally, more smoothing stopped
# helping and started actively hurting: re-measured against true arms-down absolute edge growth
# (not the retired ratio), growth bottoms out at repeat=9 and climbs again on both sides of it —
# 24.4mm at repeat=0-1, down to 12.85mm at 9, back up to 31.76mm by 90. Heavy smoothing was
# diffusing weight so broadly that nearby, genuinely different anatomy (chest vs. deltoid)
# blurred into near-identical ambiguous mixtures, which is a different, subtler version of the
# same "arms drag the ribs" failure the torso_to_arm check watches for — just not quite sharp
# enough on THIS run to trip that specific box-based check. 9 is the shipped value.
#
# Re-swept again after UNIFY changed from averaging to picking and back (see UNIFY's own comment
# in `weight()`) — 9 was still the minimum both times, so the dose this number needs does not
# depend on which of those two UNIFY strategies is in use, only on UNIFY existing at all.
SMOOTH_FACTOR = 1.0
SMOOTH_REPEAT = 9


def neighbours_of():
    """Which bones may share vertices: those meeting at a joint, and parent with child.

    Both tests are needed. An upper arm ends at `shoulder.right` while the spine ends at `chest`
    — different joint names, so the shared-joint test alone calls them strangers and puts a hard
    seam across the shoulder. The rig's parent graph knows better, so it is asked too.
    """
    out = {}
    for name, (a, b) in make_character.BONES.items():
        shares = {other for other, (c, d) in make_character.BONES.items()
                  if other != name and {a, b} & {c, d}}
        family = {make_character.PARENTS.get(name)} | {
            child for child, parent in make_character.PARENTS.items() if parent == name}
        out[name] = {name} | shares | {f for f in family if f}
    return out


def weight(body, rig, joints):
    """Bind every vertex to the bone whose capsule SURFACE it sits closest to.

    Blender's bone-heat solver was tried and is the better tool in principle — it diffuses over
    the surface rather than measuring through the body. It cannot be used here: this mesh came
    from image-to-3D with 35,393 non-manifold edges, welding got that to 5,190, and the solver
    still refuses, leaving every vertex bound to nothing. It says so in a warning rather than an
    error, which is exactly how an earlier version of this reported a flawless result on a mesh
    bound to nothing at all.

    So the binding is done here, where it always produces an answer, and the metric below is
    what makes that answer correct rather than merely plausible.

    An earlier version ranked bones by `distance_to_segment / scale[bone]` — distance to the
    bone's CENTRE LINE, divided by how fat RADIUS says that bone is. That is a sphere-vs-stick
    comparison and it is answering the wrong question. On `3d cartoon boy.glb` a deltoid vertex
    at (-0.187, -0.071, 0.706), 19 cm out from the centre line, scored 1.525 against `head`
    (distance 0.2057, scale 0.1349) and 1.531 against `upper_arm` (distance 0.0712, scale
    0.0465) — `head` won by a coin flip. A vertex 11 mm away scored the opposite way and went to
    `upper_arm`. Those two vertices are not neighbours in `neighbours_of()` (the head and the
    upper arm never meet at a joint), so nothing blended them, and the hard seam between them
    tore 18x open the moment the arm came down from a T-pose. 144 of 404 torn vertices belonged
    to `head` for exactly this reason: `head`'s RADIUS of 0.145 is wide enough that scaled
    distance called it the nearer bone for skin that is plainly deltoid, not scalp.

    The question binding actually needs answered is "how far OUTSIDE this limb's volume does
    this skin sit" — signed distance to the capsule surface, `distance_to_segment - radius`, in
    real mesh units rather than a ratio. Negative means the point is inside that bone's own
    capsule. Under this metric the same deltoid vertex scores 0.0708 against `head` and 0.0247
    against `upper_arm` — `upper_arm` wins by about 3x, and the 11mm-away vertex agrees with it
    instead of splitting the other way. The ranking, winner-take-most and BLEND sharing with
    `near[winner]` are otherwise unchanged; only what "distance" means has changed.

    Fixing that metric was not the end of it. It still left a narrower version of the identical
    mistake — `head`'s RADIUS was a guess borrowed from a different, procedurally-built character
    in `make_character.py`, not a measurement of THIS mesh, and every other entry in the table has
    the same provenance. So this now runs the bind TWICE: once against that guessed table (PASS
    1, below), purely to find out which vertices currently think they belong to each bone, and
    once more (PASS 2) against a radius actually measured from pass 1's own winners. And it still
    was not the end of it even then: the single worst remaining tear turned out to be two mesh
    vertices 1.5mm apart with genuinely different winners, which no radius or BLEND value can
    reconcile because the per-vertex ranking has no way to know its answer disagrees with its
    neighbour's. CHANGE A, the smoothing pass at the end of this function, is what fixes THAT —
    see its own comment below.

    Not even that was the end of it. A deep squat showed `spine` claiming glute and upper-thigh
    vertices that then would not travel with a thigh bending through 120 degrees — and the cause
    was the same shape of bug as `head`'s, one level subtler: PASS 2's single measured radius for
    `spine` pools EVERY vertex it won, chest and waist alike, so the percentile comes out
    chest-width and is wrong — too fat — at the waist, by about 20%. Shrinking that one number
    broke the chest instead of fixing the waist, because a torso is not a cylinder; it tapers, and
    one radius cannot be right at both ends of a bone that is not the same thickness along its
    whole length. So PASS 2 now measures TWO radii per bone, one at the head joint and one at the
    tail, and interpolates between them — see the taper comment further down for how, and for
    which bones needed the fallback it describes.
    """
    lines = {name: (Vector(joints[a]), Vector(joints[b]))
             for name, (a, b) in make_character.BONES.items()}
    near = neighbours_of()

    zs = [v.co.z for v in body.data.vertices]
    height = max(zs) - min(zs)
    # BLEND is a share of body height, the same unit RADIUS is in, converted to a real distance
    # here rather than left as a bare fraction — see the comment on BLEND above for why a share
    # of height rather than the old scaled-distance unit, which stopped meaning anything once
    # the metric below changed from a ratio to a real distance.
    blend = BLEND * height

    def reset_groups():
        """Start a binding pass from a clean slate. Needed because this now runs TWICE — once
        against the seed RADIUS, once against the measured radii below — and `add(..., "REPLACE")`
        only overwrites the (vertex, group) pairs it actually touches. Without clearing the groups
        between passes, a vertex whose winner changed between passes would keep BOTH its old and
        its new weight, which is not a blend, it is a bug that happens to look like one."""
        for group in list(body.vertex_groups):
            body.vertex_groups.remove(group)
        return {name: body.vertex_groups.new(name=name) for name in make_character.BONES}

    def segment_point(name, point):
        """Distance from `point` to bone `name`'s CENTRE LINE, with no radius subtracted yet,
        and `t`: where the closest point falls along the segment, 0 at the head joint and 1 at
        the tail. `t` is what makes the radius TAPERED rather than a single number — see the
        comment above `radius_head`/`radius_tail` below."""
        a, b = lines[name]
        axis = b - a
        length = axis.length_squared
        t = 0.0 if length < 1e-12 else max(0.0, min(1.0, (point - a).dot(axis) / length))
        return (point - (a + axis * t)).length, t

    def bind(radius_head, radius_tail):
        """One full binding pass against a per-bone radius table — two numbers per bone now, one
        at each end (real units, same shape as `RADIUS` already multiplied by `height`), linearly
        interpolated by `t` along the way. Assigns vertex groups exactly as before — rank by
        signed distance to the capsule surface, winner takes most, share with `near[winner]`
        within `blend` — and additionally returns, for every vertex, which bone WON it outright,
        the raw (pre-radius) distance to that bone's centre line, and `t`, which is the
        measurement Change B's taper needs and the blended share is not: the share already has
        BLEND's smoothing baked in, and measuring a bone's true reach from an already-smoothed
        number would be measuring the ruler against itself.
        """
        groups = reset_groups()
        winners = {}
        for vertex in body.data.vertices:
            point = vertex.co
            raw = {}
            for name in lines:
                distance, t = segment_point(name, point)
                radius = radius_head[name] + (radius_tail[name] - radius_head[name]) * t
                raw[name] = (distance - radius, distance, t)
            ranked = sorted((raw[name][0], name) for name in lines)
            winner = ranked[0][1]
            closest = ranked[0][0]
            shares = [(d, n) for d, n in ranked if n in near[winner] and d <= closest + blend]
            total = sum(1.0 / max(d - closest + blend * 0.3, 1e-6) for d, _ in shares)
            for d, name in shares:
                share = (1.0 / max(d - closest + blend * 0.3, 1e-6)) / total
                if share > 0.01:
                    groups[name].add([vertex.index], share, "REPLACE")
            _, distance, t = raw[winner]
            winners[vertex.index] = (winner, distance, t)
        return winners

    modifier = body.modifiers.get("rig")
    if not modifier:
        modifier = body.modifiers.new("rig", "ARMATURE")
    modifier.object = rig
    # Dual quaternion skinning rather than plain linear blending.
    #
    # Linear blending averages POSITIONS. A vertex on the shoulder is shared between the
    # collarbone, which does not move, and the upper arm, which swings through a right angle to
    # bring the arm down — so it lands halfway between two places that are nowhere near each
    # other, and the surface tears open into the fan visible on the shoulders. Dual quaternions
    # interpolate the ROTATION instead, which is what the joint is actually doing, so the
    # shoulder sweeps round rather than collapsing through the middle.
    modifier.use_deform_preserve_volume = True
    body.parent = rig

    # PASS 1 — bind against RADIUS, which is now a SEED rather than a set of trusted constants.
    # Every value in that table was either measured off a chest vertex (the comment above it says
    # so) or borrowed wholesale from `make_character.py`'s different, procedurally-built
    # character — `RADIUS["head"] = 0.145` turned out to be the latter: this mesh's actual skull
    # is at most 0.114 wide, measured directly off its vertices, and the gap is exactly what let
    # `head` keep winning a thin band of shoulder skin no matter how the metric or BLEND were
    # tuned. Guessing is fine as a STARTING point precisely because the mesh can correct it: pass
    # 1 only exists to find out, for each bone, which vertices currently think they belong to it.
    # One number per bone here — the seed has no taper to guess at, only PASS 2 measures one.
    seed = {name: RADIUS.get(name, 0.06) * height for name in lines}
    winners = bind(seed, seed)

    def percentile(values, pct):
        """Nearest-rank percentile, with no numpy dependency — this file has never needed the
        scientific stack and Blender's bundled Python does not reliably ship it."""
        ordered = sorted(values)
        k = min(len(ordered) - 1, int(round(pct / 100 * (len(ordered) - 1))))
        return ordered[k]

    def fit_line(points):
        """Ordinary least-squares line through (t, radius) points — again no numpy, same reason
        as `percentile`. Returns (slope, intercept) such that the fitted radius at a given t is
        `slope * t + intercept`. Tried as the fallback for a bone whose end bins are too thin to
        trust directly, per the task — see the comment on `measure_taper` for why it is no longer
        called: it measurably made things worse, not better, and the evidence for that is there
        rather than in this function, which still exists because the next person's instinct will
        be the same one that produced it, and the reason not to needs to be in reach of the code,
        not only in a report."""
        n = len(points)
        if n == 1:
            return 0.0, points[0][1]
        mean_t = sum(t for t, _ in points) / n
        mean_r = sum(r for _, r in points) / n
        denominator = sum((t - mean_t) ** 2 for t, _ in points)
        if denominator < 1e-12:
            return 0.0, mean_r
        slope = sum((t - mean_t) * (r - mean_r) for t, r in points) / denominator
        return slope, mean_r - slope * mean_t

    # PASS 2 — re-bind against a radius TAPER measured from PASS 1's winners, instead of the
    # seed, and instead of the single uniform number this used to settle on.
    #
    # A torso tapers from chest to waist. A thigh tapers from hip to knee. One radius cannot
    # describe either without being right at one end and wrong at the other — which is exactly
    # the bind this file was in: `spine`'s single measured radius (percentile of ALL its pass-1
    # winners, chest and waist pooled together) comes out chest-width, about 20% too fat for the
    # waist measured on its own (0.106 pooled vs ~0.087 measured directly off the mesh at pelvis
    # height). Shrinking that single number to fit the waist was tried and made BOTH ends worse —
    # it under-claims the chest, which is load-bearing for the very first fix in this file (the
    # chest/arm ambiguity), while still not matching the waist's own taper. A single number was
    # never going to satisfy both ends at once, because the bone has a real taper and the number
    # does not. So now each bone gets a radius at `t=0` (its head joint) and `t=1` (its tail),
    # linearly interpolated between — a capsule that narrows or widens along its own length,
    # exactly the shape `bind()`'s signed-distance formula already works in real units to use.
    def measure_taper(winners_in):
        """One taper measurement from one pass's winners: bin by `t` along each bone, and read
        the percentile of perpendicular distance in the end bins directly — but only on a bone
        with enough vertices at BOTH ends to trust that reading (`TAPER_MIN_BIN` each).

        A line fit through all the bins was tried first, per the task, for bones whose end bins
        come out too thin — `upper_arm`, `forearm` and `shin` all have far fewer winners than
        `spine` or `thigh`, a few hundred against several thousand, so their end bins empty out
        fast. Measured it rather than trusted it: the fit is an EXTRAPOLATION whenever the actual
        end bin is the thin one (that is what "too thin to trust directly" means), and on these
        smaller bones it extrapolated to a handful of millimetres at one end and came out
        asymmetric between an otherwise-mirrored left and right (`forearm.left` head 0.058m
        against `forearm.right` head 0.015m, same mesh, mirrored geometry). That is not a
        sharper measurement, it is the fit running out of real data and guessing past the edge of
        it — and it cost real ground: re-swept against both poses with the line fit live, arms-
        down rose to 1.3-2.9% (worse at every `SMOOTH_REPEAT` tried) for no corresponding gain on
        squat. A bone that does not have enough data to measure a taper directly is left UNTAPERED
        — both ends get the one plain percentile of all its winners, exactly what this already
        did before PASS 2 grew a second number — rather than handed a shape the data does not
        support. Which bones got a real taper and which stayed uniform is printed either way.
        """
        by_bone = {name: [] for name in lines}
        for winner_name, distance, t in winners_in.values():
            by_bone[winner_name].append((t, distance))

        head, tail, methods = {}, {}, {}
        for name in lines:
            samples = by_bone[name]
            if len(samples) < RADIUS_MIN_SAMPLE:
                head[name] = tail[name] = seed[name]
                methods[name] = f"only {len(samples)} vertices won it, too few — kept the seed"
                continue
            uniform = percentile([d for _, d in samples], RADIUS_PERCENTILE)
            bins = [[] for _ in range(TAPER_BINS)]
            for t, d in samples:
                bins[min(TAPER_BINS - 1, int(t * TAPER_BINS))].append(d)
            head_n, tail_n = len(bins[0]), len(bins[-1])
            if head_n >= TAPER_MIN_BIN and tail_n >= TAPER_MIN_BIN:
                r_head = percentile(bins[0], RADIUS_PERCENTILE)
                r_tail = percentile(bins[-1], RADIUS_PERCENTILE)
                methods[name] = f"tapered, end bins directly ({head_n}/{tail_n} vertices)"
            else:
                r_head = r_tail = uniform
                methods[name] = (f"uniform, not tapered — end bins too thin to trust directly "
                                  f"({head_n}/{tail_n} of {len(samples)} vertices)")
            head[name], tail[name] = r_head, r_tail
        return head, tail, methods

    # Measured once, from PASS 1's winners, and left at that — not because once was obviously
    # enough, but because measuring it AGAIN was tried and measurably made things worse, the same
    # shape of finding as the line-fit fallback above.
    #
    # A point below the PELVIS joint clamps to `t=0` on the `spine` segment regardless of how far
    # below it actually sits, so `spine`'s own `t=0` bin mixes true waist-height vertices in with
    # glute and upper-thigh vertices that only LOOK like `t=0` because they are past the end of
    # the segment — exactly the vertices this taper exists to take away from `spine`. Measuring
    # from PASS 1's seed-biased winners inherits some of that contamination, and re-measuring
    # from each pass's own (already tapered) winners does clear more of it with every repeat —
    # `spine`'s head radius moved 0.1003 -> 0.0948 -> 0.0945, settling by the third pass, and the
    # vertex count the task asked me to watch (`spine`-dominant vertices below the hip) fell with
    # it, 713 -> 611 -> 523. That part worked exactly as expected. But re-measuring `spine` this
    # way also re-measures its CHEST end from a chest-only sample instead of one pooled with the
    # (narrower) waist, and with the dilution gone the chest end drifted WIDER each pass — which
    # matters because the chest end is load-bearing for the original chest/arm fix, the same
    # tension this task's own diagnosis already named. Re-swept both poses against the number of
    # refinement passes: one pass improved both (arms-down 12.85mm -> 12.51mm, squat-bottom
    # 42.3mm -> 39.2mm); three passes — the point where `spine`'s own radius had actually
    # converged — made both worse (arms-down to 22.7mm, squat-bottom to 76.3mm), because the same
    # redistribution that correctly thins `spine` at the waist also measurably fattens it at the
    # chest and reopens the shoulder's own delicate balance a little more with every repeat. The
    # vertex-count claim is real and verified; iterating until it is fully satisfied costs more
    # than it buys. Measured once, kept it at that.
    radius_head, radius_tail, methods = measure_taper(winners)
    bind(radius_head, radius_tail)
    for name in lines:
        print(f"  radius[{name}]: seed {seed[name]:.4f} -> head {radius_head[name]:.4f} "
              f"tail {radius_tail[name]:.4f}, {methods[name]}")

    # CHANGE A — smooth the bound weights across the surface.
    #
    # Nothing above this line changes what the residual shoulder tear actually was: two mesh
    # vertices 1.5mm apart whose WEIGHT SETS differ (one majority `upper_arm`+`clavicle`, its
    # neighbour majority `spine`+`clavicle`), not which single bone is "closest". No radius and no
    # BLEND value can fix that by construction, because the nearest-capsule-per-vertex approach
    # answers each vertex independently and has no idea its answer disagrees with its neighbour's
    # until the mesh is posed and the disagreement becomes a visible tear. Smoothing works on the
    # thing that was actually wrong: it diffuses each vertex's weight vector toward its mesh
    # neighbours', so adjacent vertices end up holding similar BLENDS of bones regardless of which
    # bone the independent, per-vertex ranking above happened to prefer.
    #
    # Has to run in Edit Mode: `vertex_group_smooth`'s poll fails in Object Mode ("context is
    # incorrect"), and entering Edit Mode invalidates any Python reference to `body.data.vertices`
    # taken before this point — body.data.vertices has to be re-read fresh after, not reused.
    bpy.ops.object.select_all(action="DESELECT")
    body.select_set(True)
    bpy.context.view_layer.objects.active = body
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.object.vertex_group_smooth(group_select_mode="ALL", factor=SMOOTH_FACTOR,
                                        repeat=SMOOTH_REPEAT)
    bpy.ops.object.mode_set(mode="OBJECT")

    names = {g.index: g.name for g in body.vertex_groups}

    # UNIFY — give every vertex that shares a REST position with another vertex the exact same
    # weight vector, after smoothing rather than before it.
    #
    # This mesh has 46,261 vertices on only 26,651 distinct positions: image-to-3D duplicated a
    # vertex everywhere it put a UV seam, and never told anyone, because nothing before this
    # measured it. 34,172 vertices are stacked on another one. Of the coincident pairs, 26,304
    # are NOT joined by a mesh edge — so `body.data.edges` never sees them, which means neither
    # the stretch-ratio check nor hard-seam detection above can see them either, and 10,882 of
    # those coincident groups came out of PASS 2 and smoothing with members holding genuinely
    # different weights: one half of a duplicated vertex at the shoulder scored `upper_arm.left
    # 0.73 / clavicle.left 0.27`, its exact twin at the same rest position scored `head 0.50 /
    # neck 0.18 / upper_arm.left 0.12 / spine 0.08`. Posed, that pair does not stretch an edge —
    # there is no edge — it just comes APART, because the two faces that meet at that seam are
    # being driven by almost entirely different bones. The widest such gap measured 54.3mm, 5.4%
    # of body height, as a visible gash across the shoulder in a render while every edge-based
    # number in this file kept reporting a clean bind. That is the THIRD time in this file that a
    # check passed while posing still tore: scaled distance let a fat bone colonise a thin one and
    # the coverage check could not see it; `neighbours_of` had a hole at the hip and the stretch
    # ratio alone could not prove it; now a duplicated vertex can hold two different opinions
    # about its own skin and no EDGE-walking check, however careful, was ever going to see it,
    # because the two halves of the crack are not connected by an edge to walk.
    #
    # The fix is not to weld them — `clean()` above already tried that, left a visible dark wedge
    # of inverted faces in the REST pose before anything was even posed, and was abandoned for
    # exactly that reason. Geometry stays exactly as supplied. Instead every vertex at a shared
    # position is handed the SAME weight vector, so coincident vertices deform IDENTICALLY by
    # construction: same rest position, same weights, same blended transform, same posed
    # position, every time, for any pose, not just the ones tested here.
    #
    # AVERAGED, not picked — and that was not the first guess, it is the SURVIVOR of a direct
    # comparison. The instinct, tried first here, is that averaging manufactures a vertex that
    # belongs to neither limb: the shoulder pair above (`spine 0.607 / upper_arm.left 0.199` next
    # to `upper_arm.left 0.676 / forearm.left 0.204`) averages to something that tracks the
    # midpoint of two bones which swing apart, which sounds exactly like the ambiguity this whole
    # file exists to resolve. So picking was implemented and measured: copy the one member's full
    # vector whose OWN dominant bone sits at the smallest (most negative, most "confidently
    # inside") signed surface distance, to every other member. It closes the crack exactly as
    # well — identical weights are identical weights either way, 0.00mm either way — but on this
    # mesh it made the thing that matters (absolute edge growth, arms-down) WORSE, not better:
    # 1.29% of body height (averaging) to a measured 2.43% best case (picking, re-swept across
    # SMOOTH_REPEAT 0-30; it did not recover at any dose). Rendered and compared the shoulder
    # collar notch at picking's own best setting against averaging's — same size, same place, if
    # anything very slightly more visible. The reason: picking keeps each duplicate's SHARPEST,
    # most confident assignment, and confident-and-different is precisely what tears when the two
    # confident-but-opposite vertices sit one edge apart from a THIRD, non-duplicate vertex that
    # smoothing had carefully matched to a blend of both. Averaging's "vertex that belongs to
    # neither limb" was, on this mesh, closer to what its real smoothed neighbours already wanted
    # than either duplicate's full-confidence opinion was. Measured twice — confidence computed
    # from the post-smoothing current dominant bone, and separately from PASS 2's own winner
    # before smoothing touched anything — both landed in the same worse place, so this is not an
    # artifact of which "dominant bone" picking uses.
    #
    # Done AFTER smoothing on purpose, same as before. Smoothing diffuses each vertex toward its
    # OWN mesh neighbours, and two coincident-but-unconnected vertices are, by definition, not
    # each other's neighbours — they sit on opposite sides of a seam with different one-rings.
    # Run smoothing after this instead and it would immediately start pulling the two unified
    # duplicates apart again, in exactly the directions their real (different) neighbourhoods
    # pull, which is the crack reopening by the file's own next step.
    key_scale = 1.0 / (height * 1e-5)

    def position_key(point):
        return (round(point.x * key_scale), round(point.y * key_scale), round(point.z * key_scale))

    coincident = {}
    for vertex in body.data.vertices:
        coincident.setdefault(position_key(vertex.co), []).append(vertex.index)
    coincident_groups = [members for members in coincident.values() if len(members) > 1]

    for members in coincident_groups:
        totals = {}
        for idx in members:
            for g in body.data.vertices[idx].groups:
                name = names[g.group]
                totals[name] = totals.get(name, 0.0) + g.weight
        grand_total = sum(totals.values())
        if grand_total <= 1e-9:
            continue
        unified = {name: w / grand_total for name, w in totals.items() if w > 0.0}
        for idx in members:
            # `g.group` is read LAZILY off Blender's own per-vertex weight array, not copied out
            # when the list comprehension runs — removing the vertex from one group can shift
            # that array and leave a later `g` in this same list reading the WRONG index, so the
            # plain-int indices are pulled out in one pass BEFORE any removal starts. This was a
            # real bug, found while testing the picking variant above: it left thousands of
            # "unified" pairs still carrying a leftover sliver of a group that was never actually
            # cleared, which `cracks` below caught immediately.
            existing = [g.group for g in body.data.vertices[idx].groups]
            for group_index in existing:
                body.vertex_groups[names[group_index]].remove([idx])
            for name, w in unified.items():
                body.vertex_groups[name].add([idx], w, "REPLACE")

    arms = {"upper_arm.left", "upper_arm.right", "forearm.left", "forearm.right",
            "hand.left", "hand.right"}
    low, span = min(zs), height
    hips = joints["pelvis"][2]
    half = max(abs(joints["shoulder.left"][0]), abs(joints["shoulder.right"][0]))
    elbow = abs(joints["elbow.right"][0])

    def near_forearm(point, joints, span):
        for side in ("left", "right"):
            a = Vector(joints[f"elbow.{side}"])
            b = Vector(joints[f"wrist.{side}"])
            axis = b - a
            length = axis.length_squared
            if length < 1e-12:
                continue
            t = max(0.0, min(1.0, (point - a).dot(axis) / length))
            if (point - (a + axis * t)).length < span * 0.05:
                return True
        return False

    unbound = torso = legs = arm_owned = arm_region = 0
    dominant = {}
    for vertex in body.data.vertices:
        held = sorted(((names[g.group], g.weight) for g in vertex.groups), key=lambda p: -p[1])
        dominant[vertex.index] = held[0] if held else (None, 0.0)
        if not held:
            unbound += 1
            continue
        # Judged by distance to the forearm bone itself rather than by a box in space. Two
        # earlier versions picked a height band suited to hanging arms and selected nothing at
        # all on a T-pose model, reporting a meaningless 0/0 twice over.
        if near_forearm(vertex.co, joints, span):
            arm_region += 1
            if held[0][0] in arms:
                arm_owned += 1
        if held[0][0] not in arms:
            continue
        if abs(vertex.co.x) < half * 0.9 and hips < vertex.co.z < hips + span * 0.25:
            torso += 1
        if abs(vertex.co.x) < half * 0.8 and vertex.co.z < low + span * 0.34:
            legs += 1

    # Hard-seam detection: the check above cannot see this bug class at all. It counts arm-region
    # coverage and torso/leg misassignment, and a shoulder vertex that the new metric still hands
    # entirely to `head` falls into neither bucket — it is not in the arm region box, and `head`
    # is never counted as a torso or leg mistake. That blind spot is exactly why five earlier
    # fixes to this file were verified as "100% clean" on a mesh that tore open at the shoulder
    # the moment it was posed: the report had nothing that could have caught it.
    #
    # A hard seam is a mesh EDGE whose two endpoints are both owned outright (>= 0.85) by
    # different bones that do not share a joint, parent, or child in `neighbours_of()`. Dominant
    # weight alone is not enough to flag — two bones that DO meet at a joint are supposed to
    # disagree about a boundary vertex, that is what the BLEND zone is for — so the pair has to
    # be checked against `near` as well. Walking `body.data.edges` rather than nearby vertices
    # means only mesh connectivity that actually has to deform continuously is tested; two
    # vertices that happen to sit close together but belong to unconnected faces are not a seam.
    hard_seams = 0
    seam_examples = []
    for edge in body.data.edges:
        i, j = edge.vertices
        bone_i, weight_i = dominant[i]
        bone_j, weight_j = dominant[j]
        if bone_i is None or bone_j is None or bone_i == bone_j:
            continue
        if bone_j in near.get(bone_i, set()):
            continue
        if weight_i >= 0.85 and weight_j >= 0.85:
            hard_seams += 1
            if len(seam_examples) < 5:
                seam_examples.append((bone_i, bone_j, tuple(body.data.vertices[i].co)))

    # Crack detection: the coincident-vertex counterpart of hard-seam detection, for the failure
    # UNIFY above exists to close. A crack is a pair of vertices at the SAME rest position, not
    # joined by a mesh edge, whose weight vectors differ — which after a correct UNIFY pass
    # should never happen, because every member of a coincident group was just handed the exact
    # same vector. Reported and checked anyway, the same way `hard_seams` is, because that is
    # what catches the next person's change (or this file's own next edit to UNIFY) quietly
    # breaking the guarantee again — a weight vector compared bone-for-bone, not a distance, so
    # this needs no pose to evaluate and no render to be wrong about.
    edges_seen = {frozenset(edge.vertices) for edge in body.data.edges}

    def weight_vector(idx):
        return {names[g.group]: round(g.weight, 6) for g in body.data.vertices[idx].groups
                if g.weight > 1e-6}

    cracks = 0
    crack_examples = []
    for members in coincident_groups:
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                i, j = members[a], members[b]
                if frozenset((i, j)) in edges_seen:
                    continue
                if weight_vector(i) != weight_vector(j):
                    cracks += 1
                    if len(crack_examples) < 5:
                        crack_examples.append((i, j, tuple(body.data.vertices[i].co)))

    return {"unbound": unbound, "torso_to_arm": torso, "leg_to_arm": legs,
            "arm_region": arm_region, "arm_owned": arm_owned,
            "hard_seams": hard_seams, "seam_examples": seam_examples,
            "cracks": cracks, "crack_examples": crack_examples}


def describe(report):
    """Say whether the binding is usable, and refuse to pretend when it is not."""
    covered = report["arm_owned"] / max(report["arm_region"], 1)
    print(f"binding: {report['unbound']} vertices bound to nothing, "
          f"{report['arm_owned']}/{report['arm_region']} of the arm region owned by arm bones "
          f"({covered:.0%}), {report['torso_to_arm']} torso and {report['leg_to_arm']} leg "
          f"vertices wrongly owned by arms, {report['hard_seams']} hard seams, "
          f"{report['cracks']} cracks")
    if report["unbound"]:
        raise SystemExit("the mesh is not bound to the rig — posing it would move nothing. The "
                         "heat solver usually fails here because the mesh is not manifold; see "
                         "the cleaning step.")
    if covered < 0.75:
        raise SystemExit(f"only {covered:.0%} of the arm belongs to the arm bones, so the arms "
                         "would barely move. Check the joint placement before rendering.")
    if report["hard_seams"]:
        pairs = sorted({f"{a} / {b}" for a, b, _ in report["seam_examples"]})
        where = "; ".join(f"{a} / {b} at {tuple(round(c, 3) for c in pos)}"
                           for a, b, pos in report["seam_examples"])
        raise SystemExit(
            f"{report['hard_seams']} mesh edges are a hard seam: two bones that do not share a "
            f"joint, a parent, or a child — {', '.join(pairs)} — own neighbouring skin outright "
            "(both sides weighted >= 0.85), with nothing blending across the join. Whichever of "
            "those two bones moves, that edge gets pulled apart rather than bent, which is the "
            f"flat-sheet tear this file exists to catch. Examples: {where}.")
    if report["cracks"]:
        where = "; ".join(f"v{a}/v{b} at {tuple(round(c, 3) for c in pos)}"
                           for a, b, pos in report["crack_examples"])
        raise SystemExit(
            f"{report['cracks']} pairs of coincident vertices — duplicated at the same rest "
            "position by image-to-3D, and not joined by a mesh edge — came out of UNIFY still "
            "holding different weights. There is no edge between them for any stretch or "
            "seam check to see, so posing opens a crack with nothing reporting it except this: "
            f"each side of the duplicate answers to different bones. Examples: {where}.")


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--save", default=None)
    parser.add_argument("--still", default=None)
    args = parser.parse_args(argv)

    body = load(args.model)
    print(f"imported {len(body.data.polygons)} faces")
    decimate(body)
    print(f"decimated to {len(body.data.polygons)} faces")

    shot = Path(args.still or "/tmp/fit").with_name("silhouette.png")
    m = measure(body, shot)
    print(f"silhouette {shot}")
    print(f"height {m['height']:.3f}  floor {m['low']:.3f}  crown {m['high']:.3f}")
    for name in ("neck", "shoulder", "arms_clear", "crotch", "hand_low"):
        row = m[name]
        where = ("NOT FOUND" if row is None
                 else f"{m['frac'](row):.3f} of height, z={m['z_of'](row):.3f}")
        print(f"   measured {name:11} {where}")

    joints = joints_from(m)
    for name in sorted(joints):
        x, y, z = joints[name]
        print(f"   joint {name:16} ({x:+.3f}, {y:+.3f}, {z:.3f})")

    rig = build_rig(joints)
    report = weight(body, rig, joints)
    describe(report)

    make_character.stage(None, body)
    scene = bpy.context.scene
    make_character.use_cycles(scene, samples=48)
    scene.render.resolution_x, scene.render.resolution_y = 640, 860

    if args.still:
        scene.render.filepath = args.still
        bpy.ops.render.render(write_still=True)
        print("rendered", args.still)
    if args.save:
        bpy.ops.wm.save_as_mainfile(filepath=args.save)
        print("saved", args.save)


if __name__ == "__main__":
    main()
