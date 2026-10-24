#!/usr/bin/env python
"""Build the house mascot as a rigged 3D character, from primitives, inside Blender.

    blender --background --python pipeline/blender/make_character.py -- --save /tmp/mascot.blend

Why build it rather than convert the drawing. Image-to-3D wants an account and a GPU, Mixamo's
auto-rigger wants an Adobe login, and both hand back a mesh somebody then has to clean. This
character is a cute rounded cartoon — a big head, smooth limbs, no anatomy on show — which is
precisely the shape that primitives describe well. Spheres and capsules at measured proportions
get closer to it than a photogrammetry mesh would, and the whole thing is deterministic: the
same numbers produce the same character every run, on any machine, with nothing to log in to.

The proportions come from `pose_model.PROPORTIONS`, the same table the pose solver uses, so the
body being animated and the skeleton driving it agree by construction rather than by a mapping
somebody maintains.

The bones are named for the joints in `pose_model`, which removes the other half of the usual
pain: there is no rig-to-skeleton mapping table here because the rig IS the skeleton. A pose that
previews correctly as a stick figure cannot be mis-mapped on the way to the character.
"""

import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pose_model  # noqa: E402

P = pose_model.PROPORTIONS

# The house palette, lifted from pipeline/restyle.py so the 3D character and the drawn avatars
# are the same character family rather than two cousins.
# Sampled off reference/muscle-figure/front.png rather than guessed, so the rendered character
# and the drawn one are the same colours and not merely similar ones. Linear, because Blender
# wants linear and the PNG is sRGB: an sRGB triple pasted straight in renders washed out, which
# is what the first pass looked like.
SKIN = (0.538, 0.287, 0.132, 1.0)     # sampled: sRGB 194,146,102
BLUSH = (0.600, 0.240, 0.165, 1.0)
HAIR = (0.102, 0.059, 0.049, 1.0)     # sampled: sRGB 90,69,63
CLOTH = (0.521, 0.484, 0.465, 1.0)    # sampled: sRGB 191,185,182
WHITE = (0.930, 0.930, 0.920, 1.0)
EYE = (0.040, 0.026, 0.021, 1.0)
BROW = (0.075, 0.042, 0.034, 1.0)
MOUTH = (0.300, 0.105, 0.090, 1.0)
BACKDROP = (0.386, 0.701, 0.665, 1.0)  # restyle's #a7dad5
METAL = (0.115, 0.125, 0.135, 1.0)

# A head this size is the single thing that makes the style read. Real adults are about seven and
# a half heads tall; this character is nearer four, which is what `restyle.STYLE` means by "a big
# rounded head" and what every drawn avatar in the app already looks like.
# Bigger than anatomy. The drawn mascot is about four heads tall against a real adult's seven and
# a half, and the head is the first thing that makes the style read — at anatomical scale the
# render came back looking like a mannequin rather than like the avatars already in the app.
HEAD_RADIUS = 0.136


def use_cycles(scene, samples=96):
    """Path tracing, not EEVEE.

    EEVEE approximates: its shadows are shadow maps, its subsurface is a screen-space trick and
    its ambient occlusion has to be asked for. All three are exactly what makes the reference
    drawing look soft — light wrapping into the crease under a chin, a cheek glowing where it
    thins, contact darkening where an arm meets a body. Cycles computes them because they are
    what light does.

    It is slower, but this renders twenty-odd frames of one small figure, so the difference is
    a couple of minutes and not an afternoon. GPU where Metal offers it, with the denoiser on,
    which is what makes a low sample count usable.
    """
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.cycles.use_adaptive_sampling = True
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for kind in ("METAL", "OPTIX", "CUDA", "HIP"):
            try:
                prefs.compute_device_type = kind
            except TypeError:
                continue
            prefs.get_devices()
            for device in prefs.devices:
                device.use = True
            scene.cycles.device = "GPU"
            break
    except (KeyError, AttributeError):
        # No GPU backend available; CPU still renders, just slower, and saying so beats a
        # silent fall back that looks like a hang.
        print("cycles: no GPU backend found, rendering on CPU")
    return scene


def clear():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.armatures):
        for item in list(block):
            block.remove(item)


def material(name, colour, roughness=0.62, subsurface=0.0, sheen=0.0):
    """A shader, with real subsurface scattering where it is skin.

    The skin had none, and that was the single biggest reason the renders looked like painted
    plastic next to the drawing. Light does not stop at skin — it goes in, bounces around and
    comes back out slightly redder, which is what gives a cheek its glow and a thin ear its
    warmth. A diffuse shader with the right colour cannot fake it, however the lights are
    arranged.

    `sheen` does the equivalent small job for fabric: cloth catches a soft rim of light at
    glancing angles, and without it a vest reads as painted on.
    """
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = colour
    bsdf.inputs["Roughness"].default_value = roughness

    def put(name, value):
        if name in bsdf.inputs:
            bsdf.inputs[name].default_value = value

    put("Specular IOR Level", 0.22)
    put("Specular", 0.22)
    if subsurface:
        put("Subsurface Weight", subsurface)
        put("Subsurface", subsurface)
        # Scatters furthest in red, least in blue — the usual flesh ratio. Scaled to this
        # character, whose limbs are a few centimetres across.
        put("Subsurface Radius", (0.030, 0.012, 0.007))
        if "Subsurface Color" in bsdf.inputs:
            bsdf.inputs["Subsurface Color"].default_value = colour
        put("Subsurface Scale", 0.022)
    if sheen:
        put("Sheen Weight", sheen)
        put("Sheen", sheen)
        put("Sheen Roughness", 0.35)
    return mat


def capsule(name, start, end, radius, mat, segments=24):
    """A limb: a cylinder with a ball on each end, pointed from one joint to the next.

    Built along +Z and then rotated onto the bone's direction, which is the same convention
    `pose_model` uses, so a limb and its bone never disagree about which way they point.
    """
    start, end = Vector(start), Vector(end)
    axis = end - start
    length = axis.length
    bpy.ops.mesh.primitive_cylinder_add(vertices=segments, radius=radius, depth=length,
                                        location=(start + end) / 2)
    body = bpy.context.object
    body.name = name
    body.rotation_mode = "QUATERNION"
    body.rotation_quaternion = axis.to_track_quat("Z", "Y")

    caps = []
    for point in (start, end):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=segments, ring_count=segments // 2,
                                             radius=radius, location=point)
        caps.append(bpy.context.object)

    for part in [body] + caps:
        part.data.materials.append(mat)
        bpy.ops.object.select_all(action="DESELECT")
        part.select_set(True)
        bpy.context.view_layer.objects.active = part
        bpy.ops.object.shade_smooth()

    bpy.ops.object.select_all(action="DESELECT")
    for part in [body] + caps:
        part.select_set(True)
    bpy.context.view_layer.objects.active = body
    bpy.ops.object.join()
    return bpy.context.object


def ball(name, centre, radius, mat, squash=(1.0, 1.0, 1.0)):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=radius, location=centre)
    obj = bpy.context.object
    obj.name = name
    obj.scale = squash
    obj.data.materials.append(mat)
    bpy.ops.object.shade_smooth()
    return obj


# How far the arms are held from the body in the rest pose. Not a style choice: with the arms
# hanging against the ribs, a hand sits a couple of centimetres from a thigh, and any weighting
# by proximity — Blender's or ours — gives the thigh to the forearm. It is why rigs are built in
# an A-pose, and the drawn reference happens to stand this way too.
REST_ABDUCTION = 34.0


def rest_skeleton():
    """The character standing straight, from the same solver the animation uses.

    Deliberately solved rather than written out again: a rest pose typed by hand here would drift
    from `pose_model`'s idea of standing the first time either changed, and then every animation
    would start from a body slightly different to the one it was posing.
    """
    spread = [{"joint": "shoulder", "side": side, "measure": "abduction",
               "degrees": REST_ABDUCTION, "low": REST_ABDUCTION, "high": REST_ABDUCTION,
               "source": "rig rest pose, not from the notes", "note": ""}
              for side in ("left", "right")]
    return pose_model.solve({"phase": "rest", "orientation": "standing", "angles": spread})


def own_part(part, bone):
    """Tag a freshly built piece with the bone that moves it, and hand it back for the parts list."""
    own(part, bone)
    return part


# How thick the body is at each joint, as a radius. The Skin modifier interpolates between
# them, so these few numbers shape the whole figure — wide at the chest, narrowing to the wrist.
# Widened against the drawing rather than against anatomy. Measured off
# reference/muscle-figure/front.png: its head is 0.269 of the figure's height and its body is
# 0.42 at the widest. The first pass had the head right and the torso at 0.17 — which is why the
# head looked oversized when it was the body that was thin.
GIRTH = {
    "pelvis": 0.104, "chest": 0.112, "neck": 0.052,
    "shoulder": 0.052, "elbow": 0.041, "wrist": 0.032, "hand": 0.040,
    "hip": 0.070, "knee": 0.053, "ankle": 0.036, "foot": 0.031,
}

# Which bone owns which stretch of the body, for weighting. Order matters only in that every
# generated vertex goes to whichever of these segments it is nearest.
SEGMENTS = [
    ("spine", "pelvis", "chest"), ("neck", "chest", "neck"),
    ("upper_arm.left", "shoulder.left", "elbow.left"),
    ("forearm.left", "elbow.left", "wrist.left"),
    ("hand.left", "wrist.left", "hand.left"),
    ("upper_arm.right", "shoulder.right", "elbow.right"),
    ("forearm.right", "elbow.right", "wrist.right"),
    ("hand.right", "wrist.right", "hand.right"),
    ("thigh.left", "hip.left", "knee.left"), ("shin.left", "knee.left", "ankle.left"),
    ("foot.left", "ankle.left", "foot.left"),
    ("thigh.right", "hip.right", "knee.right"), ("shin.right", "knee.right", "ankle.right"),
    ("foot.right", "ankle.right", "foot.right"),
]


def skinned_body(skeleton, name="body"):
    """One continuous mesh grown along the skeleton, rather than a pile of separate capsules.

    This is the change that makes it look like the drawing. Capsules gave a body with a visible
    seam and a protruding ball at every joint — a doll assembled from parts, which is exactly
    what it looked like. Blender's Skin modifier grows a single surface along an edge skeleton
    with a radius at each vertex, and Subdivision smooths it, so a shoulder flows into an arm
    with no join at all.

    The cost is that the mesh no longer arrives in labelled pieces, so weights cannot simply be
    "this part belongs to that bone". They are assigned afterwards by nearest segment, which is
    only unambiguous because the rest pose holds the arms away from the body — see
    REST_ABDUCTION.
    """
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    order, coords = [], []
    for joint in ("pelvis", "chest", "neck"):
        order.append(joint); coords.append(skeleton.at(joint))
    for side in ("left", "right"):
        for joint in ("shoulder", "elbow", "wrist", "hand", "hip", "knee", "ankle", "foot"):
            order.append(f"{joint}.{side}"); coords.append(skeleton.at(f"{joint}.{side}"))
    index = {name: i for i, name in enumerate(order)}

    edges = [(index["pelvis"], index["chest"]), (index["chest"], index["neck"])]
    for side in ("left", "right"):
        edges += [
            (index["chest"], index[f"shoulder.{side}"]),
            (index[f"shoulder.{side}"], index[f"elbow.{side}"]),
            (index[f"elbow.{side}"], index[f"wrist.{side}"]),
            (index[f"wrist.{side}"], index[f"hand.{side}"]),
            (index["pelvis"], index[f"hip.{side}"]),
            (index[f"hip.{side}"], index[f"knee.{side}"]),
            (index[f"knee.{side}"], index[f"ankle.{side}"]),
            (index[f"ankle.{side}"], index[f"foot.{side}"]),
        ]
    mesh.from_pydata(coords, edges, [])
    mesh.update()

    skin = obj.modifiers.new("skin", "SKIN")
    skin.use_smooth_shade = True
    bpy.context.view_layer.objects.active = obj
    for i, joint in enumerate(order):
        radius = GIRTH[joint.split(".")[0]]
        vertex = obj.data.skin_vertices[0].data[i]
        vertex.radius = (radius, radius)
    # The root has to be set or the modifier picks one and the surface can come out inside out.
    obj.data.skin_vertices[0].data[index["pelvis"]].use_root = True

    smooth = obj.modifiers.new("smooth", "SUBSURF")
    smooth.levels = 2
    smooth.render_levels = 2

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.ops.object.modifier_apply(modifier="skin")
    bpy.ops.object.modifier_apply(modifier="smooth")
    bpy.ops.object.shade_smooth()
    return obj


def weight_to_nearest(obj, skeleton):
    """Give every vertex to the bone whose segment it lies closest to.

    Unambiguous only because of the A-pose — see REST_ABDUCTION. Hard weights with no blending,
    which on a soft rounded body shows as a slight crease at a bent joint and nothing worse; the
    alternative, smooth falloff, is what let the arms drag the legs around in the first place.
    """
    groups = {name: obj.vertex_groups.new(name=name) for name, _, _ in SEGMENTS}
    lines = [(name, Vector(skeleton.at(a)), Vector(skeleton.at(b))) for name, a, b in SEGMENTS]
    for vertex in obj.data.vertices:
        point = vertex.co
        best, best_distance = None, None
        for name, a, b in lines:
            axis = b - a
            length = axis.length_squared
            t = 0.0 if length < 1e-12 else max(0.0, min(1.0, (point - a).dot(axis) / length))
            distance = (point - (a + axis * t)).length
            if best_distance is None or distance < best_distance:
                best, best_distance = name, distance
        groups[best].add([vertex.index], 1.0, "REPLACE")


def build_body(skeleton, equipment=None):
    skin = material("skin", SKIN, roughness=0.52, subsurface=0.55)
    cloth = material("cloth", CLOTH, roughness=0.78, sheen=0.45)
    hair = material("hair", HAIR, roughness=0.38)
    eye = material("eye", EYE, roughness=0.22)
    white = material("white", WHITE, roughness=0.25)
    brow = material("brow", BROW, roughness=0.5)
    blush = material("blush", BLUSH, roughness=0.75)
    mouth = material("mouth", MOUTH, roughness=0.5)

    at = skeleton.at
    body = skinned_body(skeleton)
    body.data.materials.append(skin)
    weight_to_nearest(body, skeleton)

    parts = []
    R = HEAD_RADIUS
    # Sat high enough that the sphere's underside clears the neck joint. At 0.74 of a radius
    # the head's bottom was BELOW where the neck starts, so it ate the neck and the figure read
    # as a head balanced straight on a pair of shoulders.
    head_centre = (at("neck")[0], at("neck")[1], at("neck")[2] + R * 0.98)
    parts.append(own_part(ball("head", head_centre, R, skin, squash=(0.97, 0.95, 1.0)), "head"))

    face_y = head_centre[1] + R * 0.78

    def on_face(name, dx, dz, radius, mat, squash=(1.0, 1.0, 1.0), depth=0.0):
        return own_part(ball(name, (head_centre[0] + dx, face_y + depth, head_centre[2] + dz),
                             radius, mat, squash=squash), "head")

    for sign, side in ((-1, "l"), (1, "r")):
        parts.append(on_face(f"sclera.{side}", sign * R * 0.33, -R * 0.03, R * 0.205, white,
                             squash=(1.0, 0.42, 1.20)))
        parts.append(on_face(f"iris.{side}", sign * R * 0.335, -R * 0.04, R * 0.130, eye,
                             squash=(1.0, 0.38, 1.05), depth=R * 0.09))
        parts.append(on_face(f"spark.{side}", sign * R * 0.29, R * 0.03, R * 0.042, white,
                             squash=(1.0, 0.38, 1.0), depth=R * 0.16))
        parts.append(on_face(f"brow.{side}", sign * R * 0.34, R * 0.21, R * 0.110, brow,
                             squash=(1.40, 0.30, 0.36), depth=-R * 0.01))
        parts.append(on_face(f"cheek.{side}", sign * R * 0.53, -R * 0.25, R * 0.130, blush,
                             squash=(1.0, 0.20, 0.70), depth=-R * 0.05))
    parts.append(on_face("nose", 0.0, -R * 0.12, R * 0.105, skin, squash=(0.95, 1.05, 0.78),
                         depth=R * 0.09))
    parts.append(on_face("smile", 0.0, -R * 0.40, R * 0.165, mouth,
                         squash=(1.30, 0.20, 0.24), depth=R * 0.01))
    parts.append(own_part(ball("hair", (head_centre[0], head_centre[1] - R * 0.13,
                                        head_centre[2] + R * 0.17),
                               R * 1.035, hair, squash=(1.04, 1.06, 0.94)), "head"))

    # Clothing as its own shell over the body, which is what gives a vest a neckline and a hem
    # rather than a painted-on stripe.
    # The vest stops well below the neck and inside the shoulders, which is what gives it a
    # neckline and armholes. Run full height and full width it simply swallowed the torso — the
    # render came back with the head sitting on a sack.
    chest, pelvis = at("chest"), at("pelvis")
    top_a = (0, 0, pelvis[2] - 0.010)
    top_b = (0, 0, chest[2] - 0.034)
    parts.append(own_part(capsule("vest", top_a, top_b, 0.117, cloth, segments=32), "spine"))
    for side in ("left", "right"):
        hip = at(f"hip.{side}")
        knee = at(f"knee.{side}")
        mid = tuple(hip[i] + (knee[i] - hip[i]) * 0.46 for i in range(3))
        parts.append(own_part(capsule(f"shorts.{side}", (hip[0], hip[1], hip[2] + 0.02), mid,
                                      0.080, cloth, segments=32), f"thigh.{side}"))

    if equipment == "dumbbells":
        metal = material("metal", METAL, roughness=0.42)
        for side in ("left", "right"):
            wrist = at(f"wrist.{side}")
            hand = at(f"hand.{side}")
            grip = tuple((wrist[i] + hand[i]) / 2 for i in range(3))
            bar_a = (grip[0] - 0.060, grip[1], grip[2])
            bar_b = (grip[0] + 0.060, grip[1], grip[2])
            parts.append(own_part(capsule(f"bar.{side}", bar_a, bar_b, 0.013, metal),
                                  f"hand.{side}"))
            for end, sign in ((bar_a, -1), (bar_b, 1)):
                plate = (end[0] + sign * 0.012, end[1], end[2])
                parts.append(own_part(ball(f"plate.{side}.{sign}", plate, 0.040, metal,
                                           squash=(0.45, 1.0, 1.0)), f"hand.{side}"))

    bpy.ops.object.select_all(action="DESELECT")
    for part in parts:
        part.select_set(True)
    body.select_set(True)
    bpy.context.view_layer.objects.active = body
    bpy.ops.object.join()
    out = bpy.context.object
    out.name = "mascot"
    bpy.ops.object.shade_smooth()
    return out


# Bone name -> (head joint, tail joint). The names are the skeleton's own, which is the point:
# there is no mapping table to keep in step, because the rig and the solver share a vocabulary.
BONES = {
    "spine": ("pelvis", "chest"),
    "neck": ("chest", "neck"),
    "head": ("neck", "head"),
    # Collarbones, and they are not decoration. Without them the whole torso is two thin bones
    # down the centre line, so a vertex on the outside of the chest is about as far from the
    # spine as it is from the shoulder joint — 0.101 against 0.102 on this model. The arm wins
    # roughly half the chest on that coin flip: 4,257 torso vertices came out owned by an arm
    # bone, and raising the arms dragged the vest out into a sheet.
    #
    # A clavicle runs across the chest to the shoulder, so the yoke belongs to something that
    # barely moves. Every real humanoid rig has one for this reason. The pose solver already
    # places both ends, so it needs no changes to drive this.
    "clavicle.left": ("chest", "shoulder.left"),
    "clavicle.right": ("chest", "shoulder.right"),
    "upper_arm.left": ("shoulder.left", "elbow.left"),
    "forearm.left": ("elbow.left", "wrist.left"),
    "hand.left": ("wrist.left", "hand.left"),
    "upper_arm.right": ("shoulder.right", "elbow.right"),
    "forearm.right": ("elbow.right", "wrist.right"),
    "hand.right": ("wrist.right", "hand.right"),
    # A "pelvis.left"/"pelvis.right" bridging bone, mirroring the clavicle, was tried and measured
    # here — see TASK-1's report. It made the squat-bottom tear WORSE (41.9mm -> 75.8mm max edge
    # growth), not better, with or without a clavicle-style partial-rotation follow, and at every
    # BLEND tested. The clavicle genuinely rotates (the collarbone lifts somewhat with the arm,
    # and `with_clavicles` makes it follow the arm's swing on top of that); `pelvis`-to-`hip.left`/
    # `hip.right` is RIGID by construction in this skeleton (hip sockets don't move relative to
    # the pelvis, only the femur rotates in them — confirmed: the offset is bit-identical between
    # squat start and bottom). A bridging bone with no genuine rotation to carry just takes weight
    # away from `thigh` near the joint without giving it anywhere useful to go, sharpening the
    # transition instead of softening it. Not added.
    "thigh.left": ("hip.left", "knee.left"),
    "shin.left": ("knee.left", "ankle.left"),
    "foot.left": ("ankle.left", "foot.left"),
    "thigh.right": ("hip.right", "knee.right"),
    "shin.right": ("knee.right", "ankle.right"),
    "foot.right": ("ankle.right", "foot.right"),
}

PARENTS = {
    "neck": "spine", "head": "neck",
    "clavicle.left": "spine", "clavicle.right": "spine",
    "upper_arm.left": "clavicle.left", "forearm.left": "upper_arm.left",
    "hand.left": "forearm.left",
    "upper_arm.right": "clavicle.right", "forearm.right": "upper_arm.right",
    "hand.right": "forearm.right",
    # Without these two, `thigh.left`/`thigh.right` had NO entry here at all — not merely
    # un-neighboured with the spine for weighting purposes, but genuinely unparented, so
    # `build_rig` left them as a second and third root of the armature and the legs hung off the
    # rig detached from the torso. That is also why `fit_model.neighbours_of()`'s family test
    # could never treat `spine` and `thigh` as neighbours (it walks this exact dict), which is
    # what let a 350-edge hard seam open at the hip the moment a leg was posed.
    "thigh.left": "spine", "thigh.right": "spine",
    "shin.left": "thigh.left", "foot.left": "shin.left",
    "shin.right": "thigh.right", "foot.right": "shin.right",
}


def build_rig(skeleton):
    bpy.ops.object.armature_add(location=(0, 0, 0))
    rig = bpy.context.object
    rig.name = "rig"
    bpy.ops.object.mode_set(mode="EDIT")
    bones = rig.data.edit_bones
    for bone in list(bones):
        bones.remove(bone)

    for name, (head, tail) in BONES.items():
        bone = bones.new(name)
        bone.head = Vector(skeleton.at(head))
        bone.tail = Vector(skeleton.at(tail))
        # A zero-length bone is silently useless and poses nothing, so it is caught here.
        if (bone.tail - bone.head).length < 1e-5:
            raise SystemExit(f"bone {name!r} has no length; the skeleton put {head} and {tail} "
                             "in the same place")
    for child, parent in PARENTS.items():
        bones[child].parent = bones[parent]
        bones[child].use_connect = False

    bpy.ops.object.mode_set(mode="OBJECT")
    return rig


def own(part, bone):
    """Give every vertex of a part to one bone, at full weight.

    Explicit rather than automatic, and that is the whole point. `ARMATURE_AUTO` computes weights
    from how close a vertex is to a bone, and in this character's rest pose the hands hang
    directly beside the thighs — so the heat map handed a share of each thigh to the forearm
    that was touching it, and a lateral raise dragged the legs out into the splits along with
    the arms. Every piece of this mesh was built for a specific bone, so there is nothing to
    infer.

    The cost is rigid joints with no soft falloff. On a body made of capsules with a ball at
    every joint that is invisible, because the ball is already the only thing a viewer sees
    bending.
    """
    group = part.vertex_groups.new(name=bone)
    group.add(range(len(part.data.vertices)), 1.0, "REPLACE")


def bind(body, rig):
    bpy.ops.object.select_all(action="DESELECT")
    body.select_set(True)
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.parent_set(type="ARMATURE_NAME")


def stage(skeleton, subject=None):
    """Backdrop and lights: the same flat pastel teal and soft even studio light the drawings
    were made with, so a rendered movement sits beside a drawn avatar without a seam."""
    # Behind the character as seen from the camera, which now stands at positive Y. Left at
    # +1.6 it sat between the two and the render came back as a flat sheet of teal.
    bpy.ops.mesh.primitive_plane_add(size=12, location=(0, -1.8, 0), rotation=(math.pi / 2, 0, 0))
    wall = bpy.context.object
    wall.name = "backdrop"
    wall.data.materials.append(material("backdrop", BACKDROP, roughness=0.95))
    bpy.ops.mesh.primitive_plane_add(size=12, location=(0, 0, 0))
    floor = bpy.context.object
    floor.name = "floor"
    floor.data.materials.append(material("floor", BACKDROP, roughness=0.95))

    # Most of the light is ambient. The drawn avatars have almost no shadow on them — the style
    # line asks for "soft even studio lighting" — so a bright even world does the work and the
    # lamps only shape it. Lamps alone, however dim, gave a modelled, half-lit look that did not
    # match the flat drawn shading at all.
    world = bpy.data.worlds.new("world") if not bpy.context.scene.world else bpy.context.scene.world
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.74, 0.83, 0.82, 1.0)
    bg.inputs["Strength"].default_value = 0.45

    for name, location, energy, size in (
            # Big and close. A small lamp far away gives a hard edge to every shadow; the soft
            # wrap on the drawing comes from a source that is physically large next to the
            # subject, so the key is a two-metre panel barely more than a metre away.
            ("key", (-1.1, 1.7, 1.6), 60, 2.2),
            ("fill", (1.3, 1.5, 1.0), 30, 2.6),
            ("rim", (0.2, -1.5, 1.9), 22, 1.8)):
        bpy.ops.object.light_add(type="AREA", location=location)
        light = bpy.context.object
        light.name = name
        light.data.energy = energy
        light.data.size = size
        direction = Vector((0, 0, 0.8)) - Vector(location)
        light.rotation_mode = "QUATERNION"
        light.rotation_quaternion = direction.to_track_quat("-Z", "Y")

    # In front of the character, not behind it. `pose_model` has +Y as the direction the figure
    # faces, so a camera at negative Y renders the back of its head — which is exactly what the
    # first render was. Turned to look back along -Y, and brought close enough that the body
    # fills the frame.
    # Framed from the figure's own height rather than from numbers that looked right once. The
    # first attempt cut the feet off, and it would have done so again the moment the proportions
    # changed. Vertical field of view from the lens and the sensor, then back off far enough to
    # hold the whole body with a margin.
    # Measured off the mesh that was actually built. Every earlier version added a margin for
    # the head, then another for the hair, then another for the feet, and the figure ended up
    # floating in the middle of a mostly empty frame — a third the height of the drawing it is
    # meant to match. The object knows its own extent; ask it.
    if subject is not None:
        corners = [subject.matrix_world @ Vector(c) for c in subject.bound_box]
        top = max(c.z for c in corners)
        bottom = min(c.z for c in corners)
    else:
        top = max(joint.position[2] for joint in skeleton.joints.values()) + HEAD_RADIUS * 2.1
        bottom = -0.08
    middle = (top + bottom) / 2
    lens, sensor = 60.0, 36.0
    fov = 2 * math.atan((sensor / 2) / lens)
    distance = ((top - bottom) * 1.06 / 2) / math.tan(fov / 2)
    bpy.ops.object.camera_add(location=(0, distance, middle), rotation=(math.pi / 2, 0, math.pi))
    camera = bpy.context.object
    camera.name = "camera"
    camera.data.lens = lens
    camera.data.sensor_fit = "VERTICAL"
    camera.data.sensor_height = sensor
    bpy.context.scene.camera = camera
    return camera


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", default=None, help="write the .blend here")
    parser.add_argument("--still", default=None, help="render one frame here, to look at it")
    parser.add_argument("--dumbbells", action="store_true")
    args = parser.parse_args(argv)

    clear()
    skeleton = rest_skeleton()
    body = build_body(skeleton, equipment="dumbbells" if args.dumbbells else None)
    rig = build_rig(skeleton)
    bind(body, rig)
    camera = stage(skeleton, body)

    scene = bpy.context.scene
    use_cycles(scene)
    scene.render.resolution_x, scene.render.resolution_y = 768, 1024
    scene.render.film_transparent = False

    print(f"built: {len(body.data.vertices)} vertices, {len(rig.data.bones)} bones")
    for bone in sorted(rig.data.bones.keys()):
        print("   bone", bone)

    if args.still:
        scene.render.filepath = args.still
        bpy.ops.render.render(write_still=True)
        print("rendered", args.still)
    if args.save:
        bpy.ops.wm.save_as_mainfile(filepath=args.save)
        print("saved", args.save)


if __name__ == "__main__":
    main()
