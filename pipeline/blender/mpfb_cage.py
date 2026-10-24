#!/usr/bin/env python
"""An MPFB human, shaped and posed to a cartoon skin: the cage the skin is worn over. In Blender.

    blender --background --python pipeline/blender/mpfb_cage.py -- \
        --skin "3d-character/3d cartoon boy.glb" --out /tmp/mpfb-cage

Why. `character.py`'s cage is hand-built: its weights are assigned ring by ring and its shoulder
has a sharp weight step that task 10 traced the last of the strap slide to. MPFB (MakeHuman for
Blender; generated bodies, weights and base mesh are CC0) has a 100%-quad body with artist-made
weights and a built-in Mixamo-compatible rig (`rig.mixamo`: 52 bones named `mixamorig:*`), so a
Mixamo clip maps onto it bone for bone. This file makes that human the cage.

The skin is a cartoon and MPFB makes people, so the human is brought to the skin in two steps:

  1. **Targets** (`fit_targets`). MPFB's own parametric targets — limb `scale-vert`, hand, foot,
     torso and hip scales — are solved by bounded least squares against the skin's measured limb
     lengths, torso height, shoulder and hip widths, together with one uniform scale, and
     `HumanService.refit` moves the rig onto the reshaped body. Each target's effect on those
     numbers is measured by setting it and refitting, so nothing assumes how far a target
     reaches. The head and neck are left out (the cartoon's head is 2.7x what the targets reach;
     the brief says not to force it). What the targets cannot reach is reported, in millimetres.
  2. **The warp to the skin's joints** (`warp_to_skin`). The cartoon is outside any human's
     range: a head 25% of its height (MPFB's adult ~12%), a thigh 1.8x its shin. So every bone is
     given an affine map from where MPFB put it to where the skin's joint is — turned, stretched
     along the bone, and widened or narrowed across it to the skin's own girth — and the body is
     carried there by linear blend skinning with MPFB's own weights. The bones are then rebuilt
     at the skin's joints. The rest pose this leaves IS the skin's pose (its T-pose, its spine
     lean, its splayed legs), so `to_skin_pose` is the identity and nothing is posed at bind.
  3. **Clean-up** of what the warp does to a mesh (`relax_joints`, `trim_socket`): the joint blends
     it swung furthest are relaxed (crossing faces 96 -> 23 on the cartoon boy), and the upper
     arm's weight medial of the shoulder socket goes to the collarbone.

Then `wear` does what `animate.wear_skin` does on the house cage: `skin_wrap.region_fit`,
`skin_wrap.wrap`, `skin_wrap.bake_to_skin` — all three now read the rig's bone names through its
profile (the "rig profiles" section of `skin_wrap`). The bake makes the head rigid and the mittens rigid
(`skin_wrap._rigid_parts`).

Frame: MPFB (and Mixamo's FBX) face -Y with the character's left at +X; the house faces +Y with
its left at -X (`pose_model`, `character.py`, `equipment`, `exercise_stage`). The warp includes
the half turn, so the finished rig is unrotated in the house frame — every house function that
aims in armature space reads it correctly — and a clip is turned by `retarget.facing` as before.

The finished rig carries `rig["skin_fit"]` (what `skin_wrap.to_skin_pose`, `wrap` and
`bake_to_skin` read) and `rig["rig_profile"] = "mpfb-mixamo"` (which bone names `skin_wrap` uses).
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector
from mathutils.bvhtree import BVHTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import skin_wrap  # noqa: E402
import hand_rig  # noqa: E402

P = "mixamorig:"
SIDES = (("left", "Left", -1.0), ("right", "Right", 1.0))   # house side, Mixamo side, house x sign
TURN = Matrix.Rotation(math.pi, 3, "Z")                     # MPFB / Mixamo frame -> house frame
PROFILE = "mpfb-mixamo"

# The macros the human is built from before the targets. A grown man: the cartoon's body below
# the neck has an adult's proportions (hip height / shoulder height 0.60, MPFB's man 0.61); only its
# head and its shins are cartoon. `age` is searched over AGES and the best fit kept.
MACRO = {"gender": 1.0, "age": 0.5, "muscle": 0.5, "weight": 0.5, "proportions": 0.5,
         "height": 0.5, "cupsize": 0.5, "firmness": 0.5,
         "race": {"asian": 0.33, "caucasian": 0.34, "african": 0.33}}
AGES = (0.1875, 0.3, 0.5)

# Targets the fit may use: (name, files). A pair of files is one variable set on both sides.
# Each variable runs -1..1: positive loads the `-incr` file, negative the `-decr` one.
TARGETS = [
    ("upperleg", ("l-upperleg-scale-vert", "r-upperleg-scale-vert")),
    ("lowerleg", ("l-lowerleg-scale-vert", "r-lowerleg-scale-vert")),
    ("upperarm", ("l-upperarm-scale-vert", "r-upperarm-scale-vert")),
    ("lowerarm", ("l-lowerarm-scale-vert", "r-lowerarm-scale-vert")),
    ("hand", ("l-hand-scale", "r-hand-scale")),
    ("foot", ("l-foot-scale", "r-foot-scale")),
    ("torso-vert", ("torso-scale-vert",)),
    ("torso-horiz", ("torso-scale-horiz",)),
    ("hip-horiz", ("hip-scale-horiz",)),
    ("hip-vert", ("hip-scale-vert",)),
]
TARGET_RIDGE = 0.02         # pull toward 0 on each target variable (keeps unhelpful ones off)
THIGH_CAP = 0.9             # x the hip joint's distance from the midline: the widest a cage thigh gets

# Features the target fit matches: limb lengths and the trunk's heights and widths. NOT the head
# (the cartoon's is 246 mm on a 1 m figure; MPFB's head targets reach ~90) and not the neck (the
# skin's "neck" runs from a synthetic chest point; Mixamo's Neck bone is a short stub): the brief
# says not to force those, and with them in, the uniform scale is dragged toward the head.
FEATURES = ("thigh", "shin", "upper_arm", "forearm", "hand", "foot", "hip_height", "torso",
            "shoulder_half", "hip_half")


def services():
    from bl_ext.user_default.mpfb.services.humanservice import HumanService
    from bl_ext.user_default.mpfb.services.targetservice import TargetService
    return HumanService, TargetService


# --- the skin's joints ---------------------------------------------------------------------------

def skin_joints(meas):
    """Every joint the cage needs, on the skin, in the house frame (+Y forward, left at -X).

    All of it comes off `skin_wrap.proportions_from` — the measurement `character.py`'s cage has
    been fitted with since task 5 — and its fitted arm and leg centre lines, so the two cages are
    put on the same skeleton, measured the same way. Returns (joints, extras).
    """
    props, place = skin_wrap.proportions_from(meas)
    m, lines = place["measured"], place["lines"]
    height, low = m["height"], m["floor"]
    cx, cy = m["centre"]
    out = {}
    for side, _, sign in SIDES:
        fz, fy = (np.array(c) for c in lines["arm"][side])
        for name, s in (("shoulder", m["s_shoulder"]), ("elbow", m["s_elbow"]),
                        ("wrist", m["s_wrist"]), ("tip", m["s_tip"])):
            out[f"{name}.{side}"] = Vector((cx + sign * s, float(np.polyval(fy, s)),
                                            float(np.polyval(fz, s))))
        ls, ly = (np.array(c) for c in lines["leg"][side])
        out[f"hip.{side}"] = Vector((cx + sign * lines["hip_half"],
                                     float(np.polyval(ly, m["hip_z"])), m["hip_z"]))
        for name, z in (("knee", m["knee_z"]), ("ankle", m["ankle_z"])):
            out[f"{name}.{side}"] = Vector((cx + sign * float(np.polyval(ls, z)),
                                            float(np.polyval(ly, z)), z))
    # The wrist, re-measured. `fit_model` puts it at 338 mm out on the boy, which is on the
    # KNUCKLES: rendered with markers (task 12), the forearm is ~50 mm thick, thins to ~26 mm at
    # 290-310 mm, and the hand thickens after it. With the wrist on the knuckles the mitten's rigid
    # part was only the fingers, half the palm bent with the forearm, and the grip had a forearm
    # 125 mm long and a hand 52 mm long (they are ~82 and ~85). So: the thinnest vertical extent of
    # the arm (palms are down in the T-pose) between the elbow and the fingertips, in 6 mm slabs.
    co = meas["co"]
    notes = place["notes"]
    for side, _, sign in SIDES:
        elbow, tip = out[f"elbow.{side}"], out[f"tip.{side}"]
        s_el, s_tip = sign * (elbow.x - cx), sign * (tip.x - cx)
        fz = np.array(lines["arm"][side][0])
        s_co = sign * (co[:, 0] - cx)
        near = np.abs(co[:, 2] - np.polyval(fz, s_co)) < 0.08 * height
        best = None
        for s in np.arange(s_el + 0.3 * (s_tip - s_el), s_tip - 0.15 * (s_tip - s_el), 0.003):
            slab = co[near & (np.abs(s_co - s) < 0.003)]
            if len(slab) < 6:
                continue
            z0, z1 = np.percentile(slab[:, 2], [2, 98])
            if best is None or (z1 - z0) < best[1]:
                best = (s, z1 - z0)
        if best is not None:
            old = out[f"wrist.{side}"]
            fy = np.array(lines["arm"][side][1])
            out[f"wrist.{side}"] = Vector((cx + sign * best[0], float(np.polyval(fy, best[0])),
                                           float(np.polyval(fz, best[0]))))
            notes = notes + [f"{side} wrist re-measured at {best[0]:.3f} out (thinnest arm, "
                             f"{best[1] * 1000:.0f} mm), not fit_model's {abs(old.x - cx):.3f}"]
    place["notes"] = notes
    lean = lines["lean"]

    def lean_y(z):
        return lean["cy"] + (lean["shoulder_y"] - lean["cy"]) * (z - m["hip_z"]) / (
            m["shoulder_z"] - m["hip_z"])

    out["pelvis"] = (out["hip.left"] + out["hip.right"]) / 2
    out["chest"] = Vector((cx, lean_y(lines["chest_z"]), lines["chest_z"]))
    out["neck"] = Vector((cx, lines["neck"]["y"], m["neck_z"]))
    head = lines["head"] or {"y": lines["neck"]["y"], "half": 0.12 * height,
                             "depth": 0.12 * height}
    out["crown"] = Vector((cx, head["y"], low + height))
    extras = {"height": height, "floor": low, "centre": (cx, cy), "props": props,
              "place": place, "head": head, "foot": lines["foot"], "lines": lines,
              "arm_radius": lines["arm_radius"], "leg_radius": lines["leg_radius"]}
    return out, extras


def skin_features(j, extras):
    sides = ("left", "right")
    avg = lambda f: sum(f(s) for s in sides) / 2                      # noqa: E731
    return {
        "thigh": avg(lambda s: (j[f"knee.{s}"] - j[f"hip.{s}"]).length),
        "shin": avg(lambda s: (j[f"ankle.{s}"] - j[f"knee.{s}"]).length),
        "upper_arm": avg(lambda s: (j[f"elbow.{s}"] - j[f"shoulder.{s}"]).length),
        "forearm": avg(lambda s: (j[f"wrist.{s}"] - j[f"elbow.{s}"]).length),
        "hand": avg(lambda s: (j[f"tip.{s}"] - j[f"wrist.{s}"]).length),
        "foot": extras["foot"]["length"],
        "hip_height": j["pelvis"].z - extras["floor"],
        "torso": (j["neck"] - j["pelvis"]).length,
        "neck": (j["neck"] - j["chest"]).length,
        "head": j["crown"].z - j["neck"].z,
        "shoulder_half": abs(j["shoulder.right"].x - j["shoulder.left"].x) / 2,
        "hip_half": abs(j["hip.right"].x - j["hip.left"].x) / 2,
    }


# --- the MPFB human ------------------------------------------------------------------------------

def _co_rest(body):
    """World positions of every basemesh vertex at rest with the targets mixed in (modifiers off,
    so the helper geometry is there and vertex indices are the mesh's own)."""
    live = [m for m in body.modifiers if m.show_viewport]
    for mod in live:
        mod.show_viewport = False
    try:
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        ev = body.evaluated_get(depsgraph)
        mesh = ev.to_mesh()
        co = np.empty(len(mesh.vertices) * 3)
        mesh.vertices.foreach_get("co", co)
        ev.to_mesh_clear()
    finally:
        for mod in live:
            mod.show_viewport = True
        bpy.context.view_layer.update()
    world = np.array(body.matrix_world)
    return co.reshape(-1, 3) @ world[:3, :3].T + world[:3, 3]


def bone_weights(body, bones):
    """Vertex x bone weights from the body's vertex groups (not normalised)."""
    index = {body.vertex_groups[b].index: i for i, b in enumerate(bones) if b in body.vertex_groups}
    w = np.zeros((len(body.data.vertices), len(bones)))
    for v in body.data.vertices:
        for g in v.groups:
            k = index.get(g.group)
            if k is not None:
                w[v.index, k] = g.weight
    return w


def in_body(body):
    group = body.vertex_groups["body"].index
    keep = np.zeros(len(body.data.vertices), dtype=bool)
    for v in body.data.vertices:
        for g in v.groups:
            if g.group == group and g.weight > 0.5:
                keep[v.index] = True
    return keep


def rest_joints(rig):
    world = rig.matrix_world
    return {b.name: (world @ b.head_local, world @ b.tail_local) for b in rig.data.bones}


def mpfb_features(rig, co, keep, w, bones):
    """The same numbers as `skin_features`, read off the MPFB human (in its own frame)."""
    j = rest_joints(rig)
    h = lambda n: j[P + n][0]                                         # noqa: E731
    sides = ("Left", "Right")
    avg = lambda f: sum(f(s) for s in sides) / 2                      # noqa: E731
    hip_mid = (h("LeftUpLeg") + h("RightUpLeg")) / 2
    body = co[keep]
    floor = float(body[:, 2].min())
    crown = float(body[:, 2].max())
    dom = np.argmax(w, axis=1)
    foot_ids = [bones.index(P + n) for n in ("LeftFoot", "LeftToeBase")]
    foot = co[keep & np.isin(dom, foot_ids) & (w.max(axis=1) > 0)]
    ankle = h("LeftFoot")
    reach = float(np.percentile(ankle.y - foot[:, 1], 99)) if len(foot) else 0.0
    return {
        "thigh": avg(lambda s: (h(f"{s}Leg") - h(f"{s}UpLeg")).length),
        "shin": avg(lambda s: (h(f"{s}Foot") - h(f"{s}Leg")).length),
        "upper_arm": avg(lambda s: (h(f"{s}ForeArm") - h(f"{s}Arm")).length),
        "forearm": avg(lambda s: (h(f"{s}Hand") - h(f"{s}ForeArm")).length),
        "hand": avg(lambda s: (j[P + f"{s}HandMiddle3"][1] - h(f"{s}Hand")).length),
        "foot": reach,
        "hip_height": hip_mid.z - floor,
        "torso": (h("Head") - hip_mid).length,
        "neck": (h("Head") - h("Neck")).length,
        "head": crown - h("Head").z,
        "shoulder_half": abs(h("LeftArm").x - h("RightArm").x) / 2,
        "hip_half": abs(h("LeftUpLeg").x - h("RightUpLeg").x) / 2,
    }


def build_human(macro):
    HumanService, _ = services()
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    body = HumanService.create_human(macro_detail_dict=macro)
    rig = HumanService.add_builtin_rig(body, "mixamo", import_weights=True)
    body.name = "mpfb_body"
    rig.name = "mpfb_rig"
    return body, rig


def _target_keys(body, files):
    """Load each target file (incr and decr) once as a shape key at 0; return {file: (incr, decr)}."""
    _, TargetService = services()
    keys = {}
    for name in files:
        pair = []
        for end in ("incr", "decr"):
            path = TargetService.target_full_path(f"{name}-{end}")
            if path is None:
                raise SystemExit(f"MPFB has no target {name}-{end}")
            key = TargetService.load_target(body, path, weight=0.0)
            key.slider_min, key.slider_max = 0.0, 1.0
            pair.append(key.name)
        keys[name] = tuple(pair)
    return keys


def _set_targets(body, keys, x):
    blocks = body.data.shape_keys.key_blocks
    for (name, files), value in zip(TARGETS, x):
        for f in files:
            incr, decr = keys[f]
            blocks[incr].value = max(float(value), 0.0)
            blocks[decr].value = max(-float(value), 0.0)


def _measure(body, rig, keys, x, bones, keep, w):
    HumanService, _ = services()
    _set_targets(body, keys, x)
    HumanService.refit(body)
    bpy.context.view_layer.update()
    return mpfb_features(rig, _co_rest(body), keep, w, bones)


def fit_targets(body, rig, want, log=print):
    """Solve the target variables (each -1..1) and one uniform scale against the skin's features.

    The features' response to each variable is measured, both directions, by setting it to +-1 and
    refitting the rig (MPFB's targets move joints by moving the joint helper cubes the rig is fitted
    to). Then a bounded least squares on the RELATIVE error of every feature, the scale solved in
    closed form in between. Returns (x, scale, stats).
    """
    bones = [b.name for b in rig.data.bones]
    files = [f for _, fs in TARGETS for f in fs]
    keys = _target_keys(body, files)
    keep, w = in_body(body), bone_weights(body, bones)
    n = len(TARGETS)
    zero = np.zeros(n)
    f0 = _measure(body, rig, keys, zero, bones, keep, w)
    names = list(FEATURES)
    base = np.array([f0[k] for k in names])
    goal = np.array([want[k] for k in names])
    up, down = np.zeros((len(names), n)), np.zeros((len(names), n))
    for i in range(n):
        e = zero.copy()
        e[i] = 1.0
        got = _measure(body, rig, keys, e, bones, keep, w)
        up[:, i] = np.array([got[k] for k in names]) - base
        e[i] = -1.0
        got = _measure(body, rig, keys, e, bones, keep, w)
        down[:, i] = np.array([got[k] for k in names]) - base

    def model(x):
        return base + up @ np.maximum(x, 0) + down @ np.maximum(-x, 0)

    x = np.zeros(n)
    scale = float((base * goal / goal ** 2).sum() / (base ** 2 / goal ** 2).sum())
    for _ in range(40):
        for _ in range(200):
            f = model(x)
            r = (scale * f - goal) / goal
            jac = np.where(x >= 0, up, down) * (scale / goal)[:, None]
            grad = jac.T @ r + TARGET_RIDGE * x
            step = 1.0 / (np.linalg.norm(jac, 2) ** 2 + TARGET_RIDGE)
            x = np.clip(x - step * grad, -1.0, 1.0)
        f = model(x)
        scale = float((f * goal / goal ** 2).sum() / (f ** 2 / goal ** 2).sum())
    got = _measure(body, rig, keys, x, bones, keep, w)
    after = np.array([got[k] for k in names])
    stats = {"variables": {name: round(float(v), 3) for (name, _), v in zip(TARGETS, x)},
             "scale": scale,
             "features_mm": {k: {"skin": round(goal[i] * 1000, 1),
                                 "mpfb_before": round(scale * base[i] * 1000, 1),
                                 "mpfb_after": round(scale * after[i] * 1000, 1)}
                             for i, k in enumerate(names)},
             "rel_error_before": float(np.sqrt(np.mean(((scale * base - goal) / goal) ** 2))),
             "rel_error_after": float(np.sqrt(np.mean(((scale * after - goal) / goal) ** 2))),
             "model_vs_measured_mm": float(np.abs(scale * (model(x) - after)).max() * 1000)}
    return x, scale, stats


# --- the warp to the skin ------------------------------------------------------------------------

def _rotdiff(a, b):
    return Vector(a).normalized().rotation_difference(Vector(b).normalized()).to_matrix()


class BoneMap:
    """x -> head' + L (x - origin): an affine map of MPFB space (its frame) into the house frame."""

    def __init__(self, origin, target, linear, rotation):
        self.origin, self.target = Vector(origin), Vector(target)
        self.linear, self.rotation = Matrix(linear), Matrix(rotation)

    def __call__(self, x):
        return self.target + self.linear @ (Vector(x) - self.origin)

    def matrix(self):
        """As a 4x4 acting on MPFB world positions."""
        m = self.linear.to_4x4()
        m.translation = self.target - self.linear @ self.origin
        return m


def along_map(head, tail, head2, tail2, radial=1.0):
    """The map that carries a bone (head -> tail, MPFB frame) onto (head2 -> tail2, house frame):
    the half turn, the shortest swing onto the new direction, a stretch along the bone to the new
    length and `radial` across it."""
    d = (Vector(tail) - Vector(head)).normalized()
    d2 = Vector(tail2) - Vector(head2)
    q = _rotdiff(TURN @ d, d2) @ TURN
    stretch = d2.length / max((Vector(tail) - Vector(head)).length, 1e-9)
    dd = np.outer(d, d)
    s = Matrix((stretch * dd + radial * (np.eye(3) - dd)).tolist())
    return BoneMap(head, head2, q @ s, q)


def _girth(points, head, tail):
    """Median distance of points from the segment head-tail, over those whose projection falls
    on its middle 15-85%."""
    a, b = np.array(head), np.array(tail)
    ab = b - a
    t = ((points - a) @ ab) / max(ab @ ab, 1e-12)
    pick = (t > 0.15) & (t < 0.85)
    if pick.sum() < 8:
        return None
    foot = a + t[pick, None] * ab
    return float(np.median(np.linalg.norm(points[pick] - foot, axis=1)))


def _section_half(points, z, band):
    s = points[np.abs(points[:, 2] - z) <= band]
    if len(s) < 12:
        return None
    x0, x1 = np.percentile(s[:, 0], [2, 98])
    y0, y1 = np.percentile(s[:, 1], [2, 98])
    return (x1 - x0) / 2, (y1 - y0) / 2, (y0 + y1) / 2


def warp_to_skin(body, rig, skin_co, j, extras, log=print):
    """Carry the MPFB human onto the skin's joints and girths; rebuild the bones there.

    Returns a dict of what was done and how far each joint moved."""
    TargetService = services()[1]
    height = extras["height"]
    bones = [b.name for b in rig.data.bones]
    co = _co_rest(body)
    keep = in_body(body)
    w = bone_weights(body, bones)
    mj = rest_joints(rig)
    h = lambda n: mj[P + n][0]                                       # noqa: E731
    t = lambda n: mj[P + n][1]                                       # noqa: E731
    dom = np.argmax(w, axis=1)
    owned = lambda names: keep & np.isin(dom, [bones.index(P + n) for n in names]) & (w.max(1) > 0)  # noqa: E731

    # The trunk: one affine for hips and spine, so nothing inside it is blended between maps.
    hip_mid = (h("LeftUpLeg") + h("RightUpLeg")) / 2
    rise = h("Head") - hip_mid
    rise2 = j["neck"] - j["pelvis"]
    trunk = co[owned(("Hips", "Spine", "Spine1", "Spine2"))]
    ratios_x, ratios_y = [], []
    lines = extras["lines"]
    for key in ("pelvis", "waist", "chest"):
        sk = lines[key]
        frac = (sk["z"] - j["pelvis"].z) / rise2.z
        mine = _section_half(trunk, hip_mid.z + frac * rise.z, 0.006 * rise.length / 0.35)
        if mine:
            ratios_x.append(sk["half"] / mine[0])
            ratios_y.append(sk["depth"] / mine[1])
    sx, sy = float(np.mean(ratios_x)), float(np.mean(ratios_y))
    sz = rise2.length / rise.length
    lean = _rotdiff(TURN @ Vector((0.0, rise.y, rise.z)), Vector((0.0, rise2.y, rise2.z)))
    trunk_rot = lean @ TURN
    trunk_map = BoneMap(hip_mid, j["pelvis"], trunk_rot @ Matrix.Diagonal((sx, sy, sz)), trunk_rot)

    # The head: its box onto the skin's head's box, axis by axis. The skin's head is everything
    # above its neck joint (the T-pose arms are below it); MPFB's is what its Head bone owns above
    # the same joint. Percentiles, so an ear tip or a hair spike does not set the size. (A first
    # version scaled by one mid-height slice and stretched the cage's skull back 18 cm past the
    # skin's: the slice's depth caught the boy's hair.)
    head_pts = co[owned(("Head",))]
    head_pts = head_pts[head_pts[:, 2] >= h("Head").z]
    skin_head = skin_co[skin_co[:, 2] >= j["neck"].z]
    lo1, hi1 = np.percentile(head_pts, 1, axis=0), np.percentile(head_pts, 99, axis=0)
    lo2, hi2 = np.percentile(skin_head, 1, axis=0), np.percentile(skin_head, 99, axis=0)
    lo1[2], hi1[2] = h("Head").z, head_pts[:, 2].max()
    lo2[2], hi2[2] = j["neck"].z, j["crown"].z
    hx, hy, hz = ((hi2 - lo2) / (hi1 - lo1)).tolist()
    head_map = BoneMap(Vector((lo1 + hi1) / 2), Vector((lo2 + hi2) / 2),
                       TURN @ Matrix.Diagonal((hx, hy, hz)), TURN)

    # Limb girths: the skin's measured radii (from the same slices `proportions_from` read) over
    # the MPFB limb's own, bone by bone.
    def skin_radius(kind, side, a, b):
        rows = np.array(extras["arm_radius" if kind == "arm" else "leg_radius"][side])
        key = np.array([a, b])
        if kind == "arm":                               # rows are (outward distance s, radius)
            s = abs(key - extras["centre"][0])
            return float(np.interp(s.mean(), rows[:, 0], rows[:, 1]))
        return float(np.interp(key.mean(), rows[:, 0], rows[:, 1]))   # (z, radius)

    maps, desired, radial = {}, {}, {}
    for side, ms, _ in SIDES:
        for mname, kind, a, b in (("Arm", "arm", "shoulder", "elbow"),
                                  ("ForeArm", "arm", "elbow", "wrist"),
                                  ("UpLeg", "leg", "hip", "knee"),
                                  ("Leg", "leg", "knee", "ankle")):
            bone = ms + mname
            mine = _girth(co[owned((bone,))], h(bone), t(bone))
            if kind == "arm":
                theirs = skin_radius("arm", side, j[f"{a}.{side}"].x, j[f"{b}.{side}"].x)
            else:
                theirs = skin_radius("leg", side, j[f"{a}.{side}"].z, j[f"{b}.{side}"].z)
            radial[bone] = theirs / mine if mine else 1.0
            if mname == "UpLeg" and mine:
                # The skin's thigh slices are through its SHORTS, a tube nearly as wide as the
                # hips; a cage thigh that fat crosses the other one at the crotch (60 crossing
                # face pairs in the first build) and the shorts' crotch then bound to whichever
                # thigh was nearer and tore. Capped so the two thighs meet, not overlap.
                radial[bone] = min(radial[bone], THIGH_CAP * extras["lines"]["hip_half"] / mine)
        sh, el, wr, tip = (j[f"{n}.{side}"] for n in ("shoulder", "elbow", "wrist", "tip"))
        hp, kn, an = (j[f"{n}.{side}"] for n in ("hip", "knee", "ankle"))
        maps[ms + "Arm"] = along_map(h(ms + "Arm"), h(ms + "ForeArm"), sh, el, radial[ms + "Arm"])
        maps[ms + "ForeArm"] = along_map(h(ms + "ForeArm"), h(ms + "Hand"), el, wr,
                                         radial[ms + "ForeArm"])
        hand_tip = mj[P + ms + "HandMiddle3"][1]
        hand_r = _girth(co[owned((ms + "Hand",))], h(ms + "Hand"), hand_tip)
        skin_wrist_r = extras["props"].wrist_r_f * height
        radial[ms + "Hand"] = (skin_wrist_r / hand_r) if hand_r else 1.0
        maps[ms + "Hand"] = along_map(h(ms + "Hand"), hand_tip, wr, tip, radial[ms + "Hand"])
        maps[ms + "UpLeg"] = along_map(h(ms + "UpLeg"), h(ms + "Leg"), hp, kn, radial[ms + "UpLeg"])
        maps[ms + "Leg"] = along_map(h(ms + "Leg"), h(ms + "Foot"), kn, an, radial[ms + "Leg"])
        # the foot: its own three axes, from the ankle — length to the skin's toes, width to its
        # foot, height to its ankle
        fpts = co[owned((ms + "Foot", ms + "ToeBase"))]
        fa = h(ms + "Foot")
        floor = float(co[keep][:, 2].min())
        reach = float(np.percentile(fa.y - fpts[:, 1], 99))
        half = float(np.percentile(np.abs(fpts[:, 0] - fpts[:, 0].mean()), 98))
        fy = extras["foot"]["length"] / reach
        fx = extras["foot"]["half"] / half
        fz = (an.z - extras["floor"]) / (fa.z - floor)
        maps[ms + "Foot"] = BoneMap(fa, an, TURN @ Matrix.Diagonal((fx, fy, fz)), TURN)
        maps[ms + "ToeBase"] = maps[ms + "Foot"]
        # the collarbone: from where the trunk map puts its root to the skin's shoulder joint
        maps[ms + "Shoulder"] = along_map(h(ms + "Shoulder"), h(ms + "Arm"),
                                          trunk_map(h(ms + "Shoulder")), sh, 0.5 * (sx + sy))
        for finger in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
            for k in (1, 2, 3):
                maps[f"{ms}Hand{finger}{k}"] = maps[ms + "Hand"]
    for name in ("Hips", "Spine", "Spine1", "Spine2"):
        maps[name] = trunk_map
    neck_r = _girth(co[owned(("Neck",))], h("Neck"), h("Head"))
    radial["Neck"] = (lines["neck"]["half"] / neck_r) if neck_r else 0.5 * (sx + sy)
    maps["Neck"] = along_map(h("Neck"), h("Head"), trunk_map(h("Neck")), j["neck"], radial["Neck"])
    maps["Head"] = head_map
    missing = [b for b in bones if b[len(P):] not in maps]
    if missing:
        raise SystemExit(f"no warp map for {missing}")

    # Where each bone goes: measured joints where the skin has them, mapped MPFB joints elsewhere.
    for b in bones:
        n = b[len(P):]
        desired[b] = [maps[n](mj[b][0]), maps[n](mj[b][1])]
    for side, ms, _ in SIDES:
        sh, el, wr, tip = (j[f"{n}.{side}"] for n in ("shoulder", "elbow", "wrist", "tip"))
        hp, kn, an = (j[f"{n}.{side}"] for n in ("hip", "knee", "ankle"))
        desired[P + ms + "Shoulder"][1] = sh
        desired[P + ms + "Arm"] = [sh, el]
        desired[P + ms + "ForeArm"] = [el, wr]
        desired[P + ms + "Hand"][0] = wr
        desired[P + ms + "UpLeg"] = [hp, kn]
        desired[P + ms + "Leg"] = [kn, an]
        desired[P + ms + "Foot"][0] = an
    # The middle finger carries the mitten's front half (`skin_wrap._rigid_parts`): its three
    # joints go ON the skin's own wrist-to-tip line, where the mitten's weights bend. Mapped from
    # MPFB they sat 43 degrees above that line, outside the hand, and curling them folded the
    # fingers into a crumpled sheet.
    for side, ms, _ in SIDES:
        wr, tip = j[f"wrist.{side}"], j[f"tip.{side}"]
        stops = [a for a, _ in skin_wrap.MITTEN_JOINTS] + [1.0]
        for k in range(3):
            desired[P + f"{ms}HandMiddle{k + 1}"] = [wr + (tip - wr) * stops[k],
                                                     wr + (tip - wr) * stops[k + 1]]
        desired[P + f"{ms}Hand"][1] = wr + (tip - wr) * stops[0]
    desired[P + "Neck"][1] = j["neck"]
    desired[P + "Head"][0] = j["neck"]

    # Carry the body: baked targets, then every vertex by MPFB's own weights through the maps.
    TargetService.bake_targets(body)
    world_inv = np.array(body.matrix_world.inverted())
    total = w.sum(axis=1, keepdims=True)
    loose = (total[:, 0] <= 1e-6)
    wn = w / np.maximum(total, 1e-12)
    mats = np.array([np.array(maps[b[len(P):]].matrix()) for b in bones])
    blended = np.einsum("vb,bij->vij", wn, mats)
    hom = np.c_[co, np.ones(len(co))]
    new = np.einsum("vij,vj->vi", blended, hom)[:, :3]
    # The head's big box map applies to the head; below the head joint (MPFB's jaw and chin hang
    # under it) it hands over to the neck's map, so the cartoon-sized skull does not flare the
    # jaw out through the shoulders (31 head/torso crossings in the first build).
    hk, nk = bones.index(P + "Head"), bones.index(P + "Neck")
    fade = max((h("Head") - h("Neck")).length, 1e-6)
    below = np.clip((h("Head").z - co[:, 2]) / fade, 0.0, 1.0)
    below = below * below * (3 - 2 * below)
    moved = wn[:, hk] * below
    if moved.any():
        head_at = (np.array(head_map.matrix()) @ hom.T).T[:, :3]
        neck_at = (np.array(maps["Neck"].matrix()) @ hom.T).T[:, :3]
        new += moved[:, None] * (neck_at - head_at)
    if loose.any():
        # helper geometry with no weights: carried by the trunk (it is deleted next anyway)
        new[loose] = (np.array(trunk_map.matrix()) @ hom[loose].T).T[:, :3]
    local = new @ world_inv[:3, :3].T + world_inv[:3, 3]
    body.data.vertices.foreach_set("co", local.ravel())
    body.data.update()

    # Rebuild the bones at the desired joints, each turned as its map turned it (so its roll
    # follows), then swung onto its new direction.
    bpy.context.view_layer.objects.active = rig
    for obj in bpy.context.view_layer.objects:
        obj.select_set(obj is rig)
    rig_inv = rig.matrix_world.inverted()
    bpy.ops.object.mode_set(mode="EDIT")
    edit = rig.data.edit_bones
    connected = {e.name: e.use_connect for e in edit}
    for e in edit:
        e.use_connect = False
    for b in bones:
        e = edit[b]
        hd, tl = (rig_inv @ desired[b][0]), (rig_inv @ desired[b][1])
        frame = maps[b[len(P):]].rotation @ e.matrix.to_3x3()
        now = frame @ Vector((0.0, 1.0, 0.0))
        frame = _rotdiff(now, tl - hd) @ frame
        m = frame.to_4x4()
        m.translation = hd
        e.matrix = m
        e.length = max((tl - hd).length, 1e-4)
    for e in edit:
        e.use_connect = connected[e.name]
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.context.view_layer.update()

    after = rest_joints(rig)
    miss = max(max((after[b][0] - desired[b][0]).length, (after[b][1] - desired[b][1]).length)
               for b in bones)
    return {"trunk_scale_xyz": [sx, sy, sz], "lean_deg": math.degrees(math.acos(
                max(-1.0, min(1.0, (TURN @ rise).normalized().dot(rise2.normalized()))))),
            "head_scale_xyz": [hx, hy, hz], "radial": {k: round(v, 3) for k, v in radial.items()},
            "bones_placed_within_mm": miss * 1000}


SOCKET = (0.025, 0.055)     # x height medial of the shoulder joint: the cage's upper-arm weight fades out


def trim_socket(body, rig, height):
    """Hand the cage's upper-arm weight MEDIAL of the shoulder joint to the collarbone.

    MPFB's `rig.mixamo` weights let the upper arm reach across the top of the chest to the base of
    the neck (on the skin the bake read 8-16% upper arm on the vest's neckline, 7 cm in from the
    shoulder), so every swing of the arm — the grip on a bar is one — dragged a crease through the
    neckline. Done on the CAGE, before the bind, so Surface Deform and the bake agree on it: done on
    the skin's weights alone, the bake strayed 40 mm from Surface Deform at the back of the shoulder
    and its guard refused it. Faded between SOCKET[0] and SOCKET[1] x height inside the joint,
    measured along the upper arm."""
    world = rig.matrix_world
    moved_max, touched = 0.0, 0
    co = skin_wrap._co(body.data) @ np.array(body.matrix_world)[:3, :3].T + np.array(body.matrix_world)[:3, 3]
    for side in ("Left", "Right"):
        arm, clav = rig.data.bones[P + side + "Arm"], rig.data.bones[P + side + "Shoulder"]
        joint = np.array(world @ arm.head_local)
        axis = np.array((world.to_3x3() @ (arm.tail_local - arm.head_local)).normalized())
        inside = -((co - joint) @ axis)
        t = np.clip((inside - SOCKET[0] * height) / ((SOCKET[1] - SOCKET[0]) * height), 0.0, 1.0)
        fade = t * t * (3 - 2 * t)
        ga, gc = body.vertex_groups[arm.name], body.vertex_groups[clav.name]
        for v in np.flatnonzero(fade > 0):
            try:
                wa = ga.weight(int(v))
            except RuntimeError:
                continue
            take = wa * fade[v]
            if take <= 1e-6:
                continue
            try:
                wc = gc.weight(int(v))
            except RuntimeError:
                wc = 0.0
            ga.add([int(v)], wa - take, "REPLACE")
            gc.add([int(v)], wc + take, "REPLACE")
            moved_max = max(moved_max, take)
            touched += 1
    return {"vertices": touched, "weight_moved_max": round(moved_max, 3)}


RELAX_STEPS = 10            # Laplacian passes over the warped cage's joint blends
RELAX_FACTOR = 0.5
RELAX_BONES = ("Shoulder", "Arm", "Neck", "UpLeg")   # the joints the warp moved most


def _quads_damage(body, height):
    """`skin_wrap.damage` totals on the cage's own mesh as it stands (crossing, overlapping, folds)."""
    co = skin_wrap._co(body.data)
    quads = skin_wrap._quads(body.data)
    found = skin_wrap.damage(co, quads, height)
    return {k: found[k] for k in ("crossing", "overlapping", "folds")}


def relax_joints(body, rig, height, steps=RELAX_STEPS, factor=RELAX_FACTOR):
    """Smooth the warped cage where it was swung furthest: the shoulders, armpits and neck.

    The warp raises MPFB's A-pose arms ~50 degrees to the skin's T-pose by linear blend skinning,
    and LBS folds the armpit and crushes the top of the shoulder when it does (task 6 saw the same
    on `character.py`'s cage: Surface Deform then binds the skin to the fold). Laplacian passes
    over the vertices that blend the shoulder, arm and neck bones with anything else — and two
    rings round them — undo the fold without moving any bone or any vertex outside those zones.
    Damage (crossing faces, folds) is counted before and after; the result is kept only if it
    did not get worse."""
    bones = [b.name for b in rig.data.bones]
    w = bone_weights(body, bones)
    w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
    pick_b = [i for i, b in enumerate(bones) if any(b.endswith(n) for n in RELAX_BONES)]
    share = w[:, pick_b].sum(axis=1)
    zone = (share > 0.05) & (share < 0.95)
    edges = skin_wrap._edges(body.data)
    for _ in range(2):
        grown = zone.copy()
        grown[edges[zone[edges[:, 0]], 1]] = True
        grown[edges[zone[edges[:, 1]], 0]] = True
        zone = grown
    before = _quads_damage(body, height)
    co0 = skin_wrap._co(body.data)
    co = co0.copy()
    n = len(co)
    degree = np.bincount(edges.ravel(), minlength=n).astype(float)[:, None]
    for _ in range(steps):
        total = np.zeros_like(co)
        np.add.at(total, edges[:, 0], co[edges[:, 1]])
        np.add.at(total, edges[:, 1], co[edges[:, 0]])
        mean = total / np.maximum(degree, 1.0)
        co[zone] = (1 - factor) * co[zone] + factor * mean[zone]
    body.data.vertices.foreach_set("co", co.ravel())
    body.data.update()
    after = _quads_damage(body, height)
    kept = (after["crossing"] + after["folds"]) <= (before["crossing"] + before["folds"])
    if not kept:
        body.data.vertices.foreach_set("co", co0.ravel())
        body.data.update()
    moved = np.linalg.norm(co - co0, axis=1)
    return {"zone_vertices": int(zone.sum()), "damage_before": before, "damage_after": after,
            "kept": bool(kept), "moved_mm_max": float(moved.max() * 1000),
            "moved_mm_mean_in_zone": float(moved[zone].mean() * 1000)}


def strip_helpers(body):
    """Delete the helper geometry (joint cubes, clothes helpers) and the mask that hid it: the cage
    is the visible body and nothing else."""
    keep = in_body(body)
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(body.data)
    bm.verts.ensure_lookup_table()
    bmesh.ops.delete(bm, geom=[bm.verts[i] for i in np.flatnonzero(~keep)], context="VERTS")
    bm.to_mesh(body.data)
    bm.free()
    body.data.update()
    for mod in [m for m in body.modifiers if m.type == "MASK"]:
        body.modifiers.remove(mod)
    for mod in body.modifiers:
        if mod.type == "ARMATURE":
            mod.use_deform_preserve_volume = False      # linear, as the bake and every player do
            mod.use_vertex_groups = True
            mod.use_bone_envelopes = False
    return int((~keep).sum())


def cage_gap(cage, skin_co):
    """Distance from every skin vertex to the cage surface (the cage evaluated now), in metres."""
    co, _ = skin_wrap.evaluated(cage)
    mesh = cage.data
    mesh.calc_loop_triangles()
    tris = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int64)
    mesh.loop_triangles.foreach_get("vertices", tris)
    tree = BVHTree.FromPolygons([Vector(v) for v in co], tris.reshape(-1, 3).tolist(),
                                all_triangles=True)
    return np.array([tree.find_nearest(Vector(p))[3] for p in skin_co])


def measure_hands(skin_co, edges, j, height, log=print):
    """Both hands measured finger by finger (`hand_rig.measure`): which skin vertices are which
    finger, and where each finger's knuckle, middle and end joints are. None (and task 12's mitten
    is used) if a hand does not read as four fingers and a thumb."""
    group = skin_wrap._coincident_groups(skin_co, skin_wrap.COINCIDENT * height)
    models = {}
    for side, _, _ in SIDES:
        try:
            model, labels = hand_rig.measure(skin_co, edges, group, j[f"wrist.{side}"], j[f"tip.{side}"])
        except ValueError as err:
            log(f"hands: the {side} hand could not be read finger by finger ({err}); keeping the mitten")
            return None
        models[side] = model
        log(f"hands: {side}: " + ", ".join(
            f"{name} {f['free_length'] * 1000:.0f} mm free, {f['thickness'] * 1000:.0f} thick, "
            f"{int((labels == fid).sum())} vertices" for fid, (name, f) in enumerate(model["fingers"].items())))
    return models


def hand_shape(skin_co, j, extras, models=None):
    """What the grip needs to know about the skin's hands and shoulders, measured on the T-posed
    skin: the outer shoulder half-width (the shoulder joint plus the upper arm's radius there — the
    "shoulder width" a coach measures a grip against, deltoid to deltoid), and each mitten's length
    (wrist to tip), half thickness, and how far its PALM surface lies below the wrist's line (the
    T-posed palms face down; the mitten is not centred on the bone — palm 30 mm below, back 18 mm
    above on the boy — and a grip placed by the half thickness pushed the palm 7 mm into the bar)."""
    cx = extras["centre"][0]
    rows = np.array(extras["arm_radius"]["right"])
    s_sh = abs(j["shoulder.right"].x - cx)
    out = {"outer_shoulder_half": float(s_sh + np.interp(s_sh + 0.01, rows[:, 0], rows[:, 1]))}
    for side, _, sign in SIDES:
        wrist = j[f"wrist.{side}"]
        pts = skin_co[sign * (skin_co[:, 0] - cx) > sign * (wrist.x - cx) + 0.01]
        lo, hi = np.percentile(pts[:, 2], [5, 95])
        out[side] = {"length": float((j[f"tip.{side}"] - wrist).length),
                     "half_thickness": float((hi - lo) / 2),
                     "palm_depth": float(wrist.z - lo)}
        if models is not None:
            out[side]["model"] = models[side]
    return out


def fit_cage_to(skin_path, scratch, ages=AGES, log=print):
    """Build the MPFB cage for a skin. Returns (cage, rig, report).

    Clears the scene (MPFB's `create_human` is run fresh for each age tried), so call it first,
    as `skin_wrap.fit_cage_to` is.
    """
    t0 = time.time()
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    meas = skin_wrap.measure_skin(skin_path, scratch)
    j, extras = skin_joints(meas)
    want = skin_features(j, extras)
    log(f"[{time.time() - t0:5.1f}s] skin measured: height {extras['height']:.4f} | "
        + ", ".join(f"{k} {v * 1000:.0f}" for k, v in want.items()))
    tried = {}
    for age in ages:
        macro = dict(MACRO, age=age)
        body, rig = build_human(macro)
        x, scale, stats = fit_targets(body, rig, want, log=log)
        tried[age] = stats
        log(f"[{time.time() - t0:5.1f}s] age {age}: rel error {stats['rel_error_before']:.3f} -> "
            f"{stats['rel_error_after']:.3f}, scale {scale:.3f}, targets {stats['variables']}")
    best = min(tried, key=lambda a: tried[a]["rel_error_after"])
    body, rig = build_human(dict(MACRO, age=best))
    x, scale, stats = fit_targets(body, rig, want, log=log)
    # joints after targets, set on the skin's hips (turned into the house frame) for the report
    mj = rest_joints(rig)
    hip_mid = (mj[P + "LeftUpLeg"][0] + mj[P + "RightUpLeg"][0]) / 2
    # legs and trunk only: MPFB's arms are in its A-pose and the skin's in a T-pose, so the arm
    # is judged by its segment lengths (`features_mm`), not by where its joints are
    pairs = {"hip": "UpLeg", "knee": "Leg", "ankle": "Foot"}
    residual = {}
    for house, mix in pairs.items():
        for side, ms, _ in SIDES:
            p = j["pelvis"] + TURN @ ((mj[P + ms + mix][0] - hip_mid) * scale)
            residual[f"{house}.{side}"] = round((p - j[f"{house}.{side}"]).length * 1000, 1)
    residual["neck (Head bone)"] = round(((j["pelvis"] + TURN @ ((mj[P + "Head"][0] - hip_mid) * scale))
                                          - j["neck"]).length * 1000, 1)
    stats["joint_error_after_targets_mm"] = residual
    log(f"[{time.time() - t0:5.1f}s] after targets (age {best}): joint error mm {residual}")

    skin = skin_wrap.load_skin(skin_path, name="skin_probe")
    skin_co = skin_wrap._co(skin.data)
    skin_edges = skin_wrap._edges(skin.data)
    bpy.data.objects.remove(skin, do_unlink=True)
    warp = warp_to_skin(body, rig, skin_co, j, extras, log=log)
    removed = strip_helpers(body)
    relaxed = relax_joints(body, rig, extras["height"])
    relaxed["socket"] = trim_socket(body, rig, extras["height"])
    # The cartoon's own fingers, measured, and a bone chain fitted inside each (task 13).
    hand_models = measure_hands(skin_co, skin_edges, j, extras["height"], log=log)
    if hand_models is not None:
        hand_rig.fit_finger_bones(rig, hand_models)
    log(f"[{time.time() - t0:5.1f}s] relaxed the shoulder/neck blends: {relaxed}")
    log(f"[{time.time() - t0:5.1f}s] warped onto the skin's joints: {warp}; helpers removed "
        f"{removed} verts")
    rig.name, body.name = "rig", "cage"
    body.hide_render = True
    rig["rig_profile"] = PROFILE
    rig["skin_fit"] = json.dumps({
        "skin": str(skin_path), "rest": "skin", "pose": {}, "origin": [extras["centre"][0],
                                                                      extras["centre"][1],
                                                                      extras["floor"]],
        "measured": dict(extras["place"]["measured"], height=extras["height"]),
        "shoulder_joint": {s: list(j[f"shoulder.{s}"]) for s in ("left", "right")},
        "joints": {k: list(v) for k, v in j.items()},
        "notes": extras["place"]["notes"], "silhouette": meas["silhouette"],
        "cage": "mpfb", "age": best, "neck_radius": extras["lines"]["neck"]["half"],
        "hands": hand_shape(skin_co, j, extras, hand_models),
    })
    gap = cage_gap(body, skin_co)
    report = {"age": best, "tried": {str(a): s["rel_error_after"] for a, s in tried.items()},
              "targets": stats, "warp": warp, "helpers_removed": removed, "relax": relaxed,
              "cage_vertices": len(body.data.vertices), "cage_faces": len(body.data.polygons),
              "skin_to_cage_mm": {"mean": float(gap.mean() * 1000),
                                  "p95": float(np.percentile(gap, 95) * 1000),
                                  "max": float(gap.max() * 1000)},
              "seconds": time.time() - t0}
    rig["mpfb_cage"] = json.dumps(report, default=float)
    log(f"[{time.time() - t0:5.1f}s] cage ready: skin-to-cage gap mean "
        f"{report['skin_to_cage_mm']['mean']:.1f} / p95 {report['skin_to_cage_mm']['p95']:.1f} mm")
    return body, rig, report


# --- wearing the skin ----------------------------------------------------------------------------

# The bake's guard on how far the baked skin may stray from Surface Deform at the arm swing, as a
# fraction of height. The house cage is held to `skin_wrap.BAKE_TOLERANCE` (0.03). On this cage 8
# vertices hidden INSIDE the armpit stray 34 mm at arms-down (the rest: mean 0.27, p99 5.2 mm);
# renders of that pose, baked beside Surface Deform, are indistinguishable (task 12 report). Still
# a guard: a failed or wrong bake strays by hundreds of millimetres (131 mm caught a real bug here).
BAKE_TOLERANCE = 0.04


def wear(skin_path, scratch, ages=AGES, log=print):
    """The MPFB cage fitted to a skin, the skin pulled on and baked to the rig's bones.

    The same steps `animate.wear_skin` takes on `character.py`'s cage, in the same order:
    region fit (the cage pulled onto the skin per body region), Surface Deform bind, the skin
    parented to the rig, the bake. Returns (skin, rig, cage, report); `skin` is an ordinary
    Armature-skinned mesh on `rig` and `cage` is hidden.
    """
    remaining, refused = list(ages), {}
    while True:
        # The best-fitting age first; if `region_fit` cannot pull that cage onto the skin without
        # damaging it, the next. (A child cage fitted the boy's re-measured arm best and region_fit
        # refused it; the adult one it accepts.) `fit_cage_to` clears the scene each time.
        cage, rig, report = fit_cage_to(skin_path, scratch, ages=tuple(remaining), log=log)
        skin = skin_wrap.load_skin(skin_path)
        try:
            skin_wrap.region_fit(cage, skin)
            break
        except SystemExit as err:
            if len(remaining) == 1:
                raise
            refused[str(report["age"])] = str(err)
            log(f"wear: region_fit refused the age {report['age']} cage ({err}); trying the next age")
            remaining.remove(report["age"])
    report["region_fit"] = json.loads(cage["region_fit"])
    report["ages_refused_by_region_fit"] = refused
    mod = skin_wrap.wrap(cage, rig, skin)
    if not mod.is_bound:
        raise SystemExit("Surface Deform did not bind the skin to the MPFB cage")
    keep = skin.matrix_world.copy()
    skin.parent = rig
    skin.matrix_parent_inverse = rig.matrix_world.inverted()
    skin.matrix_world = keep
    skin_wrap.bake_to_skin(cage, rig, skin, tolerance=BAKE_TOLERANCE)
    report["bake"] = json.loads(skin["skin_bake"])
    cage.hide_render = True
    cage.hide_viewport = True
    if skin_wrap._hand_models(rig) is not None:
        report["grip"] = solve_grips(rig, skin, log=log)
    return skin, rig, cage, report


def solve_grips(rig, skin, log=print):
    """Each hand's closed grip on the house barbell, solved once on the baked skin at the bind
    (`hand_rig.solve_grip`): where the bar crosses the hand and how each finger and the thumb
    fold onto it. Kept on the rig as `rig["hand_grip"]` for `match_arms`; returns the summary.
    Then the hand's weights are smoothed where that grip turns its skin over (task 17,
    `hand_rig.relax_grip_folds`; the solved angles are kept) and written back to the skin."""
    import equipment
    t0 = time.time()
    fit = json.loads(rig["skin_fit"])
    height = fit["measured"]["height"]
    bar_r = (equipment.BARBELL_BAR_DIAMETER / 2) * height / 1.75
    rest = _rest_points(rig, skin)
    nv = len(rest)
    group = skin_wrap._coincident_groups(rest, skin_wrap.COINCIDENT * height)
    mesh_edges = skin_wrap._edges(skin.data)
    edges = hand_rig.welded_edges(mesh_edges, group)
    skin.data.calc_loop_triangles()
    tris = np.empty(len(skin.data.loop_triangles) * 3, dtype=np.int64)
    skin.data.loop_triangles.foreach_get("vertices", tris)
    tris = tris.reshape(-1, 3)
    grips, summary = {}, {}
    for side, ms, _ in SIDES:
        cols = hand_rig.hand_bones(ms)
        gi = {skin.vertex_groups[c].index: k for k, c in enumerate(cols) if c in skin.vertex_groups}
        W = np.zeros((nv, 16))
        for v in skin.data.vertices:
            for g in v.groups:
                if g.group in gi:
                    W[v.index, gi[g.group]] = g.weight
        idx = np.flatnonzero(W.sum(axis=1) > 1e-4)
        Wh = W[idx]
        Wh[:, 0] += np.clip(1.0 - Wh.sum(axis=1), 0.0, None)   # the forearm's share: still, in the hand's frame
        grip = hand_rig.solve_grip(fit["hands"][side]["model"], rest[idx], Wh,
                                   hand_rig.local_edges(edges, idx, nv), bar_r, log=log)
        grips[side] = grip
        summary[side] = {k: grip[k] for k in ("a_mm", "o_mm", "tilt_deg", "knuckle_vs_bar_deg", "fingers", "thumb")}
        log(f"grip {side}: bar {grip['a_mm']:.1f} mm along the hand, {grip['o_mm']:.1f} mm off its axis; "
            + ", ".join(f"{k} {v['coverage_deg']:.0f} deg" for k, v in grip["fingers"].items())
            + f"; thumb {grip['thumb']['coverage_deg']:.0f} deg, {grip['thumb']['separation_deg']:.0f} deg "
            f"from the fingertips")
        summary[side]["skin"] = _relax_grip_skin(skin, fit, side, cols, idx, W, Wh, rest, mesh_edges, group,
                                                 edges, tris, grip, bar_r, log)
    rig["hand_grip"] = json.dumps(grips)
    summary["seconds"] = round(time.time() - t0, 1)
    return summary


def _relax_grip_skin(skin, fit, side, cols, idx, W, Wh, rest, mesh_edges, group, edges, tris, grip, bar_r, log):
    """Task 17: the hand's weights smoothed where its solved grip turns the skin over
    (`hand_rig.relax_grip_folds`) and written back to the skin's groups. Only the palm's skin wholly
    on the hand's bones moves (not the fingers' or thumb's own pieces, not the wrist's blend with the
    forearm), and only if the grip re-measured on it (`hand_rig.grip_check`) goes no deeper into the
    bar and the thumb stays clear of the hand; otherwise the designed weights are kept."""
    nv = len(rest)
    joints = fit["joints"]
    _, lab = hand_rig.measure(rest, mesh_edges, group, joints[f"wrist.{side}"], joints[f"tip.{side}"])
    movable = (W[idx].sum(axis=1) >= 1.0 - 1e-6) & (lab[idx] < 0)
    pos = -np.ones(nv, dtype=np.int64)
    pos[idx] = np.arange(len(idx))
    faces = pos[tris]
    faces = faces[(faces >= 0).all(axis=1)]
    model = fit["hands"][side]["model"]
    before = hand_rig.grip_check(model, rest[idx], Wh, grip, bar_r)
    Wn, report = hand_rig.relax_grip_folds(model, rest[idx], Wh, faces, hand_rig.local_edges(edges, idx, nv),
                                           movable, grip["rotations"], log=log)
    after = hand_rig.grip_check(model, rest[idx], Wn, grip, bar_r)
    ok = (after["deepest_in_bar_mm"] <= before["deepest_in_bar_mm"] + 0.1
          and after["thumb_clear_of_hand_mm"] >= min(before["thumb_clear_of_hand_mm"], hand_rig.THUMB_CLEAR * 1000))
    report.update(grip_before=before, grip_after=after, applied=bool(ok))
    if ok:
        groups = [skin.vertex_groups.get(c) or skin.vertex_groups.new(name=c) for c in cols]
        changed = np.flatnonzero(movable & (np.abs(Wn - Wh).max(axis=1) > 1e-7))
        for i in changed:
            v = int(idx[i])
            for b, vg in enumerate(groups):
                if Wn[i, b] > 0.0:
                    vg.add([v], float(Wn[i, b]), "REPLACE")
                elif Wh[i, b] > 0.0:
                    vg.remove([v])
        report["written_points"] = int(len(changed))
    log(f"grip {side}: skin turned over at the thumb in the closed grip, largest patch "
        f"{report['thumb_largest_before_mm2']:.0f} -> {report['thumb_largest_after_mm2']:.0f} mm² (the whole hand's "
        f"{report['hand_largest_before_mm2']:.0f} -> {report['hand_largest_after_mm2']:.0f}; {report['changed_points']} "
        f"points smoothed, {report['rounds']} rounds); deepest in the bar {before['deepest_in_bar_mm']} -> {after['deepest_in_bar_mm']} mm, "
        f"thumb tip {before['thumb_tip_gap_mm']} -> {after['thumb_tip_gap_mm']} mm, thumb clear of the hand "
        f"{before['thumb_clear_of_hand_mm']} -> {after['thumb_clear_of_hand_mm']} mm"
        + ("" if ok else "; NOT applied (the grip would be worse): the designed weights are kept"))
    return report


BAR_DROP = 0.05             # x the hips-to-neck length: how far below the shoulder joints the bar rides


def mount_barbell(rig, skin, drop=BAR_DROP):
    """The house barbell (`equipment.barbell`, sized to the character) racked across the upper
    back, parented to Spine2 so it rides the torso through the rep, resting ON the skin of the back
    found by a ray (as `exercise_stage._mount_upper_back` places it on the house rig).

    How high: the house rig puts it 0.22 of the torso below the shoulder line; on this rig that is
    64 mm below the shoulder joints, the wrists then sat below the shoulders, and no elbow lets a
    forearm RISE to a bar that low — the user's reference (and the form notes: "on top of the
    traps", or "the shelf between the traps and the rear deltoids") has the hands at shoulder
    height with the forearms climbing to them. At BAR_DROP it sits on the traps."""
    import equipment
    import exercise_stage
    height = json.loads(rig["skin_fit"])["measured"]["height"]
    # No plates: the reference coach squats an empty bar, and seen from the side a plate is a
    # disc the size of the lifter's torso standing between the camera and everything it should show.
    bar = equipment.barbell(height, plates=False)
    world = rig.matrix_world
    pose = rig.pose.bones
    left, right = (world @ pose[P + f"{s}Arm"].head for s in ("Left", "Right"))
    hips = (world @ pose[P + "LeftUpLeg"].head + world @ pose[P + "RightUpLeg"].head) / 2
    neck = world @ pose[P + "Neck"].head
    up = (neck - hips).normalized()
    lateral = right - left
    lateral = (lateral - up * lateral.dot(up)).normalized()
    forward = up.cross(lateral)
    depth = (neck - hips).length
    along = (left + right) / 2 - up * (depth * drop)
    surface = exercise_stage.back_surface(skin, along, forward)
    radius = (equipment.BARBELL_BAR_DIAMETER / 2) * height / 1.75
    point = (surface - forward * radius) if surface is not None else along - forward * depth * 0.33
    exercise_stage.hold_at(bar, rig, P + "Spine2", point, exercise_stage.basis_from(lateral, up))
    return bar, {"on_skin": surface is not None, "at": list(point)}


# The grip, from the sourced form notes (reference/exercises/form-guidelines.json, Barbell Back
# Squat): "roughly 1.2-1.3x shoulder width", pronated, wrist "relatively straight", elbows "pointed
# down and BEHIND the torso at the top", "in line with the torso" at the bottom. Checked by eye
# against the user's reference frames (3d-character/reference-squat/14-17.png).
GRIP_WIDTH = 1.3            # x the outer shoulder width (deltoid to deltoid): the top of the sourced range
PALM_ALONG = 0.42           # the bar crosses the mitten this far from wrist to tip (the palm's middle)
FOREARM_RISE = 0.3          # the forearm aimed at: up the torso, and this much forward (elbow behind the bar)
FINGER_CURL = 55.0          # degrees at each finger joint: the mitten's front closes over the bar
GRIP_CLEAR = 0.001          # metres between palm and bar: touching, not through


def _aim_pose(pbone, direction):
    """Swing a pose bone onto `direction` (armature space) by the shortest arc, keeping its roll
    (`animate.aim`, repeated here so this module does not import the house stage)."""
    current = pbone.matrix.to_3x3()
    now = current @ Vector((0.0, 1.0, 0.0))
    turned = (now.rotation_difference(Vector(direction).normalized()).to_matrix() @ current).to_4x4()
    turned.translation = pbone.matrix.translation
    pbone.matrix = turned
    bpy.context.view_layer.update()


def _palm_rest(rig, side):
    """The skin's hand at rest (armature space = the house frame): its long axis, wrist to
    fingertip as measured, and its palm normal.

    The palm faces DOWN: the bind is the skin's T-pose (`measure_skin` accepts nothing else), and a
    T-posed character holds its palms to the floor — the boy's do (rendered, task 12). Two earlier
    versions read the hand off MPFB's bones instead: the thumb (it points forward AND down; the palm
    came out 45 degrees off and met the bar edge-on) and the middle knuckle (43 degrees above the
    skin's hand line after the warp)."""
    joints = json.loads(rig["skin_fit"])["joints"]
    s = side.lower()
    axis = (Vector(joints[f"tip.{s}"]) - Vector(joints[f"wrist.{s}"])).normalized()
    palm = Vector((0.0, 0.0, -1.0))
    return axis, (palm - axis * palm.dot(axis)).normalized()


def _frame(a, b):
    """A rotation matrix whose columns are a, b and a x b (a and b orthonormalised)."""
    a = a.normalized()
    b = (b - a * b.dot(a)).normalized()
    return Matrix((a, b, a.cross(b))).transposed()


def _curl_axis(rig, name, palm):
    """The axis, in the finger bone's own rest frame, about which a positive turn folds it toward
    the palm: (its direction x the palm normal), since turning d about a by +angle carries it toward
    a x d. Not the bone's local X: on this rig that axis is nearly vertical for the fingers, and
    turning about it swung them sideways."""
    rest = rig.data.bones[name].matrix_local.to_3x3()
    along = rest @ Vector((0.0, 1.0, 0.0))
    axis = along.cross(palm).normalized()
    return (rest.inverted() @ axis).normalized()


def grip_bar(rig, bar, first, last):
    """Hands ON the bar, as a lifter holds it, every frame; arms and fingers re-keyed.

    Mixamo's back squat holds its grip where a real lifter's bar is (on the traps, under the skull);
    the cartoon's skull fills that space, so its bar rides lower on the back and the captured
    hands were left above it, beside the shoulders, elbows flared. Per frame, per side:
      grip     the hand's centre on the bar at GRIP_WIDTH x the outer shoulder width;
      hand     its long axis continuing the forearm (a straight wrist), its palm turned onto the
               bar from behind, the bar across the palm PALM_ALONG of the way to the fingertips;
      arm      a two-bone solve to that wrist, the elbow placed where the forearm rises most
               steeply to the bar (FOREARM_RISE) — elbows down and behind, not out to the side;
      fingers  every finger joint curled FINGER_CURL toward the palm (the mitten's front half is
               weighted to the middle finger's three bones, so it closes over the bar).
    Returns, worst over the clip: the wrist's miss, the highest elbow relative to its shoulder,
    and the grip width.
    """
    scene = bpy.context.scene
    fit = json.loads(rig["skin_fit"])
    hands = fit["hands"]
    height = fit["measured"]["height"]
    import equipment
    bar_r = (equipment.BARBELL_BAR_DIAMETER / 2) * height / 1.75
    half = GRIP_WIDTH * hands["outer_shoulder_half"]
    world = rig.matrix_world
    inv = world.inverted()
    pose = rig.pose.bones
    rest = {side: _palm_rest(rig, side) for side in ("Left", "Right")}
    rest_frame = {side: _frame(*rest[side]) for side in rest}
    fingers = {side: [P + side + f"Hand{f}{k}" for f in ("Index", "Middle", "Ring", "Pinky") for k in (1, 2, 3)]
               for side in rest}
    curl = {name: Quaternion(_curl_axis(rig, name, rest[side][1]), math.radians(FINGER_CURL))
            for side in rest for name in fingers[side]}
    miss, elbow_rise, short = 0.0, -1.0, 0.0
    previous = {}
    for frame in range(first, last + 1):
        scene.frame_set(frame)
        basis = (inv @ bar.matrix_world).to_3x3().normalized()
        along = (basis @ Vector((1.0, 0.0, 0.0))).normalized()     # the character's right
        forward = (basis @ Vector((0.0, 1.0, 0.0))).normalized()
        up = (basis @ Vector((0.0, 0.0, 1.0))).normalized()
        centre = inv @ bar.matrix_world.translation
        for side, sign in (("Left", -1.0), ("Right", 1.0)):
            arm, fore, hand = (pose[P + side + n] for n in ("Arm", "ForeArm", "Hand"))
            shape = hands[side.lower()]
            grip = centre + along * (sign * half)
            shoulder = arm.head.copy()
            l1, l2 = arm.length, fore.length
            v = (up + forward * 0.15).normalized()                  # first guess at the forearm
            for _ in range(3):
                wrist = (grip - v * (PALM_ALONG * shape["length"])
                         - forward * (shape.get("palm_depth", shape["half_thickness"]) + bar_r + GRIP_CLEAR))
                to = wrist - shoulder
                dist = to.length
                u = to.normalized()
                # The elbow goes where the forearm rises most steeply to the bar: the point on the
                # solve's circle nearest the elbow a lifter's forearm would put there (straight up
                # the torso from the elbow to the hand, a little forward). A fixed down-and-back
                # pole tucked the cartoon's elbows in and left its forearms horizontal: its upper
                # arm (105 mm) is shorter than its forearm (125 mm), so the elbow cannot drop a
                # forearm's length under a hand held this low; this gets as close as it can.
                ideal = wrist - (up + forward * FOREARM_RISE).normalized() * l2
                pole = ideal - shoulder
                pole = pole - u * pole.dot(u)
                if pole.length < 1e-6:
                    pole = -up - u * (-up).dot(u)
                pole.normalize()
                if dist >= l1 + l2:
                    short = max(short, dist - (l1 + l2))
                    elbow = shoulder + u * l1
                else:
                    dist = max(dist, abs(l1 - l2) + 1e-6)
                    a = math.acos(max(-1.0, min(1.0, (l1 * l1 + dist * dist - l2 * l2) / (2 * l1 * dist))))
                    elbow = shoulder + (u * math.cos(a) + pole * math.sin(a)) * l1
                v = (wrist - elbow).normalized()
            _aim_pose(arm, elbow - shoulder)
            _aim_pose(fore, wrist - fore.head)
            # the hand: long axis along the forearm (straight wrist), palm facing forward onto the bar
            turn = _frame(v, forward) @ rest_frame[side].inverted()
            m = (turn @ hand.bone.matrix_local.to_3x3()).to_4x4()
            m.translation = hand.head
            hand.matrix = m
            bpy.context.view_layer.update()
            for name in fingers[side]:
                fb = pose[name]
                fb.rotation_mode = "QUATERNION"
                fb.rotation_quaternion = curl[name]
            bpy.context.view_layer.update()
            miss = max(miss, (hand.head - wrist).length)
            elbow_rise = max(elbow_rise, (elbow - shoulder).dot(up))
            for pb in [arm, fore, hand] + [pose[n] for n in fingers[side]]:
                q = pb.rotation_quaternion.copy()
                if pb.name in previous and q.dot(previous[pb.name]) < 0.0:
                    q.negate()
                    pb.rotation_quaternion = q
                previous[pb.name] = q.copy()
                pb.keyframe_insert("rotation_quaternion", frame=frame)
    scene.frame_set(first)
    return {"grip_half_width_mm": half * 1000,
            "grip_vs_outer_shoulder": GRIP_WIDTH,
            "wrist_miss_mm": miss * 1000, "reach_short_mm": short * 1000,
            "highest_elbow_above_shoulder_mm": elbow_rise * 1000}


def bar_contact(rig, skin, bar, first, last):
    """How the mitten meets the bar over the clip: the deepest any hand vertex goes INSIDE the bar
    (its distance to the bar's axis below the bar's radius), and how close the nearest one comes
    (a hand that floats has nothing within a few millimetres)."""
    import equipment
    fit = json.loads(rig["skin_fit"])
    height = fit["measured"]["height"]
    bar_r = (equipment.BARBELL_BAR_DIAMETER / 2) * height / 1.75
    names = [g.name for g in skin.vertex_groups]
    hand_groups = [skin.vertex_groups[n].index for n in names if "Hand" in n]   # hand, fingers, thumb
    hand = np.zeros(len(skin.data.vertices), dtype=bool)
    for v in skin.data.vertices:
        if any(g.group in hand_groups and g.weight > 0.5 for g in v.groups):
            hand[v.index] = True
    scene = bpy.context.scene
    inside, nearest = 0.0, 1e9
    per = []
    for frame in range(first, last + 1):
        scene.frame_set(frame)
        pts = skin_wrap.evaluated(skin)[0][hand]
        mw = bar.matrix_world
        c = np.array(mw.translation)
        x = np.array((mw.to_3x3() @ Vector((1.0, 0.0, 0.0))).normalized())
        rel = pts - c
        radial = np.linalg.norm(rel - np.outer(rel @ x, x), axis=1)
        gap = radial - bar_r
        inside = max(inside, float(-gap.min()))
        nearest = min(nearest, float(gap.min()))
        per.append(float(gap.min()))
    scene.frame_set(first)
    return {"hand_vertices": int(hand.sum()), "deepest_inside_bar_mm": max(inside, 0.0) * 1000,
            "nearest_gap_mm_min": nearest * 1000,
            "frames_touching_within_3mm": int(sum(1 for g in per if g < 0.003)),
            "frames": len(per)}


# --- matching a reference: the coach's measured form ---------------------------------------------
#
# The user's acceptance test: the squat must match their reference (a coach in a commercial app,
# 3d-character/reference-squat/14-17.png), measured with MediaPipe into targets.json — knee, hip,
# elbow and upper-arm angles and torso lean at standing / mid / bottom, and the grip and elbow
# widths, wrist and elbow heights and forearm line from behind. Mixamo's clip squats well below
# parallel and leans further; its TIMING and weight shift are what make it look alive. So the clip
# is kept as the clock and the poses are corrected toward the targets, frame by frame:
#
#   progress  each frame's place in the rep is the clip's own knee angle, from its straightest
#             (0) to its deepest (1); the target knee is the coach's, at the same progress, and the
#             hip, lean, elbow and upper-arm targets follow the coach's knee through his three
#             measured phases (piecewise linear in the knee angle);
#   torso     the back is held straight: the spine's side-seen bends are put back to the rest pose's
#             (`_straighten_spine`), and the lean comes from tipping the pelvis about the hip joints
#             (`_tip_pelvis`), as a squat's hip hinge does (task 16; see there);
#   legs      the hips are moved (2-D Newton on their height and depth) until the locked legs
#             have the target knee and hip angles; the legs are re-solved onto the planted ankles,
#             each foot keeping its rotation;
#   head      with the coach's neck angle and face direction in the targets (`read_form_targets`),
#             the neck and head are turned onto them (`_place_head`); otherwise a neutral nod;
#   arms      see `_solve_arm`; the grip width and the bar's height on the back are searched once
#             for the whole rep, the elbow is placed per frame.
# Everything is measured on the rig's joints with targets.json's own definitions (`form_metrics`):
# the knee and hip in 3-D, or in the side view where the file's "measure" says "flat" (`_flat_legs`).
# The torso line (hip joints -> shoulder joints, as the coach's MediaPipe landmarks) is counted from the
# rig's own upright bind pose (`rest_lean`, task 18), in every angle that uses it — the lean, the hip
# angle, the upper-arm angle and the neck angle — as the Exercise Studio counts it
# (studio/measure/metrics.py `rest_turn`): this cartoon's shoulder joints sit 76 mm behind its hip
# joints standing tall (its bind pose reads 15 degrees back), and matched joint for joint the coach's
# upright start (7 degrees) cost a visible 22-degree bow from the hips. The legs are read the way they look
# (task 20, `_leg_turns`; metrics.py `rest_leg_turns`): each thigh line is turned onto the thigh's own skin
# outline as the bind pose shows it, the shin with it, in the knee and hip angles and the depth. The cartoon's
# hip joint sits toward the front of the thigh's top: its thigh joint line slopes 10.6 degrees back standing,
# its skin outline 4.5 (MediaPipe on its bind render: about 4).
#   picture   with the Studio's targets beside the file (its targets.json), the lean and hip angle are then
#             aimed at what the Studio judges: our picture, read with the coach's own detector
#             (`match_picture`, task 19). On this cartoon the detector reads the trunk up to 12 degrees
#             more upright than the joint line at the bottom, so the joint-line targets are corrected by
#             what the picture misses, in a few rounds, the depth (hips against knees) held in tolerance.

GAZE_DOWN = 15.0            # degrees: the least the face looks down (the coach, standing, 15.png)
FORM_KEYS = ("knee_included", "hip_included", "torso_lean", "elbow_included", "upperarm_vs_torso")
# The coach's posture, as the Exercise Studio measures it (studio/measure/posture.py), aimed at when the
# targets carry it: neck_vs_torso (hip joints -> shoulder joints against shoulder joints -> the ears, +
# = head forward of the torso line) and head_pitch (the face against level, + = up), both side-seen.
POSTURE_KEYS = ("neck_vs_torso", "head_pitch")
STUDIO_PHASES = {"standing": "start", "mid": "middle", "bottom": "end"}   # the Studio's names for them
SPINE_SEGMENTS = (("Hips", "Spine"), ("Spine", "Spine1"), ("Spine1", "Spine2"), ("Spine2", "Neck"))
PITCH_SOFT = 2.0            # degrees: the face direction's scale in `_place_head`'s cost (the neck's is 1):
                            # 4 met the neck within 1.6 but left the face 14 off at mid; 2 keeps the neck
                            # within 4 and the face within 8 (task 16 sweep, the Studio's squat targets)
HEAD_SPLIT = 5.0            # degrees: ... and the scale of the neck's bend straying from NECK_PART of the two
NECK_PART = 0.4             # the neck bone's share of the neck-and-head bend (the shipped motion's: 13 of 31)
HEAD_BEND_MAX = 40.0        # degrees: the head bends no further than this from rest on the neck
NECK_BEND_MAX = 13.0        # degrees: the neck bone no further than this on the chest. The neck's skin
                            # (task 15's designed weights) was proven on a neck bone bent 9-13 degrees;
                            # bent 24-33 the nape folded 300-540 mm2 (task 16 sweep), at 13 it is 82
EAR_DEPTH = 0.10            # the Studio's ear cluster (inspect_run.ear_vertices): this far in from each
EAR_CARRIED = 0.99          # side's widest point at ear height, on skin the Head bone alone carries


def read_form_targets(path, log=print):
    """The --form-targets file. The Exercise Studio writes the coach's neck angle and face direction into
    it per position (pipeline-targets.json: {"phases": {"standing": {"neck_vs_torso": 38.6, "head_pitch":
    -36.2, ...}, ...}}, task 18; they used to be read from its targets.json beside it). A file without
    them on every position (the hand-made targets) leaves the head to its neutral nod: said in the log."""
    targets = json.loads(Path(path).read_text())
    if _has_posture(targets):
        log("form: the coach's neck and face from the targets: " + ", ".join(
            f"{p} neck {targets['phases'][p]['neck_vs_torso']:.1f} face {targets['phases'][p]['head_pitch']:+.1f}"
            for p in STUDIO_PHASES))
    else:
        log("form: the targets carry no neck/face for every position; the head keeps a neutral nod")
    return targets


def _has_posture(targets):
    return all(isinstance(targets.get("phases", {}).get(p, {}).get(k), (int, float))
               for p in STUDIO_PHASES for k in POSTURE_KEYS)


def _yz_tilt(v):
    """Degrees a vector tips forward (+Y, the house figure's front) from straight up, seen from the
    side (the y-z plane, as `form_metrics` reads the lean). A turn about +X by a lowers it by a."""
    return math.degrees(math.atan2(v.y, v.z))


def rest_lean(rig):
    """Degrees the torso line (hip joints' midpoint -> shoulder joints' midpoint) leans forward (+) or
    back (-) in the rig's rest (bind) pose, seen from the side (the y-z plane): its neutral upright
    stance. Every torso angle here is counted from it (see the section above)."""
    world, bones = rig.matrix_world, rig.data.bones
    at = lambda n: world @ bones[P + n].head_local                  # noqa: E731
    return _yz_tilt((at("LeftArm") + at("RightArm")) / 2 - (at("LeftUpLeg") + at("RightUpLeg")) / 2)


def bind_pose_warning(rig, side="right"):
    """The Exercise Studio's bind-pose rule (studio/measure/metrics.py `bind_pose_warning`, read on its side
    camera by inspect_run `bind_pose`), for this solver's log; None when the bind pose stands upright. One rule:
    the solver and the Studio say the same thing about the same rig."""
    warning = _studio().bind_pose(rig, side)[2]
    return None if warning is None else "form: " + warning


def _torso_turn(rig):
    """The turn about X that stands the rest pose's torso line upright, seen from the side: applied to
    each torso line (hip -> shoulder) before the angles that use it. A turn about +X by a lowers the
    side-seen tilt by a, so turning by the rest lean (negative: leaning back) raises it to 0."""
    return Matrix.Rotation(math.radians(rest_lean(rig)), 3, "X")


LEG_BONES = {"thigh": ("UpLeg", "Leg"), "shin": ("Leg", "Foot")}    # each limb's joint line, from -> to (bone heads)


def rest_leg_slopes(rig):
    """{"thigh.left", "thigh.right", "shin.left", "shin.right"}: degrees each leg limb's joint line (hip -> knee,
    knee -> ankle) points forward (+Y, + ) or back (-) of straight down in the rig's rest (bind) pose, seen from
    the side (the y-z plane). The cartoon's: -10.6 each (its hip joint sits toward the front of the thigh's top)."""
    world, bones = rig.matrix_world, rig.data.bones
    at = lambda n: world @ bones[P + n].head_local                  # noqa: E731
    out = {}
    for limb, (a, b) in LEG_BONES.items():
        for side in ("Left", "Right"):
            out[f"{limb}.{side.lower()}"] = _down_slope(at(side + b) - at(side + a))
    return out


def _down_slope(v):
    """Degrees a downward line points forward (+Y) or back (-) of straight down, seen from the side (y-z)."""
    return math.degrees(math.atan2(v[1], -v[2]))


def rest_thigh_outlines(rig):
    """{"left", "right"}: degrees each thigh's skin outline in the bind pose points forward (+) or back (-) of
    straight down, seen from the side (the y-z plane): the Exercise Studio's own reading of it (inspect_run
    `thigh_outlines`, on the skin it measures), None where it finds too little skin. The cartoon's: about -4.5."""
    ir = _studio()
    outline = ir.thigh_outlines(rig, ir.skin_of(rig), np.array([1.0, 0.0, 0.0]))
    return {s: (None if d is None else _down_slope(d)) for s, d in outline.items()}


def _leg_turns(rig):
    """{"thigh.left", ...}: the turns about X that lay each thigh line onto the thigh's skin outline in the bind
    pose (the shin turns with it), seen from the side: applied before the knee and hip angles and the depth, as
    the Studio counts our legs (studio/measure/metrics.py `rest_leg_turns`, task 20). A turn about +X by a moves a
    downward line's side-seen slope by +a. A side without an outline is read on its joints."""
    joints, outline = rest_leg_slopes(rig), rest_thigh_outlines(rig)
    out = {}
    for s in ("left", "right"):
        a = 0.0 if outline[s] is None else outline[s] - joints[f"thigh.{s}"]
        out[f"thigh.{s}"] = Matrix.Rotation(math.radians(a), 3, "X")
        out[f"shin.{s}"] = out[f"thigh.{s}"].copy()
    return out


def _torso_hip(S, hip, turn):
    """Where the hip joint would be for the upper-arm angle (shoulder joint S -> elbow against S ->
    hip): the torso line S -> hip turned by `turn` about S."""
    return S - turn @ (S - hip)


def _spine_tilts(rig, rest=False):
    """{bone: side-seen tilt of the segment from its head to the next one's}, Hips to Spine2 (Spine2's
    runs to the neck's base): the chain the Studio judges the back on (inspect_run.SPINE_CHAIN)."""
    if rest:
        at = {n: rig.data.bones[P + n].head_local for pair in SPINE_SEGMENTS for n in pair}
    else:
        at = {n: rig.pose.bones[P + n].head for pair in SPINE_SEGMENTS for n in pair}
    return {a: _yz_tilt(at[b] - at[a]) for a, b in SPINE_SEGMENTS}


def spine_curl(rig, rest_tilts):
    """Degrees the back curls (+, rounded) or arches (-) from the rest pose: the side-seen bends at
    Spine, Spine1 and Spine2 summed (the Studio's back_rounding, on the y-z plane)."""
    now = _spine_tilts(rig)
    return (now["Spine2"] - rest_tilts["Spine2"]) - (now["Hips"] - rest_tilts["Hips"])


def _straighten_spine(rig, rest_tilts):
    """Each spine bone turned about X (its own head) so its segment bends on its parent's as at rest,
    seen from the side: the back keeps the rest pose's curve. The clip's side bend and twist stay.
    Turning a bone carries the ones above it, so they are done from the bottom up."""
    pose = rig.pose.bones
    for (below, _), (name, _) in zip(SPINE_SEGMENTS, SPINE_SEGMENTS[1:]):
        now = _spine_tilts(rig)
        bend = (now[name] - rest_tilts[name]) - (now[below] - rest_tilts[below])
        if abs(bend) > 1e-6:
            _turn_world(pose[P + name], Matrix.Rotation(math.radians(bend), 3, "X"))


def _tip_pelvis(rig, lean, rounds=4):
    """The whole figure above the hip joints tipped about the line through them until the
    hip-mid -> shoulder-mid line leans `lean` degrees from upright, seen from the side, counted from the
    rest pose's own lean (`rest_lean`: the bind pose stands tall at 0). About the hip
    line itself, so both hip joints keep their places relative to each other: tipped about X instead,
    a pelvis the clip turns a little sideways carried one hip 5 mm forward of the other, and the
    hidden fold at the crotch at the bottom grew to 1,932 mm2 (1,040 about the hip line; task 16).
    The hip line is nearly X, so a turn moves the side-seen lean by nearly its angle; a few rounds make
    it exact. The legs go with it; the caller re-solves them onto the planted ankles."""
    root = rig.pose.bones[P + "Hips"]
    rest = rest_lean(rig)
    for _ in range(rounds):
        j = _joints(rig)
        mid = (j["hip.left"] + j["hip.right"]) / 2
        now = _yz_tilt((j["shoulder.left"] + j["shoulder.right"]) / 2 - mid) - rest
        if abs(lean - now) < 1e-4:
            break
        axis = (j["hip.right"] - j["hip.left"]).normalized()
        turn = Matrix.Rotation(-math.radians(lean - now), 4, axis)
        root.matrix = Matrix.Translation(mid) @ turn @ Matrix.Translation(-mid) @ root.matrix
        bpy.context.view_layer.update()


def ears_local(rig, skin):
    """The ears' midpoint as the Studio finds it (`inspect_run.ear_vertices`: of the skin the Head bone
    alone carries, the vertices at ear height — from the head joint up to the middle of the head —
    within EAR_DEPTH of each side's widest point), in the Head bone's rest frame, so that
    `head.matrix @ point` is where it is in any pose. None when no ears are found."""
    bones = rig.data.bones
    deform = {b.name for b in bones if b.use_deform}
    names = {g.index: g.name for g in skin.vertex_groups}
    world = np.array(rig.matrix_world.inverted() @ skin.matrix_world)     # skin -> armature space
    co = np.array([v.co for v in skin.data.vertices])
    co = co @ world[:3, :3].T + world[:3, 3]
    head = []
    for v in skin.data.vertices:
        w = {names[g.group]: g.weight for g in v.groups if names.get(g.group) in deform}
        if w and w.get(P + "Head", 0.0) >= EAR_CARRIED * sum(w.values()):
            head.append(v.index)
    if not head:
        return None
    head = np.array(head)
    joint = np.array(bones[P + "Head"].head_local)
    left = np.array(bones[P + "LeftUpLeg"].head_local - bones[P + "RightUpLeg"].head_local)
    left = np.array([left[0], left[1], 0.0]) / max(float(np.hypot(left[0], left[1])), 1e-9)
    z = co[head, 2]
    band = head[(z >= joint[2]) & (z <= (z.min() + z.max()) / 2)]
    if not len(band):
        return None
    mids = []
    for sign in (1.0, -1.0):
        out = sign * ((co[band] - joint) @ left)
        mids.append(co[band[out >= out.max() * (1 - EAR_DEPTH)]].mean(axis=0))
    mid = Vector(((mids[0] + mids[1]) / 2).tolist())
    return bones[P + "Head"].matrix_local.inverted() @ mid


def posture_metrics(rig, ears, rest_tilts):
    """The Studio's posture readings on the rig as posed now, side-seen (the y-z plane): spine_curl,
    neck_vs_torso (needs `ears`, from `ears_local`; the torso line counted from the rest pose's lean) and
    head_pitch."""
    pose = rig.pose.bones
    j = _joints(rig)
    hip = (j["hip.left"] + j["hip.right"]) / 2
    sh = (j["shoulder.left"] + j["shoulder.right"]) / 2
    head = pose[P + "Head"]
    face = (head.matrix.to_3x3() @ head.bone.matrix_local.to_3x3().inverted()) @ Vector((0.0, 1.0, 0.0))
    out = {"spine_curl": spine_curl(rig, rest_tilts), "head_pitch": math.degrees(math.atan2(face.z, face.y))}
    if ears is not None:
        out["neck_vs_torso"] = _yz_tilt(head.matrix @ ears - sh) - (_yz_tilt(sh - hip) - rest_lean(rig))
    return out


def _bend(rig, name, parent):
    """Side-seen bend of a bone's own axis on its parent's, less the same at rest (degrees)."""
    pose, bones = rig.pose.bones, rig.data.bones
    axis = lambda m: m.to_3x3() @ Vector((0.0, 1.0, 0.0))            # noqa: E731
    now = _yz_tilt(axis(pose[P + name].matrix)) - _yz_tilt(axis(pose[P + parent].matrix))
    rest = _yz_tilt(axis(bones[P + name].matrix_local)) - _yz_tilt(axis(bones[P + parent].matrix_local))
    return now - rest


def _place_head(rig, ears, neck_goal, pitch_goal):
    """The neck and head turned about X (each about its own head) so the ears sit at the coach's neck
    angle and the face points as his does. The face direction turns by the two turns' sum, and the
    ears move mostly with it (the neck bone is 2-3 cm long), so the two goals pull on nearly one
    freedom: the neck angle is aimed at first (the Studio judges it), the face is held softly
    (PITCH_SOFT), the two bones share the bend (HEAD_SPLIT) and neither bends past HEAD_BEND_MAX.
    Solved on the joints alone (no scene update per try): a coarse grid, then a fine one."""
    pose = rig.pose.bones
    neck, head = pose[P + "Neck"], pose[P + "Head"]
    j = _joints(rig)
    hip = (j["hip.left"] + j["hip.right"]) / 2
    sh = (j["shoulder.left"] + j["shoulder.right"]) / 2
    torso = _yz_tilt(sh - hip) - rest_lean(rig)        # counted from the rest pose, as the Studio does
    nh, hh, ear = neck.head.copy(), head.head.copy(), head.matrix @ ears
    face = (head.matrix.to_3x3() @ head.bone.matrix_local.to_3x3().inverted()) @ Vector((0.0, 1.0, 0.0))
    pitch0 = math.degrees(math.atan2(face.z, face.y))
    bend_n, bend_h = _bend(rig, "Neck", "Spine2"), _bend(rig, "Head", "Neck")

    def cost(a, b):
        if abs(bend_n - a) > NECK_BEND_MAX or abs(bend_h - b) > HEAD_BEND_MAX:
            return float("inf"), None
        ra, rb = Matrix.Rotation(math.radians(a), 3, "X"), Matrix.Rotation(math.radians(b), 3, "X")
        hh2 = nh + ra @ (hh - nh)
        e = hh2 + rb @ (nh + ra @ (ear - nh) - hh2)
        got = _yz_tilt(e - sh) - torso
        pitch = pitch0 + a + b
        c = (got - neck_goal) ** 2 + ((pitch - pitch_goal) / PITCH_SOFT) ** 2 \
            + (((bend_n - a) - NECK_PART * (bend_n - a + bend_h - b)) / HEAD_SPLIT) ** 2
        return c, (got, pitch)

    best = min(((cost(a, b)[0], a, b) for a in range(-90, 91, 3) for b in range(-90, 91, 3)))
    _, a0, b0 = best
    best = min(((cost(a0 + da, b0 + db)[0], a0 + da, b0 + db)
                for da in np.arange(-3.0, 3.01, 0.25) for db in np.arange(-3.0, 3.01, 0.25)))
    _, a, b = best
    _turn_world(neck, Matrix.Rotation(math.radians(a), 3, "X"))
    _turn_world(head, Matrix.Rotation(math.radians(b), 3, "X"))


def _deg(a, b):
    a, b = Vector(a), Vector(b)
    if a.length < 1e-9 or b.length < 1e-9:
        return 0.0
    return math.degrees(a.angle(b))


def _joints(rig):
    """World positions of the joints the form metrics use, both sides."""
    world, pose = rig.matrix_world, rig.pose.bones
    out = {}
    for side in ("Left", "Right"):
        for key, bone in (("hip", "UpLeg"), ("knee", "Leg"), ("ankle", "Foot"), ("shoulder", "Arm"),
                          ("elbow", "ForeArm"), ("wrist", "Hand")):
            out[f"{key}.{side.lower()}"] = world @ pose[P + side + bone].head
    return out


LEG_ANGLES = ("knee_included", "hip_included")    # read either way: in 3-D or in the side view
# the metrics the solver aims in one measure only: the lean in the side view, the arm angles in 3-D,
# the widths seen from behind (the hand-made file's definitions; the Studio's rules agree)
FIXED_MEASURE = {"torso_lean": "flat", "elbow_included": "3d", "upperarm_vs_torso": "3d",
                 "grip_ratio": "flat", "elbow_width_ratio": "flat"}


def _flat_legs(targets):
    """The leg angles to aim at in the side view ("flat") rather than in 3-D, from the targets'
    optional "measure": per metric ({"knee_included": "flat", ...}) or one word for the file
    ("flat" | "3d", which covers the knee and hip: the angles that can be read either way).
    Without it, 3-D, as the hand-made file's definitions say. A measure the solver cannot aim a
    metric in (a 3-D lean, a flat elbow) is refused rather than aimed at in another."""
    stated = targets.get("measure")
    if stated is None:
        return ()
    if isinstance(stated, str):
        stated = {k: stated for k in LEG_ANGLES}
    if not isinstance(stated, dict):
        raise SystemExit('targets: "measure" must be "flat", "3d" or {metric: "flat" | "3d"}')
    for key, measure in stated.items():
        if measure not in ("flat", "3d"):
            raise SystemExit(f'targets: "measure" of {key} is {measure!r}; it must be "flat" or "3d"')
        if FIXED_MEASURE.get(key, measure) != measure:
            raise SystemExit(f'targets: {key} is given in "{measure}", but this solver aims it only in '
                             f'"{FIXED_MEASURE[key]}"')
    return tuple(k for k in LEG_ANGLES if stated.get(k) == "flat")


def form_metrics(rig, flat=()):
    """targets.json's metrics on the rig as posed now, from joint positions (definitions in the
    file): included angles in 3-D, the lean in the side (y-z) projection, widths seen from behind
    (the x-z projection), heights in shoulder-joint widths. The knee and hip angles named in `flat`
    are taken in the side projection instead (`_flat_legs`). Each torso line (hip -> shoulder) is
    turned by the rest pose's lean first (`_torso_turn`), for the lean, hip and upper-arm angles; each
    thigh line (hip -> knee) onto the thigh's skin outline in the bind pose, the shin with it (`_leg_turns`,
    task 20), for the knee and hip angles and the hips' height against the knees (a thigh level as it looks
    reads 0)."""
    j = _joints(rig)
    sides = ("left", "right")
    mean = lambda f: sum(f(s) for s in sides) / 2                    # noqa: E731
    turn = _torso_turn(rig)
    legs = _leg_turns(rig)
    thigh = lambda s: legs[f"thigh.{s}"] @ (j[f"knee.{s}"] - j[f"hip.{s}"])       # noqa: E731
    shin = lambda s: legs[f"shin.{s}"] @ (j[f"ankle.{s}"] - j[f"knee.{s}"])       # noqa: E731
    hip_mid = (j["hip.left"] + j["hip.right"]) / 2
    sh_mid = (j["shoulder.left"] + j["shoulder.right"]) / 2
    torso = turn @ (sh_mid - hip_mid)
    behind = lambda v: Vector((v.x, 0.0, v.z))                       # noqa: E731
    sw = behind(j["shoulder.right"] - j["shoulder.left"]).length
    fore = {s: behind(j[f"wrist.{s}"] - j[f"elbow.{s}"]) for s in sides}
    seen = {k: (lambda v: Vector((0.0, v.y, v.z))) if k in flat else (lambda v: v) for k in LEG_ANGLES}
    knee, hip = seen["knee_included"], seen["hip_included"]
    return {
        "knee_included": mean(lambda s: _deg(knee(-thigh(s)), knee(shin(s)))),
        "hip_included": mean(lambda s: _deg(hip(turn @ (j[f"shoulder.{s}"] - j[f"hip.{s}"])), hip(thigh(s)))),
        "torso_lean": math.degrees(math.atan2(abs(torso.y), torso.z)),
        "elbow_included": mean(lambda s: _deg(j[f"shoulder.{s}"] - j[f"elbow.{s}"], j[f"wrist.{s}"] - j[f"elbow.{s}"])),
        "upperarm_vs_torso": mean(lambda s: _deg(j[f"elbow.{s}"] - j[f"shoulder.{s}"], -(turn @ (j[f"shoulder.{s}"] - j[f"hip.{s}"])))),
        "hip_vs_knee_mm": mean(lambda s: -thigh(s).z) * 1000,
        "grip_ratio": behind(j["wrist.right"] - j["wrist.left"]).length / sw,
        "elbow_width_ratio": behind(j["elbow.right"] - j["elbow.left"]).length / sw,
        "wrist_above_shoulder_sw": mean(lambda s: j[f"wrist.{s}"].z - j[f"shoulder.{s}"].z) / sw,
        "elbow_below_shoulder_sw": mean(lambda s: j[f"shoulder.{s}"].z - j[f"elbow.{s}"].z) / sw,
        "forearm_from_vertical_behind": mean(lambda s: _deg(fore[s], Vector((0.0, 0.0, 1.0)))),
    }


def _targets_at(targets, knee):
    """The coach's angles at a given knee angle: piecewise linear through his three phases (his
    posture too, when every phase has it)."""
    phases = sorted(targets["phases"].values(), key=lambda p: p["knee_included"])
    ks = [p["knee_included"] for p in phases]
    keys = FORM_KEYS + (POSTURE_KEYS if all(k in p for p in phases for k in POSTURE_KEYS) else ())
    return {k: float(np.interp(knee, ks, [p[k] for p in phases])) for k in keys}


def _turn_world(pb, rotation):
    """Turn a pose bone by a world (armature-space) rotation about its own head."""
    m = pb.matrix.copy()
    turned = (rotation @ m.to_3x3()).to_4x4()
    turned.translation = m.translation
    pb.matrix = turned
    bpy.context.view_layer.update()


def _two_bone_knee(hip, target, l1, l2, pole):
    to = target - hip
    dist = to.length
    u = to.normalized()
    v = pole - u * pole.dot(u)
    if v.length < 1e-9:
        v = Vector((0.0, 1.0, 0.0)) - u * u.y
    v.normalize()
    if dist >= l1 + l2:
        return hip + u * l1
    dist = max(dist, abs(l1 - l2) + 1e-6)
    a = math.acos(max(-1.0, min(1.0, (l1 * l1 + dist * dist - l2 * l2) / (2 * l1 * dist))))
    return hip + (u * math.cos(a) + v * math.sin(a)) * l1


def _key(rig, previous, frame, names=None):
    for pb in rig.pose.bones:
        if names is not None and pb.name not in names:
            continue
        q = pb.rotation_quaternion.copy()
        if pb.name in previous and q.dot(previous[pb.name]) < 0.0:
            q.negate()
            pb.rotation_quaternion = q
        previous[pb.name] = q.copy()
        pb.keyframe_insert("rotation_quaternion", frame=frame)


def _move_root(rig, delta):
    root = rig.pose.bones[P + "Hips"]
    rest = root.bone.matrix_local.to_quaternion()
    root.location = root.location + rest.inverted() @ Vector(delta)
    bpy.context.view_layer.update()


REACH_MARGIN = 0.5          # degrees: an out-of-reach hip goal is held this far below the frame's best
REACH_SOFT = 3.0            # degrees: ... reached through a smooth minimum this wide
POLE_BAND = (0.12, 0.55)    # the captured knee direction's forward part: below the band the fallback, above it the capture


def match_legs_torso(rig, targets, first, last, skin=None, log=print):
    """The legs and torso onto the coach's angles at the clip's own timing (see the section above).
    Re-keys the legs, spine, neck, head and root; returns per-frame target knee angles and the worst
    misses.

    The back (task 16). The clip bends the spine 9 degrees standing to 35 at the bottom, and this pass
    used to reach the coach's lean by bending it further (Spine, Spine1 and Spine2 each turned by a
    third: 3x the turn in curl for about 1.4x in lean), never turning the pelvis: the Studio read a
    back rounded 30-41 degrees and a head 20 degrees forward of the coach's. Now the spine keeps its
    rest curve every frame (`_straighten_spine`) and the pelvis tips about the hip joints to give the
    lean (`_tip_pelvis`) before the legs are solved; the legs then hinge at the hips. With the coach's
    neck and face in the targets, `_place_head` sets them (`skin` is needed to find the ears)."""
    scene = bpy.context.scene
    pose = rig.pose.bones
    sides = ("Left", "Right")
    rest_tilts = _spine_tilts(rig, rest=True)
    log(f"form: the torso line at rest (the bind pose, counted as upright) leans {rest_lean(rig):+.1f} degrees")
    log("form: the legs at rest (the bind pose), degrees forward of straight down: joint lines " + ", ".join(
        f"{k} {v:+.1f}" for k, v in rest_leg_slopes(rig).items()) + "; the thighs' skin outlines (counted from) "
        + ", ".join(f"{s} " + ("not found: its joints" if v is None else f"{v:+.1f}")
                    for s, v in rest_thigh_outlines(rig).items()))
    leg_turn = _leg_turns(rig)
    warning = bind_pose_warning(rig, targets.get("camera_side", "right"))
    if warning:
        log(warning)
    ears = None
    if _has_posture(targets):
        ears = ears_local(rig, skin) if skin is not None else None
        if ears is None:
            log("form: the coach's neck is in the targets, but no ears were found on the skin; "
                "the head keeps a neutral nod")
    knees = []
    for frame in range(first, last + 1):
        scene.frame_set(frame)
        knees.append(form_metrics(rig)["knee_included"])
    k_top, k_low = max(knees), min(knees)
    stand, bottom = targets["phases"]["standing"]["knee_included"], targets["phases"]["bottom"]["knee_included"]
    plan = {f: stand + (bottom - stand) * (k_top - k) / max(k_top - k_low, 1e-6)
            for f, k in zip(range(first, last + 1), knees)}
    previous, worst = {}, {k: 0.0 for k in ("knee_included", "hip_included", "torso_lean")}
    if ears is not None:
        worst.update({k: 0.0 for k in POSTURE_KEYS})
    curls = []
    legs = {s: (pose[P + s + "UpLeg"].length, pose[P + s + "Leg"].length) for s in sides}
    flat = _flat_legs(targets)            # leg angles aimed at in the side view, as they were measured
    side = lambda v: Vector((0.0, v.y, v.z))                         # noqa: E731
    last_y = None
    for frame in range(first, last + 1):
        scene.frame_set(frame)
        want = _targets_at(targets, plan[frame])
        j = _joints(rig)
        ankles = {s: j[f"ankle.{s.lower()}"].copy() for s in sides}
        feet = {s: pose[P + s + "Foot"].matrix.to_3x3().copy() for s in sides}
        poles = {}
        for s in sides:
            # the knee bends where the capture bent it (knees out included) — unless that points it
            # backward: on two frames of the squat the captured knee lay almost on the hip-ankle
            # line, the solve flipped it behind the leg and the thigh went 29 mm through the floor
            # (and blended into the fallback over a band, not switched: a hard switch at 0.3 moved the
            # knee 22 mm sideways between frames 13 and 14, task 13)
            # The direction handed to the two-bone solve is the capture's BEND: the knee's offset
            # from the capture's own hip-ankle line, square to the leg. The thigh itself (hip->knee)
            # lies only 27-35 degrees off that line; once the solve moved the hips 80-120 mm back to
            # close the hip, the new hip-ankle line swung onto the thigh, the knee's side was lost and
            # it turned out and then backward (task 14: the Studio's targets, frames 18-31, knees up
            # to 97 mm behind the leg line, a thigh 48 mm through the floor, a 245 mm jump).
            thigh = j[f"knee.{s.lower()}"] - j[f"hip.{s.lower()}"]
            line = j[f"ankle.{s.lower()}"] - j[f"hip.{s.lower()}"]
            out = Vector((-0.4 if s == "Left" else 0.4, 1.0, 0.0)).normalized()
            bend = thigh - line.normalized() * thigh.dot(line.normalized()) if line.length > 1e-6 else thigh
            if thigh.length < 1e-6 or bend.length < 1e-6:
                pole = out
            else:
                t = min(max((POLE_BAND[1] - thigh.normalized().y) / (POLE_BAND[1] - POLE_BAND[0]), 0.0), 1.0)
                t = t * t * (3.0 - 2.0 * t)
                pole = (bend.normalized() * (1.0 - t) + out * t).normalized()
            poles[s] = pole
        goal = dict(want)
        for _ in range(2):                       # second round corrects by what the first missed
            # the back straight, the lean from the pelvis (it does not depend on where the hips go)
            _straighten_spine(rig, rest_tilts)
            _tip_pelvis(rig, goal["torso_lean"])
            j = _joints(rig)
            mid = (j["hip.left"] + j["hip.right"]) / 2
            offs = {s: j[f"hip.{s.lower()}"] - mid for s in sides}
            theta = math.radians(goal["torso_lean"])
            torso = Vector((0.0, math.sin(theta), math.cos(theta)))

            # The hip height that gives each leg the hip-to-ankle distance its target knee angle
            # needs (law of cosines), then the hip depth that gives the target hip angle: two
            # nested 1-D solves. (A 2-D Newton stalled near a straight knee, where 160 and 180
            # degrees are 5 mm of hip height apart.)
            need = {s: math.sqrt(max(legs[s][0] ** 2 + legs[s][1] ** 2 - 2 * legs[s][0] * legs[s][1]
                                     * math.cos(math.radians(goal["knee_included"])), 1e-9)) for s in sides}

            def flat_knee(y, z):
                total = 0.0
                for s in sides:
                    hip = Vector((mid.x, y, z)) + offs[s]
                    knee = _two_bone_knee(hip, ankles[s], legs[s][0], legs[s][1], poles[s])
                    total += _deg(side(hip - knee), side(ankles[s] - knee))     # the shin turns with the thigh
                return total / 2

            def height_for(y):
                lo, hi = max(a.z for a in ankles.values()), max(a.z for a in ankles.values()) + sum(legs["Left"])
                if "knee_included" in flat:
                    # a side-view knee angle is not set by the hip-ankle distance alone (the knees go
                    # out), so the height is bisected on the angle itself; it opens as the hips rise.
                    # No higher than the legs reach: the two legs' mean angle could otherwise be met
                    # with one leg pulled past straight, off its planted ankle (1.2 mm at standing).
                    for s in sides:
                        flat_sq = (Vector((mid.x, y, 0.0)) + offs[s] - ankles[s]).xy.length_squared
                        if flat_sq >= sum(legs[s]) ** 2:
                            return None
                        hi = min(hi, ankles[s].z - offs[s].z + math.sqrt(sum(legs[s]) ** 2 - flat_sq))
                    if hi <= lo or flat_knee(y, lo) > goal["knee_included"]:
                        return None
                    if flat_knee(y, hi) < goal["knee_included"]:
                        return hi
                    for _ in range(50):
                        z = (lo + hi) / 2
                        lo, hi = (lo, z) if flat_knee(y, z) > goal["knee_included"] else (z, hi)
                    return (lo + hi) / 2
                if sum((Vector((mid.x, y, lo)) + offs[s] - ankles[s]).length - need[s] for s in sides) > 0:
                    return None            # this far forward or back, the legs cannot bend enough
                for _ in range(50):
                    z = (lo + hi) / 2
                    err = sum((Vector((mid.x, y, z)) + offs[s] - ankles[s]).length - need[s] for s in sides)
                    lo, hi = (lo, z) if err > 0 else (z, hi)
                return (lo + hi) / 2

            def hip_angle(y):
                z = height_for(y)
                if z is None:
                    return float("inf"), None
                total = 0.0
                for s in sides:
                    hip = Vector((mid.x, y, z)) + offs[s]
                    knee = _two_bone_knee(hip, ankles[s], legs[s][0], legs[s][1], poles[s])
                    thigh = leg_turn[f"thigh.{s.lower()}"] @ (knee - hip)     # counted from the bind pose
                    total += _deg(torso, side(thigh) if "hip_included" in flat else thigh)
                return total / 2 - goal["hip_included"], z

            # Two depths can give the same hip angle — the second with the hips far back and the
            # knees folded behind the ankles (it took frames 32 and 56 once: a thigh 29 mm through
            # the floor; with the thigh as the pole that fold was frames 18-31 of task 14, the bend
            # pole above removes it). The root nearest the last frame's answer is kept, within 15 cm
            # of the capture's own hips.
            ys = [mid.y + d for d in np.linspace(-0.15, 0.15, 61)]
            vals = [hip_angle(yy)[0] for yy in ys]
            # The coach's standing hip angle (167) is out of this figure's reach (about 158 in 3-D,
            # task 12). Aimed at anyway, the hips sat on the flat top of the angle-against-depth
            # curve, forward, and on the first frame the goal came within reach they snapped 37 mm
            # back to the other side of it (frames 5-6 and 65-66, task 13). So the goal is held
            # REACH_MARGIN below the best this frame can do: a root always exists, on the hips-back
            # side, and moves with the goal from frame to frame.
            # (a smooth minimum of the goal and that limit, over REACH_SOFT degrees: a hard one kinked
            # where the goal came within reach and the hips turned on the spot)
            finite = [v for v in vals if math.isfinite(v)]
            lift = (-REACH_SOFT * math.log1p(math.exp((REACH_MARGIN - max(finite)) / REACH_SOFT))
                    if finite else 0.0)
            vals = [v - lift for v in vals]
            near_to = last_y if last_y is not None else -1e9      # first frame: the hips-back root
            roots = []
            for a_i in range(len(ys) - 1):
                va, vb = vals[a_i], vals[a_i + 1]
                if math.isfinite(va) and math.isfinite(vb) and va * vb <= 0:
                    lo, hi = ys[a_i], ys[a_i + 1]
                    for _ in range(40):
                        m_y = (lo + hi) / 2
                        if (hip_angle(m_y)[0] - lift > 0) == (va > 0):
                            lo = m_y
                        else:
                            hi = m_y
                    roots.append((lo + hi) / 2)
            if roots:
                y = min(roots, key=lambda r: abs(r - near_to))
            else:
                finite = [(abs(v), yy) for v, yy in zip(vals, ys) if math.isfinite(v)]
                y = min(finite)[1] if finite else mid.y
            z = hip_angle(y)[1]
            if z is None:
                y, z = mid.y, mid.z
            _move_root(rig, Vector((0.0, y - mid.y, z - mid.z)))
            for s in sides:
                up, low, foot = (pose[P + s + n] for n in ("UpLeg", "Leg", "Foot"))
                l1, l2 = legs[s]
                knee = _two_bone_knee(up.head.copy(), ankles[s], l1, l2, poles[s])
                _aim_pose(up, knee - up.head)
                _aim_pose(low, ankles[s] - low.head)
                m = feet[s].to_4x4()
                m.translation = foot.head
                foot.matrix = m
                bpy.context.view_layer.update()
            if ears is not None:
                # the coach's neck angle and face direction (task 16)
                _place_head(rig, ears, want["neck_vs_torso"], want["head_pitch"])
            else:
                # The head: a neutral neck, as the coach holds it — the face pitched down by the
                # torso's lean, and never less than GAZE_DOWN. (Straightening the capture's torso to
                # the coach's lean otherwise left the head tipped chin-to-chest at the top.)
                head = pose[P + "Head"]
                face = (head.matrix.to_3x3() @ head.bone.matrix_local.to_3x3().inverted()) @ Vector((0.0, 1.0, 0.0))
                pitch = math.degrees(math.atan2(face.z, max(face.y, 1e-6)))
                aim_pitch = -max(GAZE_DOWN, goal["torso_lean"])
                nod = Matrix.Rotation(math.radians(aim_pitch - pitch) / 2, 3, "X")
                for name in ("Neck", "Head"):
                    _turn_world(pose[P + name], nod)
            got = form_metrics(rig, flat)
            for key in ("knee_included", "hip_included"):
                goal[key] += want[key] - got[key]
        got = dict(form_metrics(rig, flat))
        stance = posture_metrics(rig, ears, rest_tilts)
        got.update(stance)
        curls.append(stance["spine_curl"])
        for key in worst:
            worst[key] = max(worst[key], abs(got[key] - want[key]))
        j = _joints(rig)
        last_y = ((j["hip.left"] + j["hip.right"]) / 2).y
        _key(rig, previous, frame)
        pose[P + "Hips"].keyframe_insert("location", frame=frame)
    scene.frame_set(first)
    out = {f"worst_miss_{k}_deg": round(v, 2) for k, v in worst.items()}
    worst_curl = max(range(len(curls)), key=lambda i: abs(curls[i]))
    out["spine_curl_worst_deg"] = round(curls[worst_curl], 2)
    out["spine_curl_worst_frame"] = first + worst_curl
    return plan, out


def _bar_frame(rig, skin, drop):
    """Where `mount_barbell` would put the bar at the current frame, as a matrix in Spine2's
    posed frame (constant through the rep, as the bar is parented to Spine2)."""
    import equipment
    import exercise_stage
    height = json.loads(rig["skin_fit"])["measured"]["height"]
    world, pose = rig.matrix_world, rig.pose.bones
    left, right = (world @ pose[P + f"{s}Arm"].head for s in ("Left", "Right"))
    hips = (world @ pose[P + "LeftUpLeg"].head + world @ pose[P + "RightUpLeg"].head) / 2
    neck = world @ pose[P + "Neck"].head
    up = (neck - hips).normalized()
    lateral = right - left
    lateral = (lateral - up * lateral.dot(up)).normalized()
    forward = up.cross(lateral)
    depth = (neck - hips).length
    along = (left + right) / 2 - up * (depth * drop)
    surface = exercise_stage.back_surface(skin, along, forward)
    radius = (equipment.BARBELL_BAR_DIAMETER / 2) * height / 1.75
    point = (surface - forward * radius) if surface is not None else along - forward * depth * 0.33
    m = exercise_stage.basis_from(lateral, up).to_4x4()
    m.translation = point
    return (world @ pose[P + "Spine2"].matrix).inverted() @ m


def _grip_wrist(grip, sign, along, point, v):
    """Where the wrist goes for a hand gripping the bar as `hand_rig.solve_grip` closed it, the
    hand turned (`hand_rig.hand_turn`) so the bar runs along its knuckles and its long axis lies
    as close to the forearm `v` as it can. Returns (wrist, turn as a 3x3 numpy array)."""
    R = hand_rig.hand_turn(grip, sign, np.array(along), np.array(v))
    offset = R @ (np.array(grip["B"]) - np.array(grip["wrist"]))
    return point - Vector(offset.tolist()), R


WRIST_TOL = 3.0             # degrees: the wrist bend's weight in the arm's cost (as the other terms' 5)
WRIST_MAX = 20.0            # degrees: no arm is taken that bends the wrist more than this
# degrees: the coach's upper-arm angle's weight in a gripped arm's cost. 5 (the tolerance) while the
# back curled: the curl opened the angle 44 -> 49 through the rep, as the coach's opens 41 -> 52. With
# the back straight (task 16) the torso is rigid and at 5 the elbow sat at 48-50 all rep, failing the
# standing 41 by 2-7; at 2.5 the elbow follows him (44 / 49-51 / 52) and the wrists bend less, not more
UPPER_TOL = 2.5


def _solve_gripped_arm(S, point, along, sign, l1, l2, grip, want, hip, steps, out):
    """One arm onto a bar the hand really grips (task 13). The hand is fixed on the bar as the grip
    solve closed it; what is free is how far it is turned round the bar (the bar is round) and
    where the elbow is on the circle the arm's two lengths allow. Both are searched together, so
    the wrist stays straight: the hand's axis (wrist to middle knuckle) is held to the forearm by
    a cost on the bend (WRIST_TOL) and a limit (WRIST_MAX), beside the coach's upper-arm angle,
    elbow width and a forearm vertical from behind. (A fixed-point loop between the hand's turn and
    the elbow did not converge: the wrist sits a palm's length below the bar now, and the hand
    ended up 75-84 degrees off the forearm.) Returns (elbow, wrist, forearm direction, cost)."""
    x = np.array(along)
    x /= np.linalg.norm(x)
    kb = np.array(grip["kb"])
    target_k = -sign * x
    c = np.cross(kb, target_k)
    s_, d_ = np.linalg.norm(c), float(kb @ target_k)
    R1 = np.eye(3) if s_ < 1e-9 else hand_rig.rot(c / s_, math.atan2(s_, d_))
    w0 = np.array(grip["wrist"])
    to_bar = np.array(grip["B"]) - w0
    to_knuckle = np.array(grip["middle_mcp"]) - w0 if "middle_mcp" in grip else np.array(grip["hb"])
    to_knuckle = to_knuckle / np.linalg.norm(to_knuckle)
    P_ = np.array(point)
    Sv, Hv = np.array(S), np.array(hip)
    psi = np.linspace(0.0, 2 * math.pi, steps, endpoint=False)
    best = None
    for phi in np.radians(np.arange(0.0, 360.0, 5.0)):
        R = hand_rig.rot(target_k, phi) @ R1
        W_ = P_ - R @ to_bar
        hand_ax = R @ to_knuckle
        to = W_ - Sv
        dist = float(np.linalg.norm(to))
        if dist >= l1 + l2 - 1e-6 or dist <= abs(l1 - l2):
            continue
        u = to / dist
        a = (l1 * l1 + dist * dist - l2 * l2) / (2 * dist)
        r = math.sqrt(max(l1 * l1 - a * a, 0.0))
        e1 = np.array((0.0, 0.0, 1.0)) - u * u[2]
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(u, e1)
        E = Sv + u * a + (np.outer(np.cos(psi), e1) + np.outer(np.sin(psi), e2)) * r
        fore = W_ - E
        fn = fore / np.linalg.norm(fore, axis=1, keepdims=True)
        bend = np.degrees(np.arccos(np.clip(fn @ hand_ax, -1.0, 1.0)))
        ua = E - Sv
        down = Hv - Sv
        upper = np.degrees(np.arccos(np.clip((ua @ down) / (np.linalg.norm(ua, axis=1) * np.linalg.norm(down)), -1, 1)))
        behind = fore[:, [0, 2]]
        vert = np.degrees(np.arccos(np.clip(behind[:, 1] / np.linalg.norm(behind, axis=1), -1, 1)))
        width = (np.abs(E[:, 0]) - abs(W_[0])) / 0.02
        cost = (((upper - want["upperarm_vs_torso"]) / UPPER_TOL) ** 2 + (vert / 5.0) ** 2 + width ** 2
                + (bend / WRIST_TOL) ** 2 + 1e4 * (bend > WRIST_MAX))
        i = int(np.argmin(cost))
        if best is None or cost[i] < best[0]:
            best = (float(cost[i]), E[i], W_, fn[i], R)
    if best is None:
        # out of reach on every turn: the arm straight at the hand, as the old solve did
        R = R1
        W_ = P_ - R @ to_bar
        u = (W_ - Sv) / np.linalg.norm(W_ - Sv)
        best = (1e9, Sv + u * l1, W_, u, R)
    cost, E, W_, v, R = best
    if out is not None:
        out["turn"] = R
    return Vector(E.tolist()), Vector(W_.tolist()), Vector(v.tolist()), cost


def _solve_arm(S, bar_m, sign, g, l1, l2, shape, bar_r, want, hip, steps=72, out=None):
    """One arm onto the bar: (elbow, wrist, forearm direction, cost terms).

    The hand is on the bar `g` from its middle. With a solved grip (`shape["grip"]`, task 13) the
    wrist is where that grip puts it: a palm's length below the bar, the bar across the finger roots,
    the hand continuing the forearm. Without one (task 12), it sits behind and below the bar so the
    palm lies on the bar from behind. The elbow is on the circle the two arm lengths allow, placed
    (`steps` tries round it) where the coach's upper-arm angle, his elbow width and a forearm
    vertical from behind are best met. `out`, if given, receives the hand's turn ("turn")."""
    basis = bar_m.to_3x3().normalized()
    along, forward, up = (basis @ Vector(a) for a in ((1, 0, 0), (0, 1, 0), (0, 0, 1)))
    grip = bar_m.translation + along * (sign * g)
    v = (up + forward * 0.15).normalized()
    best = None
    gripped = shape.get("grip")
    if gripped is not None:
        return _solve_gripped_arm(S, grip, along, sign, l1, l2, gripped, want, hip, steps, out)
    for _ in range(3):
        if gripped is not None:
            wrist, turn = _grip_wrist(gripped, sign, along, grip, v)
            if out is not None:
                out["turn"] = turn
        else:
            wrist = grip - v * (PALM_ALONG * shape["length"]) - forward * (shape["palm_depth"] + bar_r + GRIP_CLEAR)
        to = wrist - S
        dist = min(to.length, l1 + l2 - 1e-6)
        u = to.normalized()
        a = (l1 * l1 + dist * dist - l2 * l2) / (2 * dist)
        r = math.sqrt(max(l1 * l1 - a * a, 0.0))
        e1 = (Vector((0.0, 0.0, 1.0)) - u * u.z).normalized()
        e2 = u.cross(e1)
        best = None
        for i in range(steps):
            phi = 2 * math.pi * i / steps
            elbow = S + u * a + (e1 * math.cos(phi) + e2 * math.sin(phi)) * r
            fore = wrist - elbow
            behind = Vector((fore.x, 0.0, fore.z))
            terms = {"upperarm_vs_torso": (_deg(elbow - S, hip - S) - want["upperarm_vs_torso"]) / 5.0,
                     "forearm_vertical": _deg(behind, Vector((0.0, 0.0, 1.0))) / 5.0,
                     "elbow_width": (abs(elbow.x) - abs(wrist.x)) / 0.02}
            cost = sum(t * t for t in terms.values())
            if best is None or cost < best[0]:
                best = (cost, elbow, wrist, terms)
        v = (best[2] - best[1]).normalized()
    return best[1], best[2], v, best[0]


# --- the Exercise Studio's own measuring (task 19) ------------------------------------------------------
# With the Studio's targets.json beside the --form-targets file (a Studio run folder), the solver picks the
# coach's positions and reads our picture with the Studio's own code (studio/blender/inspect_run.py), so it
# aims at exactly what the Studio judges. Without it (hand-made targets) the positions are found by the knee
# (`_coach_positions`) and the picture is not checked.

STUDIO_PHASE_OF = {v: k for k, v in STUDIO_PHASES.items()}     # the Studio's "start" -> "standing", ...


def _studio():
    """studio/blender/inspect_run.py, imported on first use (it imports this module)."""
    path = str(HERE.parents[1] / "studio" / "blender")
    if path not in sys.path:
        sys.path.insert(0, path)
    import inspect_run
    return inspect_run


def read_studio_targets(path, log=print):
    """The Exercise Studio's targets.json beside the --form-targets file, or None (said in the log): how the
    Studio judges - the coach's exact readings, which of them are uncertain, the side his photos were taken
    from. The hand-made targets (3d-character/reference-squat/targets.json) have none."""
    studio = Path(path).with_name("targets.json")
    why = None
    if not studio.is_file() or studio.resolve() == Path(path).resolve():
        why = "no Studio targets beside the targets file"
    else:
        try:
            t = json.loads(studio.read_text())
            if (t.get("camera") or {}).get("side") not in ("left", "right") or not isinstance(t.get("phases"), dict):
                why = "the Studio targets beside the targets file name no camera side or positions"
        except (OSError, ValueError) as e:
            why = f"the Studio targets beside the targets file can't be read ({e})"
    if why:
        log(f"form: {why}: positions by the knee, the picture not checked")
        return None
    return t


def _studio_positions(rig, studio, first, last):
    """{"standing", "mid", "bottom"}: the frames the Exercise Studio will pick for the coach's three positions,
    by its own measuring of every frame (inspect_run `measure_frames`, `pick_positions`): on the squat by the
    hip angle, never a frame that jumps. Raises the Studio's RunError when it can pick none."""
    ir = _studio()
    side = studio["camera"]["side"]
    _, _, side_normal, behind_normal, turn = ir.studio_view(rig, side)
    per_frame, joints = ir.measure_frames(rig, first, last, side, side_normal, behind_normal, turn)
    frames, driver, _, _ = ir.pick_positions(per_frame, studio, joints)
    bpy.context.scene.frame_set(first)
    return {STUDIO_PHASE_OF[p]: f for p, f in frames.items()}, {f["frame"]: f for f in per_frame}, driver


PICTURE_ROUNDS = 3          # corrections at most, each a re-solve of the legs and torso
PICTURE_DEADBAND = {"torso_lean": 2.5, "hip_included": 3.0}
                            # degrees: a row the picture reads this close is left alone, so the start, which the
                            # picture already reads like the coach, stays as it is. The hip angle's is wider: one
                            # render reads it up to 2.5 degrees apart (task 19, exp2), and the median of five
                            # still moved the start 168.0 -> 171.6 between rounds (task 19 review)
PICTURE_AIM = 1.5           # degrees: a row being corrected is aimed until the picture reads it this close
DEPTH_MARGIN = 0.8          # x the depth's tolerance: how far from the coach's the hips may sit against the knees
                            # (only when the Studio's targets carry no seat depth: `_hold`)
SEAT_HOLD = 0.02            # thigh lengths: the bottom's seat depth in our picture is held this close to the coach's
                            # (task 22; a fifth of the tolerance, about 2 degrees of hip angle)
SEAT_SLOPE = -0.008         # thigh lengths of that depth per degree of the bottom's hip-angle target (lean held): the
                            # first guess (task 20b's B: hip -12 -> 0.658, 0 -> 0.544, +8 -> 0.499); then measured
SEAT_SLOPES = (-0.02, -0.004)   # the measured slope is kept inside these: the Studio's bottom frame can shift between
                            # re-solves, and a secant through such a jump can send the target the wrong way (task 21)
SEAT_ROUNDS = 5             # re-solves at most to hold it
SEAT_STEP_MAX = 15.0        # degrees: the most the hip-angle target moves in one re-solve
DEPTH_SLOPE = -0.013        # torso lengths per degree of the bottom's hip angle (lean held): the depth's first
                            # guess (task 19, the squat: -0.063 at lean + hip 94.7 degrees, +0.040 at 86.8); then measured


def _picture_rows(studio, driver):
    """[(Studio phase, metric, the coach's value)] the Studio judges on our picture: the lean and hip angle
    where the coach's reading is certain, at a position not estimated, the middle's moment-finder left out
    (studio/measure/checks.py `_phase_rows`)."""
    from picture import PICTURE_METRICS
    rows = []
    for phase in STUDIO_PHASES.values():
        pt = studio["phases"].get(phase) or {}
        if pt.get("estimated"):
            continue
        for metric in PICTURE_METRICS:
            t = pt.get(metric)
            if not isinstance(t, dict) or t.get("uncertain") or not isinstance(t.get("value"), (int, float)):
                continue
            if phase == "middle" and metric == driver:
                continue
            rows.append((phase, metric, float(t["value"])))
    return rows


def _depth_at(rig, studio, first, last):
    """(the Studio's hip-below-knee reading at its bottom position, that frame)."""
    positions, per_frame, _ = _studio_positions(rig, studio, first, last)
    frame = positions["bottom"]
    return per_frame[frame]["flat"].get("hip_below_knee"), frame


def _hold_depth(rig, aim, studio, first, last, log, solve):
    """Keep the bottom's depth (hips against knees, the Studio's hip_below_knee, judged on the rig) within
    DEPTH_MARGIN of the tolerance of the coach's. Where the picture's corrections take it further, the
    bottom's hip-angle target gives way - not its lean - until the depth is back inside, re-solving the legs
    each time (`solve`). The lean first: it is what the user saw (him more upright than the coach), and on
    this cartoon the picture reads the bottom's thigh ~9 degrees steeper than its bones (the detector's knee
    sits low on the shin), so the picture's lean and hip angle cannot both meet the coach's with the hips
    where his are. Returns the targets used."""
    t = (studio["phases"].get("end") or {}).get("hip_below_knee")
    if not isinstance(t, dict) or not isinstance(t.get("value"), (int, float)):
        return aim
    limit = DEPTH_MARGIN * studio["tolerance"]["hip_below_knee"]
    slope, last_hip, last_depth = DEPTH_SLOPE, None, None
    for _ in range(4):
        depth, frame = _depth_at(rig, studio, first, last)
        if depth is None:
            return aim
        off = depth - t["value"]
        hip = aim["phases"]["bottom"]["hip_included"]
        if last_hip is not None and abs(hip - last_hip) > 0.1:
            slope = min((depth - last_depth) / (hip - last_hip), -0.002)     # deeper as the hip angle closes
        if abs(off) <= limit:
            return aim
        shift = -(off - math.copysign(limit, off)) / slope
        log(f"form: picture: the bottom's depth would read {depth:+.3f} (the coach {t['value']:+.3f}, frame {frame}); "
            f"its hip-angle target moved {shift:+.1f} degrees to hold it")
        last_hip, last_depth = hip, depth
        aim = json.loads(json.dumps(aim))
        aim["phases"]["bottom"]["hip_included"] += shift
        solve(aim)
    return aim


def _seat_target(studio):
    """The coach's seat below the top of his knee at the bottom (the Studio's end `seat_below_knee`, task 22), or
    None."""
    t = (studio["phases"].get("end") or {}).get("seat_below_knee")
    v = t.get("value") if isinstance(t, dict) else None
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def _seat_at(rig, skin, studio, first, last):
    """(our seat below the top of the knee at the Studio's bottom position as its stills step reads it - inspect_run
    `seat_reading`: our picture, the 3D skin when the picture gives none - that frame, which of the two). A failure
    reading either is said, never raised: the solve stands."""
    ir = _studio()
    positions, _, _ = _studio_positions(rig, studio, first, last)
    frame, side = positions["bottom"], studio["camera"]["side"]
    _, camera, side_normal, _, _ = ir.studio_view(rig, side)
    read = ir.seat_reading(rig, skin, frame, side, camera, side_normal, log=lambda _: None)
    bpy.context.scene.frame_set(first)
    if read["picture"] is not None:
        return read["picture"], frame, "picture"
    return read["skin"], frame, "skin"


def _hold(rig, skin, aim, studio, first, last, log, solve):
    """Hold the bottom where the Studio judges its depth: at the coach's seat below the top of his knee (`_hold_seat`,
    task 22), or - targets without one - its hips against its knees within the tolerance (`_hold_depth`)."""
    if _seat_target(studio) is None:
        return _hold_depth(rig, aim, studio, first, last, log, solve)
    return _hold_seat(rig, skin, aim, studio, first, last, log, solve)


def _hold_seat(rig, skin, aim, studio, first, last, log, solve):
    """Hold the bottom's seat below the top of the knee in our picture at the coach's, within SEAT_HOLD: the bottom's
    hip-angle target gives way - not its lean - re-solving the legs each time (`solve`; a secant on the measured
    slope, at most SEAT_ROUNDS times). The depth as a judge sees it, read the same way on both pictures (task 22):
    held at his, not anywhere inside the tolerance (the picture's hip angle, read through a knee landmark on the
    shin, pulled the bottom to the deep edge every time). Only our picture's reading is aimed: when only the 3D skin
    reads (its cross-check, about 0.08 below the picture on the squat), the bottom is left as solved. Returns the
    targets used."""
    want = _seat_target(studio)
    slope, last_hip, last_seen = SEAT_SLOPE, None, None
    for n in range(SEAT_ROUNDS + 1):
        seen, frame, how = _seat_at(rig, skin, studio, first, last)
        if seen is None:
            log("form: picture: the bottom's seat depth can't be read (picture or skin); the bottom is left as solved")
            return aim
        if how != "picture":        # the skin's reading is not the coach's measure: never aimed at his picture's
            log(f"form: picture: picture unreadable; depth left as solved (the 3D skin reads {seen:.3f}, the coach's "
                f"picture {want:.3f}, frame {frame})")
            return aim
        off = seen - want
        hip = aim["phases"]["bottom"]["hip_included"]
        if last_hip is not None and abs(hip - last_hip) > 0.1:
            slope = min(max((seen - last_seen) / (hip - last_hip), SEAT_SLOPES[0]), SEAT_SLOPES[1])
        if abs(off) <= SEAT_HOLD:
            return aim
        if n == SEAT_ROUNDS:
            log(f"form: picture: the bottom's seat depth reads {seen:.3f} after {n} re-solves (the coach {want:.3f})")
            return aim
        shift = max(-SEAT_STEP_MAX, min(SEAT_STEP_MAX, -off / slope))
        on = "our picture" if how == "picture" else "the 3D skin"
        log(f"form: picture: the bottom's seat depth reads {seen:.3f} thigh lengths ({on}; the coach {want:.3f}, frame "
            f"{frame}); its hip-angle target moved {shift:+.1f} degrees to hold it at his")
        last_hip, last_seen = hip, seen
        aim = json.loads(json.dumps(aim))
        aim["phases"]["bottom"]["hip_included"] += shift
        solve(aim)
    return aim


def _snapshot(rig):
    """A copy of the rig's action: the pose of every frame as solved now."""
    return rig.animation_data.action.copy()


def _restore(rig, action):
    """The rig animated by `action` again (a `_snapshot`), on the slot it was keyed on."""
    anim = rig.animation_data
    slot = anim.action_slot.identifier if getattr(anim, "action_slot", None) else None
    anim.action = action
    if slot is not None and getattr(anim, "action_slot", None) is None:
        anim.action_slot = next(s for s in action.slots if s.identifier == slot)
    bpy.context.view_layer.update()


def match_picture(rig, skin, first, last, targets, studio, folder=None, log=print):
    """The lean and hip angle aimed at what the Exercise Studio judges: our picture, read with the coach's own
    detector (task 19). The legs and torso are solved on the joints (`match_legs_torso`); this then reads our
    side view at the Studio's positions exactly as its stills step will (inspect_run `picture_at`: the same
    camera, framings and reader). A row the picture reads more than PICTURE_DEADBAND from the coach has that
    position's target moved by the miss, and the legs and torso are solved again, until the rows corrected
    read within PICTURE_AIM - at most PICTURE_ROUNDS times. The picture follows the joints almost one for one
    (tipping the trunk 10 degrees moves the picture's lean 9-10), so a round or two lands it. The bottom's
    depth is held at the coach's seat below his knee's top meanwhile (`_hold`, the hip angle giving way). The
    round whose picture reads closest to the coach - fewest rows off, then the leans, then everything - is
    kept, exactly as it was solved and read.
    The arms are not solved again: they ride the chest, and the bar with them (`match_arms` ran before).
    Returns the rounds, for the run's report."""
    import shutil
    import tempfile
    if folder is None:          # the renders are read and thrown away; the run folder keeps only the report
        folder = tempfile.mkdtemp(prefix="picture-loop-")
        try:
            return match_picture(rig, skin, first, last, targets, studio, folder, log)
        finally:
            shutil.rmtree(folder, ignore_errors=True)
    ir = _studio()
    from picture import picture_value
    side = studio["camera"]["side"]
    preview_set(560, 560)       # the stills step's own render settings
    quiet = []

    def solve(aim):
        match_legs_torso(rig, aim, first, last, skin=skin, log=quiet.append)

    keys = ("torso_lean", "hip_included")
    aim = json.loads(json.dumps(targets))
    rounds, best, corrected, held, snaps = [], None, set(), {}, []
    started = time.time()
    log("form: picture: reading our picture as the Studio will (its camera, five framings, the coach's detector)")
    aim = _hold(rig, skin, aim, studio, first, last, log, solve)     # the bottom where the Studio judges its depth
    seat_want = _seat_target(studio)
    for n in range(PICTURE_ROUNDS + 1):
        positions, per_frame, driver = _studio_positions(rig, studio, first, last)
        rows = _picture_rows(studio, driver)
        if not rows:
            log("form: picture: the Studio judges no lean or hip angle on the picture here; not checked")
            break
        _, camera, _, _, _ = ir.studio_view(rig, side)
        at = {STUDIO_PHASES[p]: f for p, f in positions.items()}
        try:
            pic = ir.picture_at(skin, at, *camera, Path(folder) / f"round-{n}")
        except Exception as e:      # noqa: BLE001 - the picture is a check; the joint solve stands
            log(f"form: picture: could not read our picture ({e}); the joint solve is kept")
            break
        misses, read = {}, {}
        for phase, metric, coach in rows:
            v = picture_value(pic[phase], "flat", metric)
            if v is not None:
                read[f"{phase} {metric}"] = round(v, 1)
                misses[(phase, metric)] = coach - v
        rig_read = {f"{p} {m}": round(per_frame[at[p]]["flat"][m], 1) for p, m, _ in rows}
        look = None
        if seat_want is not None:       # the depth as the Studio judges it: the seat against the knee's top (task 22)
            seen, _, how = _seat_at(rig, skin, studio, first, last)
            look = {"seat_below_knee": None if seen is None else round(seen, 3), "read_on": how,
                    "coach": round(seat_want, 3)}
        rounds.append({"targets": {p: {k: round(aim["phases"][p][k], 1) for k in keys} for p in STUDIO_PHASES},
                       "frames": at, "picture": read, "rig": rig_read, "depth": look,
                       "misses": {f"{p} {m}": round(v, 1) for (p, m), v in misses.items()}})
        log(f"form: picture round {n} (frames {', '.join(f'{p} {f}' for p, f in at.items())}): " + ", ".join(
            f"{p} {m} picture {read.get(f'{p} {m}', 'not read')} (rig {rig_read[f'{p} {m}']}, coach {c:.1f})"
            for p, m, c in rows) + ("" if look is None else f", end seat depth {look['seat_below_knee']} "
                                                            f"({look['read_on']}; coach {look['coach']})"))
        # the round kept: fewest rows off the coach by more than the tolerance, then the worst lean, then all
        # the misses together (the bottom's hip angle, held by the depth, reads alike in every round)
        tol = studio["tolerance"]["angles_deg"]
        score = (sum(abs(v) > tol for v in misses.values()),
                 round(max((abs(v) for (p, m), v in misses.items() if m == "torso_lean"), default=0.0), 1),
                 round(sum(v * v for v in misses.values()), 1))
        snaps.append(_snapshot(rig))
        if best is None or score < best[0]:
            best = (score, json.loads(json.dumps(aim)), n)
        # a row the depth has held back is not corrected further that way (the depth would only undo it again)
        need = {r: v for r, v in misses.items() if not (r in held and v * held[r] > 0)
                and abs(v) > (PICTURE_AIM if r in corrected else PICTURE_DEADBAND[r[1]])}
        if not need or n == PICTURE_ROUNDS:
            break
        new = json.loads(json.dumps(aim))
        for (phase, metric), miss in need.items():
            new["phases"][STUDIO_PHASE_OF[phase]][metric] += miss
        corrected |= set(need)
        asked = new["phases"]["bottom"]["hip_included"]
        solve(new)
        new = _hold(rig, skin, new, studio, first, last, log, solve)
        bottom_hip = ("end", "hip_included")
        if bottom_hip in need and abs(new["phases"]["bottom"]["hip_included"] - asked) > 0.1:
            held[bottom_hip] = math.copysign(1.0, need[bottom_hip])
            log("form: picture: the depth holds the bottom's hip angle; it is not corrected further that way")
        if all(abs(new["phases"][p][k] - aim["phases"][p][k]) < 0.5 for p in STUDIO_PHASES for k in keys):
            log("form: picture: the depth holds the targets where they were; no closer is possible")
            _restore(rig, snaps[-1])
            break
        aim = new
    if best is not None:
        _restore(rig, snaps[best[2]])
        aim = best[1]
        log(f"form: picture: kept round {best[2]} as it was solved and read: targets " + ", ".join(
            f"{p} lean {aim['phases'][p]['torso_lean']:.1f} hip {aim['phases'][p]['hip_included']:.1f}"
            for p in STUDIO_PHASES) + f"; {best[0][0]} row(s) off by more than the tolerance, the worst lean "
            f"miss {best[0][1]:.1f} degrees; {len(rounds)} reading(s) in {time.time() - started:.0f} s")
    keep = {rig.animation_data.action.name}
    for a in snaps:
        if a.name not in keep and a.users == 0:
            bpy.data.actions.remove(a)
    bpy.context.scene.frame_set(first)
    return {"rounds": rounds, "kept": None if best is None else best[2], "seconds": round(time.time() - started, 1),
            "targets": {p: {k: round(aim["phases"][p][k], 1) for k in keys} for p in STUDIO_PHASES}}


def studio_arm_rows(rig, first, last, studio, chosen_at=None, log=print):
    """The arm rows as the Exercise Studio will read them, at the positions it will pick now that the picture
    loop has moved the legs and torso (re-picked with its own picker; the arm search chose its bar and grip at
    `chosen_at`, the positions before): the upper arm and elbow in 3-D on its side camera's arm, the grip and
    elbow widths on its behind camera, each against the coach's. Logged; returned for the run's report."""
    positions, per_frame, _ = _studio_positions(rig, studio, first, last)
    tol_a, tol_r = studio["tolerance"]["angles_deg"], studio["tolerance"]["ratios"]
    rows = {}
    for p, frame in positions.items():
        phase, f = STUDIO_PHASES[p], per_frame[frame]
        rows[phase] = {"frame": frame, **{k: round(f["3d"][k], 1) for k in ("upperarm_vs_torso", "elbow_included")},
                       **{k: round(f["flat"][k], 2) for k in ("grip_ratio", "elbow_width_ratio")}}

    def line(k, tol, digits):
        def coach(ph):
            t = (studio["phases"].get(ph) or {}).get(k) if k not in ("grip_ratio", "elbow_width_ratio") else \
                (studio.get("widths") or {}).get(k)
            return t.get("value") if isinstance(t, dict) else None
        got = [rows[ph][k] for ph in STUDIO_PHASES.values()]
        want = [coach(ph) for ph in STUDIO_PHASES.values()]
        met = sum(1 for g, w in zip(got, want) if w is not None and abs(g - w) <= tol)
        return (f"{' / '.join(f'{g:.{digits}f}' for g in got)} (coach "
                f"{' / '.join('-' if w is None else f'{w:.{digits}f}' for w in want)}; {met} of 3 within tolerance)")

    before = "" if not chosen_at else " (the arm search chose at " + ", ".join(
        f"{STUDIO_PHASES[p]} {f}" for p, f in chosen_at.items()) + ")"
    log("form: the arms at the Studio's positions after the picture loop, " + ", ".join(
        f"{ph} {r['frame']}" for ph, r in rows.items()) + before + ": upper arm " + line("upperarm_vs_torso", tol_a, 1)
        + ", elbow " + line("elbow_included", tol_a, 1) + ", grip " + line("grip_ratio", tol_r, 2)
        + ", elbow width " + line("elbow_width_ratio", tol_r, 2))
    bpy.context.scene.frame_set(first)
    return rows


ARM_MARGIN = 0.8            # x the tolerance: an arm row counts as met only this far inside it (not on a knife-edge)


def match_arms(rig, skin, first, last, plan, targets, log=print, force=None, studio=None, trace=None):
    """The grip on the bar, the bar's height on the back, and every frame's arms (see the section
    above). Mounts the bar (returns it) and re-keys the arms, hands and fingers. With `studio` (the Studio's
    targets, `read_studio_targets`) the coach's positions are the ones the Studio will pick, and the widths
    are read on its behind camera (task 19). `trace`, a list, receives every candidate's score."""
    import equipment
    scene = bpy.context.scene
    fit = json.loads(rig["skin_fit"])
    hands = fit["hands"]
    # the solved grips (task 13): with them the wrist sits where the closed hand puts it
    grips = json.loads(rig.get("hand_grip", "{}"))
    shapes = {s: dict(hands[s], grip=grips.get(s)) for s in ("left", "right")}
    height = fit["measured"]["height"]
    bar_r = (equipment.BARBELL_BAR_DIAMETER / 2) * height / 1.75
    pose = rig.pose.bones
    sides = (("Left", -1.0), ("Right", 1.0))
    lens = {s: (pose[P + s + "Arm"].length, pose[P + s + "ForeArm"].length) for s, _ in sides}
    arms_t = targets["arms_all_phases"]
    scene.frame_set(first)
    frames = list(range(first, last + 1))
    sample = frames[::4] + [frames[-1]]
    positions, behind_normal = None, None
    if studio is not None:
        try:
            positions = _studio_positions(rig, studio, first, last)[0]
            behind_normal = Vector(_studio().studio_view(rig, studio["camera"]["side"])[3].tolist())
            log(f"form: the coach's positions as the Studio picks them: " + ", ".join(f"{p} {f}" for p, f in positions.items()))
        except Exception as e:      # noqa: BLE001 - said, and the knee picks them as before
            log(f"form: the Studio's positions could not be picked ({e}); found by the knee")
    if positions is None:
        positions = _coach_positions(targets, plan, frames)
    state = {}
    for frame in sorted(set(sample) | set(positions.values())):
        scene.frame_set(frame)
        j = _joints(rig)
        state[frame] = (j, (rig.matrix_world @ pose[P + "Spine2"].matrix).copy())
    j = state[sample[-1]][0]
    sw = (Vector((j["shoulder.right"].x, 0, j["shoulder.right"].z))
          - Vector((j["shoulder.left"].x, 0, j["shoulder.left"].z))).length
    scene.frame_set(first)
    # Scored by how many of the arm metrics pass on the sampled frames (both sides), ties broken by
    # the summed error in tolerance units — not by the error alone: the coach's elbow angle is out of
    # this arm's reach with the bar on its back (see the task 12 report), and as a weighted error it
    # dragged the grip and bar wherever it shrank by a degree, failing metrics that could pass.
    tol_a, tol_r = targets["tolerance"]["angles_deg"], targets["tolerance"]["ratios"]
    torso_turn = _torso_turn(rig)   # the upper arm is judged against the torso line counted from the rest pose
    cands = []
    # A hand that really closes on the bar holds its wrist a palm's length below it (task 13), so
    # the bar may ride higher on the back than task 12's (at most 0.10 of the torso above the
    # shoulder joints): up to 0.20, as long as it stays clear of the head (HEAD_CLEAR).
    # Every 0.01 of the torso from 0.20 to 0.10 above the shoulder joints (task 18): with the torso counted
    # from the bind pose the upper arm passes at the coach's three positions only in a narrow band of bar
    # heights (0.14-0.145 on the squat: 0.13 bends the left wrist 22 degrees, 0.15 lifts the standing
    # elbow 6 degrees past the coach's), which the old steps (0.16, 0.13) stepped over.
    drops = (-0.20, -0.19, -0.18, -0.17, -0.16, -0.15, -0.14, -0.13, -0.12, -0.11, -0.10, -0.06, -0.03, 0.0,
             0.03, 0.06) if grips else (-0.10, -0.06, -0.03, 0.0, 0.03, 0.06)
    clear = _head_clearance(rig, skin, state, sample, drops) if grips else {}
    for drop in drops:
        if clear.get(drop, 1.0) < HEAD_CLEAR:
            log(f"form: bar {drop:+.2f} refused: {clear[drop] * 1000:.1f} mm from the head")
            continue
        local = _bar_frame(rig, skin, drop)
        for ratio in (1.8, 1.87, 1.9, 1.93, 2.0, 2.07):
            g = ratio * sw / 2
            passes, err, grip_ok, wrist_ok = 0, 0.0, True, True
            for frame in sample:
                j, spine2 = state[frame]
                bar_m = spine2 @ local
                want = _targets_at(targets, plan[frame])
                elbows, wrists = {}, {}
                for side, sign in sides:
                    S = j[f"shoulder.{side.lower()}"]
                    hip = _torso_hip(S, j[f"hip.{side.lower()}"], torso_turn)
                    l1, l2 = lens[side]
                    elbow, wrist, v, c = _solve_arm(S, bar_m, sign, g, l1, l2, shapes[side.lower()], bar_r, want, hip, steps=36)
                    elbows[side], wrists[side] = elbow, wrist
                    wrist_ok = wrist_ok and (c < 1e3 or not grips)   # a gripped arm's cost carries 1e4 past WRIST_MAX
                    terms = [(_deg(S - elbow, wrist - elbow) - want["elbow_included"]) / tol_a,
                             (_deg(elbow - S, hip - S) - want["upperarm_vs_torso"]) / tol_a,
                             ((S.z - elbow.z) / sw - 0.5) / tol_r,
                             ((wrist.z - S.z) / sw - 0.04) / 0.16,
                             _deg(Vector((wrist.x - elbow.x, 0, wrist.z - elbow.z)), Vector((0, 0, 1))) / tol_a]
                    passes += sum(1 for t in terms if abs(t) <= 1.0)
                    err += sum(min(abs(t), 6.0) for t in terms)
                ew, gw = _grip_widths(j, elbows, wrists, behind_normal)
                # the grip counts three times: "hands right beside the shoulders" was the user's
                # first complaint, and the width is what the eye reads first from behind
                for got_r, want_r, weight in ((ew, arms_t["elbow_width_ratio"], 2),
                                              (gw, arms_t["grip_ratio"], 6)):
                    passes += weight * (abs(got_r - want_r) <= tol_r)
                    err += weight * min(abs(got_r - want_r) / tol_r, 6.0)
                grip_ok = grip_ok and abs(gw - arms_t["grip_ratio"]) <= ARM_MARGIN * tol_r
            # a grip inside the tolerance at EVERY sampled frame (with a margin: the sampled frames
            # are every fourth) outranks any number of other passes
            # ...and with a gripped bar, a wrist kept within WRIST_MAX on every sampled frame first.
            # Then (task 18) the Exercise Studio's own rows at the coach's three positions
            # (`_studio_arm_rows`): the upper arm against the torso first — the arm's side-view posture at
            # each position, which only a narrow band of bar heights meets with the torso counted from the
            # bind pose (higher bars lift the elbows past the coach's, lower ones bend the wrist past
            # WRIST_MAX) — then all eight arm rows, before the samples' proxies. Ranked by the rows alone,
            # a bar 0.18 above the shoulders met the grip and one upper arm (3 rows), tying 0.14 with all
            # three upper arms, and the proxies picked the high bar
            at = {}         # the coach's positions, solved as the final pass solves every frame (72 steps)
            for frame in set(positions.values()):
                j, spine2 = state[frame]
                want = _targets_at(targets, plan[frame])
                at[frame] = ({}, {})
                for side, sign in sides:
                    S = j[f"shoulder.{side.lower()}"]
                    at[frame][0][side], at[frame][1][side], _, _ = _solve_arm(
                        S, spine2 @ local, sign, g, *lens[side], shapes[side.lower()], bar_r, want,
                        _torso_hip(S, j[f"hip.{side.lower()}"], torso_turn))
            rows_at = _studio_arm_rows(state, at, positions, targets, torso_turn, behind_normal)
            # the grip first, at the Studio's positions too (the user's first complaint was the hands; when the
            # grip and the upper arm cannot both be met, the grip is kept: task 19's ruling), then the upper-arm
            # rows met with a margin (ARM_MARGIN) and the worst upper-arm miss, then all the rows so met
            grip_ok = grip_ok and rows_at["grip"] <= ARM_MARGIN * tol_r
            cands.append({"drop": drop, "ratio": ratio,
                          "score": list(_arm_score(wrist_ok, grip_ok, rows_at, passes, err)), **rows_at})
    if trace is not None:
        trace.extend(cands)
    chosen = _arm_choice(cands)
    (_, grip_ok, upper, _, rows, _, _), drop, ratio, rows_at = chosen["score"], chosen["drop"], chosen["ratio"], chosen
    if force is not None:
        drop, ratio = force
    log(f"form: bar {drop:+.2f} of the torso below the shoulder joints, grip {ratio:.2f} x shoulder-joint width"
        + ("" if force is not None else
           f" (at the coach's positions, met within {ARM_MARGIN:g} of the tolerance: the upper arm at {upper} of 3, "
           f"{rows} of the Studio's 8 arm rows; the grip reads {rows_at['grip_read']:.2f} at worst, the upper arm "
           f"misses by {rows_at['worst_upper'] * tol_a:.1f} degrees at worst)"))
    trade = None if force is not None else _grip_trade_note(chosen, cands)
    if trade:
        log(trade)
    scene.frame_set(first)
    bar = mount_barbell(rig, skin, drop=drop)[0]
    g = ratio * sw / 2
    rest = {s: _palm_rest(rig, s) for s, _ in sides}
    rest_frame = {s: _frame(*rest[s]) for s in rest}
    fingers = {s: [P + s + f"Hand{f}{k}" for f in ("Index", "Middle", "Ring", "Pinky") for k in (1, 2, 3)]
               for s, _ in sides}
    curl = {n: Quaternion(_curl_axis(rig, n, rest[s][1]), math.radians(FINGER_CURL))
            for s, _ in sides for n in fingers[s]}
    if grips:
        # every finger and the thumb, folded as the grip solve closed them; constant through the rep
        basis = {}
        for s, _ in sides:
            basis.update(hand_rig.pose_basis(rig, s, grips[s.lower()]["rotations"]))
        fingers = {s: [n for n in hand_rig.hand_bones(s)[1:]] for s, _ in sides}
        curl = basis
    previous, short = {}, 0.0
    names = {P + s + n for s, _ in sides for n in ("Arm", "ForeArm", "Hand")} | {n for s, _ in sides for n in fingers[s]}
    for frame in frames:
        scene.frame_set(frame)
        j = _joints(rig)
        bar_m = rig.matrix_world.inverted() @ bar.matrix_world
        want = _targets_at(targets, plan[frame])
        forward = (bar_m.to_3x3().normalized() @ Vector((0.0, 1.0, 0.0)))
        for side, sign in sides:
            arm, fore, hand = (pose[P + side + n] for n in ("Arm", "ForeArm", "Hand"))
            S = arm.head.copy()
            l1, l2 = lens[side]
            placed = {}
            elbow, wrist, v, _ = _solve_arm(S, bar_m, sign, g, l1, l2, shapes[side.lower()], bar_r, want,
                                            _torso_hip(S, j[f"hip.{side.lower()}"], torso_turn), out=placed)
            short = max(short, (wrist - S).length - (l1 + l2))
            _aim_pose(arm, elbow - S)
            _aim_pose(fore, wrist - fore.head)
            if "turn" in placed:
                turn = Matrix(placed["turn"].tolist())
            else:
                turn = _frame(v, forward) @ rest_frame[side].inverted()
            m = (turn @ hand.bone.matrix_local.to_3x3()).to_4x4()
            m.translation = hand.head
            hand.matrix = m
            bpy.context.view_layer.update()
            for n in fingers[side]:
                pose[n].rotation_mode = "QUATERNION"
                pose[n].rotation_quaternion = curl[n]
            bpy.context.view_layer.update()
        _key(rig, previous, frame, names)
    scene.frame_set(first)
    out = {"bar_drop": drop, "grip_ratio_chosen": ratio, "reach_short_mm": max(short, 0.0) * 1000,
           "positions": dict(positions)}
    if grips:
        out["bar_head_clearance_mm"] = {f"{d:+.2f}": round(c * 1000, 1) for d, c in clear.items()}
    return bar, out


HEAD_CLEAR = 0.002          # m: the least gap the bar keeps from the head (hair included) through the rep


def _coach_positions(targets, plan, frames):
    """{"standing", "mid", "bottom"}: the clip's frames standing in for the coach's three positions, by the
    plan's knee: the first frame within half a degree of the straightest (the Studio's start is the rep's
    first), the deepest, and between them the one nearest the coach's middle knee (on the way down)."""
    top = max(plan[f] for f in frames)
    stand = min(f for f in frames if plan[f] >= top - 0.5)
    bottom = min(frames, key=lambda f: (plan[f], f))
    lo, hi = sorted((stand, bottom))
    mid_knee = targets["phases"]["mid"]["knee_included"]
    mid = min((f for f in frames if lo <= f <= hi), key=lambda f: (abs(plan[f] - mid_knee), f))
    return {"standing": stand, "mid": mid, "bottom": bottom}


def _grip_widths(j, elbows, wrists, behind_normal=None):
    """(elbow width, grip width) over this frame's OWN shoulder width, seen from behind (the Studio's behind camera
    when `behind_normal` is given), as the Studio reads each position. Task 19: the standing shoulder width as the
    scale read 1.80 where the Studio read 1.67 at the bottom (the shoulders draw in as the arms hold the bar)."""
    own = _width(j["shoulder.right"] - j["shoulder.left"], behind_normal)
    return (_width(elbows["Right"] - elbows["Left"], behind_normal) / own,
            _width(wrists["Right"] - wrists["Left"], behind_normal) / own)


def _arm_score(wrist_ok, grip_ok, rows_at, passes, err):
    """An arm candidate's rank, best highest: the wrists within WRIST_MAX first; then the grip within
    ARM_MARGIN of the tolerance on every sampled frame and at the Studio's positions (the user's first
    complaint was the hands: when the grip and the upper arm can't both be met the grip is kept, task 19's
    ruling); then the upper-arm rows met within ARM_MARGIN, the worst upper-arm miss, all the rows so met; then
    the sampled frames' proxies (`passes`, `err`)."""
    return (bool(wrist_ok), bool(grip_ok), rows_at["upper"], -round(rows_at["worst_upper"], 2), rows_at["rows"],
            passes, -err)


def _arm_choice(cands):
    """The candidate (match_arms' trace entries: {"drop", "ratio", "score", ...}) ranked highest by `_arm_score`."""
    return max(cands, key=lambda c: tuple(c["score"]))


def _grip_trade_note(chosen, cands):
    """The log's words when the grip was kept over the upper arm: the chosen candidate meets the grip but not
    all three upper-arm rows, while another candidate (wrists within limits) meets all three without the grip.
    None otherwise."""
    ok_w, ok_g, upper = chosen["score"][0], chosen["score"][1], chosen["score"][2]
    if not (ok_w and ok_g) or upper >= 3:
        return None
    others = [c for c in cands if c["score"][0] and not c["score"][1] and c["score"][2] >= 3]
    if not others:
        return (f"form: the grip is kept; the upper arm is met at {upper} of 3 positions within {ARM_MARGIN:g} of the "
                f"tolerance (no bar height that keeps the wrists within {WRIST_MAX:g} degrees meets all three so)")
    return (f"form: the grip is kept over the upper arm (task 19's ruling: the grip first): {len(others)} other "
            f"candidate(s) meet the upper arm at all 3 positions within {ARM_MARGIN:g} of the tolerance but not the "
            f"grip; this one meets it at {upper} of 3")


def _width(v, normal=None):
    """A width seen from behind: `v` on the plane square to `normal` (the Studio's behind camera's axis), or
    on the x-z plane (the house figure seen from behind) without one."""
    if normal is None:
        return Vector((v.x, 0.0, v.z)).length
    n = normal.normalized()
    return (v - n * v.dot(n)).length


def _studio_arm_rows(state, at, positions, targets, torso_turn, behind_normal=None):
    """The Exercise Studio's arm rows for an arm candidate at the coach's three positions
    (studio/measure/checks.py, measured as studio/measure/metrics.py does): the upper arm against the torso
    line counted from the bind pose and the elbow angle, in 3-D, at each position, on the arm the Studio
    judges - the one on its side camera's side (the targets' "camera_side"; without one, both arms' mean,
    as `form_metrics` reads them); the grip and elbow widths seen from behind (the Studio's behind camera
    when `behind_normal` is given) against that frame's own shoulder width, at the worst of the three
    positions (one row each).
    Returns {"upper": upper-arm rows met within ARM_MARGIN of the tolerance, "rows": all rows met so,
    "worst_upper": the worst upper-arm miss in tolerances, "grip": the worst grip miss (ratio),
    "grip_read": the grip read there}. `at`: {frame: (elbows, wrists)} by "Left"/"Right"."""
    tol_a, tol_r = targets["tolerance"]["angles_deg"], targets["tolerance"]["ratios"]
    arms_t = targets["arms_all_phases"]
    camera = targets.get("camera_side")
    upper, rows, worst_upper, widths = 0, 0, 0.0, {"grip_ratio": [], "elbow_width_ratio": []}
    for phase, frame in positions.items():
        j = state[frame][0]
        elbows, wrists = at[frame]
        want = targets["phases"][phase]
        judged = [camera.capitalize()] if camera in ("left", "right") else ["Left", "Right"]
        ua_got, el_got = [], []
        for side in judged:
            S = j[f"shoulder.{side.lower()}"]
            hip = _torso_hip(S, j[f"hip.{side.lower()}"], torso_turn)
            ua_got.append(_deg(elbows[side] - S, hip - S))
            el_got.append(_deg(S - elbows[side], wrists[side] - elbows[side]))
        ua_miss = abs(sum(ua_got) / len(ua_got) - want["upperarm_vs_torso"]) / tol_a
        el_miss = abs(sum(el_got) / len(el_got) - want["elbow_included"]) / tol_a
        worst_upper = max(worst_upper, ua_miss)
        upper += ua_miss <= ARM_MARGIN
        rows += (ua_miss <= ARM_MARGIN) + (el_miss <= ARM_MARGIN)
        sw = _width(j["shoulder.right"] - j["shoulder.left"], behind_normal)
        widths["grip_ratio"].append(_width(wrists["Right"] - wrists["Left"], behind_normal) / sw)
        widths["elbow_width_ratio"].append(_width(elbows["Right"] - elbows["Left"], behind_normal) / sw)
    worst = {}
    for key, got in widths.items():
        worst[key] = max(got, key=lambda g: abs(g - arms_t[key]))
        rows += abs(worst[key] - arms_t[key]) <= ARM_MARGIN * tol_r
    return {"upper": upper, "rows": rows, "worst_upper": worst_upper,
            "grip": abs(worst["grip_ratio"] - arms_t["grip_ratio"]), "grip_read": worst["grip_ratio"]}


def _head_clearance(rig, skin, state, sample, drops):
    """For each candidate bar height, the closest the bar's surface comes to the head's skin (all of
    it above the head joint, hair included) over three of the sampled frames: the first, the
    deepest and the last."""
    import equipment
    scene = bpy.context.scene
    height = json.loads(rig["skin_fit"])["measured"]["height"]
    bar_r = (equipment.BARBELL_BAR_DIAMETER / 2) * height / 1.75
    hg = skin.vertex_groups[P + "Head"].index
    head = np.array([any(g.group == hg and g.weight > 0.5 for g in v.groups) for v in skin.data.vertices])
    deepest = min(sample, key=lambda f: state[f][0]["hip.left"].z)
    frames = sorted({sample[0], deepest, sample[-1]})
    first = scene.frame_current
    locals_ = {d: _bar_frame(rig, skin, d) for d in drops}
    out = {d: 1.0 for d in drops}
    for frame in frames:
        scene.frame_set(frame)
        pts = skin_wrap.evaluated(skin)[0][head]
        spine2 = rig.matrix_world @ rig.pose.bones[P + "Spine2"].matrix
        for d, local in locals_.items():
            m = spine2 @ local
            c = np.array(m.translation)
            x = np.array((m.to_3x3() @ Vector((1.0, 0.0, 0.0))).normalized())
            rel = pts - c
            radial = np.linalg.norm(rel - np.outer(rel @ x, x), axis=1)
            out[d] = min(out[d], float(radial.min() - bar_r))
    scene.frame_set(first)
    return out


POSTURE_TOL = 10.0          # degrees: the Studio's tolerance on the back's curl and on the neck angle


def form_report(rig, first, last, targets, skin=None):
    """targets vs ours at standing (straightest knee), bottom (deepest) and mid (the descending
    frame whose knee is nearest the coach's mid knee), every metric, pass/fail on the tolerance;
    the leg angles in the measure the targets were taken in (`_flat_legs`). With `skin`, the posture
    too: the back's curl (straight within POSTURE_TOL), and the coach's neck angle and face direction
    when the targets carry them (the face is shown, as the Studio shows it)."""
    scene = bpy.context.scene
    flat = _flat_legs(targets)
    rest_tilts = _spine_tilts(rig, rest=True)
    ears = ears_local(rig, skin) if skin is not None else None
    rows = {}
    for frame in range(first, last + 1):
        scene.frame_set(frame)
        rows[frame] = dict(form_metrics(rig, flat))
        if skin is not None:
            rows[frame].update(posture_metrics(rig, ears, rest_tilts))
    scene.frame_set(first)
    frames = sorted(rows)
    stand = max(frames, key=lambda f: rows[f]["knee_included"])
    bottom = min(frames, key=lambda f: rows[f]["knee_included"])
    mid_knee = targets["phases"]["mid"]["knee_included"]
    descent = [f for f in frames if stand <= f <= bottom] or frames
    mid = min(descent, key=lambda f: abs(rows[f]["knee_included"] - mid_knee))
    tol_a, tol_r = targets["tolerance"]["angles_deg"], targets["tolerance"]["ratios"]
    table = []
    for phase, frame in (("standing", stand), ("mid", mid), ("bottom", bottom)):
        for key in FORM_KEYS:
            t = targets["phases"][phase][key]
            ours = rows[frame][key]
            table.append({"phase": phase, "frame": frame, "metric": key, "target": t,
                          "ours": round(ours, 1), "diff": round(ours - t, 1), "pass": abs(ours - t) <= tol_a})
        arms = targets["arms_all_phases"]
        for key in ("grip_ratio", "elbow_width_ratio"):
            ours = rows[frame][key]
            table.append({"phase": phase, "frame": frame, "metric": key, "target": arms[key],
                          "ours": round(ours, 2), "diff": round(ours - arms[key], 2),
                          "pass": abs(ours - arms[key]) <= tol_r})
        table.append({"phase": phase, "frame": frame, "metric": "wrist_above_shoulder_sw",
                      "target": "about 0 (slightly above)", "ours": round(rows[frame]["wrist_above_shoulder_sw"], 2),
                      "pass": -0.12 <= rows[frame]["wrist_above_shoulder_sw"] <= 0.2})
        table.append({"phase": phase, "frame": frame, "metric": "elbow_below_shoulder_sw",
                      "target": "~0.5", "ours": round(rows[frame]["elbow_below_shoulder_sw"], 2),
                      "pass": abs(rows[frame]["elbow_below_shoulder_sw"] - 0.5) <= tol_r})
        table.append({"phase": phase, "frame": frame, "metric": "forearm_from_vertical_behind",
                      "target": 0, "ours": round(rows[frame]["forearm_from_vertical_behind"], 1),
                      "pass": rows[frame]["forearm_from_vertical_behind"] <= tol_a})
        if "spine_curl" in rows[frame]:
            table.append({"phase": phase, "frame": frame, "metric": "spine_curl", "target": 0,
                          "ours": round(rows[frame]["spine_curl"], 1),
                          "pass": abs(rows[frame]["spine_curl"]) <= POSTURE_TOL})
        for key in POSTURE_KEYS:
            t = targets["phases"][phase].get(key)
            if t is None or key not in rows[frame]:
                continue
            table.append({"phase": phase, "frame": frame, "metric": key, "target": round(t, 1),
                          "ours": round(rows[frame][key], 1), "diff": round(rows[frame][key] - t, 1),
                          "pass": abs(rows[frame][key] - t) <= POSTURE_TOL})
    table.append({"phase": "bottom", "frame": bottom, "metric": "hip_vs_knee_mm", "target": "about 0 (parallel)",
                  "ours": round(rows[bottom]["hip_vs_knee_mm"], 1),
                  "pass": abs(rows[bottom]["hip_vs_knee_mm"]) <= 0.03 * json.loads(rig["skin_fit"])["measured"]["height"] * 1000})
    return {"frames": {"standing": stand, "mid": mid, "bottom": bottom}, "table": table}


def _rest_points(rig, skin):
    """The skin at the rig's rest (= the skin's bind pose), the action set aside meanwhile."""
    anim = rig.animation_data
    action = anim.action if anim else None
    held = {pb.name: (pb.rotation_quaternion.copy(), pb.location.copy()) for pb in rig.pose.bones}
    if anim:
        anim.action = None
    for pb in rig.pose.bones:
        pb.rotation_quaternion = (1, 0, 0, 0)
        pb.location = (0, 0, 0)
    bpy.context.view_layer.update()
    rest = skin_wrap.evaluated(skin)[0]
    for pb in rig.pose.bones:
        pb.rotation_quaternion, pb.location = held[pb.name]
    if anim:
        anim.action = action
    bpy.context.view_layer.update()
    return rest


def clip_deformation(rig, skin, first, last):
    """Played back frame by frame: crack width on coincident-and-unconnected skin pairs (worst over
    the clip) and edge growth against the bind (`skin_wrap.deformation`), with the deepest frame —
    the lowest hips — reported in full: for a squat that is the bottom, where the crotch web is."""
    height = json.loads(rig["skin_fit"])["measured"]["height"]
    rest = _rest_points(rig, skin)
    scene = bpy.context.scene
    hips = rig.pose.bones[P + "Hips"]
    worst = {"crack_max_mm": 0.0, "max_growth_mm": 0.0, "growth_frame": None}
    deepest, depth_z = None, None
    for frame in range(first, last + 1):
        scene.frame_set(frame)
        posed = skin_wrap.evaluated(skin)[0]
        d = skin_wrap.deformation(skin, rest, posed, height)
        worst["crack_max_mm"] = max(worst["crack_max_mm"], d["crack_max_mm"])
        if d["max_growth_mm"] > worst["max_growth_mm"]:
            worst["max_growth_mm"], worst["growth_frame"] = d["max_growth_mm"], frame
        z = (rig.matrix_world @ hips.head).z
        if depth_z is None or z < depth_z:
            depth_z, deepest = z, (frame, d)
    scene.frame_set(first)
    frame, d = deepest
    worst["coincident_unconnected_pairs"] = d["coincident_unconnected_pairs"]
    worst["deepest_frame"] = frame
    worst["deepest"] = {k: d[k] for k in ("max_growth_mm", "max_growth_pct", "edges_over_1pct",
                                          "edges_over_0_5pct", "worst_edge_at", "worst_edge_posed_at",
                                          "worst_edge_rest_mm", "crack_max_mm", "skin_max_move_mm")}
    return worst


# --- fast preview --------------------------------------------------------------------------------
#
# EEVEE at ~560 px with three area lights and the house teal, for judging motion (Workbench is
# not in this Blender's engine list). Camera azimuths are `exercise_stage.orbit`'s: 0 in front,
# +90 the figure's left.

BACKDROP = (0.70, 0.80, 0.79, 1.0)


def _flat_material(name, colour, roughness=0.6):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = colour
    bsdf.inputs["Roughness"].default_value = roughness
    mat.diffuse_color = colour
    return mat


def preview_set(width=560, height=560, samples=16):
    """Floor, world, lights and EEVEE settings. Returns the scene."""
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = samples
    scene.render.resolution_x, scene.render.resolution_y = width, height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    world = scene.world or bpy.data.worlds.new("world")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = BACKDROP
    bg.inputs["Strength"].default_value = 0.8
    if "floor" not in bpy.data.objects:
        bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 0))
        floor = bpy.context.object
        floor.name = "floor"
        floor.data.materials.append(_flat_material("floor", (0.60, 0.72, 0.70, 1.0), 0.9))
    for name, location, energy in (("key", (-2.0, 3.0, 3.2), 380), ("fill", (2.6, 2.2, 1.8), 160),
                                   ("rim", (0.5, -2.8, 2.6), 200)):
        light = bpy.data.objects.get(name)
        if light is None:
            bpy.ops.object.light_add(type="AREA", location=location)
            light = bpy.context.object
            light.name = name
        light.data.energy = energy
        light.data.size = 2.5
        light.rotation_mode = "QUATERNION"
        light.rotation_quaternion = (Vector((0, 0, 0.5)) - Vector(location)).to_track_quat("-Z", "Y")
    return scene


def orbit_lights(azimuth):
    """Turn the three lamps about Z so they keep their place relative to a camera at `azimuth`."""
    turn = Matrix.Rotation(math.radians(azimuth), 3, "Z")
    for name, location in (("key", (-2.0, 3.0, 3.2)), ("fill", (2.6, 2.2, 1.8)),
                           ("rim", (0.5, -2.8, 2.6))):
        light = bpy.data.objects.get(name)
        if light is None:
            continue
        light.location = turn @ Vector(location)
        light.rotation_quaternion = (Vector((0, 0, 0.5)) - light.location).to_track_quat("-Z", "Y")


def camera_on(centre, azimuth, elevation, distance, lens=50.0):
    camera = bpy.data.objects.get("preview_cam")
    if camera is None:
        data = bpy.data.cameras.new("preview_cam")
        camera = bpy.data.objects.new("preview_cam", data)
        bpy.context.scene.collection.objects.link(camera)
    a, e = math.radians(azimuth), math.radians(elevation)
    offset = Vector((-math.sin(a) * math.cos(e), math.cos(a) * math.cos(e), math.sin(e))) * distance
    camera.location = Vector(centre) + offset
    camera.rotation_mode = "QUATERNION"
    camera.rotation_quaternion = (-offset).to_track_quat("-Z", "Y")
    camera.data.lens = lens
    camera.data.sensor_fit = "VERTICAL"
    camera.data.sensor_height = 24.0
    bpy.context.scene.camera = camera
    orbit_lights(azimuth)
    return camera


def distance_for(extent, lens=50.0, sensor=24.0, margin=1.15):
    """Camera distance that holds `extent` (metres, vertical) in frame."""
    return extent * margin / 2 / math.tan(math.atan(sensor / 2 / lens))


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="build the MPFB cage for a skin")
    parser.add_argument("--skin", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ages", default=None, help="comma-separated MPFB ages to try")
    args = parser.parse_args(argv)
    ages = tuple(float(a) for a in args.ages.split(",")) if args.ages else AGES
    out = Path(args.out)
    cage, rig, report = fit_cage_to(args.skin, out, ages=ages)
    (out / "mpfb_cage.json").write_text(json.dumps(report, indent=1, default=float))
    bpy.ops.wm.save_as_mainfile(filepath=str(out / "mpfb_cage.blend"))


if __name__ == "__main__":
    main()
