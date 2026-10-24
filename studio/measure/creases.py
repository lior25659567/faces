"""Which folds of the skin are the creases of a bent hinge (pure numpy; inspect_run feeds it the posed
joints). A crease is skin pinched inside a knee, elbow, hip or ankle bent far enough (in the wedge
between its two segments, near the joint): normal in a deep bend, so the folded-skin check never fails
on it (studio/measure/checks.py). Knees, elbows and hips bend from straight; an ankle from its rest
angle, where the foot already stands square to the shin."""
from collections import namedtuple

import numpy as np

CREASE_BEND_DEG = 60.0  # a hinge bent further than this creases inside the bend
CREASE_NEAR = 0.5       # x the hinge's shorter segment (the thigh alone for the hip): how near its joint
CREASE_SLACK = 0.004    # x the character's height: how far past a segment's axis a crease's centre may lie

# name ("left knee"), the bones whose patches can be its crease, the joint (world), the unit directions
# from the joint along the parent segment (u) and the child segment (v), the reach (m), and the angle
# between u and v at which the hinge counts as unbent (degrees)
Hinge = namedtuple("Hinge", "name carry joint u v reach straight")


P = "mixamorig:"
# hinge: the bones whose patches can be its crease, the joint, the parent segment (from, to) and the
# child segment (from, to), each a bone head; "/" marks a bone with no side. The hip's parent segment
# is the torso (Hips to Spine1). The crotch (Hips) carries no crease: the shorts' crotch web is a
# defect wherever it is seen; the groin (UpLeg) does.
HINGES = (("knee", ("UpLeg", "Leg"), "Leg", ("Leg", "UpLeg"), ("Leg", "Foot")),
          ("elbow", ("Arm", "ForeArm"), "ForeArm", ("ForeArm", "Arm"), ("ForeArm", "Hand")),
          ("ankle", ("Leg", "Foot"), "Foot", ("Foot", "Leg"), ("Foot", "ToeBase")),
          ("hip", ("UpLeg",), "UpLeg", ("/Hips", "/Spine1"), ("UpLeg", "Leg")))


def hinges_from(posed, rest):
    """Hinge for each knee, elbow, ankle and hip a rig has, from its bones' heads: `posed` {bone: world
    head} now, `rest` {bone: head} in the rest pose. The hip's reach counts the thigh alone; the ankle
    is unbent at its rest angle (the foot stands square to the shin), the others when straight."""
    out = []
    for side in ("Left", "Right"):
        def bone(n):
            return P + (n[1:] if n.startswith("/") else side + n)
        for name, carry, joint, parent, child in HINGES:
            names = [bone(n) for n in (joint, *parent, *child)]
            if any(n not in posed for n in names):
                continue
            j, p0, p1, c0, c1 = (np.asarray(posed[n], dtype=float) for n in names)
            u, v = p1 - p0, c1 - c0
            lu, lv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
            if lu < 1e-9 or lv < 1e-9:
                continue
            straight = 180.0
            if name == "ankle":
                r = [np.asarray(rest[n], dtype=float) for n in names]
                straight = included_deg(r[2] - r[1], r[4] - r[3])
            out.append(Hinge(f"{side.lower()} {name}", {bone(c) for c in carry}, j, u / lu, v / lv,
                             CREASE_NEAR * (lv if name == "hip" else min(lu, lv)), straight))
    return out


def included_deg(u, v):
    """The angle between two directions (any length), degrees."""
    u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
    cos = float(u @ v) / max(float(np.linalg.norm(u) * np.linalg.norm(v)), 1e-30)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def bend_deg(hinge):
    """How far the hinge is bent from its unbent angle, degrees."""
    return hinge.straight - included_deg(hinge.u, hinge.v)


def inside_bend(hinge, centre, slack=0.0):
    """Whether the point lies in the wedge between the two segments: on the inner side of the parent's
    axis (towards the child) and of the child's axis (towards the parent), or no more than `slack`
    (m) past either. The outer surfaces of both segments near the joint lie a limb's radius outside
    it, though on the bisector's side of the joint; a crease pinched onto the axis of a joint folded
    tight (the squat's elbows, 2-3 mm past) is within the slack."""
    d = np.asarray(centre, dtype=float) - hinge.joint
    u, v = hinge.u, hinge.v
    towards_child = v - (v @ u) * u           # square to the parent's axis, pointing at the child
    towards_parent = u - (u @ v) * v
    size = float(np.linalg.norm(towards_child))
    if size < 1e-9:                           # unbent or folded flat: no inside to speak of
        return False
    return float(d @ towards_child) / size > -slack and float(d @ towards_parent) / size > -slack


def crease_of(hinges, bone, centre, slack=0.0):
    """The name of the hinge a patch is a crease of, or None: its bone is one of the hinge's, the
    nearest such joint is within the hinge's reach of its centre, the centre is inside the bend
    (inside_bend, with `slack`: CREASE_SLACK x the character's height) and the hinge is bent more than
    CREASE_BEND_DEG."""
    centre = np.asarray(centre, dtype=float)
    near = [(float(np.linalg.norm(centre - h.joint)), h) for h in hinges if bone in h.carry]
    if not near:
        return None
    distance, hinge = min(near, key=lambda n: n[0])
    if distance > hinge.reach or bend_deg(hinge) <= CREASE_BEND_DEG or not inside_bend(hinge, centre, slack):
        return None
    return hinge.name
