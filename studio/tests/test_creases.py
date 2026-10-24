"""creases.crease_of on synthetic hinges: each guard (the bone, the reach, the bend, the inside of the
bend) pinned by its own case, so deleting any one of them fails a test."""
import math

import numpy as np

from creases import CREASE_BEND_DEG, Hinge, bend_deg, crease_of, included_deg

P = "mixamorig:"
KNEE = np.array([0.0, 0.0, 0.5])
UP = np.array([0.0, 0.0, 1.0])          # along the thigh, from the knee to the hip
BACK = np.array([0.0, 1.0, 0.0])


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def knee(included, reach=0.1, straight=180.0):
    """A left knee whose shin makes `included` degrees with the thigh, the shin swinging backwards."""
    a = math.radians(included)
    shin = unit(math.cos(a) * UP + math.sin(a) * BACK)
    return Hinge("left knee", {P + "LeftUpLeg", P + "LeftLeg"}, KNEE, UP, shin, reach, straight)


def inside(h, mm=30):
    """A point `mm` from the joint along the bisector: the back of the knee, inside the bend."""
    return h.joint + unit(h.u + h.v) * mm / 1000


def test_the_back_of_a_deeply_bent_knee_is_a_crease():
    h = knee(60)                                         # bent 120
    assert crease_of([h], P + "LeftLeg", inside(h)) == "left knee"
    assert crease_of([h], P + "LeftUpLeg", inside(h)) == "left knee"


def test_the_outer_surface_near_a_bent_knee_is_not_a_crease():
    h = knee(60)
    # the front of the thigh, 30 mm from the joint: past the joint along the thigh, in front of its axis.
    # It is on the bisector's side of the joint (a half-space test admits it) but outside the bend.
    front = -unit(h.v - (h.v @ h.u) * h.u)               # away from the shin, square to the thigh
    point = h.joint + unit(0.6 * h.u + 0.8 * front) * 0.030
    assert float((point - h.joint) @ (h.u + h.v)) > 0    # the half-space would have called it a crease
    assert crease_of([h], P + "LeftLeg", point) is None
    # the front of the calf, likewise, and the kneecap itself
    front_shin = -unit(h.u - (h.u @ h.v) * h.v)
    assert crease_of([h], P + "LeftLeg", h.joint + unit(0.6 * h.v + 0.8 * front_shin) * 0.030) is None
    assert crease_of([h], P + "LeftLeg", h.joint - unit(h.u + h.v) * 0.030) is None


def test_a_knee_bent_less_than_the_limit_has_no_crease():
    shallow, deep = knee(180 - CREASE_BEND_DEG + 5), knee(180 - CREASE_BEND_DEG - 5)
    assert bend_deg(shallow) < CREASE_BEND_DEG < bend_deg(deep)
    assert crease_of([shallow], P + "LeftLeg", inside(shallow)) is None
    assert crease_of([deep], P + "LeftLeg", inside(deep)) == "left knee"


def test_a_fold_beyond_the_reach_is_not_the_knee_crease():
    h = knee(60, reach=0.05)
    assert crease_of([h], P + "LeftLeg", inside(h, mm=40)) == "left knee"
    assert crease_of([h], P + "LeftLeg", inside(h, mm=60)) is None


def test_only_the_hinge_bones_make_its_crease():
    h = knee(60)
    assert crease_of([h], P + "LeftArm", inside(h)) is None
    assert crease_of([h], P + "RightLeg", inside(h)) is None


def test_the_nearest_carrying_joint_judges_the_patch():
    h = knee(60)
    far = Hinge("left hip", {P + "LeftUpLeg", P + "Hips"}, KNEE + np.array([0.0, 0.0, 0.4]), UP, -UP, 0.1, 180.0)
    assert crease_of([far, h], P + "LeftUpLeg", inside(h)) == "left knee"


def ankle(included, rest=100.0):
    """A left ankle: the shin up from it, the foot forward; unbent at its rest angle."""
    a = math.radians(included)
    foot = unit(math.cos(a) * UP + math.sin(a) * -BACK)
    return Hinge("left ankle", {P + "LeftLeg", P + "LeftFoot"}, np.zeros(3), UP, foot, 0.04, rest)


def test_an_ankle_bends_from_its_rest_angle():
    # standing, the foot already makes ~100 degrees with the shin: that is not a 80-degree bend
    standing = ankle(100)
    assert abs(bend_deg(standing)) < 1e-9
    assert crease_of([standing], P + "LeftFoot", inside(standing, mm=20)) is None
    shallow = ankle(100 - CREASE_BEND_DEG + 5)
    assert crease_of([shallow], P + "LeftFoot", inside(shallow, mm=20)) is None
    deep = ankle(100 - CREASE_BEND_DEG - 5)
    assert crease_of([deep], P + "LeftFoot", inside(deep, mm=20)) == "left ankle"


def test_included_angle_takes_directions_of_any_length():
    assert abs(included_deg([0, 0, 3.0], [0, 2.0, 0]) - 90) < 1e-9
    assert abs(included_deg([1.0, 0, 0], [-0.5, 0.5 * math.sqrt(3), 0]) - 120) < 1e-9


def past_the_thigh_axis(h, mm, shin_mm=None):
    """A point in the bend's plane `mm` past the thigh's axis (on the side away from the shin) and 10 mm
    inside the shin's axis, or, with `shin_mm`, 10 mm inside the thigh's axis and `shin_mm` past the
    shin's: square distances to the two axes, solved for the point."""
    to_shin = unit(h.v - (h.v @ h.u) * h.u)            # square to the thigh, towards the shin
    to_thigh = unit(h.u - (h.u @ h.v) * h.v)           # square to the shin, towards the thigh
    c = float(to_shin @ to_thigh)
    s1, s2 = -mm / 1000, 0.010                          # wanted: d.to_shin = s1, d.to_thigh = s2
    if shin_mm is not None:
        s1, s2 = 0.010, -shin_mm / 1000
    a, b = (s1 - c * s2) / (1 - c * c), (s2 - c * s1) / (1 - c * c)
    d = a * to_shin + b * to_thigh
    assert abs(d @ to_shin - s1) < 1e-12 and abs(d @ to_thigh - s2) < 1e-12
    return h.joint + d


def test_a_crease_on_the_joint_axis_is_allowed_a_few_mm_by_the_character_size():
    from creases import CREASE_SLACK
    h = knee(40)                                        # folded tight, as the squat's elbows
    slack = CREASE_SLACK * 1.0                          # a 1 m character: 4 mm
    near = past_the_thigh_axis(h, 2.5)
    assert crease_of([h], P + "LeftLeg", near) is None                    # no slack: outside the wedge
    assert crease_of([h], P + "LeftLeg", near, slack=slack) == "left knee"
    assert crease_of([h], P + "LeftLeg", past_the_thigh_axis(h, 8.0), slack=slack) is None
    # past the shin's axis alike
    assert crease_of([h], P + "LeftLeg", past_the_thigh_axis(h, 0, shin_mm=2.5), slack=slack) == "left knee"
    assert crease_of([h], P + "LeftLeg", past_the_thigh_axis(h, 0, shin_mm=8.0), slack=slack) is None
    # twice the size, twice the slack
    assert crease_of([h], P + "LeftLeg", past_the_thigh_axis(h, 6.0), slack=CREASE_SLACK * 2.0) == "left knee"


def squat_skeleton(hip_bend=100.0):
    """Heads of a crouched left side (and the pelvis), as a rig gives them: the torso up, the thigh
    swung forward `hip_bend` degrees from hanging, the knee folded, the foot flat."""
    a = math.radians(hip_bend)
    hip = np.array([-0.08, 0.0, 0.40])
    thigh = np.array([0.0, -math.sin(a), -math.cos(a)])            # -Y is forward here
    knee = hip + 0.25 * thigh
    ankle = knee + 0.22 * unit([0.0, 0.3, -1.0])
    return {P + "Hips": np.array([0.0, 0.0, 0.42]), P + "Spine1": np.array([0.0, 0.0, 0.62]),
            P + "LeftUpLeg": hip, P + "LeftLeg": knee, P + "LeftFoot": ankle,
            P + "LeftToeBase": ankle + np.array([0.0, -0.08, -0.03])}


def test_hinges_from_a_rig_s_heads_and_the_crotch_is_never_a_crease():
    from creases import hinges_from
    heads = squat_skeleton()
    hinges = {h.name: h for h in hinges_from(heads, heads)}
    assert set(hinges) == {"left hip", "left knee", "left ankle"}          # no right side, no arms here
    hip = hinges["left hip"]
    assert abs(bend_deg(hip) - 100) < 1 and abs(bend_deg(hinges["left ankle"])) < 1e-9   # ankle at rest
    assert hip.carry == {P + "LeftUpLeg"}                                  # the groin, not the crotch
    groin = hip.joint + unit(hip.u + hip.v) * 0.02
    assert crease_of(list(hinges.values()), P + "LeftUpLeg", groin) == "left hip"
    # the shorts' crotch web is a defect wherever it is seen, however bent the hips (ruling)
    assert crease_of(list(hinges.values()), P + "Hips", groin) is None
