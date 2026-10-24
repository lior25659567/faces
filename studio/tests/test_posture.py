"""posture.py on synthetic side views: masks drawn in the torso's own frame, points with known angles."""
import math

import numpy as np
import pytest
from pytest import approx

from posture import (BACK_MIN_SEEN, back_profile, back_rounding, classify_back, curl_from_rounding, head_pitch,
                     neck_vs_torso, neutral_rounding, spine_bend)

SIZE = (400, 400)                    # rows, columns; image coordinates (x right, y down)
HIP, SHOULDER = (200.3, 300.0), (200.3, 100.0)    # an upright torso 200 px long (edges off the pixel centres)
BOW = 0.08                           # torso lengths


def bump(s):
    """0 at the measured stretch's ends (10% and 90% of the torso), 1 at its middle, 0 outside."""
    s = np.asarray(s, dtype=float)
    return np.where((s > 0.1) & (s < 0.9), 4 * (s - 0.1) * (0.9 - s) / 0.64, 0.0)


def torso_mask(hip=HIP, shoulder=SHOULDER, facing="right", bow=0.0, back=0.25, front=0.25, size=SIZE):
    """A torso band around the hip->shoulder chord: `front` torso lengths on the face's side, `back` (plus
    bow x bump(s)) on the other; + bow bulges the back outward (rounded), - draws it in (arched)."""
    hip, shoulder = np.asarray(hip, float), np.asarray(shoulder, float)
    length = float(np.linalg.norm(shoulder - hip))
    t = (shoulder - hip) / length
    n = np.array([-t[1], t[0]])
    if n @ np.array([1.0 if facing == "right" else -1.0, 0.0]) > 0:
        n = -n                                       # n points to the back
    ys, xs = np.mgrid[0:size[0], 0:size[1]]
    d = np.stack([xs - hip[0], ys - hip[1]], axis=-1)
    s, r = d @ t / length, d @ n / length
    return (s >= -0.2) & (s <= 1.2) & (r >= -front) & (r <= back + bow * bump(s))


def mirrored(mask, *points):
    width = mask.shape[1]
    return (mask[:, ::-1], *[(width - 1 - p[0], p[1]) for p in points])


def lean(deg, facing="right"):
    """Hip and shoulder of a torso leaning `deg` toward the face."""
    a = math.radians(deg)
    sign = 1 if facing == "right" else -1
    hip = np.array([200.0, 320.0])
    return hip, hip + 200 * np.array([sign * math.sin(a), -math.cos(a)])


# --- back_rounding -------------------------------------------------------------------------------------

def test_a_straight_back_reads_zero():
    assert back_rounding(torso_mask(), SHOULDER, HIP, "right") == approx(0.0, abs=0.004)


def test_a_bowed_back_reads_the_bow_over_the_torso_length():
    assert back_rounding(torso_mask(bow=BOW), SHOULDER, HIP, "right") == approx(BOW, abs=0.006)


def test_an_arched_back_reads_negative():
    assert back_rounding(torso_mask(bow=-BOW), SHOULDER, HIP, "right") == approx(-BOW, abs=0.006)


def test_a_thicker_or_thinner_body_with_a_straight_back_still_reads_zero():
    assert back_rounding(torso_mask(back=0.4, front=0.1), SHOULDER, HIP, "right") == approx(0.0, abs=0.004)


@pytest.mark.parametrize("bow", [0.0, BOW, -BOW])
def test_facing_left_mirrors(bow):
    right = back_rounding(torso_mask(bow=bow), SHOULDER, HIP, "right")
    mask, shoulder, hip = mirrored(torso_mask(bow=bow), SHOULDER, HIP)
    assert back_rounding(mask, shoulder, hip, "left") == approx(right, abs=0.003)
    # the back is the side away from the face: asked the wrong way, the front's straight edge is read
    if bow:
        assert back_rounding(mask, shoulder, hip, "right") == approx(0.0, abs=0.004)


@pytest.mark.parametrize("facing", ["right", "left"])
def test_a_leaning_torso_is_measured_square_to_its_own_line(facing):
    hip, shoulder = lean(35, facing)
    assert back_rounding(torso_mask(hip, shoulder, facing), shoulder, hip, facing) == approx(0.0, abs=0.004)
    assert back_rounding(torso_mask(hip, shoulder, facing, bow=BOW), shoulder, hip, facing) == approx(BOW, abs=0.006)


def arm_behind_the_back(mask, hip=HIP, shoulder=SHOULDER, top=0.9, bottom=0.6, out=0.45):
    """An upper arm hanging behind the back from the shoulder to an elbow out at `out` torso lengths:
    returns the mask with the arm drawn in, and the arm's segment (shoulder -> elbow)."""
    hip, shoulder = np.asarray(hip, float), np.asarray(shoulder, float)
    length = float(np.linalg.norm(shoulder - hip))
    a = hip + top * (shoulder - hip) + np.array([-0.15 * length, 0.0])
    b = hip + bottom * (shoulder - hip) + np.array([-out * length, 0.0])
    ys, xs = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
    p = np.stack([xs, ys], axis=-1).astype(float)
    ab = b - a
    k = np.clip(((p - a) @ ab) / (ab @ ab), 0, 1)
    near = np.linalg.norm(p - (a + k[..., None] * ab), axis=-1) < 0.12 * length
    return mask | near, (tuple(a), tuple(b))


def test_an_arm_behind_the_back_is_left_out():
    mask, arm = arm_behind_the_back(torso_mask())
    # read blindly, the arm's outline looks like a hugely rounded back
    assert back_rounding(mask, SHOULDER, HIP, "right") > 0.15
    profile = back_profile(mask, SHOULDER, HIP, "right", arms=[arm])
    assert profile["rounding"] == approx(0.0, abs=0.004)
    assert 0.3 < profile["seen"] < 0.9               # the rows the arm covers are not counted
    lo, hi = profile["span"]
    assert 0.1 <= lo < hi <= 0.9
    # the arm's samples are the ones left out: none of the kept edge points is within reach of it
    kept = [p for p in profile["samples"] if p["kept"]]
    assert kept and all(p["clear"] >= 0 for p in kept)
    assert any(not p["kept"] for p in profile["samples"])


def test_a_back_mostly_hidden_by_an_arm_is_not_measured():
    mask, arm = arm_behind_the_back(torso_mask(), top=0.95, bottom=0.05, out=0.3)
    profile = back_profile(mask, SHOULDER, HIP, "right", arms=[arm])
    assert profile["rounding"] is None and profile["seen"] < 0.25
    assert back_rounding(mask, SHOULDER, HIP, "right", arms=[arm]) is None


def test_a_wider_arm_radius_leaves_out_more():
    mask, arm = arm_behind_the_back(torso_mask())
    narrow = back_profile(mask, SHOULDER, HIP, "right", arms=[arm], arm_radius=0.2)
    wide = back_profile(mask, SHOULDER, HIP, "right", arms=[arm], arm_radius=0.35)
    assert wide["seen"] < narrow["seen"]


def test_each_arm_segment_can_carry_its_own_radius():
    mask, (a, b) = arm_behind_the_back(torso_mask())
    assert back_profile(mask, SHOULDER, HIP, "right", arms=[(a, b, 0.2)])["seen"] == approx(
        back_profile(mask, SHOULDER, HIP, "right", arms=[(a, b)], arm_radius=0.2)["seen"])
    # an arm drawn 0.12 thick read with 0.10: its outline creeps in as a bulge
    thin = back_profile(mask, SHOULDER, HIP, "right", arms=[(a, b, 0.10)])
    assert thin["rounding"] > 0.04


def test_a_mask_of_the_arms_alone_leaves_out_exactly_their_outline():
    body = torso_mask()
    with_arm, arm = arm_behind_the_back(body)
    arm_only = with_arm & ~body
    profile = back_profile(with_arm, SHOULDER, HIP, "right", arm_mask=arm_only)
    assert profile["rounding"] == approx(0.0, abs=0.004)
    kept = [p for p in profile["samples"] if p["kept"]]
    assert kept and not any(p["on_arm"] for p in kept)
    # the arm's own outline is all it leaves out: more of the back is read than a capsule wide enough for it
    capsule = back_profile(with_arm, SHOULDER, HIP, "right", arms=[arm])
    assert profile["seen"] > capsule["seen"]
    # an arm in front of the back (inside the silhouette, nowhere near its edge) leaves out nothing
    inside = np.zeros(SIZE, bool)
    inside[150:250, 190:215] = True
    assert back_profile(body, SHOULDER, HIP, "right", arm_mask=inside)["seen"] == approx(1.0)


def test_a_span_reads_only_that_stretch_of_the_back():
    # a back bowed only in its upper half (s 0.5-0.9): straight on the lower stretch the coach shows
    def upper(s):
        s = np.asarray(s, dtype=float)
        return np.where((s > 0.5) & (s < 0.9), 4 * (s - 0.5) * (0.9 - s) / 0.16, 0.0)
    hip, shoulder = np.asarray(HIP), np.asarray(SHOULDER)
    t = (shoulder - hip) / 200.0
    ys, xs = np.mgrid[0:SIZE[0], 0:SIZE[1]]
    d = np.stack([xs - hip[0], ys - hip[1]], axis=-1)
    s, r = d @ t / 200.0, d @ np.array([-1.0, 0.0]) / 200.0
    mask = (s >= -0.2) & (s <= 1.2) & (r >= -0.25) & (r <= 0.25 + BOW * upper(s))
    whole = back_profile(mask, SHOULDER, HIP, "right")
    lower = back_profile(mask, SHOULDER, HIP, "right", span=(0.1, 0.45))
    assert whole["rounding"] > 0.03 and lower["rounding"] == approx(0.0, abs=0.004)
    assert lower["span"][0] == approx(0.1) and lower["span"][1] == approx(0.45, abs=0.01)
    assert lower["seen"] == approx(0.35 / 0.8, abs=0.01)          # of the whole 10-90% stretch
    assert all(0.1 <= p["s"] <= 0.45 + 1e-9 for p in lower["samples"])
    # never past the 10% the ends leave out, nor shorter than MIN_SPAN
    assert back_profile(mask, SHOULDER, HIP, "right", span=(0.0, 1.0))["span"] == approx((0.1, 0.9))
    assert back_profile(mask, SHOULDER, HIP, "right", span=(0.2, 0.3))["rounding"] is None


def test_small_holes_in_the_mask_are_not_the_outside():
    mask = torso_mask()
    mask[:, 160:162] = False                         # a 2 px seam just inside the back edge (x=150)
    assert back_rounding(mask, SHOULDER, HIP, "right") == approx(0.0, abs=0.004)


def test_no_reading_without_a_facing_a_body_or_a_torso():
    assert back_rounding(torso_mask(), SHOULDER, HIP, None) is None
    assert back_rounding(np.zeros(SIZE, bool), SHOULDER, HIP, "right") is None
    assert back_rounding(torso_mask(), HIP, HIP, "right") is None
    # a torso lying level: no side is "away from the face" in the picture's left/right
    hip, shoulder = (100.0, 200.0), (300.0, 200.0)
    assert back_rounding(torso_mask(hip, shoulder), shoulder, hip, "right") is None


def test_the_profile_reads_in_torso_lengths_whatever_the_picture_size():
    small = torso_mask(bow=BOW)
    big = np.kron(small, np.ones((2, 2), dtype=bool))      # the same body, twice the pixels
    a = back_rounding(small, SHOULDER, HIP, "right")
    b = back_rounding(big, (2 * SHOULDER[0], 2 * SHOULDER[1]), (2 * HIP[0], 2 * HIP[1]), "right")
    assert a == approx(b, abs=0.005)


# --- neck_vs_torso -------------------------------------------------------------------------------------

RIGHT, LEFT = (1.0, 0.0), (-1.0, 0.0)


def test_neck_in_line_is_zero():
    assert neck_vs_torso((0, 0), (0, -10), (0, -15), RIGHT) == approx(0.0, abs=1e-9)


def test_neck_forward_is_positive_and_back_is_negative():
    assert neck_vs_torso((0, 0), (0, -10), (5, -15), RIGHT) == approx(45.0)
    assert neck_vs_torso((0, 0), (0, -10), (-5, -15), RIGHT) == approx(-45.0)


def test_neck_mirrors_with_the_facing():
    assert neck_vs_torso((0, 0), (0, -10), (-5, -15), LEFT) == approx(45.0)
    assert neck_vs_torso((0, 0), (0, -10), (5, -15), LEFT) == approx(-45.0)


def test_neck_is_measured_from_the_torso_line_not_from_upright():
    # torso leaning 30 degrees forward, neck 50 degrees forward of upright: 20 degrees forward of the torso
    def up_at(deg):
        a = math.radians(deg)
        return np.array([math.sin(a), -math.cos(a)])
    hip = np.zeros(2)
    shoulder = hip + 10 * up_at(30)
    ear = shoulder + 4 * up_at(50)
    assert neck_vs_torso(hip, shoulder, ear, RIGHT) == approx(20.0)


def test_neck_in_3d_drops_the_sideways_part_when_given_the_side_plane():
    # x forward, y to the body's left, z up: the ear 0.1 forward and 0.1 sideways of the shoulders
    hip, shoulder, ear = (0, 0, 0), (0, 0, 0.5), (0.1, 0.1, 0.6)
    assert neck_vs_torso(hip, shoulder, ear, (1, 0, 0)) == approx(math.degrees(math.acos(1 / math.sqrt(3))))
    assert neck_vs_torso(hip, shoulder, ear, (1, 0, 0), normal=(0, 1, 0)) == approx(45.0)


def test_neck_against_a_torso_line_turned_by_the_rest_lean():
    # task 18: our rig's torso line (hip joints -> shoulder joints) leans back 11.3 degrees standing tall, so
    # it is turned upright by its rest pose's lean (metrics.rest_turn) before the neck is read against it.
    # x to the body's left, y forward, z up; the shoulders 0.1 behind the hips, the ears straight above them
    hip, shoulder, ear = (0, 0, 0), (0, -0.1, 0.5), (0, -0.1, 0.7)
    a = math.atan(0.1 / 0.5)
    turn = np.array([[1, 0, 0], [0, math.cos(a), math.sin(a)], [0, -math.sin(a), math.cos(a)]])   # tips +z to +y
    forward, up, side = (0, 1, 0), (0, 0, 1), (1, 0, 0)
    assert neck_vs_torso(hip, shoulder, ear, forward, normal=side, up=up) == approx(math.degrees(a))
    assert neck_vs_torso(hip, shoulder, ear, forward, normal=side, up=up, torso_turn=turn) == approx(0.0, abs=1e-9)
    # the neck line itself is not turned: it still runs from the shoulder joints to the ears
    ear_forward = (0, -0.1 + 0.2 * math.tan(math.radians(30)), 0.7)
    assert neck_vs_torso(hip, shoulder, ear_forward, forward, normal=side, up=up, torso_turn=turn) == approx(30.0)
    assert neck_vs_torso(hip, shoulder, ear, forward, normal=side, up=up, torso_turn=None) == \
        neck_vs_torso(hip, shoulder, ear, forward, normal=side, up=up)


def test_neck_without_a_length_is_none():
    assert neck_vs_torso((0, 0), (0, 0), (0, -5), RIGHT) is None
    assert neck_vs_torso((0, 0), (0, -10), (0, -10), RIGHT) is None


# --- head_pitch ----------------------------------------------------------------------------------------

UP_IMG = (0.0, -1.0)    # image y runs down


def test_level_up_and_down_faces():
    assert head_pitch((1, 0), UP_IMG, RIGHT) == approx(0.0)
    assert head_pitch((1, -1), UP_IMG, RIGHT) == approx(45.0)
    assert head_pitch((1, 1), UP_IMG, RIGHT) == approx(-45.0)


def test_pitch_mirrors_with_the_facing_and_takes_an_offset():
    assert head_pitch((-1, -1), UP_IMG, LEFT) == approx(45.0)
    assert head_pitch((-1, 0), UP_IMG, LEFT, offset=12.0) == approx(12.0)


def test_pitch_in_3d_on_the_side_plane():
    # x forward, y sideways, z up: a face turned partly sideways but level reads level on the side plane
    assert head_pitch((1, 0.5, 0), (0, 0, 1), (1, 0, 0), normal=(0, 1, 0)) == approx(0.0)
    assert head_pitch((1, 0.5, 1), (0, 0, 1), (1, 0, 0), normal=(0, 1, 0)) == approx(45.0)


def test_pitch_without_a_direction_is_none():
    assert head_pitch((0, 0), UP_IMG, RIGHT) is None


# --- review fixes: what is kept, unbroken stretches, the coach's classification, the spine, level ---------

def forearm_across(mask, lo, hi, hip=HIP, shoulder=SHOULDER):
    """A forearm lying along the back from s=lo to s=hi and sticking out behind it (r 0.15-0.40, the
    back's edge at 0.25): (the mask with it drawn in, the forearm alone)."""
    hip, shoulder = np.asarray(hip, float), np.asarray(shoulder, float)
    length = float(np.linalg.norm(shoulder - hip))
    t = (shoulder - hip) / length
    ys, xs = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
    d = np.stack([xs - hip[0], ys - hip[1]], axis=-1)
    s, r = d @ t / length, d @ np.array([-1.0, 0.0]) / length
    arm = (s >= lo) & (s <= hi) & (r >= 0.15) & (r <= 0.40)
    return mask | arm, arm


def test_seen_is_the_share_of_the_back_actually_kept():
    with_arm, arm = forearm_across(torso_mask(), 0.3, 0.7)
    p = back_profile(with_arm, SHOULDER, HIP, "right", arm_mask=arm)
    kept = [x["s"] for x in p["samples"] if x["kept"]]
    assert min(kept) == approx(0.1) and max(kept) == approx(0.9)       # both ends of the back are in view
    assert p["seen"] == approx(0.36 / 0.8, abs=0.03)                  # but only the two ends: 0.1-0.28 and 0.72-0.9


def test_a_bow_hidden_mid_back_by_a_forearm_is_not_bridged_and_never_reads_straight():
    with_arm, arm = forearm_across(torso_mask(bow=BOW), 0.3, 0.7)     # the bow peaks at s=0.5, under the forearm
    p = back_profile(with_arm, SHOULDER, HIP, "right", arm_mask=arm)
    assert p["seen"] < 0.6
    lo, hi = p["span"]                                                # the longest unbroken stretch, one side only
    assert hi - lo < 0.22 and (hi < 0.3 or lo > 0.7)
    c = classify_back(p["rounding"], p["seen"], p["span"], p["why"])
    assert (c["state"], c["cause"]) == ("unreadable", "hidden") and c["curl"] is None


def test_a_back_broken_up_by_an_arm_is_unreadable_even_when_most_is_in_view():
    with_arm, arm = forearm_across(torso_mask(bow=BOW), 0.45, 0.55)
    p = back_profile(with_arm, SHOULDER, HIP, "right", arm_mask=arm)
    assert p["seen"] >= BACK_MIN_SEEN - 0.15 and p["run"] < BACK_MIN_SEEN
    assert classify_back(p["rounding"], p["seen"], p["span"], p["why"])["cause"] in ("broken", "hidden")


def test_a_small_gap_does_not_break_the_back():
    mask = torso_mask()
    arm = np.zeros(SIZE, bool)
    arm[198:200, 140:160] = True            # 2 px of arm on the edge at mid-back: 0.01 torso lengths
    p = back_profile(mask | arm, SHOULDER, HIP, "right", arm_mask=arm)
    assert p["run"] == approx(1.0, abs=0.01) and p["rounding"] == approx(0.0, abs=0.004)


def test_lying_level_says_so():
    hip, shoulder = (100.0, 200.0), (300.0, 200.0)
    p = back_profile(torso_mask(hip, shoulder), shoulder, hip, "right")
    assert p["rounding"] is None and p["why"] == "level"
    assert classify_back(p["rounding"], p["seen"], p["span"], p["why"])["cause"] == "level"


def test_the_gate_rests_on_a_30_degree_curl():
    # back-calibration.json, worst case over where the stretch sits: a 30-degree curl reads beyond the 0.04
    # "straight" band only with 87.5% of the back in view (0.70 of the 0.8 stretch)
    assert BACK_MIN_SEEN == 0.875


def test_the_human_neutral_and_curls_come_from_the_calibration():
    whole = (0.1, 0.9)
    assert neutral_rounding(whole) == approx(-0.096, abs=0.002)
    for span in (whole, (0.1, 0.8), (0.2, 0.9)):
        assert curl_from_rounding(neutral_rounding(span), span) == approx(0.0, abs=0.5)
    # the calibration's own +30 and -30 readings come back as 30 and -30; halfway reads in between
    import json
    from pathlib import Path
    import posture
    cal = json.loads((Path(posture.__file__).parent / "back-calibration.json").read_text())
    s = np.array(cal["s"])
    def sag(d):
        d = np.array(d)
        dev = d - (d[0] + (d[-1] - d[0]) * (s - s[0]) / (s[-1] - s[0]))
        return float(dev[np.argmax(np.abs(dev))])
    up, down = sag(cal["profiles"]["30"]), sag(cal["profiles"]["-30"])
    assert curl_from_rounding(up, whole) == approx(30, abs=0.5)
    assert curl_from_rounding(down, whole) == approx(-30, abs=0.5)
    assert 15 < curl_from_rounding((sag(cal["profiles"]["15"]) + up) / 2, whole) < 30


def test_the_coach_back_is_classified_only_when_decisive():
    whole, n = (0.1, 0.9), neutral_rounding((0.1, 0.9))
    assert classify_back(n, 1.0, whole)["state"] == "straight"
    assert classify_back(n + 0.03, 1.0, whole)["state"] == "straight"          # within 1 x 0.04
    assert classify_back(n + 0.06, 1.0, whole)["state"] == "cant-tell"         # between 1x and 2x
    rounded = classify_back(n + 0.09, 1.0, whole)                              # beyond 2 x 0.04
    assert rounded["state"] == "rounded" and rounded["curl"] > 30
    arched = classify_back(n - 0.09, 1.0, whole)
    assert arched["state"] == "arched" and arched["curl"] < -30
    assert classify_back(n, 0.25, (0.1, 0.3))["cause"] == "hidden"             # the squat photos
    assert classify_back(n, 0.9, (0.1, 0.45))["cause"] == "broken"             # in view, but in pieces
    assert classify_back(None, 0.0, None, "level")["cause"] == "level"
    assert classify_back(n, None, whole)["cause"] == "no-seen"                  # never judged without it
    assert classify_back(None, 0.0, None)["cause"] == "hidden"
    for c in (classify_back(n, 0.25, (0.1, 0.3)), classify_back(None, 0.0, None, "level")):
        assert c["state"] == "unreadable" and c["curl"] is None


def test_spine_bend_sums_the_side_seen_bends():
    # x forward, y sideways, z up: four segments up the back
    straight = [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 0, 4)]
    assert spine_bend(straight, (1, 0, 0), (0, 0, 1), (0, 1, 0)) == approx(0.0)

    def chain(step_deg, sideways=0.0):
        pts, angle, p = [np.zeros(3)], 0.0, np.zeros(3)
        for _ in range(4):
            p = p + np.array([math.sin(math.radians(angle)), sideways, math.cos(math.radians(angle))])
            pts.append(p)
            angle += step_deg
        return pts
    assert spine_bend(chain(10), (1, 0, 0), (0, 0, 1), (0, 1, 0)) == approx(30.0)      # curled forward
    assert spine_bend(chain(-10), (1, 0, 0), (0, 0, 1), (0, 1, 0)) == approx(-30.0)    # arched
    assert spine_bend(chain(10), (-1, 0, 0), (0, 0, 1), (0, 1, 0)) == approx(-30.0)    # facing the other way
    assert spine_bend(chain(10, sideways=0.3), (1, 0, 0), (0, 0, 1), (0, 1, 0)) == approx(30.0)


@pytest.mark.parametrize("deg", [14.0, -14.0])
def test_no_neck_reading_within_15_degrees_of_level(deg):
    # a torso lying near level (a plank), its line tipped 14 degrees above or below level: the head's
    # "forward of the back" has no sign the picture's left/right can give
    a = math.radians(deg)
    hip, shoulder = np.zeros(2), 10 * np.array([math.cos(a), -math.sin(a)])
    ear = shoulder + 4 * np.array([1.0, 0.0])
    assert neck_vs_torso(hip, shoulder, ear, RIGHT) is None
    assert neck_vs_torso(hip, shoulder, ear, RIGHT, up=UP_IMG) is None


@pytest.mark.parametrize("deg", [16.0, -16.0])
def test_a_neck_reading_just_past_15_degrees_of_level(deg):
    a = math.radians(deg)
    hip, shoulder = np.zeros(2), 10 * np.array([math.cos(a), -math.sin(a)])
    ear = shoulder + 4 * np.array([math.cos(a), -math.sin(a)])
    assert neck_vs_torso(hip, shoulder, ear, RIGHT, up=UP_IMG) == approx(0.0, abs=1e-6)


# ---- the depth at the bottom of a squat: the seat against the top of the knee (task 22) ----
# The judge's criterion, read the same way on both silhouettes - the coach's MediaPipe person mask, our rendered one:
# how far the seat's lowest point (behind the hip) sits below the top of whichever knee shows, in thigh lengths.
import json as _json
from pathlib import Path as _Path

from PIL import Image as _Image

_THIGH = _Path(__file__).parent / "fixtures" / "thigh"


def _thigh_fixture(name):
    """(mask, hip, knee) of a real bottom: the coach's 17.png or one of our side stills' mask renders, cropped."""
    meta = _json.loads((_THIGH / "thigh.json").read_text())[name]
    mask = np.asarray(_Image.open(_THIGH / f"{name}.png").convert("L"), dtype=float) / 255.0
    return mask, meta["hip"], meta["knee"]


def test_the_coachs_seat_against_his_knee_top():
    """17.png: his seat's lowest point (on his shorts, under his hip) 0.646 thigh lengths below the top of the knee
    that shows (the overlay: t22 ov2-grid.png)."""
    from posture import seat_depth
    mask, hip, knee = _thigh_fixture("coach-17-end")
    r = seat_depth(mask, hip, knee, "right")
    length = math.dist(hip, knee)
    assert r["why"] is None and r["depth"] == approx(0.646, abs=0.005), r
    sx, sy = r["seat"]
    assert hip[0] - 0.8 * length <= sx <= hip[0] and 0.2 * length < sy - hip[1] < 0.35 * length, r   # under the hip
    kx, ky = r["knee_top"]
    # on the window's back edge (0.15 thigh lengths, to the whole pixel): his outline still rises toward the hip there
    assert abs(kx - knee[0]) <= 0.15 * length + 1 and 0.35 * length < knee[1] - ky < 0.5 * length, r


BOTTOMS = ("ours-hips-high-end", "ours-20b-end", "ours-21-end", "ours-before-end", "ours-22-end")   # shallow to deep


def test_the_seat_depth_tells_the_bottoms_apart():
    """Our bottoms in order, hips over the knees (task 20b's A) to task 22's held at the coach's. Within 0.10 thigh
    lengths (the tolerance) of the coach's: task 22's, the old one (before task 20) and task 21's pass, the closest
    0.014 inside the line; A fails by 0.04 past it. Task 20b's B lies 0.002 past the line, inside the fixtures'
    pinning, so only its place in the order is asserted, not a verdict."""
    from build_targets import TOLERANCE
    from posture import seat_depth
    tol = TOLERANCE["seat_below_knee"]
    assert tol == 0.10
    coach = seat_depth(*_thigh_fixture("coach-17-end"), "right")["depth"]
    reads = {n: seat_depth(*_thigh_fixture(n), "right")["depth"] for n in BOTTOMS}
    assert reads == approx({"ours-hips-high-end": 0.507, "ours-20b-end": 0.544, "ours-21-end": 0.560,
                            "ours-before-end": 0.589, "ours-22-end": 0.639}, abs=0.005), reads
    assert [reads[n] for n in BOTTOMS] == sorted(reads.values()), reads
    for n in ("ours-22-end", "ours-before-end", "ours-21-end"):
        assert abs(coach - reads[n]) <= tol - 0.01, (n, reads[n])
    assert coach - reads["ours-hips-high-end"] >= tol + 0.03, reads


def test_the_knee_window_moves_both_readings_alike():
    """The knee's top lies on the window's back edge (the thigh's top just behind the knee cap) on the coach's and
    ours alike, so the window's width moves the reading - by about as much on both: at 0.10 / 0.15 / 0.20 thigh
    lengths the coach reads 0.636 / 0.646 / 0.661, task 22's bottom 0.621 / 0.639 / 0.655; within 0.02 of his at each."""
    from posture import seat_depth
    sweep = {n: [seat_depth(*_thigh_fixture(n), "right", window=w)["depth"] for w in (0.10, 0.15, 0.20)]
             for n in ("coach-17-end", "ours-22-end")}
    assert sweep["coach-17-end"] == approx([0.636, 0.646, 0.661], abs=0.003), sweep
    assert sweep["ours-22-end"] == approx([0.621, 0.639, 0.655], abs=0.003), sweep
    assert all(abs(c - o) <= 0.02 for c, o in zip(sweep["coach-17-end"], sweep["ours-22-end"])), sweep


def _squat_scene(far_knee=None, torso_to=200, torso_low=215, size=500):
    """A squat's bottom drawn facing right, all in pixels (y down): the hip at (150, 250) and the knee at (350, 250)
    (a thigh 200 long); the seat a block x 60-149 down to y 320; the thigh a band y 215-285; the knee a disc of
    radius 40 round the knee (its top y 210); the calf hanging under the knee; the torso x 60-`torso_to` above y
    `torso_low`.
    The seat stops a pixel short of the hip's column, so read facing the wrong way it is not found.
    `far_knee`: (x, y) of the far knee's disc, radius 40."""
    yy, xx = np.mgrid[0:size, 0:size]
    mask = (xx >= 60) & (xx <= 149) & (yy >= 200) & (yy <= 320)                 # the seat
    mask |= (xx >= 150) & (xx <= 350) & (yy >= 215) & (yy <= 285)               # the thigh
    mask |= (xx - 350) ** 2 + (yy - 250) ** 2 <= 40 ** 2                         # the near knee
    mask |= (xx >= 330) & (xx <= 370) & (yy >= 250) & (yy <= 420)               # the calf
    mask |= (xx >= 60) & (xx <= torso_to) & (yy >= 0) & (yy <= torso_low)       # the torso
    if far_knee is not None:
        mask |= (xx - far_knee[0]) ** 2 + (yy - far_knee[1]) ** 2 <= 40 ** 2
    return mask.astype(float), (150.0, 250.0), (350.0, 250.0)


def test_the_seat_depth_reads_the_seat_and_the_top_of_whichever_knee_shows():
    from posture import seat_depth
    mask, hip, knee = _squat_scene()
    r = seat_depth(mask, hip, knee, "right")
    assert r["depth"] == approx((320 - 210) / 200) and r["seat"][1] == 320 and r["knee_top"] == (350, 210), r
    # the far knee higher (a knee out, or the camera's perspective): its top is the one that shows, and is read
    up = seat_depth(*_squat_scene(far_knee=(360, 240)), "right")
    assert up["depth"] == approx((320 - 200) / 200) and up["knee_top"] == (360, 200), up
    # the far knee lower, hidden behind the near one: the near knee's top is read, as before
    hidden = seat_depth(*_squat_scene(far_knee=(345, 262)), "right")
    assert hidden["depth"] == approx(r["depth"]) and hidden["knee_top"] == (350, 210), hidden


def test_the_seat_depth_faces_either_way():
    from posture import seat_depth
    mask, hip, knee = _squat_scene()
    flipped = mask[:, ::-1]
    r = seat_depth(flipped, (499 - hip[0], hip[1]), (499 - knee[0], knee[1]), "left")
    assert r["depth"] == approx((320 - 210) / 200), r
    # the seat is looked for behind the hip only: read facing the wrong way, the body behind the "hip" is the thigh
    wrong = seat_depth(mask, hip, knee, "left")
    assert wrong["depth"] != approx(r["depth"]), wrong


def test_a_knee_under_the_torso_has_no_top():
    """A column of body reaching a thigh length above the knee is the torso (leaning over the knees), not a knee."""
    from posture import seat_depth
    r = seat_depth(*_squat_scene(torso_to=420, torso_low=240), "right")
    assert r["depth"] is None and r["knee_top"] is None and "knee" in r["why"], r
    blank = seat_depth(np.zeros((500, 500)), (150.0, 250.0), (350.0, 250.0), "right")
    assert blank["depth"] is None and "seat" in blank["why"], blank
    assert seat_depth(*_squat_scene(), None)["why"], "no facing: no side view"
