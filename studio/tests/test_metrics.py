import math

import numpy as np
import pytest
from pytest import approx

from metrics import jump_frames, jump_sizes, jump_steps, measure, phase_driver, pick_phase_frames

UP = np.array([0.0, 1.0, 0.0])
TOLERANCE = {"angles_deg": 5, "ratios": 0.12, "hip_below_knee": 0.10}


def _target(value):
    return {"value": value, "method": "flat", "cross": None, "uncertain": False, "image": None}


def _knee(f):
    """175 at frame 1, straight down to 50 at 30, straight back up to 175 at 60."""
    return 175 - 125 * (f - 1) / 29 if f <= 30 else 50 + 125 * (f - 30) / 30


# torso lean rises with the bend but moves far less than the knee, so the knee drives the frames
PER_FRAME = [{"frame": f, "flat": {"knee_included": _knee(f), "torso_lean": 5 + 15 * (175 - _knee(f)) / 125},
              "3d": {}} for f in range(1, 61)]
TARGETS = {"phases": {"start": {"knee_included": _target(175), "torso_lean": _target(5)},
                      "middle": {"knee_included": _target(105), "torso_lean": _target(14)},
                      "end": {"knee_included": _target(47), "torso_lean": _target(20)}},
           "widths": {}, "tolerance": TOLERANCE}


def standing_joints():
    """A straight figure, y up, facing -z. Shoulders 0.3 apart, arms hanging."""
    j = {}
    for s, x in (("left", 0.15), ("right", -0.15)):
        j[f"shoulder.{s}"] = np.array([x, 1.4, 0.0])
        j[f"elbow.{s}"] = np.array([x, 1.1, 0.0])
        j[f"wrist.{s}"] = np.array([x, 0.8, 0.0])
        j[f"hip.{s}"] = np.array([x * 0.6, 0.9, 0.0])
        j[f"knee.{s}"] = np.array([x * 0.6, 0.5, 0.0])
        j[f"ankle.{s}"] = np.array([x * 0.6, 0.1, 0.0])
    return j


def test_right_angle_knee_flat_and_3d():
    j = standing_joints()
    j["ankle.left"] = j["knee.left"] + np.array([0, 0, -0.4])
    assert measure(j, "left", UP, None)["knee_included"] == approx(90, abs=0.1)
    flat = measure(j, "left", UP, np.array([1.0, 0, 0]))
    assert flat["knee_included"] == approx(90, abs=0.1)


def test_flat_projection_hides_out_of_plane_bend():
    j = standing_joints()
    j["wrist.left"] = j["elbow.left"] + np.array([0, 0, -0.25])
    assert measure(j, "left", UP, view_normal=np.array([0, 0, 1.0]))["elbow_included"] == approx(180, abs=0.5)
    assert measure(j, "left", UP, None)["elbow_included"] == approx(90, abs=0.5)


def test_torso_lean_and_hip_below_knee_signs():
    # torso tipped 20 degrees forward (toward -z), hips 0.05 torso lengths below the knees
    j = standing_joints()
    t = np.radians(20)
    torso = 0.5
    for s in ("left", "right"):
        hip = j[f"hip.{s}"].copy()
        hip[1] = 0.5 - 0.05 * torso
        j[f"hip.{s}"] = hip
        j[f"shoulder.{s}"] = hip + np.array([0.0, np.cos(t) * torso, -np.sin(t) * torso])
        j[f"shoulder.{s}"][0] = j[f"shoulder.{s}"][0] + (0.15 if s == "left" else -0.15) - hip[0]
    m = measure(j, "left", UP, np.array([1.0, 0, 0]))
    assert m["torso_lean"] == approx(20, abs=0.2)
    assert m["hip_below_knee"] == approx(0.05, abs=0.01)


def test_width_ratios():
    j = standing_joints()
    j["wrist.left"][0], j["wrist.right"][0] = 0.3, -0.3
    j["elbow.left"][0], j["elbow.right"][0] = 0.25, -0.25
    m = measure(j, "left", UP, None)
    assert m["grip_ratio"] == approx(2.0) and m["elbow_width_ratio"] == approx(1.6667, abs=1e-3)


def test_pick_phase_frames_squat():
    # knee per frame: 175 at 1, falls to 50 at 30, back to 175 at 60; targets start 175, middle 105, end 47
    f = pick_phase_frames(PER_FRAME, TARGETS)
    assert f["start"] == 1 and f["end"] == 30 and 10 < f["middle"] < 30


def test_pick_phase_frames_driver_is_largest_change_in_tolerance_units():
    # lean changes 20 deg (4 tolerances), hip-vs-knee 0.6 torso lengths (6 tolerances): hip-vs-knee drives,
    # though 20 > 0.6 in raw numbers. Lean peaks at frame 41, hip-vs-knee at 21. A missing reading is skipped.
    per_frame = [{"frame": f, "flat": {"torso_lean": 20 * (f - 1) / 40,
                                       "hip_below_knee": -0.5 + 0.6 * (1 - abs(f - 21) / 20)}, "3d": {}}
                 for f in range(1, 42)]
    per_frame[4]["flat"]["hip_below_knee"] = None
    targets = {"phases": {"start": {"torso_lean": _target(0), "hip_below_knee": _target(-0.5)},
                          "middle": {"torso_lean": _target(10), "hip_below_knee": _target(-0.2), "estimated": True},
                          "end": {"torso_lean": _target(20), "hip_below_knee": _target(0.1)}},
               "widths": {}, "tolerance": TOLERANCE}
    assert pick_phase_frames(per_frame, targets) == {"start": 1, "middle": 11, "end": 21}


def test_pick_phase_frames_skips_single_frame_jumps():
    # The saved squat scene jumps on single frames, every other frame (frame 18: flat knee 175 between 129
    # and 122, the hips 9 cm further back). Such a frame is not a moment of the movement, even when its
    # reading lands nearest a target.
    per_frame = [{"frame": f["frame"], "flat": dict(f["flat"]), "3d": {}} for f in PER_FRAME]
    for frame in (12, 14):
        per_frame[frame - 1]["flat"]["knee_included"] = 174.9
    targets = {**TARGETS, "phases": {**TARGETS["phases"], "start": {"knee_included": _target(174.9)}}}
    f = pick_phase_frames(per_frame, targets)
    assert f["start"] == 1 and f["end"] == 30 and 10 < f["middle"] < 30


def test_jump_frames_are_the_frames_picking_skips():
    per_frame = [{"frame": f["frame"], "flat": dict(f["flat"]), "3d": {}} for f in PER_FRAME]
    for frame in (12, 14):
        per_frame[frame - 1]["flat"]["knee_included"] = 174.9
    assert jump_frames(per_frame, TARGETS) == [12, 14]
    assert jump_frames(PER_FRAME, TARGETS) == []


Z_UP = np.array([0.0, 0.0, 1.0])
SIDE = np.array([1.0, 0.0, 0.0])
SQUAT_FRAMES = 69     # form5's length; the knee moves at most ~6.5 deg a frame, about its pace


def squat_joints(f):
    """A smooth squat, z up, facing +y: standing at frame 1 and 69, deepest at 35. Torso about 0.3 long."""
    p = (1 - np.cos(2 * np.pi * (f - 1) / (SQUAT_FRAMES - 1))) / 2
    hip_mid = np.array([0.0, -0.15 * p, 0.40 - 0.18 * p])
    shoulder_mid = hip_mid + np.array([0.0, 0.10 * p, 0.30])
    j = {}
    for s, x in (("left", 1), ("right", -1)):
        j[f"hip.{s}"] = hip_mid + [0.05 * x, 0, 0]
        j[f"knee.{s}"] = np.array([0.08 * x, 0.12 * p, 0.21 - 0.03 * p])
        j[f"ankle.{s}"] = np.array([0.08 * x, 0.0, 0.03])
        j[f"shoulder.{s}"] = shoulder_mid + [0.12 * x, 0, 0]
        j[f"elbow.{s}"] = j[f"shoulder.{s}"] + [0.05 * x, -0.03, -0.10]
        j[f"wrist.{s}"] = j[f"elbow.{s}"] + [0, 0.02, 0.10]
    return j


def squat_run(joints):
    per_frame = [{"frame": f, "flat": measure(j, "right", Z_UP, SIDE), "3d": {}} for f, j in sorted(joints.items())]
    knee = {f["frame"]: f["flat"]["knee_included"] for f in per_frame}
    targets = {"phases": {"start": {"knee_included": _target(knee[1])},
                          "middle": {"knee_included": _target(knee[18])},
                          "end": {"knee_included": _target(knee[35])}},
               "widths": {}, "tolerance": TOLERANCE}
    return per_frame, targets


def test_smooth_squat_has_no_jumps():
    joints = {f: squat_joints(f) for f in range(1, SQUAT_FRAMES + 1)}
    per_frame, targets = squat_run(joints)
    assert jump_frames(per_frame, targets, joints) == [] and jump_steps(joints) == []


def test_position_jumps_a_three_frame_lurch_and_a_drop_the_angles_miss():
    # Frame 10: the whole body drops 100 mm for one frame (form5's frame 56: knees on the floor), so
    # every angle is unchanged. Frames 30-32: everything but the feet lurches 150 mm forward (form5's
    # 61-63), three in a row. Sizes are measured against the window's steady frames, not a neighbour;
    # the lurch sits off-centre in its windows, so its sizes carry up to ~1.5 frames of motion (10 mm).
    joints = {f: squat_joints(f) for f in range(1, SQUAT_FRAMES + 1)}
    joints[10] = {k: v + [0, 0, -0.10] for k, v in joints[10].items()}
    for f in (30, 31, 32):
        joints[f] = {k: (v if k.startswith("ankle") else v + [0, 0.15, 0]) for k, v in joints[f].items()}
    per_frame, targets = squat_run(joints)
    assert per_frame[9]["flat"]["knee_included"] == approx(measure(squat_joints(10), "right", Z_UP, SIDE)
                                                           ["knee_included"])      # the angle misses the drop
    frames = jump_frames(per_frame, targets, joints)
    assert frames == [10, 30, 31, 32]
    sizes = {s["frame"]: s for s in jump_sizes(joints, frames)}
    assert sizes[10]["hips_mm"] == approx(100, abs=5) and sizes[10]["knees_mm"] == approx(100, abs=5)
    for f in (30, 31, 32):
        assert sizes[f]["hips_mm"] == approx(150, abs=10) and sizes[f]["knees_mm"] == approx(150, abs=10)
    picked = pick_phase_frames(per_frame, targets, skip=frames)
    assert not set(picked.values()) & set(frames)


def test_picking_skips_the_frames_it_is_given():
    # frames 1 and 60 both read 175; skipping 1 moves the start to 60
    assert pick_phase_frames(PER_FRAME, TARGETS, skip=[1])["start"] == 60


def shifted(frames, dy=0.15, feet=False):
    """The smooth squat with everything but the feet (with them too if `feet`) moved dy forward on `frames`."""
    joints = {f: squat_joints(f) for f in range(1, SQUAT_FRAMES + 1)}
    for f in frames:
        joints[f] = {k: (v if k.startswith("ankle") and not feet else v + [0, dy, 0]) for k, v in joints[f].items()}
    return joints


@pytest.mark.parametrize("length", [3, 4, 5])
def test_a_displaced_block_is_caught_by_its_steps(length):
    # The reviewer's probe: 150 mm forward on frames 30..29+length. From 4 frames on, the 7-frame window's
    # median moves with the block, so the window rule alone sees nothing; the steps in and out catch it.
    block = list(range(30, 30 + length))
    joints = shifted(block)
    per_frame, targets = squat_run(joints)
    steps = jump_steps(joints)
    assert [s["frame"] for s in steps] == [30, 30 + length]
    assert all(s["mm"] == approx(150, abs=15) for s in steps)
    frames = jump_frames(per_frame, targets, joints)
    assert frames == block
    assert not set(pick_phase_frames(per_frame, targets, skip=frames).values()) & set(block)


def test_the_good_frame_between_two_pops_is_not_a_jump():
    # form5's 18 and 20: two single-frame pops with a good frame between. Steps fire on 18, 19, 20 and 21,
    # so 19 sits between two steps, but it is on the motion around it: not displaced. The whole body moves
    # (feet too), so no angle changes and only the position rules speak.
    joints = shifted([18, 20], feet=True)
    per_frame, targets = squat_run(joints)
    assert [s["frame"] for s in jump_steps(joints)] == [18, 19, 20, 21]
    assert jump_frames(per_frame, targets, joints) == [18, 20]


def test_jump_sizes_are_null_when_the_window_has_no_steady_frame():
    # frames 20..29 alternate 150 mm forward and back: frame 24's whole window (21..27) jumps
    joints = {f: squat_joints(f) for f in range(1, SQUAT_FRAMES + 1)}
    for f in range(20, 30):
        joints[f] = {k: (v if k.startswith("ankle") else v + [0, 0.15 if f % 2 else -0.15, 0])
                     for k, v in joints[f].items()}
    per_frame, targets = squat_run(joints)
    frames = jump_frames(per_frame, targets, joints)
    assert set(range(20, 30)) <= set(frames)
    sizes = {s["frame"]: s for s in jump_sizes(joints, frames)}
    assert sizes[24]["hips_mm"] is None and sizes[24]["knees_mm"] is None
    assert sizes[20]["hips_mm"] > 100 and sizes[20]["knees_mm"] > 100      # its window still has 17..19


def test_certain_targets_drive_ahead_of_uncertain_ones():
    # The knee moves most (24 tolerances) but the coach's two knee readings disagree: uncertain. The hip
    # (12, certain) drives, ahead of the lean (2, certain): the end is the hip's frame 21, not 41 or 31.
    per_frame = [{"frame": f, "flat": {"knee_included": 175 - 3 * (f - 1), "hip_included": 110 + 3 * abs(f - 21),
                                       "torso_lean": 15 - abs(f - 31)}, "3d": {}} for f in range(1, 42)]
    unsure = lambda v: {**_target(v), "uncertain": True}                        # noqa: E731
    targets = {"phases": {"start": {"knee_included": unsure(175), "hip_included": _target(170), "torso_lean": _target(5)},
                          "middle": {"knee_included": unsure(115), "hip_included": _target(140), "torso_lean": _target(10)},
                          "end": {"knee_included": unsure(55), "hip_included": _target(110), "torso_lean": _target(15)}},
               "widths": {}, "tolerance": TOLERANCE}
    assert pick_phase_frames(per_frame, targets)["end"] == 21


def test_picking_refuses_when_every_frame_jumps():
    with pytest.raises(ValueError, match="Every frame of the animation jumps; no position can be measured"):
        pick_phase_frames(PER_FRAME, TARGETS, skip=range(1, 61))


def test_start_and_end_are_the_drivers_turning_points_not_matches():
    # Our hip stands straighter (179) than the coach's start (171) and never gets as deep as his end (69).
    # The frame nearest 171 is mid-descent (frame 3, 172.4); matching it would make the start check pass
    # by construction. The start is the standing turning point, the end the deepest; the middle is matched.
    hip = [179 - 3.3 * (f - 1) if f <= 31 else 80 + 3.3 * (f - 31) for f in range(1, 62)]
    per_frame = [{"frame": f, "flat": {"hip_included": hip[f - 1]}, "3d": {}} for f in range(1, 62)]
    targets = {"phases": {"start": {"hip_included": _target(171)}, "middle": {"hip_included": _target(109)},
                          "end": {"hip_included": _target(69)}}, "widths": {}, "tolerance": TOLERANCE}
    assert pick_phase_frames(per_frame, targets) == {"start": 1, "middle": 22, "end": 31}
    assert phase_driver(per_frame, targets) == "hip_included" and phase_driver(PER_FRAME, TARGETS) == "knee_included"


def test_posture_never_times_the_positions():
    # posture targets sit in the same phases (build_targets, Task 6c) but are judged, never matched: a
    # neck that swings far more than the knee in tolerance units must still not drive
    per_frame = [{**f, "flat": {**f["flat"], "neck_vs_torso": 2000 - 10 * f["frame"], "head_pitch": 3 * f["frame"],
                                "back_rounding": 0.01 * f["frame"]}} for f in PER_FRAME]
    targets = {**TARGETS, "phases": {p: {**v, "neck_vs_torso": _target(n), "head_pitch": _target(n),
                                         "back_rounding": _target(n / 1000)}
                                     for (p, v), n in zip(TARGETS["phases"].items(), (2000, 1000, 10))}}
    assert phase_driver(per_frame, targets) == "knee_included"
    assert pick_phase_frames(per_frame, targets) == pick_phase_frames(PER_FRAME, TARGETS)
    only_posture = {**targets, "phases": {p: {k: v for k, v in ph.items() if k in ("neck_vs_torso", "back_rounding")}
                                          for p, ph in targets["phases"].items()}}
    with pytest.raises(ValueError):
        phase_driver(per_frame, only_posture)


# --- the torso line counted from the character's own upright bind pose (task 18) ----------------------
# Our rig's shoulder joints sit 76 mm behind its hip joints when it stands tall, so its torso line (hip
# joints -> shoulder joints) reads 15 degrees back at rest where a person's MediaPipe landmarks read about
# 0. Ours is turned by its rest pose's lean, about the side camera's axis (rest_turn); the coach's is not.
SIDE = np.array([1.0, 0.0, 0.0])        # the figure faces -z, so a side camera looks along x


def rig_rest(back=0.1):
    """standing_joints with the shoulders (and the arms hanging from them) `back` behind the hips: the
    torso line leans back atan(back / 0.5) standing tall, as the cartoon rig's does."""
    j = standing_joints()
    for s in ("left", "right"):
        for k in ("shoulder", "elbow", "wrist"):
            j[f"{k}.{s}"] = j[f"{k}.{s}"] + np.array([0.0, 0.0, back])
    return j


def tipped(j, degrees):
    """Everything above the hips (shoulders, elbows, wrists) turned `degrees` forward (toward -z) about
    the hips' line (the x axis through the hips' midpoint)."""
    out = {k: v.copy() for k, v in j.items()}
    mid = (j["hip.left"] + j["hip.right"]) / 2
    a = np.radians(degrees)
    turn = np.array([[1, 0, 0], [0, np.cos(a), np.sin(a)], [0, -np.sin(a), np.cos(a)]])  # +y toward -z
    for s in ("left", "right"):
        for k in ("shoulder", "elbow", "wrist"):
            out[f"{k}.{s}"] = mid + turn @ (j[f"{k}.{s}"] - mid)
    return out


def test_the_rest_lean_is_signed_toward_the_front():
    from metrics import rest_lean
    front = np.array([0.0, 0.0, -1.0])
    assert rest_lean(rig_rest(), UP, SIDE, front) == approx(-np.degrees(np.arctan(0.1 / 0.5)), abs=1e-6)
    assert rest_lean(rig_rest(-0.1), UP, SIDE, front) == approx(np.degrees(np.arctan(0.1 / 0.5)), abs=1e-6)
    assert rest_lean(standing_joints(), UP, SIDE, front) == approx(0.0, abs=1e-9)


def test_the_rest_pose_reads_upright_once_turned():
    from metrics import rest_turn
    j = rig_rest()
    turn = rest_turn(j, UP, SIDE)
    assert measure(j, "left", UP, SIDE)["torso_lean"] == approx(11.31, abs=0.01)     # the joint line: leaning back
    assert measure(j, "left", UP, None, torso_turn=turn)["torso_lean"] == approx(0.0, abs=1e-6)
    m = measure(j, "left", UP, SIDE, torso_turn=turn)
    assert m["torso_lean"] == approx(0.0, abs=1e-6)
    assert m["hip_included"] == approx(180.0, abs=1e-4)            # legs straight down under an upright torso
    assert m["upperarm_vs_torso"] == approx(0.0, abs=1e-4)         # the arm hangs along the upright torso


def test_a_20_degree_forward_tilt_reads_20():
    from metrics import rest_turn
    rest = rig_rest()
    turn = rest_turn(rest, UP, SIDE)
    j = tipped(rest, 20)
    assert measure(j, "left", UP, SIDE)["torso_lean"] == approx(20 - 11.31, abs=0.01)   # what the joints said
    assert measure(j, "left", UP, None, torso_turn=turn)["torso_lean"] == approx(20.0, abs=1e-6)
    m = measure(j, "left", UP, SIDE, torso_turn=turn)
    assert m["torso_lean"] == approx(20.0, abs=1e-6)
    assert m["hip_included"] == approx(160.0, abs=1e-4)
    assert m["upperarm_vs_torso"] == approx(0.0, abs=1e-4)         # the arms went with the torso
    # in 3D each side's torso line keeps its own sideways part (shoulders wider than hips); only its
    # side-seen part is turned
    m3 = measure(j, "left", UP, None, torso_turn=turn)
    side_torso = np.array([0.06, 0.5 / np.cos(np.radians(11.31)), 0.0])     # turned upright, then tipped 20
    assert m3["upperarm_vs_torso"] == approx(np.degrees(np.arctan(0.06 / side_torso[1])), abs=0.05)
    # the turn is the rest pose's, fixed: the same figure tipped back 10 reads 10
    assert measure(tipped(rest, -10), "left", UP, SIDE, torso_turn=turn)["torso_lean"] == approx(10.0, abs=1e-6)


def test_the_turn_is_about_the_view_axis_and_leaves_the_rest_alone():
    from metrics import rest_turn
    rest = rig_rest()
    turn = rest_turn(rest, UP, SIDE)
    assert np.allclose(turn @ turn.T, np.eye(3)) and np.linalg.det(turn) == approx(1.0)
    assert np.allclose(turn @ SIDE, SIDE)                           # a turn about the side camera's axis
    j = tipped(rest, 20)
    plain, turned = measure(j, "left", UP, SIDE), measure(j, "left", UP, SIDE, torso_turn=turn)
    for k in ("knee_included", "elbow_included", "hip_below_knee", "grip_ratio", "elbow_width_ratio"):
        assert turned[k] == approx(plain[k], nan_ok=True), k
    # an upright figure (a person's landmarks) needs no turn
    assert np.allclose(rest_turn(standing_joints(), UP, SIDE), np.eye(3))
    none, default = measure(j, "left", UP, SIDE, torso_turn=None), measure(j, "left", UP, SIDE)
    assert none.keys() == default.keys() and all(none[k] == approx(default[k], nan_ok=True) for k in none)


# ---- the bind pose must stand upright (task 19 review: one rule, the Studio's, the solver imports it) ----

def test_bind_pose_warning():
    from metrics import BIND_LEAN_MAX, CHEST_TILT_MAX, bind_pose_warning
    assert (BIND_LEAN_MAX, CHEST_TILT_MAX) == (25.0, 30.0)
    assert bind_pose_warning(-15.4, -5.0) is None           # the cartoon: shoulders behind the hips, chest upright
    assert bind_pose_warning(1.8, -4.0) is None             # a stock MPFB human
    assert bind_pose_warning(25.0, 30.0) is None            # on the line is still upright
    w = bind_pose_warning(-26.0, None)
    assert w == ("WARNING: the rig's bind pose does not stand upright (its torso line leans -26.0 degrees); every "
                 "torso angle of ours is counted from it, so they may read off")
    assert "its chest tips +31.0 degrees" in bind_pose_warning(3.0, 31.0)
    assert bind_pose_warning(30.0, -40.0).count("degrees") == 2


# ---- the legs counted from how their skin looks in the bind pose (task 20) ----
# The cartoon's hip joint sits toward the front of the thigh's top, so its thigh joint line slopes 10.6 degrees back
# standing while the thigh's skin outline leans back 4.5 (MediaPipe on its bind render: about 4). Each thigh line is
# turned onto the thigh's own skin outline as the bind pose shows it (rest_leg_turns, about the side camera's axis);
# the shin turns with it (the knee keeps its joints' reading: the cartoon's shin outline is all knee and heel). The
# coach's landmarks are not turned.

LEG = 0.4               # thigh and shin length in standing_joints


def _down(degrees):
    """Unit direction `degrees` forward (toward -z, the figure's front) of straight down."""
    a = np.radians(degrees)
    return np.array([0.0, -np.cos(a), -np.sin(a)])


def legs_at(j, thigh, shin):
    """j with each leg's knee and ankle joints placed with the thigh's joint line `thigh` and the shin's `shin`
    degrees forward of straight down."""
    out = {k: v.copy() for k, v in j.items()}
    for s in ("left", "right"):
        out[f"knee.{s}"] = out[f"hip.{s}"] + LEG * _down(thigh)
        out[f"ankle.{s}"] = out[f"knee.{s}"] + LEG * _down(shin)
    return out


JOINT, SKIN = -10.6, -4.5   # the cartoon: thigh joint line and thigh skin outline at rest, + forward of straight down


def outlines(degrees=SKIN):
    return {"left": _down(degrees), "right": _down(degrees)}


def test_the_bind_pose_reads_as_its_skin_stands():
    """hip = 180 less the thigh outline's lean back (the torso upright); the knee as its joints (straight)."""
    from metrics import rest_leg_turns, rest_turn
    rest = legs_at(rig_rest(), JOINT, JOINT)
    legs, torso = rest_leg_turns(rest, SIDE, outlines()), rest_turn(rest, UP, SIDE)
    assert set(legs) == {"thigh.left", "thigh.right", "shin.left", "shin.right"}
    assert measure(rest, "left", UP, SIDE, torso_turn=torso)["hip_included"] == approx(180 + JOINT, abs=1e-4)
    m = measure(rest, "left", UP, SIDE, torso_turn=torso, leg_turns=legs)
    assert m["hip_included"] == approx(180 + SKIN, abs=1e-4)                  # 175.5: the skin leans back 4.5
    assert m["knee_included"] == approx(180.0, abs=1e-4)
    assert m["hip_below_knee"] == approx(-LEG * np.cos(np.radians(SKIN)) / np.hypot(0.5, 0.1), abs=1e-6)


def test_a_thigh_level_as_its_skin_reads_parallel():
    from metrics import rest_leg_turns, rest_turn
    rest = legs_at(rig_rest(), JOINT, JOINT)
    legs, torso = rest_leg_turns(rest, SIDE, outlines()), rest_turn(rest, UP, SIDE)
    level = 90 - (SKIN - JOINT)                     # the joint line where the skin outline is level
    j = legs_at(rest, level, 0.0)
    plain = measure(j, "left", UP, SIDE, torso_turn=torso)
    assert plain["hip_below_knee"] == approx(-LEG * np.sin(np.radians(6.1)) / np.hypot(0.5, 0.1), abs=1e-6)
    m = measure(j, "left", UP, SIDE, torso_turn=torso, leg_turns=legs)
    assert m["hip_below_knee"] == approx(0.0, abs=1e-9) and m["thigh_level"] == approx(0.0, abs=1e-6)
    assert plain["thigh_level"] == approx(-6.1, abs=1e-6)                   # the joint line falls to the knee
    assert m["hip_included"] == approx(90.0, abs=1e-4)
    assert m["knee_included"] == approx(plain["knee_included"], abs=1e-9)   # the shin turns with the thigh
    deep = measure(legs_at(rest, level + 5, 0.0), "left", UP, SIDE, torso_turn=torso, leg_turns=legs)
    assert deep["hip_below_knee"] == approx(LEG * np.sin(np.radians(5)) / np.hypot(0.5, 0.1), abs=1e-6)
    assert deep["thigh_level"] == approx(5.0, abs=1e-6)                     # + = the knee higher: deep


def test_where_the_joints_sit_inside_the_skin_does_not_matter():
    """Two rigs with the same skin, their hip joints placed differently (thigh joint lines 10.6 and 2 degrees
    back), read the same when their skins are posed alike."""
    from metrics import rest_leg_turns
    reads = []
    for joint in (JOINT, -2.0):
        rest = legs_at(rig_rest(), joint, joint)
        legs = rest_leg_turns(rest, SIDE, outlines())
        seen = 70.0                                  # the thigh's skin 70 degrees forward of straight down
        j = legs_at(rest, seen - (SKIN - joint), -30.0 - (SKIN - joint))
        reads.append(measure(j, "left", UP, SIDE, leg_turns=legs))
    for k in ("hip_included", "knee_included", "hip_below_knee"):
        assert reads[0][k] == approx(reads[1][k], abs=1e-6), k


def test_a_side_without_an_outline_is_read_on_its_joints():
    from metrics import rest_leg_turns
    rest = legs_at(rig_rest(), JOINT, JOINT)
    legs = rest_leg_turns(rest, SIDE, {"left": None, "right": _down(SKIN)})
    assert np.allclose(legs["thigh.left"], np.eye(3)) and np.allclose(legs["shin.left"], np.eye(3))
    assert not np.allclose(legs["thigh.right"], np.eye(3))
    assert np.allclose(legs["shin.right"], legs["thigh.right"])


def test_the_leg_turns_are_about_the_view_axis_and_leave_the_rest_alone():
    from metrics import rest_leg_turns, rest_turn
    rest = legs_at(rig_rest(), JOINT, JOINT)
    legs, torso = rest_leg_turns(rest, SIDE, outlines()), rest_turn(rest, UP, SIDE)
    for r in legs.values():
        assert np.allclose(r @ r.T, np.eye(3)) and np.linalg.det(r) == approx(1.0)
        assert np.allclose(r @ SIDE, SIDE)
    j = tipped(legs_at(rest, 70.0, -40.0), 25)
    plain, turned = measure(j, "left", UP, SIDE, torso_turn=torso), measure(j, "left", UP, SIDE, torso_turn=torso,
                                                                           leg_turns=legs)
    for k in ("torso_lean", "elbow_included", "upperarm_vs_torso", "grip_ratio", "elbow_width_ratio"):
        assert turned[k] == approx(plain[k], nan_ok=True), k
    none, default = measure(j, "left", UP, SIDE, leg_turns=None), measure(j, "left", UP, SIDE)
    assert all(none[k] == approx(default[k], nan_ok=True) for k in none)


def _tube(top, bottom, r_top, r_bottom, rings=12, around=16):
    """A tapered tube of skin from `top` to `bottom` (the limb's own axis): its vertices and edges (around each
    ring and down between rings), x across the body."""
    pts, edges = [], []
    axis = (bottom - top) / np.linalg.norm(bottom - top)
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.cross(axis, e1)
    for i in range(rings):
        t = i / (rings - 1)
        c, r = top + t * (bottom - top), r_top + t * (r_bottom - r_top)
        for k in range(around):
            a = 2 * np.pi * k / around
            pts.append(c + r * (np.cos(a) * e1 + np.sin(a) * e2))
            edges.append((i * around + k, i * around + (k + 1) % around))
            if i:
                edges.append(((i - 1) * around + k, i * around + k))
    return np.array(pts), np.array(edges)


def test_the_skin_outline_is_the_midline_of_the_limbs_own_skin():
    """A thigh of skin whose own axis leans back 4.5 degrees, tapering 80 -> 50 mm, around a joint line from a hip
    joint 30 mm toward its front to the knee at its middle (10.6 back): the outline reads the skin's 4.5, not the
    joints'; in another plane than the side's, the same tube reads its slope there."""
    from metrics import limb_outline, limb_slope
    front = np.array([0.0, 0.0, -1.0])
    knee = np.array([0.1, 0.5, 0.0])
    top = knee - 0.42 * _down(SKIN)
    hip = top + np.array([0.0, 0.0, -0.03]) + 0.03 * _down(SKIN)       # toward the front of the thigh's top
    pts, edges = _tube(top, knee, 0.08, 0.05)
    d = limb_outline(pts, edges, hip, knee, SIDE)
    assert limb_slope(knee - hip, UP, SIDE, front) < -8                     # the joint line slopes further back
    assert limb_slope(d, UP, SIDE, front) == approx(SKIN, abs=0.3)
    assert np.linalg.norm(d) == approx(1.0) and abs(d @ SIDE) < 1e-9       # a unit direction in the side plane
    assert limb_outline(pts[:10], edges[:5], hip, knee, SIDE) is None      # too little skin: no outline
    # its front edge (task 21: the line that is the thigh's top at the bottom of a squat) runs in toward the axis
    # with the taper, 80 -> 50 mm over 0.42 m: 4.1 degrees further back; the back edge as much forward
    taper = np.degrees(np.arctan(0.03 / 0.42))
    assert limb_slope(limb_outline(pts, edges, hip, knee, SIDE, edge=front), UP, SIDE, front) == approx(SKIN - taper, abs=0.3)
    assert limb_slope(limb_outline(pts, edges, hip, knee, SIDE, edge=-front), UP, SIDE, front) == approx(SKIN + taper, abs=0.3)
    # seen on other planes - a side camera 5 degrees up, one turned 30 degrees toward the front - the outline reads
    # the skin's own axis as that plane shows it
    for normal in (np.array([np.cos(np.radians(5)), np.sin(np.radians(5)), 0.0]),
                   np.array([np.cos(np.radians(30)), 0.0, -np.sin(np.radians(30))])):
        seen = limb_outline(pts, edges, hip, knee, normal)
        assert abs(seen @ normal) < 1e-9
        assert limb_slope(seen, UP, normal, front) == approx(limb_slope(knee - top, UP, normal, front), abs=0.3), normal


def test_the_rest_leg_slopes_are_signed_toward_the_front():
    from metrics import rest_leg_slopes
    front = np.array([0.0, 0.0, -1.0])
    s = rest_leg_slopes(legs_at(rig_rest(), -10.6, -6.2), UP, SIDE, front)
    assert s == {k: approx(v, abs=1e-6) for k, v in
                 {"thigh.left": -10.6, "thigh.right": -10.6, "shin.left": -6.2, "shin.right": -6.2}.items()}


def test_the_bind_legs_warning():
    """Task 21: the bind pose's thigh as its skin outline shows it, cross-checked with MediaPipe on a render of the
    bind pose: the slope of its camera-side hip -> knee landmarks, signed (task 22: a hip angle less the lean could not
    tell a thigh forward from one back). The cartoon: outline -4.5, MediaPipe about -4: quiet. Past BIND_THIGH_MAX,
    or a thigh with no outline: said."""
    from metrics import BIND_THIGH_MAX, bind_legs_warning
    assert BIND_THIGH_MAX == 5.0
    mp = {"hip_included": 172.0, "torso_lean": 4.1, "thigh_slope": -3.9}
    assert bind_legs_warning({"thigh.left": -5.0, "thigh.right": -4.5}, mp, "right") is None
    w = bind_legs_warning({"thigh.left": -5.0, "thigh.right": -11.0}, mp, "right")
    assert w == ("The bind pose's thigh reads -11.0 degrees from straight down by its skin outline but -3.9 by "
                 "MediaPipe on its render: the legs, counted from the outline, may read off")
    # the same size of slope the other way (the thigh forward): said - the hip angle alone read both as -3.9
    forward = {**mp, "thigh_slope": 3.9}
    assert "+3.9 by MediaPipe" in bind_legs_warning({"thigh.left": -5.0, "thigh.right": -4.5}, forward, "right")
    assert "left thigh's outline" in bind_legs_warning({"thigh.left": None, "thigh.right": -4.5}, mp, "right")
    assert bind_legs_warning({"thigh.left": -5.0, "thigh.right": -4.5}, {"why": "no body"}, "right") is None
    assert bind_legs_warning({"thigh.left": -5.0, "thigh.right": -4.5}, None, "right") is None
    assert bind_legs_warning({"thigh.left": -5.0, "thigh.right": -4.5}, {**mp, "thigh_slope": None}, "right") is None


def test_a_limbs_slope_in_a_picture():
    """Degrees a hip -> knee line in a picture (y down) points toward where the body faces, + forward of straight
    down; None without a facing (not a side view)."""
    from metrics import picture_slope
    assert picture_slope((100.0, 100.0), (90.0, 200.0), "right") == approx(math.degrees(math.atan2(-10, 100)))
    assert picture_slope((100.0, 100.0), (90.0, 200.0), "left") == approx(math.degrees(math.atan2(10, 100)))
    assert picture_slope((100.0, 100.0), (200.0, 100.0), "right") == approx(90.0)        # level, forward
    assert picture_slope((100.0, 100.0), (90.0, 200.0), None) is None
