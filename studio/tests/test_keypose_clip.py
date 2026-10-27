"""keypose_clip's numpy core (pipeline/blender/keypose_clip.py): the reference pictures' key positions turned
into bone turns and timed as a real lift. The Blender half (`clip_from_keyposes`, a retarget.Clip) is tested
by studio/tests/blender/test_keypose.py."""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from synthetic_pose import mp_pose

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("keypose_clip", ROOT / "pipeline" / "blender" / "keypose_clip.py")
kc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kc)

# The Mixamo Y Bot's rest (3d-character/mixamo/squat.fbx, read in Blender: it faces -Y, its left is +X, up +Z),
# which is the character frame keypose_clip works in.
REST = {k: np.array(v) for k, v in {
    "Head": [0.0, 0.0152, 1.5993], "HeadTop_End": [0.0, -0.0598, 1.8197], "Neck": [0.0, 0.032, 1.5031],
    "LeftShoulder": [0.0457, 0.0332, 1.4459], "LeftArm": [0.1516, 0.0555, 1.4406],
    "LeftForeArm": [0.43, 0.0555, 1.4406], "LeftHand": [0.7133, 0.0555, 1.4406],
    "LeftHandMiddle1": [0.8087, 0.0555, 1.4406], "LeftUpLeg": [0.0821, 0.0005, 0.9752],
    "LeftLeg": [0.0821, -0.003, 0.5315], "LeftFoot": [0.0821, 0.0274, 0.0873], "LeftToeBase": [0.0821, -0.0797, 0.0],
}.items()}
REST.update({k.replace("Left", "Right"): v * np.array([-1.0, 1.0, 1.0]) for k, v in list(REST.items())
             if k.startswith("Left")})


def side_angles(points):
    """(lean, hip, knee) on the character's side plane (its y-z: Y back, Z up), left side."""
    def yz(n):
        return np.array([-points[n][1], points[n][2]])      # (forward, up)

    def included(a, b, c):
        u, v = a - b, c - b
        return math.degrees(math.acos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)))
    hip = (yz("left_hip") + yz("right_hip")) / 2
    sh = (yz("left_shoulder") + yz("right_shoulder")) / 2
    t = sh - hip
    return (math.degrees(math.atan2(t[0], t[1])), included(sh, yz("left_hip"), yz("left_knee")),
            included(yz("left_hip"), yz("left_knee"), yz("left_ankle")))


def angle(u, v):
    u, v = np.asarray(u, float), np.asarray(v, float)
    return math.degrees(math.acos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)))


# --- reading a key position into the character's frame -------------------------------------------

def test_a_side_picture_is_read_square_to_the_picture():
    # person's left -> +X, up -> +Z, facing -Y; in a side picture the picture is the side plane
    entry = mp_pose(0, 180, 180, facing="right")
    rows, swapped = kc.character_frame({k: np.array(v) for k, v in entry["world"].items()}, "right")
    assert not swapped
    assert np.allclose(rows, [[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    rows, _ = kc.character_frame({k: np.array(v) for k, v in mp_pose(0, 180, 180, facing="left")["world"].items()},
                                 "left")
    assert np.allclose(rows, [[0, 0, -1], [1, 0, 0], [0, -1, 0]])


def test_off_a_side_view_the_frame_is_the_hips_line_made_level():
    entry = mp_pose(0, 180, 180)
    world = {k: np.array(v) for k, v in entry["world"].items()}
    world["left_hip"] = world["left_hip"] + np.array([0.03, -0.02, 0.0])   # turned and tipped a little
    rows, _ = kc.character_frame(world, None)
    left = world["left_hip"] - world["right_hip"]
    left[1] = 0.0
    assert np.allclose(rows[0], left / np.linalg.norm(left))
    assert np.allclose(rows[2], [0, -1, 0])
    assert np.allclose(np.cross(rows[0], rows[1]), rows[2])          # a proper frame, never a mirror


def test_sides_labelled_the_wrong_way_round_are_swapped_back():
    entry = mp_pose(20, 90, 90)
    entry["world"]["left_knee"][2] = 0.18          # the person's left knee, out to the left (+z facing right)
    for key in ("world", "pixels"):                # ... and MediaPipe calls every left point right
        for n in [n for n in entry[key] if n.startswith("left_")]:
            r = "right_" + n[5:]
            entry[key][n], entry[key][r] = entry[key][r], entry[key][n]
    body, arms, found = kc.key_points(entry)
    assert found["swapped_sides"]
    assert body["left_hip"][0] > 0 > body["right_hip"][0]
    assert body["left_knee"][0] == pytest.approx(0.14)      # its 0.18 out, with the right knee's 0.1 out


def test_a_side_picture_is_made_left_right_symmetric():
    # in a side picture left-right is MediaPipe's depth guess: on the squat's it put the near ankle on the far
    # side at the bottom and the hidden arm straight out sideways. The exercises are left-right symmetric
    entry = mp_pose(20, 90, 90)                    # the camera sees his right side
    w = entry["world"]
    w["left_knee"][2] += 0.10                      # left knee 10 cm further out (left is +z facing right)
    w["right_ankle"][2] += 0.12                    # right ankle 12 cm toward the left
    w["left_elbow"] = [w["left_shoulder"][0], w["left_shoulder"][1], w["left_shoulder"][2] + 0.28]
    w["nose"][2] += 0.05
    picture, _, _ = kc.key_points({**entry, "world": mp_pose(20, 90, 90)["world"]})
    body, arms, found = kc.key_points(entry)
    for j in ("hip", "knee", "ankle", "heel", "foot_index", "shoulder", "ear"):
        assert body[f"left_{j}"][0] == pytest.approx(-body[f"right_{j}"][0])
        for s in ("left", "right"):
            assert body[f"{s}_{j}"][1:] == pytest.approx(picture[f"{s}_{j}"][1:])     # side plane untouched
    assert body["left_knee"][0] == pytest.approx(0.15)       # (0.2 out + 0.1 out) / 2
    assert body["left_ankle"][0] == pytest.approx(0.04)      # (0.1 out - 0.02 out) / 2
    assert body["nose"][0] == 0.0
    for j in ("shoulder", "elbow", "wrist", "index"):
        assert arms[f"left_{j}"] == pytest.approx(arms[f"right_{j}"] * np.array([-1.0, 1.0, 1.0]))
    assert found["mirrored_arm"] == "left"


def test_side_angles_come_from_the_picture_not_the_noisy_depth():
    entry = mp_pose(20, 90, 90)
    clean = {k: list(v) for k, v in entry["world"].items()}
    # MediaPipe's 3D lift of a side picture: knees and ankles off in depth, and its own side-plane guess off
    for n, (dx, dy, dz) in {"left_knee": (0.04, -0.03, 0.07), "right_knee": (0.02, 0.03, -0.06),
                            "left_ankle": (-0.03, 0.0, 0.09), "left_shoulder": (0.05, 0.0, 0.0)}.items():
        entry["world"][n] = [a + b for a, b in zip(entry["world"][n], (dx, dy, dz))]
    raw, _, _ = kc.key_points({**entry, "pixels": None})            # depth alone: off
    assert max(abs(a - b) for a, b in zip(side_angles(raw), (20, 90, 90))) > 4
    body, _, _ = kc.key_points(entry)
    assert side_angles(body) == pytest.approx((20, 90, 90), abs=0.3)
    clean_body, _, _ = kc.key_points({**entry, "world": clean})
    assert body["left_knee"][0] == pytest.approx(clean_body["left_knee"][0] + 0.065)   # both knees' depth, kept


def test_a_straight_leg_is_not_bent_sideways_by_depth_noise():
    entry = mp_pose(0, 180, 175)
    entry["world"]["left_ankle"][2] += 0.15        # the coach's standing picture: ankle 15 cm out, knee in line
    body, _, found = kc.key_points(entry)
    h, k, a = body["left_hip"], body["left_knee"], body["left_ankle"]
    assert angle(k - h, a - k) == pytest.approx(5.0, abs=1.0)     # unclamped: 25 degrees, knee bent inward
    assert side_angles(body)[2] == pytest.approx(175, abs=0.1)      # the side view is untouched
    assert found["knees"]["left"]["clamped"]


def test_knees_out_in_a_deep_squat_are_kept():
    entry = mp_pose(20, 70, 50)
    for side, sign in (("left", 1.0), ("right", -1.0)):
        entry["world"][f"{side}_knee"][2] += sign * 0.12 * (1 if entry["facing"] == "right" else -1)
    body, _, found = kc.key_points(entry)
    assert body["left_knee"][0] - body["left_hip"][0] == pytest.approx(0.12, abs=0.001)
    assert not found["knees"]["left"]["clamped"]


# --- the bones -------------------------------------------------------------------------------------

def test_each_bone_lands_on_its_landmark_direction():
    entry = mp_pose(20, 90, 90)                     # the seen (right) arm bent: elbow down and back, hand up
    sh = np.array(entry["world"]["right_shoulder"])
    entry["world"]["right_elbow"] = list(sh + np.array([-0.1, 0.22, -0.12]))
    entry["world"]["right_wrist"] = list(sh + np.array([0.05, -0.02, -0.2]))
    body, arms, _ = kc.key_points(entry)
    turns = kc.bone_turns(body, arms, REST)
    for side, S in (("left", "Left"), ("right", "Right")):
        down = np.array([0, 0, -1.0])               # legs: counted from a straight, vertical leg
        assert angle(turns[f"{S}UpLeg"] @ down, body[f"{side}_knee"] - body[f"{side}_hip"]) < 0.01
        assert angle(turns[f"{S}Leg"] @ down, body[f"{side}_ankle"] - body[f"{side}_knee"]) < 0.01
        assert angle(turns[f"{S}Arm"] @ (REST[f"{S}ForeArm"] - REST[f"{S}Arm"]),
                     arms[f"{side}_elbow"] - arms[f"{side}_shoulder"]) < 0.01
        assert angle(turns[f"{S}ForeArm"] @ (REST[f"{S}Hand"] - REST[f"{S}ForeArm"]),
                     arms[f"{side}_wrist"] - arms[f"{side}_elbow"]) < 0.01
        assert np.allclose(turns[f"{S}Hand"], turns[f"{S}ForeArm"])            # hands keep the forearm's turn
        assert not np.allclose(turns[f"{S}Hand"], turns[f"{S}Arm"])            # ... not the upper arm's
    trunk = (body["left_shoulder"] + body["right_shoulder"]) / 2 - (body["left_hip"] + body["right_hip"]) / 2
    for bone in ("Hips", "Spine", "Spine1", "Spine2"):
        assert angle(turns[bone] @ np.array([0, 0, 1.0]), trunk) < 0.01


def test_a_forearm_bends_on_its_upper_arm_without_twisting():
    # the squat's arms: elbow down and back, hand up by the shoulder. A roll the upper arm does not share would
    # twist the skin at the elbow, so the forearm's turn on the upper arm is a bend about an axis square to it
    entry = mp_pose(20, 90, 90)                     # the camera sees his right arm (right is -z facing right)
    sh = np.array(entry["world"]["right_shoulder"])
    entry["world"]["right_elbow"] = list(sh + np.array([-0.1, 0.22, -0.12]))
    entry["world"]["right_wrist"] = list(sh + np.array([0.05, -0.02, -0.2]))
    body, arms, _ = kc.key_points(entry)
    turns = kc.bone_turns(body, arms, REST)
    for s in ("Left", "Right"):
        upper, fore = turns[f"{s}Arm"], turns[f"{s}ForeArm"]
        bend = fore @ upper.T
        axis = np.array([bend[2, 1] - bend[1, 2], bend[0, 2] - bend[2, 0], bend[1, 0] - bend[0, 1]])
        carried = upper @ (REST[f"{s}Hand"] - REST[f"{s}ForeArm"])
        assert np.linalg.norm(axis) > 0.5                       # it does bend
        assert abs(axis @ carried) / np.linalg.norm(axis) / np.linalg.norm(carried) < 1e-9


def test_every_turn_is_a_rotation():
    body, arms, _ = kc.key_points(mp_pose(20, 90, 90))
    for bone, r in kc.bone_turns(body, arms, REST).items():
        assert np.allclose(r @ r.T, np.eye(3), atol=1e-9), bone
        assert np.linalg.det(r) == pytest.approx(1.0), bone


def test_standing_tall_turns_the_trunk_nowhere():
    # the Y Bot's rest joint line leans back 6.7 degrees; a person standing tall reads 0 in MediaPipe, and the
    # Studio counts our lean from the bind pose: upright is upright, not the rest line
    body, arms, _ = kc.key_points(mp_pose(0, 180, 180))
    turns = kc.bone_turns(body, arms, REST)
    for bone in ("Hips", "Spine", "Spine1", "Spine2", "Neck", "Head"):
        assert angle(turns[bone] @ np.array([0, 0, 1.0]), [0, 0, 1]) < 0.01, bone


def test_a_straight_vertical_leg_turns_the_legs_nowhere():
    # the Y Bot's rest knee is bent 4.4 degrees (its ankle 3 cm behind the knee); the rig's (the skin's) is
    # straight, and the Studio reads our knee off the rig's joints - so the legs are counted from a straight,
    # vertical leg, as the trunk is from upright
    body, arms, _ = kc.key_points(mp_pose(0, 180, 180))
    turns = kc.bone_turns(body, arms, REST)
    for bone in ("LeftUpLeg", "LeftLeg", "RightUpLeg", "RightLeg"):
        assert kc._turned_by(turns[bone]) < 1e-6, bone
    body, arms, _ = kc.key_points(mp_pose(20, 90, 90))
    turns = kc.bone_turns(body, arms, REST)
    down = np.array([0, 0, -1.0])
    assert angle(turns["LeftUpLeg"] @ down, turns["LeftLeg"] @ down) == pytest.approx(90.0, abs=1e-6)


def test_the_back_stays_straight_while_the_trunk_leans():
    body, arms, _ = kc.key_points(mp_pose(35, 80, 70))
    turns = kc.bone_turns(body, arms, REST)
    for bone in ("Spine", "Spine1", "Spine2"):
        assert np.allclose(turns[bone], turns["Hips"], atol=1e-9)


def test_a_foot_flat_on_the_floor_stays_flat():
    # MediaPipe reads a flat foot's heel->toe line a few degrees off level (the coach: 9); the planted foot is level
    for pitch in (9.0, -6.0):
        body, arms, _ = kc.key_points(mp_pose(0, 180, 180, foot_pitch=pitch))
        turns = kc.bone_turns(body, arms, REST)
        sole = turns["LeftFoot"] @ np.array([0, -1.0, 0])
        assert abs(sole[2]) < 1e-9
    # a foot on its toes (a plank's) keeps its pitch
    body, arms, _ = kc.key_points(mp_pose(0, 180, 180, foot_pitch=60.0))
    sole = kc.bone_turns(body, arms, REST)["LeftFoot"] @ np.array([0, -1.0, 0])
    assert math.degrees(math.asin(-sole[2])) == pytest.approx(60.0, abs=0.5)


def test_the_head_follows_the_face():
    # the ear->nose line of a level gaze points 12 degrees below level (studio/measure/posture.py)
    body, arms, _ = kc.key_points(mp_pose(0, 180, 180, face_down=12.0))
    level = kc.bone_turns(body, arms, REST)
    assert angle(level["Head"] @ np.array([0, -1.0, 0]), [0, -1, 0]) < 0.01
    body, arms, _ = kc.key_points(mp_pose(0, 180, 180, face_down=42.0))
    down = kc.bone_turns(body, arms, REST)
    gaze = down["Head"] @ np.array([0, -1.0, 0])
    assert math.degrees(math.asin(-gaze[2])) == pytest.approx(30.0, abs=0.1)
    neck = down["Neck"] @ np.array([0, -1.0, 0])                  # the neck takes part of it
    assert math.degrees(math.asin(-neck[2])) == pytest.approx(30.0 * kc.NECK_FOLLOW, abs=0.1)


# --- timing ------------------------------------------------------------------------------------------

def test_the_rep_is_timed_like_a_real_lift():
    s = kc.timeline(30)
    down, hold, up = round(1.5 * 30), round(0.2 * 30), round(1.0 * 30)
    assert len(s) == down + hold + up                     # the last frame leads into the first: no repeat at the seam
    assert s[0] == 0.0 and s[down] == 1.0 and 0.0 < s[-1] < 0.01
    assert all(v == 1.0 for v in s[down:down + hold + 1])                  # the hold at the bottom
    assert all(b > a for a, b in zip(s[:down], s[1:down + 1]))             # down, steadily
    assert all(b < a for a, b in zip(s[down + hold:-1], s[down + hold + 1:]))
    # eased in and out: the first and last steps of each stroke are far smaller than the middle one
    steps = np.diff(s[:down + 1])
    assert steps[0] < steps[down // 2] / 10 and steps[-1] < steps[down // 2] / 10


def test_pullup_pulls_up_then_lowers_more_slowly_with_matching_recorded_phases():
    # Photos give positions, not a timestamp. The declared tempo must follow
    # exercise semantics and the inspector must use the changed actual frames.
    keys = [kc.key_pose(mp_pose(0, 180, 180), REST),
            kc.key_pose(mp_pose(20, 90, 90, phase='end'), REST)]
    fps = 16
    parameters = kc.timeline(fps, 'Pull-Up')
    peak = parameters.index(1.)
    assert peak == fps
    assert len(parameters) == len(kc.timeline(fps)) == 43
    returning = len(parameters) - (peak + round(.2 * fps))
    assert returning > peak
    assert all(a <= b for a, b in zip(parameters[:peak], parameters[1:peak + 1]))
    assert all(a >= b for a, b in zip(parameters[peak:], parameters[peak + 1:]))
    poses = kc.motion(keys, fps=fps, feet='free', exercise='Pull-Up')
    for name in keys[1]:
        assert np.allclose(poses[peak][name], keys[1][name])
    recorded = kc.reference_timing({'exercise': 'Pull-Up'}, ['start', 'end'], keys, fps=fps)
    assert recorded['phase_frames'] == {'start': 1, 'middle': 9, 'end': 17}
    assert 'not measured' in recorded['timing']['source']
    bench = kc.reference_timing({'exercise': 'Flat Barbell Bench Press'}, ['start', 'end'], keys, fps=fps)
    assert bench['phase_frames']['end'] == 25
    assert bench['timing']['forward_s'] > bench['timing']['return_s']
    pulldown = kc.reference_timing({'exercise': 'Lat Pulldown (seated, to the front)'},
        ['start', 'end'], keys, fps=12)
    assert pulldown['phase_frames'] == {'start': 1, 'middle': 7, 'end': 13}
    assert pulldown['timing']['forward_s'] == 1.
    assert pulldown['timing']['return_s'] == 1.5
    assert len(kc.timeline(12, 'Lat Pulldown (seated, to the front)')) == 32
    for exercise in ('Standing Barbell Overhead Press (Shoulder Press)',
                     'Dumbbell Shoulder Press (seated overhead press)'):
        press = kc.reference_timing({'exercise': exercise}, ['start', 'end'], keys, fps=12)
        assert press['phase_frames']['end'] == 13
        assert press['timing']['forward_action'] == 'press'
        assert press['timing']['forward_s'] < press['timing']['return_s']


def test_the_motion_passes_through_the_middle_position():
    keys = [kc.key_pose(mp_pose(*a, phase=p), REST) for a, p in
            (((0, 180, 180), "start"), ((15, 120, 120), "middle"), ((20, 90, 90), "end"))]
    marks = kc.path_marks(keys)
    assert marks[0] == 0.0 and marks[-1] == 1.0 and 0.0 < marks[1] < 1.0
    at_middle = kc.pose_at(keys, marks, marks[1])
    for bone, r in keys[1].items():
        assert np.allclose(at_middle[bone], r, atol=1e-9), bone
    halfway = kc.pose_at(keys, marks, marks[1] / 2)
    knee = angle(halfway["LeftUpLeg"] @ (REST["LeftLeg"] - REST["LeftUpLeg"]),
                 halfway["LeftLeg"] @ (REST["LeftFoot"] - REST["LeftLeg"]))
    assert 0 < knee < 60                       # between standing (0 bend) and the middle's (60 bend)


def test_two_positions_without_a_middle_still_move():
    keys = [kc.key_pose(mp_pose(0, 180, 180), REST), kc.key_pose(mp_pose(20, 90, 90, phase="end"), REST)]
    assert kc.path_marks(keys) == [0.0, 1.0]
    poses = kc.motion(keys, fps=30)
    assert len(poses) == len(kc.timeline(30))


def test_planted_feet_keep_the_first_position_s_foot():
    keys = [kc.key_pose(mp_pose(0, 180, 180), REST),
            kc.key_pose(mp_pose(20, 90, 90, foot_pitch=-30.0, phase="end"), REST)]
    assert not np.allclose(keys[0]["LeftFoot"], keys[1]["LeftFoot"])
    for pose in kc.motion(keys, fps=30, feet="planted"):
        for bone in ("LeftFoot", "RightFoot", "LeftToeBase", "RightToeBase"):
            assert np.allclose(pose[bone], keys[0][bone]), bone
    free = kc.motion(keys, fps=30, feet="free")
    assert np.allclose(free[round(1.5 * 30)]["LeftFoot"], keys[1]["LeftFoot"])


def test_planted_feet_keep_the_first_position_s_stance():
    # the coach's side pictures read his ankles 17 cm out standing and 9 cm out at the bottom (depth noise: the
    # feet do not move); with the ankles locked where they stood, a narrower stance in a later position made the
    # lock pull each leg outward and straightened the bottom's knee from 47 to 55 degrees
    start, end = mp_pose(0, 180, 178), mp_pose(20, 70, 50, phase="end")
    for side, sign in (("left", 1.0), ("right", -1.0)):
        for j in ("ankle", "heel", "foot_index"):
            start["world"][f"{side}_{j}"][2] += sign * 0.07          # wide standing stance
    body0, _, _ = kc.key_points(start)
    stance = kc.stance_of(body0)
    assert stance["left_ankle"] == pytest.approx(0.17) and stance["right_ankle"] == pytest.approx(-0.17)
    body, arms, found = kc.key_points(end, stance=stance)
    for n, x in stance.items():
        assert body[n][0] == pytest.approx(x)
    free, _, _ = kc.key_points(end)
    assert free["left_ankle"][0] == pytest.approx(0.1)
    assert body["left_ankle"][1:] == pytest.approx(free["left_ankle"][1:])           # the side plane stays
    planted = dict(kc.key_poses({"phases": [start, end]}, REST, feet="planted"))
    assert all(np.allclose(planted["end"][b], r) for b, r in kc.bone_turns(body, arms, REST, square=True).items())
    loose = dict(kc.key_poses({"phases": [start, end]}, REST, feet="free"))
    assert not np.allclose(loose["end"]["LeftLeg"], planted["end"]["LeftLeg"])


def test_the_legs_points_keep_the_ankles_where_they_stand():
    keys = [kc.key_pose(mp_pose(0, 180, 180), REST), kc.key_pose(mp_pose(20, 90, 90, phase="end"), REST)]
    frames = [kc.leg_points(pose, REST) for pose in kc.motion(keys, fps=30)]
    for f in frames:
        assert np.allclose((f["LeftFoot"] + f["RightFoot"]) / 2, (REST["LeftFoot"] + REST["RightFoot"]) / 2)
    bottom = frames[round(1.5 * 30)]
    hips = (bottom["LeftUpLeg"] + bottom["RightUpLeg"]) / 2
    assert hips[2] < (REST["LeftUpLeg"][2] - 0.25)                    # the hips went down
    knee = angle(bottom["LeftUpLeg"] - bottom["LeftLeg"], bottom["LeftFoot"] - bottom["LeftLeg"])
    assert knee == pytest.approx(90.0, abs=0.01)


def test_too_few_positions_are_refused_in_plain_words():
    with pytest.raises(ValueError, match="start and an end"):
        kc.key_poses({"phases": [mp_pose(0, 180, 180)]}, REST)
    poses = kc.key_poses({"phases": [mp_pose(20, 90, 90, phase="end"), mp_pose(0, 180, 180, phase="start")]}, REST)
    assert [p for p, _ in poses] == ["start", "end"]                  # put in start, middle, end order


def test_the_picture_s_scale_is_fitted_not_assumed():
    entry = mp_pose(20, 90, 90, px_per_m=300.0)
    entry["world"]["left_knee"][2] += 0.12           # knees out 12 cm (left is +z facing right)
    entry["world"]["right_knee"][2] -= 0.12
    body, _, found = kc.key_points(entry)
    assert found["metres_per_pixel"] == pytest.approx(1 / 300.0)
    assert side_angles(body) == pytest.approx((20, 90, 90), abs=0.01)
    # the side plane in metres, so the knees' left-right offset (metres, from the depth) sits on the right leg
    assert np.linalg.norm(body["left_knee"][1:] - body["left_hip"][1:]) == pytest.approx(0.42, abs=1e-6)


def test_the_middle_sits_where_its_share_of_the_turning_puts_it():
    keys = [kc.key_pose(mp_pose(*a, phase=p), REST) for a, p in
            (((0, 180, 180), "start"), ((2, 172, 172), "middle"), ((20, 90, 90), "end"))]
    marks = kc.path_marks(keys)
    gap = [sum(kc._turned_by(a[n].T @ b[n]) for n in a) for a, b in zip(keys, keys[1:])]
    assert marks[1] == pytest.approx(gap[0] / sum(gap))
    assert marks[1] < 0.15                          # nearly standing: early in the stroke, not half way


def test_slerp_takes_the_shorter_arc():
    # half way along is half the turn between them, which is never more than 180: q and -q are the same turn, and
    # without picking the nearer one half the pairs would swing the long way round
    rng = np.random.default_rng(9)
    flipped = 0
    for _ in range(200):
        r0, r1 = (kc._rotation(kc._unit(rng.normal(size=3)), rng.uniform(0, math.pi)) for _ in range(2))
        flipped += kc._quat(r0) @ kc._quat(r1) < 0
        half = kc.slerp(r0, r1, 0.5)
        assert kc._turned_by(r0.T @ half) == pytest.approx(kc._turned_by(r0.T @ r1) / 2, abs=1e-6)
    assert flipped > 20                                  # the case the flip is for did come up


def test_a_three_quarter_picture_set_to_side_gets_its_facing_from_the_picture():
    for facing in ("right", "left"):
        entry = {**mp_pose(20, 90, 90, facing=facing), "facing": None}     # the detector gave none
        body, _, found = kc.key_points(entry)
        assert found["square"] and found["facing"] == facing
        assert side_angles(body) == pytest.approx((20, 90, 90), abs=0.01)
    off = {**mp_pose(20, 90, 90), "facing": None, "view": "three-quarter"}
    assert not kc.key_points(off)[2]["square"]           # not set to Side: the 3D path
