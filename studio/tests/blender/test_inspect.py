"""inspect_run on a tiny scene: an MPFB human, knees bent to a known angle (the skin behind the knees
past 60 degrees is a crease: the folded-skin row passes); and its fold finder on one knee folded to 30
degrees (a crease behind that knee, none at rest) and on an arm raised overhead (a fold on the
shoulder the GIF's cameras see: the row fails; behind a wall: hidden, info). Runs inside Blender (no
other Blender running):

    /Applications/Blender.app/Contents/MacOS/Blender --background --python-exit-code 1 \
        --python studio/tests/blender/test_inspect.py

Any failed assert exits 1. The run folder is left in the system temp dir (printed) so its stills
can be looked at.
"""
import json
import math
import sys
import tempfile
from pathlib import Path

try:
    import bpy
except ImportError:  # collected by pytest outside Blender: this one runs inside Blender only
    import pytest
    pytest.skip("runs inside Blender (see the module docstring)", allow_module_level=True)
import numpy as np
from mathutils import Matrix, Vector

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "studio" / "blender"))
import inspect_run  # noqa: E402
import checks  # noqa: E402  (studio/measure, on the path inspect_run set)
import creases  # noqa: E402

P = "mixamorig:"
BENT = 20


def _yz(v):
    return np.array([v.y, v.z])


def _signed(a, b):
    """Signed angle (radians) turning 2-D vector a onto b."""
    return math.atan2(a[0] * b[1] - a[1] * b[0], a @ b)


def side_knee(rig, side):
    """The knee's included angle seen straight from the side: the y-z projection (the test's own
    arithmetic, independent of inspect_run's cameras)."""
    pb = rig.pose.bones
    hip, knee, ankle = (_yz(rig.matrix_world @ pb[P + side + b].head) for b in ("UpLeg", "Leg", "Foot"))
    u, v = hip - knee, ankle - knee
    return math.degrees(math.acos(u @ v / np.linalg.norm(u) / np.linalg.norm(v)))


def key_rotation(pb, frame):
    pb.keyframe_insert("rotation_quaternion" if pb.rotation_mode == "QUATERNION" else "rotation_euler", frame=frame)


def bend_knee(rig, side, forward):
    """Turn the shin about the world X axis through the knee so the side-seen knee angle is 90 with
    the foot behind the knee. A turn about X turns the y-z projection by exactly that angle."""
    pb = rig.pose.bones[P + side + "Leg"]
    w = rig.matrix_world
    hip, knee, ankle = (w @ rig.pose.bones[P + side + b].head for b in ("UpLeg", "Leg", "Foot"))
    thigh, shin = _yz(hip - knee), _yz(ankle - knee)
    fwd = np.array([forward[1], forward[2]])
    # the shin direction 90 degrees from the thigh, on the behind side
    options = [np.array([-thigh[1], thigh[0]]), np.array([thigh[1], -thigh[0]])]
    want = min(options, key=lambda o: o @ fwd)
    turn = Matrix.Translation(knee) @ Matrix.Rotation(_signed(shin, want), 4, "X") @ Matrix.Translation(-knee)
    pb.matrix = w.inverted() @ turn @ w @ pb.matrix
    bpy.context.view_layer.update()


def new_human():
    from bl_ext.user_default.mpfb.services.humanservice import HumanService
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    body = HumanService.create_human()
    return body, HumanService.add_builtin_rig(body, "mixamo", import_weights=True)


def fold_knee(rig, side, forward, included):
    """Like bend_knee, to any side-seen knee angle: `included` 30 folds the heel up to the buttock,
    far past where the skin at the back of the knee can follow its bones."""
    pb = rig.pose.bones[P + side + "Leg"]
    w = rig.matrix_world
    hip, knee, ankle = (w @ rig.pose.bones[P + side + b].head for b in ("UpLeg", "Leg", "Foot"))
    thigh, shin = _yz(hip - knee), _yz(ankle - knee)
    fwd = np.array([forward[1], forward[2]])
    c, s = math.cos(math.radians(included)), math.sin(math.radians(included))
    options = [np.array([c * thigh[0] - s * thigh[1], s * thigh[0] + c * thigh[1]]),
               np.array([c * thigh[0] + s * thigh[1], -s * thigh[0] + c * thigh[1]])]
    want = min(options, key=lambda o: o @ fwd)
    turn = Matrix.Translation(knee) @ Matrix.Rotation(_signed(shin, want), 4, "X") @ Matrix.Translation(-knee)
    pb.matrix = w.inverted() @ turn @ w @ pb.matrix
    bpy.context.view_layer.update()


def build_scene(run):
    body, rig = new_human()
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, BENT
    scene.frame_set(1)
    legs = [rig.pose.bones[P + s + "Leg"] for s in ("Left", "Right")]
    for pb in legs:
        key_rotation(pb, 1)
    scene.frame_set(BENT)
    left = rig.matrix_world @ rig.pose.bones[P + "LeftUpLeg"].head - rig.matrix_world @ rig.pose.bones[P + "RightUpLeg"].head
    left.z = 0
    forward = left.normalized().cross(Vector((0, 0, 1)))
    for side, pb in zip(("Left", "Right"), legs):
        bend_knee(rig, side, forward)
        key_rotation(pb, BENT)
    scene.frame_set(BENT)
    for side in ("Left", "Right"):
        got = side_knee(rig, side)
        assert abs(got - 90) < 0.5, f"test scene: {side} knee seen from the side is {got:.2f}, not 90"
    scene.frame_set(1)
    print(f"test: rest knee seen from the side {side_knee(rig, 'Left'):.1f}; figure faces {tuple(round(c, 2) for c in forward)}")
    bpy.ops.wm.save_as_mainfile(filepath=str(run / "scene.blend"), copy=True)
    return rig


def save_turned(rig, run):
    """The same scene with the armature (and the body it carries) turned 180 degrees about Z: it faces +Y,
    as the run scenes do."""
    rig.matrix_world = Matrix.Rotation(math.pi, 4, "Z") @ rig.matrix_world
    bpy.context.view_layer.update()
    bpy.ops.wm.save_as_mainfile(filepath=str(run / "scene.blend"), copy=True)


def write_run_files(run, exercise="Test Squat"):
    def t(v):
        return {"value": v, "method": "flat", "cross": None, "uncertain": False, "image": None}

    back = {**t(0.0), "seen": 0.25, "span": [0.1, 0.3]}     # the stretch of back a squat photo shows
    targets = {"exercise": exercise,
               "phases": {"start": {"knee_included": t(175), "back_rounding": back},
                          "middle": {"knee_included": t(132.5), "estimated": True},
                          "end": {"knee_included": t(90), "back_rounding": back}},
               "widths": {}, "tolerance": {"angles_deg": 5, "ratios": 0.12, "hip_below_knee": 0.10},
               "images": {}, "camera": {"side": "right", "facing": "left"}}
    base = {"phaseSetBy": "user", "contextOnly": False, "deleted": False}
    images = [
        # view set by the user
        {**base, "id": "img-start", "file": "img-start.png", "original": "a.png", "order": 1, "phase": "start",
         "view": "side", "viewSetBy": "user", "path": str(run / "img-start.png")},
        # view left to the detector: read from measurements.json
        {**base, "id": "img-end", "file": "img-end.png", "original": "b.png", "order": 2, "phase": "end",
         "view": None, "viewSetBy": None, "path": str(run / "img-end.png")},
        # context only: no still
        {**base, "id": "img-ctx", "file": "img-ctx.png", "original": "c.png", "order": 3, "phase": "middle",
         "view": None, "viewSetBy": None, "contextOnly": True, "path": str(run / "img-ctx.png")},
    ]
    seen = {"view": "side", "camera_side": "right", "facing": "left"}
    (run / "targets.json").write_text(json.dumps(targets))
    (run / "references.json").write_text(json.dumps({"exercise": exercise, "slug": "test-squat", "images": images}))
    (run / "measurements.json").write_text(json.dumps({"img-start": seen, "img-end": seen,
                                                       "img-ctx": {"error": "No body found in this image"}}))


def check(run):
    text = (run / "ours.json").read_text()
    ours = json.loads(text, parse_constant=lambda c: (_ for _ in ()).throw(AssertionError(f"ours.json has {c}")))
    print("test: phase_frames", ours["phase_frames"], "stills", ours["stills"])
    assert ours["first"] == 1 and ours["last"] == BENT and ours["camera_side"] == "right"
    assert ours["phase_frames"]["end"] == BENT and ours["phase_driver"] == "knee_included", ours["phase_frames"]
    assert ours["per_frame"][BENT - 1]["frame"] == BENT
    knee = ours["per_frame"][BENT - 1]["flat"]["knee_included"]
    assert abs(knee - 90) <= 2, f"flat knee at frame {BENT} is {knee}"     # the knee is read on its joints
    assert all(v > 0 for v in ours["arm"].values()), ours["arm"]
    assert ours["jumps"] == [] and ours["steps"] == [], (ours["jumps"], ours["steps"])   # the test motion is smooth
    # rest pose at frame 1, knees bending to 90 by frame 20. Up to 45 degrees (knee >= 135, frames
    # 1-10) no skin turns over. Past about 60 MPFB's own weights pinch the skin behind the knee through
    # itself (about 1,000 mm² on the right knee at 90): a crease of a deep bend, so the row passes.
    knee = {f["frame"]: f["3d"]["knee_included"] for f in ours["per_frame"]}
    assert [knee[f] for f in range(1, 11)] == sorted((knee[f] for f in range(1, 11)), reverse=True) and knee[10] > 134
    assert 1.5 < ours["height_m"] < 2.0, ours["height_m"]
    limit = checks.fold_limit(ours["height_m"])
    assert ours["folds"], "no crease behind the knees bent to 90"
    assert not [f for f in ours["folds"] if knee[f["frame"]] >= 134], ours["folds"]
    assert all(f["kind"] == "crease" and f["bone"] in (P + "LeftLeg", P + "RightLeg") for f in ours["folds"]), ours["folds"]
    assert all(f["area_mm2"] > limit and 0 <= f["visible_mm2"] <= f["area_mm2"] for f in ours["folds"]), ours["folds"]
    assert ours["folds_worst"] == inspect_run.worst_folds(ours["folds"]), ours["folds_worst"]
    row = checks._fold_rows(ours)[0]
    assert row["status"] == "pass" and row["target"] == limit and "knee" in row["note"], row
    assert row["ours"] == max(f["visible_mm2"] for f in ours["folds"]), row     # the most seen crease

    stills = {s["image"]: s for s in ours["stills"]}
    assert set(stills) == {"img-start", "img-end"}, stills
    assert stills["img-end"]["frame"] == BENT and stills["img-start"]["frame"] == ours["phase_frames"]["start"]
    # unturned, MPFB faces -Y; the right side's camera sits at -X looking +X, so image right is -Y: the nose
    # points right. Turned, everything is mirrored through the Z axis and the answer is the same.
    assert all(s["view"] == "side" and s["facing"] == "right" for s in stills.values()), stills
    files = sorted(p.name for p in (run / "stills").iterdir())
    assert files == ["img-end.png", "img-start.png"], files     # the posture masks are not stills
    # posture (Task 6c): the neck and the face's direction on every frame; the back at the positions
    for f in ours["per_frame"]:
        assert isinstance(f["flat"]["neck_vs_torso"], float) and isinstance(f["flat"]["head_pitch"], float), f
        assert isinstance(f["3d"]["neck_vs_torso"], float), f
    at = {f["frame"]: f for f in ours["per_frame"]}
    # our back is judged from the rig (its spine's curl from rest, every frame), not from its silhouette;
    # this motion bends only the knees, so the spine stays as it stands
    for f in ours["per_frame"]:
        assert isinstance(f["flat"]["spine_curl"], float) and abs(f["flat"]["spine_curl"]) < 1.0, f
        assert "back_rounding" not in f["flat"] and "back_seen" not in f["flat"], f
    assert isinstance(ours["spine_rest"], float), ours.get("spine_rest")
    rest = at[1]["flat"]                        # frame 1 is MPFB's standing pose
    assert abs(rest["spine_curl"]) < 0.5 and abs(rest["head_pitch"]) < 1.0, rest
    # the torso line is counted from the rig's own upright bind pose (task 18): standing, it reads upright
    assert isinstance(ours["torso_rest_lean"], float), ours.get("torso_rest_lean")
    assert abs(rest["torso_lean"]) < 0.01 and abs(at[1]["3d"]["torso_lean"]) < 0.01, (rest, at[1]["3d"])
    # ... and its thighs from their skin outline in the bind pose (task 20): standing, the hip reads 180 less the
    # outline's own lean; the bind pose is also read by MediaPipe on a render of it, as a person's would be
    legs = ours["legs_at_rest"]
    assert set(legs["joints"]) == {"thigh.left", "thigh.right", "shin.left", "shin.right"}, legs
    out = legs["outline"]["thigh.right"]
    assert abs(out) < 3 and abs(rest["hip_included"] - (180 - abs(out))) < 0.5, (out, rest)
    bind_mp = legs["read"]["mediapipe"]
    print(f"test: bind pose read as ours {legs['read']['ours']}, by MediaPipe {bind_mp}")
    assert "why" not in bind_mp and abs(bind_mp["hip_included"] - legs["read"]["ours"]["hip_included"]) < 6, legs["read"]
    # MediaPipe's thigh: its camera-side hip -> knee landmarks' slope, signed (task 22), beside the outline's
    assert isinstance(bind_mp["thigh_slope"], float) and abs(bind_mp["thigh_slope"] - out) <= 5, (bind_mp, out)
    assert legs["warning"] is None, legs["warning"]          # a person-shaped human: the two agree (task 21)
    # the depth as a judge sees it (task 22): for an exercise whose kit row says so, at the end position, off our
    # silhouette and off the 3D skin, each with a value or its reason (this test's bottom is a standing bent knee)
    exercise = json.loads((run / "targets.json").read_text())["exercise"]
    if checks.has_depth(exercise):
        end = ours["seat_below_knee"]["end"]
        assert set(end) == {"frame", "picture", "seat", "knee_top", "hip", "knee", "why", "skin", "skin_why"}, end
        assert end["frame"] == ours["phase_frames"]["end"], end
        assert (end["picture"] is None) != (end["why"] is None), end
        assert (end["skin"] is None) != (end["skin_why"] is None), end
        print(f"test: seat at the end: {end}")
        # this test's bottom is a standing bent knee: the thigh stands up over the knee, so neither finds a knee's top
        assert end["picture"] is None and end["skin"] is None and "knee top" in end["skin_why"], end
    else:
        assert "seat_below_knee" not in ours, ours["seat_below_knee"]
    assert "thigh_outline" not in ours and all("thigh_outline" not in f["flat"] for f in ours["per_frame"])
    # the neck runs to the ears found on the skin (no baseline); ours.json says where they were found
    assert "neck_rest" not in ours and set(ours["ears"]) == {"left", "right", "count"}, ours.get("ears")
    assert all(n >= 10 for n in ours["ears"]["count"]), ours["ears"]
    # ours read from the picture as the coach is (task 19): every position, five framings each, the coach's
    # reader in the Studio's Python; a person-proportioned human standing is read clearly
    pic = ours["picture"]
    assert pic["framings"] == [list(f) for f in inspect_run.FRAMINGS] and set(pic["camera"]) == {"azimuth", "elevation"}
    for phase, frame in ours["phase_frames"].items():
        assert pic[phase]["frame"] == frame and pic[phase]["reads"] == 5, (phase, pic[phase])
    start = pic["start"]
    rig_lean = at[ours["phase_frames"]["start"]]["flat"]["torso_lean"]
    print(f"test: picture at the start: {start['flat'] and start['flat']['torso_lean']} (rig {rig_lean}), "
          f"{start['seen']} of 5 framings seen, torso visibility {start['visibility']['torso']}")
    assert start["ok"] and start["seen"] >= 4, start
    assert abs(start["flat"]["torso_lean"] - rig_lean) < 6, (start["flat"], rig_lean)
    pixels = {}
    for s in stills.values():
        path = run / s["file"]
        img = bpy.data.images.load(str(path))
        assert tuple(img.size) == (560, 560), (path, tuple(img.size))
        px = np.empty(560 * 560 * 4, dtype=np.float32)
        img.pixels.foreach_get(px)
        pixels[s["image"]] = px.reshape(-1, 4)[:, :3]
        spread = np.ptp(pixels[s["image"]], axis=0).max()
        assert spread > 0.05, f"{path} is one flat colour"
    print(f"test: checked {run}")
    return ours, pixels


FOLDED = 6


def run_fold_test():
    """An MPFB human, rest pose at frame 1, its left knee folded to 30 degrees at frame FOLDED: the
    skin at the back of that knee turns over there, nowhere else, and not at rest."""
    body, rig = new_human()
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, FOLDED
    scene.frame_set(1)
    pb = rig.pose.bones[P + "LeftLeg"]
    key_rotation(pb, 1)
    left = rig.matrix_world @ rig.pose.bones[P + "LeftUpLeg"].head - rig.matrix_world @ rig.pose.bones[P + "RightUpLeg"].head
    left.z = 0
    forward = left.normalized().cross(Vector((0, 0, 1)))
    scene.frame_set(FOLDED)
    fold_knee(rig, "Left", forward, 30)
    key_rotation(pb, FOLDED)
    scene.frame_set(FOLDED)
    assert abs(side_knee(rig, "Left") - 30) < 0.5, side_knee(rig, "Left")
    cameras = gif_of(rig, body, FOLDED)
    measured = inspect_run.skin_folds(rig, body, 1, FOLDED, lambda frame: cameras)
    folds = measured["folds"]
    for f in folds:
        print("test: fold", f)
    assert not [f for f in folds if f["frame"] == 1], "the rest pose has turned-over skin"
    at = [f for f in folds if f["frame"] == FOLDED]
    assert at, "no fold behind the knee folded to 30 degrees"
    worst = max(at, key=lambda f: f["area_mm2"])
    assert worst["bone"] == P + "LeftLeg" and worst["kind"] == "crease", worst
    knee = np.array(rig.matrix_world @ rig.pose.bones[P + "LeftLeg"].head)
    assert np.linalg.norm(np.array(worst["centre"]) - knee) < 0.15, (worst, knee)
    assert all(f["area_mm2"] > checks.fold_limit(measured["height_m"]) for f in folds)
    assert not [f for f in folds if "Right" in f["bone"]], "the straight right leg folded"
    assert inspect_run.worst_folds(folds)[P + "LeftLeg"] == worst["area_mm2"]
    print(f"test: fold PASS (worst {worst})")


def gif_of(rig, body, last):
    box = inspect_run.gif_box(tempfile.gettempdir() + "/no-such-run", body, 1, last)
    return inspect_run.gif_cameras(box, inspect_run.figure_yaw(inspect_run.joints_at(rig, 1)))


def wall_between(camera, target, size=3.0, collection=None):
    """A square plane halfway between a camera and a point, facing the camera: nothing behind it is
    seen, if it is rendered (in `collection`, the scene's own by default)."""
    camera, target = Vector(camera), Vector(target)
    mid, n = (camera + target) / 2, (camera - target).normalized()
    a = n.cross(Vector((0, 0, 1))).normalized() * size / 2
    b = n.cross(a).normalized() * size / 2
    mesh = bpy.data.meshes.new("wall")
    mesh.from_pydata([mid - a - b, mid + a - b, mid + a + b, mid - a + b], [], [(0, 1, 2, 3)])
    wall = bpy.data.objects.new("wall", mesh)
    (collection or bpy.context.scene.collection).objects.link(wall)
    bpy.context.view_layer.update()
    return wall


def figure_forward(rig):
    w = rig.matrix_world
    left = w @ rig.pose.bones[P + "LeftUpLeg"].head - w @ rig.pose.bones[P + "RightUpLeg"].head
    left.z = 0
    return left.normalized().cross(Vector((0, 0, 1)))


SPINE = ("Spine", "Spine1", "Spine2")


def bend_spine(rig, forward, degrees):
    """From the rest pose, bend each spine bone `degrees` forward (negative: back) about the figure's
    left-right axis through its own joint: the torso curls (rounds) or arches. The head follows."""
    w = rig.matrix_world
    axis = Vector((0, 0, 1)).cross(forward)         # turning up toward forward
    for name in SPINE:
        pb = rig.pose.bones[P + name]
        pb.matrix_basis = Matrix()
    bpy.context.view_layer.update()
    for name in SPINE:
        pb = rig.pose.bones[P + name]
        joint = w @ pb.head
        turn = Matrix.Translation(joint) @ Matrix.Rotation(math.radians(degrees), 4, axis) @ Matrix.Translation(-joint)
        pb.matrix = w.inverted() @ turn @ w @ pb.matrix
        bpy.context.view_layer.update()


def side_neck(rig, forward, ear):
    """The neck against the torso seen straight from the side (the y-z plane; the figure faces -Y): hip
    joints' midpoint -> shoulder joints' midpoint, turned by that same line's angle in the rest pose (task
    18: the torso line is counted from the rig's own upright bind pose), against shoulder joints'
    midpoint -> the ears' midpoint `ear` (the test's own arithmetic, independent of posture.py and the
    camera)."""
    pb, bones, w = rig.pose.bones, rig.data.bones, rig.matrix_world

    def yz(name, rest=False):
        return _yz(w @ (bones[P + name].head_local if rest else pb[P + name].head))
    hip = (yz("LeftUpLeg") + yz("RightUpLeg")) / 2
    shoulder = (yz("LeftArm") + yz("RightArm")) / 2
    rest_t = (yz("LeftArm", True) + yz("RightArm", True)) / 2 - (yz("LeftUpLeg", True) + yz("RightUpLeg", True)) / 2
    t, k = shoulder - hip, _yz(Vector(ear)) - shoulder
    a = _signed(rest_t, np.array([0.0, 1.0]))           # turns the rest torso line upright
    t = np.array([math.cos(a) * t[0] - math.sin(a) * t[1], math.sin(a) * t[0] + math.cos(a) * t[1]])
    angle = math.degrees(math.atan2(t[0] * k[1] - t[1] * k[0], t @ k))
    return angle if forward.y < 0 else -angle


def arms_forward(rig, forward):
    """Raise both upper arms to point forward (15 degrees below level): clear of the back."""
    w = rig.matrix_world
    for side in ("Left", "Right"):
        pb = rig.pose.bones[P + side + "Arm"]
        head, tail = w @ pb.head, w @ pb.tail
        want = (forward * math.cos(math.radians(15)) - Vector((0, 0, 1)) * math.sin(math.radians(15))).normalized()
        turn = (Matrix.Translation(head) @ (tail - head).normalized().rotation_difference(want).to_matrix().to_4x4()
                @ Matrix.Translation(-head))
        pb.matrix = w.inverted() @ turn @ w @ pb.matrix
        bpy.context.view_layer.update()


def run_posture_test():
    """Posture on an MPFB human (Task 6c). Our back is judged from the rig: the spine's side-seen curl
    from rest reads 0 standing, +45 with each spine bone bent 15 degrees forward, -30 with each bent 10
    back. The ears are found on the skin where MPFB's own "ears" group is; the neck (shoulder joints to
    the ears' midpoint) reads what that arithmetic says seen from the side; standing, the face reads level.
    The silhouette measure (used on the coach's photos): standing with the arms raised clear, it
    reproduces back-calibration.json's neutral; with the arms clear the whole back reads the waist's
    hollow standing, rounds with the curl and arches further back. The whole back reads the waist's hollow (a neutral back is an S, not a
    line); each spine bone bent 15 degrees forward rounds it past 0 (and tips the face 45 degrees down),
    bent 10 degrees back arches it further. On the lowest stretch a squat photo shows (0.1-0.3 of the
    torso) the standing back reads about straight, and even the 45-degree curl barely moves it: why the
    check needs most of the back in view. The mask is the body alone: no floor, no backdrop; the arms'
    own mask is a part of it, and the body is left as it was."""
    body, rig = new_human()
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, 4
    spine = [rig.pose.bones[P + n] for n in SPINE]
    arms = [rig.pose.bones[P + s + "Arm"] for s in ("Left", "Right")]
    scene.frame_set(1)
    for pb in spine + arms:
        key_rotation(pb, 1)
    forward = figure_forward(rig)
    for frame, deg in ((2, 15.0), (3, -10.0), (4, 0.0)):    # arms hanging at 1, raised clear of the back after
        scene.frame_set(frame)
        for pb in arms:
            pb.matrix_basis = Matrix()
        bend_spine(rig, forward, deg)
        arms_forward(rig, forward)
        for pb in spine + arms:
            key_rotation(pb, frame)
    import mpfb_cage
    mpfb_cage.preview_set(560, 560)
    yaw = inspect_run.figure_yaw(inspect_run.joints_at(rig, 1))
    azimuth, elevation, _ = inspect_run.camera_for("side", "right", yaw)
    side_normal = inspect_run.view_direction(azimuth, elevation)
    local_forward = inspect_run.head_forward(rig)
    # the ears, found on the skin, are where MPFB's own "ears" vertex group is
    found = inspect_run.ear_vertices(rig, body)
    group = body.vertex_groups["ears"].index
    in_group = {v.index for v in body.data.vertices if any(g.group == group and g.weight > 0 for g in v.groups)}
    co = np.array([body.matrix_world @ v.co for v in body.data.vertices])
    rest_ears = inspect_run.ear_points(rig, body)
    for side in ("left", "right"):
        verts = found[side]
        share = len(in_group & set(verts)) / len(verts)
        lateral = np.sign(co[verts, 0].mean())
        truth = co[[i for i in in_group if np.sign(co[i, 0]) == lateral]]
        miss = float(np.linalg.norm(rest_ears[side] - truth.mean(axis=0)))
        print(f"test: {side} ear: {len(verts)} vertices, {share:.0%} of them MPFB's ears, {miss * 1000:.1f} mm "
              f"from the ears group's middle")
        assert len(verts) >= 10 and share > 0.8 and miss < 0.02, (side, len(verts), share, miss)
        assert truth[:, 2].min() - 0.005 <= rest_ears[side][2] <= truth[:, 2].max() + 0.005
    assert np.sign(rest_ears["left"][0]) != np.sign(rest_ears["right"][0])
    got = {}
    for frame in (1, 2, 3):
        joints = inspect_run.joints_at(rig, frame)
        head, face = inspect_run.head_at(rig, local_forward)
        ear = inspect_run.ears_at(rig, rest_ears)
        posture = inspect_run.posture_frame(joints, ear, face, side_normal, inspect_run.torso_turn(rig, side_normal))
        back = inspect_run.back_at(rig, body, frame, azimuth, elevation, local_forward)
        lower = inspect_run.back_at(rig, body, frame, azimuth, elevation, local_forward, span=(0.1, 0.3))["profile"]
        got[frame] = (posture, back, lower)
        print(f"test: lower back (0.1-0.3) frame {frame}: {lower['rounding']} seen {lower['seen']:.2f}")
        print(f"test: posture frame {frame}: {posture}; back {back['profile']['rounding']} "
              f"seen {back['profile']['seen']:.2f} facing {back['facing']}")
        mask = back["mask"]
        assert mask.shape == (1120, 1120), mask.shape                  # twice the stills' size
        assert max(mask[0, 0], mask[0, -1], mask[-1, 0], mask[-1, -1]) == 0, "the mask has a backdrop or floor"
        assert 0.02 < float((mask >= 0.5).mean()) < 0.5
        arms = back["arm_mask"] >= 0.5          # the arms alone: a part of the body's mask
        assert arms.any() and float((mask[arms] >= 0.5).mean()) > 0.99, float((mask[arms] >= 0.5).mean())
        assert arms.sum() < 0.3 * (mask >= 0.5).sum()
        # MPFB's own Mask modifier (it hides the helper geometry) stays; ours is gone
        assert inspect_run.ARM_GROUP not in body.modifiers and inspect_run.ARM_GROUP not in body.vertex_groups
        assert back["facing"] == "right", back["facing"]
        scene.frame_set(frame)
        neck = side_neck(rig, forward, ear)
        assert abs(posture["flat"]["neck_vs_torso"] - neck) < 1.0, (frame, posture, neck)
    spine_rest = inspect_run.rest_spine(rig, side_normal)
    for frame, want in ((1, 0.0), (2, 45.0), (3, -30.0), (4, 0.0)):
        scene.frame_set(frame)
        curl = inspect_run.spine_curl(rig, side_normal, spine_rest)
        print(f"test: spine curl frame {frame}: {curl:.2f} (want {want})")
        assert abs(curl - want) < 1.0, (frame, curl, want)
    # the silhouette measure, arms clear: the stored human neutral is what it reads now
    cal = json.loads((REPO / "studio" / "measure" / "back-calibration.json").read_text())
    clear = inspect_run.back_at(rig, body, 4, azimuth, elevation, local_forward)["profile"]
    assert clear["seen"] == 1.0 and clear["run"] == 1.0, (clear["seen"], clear["run"])
    s_now = np.array([p["s"] for p in clear["samples"]])
    d_now = np.interp(cal["s"], s_now, [p["d"] for p in clear["samples"]])
    drift = float(np.abs(d_now - np.array(cal["profiles"]["0"])).max())
    print(f"test: calibration neutral reproduced within {drift:.4f} torso lengths")
    assert drift < 0.003, drift
    rest, bent, arched = (got[f][1]["profile"]["rounding"] for f in (1, 2, 3))
    assert all(got[f][1]["profile"]["run"] > 0.99 for f in (1, 2, 3)), "the arms are clear of the back"
    assert -0.15 < rest < -0.03, rest                   # the waist's hollow
    assert bent > 0 and bent > rest + 0.08, (rest, bent)
    assert arched < rest - 0.05, (rest, arched)
    low = [got[f][2]["rounding"] for f in (1, 2, 3)]
    assert abs(low[0]) < 0.02 and all(abs(v - low[0]) < 0.01 for v in low), low
    assert abs(got[1][0]["flat"]["head_pitch"]) < 1.0, got[1][0]
    assert abs(got[2][0]["flat"]["head_pitch"] + 45) < 3.0, got[2][0]
    assert abs(got[3][0]["flat"]["head_pitch"] - 30) < 3.0, got[3][0]
    print(f"test: posture PASS (whole back: rest {rest:.3f}, bent forward {bent:.3f}, arched {arched:.3f}; "
          f"lower back: {', '.join(f'{v:.3f}' for v in low)})")


def turn_about(rig, pb, pivot, axis, degrees):
    """Turn a pose bone (and all it carries) by `degrees` about a world axis through a world point."""
    w = rig.matrix_world
    turn = Matrix.Translation(pivot) @ Matrix.Rotation(math.radians(degrees), 4, axis) @ Matrix.Translation(-pivot)
    pb.matrix = w.inverted() @ turn @ w @ pb.matrix
    bpy.context.view_layer.update()


def run_lean_test():
    """Task 18: our torso line (hip joints -> shoulder joints, as the coach's MediaPipe landmarks) is counted
    from the rig's own upright bind pose, so a rig whose shoulder joints sit behind its hips does not read
    a backward lean standing tall. An MPFB human, rest pose at frame 1; at frame 2 everything above the hips
    tipped 20 degrees forward about the hip joints' line (the legs turned back, so they stay put): the lean
    reads 0, then 20, on the side camera's plane and in 3D; the hip angle closes by the same 20; the
    upper arm, carried with the torso, keeps its angle to it. The rig itself is then given a rest lean (its
    shoulders moved back as on the cartoon) and still reads 0 standing."""
    body, rig = new_human()
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, 2
    scene.frame_set(1)
    hips = rig.pose.bones[P + "Hips"]
    legs = [rig.pose.bones[P + s + "UpLeg"] for s in ("Left", "Right")]
    for pb in [hips] + legs:
        key_rotation(pb, 1)
    forward = figure_forward(rig)
    w = rig.matrix_world
    hip_l, hip_r = (w @ rig.pose.bones[P + s + "UpLeg"].head for s in ("Left", "Right"))
    axis = (hip_l - hip_r).normalized()
    if axis.cross(Vector((0, 0, 1))).dot(forward) < 0:      # turning up toward forward
        axis = -axis
    scene.frame_set(2)
    turn_about(rig, hips, (hip_l + hip_r) / 2, axis, 20.0)
    for pb in legs:
        turn_about(rig, pb, w @ pb.head, axis, -20.0)
    for pb in [hips] + legs:
        key_rotation(pb, 2)
    yaw = inspect_run.figure_yaw(inspect_run.joints_at(rig, 1))
    side_normal = inspect_run.view_direction(*inspect_run.camera_for("side", "right", yaw)[:2])
    behind_normal = inspect_run.view_direction(*inspect_run.camera_for("behind", "right", yaw)[:2])
    turn = inspect_run.torso_turn(rig, side_normal)
    read = {f: inspect_run.measure_frame(inspect_run.joints_at(rig, f), "right", side_normal, behind_normal, turn)
            for f in (1, 2)}
    plain = {f: inspect_run.measure_frame(inspect_run.joints_at(rig, f), "right", side_normal, behind_normal)
             for f in (1, 2)}
    print(f"test: MPFB human's own rest lean {inspect_run.rest_lean(rig, side_normal):.2f}; lean turned "
          f"{read[1]['flat']['torso_lean']} -> {read[2]['flat']['torso_lean']}, joints "
          f"{plain[1]['flat']['torso_lean']} -> {plain[2]['flat']['torso_lean']}")
    assert abs(read[1]["flat"]["torso_lean"]) < 0.01 and abs(read[1]["3d"]["torso_lean"]) < 0.01, read[1]
    assert abs(read[2]["flat"]["torso_lean"] - 20) < 0.3 and abs(read[2]["3d"]["torso_lean"] - 20) < 0.3, read[2]
    assert abs((read[1]["flat"]["hip_included"] - read[2]["flat"]["hip_included"]) - 20) < 0.5, read
    assert abs(read[1]["3d"]["upperarm_vs_torso"] - read[2]["3d"]["upperarm_vs_torso"]) < 0.5, read
    assert read[2]["flat"]["knee_included"] == plain[2]["flat"]["knee_included"]
    # a rig whose shoulder joints sit 76 mm behind its hips (the cartoon's): standing it reads upright,
    # where the joints alone read the backward lean as a lean
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    back = (w.inverted().to_3x3() @ (-forward * 0.076))
    for name in ("LeftShoulder", "RightShoulder"):
        eb = rig.data.edit_bones[P + name]
        eb.tail += back
    for name in ("LeftArm", "RightArm"):
        eb = rig.data.edit_bones[P + name]
        eb.head += back
        eb.tail += back
    bpy.ops.object.mode_set(mode="OBJECT")
    scene.frame_set(1)
    lean = inspect_run.rest_lean(rig, side_normal)
    turn = inspect_run.torso_turn(rig, side_normal)
    j = inspect_run.joints_at(rig, 1)
    turned = inspect_run.measure_frame(j, "right", side_normal, behind_normal, turn)["flat"]["torso_lean"]
    joints = inspect_run.measure_frame(j, "right", side_normal, behind_normal)["flat"]["torso_lean"]
    print(f"test: shoulders 76 mm back: rest lean {lean:.2f}; standing reads {turned} (joints alone {joints})")
    assert lean < -10 and abs(turned) < 0.01 and joints > 10, (lean, turned, joints)
    scene.frame_set(2)
    tipped = inspect_run.measure_frame(inspect_run.joints_at(rig, 2), "right", side_normal, behind_normal, turn)
    assert abs(tipped["flat"]["torso_lean"] - 20) < 0.3, tipped
    print("test: lean PASS")


def house_human():
    """An MPFB human in the house frame, as the run scenes' rig is (mpfb_cage: "the finished rig is unrotated in
    the house frame"): its bones and skin turned half round in their own data, so the armature itself faces +Y
    and the objects stay unrotated (the solver reads the rig's armature space as the house frame)."""
    body, rig = new_human()
    turn = Matrix.Rotation(math.pi, 4, "Z")
    assert rig.matrix_world == Matrix() and body.matrix_world == Matrix(), (rig.matrix_world, body.matrix_world)
    body.data.transform(turn)
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    # unhooked first: turning a parent drags a connected child's head along, which its own turn then turns
    # again (task 20: the knees crossed to the other side of the body and the ankles came back)
    connected = {eb.name for eb in rig.data.edit_bones if eb.use_connect}
    for eb in rig.data.edit_bones:
        eb.use_connect = False
    for eb in rig.data.edit_bones:
        eb.transform(turn)
    for eb in rig.data.edit_bones:
        eb.use_connect = eb.name in connected
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.context.view_layer.update()
    assert figure_forward(rig).y > 0.99, figure_forward(rig)
    w, pb = rig.matrix_world, rig.pose.bones
    for side, sign in (("Left", -1), ("Right", 1)):     # each leg on its own side; facing +Y the right is at +x
        assert all(sign * (w @ pb[P + side + b].head).x > 0.03 for b in ("UpLeg", "Leg", "Foot")), side
    return body, rig


def run_agreement_test():
    """The solver's own readings (pipeline/blender/mpfb_cage.py form_metrics, posture_metrics) and the Studio's
    (inspect_run.measure_frame, posture_frame, spine_curl) agree on one posed rig (task 19, carried from task
    18's review). An MPFB human turned to face +Y, as the run scenes do; the body above the hips tipped 20
    degrees forward about the hip line (legs kept), both knees bent, both arms raised forward, the head nodded.
    The solver reads the side view on the y-z plane and the widths on the x-z plane; the Studio through its
    side and behind cameras (5 degrees up): they may differ by the camera's tilt, which this bounds."""
    import mpfb_cage
    body, rig = house_human()
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, 1
    scene.frame_set(1)
    forward = figure_forward(rig)
    w = rig.matrix_world
    hips = rig.pose.bones[P + "Hips"]
    legs = [rig.pose.bones[P + s + "UpLeg"] for s in ("Left", "Right")]
    hip_l, hip_r = (w @ rig.pose.bones[P + s + "UpLeg"].head for s in ("Left", "Right"))
    axis = (hip_l - hip_r).normalized()
    if axis.cross(Vector((0, 0, 1))).dot(forward) < 0:
        axis = -axis
    turn_about(rig, hips, (hip_l + hip_r) / 2, axis, 20.0)
    for pb in legs:
        turn_about(rig, pb, w @ pb.head, axis, -20.0 - 35.0)        # the thighs come forward 35 degrees
    for side in ("Left", "Right"):
        knee = rig.pose.bones[P + side + "Leg"]
        turn_about(rig, knee, w @ knee.head, axis, 60.0)            # and the shins back: knees bent
    arms_forward(rig, forward)
    for name in ("Neck", "Head"):
        pb = rig.pose.bones[P + name]
        turn_about(rig, pb, w @ pb.head, axis, 8.0)
    for pb in rig.pose.bones:
        key_rotation(pb, 1)
    pb_root = rig.pose.bones[P + "Hips"]
    scene.frame_set(1)
    yaw, _, side_normal, behind_normal, turn = inspect_run.studio_view(rig, "right")
    joints = inspect_run.joints_at(rig, 1)
    # the legs counted from the bind pose too, as measure_frames counts them (task 20)
    studio = inspect_run.measure_frame(joints, "right", side_normal, behind_normal, turn,
                                       inspect_run.leg_turns(rig, side_normal))
    ears = inspect_run.ear_points(rig, body)
    face = inspect_run.head_at(rig, inspect_run.head_forward(rig))[1]
    posture = inspect_run.posture_frame(joints, inspect_run.ears_at(rig, ears), face, side_normal, turn)
    curl = inspect_run.spine_curl(rig, side_normal, inspect_run.rest_spine(rig, side_normal))
    solver = mpfb_cage.form_metrics(rig, flat=("knee_included", "hip_included"))
    stance = mpfb_cage.posture_metrics(rig, mpfb_cage.ears_local(rig, body), mpfb_cage._spine_tilts(rig, rest=True))
    pairs = {"torso_lean": (solver["torso_lean"], studio["flat"]["torso_lean"]),
             "hip_included": (solver["hip_included"], studio["flat"]["hip_included"]),
             "knee_included": (solver["knee_included"], studio["flat"]["knee_included"]),
             "elbow_included": (solver["elbow_included"], studio["3d"]["elbow_included"]),
             "upperarm_vs_torso": (solver["upperarm_vs_torso"], studio["3d"]["upperarm_vs_torso"]),
             "grip_ratio": (solver["grip_ratio"], studio["flat"]["grip_ratio"]),
             "elbow_width_ratio": (solver["elbow_width_ratio"], studio["flat"]["elbow_width_ratio"]),
             "neck_vs_torso": (stance["neck_vs_torso"], posture["flat"]["neck_vs_torso"]),
             "head_pitch": (stance["head_pitch"], posture["flat"]["head_pitch"]),
             "spine_curl": (stance["spine_curl"], curl)}
    for k, (a, b) in pairs.items():
        print(f"test: agreement {k}: solver {a:.3f}, Studio {b:.3f}")
    assert 15 < pairs["torso_lean"][1] < 25 and pairs["knee_included"][1] < 150, pairs     # the pose is the one meant
    for k, (a, b) in pairs.items():
        limit = 0.02 if k.endswith("ratio") else 1.0
        assert abs(a - b) <= limit, (k, a, b)
    # the solver's grip width read on the Studio's behind camera (match_arms, task 19) is the Studio's own
    j = mpfb_cage._joints(rig)
    bn = Vector(behind_normal.tolist())
    grip = mpfb_cage._width(j["wrist.right"] - j["wrist.left"], bn) / mpfb_cage._width(j["shoulder.right"] - j["shoulder.left"], bn)
    assert abs(grip - studio["flat"]["grip_ratio"]) <= 5e-4, (grip, studio["flat"]["grip_ratio"])   # ours.json: 3 decimals
    print("test: agreement PASS")


def slope_legs(rig, degrees):
    """The rig's bind pose edited so each leg's knee and ankle joints sit `degrees` back of where they were,
    turned about the hip joint (the shin with the thigh, the foot moved along): the joint lines then slope back
    as the cartoon's do (10.6), while the skin stays where it stood."""
    forward = figure_forward(rig)
    w = rig.matrix_world
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    eb = rig.data.edit_bones
    local = w.inverted().to_3x3()
    up_axis = Vector((0.0, 0.0, 1.0)).cross(forward).normalized()       # turning down toward the back by +degrees
    turn = local @ Matrix.Rotation(math.radians(degrees), 3, up_axis) @ w.to_3x3()
    for side in ("Left", "Right"):
        hip = eb[P + side + "UpLeg"].head.copy()
        knee, ankle = eb[P + side + "Leg"].head.copy(), eb[P + side + "Foot"].head.copy()
        knee2 = hip + turn @ (knee - hip)
        ankle2 = knee2 + turn @ (ankle - knee)
        move = ankle2 - ankle
        feet = {b.name: (b.head + move, b.tail + move) for b in eb
                if b.name.startswith(P + side + "Foot") or b.name.startswith(P + side + "Toe")}
        for b in eb:                # unhooked first, so no connected joint drags another along
            if b.name.startswith(P + side):
                b.use_connect = False
        eb[P + side + "UpLeg"].tail = knee2
        eb[P + side + "Leg"].head, eb[P + side + "Leg"].tail = knee2, ankle2
        for name, (head, tail) in feet.items():
            eb[name].head, eb[name].tail = head, tail
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.context.view_layer.update()


def posed_legs(sloped):
    """An MPFB human (`sloped`: its knee and ankle joints first moved 10.6 degrees back inside the same skin, as the
    cartoon's sit) standing at frame 1, and at frame 2 both thighs turned 90 degrees forward and the shins back
    down: the thighs' skin then lies 90 degrees from where it stood. Returns (rig, the Studio's readings
    {frame: {"flat", "3d"}}, its leg slopes, the solver's {frame: form_metrics})."""
    import mpfb_cage
    body, rig = new_human()
    if sloped:
        slope_legs(rig, 10.6)
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, 2
    scene.frame_set(1)
    legs = [rig.pose.bones[P + s + n] for s in ("Left", "Right") for n in ("UpLeg", "Leg")]
    for pb in legs:
        key_rotation(pb, 1)
    forward = figure_forward(rig)
    w = rig.matrix_world
    hip_l, hip_r = (w @ rig.pose.bones[P + s + "UpLeg"].head for s in ("Left", "Right"))
    axis = (hip_l - hip_r).normalized()
    if axis.cross(Vector((0, 0, 1))).dot(forward) < 0:      # + turns up toward the front, down toward the back
        axis = -axis
    scene.frame_set(2)
    for side in ("Left", "Right"):
        thigh, shin = rig.pose.bones[P + side + "UpLeg"], rig.pose.bones[P + side + "Leg"]
        turn_about(rig, thigh, w @ thigh.head, axis, -90.0)     # the thigh forward by a right angle
        turn_about(rig, shin, w @ shin.head, axis, 90.0)        # the shin back down
    for pb in legs:
        key_rotation(pb, 2)
    yaw, _, side_normal, behind_normal, turn = inspect_run.studio_view(rig, "right")
    per_frame, _ = inspect_run.measure_frames(rig, 1, 2, "right", side_normal, behind_normal, turn)
    solver = {}
    for f in (1, 2):
        scene.frame_set(f)
        solver[f] = mpfb_cage.form_metrics(rig, flat=("knee_included", "hip_included"))
    return rig, {f["frame"]: f for f in per_frame}, inspect_run.leg_slopes(rig, side_normal), solver


def run_leg_test():
    """Task 20: our thigh line (hip -> knee joints, as the coach's MediaPipe landmarks) is laid onto the thigh's
    own skin outline as the bind pose shows it, the shin turned with it, so the legs read the way they look
    wherever the rig's joints sit inside the skin (the cartoon's hip joint sits toward the front of the thigh's
    top: its thigh joint line slopes 10.6 degrees back standing, its skin 4.5). An MPFB human as it comes and the
    same human with its knee and ankle joints moved 10.6 degrees back inside the same skin read alike, standing
    and with the thighs turned a right angle forward; the bind pose's hip reads 180 less the outline's own lean,
    the knee as its joints; a thigh whose skin lies level reads depth 0; and the solver (form_metrics, the y-z
    plane) reads the hip and knee as the Studio does. (Not house_human: see there.)"""
    rig, plain, slopes, solver = posed_legs(False)
    _, moved, slopes2, solver2 = posed_legs(True)
    print(f"test: legs: a stock MPFB human's slopes {slopes}; joints moved back inside its skin: {slopes2}")
    for f in (1, 2):
        print(f"test: legs frame {f}: as it comes " + ", ".join(f"{k} {plain[f]['flat'][k]}" for k in
              ("hip_included", "knee_included", "hip_below_knee")) + "; joints moved " + ", ".join(
              f"{k} {moved[f]['flat'][k]}" for k in ("hip_included", "knee_included", "hip_below_knee")))
    out = slopes["outline"]["thigh.right"]
    assert all(abs(slopes2["joints"][k] - slopes["joints"][k] + 10.6) < 1.0 for k in ("thigh.left", "thigh.right"))
    # the outline's cuts are square to the joint line, so where the joints sit moves it a little (0.5 degrees here)
    assert all(abs(slopes2["outline"][k] - slopes["outline"][k]) < 1.0 for k in slopes["outline"]), (slopes, slopes2)
    assert abs(out) < 3, slopes                                  # a person-shaped human's thigh stands about straight
    for f in (1, 2):
        for k, tol in (("hip_included", 1.0), ("knee_included", 0.5), ("hip_below_knee", 0.01)):
            assert abs(plain[f]["flat"][k] - moved[f]["flat"][k]) <= tol, (f, k, plain[f]["flat"][k], moved[f]["flat"][k])
    # the bind pose (frame 1): the hip 180 less the outline's lean, the knee as its joints; frame 2: the outline a
    # right angle forward of where it stood (90 + its slope from straight down): hip 90 - slope, depth sin(slope)
    assert abs(plain[1]["flat"]["hip_included"] - (180 - abs(out))) < 0.5, (plain[1], out)
    assert abs(plain[2]["flat"]["hip_included"] - (90 - out)) < 1.0, (plain[2], out)
    assert abs(plain[2]["flat"]["hip_below_knee"]) < 0.02, plain[2]        # the skin about level: about parallel
    for got, sol in ((plain, solver), (moved, solver2)):
        for f in (1, 2):
            for k in ("hip_included", "knee_included"):
                assert abs(sol[f][k] - got[f]["flat"][k]) <= 1.0, (f, k, sol[f][k], got[f]["flat"][k])
    print("test: legs PASS")


SEAT_POSES = {2: (80.0, 80.0), 3: (90.0, 90.0), 4: (100.0, 100.0), 5: (90.0, 105.0), 6: (90.0, 75.0), 7: (90.0, 60.0)}
# frame: (the near (right) thigh's, the far (left) thigh's) degrees forward from the bind pose, the shin back down
SEAT_AGREE = 0.05       # thigh lengths: the most the 3D skin may read from the picture on this human (0.02 measured)


def seat_scene():
    """(body, rig, the side camera, its axis): an MPFB human posed per SEAT_POSES, the torso upright and the arms
    raised forward (MPFB's own hands rest on the thighs), keyed at frames 1-6."""
    body, rig = new_human()
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, max(SEAT_POSES)
    scene.frame_set(1)
    legs = [rig.pose.bones[P + s + n] for s in ("Left", "Right") for n in ("UpLeg", "Leg", "Arm")]
    for pb in legs:
        key_rotation(pb, 1)
    forward = figure_forward(rig)
    w = rig.matrix_world
    hip_l, hip_r = (w @ rig.pose.bones[P + s + "UpLeg"].head for s in ("Left", "Right"))
    axis = (hip_l - hip_r).normalized()
    if axis.cross(Vector((0, 0, 1))).dot(forward) < 0:
        axis = -axis
    for frame, (near, far) in SEAT_POSES.items():
        scene.frame_set(frame)
        for pb in legs:                                      # from the bind pose, not the last frame's key
            pb.matrix_basis = Matrix()
        bpy.context.view_layer.update()
        for side, degrees in (("Right", near), ("Left", far)):
            thigh, shin = rig.pose.bones[P + side + "UpLeg"], rig.pose.bones[P + side + "Leg"]
            turn_about(rig, thigh, w @ thigh.head, axis, -degrees)
            turn_about(rig, shin, w @ shin.head, axis, degrees)
        arms_forward(rig, forward)
        for pb in legs:
            key_rotation(pb, frame)
    _, side_camera, side_normal, _, _ = inspect_run.studio_view(rig, "right")
    return body, rig, side_camera, side_normal


def run_seat_test():
    """Task 22: the depth as a judge sees it - the seat's lowest point against the top of the knee, in thigh lengths
    - on our silhouette (inspect_run.seat_picture, posture.seat_depth, the coach's own function) and its cross-check
    off the 3D skin (seat_skin), on real geometry. Both thighs 80, 90 and 100 degrees forward: the depth rises with
    them (the knee rises 0.17 thigh lengths a step, the seat a little with it). The far thigh 15 degrees above the
    near one: its knee's top shows above the near one's and is the one read, on the picture and the skin alike (the
    top of whichever knee shows). Below it, hidden behind the near knee: the near knee's top is read, however low
    the far one goes. Level, the far knee seen from the camera's 5 degrees up rides higher in the picture and on the
    camera's plane than the near one, so it is the one read: lowered out of sight the depth reads less. The skin
    reads within SEAT_AGREE of the picture throughout. The masks and points are kept (seat-<frame>.npz)."""
    body, rig, side_camera, side_normal = seat_scene()
    folder = Path(tempfile.mkdtemp(prefix="seat-test-"))
    seen, skin = {}, {}
    for frame in SEAT_POSES:
        read = inspect_run.seat_picture(rig, body, frame, "right", side_camera, keep=True)
        np.savez(folder / f"seat-{frame}.npz", mask=read["mask"].astype(np.float32), hip=read["hip"], knee=read["knee"],
                 seat=read["seat"] or [-1, -1], knee_top=read["knee_top"] or [-1, -1], depth=read["picture"] or np.nan)
        seen[frame] = read["picture"]
        skin[frame] = inspect_run.seat_skin(rig, body, frame, "right", side_normal)["skin"]
    print(f"test: seat (kept in {folder}): " + ", ".join(
        f"f{f} {SEAT_POSES[f]} picture {seen[f]} skin {skin[f]}" for f in SEAT_POSES))
    assert all(seen[f] is not None and skin[f] is not None for f in SEAT_POSES), (seen, skin)
    for read in (seen, skin):
        assert all(0.08 <= read[b] - read[a] <= 0.2 for a, b in ((2, 3), (3, 4))), read      # deeper as they rise
        assert read[5] - read[3] >= 0.6 * math.sin(math.radians(15.0)), read    # the far knee above: its top read
        # the far knee hidden: it no longer counts (the skin's knee top is the same vertex at both; lowered to 60, the
        # far thigh's underside at the hip becomes the skin's lowest seat point, 0.018 lower)
        assert abs(read[7] - read[6]) <= 0.03, read
        assert read[3] - read[6] >= 0.02, read                  # level: the far knee rides higher, seen from above
    assert all(abs(seen[f] - skin[f]) <= SEAT_AGREE for f in SEAT_POSES), (seen, skin)
    # each reading on its own: a failure rendering our silhouette costs the picture's reading, said, never the skin's
    def broken(*args, **kwargs):
        raise RuntimeError("the mask render failed")

    kept, inspect_run.seat_picture = inspect_run.seat_picture, broken
    try:
        read = inspect_run.seat_reading(rig, body, 3, "right", side_camera, side_normal)
    finally:
        inspect_run.seat_picture = kept
    assert read["picture"] is None and "the mask render failed" in read["why"] and read["skin"] == skin[3], read
    print("test: seat PASS")


def run_seat_hold_test():
    """The solver holds the bottom's seat depth at the coach's (mpfb_cage._hold_seat, task 22): on a stand-in for our
    picture whose depth falls 0.009 per degree of the bottom's hip-angle target, it moves that target until the depth
    reads within SEAT_HOLD of his, re-solving each time; when the depth can't be read, or only the 3D skin reads it (it
    runs about 0.08 below the picture on the squat), it leaves the bottom as solved."""
    import mpfb_cage
    studio = {"phases": {"end": {"seat_below_knee": {"value": 0.646}}}}
    aim = {"phases": {"bottom": {"hip_included": 80.0, "torso_lean": 40.0}}}
    solved, said = [], []
    kept = mpfb_cage._seat_at
    try:
        mpfb_cage._seat_at = lambda *a: (0.56 - 0.009 * (solved[-1] - 80.0) if solved else 0.56, 38, "picture")
        out = mpfb_cage._hold_seat(None, None, aim, studio, 1, 2, said.append,
                                   lambda a: solved.append(a["phases"]["bottom"]["hip_included"]))
        assert solved and abs(0.56 - 0.009 * (solved[-1] - 80.0) - 0.646) <= mpfb_cage.SEAT_HOLD, (solved, said)
        assert out["phases"]["bottom"]["hip_included"] == solved[-1] and out["phases"]["bottom"]["torso_lean"] == 40.0
        assert aim["phases"]["bottom"]["hip_included"] == 80.0                     # the caller's targets untouched
        mpfb_cage._seat_at = lambda *a: (None, 38, "skin")
        assert mpfb_cage._hold_seat(None, None, aim, studio, 1, 2, said.append, None) is aim
        assert "can't be read" in said[-1], said
        mpfb_cage._seat_at = lambda *a: (0.56, 38, "skin")         # the picture unreadable, the skin read
        assert mpfb_cage._hold_seat(None, None, aim, studio, 1, 2, said.append, None) is aim
        assert "picture unreadable; depth left as solved" in said[-1], said
    finally:
        mpfb_cage._seat_at = kept
    print(f"test: seat hold PASS ({len(solved)} re-solves: {[round(v, 1) for v in solved]})")


def run_bind_pose_test():
    """Task 19 review: the one bind-pose rule (studio/measure/metrics.py), read on the Studio's side camera, fires
    for a rig whose bind pose leans past 25 degrees or whose chest tips past 30 - in the Studio (inspect_run
    `bind_pose`) and in the solver's log (mpfb_cage `bind_pose_warning`) alike - and stays quiet on a stock human."""
    import mpfb_cage
    body, rig = new_human()
    assert inspect_run.bind_pose(rig, "right")[2] is None and mpfb_cage.bind_pose_warning(rig) is None

    def move(names_heads, names_tails, offset):
        bpy.context.view_layer.objects.active = rig
        rig.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")
        for n in names_heads:
            eb = rig.data.edit_bones[P + n]
            eb.head += offset
            eb.tail += offset
        for n in names_tails:
            rig.data.edit_bones[P + n].tail += offset
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.context.view_layer.update()

    forward = figure_forward(rig)
    back = rig.matrix_world.inverted().to_3x3() @ (-forward * 0.30)       # the shoulder joints 30 cm behind the hips
    move(("LeftArm", "RightArm"), ("LeftShoulder", "RightShoulder"), back)
    lean, chest, studio = inspect_run.bind_pose(rig, "right")
    solver = mpfb_cage.bind_pose_warning(rig)
    print(f"test: bind pose with the shoulders 30 cm back: lean {lean:.1f}, chest {chest:.1f}; {studio}")
    assert lean < -25 and "its torso line leans" in studio, (lean, studio)
    assert solver == "form: " + studio, solver
    move(("LeftArm", "RightArm"), ("LeftShoulder", "RightShoulder"), -back)          # put them back
    ahead = rig.matrix_world.inverted().to_3x3() @ (forward * 0.25)
    move(("Neck", "Head"), ("Spine2",), ahead)                                        # the chest tipped forward
    lean, chest, studio = inspect_run.bind_pose(rig, "right")
    print(f"test: bind pose with the neck 25 cm forward: lean {lean:.1f}, chest {chest:.1f}; {studio}")
    assert chest > 30 and "its chest tips" in studio, (chest, studio)
    print("test: bind pose PASS")


def run_arm_rules_test():
    """Task 19 review, the arm search's rules on hand-made numbers (pipeline/blender/mpfb_cage.py):
    an arm row counts as met only within ARM_MARGIN (0.8) of the tolerance - at 0.9 it does not; a candidate that
    meets the grip outranks one meeting all three upper arms without it, and the log says the grip was kept;
    the grip is read against each frame's own shoulder width, not the standing one."""
    import mpfb_cage
    tol = 5.0
    targets = {"phases": {p: {"upperarm_vs_torso": 41.3, "elbow_included": 90.0} for p in ("standing", "mid", "bottom")},
               "tolerance": {"angles_deg": tol, "ratios": 0.12}, "camera_side": "right",
               "arms_all_phases": {"grip_ratio": 1.87, "elbow_width_ratio": 1.97}}
    S, H = Vector((-0.15, 0.0, 1.0)), Vector((-0.10, 0.0, 0.5))
    j = {"shoulder.right": S, "shoulder.left": Vector((0.15, 0.0, 1.0)), "hip.right": H, "hip.left": Vector((0.10, 0.0, 0.5))}

    def arm_at(miss):
        """An elbow whose upper arm is `miss` x the tolerance past the coach's 41.3 degrees from the torso line."""
        down = (H - S).normalized()
        a = math.radians(41.3 + miss * tol)
        out = Vector((0.0, 1.0, 0.0))
        return S + (down * math.cos(a) + out * math.sin(a)) * 0.3

    def rows(miss):
        el = {"Right": arm_at(miss), "Left": Vector((0.3, 0.2, 0.8))}
        wr = {"Right": Vector((-0.28, 0.1, 1.0)), "Left": Vector((0.28, 0.1, 1.0))}
        state = {7: (j, None)}
        return mpfb_cage._studio_arm_rows(state, {7: (el, wr)}, {"standing": 7, "mid": 7, "bottom": 7}, targets,
                                          Matrix.Identity(3))
    r9, r7 = rows(0.9), rows(0.7)
    print(f"test: arm rows at 0.9 x tolerance {r9['upper']} met (worst {r9['worst_upper']:.2f}), at 0.7 {r7['upper']}")
    assert r9["upper"] == 0 and abs(r9["worst_upper"] - 0.9) < 1e-6, r9
    assert r7["upper"] == 3 and abs(r7["worst_upper"] - 0.7) < 1e-6, r7

    # the ranking, on candidates in match_arms' own trace format
    grip = {"upper": 2, "worst_upper": 1.9, "rows": 3}
    upper = {"upper": 3, "worst_upper": 0.3, "rows": 5}
    cands = [{"drop": -0.14, "ratio": 1.8, "score": list(mpfb_cage._arm_score(True, False, upper, 40, 10.0)), **upper},
             {"drop": -0.16, "ratio": 1.93, "score": list(mpfb_cage._arm_score(True, True, grip, 30, 20.0)), **grip},
             {"drop": -0.03, "ratio": 1.93, "score": list(mpfb_cage._arm_score(False, True, upper, 50, 1.0)), **upper}]
    chosen = mpfb_cage._arm_choice(cands)
    note = mpfb_cage._grip_trade_note(chosen, cands)
    print(f"test: arm search chose {chosen['drop']} / {chosen['ratio']}; {note}")
    assert (chosen["drop"], chosen["ratio"]) == (-0.16, 1.93), chosen       # the grip, wrists within limits
    assert note.startswith("form: the grip is kept over the upper arm") and "1 other candidate(s)" in note, note
    assert mpfb_cage._grip_trade_note(cands[0], cands) is None               # no grip kept: nothing to say

    # the grip width against this frame's own shoulder width: shoulders drawn in to 0.30 (standing 0.36)
    jj = {"shoulder.right": Vector((-0.15, 0.0, 1.0)), "shoulder.left": Vector((0.15, 0.0, 1.0))}
    wr = {"Right": Vector((-0.27, 0.05, 1.02)), "Left": Vector((0.27, 0.05, 1.02))}
    el = {"Right": Vector((-0.25, -0.05, 0.85)), "Left": Vector((0.25, -0.05, 0.85))}
    ew, gw = mpfb_cage._grip_widths(jj, el, wr)
    assert abs(gw - 1.80) < 1e-6 and abs(ew - 0.50 / 0.30) < 1e-6, (ew, gw)   # 0.54 / 0.36 would read 1.50
    tilted = Vector((0.0, -math.cos(math.radians(5)), math.sin(math.radians(5))))
    ew2, gw2 = mpfb_cage._grip_widths(jj, el, wr, tilted)
    assert abs(gw2 - 1.80) < 1e-3, gw2            # level widths read alike through the 5-degree behind camera
    print("test: arm rules PASS")


RAISED = 2


def run_shoulder_test():
    """An MPFB human, rest pose at frame 1, its left arm raised 150 degrees sideways (overhead) at
    frame RAISED: the skin on top of that shoulder folds over and the GIF's cameras see it, so the row
    fails naming the left shoulder. Behind a wall from both cameras the same fold is hidden: info."""
    body, rig = new_human()
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, RAISED
    scene.frame_set(1)
    # the ankles count their bend from the rest angle, the others from straight: in the rest pose an
    # ankle is unbent, and standing in MPFB's default pose (frame 1) nothing is bent enough to crease
    rig.data.pose_position = "REST"
    bpy.context.view_layer.update()
    rest = {h.name: creases.bend_deg(h) for h in inspect_run.hinges(rig)}
    rig.data.pose_position = "POSE"
    bpy.context.view_layer.update()
    standing = {h.name: creases.bend_deg(h) for h in inspect_run.hinges(rig)}
    print("test: hinge bends at rest", {k: round(v, 1) for k, v in rest.items()},
          "standing", {k: round(v, 1) for k, v in standing.items()})
    assert len(rest) == 8 and all(abs(rest[f"{s} ankle"]) < 1e-3 for s in ("left", "right")), rest
    assert all(b < creases.CREASE_BEND_DEG for b in standing.values()), standing
    pb = rig.pose.bones[P + "LeftArm"]
    key_rotation(pb, 1)
    w = rig.matrix_world
    left = w @ rig.pose.bones[P + "LeftUpLeg"].head - w @ rig.pose.bones[P + "RightUpLeg"].head
    left.z = 0
    forward = left.normalized().cross(Vector((0, 0, 1)))
    scene.frame_set(RAISED)
    shoulder = w @ pb.head
    turn = Matrix.Translation(shoulder) @ Matrix.Rotation(math.radians(150), 4, -forward) @ Matrix.Translation(-shoulder)
    pb.matrix = w.inverted() @ turn @ w @ pb.matrix
    bpy.context.view_layer.update()
    key_rotation(pb, RAISED)
    cameras = gif_of(rig, body, RAISED)

    def judged(measured):
        ours = {"first": 1, "last": RAISED, "height_m": measured["height_m"], "folds": measured["folds"]}
        return checks._fold_rows(ours)[0]

    seen = inspect_run.skin_folds(rig, body, 1, RAISED, lambda frame: cameras)
    for f in seen["folds"]:
        print("test: shoulder fold", f)
    assert not [f for f in seen["folds"] if f["frame"] == 1], "the rest pose has turned-over skin"
    top = [f for f in seen["folds"] if f["bone"] == P + "LeftArm" and f["kind"] == "defect"]
    assert top, seen["folds"]
    row = judged(seen)
    assert row["status"] == "fail" and row["note"].startswith("Skin folds over at the left shoulder ("), row
    centre = np.array([f["centre"] for f in top]).mean(axis=0)
    # walls that are not rendered hide nothing: one in a collection off for render, one in a collection
    # left out of the view layer
    off, out = bpy.data.collections.new("off"), bpy.data.collections.new("out")
    for c in (off, out):
        scene.collection.children.link(c)
    off.hide_render = True
    bpy.context.view_layer.layer_collection.children["out"].exclude = True
    ghosts = [wall_between(c, centre, collection=col) for c in cameras for col in (off, out)]
    unhidden = inspect_run.skin_folds(rig, body, 1, RAISED, lambda frame: cameras)
    for wall in ghosts:
        bpy.data.objects.remove(wall, do_unlink=True)
    assert [(f["kind"], f["visible_mm2"]) for f in unhidden["folds"]] == [(f["kind"], f["visible_mm2"]) for f in seen["folds"]], \
        (unhidden["folds"], seen["folds"])
    walls = [wall_between(c, centre) for c in cameras]
    hidden = inspect_run.skin_folds(rig, body, 1, RAISED, lambda frame: cameras)
    for wall in walls:
        bpy.data.objects.remove(wall, do_unlink=True)
    assert hidden["folds"] and all(f["kind"] == "hidden" and f["visible_mm2"] == 0 for f in hidden["folds"]), hidden["folds"]
    row = judged(hidden)
    assert row["status"] == "info" and row["note"].startswith("Folds hidden from every view at the left shoulder ("), row
    print(f"test: shoulder PASS ({top[0]['area_mm2']} mm² seen {top[0]['visible_mm2']} mm²)")


def run_test():
    base = Path(tempfile.mkdtemp(prefix="inspect-test-"))
    plain, turned = base / "plain", base / "turned"
    plain.mkdir()
    turned.mkdir()
    print(f"test: run folders {plain} and {turned}")
    rig = build_scene(plain)
    save_turned(rig, turned)
    write_run_files(plain, "Barbell Back Squat")      # its depth read (exercise-kit.json `depth`, task 22)
    write_run_files(turned)
    assert inspect_run.main(["--run", str(plain)]) == 0
    a, a_px = check(plain)
    assert inspect_run.main(["--run", str(turned)]) == 0
    b, b_px = check(turned)
    # the cameras follow the figure, not the world: the turned run measures and renders the same
    assert b["phase_frames"] == a["phase_frames"], (a["phase_frames"], b["phase_frames"])
    knee_a, knee_b = (o["per_frame"][BENT - 1]["flat"]["knee_included"] for o in (a, b))
    assert abs(knee_a - knee_b) < 0.5, (knee_a, knee_b)
    assert [s["facing"] for s in a["stills"]] == [s["facing"] for s in b["stills"]]
    # posture too: the turned figure is seen by cameras that turn with it
    for fa, fb in zip(a["per_frame"], b["per_frame"]):
        for method, metric in (("flat", "neck_vs_torso"), ("flat", "head_pitch"), ("3d", "neck_vs_torso"),
                               ("flat", "spine_curl")):
            va, vb = fa[method].get(metric), fb[method].get(metric)
            assert (va is None) == (vb is None), (fa["frame"], metric, va, vb)
            if va is not None:
                assert abs(va - vb) <= 0.5, (fa["frame"], metric, va, vb)
    # turned skin is measured against the figure's own bones and seen by cameras that turn with it, not
    # the world's: the same folds either way
    fa, fb = ([(f["frame"], f["bone"], f["kind"], f["area_mm2"], f["visible_mm2"]) for f in o["folds"]] for o in (a, b))
    assert [x[:3] for x in fa] == [x[:3] for x in fb], (fa, fb)
    assert all(abs(x[3] - y[3]) <= 0.2 and abs(x[4] - y[4]) <= max(1.0, 0.05 * x[3]) for x, y in zip(fa, fb)), (fa, fb)
    for image in a_px:
        diff = float(np.abs(a_px[image] - b_px[image]).mean())
        print(f"test: {image} turned vs plain, mean pixel difference {diff:.4f}")
        assert diff < 0.01, f"{image}: the turned figure's still differs from the plain one ({diff:.4f})"
    # the picture too (the same pictures, read by the same detector)
    for phase in ("start", "end"):
        pa, pb = a["picture"][phase], b["picture"][phase]
        assert pa["ok"] == pb["ok"], (phase, pa, pb)
        if pa["ok"]:
            for m in ("torso_lean", "hip_included"):
                assert abs(pa["flat"][m] - pb["flat"][m]) <= 1.5, (phase, m, pa["flat"], pb["flat"])
    positions_test(plain, a)
    gif_box_test(plain)
    fold_failure_test(plain)
    picture_failure_test(plain)
    print(f"test: PASS ({base})")


def positions_test(run, ours):
    """The solver picks the coach's positions as the Studio does (task 19): mpfb_cage._studio_positions on the
    run's scene and targets gives the Studio's own phase frames."""
    import mpfb_cage
    bpy.ops.wm.open_mainfile(filepath=str(run / "scene.blend"))
    rig, _ = inspect_run.find_rig_and_skin()
    targets = json.loads((run / "targets.json").read_text())
    scene = bpy.context.scene
    positions, _, driver = mpfb_cage._studio_positions(rig, targets, scene.frame_start, scene.frame_end)
    want = {mpfb_cage.STUDIO_PHASE_OF[p]: f for p, f in ours["phase_frames"].items()}
    assert positions == want and driver == ours["phase_driver"], (positions, want, driver)
    print("test: positions PASS", positions)


def picture_failure_test(run):
    """A failure reading our picture costs the picture, never the stills: ours.json is written with every
    position "not read" and the reason, and the torso rows then fall back to the rig, saying why."""
    import contextlib
    import io

    def broken(files):
        raise inspect_run.RunError("MediaPipe could not read our pictures (test).")

    kept, inspect_run.read_pictures = inspect_run.read_pictures, broken
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            code = inspect_run.main(["--run", str(run)])
    finally:
        inspect_run.read_pictures = kept
    assert code == 0, out.getvalue()
    assert "picture not read: MediaPipe could not read our pictures (test)." in out.getvalue(), out.getvalue()
    ours = json.loads((run / "ours.json").read_text())
    for phase in ("start", "middle", "end"):
        p = ours["picture"][phase]
        assert p["ok"] is False and p["why"] == "MediaPipe could not read our pictures (test).", p
    assert len(ours["stills"]) == 2
    print("test: picture failure PASS")


def gif_box_test(run):
    """The GIF's cameras stand where animate placed them: around frames/retarget.json's contact box when
    the run has one, else around the skin's box over the clip, measured."""
    bpy.ops.wm.open_mainfile(filepath=str(run / "scene.blend"))
    rig, skin = inspect_run.find_rig_and_skin()
    scene = bpy.context.scene
    first, last = scene.frame_start, scene.frame_end
    measured = inspect_run.gif_box(run, skin, first, last)
    pts = np.concatenate([inspect_run.figure_points(skin, f) for f in range(first, last + 1)])
    assert np.allclose(measured, [pts.min(axis=0), pts.max(axis=0)], atol=1e-6), measured
    box = [[-0.5, -0.4, 0.0], [0.5, 0.3, 1.8]]
    (run / "frames").mkdir(exist_ok=True)
    (run / "frames" / "retarget.json").write_text(json.dumps({"contact": {"box": box}}))
    assert np.allclose(inspect_run.gif_box(run, skin, first, last), box)
    (run / "frames" / "retarget.json").write_text("{not json")
    assert np.allclose(inspect_run.gif_box(run, skin, first, last), measured)
    (run / "frames" / "retarget.json").unlink()


def fold_failure_test(run):
    """A failure inside the fold measurement costs the folds, never the stills: inspect_run still
    writes ours.json and the stills, without the fold keys, so the check reads "not measured"."""
    import contextlib
    import io

    def broken(*args, **kwargs):
        raise RuntimeError("the BVH could not be built")

    kept, inspect_run.skin_folds = inspect_run.skin_folds, broken
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            code = inspect_run.main(["--run", str(run)])
    finally:
        inspect_run.skin_folds = kept
    print(out.getvalue())
    assert code == 0, out.getvalue()
    assert "folds not measured: RuntimeError: the BVH could not be built" in out.getvalue(), out.getvalue()
    ours = json.loads((run / "ours.json").read_text())
    assert not {"folds", "folds_worst", "height_m"} & set(ours) and len(ours["stills"]) == 2, list(ours)
    row = checks._fold_rows(ours)[0]
    assert (row["status"], row["note"]) == ("info", "Not measured on the character"), row
    print("test: fold failure PASS")


run_leg_test()
run_seat_test()
run_seat_hold_test()
run_bind_pose_test()
run_arm_rules_test()
run_agreement_test()
run_lean_test()
run_posture_test()
run_fold_test()
run_shoulder_test()
run_test()
