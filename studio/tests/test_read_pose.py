import pytest
from PIL import Image
from pytest import approx

from conftest import REF
from read_pose import NoBody, read_pose


@pytest.mark.parametrize("name,knee_f,knee_3,hip_f,hip_3,elbow_3,ua_3,lean", [
    ("15.png", 174.7, 160.4, 170.8, 167.1, 90.5, 41.3, 6.7),
    ("16.png", 104.7, 87.9, 109.2, 97.9, 99.9, 50.5, 18.4),
    ("17.png", 47.2, 64.5, 69.3, 77.3, 82.3, 52.0, 22.4)])
def test_side_views(name, knee_f, knee_3, hip_f, hip_3, elbow_3, ua_3, lean):
    r = read_pose(REF / name)
    assert (r["view"], r["camera_side"], r["facing"]) == ("side", "right", "right")
    assert r["flat"]["knee_included"] == approx(knee_f, abs=1.5)
    assert r["3d"]["knee_included"] == approx(knee_3, abs=1.5)
    assert r["flat"]["hip_included"] == approx(hip_f, abs=1.5)
    assert r["3d"]["hip_included"] == approx(hip_3, abs=1.5)
    assert r["3d"]["elbow_included"] == approx(elbow_3, abs=1.5)
    assert r["3d"]["upperarm_vs_torso"] == approx(ua_3, abs=1.5)
    assert r["flat"]["torso_lean"] == approx(lean, abs=1.0)


# ---- posture (studio/measure/posture.py), looked at on the pictures before pinning ----
# 15 standing: head bent down and forward (chin toward the chest, eyes on the floor ahead), back straight.
# 16 half way and 17 at the bottom: back straight, head a little forward of the back's line, eyes ahead
# and a little down. The arms (elbows back under the bar) hide the upper back in all three, so the
# back is read on the lower back only (green in scratchpad overlays: about 22-26% of the 10-90% stretch).
# A straight lower back has a slight S (buttock, then the waist's hollow), so a value this near 0 can
# take either sign.
@pytest.mark.parametrize("name,rounding,neck_f,neck_3,pitch", [
    ("15.png", -0.011, 38.6, 52.1, -36.2),
    ("16.png", 0.026, 13.4, 27.0, -14.3),
    ("17.png", -0.010, 13.5, 16.6, -11.4)])
def test_coach_posture(name, rounding, neck_f, neck_3, pitch):
    r = read_pose(REF / name)
    flat = r["flat"]
    assert abs(flat["back_rounding"]) < 0.04                  # the coach's back is straight
    assert flat["back_rounding"] == approx(rounding, abs=0.01)
    assert 0.15 <= flat["back_seen"] <= 0.5                   # lower back only: the arms cover the rest
    lo, hi = flat["back_span"]                                # the unbroken stretch read: from 10% up
    assert lo == approx(0.1) and hi - lo <= 0.8 * flat["back_seen"] + 0.01
    assert flat["back_why"] is None                           # read, but far too little to classify
    assert flat["neck_vs_torso"] == approx(neck_f, abs=2.0)
    assert r["3d"]["neck_vs_torso"] == approx(neck_3, abs=3.0)
    assert abs(flat["neck_vs_torso"] - r["3d"]["neck_vs_torso"]) <= 20     # the two readings agree
    assert flat["head_pitch"] == approx(pitch, abs=2.0)


def test_the_coachs_seat_below_his_knee_top():
    """Task 22: the bottom's depth as a judge sees it - his seat's lowest point 0.646 thigh lengths below the top of
    the knee that shows, on MediaPipe's person mask by his camera-side hip and knee (posture.seat_depth). Off a side
    view: none."""
    r = read_pose(REF / "17.png")
    assert r["flat"]["seat_below_knee"] == approx(0.646, abs=0.005)
    assert "thigh_outline" not in r["flat"]
    assert read_pose(REF / "behind-thumb.png")["flat"]["seat_below_knee"] is None


def test_no_posture_off_a_side_view():
    r = read_pose(REF / "behind-thumb.png")
    for k in ("back_rounding", "back_seen", "back_span", "back_why", "neck_vs_torso", "head_pitch"):
        assert r["flat"][k] is None, k
    assert r["3d"]["neck_vs_torso"] is None


def test_the_angles_are_the_same_with_the_silhouette_switched_on():
    # the detector now also returns the person's mask; the landmarks it gives must not move
    r = read_pose(REF / "17.png")
    assert r["flat"]["knee_included"] == approx(47.2, abs=1.5)
    assert r["flat"]["torso_lean"] == approx(22.4, abs=1.0)


def test_bottom_is_parallel():
    assert read_pose(REF / "17.png")["flat"]["hip_below_knee"] == approx(-0.05, abs=0.05)


def test_behind_view_widths():
    r = read_pose(REF / "behind-thumb.png")
    assert r["view"] == "frontal"
    assert r["flat"]["grip_ratio"] == approx(1.87, abs=0.05)
    assert r["flat"]["elbow_width_ratio"] == approx(1.97, abs=0.05)
    assert r["3d"]["grip_ratio"] == approx(1.64, abs=0.08)


def test_no_body(tmp_path):
    Image.new("RGB", (400, 400), "white").save(tmp_path / "blank.png")
    with pytest.raises(NoBody):
        read_pose(tmp_path / "blank.png")


# ---- crop: the body's box, so the page can show the coach instead of the whole phone screen ----

# The coach in 15.png, measured by eye on the picture (fractions of 1260 x 2736): back of the elbow
# to the face and toes, top of the hair to the soles.
COACH_15 = (0.345, 0.44, 0.54, 0.75)


def test_crop_contains_the_coach_and_is_much_smaller_than_the_screenshot():
    x0, y0, x1, y1 = read_pose(REF / "15.png")["crop"]
    cx0, cy0, cx1, cy1 = COACH_15
    assert x0 <= cx0 and y0 <= cy0 and x1 >= cx1 and y1 >= cy1
    assert (x1 - x0) * (y1 - y0) < 0.2          # the screenshot is 1.0
    assert 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1


@pytest.mark.parametrize("name,crown", [("15.png", 0.44), ("16.png", 0.475), ("17.png", 0.535)])
def test_crop_keeps_the_top_of_the_head(name, crown):
    # The crown (top of the hair, measured by eye) is not a landmark; the box must still clear it.
    assert read_pose(REF / name)["crop"][1] < crown - 0.01


def test_body_crop_adds_12_percent_of_the_longer_side_and_clamps():
    from read_pose import body_crop
    # box 200 x 400 px -> 48 px each side
    assert body_crop([(100, 200), (300, 600)], (1000, 1000)) == approx([0.052, 0.152, 0.348, 0.648])
    # reaching past the picture: clamped to it
    assert body_crop([(0, 0), (1000, 500)], (1000, 1000)) == approx([0.0, 0.0, 1.0, 0.62])
    # a body wholly outside the picture, or no points at all: nothing to crop to
    assert body_crop([(1200, 100), (1500, 400)], (1000, 1000)) is None
    assert body_crop([], (1000, 1000)) is None


def test_run_read_carries_the_crop_into_measurements(tmp_path):
    import json
    from run_read import main
    Image.new("RGB", (400, 400), "white").save(tmp_path / "blank.png")
    images = [
        {"id": "img-15", "path": str(REF / "15.png"), "original": "15.png", "order": 1, "phase": "start", "contextOnly": False},
        {"id": "img-blank", "path": str(tmp_path / "blank.png"), "original": "blank.png", "order": 2, "phase": "end", "contextOnly": False},
    ]
    (tmp_path / "references.json").write_text(json.dumps({"exercise": "Barbell Back Squat", "images": images}))
    main(tmp_path)
    m = json.loads((tmp_path / "measurements.json").read_text())
    assert m["img-15"]["crop"] == approx(read_pose(REF / "15.png")["crop"])
    assert m["img-blank"]["crop"] is None


def test_a_posture_failure_costs_only_the_posture(monkeypatch, capsys):
    import read_pose as rp

    def broken(*args, **kwargs):
        raise RuntimeError("mask went wrong")
    monkeypatch.setattr(rp, "back_profile", broken)
    r = rp.read_pose(REF / "17.png")
    assert r["flat"]["knee_included"] == approx(47.2, abs=1.5) and r["crop"] is not None
    for k in ("back_rounding", "back_seen", "back_span", "back_why", "neck_vs_torso", "head_pitch"):
        assert r["flat"][k] is None, k
    assert r["3d"]["neck_vs_torso"] is None
    assert "posture not measured" in capsys.readouterr().err


# ---- any picture PIL reads: WebP, grey + alpha PNG, and EXIF-rotated phone photos (final fix 3) ----
# 15.png as read before the loader went through PIL (mp.Image.create_from_file on the RGBA PNG).
BEFORE_15 = {"flat": {"knee_included": 174.668, "hip_included": 170.821, "torso_lean": 6.685,
                      "neck_vs_torso": 38.618, "head_pitch": -36.225},
             "3d": {"knee_included": 160.415, "hip_included": 167.124, "elbow_included": 90.531,
                    "upperarm_vs_torso": 41.275, "neck_vs_torso": 52.132},
             "crop": [0.2895, 0.3998, 0.6404, 0.7847]}


def assert_like_before_15(r):
    assert (r["view"], r["camera_side"], r["facing"], r["size"]) == ("side", "right", "right", [1260, 2736])
    for method, values in (("flat", BEFORE_15["flat"]), ("3d", BEFORE_15["3d"])):
        for k, v in values.items():
            assert r[method][k] == approx(v, abs=0.5), (method, k)
    assert r["crop"] == approx(BEFORE_15["crop"], abs=0.002)


def test_15_reads_as_before_through_pil():
    assert_like_before_15(read_pose(REF / "15.png"))


def test_a_webp_reads_like_the_png(tmp_path):
    Image.open(REF / "15.png").save(tmp_path / "15.webp", lossless=True)
    assert_like_before_15(read_pose(tmp_path / "15.webp"))


@pytest.mark.parametrize("mode", ["LA", "L"])   # grey + alpha, and plain grey: both crashed the old loader
def test_a_grey_png_is_read(tmp_path, mode):
    Image.open(REF / "15.png").convert(mode).save(tmp_path / "15-grey.png")
    r = read_pose(tmp_path / "15-grey.png")
    assert (r["view"], r["size"]) == ("side", [1260, 2736])
    assert r["flat"]["knee_included"] == approx(174.7, abs=3)


def test_an_exif_rotated_photo_is_read_upright_and_its_crop_lines_up(tmp_path):
    # stored on its side, with the EXIF tag (6: turn 90° clockwise) a phone writes: shown upright
    im = Image.open(REF / "15.png").transpose(Image.Transpose.ROTATE_90)
    exif = Image.Exif()
    exif[0x0112] = 6
    im.save(tmp_path / "rotated.png", exif=exif)
    assert Image.open(tmp_path / "rotated.png").size == (2736, 1260)
    assert_like_before_15(read_pose(tmp_path / "rotated.png"))


# ---- visibility (task 19): how surely MediaPipe sees each joint's landmark, for picture.py's gate ----

@pytest.mark.parametrize("name", ["15.png", "16.png", "17.png"])
def test_the_coach_s_shoulders_and_hips_are_seen_clearly(name):
    vis = read_pose(REF / name)["visibility"]
    assert set(vis) == {f"{s}_{j}" for s in ("left", "right") for j in
                        ("shoulder", "elbow", "wrist", "hip", "knee", "ankle")}
    assert min(vis[f"{s}_{j}"] for s in ("left", "right") for j in ("shoulder", "hip")) >= 0.99
    assert all(0.0 <= v <= 1.0 for v in vis.values())


# ---- the picture's own landmarks and the key positions (task 9: motion from the pictures) ----

def test_the_picture_s_own_landmarks_give_the_flat_readings():
    import numpy as np
    from metrics import measure
    from read_pose import JOINTS, NAMES, UP, VIEW_NORMAL
    r = read_pose(REF / "16.png")
    assert set(r["pixels"]) == set(NAMES)
    w, h = r["size"]
    assert all(-w <= x <= 2 * w and -h <= y <= 2 * h for x, y in r["pixels"].values())
    joints = {f"{j}.{s}": np.array([*r["pixels"][f"{s}_{j}"], 0.0]) for j in JOINTS for s in ("left", "right")}
    flat = measure(joints, r["camera_side"], UP, VIEW_NORMAL)
    for k in ("knee_included", "hip_included", "torso_lean"):
        assert flat[k] == approx(r["flat"][k], abs=0.05)


def _squat_refs(tmp_path, names):
    import json
    images = [{"id": f"img-{n}", "path": str(REF / f"{n}.png"), "original": f"{n}.png", "order": i + 1,
               "phase": phase, "contextOnly": context}
              for i, (n, phase, context) in enumerate(names)]
    (tmp_path / "references.json").write_text(json.dumps({"exercise": "Back Squat from images", "images": images}))
    return images


def test_run_read_writes_the_key_positions_the_targets_were_read_from(tmp_path):
    import json
    from run_read import main
    _squat_refs(tmp_path, [("14", "start", True), ("15", "start", False), ("16", "middle", False),
                           ("17", "end", False), ("behind-thumb", "start", False)])
    assert main(tmp_path) == 0
    kp = json.loads((tmp_path / "keyposes.json").read_text())
    m = json.loads((tmp_path / "measurements.json").read_text())
    assert kp["exercise"] == "Back Squat from images"
    assert [(p["phase"], p["image"]) for p in kp["phases"]] == [("start", "img-15"), ("middle", "img-16"),
                                                                ("end", "img-17")]
    for p in kp["phases"]:
        assert (p["view"], p["facing"], p["camera_side"]) == ("side", "right", "right")
        assert p["world"] == m[p["image"]]["world"]
        assert p["pixels"] == m[p["image"]]["pixels"]
    # a clip-less exercise: no solver targets, only the key positions
    assert not (tmp_path / "pipeline-targets.json").exists()


def test_without_a_middle_picture_the_key_positions_are_start_and_end(tmp_path):
    import json
    from run_read import main
    _squat_refs(tmp_path, [("15", "start", False), ("17", "end", False)])
    assert main(tmp_path) == 0
    kp = json.loads((tmp_path / "keyposes.json").read_text())
    assert [p["phase"] for p in kp["phases"]] == ["start", "end"]


def test_key_positions_skip_a_picture_read_before_landmarks_were_kept():
    from run_read import key_positions
    targets = {"exercise": "X", "phases": {
        "start": {"knee_included": {"value": 170, "image": "a"}},
        "middle": {"knee_included": {"value": 120, "image": None}, "estimated": True},
        "end": {"knee_included": {"value": 90, "image": "b"}}}}
    side = {"view": "side", "facing": "right", "camera_side": "right", "world": {"nose": [0, 0, 0]}}
    assert [p["phase"] for p in key_positions(targets, {"a": side, "b": side})["phases"]] == ["start", "end"]
    assert key_positions(targets, {"a": side, "b": {**side, "world": None}}) is None     # no end: no motion


def test_a_picture_set_to_side_is_a_side_key_position_whatever_the_detector_said():
    # the user tagged it Side; the detector called it three-quarter and gave no facing (task 9 review)
    from run_read import key_positions
    targets = {"exercise": "X", "images": {"a": {"view": "side"}, "b": {"view": "side"}}, "phases": {
        "start": {"knee_included": {"value": 170, "image": "a"}},
        "end": {"knee_included": {"value": 90, "image": "b"}}}}
    seen = {"view": "three-quarter", "facing": None, "camera_side": "right", "world": {"nose": [0, 0, 0]}}
    kp = key_positions(targets, {"a": seen, "b": {**seen, "view": "side", "facing": "right"}})
    assert [(p["view"], p["facing"]) for p in kp["phases"]] == [("side", None), ("side", "right")]


def test_targets_carry_the_exercise_s_equipment_word(tmp_path):
    import json
    from run_read import main
    _squat_refs(tmp_path, [("15", "start", False), ("17", "end", False)])
    snap = json.loads((tmp_path / "references.json").read_text())
    (tmp_path / "references.json").write_text(json.dumps({**snap, "equipment": "barbell"}))
    assert main(tmp_path) == 0
    assert json.loads((tmp_path / "targets.json").read_text())["equipment"] == "barbell"
