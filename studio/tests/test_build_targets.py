import json
import subprocess
import sys

import pytest
from pytest import approx

from conftest import ROOT, REF  # noqa: F401
from build_targets import build_targets, to_pipeline_targets, TooFewPositions
from read_pose import NoBody

RAW = json.loads((ROOT / "studio/tests/fixtures/squat-measurements.json").read_text())
FILES = {"img-14": "14.png", "img-15": "15.png", "img-16": "16.png", "img-17": "17.png",
         "img-bt": "behind-thumb.png"}


def meta(i, phase, order, **kw):
    m = {"id": i, "file": i + ".png", "original": FILES[i], "order": order, "phase": phase,
         "phaseSetBy": "user", "view": None, "viewSetBy": None, "contextOnly": False, "deleted": False}
    m.update(kw)
    return m


IMAGES = [
    meta("img-14", "start", 1, contextOnly=True),
    meta("img-15", "start", 2),
    meta("img-16", "middle", 3),
    meta("img-17", "end", 4),
    meta("img-bt", "start", 5, view="behind", viewSetBy="user"),
]
MEASURED = {i: RAW[f] for i, f in FILES.items()}


def subset(*ids):
    return [m for m in IMAGES if m["id"] in ids], {i: MEASURED[i] for i in ids}


def test_squat_targets():
    t = build_targets("Barbell Back Squat", IMAGES, MEASURED)
    knee = t["phases"]["start"]["knee_included"]
    assert (knee["method"], knee["uncertain"]) == ("flat", True)
    assert knee["value"] == approx(174.7, abs=1.5) and knee["cross"] == approx(160.4, abs=1.5)
    assert t["phases"]["start"]["hip_included"]["uncertain"] is False
    assert t["phases"]["middle"]["hip_included"]["uncertain"] is True
    assert t["phases"]["end"]["hip_included"]["uncertain"] is False
    elbow = t["phases"]["end"]["elbow_included"]
    assert (elbow["method"], elbow["cross"], elbow["uncertain"]) == ("3d", None, False)
    assert t["phases"]["end"]["hip_below_knee"]["value"] == approx(-0.05, abs=0.05)
    grip = t["widths"]["grip_ratio"]
    assert grip["value"] == approx(1.87, abs=0.05) and grip["uncertain"] is False
    hbk = t["phases"]["end"]["hip_below_knee"]
    assert (hbk["method"], hbk["cross"], hbk["uncertain"]) == ("flat", None, False)
    ew = t["widths"]["elbow_width_ratio"]
    assert ew["value"] == approx(1.97, abs=0.05) and ew["cross"] == approx(1.76, abs=0.05)
    assert ew["uncertain"] is False
    assert t["widths"]["grip_ratio"]["cross"] == approx(1.64, abs=0.05)
    assert t["images"]["img-14"] == {"used": False, "why": "Context only", "view": "side"}
    assert t["camera"] == {"side": "right", "facing": "right"}
    assert t["tolerance"] == {"angles_deg": 5, "ratios": 0.12, "hip_below_knee": 0.10, "seat_below_knee": 0.10,
                              "back_rounding": 0.04, "neck_vs_torso": 10, "spine_curl": 10}
    assert "estimated" not in t["phases"]["middle"]
    assert t["phases"]["start"]["torso_lean"]["method"] == "flat"


def test_two_positions_estimate_the_middle():
    imgs, meas = subset("img-15", "img-17")
    t = build_targets("Barbell Back Squat", imgs, meas)
    mid = t["phases"]["middle"]
    assert mid["estimated"] is True
    s, e = t["phases"]["start"]["knee_included"]["value"], t["phases"]["end"]["knee_included"]["value"]
    assert mid["knee_included"]["value"] == approx((s + e) / 2)


def test_one_position_is_too_few():
    imgs, meas = subset("img-15")
    with pytest.raises(TooFewPositions, match="missing: end"):
        build_targets("Barbell Back Squat", imgs, meas)


def test_a_bad_image_fails_only_itself():
    meas = dict(MEASURED)
    meas["img-16"] = NoBody("no person")
    t = build_targets("Barbell Back Squat", IMAGES, meas)
    assert t["images"]["img-16"]["why"] == "No body found in this image"
    assert t["images"]["img-16"]["used"] is False
    assert t["phases"]["middle"]["estimated"] is True


def test_duplicate_phase_first_wins():
    imgs = [m for m in IMAGES if m["id"] != "img-16"] + [meta("img-16", "start", 9)]
    t = build_targets("Barbell Back Squat", imgs, MEASURED)
    assert t["images"]["img-16"]["why"] == "Another image already gives the start position"


def test_pipeline_format():
    t = build_targets("Barbell Back Squat", IMAGES, MEASURED)
    p = to_pipeline_targets(t)
    assert p["phases"]["bottom"]["knee_included"] == 47
    assert p["phases"]["standing"]["knee_included"] == 175
    assert p["arms_all_phases"]["grip_ratio"] == 1.87
    assert p["tolerance"] == {"angles_deg": 5, "ratios": 0.12}
    # the side the Studio judges one-sided rows (the upper arm, the elbow) on: its side camera's (task 18)
    assert p["camera_side"] == t["camera"]["side"] in ("left", "right")
    imgs, meas = subset("img-15", "img-16", "img-17")
    t2 = build_targets("Barbell Back Squat", imgs, meas)
    assert t2["widths"] == {}
    assert to_pipeline_targets(t2) is None


def test_pipeline_targets_carry_the_measure_map():
    p = to_pipeline_targets(build_targets("Barbell Back Squat", IMAGES, MEASURED))
    assert p["measure"] == {"knee_included": "flat", "hip_included": "flat", "torso_lean": "flat",
                            "elbow_included": "3d", "upperarm_vs_torso": "3d",
                            "neck_vs_torso": "flat", "head_pitch": "flat"}


def test_mixed_methods_across_phases_raise():
    t = build_targets("Barbell Back Squat", IMAGES, MEASURED)
    t["phases"]["end"]["knee_included"] = {**t["phases"]["end"]["knee_included"], "method": "3d"}
    with pytest.raises(ValueError, match="knee_included"):
        to_pipeline_targets(t)


def run_cli(run_dir):
    return subprocess.run([sys.executable, str(ROOT / "studio/measure/run_read.py"), str(run_dir)],
                          capture_output=True, text=True)


def snapshot(tmp_path, ids):
    imgs = []
    for m in IMAGES:
        if m["id"] in ids:
            imgs.append({**m, "path": str(REF / FILES[m["id"]])})
    (tmp_path / "references.json").write_text(json.dumps(
        {"exercise": "Barbell Back Squat", "slug": "barbell-back-squat", "images": imgs}))


@pytest.mark.skipif(not (REF / "15.png").exists(), reason="reference images missing")
def test_run_read_cli(tmp_path):
    snapshot(tmp_path, {"img-15", "img-17", "img-bt"})
    r = run_cli(tmp_path)
    assert r.returncode == 0, r.stderr
    assert "image 1/3" in r.stdout
    for f in ("measurements.json", "targets.json", "pipeline-targets.json"):
        assert (tmp_path / f).exists()
    pt = json.loads((tmp_path / "pipeline-targets.json").read_text())
    assert pt["phases"]["bottom"]["knee_included"] == 47

    bad = tmp_path / "bad"
    bad.mkdir()
    snapshot(bad, {"img-15", "img-17", "img-bt"})
    snap = json.loads((bad / "references.json").read_text())
    snap["images"][2]["path"] = str(bad / "missing.png")
    (bad / "references.json").write_text(json.dumps(snap))
    assert run_cli(bad).returncode == 0
    m = json.loads((bad / "measurements.json").read_text())
    assert "error" in m["img-bt"]
    assert not (bad / "pipeline-targets.json").exists()

    one = tmp_path / "one"
    one.mkdir()
    snapshot(one, {"img-15"})
    r = run_cli(one)
    assert r.returncode == 3
    assert "missing: end" in r.stderr.strip().splitlines()[-1]


def test_start_and_middle_miss_the_end():
    imgs, meas = subset("img-15", "img-16")
    with pytest.raises(TooFewPositions) as e:
        build_targets("Barbell Back Squat", imgs, meas)
    assert str(e.value) == ("Need a start and an end position from side-view images with a body found; "
                            "missing: end.")


def test_middle_and_end_miss_the_start():
    imgs, meas = subset("img-16", "img-17")
    with pytest.raises(TooFewPositions, match="missing: start."):
        build_targets("Barbell Back Squat", imgs, meas)


def test_unexpected_read_failure_is_logged(tmp_path, monkeypatch, capsys):
    import run_read
    snapshot(tmp_path, {"img-15", "img-17"})

    def boom(path):
        raise RuntimeError("boom")
    monkeypatch.setattr(run_read, "read_pose", boom)
    assert run_read.main(tmp_path) == 3
    err = capsys.readouterr().err
    assert "boom" in err and "15.png" in err
    m = json.loads((tmp_path / "measurements.json").read_text())
    assert m["img-15"] == {"error": "Could not read this image"}


# ---- posture (Task 6c): the coach's own posture in each side view is the target ----

@pytest.mark.parametrize("phase,image", [("start", "img-15"), ("middle", "img-16"), ("end", "img-17")])
def test_posture_targets_from_each_side_view(phase, image):
    t = build_targets("Barbell Back Squat", IMAGES, MEASURED)
    p, raw = t["phases"][phase], MEASURED[image]
    back = p["back_rounding"]
    assert (back["method"], back["cross"], back["uncertain"], back["image"]) == ("flat", None, False, image)
    assert back["value"] == raw["flat"]["back_rounding"] and back["seen"] == raw["flat"]["back_seen"]
    assert back["span"] == raw["flat"]["back_span"]
    # the squat photos show a quarter of the back: the coach's back is classified as unreadable
    assert (back["state"], back["cause"], back["curl"]) == ("unreadable", "hidden", None)
    assert back["run"] == approx((back["span"][1] - back["span"][0]) / 0.8)
    neck = p["neck_vs_torso"]
    assert (neck["method"], neck["uncertain"]) == ("flat", False)
    assert neck["value"] == raw["flat"]["neck_vs_torso"] and neck["cross"] == raw["3d"]["neck_vs_torso"]
    pitch = p["head_pitch"]
    assert (pitch["method"], pitch["cross"], pitch["value"]) == ("flat", None, raw["flat"]["head_pitch"])


def test_the_neck_is_uncertain_when_flat_and_3d_differ_by_more_than_twice_its_tolerance():
    meas = json.loads(json.dumps(MEASURED))
    meas["img-17"]["3d"]["neck_vs_torso"] = meas["img-17"]["flat"]["neck_vs_torso"] + 20.5
    meas["img-16"]["3d"]["neck_vs_torso"] = meas["img-16"]["flat"]["neck_vs_torso"] - 19.5
    t = build_targets("Barbell Back Squat", IMAGES, meas)
    assert t["phases"]["end"]["neck_vs_torso"]["uncertain"] is True
    assert t["phases"]["middle"]["neck_vs_torso"]["uncertain"] is False


def test_a_posture_the_coach_picture_cannot_show_is_kept_as_not_measured():
    # e.g. the arms hide the whole back: the reading is there with no value, so the check can say so
    meas = json.loads(json.dumps(MEASURED))
    meas["img-17"]["flat"]["back_rounding"] = None
    t = build_targets("Barbell Back Squat", IMAGES, meas)
    assert t["phases"]["end"]["back_rounding"]["value"] is None
    assert t["phases"]["end"]["back_rounding"]["image"] == "img-17"
    # measurements from before posture was read have no keys at all: no posture targets, nothing breaks
    old = json.loads(json.dumps(MEASURED))
    for r in old.values():
        for k in ("back_rounding", "back_seen", "neck_vs_torso", "head_pitch"):
            r["flat"].pop(k, None)
        r["3d"].pop("neck_vs_torso", None)
    t = build_targets("Barbell Back Squat", IMAGES, old)
    assert not {"back_rounding", "neck_vs_torso", "head_pitch"} & set(t["phases"]["end"])


def test_an_estimated_middle_carries_the_posture_halfway_and_none_when_a_side_has_none():
    imgs, meas = subset("img-15", "img-17")
    t = build_targets("Barbell Back Squat", imgs, meas)
    s, e = (t["phases"][p]["neck_vs_torso"] for p in ("start", "end"))
    assert t["phases"]["middle"]["neck_vs_torso"]["value"] == approx((s["value"] + e["value"]) / 2)
    meas = json.loads(json.dumps(meas))
    meas["img-17"]["flat"]["back_rounding"] = None
    t = build_targets("Barbell Back Squat", imgs, meas)
    assert t["phases"]["middle"]["back_rounding"]["value"] is None and t["phases"]["middle"]["estimated"] is True


def test_the_coach_neck_and_face_reach_the_solver_per_phase():
    # task 18: the solver aims the head at the coach's neck angle and face direction; the Studio writes them
    # into its own file (they used to be read from targets.json beside it). The back is never aimed at: ours
    # is held straight by the solver and judged from the rig.
    t = build_targets("Barbell Back Squat", IMAGES, MEASURED)
    p = to_pipeline_targets(t)
    names = {"start": "standing", "middle": "mid", "end": "bottom"}
    for src, dst in names.items():
        ph = p["phases"][dst]
        assert set(ph) == {"knee_included", "hip_included", "torso_lean", "elbow_included", "upperarm_vs_torso",
                           "neck_vs_torso", "head_pitch"}
        for k in ("neck_vs_torso", "head_pitch"):
            assert ph[k] == round(t["phases"][src][k]["value"], 1)
    assert p["phases"]["standing"]["neck_vs_torso"] == approx(38.6, abs=0.05)


def test_no_posture_for_the_solver_when_a_position_lacks_it():
    # the coach's picture can't show the neck at one position (or the reading predates posture): the solver
    # gets no neck or face at all and keeps its neutral nod, rather than a head aimed at two positions of three
    t = build_targets("Barbell Back Squat", IMAGES, MEASURED)
    t["phases"]["end"]["neck_vs_torso"]["value"] = None
    p = to_pipeline_targets(t)
    assert not {"neck_vs_torso", "head_pitch"} & {k for ph in p["phases"].values() for k in ph}
    assert not {"neck_vs_torso", "head_pitch"} & set(p["measure"])
    t = build_targets("Barbell Back Squat", IMAGES, MEASURED)
    del t["phases"]["middle"]["head_pitch"]
    p = to_pipeline_targets(t)
    assert not {"neck_vs_torso", "head_pitch"} & {k for ph in p["phases"].values() for k in ph}
    assert "back_rounding" not in {k for ph in p["phases"].values() for k in ph}


def test_the_coach_back_is_classified_from_a_clear_reading():
    from posture import neutral_rounding
    meas = json.loads(json.dumps(MEASURED))
    f = meas["img-17"]["flat"]
    f.update(back_seen=1.0, back_span=[0.1, 0.9], back_rounding=neutral_rounding((0.1, 0.9)) + 0.09, back_why=None)
    meas["img-15"]["flat"].update(back_seen=1.0, back_span=[0.1, 0.9], back_rounding=neutral_rounding((0.1, 0.9)))
    meas["img-16"]["flat"].update(back_rounding=None, back_seen=0.0, back_span=None, back_why="level")
    t = build_targets("Barbell Back Squat", IMAGES, meas)
    end, start, middle = (t["phases"][p]["back_rounding"] for p in ("end", "start", "middle"))
    assert end["state"] == "rounded" and end["curl"] > 30
    assert start["state"] == "straight" and start["curl"] is None
    assert (middle["state"], middle["cause"], middle["value"]) == ("unreadable", "level", None)


# ---- a failed read names each picture it couldn't use, and why (final fix 2) ----

def test_too_few_positions_names_each_skipped_picture_and_why():
    meas = json.loads(json.dumps(MEASURED))
    meas["img-17"] = NoBody("no person")
    meas["img-16"]["view"] = "frontal"
    with pytest.raises(TooFewPositions) as e:
        build_targets("Barbell Back Squat", IMAGES, meas)
    assert str(e.value) == ("Need a start and an end position from side-view images with a body found; missing: end. "
                            "16.png was read as a front or behind view — set it to Side if it isn't; "
                            "17.png: no body found.")
    # the per-picture reasons the strip shows ride along with it
    assert e.value.images["img-17"] == {"used": False, "why": "No body found in this image", "view": None}
    assert e.value.images["img-14"]["why"] == "Context only"


def test_the_message_says_how_to_fix_a_misread_or_doubled_picture():
    meas = json.loads(json.dumps(MEASURED))
    meas["img-17"]["view"] = "three-quarter"
    imgs = [m for m in IMAGES if m["id"] != "img-16"] + [meta("img-16", "start", 3)]
    with pytest.raises(TooFewPositions) as e:
        build_targets("Barbell Back Squat", imgs, meas)
    msg = str(e.value)
    assert "16.png is tagged Start, but another picture already gives that position" in msg
    assert "17.png was read as a three-quarter view — set it to Side if it isn't" in msg
    # pictures the user set that way on purpose (context only, a behind view for the widths) aren't blamed
    assert "14.png" not in msg and "behind-thumb.png" not in msg


def test_an_unreadable_picture_is_named():
    imgs, meas = subset("img-15", "img-17")
    meas = {**meas, "img-17": RuntimeError("broken file")}
    with pytest.raises(TooFewPositions, match=r"missing: end\. 17\.png: could not be read\.$"):
        build_targets("Barbell Back Squat", imgs, meas)


def test_a_failed_read_writes_this_runs_own_reasons(tmp_path, monkeypatch, capsys):
    import run_read
    snapshot(tmp_path, {"img-15", "img-17"})

    def fake(path):
        if path.name == "17.png":
            raise NoBody("no person found in 17.png")
        return MEASURED["img-15"]
    monkeypatch.setattr(run_read, "read_pose", fake)
    assert run_read.main(tmp_path) == 3
    last = capsys.readouterr().err.strip().splitlines()[-1]
    assert last == ("Need a start and an end position from side-view images with a body found; missing: end. "
                    "17.png: no body found.")
    problems = json.loads((tmp_path / "read-problems.json").read_text())
    assert problems["message"] == last
    assert problems["images"]["img-17"] == {"used": False, "why": "No body found in this image", "view": None}
    assert problems["images"]["img-15"]["used"] is True
    assert not (tmp_path / "targets.json").exists()


def test_a_good_read_writes_no_problems_file(tmp_path, monkeypatch):
    import run_read
    snapshot(tmp_path, {"img-15", "img-17"})
    monkeypatch.setattr(run_read, "read_pose", lambda path: MEASURED["img-15" if path.name == "15.png" else "img-17"])
    assert run_read.main(tmp_path) == 0
    assert (tmp_path / "targets.json").exists() and not (tmp_path / "read-problems.json").exists()


def test_the_bottoms_seat_depth_is_a_target_at_the_end_only():
    """Task 22: the depth as a judge sees it - the coach's seat against the top of his knee at the bottom (read_pose's
    seat_below_knee, his 17.png: 0.646 thigh lengths), judged at the end position only, within 0.10 thigh lengths."""
    t = build_targets("Barbell Back Squat", IMAGES, MEASURED)
    seat = t["phases"]["end"]["seat_below_knee"]
    assert seat["value"] == approx(0.646, abs=0.005)
    assert (seat["method"], seat["cross"], seat["uncertain"], seat["image"]) == ("flat", None, False, "img-17")
    assert "seat_below_knee" not in t["phases"]["start"] and "seat_below_knee" not in t["phases"]["middle"]
    assert t["tolerance"]["seat_below_knee"] == 0.10
    assert "thigh_outline" not in t["phases"]["end"]


def test_the_depth_is_read_only_for_an_exercise_whose_kit_row_says_so():
    """The deadlift's or the press's end position has no depth (exercise-kit.json `depth`: the back squat only), nor
    an exercise added in the Studio, whatever its equipment word."""
    for name, equipment in (("Conventional Barbell Deadlift", None), ("My Own Squat", "barbell")):
        end = build_targets(name, IMAGES, MEASURED, equipment)["phases"]["end"]
        assert "seat_below_knee" not in end and "hip_below_knee" not in end, (name, end.keys())
        assert "knee_included" in end


def test_a_coach_picture_that_does_not_show_the_seat_keeps_the_row_without_a_value():
    """His picture read but no seat or knee top found: the target is kept with no value (the check says the coach's
    picture doesn't show it); measured before task 22 (no key at all): no target (the check says to run it again)."""
    import copy
    blind = copy.deepcopy(MEASURED)
    blind["img-17"]["flat"]["seat_below_knee"] = None
    seat = build_targets("Barbell Back Squat", IMAGES, blind)["phases"]["end"]["seat_below_knee"]
    assert seat["value"] is None and seat["image"] == "img-17", seat
    old = copy.deepcopy(MEASURED)
    del old["img-17"]["flat"]["seat_below_knee"]
    assert "seat_below_knee" not in build_targets("Barbell Back Squat", IMAGES, old)["phases"]["end"]