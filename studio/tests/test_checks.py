import copy
import json
import subprocess
import sys

import pytest
from pytest import approx

from conftest import ROOT
from checks import STATUSES, build_checks
from test_build_targets import IMAGES, MEASURED
from build_targets import build_targets

FIX = ROOT / "studio/tests/fixtures"
TARGETS = build_targets("Barbell Back Squat", IMAGES, MEASURED)
OURS = json.loads((FIX / "ours-squat.json").read_text())
REPORT = json.loads((FIX / "retarget-squat.json").read_text())
EX = "Barbell Back Squat"


def row(c, g, m):
    return next(r for r in c["rows"] if (r["group"], r["metric"]) == (g, m))


def with_grip(v):
    o = copy.deepcopy(OURS)
    for f in o["per_frame"]:
        if "grip_ratio" in f["flat"]:
            f["flat"]["grip_ratio"] = v
    return o


def test_statuses():
    c = build_checks(TARGETS, OURS, REPORT, EX)
    assert row(c, "start", "knee_included")["status"] == "uncertain"
    assert row(c, "start", "knee_included")["note"] == "Two readings of the coach disagree: 175° flat, 160° in 3D"
    assert row(c, "end", "hip_included")["status"] == "pass"            # ours 72.0 vs 69.3
    assert row(c, "end", "torso_lean")["status"] == "fail"              # ours 30.0 vs 22.4
    g = row(c, "all positions", "grip_ratio")   # ours 1.70; a human arm also misses with these targets
    assert g["status"] == "uncertain" and g["uncertain_why"] == "coach"   # the coach's numbers don't fit one arm
    assert "don't all fit together" in g["note"] and "×" in g["note"]
    assert row(c, "body", "feet_in_floor")["status"] == "fail"          # 28.91 mm in the fixture
    assert row(c, "body", "hands_on_bar")["status"] == "pass"           # 1.92 mm, 61 of 69 frames
    assert row(c, "body", "skin_stretch")["status"] == "info"
    assert sum(c["summary"][s] for s in STATUSES) == len(c["rows"])
    assert c["summary"]["fail"] >= 2


def test_row_shape():
    c = build_checks(TARGETS, OURS, REPORT, EX)
    r = row(c, "end", "torso_lean")
    assert (r["unit"], r["frame"]) == ("deg", 38)
    assert r["ours"] == 30.0 and abs(r["target"] - 22.383) < 1e-6
    assert row(c, "end", "hip_below_knee")["unit"] == "torso"
    assert row(c, "all positions", "grip_ratio")["unit"] == "x"
    assert row(c, "body", "feet_in_floor")["unit"] == "mm"
    assert row(c, "body", "skin_stretch")["unit"] == "%"


def test_grip_far_below_reach_is_a_fail():
    c = build_checks(TARGETS, with_grip(1.40), REPORT, EX)
    assert row(c, "all positions", "grip_ratio")["status"] == "fail"


def test_limited_when_a_human_arm_meets_it_and_the_cartoon_cannot():
    from checks import human_arm_like, _want
    from feasibility import closest_arm_pose
    human = {"shoulder_half_width": 0.19, "upper_arm": 0.30, "forearm": 0.27}
    hw = closest_arm_pose(human, _want(TARGETS, "end"))   # a human meets this by construction
    t, o = copy.deepcopy(TARGETS), copy.deepcopy(OURS)
    for p in t["phases"].values():
        p["elbow_included"]["value"], p["upperarm_vs_torso"]["value"] = hw["elbow_included"], hw["upperarm_vs_torso"]
    for m in ("grip_ratio", "elbow_width_ratio"):
        t["widths"][m]["value"] = hw[m]
    char = closest_arm_pose(o["arm"], hw)
    humanlike = closest_arm_pose(human_arm_like(o["arm"]), hw)
    tol = {"grip_ratio": 0.12, "elbow_width_ratio": 0.12, "elbow_included": 5, "upperarm_vs_torso": 5}
    for f in o["per_frame"]:
        f["flat"].update(grip_ratio=char["grip_ratio"], elbow_width_ratio=char["elbow_width_ratio"])
        f["3d"].update(elbow_included=char["elbow_included"], upperarm_vs_torso=char["upperarm_vs_torso"])
    c = build_checks(t, o, REPORT, EX)
    rows = {"grip_ratio": row(c, "all positions", "grip_ratio"), "elbow_included": row(c, "end", "elbow_included")}
    for m, r in rows.items():
        cartoon_misses = abs(char[m] - hw[m]) > tol[m]
        human_meets = abs(humanlike[m] - hw[m]) <= tol[m]
        assert r["status"] == ("limited" if cartoon_misses and human_meets else "pass"), (m, r)
    assert rows["grip_ratio"]["status"] == "limited"
    assert "human-proportioned arm reaches" in rows["grip_ratio"]["note"]


def test_other_exercise_never_gets_limited():
    c = build_checks(TARGETS, OURS, REPORT, "Barbell Deadlift")
    assert row(c, "all positions", "grip_ratio")["status"] == "fail"


def test_estimated_middle_is_info():
    t = copy.deepcopy(TARGETS)
    t["phases"]["middle"]["estimated"] = True
    c = build_checks(t, OURS, REPORT, EX)
    mids = [r for r in c["rows"] if r["group"] == "middle"]
    assert mids and all(r["status"] == "info" for r in mids)


def test_body_checks_pass_when_clean():
    rep = copy.deepcopy(REPORT)
    rep["contact"]["penetration_mm"] = 0.0
    c = build_checks(TARGETS, OURS, rep, EX)
    assert row(c, "body", "feet_in_floor")["status"] == "pass"
    assert row(c, "body", "cracks")["status"] == "pass"
    rep["contact"]["sides"]["left"]["sole_slide_mm"] = 9.0
    rep["form"]["contact"]["frames_touching_within_3mm"] = 10
    c = build_checks(TARGETS, OURS, rep, EX)
    assert row(c, "body", "foot_slide_left")["status"] == "fail"
    assert row(c, "body", "hands_on_bar")["status"] == "fail"


def run(run_dir):
    return subprocess.run([sys.executable, str(ROOT / "studio/measure/run_check.py"), str(run_dir)],
                          capture_output=True, text=True)


def test_run_check_cli(tmp_path):
    (tmp_path / "frames").mkdir()
    (tmp_path / "targets.json").write_text(json.dumps(TARGETS))
    (tmp_path / "frames/retarget.json").write_text((FIX / "retarget-squat.json").read_text())
    r = run(tmp_path)
    assert r.returncode == 1 and "ours.json" in r.stderr and "Traceback" not in r.stderr
    (tmp_path / "ours.json").write_text((FIX / "ours-squat.json").read_text())
    r = run(tmp_path)
    assert r.returncode == 0, r.stderr
    out = json.loads((tmp_path / "checks.json").read_text())
    assert sum(out["summary"][s] for s in STATUSES) == len(out["rows"])


def drop_frame(o, frame):
    o = copy.deepcopy(o)
    o["per_frame"] = [f for f in o["per_frame"] if f["frame"] != frame]
    return o


def test_partly_judged_width_says_so():
    c = build_checks(TARGETS, drop_frame(OURS, 50), REPORT, EX)
    assert "Judged at 2 of 3 positions" in row(c, "all positions", "grip_ratio")["note"]
    full = build_checks(TARGETS, OURS, REPORT, EX)
    assert "Judged at" not in (row(full, "all positions", "grip_ratio")["note"] or "")


def test_width_with_no_measured_position_is_info():
    o = copy.deepcopy(OURS)
    o["per_frame"] = []
    c = build_checks(TARGETS, o, REPORT, EX)
    r = row(c, "all positions", "grip_ratio")
    assert (r["status"], r["kind"], r["note"]) == ("info", "not-measured", "Not measured on the character")


def test_worst_position_is_reported():
    o = copy.deepcopy(OURS)
    vals = {68: 1.866, 50: 1.2, 38: 1.5}
    for f in o["per_frame"]:
        if f["frame"] in vals:
            f["flat"]["grip_ratio"] = vals[f["frame"]]
    r = row(build_checks(TARGETS, o, REPORT, EX), "all positions", "grip_ratio")
    assert (r["status"], r["frame"], r["ours"]) == ("fail", 50, 1.2)


def test_fail_note_only_blames_the_model_when_the_character_cannot_reach_it():
    from checks import _want
    from feasibility import closest_arm_pose
    reach = closest_arm_pose(OURS["arm"], _want(TARGETS, "end"))
    t, o = copy.deepcopy(TARGETS), copy.deepcopy(OURS)
    t["widths"]["grip_ratio"]["value"] = reach["grip_ratio"]        # the character's own best grip
    for ph in ("start", "middle", "end"):                             # whatever each position's want, it reaches this
        assert abs(closest_arm_pose(o["arm"], _want(t, ph))["grip_ratio"] - reach["grip_ratio"]) <= 0.12
    for f in o["per_frame"]:
        f["flat"]["grip_ratio"] = 1.2
    r = row(build_checks(t, o, REPORT, EX), "all positions", "grip_ratio")
    assert r["status"] == "fail" and r["note"].startswith("Off by")
    assert "don't all fit together" not in r["note"]
    # when the coach's own numbers fit no arm (a human misses it too), it is the coach's reading, not a
    # character fail: reported uncertain, still saying why
    g = row(build_checks(TARGETS, OURS, REPORT, EX), "all positions", "grip_ratio")
    assert g["status"] == "uncertain" and "don't all fit together" in g["note"]


def test_coach_arm_numbers_that_dont_fit_are_uncertain_not_a_character_fail():
    # The squat fixture's grip target is one no arm can meet alongside the other arm numbers
    # (a human arm misses it too). That is the coach's reading not fitting together — read off a
    # foreshortened side photo — not a fault in the character: report it uncertain, with the reason,
    # never a hard fail that blames the character.
    g = row(build_checks(TARGETS, OURS, REPORT, EX), "all positions", "grip_ratio")
    assert g["status"] == "uncertain"
    assert g["uncertain_why"] == "coach"
    assert "don't all fit together" in g["note"]


def test_no_hand_block_is_a_not_measured_hands_row_on_a_bar_exercise_and_no_crash():
    rep = {k: v for k, v in REPORT.items() if k not in ("form", "grip")}
    c = build_checks(TARGETS, OURS, rep, EX)
    r = row(c, "body", "hands_on_bar")
    assert (r["status"], r["kind"], r["ours"], r["note"]) == ("info", "not-measured", None, "Not measured on the character")
    assert row(c, "body", "cracks")["status"] == "pass"
    # an exercise the pipeline holds no bar for has no hands row at all
    other = build_checks(TARGETS, OURS, rep, "Dumbbell Lateral Raise")
    assert not [r for r in other["rows"] if r["metric"] == "hands_on_bar"]


def test_unmeasured_things_are_info_never_pass():
    o = copy.deepcopy(OURS)
    del o["arm"]
    del o["phase_frames"]["end"]
    o["per_frame"] = [f for f in o["per_frame"] if f["frame"] != 50]
    for f in o["per_frame"]:
        f["flat"].pop("hip_below_knee", None)
    c = build_checks(TARGETS, o, REPORT, EX)
    for g in ("middle", "end"):
        for r in c["rows"]:
            if r["group"] == g and r["status"] != "uncertain" and r["metric"] not in ("hip_below_knee", "seat_below_knee"):
                assert r["status"] == "info" and r["note"] == "Not measured on the character", r
    # the MediaPipe-line depth is only ever shown (task 21): the seat against the top of the knee is the judged depth
    assert row(c, "end", "hip_below_knee")["kind"] == "shown-only" and row(c, "end", "seat_below_knee")["kind"] == "not-measured"
    assert row(c, "start", "hip_included")["status"] == "pass"
    # no arm at all: arm metrics are judged plainly, never limited
    assert row(c, "all positions", "grip_ratio")["status"] in ("fail", "pass")
    assert "limited" not in {r["status"] for r in c["rows"]}


def test_non_solver_exercise_never_gets_limited():
    from checks import _want
    from feasibility import closest_arm_pose
    human = {"shoulder_half_width": 0.19, "upper_arm": 0.30, "forearm": 0.27}
    hw = closest_arm_pose(human, _want(TARGETS, "end"))
    t, o = copy.deepcopy(TARGETS), copy.deepcopy(OURS)
    for p in t["phases"].values():
        p["elbow_included"]["value"], p["upperarm_vs_torso"]["value"] = hw["elbow_included"], hw["upperarm_vs_torso"]
    for m in ("grip_ratio", "elbow_width_ratio"):
        t["widths"][m]["value"] = hw[m]
    char = closest_arm_pose(o["arm"], hw)
    for f in o["per_frame"]:
        f["flat"].update(grip_ratio=char["grip_ratio"], elbow_width_ratio=char["elbow_width_ratio"])
        f["3d"].update(elbow_included=char["elbow_included"], upperarm_vs_torso=char["upperarm_vs_torso"])
    assert build_checks(t, o, REPORT, EX)["summary"]["limited"] >= 2          # the scenario is real
    other = build_checks(t, o, REPORT, "Barbell Deadlift")
    assert other["summary"]["limited"] == 0
    assert row(other, "all positions", "grip_ratio")["status"] == "fail"


def test_nan_is_not_measured_and_partial_blocks_are_info():
    nan = float("nan")
    o = copy.deepcopy(OURS)
    for f in o["per_frame"]:
        if f["frame"] == 38:
            f["flat"]["torso_lean"] = nan
    rep = copy.deepcopy(REPORT)
    rep["contact"]["penetration_mm"] = nan
    del rep["contact"]["sides"]["left"]["sole_slide_mm"]
    rep["deformation"] = {}
    rep["form"]["contact"] = {"deepest_inside_bar_mm": 1.0}
    c = build_checks(TARGETS, o, rep, EX)
    assert row(c, "end", "torso_lean")["status"] == "info"
    for m in ("feet_in_floor", "foot_slide_left", "cracks", "hands_on_bar"):
        r = row(c, "body", m)
        assert r["status"] == "info" and r["note"] == "Not measured on the character", m
    json.dumps(c, allow_nan=False)


def test_checks_json_never_has_nan_and_errors_are_plain(tmp_path):
    (tmp_path / "frames").mkdir()
    (tmp_path / "targets.json").write_text(json.dumps(TARGETS))
    rep = copy.deepcopy(REPORT)
    rep["contact"]["penetration_mm"] = float("nan")
    (tmp_path / "frames/retarget.json").write_text(json.dumps(rep))   # writes the NaN token
    (tmp_path / "ours.json").write_text((FIX / "ours-squat.json").read_text())
    assert run(tmp_path).returncode == 0
    text = (tmp_path / "checks.json").read_text()
    assert "NaN" not in text and row(json.loads(text), "body", "feet_in_floor")["status"] == "info"
    (tmp_path / "ours.json").write_text("{not json")
    r = run(tmp_path)
    assert r.returncode == 1 and "Traceback" not in r.stderr and "Error(" not in r.stderr
    assert r.stderr.startswith("Could not compare this run")


def with_jumps(jumps):
    o = copy.deepcopy(OURS)
    if jumps is None:
        o.pop("jumps", None)
    else:
        o["jumps"] = jumps
    return o


def test_smooth_motion_passes_without_jumps():
    r = row(build_checks(TARGETS, OURS, REPORT, EX), "body", "smooth_motion")      # the fixture has jumps: []
    assert r["status"] == "pass" and r["ours"] == 0 and r["target"] == 0 and r["unit"] == "frames"


def test_jumpy_motion_is_a_fail_naming_the_frames():
    jumps = [{"frame": 18, "hips_mm": 91.6, "knees_mm": 40.2}, {"frame": 20, "hips_mm": 88.0, "knees_mm": 45.0},
             {"frame": 59, "hips_mm": 30.0, "knees_mm": 12.0}]
    r = row(build_checks(TARGETS, with_jumps(jumps), REPORT, EX), "body", "smooth_motion")
    assert r["status"] == "fail" and r["ours"] == 3 and r["frame"] == 18
    assert r["note"] == "Jumps on frames 18, 20 and 59 (up to 92 mm)"
    many = [{"frame": f, "hips_mm": 10.0, "knees_mm": 5.0} for f in range(1, 21, 2)]
    r = row(build_checks(TARGETS, with_jumps(many), REPORT, EX), "body", "smooth_motion")
    assert r["status"] == "fail" and r["ours"] == 10
    assert r["note"] == "Jumps on frames 1, 3, 5, 7, 9, 11, 13, 15 … (10 mm)"


def test_smooth_motion_without_a_reading_is_info():
    r = row(build_checks(TARGETS, with_jumps(None), REPORT, EX), "body", "smooth_motion")
    assert r["status"] == "info" and r["ours"] is None and r["note"].startswith("Not measured")


def with_motion(jumps, steps):
    o = with_jumps(jumps)
    if steps is None:
        o.pop("steps", None)
    else:
        o["steps"] = steps
    return o


def test_a_displaced_block_fails_naming_its_steps():
    # four frames shifted as a block: the steps in (30) and out (34) name it; the block's frames are covered
    jumps = [{"frame": f, "hips_mm": 150.0, "knees_mm": 150.0} for f in range(30, 34)]
    steps = [{"frame": 30, "mm": 150.2}, {"frame": 34, "mm": 149.8}]
    r = row(build_checks(TARGETS, with_motion(jumps, steps), REPORT, EX), "body", "smooth_motion")
    assert r["status"] == "fail" and r["ours"] == 2 and r["frame"] == 30
    assert r["note"] == "Jumps on frames 30 and 34 (150 mm)"


def test_a_step_alone_fails():
    r = row(build_checks(TARGETS, with_motion([], [{"frame": 12, "mm": 80.0}]), REPORT, EX), "body", "smooth_motion")
    assert r["status"] == "fail" and r["note"] == "Jump on frame 12 (80 mm)"


def test_no_steps_reading_is_never_a_pass():
    r = row(build_checks(TARGETS, with_motion([], None), REPORT, EX), "body", "smooth_motion")
    assert r["status"] == "info" and r["note"].startswith("Not measured")
    r = row(build_checks(TARGETS, with_motion([{"frame": 5, "hips_mm": 70.0, "knees_mm": None}], None), REPORT, EX),
            "body", "smooth_motion")
    assert r["status"] == "fail" and r["note"] == "Jump on frame 5 (70 mm)"


def test_the_driver_middle_row_is_info_and_nothing_else_changes():
    # the middle frame was found by matching the driver there, so its own middle row can't be judged;
    # its start and end rows (turning points, not matches) and every other row are judged as before
    plain = copy.deepcopy(OURS)
    plain.pop("phase_driver", None)
    a, b = build_checks(TARGETS, plain, REPORT, EX), build_checks(TARGETS, OURS, REPORT, EX)   # fixture: hip_included
    r = row(b, "middle", "hip_included")
    assert r["status"] == "info" and r["note"] == "Used to find this moment" and r["ours"] == 109.0 and r["frame"] == 50
    others = lambda c: [x for x in c["rows"] if (x["group"], x["metric"]) != ("middle", "hip_included")]   # noqa: E731
    assert others(a) == others(b)


# --- folded skin (ours.json `folds`, written by studio/blender/inspect_run.py) ------------------------
# Each record is one patch of turned-over skin over the limit, at one frame: kind "crease" (inside a
# hinge bent past 60 degrees: status-neutral), "defect" (seen by a camera the user looks at: fails) or
# "hidden" (no such camera sees enough of it: the row can't pass, so it is info).

def with_folds(folds, first=1, last=69, height=1.0):
    o = copy.deepcopy(OURS)
    o["first"], o["last"] = first, last
    for k in ("folds", "folds_worst", "height_m"):
        o.pop(k, None)
    if height is not None:
        o["height_m"] = height
    if folds is not None:
        o["folds"] = folds
    return o


def fold(frame, bone, area, kind="defect", visible=None):
    return {"frame": frame, "bone": bone, "kind": kind, "area_mm2": area,
            "visible_mm2": area if visible is None else visible, "centre": [0.1, 0.0, 1.2]}


def every(bone, area, kind="defect", frames=range(1, 70)):
    return [fold(f, bone, area, kind) for f in frames]


def folded(folds, **kw):
    return row(build_checks(TARGETS, with_folds(folds, **kw), REPORT, EX), "body", "folded_skin")


def test_folded_skin_passes_with_no_fold_over_the_limit():
    from checks import FOLDED_SKIN_MAX_MM2
    r = folded([])
    assert (r["status"], r["ours"], r["target"], r["unit"]) == ("pass", 0, FOLDED_SKIN_MAX_MM2, "mm2")
    assert r["note"] is None and r["frame"] is None


def test_folded_skin_limit_scales_with_the_character_height():
    from checks import FOLDED_SKIN_MAX_MM2
    r = folded([], height=1.7)
    assert r["status"] == "pass" and abs(r["target"] - FOLDED_SKIN_MAX_MM2 * 1.7 ** 2) < 1e-9


def test_creases_alone_pass_showing_the_most_seen_one():
    folds = (every("mixamorig:LeftLeg", 1033.0, "crease", range(15, 21)) + every("mixamorig:RightLeg", 181.0, "crease")
             + every("mixamorig:RightForeArm", 480.0, "crease"))
    for f in folds:
        f["visible_mm2"] = {"mixamorig:LeftLeg": 40.0, "mixamorig:RightLeg": 0.0}.get(f["bone"], 12.5)
    folds[2]["visible_mm2"] = 231.8                                    # the left knee at frame 17
    r = folded(folds)
    assert (r["status"], r["ours"], r["frame"]) == ("pass", 231.8, 17)
    assert r["note"] == ("Creases only, at the knees and the right elbow, which are normal in deep bends; "
                         "the most seen is 232 mm² at the left knee")
    for f in folds:
        f["visible_mm2"] = 0.0
    r = folded(folds)
    assert (r["status"], r["ours"], r["frame"]) == ("pass", 0.0, None)
    assert r["note"] == "Creases only, at the knees and the right elbow, which are normal in deep bends; none is seen"


def test_a_visible_fold_fails_naming_every_place_largest_first():
    folds = (every("mixamorig:RightArm", 1007.6) + every("mixamorig:Head", 3268.3) + every("mixamorig:Neck", 1147.3)
             + every("mixamorig:LeftArm", 859.2) + [fold(38, "mixamorig:Spine1", 367.2)]
             + every("mixamorig:RightLeg", 1610.5, "crease", range(15, 60)) + every("mixamorig:LeftLeg", 1554.8, "crease", range(15, 60))
             + every("mixamorig:LeftForeArm", 406.4, "crease") + every("mixamorig:Hips", 978.5, "defect", range(9, 17))
             + every("mixamorig:LeftUpLeg", 358.3, "crease", range(21, 54)) + every("mixamorig:RightUpLeg", 359.9, "crease")
             + every("mixamorig:RightHandThumb1", 1368.4, "hidden") + every("mixamorig:LeftHandIndex1", 1323.6, "hidden"))
    r = folded(folds)
    assert (r["status"], r["ours"], r["frame"], r["unit"]) == ("fail", 3268.3, 1, "mm2")
    # the crotch (the Hips bone) and the hips' groin creases (UpLeg) are told apart
    assert r["note"] == ("Skin folds over at the neckline (up to 3,268 mm², every frame), the right shoulder "
                         "(up to 1,008 mm², every frame), the crotch (up to 978 mm², frames 9–16), the left shoulder "
                         "(up to 859 mm², every frame) and the belly (up to 367 mm², frame 38); plus creases at the "
                         "knees, the left elbow and the hips, which are normal in deep bends, and folds hidden from "
                         "every view at the hands")


def test_every_failing_place_is_named_however_many():
    bones = ["mixamorig:LeftArm", "mixamorig:RightArm", "mixamorig:Neck", "mixamorig:Spine1", "mixamorig:Spine2",
             "mixamorig:LeftHand", "mixamorig:RightFoot"]
    r = folded([fold(10, b, 500.0 - i) for i, b in enumerate(bones)])
    assert r["status"] == "fail"
    for place in ("the left shoulder", "the right shoulder", "the neckline", "the belly", "the upper torso",
                  "the left wrist", "the right ankle"):
        assert place in r["note"], place
    assert "more place" not in r["note"]


def test_folds_hidden_from_every_view_are_info():
    r = folded(every("mixamorig:RightHandThumb1", 1368.4, "hidden") + every("mixamorig:RightLeg", 900.0, "crease"))
    assert (r["status"], r["ours"], r["frame"]) == ("info", 1368.4, 1)
    assert r["note"] == ("Folds hidden from every view at the right hand (up to 1,368 mm², every frame); "
                         "plus creases at the right knee, which are normal in deep bends")


def test_the_frames_are_named_when_not_every_frame():
    r = folded([fold(f, "mixamorig:LeftArm", 400.0) for f in range(36, 41)]
               + [fold(f, "mixamorig:RightArm", 300.0) for f in (3, 9, 60)])
    assert r["note"] == ("Skin folds over at the left shoulder (up to 400 mm², frames 36–40) and the right shoulder "
                         "(up to 300 mm², 3 of 69 frames)")


def test_folded_skin_without_a_reading_is_info():
    from checks import FOLDED_SKIN_MAX_MM2
    for folds, height in ((None, 1.0), ("not a list", 1.0), ([1, 2], 1.0), ([], None), ([], float("nan")), ([], 0)):
        r = folded(folds, height=height)
        assert (r["status"], r["ours"], r["note"]) == ("info", None, "Not measured on the character"), (folds, height)
        assert r["unit"] == "mm2"
    assert folded(None)["target"] == FOLDED_SKIN_MAX_MM2


def test_folded_skin_never_nan_and_unreadable_records_are_never_a_pass():
    nan = float("nan")
    for bad in (fold(4, "mixamorig:LeftArm", nan), fold(5, "mixamorig:LeftArm", "big"),
                fold(6, "mixamorig:LeftArm", 300.0, kind="odd"), {**fold(7, "mixamorig:LeftArm", 300.0), "kind": None}):
        r = folded([bad])
        assert r["status"] == "info" and r["ours"] is None and r["note"] == "Not measured on the character", bad
    # beside a visible fold, the fold still fails
    r = folded([fold(4, "mixamorig:LeftArm", nan), fold(6, "mixamorig:RightArm", 300.0)])
    assert r["status"] == "fail" and r["ours"] == 300.0
    json.dumps(build_checks(TARGETS, with_folds([fold(4, "mixamorig:LeftArm", nan)]), REPORT, EX), allow_nan=False)


def test_folded_skin_place_names():
    from checks import fold_place
    assert fold_place("mixamorig:RightArm") == "the right shoulder"
    assert fold_place("mixamorig:LeftForeArm") == "the left elbow"
    assert fold_place("mixamorig:LeftHand") == "the left wrist"
    assert fold_place("mixamorig:LeftHandThumb1") == "the left hand"
    assert fold_place("mixamorig:RightUpLeg") == "the right hip"
    assert fold_place("mixamorig:RightLeg") == "the right knee"
    assert fold_place("mixamorig:LeftFoot") == "the left ankle"
    assert fold_place("mixamorig:LeftToeBase") == "the left toes"
    assert fold_place("mixamorig:LeftShoulder") == "the left collarbone"
    assert fold_place("mixamorig:Hips") == "the crotch"
    assert fold_place("mixamorig:Spine1") == "the belly"
    assert fold_place("mixamorig:Neck") == "the neckline"
    assert fold_place("mixamorig:Head") == "the neckline"
    assert fold_place("Bone.004") == "Bone.004"
    assert fold_place(None) == "an unnamed place"


# --- posture (Task 6c): the coach's own posture in each side view, per position ----------------------
# Coach (fixture, read off 15/16/17.png): back_rounding -0.011 / 0.026 / -0.009 torso lengths with 26% /
# 26% / 23% of the back in view (the elbows hide the rest): the coach's back is unreadable, and the
# squat's form notes ask for a straight back, so ours is judged against straight. neck_vs_torso 38.6 /
# 13.4 / 13.5 deg (3D 52.1 / 27.0 / 16.6); head_pitch -36.2 / -14.3 / -11.4 deg. Our back is its spine's
# curl from its rest pose (ours.json flat spine_curl, degrees).

POSTURE = ("back_rounding", "neck_vs_torso", "head_pitch")
JUDGED_STRAIGHT = "judged against a straight back, as the exercise's form notes require"


def with_posture(**phases):
    """OURS with posture readings at the phase frames: with_posture(start={"neck_vs_torso": 40}, ...)."""
    o = copy.deepcopy(OURS)
    for phase, values in phases.items():
        frame = o["phase_frames"][phase]
        for f in o["per_frame"]:
            if f["frame"] == frame:
                f["flat"].update(values)
    return o


def curls(**phases):
    """OURS with spine curls (degrees) at the phase frames: curls(start=4, end=18)."""
    return with_posture(**{p: {"spine_curl": v} for p, v in phases.items()})


def coach(phase, metric):
    return TARGETS["phases"][phase][metric]["value"]


def coach_back(**phases):
    """TARGETS with the coach's back classified as given: coach_back(start={"state": "straight"}, ...)."""
    t = copy.deepcopy(TARGETS)
    for p, fields in phases.items():
        t["phases"][p]["back_rounding"].update(fields)
    return t


def back(c, phase):
    return row(c, phase, "back_rounding")


def test_posture_rows_for_every_position():
    c = build_checks(TARGETS, OURS, REPORT, EX)
    for phase in ("start", "middle", "end"):
        for m in POSTURE:
            r = row(c, phase, m)
            assert r["unit"] == "deg" and r["frame"] == OURS["phase_frames"][phase]
            # the fixture's character has no posture readings: never a pass
            assert (r["status"], r["ours"], r["note"]) == ("info", None, "Not measured on the character"), r
        assert row(c, phase, "neck_vs_torso")["target"] == coach(phase, "neck_vs_torso")
        assert back(c, phase)["target"] == 0          # judged against a straight back (see below)


def test_the_squat_coach_back_is_hidden_so_ours_is_judged_against_straight():
    assert all(TARGETS["phases"][p]["back_rounding"]["state"] == "unreadable" for p in ("start", "middle", "end"))
    c = build_checks(TARGETS, curls(start=4.0, middle=-12.4, end=18.2), REPORT, EX)
    r = back(c, "start")
    assert (r["status"], r["target"], r["ours"], r["unit"]) == ("pass", 0, 4.0, "deg")
    assert r["note"] == f"The coach's back isn't visible in these photos (the arms hide 74%); {JUDGED_STRAIGHT}"
    r = back(c, "end")
    assert r["status"] == "fail"
    assert r["note"] == f"His back rounds 18°. The coach's back isn't visible in these photos (the arms hide 77%); {JUDGED_STRAIGHT}"
    r = back(c, "middle")
    assert r["status"] == "fail" and r["note"].startswith("His back arches 12°. The coach's back isn't visible")
    # exactly 10 degrees from straight is still straight
    assert back(build_checks(TARGETS, curls(start=-10.0), REPORT, EX), "start")["status"] == "pass"


def test_without_a_form_rule_an_unreadable_coach_back_is_info_with_the_plain_reason():
    c = build_checks(TARGETS, curls(start=4.0, end=30.0), REPORT, "Mystery Lift")
    for phase, seen in (("start", 26), ("end", 23)):
        r = back(c, phase)
        assert (r["status"], r["target"]) == ("info", None)
        assert r["note"] == f"Not judged: only {seen}% of his back is in view — the arms hide the rest"


def test_a_straight_coach_back():
    t = coach_back(start={"state": "straight", "cause": None}, middle={"state": "straight", "cause": None},
                   end={"state": "straight", "cause": None})
    c = build_checks(t, curls(start=4.0, middle=-12.0, end=18.0), REPORT, EX)
    assert (back(c, "start")["status"], back(c, "start")["target"], back(c, "start")["note"]) == (
        "pass", 0, "Straight like the coach's")
    assert (back(c, "end")["status"], back(c, "end")["note"]) == ("fail", "His back rounds 18° where the coach keeps it straight")
    assert (back(c, "middle")["status"], back(c, "middle")["note"]) == (
        "fail", "His back arches 12° where the coach keeps it straight")


def test_a_rounded_coach_back_is_matched_in_direction_and_size():
    t = coach_back(**{p: {"state": "rounded", "cause": None, "curl": 35.0} for p in ("start", "middle", "end")})
    c = build_checks(t, curls(start=30.0, middle=12.0, end=-15.0), REPORT, EX)
    r = back(c, "start")
    assert (r["status"], r["target"], r["note"]) == ("pass", 35.0, "His back rounds 30°; the coach's rounds about 35°")
    assert (back(c, "middle")["status"], back(c, "middle")["note"]) == (
        "fail", "His back rounds 12°; the coach's rounds about 35°")
    assert (back(c, "end")["status"], back(c, "end")["note"]) == (
        "fail", "His back arches 15°; the coach's rounds about 35°")
    c = build_checks(t, curls(start=2.0), REPORT, EX)
    assert back(c, "start")["note"] == "His back stays straight; the coach's rounds about 35°"
    arched = coach_back(start={"state": "arched", "cause": None, "curl": -32.0})
    r = back(build_checks(arched, curls(start=-25.0), REPORT, EX), "start")
    assert (r["status"], r["note"]) == ("pass", "His back arches 25°; the coach's arches about 32°")


def test_plain_reasons_when_the_coach_back_cannot_be_read():
    level = {"state": "unreadable", "cause": "level", "value": None, "seen": 0.0, "span": None}
    broken = {"state": "unreadable", "cause": "broken", "seen": 0.92, "run": 0.4}
    unsure = {"state": "cant-tell", "cause": None, "seen": 1.0, "run": 1.0}
    t = coach_back(start=level, middle=broken, end=unsure)
    c = build_checks(t, curls(start=0.0, middle=0.0, end=0.0), REPORT, "Mystery Lift")
    assert back(c, "start")["note"] == "Not judged: he's lying level"
    assert back(c, "middle")["note"] == "Not judged: the arms break his back up — the longest unbroken stretch in view is 40%"
    assert back(c, "end")["note"] == "Not judged: his back reads neither clearly straight nor clearly rounded"
    assert {back(c, p)["status"] for p in ("start", "middle", "end")} == {"info"}
    c = build_checks(t, curls(start=0.0, middle=0.0, end=0.0), REPORT, EX)        # the squat's form notes
    assert back(c, "start")["note"] == f"The coach's back can't be read in these photos (he's lying level); {JUDGED_STRAIGHT}"
    assert back(c, "end")["note"] == (f"The coach's back reads neither clearly straight nor clearly rounded in these "
                                      f"photos; {JUDGED_STRAIGHT}")
    assert {back(c, p)["status"] for p in ("start", "middle", "end")} == {"pass"}


def test_a_coach_reading_without_seen_is_never_judged():
    from posture import classify_back
    state = classify_back(-0.011, None, [0.1, 0.9])
    t = coach_back(start={**state, "seen": None})
    r = back(build_checks(t, curls(start=0.0), REPORT, EX), "start")            # even with the squat's notes
    assert (r["status"], r["target"]) == ("info", None)
    assert r["note"] == "Not judged: the coach's reading doesn't say how much of his back was in view"


def test_the_straight_back_table_quotes_the_form_notes_verbatim():
    from checks import STRAIGHT_BACK, straight_back_rule
    notes = json.loads((ROOT / "reference/exercises/form-guidelines.json").read_text())
    text = {ex["name"]: " ".join(ex.get(f) or "" for f in ("start_position", "end_position", "body_alignment_cues"))
            for ex in notes["exercises"]}
    for exercise, cue in STRAIGHT_BACK.items():
        assert exercise in text, exercise
        assert cue in text[exercise], (exercise, cue)              # quoted, not paraphrased
        assert straight_back_rule(exercise) == cue
    assert straight_back_rule("Barbell Back Squat") == "SPINE: natural standing lordotic arch, neither rounded nor over-arched."
    assert "Conventional Barbell Deadlift" in STRAIGHT_BACK
    # a deliberate arch is not a straight back
    assert straight_back_rule("Flat Barbell Bench Press") is None
    assert straight_back_rule("Standing Barbell Overhead Press (Shoulder Press)") is None
    assert straight_back_rule("Mystery Lift") is None


@pytest.mark.parametrize("exercise", ["Flat Barbell Bench Press", "Standing Barbell Overhead Press (Shoulder Press)",
                                      "Mystery Lift"])
def test_exercises_off_the_table_leave_an_unreadable_coach_back_unjudged(exercise):
    c = build_checks(TARGETS, curls(start=30.0), REPORT, exercise)
    r = back(c, "start")
    assert (r["status"], r["target"]) == ("info", None)
    assert r["note"] == "Not judged: only 26% of his back is in view — the arms hide the rest"


def test_the_head_forward_of_the_coach_fails():
    # both neck lines run from the shoulder joints to the ears (ours found on the skin), no baseline
    o = with_posture(start={"neck_vs_torso": coach("start", "neck_vs_torso") + 14},
                     middle={"neck_vs_torso": coach("middle", "neck_vs_torso") - 12.2},
                     end={"neck_vs_torso": coach("end", "neck_vs_torso") + 9.6})
    c = build_checks(TARGETS, o, REPORT, EX)
    r = row(c, "start", "neck_vs_torso")
    assert (r["status"], r["note"]) == ("fail", "Head 14° further forward than the coach")
    r = row(c, "middle", "neck_vs_torso")
    assert (r["status"], r["note"]) == ("fail", "Head 12° further back than the coach")
    r = row(c, "end", "neck_vs_torso")
    assert (r["status"], r["note"]) == ("pass", None)
    o = with_posture(end={"neck_vs_torso": coach("end", "neck_vs_torso") + 10.3})
    r = row(build_checks(TARGETS, o, REPORT, EX), "end", "neck_vs_torso")
    assert r["status"] == "fail" and r["note"].startswith("Head 10.3° further forward")


def test_where_the_face_points_is_shown_never_judged():
    o = with_posture(start={"head_pitch": coach("start", "head_pitch") + 30},
                     end={"head_pitch": coach("end", "head_pitch") - 0.2})
    c = build_checks(TARGETS, o, REPORT, EX)
    r = row(c, "start", "head_pitch")
    assert r["status"] == "info" and r["ours"] == approx(coach("start", "head_pitch") + 30)
    assert r["note"] == ("Shown, not judged: the coach's reading assumes a level gaze puts the nose 12° below "
                         "the ear. Ours looks 30° further up than the coach")
    r = row(c, "end", "head_pitch")
    assert r["status"] == "info" and r["note"].endswith("Ours looks the same way as the coach")
    assert all(r["status"] == "info" for r in c["rows"] if r["metric"] == "head_pitch")


def test_a_posture_the_coach_picture_does_not_show_is_info():
    t = copy.deepcopy(TARGETS)
    t["phases"]["end"]["neck_vs_torso"]["value"] = None
    o = with_posture(end={"neck_vs_torso": 5.0})
    r = row(build_checks(t, o, REPORT, EX), "end", "neck_vs_torso")
    assert (r["status"], r["target"], r["ours"]) == ("info", None, 5.0)
    assert r["note"] == "Not measured in the coach's picture"


def test_an_uncertain_coach_neck_is_uncertain():
    t = copy.deepcopy(TARGETS)
    t["phases"]["end"]["neck_vs_torso"].update(cross=40.0, uncertain=True)
    o = with_posture(end={"neck_vs_torso": coach("end", "neck_vs_torso")})
    r = row(build_checks(t, o, REPORT, EX), "end", "neck_vs_torso")
    assert r["status"] == "uncertain"
    assert r["note"] == "Two readings of the coach disagree: 13° flat, 40° in 3D"


def test_an_estimated_middle_posture_is_info():
    t = copy.deepcopy(TARGETS)
    t["phases"]["middle"]["estimated"] = True
    o = with_posture(middle={"spine_curl": 40.0, "neck_vs_torso": 80.0, "head_pitch": 0.0})
    c = build_checks(t, o, REPORT, EX)
    for m in POSTURE:
        r = row(c, "middle", m)
        assert r["status"] == "info" and r["note"] == "Estimated between the other two positions", r


def test_posture_never_nan_and_targets_without_posture_add_no_rows():
    nan = float("nan")
    o = with_posture(start={"spine_curl": nan, "neck_vs_torso": nan, "head_pitch": "up"})
    c = build_checks(TARGETS, o, REPORT, EX)
    for m in POSTURE:
        r = row(c, "start", m)
        assert (r["status"], r["ours"], r["note"]) == ("info", None, "Not measured on the character"), r
    json.dumps(c, allow_nan=False)
    old = copy.deepcopy(TARGETS)
    for p in old["phases"].values():
        for m in POSTURE:
            p.pop(m, None)
    c = build_checks(old, OURS, REPORT, EX)
    assert not [r for r in c["rows"] if r["metric"] in POSTURE]
    assert sum(c["summary"][s] for s in STATUSES) == len(c["rows"])


# --- info rows say which kind they are, and not-measured rows stop a green exercise (final fix 1) -----
KINDS = ("not-measured", "not-judged", "shown-only")


def kinds_ok(c):
    for r in c["rows"]:
        if r["status"] == "info":
            assert r["kind"] in KINDS, r
        else:
            assert r["kind"] is None, r
    assert sum(c["summary"][s] for s in STATUSES) == len(c["rows"])


def test_every_info_row_says_which_kind_and_judged_rows_none():
    t = copy.deepcopy(TARGETS)
    t["phases"]["middle"]["estimated"] = True
    for c in (build_checks(TARGETS, OURS, REPORT, EX), build_checks(t, with_jumps(None), REPORT, EX),
              build_checks(TARGETS, curls(start=4.0, end=30.0), REPORT, "Mystery Lift"),
              build_checks(TARGETS, drop_frame(OURS, 38), {}, EX)):
        kinds_ok(c)


def test_the_three_kinds():
    o = with_posture(start={"head_pitch": coach("start", "head_pitch")})
    c = build_checks(TARGETS, o, REPORT, EX)
    assert row(c, "middle", "hip_included")["kind"] == "shown-only"              # "Used to find this moment"
    assert row(c, "start", "head_pitch")["kind"] == "shown-only"                 # face direction, shown only
    assert row(c, "body", "skin_stretch")["kind"] == "shown-only"
    r = row(c, "end", "head_pitch")          # no reading on the character: still shown only, never counted (task 18)
    assert (r["kind"], r["note"]) == ("shown-only", "Not measured on the character")
    assert row(c, "end", "neck_vs_torso")["kind"] == "not-measured"
    assert row(build_checks(TARGETS, with_jumps(None), REPORT, EX), "body", "smooth_motion")["kind"] == "not-measured"
    # not judged: the reason is on the coach's side, or the row can't be judged by design
    t = copy.deepcopy(TARGETS)
    t["phases"]["middle"]["estimated"] = True
    plain = {k: v for k, v in OURS.items() if k != "phase_driver"}
    assert {r["kind"] for r in build_checks(t, plain, REPORT, EX)["rows"] if r["group"] == "middle"} == {"not-judged"}
    assert back(build_checks(TARGETS, curls(start=4.0), REPORT, "Mystery Lift"), "start")["kind"] == "not-judged"
    t = copy.deepcopy(TARGETS)
    t["phases"]["end"]["neck_vs_torso"]["value"] = None
    assert row(build_checks(t, with_posture(end={"neck_vs_torso": 5.0}), REPORT, EX), "end", "neck_vs_torso")["kind"] == "not-judged"
    assert folded(every("mixamorig:RightHandThumb1", 1368.4, "hidden"))["kind"] == "not-judged"


def test_unmeasured_counts_not_measured_rows_except_shown_only_ones():
    o = with_posture(start={"spine_curl": 4.0, "neck_vs_torso": 38.0, "head_pitch": -30.0},
                     middle={"spine_curl": 4.0, "neck_vs_torso": 14.0, "head_pitch": -14.0},
                     end={"spine_curl": 4.0, "neck_vs_torso": 14.0, "head_pitch": -11.0})
    o.update(folds=[], height_m=1.0)          # the fixture has no fold reading
    o["seat_below_knee"] = {"end": {"frame": 38, "picture": 0.62, "skin": 0.6, "why": None}}   # nor a seat depth
    c = build_checks(TARGETS, o, REPORT, EX)
    assert c["summary"]["unmeasured"] == 0
    no_jumps = {k: v for k, v in o.items() if k != "jumps"}
    assert build_checks(TARGETS, no_jumps, REPORT, EX)["summary"]["unmeasured"] == 1
    # the fixture's character has no posture, no folds and no seat depth: 3 back + 3 neck rows + the folds row + the
    # depth row (task 22: it predates it) count; its 3 face-direction rows, shown only, don't — and say so (task 18)
    c = build_checks(TARGETS, OURS, REPORT, EX)
    assert c["summary"]["unmeasured"] == 8
    assert len([r for r in c["rows"] if r["kind"] == "not-measured"]) == 8
    # a whole position missing on the character: every judged metric there counts, the body rows too
    c = build_checks(TARGETS, drop_frame(o, 38), {}, EX)
    end_judged = [r for r in c["rows"] if r["group"] == "end" and r["kind"] == "not-measured" and r["metric"] != "head_pitch"]
    body = [r for r in c["rows"] if r["group"] == "body" and r["kind"] == "not-measured"]
    assert end_judged and body
    assert c["summary"]["unmeasured"] == len(end_judged) + len(body)


def test_width_rows_without_a_front_or_behind_picture_are_not_judged():
    from test_build_targets import subset
    t = build_targets(EX, *subset("img-15", "img-16", "img-17"))
    assert t["widths"] == {}
    c = build_checks(t, OURS, REPORT, EX)
    for m in ("grip_ratio", "elbow_width_ratio"):
        r = row(c, "all positions", m)
        assert (r["status"], r["kind"], r["target"], r["ours"]) == ("info", "not-judged", None, None)
        assert r["note"] == "Not judged: no front or behind picture"
    # with one, a width it couldn't read says so instead
    t = copy.deepcopy(TARGETS)
    del t["widths"]["elbow_width_ratio"]
    r = row(build_checks(t, OURS, REPORT, EX), "all positions", "elbow_width_ratio")
    assert (r["kind"], r["note"]) == ("not-judged", "Not judged: not measured in the front or behind picture")


def test_a_coach_reading_missing_from_a_position_is_a_not_judged_row():
    t = copy.deepcopy(TARGETS)
    del t["phases"]["start"]["torso_lean"]
    t["phases"]["end"]["knee_included"]["value"] = None
    c = build_checks(t, OURS, REPORT, EX)
    for phase, m in (("start", "torso_lean"), ("end", "knee_included")):
        r = row(c, phase, m)
        assert (r["status"], r["kind"], r["target"]) == ("info", "not-judged", None)
        assert r["note"] == "Not judged: not measured in the coach's picture"


# --- carried minors (task 18): every amber "Not measured" chip is counted in the header, and only those -----

def not_measured_chips(c):
    return len([r for r in c["rows"] if r["kind"] == "not-measured"])


def test_shown_only_rows_without_a_reading_say_shown_only():
    # face direction and skin stretch never hold an exercise back, so without a reading their chip is
    # "Shown only", not the amber "Not measured" the header doesn't count
    c = build_checks(TARGETS, OURS, REPORT, EX)            # the fixture's character has no posture readings
    for phase in ("start", "middle", "end"):
        r = row(c, phase, "head_pitch")
        assert (r["status"], r["kind"], r["ours"], r["note"]) == ("info", "shown-only", None,
                                                                  "Not measured on the character"), r
    report = copy.deepcopy(REPORT)
    report["deformation"]["deepest"].pop("max_growth_pct", None)
    r = row(build_checks(TARGETS, OURS, report, EX), "body", "skin_stretch")
    assert (r["status"], r["kind"], r["ours"], r["note"]) == ("info", "shown-only", None, "Not measured on the character")
    r = row(build_checks(TARGETS, OURS, {}, EX), "body", "skin_stretch")
    assert (r["status"], r["kind"]) == ("info", "shown-only")


def test_an_uncertain_row_whose_ours_is_not_measured_is_not_measured():
    # the coach's two readings disagree, but ours has no reading at all: that is a gap on our side, counted
    assert row(build_checks(TARGETS, OURS, REPORT, EX), "start", "knee_included")["status"] == "uncertain"
    c = build_checks(TARGETS, drop_frame(OURS, 68), REPORT, EX)
    r = row(c, "start", "knee_included")
    assert (r["status"], r["kind"], r["note"]) == ("info", "not-measured", "Not measured on the character"), r
    # the neck too (the fixture's character has no neck reading)
    t = copy.deepcopy(TARGETS)
    t["phases"]["end"]["neck_vs_torso"].update(cross=40.0, uncertain=True)
    r = row(build_checks(t, OURS, REPORT, EX), "end", "neck_vs_torso")
    assert (r["status"], r["kind"]) == ("info", "not-measured"), r
    o = with_posture(end={"neck_vs_torso": coach("end", "neck_vs_torso")})
    assert row(build_checks(t, o, REPORT, EX), "end", "neck_vs_torso")["status"] == "uncertain"
    # and a width: ours is shown (the position furthest from the coach) when measured, else not measured
    t = copy.deepcopy(TARGETS)
    t["widths"]["grip_ratio"].update(cross=2.5, uncertain=True)
    r = row(build_checks(t, OURS, REPORT, EX), "all positions", "grip_ratio")
    assert (r["status"], r["ours"], r["frame"]) == ("uncertain", approx(1.7), 68), r
    o = copy.deepcopy(OURS)
    o["per_frame"] = []
    r = row(build_checks(t, o, REPORT, EX), "all positions", "grip_ratio")
    assert (r["status"], r["kind"]) == ("info", "not-measured"), r


def test_the_amber_chips_add_up_to_the_header():
    t = copy.deepcopy(TARGETS)
    t["phases"]["end"]["neck_vs_torso"].update(cross=40.0, uncertain=True)
    t["widths"]["grip_ratio"].update(cross=2.5, uncertain=True)
    report = copy.deepcopy(REPORT)
    report["deformation"]["deepest"].pop("max_growth_pct", None)
    empty = copy.deepcopy(OURS)
    empty["per_frame"] = []
    for c in (build_checks(TARGETS, OURS, REPORT, EX), build_checks(t, drop_frame(OURS, 68), report, EX),
              build_checks(t, empty, {}, EX), build_checks(TARGETS, with_jumps(None), REPORT, EX),
              build_checks(t, with_posture(start={"head_pitch": -30.0}), report, EX)):
        assert c["summary"]["unmeasured"] == not_measured_chips(c) > 0
        kinds_ok(c)


# --- ours read from the picture, as the coach is (task 19; studio/measure/picture.py) -------------------
# The fixture's rig reads (frames 68 / 50 / 38): lean 6.7 / 18.4 / 30.0, hip 171 / 109 / 72, knee 175 / 105 / 47.

def picture(flat=None, ok=True, why=None, d3=None):
    return {"ok": ok, "reads": 5, "seen": 5 if ok else 1, "flat": flat if ok else None, "3d": d3 if ok else None,
            "spread": None, "visibility": {"torso": 1.0 if ok else 0.3, "knee": 0.2}, "why": why}


def with_picture(**phases):
    o = copy.deepcopy(OURS)
    o["picture"] = phases
    return o


def test_the_picture_is_ours_for_the_lean_and_hip():
    o = with_picture(end=picture({"torso_lean": 21.5, "hip_included": 70.0, "knee_included": 48.0}))
    c = build_checks(TARGETS, o, REPORT, EX)
    lean, hip = row(c, "end", "torso_lean"), row(c, "end", "hip_included")
    assert (lean["ours"], lean["status"], lean["frame"]) == (21.5, "pass", 38)     # the rig's 30.0 failed
    assert lean["note"] == "Read from the picture, as the coach's; the rig reads 30°"
    assert (hip["ours"], hip["status"]) == (70.0, "pass")
    # the picture's hip angle runs to its knee landmark: said when MediaPipe barely sees it (task 19 review)
    assert hip["note"] == "Read from the picture, as the coach's; the rig reads 72°; the picture's knee landmark is seen at 0.20"
    assert lean["uncertain_why"] is None


def test_a_picture_far_from_the_rig_is_uncertain_with_both_numbers():
    # more than 2 x the 5-degree tolerance apart: the coach's own rule for two readings that disagree
    o = with_picture(end=picture({"torso_lean": 10.0, "hip_included": 99.0}))
    c = build_checks(TARGETS, o, REPORT, EX)
    lean = row(c, "end", "torso_lean")
    assert (lean["ours"], lean["status"], lean["uncertain_why"]) == (10.0, "uncertain", "ours")
    assert lean["note"] == "The picture reads 10°, the rig reads 30°"
    hip = row(c, "end", "hip_included")      # the rig passes (72 vs 69.3): which is right is a question for the eye
    assert (hip["status"], hip["note"]) == ("uncertain", "The picture reads 99°, the rig reads 72°; the picture's knee "
                                                         "landmark is seen at 0.20")
    assert c["summary"]["uncertain"] == build_checks(TARGETS, OURS, REPORT, EX)["summary"]["uncertain"] + 2


def test_exactly_twice_the_tolerance_apart_is_still_judged():
    o = with_picture(end=picture({"torso_lean": 20.0, "hip_included": 70.0}))     # rig 30.0: 10.0 apart
    lean = row(build_checks(TARGETS, o, REPORT, EX), "end", "torso_lean")
    assert (lean["status"], lean["ours"]) == ("pass", 20.0)


def test_a_picture_judged_off_says_so_and_names_the_rig():
    o = with_picture(middle=picture({"torso_lean": 12.4, "hip_included": 115.3}))   # rig 18.4
    lean = row(build_checks(TARGETS, o, REPORT, EX), "middle", "torso_lean")
    assert (lean["ours"], lean["status"]) == (12.4, "fail")
    assert lean["note"] == "Off by 6.0°. Read from the picture, as the coach's; the rig reads 18°"


def test_an_unreadable_picture_falls_back_to_the_rig_and_says_why():
    why = "MediaPipe saw our shoulders and hips clearly in 1 of 5 framings"
    o = with_picture(end=picture(ok=False, why=why))
    lean = row(build_checks(TARGETS, o, REPORT, EX), "end", "torso_lean")
    assert (lean["ours"], lean["status"]) == (30.0, "fail")
    assert lean["note"] == f"Off by 7.6°. Read on the rig: {why}"
    # a position the stills step read no picture for at all
    lean = row(build_checks(TARGETS, with_picture(), REPORT, EX), "end", "torso_lean")
    assert lean["ours"] == 30.0 and lean["note"].endswith("Read on the rig: no picture of ours at this position")
    # neither the picture nor the rig read there: not measured, and the picture's reason is kept (task 22)
    gone = drop_frame(o, 38)
    lean = row(build_checks(TARGETS, gone, REPORT, EX), "end", "torso_lean")
    assert (lean["kind"], lean["note"]) == ("not-measured", f"Not measured on the character: {why}")


def test_a_run_without_a_picture_reading_is_judged_on_the_rig_as_before():
    c = build_checks(TARGETS, OURS, REPORT, EX)
    lean = row(c, "end", "torso_lean")
    assert (lean["ours"], lean["status"], lean["note"]) == (30.0, "fail", "Off by 7.6°")


def test_knees_and_arms_stay_on_the_rig_and_show_the_picture():
    o = with_picture(end=picture({"torso_lean": 21.0, "hip_included": 70.0, "knee_included": 25.0},
                                 d3={"elbow_included": 60.0, "upperarm_vs_torso": 30.0}))
    c = build_checks(TARGETS, o, REPORT, EX)
    knee = row(c, "end", "knee_included")      # the coach's two readings disagree: uncertain, as before
    assert (knee["ours"], knee["status"], knee["uncertain_why"]) == (47.0, "uncertain", "coach")
    assert knee["note"].endswith("The picture of ours reads 25° (shown only; its knee landmark is seen at 0.20)")
    elbow = row(c, "end", "elbow_included")    # the coach's elbow is a 3-D reading: so is the picture's shown
    arms = "the detector doesn't read the cartoon's arms reliably"
    assert elbow["ours"] == 82.3 and elbow["note"] == f"The picture of ours reads 60° (shown only; {arms})"
    assert row(c, "end", "upperarm_vs_torso")["note"] == f"The picture of ours reads 30° (shown only; {arms})"


def test_the_moment_finder_stays_on_the_rig_and_shows_the_picture():
    o = with_picture(middle=picture({"torso_lean": 18.0, "hip_included": 115.3}))
    r = row(build_checks(TARGETS, o, REPORT, EX), "middle", "hip_included")
    assert (r["status"], r["ours"]) == ("info", 109.0)      # the moment was found on the rig's reading
    assert r["note"] == "Used to find this moment; the picture reads 115°"


def test_uncertain_rows_say_whose_readings_disagree():
    c = build_checks(TARGETS, OURS, REPORT, EX)
    assert row(c, "start", "knee_included")["uncertain_why"] == "coach"
    assert all(r["uncertain_why"] is None for r in c["rows"] if r["status"] != "uncertain")


def test_a_note_ending_in_a_full_stop_is_not_given_another():
    o = with_picture(end=picture({"torso_lean": 21.0, "hip_included": 70.0},
                                 d3={"elbow_included": 60.0, "upperarm_vs_torso": 30.0}))
    note = row(build_checks(TARGETS, o, REPORT, EX), "all positions", "grip_ratio")["note"]   # no picture note: unchanged
    assert ".." not in note
    from checks import _joined
    assert _joined("a human-proportioned arm reaches 41°.", "The picture of ours reads 63° (shown only)") == \
        "a human-proportioned arm reaches 41°. The picture of ours reads 63° (shown only)"
    assert _joined("Off by 6.0°", "Read from the picture") == "Off by 6.0°. Read from the picture"
    assert _joined(None, "x") == "x" and _joined(None, None) is None


# --- task 19 review: when the two readings of ours disagree ----------------------------------------------

def with_rig(o, frame, **values):
    for f in o["per_frame"]:
        if f["frame"] == frame:
            f["flat"].update(values)
    return o


def test_both_readings_off_on_the_same_side_is_a_fail_with_both_numbers():
    # picture 120, rig 100, coach 69.3: far apart, but both say "too open" - not a question of which is right
    o = with_rig(with_picture(end=picture({"torso_lean": 45.0, "hip_included": 120.0})), 38, hip_included=100.0)
    c = build_checks(TARGETS, o, REPORT, EX)
    hip = row(c, "end", "hip_included")
    assert (hip["status"], hip["ours"], hip["uncertain_why"]) == ("fail", 120.0, None)
    assert hip["note"] == ("Off by 50.7°. The picture reads 120°, the rig reads 100°; the picture's knee landmark is "
                           "seen at 0.20")
    lean = row(c, "end", "torso_lean")       # picture 45, rig 30, coach 22.4: both lean too far
    assert (lean["status"], lean["note"]) == ("fail", "Off by 22.6°. The picture reads 45°, the rig reads 30°")


def test_readings_either_side_of_the_coach_or_one_passing_are_uncertain():
    # straddling: picture 10 below the coach's 22.4, rig 30 above it
    o = with_picture(end=picture({"torso_lean": 10.0, "hip_included": 50.0}))
    c = build_checks(TARGETS, o, REPORT, EX)
    assert row(c, "end", "torso_lean")["status"] == "uncertain"
    # rig 72 passes (69.3), picture 50 is far below: uncertain
    assert row(c, "end", "hip_included")["status"] == "uncertain"
    # the picture passes and the rig is far off: uncertain too
    o = with_rig(with_picture(end=picture({"torso_lean": 22.0, "hip_included": 70.0})), 38, torso_lean=34.0)
    lean = row(build_checks(TARGETS, o, REPORT, EX), "end", "torso_lean")
    assert (lean["status"], lean["ours"]) == ("uncertain", 22.0)


def test_a_gap_near_the_uncertain_line_says_so():
    near = "Close to the uncertain line; a later run may read uncertain"
    o = with_picture(end=picture({"torso_lean": 20.5, "hip_included": 70.0}))     # rig 30.0: 9.5 apart
    lean = row(build_checks(TARGETS, o, REPORT, EX), "end", "torso_lean")
    assert lean["status"] == "pass" and lean["note"] == f"Read from the picture, as the coach's; the rig reads 30°. {near}"
    o = with_picture(end=picture({"torso_lean": 19.5, "hip_included": 70.0}))     # 10.5 apart: uncertain, near the line
    lean = row(build_checks(TARGETS, o, REPORT, EX), "end", "torso_lean")
    assert lean["status"] == "uncertain"
    assert lean["note"] == "The picture reads 20°, the rig reads 30°. Close to the uncertain line; a later run may be judged"
    o = with_picture(end=picture({"torso_lean": 21.5, "hip_included": 70.0}))     # 8.5 apart: nothing said
    assert "uncertain line" not in row(build_checks(TARGETS, o, REPORT, EX), "end", "torso_lean")["note"]


def test_a_well_seen_knee_is_not_mentioned():
    p = picture({"torso_lean": 21.5, "hip_included": 70.0, "knee_included": 48.0})
    p["visibility"]["knee"] = 0.8
    c = build_checks(TARGETS, with_picture(end=p), REPORT, EX)
    assert row(c, "end", "hip_included")["note"] == "Read from the picture, as the coach's; the rig reads 72°"
    assert row(c, "end", "knee_included")["note"].endswith("The picture of ours reads 48° (shown only)")


# ---- the equipment the hands should hold (task 9 review): never silent when the path doesn't build it ----

def _no_bar_report():
    r = copy.deepcopy(REPORT)
    r.pop("form", None)
    r.pop("grip", None)
    return r


@pytest.mark.parametrize("exercise,thing", [("Standing Barbell Overhead Press (Shoulder Press)", "bar"),
                                            ("Tricep Pushdown (cable, bar or rope attachment)", "cable handle")])
def test_equipment_the_path_does_not_build_is_a_not_measured_row(exercise, thing):
    c = build_checks({**TARGETS, "exercise": exercise}, OURS, _no_bar_report(), exercise)
    r = row(c, "body", "equipment")
    assert (r["status"], r["kind"], r["ours"]) == ("info", "not-measured", None)
    assert r["note"] == f"No {thing} on this path yet; the hands hold nothing"
    assert c["summary"]["unmeasured"] >= 1                     # so the exercise can never show green


def test_an_exercise_of_our_own_goes_by_its_equipment_word():
    targets = {**TARGETS, "exercise": "Back Squat from images", "equipment": "barbell"}
    c = build_checks(targets, OURS, _no_bar_report(), "Back Squat from images")
    assert row(c, "body", "equipment")["note"] == "No bar on this path yet; the hands hold nothing"


def test_an_older_dumbbell_run_without_physical_measurements_cannot_pass():
    c = build_checks({**TARGETS, 'exercise': 'Dumbbell Lateral Raise'}, OURS, _no_bar_report(), 'Dumbbell Lateral Raise')
    r = row(c, 'body', 'equipment_contact')
    assert r['kind'] == 'not-measured'
    assert c['summary']['unmeasured'] >= 1


@pytest.mark.parametrize("exercise", ["Barbell Back Squat", "Front Plank (forearm plank)"])
def test_no_equipment_row_when_the_bar_is_built_or_nothing_is_held(exercise):
    report = REPORT if exercise == "Barbell Back Squat" else _no_bar_report()
    c = build_checks({**TARGETS, "exercise": exercise}, OURS, report, exercise)
    assert not [r for r in c["rows"] if r["metric"] == "equipment"]


# --- the depth as a judge sees it: the seat against the top of the knee (task 22) ------------------------------
# The coach's (TARGETS: 17.png's MediaPipe person mask and landmarks, posture.seat_depth): 0.646 thigh lengths. Ours:
# our side still's mask render at the end frame (38 in the fixture), the 3D skin at that frame its cross-check.
from checks import NOT_ON_CHARACTER, PREDATES, SEAT_ABOUT


def with_seat(picture=None, skin=None, why=None, skin_why=None, frame=38, warning=None):
    o = copy.deepcopy(OURS)
    o["seat_below_knee"] = {"end": {"frame": frame, "picture": picture, "skin": skin, "why": why, "skin_why": skin_why}}
    if warning:
        o["legs_at_rest"] = {"warning": warning}
    return o


def seat(targets=TARGETS, ours=None, exercise=EX):
    return row(build_checks(targets, OURS if ours is None else ours, REPORT, exercise), "end", "seat_below_knee")


def test_the_depth_is_judged_on_the_seat_against_the_top_of_the_knee():
    r = seat(ours=with_seat(picture=0.589, skin=0.571))         # the old bottom: within 0.10 of his 0.646
    assert (r["ours"], r["status"], r["unit"], r["frame"]) == (0.589, "pass", "thigh", 38)
    assert r["target"] == approx(0.646, abs=0.005)
    assert r["note"] == f"Read from our picture, as the coach's; the 3D skin reads 0.57. {SEAT_ABOUT}"
    assert "whichever knee shows" in SEAT_ABOUT and "thicker thigh and seat" in SEAT_ABOUT
    # the MediaPipe-line depth is shown only: the rig's thigh line is not his landmarks
    d = row(build_checks(TARGETS, with_seat(picture=0.589, skin=0.571), REPORT, EX), "end", "hip_below_knee")
    assert (d["status"], d["kind"], d["unit"]) == ("info", "shown-only", "torso")
    assert d["note"].startswith("Shown only")


def test_the_hips_high_bottom_fails():
    high = seat(ours=with_seat(picture=0.507, skin=0.49))
    assert (high["status"], high["note"]) == ("fail", "Off by 0.14 thigh lengths. Read from our picture, as the "
                                                      f"coach's; the 3D skin reads 0.49. {SEAT_ABOUT}")


def test_a_miss_just_past_the_tolerance_shows_its_third_digit():
    """0.102 off reads "0.102", never "off by 0.10" beside a 0.10 tolerance (checks._amount)."""
    r = seat(ours=with_seat(picture=0.544, skin=0.53))
    assert r["status"] == "fail" and r["note"].startswith("Off by 0.102 thigh lengths."), r


def test_a_seat_deeper_than_the_coach_by_half_the_tolerance_is_said():
    """The deep side gets its own words past half the tolerance (0.05), the verdict unchanged: a pass stays a pass."""
    deep = seat(ours=with_seat(picture=0.71, skin=0.69))         # 0.064 deeper than his 0.646
    assert deep["status"] == "pass" and deep["note"] == ("Deeper than the coach by 0.06 thigh lengths. Read from our "
                                                         f"picture, as the coach's; the 3D skin reads 0.69. {SEAT_ABOUT}")
    for picture in (0.68, 0.58, 0.646):                          # 0.034 deeper, 0.066 shallower, his own
        assert "Deeper" not in seat(ours=with_seat(picture=picture, skin=picture))["note"], picture
    far = seat(ours=with_seat(picture=0.80, skin=0.79))          # a deep fail says which way it is off
    assert far["status"] == "fail" and far["note"].startswith("Off by 0.15 thigh lengths. Deeper than the coach by 0.15")


def test_a_seat_depth_far_from_the_skins_is_uncertain_only_when_they_could_disagree():
    split = seat(ours=with_seat(picture=0.62, skin=0.30))
    assert (split["status"], split["uncertain_why"]) == ("uncertain", "ours")
    assert split["note"] == f"Our picture reads 0.62, the 3D skin reads 0.30. {SEAT_ABOUT}"
    same = seat(ours=with_seat(picture=0.40, skin=0.15))
    assert (same["status"], same["note"]) == ("fail", "Off by 0.25 thigh lengths. Our picture reads 0.40, the 3D skin "
                                                      f"reads 0.15. {SEAT_ABOUT}")


def test_a_seat_depth_not_read_on_the_picture_falls_back_to_the_skin_and_says_why():
    why = "no knee top: the body over the knee reaches up into the torso"
    r = seat(ours=with_seat(picture=None, skin=0.60, why=why))
    assert (r["ours"], r["status"]) == (0.60, "pass") and r["note"] == f"Read on the 3D skin: {why}. {SEAT_ABOUT}"
    # neither read: the row keeps both reasons, counted as not measured
    both = seat(ours=with_seat(picture=None, skin=None, why=why, skin_why="no seat vertices behind the hip"))
    assert (both["status"], both["kind"]) == ("info", "not-measured")
    assert both["note"] == f"{NOT_ON_CHARACTER}: {why}; the 3D skin: no seat vertices behind the hip", both


def test_a_run_from_before_the_seat_depth_says_to_run_it_again():
    """Re-checking a run whose stills step or read predates task 22: not measured (it counts), and says so."""
    old_ours = seat()                                           # ours.json without the reading
    assert (old_ours["status"], old_ours["kind"], old_ours["note"]) == ("info", "not-measured", PREDATES)
    assert PREDATES == "This run predates the new depth reading; run it again"
    t = copy.deepcopy(TARGETS)
    del t["phases"]["end"]["seat_below_knee"]                   # targets read before it
    old_read = seat(targets=t, ours=with_seat(picture=0.6, skin=0.6))
    assert (old_read["kind"], old_read["note"], old_read["target"]) == ("not-measured", PREDATES, None)
    del t["tolerance"]["seat_below_knee"]                       # an old targets.json has no tolerance for it either
    assert seat(targets=t, ours=with_seat(picture=0.6, skin=0.6))["kind"] == "not-measured"


def test_no_coach_seat_depth_is_not_judged():
    t = copy.deepcopy(TARGETS)
    t["phases"]["end"]["seat_below_knee"]["value"] = None       # his picture read, the seat or knee top not found
    r = seat(targets=t, ours=with_seat(picture=0.6, skin=0.6))
    assert (r["status"], r["kind"], r["note"]) == ("info", "not-judged", "Not judged: not measured in the coach's picture")


def test_depth_rows_only_for_an_exercise_whose_kit_row_says_so():
    """The deadlift's end position has no depth rows, even with targets that carry them (an old read)."""
    lift = "Conventional Barbell Deadlift"
    c = build_checks({**TARGETS, "exercise": lift}, with_seat(picture=0.6, skin=0.6), REPORT, lift)
    assert not [r for r in c["rows"] if r["metric"] in ("seat_below_knee", "hip_below_knee", "thigh_outline")]
    c = build_checks(TARGETS, with_seat(picture=0.6, skin=0.6), REPORT, EX)
    assert [r["metric"] for r in c["rows"] if r["metric"] in ("seat_below_knee", "hip_below_knee")] == [
        "seat_below_knee", "hip_below_knee"]
    assert not [r for r in c["rows"] if r["metric"] == "thigh_outline"]     # task 21's row is gone


def test_a_bind_pose_warning_is_said_on_the_leg_rows():
    """The leg rows counted from the thigh's outline in the bind pose; not the seat depth, read off the silhouette and
    the skin alone."""
    w = "MediaPipe reads the bind pose's thigh 6.0 degrees from its outline"
    c = build_checks(TARGETS, with_seat(picture=0.6, skin=0.6, warning=w), REPORT, EX)
    for phase, metric in (("end", "hip_included"), ("start", "knee_included"), ("end", "hip_below_knee")):
        assert w in (row(c, phase, metric)["note"] or ""), (phase, metric)
    for phase, metric in (("end", "torso_lean"), ("end", "seat_below_knee")):
        assert w not in (row(c, phase, metric)["note"] or ""), (phase, metric)
