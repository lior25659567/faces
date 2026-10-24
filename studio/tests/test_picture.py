"""Our character read from its picture, as the coach is (studio/measure/picture.py, task 19)."""
import pytest
from pytest import approx

from picture import (FRAMINGS, MIN_SEEN, PICTURE_METRICS, REPORTED_METRICS, SEEN_MIN, TORSO_LANDMARKS,
                     picture_value, summarize, torso_visibility)


def read(lean, hip, knee=100.0, vis=1.0, knee_vis=0.2, elbow=40.0, upper=45.0, side="right"):
    """A read_pose result as the CLI prints it (only the keys picture.py reads)."""
    visibility = {n: vis for n in TORSO_LANDMARKS}
    visibility.update({"left_knee": knee_vis, "right_knee": knee_vis, "left_elbow": 0.5, "right_elbow": 0.5})
    return {"image": "x.png", "camera_side": side,
            "flat": {"torso_lean": lean, "hip_included": hip, "knee_included": knee, "elbow_included": 30.0,
                     "upperarm_vs_torso": 20.0},
            "3d": {"torso_lean": lean + 1, "hip_included": hip + 2, "knee_included": knee + 3,
                   "elbow_included": elbow, "upperarm_vs_torso": upper},
            "visibility": visibility}


def test_constants_are_what_the_report_justifies():
    assert PICTURE_METRICS == ("torso_lean", "hip_included")
    assert set(REPORTED_METRICS) == {"knee_included", "elbow_included", "upperarm_vs_torso"}
    assert SEEN_MIN == 0.5                      # MediaPipe's own drawing threshold for "visible"
    assert len(FRAMINGS) == 5 and FRAMINGS[0] == (0.0, 0.0, 1.0)    # the still itself comes first
    assert MIN_SEEN == 3                        # a majority of the framings


def test_torso_visibility_is_the_least_seen_of_the_shoulders_and_hips():
    r = read(10, 90)
    r["visibility"]["left_hip"] = 0.4
    assert torso_visibility(r) == 0.4
    assert torso_visibility({"flat": {}}) is None                   # an older reading without visibility
    assert torso_visibility({"error": "no person found"}) is None


def test_summarize_takes_the_median_of_the_framings_seen():
    s = summarize([read(10.5, 99.0), read(11.3, 96.0), read(11.5, 94.5), read(12.0, 96.9), read(13.1, 94.0)])
    assert s["ok"] is True and s["reads"] == 5 and s["seen"] == 5 and s["why"] is None
    assert s["flat"]["torso_lean"] == approx(11.5) and s["flat"]["hip_included"] == approx(96.0)
    assert s["3d"]["elbow_included"] == approx(40.0)
    assert s["spread"]["torso_lean"] == approx([10.5, 13.1])
    assert s["visibility"]["torso"] == approx(1.0) and s["visibility"]["knee"] == approx(0.2)


def test_a_framing_whose_shoulders_or_hips_are_not_seen_is_left_out():
    bad = read(40.0, 40.0, vis=0.3)
    s = summarize([read(10.0, 90.0), bad, read(12.0, 92.0), read(11.0, 91.0), {"image": "y.png", "error": "no person"}])
    assert s["ok"] is True and s["seen"] == 3 and s["reads"] == 5
    assert s["flat"]["torso_lean"] == approx(11.0) and s["flat"]["hip_included"] == approx(91.0)


def test_too_few_framings_seen_is_not_a_reading():
    s = summarize([read(10.0, 90.0, vis=0.2), read(12.0, 92.0, vis=0.45), read(11.0, 91.0)])
    assert s["ok"] is False and s["seen"] == 1
    assert s["flat"] is None and s["3d"] is None
    assert s["why"] == "MediaPipe saw our shoulders and hips clearly in 1 of 3 framings"
    assert picture_value(s, "flat", "torso_lean") is None


def test_no_body_in_any_framing():
    s = summarize([{"image": "a.png", "error": "no person found in a.png"}] * 5)
    assert s["ok"] is False and s["seen"] == 0 and s["why"] == "MediaPipe found no body in our picture"
    assert summarize([])["why"] == "MediaPipe found no body in our picture"


def test_picture_value():
    s = summarize([read(10.0, 90.0)] * 5)
    assert picture_value(s, "flat", "torso_lean") == approx(10.0)
    assert picture_value(s, "3d", "hip_included") == approx(92.0)
    assert picture_value(s, "flat", "grip_ratio") is None
    assert picture_value(None, "flat", "torso_lean") is None
    assert picture_value({"ok": True, "flat": {"torso_lean": float("nan")}}, "flat", "torso_lean") is None


def test_a_missing_value_in_one_framing_does_not_drop_the_others():
    a, b, c = read(10.0, 90.0), read(12.0, 92.0), read(11.0, 91.0)
    b["flat"]["hip_included"] = None
    s = summarize([a, b, c])
    assert s["flat"]["hip_included"] == approx(90.5) and s["flat"]["torso_lean"] == approx(11.0)


def test_ours_squat_bottom_still_is_read_and_reads_upright():
    """The latest run's bottom still (task 19's starting point): MediaPipe sees our shoulders and hips
    clearly and reads the trunk 10.5 degrees forward, where the rig's line (counted from the bind pose)
    reads 22.1 and the coach 22.4. The knee is barely 'visible' (0.11), as the coach's own (0.03)."""
    from conftest import ROOT
    from read_pose import read_pose
    r = read_pose(ROOT / "studio/tests/fixtures/ours-squat-end-still.png")
    assert torso_visibility(r) >= 0.99
    assert r["flat"]["torso_lean"] == approx(10.5, abs=0.5)
    assert r["flat"]["hip_included"] == approx(99.0, abs=0.5)
    assert r["visibility"]["right_knee"] < SEEN_MIN
