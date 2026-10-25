"""Estimated positions keep rig measurements without an unjudged camera search."""
from copy import deepcopy
from picture import comparison_frames


def test_only_estimated_middle_avoids_the_picture_search():
    frames = {'start': 1, 'middle': 24, 'end': 46}
    targets = {'phases': {'middle': {'estimated': True}, 'start': {}, 'end': {}}}
    original = deepcopy((frames, targets))
    assert comparison_frames(frames, targets) == {'start': 1, 'end': 46}
    assert (frames, targets) == original


def test_measured_middle_and_legacy_targets_keep_all_picture_readings():
    frames = {'start': 1, 'middle': 24, 'end': 46}
    for targets in (None, {}, {'phases': {}}, {'phases': {'middle': {'estimated': False}}}):
        assert comparison_frames(frames, targets) == frames


def test_missing_middle_does_not_invent_a_new_frame():
    assert comparison_frames({'start': 7}, {'phases': {'middle': {'estimated': True}}}) == {'start': 7}
