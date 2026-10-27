"""A toe touching the floor cannot certify a flat planted bench-press foot."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'pipeline/blender'))
from bench_feet import floor_report


def feet():
    # Distinct three-vertex heel/toe patches on each fitted rest sole.
    left = np.array([[-.1, y, 0.] for y in (0., .01, .02, .08, .09, .1)])
    right = left + np.array([.2, 0., 0.])
    rest = np.vstack([left, right])
    masks = {'left': np.arange(12) < 6, 'right': np.arange(12) >= 6}
    return rest, masks


def test_toe_only_contact_exposes_the_lifted_heel():
    rest, masks = feet()
    raised = rest.copy()
    raised[:3, 2] = .015
    report = floor_report(rest, masks, [(1, rest), (2, raised), (3, rest)])
    assert report['surfaces']['foot_left_heel']['max_gap_mm'] == 15.
    assert report['surfaces']['foot_left_toe']['max_gap_mm'] == 0.
    assert report['surfaces']['foot_left_heel']['measured_frames'] == 3


def test_the_entire_patch_and_all_frames_are_measured():
    rest, masks = feet()
    changed = rest.copy()
    changed[0, 2] = -.004  # Nearby heel vertices still touch, but one penetrates.
    changed[6:, 0] += .01  # The other sole slides without lifting.
    report = floor_report(rest, masks, [(7, rest), (8, changed), (9, rest)])
    assert report['surfaces']['foot_left_heel']['max_penetration_mm'] == 4.
    assert report['planted_feet']['max_slide_mm'] == pytest.approx(10.)
    assert report['planted_feet']['first_frame'] == 7
    assert report['planted_feet']['last_frame'] == 9


@pytest.mark.parametrize('frames', [[], [(1, None)], [(1, 'rest'), (3, 'rest')]])
def test_incomplete_or_unreadable_floor_samples_are_rejected(frames):
    rest, masks = feet()
    frames = [(frame, rest if isinstance(points, str) else points) for frame, points in frames]
    with pytest.raises(ValueError):
        floor_report(rest, masks, frames)


def test_missing_heel_or_toe_surface_is_not_a_measured_contact():
    rest, masks = feet()
    rest[:6, 1] = 0.
    with pytest.raises(ValueError, match='distinct heel and toe'):
        floor_report(rest, masks, [(1, rest)])
