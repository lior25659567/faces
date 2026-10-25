"""A generated deadlift cannot use source-depth uncertainty to excuse bad form."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'pipeline/blender'))
from deadlift_range import measure_frame, summarize


def joints(hinged=False, wide=False):
    result = {}
    for side, sign in [('left', -1), ('right', 1)]:
        x = sign * .1
        spread = .2 if wide else 0.
        result[side] = {
            'hip': [x, -.1 if hinged else 0., .5],
            'shoulder': [sign * .16, .16 if hinged else 0., .75 if hinged else .8],
            'elbow': [sign * .16, .16 if hinged else 0., .50 if hinged else .55],
            'wrist': [sign * .16, .16 if hinged else 0., .25 if hinged else .3],
            'knee': [x + sign * spread, .1 if hinged else 0., .28],
            'ankle': [x + sign * spread * 2, 0., .06]}
    return result


def valid_samples():
    start = measure_frame(1, joints(hinged=True), [0., .16, .15])
    end = measure_frame(3, joints(), [0., 0., .4])
    middle = copy.deepcopy(start)
    middle['frame'] = 2
    middle['bar_height_m'] = .25
    return [start, middle, end]


def test_complete_deadlift_has_straight_arms_and_simultaneous_lockout():
    report = summarize(valid_samples(), 1., {'start': 1, 'end': 3})
    assert report['ready']
    assert report['minimum_elbow_deg'] == pytest.approx(180.)
    assert report['start_bar_below_knees_mm'] == pytest.approx(130.)
    assert report['forward_bar_travel_mm'] == pytest.approx(250.)


def test_outward_stance_is_not_mistaken_for_hip_flexion():
    narrow = measure_frame(1, joints(), [0., 0., .3])
    wide = measure_frame(1, joints(wide=True), [0., 0., .3])
    assert wide['hips_deg'] == pytest.approx(narrow['hips_deg'])
    assert min(wide['hips_deg'].values()) == pytest.approx(180.)


def test_world_translation_and_body_yaw_preserve_angles_and_relative_bar_height():
    points = joints(hinged=True, wide=True)
    bar = np.array([0., .16, .15])
    original = measure_frame(1, points, bar)
    a = .82
    rotation = np.array([[np.cos(a), -np.sin(a), 0.], [np.sin(a), np.cos(a), 0.], [0., 0., 1.]])
    offset = np.array([2., -3., 5.])
    transformed = {side: {name: rotation @ point + offset for name, point in values.items()}
                   for side, values in points.items()}
    moved = measure_frame(1, transformed, rotation @ bar + offset)
    for key in ('elbows_deg', 'knees_deg', 'hips_deg'):
        assert moved[key] == pytest.approx(original[key])
    assert moved['torso_lean_deg'] == pytest.approx(original['torso_lean_deg'])
    assert moved['bar_below_knees_m'] == pytest.approx(original['bar_below_knees_m'])


def test_bent_intermediate_lifting_arm_blocks_otherwise_good_endpoints():
    samples = valid_samples()
    samples[1]['elbows_deg']['left'] = 120.
    report = summarize(samples, 1., {'start': 1, 'end': 3})
    assert not report['ready']
    assert report['failures'] == ['arms_straight_throughout']
    assert report['worst_elbow_frame'] == 2


def test_hips_and_knees_must_lock_out_together_at_recorded_end_phase():
    samples = valid_samples()
    samples[1]['hips_deg'] = {'left': 180., 'right': 180.}
    samples[1]['knees_deg'] = {'left': 130., 'right': 130.}
    samples[2]['hips_deg'] = {'left': 145., 'right': 145.}
    report = summarize(samples, 1., {'start': 1, 'end': 3})
    assert report['failures'] == ['hip_lockout']
    assert report['lockout_frame'] == 3


def test_recorded_source_phase_cannot_be_replaced_by_later_better_pose():
    samples = valid_samples()
    samples[1]['hips_deg'] = {'left': 140., 'right': 140.}
    report = summarize(samples, 1., {'start': 1, 'end': 2})
    assert 'hip_lockout' in report['failures']
    assert report['lockout_frame'] == 2


def test_high_bar_rack_pull_and_trivial_motion_do_not_pass_conventional_range():
    samples = valid_samples()
    samples[0]['bar_below_knees_m'] = -.12
    samples[2]['bar_height_m'] = samples[0]['bar_height_m'] + .03
    report = summarize(samples, 1., {'start': 1, 'end': 3})
    assert report['failures'] == ['start_bar_below_knees', 'forward_bar_travel']


def test_standing_arm_lift_is_not_a_deadlift_start():
    samples = valid_samples()
    samples[0]['hips_deg'] = {'left': 180., 'right': 180.}
    samples[0]['knees_deg'] = {'left': 180., 'right': 180.}
    report = summarize(samples, 1., {'start': 1, 'end': 3})
    assert report['failures'] == ['start_hip_hinge', 'start_knee_bend']


def test_inside_leg_grip_is_not_certified_as_a_conventional_deadlift():
    samples = valid_samples()
    points = joints(hinged=True)
    for side, sign in [('left', -1), ('right', 1)]:
        for joint in ('shoulder', 'elbow', 'wrist'):
            points[side][joint][0] = sign * .08
    samples[0] = measure_frame(1, points, [0., .16, .15])
    report = summarize(samples, 1., {'start': 1, 'end': 3})
    assert report['start_grip_outside_knees_mm'] == pytest.approx(-20.)
    assert report['failures'] == ['conventional_grip_outside_knees']


@pytest.mark.parametrize('phases', [{}, {'start': 1, 'end': 99}, {'start': 3, 'end': 1}, {'start': True, 'end': 3}])
def test_unverifiable_phase_timing_cannot_produce_a_form_pass(phases):
    with pytest.raises(ValueError):
        summarize(valid_samples(), 1., phases)


def test_skipped_duplicate_and_nonfinite_samples_cannot_produce_a_form_pass():
    samples = valid_samples()
    with pytest.raises(ValueError):
        summarize([samples[0], samples[2]], 1., {'start': 1, 'end': 3})
    with pytest.raises(ValueError):
        summarize([samples[0], samples[0], samples[2]], 1., {'start': 1, 'end': 3})
    samples[1]['knees_deg']['left'] = float('nan')
    with pytest.raises(ValueError):
        summarize(samples, 1., {'start': 1, 'end': 3})


def test_missing_body_side_and_degenerate_bones_are_rejected():
    with pytest.raises(ValueError):
        measure_frame(1, {'left': joints()['left']}, [0., 0., .2])
    points = joints()
    points['left']['elbow'] = points['left']['shoulder']
    with pytest.raises(ValueError):
        measure_frame(1, points, [0., 0., .2])
