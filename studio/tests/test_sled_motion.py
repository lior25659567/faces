"""Support contact cannot certify a leg press whose rider rocks with the sled."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'pipeline/blender'))
from sled_motion import summarize, stationary_seat


def original():
    return json.loads((Path(__file__).with_name('fixtures') / 'sled-original-motion.json').read_text())


def measured(raw):
    return summarize(raw['per_frame'], raw['height_m'], raw['phase_frames'])


def test_actual_original_rider_moves_despite_a_good_leg_range():
    result = measured(original())
    assert result['max_supported_body_drift_mm'] > 26.
    assert result['max_hand_drift_mm'] > 60.
    assert result['top_extension_shortfall_deg'] == 0.
    assert result['bottom_flexion_shortfall_deg'] == 0.
    assert result['bottom_overfold_deg'] == 0.
    assert result['max_knee_asymmetry_deg'] < 1.


def test_stationary_supported_body_has_no_drift_without_changing_measured_leg_motion():
    raw = original()
    start = raw['per_frame'][0]['points_m']
    for sample in raw['per_frame']:
        for part in ('hip', 'shoulder', 'wrist', 'head'):
            sample['points_m'][part] = deepcopy(start[part])
    result = measured(raw)
    assert result['max_supported_body_drift_mm'] == 0.
    assert result['max_hand_drift_mm'] == 0.
    assert result['top_extension_shortfall_deg'] == 0.


@pytest.mark.parametrize('defect', ['missing_frame', 'duplicate', 'missing_arm', 'nonfinite', 'phase'])
def test_missing_joint_coverage_or_wrong_phase_cannot_pass(defect):
    raw = original()
    if defect == 'missing_frame':
        raw['per_frame'].pop(5)
    elif defect == 'duplicate':
        raw['per_frame'][5] = deepcopy(raw['per_frame'][4])
    elif defect == 'missing_arm':
        del raw['per_frame'][5]['points_m']['wrist']['right']
    elif defect == 'nonfinite':
        raw['per_frame'][5]['points_m']['head'][0] = float('nan')
    else:
        raw['phase_frames']['end'] = 0
    with pytest.raises(ValueError):
        measured(raw)


def test_measuring_raw_samples_does_not_modify_them_or_trust_saved_maxima():
    raw = original()
    before = deepcopy(raw)
    raw['max_supported_body_drift_mm'] = raw['max_hand_drift_mm'] = 0.
    result = measured(raw)
    assert result['max_hand_drift_mm'] > 60.
    del raw['max_supported_body_drift_mm'], raw['max_hand_drift_mm']
    assert raw == before


def test_one_stationary_posterior_pad_keeps_contact_despite_anterior_thigh_deformation():
    posterior = np.array([[x, y, .20] for x in (-.04, 0., .04) for y in (-.12, -.11, -.10)])
    anterior = np.array([[x, y, .15] for x in (-.04, 0., .04) for y in (.0, .01, .02)])
    frames = np.array([np.vstack([posterior, anterior + [0., 0., shift]]) for shift in (0., .02, .04)])
    original = frames.copy()
    axis = np.array([0., -1., .2])
    seat = stationary_seat(frames, 1., .18, axis)
    assert seat['centre_m'][1] < -.05
    assert .195 <= seat['centre_m'][2] <= .203
    assert seat['frames'] == 3 and seat['max_gap_mm'] <= 5. and seat['max_penetration_mm'] <= 3.
    np.testing.assert_array_equal(frames, original)
    np.testing.assert_array_equal(axis, [0., -1., .2])


def test_a_body_that_lifts_off_every_possible_fixed_pad_is_rejected():
    patch = np.array([[x, y, .20] for x in (-.04, 0., .04) for y in (-.04, 0., .04)])
    frames = np.array([patch, patch + [0., 0., .03]])
    with pytest.raises(ValueError, match='fixed pelvis pad'):
        stationary_seat(frames, 1., .18, [0., -1., .2])
