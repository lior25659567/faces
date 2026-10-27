"""Actual old isolation defects must not pass by keeping contact with a weight."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'pipeline/blender'))
from isolation_poses import CURL, RAISE, PUSHDOWN, prepare
from isolation_range import summarize
from keypose_clip import motion


def angle(a, b):
    return float(np.degrees(np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1., 1.))))


def keys():
    return [{name: np.eye(3) for name in ('Hips', 'Spine', 'Spine1', 'Spine2', 'Head', 'Neck',
        'LeftArm', 'RightArm', 'LeftForeArm', 'RightForeArm', 'LeftHand', 'RightHand',
        'LeftUpLeg', 'RightUpLeg', 'LeftLeg', 'RightLeg', 'LeftFoot', 'RightFoot',
        'LeftToeBase', 'RightToeBase')} for _ in range(2)]


@pytest.mark.parametrize('exercise,start,end', [(CURL, 175., 55.), (PUSHDOWN, 90., 175.)])
def test_curl_and_pushdown_keep_the_upper_arms_fixed_and_complete_the_declared_range(exercise, start, end):
    original = keys()
    before = deepcopy(original)
    prepared = prepare(exercise, original)
    poses = motion(prepared, fps=12, feet='planted')
    for side, sign in (('Left', 1.), ('Right', -1.)):
        axis = np.array([sign, 0., 0.])
        angles = [angle(-(k[side + 'Arm'] @ axis), k[side + 'ForeArm'] @ axis) for k in poses]
        assert angles[0] == pytest.approx(start)
        assert angles[18] == pytest.approx(end)
        assert all(np.allclose(k[side + 'Arm'], poses[0][side + 'Arm']) for k in poses)
    assert all(np.array_equal(a[name], b[name]) for a, b in zip(original, before) for name in a)


def test_raise_keeps_its_elbow_bend_constant_through_the_entire_shoulder_height_path():
    prepared = prepare(RAISE, keys())
    poses = motion(prepared, fps=12, feet='planted')
    for side, sign in (('Left', 1.), ('Right', -1.)):
        axis = np.array([sign, 0., 0.])
        for key in poses:
            assert angle(-(key[side + 'Arm'] @ axis), key[side + 'ForeArm'] @ axis) == pytest.approx(165.)
        assert angle([0., 0., -1.], poses[0][side + 'Arm'] @ axis) == pytest.approx(10.)
        assert angle([0., 0., -1.], poses[18][side + 'Arm'] @ axis) == pytest.approx(90.)


def recorded(exercise):
    data = json.loads((Path(__file__).with_name('fixtures') / 'isolation-original-ranges.json').read_text())
    return next(value for value in data if value['exercise'] == exercise)


def recheck(record):
    return summarize(record['per_frame'], record['height_m'], record['phase_frames'], record['exercise'])


def test_actual_old_curl_exposes_the_crouch_upperarm_swing_and_incomplete_curl():
    result = recheck(recorded(CURL))
    assert result['knee_extension_shortfall_deg'] > 9.
    assert result['upperarm_drift_deg'] > 30.
    assert result['end_flexion_excess_deg'] > 15.


def test_actual_old_raise_exposes_elbow_bend_drift_and_above_shoulder_elevation():
    result = recheck(recorded(RAISE))
    assert result['end_elevation_excess_deg'] > 10.
    assert result['elbow_bend_variation_deg'] > 15.
    assert result['elbow_bend_excess_deg'] > 5.


def test_actual_old_pushdown_does_not_establish_full_extension():
    result = recheck(recorded(PUSHDOWN))
    assert result['end_extension_shortfall_deg'] > 12.


def test_raw_joints_take_precedence_over_forged_angles_or_success_summaries():
    record = recorded(CURL)
    record.update(ready=True, upperarm_drift_deg=0., end_flexion_excess_deg=0.)
    for sample in record['per_frame']:
        sample['elbows_deg'] = {'left': 55., 'right': 55.}
    assert recheck(record)['upperarm_drift_deg'] > 30.
    assert recheck(record)['end_flexion_excess_deg'] > 15.


@pytest.mark.parametrize('defect', ['missing_frame', 'duplicated_frame', 'missing_joint', 'nonfinite', 'wrong_phase'])
def test_missing_or_inconsistent_measurements_are_rejected(defect):
    record = recorded(CURL)
    if defect == 'missing_frame':
        record['per_frame'].pop(7)
    elif defect == 'duplicated_frame':
        record['per_frame'][8]['frame'] = 8
    elif defect == 'missing_joint':
        del record['per_frame'][8]['knees_m']['right']
    elif defect == 'nonfinite':
        record['per_frame'][8]['wrists_m']['left'][2] = float('nan')
    else:
        record['phase_frames']['end'] = 1000
    with pytest.raises(ValueError):
        recheck(record)


def test_remeasurement_never_mutates_the_saved_source_geometry():
    record = recorded(RAISE)
    before = deepcopy(record)
    recheck(record)
    assert record == before
