"""An overhead handle position alone cannot certify a complete press."""
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'pipeline/blender'))
from overhead_range import summarize, progress_at


def samples():
    records = []
    for frame, p in enumerate((0., .5, 1., 1., .6, .1), 1):
        points = {key: {} for key in ('shoulders_m', 'hips_m', 'elbows_m', 'wrists_m', 'handles_m')}
        for side, sign in (('left', -1), ('right', 1)):
            shoulder = [sign * .15, 0., .6]
            elbow = [shoulder[0] + sign * .2 * math.cos(math.pi / 2 * p), 0., .6 + .2 * math.sin(math.pi / 2 * p)]
            wrist = [elbow[0], 0., elbow[2] + .16]
            points['shoulders_m'][side] = shoulder
            points['hips_m'][side] = [sign * .1, 0., .3]
            points['elbows_m'][side] = elbow
            points['wrists_m'][side] = wrist
            points['handles_m'][side] = [wrist[0], wrist[1], wrist[2] + .05]
        records.append({'frame': frame, **points, 'hip_m': [0., 0., .3],
            'spine_m': [0., 0., .5], 'head_m': [0., 0., .8], 'head_top_m': .96})
    return records, {'start': 1, 'middle': 2, 'end': 3}


def test_complete_press_reaches_lockout_and_preserves_raw_measurements():
    raw, phases = samples()
    old = deepcopy(raw)
    report = summarize(raw, 1., phases)
    assert report['top_elbows_deg'] == {'left': 180., 'right': 180.}
    assert report['bottom_elbows_deg'] == {'left': 90., 'right': 90.}
    for name in ('lockout_shortfall_deg', 'bottom_flexion_excess_deg', 'bottom_overfold_deg',
                 'elbow_above_wrist_mm', 'travel_shortfall_mm', 'peak_stack_excess_mm',
                 'lockout_head_shortfall_mm', 'elbow_asymmetry_deg', 'hip_drift_mm', 'torso_lean_deg'):
        assert report[name] == pytest.approx(0.)
    assert raw == old


def test_high_handles_with_bent_elbows_are_not_lockout():
    raw, phases = samples()
    for side in ('left', 'right'):
        raw[2]['wrists_m'][side][1] = .12
    result = summarize(raw, 1., phases)
    assert result['lockout_shortfall_deg'] > 30.
    assert result['lockout_head_shortfall_mm'] == 0.


def test_each_arm_and_each_recorded_phase_must_pass():
    raw, phases = samples()
    raw[2]['wrists_m']['left'][1] = .12
    raw[0]['wrists_m']['right'][2] = raw[0]['elbows_m']['right'][2] - .1
    result = summarize(raw, 1., phases)
    assert result['lockout_shortfall_deg'] > 30.
    assert result['elbow_asymmetry_deg'] > 30.
    assert result['elbow_above_wrist_mm'] == pytest.approx(100.)


@pytest.mark.parametrize('defect', ['missing', 'duplicate', 'nonfinite', 'missing_arm', 'phase'])
def test_incomplete_or_invalid_geometry_cannot_pass(defect):
    raw, phases = samples()
    if defect == 'missing': raw.pop(1)
    elif defect == 'duplicate': raw[2]['frame'] = 2
    elif defect == 'nonfinite': raw[2]['handles_m']['left'][2] = float('nan')
    elif defect == 'missing_arm': del raw[2]['wrists_m']['right']
    else: phases['end'] = len(raw) + 1
    with pytest.raises(ValueError):
        summarize(raw, 1., phases)


def test_saved_or_fabricated_elbow_angles_cannot_override_actual_joints():
    raw, phases = samples()
    for side in ('left', 'right'):
        raw[2]['wrists_m'][side][1] = .12
    raw[2]['elbows_deg'] = {'left': 180., 'right': 180.}
    assert summarize(raw, 1., phases)['lockout_shortfall_deg'] > 30.


def test_reclined_torso_cannot_be_hidden_by_raised_shoulders():
    raw, phases = samples()
    for sample in raw:
        for side in ('left', 'right'):
            sample['shoulders_m'][side][1] = .3
    assert summarize(raw, 1., phases)['torso_lean_deg'] == pytest.approx(45.)


@pytest.mark.parametrize('profile,minimum_shortfall', [('standing', 16.), ('seated', 29.)])
def test_actual_original_press_frames_fail_lockout_despite_overhead_handles(profile, minimum_shortfall):
    fixture = Path(__file__).with_name('fixtures') / f'overhead-original-{profile}.json'
    raw = json.loads(fixture.read_text())
    result = summarize(raw['per_frame'], raw['height_m'], raw['phase_frames'])
    assert result['lockout_shortfall_deg'] > minimum_shortfall
    assert result['lockout_head_shortfall_mm'] == 0.
    if profile == 'seated':
        assert result['torso_lean_deg'] > 30.


def test_tempo_has_a_real_hold_and_controlled_return_without_a_loop_snap():
    phases = {'start': 1, 'middle': 7, 'end': 13}
    p = [progress_at(f, 1, 32, phases) for f in range(1, 33)]
    assert p[0] == 0.
    assert p[12:15] == [1., 1., 1.]
    assert 0. < p[-1] < .02
    assert all(a <= b for a, b in zip(p[:12], p[1:13]))
    assert all(a >= b for a, b in zip(p[14:-1], p[15:]))
