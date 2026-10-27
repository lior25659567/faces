"""Arm snaps cannot hide behind stationary hips and knees or a cached pass."""
from copy import deepcopy
import math
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'pipeline/blender'))
from upper_limb_motion import POINTS, TURNS, summarize, remeasure
from checks import _upper_limb_rows
from skin_preflight import upper_limb_verdict


def samples(count=8, displacement=.08):
    return [{'frame': f + 1,
             'points_m': {name: [0., displacement * math.sin(f * 2 * math.pi / count), .5] for name in POINTS},
             'turns_wxyz': {name: [1., 0., 0., 0.] for name in TURNS}} for f in range(count)]


def checked(data=None, fps=16., height=1., loop=True):
    data = samples() if data is None else data
    return summarize(data, height, fps, 1, len(data), loop=loop)


def rows(value):
    return {r['metric']: r for r in _upper_limb_rows({'first': 1, 'last': value['last_frame'],
                                                    'upper_limb_motion': value})}


def test_ordinary_motion_and_static_arms_pass_without_claiming_form_approval():
    for value in (checked(), checked(samples(displacement=0.))):
        result = rows(value)
        assert all(r['status'] == 'pass' for r in result.values())
        assert all('does not establish exercise technique' in r['note'] for r in result.values())


def test_elbow_branch_flip_fails_even_with_stationary_hips_knees_and_wrists():
    data = samples(displacement=0.)
    data[4]['points_m']['elbow.right'][1] += .2
    result = rows(checked(data))
    assert result['upper_limb_speed']['status'] == 'fail'
    assert result['upper_limb_speed']['ours'] == pytest.approx(3.2)
    assert result['upper_limb_speed']['frame'] == 5


def test_twisted_forearm_rotation_fails_even_when_joint_positions_are_unchanged():
    data = samples(displacement=0.)
    data[4]['turns_wxyz']['forearm.left'] = [.5, math.sqrt(3.) / 2, 0., 0.]
    result = rows(checked(data))
    assert result['upper_limb_speed']['status'] == 'pass'
    assert result['upper_limb_turn_speed']['status'] == 'fail'
    assert result['upper_limb_turn_speed']['ours'] == pytest.approx(1920.)


def test_quaternion_hemisphere_change_is_not_a_physical_spin():
    data = samples(displacement=0.)
    data[4]['turns_wxyz']['forearm.left'] = [-1., 0., 0., 0.]
    value = checked(data)
    assert value['max_turn_deg_s'] == 0.


def test_speeds_use_character_height_and_actual_playback_rate():
    data = samples()
    base = checked(data)
    scaled = deepcopy(data)
    for sample in scaled:
        sample['points_m'] = {name: [2 * v for v in point] for name, point in sample['points_m'].items()}
    assert checked(scaled, height=2.)['max_speed_heights_s'] == pytest.approx(base['max_speed_heights_s'])
    assert checked(data, fps=32.)['max_speed_heights_s'] == pytest.approx(2 * base['max_speed_heights_s'])


def test_closed_cycle_includes_last_to_first_transition():
    data = samples(count=6, displacement=0.)
    for index, sample in enumerate(data):
        sample['points_m']['wrist.left'][1] = index * .06
    assert checked(data, fps=10., loop=False)['max_speed_heights_s'] == pytest.approx(.6)
    result = rows(checked(data, fps=10., loop=True))
    assert result['upper_limb_speed']['status'] == 'fail'
    assert result['upper_limb_speed']['frame'] == 1


@pytest.mark.parametrize('defect', ['missing_frame', 'duplicate_frame', 'missing_arm', 'nonfinite', 'invalid_rotation'])
def test_missing_or_invalid_raw_measurements_cannot_retain_a_good_summary(defect):
    value = checked()
    if defect == 'missing_frame':
        value['per_frame'].pop()
    elif defect == 'duplicate_frame':
        value['per_frame'][3]['frame'] = 3
    elif defect == 'missing_arm':
        del value['per_frame'][3]['points_m']['elbow.right']
    elif defect == 'nonfinite':
        value['per_frame'][3]['points_m']['elbow.right'][0] = float('nan')
    else:
        value['per_frame'][3]['turns_wxyz']['arm.left'] = [0., 0., 0., 0.]
    assert all(r['kind'] == 'not-measured' for r in rows(value).values())


def test_cached_maximum_and_limits_are_recomputed_from_actual_arm_samples():
    value = checked(samples(displacement=0.))
    value['per_frame'][4]['points_m']['elbow.right'][1] = .2
    value['max_speed_heights_s'] = 0.
    value['limits']['speed_heights_s'] = 100.
    assert rows(value)['upper_limb_speed']['status'] == 'fail'


@pytest.mark.parametrize('height,fps', [(True, 16.), (1., True), (0., 16.), (1., float('inf'))])
def test_invalid_scale_or_playback_rate_is_rejected(height, fps):
    with pytest.raises(ValueError):
        checked(height=height, fps=fps)


def test_pre_render_and_final_checks_share_identical_arm_verdicts():
    value = checked()
    verdict, ready = upper_limb_verdict(value, 1, value['last_frame'])
    assert ready
    assert verdict == list(rows(value).values())
    value['per_frame'][4]['points_m']['elbow.right'][1] += .2
    verdict, ready = upper_limb_verdict(value, 1, value['last_frame'])
    assert not ready
    assert verdict == list(rows(value).values())
    verdict, ready = upper_limb_verdict(None, 1, 8)
    assert not ready and all(r['kind'] == 'not-measured' for r in verdict)


def test_inconsistent_report_coverage_cannot_certify_motion():
    value = checked()
    value['measured_frames'] -= 1
    with pytest.raises(ValueError):
        remeasure(value, 1, value['last_frame'])
