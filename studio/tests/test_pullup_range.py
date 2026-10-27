"""Incomplete pull-ups must not be certified by contact alone."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'pipeline/blender'), str(ROOT / 'studio/measure')]
from pullup_range import chin_region, included, summarize, elbow_width_timeline, grip_candidates, pole_summary
from physical_checks import rows_for
from row_motion import solve_contact


def sample(frame, chin, left, right):
    return {'frame': frame, 'chin_clearance_mm': chin, 'elbows_deg': {'left': left, 'right': right},
            'hip_height_m': 1. - min(left, right) / 180.,
            'elbow_bar_clearance_mm': {'left': 20., 'right': 20.},
            'elbow_wrist_clearance_mm': {'left': 10., 'right': 10.},
            'elbow_shoulder_clearance_mm': {'left': -100. if min(left, right) > 170. else 10.,
                                             'right': -100. if min(left, right) > 170. else 10.}}


def report(samples):
    motion = summarize(samples)
    return {'frames': len(samples), 'physical': {'profile': {'family': 'hanging'}, 'contacts': {
        'frames': len(samples), 'pullup_range': motion, 'fixed_bar_drift_mm': 0., 'fixed_bar_rotation_deg': 0.,
        'hang_extension': {'max_body_drop_mm': 3., 'max_body_raise_mm': 0.}}}}


def range_rows(value):
    return {row['metric']: row for row in rows_for(value) if row['metric'].startswith('pullup_')}


def test_dead_hang_and_chin_clearance_are_required_at_different_frames():
    value = report([sample(1, -100., 176., 179.), sample(19, 4., 58., 60.)])
    checks = range_rows(value)
    assert all(row['status'] == 'pass' for metric, row in checks.items()
               if metric not in ('pullup_elbow_path_speed', 'pullup_return_path_error'))
    assert checks['pullup_elbow_path_speed']['kind'] == 'not-measured'
    assert checks['pullup_return_path_error']['kind'] == 'not-measured'
    assert value['physical']['contacts']['pullup_range']['top_frame'] == 19
    assert value['physical']['contacts']['pullup_range']['bottom_frame'] == 1
    assert checks['pullup_extension_shortfall']['unit'] == 'deg'


def test_bent_bottom_is_rejected_despite_top_chin_clearance():
    checks = range_rows(report([sample(1, -100., 152., 153.), sample(19, 8., 59., 60.)]))
    assert checks['pullup_chin_shortfall']['status'] == 'pass'
    assert checks['pullup_extension_shortfall']['status'] == 'fail'
    assert checks['pullup_extension_shortfall']['ours'] == 28.


def test_both_arms_must_extend_together():
    checks = range_rows(report([sample(1, 1., 179., 170.), sample(2, 1., 170., 179.)]))
    assert checks['pullup_extension_shortfall']['status'] == 'fail'


def test_head_above_bar_is_insufficient_if_chin_remains_below_it():
    checks = range_rows(report([sample(1, -.01, 179., 179.)]))
    assert checks['pullup_chin_shortfall']['status'] == 'fail'


def test_unmeasured_or_incomplete_rom_does_not_pass():
    value = report([sample(1, 1., 179., 179.)])
    value['physical']['contacts']['pullup_range']['per_frame'] = []
    checks = range_rows(value)
    assert checks['pullup_chin_shortfall']['kind'] == 'not-measured'
    assert checks['pullup_extension_shortfall']['kind'] == 'not-measured'
    checks = range_rows({'physical': {'profile': {'family': 'hanging'}}})
    assert all(row['status'] == 'info' for row in checks.values())


@pytest.mark.parametrize('value', [None, True, float('nan'), float('inf')])
def test_invalid_range_samples_are_rejected(value):
    point = sample(1, 0., 180., 180.)
    point['elbows_deg']['right'] = value
    with pytest.raises(ValueError):
        summarize([point])


def test_chin_proxy_uses_lower_front_face_and_excludes_neck_ears_and_hair():
    # The head-owned neck is lower, but farther back. The nose and hair are high;
    # the ear is lateral. A torso point is not head owned even when in the band.
    points = np.array([[0., .08, .8], [.001, .08, .801], [0., .0, .74],
                       [0., .12, .9], [0., -.10, 1.], [.1, .08, .78],
                       [-.1, .08, .78], [0., .08, .7]])
    indices = chin_region(points, np.array([True] * 7 + [False]))
    assert set(indices) == {0, 1}
    assert included(np.array([0., 1., 0.]), np.zeros(3), np.array([0., -1., 0.])) == pytest.approx(180.)


def test_a_moving_bar_and_excessive_body_adaptation_are_rejected():
    value = report([sample(1, 1., 179., 179.)])
    contacts = value['physical']['contacts']
    contacts.update(fixed_bar_drift_mm=2., fixed_bar_rotation_deg=.2,
                    hang_extension={'max_body_drop_mm': 21., 'max_body_raise_mm': 21.})
    checks = range_rows(value)
    assert all(checks[name]['status'] == 'fail' for name in
               ('pullup_bar_drift', 'pullup_bar_rotation', 'pullup_hang_body_adaptation', 'pullup_hang_raise_adaptation'))


def test_photographed_widths_follow_source_positions_hold_and_return():
    widths = {'start': {'elbow_width_ratio': {'value': 1.616, 'image': 'start'}},
              'end': {'elbow_width_ratio': {'value': 2.048, 'image': 'end'}}}
    fractions = [0., .5, 1., 1., .5, 0.]
    values = elbow_width_timeline(widths, {'start': 1, 'end': 3}, fractions)
    assert values.tolist() == pytest.approx([1.616, 1.832, 2.048, 2.048, 1.832, 1.616])
    assert elbow_width_timeline(None, None, fractions) is None


def test_widths_require_recorded_distinct_photographed_positions():
    widths = {'start': {'value': 1.616}, 'end': {'value': 2.048}}
    with pytest.raises(ValueError):
        elbow_width_timeline(widths, None, [0., 1.])
    with pytest.raises(ValueError):
        elbow_width_timeline(widths, {'start': 1, 'end': 8}, [0., 1.])
    with pytest.raises(ValueError):
        elbow_width_timeline(widths, {'start': 1, 'end': 2}, [0., 0.])


def test_source_elbow_width_can_choose_the_ik_pole_without_moving_the_hand_or_stretching():
    grip = {'B': [0., .08, 0.], 'wrist': [0., 0., 0.], 'kb': [1., 0., 0.], 'hb': [0., 1., 0.]}
    shoulder, target = np.array([0., 0., 1.]), np.array([.05, .2, .5])
    solution = solve_contact(shoulder, target, [1., 0., 0.], -1, .3, .3, grip, [0., .05, .75],
        preference={'elbow_lateral': .1, 'lateral_axis': [1., 0., 0.], 'lateral_scale': .003})
    assert solution['elbow'][0] == pytest.approx(.1, abs=.005)
    assert np.linalg.norm(solution['elbow'] - shoulder) == pytest.approx(.3)
    assert np.linalg.norm(solution['wrist'] - solution['elbow']) == pytest.approx(.3)
    np.testing.assert_allclose(solution['wrist'] + solution['turn'] @ grip['B'], target, atol=1e-9)
    assert solution['bend_deg'] <= 25.


def test_the_dead_hang_minimum_is_a_constraint_even_with_a_preferred_elbow_pose():
    grip = {'B': [0., .08, 0.], 'wrist': [0., 0., 0.], 'kb': [1., 0., 0.], 'hb': [0., 1., 0.]}
    solution = solve_contact([0., 0., 1.], [0., 0., .321], [1., 0., 0.], -1, .3, .3, grip, [0., .05, .75],
        preference={'elbow_deg': 176., 'min_elbow_deg': 175.})
    assert solution['elbow_deg'] >= 175.
    with pytest.raises(ValueError):
        solve_contact([0., 0., 1.], [0., 0., .45], [1., 0., 0.], -1, .3, .3, grip, [0., .05, .75],
            preference={'elbow_deg': 176., 'min_elbow_deg': 175.})


def test_bounded_grip_fit_is_nearest_first_and_never_changes_the_source_tolerance():
    values = grip_candidates(1.762, .012)
    assert values == pytest.approx([1.762, 1.767, 1.757, 1.772, 1.752, 1.774, 1.750])
    assert max(abs(value - 1.762) for value in values) <= .012 + 1e-12
    assert grip_candidates(1.762) == [1.762]


@pytest.mark.parametrize('requested,tolerance', [(0., .12), (1., -.1), (True, .12), (1., float('nan'))])
def test_invalid_grip_fit_limits_are_rejected(requested, tolerance):
    with pytest.raises(ValueError):
        grip_candidates(requested, tolerance)


def test_inverted_elbows_fail_even_when_the_chin_grip_and_dead_hang_pass():
    bottom, top = sample(1, -100., 176., 179.), sample(19, 11., 57., 58.)
    top['elbow_bar_clearance_mm'] = {'left': -66., 'right': -65.}
    top['elbow_wrist_clearance_mm'] = {'left': -45., 'right': -44.}
    top['elbow_shoulder_clearance_mm'] = {'left': -137., 'right': -136.}
    checks = range_rows(report([bottom, top]))
    assert checks['pullup_chin_shortfall']['status'] == 'pass'
    assert checks['pullup_extension_shortfall']['status'] == 'pass'
    assert all(checks[metric]['status'] == 'fail' for metric in
               ('pullup_elbow_above_bar', 'pullup_elbow_above_wrist', 'pullup_peak_elbow_above_shoulder'))


def test_legacy_range_without_elbow_pole_measurements_is_unmeasured():
    value = report([sample(1, -100., 176., 179.), sample(19, 11., 57., 58.)])
    value['physical']['contacts']['pullup_range'].pop('elbow_pole')
    checks = range_rows(value)
    assert checks['pullup_elbow_above_bar']['kind'] == 'not-measured'
    assert checks['pullup_peak_elbow_above_shoulder']['kind'] == 'not-measured'


def test_peak_shoulder_pole_is_measured_at_the_highest_body_position():
    bottom, top = sample(1, -100., 176., 179.), sample(19, 11., 57., 58.)
    result = pole_summary([bottom, top])
    assert result['peak_frame'] == 19
    assert result['peak_elbow_above_shoulder_mm'] == 0.
    top['elbow_wrist_clearance_mm']['right'] = float('nan')
    with pytest.raises(ValueError):
        pole_summary([bottom, top])


def test_actual_former_pullup_branch_flip_fails_motion_continuity_and_retrace():
    import json
    from pullup_range import motion_summary
    fixture = json.loads((Path(__file__).with_name('fixtures') / 'pullup-branch-regression.json').read_text())
    result = motion_summary(fixture['samples'], fixture['fps'], fixture['phase_frames'])
    assert result['max_elbow_speed_arm_lengths_per_s'] > 6.
    assert result['max_return_path_error_arm_lengths'] > .12


def smooth_path(fractions=None):
    fractions = fractions or [0., .1, .5, .9, 1., .9, .5, .1, 0.]
    return [{'frame': i + 1, 'hip_height_m': 1. + fraction * .3,
             'elbow_offsets_m': {side: [sign * .05, .2 * np.sin(.3 + fraction), .2 * np.cos(.3 + fraction)]
                                  for side, sign in (('left', -1), ('right', 1))}}
            for i, fraction in enumerate(fractions)]


def test_same_smooth_path_can_have_different_pull_and_lowering_tempo():
    from pullup_range import motion_summary
    samples = smooth_path([0., .2, .65, 1., 1., .9, .7, .4, .1, 0.])
    result = motion_summary(samples, 8., {'start': 1, 'end': 4})
    assert result['frames'] == len(samples)
    assert result['max_elbow_speed_arm_lengths_per_s'] < 6.
    assert result['max_return_path_error_arm_lengths'] < .04


def test_a_second_elbow_branch_on_lowering_cannot_hide_behind_good_endpoints():
    from pullup_range import motion_summary
    samples = smooth_path()
    for sample in samples[5:8]:
        for point in sample['elbow_offsets_m'].values():
            point[1] *= -1
    result = motion_summary(samples, 8., {'start': 1, 'end': 5})
    assert result['max_return_path_error_arm_lengths'] > .12


@pytest.mark.parametrize('defect', ['missing', 'nan', 'duplicate', 'fps', 'phase', 'stretch'])
def test_motion_path_requires_complete_measured_geometry(defect):
    from pullup_range import motion_summary
    samples, fps, phases = smooth_path(), 8., {'start': 1, 'end': 5}
    if defect == 'missing': del samples[1]['elbow_offsets_m']['left']
    if defect == 'nan': samples[1]['elbow_offsets_m']['left'][0] = float('nan')
    if defect == 'duplicate': samples[1]['frame'] = 1
    if defect == 'fps': fps = True
    if defect == 'phase': phases['end'] = 4
    if defect == 'stretch': samples[1]['elbow_offsets_m']['left'][0] = 1.
    with pytest.raises(ValueError):
        motion_summary(samples, fps, phases)


def test_continuous_pullup_pole_does_not_swap_the_elbow_depth_branch_near_the_top():
    from pullup_range import continuous_pole
    shoulders = {'left': np.array([-.1, 0., 1.]), 'right': np.array([.1, 0., 1.])}
    lengths = {side: [.2, .15] for side in shoulders}
    before = continuous_pole(shoulders, [0., .1, 1.1], [1., 0., 0.], 2., .999, lengths, 1.)
    top = continuous_pole(shoulders, [0., .1, 1.1], [1., 0., 0.], 2., 1., lengths, 1.)
    for side in shoulders:
        assert before[0][side][1] > shoulders[side][1]
        assert top[1][side]['min_elbow_y'] == pytest.approx(.035)
        assert abs(before[1][side]['max_elbow_z'] - top[1][side]['max_elbow_z']) < .001
