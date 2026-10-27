"""The bench profile must fail visibly incomplete or looping presses."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'pipeline/blender'))
from bench_press import path_at, progress_at, summarize
from pullup_range import included
from row_motion import solve_contact


def measurements():
    samples = []
    phases = {'start': 1, 'middle': 2, 'end': 3}
    for frame, fraction in enumerate((0., .5, 1., 1., .5), 1):
        shoulders, elbows, wrists, angles = {}, {}, {}, {}
        for side, sign in (('left', -1), ('right', 1)):
            shoulders[side] = [sign * .2, 0., .5]
            elbows[side] = [sign * (.2 + .2 * fraction), .25 * fraction, .8 - .35 * fraction]
            wrists[side] = [sign * (.2 + .2 * fraction), .25 * fraction, 1.1 - .35 * fraction]
            angles[side] = included(np.array(shoulders[side]), np.array(elbows[side]), np.array(wrists[side]))
        samples.append({'frame': frame, 'bar_m': [0., .25 * fraction, 1.16 - .35 * fraction],
            'hip_m': [0., .5, .5], 'shoulders_m': shoulders, 'elbows_m': elbows,
            'wrists_m': wrists, 'elbows_deg': angles})
    return samples, phases


def test_complete_press_has_simultaneous_lockout_bend_and_fixed_body():
    samples, phases = measurements()
    result = summarize(samples, 1., phases)
    for metric in ('lockout_shortfall_deg', 'bottom_flexion_excess_deg', 'bottom_overfold_deg',
            'travel_shortfall_mm', 'forearm_tilt_excess_deg', 'bar_torso_path_excess_mm',
            'bottom_chest_shortfall_mm', 'lockout_shoulder_offset_mm', 'stroke_reversal_mm',
            'stationary_body_drift_mm', 'elbow_asymmetry_deg'):
        assert result[metric] == pytest.approx(0.)


def test_bar_cannot_sweep_toward_hips_and_reverse_before_chest():
    samples, phases = measurements()
    samples[1]['bar_m'][1] = .5
    result = summarize(samples, 1., phases)
    assert result['bar_torso_path_excess_mm'] == pytest.approx(200.)
    assert result['stroke_reversal_mm'] == pytest.approx(250.)


def test_lockout_is_measured_at_the_recorded_source_start_not_another_good_frame():
    samples, phases = measurements()
    for side in ('left', 'right'):
        samples[0]['elbows_m'][side][1] = .15
        samples[0]['elbows_deg'][side] = included(np.array(samples[0]['shoulders_m'][side]),
            np.array(samples[0]['elbows_m'][side]), np.array(samples[0]['wrists_m'][side]))
    assert summarize(samples, 1., phases)['lockout_shortfall_deg'] > 40.


def test_missing_frames_and_fabricated_elbow_angles_cannot_pass():
    samples, phases = measurements()
    missing = deepcopy(samples)
    missing.pop(1)
    with pytest.raises(ValueError, match='every movement frame'):
        summarize(missing, 1., phases)
    samples[0]['elbows_deg']['left'] = 120.
    with pytest.raises(ValueError, match='disagree'):
        summarize(samples, 1., phases)


def test_bottom_must_reach_lower_chest_without_entering_face_side_of_shoulders():
    samples, phases = measurements()
    samples[2]['bar_m'][1] = 0.
    assert summarize(samples, 1., phases)['bottom_chest_shortfall_mm'] == pytest.approx(100.)
    samples[2]['bar_m'][1] = -.08
    assert summarize(samples, 1., phases)['bar_torso_path_excess_mm'] == pytest.approx(50.)


def test_declared_trajectory_reverses_only_after_bottom_hold_and_keeps_source_timing():
    phases = {'start': 1, 'middle': 10, 'end': 19}
    fractions = [progress_at(frame, 1, 32, phases) for frame in range(1, 33)]
    assert fractions[0] == 0.
    assert fractions[9] == pytest.approx(.5)
    assert fractions[18:21] == [1., 1., 1.]
    assert all(a <= b for a, b in zip(fractions[:18], fractions[1:19]))
    assert all(a >= b for a, b in zip(fractions[20:-1], fractions[21:]))
    top, bottom = np.array([0., 0., 1.]), np.array([0., .2, .6])
    assert np.allclose(path_at(top, bottom, 0.), top)
    assert np.allclose(path_at(top, bottom, 1.), bottom)


def test_fixed_grip_branch_prevents_an_opposite_palm_roll_despite_a_bad_elbow_hint():
    grip = {'wrist': [0., 0., 0.], 'B': [0., .04, 0.], 'kb': [1., 0., 0.], 'hb': [0., 1., 0.]}
    reference = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    shoulder, target, hint = np.zeros(3), np.array([0., .2, .4]), np.array([0., -.1, .2])
    ordinary = solve_contact(shoulder, target, [1., 0., 0.], -1, .3, .3, grip, hint)
    assert ordinary['elbow'][1] < 0.
    corrected = solve_contact(shoulder, target, [1., 0., 0.], -1, .3, .3, grip, hint,
        preference={'reference_hand_turn': reference, 'max_hand_turn_deg': 5.,
            'minimum_forearm_vertical_fraction': np.cos(np.radians(30.)),
            'elbow_forward_axis': [0., 1., 0.], 'minimum_elbow_forward': .01})
    assert corrected['elbow'][1] > 0.
    change = np.degrees(np.arccos(np.clip((np.trace(reference.T @ corrected['turn']) - 1.) / 2., -1., 1.)))
    assert change <= 5. + 1e-8
    assert (corrected['wrist'][2] - corrected['elbow'][2]) / .3 >= np.cos(np.radians(30.))
    np.testing.assert_allclose(corrected['wrist'] + corrected['turn'] @ grip['B'], target, atol=1e-9)
    assert np.linalg.norm(corrected['elbow'] - shoulder) == pytest.approx(.3)
    assert np.linalg.norm(corrected['wrist'] - corrected['elbow']) == pytest.approx(.3)


@pytest.mark.parametrize('limit', [None, True, -1., 181., float('nan')])
def test_grip_branch_cannot_use_an_invalid_rotation_limit(limit):
    grip = {'wrist': [0., 0., 0.], 'B': [0., .04, 0.], 'kb': [1., 0., 0.], 'hb': [0., 1., 0.]}
    with pytest.raises(ValueError, match='bounded hand turn'):
        solve_contact([0., 0., 0.], [0., .2, .4], [1., 0., 0.], -1, .3, .3, grip, [0., -.1, .2],
            preference={'reference_hand_turn': np.eye(3), 'max_hand_turn_deg': limit})
