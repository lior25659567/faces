"""A real observed step must keep contact identities without inventing a loop."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'pipeline/blender'))
from lunge_motion import samples_from, named_contacts, timeline, descent_progress, contact_fit, knee_position, root_height, carrying_arms, knee_clock, root_from_knee, ankle_from_knee
from lunge_range import summarize

ACTUAL = json.loads((Path(__file__).with_name('fixtures') / 'lunge-one-step-contacts.json').read_text())


def test_actual_dense_step_has_observed_named_handoffs_and_a_real_swing_ending():
    samples = samples_from(ACTUAL)
    assert len(samples) == 26
    assert named_contacts(samples[6]) == {'left.toe'}  # Right heel approach stays uncertain.
    assert named_contacts(samples[7]) == {'left.toe', 'right.heel', 'right.toe'}
    assert named_contacts(samples[-1]) == {'right.heel', 'right.toe'}
    assert samples[-1]['footContacts']['left'] == 'airborne'
    rows = timeline(samples, [sample['timeSeconds'] for sample in samples])
    assert rows[-1]['foot_states']['left'] == 'airborne'
    progress = descent_progress(samples, [sample['timeSeconds'] for sample in samples])
    assert progress[0] == 0. and progress[15] == 1.
    assert progress[-1] < .1


def test_unknown_contact_cannot_be_promoted_to_a_heel_anchor():
    reference = deepcopy(ACTUAL)
    reference['sequence']['samples'][6]['observedContactPoints']['right'] = ['heel']
    with pytest.raises(ValueError, match='visibly reviewed'):
        samples_from(reference)


def test_a_missing_shared_named_support_blocks_the_step_instead_of_resetting_root():
    reference = deepcopy(ACTUAL)
    sample = reference['sequence']['samples'][6]
    sample['footContacts'] = {'left': 'airborne', 'right': 'heel'}
    sample['observedContactPoints'] = {'left': [], 'right': ['heel']}
    with pytest.raises(ValueError, match='unobserved support handoff'):
        samples_from(reference)


def test_interpolated_contacts_are_conservative_and_source_times_are_not_extended():
    samples = samples_from(ACTUAL)
    time = (samples[6]['timeSeconds'] + samples[7]['timeSeconds']) / 2
    row = timeline(samples, [time])[0]
    assert row['contacts'] == ['left.toe']
    assert row['foot_states']['right'] == 'uncertain'
    with pytest.raises(ValueError, match='extrapolate'):
        timeline(samples, [samples[-1]['timeSeconds'] + .01])


def test_contact_fit_keeps_toe_identity_while_a_second_foot_lands():
    points = []
    for advance in (0., .1, .2):
        points.append({name: np.array([-.1 if name.startswith('left') else .1,
            (0. if name.startswith('left') else .3) - advance + (.08 if name.endswith('toe') else 0.), -.4])
            for name in ('left.heel', 'left.toe', 'right.heel', 'right.toe')})
    shifts, anchors = contact_fit(points, [{'left.heel', 'left.toe'},
        {'left.toe', 'right.heel', 'right.toe'}, {'right.heel', 'right.toe'}])
    assert shifts[:, 1] == pytest.approx([0., .1, .2])
    np.testing.assert_allclose(anchors[0]['left.toe'], anchors[1]['left.toe'])
    np.testing.assert_allclose(anchors[1]['right.heel'], anchors[2]['right.heel'])


def test_fixed_length_leg_reaches_with_a_forward_knee_and_refuses_stretch():
    hip, ankle = np.array([0., 0., .4]), np.array([0., .1, .08])
    knee = knee_position(hip, ankle, .2, .2)
    assert np.linalg.norm(knee - hip) == pytest.approx(.2)
    assert np.linalg.norm(ankle - knee) == pytest.approx(.2)
    assert knee[1] > .05
    with pytest.raises(ValueError, match='unchanged leg reach'):
        knee_position(hip, [0., 1., 0.], .2, .2)


def test_source_timed_angle_clock_avoids_a_linear_height_lockout_singularity():
    start, middle, end = 2.001667, 3.501667, 4.503333
    assert knee_clock(start, start, middle, end) == 175.
    assert knee_clock(middle, start, middle, end) == 100.
    assert knee_clock(end, start, middle, end) == 170.
    times = np.linspace(middle, end, 17)
    angles = [knee_clock(time, start, middle, end) for time in times]
    assert all(a <= b for a, b in zip(angles, angles[1:]))
    assert max(np.diff(angles)) < 9.
    hip_z = root_from_knee([0., .1], [0., 0., 0.], [0., .2, .06], [.2, .2], 100.)
    ankle_z = ankle_from_knee([0., .1, hip_z], [0., .2], [.2, .2], 100.)
    assert ankle_z == pytest.approx(.06)


def test_carrying_pose_matches_visible_widths_without_stretching_either_arm():
    shoulders = {'left': np.array([-.117, 0., .69]), 'right': np.array([.117, 0., .69])}
    lengths = {'left': [.1044, .0779], 'right': [.1047, .0750]}
    result = carrying_arms(shoulders, lengths, 1.592, 1.352)
    shoulder_span = np.linalg.norm(shoulders['right'] - shoulders['left'])
    assert np.linalg.norm(result['right']['wrist'] - result['left']['wrist']) / shoulder_span == pytest.approx(1.592, abs=.001)
    assert np.linalg.norm(result['right']['elbow'] - result['left']['elbow']) / shoulder_span == pytest.approx(1.352, abs=.001)
    for side in shoulders:
        assert np.linalg.norm(result[side]['elbow'] - shoulders[side]) == pytest.approx(lengths[side][0])
        assert np.linalg.norm(result[side]['wrist'] - result[side]['elbow']) == pytest.approx(lengths[side][1])
        assert result[side]['wrist'][2] < result[side]['elbow'][2] < shoulders[side][2]
    with pytest.raises(ValueError, match='unchanged arm reach'):
        carrying_arms(shoulders, lengths, 5., 1.352)


def measured_step():
    times = [0., .1, .2, .3, .4]
    phases = {'start': 1, 'middle': 3, 'end': 5}
    states = [{'left': 'flat', 'right': 'flat'}, {'left': 'toe', 'right': 'airborne'},
        {'left': 'toe', 'right': 'flat'}, {'left': 'toe', 'right': 'flat'},
        {'left': 'airborne', 'right': 'flat'}]
    rows = []
    for index, time in enumerate(times):
        root = np.array([0., index * .05, [.4596, .4, .32, .4, .4596][index]])
        ankles = {'left': np.array([-.05, 0., .06]),
                  'right': np.array([.05, 0. if index == 0 else .1 if index == 1 else .2, .06])}
        if index == 1: ankles['right'][2] = .1
        if index == 4: ankles['left'] = np.array([-.05, .15, .12])
        hips = {side: root + np.array([-.05 if side == 'left' else .05, 0., 0.]) for side in ('left', 'right')}
        knees = {side: knee_position(hips[side], ankles[side], .2, .2) for side in hips}
        points, anchors, floor = {}, {}, {}
        for side in ('left', 'right'):
            y = 0. if side == 'left' or index == 0 else .2
            for point, offset in (('heel', -.03), ('toe', .06)):
                name = side + '.' + point
                active = states[index][side] == 'flat' or states[index][side] == point
                position = np.array([-.05 if side == 'left' else .05, y + offset, 0. if active else .02])
                points[name] = position.tolist()
                floor[name] = {'min_z_m': position[2], 'max_z_m': position[2] + .0001}
                if active: anchors[name] = position.tolist()
        rows.append({'frame': index + 1, 'time_s': time, 'foot_states': states[index],
            'contact_points_m': points, 'contact_anchors_m': anchors, 'contact_floor': floor,
            'hips_m': {side: value.tolist() for side, value in hips.items()},
            'knees_m': {side: value.tolist() for side, value in knees.items()},
            'ankles_m': {side: value.tolist() for side, value in ankles.items()},
            'leg_lengths_m': {'left': [.2, .2], 'right': [.2, .2]},
            'foot_min_z_m': {side: .02 if states[index][side] == 'airborne' else 0. for side in hips},
            'body_min_z_m': 0., 'root_m': root.tolist()})
    return rows, phases, times


def test_actual_geometry_checks_a_one_shot_step_without_requiring_a_final_two_foot_stand():
    rows, phases, times = measured_step()
    measured = summarize(rows, 1., phases, times, 10.)
    for metric in ('max_contact_gap_mm', 'max_contact_penetration_mm', 'max_anchor_drift_mm',
            'max_swing_clearance_shortfall_mm', 'max_leg_length_error_mm', 'max_backward_knee_mm',
            'max_body_floor_penetration_mm', 'missing_forward_travel', 'max_reverse_root_mm',
            'start_extension_shortfall_deg', 'middle_knee_overfold_deg', 'middle_knee_depth_shortfall_deg'):
        assert measured[metric] == pytest.approx(0., abs=1e-8), metric
    assert measured['playback'] == 'once' and measured['source_inferred_depth']
    assert measured['per_frame'][-1]['foot_states']['left'] == 'airborne'


def test_a_sliding_contact_and_an_airborne_foot_scraping_the_floor_are_rejected():
    rows, phases, times = measured_step()
    rows[2]['contact_points_m']['right.heel'][0] += .01
    rows[1]['foot_min_z_m']['right'] = 0.
    measured = summarize(rows, 1., phases, times, 10.)
    assert measured['max_anchor_drift_mm'] == pytest.approx(10.)
    assert measured['max_swing_clearance_shortfall_mm'] == pytest.approx(2.)


@pytest.mark.parametrize('defect', ['missing_frame', 'wrong_clock', 'reset_anchor', 'missing_patch', 'stretched_leg'])
def test_incomplete_or_forged_measurements_do_not_keep_an_old_passing_summary(defect):
    rows, phases, times = measured_step()
    if defect == 'missing_frame': rows.pop(2)
    elif defect == 'wrong_clock': rows[2]['time_s'] += .03
    elif defect == 'reset_anchor': rows[3]['contact_anchors_m']['right.heel'][1] += .1
    elif defect == 'missing_patch': del rows[2]['contact_floor']['right.heel']
    else:
        rows[2]['knees_m']['right'][0] += .04
        assert summarize(rows, 1., phases, times, 10.)['max_leg_length_error_mm'] > 1.
        return
    with pytest.raises(ValueError): summarize(rows, 1., phases, times, 10.)
