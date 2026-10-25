"""Chronological stepping never becomes a reversed lift or inferred hidden contact."""
from copy import deepcopy
import json
import sys

import numpy as np
import pytest

from conftest import ROOT

sys.path.insert(0, str(ROOT / 'pipeline/blender'))
from sequence_clip import SequenceError, contact_roots, inspect_sequence, motion_from_sequence, visible_support
from synthetic_pose import mp_pose
from test_keypose_clip import REST

ACTUAL = json.loads((ROOT / 'studio/tests/fixtures/lunge-contact-samples.json').read_text())


def test_shared_visible_foot_recovers_forward_root_motion_without_fixing_the_swing_foot():
    # The left foot stays at the same world point while the hips advance 30 cm.
    points = [{'left': np.array([0., 0., -1.]), 'right': np.array([.2, -.1, -1.])},
              {'left': np.array([0., .3, -1.]), 'right': np.array([.2, -.5, -.8])}]
    result = contact_roots(points, [{'left'}, {'left'}])
    assert result['ready']
    assert result['roots'][1] == pytest.approx([0., -.3, 0.])
    assert np.asarray(result['roots'][1]) + points[1]['left'] == pytest.approx(points[0]['left'])
    assert np.linalg.norm(np.asarray(result['roots'][1]) + points[1]['right'] - points[0]['right']) > .2


def test_visible_double_support_allows_a_contact_handoff_without_resetting_the_root():
    points = [{'left': np.array([0., 0., -1.]), 'right': np.array([.2, -.1, -.8])},
              {'left': np.array([0., .2, -1.]), 'right': np.array([.2, -.3, -1.])},
              {'left': np.array([0., .4, -.8]), 'right': np.array([.2, -.1, -1.])}]
    result = contact_roots(points, [{'left'}, {'left', 'right'}, {'right'}])
    assert result['ready']
    assert result['roots'] == pytest.approx(np.array([[0., 0., 0.], [0., -.2, 0.], [0., -.4, 0.]]))


def test_unobserved_handoff_is_blocked_instead_of_extrapolating_or_pinning_an_occluded_foot():
    points = [{'left': np.zeros(3), 'right': np.ones(3)}, {'left': np.ones(3), 'right': np.zeros(3)}]
    result = contact_roots(points, [{'left'}, {'right'}])
    assert not result['ready']
    assert result['roots'][1] is None
    assert 'no shared visible planted foot' in result['blockers'][0]


def test_incompatible_flat_feet_cannot_both_be_declared_locked():
    points = [{'left': np.zeros(3), 'right': np.ones(3)},
              {'left': np.array([0., .02, 0.]), 'right': np.ones(3)}]
    result = contact_roots(points, [{'left', 'right'}, {'left', 'right'}], tolerance=.003)
    assert not result['ready']
    assert result['roots'][1] is None
    assert 'incompatible visible foot anchors' in result['blockers'][0]


def test_uncertain_airborne_low_visibility_and_unobserved_feet_do_not_anchor():
    sample = {'observedSupport': 'both', 'footContacts': {'left': 'uncertain', 'right': 'airborne'},
              'visibility': {'left_ankle': 1., 'right_ankle': 1.}}
    assert visible_support(sample) == set()
    sample['footContacts'] = {'left': 'flat', 'right': 'flat'}
    sample['visibility']['left_ankle'] = .7
    assert visible_support(sample) == {'right'}
    sample['observedSupport'] = 'left'
    assert visible_support(sample) == set()


def test_actual_twelve_frame_lunge_retains_chronology_and_stops_at_unobservable_travel():
    result = inspect_sequence(ACTUAL)
    assert not result['ready']
    assert result['sampleCount'] == 12
    assert result['sourceTimes'] == [sample['timeSeconds'] for sample in ACTUAL['sequence']['samples']]
    assert result['sourceTimes'][5] == pytest.approx(4.503333)
    assert result['playback'] == 'once'
    assert result['completeClosedCycle'] is False
    assert result['calibrated'] is False and result['sourceInferredDepth'] is True
    assert result['roots'][1] is not None
    assert result['roots'][2:] == [None] * 10
    assert any('Samples 1→2' in blocker for blocker in result['blockers'])
    # The adapter refuses before manufacturing a complete clip or a reverse path.
    with pytest.raises(SequenceError, match='unobserved') as failure:
        motion_from_sequence(ACTUAL, {})
    assert failure.value.report['sampleCount'] == 12


def test_valid_sequence_runs_once_by_source_time_and_moves_the_root_around_the_support():
    samples = []
    for index, (time, phase, angles) in enumerate([(2., 'start', (0, 180, 180)),
                                                 (4.3, 'middle', (0, 190, 175)),
                                                 (6.1, 'end', (0, 205, 170))]):
        entry = mp_pose(*angles)
        entry.update(sequenceIndex=index, timeSeconds=time, observedSupport='left',
                     footContacts={'left': 'flat', 'right': 'airborne'},
                     visibility={'left_ankle': 1., 'right_ankle': 1.}, source={'phase': phase}, motionOnly=False)
        samples.append(entry)
    result = motion_from_sequence({'motion': {'mode': 'sequence'}, 'sequence': {'samples': samples}}, REST, fps=10)
    assert len(result['turns']) == 42
    assert result['reference_motion']['phase_frames'] == {'start': 1, 'middle': 24, 'end': 42}
    assert result['reference_motion']['playback'] == 'once'
    planted = np.asarray([frame['LeftToeBase'] for frame in result['frames']])
    assert np.max(np.linalg.norm(planted - planted[0], axis=1)) < 1e-8
    hips = np.asarray([(frame['LeftUpLeg'] + frame['RightUpLeg']) / 2 for frame in result['frames']])
    assert hips[-1, 1] < hips[0, 1] - .2  # character forward is -Y
    assert not np.allclose(result['turns'][0]['LeftLeg'], result['turns'][-1]['LeftLeg'])


@pytest.mark.parametrize('edit', ['duplicate_time', 'reverse', 'missing_sample', 'bad_point', 'missing_body'])
def test_malformed_or_incomplete_sequence_cannot_silently_fall_back_to_a_cycle(edit):
    keys = deepcopy(ACTUAL)
    samples = keys['sequence']['samples']
    if edit == 'duplicate_time':
        samples[1]['timeSeconds'] = samples[0]['timeSeconds']
    elif edit == 'reverse':
        samples.reverse()
    elif edit == 'missing_sample':
        samples.pop(1)
    elif edit == 'bad_point':
        samples[1]['world']['left_ankle'][0] = float('nan')
    else:
        samples[1]['error'] = 'No body'
    with pytest.raises(SequenceError):
        inspect_sequence(keys)


def test_contact_fixture_is_tied_to_reviewed_public_images_and_annotations():
    manifest = json.loads((ROOT / 'reference/exercise-lunge/sequence.json').read_text())
    for sample, frame in zip(ACTUAL['sequence']['samples'], manifest['frames']):
        assert sample['imageSha256'] == frame['sha256']
        assert sample['sourceFrameIndex'] == frame['sourceFrameIndex']
        assert sample['footContacts'] == frame['footContacts']
