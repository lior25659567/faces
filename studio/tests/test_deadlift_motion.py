"""Real fitted proportions must support a floor pull, not a hovering bent-arm lift."""
import copy
import json
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'pipeline/blender'))
from deadlift_motion import knee_position, plan
from deadlift_range import measure_frame, summarize
from keypose_clip import timeline


def geometry(scale=1.8):
    data = json.loads((Path(__file__).with_name('fixtures') / 'deadlift-fit-geometry.json').read_text())
    for side in data['rest']:
        data['rest'][side]['arm_lengths'] = [length * scale for length in data['rest'][side]['arm_lengths']]
    return data


def test_fitted_fitness_character_completes_the_actual_floor_pull_and_lockout():
    data = geometry()
    original = copy.deepcopy(data)
    result = plan(data['rest'], data['grips'], data['height'], timeline(12))
    samples = []
    for frame, pose in enumerate(result['poses'], start=1):
        joints = {}
        for side in ('left', 'right'):
            arm = pose['arms'][side]
            joints[side] = {'shoulder': pose['shoulders'][side], 'elbow': arm['elbow'], 'wrist': arm['wrist'],
                            'hip': pose['hips'][side], 'knee': pose['knees'][side], 'ankle': result['feet'][side]}
            upper, lower = data['rest'][side]['leg_lengths']
            assert np.linalg.norm(pose['knees'][side] - pose['hips'][side]) == pytest.approx(upper)
            assert np.linalg.norm(result['feet'][side] - pose['knees'][side]) == pytest.approx(lower)
            assert arm['bend_deg'] <= 25.
        samples.append(measure_frame(frame, joints, pose['bar']))
    measured = summarize(samples, data['height'], {'start': 1, 'middle': 10, 'end': 19})
    assert measured['ready'] is True, measured
    assert measured['minimum_elbow_deg'] >= 175.
    assert min(measured['lockout_hips_deg'].values()) >= 170.
    assert result['poses'][0]['bar'][2] == pytest.approx(.225 * data['height'] / 1.75)
    assert result['poses'][0]['bar'][2] + .02 * data['height'] < min(result['poses'][0]['knees'][s][2] for s in ('left', 'right'))
    assert result['poses'][18]['hip'][1] == pytest.approx(np.mean([v[1] for v in result['feet'].values()]))
    assert data == original  # planning cannot alter the source character or grip models


def test_original_short_arm_fit_is_refused_instead_of_bending_or_floating_the_bar():
    data = geometry(scale=1.)
    with pytest.raises(ValueError, match='conventional|short-arm|fitness'):
        plan(data['rest'], data['grips'], data['height'], [0., 1.])


def test_unreachable_leg_target_is_not_stretched():
    with pytest.raises(ValueError, match='fixed leg reach'):
        knee_position(np.array([0., 0., 1.]), np.zeros(3), .2, .2)


@pytest.mark.parametrize('progress', [[-1., 1.], [0., np.nan], [0., 2.], []])
def test_invalid_reference_clock_is_not_fabricated(progress):
    data = geometry()
    with pytest.raises(ValueError, match='progress'):
        plan(data['rest'], data['grips'], data['height'], progress)
