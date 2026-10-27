"""A declared squat preset must show a complete supported rep in actual geometry."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'pipeline/blender'))
from squat_range import summarize


def measured():
    return json.loads((Path(__file__).with_name('fixtures') / 'squat-preset-range.json').read_text())


def check(value):
    return summarize(value['per_frame'], value['height_m'], value['phase_frames'],
                     value['torso_turn'], value['leg_turns'])


def test_real_pinned_squat_contains_lockout_depth_travel_symmetry_and_a_closed_loop():
    raw = measured()
    before = deepcopy(raw)
    result = check(raw)
    assert result['frames'] == 69
    assert min(result['start']['knees_deg'].values()) >= 165.
    assert max(result['bottom']['knees_deg'].values()) <= 100.
    assert result['depth_shortfall_mm'] == 0.
    assert result['pelvis_travel_mm'] > 280.
    assert result['loop_position_error_mm'] < 1.
    assert result['spine_deviation_deg'] < 15.
    assert raw == before


def test_a_shallow_or_static_clip_cannot_claim_full_squat_range():
    raw = measured()
    first = deepcopy(raw['per_frame'][0])
    first['frame'] = 38
    raw['per_frame'][37] = first
    result = check(raw)
    assert result['bottom_knee_flexion_excess_deg'] > 60.
    assert result['depth_shortfall_mm'] > 100.
    assert result['pelvis_travel_shortfall_mm'] > 100.


def test_a_different_good_frame_cannot_replace_the_pinned_bottom_position():
    raw = measured()
    raw['phase_frames']['end'] = 37
    with pytest.raises(ValueError, match='pinned source phases'):
        check(raw)


@pytest.mark.parametrize('defect', ['missing', 'duplicate', 'nonfinite', 'missing_side', 'bad_rest_axis'])
def test_missing_or_inconsistent_measurements_are_rejected(defect):
    raw = measured()
    if defect == 'missing':
        raw['per_frame'].pop(20)
    elif defect == 'duplicate':
        raw['per_frame'][20]['frame'] = 20
    elif defect == 'nonfinite':
        raw['per_frame'][20]['knees_m']['left'][2] = float('nan')
    elif defect == 'missing_side':
        del raw['per_frame'][20]['ankles_m']['right']
    else:
        raw['torso_turn'][0][0] = 2.
    with pytest.raises(ValueError):
        check(raw)


def test_a_tilted_bar_and_large_loop_seam_remain_measured_failures():
    raw = measured()
    raw['per_frame'][20]['bar_axis_m'] = [0., .8, .6]
    raw['per_frame'][-1]['bar_m'][0] += .02
    result = check(raw)
    assert result['bar_level_excess_deg'] > 30.
    assert result['loop_position_error_mm'] > 19.
