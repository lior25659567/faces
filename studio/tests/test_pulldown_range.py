"""Smooth fixed-grip motion still fails when elbows use the inverted branch."""
import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

BLENDER = Path(__file__).resolve().parents[2] / 'pipeline/blender'
sys.path.insert(0, str(BLENDER))
from pulldown_range import progress_at, summarize


def samples():
    result = []
    for frame, p in enumerate((0., .5, 1., .5, 0.), 1):
        shoulders = {s: [sign * .1, 0., .5] for s, sign in (('left', -1), ('right', 1))}
        elbows = {s: [sign * (.1 + .02 * p), .03 * p, .68 - .33 * p] for s, sign in (('left', -1), ('right', 1))}
        wrists = {s: [sign * (.1 + .04 * p), .04 * p, .82 - .32 * p] for s, sign in (('left', -1), ('right', 1))}
        result.append({'frame': frame, 'shoulders_m': shoulders, 'elbows_m': elbows, 'wrists_m': wrists,
                       'hip_m': [0., 0., .25], 'bar_m': [0., .04 * p, .87 - .32 * p]})
    return result


PHASES = {'start': 1, 'middle': 2, 'end': 3}


def test_downward_pull_has_upward_forearms_and_full_top_extension():
    result = summarize(samples(), 1., PHASES)
    assert result['max_elbow_above_wrist_mm'] == 0.
    assert result['bottom_elbow_above_shoulder_mm'] == 0.
    assert result['bottom_upperarm_excess_deg'] == 0.
    assert result['top_extension_shortfall_deg'] == 0.
    assert result['travel_shortfall_mm'] == 0.


def test_smooth_inverted_bottom_cannot_pass_contact_alone():
    motion = samples()
    for sample in motion:
        p = 1. - abs(sample['frame'] - 3) / 2
        for point in sample['elbows_m'].values():
            point[2] = .68 - .03 * p
    result = summarize(motion, 1., PHASES)
    assert result['max_elbow_above_wrist_mm'] > 140.
    assert result['bottom_elbow_above_shoulder_mm'] > 140.
    assert result['bottom_upperarm_excess_deg'] > 60.


def test_actual_previously_accepted_scene_reproduces_the_inverted_elbows():
    source = json.loads((Path(__file__).with_name('fixtures') / 'pulldown-original-range.json').read_text())
    result = summarize(source['per_frame'], source['height_m'], source['phase_frames'])
    assert result['max_elbow_above_wrist_mm'] > 80.
    assert result['bottom_elbow_above_shoulder_mm'] > 100.
    assert result['bottom_upperarm_excess_deg'] > 20.


def test_forged_angle_summaries_do_not_hide_a_bent_start():
    motion = samples()
    for side in ('left', 'right'):
        motion[0]['elbows_m'][side][1] = .1
    motion[0]['elbows_deg'] = {'left': 180., 'right': 180.}
    assert summarize(motion, 1., PHASES)['top_extension_shortfall_deg'] > 20.


def test_remeasurement_does_not_mutate_the_input():
    motion = samples()
    original = copy.deepcopy(motion)
    summarize(motion, 1., PHASES)
    assert motion == original


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'nonfinite', 'one_arm', 'collapsed'])
def test_incomplete_or_unreadable_geometry_is_rejected(change):
    motion = samples()
    if change == 'missing': motion.pop(1)
    elif change == 'duplicate': motion[1]['frame'] = 1
    elif change == 'nonfinite': motion[2]['wrists_m']['left'][0] = float('nan')
    elif change == 'one_arm': del motion[2]['elbows_m']['right']
    elif change == 'collapsed': motion[2]['elbows_m']['left'] = motion[2]['shoulders_m']['left']
    with pytest.raises(ValueError): summarize(motion, 1., PHASES)


def test_recorded_phase_order_and_real_range_are_required():
    with pytest.raises(ValueError): summarize(samples(), 1., {'start': 1, 'end': 6})
    motion = samples()
    for sample in motion: sample['bar_m'][2] = .87
    assert summarize(motion, 1., PHASES)['travel_shortfall_mm'] == pytest.approx(120.)
    assert progress_at(1, 1, 5, PHASES) == 0.
    assert progress_at(3, 1, 5, PHASES) == 1.


def test_upper_chest_hold_and_controlled_return_use_the_generated_clock():
    phases = {'start': 1, 'middle': 7, 'end': 13}
    progress = [progress_at(frame, 1, 32, phases) for frame in range(1, 33)]
    assert progress[0] == 0.
    assert progress[12:15] == [1., 1., 1.]
    assert all(a <= b for a, b in zip(progress[:12], progress[1:13]))
    assert all(a >= b for a, b in zip(progress[14:], progress[15:]))
    assert 0. < progress[-1] < .01  # The loop seam leads naturally into frame 1.
