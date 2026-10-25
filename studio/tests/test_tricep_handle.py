"""A cable pushdown has a short finite handle, sized for both closed hands."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'pipeline/blender'))
from motion_geometry import cable_handle_half_length
from pullup_range import reference_timeline


def test_standard_attachment_is_minimum_but_palms_cannot_overhang_its_ends():
    assert cable_handle_half_length(1.75, .1) == pytest.approx(.46 / 2)
    assert cable_handle_half_length(1., .1405) == pytest.approx(.1755)
    assert cable_handle_half_length(1., .1405) < 2.2 / 2 / 1.75


def test_scaled_characters_keep_relative_attachment_size():
    assert cable_handle_half_length(2., .281) == pytest.approx(2 * cable_handle_half_length(1., .1405))


@pytest.mark.parametrize('height,grip,length', [(0., .1, .46), (1., 0., .46), (1., .1, -1.),
    (float('nan'), .1, .46), (1., float('inf'), .46), (True, .1, .46)])
def test_unmeasured_or_invalid_attachment_cannot_be_built(height, grip, length):
    with pytest.raises(ValueError):
        cable_handle_half_length(height, grip, length)


def test_reference_arm_angles_use_measured_positions_and_skip_estimated_middle():
    targets = {
        'start': {'upperarm_vs_torso': {'value': 0., 'image': 'start'}},
        'middle': {'estimated': True, 'upperarm_vs_torso': {'value': 99., 'image': None}},
        'end': {'upperarm_vs_torso': {'value': 20., 'image': 'end'}},
    }
    values = reference_timeline(targets, {'start': 1, 'middle': 2, 'end': 3}, [0., .5, 1., .5, 0.], 'upperarm_vs_torso')
    assert values.tolist() == pytest.approx([0., 10., 20., 10., 0.])
