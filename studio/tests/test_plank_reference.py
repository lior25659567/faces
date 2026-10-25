"""Keep the real source body calibration and arm targets through support fitting."""
import json
import sys
from copy import deepcopy
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'pipeline/blender'))
from plank_reference import read


def source():
    return json.loads((Path(__file__).parent / 'fixtures/plank-keypose.json').read_text())


def test_actual_plank_keeps_its_measured_inclined_forearms_and_straight_body():
    result = read(source())
    assert result['torso_above_horizontal_deg'] == pytest.approx(4.858, abs=.01)
    assert result['hip_included_deg'] == pytest.approx(177.487, abs=.01)
    assert result['knee_included_deg'] == pytest.approx(175.515, abs=.01)
    assert result['elbow_included_deg'] == pytest.approx(67.657, abs=.01)
    assert result['upperarm_vs_torso_deg'] == pytest.approx(74.063, abs=.01)
    assert result['pixel_elbow_included_deg'] == pytest.approx(73.404, abs=.02)


def test_a_front_view_or_hidden_arm_cannot_supply_a_calibrated_side_hold():
    for field, value in [('view', 'front'), ('camera_side', None), ('facing', 'unknown')]:
        reference = source()
        reference['phases'][0][field] = value
        assert read(reference) is None
    reference = source()
    reference['phases'][0]['visibility']['right_elbow'] = .2
    assert read(reference) is None


def test_degenerate_or_invalid_landmarks_cannot_silently_make_a_default_pose():
    for value in ([float('nan'), 1.], [1., 2., 3.], None):
        reference = source()
        reference['phases'][0]['pixels']['right_elbow'] = value
        assert read(reference) is None
    reference = source()
    reference['phases'][0]['world']['right_elbow'] = reference['phases'][0]['world']['right_shoulder']
    assert read(reference) is None


def test_a_left_facing_image_preserves_the_same_body_angles_without_changing_input():
    reference = source()
    expected = read(reference)
    mirrored = deepcopy(reference)
    mirrored['phases'][0]['facing'] = 'left'
    for point in mirrored['phases'][0]['pixels'].values():
        point[0] = 850 - point[0]
    actual = read(mirrored)
    for name in ('torso_above_horizontal_deg', 'thigh_angle_deg', 'shin_angle_deg', 'hip_included_deg'):
        assert actual[name] == pytest.approx(expected[name])
    assert reference == source()
