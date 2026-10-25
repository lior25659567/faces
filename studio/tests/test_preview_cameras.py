import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'pipeline/blender'))
from preview_camera_policy import angles_from_report, padded_box, views_for


def test_horizontal_unpropped_character_uses_full_length_instead_of_height():
    low, high = [-.2, -.6, 0.], [.2, .6, .3]
    box = padded_box(low, high)
    assert all(abs(box[1][i] - box[0][i] - 1.32) < 1e-10 for i in range(3))
    assert all(box[0][i] <= low[i] <= high[i] <= box[1][i] for i in range(3))
    assert low == [-.2, -.6, 0.] and high == [.2, .6, .3]


def test_actual_machine_camera_choices_are_recorded_without_changing_legacy_views():
    views = views_for('Leg Press (seated machine)')
    assert views == ((90., 5.), (-90., 5.))
    assert angles_from_report({'preview_cameras': [list(v) for v in views], 'preview_camera_space': 'world'}) == views
    assert views_for('Barbell Bent-Over Row') == ((35., 10.), (90., 5.))
    assert views_for('Leg Press (seated machine)', False) == (views[0],)
    assert angles_from_report({}) is None


@pytest.mark.parametrize('views,space', [([], 'world'), ([[90, 5]], 'relative'),
    ([[True, 5]], 'world'), ([[90, float('nan')]], 'world'), ([[90, 90]], 'world'),
    ([[90]], 'world'), ([[90, 5], [-90, 5], [0, 0]], 'world')])
def test_corrupt_recorded_cameras_cannot_silently_change_visibility_checks(views, space):
    with pytest.raises(ValueError):
        angles_from_report({'preview_cameras': views, 'preview_camera_space': space})
