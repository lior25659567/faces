"""Pre-render visibility uses all real views and preserves the existing verdict."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'pipeline/blender'))
from skin_preflight import fold_verdict, motion_verdict, preview_bounds, source_camera_plan
from checks import _fold_rows, _motion_rows


def test_horizontal_character_and_animated_props_fit_the_complete_preview_cube():
    skin = np.array([[-.5, -.1, 0.], [.5, .1, .3]])
    props = [np.array([[-.7, -.1, .2], [.7, .1, .5]]), np.array([[-.7, -.1, .8], [.7, .1, 1.1]])]
    box = np.array(preview_bounds(skin, props))
    all_points = np.concatenate([skin, *props])
    assert (all_points >= box[0]).all() and (all_points <= box[1]).all()
    np.testing.assert_allclose(box[1] - box[0], [1.54] * 3)
    # A plank with no held equipment must still fit its full horizontal length.
    plank = np.array(preview_bounds(skin))
    assert plank[1, 2] - plank[0, 2] == pytest.approx(1.1)


def camera_for(view, side, yaw):
    return {'behind': (180. + yaw, 5., 'behind'), 'front': (yaw, 5., 'front'),
            'side': (yaw + (90. if side == 'left' else -90.), 5., 'side')}[view]


def test_behind_comparison_at_actual_end_frame_cannot_be_omitted():
    snapshot = {'images': [{'id': 'front', 'phase': 'start', 'view': 'front'},
                           {'id': 'back', 'phase': 'end', 'view': 'behind'},
                           {'id': 'context', 'phase': 'end', 'view': 'side', 'contextOnly': True},
                           {'id': 'motion', 'phase': 'middle', 'view': 'side', 'source': {'motionOnly': True}},
                           {'id': 'deleted', 'phase': 'middle', 'view': 'side', 'deleted': True}]}
    result = source_camera_plan(snapshot, {}, {'camera': {'side': 'left'}},
                                {'start': 1, 'middle': 10, 'end': 19}, 2., camera_for,
                                lambda image, _: image['view'])
    assert [camera['image'] for camera in result] == ['front', 'back']
    assert result[1] == {'image': 'back', 'phase': 'end', 'frame': 19, 'view': 'behind', 'azimuth': 182., 'elevation': 5.}


def test_active_reference_without_a_timed_camera_is_an_error():
    with pytest.raises(ValueError, match='recorded comparison frame'):
        source_camera_plan({'images': [{'id': 'x', 'phase': 'end'}]}, {}, {'camera': {'side': 'left'}},
                           {'start': 1}, 0., camera_for, lambda *_: 'side')


@pytest.mark.parametrize('measured,expected', [
    ({'height_m': 1., 'folds': []}, True),
    ({'height_m': 1., 'folds': [{'frame': 19, 'bone': 'mixamorig:LeftForeArm', 'kind': 'defect', 'area_mm2': 313., 'visible_mm2': 120.}]}, False),
    ({'height_m': 1., 'folds': [{'frame': 19, 'bone': 'mixamorig:LeftForeArm', 'kind': 'hidden', 'area_mm2': 313., 'visible_mm2': 20.}]}, True),
    ({'height_m': 1., 'folds': [{'frame': 19, 'bone': 'mixamorig:LeftForeArm', 'kind': 'crease', 'area_mm2': 313., 'visible_mm2': 120.}]}, True),
    (None, False), ({'height_m': 1., 'folds': None}, False),
    ({'height_m': 1., 'folds': [{'kind': 'hidden'}]}, False),
])
def test_verdict_is_exactly_the_studio_classifier_without_promoting_missing_or_hidden_data(measured, expected):
    rows, ready = fold_verdict(measured, 1, 32)
    assert rows == _fold_rows({'first': 1, 'last': 32, **(measured or {})})
    assert ready is expected
    if measured and measured.get('folds') and measured['folds'][0].get('kind') == 'hidden' and expected:
        assert rows[0]['status'] == 'info'


def test_failed_scan_cannot_keep_a_pass_from_partial_results():
    rows, ready = fold_verdict({'height_m': 1., 'folds': []}, 1, 32, error='scan interrupted')
    assert ready is False
    assert rows[0]['kind'] == 'not-measured'


@pytest.mark.parametrize('motion,ready', [({'jumps': [], 'steps': []}, True),
    ({'jumps': [], 'steps': [{'frame': 31, 'mm': 51.}]}, False),
    ({'jumps': [{'frame': 31, 'hips_mm': 51., 'knees_mm': 10.}], 'steps': []}, False),
    ({'jumps': []}, False), (None, False)])
def test_movement_gate_preserves_the_existing_jump_and_missing_measurement_rules(motion, ready):
    rows, actual = motion_verdict(motion)
    assert rows == _motion_rows(motion or {})
    assert actual is ready


@pytest.mark.parametrize('box', [[[0., 0., 0.], [np.nan, 1., 1.]], [[1., 0., 0.], [0., 1., 1.]]])
def test_unknown_or_reversed_bounds_are_not_used_for_visibility(box):
    with pytest.raises(ValueError, match='bounds'):
        preview_bounds(box)
