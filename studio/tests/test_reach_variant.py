"""Rest-space reach variants preserve hands/face and never change original mode."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('reach_variant', ROOT / 'pipeline/blender/reach_variant.py')
rv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rv)


def stretch(points, weights, **options):
    return rv.displacement(np.array(points, dtype=float), [0., 0., 0.], [1., 0., 0.], [2., 0., 0.],
                           *np.array(weights, dtype=float).T, 1.5, 2., **options)


def test_longer_segments_keep_shoulder_fixed_and_translate_fingers_without_scaling():
    points = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.], [2.3, .1, 0.], [2.5, -.1, .1]])
    before = points.copy()
    delta = stretch(points, [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]])
    assert np.array_equal(points, before)
    moved = points + delta
    assert np.allclose(moved[:3, 0], [0., 1.5, 3.5])
    assert np.allclose(moved[3:] - moved[2], points[3:] - points[2])


def test_stray_inner_collar_arm_weights_cannot_pull_the_neck_outward():
    delta = stretch([[-.1, .2, 0.], [-.3, 0., .1]], [[.25, 0, 0], [.25, .2, .1]])
    assert np.array_equal(delta, np.zeros((2, 3)))


def test_face_vertices_stay_fixed_even_with_erroneous_arm_weights():
    delta = stretch([[2., .2, .1]], [[.25, .25, .5]], fixed=[True])
    assert np.array_equal(delta, [[0., 0., 0.]])


def test_duplicate_seams_share_displacement_and_a_fixed_face_seam_wins():
    points = [[1.3, .1, 0.], [1.3, .1, 0.]]
    weights = [[.5, .5, 0.], [.1, .9, 0.]]
    delta = stretch(points, weights, seams=[4, 4])
    assert np.array_equal(delta[0], delta[1])
    delta = stretch(points, weights, seams=[4, 4], fixed=[True, False])
    assert np.array_equal(delta, np.zeros((2, 3)))


@pytest.mark.parametrize('scale', [0., .9, 2.1, True, None, '1.8', float('nan'), float('inf')])
def test_invalid_scales_never_become_a_mesh_change(scale):
    with pytest.raises(ValueError):
        rv.displacement([[0, 0, 0]], [0, 0, 0], [1, 0, 0], [2, 0, 0], [1], [0], [0], scale, 1.5)


def test_zero_length_bones_and_nonfinite_geometry_are_rejected():
    with pytest.raises(ValueError):
        rv.displacement([[0, 0, 0]], [0, 0, 0], [0, 0, 0], [2, 0, 0], [1], [0], [0], 1.5, 1.5)
    with pytest.raises(ValueError):
        stretch([[float('nan'), 0, 0]], [[1, 0, 0]])


def test_original_and_unrelated_auto_profiles_never_load_or_modify_a_model():
    assert rv.apply_if_needed({}, None, 'Dumbbell Shoulder Press (seated overhead press)', 'original') is None
    assert rv.apply_if_needed({}, None, 'Dumbbell Bicep Curl (standing, both arms)', 'auto') is None
    with pytest.raises(ValueError):
        rv.apply_if_needed({}, None, 'Pull-Up', 'elastic')


def test_auto_needs_a_measured_reach_deficit_and_fitness_is_an_explicit_fixed_variant(monkeypatch):
    rig = {'skin_fit': json.dumps({'measured': {'height': 1.}})}
    calls = []
    monkeypatch.setattr(rv, 'measure_reach', lambda *_: {'head_top_m': 1., 'sides': {'left': {'vertical_handle_bound_m': 1.4}}})
    monkeypatch.setattr(rv, 'apply', lambda *_args, **kwargs: calls.append(kwargs) or {'kind': 'fitness arm reach'})
    assert rv.apply_if_needed(rig, None, 'Pull-Up', 'auto') is None
    result = rv.apply_if_needed(rig, None, 'Dumbbell Bicep Curl (standing, both arms)', 'fitness')
    assert calls == [{'upper_scale': 1.8, 'fore_scale': 1.8}]
    assert result['requested_mode'] == 'fitness'
