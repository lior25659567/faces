"""Local surface-correction influence without Blender."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'pipeline/blender'))
from pose_cleanup import collar_weights, arm_emphasis


def test_collar_influence_is_local_and_protects_face_and_hands():
    centres = np.array([[-.05, 0., .7], [.05, 0., .7]])
    points = np.array([[-.05, 0., .7], [-.05, .15, .7], [-.05, .3, .7], [.05, 0., .7]])
    weights = collar_weights(points, centres, 1., [False, False, False, True])
    assert weights[0] == 1.
    assert 0. < weights[1] < 1.
    assert weights[2] == 0.
    assert weights[3] == 0.
    assert np.isfinite(weights).all()


def test_local_region_scales_with_character_without_mutating_coordinates():
    points = np.array([[0., 0., .7], [0., .17, .7], [0., .21, .7]])
    centres = np.array([[0., 0., .7], [0., 0., .7]])
    before = points.copy()
    assert np.allclose(collar_weights(points, centres, 1.), collar_weights(points * 3, centres * 3, 3.))
    assert np.array_equal(points, before)


@pytest.mark.parametrize('height', [0., -1., float('nan'), float('inf')])
def test_invalid_scale_cannot_generate_a_cleanup_region(height):
    with pytest.raises(ValueError):
        collar_weights(np.zeros((1, 3)), np.zeros((2, 3)), height)


def test_invalid_geometry_and_mismatched_protected_mask_fail_before_mesh_mutation():
    with pytest.raises(ValueError):
        collar_weights(np.array([[float('nan'), 0., 0.]]), np.zeros((2, 3)), 1.)
    with pytest.raises(ValueError):
        collar_weights(np.zeros((2, 3)), np.zeros((2, 3)), 1., [False])


def test_arm_emphasis_counterbalances_inner_collar_weight_and_protects_zero_weights():
    original = np.array([1., 1., 1., 0.])
    shares = np.array([.25, .40, .55, .8])
    result = arm_emphasis(original, shares, 1.5)
    displacement = result * 1.5
    assert displacement[0] == pytest.approx(original[0])
    assert displacement[1] == pytest.approx(1.25)
    assert displacement[2] == pytest.approx(1.5)
    assert displacement[3] == 0.
    assert np.array_equal(original, [1., 1., 1., 0.])
    assert np.array_equal(shares, [.25, .40, .55, .8])


@pytest.mark.parametrize('factor', [0., .9, float('nan'), float('inf')])
def test_arm_emphasis_rejects_invalid_strength(factor):
    with pytest.raises(ValueError):
        arm_emphasis(np.ones(2), np.ones(2), factor)


def test_arm_emphasis_rejects_unmatched_and_nonfinite_influences():
    with pytest.raises(ValueError):
        arm_emphasis(np.ones(2), np.ones(3))
    with pytest.raises(ValueError):
        arm_emphasis(np.ones(2), [1., float('nan')])
