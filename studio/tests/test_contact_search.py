"""Collision-safe reach corrections: a reach solve cannot undo body clearance."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'pipeline/blender'))
from motion_geometry import collision_safe_target


def outside_body(point):
    point = np.array(point, dtype=float)
    projected = point.copy()
    projected[1] = max(projected[1], 0.)
    return projected, float(np.linalg.norm(point - projected))


def test_reach_correction_stays_outside_body_and_reports_its_actual_displacement():
    calls = []
    def solve(point):
        calls.append(point.copy())
        assert point[1] >= 0.
        if point[2] < -.99000001:
            raise ValueError('unreachable')
        return {'wrist': point.copy()}
    start = np.array([0., 0., -1.])
    point, result, adjustment = collision_safe_target(start, ([0., -1., 1.], [0., 0., 1.]), outside_body, solve)
    assert point[1] == 0.
    assert adjustment == pytest.approx(.01)
    assert np.array_equal(result['wrist'], point)
    assert all(p[1] >= 0. for p in calls)


def test_a_reachable_target_inside_the_body_is_not_an_acceptable_fallback():
    def solve(point):
        if point[1] < -.001:
            return 'would reach only by penetrating'
        raise ValueError('unreachable outside body')
    with pytest.raises(ValueError, match='collision-free'):
        collision_safe_target([0., 0., 0.], ([0., -1., 0.],), outside_body, solve)


def test_collision_projection_cannot_hide_a_correction_beyond_the_limit():
    def guard(point):
        return np.array([0., .03, 0.]), .03
    def solve(_):
        pytest.fail('The projected point exceeded the allowed 20 mm correction.')
    with pytest.raises(ValueError):
        collision_safe_target([0., 0., 0.], ([0., 1., 0.],), guard, solve)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), 0., -1.])
def test_invalid_search_limits_are_refused(bad):
    with pytest.raises(ValueError):
        collision_safe_target([0., 0., 0.], ([0., 1., 0.],), outside_body, lambda p: p, max_adjustment=bad)
