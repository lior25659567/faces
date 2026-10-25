"""Numpy-only finite weight-head contact geometry."""
import numpy as np


def cable_handle_half_length(height, half_grip, standard_length=.46):
    """A short straight attachment long enough for both closed cartoon hands.

    The catalogue's 0.46 m attachment is the minimum, scaled like other props
    against a 1.75 m adult. A large-hand character may need a longer attachment;
    disclose the resulting size rather than putting its palms beyond the ends.
    This is prop fitting, not a calibrated width read from a side photograph.
    """
    values = (height, half_grip, standard_length)
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not np.isfinite(value) or value <= 0 for value in values):
        raise ValueError('A cable attachment needs finite positive height, grip and standard length.')
    return max(standard_length / 2 * height / 1.75, half_grip + .035 * height)


def collision_safe_target(preferred, directions, guard, solve, max_adjustment=.020, step=.001):
    """Find a bounded reachable target without undoing its collision projection.

    ``solve`` must reject an unreachable target instead of adapting it internally.
    Every candidate, including the preferred one, is projected by ``guard`` before
    being tried. The displacement limit applies after projection, not before it.
    """
    preferred = np.asarray(preferred, dtype=float)
    if preferred.shape != (3,) or not np.isfinite(preferred).all():
        raise ValueError('A finite three-dimensional contact target is required.')
    if not np.isfinite(max_adjustment) or not np.isfinite(step) or min(max_adjustment, step) <= 0:
        raise ValueError('Contact search limits must be finite and positive.')
    axes = []
    for value in directions:
        value = np.asarray(value, dtype=float)
        length = np.linalg.norm(value)
        if value.shape == (3,) and np.isfinite(value).all() and length > 1e-10:
            axis = value / length
            if not any(np.allclose(axis, old) for old in axes):
                axes.append(axis)
    seen = set()
    last_error = None
    radii = [0.] + list(np.arange(step, max_adjustment + step * .5, step))
    for distance in radii:
        for axis in axes if distance else [np.zeros(3)]:
            projected, _ = guard(preferred + distance * axis)
            target = np.asarray(projected, dtype=float)
            adjustment = float(np.linalg.norm(target - preferred))
            if not np.isfinite(target).all() or adjustment > max_adjustment + 1e-9:
                continue
            key = tuple(np.round(target, 9))
            if key in seen:
                continue
            seen.add(key)
            try:
                return target, solve(target), adjustment
            except ValueError as error:
                last_error = error
    raise ValueError('No collision-free target is reachable within the bounded contact correction.') from last_error


def hex_head_depth(points, centre_x, radius, thickness):
    """Positive penetration into a six-sided prism whose cylinder axis is X.

    Blender's six cylinder vertices rotate from its Z axis onto X. The six face
    normals are midway between those vertices. Points beyond the finite end caps
    cannot count as contact, regardless of their distance from the infinite axis.
    """
    points = np.asarray(points, dtype=float)
    angles = (np.arange(6) + .5) * np.pi / 3
    normals = np.column_stack([np.sin(angles), -np.cos(angles)])
    sides = radius * np.cos(np.pi / 6) - (points[:, 1:] @ normals.T).max(axis=1)
    ends = thickness / 2 - abs(points[:, 0] - centre_x)
    return np.maximum(0., np.minimum(sides, ends))
