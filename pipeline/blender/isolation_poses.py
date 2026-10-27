"""Declared standing curl, lateral-raise and pushdown form, not photo depth.

The photographs select exercise and comparison phases. These mechanical
profiles correct visibly wrong crouches and upper-arm swings rather than
presenting uncertain monocular angles as accurate exercise technique.
"""
import numpy as np

from standing_posture import upright

CURL = 'Dumbbell Bicep Curl (standing, both arms)'
RAISE = 'Dumbbell Lateral Raise'
PUSHDOWN = 'Tricep Pushdown (cable, bar or rope attachment)'
EXERCISES = {CURL, RAISE, PUSHDOWN}


def _rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + np.sin(angle) * cross + (1. - np.cos(angle)) * cross @ cross


def _swing(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    axis = np.cross(a, b)
    if np.linalg.norm(axis) < 1e-9:
        return np.eye(3) if a @ b > 0 else _rotation([0., 1., 0.], np.pi)
    return _rotation(axis, np.arctan2(np.linalg.norm(axis), a @ b))


def pose(exercise, key, progress):
    if exercise not in EXERCISES or not isinstance(progress, (int, float)) or isinstance(progress, bool) or not np.isfinite(progress) or not 0. <= progress <= 1.:
        raise ValueError('An isolation pose needs a supported exercise and finite phase progress.')
    result = upright([key])[0]
    down, forward = np.array([0., 0., -1.]), np.array([0., -1., 0.])
    for side, sign in (('Left', 1.), ('Right', -1.)):
        rest = np.array([sign, 0., 0.])
        if exercise in (CURL, PUSHDOWN):
            # Fixed upper arms just clear the mascot's torso. Only elbows bend.
            direction = np.array([sign * .174, -.087, -.981]) if exercise == CURL else np.array([sign * .087, -.174, -.981])
            direction /= np.linalg.norm(direction)
            upper = _swing(rest, direction)
            axis = np.cross(direction, forward)
            included = 175. - 120. * progress if exercise == CURL else 90. + 85. * progress
            fore = _rotation(axis, np.radians(180. - included)) @ upper
        else:
            # Raise in the scapular plane with a constant 15-degree elbow bend.
            outward = np.array([sign * np.cos(np.radians(15.)), -np.sin(np.radians(15.)), 0.])
            axis = np.cross(down, outward)
            lift = _rotation(axis, np.radians(10. + 80. * progress))
            upper = lift @ _swing(rest, down)
            bent = down * np.cos(np.radians(15.)) + forward * np.sin(np.radians(15.))
            fore = lift @ _swing(rest, bent)
        for name, turn in (('Arm', upper), ('ForeArm', fore), ('Hand', fore)):
            if side + name in result:
                result[side + name] = turn
    return result


def prepare(exercise, keys):
    """The same fixed-axis paths interpolate without changing the elbow hinge."""
    if not keys:
        return keys
    return [pose(exercise, key, index / max(1, len(keys) - 1)) for index, key in enumerate(keys)]
