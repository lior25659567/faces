"""Recorded preview angles keep rendered visibility and fold checks consistent."""
import math

DEFAULT_VIEWS = ((35., 10.), (90., 5.))


def padded_box(low, high):
    """Fit the entire character even when it lies horizontally with no props."""
    low, high = tuple(map(float, low)), tuple(map(float, high))
    if len(low) != 3 or len(high) != 3 or any(not math.isfinite(v) for v in low + high) or any(b < a for a, b in zip(low, high)):
        raise ValueError('Preview bounds must contain finite ordered 3D corners.')
    centre = [(a + b) / 2 for a, b in zip(low, high)]
    half = max(max(b - a for a, b in zip(low, high)), .5) * .55
    return [[value - half for value in centre], [value + half for value in centre]]


def views_for(exercise, side_view=True):
    # An opaque footplate hides the body from the foot-end oblique camera.
    # Both sides expose the actual soles, backrest, headrest and rail motion.
    views = ((90., 5.), (-90., 5.)) if exercise == 'Leg Press (seated machine)' else DEFAULT_VIEWS
    return views if side_view else views[:1]


def angles_from_report(report):
    """Return explicit world angles; old reports retain their legacy cameras.

    Malformed recorded views are an error, not permission to measure visibility
    from a different camera than the one used to render the result.
    """
    if 'preview_cameras' not in report:
        return None
    values = report['preview_cameras']
    if report.get('preview_camera_space') != 'world' or not isinstance(values, list) or len(values) not in (1, 2):
        raise ValueError('The recorded preview cameras are incomplete.')
    result = []
    for view in values:
        if not isinstance(view, (tuple, list)) or len(view) != 2 or any(
                type(value) not in (float, int) or not math.isfinite(value) for value in view):
            raise ValueError('The recorded preview camera angles are invalid.')
        azimuth, elevation = view
        if not -360. <= azimuth <= 360. or not -89. <= elevation <= 89.:
            raise ValueError('The recorded preview camera angles are outside the supported range.')
        result.append((float(azimuth), float(elevation)))
    return tuple(result)
