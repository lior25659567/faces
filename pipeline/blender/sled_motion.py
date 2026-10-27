"""Measured leg-press range and stationary body support, without Blender.

The reference rail remains an inferred fit. Known generated joint coordinates
must still show a leg-only press with the supported torso and hands stationary.
"""
import math
import numpy as np

SIDES = ('left', 'right')
PARTS = ('hip', 'knee', 'ankle', 'shoulder', 'wrist')


def stationary_seat(points, height, width, body_axis):
    """Fit one finite posterior pad to every posed pelvis surface.

    Hip/upper-thigh blending changes the lowest anterior vertex during knee
    flexion. A broad pad placed under that one initial vertex can lose contact.
    Find a stationary posterior pressure patch; do not move the pad per frame.
    """
    points = np.asarray(points, dtype=float)
    axis = np.asarray(body_axis, dtype=float)[:2].copy()
    if (points.ndim != 3 or points.shape[2] != 3 or not len(points) or points.shape[1] < 6
            or not np.isfinite(points).all() or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-8
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0
                       for v in (height, width))):
        raise ValueError('A stationary leg-press seat needs complete finite pelvis surfaces.')
    axis /= np.linalg.norm(axis)
    origin = np.median(points[0, :, :2], axis=0)
    candidates = []
    for offset in np.arange(0., .1001, .01):
        centre = origin + axis * offset * height
        for ratio in (.22, .20, .18, .16, .14, .12):
            length = ratio * height
            samples = [sample[(abs(sample[:, 0] - centre[0]) <= width / 2 - .001)
                              & (abs(sample[:, 1] - centre[1]) <= length / 2 - .001)] for sample in points]
            if any(len(sample) < 6 for sample in samples):
                continue
            lows = [float(sample[:, 2].min()) for sample in samples]
            lower, upper = max(lows) - .005, min(lows) + .003
            if lower > upper:
                continue
            plane = (lower + upper) / 2
            if min(int(np.count_nonzero(abs(sample[:, 2] - plane) <= .005)) for sample in samples) < 3:
                continue
            candidates.append((offset + max(0., .16 - ratio), -ratio, centre, length, plane,
                               max(lows) - plane, plane - min(lows)))
    if not candidates:
        raise ValueError('No finite fixed pelvis pad can maintain the leg-press seat contact through the rep.')
    _, _, centre, length, plane, gap, depth = min(candidates, key=lambda value: value[:2])
    return {'centre_m': [float(centre[0]), float(centre[1]), plane], 'width_m': float(width),
            'length_m': float(length), 'frames': len(points), 'max_gap_mm': max(0., gap) * 1000,
            'max_penetration_mm': max(0., depth) * 1000,
            'method': 'One fixed finite posterior pelvis pad fitted to actual all-frame skin contact.'}


def _point(value):
    if (not isinstance(value, (list, tuple)) or len(value) != 3
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value)):
        raise ValueError('Leg-press motion requires finite measured joint coordinates.')
    return np.asarray(value, dtype=float)


def _included(a, joint, b):
    a, b = a - joint, b - joint
    scale = float(np.linalg.norm(a) * np.linalg.norm(b))
    if scale <= 1e-12:
        raise ValueError('Leg-press knee measurements require two nonzero segments.')
    return float(np.degrees(np.arccos(np.clip(a @ b / scale, -1., 1.))))


def summarize(samples, height, phase_frames):
    if (isinstance(height, bool) or not isinstance(height, (int, float)) or not math.isfinite(height) or height <= 0
            or not isinstance(samples, list) or len(samples) < 2 or not all(isinstance(s, dict) for s in samples)):
        raise ValueError('Leg-press motion requires every frame and a finite positive character height.')
    first, last = samples[0].get('frame'), samples[-1].get('frame')
    if (type(first) is not int or type(last) is not int or first >= last
            or [s.get('frame') for s in samples] != list(range(first, last + 1))
            or not isinstance(phase_frames, dict) or phase_frames.get('start') != first
            or type(phase_frames.get('end')) is not int or not first < phase_frames['end'] <= last
            or any(type(v) is not int or not first <= v <= last for v in phase_frames.values())):
        raise ValueError('Leg-press motion requires a complete timeline and recorded source positions.')
    measured = []
    for sample in samples:
        points = sample.get('points_m')
        if not isinstance(points, dict) or set(points) != {*PARTS, 'head'}:
            raise ValueError('Leg-press motion requires torso, head, arms and both leg measurements.')
        if any(not isinstance(points[name], dict) or set(points[name]) != set(SIDES) for name in PARTS):
            raise ValueError('Both sides must be measured in every leg-press frame.')
        values = {name: {side: _point(points[name][side]) for side in SIDES} for name in PARTS}
        values['head'] = _point(points['head'])
        knees = {side: _included(values['hip'][side], values['knee'][side], values['ankle'][side]) for side in SIDES}
        measured.append((values, knees))
    start, end = measured[0], measured[phase_frames['end'] - first]
    body_drift = hand_drift = asymmetry = 0.
    for values, knees in measured:
        body_drift = max(body_drift, float(np.linalg.norm(values['head'] - start[0]['head'])))
        for side in SIDES:
            for part in ('hip', 'shoulder'):
                body_drift = max(body_drift, float(np.linalg.norm(values[part][side] - start[0][part][side])))
            hand_drift = max(hand_drift, float(np.linalg.norm(values['wrist'][side] - start[0]['wrist'][side])))
        asymmetry = max(asymmetry, abs(knees['left'] - knees['right']))
    return {'frames': len(samples), 'measured_frames': len(samples), 'first_frame': first, 'last_frame': last,
            'height_m': float(height), 'phase_frames': dict(phase_frames), 'per_frame': samples,
            'max_supported_body_drift_mm': body_drift * 1000,
            'max_hand_drift_mm': hand_drift * 1000,
            'top_knees_deg': dict(start[1]), 'bottom_knees_deg': dict(end[1]),
            'top_extension_shortfall_deg': max(0., 170. - min(start[1].values())),
            'bottom_flexion_shortfall_deg': max(0., max(end[1].values()) - 110.),
            'bottom_overfold_deg': max(0., 60. - min(end[1].values())),
            'max_knee_asymmetry_deg': asymmetry,
            'method': 'Actual all-frame joint coordinates; fixed supported torso and hands, source-phase leg range.'}
