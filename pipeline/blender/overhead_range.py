"""All-frame mechanical range of a standing or seated overhead press.

The positions are measured on the fitted character, not inferred from a photo.
The declared demonstration phases still identify the bottom and lockout frames.
"""
import numpy as np

SIDES = ('left', 'right')


def progress_at(frame, first, last, phases):
    if (type(first) is not int or type(last) is not int or first >= last
            or not isinstance(phases, dict) or phases.get('start') != first
            or type(phases.get('end')) is not int or not first < phases['end'] < last
            or type(frame) is not int or not first <= frame <= last):
        raise ValueError('An overhead press needs every frame and its recorded bottom/lockout phases.')
    top = phases['end']
    # Both press profiles use one second of effort and a controlled 1.5 s
    # return. The generated loop excludes a duplicated final bottom sample.
    hold = max(1, round((last - first + 1) * .2 / 2.7))
    if frame <= top:
        t = (frame - first) / (top - first)
    elif frame <= top + hold:
        t = 1.
    else:
        t = 1. - (frame - top - hold) / (last + 1 - top - hold)
    t = float(np.clip(t, 0., 1.))
    return t * t * (3. - 2. * t)


def _point(value):
    if (not isinstance(value, (list, tuple)) or len(value) != 3
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) for v in value)):
        raise ValueError('An overhead measurement needs finite three-dimensional landmarks.')
    return np.asarray(value, dtype=float)


def _included(a, joint, b):
    u, v = a - joint, b - joint
    length = np.linalg.norm(u) * np.linalg.norm(v)
    if length <= 1e-10:
        raise ValueError('An overhead arm cannot have zero-length segments.')
    return float(np.degrees(np.arccos(np.clip(u @ v / length, -1., 1.))))


def summarize(samples, height_m, phase_frames):
    if (not isinstance(samples, list) or len(samples) < 3
            or isinstance(height_m, bool) or not isinstance(height_m, (int, float))
            or not np.isfinite(height_m) or height_m <= 0
            or not all(isinstance(sample, dict) for sample in samples)):
        raise ValueError('An overhead press needs complete measured frames and a finite positive height.')
    first, last = samples[0].get('frame'), samples[-1].get('frame')
    progress_at(first, first, last, phase_frames)
    if [s.get('frame') for s in samples] != list(range(first, last + 1)):
        raise ValueError('Every overhead press frame must be measured exactly once.')
    measured = []
    for sample in samples:
        points = {}
        for name in ('shoulders_m', 'hips_m', 'elbows_m', 'wrists_m', 'handles_m'):
            values = sample.get(name)
            if not isinstance(values, dict) or set(values) != set(SIDES):
                raise ValueError('Every overhead frame needs both arms and handles.')
            points[name] = {side: _point(values[side]) for side in SIDES}
        head = sample.get('head_top_m')
        if isinstance(head, bool) or not isinstance(head, (int, float)) or not np.isfinite(head):
            raise ValueError('The overhead head clearance must be measured in every frame.')
        points['hip_m'] = _point(sample.get('hip_m'))
        points['spine_m'] = _point(sample.get('spine_m'))
        points['head_m'] = _point(sample.get('head_m'))
        points['head_top_m'] = head
        points['elbows_deg'] = {side: _included(points['shoulders_m'][side], points['elbows_m'][side],
            points['wrists_m'][side]) for side in SIDES}
        measured.append(points)
    bottom, top = measured[0], measured[phase_frames['end'] - first]
    angles = [s['elbows_deg'] for s in measured]
    travels = {side: float((top['handles_m'][side][2] - bottom['handles_m'][side][2]) * 1000) for side in SIDES}
    stack = max(np.linalg.norm((top['wrists_m'][side] - top['shoulders_m'][side])[:2]) for side in SIDES)
    lean = []
    for sample in measured:
        shoulder_mid = (sample['shoulders_m']['left'] + sample['shoulders_m']['right']) / 2
        hip_mid = (sample['hips_m']['left'] + sample['hips_m']['right']) / 2
        axis = shoulder_mid - hip_mid
        lean.append(float(np.degrees(np.arctan2(np.linalg.norm(axis[:2]), axis[2]))))
    return {'frames': len(samples), 'measured_frames': len(samples), 'first_frame': first, 'last_frame': last,
        'height_m': float(height_m), 'phase_frames': dict(phase_frames), 'per_frame': samples,
        'bottom_elbows_deg': dict(bottom['elbows_deg']), 'top_elbows_deg': dict(top['elbows_deg']),
        'lockout_shortfall_deg': max(0., 175. - min(top['elbows_deg'].values())),
        'bottom_flexion_excess_deg': max(0., max(bottom['elbows_deg'].values()) - 110.),
        'bottom_overfold_deg': max(0., 65. - min(bottom['elbows_deg'].values())),
        'elbow_above_wrist_mm': max(0., max(float(s['elbows_m'][side][2] - s['wrists_m'][side][2]) * 1000
            for s in measured for side in SIDES)),
        'handle_travel_mm': travels, 'travel_shortfall_mm': max(0., .12 * height_m * 1000 - min(travels.values())),
        'peak_stack_excess_mm': max(0., float(stack - .10 * height_m) * 1000),
        'lockout_head_shortfall_mm': max(0., float(top['head_top_m'] + .01 * height_m
            - min(top['handles_m'][s][2] for s in SIDES)) * 1000),
        'elbow_asymmetry_deg': max(abs(a['left'] - a['right']) for a in angles),
        'hip_drift_mm': max(float(np.linalg.norm(s['hip_m'] - bottom['hip_m'])) * 1000 for s in measured),
        'torso_lean_deg': max(lean),
        'method': 'Actual bilateral press joints and handles at the recorded phases and every intervening frame; authored fitted trajectory, not calibrated photo depth'}
