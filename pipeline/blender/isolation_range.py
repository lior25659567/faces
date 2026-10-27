"""All-frame mechanical form measurements for standing isolation exercises.

These declared form limits do not approve inferred source depth. Genuine photo
differences remain in the separate comparison rows, including incomplete
demonstrated pushdown extension.
"""
import numpy as np

CURL = 'Dumbbell Bicep Curl (standing, both arms)'
RAISE = 'Dumbbell Lateral Raise'
PUSHDOWN = 'Tricep Pushdown (cable, bar or rope attachment)'
EXERCISES = {CURL, RAISE, PUSHDOWN}
SIDES = ('left', 'right')


def _point(value):
    if (not isinstance(value, (list, tuple)) or len(value) != 3
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) for v in value)):
        raise ValueError('Isolation form needs finite measured three-dimensional joints.')
    return np.asarray(value, dtype=float)


def _angle(a, b):
    size = np.linalg.norm(a) * np.linalg.norm(b)
    if not np.isfinite(size) or size < 1e-10:
        raise ValueError('Isolation form needs nonzero measured body segments.')
    return float(np.degrees(np.arccos(np.clip(a @ b / size, -1., 1.))))


def summarize(samples, height, phase_frames, exercise):
    if (exercise not in EXERCISES or isinstance(height, bool) or not isinstance(height, (int, float))
            or not np.isfinite(height) or height <= 0 or not isinstance(samples, list) or len(samples) < 2
            or not all(isinstance(sample, dict) for sample in samples) or not isinstance(phase_frames, dict)):
        raise ValueError('Isolation form needs its exercise, measured height and complete movement samples.')
    first, last = samples[0].get('frame'), samples[-1].get('frame')
    if (type(first) is not int or type(last) is not int or last <= first
            or [sample.get('frame') for sample in samples] != list(range(first, last + 1))
            or phase_frames.get('start') != first or type(phase_frames.get('end')) is not int
            or not first < phase_frames['end'] <= last):
        raise ValueError('Isolation form needs every generated frame and its recorded start/end positions.')
    measured = []
    for sample in samples:
        points = {}
        for name in ('shoulders_m', 'elbows_m', 'wrists_m', 'hips_m', 'knees_m', 'ankles_m'):
            pair = sample.get(name)
            if not isinstance(pair, dict) or set(pair) != set(SIDES):
                raise ValueError('Isolation form needs both sides of every required joint.')
            points[name] = {side: _point(pair[side]) for side in SIDES}
        spine = _point(sample.get('spine2_m'))
        hip = np.mean(list(points['hips_m'].values()), axis=0)
        torso = np.mean(list(points['shoulders_m'].values()), axis=0) - hip
        values = {'frame': sample['frame'], 'body_lean_deg': _angle(torso, np.array([0., 0., 1.])),
            'hip_mid': hip, 'spine': spine, 'elbows_deg': {}, 'upperarms_deg': {},
            'knees_deg': {}, 'hips_deg': {}, 'upper_vectors': {}}
        for side in SIDES:
            shoulder, elbow, wrist, pelvis, knee, ankle = [points[name][side] for name in
                ('shoulders_m', 'elbows_m', 'wrists_m', 'hips_m', 'knees_m', 'ankles_m')]
            values['elbows_deg'][side] = _angle(shoulder - elbow, wrist - elbow)
            values['upperarms_deg'][side] = _angle(-torso, elbow - shoulder)
            values['knees_deg'][side] = _angle(pelvis - knee, ankle - knee)
            values['hips_deg'][side] = _angle(torso, knee - pelvis)
            values['upper_vectors'][side] = elbow - shoulder
        measured.append(values)
    start, end = measured[0], measured[phase_frames['end'] - first]
    all_elbows = [m['elbows_deg'][s] for m in measured for s in SIDES]
    result = {'exercise': exercise, 'frames': len(samples), 'measured_frames': len(samples),
        'first_frame': first, 'last_frame': last, 'height_m': float(height),
        'phase_frames': dict(phase_frames), 'per_frame': samples,
        'start_elbows_deg': start['elbows_deg'], 'end_elbows_deg': end['elbows_deg'],
        'start_upperarms_deg': start['upperarms_deg'], 'end_upperarms_deg': end['upperarms_deg'],
        'body_lean_excess_deg': max(0., max(m['body_lean_deg'] for m in measured) - 10.),
        'knee_extension_shortfall_deg': max(0., 170. - min(m['knees_deg'][s] for m in measured for s in SIDES)),
        'hip_extension_shortfall_deg': max(0., 170. - min(m['hips_deg'][s] for m in measured for s in SIDES)),
        'body_drift_mm': max(np.linalg.norm(m[key] - start[key]) for m in measured for key in ('hip_mid', 'spine')) * 1000,
        'elbow_asymmetry_deg': max(abs(m['elbows_deg']['left'] - m['elbows_deg']['right']) for m in measured),
        'upperarm_drift_deg': max(_angle(m['upper_vectors'][s], start['upper_vectors'][s]) for m in measured for s in SIDES),
        'method': 'Declared standing isolation technique, measured from actual body and arm joints at every frame; separate photo comparisons retain uncertainty and source conflicts.'}
    if exercise == CURL:
        result.update(upperarm_position_excess_deg=max(0., max(m['upperarms_deg'][s] for m in measured for s in SIDES) - 25.),
            start_extension_shortfall_deg=max(0., 170. - min(start['elbows_deg'].values())),
            end_flexion_excess_deg=max(0., max(end['elbows_deg'].values()) - 65.),
            end_overfold_deg=max(0., 45. - min(end['elbows_deg'].values())))
    elif exercise == RAISE:
        result.update(start_elevation_excess_deg=max(0., max(start['upperarms_deg'].values()) - 20.),
            end_elevation_shortfall_deg=max(0., 85. - min(end['upperarms_deg'].values())),
            end_elevation_excess_deg=max(0., max(end['upperarms_deg'].values()) - 95.),
            elbow_bend_excess_deg=max(0., 160. - min(all_elbows), max(all_elbows) - 170.),
            elbow_bend_variation_deg=max(max(m['elbows_deg'][s] for m in measured) - min(m['elbows_deg'][s] for m in measured) for s in SIDES))
    else:
        result.update(upperarm_position_excess_deg=max(0., max(m['upperarms_deg'][s] for m in measured for s in SIDES) - 25.),
            start_bend_excess_deg=max(0., max(start['elbows_deg'].values()) - 105.),
            start_overfold_deg=max(0., 75. - min(start['elbows_deg'].values())),
            end_extension_shortfall_deg=max(0., 175. - min(end['elbows_deg'].values())))
    return result
