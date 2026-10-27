"""Mechanical bench-press trajectory and independent all-frame range checks.

An oblique photograph does not reliably reconstruct elbow depth. This profile
uses the photographs' phase timing, then enforces the declared flat-bench
movement: lockout over shoulders, lower-chest descent, fixed grip and support.
Its fitted path is an explicit correction, not a calibrated source measurement.
"""
import numpy as np

SIDES = ('left', 'right')


def progress_at(frame, first, last, phase_frames):
    if (type(first) is not int or type(last) is not int or first >= last
            or not isinstance(phase_frames, dict)
            or phase_frames.get('start') != first
            or type(phase_frames.get('end')) is not int
            or not first < phase_frames['end'] <= last):
        raise ValueError('A bench press needs recorded start and lower-chest frames.')
    low = phase_frames['end']
    hold = max(1, round((low - first) * .2 / 1.5))
    if frame <= low:
        t = (frame - first) / (low - first)
    elif frame <= min(last, low + hold):
        t = 1.
    else:
        t = 1. - (frame - low - hold) / (last + 1 - low - hold)
    t = float(np.clip(t, 0., 1.))
    return t * t * (3. - 2. * t)


def path_at(top, bottom, progress):
    top, bottom = np.asarray(top, dtype=float), np.asarray(bottom, dtype=float)
    if (top.shape != (3,) or bottom.shape != (3,) or not np.isfinite([*top, *bottom, progress]).all()
            or not 0. <= progress <= 1. or top[2] <= bottom[2]):
        raise ValueError('A bench path needs a finite lockout above its lower-chest position.')
    return top * (1. - progress) + bottom * progress


def summarize(samples, height_m, phase_frames):
    if not isinstance(samples, list) or not samples or not np.isfinite(height_m) or height_m <= 0:
        raise ValueError('Bench range needs complete measured movement frames and a positive height.')
    first, last = samples[0].get('frame'), samples[-1].get('frame')
    progress_at(first, first, last, phase_frames)
    if [sample.get('frame') for sample in samples] != list(range(first, last + 1)):
        raise ValueError('Bench measurements must cover every movement frame exactly once.')
    points = []
    for sample in samples:
        try:
            values = [*sample['bar_m'], *sample['hip_m']]
            for name in ('shoulders_m', 'elbows_m', 'wrists_m'):
                if set(sample[name]) != set(SIDES):
                    raise ValueError('Missing bilateral bench landmarks.')
                for side in SIDES:
                    if len(sample[name][side]) != 3:
                        raise ValueError('Invalid bench point.')
                    values.extend(sample[name][side])
            if set(sample['elbows_deg']) != set(SIDES):
                raise ValueError('Missing bench elbows.')
            values.extend(sample['elbows_deg'].values())
            if (len(sample['bar_m']) != 3 or len(sample['hip_m']) != 3
                    or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) for v in values)
                    or any(not 0. <= a <= 180. for a in sample['elbows_deg'].values())):
                raise ValueError('Nonfinite bench landmarks.')
            for side in SIDES:
                upper = np.asarray(sample['shoulders_m'][side]) - sample['elbows_m'][side]
                fore = np.asarray(sample['wrists_m'][side]) - sample['elbows_m'][side]
                divisor = np.linalg.norm(upper) * np.linalg.norm(fore)
                if divisor <= 1e-10:
                    raise ValueError('Bench elbow has zero-length segments.')
                measured_angle = float(np.degrees(np.arccos(np.clip(upper @ fore / divisor, -1., 1.))))
                if abs(measured_angle - sample['elbows_deg'][side]) > .1:
                    raise ValueError('Bench elbow angles disagree with the measured landmarks.')
        except (KeyError, TypeError):
            raise ValueError('Bench movement lacks measured bilateral landmarks.') from None
        points.append(np.asarray(sample['bar_m'], dtype=float))
    bars = np.asarray(points)
    top, bottom = samples[0], samples[phase_frames['end'] - first]
    shoulder = np.mean([top['shoulders_m'][s] for s in SIDES], axis=0)
    hip = np.asarray(top['hip_m'])
    axis = hip - shoulder
    axis[2] = 0.
    length = float(np.linalg.norm(axis))
    if length <= 1e-6:
        raise ValueError('Bench torso landmarks overlap.')
    axis /= length
    longitudinal = (bars - shoulder) @ axis
    tilt, drift, asymmetry = [], [], []
    for sample in samples:
        drift.append(np.linalg.norm(np.asarray(sample['hip_m']) - hip))
        for side in SIDES:
            drift.append(np.linalg.norm(np.asarray(sample['shoulders_m'][side]) - top['shoulders_m'][side]))
            forearm = np.asarray(sample['wrists_m'][side]) - sample['elbows_m'][side]
            size = np.linalg.norm(forearm)
            if size <= 1e-9:
                raise ValueError('Bench forearm has zero length.')
            tilt.append(float(np.degrees(np.arccos(np.clip(forearm[2] / size, -1., 1.)))))
        asymmetry.append(abs(sample['elbows_deg']['left'] - sample['elbows_deg']['right']))
    lower_index = phase_frames['end'] - first
    # Changes during the short bottom hold may only be floating-point noise.
    hold = max(1, round((phase_frames['end'] - first) * .2 / 1.5))
    wrong = [0.]
    for index in range(1, len(samples)):
        dz = bars[index, 2] - bars[index - 1, 2]
        dy = longitudinal[index] - longitudinal[index - 1]
        if index <= lower_index:
            wrong.extend((max(0., dz), max(0., -dy)))
        elif index <= lower_index + hold:
            wrong.extend((abs(dz), abs(dy)))
        else:
            wrong.extend((max(0., -dz), max(0., dy)))
    return {'frames': len(samples), 'first_frame': first, 'last_frame': last, 'height_m': float(height_m),
        'phase_frames': dict(phase_frames), 'per_frame': samples,
        'top_elbows_deg': top['elbows_deg'], 'bottom_elbows_deg': bottom['elbows_deg'],
        'lockout_shortfall_deg': max(0., 175. - min(top['elbows_deg'].values())),
        'bottom_flexion_excess_deg': max(0., max(bottom['elbows_deg'].values()) - 105.),
        'bottom_overfold_deg': max(0., 65. - min(bottom['elbows_deg'].values())),
        'bar_vertical_travel_mm': float((bars[0, 2] - bars[lower_index, 2]) * 1000),
        'travel_shortfall_mm': max(0., float(.12 * height_m - (bars[0, 2] - bars[lower_index, 2]))) * 1000,
        'max_forearm_tilt_deg': max(tilt), 'forearm_tilt_excess_deg': max(0., max(tilt) - 30.),
        'bar_torso_path_excess_mm': max(0., float(-longitudinal.min() - .03 * height_m),
            float(longitudinal.max() - .6 * length)) * 1000,
        'bottom_chest_shortfall_mm': max(0., float(.2 * length - longitudinal[lower_index])) * 1000,
        'lockout_shoulder_offset_mm': max(0., abs(float(longitudinal[0])) - .03 * height_m) * 1000,
        'stroke_reversal_mm': max(wrong) * 1000, 'stationary_body_drift_mm': max(drift) * 1000,
        'elbow_asymmetry_deg': max(asymmetry)}
