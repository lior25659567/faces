"""Technique measurements for a front lat pulldown, independent of Blender.

These checks establish the direction of the pull on the fitted character. They
do not turn depth inferred from a reference photograph into calibrated anatomy.
"""
import numpy as np

SIDES = ('left', 'right')


def _included(a, joint, b):
    u, v = np.asarray(a, dtype=float) - joint, np.asarray(b, dtype=float) - joint
    denominator = float(np.linalg.norm(u) * np.linalg.norm(v))
    if not np.isfinite(denominator) or denominator <= 1e-12:
        raise ValueError('Pulldown angles require two finite nonzero segments.')
    return float(np.degrees(np.arccos(np.clip(u @ v / denominator, -1., 1.))))


def progress_at(frame, first, last, phase_frames):
    if (type(first) is not int or type(last) is not int or first > last or
            not isinstance(phase_frames, dict) or phase_frames.get('start') != first or
            type(phase_frames.get('end')) is not int or not first < phase_frames['end'] <= last or
            type(frame) is not int or not first <= frame <= last):
        raise ValueError('The pulldown needs its complete recorded start/end timeline.')
    end = phase_frames['end']
    # The generated demonstration contains a 0.2-second bottom hold within
    # its 2.7-second loop. Preserve those samples instead of starting the cable
    # return immediately after the recorded chest position.
    hold = max(1, round((last - first + 1) * .2 / 2.7))
    if frame <= end:
        t = (frame - first) / (end - first)
    elif frame <= min(last, end + hold):
        t = 1.
    else:
        # The generated loop excludes its duplicated starting pose. Preserve
        # that convention instead of snapping to a second arbitrary endpoint.
        t = (last + 1 - frame) / (last + 1 - end - hold)
    t = float(np.clip(t, 0., 1.))
    return t * t * (3. - 2. * t)


def summarize(samples, height, phase_frames):
    if isinstance(height, bool) or not isinstance(height, (int, float)) or not np.isfinite(height) or height <= 0:
        raise ValueError('The pulldown needs a finite positive character height.')
    if not isinstance(samples, list) or len(samples) < 2:
        raise ValueError('A pulldown needs every generated frame.')
    first, last = samples[0].get('frame'), samples[-1].get('frame')
    progress_at(first, first, last, phase_frames)
    if [s.get('frame') for s in samples] != list(range(first, last + 1)):
        raise ValueError('The pulldown measurements must cover every frame in order.')
    computed = []
    for original in samples:
        sample = dict(original)
        for key in ('shoulders_m', 'elbows_m', 'wrists_m'):
            values = sample.get(key)
            if not isinstance(values, dict) or set(values) != set(SIDES):
                raise ValueError('Both pulldown arms must be measured in every frame.')
            if any(np.asarray(v).shape != (3,) or not np.isfinite(v).all() for v in values.values()):
                raise ValueError('Pulldown joint positions must be finite three-dimensional points.')
        for key in ('bar_m', 'hip_m'):
            value = np.asarray(sample.get(key))
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError('The pulldown bar and torso must be measured in every frame.')
        shoulders = {side: np.asarray(sample['shoulders_m'][side], dtype=float) for side in SIDES}
        torso_down = np.asarray(sample['hip_m']) - (shoulders['left'] + shoulders['right']) / 2
        sample['elbows_deg'] = {side: _included(shoulders[side], np.asarray(sample['elbows_m'][side]),
            np.asarray(sample['wrists_m'][side])) for side in SIDES}
        sample['upperarms_deg'] = {side: _included(torso_down, np.zeros(3),
            np.asarray(sample['elbows_m'][side]) - shoulders[side]) for side in SIDES}
        computed.append(sample)
    bottom = computed[phase_frames['end'] - first]
    above_wrist = max((s['elbows_m'][side][2] - s['wrists_m'][side][2]) * 1000
                      for s in computed for side in SIDES)
    above_shoulder = max((bottom['elbows_m'][side][2] - bottom['shoulders_m'][side][2]) * 1000 for side in SIDES)
    travel = float((computed[0]['bar_m'][2] - bottom['bar_m'][2]) * 1000)
    return {'frames': len(samples), 'measured_frames': len(samples), 'first_frame': first, 'last_frame': last,
            'height_m': float(height), 'phase_frames': dict(phase_frames), 'per_frame': computed,
            'max_elbow_above_wrist_mm': max(0., float(above_wrist)),
            'bottom_elbow_above_shoulder_mm': max(0., float(above_shoulder)),
            'bottom_upperarm_excess_deg': max(0., max(bottom['upperarms_deg'].values()) - 70.),
            'bottom_upperarm_deg': dict(bottom['upperarms_deg']),
            'top_elbows_deg': dict(computed[0]['elbows_deg']),
            'top_extension_shortfall_deg': max(0., 175. - min(computed[0]['elbows_deg'].values())),
            'bar_travel_mm': travel, 'travel_shortfall_mm': max(0., 120. * height - travel),
            'method': 'all-frame elbow-to-wrist direction and recorded bottom-phase elbow/torso geometry'}
