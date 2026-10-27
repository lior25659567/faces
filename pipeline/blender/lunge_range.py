"""Actual one-shot lunge contacts and leg mechanics, independent of Blender."""
import math
import numpy as np

SIDES = ('left', 'right')
POINTS = tuple(f'{side}.{point}' for side in SIDES for point in ('heel', 'toe'))


def _point(value):
    point = np.asarray(value, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError('Lunge measurements need finite three-dimensional points.')
    return point


def included(a, joint, b):
    u, v = _point(a) - joint, _point(b) - joint
    denominator = float(np.linalg.norm(u) * np.linalg.norm(v))
    if denominator <= 1e-12:
        raise ValueError('Lunge range cannot be measured on collapsed leg segments.')
    return float(np.degrees(np.arccos(np.clip(u @ v / denominator, -1., 1.))))


def summarize(samples, height_m, phase_frames, source_times, fps):
    """Independently judge actual skin contacts, leg geometry and forward travel."""
    lead_leg = next((side for sample in samples if isinstance(sample, dict) for side in SIDES
                     if (sample.get('foot_states') or {}).get(side) == 'airborne'), None) if isinstance(samples, list) else None
    if (isinstance(height_m, bool) or not isinstance(height_m, (int, float)) or not np.isfinite(height_m)
            or height_m <= 0 or not isinstance(samples, list) or len(samples) < 3 or lead_leg not in SIDES
            or isinstance(fps, bool) or not isinstance(fps, (int, float)) or not np.isfinite(fps) or fps <= 0
            or not isinstance(source_times, list) or len(source_times) < 3
            or any(isinstance(t, bool) or not isinstance(t, (int, float)) or not np.isfinite(t) for t in source_times)
            or any(a >= b for a, b in zip(source_times, source_times[1:]))):
        raise ValueError('Lunge checks require complete actual movement measurements.')
    first, last = samples[0].get('frame'), samples[-1].get('frame')
    if (type(first) is not int or type(last) is not int
            or [sample.get('frame') for sample in samples] != list(range(first, last + 1))
            or not isinstance(phase_frames, dict) or phase_frames.get('start') != first
            or phase_frames.get('end') != last or type(phase_frames.get('middle')) is not int
            or not first < phase_frames['middle'] < last):
        raise ValueError('Lunge checks require every frame at its actual source comparison times.')
    duration = source_times[-1] - source_times[0]
    if len(samples) != math.ceil(duration * fps) + 1:
        raise ValueError('The lunge frame count must cover its actual one-shot source duration.')
    expected_times = [source_times[0] + min(index / fps, duration) for index in range(len(samples))]
    if any(not isinstance(sample, dict) or not isinstance(sample.get('time_s'), (int, float))
           or abs(sample['time_s'] - time) > 1e-5 for sample, time in zip(samples, expected_times)):
        raise ValueError('Every rendered lunge sample must use the actual chronological source clock.')
    gaps, penetration, drift, airborne, lengths, backward, floor = [], [], [], [], [], [], []
    knees, roots, previous_anchors, previous_time = [], [], {}, -math.inf
    for sample in samples:
        time = sample.get('time_s')
        if isinstance(time, bool) or not isinstance(time, (int, float)) or not np.isfinite(time) or time <= previous_time:
            raise ValueError('Every lunge frame must retain increasing source time.')
        previous_time = time
        points, anchors = sample.get('contact_points_m'), sample.get('contact_anchors_m')
        states = sample.get('foot_states')
        if not isinstance(points, dict) or set(points) != set(POINTS) or not isinstance(anchors, dict) or not anchors:
            raise ValueError('Actual heel and toe measurements and observed anchors are required.')
        expected = set()
        for side in SIDES:
            state = (states or {}).get(side)
            if state not in ('flat', 'heel', 'toe', 'airborne', 'uncertain'):
                raise ValueError('Lunge foot states are missing.')
            expected.update(f'{side}.{point}' for point in ('heel', 'toe') if state == 'flat' or state == point)
        if set(anchors) != expected:
            raise ValueError('Actual contacts cannot add an unobserved heel or toe support.')
        for name in POINTS:
            _point(points[name])
        for name, target in anchors.items():
            target = _point(target)
            if abs(target[2]) > 1e-7:
                raise ValueError('Lunge support anchors must lie on the actual floor.')
            if name in previous_anchors and np.linalg.norm(target - previous_anchors[name]) > 1e-7:
                raise ValueError('A planted heel/toe anchor cannot move or reset within its contact episode.')
            patch = (sample.get('contact_floor') or {}).get(name) or {}
            low, high = patch.get('min_z_m'), patch.get('max_z_m')
            if (isinstance(low, bool) or isinstance(high, bool) or not isinstance(low, (int, float))
                    or not isinstance(high, (int, float)) or not np.isfinite([low, high]).all() or high < low):
                raise ValueError('The actual supported heel/toe surface must be measured.')
            gaps.append(abs(low) * 1000)
            penetration.append(max(0., -low) * 1000)
            drift.append(float(np.linalg.norm(_point(points[name])[:2] - target[:2])) * 1000)
        previous_anchors = {name: _point(point) for name, point in anchors.items()}
        angles = {}
        for side in SIDES:
            hip = _point((sample.get('hips_m') or {}).get(side))
            knee = _point((sample.get('knees_m') or {}).get(side))
            ankle = _point((sample.get('ankles_m') or {}).get(side))
            rest = (sample.get('leg_lengths_m') or {}).get(side)
            if (not isinstance(rest, (list, tuple)) or len(rest) != 2
                    or not np.isfinite(rest).all() or min(rest) <= 0):
                raise ValueError('Both unchanged leg lengths must accompany every frame.')
            lengths.extend((abs(np.linalg.norm(knee - hip) - rest[0]) * 1000,
                            abs(np.linalg.norm(ankle - knee) - rest[1]) * 1000))
            axis = ankle - hip
            projection = hip + axis * float((knee - hip) @ axis) / max(float(axis @ axis), 1e-12)
            backward.append(max(0., -float((knee - projection)[1])) * 1000)
            angles[side] = included(hip, knee, ankle)
            clearance = (sample.get('foot_min_z_m') or {}).get(side)
            if not isinstance(clearance, (int, float)) or isinstance(clearance, bool) or not np.isfinite(clearance):
                raise ValueError('Both actual foot surfaces must be measured in every frame.')
            if states[side] == 'airborne':
                airborne.append(max(0., .002 - clearance) * 1000)
        body_floor = sample.get('body_min_z_m')
        if not isinstance(body_floor, (int, float)) or isinstance(body_floor, bool) or not np.isfinite(body_floor):
            raise ValueError('The full moving body must be measured against the floor.')
        floor.append(max(0., -body_floor) * 1000)
        knees.append(angles)
        roots.append(_point(sample.get('root_m')))
    if not airborne:
        raise ValueError('The observed forward step must include an actually airborne foot.')
    bottom = knees[phase_frames['middle'] - first][lead_leg]
    travel = float(roots[-1][1] - roots[0][1])
    return {'frames': len(samples), 'measured_frames': len(samples), 'first_frame': first, 'last_frame': last,
        'height_m': float(height_m), 'phase_frames': dict(phase_frames), 'lead_leg': lead_leg,
        'fps': float(fps), 'source_times': list(source_times),
        'per_frame': samples, 'mode': 'sequence', 'playback': 'once', 'source_inferred_depth': True,
        'max_contact_gap_mm': max(gaps), 'max_contact_penetration_mm': max(penetration),
        'max_anchor_drift_mm': max(drift), 'max_swing_clearance_shortfall_mm': max(airborne),
        'max_leg_length_error_mm': max(lengths), 'max_backward_knee_mm': max(backward),
        'max_body_floor_penetration_mm': max(floor), 'forward_travel_mm': travel * 1000,
        'missing_forward_travel': int(travel <= 1e-5),
        'max_reverse_root_mm': max(0., max(-float(b[1] - a[1]) for a, b in zip(roots, roots[1:]))) * 1000,
        'start_knees_deg': knees[0], 'middle_lead_knee_deg': bottom,
        'start_extension_shortfall_deg': max(0., 170. - min(knees[0].values())),
        'middle_knee_overfold_deg': max(0., 80. - bottom), 'middle_knee_depth_shortfall_deg': max(0., bottom - 110.),
        'method': 'actual named heel/toe contact episodes, fixed-length forward-knee mechanics and observed one-shot timing; monocular depth remains approximate'}
