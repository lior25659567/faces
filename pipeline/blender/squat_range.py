"""All-frame geometry requirements for the declared squat movement preset.

The fitted character's measured rest-axis corrections account for its offset
cartoon hip/shoulder joints. They do not calibrate any reference photograph.
"""
import math

import numpy as np

SIDES = ('left', 'right')
PHASE_FRAMES = {'start': 1, 'middle': 19, 'end': 38}
POINT_FIELDS = ('hips_m', 'knees_m', 'ankles_m', 'shoulders_m')
SPINE = ('Hips', 'Spine', 'Spine1', 'Spine2', 'Neck')


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _point(value):
    if not isinstance(value, (list, tuple)) or len(value) != 3 or not all(_finite(v) for v in value):
        raise ValueError('Squat measurements require finite three-dimensional points.')
    return np.asarray(value, dtype=float)


def _turn(value):
    matrix = np.asarray(value, dtype=float)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(matrix), 1., atol=1e-5)):
        raise ValueError('Squat measurements require the fitted rest-axis rotations.')
    return matrix


def _angle(a, b):
    length = np.linalg.norm(a) * np.linalg.norm(b)
    if length <= 1e-10:
        raise ValueError('Squat joint segments must have nonzero length.')
    return float(np.degrees(np.arccos(np.clip(a @ b / length, -1., 1.))))


def summarize(samples, height_m, phase_frames, torso_turn, leg_turns):
    """Recompute movement quality from every keyed frame and measured rest axes."""
    if not _finite(height_m) or height_m <= 0 or not isinstance(samples, list) or len(samples) < 3:
        raise ValueError('A squat needs a positive character height and a complete movement.')
    if not all(isinstance(sample, dict) for sample in samples):
        raise ValueError('Squat measurements need a record for every frame.')
    first, last = samples[0].get('frame'), samples[-1].get('frame')
    if (type(first) is not int or type(last) is not int or first >= last
            or [sample.get('frame') for sample in samples] != list(range(first, last + 1))
            or phase_frames != PHASE_FRAMES or first != 1 or last != 69):
        raise ValueError('The squat preset must preserve all 69 frames and its pinned source phases.')
    torso = _turn(torso_turn)
    if not isinstance(leg_turns, dict) or set(leg_turns) != {f'{part}.{side}' for part in ('thigh', 'shin') for side in SIDES}:
        raise ValueError('The squat needs both fitted thigh and shin rest axes.')
    legs = {name: _turn(value) for name, value in leg_turns.items()}
    measured, hips, knee_symmetry, bar_tilts, spines = [], [], [], [], []
    for sample in samples:
        points = {}
        for name in POINT_FIELDS:
            values = sample.get(name)
            if not isinstance(values, dict) or set(values) != set(SIDES):
                raise ValueError('Both squat legs and shoulders must be measured in every frame.')
            points[name] = {side: _point(values[side]) for side in SIDES}
        bar, axis = _point(sample.get('bar_m')), _point(sample.get('bar_axis_m'))
        if abs(np.linalg.norm(axis) - 1.) > .001:
            raise ValueError('The measured squat shaft direction must be normalized.')
        spine = sample.get('spine_m')
        if not isinstance(spine, dict) or set(spine) != set(SPINE):
            raise ValueError('The complete squat spine must be measured in every frame.')
        spine = {name: _point(value) for name, value in spine.items()}
        hip = np.mean(list(points['hips_m'].values()), axis=0)
        shoulder = np.mean(list(points['shoulders_m'].values()), axis=0)
        trunk = torso @ (shoulder - hip)
        knees, hip_angles = {}, {}
        for side in SIDES:
            thigh = legs['thigh.' + side] @ (points['knees_m'][side] - points['hips_m'][side])
            shin = legs['shin.' + side] @ (points['ankles_m'][side] - points['knees_m'][side])
            knees[side] = _angle(-thigh, shin)
            hip_angles[side] = _angle(trunk, thigh)
        measured.append({'frame': sample['frame'], 'knees_deg': knees, 'hips_deg': hip_angles,
                         'torso_lean_deg': _angle(trunk, np.array([0., 0., 1.]))})
        hips.append(hip)
        knee_symmetry.append(abs(knees['left'] - knees['right']))
        bar_tilts.append(float(np.degrees(np.arcsin(np.clip(abs(axis[2]), 0., 1.)))))
        directions = [spine[b] - spine[a] for a, b in zip(SPINE, SPINE[1:])]
        spines.append([_angle(a, b) for a, b in zip(directions, directions[1:])])
    start, bottom, finish = measured[0], measured[PHASE_FRAMES['end'] - first], measured[-1]
    bottom_sample = samples[PHASE_FRAMES['end'] - first]
    hip_bottom = np.mean([bottom_sample['hips_m'][side][2] for side in SIDES])
    knee_bottom = np.mean([bottom_sample['knees_m'][side][2] for side in SIDES])
    pelvis_travel = float(hips[0][2] - hips[PHASE_FRAMES['end'] - first][2])
    seam_points = [np.linalg.norm(_point(samples[-1][field][side]) - _point(samples[0][field][side]))
                   for field in POINT_FIELDS for side in SIDES]
    seam_points.append(np.linalg.norm(_point(samples[-1]['bar_m']) - _point(samples[0]['bar_m'])))
    spine_deviation = float(np.max(abs(np.asarray(spines) - spines[0])))
    return {'frames': len(samples), 'measured_frames': len(samples), 'first_frame': first, 'last_frame': last, 'height_m': float(height_m),
            'phase_frames': dict(phase_frames), 'torso_turn': torso.tolist(),
            'leg_turns': {name: value.tolist() for name, value in legs.items()}, 'per_frame': samples,
            'start': start, 'bottom': bottom, 'finish': finish,
            'top_knee_extension_shortfall_deg': max(0., 165. - min(v for pose in (start, finish) for v in pose['knees_deg'].values())),
            'top_hip_extension_shortfall_deg': max(0., 160. - min(v for pose in (start, finish) for v in pose['hips_deg'].values())),
            'bottom_knee_flexion_excess_deg': max(0., max(bottom['knees_deg'].values()) - 100.),
            'bottom_hip_flexion_excess_deg': max(0., max(bottom['hips_deg'].values()) - 110.),
            'depth_shortfall_mm': max(0., float(hip_bottom - knee_bottom - .01 * height_m)) * 1000,
            'pelvis_travel_mm': pelvis_travel * 1000,
            'pelvis_travel_shortfall_mm': max(0., .12 * height_m - pelvis_travel) * 1000,
            'knee_asymmetry_deg': max(knee_symmetry), 'bar_level_excess_deg': max(0., max(bar_tilts) - 2.),
            'spine_deviation_deg': spine_deviation, 'loop_position_error_mm': max(seam_points) * 1000,
            'method': 'Pinned 3D clip with declared fitted back-bar grip; full keyed bilateral range, depth, spine, shaft and loop measurements; photos are context only.'}
