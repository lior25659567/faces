"""Read a visible-side plank without treating monocular depth as floor calibration."""
import math
import numpy as np


def _angle(a, b):
    scale = np.linalg.norm(a) * np.linalg.norm(b)
    if scale <= 1e-9:
        raise ValueError('Coincident plank landmarks.')
    return math.degrees(math.acos(float(np.clip(np.dot(a, b) / scale, -1., 1.))))


def read(reference):
    try:
        phase = reference['phases'][0]
        side = phase['camera_side']
        if phase['view'] != 'side' or side not in ('left', 'right') or phase['facing'] not in ('left', 'right'):
            return None
        forward = 1. if phase['facing'] == 'right' else -1.
        pixels, world = phase['pixels'], phase['world']
        names = [side + '_' + part for part in ('shoulder', 'hip', 'knee', 'ankle', 'elbow', 'wrist')]
        if any(phase.get('visibility', {}).get(name, 0.) < .5 for name in names):
            return None
        p = {name: np.asarray(pixels[name], dtype=float) for name in names}
        w = {name: np.asarray(world[name], dtype=float) for name in names}
        if any(value.shape != (2,) or not np.isfinite(value).all() for value in p.values()):
            return None
        if any(value.shape != (3,) or not np.isfinite(value).all() for value in w.values()):
            return None
        def flat(name):
            point = p[side + '_' + name]
            return np.array([point[0] * forward, -point[1]])
        shoulder, hip, knee, ankle, elbow, wrist = map(flat, ('shoulder', 'hip', 'knee', 'ankle', 'elbow', 'wrist'))
        shoulder_mid = (np.array(pixels['left_shoulder']) + np.array(pixels['right_shoulder'])) / 2
        hip_mid = (np.array(pixels['left_hip']) + np.array(pixels['right_hip'])) / 2
        torso = np.array([(shoulder_mid[0] - hip_mid[0]) * forward, hip_mid[1] - shoulder_mid[1]])
        beta = math.degrees(math.atan2(torso[1], torso[0]))
        upper = w[side + '_elbow'] - w[side + '_shoulder']
        fore = w[side + '_wrist'] - w[side + '_elbow']
        back = w[side + '_hip'] - w[side + '_shoulder']
        result = {'image': phase.get('image'), 'camera_side': side,
            'method': 'visible-side pixels for body; original 3D arm angles',
            'torso_above_horizontal_deg': beta,
            'thigh_angle_deg': math.degrees(math.atan2(*(knee - hip)[::-1])),
            'shin_angle_deg': math.degrees(math.atan2(*(ankle - knee)[::-1])),
            'knee_included_deg': _angle(hip - knee, ankle - knee),
            'hip_included_deg': _angle(shoulder - hip, knee - hip),
            'elbow_included_deg': _angle(-upper, fore),
            'upperarm_vs_torso_deg': _angle(upper, back),
            'pixel_elbow_included_deg': _angle(shoulder - elbow, wrist - elbow)}
        if not -15 <= beta <= 20 or result['hip_included_deg'] < 150:
            return None
        return result
    except (KeyError, TypeError, ValueError, IndexError):
        return None
