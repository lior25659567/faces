"""Continuity of the actual keyed arms, independent of Blender and the camera.

These generous speed limits detect arm snaps; they do not certify technique.
Positions are relative to the hips so whole-body travel cannot mask or invent
an elbow/wrist jump. Quaternion signs do not change the measured rotation.
"""
import math
import numpy as np

POINTS = tuple(f'{part}.{side}' for side in ('left', 'right') for part in ('elbow', 'wrist'))
TURNS = tuple(f'{part}.{side}' for side in ('left', 'right') for part in ('arm', 'forearm', 'hand'))
MAX_SPEED_HEIGHTS_S = 2.
MAX_TURN_DEG_S = 720.


def _positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def _vector(value, size):
    if (not isinstance(value, (list, tuple)) or len(value) != size
            or any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) for v in value)):
        raise ValueError('Arm continuity requires finite measured positions and rotations.')
    return np.asarray(value, dtype=float)


def summarize(samples, height_m, fps, first, last, loop=False):
    """Recalculate all speeds from complete samples rather than saved maxima."""
    if (not _positive(height_m) or not _positive(fps) or type(first) is not int or type(last) is not int
            or last < first or type(loop) is not bool or not isinstance(samples, list)
            or len(samples) != last - first + 1 or not all(isinstance(s, dict) for s in samples)
            or any(type(s.get('frame')) is not int for s in samples)
            or [s.get('frame') for s in samples] != list(range(first, last + 1))):
        raise ValueError('Arm continuity requires every recorded movement frame and valid height/fps.')
    measured = []
    for sample in samples:
        if (not isinstance(sample.get('points_m'), dict) or set(sample['points_m']) != set(POINTS)
                or not isinstance(sample.get('turns_wxyz'), dict) or set(sample['turns_wxyz']) != set(TURNS)):
            raise ValueError('Both arms must be measured in every movement frame.')
        points = {name: _vector(sample['points_m'][name], 3) for name in POINTS}
        turns = {name: _vector(sample['turns_wxyz'][name], 4) for name in TURNS}
        for name, quat in turns.items():
            norm = float(np.linalg.norm(quat))
            if abs(norm - 1.) > .001:
                raise ValueError('Arm continuity requires normalized measured rotations.')
            turns[name] = quat / norm
        measured.append((sample['frame'], points, turns))
    pairs = list(zip(measured, measured[1:]))
    if loop and len(measured) > 1:
        pairs.append((measured[-1], measured[0]))
    speed = {'value': 0., 'frame': first, 'joint': None}
    turn = {'value': 0., 'frame': first, 'joint': None}
    for (_, before, rotations), (frame, after, next_rotations) in pairs:
        for name in POINTS:
            value = float(np.linalg.norm(after[name] - before[name])) * fps / height_m
            if value > speed['value']:
                speed = {'value': value, 'frame': frame, 'joint': name}
        for name in TURNS:
            dot = abs(float(rotations[name] @ next_rotations[name]))
            value = math.degrees(2 * math.acos(min(1., max(0., dot)))) * fps
            if value > turn['value']:
                turn = {'value': value, 'frame': frame, 'joint': name}
    return {'frames': len(samples), 'measured_frames': len(samples), 'first_frame': first, 'last_frame': last,
            'height_m': float(height_m), 'fps': float(fps), 'loop': loop, 'per_frame': samples,
            'max_speed_heights_s': speed['value'], 'speed_frame': speed['frame'], 'speed_joint': speed['joint'],
            'max_turn_deg_s': turn['value'], 'turn_frame': turn['frame'], 'turn_joint': turn['joint'],
            'limits': {'speed_heights_s': MAX_SPEED_HEIGHTS_S, 'turn_deg_s': MAX_TURN_DEG_S},
            'method': 'All-frame hip-relative elbow/wrist speed and shortest world arm/forearm/hand turns; snap detection only.'}


def remeasure(report, first, last):
    if (type(first) is not int or type(last) is not int or last < first
            or not isinstance(report, dict) or report.get('first_frame') != first or report.get('last_frame') != last
            or any(type(report.get(name)) is not int for name in ('frames', 'measured_frames', 'first_frame', 'last_frame'))
            or report.get('frames') != last - first + 1 or report.get('measured_frames') != last - first + 1
            or report.get('measurement_error')):
        raise ValueError('Arm continuity has missing or inconsistent movement coverage.')
    return summarize(report.get('per_frame'), report.get('height_m'), report.get('fps'), first, last,
                     loop=report.get('loop'))
