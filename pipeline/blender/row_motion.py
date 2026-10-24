"""Keep the body hinge stable in a reference-driven bent-over row."""
def stabilize_body(keys):
    """Hold the starting hinge and lower body while allowing the row's arms to move.

    Matrices are absolute turns, so holding the trunk alone would leave the head or
    thighs drifting. Preserve every non-arm turn, including neck and feet.
    """
    if not keys:
        return []
    def arm(name):
        return name.startswith(('LeftArm', 'RightArm', 'LeftForeArm', 'RightForeArm',
                                'LeftHand', 'RightHand'))
    return [{name: value if arm(name) else keys[0][name].copy() for name, value in key.items()}
            for key in keys]


# Numpy-only geometry used by the Blender row contact solve.
import math
import numpy as np


def _unit(value):
    value = np.asarray(value, dtype=float)
    length = np.linalg.norm(value)
    if not np.isfinite(length) or length < 1e-10:
        raise ValueError('Cannot solve a row with overlapping or nonfinite landmarks.')
    return value / length


def _rotation_between(a, b):
    a, b = _unit(a), _unit(b)
    axis, cosine = np.cross(a, b), float(a @ b)
    sine = np.linalg.norm(axis)
    if sine < 1e-9:
        if cosine > 0:
            return np.eye(3)
        axis = _unit(np.cross(a, np.eye(3)[np.argmin(np.abs(a))]))
        return 2 * np.outer(axis, axis) - np.eye(3)
    cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + cross + cross @ cross * ((1 - cosine) / sine ** 2)


def solve_contact(shoulder, point, along, sign, upper, forearm, grip, hint, previous=None, max_bend=25, previous_turn=None):
    """Exact two-bone arm and closed-hand contact with one point on a straight bar.

    Free variables are the hand's roll around the bar and the elbow on its IK circle.
    Neither limb is stretched. Unreachable or excessively bent wrists fail explicitly.
    ``turn`` maps the fitted hand's rest coordinates onto its gripping pose.
    """
    shoulder, point, hint = (np.asarray(v, dtype=float) for v in (shoulder, point, hint))
    if not all(np.isfinite(v).all() for v in (shoulder, point, hint)) or min(upper, forearm) <= 0:
        raise ValueError('Invalid row arm geometry.')
    axis = -sign * _unit(along)
    base = _rotation_between(grip['kb'], axis)
    offset = np.asarray(grip['B']) - grip['wrist']
    hand_axis = _unit(np.asarray(grip['middle_mcp']) - grip['wrist'] if 'middle_mcp' in grip else grip['hb'])
    psi = np.linspace(0, 2 * math.pi, 72, endpoint=False)
    cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    best = None
    for phi in np.radians(np.arange(0, 360, 5)):
        turn = (np.eye(3) + math.sin(phi) * cross + (1 - math.cos(phi)) * cross @ cross) @ base
        wrist = point - turn @ offset
        delta = wrist - shoulder
        distance = np.linalg.norm(delta)
        if not abs(upper - forearm) + 1e-6 < distance < upper + forearm - 1e-6:
            continue
        direction = delta / distance
        reach = (upper ** 2 + distance ** 2 - forearm ** 2) / (2 * distance)
        radius = math.sqrt(max(0, upper ** 2 - reach ** 2))
        pole = np.eye(3)[np.argmin(np.abs(direction))]
        e1 = _unit(pole - direction * (pole @ direction))
        e2 = np.cross(direction, e1)
        elbows = shoulder + reach * direction + radius * (np.outer(np.cos(psi), e1) + np.outer(np.sin(psi), e2))
        fore = (wrist - elbows) / forearm
        bends = np.degrees(np.arccos(np.clip(fore @ (turn @ hand_axis), -1, 1)))
        cost = (bends / 5) ** 2 + np.sum(((elbows - hint) / (upper * .2)) ** 2, axis=1)
        if previous is not None:
            cost += .3 * np.sum(((elbows - previous) / (upper * .2)) ** 2, axis=1)
        if previous_turn is not None:
            # Quaternion-sign continuity alone cannot stop physical half-turn jumps.
            relative = previous_turn.T @ turn
            change = math.degrees(math.acos(float(np.clip((np.trace(relative) - 1) / 2, -1, 1))))
            cost += (change / 12) ** 2
        cost[bends > max_bend] = np.inf
        index = int(np.argmin(cost))
        if np.isfinite(cost[index]) and (best is None or cost[index] < best['cost']):
            best = dict(elbow=elbows[index], wrist=wrist, turn=turn,
                        bend_deg=float(bends[index]), cost=float(cost[index]))
    if best is None:
        raise ValueError('The row bar cannot be reached with an unstrained wrist.')
    return best


def solve_shared_bar(shoulders, centre, along, half_width, lengths, grips, hints, previous=None, allow_adaptation=True):
    """Find the nearest inward adaptation of one shared bar that both hands can grip.

    Adaptation is uniform for the two hands, with fixed grip width and level bar. The
    caller reports the displacement; a reference target beyond this range is refused.
    """
    previous = previous or {}
    centre, along = np.asarray(centre), _unit(along)
    origin = (np.asarray(shoulders['left']) + shoulders['right']) / 2
    for scale in (np.linspace(1, .65, 36) if allow_adaptation else [1.]):
        adapted = origin + scale * (centre - origin)
        solutions = {}
        try:
            for side, sign in (('left', -1), ('right', 1)):
                old = previous.get(side)
                elbow = old.get('elbow') if isinstance(old, dict) else old
                turn = old.get('turn') if isinstance(old, dict) else None
                solutions[side] = solve_contact(shoulders[side], adapted + sign * half_width * along,
                    along, sign, *lengths[side], grips[side], hints[side], elbow, previous_turn=turn)
        except ValueError:
            continue
        return adapted, solutions, float(np.linalg.norm(adapted - centre))
    raise ValueError('The reference row path is outside the character\'s reachable two-hand grip.')



def shaft_gaps(points, centre, along, half_length, radius):
    """Signed surface distances to a finite capsule shaft, including its rounded ends."""
    if half_length <= 0 or radius <= 0:
        raise ValueError('The shaft must have a positive length and radius.')
    relative = np.asarray(points, dtype=float) - np.asarray(centre, dtype=float)
    axis = _unit(along)
    axial = relative @ axis
    nearest = np.clip(axial, -half_length, half_length)
    return np.linalg.norm(relative - np.outer(nearest, axis), axis=1) - radius


def reference_arms(keyposes):
    """Direct shoulder-relative arm directions and elbow angles from measured references.

    Bone retarget rotations are unsuitable after the trunk is held fixed: carrying an arm
    rotation from another trunk pose can reverse its apparent pull direction. Read the
    original landmarks in the same canonical body frame instead. Elbow angle, rather than
    human arm-length proportions, sets the cartoon's reachable shoulder-wrist distance.
    """
    from keypose_clip import key_points
    convert = np.diag([-1., -1., 1.])  # canonical person-left/back/up -> house right/front/up
    phases = {}
    for entry in keyposes.get('phases', []):
        phase = entry.get('phase')
        if phase not in ('start', 'middle', 'end'):
            continue
        _, arms, found = key_points(entry)
        phases[phase] = {}
        seen = entry.get('camera_side')
        if found.get('swapped_sides') and seen in ('left', 'right'):
            seen = 'left' if seen == 'right' else 'right'
        for side in ('left', 'right'):
            # A bilateral row uses the visible arm on both sides. The occluded arm's
            # monocular depth can otherwise turn a 108-degree pull into 145 degrees.
            measured_side = seen if seen in ('left', 'right') else side
            shoulder, elbow, wrist = [convert @ arms[f'{measured_side}_{joint}'] for joint in ('shoulder','elbow','wrist')]
            if measured_side != side:
                mirror = np.array([-1., 1., 1.])
                shoulder, elbow, wrist = shoulder * mirror, elbow * mirror, wrist * mirror
            upper, fore = elbow - shoulder, wrist - elbow
            angle = math.degrees(math.acos(float(np.clip(_unit(-upper) @ _unit(fore), -1, 1))))
            phases[phase][side] = {'wrist_direction': _unit(wrist - shoulder),
                                  'elbow_direction': _unit(upper), 'forearm_direction': _unit(fore),
                                  'elbow_deg': angle}
    if not all(p in phases for p in ('start','end')):
        raise ValueError('Row contact needs measured Start and End arm positions.')
    return phases


def reference_arm_at(path, side, progress, upper, forearm):
    """Sample direct reference motion, preserving elbow bend for different limb lengths."""
    phases = [p for p in ('start','middle','end') if p in path]
    marks = [dict(start=0., middle=.5, end=1.)[p] for p in phases]
    p = float(np.clip(progress, 0, 1))
    angle = float(np.interp(p, marks, [path[phase][side]['elbow_deg'] for phase in phases]))
    directions = {}
    for name in ('wrist_direction','elbow_direction','forearm_direction'):
        directions[name] = _unit([np.interp(p, marks, [path[phase][side][name][axis] for phase in phases])
                                  for axis in range(3)])
    reach = math.sqrt(max(0., upper * upper + forearm * forearm -
                         2 * upper * forearm * math.cos(math.radians(angle))))
    return {**directions, 'wrist': directions['wrist_direction'] * reach,
            'elbow': directions['elbow_direction'] * upper, 'elbow_deg': angle}


def symmetric_body(points):
    """Keep a bilateral row level rather than reproducing monocular shoulder/hip twist."""
    result = {}
    for name, point in points.items():
        value = np.asarray(point, dtype=float).copy()
        if name.startswith(('left_', 'right_')):
            side, joint = name.split('_', 1)
            other = points.get(('right_' if side == 'left' else 'left_') + joint)
            if other is not None:
                value[0] = (point[0] - other[0]) / 2
                value[1:] = (np.asarray(point)[1:] + np.asarray(other)[1:]) / 2
        else:
            value[0] = 0
        result[name] = value
    return result
