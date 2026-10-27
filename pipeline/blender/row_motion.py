"""Keep the body hinge stable in a reference-driven bent-over row."""
def stabilize_body(keys, legs=None):
    """Hold the starting hinge while allowing the row's arms to move.

    With measured leg directions, hold the thigh direction and knee bend at their
    observed means instead of adding a knee bounce. The feet keep their starting
    turns. This constant stance stays within the observed motion of both legs.
    Matrices are absolute turns, so holding the trunk alone would leave the head or
    thighs drifting. Preserve every non-arm turn, including neck and feet.
    """
    if not keys:
        return []
    def arm(name):
        return name.startswith(('LeftArm', 'RightArm', 'LeftForeArm', 'RightForeArm',
                                'LeftHand', 'RightHand'))
    held = {name: value.copy() for name, value in keys[0].items() if not arm(name)}
    if legs is not None:
        down = _unit(legs)
        for side in ('Left', 'Right'):
            thigh, shin = side + 'UpLeg', side + 'Leg'
            if not all(thigh in key and shin in key for key in keys):
                continue
            angles = [math.acos(float(np.clip(_unit(-(key[thigh] @ down)) @ _unit(key[shin] @ down), -1, 1)))
                      for key in keys]
            target = float(np.mean(angles))
            observed = [_unit(key[thigh] @ down) for key in keys]
            average = np.mean(observed, axis=0)
            if np.linalg.norm(average) < 1e-9:
                # Opposing thigh directions do not establish a standing row.
                raise ValueError('The row references do not establish a consistent thigh direction.')
            thigh_turn = _rotation_between(observed[0], _unit(average))
            held[thigh] = thigh_turn @ keys[0][thigh]
            up = _unit(-(held[thigh] @ down))
            original = _unit(thigh_turn @ keys[0][shin] @ down)
            pole = original - up * (original @ up)
            if np.linalg.norm(pole) < 1e-9:
                # A straight starting knee has no stable bend plane to average.
                # Keep its measured stance rather than invent a lateral knee.
                held[thigh] = keys[0][thigh].copy()
                continue
            desired = up * math.cos(target) + _unit(pole) * math.sin(target)
            held[shin] = _rotation_between(original, desired) @ thigh_turn @ keys[0][shin]
    return [{name: value if arm(name) else held[name].copy() for name, value in key.items()}
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


def solve_contact(shoulder, point, along, sign, upper, forearm, grip, hint, previous=None, max_bend=25, previous_turn=None,
                  preference=None):
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
    lateral_axis = None
    if preference and preference.get('elbow_lateral') is not None:
        lateral_axis = _unit(preference['lateral_axis'])
        lateral_goal = float(preference['elbow_lateral'])
        lateral_scale = float(preference.get('lateral_scale', upper * .03))
        if not np.isfinite([lateral_goal, lateral_scale]).all() or lateral_scale <= 0:
            raise ValueError('The elbow lateral goal and scale must be finite, with a positive scale.')
    minimum_elbow = None if not preference else preference.get('min_elbow_deg')
    if minimum_elbow is not None:
        if isinstance(minimum_elbow, bool) or not np.isfinite(minimum_elbow) or not 0 <= minimum_elbow <= 180:
            raise ValueError('The minimum elbow angle must be between zero and 180 degrees.')
    maximum_elbow_z = None if not preference else preference.get('max_elbow_z')
    if maximum_elbow_z is not None and (isinstance(maximum_elbow_z, bool) or not np.isfinite(maximum_elbow_z)):
        raise ValueError('The maximum elbow height must be finite.')
    minimum_elbow_y = None if not preference else preference.get('min_elbow_y')
    if minimum_elbow_y is not None and (isinstance(minimum_elbow_y, bool) or not np.isfinite(minimum_elbow_y)):
        raise ValueError('The minimum elbow forward position must be finite.')
    below_wrist = bool((preference or {}).get('elbow_below_wrist', False))
    clearance = float((preference or {}).get('elbow_clearance_m', 0.))
    vertical_fraction = (preference or {}).get('forearm_vertical_fraction')
    minimum_vertical_fraction = (preference or {}).get('minimum_forearm_vertical_fraction')
    forward_axis = (preference or {}).get('elbow_forward_axis')
    minimum_forward = (preference or {}).get('minimum_elbow_forward')
    if minimum_forward is not None:
        forward_axis = _unit(forward_axis)
        if isinstance(minimum_forward, bool) or not np.isfinite(minimum_forward):
            raise ValueError('The minimum elbow-forward projection must be finite.')
    reference_hand_turn = (preference or {}).get('reference_hand_turn')
    maximum_hand_turn = (preference or {}).get('max_hand_turn_deg')
    if reference_hand_turn is not None:
        reference_hand_turn = np.asarray(reference_hand_turn, dtype=float)
        if (reference_hand_turn.shape != (3, 3) or not np.isfinite(reference_hand_turn).all()
                or not np.allclose(reference_hand_turn.T @ reference_hand_turn, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(reference_hand_turn), 1., atol=1e-6)
                or isinstance(maximum_hand_turn, bool) or not isinstance(maximum_hand_turn, (int, float))
                or not np.isfinite(maximum_hand_turn) or not 0. <= maximum_hand_turn <= 180.):
            raise ValueError('A bounded hand turn needs a finite reference rotation and degree limit.')
    if not np.isfinite(clearance) or clearance < 0:
        raise ValueError('The elbow height clearance must be finite and nonnegative.')
    if vertical_fraction is not None and (isinstance(vertical_fraction, bool) or not np.isfinite(vertical_fraction)
                                         or not 0 <= vertical_fraction <= 1):
        raise ValueError('The forearm vertical fraction must lie between zero and one.')
    if minimum_vertical_fraction is not None and (isinstance(minimum_vertical_fraction, bool)
            or not np.isfinite(minimum_vertical_fraction) or not 0 <= minimum_vertical_fraction <= 1):
        raise ValueError('The minimum forearm vertical fraction must lie between zero and one.')
    def candidate(degrees):
        phi = math.radians(degrees)
        turn = (np.eye(3) + math.sin(phi) * cross + (1 - math.cos(phi)) * cross @ cross) @ base
        if reference_hand_turn is not None:
            angle = math.degrees(math.acos(float(np.clip((np.trace(reference_hand_turn.T @ turn) - 1.) / 2., -1., 1.))))
            if angle > maximum_hand_turn:
                return None
        wrist = point - turn @ offset
        delta = wrist - shoulder
        distance = np.linalg.norm(delta)
        if not abs(upper - forearm) + 1e-6 < distance < upper + forearm - 1e-6:
            return None
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
        elbow_deg = math.degrees(math.acos(float(np.clip((upper * upper + forearm * forearm - distance * distance)
                                                         / (2 * upper * forearm), -1, 1))))
        if minimum_elbow is not None and elbow_deg < minimum_elbow:
            return None
        upperarm_deg = None
        if preference:
            if preference.get('elbow_deg') is not None:
                cost += ((elbow_deg - float(preference['elbow_deg'])) / 1.5) ** 2
            if preference.get('upperarm_deg') is not None and preference.get('torso_down') is not None:
                down = _unit(preference['torso_down'])
                upperarm_deg = np.degrees(np.arccos(np.clip(((elbows - shoulder) / upper) @ down, -1, 1)))
                cost += ((upperarm_deg - float(preference['upperarm_deg'])) / 3.) ** 2
            if lateral_axis is not None:
                cost += ((elbows @ lateral_axis - lateral_goal) / lateral_scale) ** 2
            if vertical_fraction is not None:
                cost += ((elbows[:, 2] - (wrist[2] - vertical_fraction * forearm)) / (forearm * .1)) ** 2
        if previous is not None:
            cost += .3 * np.sum(((elbows - previous) / (upper * .2)) ** 2, axis=1)
        if previous_turn is not None:
            # Quaternion-sign continuity alone cannot stop physical half-turn jumps.
            relative = previous_turn.T @ turn
            change = math.degrees(math.acos(float(np.clip((np.trace(relative) - 1) / 2, -1, 1))))
            # The finer reference grid exposes neighboring roll solutions. Give
            # continuity enough weight to follow those gradually rather than
            # hopping between individually good elbow fits.
            cost += (change / (4. if preference else 12.)) ** 2
        cost[bends > max_bend] = np.inf
        if maximum_elbow_z is not None:
            cost[elbows[:, 2] > maximum_elbow_z] = np.inf
        if minimum_elbow_y is not None:
            cost[elbows[:, 1] < minimum_elbow_y] = np.inf
        if below_wrist:
            cost[elbows[:, 2] > wrist[2] - clearance] = np.inf
        if minimum_vertical_fraction is not None:
            cost[fore[:, 2] < minimum_vertical_fraction] = np.inf
        if minimum_forward is not None:
            cost[elbows @ forward_axis < minimum_forward] = np.inf
        index = int(np.argmin(cost))
        if np.isfinite(cost[index]):
            return dict(elbow=elbows[index], wrist=wrist, turn=turn,
                        bend_deg=float(bends[index]), elbow_deg=elbow_deg,
                        upperarm_deg=None if upperarm_deg is None else float(upperarm_deg[index]), cost=float(cost[index]))
        return None

    coarse = [(angle, candidate(angle)) for angle in range(0, 360, 5)]
    viable = [(angle, result) for angle, result in coarse if result is not None]
    best = min((result for _, result in viable), key=lambda result: result['cost'], default=None)
    if preference:
        # Nearly straight arms can be reachable over a narrow roll interval.
        # Refine the best coarse regions to half-degree accuracy, rather than
        # making every frame pay for an exhaustive fine grid. If the coarse
        # grid missed all feasible regions, the full fine scan is necessary.
        if viable:
            centres = [angle for angle, _ in sorted(viable, key=lambda pair: pair[1]['cost'])[:3]]
            fine = sorted({float((angle + step) % 360) for angle in centres
                           for step in np.arange(-4.5, 5., .5)} - set(range(0, 360, 5)))
        else:
            fine = [angle for angle in np.arange(.5, 360., .5) if angle % 5]
        for angle in fine:
            result = candidate(angle)
            if result is not None and (best is None or result['cost'] < best['cost']):
                best = result
    if best is None:
        raise ValueError('The row bar cannot be reached with an unstrained wrist.')
    return best


def solve_shared_bar(shoulders, centre, along, half_width, lengths, grips, hints, previous=None, allow_adaptation=True,
                     preferences=None):
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
                    along, sign, *lengths[side], grips[side], hints[side], elbow, previous_turn=turn,
                    preference=(preferences or {}).get(side))
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
            hip = convert @ arms[f'{measured_side}_hip']
            if measured_side != side:
                hip = hip * mirror
            upperarm = math.degrees(math.acos(float(np.clip(_unit(upper) @ _unit(hip - shoulder), -1, 1))))
            phases[phase][side] = {'wrist_direction': _unit(wrist - shoulder),
                                  'elbow_direction': _unit(upper), 'forearm_direction': _unit(fore),
                                  'elbow_deg': angle, 'upperarm_deg': upperarm}
    if not all(p in phases for p in ('start','end')):
        raise ValueError('Row contact needs measured Start and End arm positions.')
    return phases


def reference_arm_at(path, side, progress, upper, forearm):
    """Sample direct reference motion, preserving elbow bend for different limb lengths."""
    phases = [p for p in ('start','middle','end') if p in path]
    marks = [dict(start=0., middle=.5, end=1.)[p] for p in phases]
    p = float(np.clip(progress, 0, 1))
    angle = float(np.interp(p, marks, [path[phase][side]['elbow_deg'] for phase in phases]))
    upperarm = float(np.interp(p, marks, [path[phase][side]['upperarm_deg'] for phase in phases]))
    directions = {}
    for name in ('wrist_direction','elbow_direction','forearm_direction'):
        directions[name] = _unit([np.interp(p, marks, [path[phase][side][name][axis] for phase in phases])
                                  for axis in range(3)])
    reach = math.sqrt(max(0., upper * upper + forearm * forearm -
                         2 * upper * forearm * math.cos(math.radians(angle))))
    return {**directions, 'wrist': directions['wrist_direction'] * reach,
            'elbow': directions['elbow_direction'] * upper, 'elbow_deg': angle, 'upperarm_deg': upperarm}


def symmetric_body(points, visible_leg=None):
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
    if visible_leg in ('left', 'right'):
        # A bilateral row stands symmetrically, but averaging the occluded leg's
        # inferred depth changes the very knee/hip angles judged on the near leg.
        # Preserve that leg's measured vectors and mirror them onto the far side.
        origin = np.asarray(points[visible_leg + '_hip'], dtype=float)
        target = result[visible_leg + '_hip']
        for joint in ('knee', 'ankle', 'heel', 'foot_index'):
            name = visible_leg + '_' + joint
            if name not in points:
                continue
            near = np.asarray(points[name], dtype=float) - origin + target
            result[name] = near.copy()
            far = ('right' if visible_leg == 'left' else 'left') + '_' + joint
            if far in points:
                result[far] = near * np.array([-1., 1., 1.])
    return result
