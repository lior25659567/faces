"""Body-aligned exercise supports and measured contact diagnostics, in Blender.

Equipment dimensions follow the posed mascot. Placement is measured from the
evaluated skin, not a detached decorative prop. Failed contact stays a failed check.
"""
import json
import math
import bpy
import numpy as np
from mathutils import Matrix, Vector
import equipment
import mpfb_cage as cage
import skin_wrap

SEATED = {'Lat Pulldown (seated, to the front)', 'Dumbbell Shoulder Press (seated overhead press)'}
BENCH = 'Flat Barbell Bench Press'
HANG = 'Pull-Up'
PLANK = 'Front Plank (forearm plank)'
SLED = 'Leg Press (seated machine)'
CABLE = {'Lat Pulldown (seated, to the front)', 'Tricep Pushdown (cable, bar or rope attachment)'}


def _mask(skin, names):
    groups = {g.index for g in skin.vertex_groups if any(name in g.name for name in names)}
    result = np.array([sum(g.weight for g in vertex.groups if g.group in groups) > .5 for vertex in skin.data.vertices])
    if not result.any():
        raise ValueError('No measurable skin surface for ' + ', '.join(names))
    return result


def _pad(name, top, width, length, thickness, material):
    return equipment._pad(name, (top[0], top[1], top[2] - thickness / 2), (width, length, thickness), material)


def _joint_mid(rig, suffix):
    return sum((rig.pose.bones[cage.P + side + suffix].head for side in ('Left', 'Right')), Vector()) / 2


def _rotate_root(rig, angle):
    root = rig.pose.bones[cage.P + 'Hips']
    matrix = root.matrix.copy()
    changed = (Matrix.Rotation(angle, 3, 'X') @ matrix.to_3x3()).to_4x4()
    changed.translation = matrix.translation
    root.matrix = changed
    bpy.context.view_layer.update()


def _neutral_neck(rig):
    pose = rig.pose.bones
    chest = pose[cage.P + 'Spine2']
    turn = chest.matrix.to_3x3() @ chest.bone.matrix_local.to_3x3().inverted()
    for name in ('Neck', 'Head'):
        bone = pose[cage.P + name]
        matrix = (turn @ bone.bone.matrix_local.to_3x3()).to_4x4()
        matrix.translation = bone.head
        bone.matrix = matrix
        bpy.context.view_layer.update()


def _leg_to(rig, side, target, foot_rotation, pole=None):
    """Exact two-bone leg endpoint without stretching; return ankle residual."""
    up, low, foot = [rig.pose.bones[cage.P + side + name] for name in ('UpLeg', 'Leg', 'Foot')]
    hip = up.head.copy()
    if (target - hip).length > up.length + low.length + .002:
        raise ValueError(f'{side} support target is outside the leg reach.')
    knee = cage._two_bone_knee(hip, target, up.length, low.length,
                               pole if pole is not None else low.head - hip)
    cage._aim_pose(up, knee - up.head)
    cage._aim_pose(low, target - low.head)
    matrix = foot_rotation.to_4x4()
    matrix.translation = foot.head
    foot.matrix = matrix
    bpy.context.view_layer.update()
    return float((foot.head - target).length)


def _arm_to(rig, side, wrist, elbow_pole):
    up, low, hand = [rig.pose.bones[cage.P + side + name] for name in ('Arm', 'ForeArm', 'Hand')]
    shoulder = up.head.copy()
    if (wrist - shoulder).length > up.length + low.length + .002:
        raise ValueError(f'{side} forearm support target is outside the arm reach.')
    elbow = cage._two_bone_knee(shoulder, wrist, up.length, low.length, elbow_pole)
    cage._aim_pose(up, elbow - shoulder)
    cage._aim_pose(low, wrist - low.head)
    return float((hand.head - wrist).length)


def _reference_plank(rig, skin, first, last, source):
    """Fit supports while retaining the source's calibrated body/arm angles."""
    scene, pose = bpy.context.scene, rig.pose.bones
    scene.frame_set(first)
    base = {bone.name: bone.matrix_basis.copy() for bone in pose}
    feet = {side: pose[cage.P + side + 'Foot'].matrix.copy() for side in ('Left', 'Right')}
    head_turn = pose[cage.P + 'Head'].matrix.to_3x3().copy()
    fore_masks = {side: _mask(skin, [side + 'ForeArm']) for side in ('Left', 'Right')}
    toe_masks = {side: _mask(skin, [side + 'ToeBase']) for side in ('Left', 'Right')}
    turn = cage._leg_turns(rig)
    leg_offsets = {side: math.atan2(turn['thigh.' + side.lower()][2][1], turn['thigh.' + side.lower()][1][1])
                   for side in ('Left', 'Right')}
    beta = math.radians(source['torso_above_horizontal_deg'])
    raw_torso = beta - math.radians(cage.rest_lean(rig))
    alpha = math.pi / 2 - math.radians(source['upperarm_vs_torso_deg']) - beta
    gamma = math.pi / 2 - math.radians(source['elbow_included_deg']) - alpha
    def reset():
        for bone in pose:
            bone.matrix_basis = base[bone.name]
        bpy.context.view_layer.update()
    def trial(leg_delta):
        reset()
        torso = _joint_mid(rig, 'Arm') - _joint_mid(rig, 'UpLeg')
        pitch = raw_torso - math.atan2(torso.z, torso.y)
        _rotate_root(rig, pitch)
        hip_mid = _joint_mid(rig, 'UpLeg')
        hip_goals, poles = [], {}
        for side in ('Left', 'Right'):
            up, low = [pose[cage.P + side + name] for name in ('UpLeg', 'Leg')]
            lateral = feet[side].translation.x - up.head.x
            projection = math.sqrt(max(0., 1 - (lateral / (up.length + low.length)) ** 2))
            thigh = math.radians(source['thigh_angle_deg']) - leg_offsets[side] + leg_delta
            shin = math.radians(source['shin_angle_deg']) - leg_offsets[side] + leg_delta
            upper = Vector((lateral * up.length / (up.length + low.length),
                up.length * projection * math.cos(thigh), up.length * projection * math.sin(thigh)))
            lower = Vector((lateral * low.length / (up.length + low.length),
                low.length * projection * math.cos(shin), low.length * projection * math.sin(shin)))
            hip_goals.append(feet[side].translation - upper - lower - (up.head - hip_mid))
            poles[side] = upper
        cage._move_root(rig, sum(hip_goals, Vector()) / 2 - hip_mid)
        leg_error = arm_error = 0.
        for side in ('Left', 'Right'):
            leg_error = max(leg_error, _leg_to(rig, side, feet[side].translation,
                feet[side].to_3x3(), poles[side]))
            up, low = [pose[cage.P + side + name] for name in ('Arm', 'ForeArm')]
            elbow = up.head + Vector((0., -up.length * math.sin(alpha), -up.length * math.cos(alpha)))
            wrist = elbow + Vector((0., low.length * math.cos(gamma), low.length * math.sin(gamma)))
            arm_error = max(arm_error, _arm_to(rig, side, wrist, elbow - up.head))
        matrix = head_turn.to_4x4()
        matrix.translation = pose[cage.P + 'Head'].head
        pose[cage.P + 'Head'].matrix = matrix
        bpy.context.view_layer.update()
        points = skin_wrap.evaluated(skin)[0]
        gaps = {side: float(points[mask, 2].min()) for side, mask in fore_masks.items()}
        return sum(gaps.values()) / 2, {'body_pitch_adjustment_deg': math.degrees(pitch),
            'leg_direction_adjustment_deg': math.degrees(leg_delta),
            'ankle_residual_mm': leg_error * 1000, 'wrist_residual_mm': arm_error * 1000,
            'forearm_gaps_mm': {side.lower(): value * 1000 for side, value in gaps.items()},
            'toe_gaps_mm': {side.lower(): float(points[mask, 2].min()) * 1000 for side, mask in toe_masks.items()},
            'floor_penetration_mm': max(0., -float(points[:, 2].min())) * 1000,
            'raw_torso_above_horizontal_deg': math.degrees(raw_torso), 'reference': source}
    try:
        lo, hi = -math.radians(5), math.radians(5)
        left, _ = trial(lo)
        right, _ = trial(hi)
        if left * right > 0:
            raise ValueError('The photographed plank cannot meet the floor within five degrees of its measured leg direction.')
        for _ in range(18):
            middle = (lo + hi) / 2
            gap, outcome = trial(middle)
            if abs(gap) < .00005:
                break
            if gap * left > 0:
                lo, left = middle, gap
            else:
                hi = middle
        if (outcome['floor_penetration_mm'] > 3 or outcome['ankle_residual_mm'] > 2
                or outcome['wrist_residual_mm'] > 2
                or max(abs(value) for value in outcome['forearm_gaps_mm'].values()) > 3
                or max(abs(value) for value in outcome['toe_gaps_mm'].values()) > 3):
            raise ValueError('The source-aligned plank cannot satisfy forearm/toe contact and full-body floor clearance.')
        fixed = {bone.name: bone.matrix_basis.copy() for bone in pose}
        previous = {}
        for frame in range(first, last + 1):
            for bone in pose:
                bone.matrix_basis = fixed[bone.name]
            cage._key(rig, previous, frame)
            pose[cage.P + 'Hips'].keyframe_insert('location', frame=frame)
        scene.frame_set(first)
        return {'applied': True, **outcome}
    except ValueError as error:
        reset()
        return {'applied': False, 'reason': str(error), 'reference': source}


def correct_plank(rig, skin, first, last, reference=None):
    """Fit the observed hold without replacing its body line or arm angles."""
    import plank_reference
    source = plank_reference.read(reference)
    if source is None:
        return {'applied': False, 'reason': 'A clear visible-side plank reference is required to fit the body and arm angles.'}
    return _reference_plank(rig, skin, first, last, source)


def correct_bench(rig, skin, first, last):
    """Settle pelvis/back on one flat pad while feet keep their actual floor marks."""
    scene, pose = bpy.context.scene, rig.pose.bones
    scene.frame_set(first)
    pelvis, back = _mask(skin, ['Hips']), _mask(skin, ['Spine1', 'Spine2'])
    initial = skin_wrap.evaluated(skin)[0]
    plane = float(initial[pelvis, 2].min())
    base = {bone.name: bone.matrix_basis.copy() for bone in pose}
    foot = {side: pose[cage.P + side + 'Foot'].matrix.copy() for side in ('Left', 'Right')}
    # Keep the photographed floor marks while using flat, neutral plantar
    # surfaces. Fitted rest soles are horizontal; toe-only contact must not
    # masquerade as a planted foot. The original rig and mesh stay unchanged.
    foot_adjustments = {}
    for side in ('Left', 'Right'):
        bone = pose[cage.P + side + 'Foot']
        matrix = bone.bone.matrix_local.to_3x3().to_4x4()
        matrix.translation = foot[side].translation
        bone.matrix = matrix
        bpy.context.view_layer.update()
        toe = pose[cage.P + side + 'ToeBase']
        matrix = toe.bone.matrix_local.to_3x3().to_4x4()
        matrix.translation = toe.head
        toe.matrix = matrix
        bpy.context.view_layer.update()
        points = skin_wrap.evaluated(skin)[0]
        selected = _mask(skin, [side + 'Foot', side + 'ToeBase'])
        dz = -float(points[selected, 2].min())
        foot_adjustments[side.lower()] = dz * 1000
        foot[side] = bone.matrix.copy()
        foot[side].translation.z += dz
    for bone in pose:
        bone.matrix_basis = base[bone.name]
    bpy.context.view_layer.update()
    def settle(angle, basis, goals):
        for bone in pose:
            bone.matrix_basis = basis[bone.name]
        bpy.context.view_layer.update()
        _neutral_neck(rig)
        _rotate_root(rig, angle)
        shift = residual = 0.
        for _ in range(3):
            points = skin_wrap.evaluated(skin)[0]
            dz = plane - float(points[pelvis, 2].min())
            shift += dz
            cage._move_root(rig, Vector((0., 0., dz)))
            for side in ('Left', 'Right'):
                residual = max(residual, _leg_to(rig, side, goals[side].translation, goals[side].to_3x3()))
                toe = pose[cage.P + side + 'ToeBase']
                matrix = toe.bone.matrix_local.to_3x3().to_4x4()
                matrix.translation = toe.head
                toe.matrix = matrix
                bpy.context.view_layer.update()
        points = skin_wrap.evaluated(skin)[0]
        return float(points[back, 2].min()) - plane, residual, shift
    try:
        height = json.loads(rig['skin_fit'])['measured']['height']
        if max(abs(value) for value in foot_adjustments.values()) > height * 50.:
            raise ValueError('Bench floor contact needs more than 5% of character height in ankle correction.')
        lo, hi = -math.radians(12), math.radians(12)
        left, *_ = settle(lo, base, foot)
        right, *_ = settle(hi, base, foot)
        if left * right > 0:
            raise ValueError('No flat bench pose fits both pelvis and back within the 12-degree correction bound.')
        for _ in range(16):
            angle = (lo + hi) / 2
            value, *_ = settle(angle, base, foot)
            if value * left > 0:
                lo, left = angle, value
            else:
                hi = angle
        angle = (lo + hi) / 2
        # The bracket probes must not overwrite original animation before the
        # per-frame corrections capture it.
        for bone in pose:
            bone.matrix_basis = base[bone.name]
        bpy.context.view_layer.update()
        names = {cage.P + 'Hips'} | {cage.P + side + name for side in ('Left', 'Right') for name in ('UpLeg', 'Leg', 'Foot', 'ToeBase')}
        errors, shifts, head_turns, states = [], [], [], []
        for frame in range(first, last + 1):
            scene.frame_set(frame)
            basis = {bone.name: bone.matrix_basis.copy() for bone in pose}
            gap, error, shift = settle(angle, basis, foot)
            head_turns.append(_head_on_plane(rig, skin, plane))
            errors.append(error); shifts.append(abs(shift))
            states.append((frame, {bone.name: bone.matrix_basis.copy() for bone in pose}))
        previous = {}
        for frame, basis in states:
            for bone in pose:
                bone.matrix_basis = basis[bone.name]
            cage._key(rig, previous, frame, names | {cage.P + 'Neck', cage.P + 'Head'})
            pose[cage.P + 'Hips'].keyframe_insert('location', frame=frame)
        scene.frame_set(first)
        return {'applied': True, 'body_pitch_adjustment_deg': math.degrees(angle), 'pad_height': plane,
                'max_ankle_residual_mm': max(errors) * 1000, 'max_settling_shift_mm': max(shifts) * 1000,
                'max_head_pitch_adjustment_deg': max(abs(v) for v in head_turns),
                'floor_ankle_adjustments_mm': foot_adjustments,
                'foot_orientation': 'neutral fitted rest soles on fixed floor marks'}
    except ValueError as error:
        for bone in pose:
            bone.matrix_basis = base[bone.name]
        bpy.context.view_layer.update()
        return {'applied': False, 'reason': str(error)}


def _head_on_plane(rig, skin, plane, normal=None):
    """Settle the rear skull using a bounded head pitch from the neutral neck.

    Translating a large head by rotating its very short neck demands an extreme
    neck angle. A small head pitch supplies the real joint motion instead; face
    geometry and the fitted rest skeleton stay untouched.
    """
    pose = rig.pose.bones
    head = pose[cage.P + 'Head']
    basis = head.matrix.copy()
    mask = _mask(skin, ['Head'])
    normal = np.array([0., 0., 1.]) if normal is None else np.array(normal)
    def test(angle):
        matrix = (Matrix.Rotation(angle, 3, 'X') @ basis.to_3x3()).to_4x4()
        matrix.translation = basis.translation
        head.matrix = matrix
        bpy.context.view_layer.update()
        return float((skin_wrap.evaluated(skin)[0][mask] @ normal).min()) - plane
    grid = [math.radians(v) for v in (-20, -15, -10, -5, 0, 5, 10, 15, 20)]
    values = [(angle, test(angle)) for angle in grid]
    brackets = [(a, b) for a, b in zip(values, values[1:]) if a[1] * b[1] <= 0]
    if not brackets:
        best = min(values, key=lambda value: abs(value[1]))
        if abs(best[1]) <= .003:
            test(best[0])
            return math.degrees(best[0])
        test(0.)
        raise ValueError('The head support cannot be reached within a 20-degree head-pitch correction.')
    start, end = min(brackets, key=lambda pair: min(abs(pair[0][0]), abs(pair[1][0])))
    (lo, left), (hi, _) = start, end
    for _ in range(16):
        angle = (lo + hi) / 2
        value = test(angle)
        if value * left > 0:
            lo, left = angle, value
        else:
            hi = angle
    angle = (lo + hi) / 2
    test(angle)
    return math.degrees(angle)


def _head_support_range(rig, skin, normal):
    """Attainable rear-skull planes within the same 20-degree head bound."""
    head = rig.pose.bones[cage.P + 'Head']
    basis = head.matrix.copy()
    mask = _mask(skin, ['Head'])
    values = []
    for degrees in (-20, -15, -10, -5, 0, 5, 10, 15, 20):
        matrix = (Matrix.Rotation(math.radians(degrees), 3, 'X') @ basis.to_3x3()).to_4x4()
        matrix.translation = basis.translation
        head.matrix = matrix
        bpy.context.view_layer.update()
        values.append(float((skin_wrap.evaluated(skin)[0][mask] @ normal).min()))
    head.matrix = basis
    bpy.context.view_layer.update()
    return min(values), max(values)




def apply(rig, skin, first, last, exercise, reference=None, reference_phase_frames=None):
    scene = bpy.context.scene
    height = json.loads(rig['skin_fit'])['measured']['height']
    scene.frame_set(first)
    points = skin_wrap.evaluated(skin)[0]
    metal, rubber, _ = equipment._materials()
    props, requirements, oriented_requirements = [], [], []
    report = {'frames': last - first + 1, 'surfaces': {}}
    if exercise == PLANK:
        report['correction'] = correct_plank(rig, skin, first, last, reference=reference)
        points = skin_wrap.evaluated(skin)[0]
    supported = exercise in SEATED | {BENCH, HANG, PLANK, SLED}
    if not supported and exercise not in CABLE:
        return [], None
    pelvis = _mask(skin, ['Hips'])
    torso = _mask(skin, ['Spine', 'Hips'])
    # Preserve pelvis position on supports. Free-foot retargeting follows the
    # ankles by default, which otherwise moves a seat or a bench through the rider.
    if exercise in SEATED | {BENCH, SLED, HANG}:
        previous = {}
        pb = rig.pose.bones[cage.P + 'Hips']
        if exercise == HANG:
            def anchor():
                return sum((rig.pose.bones[cage.P + side + 'Hand'].head for side in ('Left', 'Right')), Vector()) / 2
        else:
            def anchor():
                return sum((rig.pose.bones[cage.P + side + 'UpLeg'].head for side in ('Left', 'Right')), Vector()) / 2
        origin = anchor().copy()
        if exercise == HANG:
            origin.z += height * .10
        for frame in range(first, last + 1):
            scene.frame_set(frame)
            cage._move_root(rig, origin - anchor())
            pb.keyframe_insert('location', frame=frame)
        scene.frame_set(first)
        points = skin_wrap.evaluated(skin)[0]
    if exercise == 'Dumbbell Shoulder Press (seated overhead press)':
        import seated_press_body
        report['body_stabilization'] = seated_press_body.correct(rig, skin, first, last)
        points = skin_wrap.evaluated(skin)[0]
    if exercise == BENCH:
        report['correction'] = correct_bench(rig, skin, first, last)
        points = skin_wrap.evaluated(skin)[0]
    if exercise == SLED:
        import time
        import sled_body
        started = time.perf_counter()
        report['correction'], report['body_stabilization'] = sled_body.fit(
            rig, skin, first, last, height, reference_phase_frames)
        report['body_stabilization']['fitting_seconds'] = time.perf_counter() - started
        report['sled_motion'] = sled_body.measure(rig, first, last, reference_phase_frames)
        print(f"supports: fixed-body rail fitted {last - first + 1} frames in "
              f"{report['body_stabilization']['fitting_seconds']:.2f}s", flush=True)
        points = skin_wrap.evaluated(skin)[0]
    if exercise in SEATED | {BENCH, SLED}:
        selected = points[pelvis]
        top = np.array([np.median(selected[:, 0]), np.median(selected[:, 1]), selected[:, 2].min()])
        width = max(float(np.ptp(selected[:, 0])) * .9, height * .18)
        length = height * (.65 if exercise == BENCH else .22)
        if exercise == SLED and report.get('body_stabilization', {}).get('applied'):
            from sled_motion import stationary_seat
            samples = []
            for frame in range(first, last + 1):
                scene.frame_set(frame)
                samples.append(skin_wrap.evaluated(skin)[0][pelvis])
            body_axis = report['correction']['backrest']['body_axis']
            seat = stationary_seat(samples, height, width, body_axis)
            report['pelvis_seat_fit'] = seat
            top, length = np.array(seat['centre_m']), seat['length_m']
            scene.frame_set(first)
        if exercise == BENCH:
            # Bench long axis follows the torso's actual horizontal axis.
            upper = points[torso | _mask(skin, ['Head'])]
            top[1] = (upper[:, 1].min() + upper[:, 1].max()) / 2
            length = max(length, float(np.ptp(upper[:, 1])) * 1.1)
        props.append(_pad('body_support_pad', top, width, length, height * .035, rubber))
        for sign in (-1, 1):
            base = (top[0], top[1] + sign * length * .3, height * .015)
            end = (base[0], base[1], max(height * .02, top[2] - height * .035))
            props.append(equipment.capsule('support_leg', base, end, height * .018, metal))
            props.append(_pad('support_base', (base[0], base[1], height * .02), width * 1.2, height * .09, height * .02, metal))
        requirements.append(('pelvis', pelvis, float(top[2]), top[:2].copy(), np.array([width, length]) / 2))
        if exercise == BENCH:
            requirements.append(('back', _mask(skin, ['Spine1', 'Spine2']), float(top[2]), top[:2].copy(), np.array([width, length]) / 2))
            requirements.append(('head', _mask(skin, ['Head']), float(top[2]), top[:2].copy(), np.array([width, length]) / 2))
        elif exercise in SEATED:
            upper = points[_mask(skin, ['Spine1', 'Spine2'])]
            low, high = upper.min(0), upper.max(0)
            centre = ((low + high) / 2).tolist()
            centre[1] = float(low[1] - height * .015)
            props.append(equipment._pad('seat_back', centre,
                (width, height * .03, max(float(high[2] - low[2]), height * .2)), rubber))
            report['backrest_plane_y'] = float(low[1])
            if exercise == 'Dumbbell Shoulder Press (seated overhead press)':
                # Measure the finite front face of the actual pad in every
                # frame; the seat alone cannot certify a supported press.
                axes = np.column_stack([np.array([1., 0., 0.]), np.array([0., 0., -1.]), np.array([0., 1., 0.])])
                surface = np.array([centre[0], float(low[1]), centre[2]])
                oriented_requirements.append(('back', _mask(skin, ['Spine1', 'Spine2']), surface, axes,
                    np.array([width, max(float(high[2] - low[2]), height * .2)]) / 2))
    if exercise == SLED:
        back_mask, head_mask = _mask(skin, ['Spine1', 'Spine2']), _mask(skin, ['Head'])
        backrest = report.get('correction', {}).get('backrest')
        if backrest:
            normal, body_axis = np.array(backrest['normal']), np.array(backrest['body_axis'])
            plane = backrest['plane']
        else:
            body_axis = np.array(_joint_mid(rig, 'Arm') - _joint_mid(rig, 'UpLeg'))
            body_axis[0] = 0.
            body_axis /= np.linalg.norm(body_axis)
            normal = np.array([0., body_axis[2], -body_axis[1]])
            if normal[2] < 0:
                normal = -normal
            plane = float((points[back_mask] @ normal).min())
        lateral, along = np.array([1., 0., 0.]), -body_axis
        if np.dot(np.cross(lateral, along), normal) < 0:
            along = -along
        axes = np.column_stack([lateral, along, normal])
        upper = points[back_mask | head_mask] @ axes
        low, high = upper.min(0), upper.max(0)
        width = max(float(high[0] - low[0]) * 1.1, height * .26)
        length = float(high[1] - low[1]) * 1.1
        top = axes @ np.array([(low[0] + high[0]) / 2, (low[1] + high[1]) / 2, plane])
        thickness = height * .035
        pad = equipment._pad('leg_press_backrest', (0., 0., 0.), (width, length, thickness), rubber)
        matrix = Matrix(axes.tolist()).to_4x4()
        matrix.translation = Vector(top - normal * thickness / 2)
        pad.matrix_world = matrix
        props.append(pad)
        for sign in (-1, 1):
            rear = top - normal * thickness + lateral * sign * width * .32
            props.append(equipment.capsule('leg_press_backrest_stand',
                (rear[0], rear[1], height * .015), rear, height * .018, metal))
        oriented_requirements.append(('back', back_mask, top, axes, np.array([width, length]) / 2))
        head_plane = backrest.get('head_plane', plane) if backrest else plane
        head_local = points[head_mask] @ axes
        low_head, high_head = head_local.min(0), head_local.max(0)
        head_width = max(float(high_head[0] - low_head[0]) * 1.1, width)
        head_length = float(high_head[1] - low_head[1]) * 1.1
        head_top = axes @ np.array([(low_head[0] + high_head[0]) / 2, (low_head[1] + high_head[1]) / 2, head_plane])
        head_thickness = max(height * .02, head_plane - plane + height * .02)
        headrest = equipment._pad('leg_press_headrest', (0., 0., 0.), (head_width, head_length, head_thickness), rubber)
        matrix = Matrix(axes.tolist()).to_4x4()
        matrix.translation = Vector(head_top - normal * head_thickness / 2)
        headrest.matrix_world = matrix
        props.append(headrest)
        oriented_requirements.append(('head', head_mask, head_top, axes, np.array([head_width, head_length]) / 2))
        report['backrest'] = {'normal': normal.tolist(), 'plane': float(plane),
            'headrest_padding_mm': float(head_plane - plane) * 1000,
            'source': 'fixed pad aligned to the measured torso; not a calibrated machine angle'}
    # Pull-up posts are built against the final fixed bar in exercise_contacts.
    if exercise == PLANK:
        for side in ('Left', 'Right'):
            requirements.append((side.lower() + '_forearm', _mask(skin, [side + 'ForeArm']), 0., None, None))
            requirements.append((side.lower() + '_toe', _mask(skin, [side + 'ToeBase']), 0., None, None))
    if exercise in CABLE:
        shoulders = np.array([rig.matrix_world @ rig.pose.bones[cage.P + side + 'Arm'].head for side in ('Left', 'Right')])
        pulley = shoulders.mean(0) + np.array([0., .45 * height, .45 * height])
        props.append(equipment.capsule('cable_tower_post', (pulley[0], pulley[1], .02 * height), pulley, .02 * height, metal))
        report['pulley_anchor'] = pulley.tolist()
    rows = {name: [] for name, *_ in requirements}
    rows.update({name: [] for name, *_ in oriented_requirements})
    for frame in range(first, last + 1):
        scene.frame_set(frame)
        points = skin_wrap.evaluated(skin)[0]
        for name, mask, plane, centre, half in requirements:
            surface = points[mask]
            if centre is not None:
                surface = surface[np.all(abs(surface[:, :2] - centre) <= half, axis=1)]
            if not len(surface):
                rows[name].append(None)
            else:
                # Distance to the plane only counts within the finite support pad.
                rows[name].append((float(abs(surface[:, 2] - plane).min()), max(0., float(plane - surface[:, 2].min()))))
        for name, mask, centre, axes, half in oriented_requirements:
            local = (points[mask] - centre) @ axes
            local = local[np.all(abs(local[:, :2]) <= half, axis=1)]
            rows[name].append(None if not len(local) else
                (float(abs(local[:, 2]).min()), max(0., -float(local[:, 2].min()))))
    for name, values in rows.items():
        measured = [v for v in values if v is not None]
        report['surfaces'][name] = {'frames': len(values), 'measured_frames': len(measured),
            'max_gap_mm': max((v[0] * 1000 for v in measured), default=None),
            'max_penetration_mm': max((v[1] * 1000 for v in measured), default=None)}
    if exercise in (BENCH, 'Dumbbell Shoulder Press (seated overhead press)'):
        from bench_feet import floor_report
        rest = np.array([vertex.co for vertex in skin.data.vertices])
        feet = {side.lower(): _mask(skin, [side + 'Foot', side + 'ToeBase']) for side in ('Left', 'Right')}
        def measured_frames():
            for frame in range(first, last + 1):
                scene.frame_set(frame)
                yield frame, skin_wrap.evaluated(skin)[0]
        measured = floor_report(rest, feet, measured_frames())
        report['surfaces'].update(measured['surfaces'])
        report['planted_feet'] = measured['planted_feet']
    if exercise == SLED:
        sled, measured = moving_sled(rig, skin, first, last, height, metal, rubber)
        props.extend(sled)
        report['surfaces'].update(measured['surfaces'])
        report['rail_path'] = measured['rail_path']
    scene.frame_set(first)
    return props, report


def moving_sled(rig, skin, first, last, height, metal, rubber):
    """A finite footplate on straight rails; report incompatible curved reference paths.

    Rails follow the measured travel axis. An inferred oblique angle is explicitly
    not claimed to be a calibrated 45-degree machine.
    """
    scene = bpy.context.scene
    feet = {side: _mask(skin, [title + 'Foot', title + 'ToeBase']) for side, title in (('left', 'Left'), ('right', 'Right'))}
    rest_points = np.array([vertex.co for vertex in skin.data.vertices])
    patches = {}
    for side, mask in feet.items():
        sole = mask & (rest_points[:, 2] <= float(rest_points[mask, 2].min()) + .002)
        low, high = float(rest_points[sole, 1].min()), float(rest_points[sole, 1].max())
        patches[side + '_heel'] = sole & (rest_points[:, 1] <= low + (high - low) * .3)
        patches[side + '_toe'] = sole & (rest_points[:, 1] >= high - (high - low) * .3)
    positions = []
    for frame in range(first, last + 1):
        scene.frame_set(frame)
        positions.append(np.mean([np.array(rig.matrix_world @ rig.pose.bones[cage.P + side + 'Foot'].head) for side in ('Left', 'Right')], axis=0))
    positions = np.array(positions)
    centre = positions.mean(0)
    _, singular, directions = np.linalg.svd(positions - centre)
    if singular[0] < height * .01:
        raise ValueError('The leg-press references provide no readable sled travel.')
    axis = directions[0]
    if axis[2] < 0:
        axis = -axis
    distance = (positions - centre) @ axis
    line = centre + distance[:, None] * axis
    residual = np.linalg.norm(positions - line, axis=1)
    scene.frame_set(first)
    points = skin_wrap.evaluated(skin)[0]
    offset = max(float(((points[mask] - positions[0]) @ axis).max()) for mask in feet.values())
    line += axis * offset
    lateral = np.array([1., 0., 0.])
    lateral -= axis * (lateral @ axis)
    lateral /= np.linalg.norm(lateral)
    tangent = np.cross(axis, lateral)
    width, length, thickness = height * .5, height * .35, height * .025
    plate = equipment._pad('leg_press_footplate', (0, 0, 0), (width, length, thickness), rubber)
    rotation = Matrix((Vector(lateral), Vector(tangent), Vector(axis))).transposed().to_4x4()
    contact_masks = {**feet, **patches}
    values = {side: [] for side in contact_masks}
    for frame, target in zip(range(first, last + 1), line):
        scene.frame_set(frame)
        # The skin touches the near face, whose normal points toward the rider.
        matrix = rotation.copy()
        matrix.translation = Vector(target + axis * thickness / 2)
        plate.matrix_world = matrix
        plate.rotation_mode = 'QUATERNION'
        plate.keyframe_insert('location', frame=frame)
        plate.keyframe_insert('rotation_quaternion', frame=frame)
        points = skin_wrap.evaluated(skin)[0]
        for side, mask in contact_masks.items():
            local = points[mask] - target
            within = (abs(local @ lateral) <= width / 2) & (abs(local @ tangent) <= length / 2)
            gaps = local[within] @ axis
            values[side].append(None if not len(gaps) else (float(abs(gaps).min()), max(0., float(gaps.max()))))
    rails = []
    start = line[int(np.argmin(distance))] - axis * height * .1
    end = line[int(np.argmax(distance))] + axis * height * .1
    for sign in (-1, 1):
        offset = lateral * sign * width * .6
        rails.append(equipment.capsule('leg_press_rail', start + offset, end + offset, height * .018, metal))
        for point in (start + offset, end + offset):
            rails.append(equipment.capsule('leg_press_rail_stand', (point[0], point[1], 0.), point, height * .015, metal))
    surfaces = {}
    for side, rows in values.items():
        measured = [row for row in rows if row is not None]
        surfaces['footplate_' + side] = {'frames': len(rows), 'measured_frames': len(measured),
            'max_gap_mm': max((r[0] * 1000 for r in measured), default=None),
            'max_penetration_mm': max((r[1] * 1000 for r in measured), default=None)}
    return [plate, *rails], {'surfaces': surfaces, 'rail_path': {'frames': last - first + 1, 'max_deviation_mm': float(residual.max() * 1000),
        'incline_deg': float(np.degrees(np.arcsin(axis[2]))), 'source': 'reference-inferred travel axis; not calibrated'}}
