"""Held equipment fitted to the mascot's actual closed-hand anchors, in Blender.

Bars constrain both arms to one rigid shaft. Dumbbells follow independent hands.
Contact reports measure rendered skin against finite handles at every frame.
"""
import json
import bpy
import numpy as np
from mathutils import Matrix, Vector

import equipment
import hand_rig
import mpfb_cage as cage
import skin_wrap
from row_motion import shaft_gaps, solve_shared_bar
from motion_profiles import profile_for

SIDES = (('left', 'Left', -1), ('right', 'Right', 1))
DUMBBELLS = {'Dumbbell Bicep Curl (standing, both arms)', 'Dumbbell Lateral Raise',
             'Dumbbell Shoulder Press (seated overhead press)', 'Walking Lunge (dumbbells optional)'}
BARS = {'Conventional Barbell Deadlift', 'Flat Barbell Bench Press',
        'Standing Barbell Overhead Press (Shoulder Press)',
        'Lat Pulldown (seated, to the front)', 'Tricep Pushdown (cable, bar or rope attachment)', 'Pull-Up'}
PUSHDOWN = 'Tricep Pushdown (cable, bar or rope attachment)'


def _turn(hand):
    return np.array(hand.matrix.to_3x3() @ hand.bone.matrix_local.to_3x3().inverted())


def _anchor(hand, grip):
    return np.array(hand.head) + _turn(hand) @ (np.array(grip['B']) - grip['wrist'])


def _bar_anchors(pose, grips, along):
    """Requested contact points after the closed hands are aligned to the shaft.

    The reference clip only sets limb directions. Its inherited hand roll is not
    a bar grip, so projecting its unaligned palm offset creates a different target
    from the one the arm solve must actually reach.
    """
    anchors = {}
    for side, title, sign in SIDES:
        hand = pose[cage.P + title + 'Hand']
        forearm = np.array(hand.head - pose[cage.P + title + 'ForeArm'].head)
        turn = hand_rig.hand_turn(grips[side], sign, along, forearm)
        anchors[side] = np.array(hand.head) + turn @ (np.array(grips[side]['B']) - grips[side]['wrist'])
    return anchors


def _matrix(along, centre):
    x = Vector(along).normalized()
    hint = Vector((0, 0, 1)) if abs(x.z) < .9 else Vector((0, 1, 0))
    y = hint.cross(x).normalized()
    result = Matrix((x, y, x.cross(y))).transposed().to_4x4()
    result.translation = Vector(centre)
    return result


def _hand_masks(skin):
    result = {}
    for side, title, _ in SIDES:
        groups = {g.index for g in skin.vertex_groups if title + 'Hand' in g.name}
        result[side] = np.array([sum(g.weight for g in v.groups if g.group in groups) > .5 for v in skin.data.vertices])
        if not result[side].any():
            raise ValueError(f'Cannot measure the {side} hand surface.')
    return result


def _elbows(pose):
    from pullup_range import included
    return {side: included(np.array(pose[cage.P + title + 'Arm'].head),
                           np.array(pose[cage.P + title + 'ForeArm'].head),
                           np.array(pose[cage.P + title + 'Hand'].head))
            for side, title, _ in SIDES}


def _upperarms(rig):
    from pullup_range import included
    pose, turn = rig.pose.bones, np.array(cage._torso_turn(rig))
    return {side: included(turn @ np.array(pose[cage.P + title + 'UpLeg'].head - pose[cage.P + title + 'Arm'].head),
                           np.zeros(3), np.array(pose[cage.P + title + 'ForeArm'].head - pose[cage.P + title + 'Arm'].head))
            for side, title, _ in SIDES}


def pullup_range(rig, skin, bar, first, last):
    """Measure the final keyed skin's lower face against the top of a fixed bar."""
    from pullup_range import chin_region, summarize, motion_summary
    head = {g.index for g in skin.vertex_groups if g.name == cage.P + 'Head'}
    mask = np.array([sum(g.weight for g in v.groups if g.group in head) > .99 for v in skin.data.vertices])
    transform = rig.matrix_world.inverted() @ skin.matrix_world
    rest = np.array([transform @ v.co for v in skin.data.vertices])
    region = chin_region(rest, mask)
    height = json.loads(rig['skin_fit'])['measured']['height']
    radius = equipment.BARBELL_BAR_DIAMETER / 2 * height / 1.75
    samples = []
    for frame in range(first, last + 1):
        bpy.context.scene.frame_set(frame)
        points = skin_wrap.evaluated(skin)[0]
        chin = float(points[region, 2].min())
        top = float(bar.matrix_world.translation.z) + radius
        pose, world = rig.pose.bones, rig.matrix_world
        elbows = {side: float((world @ pose[cage.P + title + 'ForeArm'].head).z) for side, title, _ in SIDES}
        wrists = {side: float((world @ pose[cage.P + title + 'Hand'].head).z) for side, title, _ in SIDES}
        shoulders = {side: float((world @ pose[cage.P + title + 'Arm'].head).z) for side, title, _ in SIDES}
        samples.append({'frame': frame, 'chin_clearance_mm': (chin - top) * 1000,
                        'elbows_deg': _elbows(pose), 'hip_height_m': float((world @ pose[cage.P + 'Hips'].head).z),
                        'elbow_offsets_m': {side: list(world.to_3x3() @ (pose[cage.P + title + 'ForeArm'].head - pose[cage.P + title + 'Arm'].head))
                                            for side, title, _ in SIDES},
                        'elbow_bar_clearance_mm': {side: (float(bar.matrix_world.translation.z) - elbow) * 1000 for side, elbow in elbows.items()},
                        'elbow_wrist_clearance_mm': {side: (wrists[side] - elbow) * 1000 for side, elbow in elbows.items()},
                        'elbow_shoulder_clearance_mm': {side: (shoulders[side] - elbow) * 1000 for side, elbow in elbows.items()}})
    result = summarize(samples)
    reference = json.loads(rig.get('reference_motion', '{}')) if 'reference_motion' in rig else None
    # The caller adds the recorded source phases once retargeting has finished;
    # saved-scene replays read those same phases from the rig custom property.
    if reference and reference.get('phase_frames'):
        result['motion_continuity'] = motion_summary(samples,
            bpy.context.scene.render.fps / bpy.context.scene.render.fps_base, reference['phase_frames'])
    result['chin_proxy'] = {'method': 'lowest central 20% / front 40% rigid head rest-surface band; geometric lower-face proxy, not anatomical calibration',
                           'rest_indices': region.tolist(), 'rest_points_m': rest[region].tolist()}
    bpy.context.scene.frame_set(first)
    return result


def apply(rig, skin, first, last, exercise, height=None, pulley_anchor=None, grip_ratio=None, source_grip_ratio=None,
          source_elbow_widths=None, reference_phase_frames=None, grip_ratio_tolerance=0., bar_path=None,
          source_arm_targets=None, source_angle_tolerance_deg=5.):
    if exercise not in DUMBBELLS | BARS:
        return [], None
    if bar_path is not None and exercise != 'Conventional Barbell Deadlift':
        raise ValueError('An explicit floor-pull bar path is only supported for a conventional deadlift.')
    scene, pose = bpy.context.scene, rig.pose.bones
    height = height or json.loads(rig['skin_fit'])['measured']['height']
    unit = height / 1.75
    for value in (grip_ratio, source_grip_ratio):
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not np.isfinite(value) or value <= 0):
            raise ValueError('The requested grip ratio must be a finite positive number.')
    from pullup_range import grip_candidates
    ratio_candidates = grip_candidates(grip_ratio, grip_ratio_tolerance) if grip_ratio is not None else None
    if grip_ratio is None and grip_ratio_tolerance != 0:
        raise ValueError('A bounded grip fit needs the source grip ratio.')
    if (isinstance(source_angle_tolerance_deg, bool) or not isinstance(source_angle_tolerance_deg, (int, float))
            or not np.isfinite(source_angle_tolerance_deg) or source_angle_tolerance_deg <= 0):
        raise ValueError('The reference angle tolerance must be a finite positive number.')
    source_fit_limit = max(0., float(source_angle_tolerance_deg) - .5)
    grips = json.loads(rig['hand_grip'])
    if any(side not in grips for side, _, _ in SIDES):
        raise ValueError('Equipment requires both fitted hand grip models.')
    separate = exercise in DUMBBELLS
    # The cache solves its grips for a 28 mm handle. Match the physical handle
    # radius rather than pushing the fingers into a thicker nominal dumbbell.
    radius = equipment.BARBELL_BAR_DIAMETER / 2 * unit
    props = {}
    if separate:
        for side, _, _ in SIDES:
            obj = equipment.dumbbell(height, kg=5)
            # Only the central handle's radius needs adjustment, preserving heads.
            handle_radius = equipment.DUMBBELL_HANDLE_DIAMETER / 2 * unit
            half = equipment.DUMBBELL_HANDLE_LENGTH / 2 * unit
            for vertex in obj.data.vertices:
                if abs(vertex.co.x) < half + 1e-5 and np.hypot(vertex.co.y, vertex.co.z) < handle_radius * 1.1:
                    vertex.co.y *= radius / handle_radius
                    vertex.co.z *= radius / handle_radius
            obj.name = 'dumbbell_' + side
            props[side] = obj
    elif exercise == PUSHDOWN:
        half = equipment.CABLE_BAR_LENGTH / 2 * unit
        metal, _, _ = equipment._materials()
        # Primitive rods keep their long axis in the object's transform. Join
        # onto an identity anchor, as equipment.barbell does, before replacing
        # the object transform with the animated grip frame. The mesh's own X
        # axis must agree with the finite-shaft contact proxy.
        props['bar'] = equipment._join('cable_handle', [equipment._origin('cable_handle'),
            equipment.capsule('handle', (-half, 0., 0.), (half, 0., 0.), radius, metal)])
        props['bar']['contact_half_length'] = half
    else:
        props['bar'] = equipment.barbell(height, plates=exercise in ('Conventional Barbell Deadlift', 'Flat Barbell Bench Press',
            'Standing Barbell Overhead Press (Shoulder Press)'))
    curl = {}
    for side, title, _ in SIDES:
        curl.update(hand_rig.pose_basis(rig, title, grips[side]['rotations']))
    names = set(curl)
    names.update(cage.P + title + 'Hand' for _, title, _ in SIDES)
    if not separate or exercise == 'Dumbbell Shoulder Press (seated overhead press)':
        names.update(cage.P + title + part for _, title, _ in SIDES for part in ('Arm', 'ForeArm', 'Hand'))
    previous, elbows, path, root_adjustments, source_elbow_goals = {}, {}, [], [], []
    half_grip = None
    fixed_bar = None
    initial_shoulder_width = None
    requested_half_grip = None
    lengths = {side: tuple(pose[cage.P + title + part].length for part in ('Arm', 'ForeArm')) for side, title, _ in SIDES}
    worst_anchor = worst_bend = 0.
    worst_surface_shift = 0.
    arm_goals = None
    pulldown_plan = None
    if exercise == PUSHDOWN and source_arm_targets is not None:
        observed = []
        for frame in range(first, last + 1):
            scene.frame_set(frame)
            observed.append(float(np.mean(list(_elbows(pose).values()))))
        fraction = (np.array(observed) - min(observed)) / max(max(observed) - min(observed), 1e-9)
        from pullup_range import reference_timeline
        arm_goals = {metric: reference_timeline(source_arm_targets, reference_phase_frames, fraction, metric, first)
            for metric in ('elbow_included', 'upperarm_vs_torso')}
        scene.frame_set(first)
    bench = None
    if exercise == 'Flat Barbell Bench Press':
        import bench_contacts
        scene.frame_set(first)
        shoulder_line = np.array(pose[cage.P + 'RightArm'].head - pose[cage.P + 'LeftArm'].head)
        shoulder_line[2] = 0.; shoulder_line /= np.linalg.norm(shoulder_line)
        bench_guard = make_body_guard(rig, skin, shoulder_line, np.array([0., 0., 1.]), height)
        bench = bench_contacts.prepare(rig, skin, first, last, height, grips, lengths, bench_guard, reference_phase_frames)
        half_grip = requested_half_grip = bench['half_grip']
    overhead = None
    if exercise in ('Standing Barbell Overhead Press (Shoulder Press)', 'Dumbbell Shoulder Press (seated overhead press)'):
        import overhead_contacts
        overhead = overhead_contacts.prepare(rig, skin, first, last, reference_phase_frames,
            height, grips, lengths, separate=separate)
        if not separate:
            half_grip = requested_half_grip = overhead['half_grip']
    hang = None
    if exercise == 'Pull-Up':
        if reference_phase_frames is not None:
            rig['reference_motion'] = json.dumps({'phase_frames': reference_phase_frames})
        roots = []
        for frame in range(first, last + 1):
            scene.frame_set(frame)
            roots.append(float(pose[cage.P + 'Hips'].head.z))
        scene.frame_set(first)
        hang = {'source_bottom_elbows_deg': _elbows(pose), 'root_down_m': None, 'grip_fit_satisfied': False,
                'hand_turns': {}, 'max_hand_roll_deg': 0.,
                'fractions': np.clip((np.array(roots) - min(roots)) / max(max(roots) - min(roots), 1e-9), 0., 1.)}
        # With the head/torso held neutral, a few millimetres of extra body
        # travel may be needed for the mascot's actual chin to clear its grip.
        # Measure it before solving, then ease it over the whole pull/return;
        # never snap the body upward only at the last source frame.
        from pullup_range import chin_region
        head_groups = {g.index for g in skin.vertex_groups if g.name == cage.P + 'Head'}
        head_mask = np.array([sum(g.weight for g in vertex.groups if g.group in head_groups) > .99 for vertex in skin.data.vertices])
        transform = rig.matrix_world.inverted() @ skin.matrix_world
        chin_indices = chin_region(np.array([transform @ vertex.co for vertex in skin.data.vertices]), head_mask)
        peak_frame = first + int(np.argmax(roots))
        scene.frame_set(peak_frame)
        for name, value in curl.items():
            pose[name].rotation_mode = 'QUATERNION'
            pose[name].rotation_quaternion = value
        bpy.context.view_layer.update()
        peak_along = np.array(pose[cage.P + 'RightArm'].head - pose[cage.P + 'LeftArm'].head)
        peak_along[2] = 0.
        peak_along /= np.linalg.norm(peak_along)
        peak_anchors = _bar_anchors(pose, grips, peak_along)
        peak_bar_z = float((peak_anchors['left'][2] + peak_anchors['right'][2]) / 2)
        peak_chin = float(skin_wrap.evaluated(skin)[0][chin_indices, 2].min())
        hang['peak_raise_m'] = max(0., peak_bar_z + radius + .001 * height - peak_chin)
        if hang['peak_raise_m'] > .020:
            raise ValueError('The neutral pull-up cannot clear the chin within the bounded 20 mm body-travel correction.')
        scene.frame_set(first)
        from pullup_range import elbow_width_timeline
        hang['widths'] = elbow_width_timeline(source_elbow_widths, reference_phase_frames, hang['fractions'], first)
    for frame in range(first, last + 1):
        scene.frame_set(frame)
        for name, value in curl.items():
            pose[name].rotation_mode = 'QUATERNION'
            pose[name].rotation_quaternion = value
        bpy.context.view_layer.update()
        if separate:
            if overhead is not None:
                shoulders = {side: np.array(pose[cage.P + title + 'Arm'].head) for side, title, _ in SIDES}
                press_goals, press_solutions, _ = overhead_contacts.solve(overhead, frame, shoulders,
                    lengths, grips, elbows, height)
            for side, title, sign in SIDES:
                hand = pose[cage.P + title + 'Hand']
                # Explicit grip roll prevents a supinated biceps curl silently
                # becoming a neutral-grip hammer curl. Lateral-raise handles
                # run forward/back while press/curl handles run across the body.
                if overhead is not None:
                    solution = press_solutions[side]
                    overhead_contacts.apply_solution(rig, side, solution)
                    along, turn = overhead['along'], solution['turn']
                    elbows[side] = {'elbow': solution['elbow'], 'turn': turn}
                else:
                    neutral = exercise in ('Dumbbell Lateral Raise', 'Walking Lunge (dumbbells optional)')
                    along = np.array([0., 1., 0.]) if neutral else np.array([1., 0., 0.])
                    forearm = np.array(hand.head - pose[cage.P + title + 'ForeArm'].head)
                    if exercise == 'Dumbbell Bicep Curl (standing, both arms)':
                        palm = np.cross([1., 0., 0.], forearm)
                    elif exercise == 'Dumbbell Lateral Raise':
                        palm = sign * np.cross([0., 1., 0.], forearm)
                    elif exercise == 'Walking Lunge (dumbbells optional)':
                        palm = np.array([-sign, 0., 0.])
                    else:
                        palm = np.array([0., 1., 0.])
                    candidates = [hand_rig.hand_turn(grips[side], sign, axis, forearm) for axis in (along, -along)]
                    turn = max(candidates, key=lambda rotation: float((rotation @ np.array(grips[side]['n'])) @ palm))
                matrix = (Matrix(turn.tolist()) @ hand.bone.matrix_local.to_3x3()).to_4x4()
                matrix.translation = hand.head
                hand.matrix = matrix
                bpy.context.view_layer.update()
                centre = _anchor(hand, grips[side])
                if overhead is not None:
                    worst_anchor = max(worst_anchor, float(np.linalg.norm(centre - press_goals[side])))
                    worst_bend = max(worst_bend, solution['bend_deg'])
                obj = props[side]
                obj.matrix_world = rig.matrix_world @ _matrix(along, centre)
                obj.keyframe_insert('location', frame=frame)
                obj.rotation_mode = 'QUATERNION'
                obj.keyframe_insert('rotation_quaternion', frame=frame)
            if overhead is not None:
                import overhead_arm_roll
                roll = overhead_arm_roll.orient_upperarms(rig)
                roll['max_change_degrees'] = max(roll['max_change_degrees'],
                    (overhead.get('arm_roll') or {}).get('max_change_degrees', 0.))
                overhead['arm_roll'] = roll
        else:
            shoulders = {side: np.array(pose[cage.P + title + 'Arm'].head) for side, title, _ in SIDES}
            if initial_shoulder_width is None:
                initial_shoulder_width = float(np.linalg.norm(shoulders['right'] - shoulders['left']))
                if not np.isfinite(initial_shoulder_width) or initial_shoulder_width <= 1e-8:
                    raise ValueError('Cannot place a bar on overlapping or invalid shoulders.')
            along = shoulders['right'] - shoulders['left']
            along[2] = 0.
            along /= np.linalg.norm(along)
            requested = None
            if bar_path is not None:
                entry = bar_path.get(frame, bar_path.get(str(frame))) if isinstance(bar_path, dict) else None
                if not isinstance(entry, dict):
                    raise ValueError('The floor-pull bar path must contain every keyed frame.')
                requested = np.asarray(entry.get('centre'), dtype=float)
                if requested.shape != (3,) or not np.isfinite(requested).all():
                    raise ValueError('The floor-pull bar centre must be a finite rig-space point.')
                if entry.get('along') is not None:
                    axis = np.asarray(entry['along'], dtype=float)
                    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-9 or abs(axis[2]) > 1e-6:
                        raise ValueError('The floor-pull shaft must have a finite horizontal axis.')
                    along = axis / np.linalg.norm(axis)
            anchors = _bar_anchors(pose, grips, along)
            if half_grip is None:
                half_grip = abs((anchors['right'] - anchors['left']) @ along) / 2
                requested_half_grip = half_grip
                half_grip = float(np.clip(half_grip, .4 * np.linalg.norm(shoulders['right'] - shoulders['left']), .4 * height))
                if grip_ratio is not None:
                    half_grip = float(grip_ratio) * initial_shoulder_width / 2
                    handle_half = (equipment.BARBELL_LENGTH / 2 - equipment.BARBELL_SLEEVE_LENGTH - equipment.COLLAR_THICKNESS) * unit
                    if half_grip >= handle_half:
                        raise ValueError('The requested grip falls on the thick sleeves or collars instead of the handle.')
                if exercise == PUSHDOWN:
                    from motion_geometry import cable_handle_half_length
                    handle = props['bar']
                    old_half = float(handle['contact_half_length'])
                    new_half = cable_handle_half_length(height, half_grip, equipment.CABLE_BAR_LENGTH)
                    for vertex in handle.data.vertices:
                        vertex.co.x *= new_half / old_half
                    handle['contact_half_length'] = new_half
            centre = (anchors['left'] + anchors['right']) / 2
            if requested is not None:
                centre = requested.copy()
            if exercise == 'Pull-Up':
                if fixed_bar is None:
                    fixed_bar = (centre.copy(), along.copy())
                # Support staging anchors the wrists, but a rotating closed hand
                # has a moving palm-to-shaft offset. Anchor the actual two-hand
                # contact midpoint before solving against the fixed bar.
                root_shift = fixed_bar[0] - centre
                if np.linalg.norm(root_shift) > 1e-10:
                    cage._move_root(rig, Vector(root_shift))
                    pose[cage.P + 'Hips'].keyframe_insert('location', frame=frame)
                    bpy.context.view_layer.update()
                    shoulders = {s: np.array(pose[cage.P + t + 'Arm'].head) for s, t, _ in SIDES}
                centre, along = (v.copy() for v in fixed_bar)
                direction = np.cross([0., 0., 1.], along)
                _, shift = guard_body(rig, skin, centre, along, direction, height)
                if shift:
                    cage._move_root(rig, -Vector(direction) * shift)
                    pose[cage.P + 'Hips'].keyframe_insert('location', frame=frame)
                    bpy.context.view_layer.update()
                    shoulders = {s: np.array(pose[cage.P + t + 'Arm'].head) for s, t, _ in SIDES}
                    worst_surface_shift = max(worst_surface_shift, shift)
                    root_shift = root_shift - direction * shift
                rise = hang['peak_raise_m'] * float(hang['fractions'][frame - first])
                if rise:
                    delta = np.array([0., 0., rise])
                    cage._move_root(rig, Vector(delta))
                    pose[cage.P + 'Hips'].keyframe_insert('location', frame=frame)
                    bpy.context.view_layer.update()
                    shoulders = {s: np.array(pose[cage.P + t + 'Arm'].head) for s, t, _ in SIDES}
                    root_shift = root_shift + delta
                root_adjustments.append({'frame': frame, 'delta_mm': (root_shift * 1000).tolist()})
            if exercise not in ('Pull-Up', 'Flat Barbell Bench Press', 'Standing Barbell Overhead Press (Shoulder Press)'):
                direction = np.array([0., 0., 1.]) if exercise == 'Flat Barbell Bench Press' else np.cross([0., 0., 1.], along)
                guard = make_body_guard(rig, skin, along, direction, height,
                                        half_length=props['bar'].get('contact_half_length'))
                centre, shift = guard(centre)
                worst_surface_shift = max(worst_surface_shift, shift)
            hints = {side: np.array(pose[cage.P + title + 'ForeArm'].head) for side, title, _ in SIDES}
            if exercise == 'Conventional Barbell Deadlift':
                preferences = {side: {'elbow_deg': 178., 'min_elbow_deg': 175.} for side, _, _ in SIDES}
                centre, solutions, adjustment = bounded_reach(rig, skin, shoulders, centre, along, half_grip,
                    lengths, grips, hints, elbows, height, guard=guard, preferences=preferences)
            elif exercise == 'Flat Barbell Bench Press':
                centre, solutions, adjustment = bench_contacts.solve(bench, frame, shoulders, lengths, grips, elbows, height)
            elif exercise == 'Standing Barbell Overhead Press (Shoulder Press)':
                guard = make_body_guard(rig, skin, along, np.cross([0., 0., 1.], along), height)
                centre, solutions, adjustment = overhead_contacts.solve(overhead, frame, shoulders,
                    lengths, grips, elbows, height, guard=guard)
            elif exercise == 'Lat Pulldown (seated, to the front)':
                import pulldown_contacts
                if pulldown_plan is None:
                    pulldown_plan = pulldown_contacts.prepare(rig, first, last, reference_phase_frames,
                        shoulders, along, half_grip, lengths, grips, height, guard)
                centre, solutions, adjustment = pulldown_contacts.solve(rig, pulldown_plan, frame, first, last,
                    reference_phase_frames, shoulders, lengths, grips, elbows, height, guard)
            elif exercise == PUSHDOWN:
                import pushdown_contacts
                source_fk = _elbows(pose)
                observed = source_fk if arm_goals is None else {
                    side: float(arm_goals['elbow_included'][frame - first]) for side, _, _ in SIDES}
                source_elbow_goals.append({'frame': frame, 'elbows_deg': observed, 'source_fk_elbows_deg': source_fk})
                if arm_goals is not None:
                    source_elbow_goals[-1].update(upperarm_goal_deg=float(arm_goals['upperarm_vs_torso'][frame - first]),
                        source_fk_upperarm_deg=_upperarms(rig))
                centre, solutions, adjustment = pushdown_contacts.solve(frame, first, last, reference_phase_frames,
                    shoulders, centre, along, half_grip, lengths, grips, hints, elbows, height, guard,
                    scene.render.fps / scene.render.fps_base)
            elif exercise != 'Pull-Up':
                origin = (shoulders['left'] + shoulders['right']) / 2
                toward = origin - centre
                toward -= along * (toward @ along)
                tangent = toward - direction * (toward @ direction) / float(direction @ direction)
                preferences = None
                def solve(target):
                    return solve_shared_bar(shoulders, target, along, half_grip, lengths,
                        grips, hints, elbows, allow_adaptation=False, preferences=preferences)[1]
                from motion_geometry import collision_safe_target
                centre, solutions, adjustment = collision_safe_target(centre, (toward, tangent), guard, solve)
            else:
                try:
                    # Near-straight arms need a fine hand-roll search. The generic
                    # wrist-first objective can choose a bent elbow even when a
                    # full dead hang is reachable. Keep the source's body pose
                    # and top position; taper a declared, at-most-20mm body drop
                    # from the photographed bottom toward the photographed top.
                    taper = float((1. - hang['fractions'][frame - first]) ** 2)
                    observed = _elbows(pose)
                    bottom_goal = 175.5 if hang['widths'] is not None else 179.
                    preferences = {side: {'elbow_deg': min(179., observed[side]
                        + (bottom_goal - hang['source_bottom_elbows_deg'][side]) * taper)} for side, _, _ in SIDES}
                    for side, _, _ in SIDES:
                        preferences[side].update(max_elbow_z=float(centre[2] - .005 * height), elbow_below_wrist=True,
                            elbow_clearance_m=.005 * height, forearm_vertical_fraction=.9)
                    if hang['widths'] is not None:
                        width = float(hang['widths'][frame - first])
                        from pullup_range import continuous_pole
                        hints, poles = continuous_pole(shoulders, centre, along, width,
                            hang['fractions'][frame - first], lengths, height)
                        shoulder_width = float(np.linalg.norm(shoulders['right'] - shoulders['left']))
                        for side, _, sign in SIDES:
                            # The inferred FK elbow angle can favor the other
                            # depth branch. Fixed grip, fixed limb lengths and
                            # the continuous pole determine the intervening
                            # bend; only the photographed dead hang is required
                            # to be nearly straight.
                            preferences[side].pop('elbow_deg', None)
                            preferences[side].update(poles[side])
                            if side in hang['hand_turns']:
                                preferences[side].update(reference_hand_turn=hang['hand_turns'][side],
                                    max_hand_turn_deg=20. * float(np.sqrt(hang['fractions'][frame - first])))
                            if frame == first:
                                preferences[side]['min_elbow_deg'] = 175.
                    if hang['root_down_m'] is None:
                        candidates = ratio_candidates if hang['widths'] is not None and ratio_candidates else [2 * half_grip / initial_shoulder_width]
                        for ratio in candidates:
                            candidate_half = ratio * initial_shoulder_width / 2
                            handle_half = (equipment.BARBELL_LENGTH / 2 - equipment.BARBELL_SLEEVE_LENGTH - equipment.COLLAR_THICKNESS) * unit
                            if candidate_half >= handle_half:
                                continue
                            for down in np.arange(0., .02001, .001):
                                shifted = {side: point - np.array([0., 0., down]) for side, point in shoulders.items()}
                                try:
                                    _, trial, _ = solve_shared_bar(shifted, centre, along, candidate_half, lengths, grips,
                                        hints, {}, allow_adaptation=False, preferences=preferences)
                                except ValueError:
                                    continue
                                if min(solution['elbow_deg'] for solution in trial.values()) < 175.:
                                    continue
                                if hang['widths'] is not None and grip_ratio_tolerance > 0:
                                    trial_width = float((trial['right']['elbow'] - trial['left']['elbow']) @ along / shoulder_width)
                                    if abs(trial_width - width) > grip_ratio_tolerance:
                                        continue
                                half_grip = candidate_half
                                hang['root_down_m'] = float(down)
                                hang['grip_fit_satisfied'] = True
                                break
                            if hang['root_down_m'] is not None:
                                break
                        if hang['root_down_m'] is None:
                            hang['root_down_m'] = 0.
                    down = hang['root_down_m'] * taper
                    if down:
                        delta = np.array([0., 0., -down])
                        cage._move_root(rig, Vector(delta))
                        pose[cage.P + 'Hips'].keyframe_insert('location', frame=frame)
                        bpy.context.view_layer.update()
                        shoulders = {s: np.array(pose[cage.P + t + 'Arm'].head) for s, t, _ in SIDES}
                        hints = {s: np.array(pose[cage.P + t + 'ForeArm'].head) for s, t, _ in SIDES}
                        root_adjustments[-1]['delta_mm'] = (np.array(root_adjustments[-1]['delta_mm']) + delta * 1000).tolist()
                    centre, solutions, adjustment = solve_shared_bar(shoulders, centre, along, half_grip, lengths,
                        grips, hints, {} if hang['widths'] is not None else elbows,
                        allow_adaptation=False, preferences=preferences)
                except ValueError as error:
                    raise ValueError('The fixed pull-up bar cannot be gripped while keeping the head clear and the arms within reach.') from error
            bar = props['bar']
            bar.matrix_world = rig.matrix_world @ _matrix(along, centre)
            bar.rotation_mode = 'QUATERNION'
            bar.keyframe_insert('location', frame=frame)
            bar.keyframe_insert('rotation_quaternion', frame=frame)
            for side, title, sign in SIDES:
                solution = solutions[side]
                if exercise == 'Pull-Up':
                    hang['hand_turns'].setdefault(side, solution['turn'].copy())
                    turn = hang['hand_turns'][side].T @ solution['turn']
                    hang['max_hand_roll_deg'] = max(hang['max_hand_roll_deg'],
                        float(np.degrees(np.arccos(np.clip((np.trace(turn) - 1.) / 2., -1., 1.)))))
                arm, fore, hand = [pose[cage.P + title + part] for part in ('Arm', 'ForeArm', 'Hand')]
                cage._aim_pose(arm, Vector(solution['elbow']) - arm.head)
                cage._aim_pose(fore, Vector(solution['wrist']) - fore.head)
                matrix = (Matrix(solution['turn'].tolist()) @ hand.bone.matrix_local.to_3x3()).to_4x4()
                matrix.translation = hand.head
                hand.matrix = matrix
                bpy.context.view_layer.update()
                worst_anchor = max(worst_anchor, float(np.linalg.norm(_anchor(hand, grips[side]) - (centre + sign * half_grip * along))))
                worst_bend = max(worst_bend, solution['bend_deg'])
                elbows[side] = {'elbow': solution['elbow'], 'turn': solution['turn']}
            if bench is not None:
                import bench_arm_roll
                roll = bench_arm_roll.orient_upperarms(rig)
                roll['max_change_degrees'] = max(roll['max_change_degrees'],
                    (bench.get('arm_roll') or {}).get('max_change_degrees', 0.))
                bench['arm_roll'] = roll
            if pulldown_plan is not None:
                import pulldown_arm_roll
                roll = pulldown_arm_roll.orient_upperarms(rig)
                roll['max_change_degrees'] = max(roll['max_change_degrees'],
                    (pulldown_plan.get('arm_roll') or {}).get('max_change_degrees', 0.))
                pulldown_plan['arm_roll'] = roll
            if exercise == PUSHDOWN:
                source_elbow_goals[-1]['solved_elbows_deg'] = _elbows(pose)
                if arm_goals is not None:
                    source_elbow_goals[-1]['solved_upperarm_deg'] = _upperarms(rig)
            path_row = {'frame': frame, 'adjustment_mm': float(adjustment * 1000)}
            if requested is not None:
                path_row.update(requested_centre=requested.tolist(), actual_centre=centre.tolist())
            path.append(path_row)
        cage._key(rig, previous, frame, names)
    masks, rows = _hand_masks(skin), {side: [] for side, _, _ in SIDES}
    fixed_pose = None
    bar_drift = bar_rotation = 0.
    for frame in range(first, last + 1):
        scene.frame_set(frame)
        points = skin_wrap.evaluated(skin)[0]
        if exercise == 'Pull-Up':
            matrix = props['bar'].matrix_world
            if fixed_pose is None:
                fixed_pose = (matrix.translation.copy(), matrix.to_quaternion())
            bar_drift = max(bar_drift, (matrix.translation - fixed_pose[0]).length)
            bar_rotation = max(bar_rotation, float(np.degrees(matrix.to_quaternion().rotation_difference(fixed_pose[1]).angle)))
        for side, _, _ in SIDES:
            obj = props[side] if separate else props['bar']
            centre = np.array(obj.matrix_world.translation)
            along = np.array((obj.matrix_world.to_3x3() @ Vector((1, 0, 0))).normalized())
            half = equipment.DUMBBELL_HANDLE_LENGTH / 2 * unit if separate else float(obj.get('contact_half_length', equipment.BARBELL_LENGTH / 2 * unit))
            gaps = shaft_gaps(points[masks[side]], centre, along, half, radius)
            rows[side].append((float(abs(gaps).min()), max(0., float(-gaps.min()))))
    sides = {side: {'frames': len(values), 'frames_touching_within_3mm': sum(gap <= .003 for gap, _ in values),
                    'nearest_gap_mm_max': max(gap for gap, _ in values) * 1000,
                    'deepest_inside_bar_mm': max(depth for _, depth in values) * 1000} for side, values in rows.items()}
    report = {'frames': last - first + 1, 'equipment': 'dumbbells' if separate else 'bar',
              'max_grip_anchor_error_mm': worst_anchor * 1000, 'worst_wrist_bend_deg': worst_bend,
              'max_surface_path_adjustment_mm': worst_surface_shift * 1000,
              'path_adjustments': path, 'sides': sides}
    if exercise in ('Dumbbell Bicep Curl (standing, both arms)', 'Dumbbell Lateral Raise', PUSHDOWN):
        import isolation_contacts
        report['isolation_range'] = isolation_contacts.measure(rig, first, last, height, reference_phase_frames, exercise)
        report['isolation_profile'] = {'method': 'declared upright isolation technique; source depth is not calibrated',
            'source_comparison_retained': True,
            'source_conflict': 'The technique profile completes 175-degree elbow extension; the demonstrated endpoint is incomplete and its comparison failures are retained.' if exercise == PUSHDOWN else None}
    if exercise == 'Conventional Barbell Deadlift':
        report['loaded_plates'] = True
        report['plate_radius_character_units'] = equipment.PLATE_DIAMETER / 2 * unit
        report['bar_path_source'] = 'provided rig-space floor-pull trajectory' if bar_path is not None else 'reference-driven fitted closed-hand anchors'
    if bench is not None:
        report['bench_range'] = bench_contacts.measure(rig, props['bar'], first, last, height, reference_phase_frames)
        report['bench_profile'] = {'method': bench['method'], 'grip_ratio': 2.,
            'top_rig_m': bench['top'].tolist(), 'bottom_rig_m': bench['bottom'].tolist(),
            'bottom_surface_shift_mm': float(bench['bottom_guard_shift'] * 1000), 'loaded_plates': True,
            'lockout_lateral_fit_mm': bench['lockout_lateral_fit_mm'],
            'max_hand_roll_from_lower_chest_deg': 30., 'upper_arm_roll': bench.get('arm_roll')}
    if overhead is not None:
        report['overhead_range'] = overhead_contacts.measure(rig, skin, props, separate, grips,
            first, last, height, reference_phase_frames)
        report['overhead_profile'] = {'method': overhead['method'],
            'clearance_shifts': overhead['clearance_shifts'], 'hand_roll_limit_deg': overhead['motion_hand_roll_limit_deg'],
            'grip_roll_reference': 'eased lower-to-lockout grip' if separate else 'starting grip',
            'source_depth_calibrated': False, 'loaded_plates': not separate,
            'upper_arm_roll': overhead.get('arm_roll')}
    if exercise == 'Lat Pulldown (seated, to the front)':
        import pulldown_contacts
        report['pulldown_range'] = pulldown_contacts.measure(rig, props['bar'], first, last, reference_phase_frames)
        report['pulldown_profile'] = {'method': pulldown_plan['method'],
            'source_depth': 'inferred, not calibrated', 'bottom_upperarm_limit_deg': 70.,
            'top_rig_m': pulldown_plan['top'].tolist(), 'bottom_rig_m': pulldown_plan['bottom'].tolist(),
            'bottom_guard_shift_mm': pulldown_plan['bottom_guard_shift_mm'], 'max_hand_turn_deg': 25.,
            'max_actual_hand_roll_deg': pulldown_plan['max_hand_roll_deg'],
            'arm_roll': pulldown_plan.get('arm_roll'),
            'clearance_path_shifts': pulldown_plan['clearance_path_shifts']}
    if root_adjustments:
        report['root_adjustments'] = root_adjustments
        report['fixed_bar_drift_mm'] = bar_drift * 1000
        report['fixed_bar_rotation_deg'] = bar_rotation
        report['hang_extension'] = {'max_body_drop_mm': hang['root_down_m'] * 1000,
            'max_body_raise_mm': hang['peak_raise_m'] * 1000,
            'limit_mm': 20., 'source_bottom_elbows_deg': hang['source_bottom_elbows_deg'],
            'method': 'fixed bar and limb lengths; bounded lowering for dead hang and measured neutral-chin travel correction, eased by source root height'}
        report['pullup_range'] = pullup_range(rig, skin, props['bar'], first, last)
        report['pullup_grip_roll'] = {'max_turn_deg': hang['max_hand_roll_deg'], 'limit_deg': 20.,
            'method': 'One palm-roll branch within 20 degrees of the dead-hang grip; available roll eases with square-root body travel on both strokes.'}
        if hang['widths'] is not None:
            report['elbow_width_references'] = source_elbow_widths
            report['elbow_width_phase_frames'] = reference_phase_frames
            report['elbow_width_goals'] = [{'frame': frame, 'ratio': float(hang['widths'][frame - first])}
                for frame in range(first, last + 1)]
    if not separate:
        report['grip_ratio'] = 2 * half_grip / initial_shoulder_width
        report['requested_grip_ratio'] = source_grip_ratio if source_grip_ratio is not None else (grip_ratio if grip_ratio is not None else 2 * requested_half_grip / initial_shoulder_width)
        report['grip_source'] = 'provided shoulder-relative ratio' if grip_ratio is not None else 'reference-driven fitted hand span'
        if bench is not None:
            report['grip_source'] = 'declared medium-grip bench profile; not calibrated from source photographs'
            report['requested_grip_ratio'] = 2.
            report['grip_calibration'] = 'The model uses a declared two-shoulder-width span; source width and depth remain uncalibrated.'
        if overhead is not None:
            report['grip_source'] = 'declared fitted overhead-press grip; not calibrated from source photographs'
            report['requested_grip_ratio'] = 1.7
            report['grip_calibration'] = 'The overhead press uses a declared medium grip and full lockout; photo depth remains inferred.'
        if exercise == 'Pull-Up' and grip_ratio is not None and grip_ratio_tolerance > 0 and hang['widths'] is not None:
            report['grip_adaptation'] = {
                'requested_ratio': float(grip_ratio), 'chosen_ratio': report['grip_ratio'],
                'allowed_ratio_difference': float(grip_ratio_tolerance),
                'span_change_mm': (report['grip_ratio'] - grip_ratio) * initial_shoulder_width * 1000,
                'fit_satisfied': hang['grip_fit_satisfied'],
                'method': 'nearest source grip search within the unchanged dead-hang, wrist and photographed elbow-width limits',
            }
            if abs(report['grip_ratio'] - grip_ratio) > 1e-9:
                report['grip_source'] = 'bounded fit within the photographed grip tolerance'
        if exercise == PUSHDOWN:
            report['attachment'] = {'kind': 'short straight cable handle',
                'length_character_units': float(props['bar']['contact_half_length']) * 2,
                'length_adult_m': float(props['bar']['contact_half_length']) * 2 / unit,
                'minimum_standard_length_adult_m': equipment.CABLE_BAR_LENGTH,
                'source_width_calibrated': False,
                'note': 'Attachment length fits the closed hand span with palm margins; the side photograph does not calibrate its width.'}
            report['source_elbow_goals'] = source_elbow_goals
            report['max_source_elbow_error_deg'] = max(abs(row['solved_elbows_deg'][side] - row['elbows_deg'][side])
                for row in source_elbow_goals for side, _, _ in SIDES)
            if arm_goals is not None:
                report['arm_references'] = source_arm_targets
                report['source_angle_tolerance_deg'] = float(source_angle_tolerance_deg)
                report['max_source_upperarm_error_deg'] = max(abs(row['solved_upperarm_deg'][side] - row['upperarm_goal_deg'])
                    for row in source_elbow_goals for side, _, _ in SIDES)
    if pulley_anchor is not None:
        curve = bpy.data.curves.new('equipment_cable', 'CURVE')
        curve.dimensions = '3D'
        curve.bevel_depth = .002 * unit
        spline = curve.splines.new('POLY')
        spline.points.add(1)
        cable = bpy.data.objects.new('equipment_cable', curve)
        scene.collection.objects.link(cable)
        for frame in range(first, last + 1):
            scene.frame_set(frame)
            for point, position in zip(spline.points, (props['bar'].matrix_world.translation, pulley_anchor)):
                point.co = (*position, 1.)
                point.keyframe_insert('co', frame=frame)
        report['cable_anchor'] = list(pulley_anchor)
    if not separate:
        from row_bar import body_clearance
        report['body_clearance'] = body_clearance(rig, skin, props['bar'], first, last, include_head=True)
    else:
        report['head_body_clearance'] = dumbbell_clearance(skin, props, first, last, height)
    if (profile_for(exercise) or {}).get('overhead'):
        head = {g.index for g in skin.vertex_groups if 'Head' in g.name}
        mask = np.array([sum(g.weight for g in v.groups if g.group in head) > .5 for v in skin.data.vertices])
        clearances = []
        for frame in range(first, last + 1):
            scene.frame_set(frame)
            highest = float(skin_wrap.evaluated(skin)[0][mask, 2].max())
            clearances.append(min(float(obj.matrix_world.translation.z) for obj in props.values()) - highest)
        report['overhead_shortfall_mm'] = max(0., -max(clearances)) * 1000
    scene.frame_set(first)
    result = list(props.values())
    if exercise == 'Pull-Up':
        centre = props['bar'].matrix_world.translation
        along = (props['bar'].matrix_world.to_3x3() @ Vector((1, 0, 0))).normalized()
        metal, _, _ = equipment._materials()
        for sign in (-1, 1):
            top = centre + along * sign * height * .4
            result.append(equipment.capsule('pullup_upright', (top.x, top.y, 0.), top, height * .015, metal))
    return result, report


def guard_body(rig, skin, centre, along, forward, height):
    """Move a bar outside actual torso/head surfaces before solving the grip.

    Standing bars move forward, supine bars move up. Every adjustment is reported;
    an incompatible arm reach remains visible in the subsequent measured checks.
    """
    return make_body_guard(rig, skin, along, forward, height)(centre)


def make_body_guard(rig, skin, along, forward, height, half_length=None):
    """Freeze this frame's body surface once for a bounded contact search."""
    from mathutils.bvhtree import BVHTree
    groups = {g.index for g in skin.vertex_groups if any(n in g.name for n in ('Hips', 'Spine', 'UpLeg', 'Leg', 'Neck', 'Head')) and 'Hand' not in g.name}
    mask = np.array([sum(g.weight for g in v.groups if g.group in groups) > .5 for v in skin.data.vertices])
    skin.data.calc_loop_triangles()
    faces = [tuple(t.vertices) for t in skin.data.loop_triangles if all(mask[v] for v in t.vertices)]
    if not faces:
        raise ValueError('Cannot measure equipment clearance against the body.')
    points = skin_wrap.evaluated(skin)[0]
    tree = BVHTree.FromPolygons([Vector(p) for p in points], faces, all_triangles=True)
    forward = np.asarray(forward) / np.linalg.norm(forward)
    perpendicular = np.cross(along, forward)
    perpendicular /= np.linalg.norm(perpendicular)
    radius = equipment.BARBELL_BAR_DIAMETER / 2 * height / 1.75
    half = equipment.BARBELL_LENGTH / 2 * height / 1.75 if half_length is None else float(half_length)
    if not np.isfinite(half) or half <= 0:
        raise ValueError('Cannot check equipment against a nonpositive or invalid handle length.')
    world, inverse = rig.matrix_world, rig.matrix_world.inverted()
    ray = (world.to_3x3() @ Vector(forward)).normalized()
    count = int(np.ceil(2 * half / (radius / 2))) + 1
    axial_samples = np.linspace(-half, half, count)
    sections = [(offset, np.sqrt(max(0., radius ** 2 - offset ** 2))) for offset in np.linspace(-radius, radius, 9)]
    def guard(centre):
        centre = np.asarray(centre, dtype=float)
        required = float(centre @ forward)
        for offset, section in sections:
            for axial in axial_samples:
                sample = centre + along * axial + perpendicular * offset
                start = world @ Vector(sample + forward * height * 2)
                hit, _, _, _ = tree.ray_cast(start, -ray, height * 4)
                if hit is not None:
                    position = np.array(inverse @ hit)
                    required = max(required, float(position @ forward) + section + .003)
        shift = max(0., required - float(centre @ forward))
        return centre + forward * shift, shift
    return guard


def bounded_reach(rig, skin, shoulders, centre, along, half, lengths, grips, hints, previous, height, guard=None, preferences=None):
    """Keep the shaft outside the body and adapt its height by at most 20 mm.

    Moving a target back toward the shoulder can put it inside the thigh. A small
    vertical adjustment is tried instead; an infeasible safe pose is rejected.
    """
    try:
        return solve_shared_bar(shoulders, centre, along, half, lengths, grips, hints, previous, allow_adaptation=False,
                                preferences=preferences)
    except ValueError as error:
        sign = 1 if np.mean([p[2] for p in shoulders.values()]) >= centre[2] else -1
        direction = np.cross([0., 0., 1.], along)
        guard = guard or make_body_guard(rig, skin, along, direction, height)
        for step in range(1, 21):
            target, _ = guard(centre + np.array([0., 0., sign * step * .001]))
            try:
                point, solutions, _ = solve_shared_bar(shoulders, target, along, half, lengths, grips, hints, previous, allow_adaptation=False,
                                                      preferences=preferences)
                adaptation = float(np.linalg.norm(point - centre))
                if adaptation <= .020:
                    return point, solutions, adaptation
            except ValueError:
                continue
        raise ValueError('The safe deadlift bar path is outside this character\'s arm reach, even after a bounded 20 mm height correction.') from error


def dumbbell_clearance(skin, props, first, last, height):
    """Test body vertices against the actual finite convex hex weight heads.

    Hands and forearms are excluded: their contact with handles is intentional and
    measured separately. These are surface samples, with the limitation disclosed.
    """
    groups = {g.index for g in skin.vertex_groups if any(n in g.name for n in ('Head', 'Neck', 'Spine', 'Hips', 'UpLeg', 'Leg')) and 'Hand' not in g.name}
    mask = np.array([sum(g.weight for g in v.groups if g.group in groups) > .5 for v in skin.data.vertices])
    if not mask.any():
        return {'frames': 0, 'max_penetration_mm': None}
    unit, scale = height / 1.75, (5. / 10.) ** (1. / 3.)
    radius = equipment.DUMBBELL_HEAD_RADIUS_10KG * scale * unit
    thickness = equipment.DUMBBELL_HEAD_THICKNESS_10KG * scale * unit
    along = equipment.DUMBBELL_HANDLE_LENGTH / 2 * unit + thickness / 2
    from motion_geometry import hex_head_depth
    worst = 0.
    for frame in range(first, last + 1):
        bpy.context.scene.frame_set(frame)
        points = skin_wrap.evaluated(skin)[0][mask]
        for obj in props.values():
            inverse = np.array(obj.matrix_world.inverted())
            local = points @ inverse[:3, :3].T + inverse[:3, 3]
            for sign in (-1, 1):
                worst = max(worst, float(hex_head_depth(local, sign * along, radius, thickness).max()))
    return {'frames': last - first + 1, 'max_penetration_mm': worst * 1000,
            'method': 'finite convex weight-head/body surface samples; hand contact checked separately'}
