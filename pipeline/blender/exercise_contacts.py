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
             'Dumbbell Shoulder Press (seated overhead press)'}
BARS = {'Conventional Barbell Deadlift', 'Flat Barbell Bench Press',
        'Standing Barbell Overhead Press (Shoulder Press)',
        'Lat Pulldown (seated, to the front)', 'Tricep Pushdown (cable, bar or rope attachment)', 'Pull-Up'}


def _turn(hand):
    return np.array(hand.matrix.to_3x3() @ hand.bone.matrix_local.to_3x3().inverted())


def _anchor(hand, grip):
    return np.array(hand.head) + _turn(hand) @ (np.array(grip['B']) - grip['wrist'])


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


def apply(rig, skin, first, last, exercise, height=None, pulley_anchor=None):
    if exercise not in DUMBBELLS | BARS:
        return [], None
    scene, pose = bpy.context.scene, rig.pose.bones
    height = height or json.loads(rig['skin_fit'])['measured']['height']
    unit = height / 1.75
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
    else:
        props['bar'] = equipment.barbell(height, plates=False)
    curl = {}
    for side, title, _ in SIDES:
        curl.update(hand_rig.pose_basis(rig, title, grips[side]['rotations']))
    names = set(curl)
    names.update(cage.P + title + 'Hand' for _, title, _ in SIDES)
    if not separate:
        names.update(cage.P + title + part for _, title, _ in SIDES for part in ('Arm', 'ForeArm', 'Hand'))
    previous, elbows, path = {}, {}, []
    half_grip = None
    fixed_bar = None
    lengths = {side: tuple(pose[cage.P + title + part].length for part in ('Arm', 'ForeArm')) for side, title, _ in SIDES}
    worst_anchor = worst_bend = 0.
    worst_surface_shift = 0.
    for frame in range(first, last + 1):
        scene.frame_set(frame)
        for name, value in curl.items():
            pose[name].rotation_mode = 'QUATERNION'
            pose[name].rotation_quaternion = value
        bpy.context.view_layer.update()
        if separate:
            for side, title, sign in SIDES:
                hand = pose[cage.P + title + 'Hand']
                # Explicit grip roll prevents a supinated biceps curl silently
                # becoming a neutral-grip hammer curl. Lateral-raise handles
                # run forward/back while press/curl handles run across the body.
                along = np.array([0., 1., 0.]) if exercise == 'Dumbbell Lateral Raise' else np.array([1., 0., 0.])
                forearm = np.array(hand.head - pose[cage.P + title + 'ForeArm'].head)
                if exercise == 'Dumbbell Bicep Curl (standing, both arms)':
                    palm = np.cross([1., 0., 0.], forearm)
                elif exercise == 'Dumbbell Lateral Raise':
                    palm = sign * np.cross([0., 1., 0.], forearm)
                else:
                    palm = np.array([0., 1., 0.])
                candidates = [hand_rig.hand_turn(grips[side], sign, axis, forearm) for axis in (along, -along)]
                turn = max(candidates, key=lambda rotation: float((rotation @ np.array(grips[side]['n'])) @ palm))
                matrix = (Matrix(turn.tolist()) @ hand.bone.matrix_local.to_3x3()).to_4x4()
                matrix.translation = hand.head
                hand.matrix = matrix
                bpy.context.view_layer.update()
                centre = _anchor(hand, grips[side])
                obj = props[side]
                obj.matrix_world = rig.matrix_world @ _matrix(along, centre)
                obj.keyframe_insert('location', frame=frame)
                obj.rotation_mode = 'QUATERNION'
                obj.keyframe_insert('rotation_quaternion', frame=frame)
        else:
            shoulders = {side: np.array(pose[cage.P + title + 'Arm'].head) for side, title, _ in SIDES}
            along = shoulders['right'] - shoulders['left']
            along[2] = 0.
            along /= np.linalg.norm(along)
            anchors = {side: _anchor(pose[cage.P + title + 'Hand'], grips[side]) for side, title, _ in SIDES}
            if half_grip is None:
                half_grip = abs((anchors['right'] - anchors['left']) @ along) / 2
                half_grip = float(np.clip(half_grip, .4 * np.linalg.norm(shoulders['right'] - shoulders['left']), .4 * height))
            centre = (anchors['left'] + anchors['right']) / 2
            if exercise == 'Pull-Up':
                if fixed_bar is None:
                    fixed_bar = (centre.copy(), along.copy())
                centre, along = (v.copy() for v in fixed_bar)
                direction = np.cross([0., 0., 1.], along)
                _, shift = guard_body(rig, skin, centre, along, direction, height)
                if shift:
                    cage._move_root(rig, -Vector(direction) * shift)
                    pose[cage.P + 'Hips'].keyframe_insert('location', frame=frame)
                    bpy.context.view_layer.update()
                    shoulders = {s: np.array(pose[cage.P + t + 'Arm'].head) for s, t, _ in SIDES}
                    worst_surface_shift = max(worst_surface_shift, shift)
            if exercise != 'Pull-Up':
                direction = np.array([0., 0., 1.]) if exercise == 'Flat Barbell Bench Press' else np.cross([0., 0., 1.], along)
                centre, shift = guard_body(rig, skin, centre, along, direction, height)
                worst_surface_shift = max(worst_surface_shift, shift)
            hints = {side: np.array(pose[cage.P + title + 'ForeArm'].head) for side, title, _ in SIDES}
            if exercise == 'Conventional Barbell Deadlift':
                centre, solutions, adjustment = bounded_reach(rig, skin, shoulders, centre, along, half_grip,
                    lengths, grips, hints, elbows, height)
            else:
                centre, solutions, adjustment = solve_shared_bar(shoulders, centre, along, half_grip, lengths,
                                                                 grips, hints, elbows, allow_adaptation=exercise != 'Pull-Up')
            bar = props['bar']
            bar.matrix_world = rig.matrix_world @ _matrix(along, centre)
            bar.rotation_mode = 'QUATERNION'
            bar.keyframe_insert('location', frame=frame)
            bar.keyframe_insert('rotation_quaternion', frame=frame)
            for side, title, sign in SIDES:
                solution = solutions[side]
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
            path.append({'frame': frame, 'adjustment_mm': float(adjustment * 1000)})
        cage._key(rig, previous, frame, names)
    masks, rows = _hand_masks(skin), {side: [] for side, _, _ in SIDES}
    for frame in range(first, last + 1):
        scene.frame_set(frame)
        points = skin_wrap.evaluated(skin)[0]
        for side, _, _ in SIDES:
            obj = props[side] if separate else props['bar']
            centre = np.array(obj.matrix_world.translation)
            along = np.array((obj.matrix_world.to_3x3() @ Vector((1, 0, 0))).normalized())
            half = (equipment.DUMBBELL_HANDLE_LENGTH if separate else equipment.BARBELL_LENGTH) / 2 * unit
            gaps = shaft_gaps(points[masks[side]], centre, along, half, radius)
            rows[side].append((float(abs(gaps).min()), max(0., float(-gaps.min()))))
    sides = {side: {'frames': len(values), 'frames_touching_within_3mm': sum(gap <= .003 for gap, _ in values),
                    'nearest_gap_mm_max': max(gap for gap, _ in values) * 1000,
                    'deepest_inside_bar_mm': max(depth for _, depth in values) * 1000} for side, values in rows.items()}
    report = {'frames': last - first + 1, 'equipment': 'dumbbells' if separate else 'bar',
              'max_grip_anchor_error_mm': worst_anchor * 1000, 'worst_wrist_bend_deg': worst_bend,
              'max_surface_path_adjustment_mm': worst_surface_shift * 1000,
              'path_adjustments': path, 'sides': sides}
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
    half = equipment.BARBELL_LENGTH / 2 * height / 1.75
    required = float(centre @ forward)
    for offset in np.linspace(-radius, radius, 5):
        section = np.sqrt(max(0., radius ** 2 - offset ** 2))
        for axial in np.linspace(-half, half, 81):
            sample = centre + along * axial + perpendicular * offset
            start = rig.matrix_world @ Vector(sample + forward * height * 2)
            hit, _, _, _ = tree.ray_cast(start, -Vector(forward), height * 4)
            if hit is not None:
                position = np.array(rig.matrix_world.inverted() @ hit)
                required = max(required, float(position @ forward) + section + .003)
    shift = max(0., required - float(centre @ forward))
    return centre + forward * shift, shift


def bounded_reach(rig, skin, shoulders, centre, along, half, lengths, grips, hints, previous, height):
    """Keep the shaft outside the body and adapt its height by at most 20 mm.

    Moving a target back toward the shoulder can put it inside the thigh. A small
    vertical adjustment is tried instead; an infeasible safe pose is rejected.
    """
    try:
        return solve_shared_bar(shoulders, centre, along, half, lengths, grips, hints, previous, allow_adaptation=False)
    except ValueError as error:
        sign = 1 if np.mean([p[2] for p in shoulders.values()]) >= centre[2] else -1
        direction = np.cross([0., 0., 1.], along)
        for step in range(1, 21):
            target, _ = guard_body(rig, skin, centre + np.array([0., 0., sign * step * .001]), along, direction, height)
            try:
                point, solutions, _ = solve_shared_bar(shoulders, target, along, half, lengths, grips, hints, previous, allow_adaptation=False)
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
