"""A reversible arm-reach variant of an already fitted, unposed character.

The face, torso, legs, hands and fingers keep their shapes. Only upper-arm and
forearm segments become longer; distal hand bones/vertices translate rigidly.
Apply after ``mpfb_cage.wear`` and before retargeting. The original GLB and fitted
cache are never written. This is a declared body-proportion variant, not an
invisible per-frame IK stretch or permission to waive movement/contact checks.
"""
import copy
import json
import math

import numpy as np

P = 'mixamorig:'
AUTO_EXERCISES = frozenset({
    'Conventional Barbell Deadlift',
    'Flat Barbell Bench Press',
    'Standing Barbell Overhead Press (Shoulder Press)',
    'Dumbbell Shoulder Press (seated overhead press)',
    'Lat Pulldown (seated, to the front)', 'Pull-Up', 'Front Plank (forearm plank)',
    'Tricep Pushdown (cable, bar or rope attachment)',
})


def measure_reach(rig, skin):
    """An optimistic straight-arm bound, measured on this instance's rest mesh.

    If even a vertical arm cannot clear the head, source-pose IK cannot fix it.
    A positive bound is not approval: the final source pose still needs all gates.
    """
    inverse = rig.matrix_world.inverted()
    co = np.array([inverse @ (skin.matrix_world @ vertex.co) for vertex in skin.data.vertices])
    head_groups = {g.index for g in skin.vertex_groups if g.name == P + 'Head'}
    selected = np.array([sum(g.weight for g in v.groups if g.group in head_groups) > .5 for v in skin.data.vertices])
    if not selected.any():
        raise ValueError('Cannot measure the character head for a reach policy.')
    top = float(co[selected, 2].max())
    grips = json.loads(rig['hand_grip'])
    sides = {}
    for side in ('left', 'right'):
        arm, fore = (rig.data.bones[P + side.title() + part] for part in ('Arm', 'ForeArm'))
        offset = float(np.linalg.norm(np.array(grips[side]['B']) - grips[side]['wrist']))
        bound = float(arm.head_local.z + arm.length + fore.length + offset)
        sides[side] = {'upper_m': float(arm.length), 'fore_m': float(fore.length),
                       'shoulder_z_m': float(arm.head_local.z), 'grip_offset_m': offset,
                       'vertical_handle_bound_m': bound, 'head_shortfall_mm': max(0., top - bound) * 1000}
    return {'head_top_m': top, 'sides': sides}


def apply_if_needed(rig, skin, exercise, mode='auto'):
    """Select original or fixed fitness proportions for one fresh fitted instance.

    ``auto`` adapts only overhead/hanging/forearm-support profiles whose optimistic
    original reach misses the head clearance. ``fitness`` explicitly uses the same
    proportions for every profile. ``original`` leaves the fitted model unchanged.
    """
    if mode not in ('auto', 'original', 'fitness'):
        raise ValueError('Choose auto, original or fitness proportions.')
    existing = rig.get('reach_variant')
    if existing:
        if mode == 'original':
            raise ValueError('Load a fresh fitted character to restore original proportions.')
        return json.loads(existing)
    if mode == 'original' or (mode == 'auto' and exercise not in AUTO_EXERCISES):
        return None
    original = measure_reach(rig, skin)
    height = float(json.loads(rig['skin_fit'])['measured']['height'])
    margin = .02 * height
    deficient = any(side['vertical_handle_bound_m'] < original['head_top_m'] + margin
                    for side in original['sides'].values())
    if mode == 'auto' and not deficient:
        return None
    report = apply(rig, skin, upper_scale=1.8, fore_scale=1.8)
    support_reasons = {
        'Conventional Barbell Deadlift': 'The short-arm rest character uses the validated fitness proportions for a loaded conventional floor pull.',
        'Flat Barbell Bench Press': 'The short-arm rest character uses the validated fitness proportions for a supported bench grip.',
        'Tricep Pushdown (cable, bar or rope attachment)': 'The short-arm rest character uses the validated fitness proportions for a collision-free cable grip.',
    }
    reason = (support_reasons[exercise] if deficient and exercise in support_reasons else
              'Original straight-arm reach cannot clear the head.' if deficient else 'Fitness proportions selected explicitly.')
    report.update(requested_mode=mode, original_reach=original, actual_reach=measure_reach(rig, skin), reason=reason)
    rig['reach_variant'] = json.dumps(report)
    return report


def _scale(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 1. <= value <= 2.:
        raise ValueError('Arm reach scales must be finite numbers between 1 and 2.')
    return float(value)


def displacement(points, shoulder, elbow, wrist, upper_weights, fore_weights, hand_weights,
                 upper_scale, fore_scale, *, fixed=None, seams=None):
    """Rest-space mesh displacement with fixed segment ends and no finger scaling.

    Longitudinal masks protect inner-collar/torso vertices from stray Arm weights.
    Coincident mesh seams share displacement even when weights differ slightly.
    """
    upper_scale, fore_scale = _scale(upper_scale), _scale(fore_scale)
    points = np.asarray(points, dtype=float)
    shoulder, elbow, wrist = (np.asarray(point, dtype=float) for point in (shoulder, elbow, wrist))
    weights = np.column_stack([upper_weights, fore_weights, hand_weights]).astype(float)
    if points.ndim != 2 or points.shape[1] != 3 or weights.shape != (len(points), 3):
        raise ValueError('Reach deformation needs one finite point/weight row per vertex.')
    if not np.isfinite(points).all() or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('Reach deformation cannot use invalid points or weights.')
    upper, fore = elbow - shoulder, wrist - elbow
    upper_sq, fore_sq = float(upper @ upper), float(fore @ fore)
    if not np.isfinite([upper_sq, fore_sq]).all() or min(upper_sq, fore_sq) < 1e-12:
        raise ValueError('Reach deformation needs nonzero finite arm segments.')
    upper_progress = np.clip((points - shoulder) @ upper / upper_sq, 0., 1.)
    fore_progress = np.clip((points - elbow) @ fore / fore_sq, 0., 1.)
    du, df = upper * (upper_scale - 1.), fore * (fore_scale - 1.)
    delta = np.outer(weights.sum(axis=1) * upper_progress, du)
    delta += np.outer(weights[:, 1:].sum(axis=1) * fore_progress, df)
    rigid_hand = (weights[:, 2] > .99) & (upper_progress > .999) & (fore_progress > .75)
    delta[rigid_hand] = du + df
    if fixed is not None:
        fixed = np.asarray(fixed, dtype=bool)
        if fixed.shape != (len(points),):
            raise ValueError('The fixed vertex mask does not match the mesh.')
        delta[fixed] = 0.
    if seams is not None:
        seams = np.asarray(seams)
        if seams.shape != (len(points),):
            raise ValueError('The seam map does not match the mesh.')
        _, inverse = np.unique(seams, return_inverse=True)
        counts = np.bincount(inverse)
        sums = np.zeros((len(counts), 3))
        np.add.at(sums, inverse, delta)
        delta = (sums / counts[:, None])[inverse]
        rigid_seams = np.zeros(len(counts), dtype=bool)
        np.logical_or.at(rigid_seams, inverse, rigid_hand)
        delta[rigid_seams[inverse]] = du + df
        if fixed is not None:
            # If a face seam shares an index, that entire seam remains fixed.
            fixed_seams = np.zeros(len(counts), dtype=bool)
            np.logical_or.at(fixed_seams, inverse, fixed)
            delta[fixed_seams[inverse]] = 0.
    return delta


def _translated_hand_metadata(fit, grips, side, elbow_delta, wrist_delta):
    for part, delta in (('elbow', elbow_delta), ('wrist', wrist_delta), ('tip', wrist_delta)):
        key = part + '.' + side
        if key in fit.get('joints', {}):
            fit['joints'][key] = (np.array(fit['joints'][key]) + delta).tolist()
    model = ((fit.get('hands') or {}).get(side) or {}).get('model')
    if model:
        model['wrist'] = (np.array(model['wrist']) + wrist_delta).tolist()
        for finger in model.get('fingers', {}).values():
            for name in ('joints', 'centreline'):
                if name in finger:
                    finger[name] = (np.array(finger[name]) + wrist_delta).tolist()
    grip = grips[side]
    for name in ('B', 'wrist', 'middle_mcp'):
        if name in grip:
            grip[name] = (np.array(grip[name]) + wrist_delta).tolist()


def apply(rig, skin, upper_scale=1.8, fore_scale=1.8):
    """Extend one local fitted instance and return the measured variant report.

    The character must be unposed and have no action/NLA. Repeated applications
    are rejected, so a caller cannot accidentally compound the proportions.
    """
    import bpy
    from mathutils import Vector
    import skin_wrap

    upper_scale, fore_scale = _scale(upper_scale), _scale(fore_scale)
    if rig.get('reach_variant'):
        raise ValueError('This rig already has an arm-reach variant; load a fresh fitted character.')
    animation = rig.animation_data
    if animation and (animation.action or len(animation.nla_tracks)):
        raise ValueError('Apply the reach variant before adding animation.')
    if any(np.max(abs(np.array(pb.matrix_basis) - np.eye(4))) > 1e-6 for pb in rig.pose.bones):
        raise ValueError('Apply the reach variant to the unposed fitted character.')
    if rig.type != 'ARMATURE' or skin.type != 'MESH':
        raise ValueError('The reach variant needs its fitted armature and skin.')
    fit = copy.deepcopy(json.loads(rig['skin_fit']))
    grips = copy.deepcopy(json.loads(rig['hand_grip']))
    data = rig.data
    original_bones = {bone.name: (bone.head_local.copy(), bone.tail_local.copy()) for bone in data.bones}
    inverse = rig.matrix_world.inverted()
    co = np.array([inverse @ (skin.matrix_world @ vertex.co) for vertex in skin.data.vertices])
    weights = {side: np.zeros((len(co), 3)) for side in ('left', 'right')}
    head_groups = {vg.index for vg in skin.vertex_groups if P + 'Head' == vg.name}
    face = np.zeros(len(co), dtype=bool)
    for vertex in skin.data.vertices:
        face[vertex.index] = sum(g.weight for g in vertex.groups if g.group in head_groups) > .05
        for side in ('left', 'right'):
            title = side.title()
            for membership in vertex.groups:
                name = skin.vertex_groups[membership.group].name
                if name == P + title + 'Arm':
                    weights[side][vertex.index, 0] += membership.weight
                elif name == P + title + 'ForeArm':
                    weights[side][vertex.index, 1] += membership.weight
                elif name.startswith(P + title + 'Hand'):
                    weights[side][vertex.index, 2] += membership.weight
    height = float(fit['measured']['height'])
    seams = skin_wrap._coincident_groups(co, skin_wrap.COINCIDENT * height)
    delta = np.zeros_like(co)
    movements, reports = {}, {}
    for side in ('left', 'right'):
        title = side.title()
        arm, fore, hand = (data.bones[P + title + name] for name in ('Arm', 'ForeArm', 'Hand'))
        shoulder, elbow, wrist = (np.array(bone.head_local) for bone in (arm, fore, hand))
        du, df = (elbow - shoulder) * (upper_scale - 1.), (wrist - elbow) * (fore_scale - 1.)
        dw = du + df
        delta += displacement(co, shoulder, elbow, wrist, *weights[side].T,
                              upper_scale, fore_scale, fixed=face, seams=seams)
        movements[side] = (du, df, dw)
        reports[side] = {'upper_before_m': float(arm.length), 'fore_before_m': float(fore.length),
                         'upper_after_m': float(arm.length * upper_scale), 'fore_after_m': float(fore.length * fore_scale),
                         'wrist_translation_m': dw.tolist()}
        _translated_hand_metadata(fit, grips, side, du, dw)
    delta[face] = 0.
    mesh_inverse = skin.matrix_world.inverted() @ rig.matrix_world
    new_co = np.array([mesh_inverse @ Vector(point) for point in co + delta])
    local_delta = new_co - np.array([v.co[:] for v in skin.data.vertices])
    if skin.data.users > 1:
        skin.data = skin.data.copy()
    if skin.data.shape_keys:
        for key in skin.data.shape_keys.key_blocks:
            values = np.array([point.co[:] for point in key.data]) + local_delta
            key.data.foreach_set('co', values.reshape(-1))
    else:
        skin.data.vertices.foreach_set('co', new_co.reshape(-1))
    skin.data.update()
    bpy.context.view_layer.objects.active = rig
    for obj in bpy.context.view_layer.objects:
        obj.select_set(obj == rig)
    bpy.ops.object.mode_set(mode='EDIT')
    try:
        connected = {bone.name: bone.use_connect for bone in rig.data.edit_bones}
        for bone in rig.data.edit_bones:
            bone.use_connect = False
        for side in ('left', 'right'):
            title, (du, df, dw) = side.title(), movements[side]
            edit = rig.data.edit_bones
            upper = edit[P + title + 'Arm']
            fore = edit[P + title + 'ForeArm']
            upper.head, upper.tail = original_bones[upper.name][0], original_bones[upper.name][1] + Vector(du)
            fore.head, fore.tail = original_bones[fore.name][0] + Vector(du), original_bones[fore.name][1] + Vector(dw)
            for bone in edit:
                if bone.name.startswith(P + title + 'Hand'):
                    bone.head = original_bones[bone.name][0] + Vector(dw)
                    bone.tail = original_bones[bone.name][1] + Vector(dw)
        for bone in rig.data.edit_bones:
            bone.use_connect = connected[bone.name]
    finally:
        bpy.ops.object.mode_set(mode='OBJECT')
    rig['skin_fit'] = json.dumps(fit)
    rig['hand_grip'] = json.dumps(grips)
    bpy.context.view_layer.update()
    report = {'kind': 'fitness arm reach', 'label': 'Fitness proportions', 'applied': True,
              'upper_scale': upper_scale, 'fore_scale': fore_scale,
              'source_model_unchanged': True, 'face_max_displacement_mm': float(np.linalg.norm(delta[face], axis=1).max(initial=0) * 1000),
              'hand_scale': 1., 'per_frame_stretch': False, 'sides': reports}
    rig['reach_variant'] = json.dumps(report)
    return report
