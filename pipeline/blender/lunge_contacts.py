"""Fit one observed forward lunge to actual named heel/toe skin supports.

The camera is frontal and uncalibrated. Root travel and the hidden joint depths
are a declared constrained fit, not recovered motion-capture coordinates.
"""
import json
import math

import bpy
import numpy as np
from mathutils import Matrix, Vector

import mpfb_cage as cage
import skin_wrap
from lunge_motion import SIDES, POINTS, samples_from, timeline, foot_pitch, contact_fit, knee_position, root_height, eased, knee_clock, root_from_knee, ankle_from_knee, carrying_arms
from lunge_range import summarize, included


def _mask(skin, side):
    groups = {group.index for group in skin.vertex_groups
              if group.name in (cage.P + side.title() + 'Foot', cage.P + side.title() + 'ToeBase')}
    return np.array([sum(group.weight for group in vertex.groups if group.group in groups) > .5
                     for vertex in skin.data.vertices])


def _rest_geometry(rig, skin):
    previous = rig.data.pose_position
    rig.data.pose_position = 'REST'
    bpy.context.view_layer.update()
    try:
        rest = skin_wrap.evaluated(skin)[0]
    finally:
        rig.data.pose_position = previous
        bpy.context.view_layer.update()
    # Fitted Studio rigs use the canonical +Y forward frame.
    transform = rig.matrix_world.inverted()
    rest = np.array([transform @ Vector(point) for point in rest])
    feet, patches, offsets = {}, {}, {}
    for side in SIDES:
        foot = rig.pose.bones[cage.P + side.title() + 'Foot']
        selected = _mask(skin, side)
        if selected.sum() < 12:
            raise ValueError('Both lunge feet need readable actual sole geometry.')
        feet[side] = selected
        sole = selected & (rest[:, 2] <= rest[selected, 2].min() + .002)
        low, high = rest[sole, 1].min(), rest[sole, 1].max()
        if high - low <= .02:
            raise ValueError('The lunge needs distinct fitted heel and toe contact regions.')
        for point, chosen in (('heel', rest[:, 1] <= low + (high - low) * .08),
                              ('toe', rest[:, 1] >= high - (high - low) * .08)):
            patch = sole & chosen
            if patch.sum() < 3:
                raise ValueError('A lunge heel/toe contact patch has too few measured vertices.')
            name = side + '.' + point
            patches[name] = patch
            offsets[name] = rest[patch].mean(axis=0) - np.array(foot.bone.head_local)
    return rest, feet, patches, offsets


def _rotate_x(degrees):
    a = math.radians(float(degrees))
    c, s = math.cos(a), math.sin(a)
    return np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])


def _leg(rig, side, ankle, foot_change, lengths):
    title = side.title()
    up, low, foot, toe = [rig.pose.bones[cage.P + title + part] for part in ('UpLeg', 'Leg', 'Foot', 'ToeBase')]
    hip = np.array(up.head)
    knee = knee_position(hip, ankle, *lengths[side])
    # Joint locations from a frontal lift do not establish axial twist. Build
    # a stable anatomical leg frame, then retain the solved distal targets.
    for bone, target in ((up, knee), (low, ankle)):
        reference = (bone.parent.matrix.to_3x3() @ bone.parent.bone.matrix_local.to_3x3().inverted()
                     @ bone.bone.matrix_local.to_3x3())
        axis = (Vector(target) - bone.head).normalized()
        turn = (reference @ Vector((0., 1., 0.))).rotation_difference(axis).to_matrix() @ reference
        matrix = turn.to_4x4()
        matrix.translation = bone.head
        bone.matrix = matrix
        bpy.context.view_layer.update()
    matrix = (Matrix(foot_change.tolist()) @ foot.bone.matrix_local.to_3x3()).to_4x4()
    matrix.translation = foot.head
    foot.matrix = matrix
    bpy.context.view_layer.update()
    matrix = (Matrix(foot_change.tolist()) @ toe.bone.matrix_local.to_3x3()).to_4x4()
    matrix.translation = toe.head
    toe.matrix = matrix
    bpy.context.view_layer.update()
    error = float(np.linalg.norm(np.array(foot.head) - ankle))
    if error > .002:
        raise ValueError('The observed lunge support could not be reached without stretching the leg.')
    return error


def _carry(rig, grip_ratio, elbow_ratio):
    shoulders = {side: np.array(rig.pose.bones[cage.P + side.title() + 'Arm'].head) for side in SIDES}
    lengths = {side: tuple(rig.pose.bones[cage.P + side.title() + part].length
        for part in ('Arm', 'ForeArm')) for side in SIDES}
    goals = carrying_arms(shoulders, lengths, grip_ratio, elbow_ratio)
    for side in SIDES:
        for part, target in (('Arm', goals[side]['elbow']), ('ForeArm', goals[side]['wrist'])):
            bone = rig.pose.bones[cage.P + side.title() + part]
            reference = (bone.parent.matrix.to_3x3() @ bone.parent.bone.matrix_local.to_3x3().inverted()
                         @ bone.bone.matrix_local.to_3x3())
            axis = (Vector(target) - bone.head).normalized()
            turn = (reference @ Vector((0., 1., 0.))).rotation_difference(axis).to_matrix() @ reference
            matrix = turn.to_4x4()
            matrix.translation = bone.head
            bone.matrix = matrix
            bpy.context.view_layer.update()


def apply(rig, skin, first, last, reference, phase_frames):
    samples = samples_from(reference)
    scene, pose = bpy.context.scene, rig.pose.bones
    fps = scene.render.fps / scene.render.fps_base
    source_times = [sample['timeSeconds'] for sample in samples]
    duration = source_times[-1] - source_times[0]
    times = [source_times[0] + min((frame - first) / fps, duration) for frame in range(first, last + 1)]
    contacts = timeline(samples, times)
    pitch = {side: foot_pitch(samples, times, side) for side in SIDES}
    height = json.loads(rig['skin_fit'])['measured']['height']
    source_pixels = samples[0]['pixels']
    source_shoulder = np.linalg.norm(np.asarray(source_pixels['left_shoulder']) - source_pixels['right_shoulder'])
    carry_grip = float(np.linalg.norm(np.asarray(source_pixels['left_wrist']) - source_pixels['right_wrist']) / source_shoulder)
    carry_elbow = float(np.linalg.norm(np.asarray(source_pixels['left_elbow']) - source_pixels['right_elbow']) / source_shoulder)
    rest, feet, patches, offsets = _rest_geometry(rig, skin)
    lengths = {side: tuple(pose[cage.P + side.title() + part].length for part in ('UpLeg', 'Leg')) for side in SIDES}
    hip_width = abs(float(rig.data.bones[cage.P + 'RightUpLeg'].head_local.x
                          - rig.data.bones[cage.P + 'LeftUpLeg'].head_local.x))
    lateral = {side: [] for side in SIDES}
    for sample in samples:
        pixels = sample['pixels']
        centre = (pixels['left_hip'][0] + pixels['right_hip'][0]) / 2
        width = abs(pixels['left_hip'][0] - pixels['right_hip'][0])
        if width <= 5.:
            raise ValueError('The frontal lunge needs readable hip width for lateral foot placement.')
        for side in SIDES:
            lateral[side].append(-(pixels[side + '_ankle'][0] - centre) / width * hip_width)
    lateral_origin = float(np.mean([lateral[side][0] for side in SIDES]))
    lateral = {side: np.interp(times, source_times, np.asarray(values) - lateral_origin)
               for side, values in lateral.items()}
    base, virtual, hips, ankles, hip_offsets = [], [], [], [], []
    for index, frame in enumerate(range(first, last + 1)):
        scene.frame_set(frame)
        base.append({bone.name: bone.matrix_basis.copy() for bone in pose})
        hip = np.mean([np.array(pose[cage.P + side.title() + 'UpLeg'].head) for side in SIDES], axis=0)
        hips.append(hip)
        ankle = {side: np.array(pose[cage.P + side.title() + 'Foot'].head) for side in SIDES}
        ankles.append(ankle)
        hip_offsets.append({side: np.array(pose[cage.P + side.title() + 'UpLeg'].head) - hip for side in SIDES})
        virtual.append({name: ankle[name.split('.')[0]] + _rotate_x(pitch[name.split('.')[0]][index]) @ offset
                        for name, offset in offsets.items()})
        for name in virtual[-1]:
            side = name.split('.')[0]
            virtual[-1][name][0] = hip[0] + lateral[side][index] + offsets[name][0]
    # The visible first pose is a parallel two-foot stand. Monocular inferred
    # foot depth must not put one ankle a stride behind the other. Keep its
    # observed frontal width and every subsequent relative step displacement.
    start_y = float(np.mean([ankles[0][side][1] for side in SIDES]))
    stance_bias = {}
    for side in SIDES:
        stance_bias[side] = np.array([0., start_y - ankles[0][side][1], 0.])
    for points in virtual:
        for name in points:
            points[name] += stance_bias[name.split('.')[0]]
    lead = next(side for row in contacts for side in SIDES if row['foot_states'][side] == 'airborne')
    shifts, anchors = contact_fit(virtual, [row['contacts'] for row in contacts], preferred_side=lead)
    raw_roots = np.asarray(hips) + shifts
    roots = raw_roots.copy()
    foot_goals = []
    for index, row in enumerate(contacts):
        foot_goals.append({side: np.mean([anchors[index][point] - _rotate_x(pitch[side][index]) @ offsets[point]
                            for point in row['contacts'] if point.startswith(side + '.')], axis=0)
                           for side in SIDES if any(point.startswith(side + '.') for point in row['contacts'])})
    rear = next(side for side in SIDES if side != lead)
    land_index = next(i for i in range(1, len(times)) if lead in foot_goals[i] and lead not in foot_goals[i - 1])
    release_index = max(i for i in range(len(times)) if rear in foot_goals[i])
    landing_time = next(sample['timeSeconds'] for i, sample in enumerate(samples[1:], 1)
        if sample['footContacts'][lead] in ('flat', 'heel', 'toe')
        and samples[i - 1]['footContacts'][lead] in ('airborne', 'uncertain'))
    release_time = max(sample['timeSeconds'] for sample in samples
        if sample['footContacts'][rear] in ('flat', 'heel', 'toe'))
    middle_time = next(sample['timeSeconds'] for sample in samples
        if sample.get('phase') == 'middle' or (sample.get('source') or {}).get('phase') == 'middle')
    initial_y = float(np.mean([goal[1] for goal in foot_goals[0].values()]))
    middle_goals = foot_goals[phase_frames['middle'] - first]
    middle_y = float(np.mean([goal[1] for goal in middle_goals.values()]))
    final_y = float(foot_goals[-1][lead][1])
    if not initial_y < middle_y < final_y:
        raise ValueError('The observed planted points do not establish a forward lunge path.')
    for index, time in enumerate(times):
        roots[index, 0] = hips[0][0]
        roots[index, 1] = (initial_y + (middle_y - initial_y) * eased(time, times[0], middle_time)
            if time <= middle_time else middle_y + (final_y - middle_y) * eased(time, middle_time, times[-1]))
    _, _, standing_height = root_height(roots[0, :2], hip_offsets[0], foot_goals[0], lengths, 10.)
    end_world = {name: np.asarray(value) for name, value in samples[-1]['world'].items()}
    from keypose_clip import character_frame
    character_rows, _ = character_frame({name: np.asarray(value) for name, value in samples[0]['world'].items()})
    human_leg = np.mean([np.linalg.norm(np.asarray(sample['world'][side + '_hip']) - sample['world'][side + '_knee'])
        + np.linalg.norm(np.asarray(sample['world'][side + '_knee']) - sample['world'][side + '_ankle'])
        for sample in samples for side in SIDES])
    model_scale = float(np.mean([sum(value) for value in lengths.values()]) / human_leg)
    end_separation = -float((character_rows @ (end_world[rear + '_ankle'] - end_world[lead + '_ankle']))[1]) * model_scale
    recovery_end = foot_goals[-1][lead].copy()
    recovery_end[0] = hips[0][0] + lateral[rear][-1]
    recovery_end[1] += end_separation
    previous, last_knees = {}, {}
    release_angle = None
    residuals, projections = [], []
    names = {cage.P + 'Hips'} | {cage.P + side.title() + part for side in SIDES
        for part in ('UpLeg', 'Leg', 'Foot', 'ToeBase', 'Arm', 'ForeArm')}
    for index, frame in enumerate(range(first, last + 1)):
        scene.frame_set(frame)
        for bone in pose:
            bone.matrix_basis = base[index][bone.name]
        bpy.context.view_layer.update()
        goals = foot_goals[index]
        target_angle = knee_clock(times[index], times[0], middle_time, times[-1])
        lead_target = goals.get(lead, foot_goals[land_index][lead])
        lead_height = root_from_knee(roots[index, :2], hip_offsets[index][lead], lead_target, lengths[lead], target_angle)
        if times[index] < landing_time:
            rear_target = goals[rear]
            rear_angle = 175. - 25. * eased(times[index], times[0], landing_time)
            rear_height = root_from_knee(roots[index, :2], hip_offsets[index][rear], rear_target, lengths[rear], rear_angle)
            arrival = eased(times[index], times[0], landing_time)
            desired = rear_height * (1. - arrival) + lead_height * arrival
        else:
            desired = lead_height
        roots[index, 2], _, _ = root_height(roots[index, :2], hip_offsets[index], goals, lengths, desired)
        projections.append(abs(roots[index, 2] - desired))
        cage._move_root(rig, Vector(roots[index] - hips[index]))
        bpy.context.view_layer.update()
        for side in SIDES:
            change = _rotate_x(pitch[side][index])
            if side in goals:
                goal = goals[side].copy()
            else:
                if side == lead and times[index] <= landing_time:
                    fraction = eased(times[index], times[0], landing_time)
                    goal = foot_goals[0][lead] * (1. - fraction) + foot_goals[land_index][lead] * fraction
                    middle_swing = (times[0] + landing_time) / 2
                    landing_angle = knee_clock(landing_time, times[0], middle_time, times[-1])
                    angle = (175. - 40. * eased(times[index], times[0], middle_swing)
                        if times[index] <= middle_swing else 135. + (landing_angle - 135.) * eased(times[index], middle_swing, landing_time))
                elif side == rear and times[index] >= release_time:
                    if release_angle is None:
                        release_angle = last_knees[rear]
                    fraction = eased(times[index], release_time, times[-1])
                    goal = foot_goals[release_index][rear] * (1. - fraction) + recovery_end * fraction
                    swing_peak = next(sample['timeSeconds'] for sample in samples if sample['footContacts'][rear] == 'airborne')
                    angle = (release_angle + (110. - release_angle) * eased(times[index], release_time, swing_peak)
                        if times[index] <= swing_peak else 110. + 40. * eased(times[index], swing_peak, times[-1]))
                else:
                    raise ValueError('The source contains another unobserved step beyond the supported one-shot fit.')
                hip = np.array(pose[cage.P + side.title() + 'UpLeg'].head)
                goal[2] = ankle_from_knee(hip, goal[:2], lengths[side], angle)
                foot = pose[cage.P + side.title() + 'Foot']
                cloud = (rest[feet[side]] - np.array(foot.bone.head_local)) @ change.T
                clearance = .004 if contacts[index]['foot_states'][side] == 'airborne' else .001
                goal[2] = max(goal[2], clearance - float(cloud[:, 2].min()))
            residuals.append(_leg(rig, side, goal, change, lengths))
            title = side.title()
            last_knees[side] = included(np.array(pose[cage.P + title + 'UpLeg'].head),
                np.array(pose[cage.P + title + 'Leg'].head), np.array(pose[cage.P + title + 'Foot'].head))
        _carry(rig, carry_grip, carry_elbow)
        cage._key(rig, previous, frame, names)
        pose[cage.P + 'Hips'].keyframe_insert('location', frame=frame)
    scene.frame_set(first)
    readings = []
    for index, frame in enumerate(range(first, last + 1)):
        scene.frame_set(frame)
        points = skin_wrap.evaluated(skin)[0]
        world = rig.matrix_world
        readings.append({'frame': frame, 'time_s': times[index], 'foot_states': contacts[index]['foot_states'],
            'contact_points_m': {name: points[mask].mean(axis=0).tolist() for name, mask in patches.items()},
            'contact_anchors_m': {name: list(world @ Vector(point)) for name, point in anchors[index].items()},
            'contact_floor': {name: {'min_z_m': float(points[mask, 2].min()), 'max_z_m': float(points[mask, 2].max())}
                              for name, mask in patches.items()},
            'foot_min_z_m': {side: float(points[mask, 2].min()) for side, mask in feet.items()},
            'hips_m': {side: list(world @ pose[cage.P + side.title() + 'UpLeg'].head) for side in SIDES},
            'knees_m': {side: list(world @ pose[cage.P + side.title() + 'Leg'].head) for side in SIDES},
            'ankles_m': {side: list(world @ pose[cage.P + side.title() + 'Foot'].head) for side in SIDES},
            'leg_lengths_m': {side: list(value) for side, value in lengths.items()},
            'body_min_z_m': float(points[:, 2].min()),
            'root_m': list(world @ Vector(np.mean([np.array(pose[cage.P + side.title() + 'UpLeg'].head) for side in SIDES], axis=0)))})
    report = summarize(readings, height, phase_frames, source_times, fps)
    report['fit'] = {'max_ankle_residual_mm': max(residuals) * 1000,
        'carrying_arm_widths': {'grip_ratio': carry_grip, 'elbow_width_ratio': carry_elbow,
            'method': 'visible initial frontal wrist/elbow widths with fixed downward arm lengths; depth remains uncalibrated'},
        'max_source_root_adjustment_mm': float(np.linalg.norm(roots - raw_roots, axis=1).max()) * 1000,
        'max_root_height_projection_mm': max(projections) * 1000,
        'foot_pitch_policy': 'declared flat sole, 15-degree rear toe pivot and smooth knee-angle swing; inferred source depth is not calibrated',
        'root_policy': 'one global source-timed support path with visible hip-relative lateral stance, eased knee-angle descent/recovery and fixed observed heel/toe anchors',
        'complete_closed_cycle': False, 'final_standing_pose_invented': False}
    report['patch_indices'] = {name: np.where(mask)[0].tolist() for name, mask in patches.items()}
    rig['lunge_source_times'] = json.dumps(source_times)
    rig['lunge_phase_frames'] = json.dumps(phase_frames)
    rig['lunge_contact_samples'] = json.dumps([{'frame': row['frame'], 'time_s': row['time_s'],
        'foot_states': row['foot_states'], 'contact_anchors_m': row['contact_anchors_m']} for row in readings])
    rig['lunge_patch_indices'] = json.dumps(report['patch_indices'])
    scene.frame_set(first)
    return report
