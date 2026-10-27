"""Keep the leg-press rider on the fixed support while only the legs move."""
import json
import bpy
from mathutils import Vector
import numpy as np

import mpfb_cage as cage
from sled_motion import summarize

SIDES = (('left', 'Left'), ('right', 'Right'))


def fit(rig, skin, first, last, height, phase_frames):
    """Fit a source-aligned rail from one body pose, without torso searches.

    The source supplies the rail direction and lower knee bend. A declared
    175-degree soft extension is approached on the same eased phase clock as
    the source comparisons. Solving distance from knee bend avoids the sudden
    near-extension knee motion caused by clamping a linear rail trajectory.
    """
    from exercise_supports import _leg_to, _mask, _joint_mid, _neutral_neck, _head_support_range, _head_on_plane
    from keypose_clip import timeline
    from pullup_range import included
    scene, pose = bpy.context.scene, rig.pose.bones
    if (not isinstance(phase_frames, dict) or phase_frames.get('start') != first
            or type(phase_frames.get('end')) is not int or not first < phase_frames['end'] <= last):
        raise ValueError('A fixed-body leg press requires its recorded source phases.')
    progress = np.asarray(timeline(scene.render.fps / scene.render.fps_base, 'Leg Press (seated machine)'))
    if len(progress) != last - first + 1 or abs(progress[phase_frames['end'] - first] - 1.) > 1e-8:
        raise ValueError('The leg-press knee clock must match the generated clip and recorded phases.')
    ankles = []
    for frame in range(first, last + 1):
        scene.frame_set(frame)
        ankles.append([np.array(pose[cage.P + title + 'Foot'].head) for _, title in SIDES])
    ankles = np.asarray(ankles)
    middle = ankles.mean(1)
    centre = middle.mean(0)
    _, singular, directions = np.linalg.svd(middle - centre)
    if singular[0] < .01 * height:
        raise ValueError('The source leg press has no readable rail travel.')
    axis = directions[0]
    if axis[2] < 0:
        axis = -axis
    source_distance = (middle - centre) @ axis
    rail = centre + source_distance[:, None] * axis
    source_deviation = float(np.linalg.norm(middle - rail, axis=1).max())
    # The two source poses can disagree on depth-derived stance width. One
    # rigid footplate needs a single span; use the median observed offsets,
    # not whichever noisy first pose happens to be selected.
    offsets = np.median(ankles - middle[:, None, :], axis=0)
    offsets -= offsets.mean(0)
    offsets -= (offsets @ axis)[:, None] * axis
    scene.frame_set(phase_frames['end'])
    source_bottom, poles = {}, {}
    for side, title in SIDES:
        hip, knee, ankle = [np.array(pose[cage.P + title + part].head) for part in ('UpLeg', 'Leg', 'Foot')]
        source_bottom[side] = included(hip, knee, ankle)
        poles[title] = Vector(knee - hip)
    bottom = float(np.mean(list(source_bottom.values())))
    if not 60. <= bottom <= 110.:
        raise ValueError('The source lower leg-press position is outside the supported 60–110 degree knee range.')
    scene.frame_set(first)
    _neutral_neck(rig)
    body_axis = np.array(_joint_mid(rig, 'Arm') - _joint_mid(rig, 'UpLeg'))
    body_axis[0] = 0.
    body_axis /= np.linalg.norm(body_axis)
    normal = np.array([0., body_axis[2], -body_axis[1]])
    if normal[2] < 0:
        normal = -normal
    import skin_wrap
    initial = skin_wrap.evaluated(skin)[0]
    back_plane = float((initial[_mask(skin, ['Spine1', 'Spine2'])] @ normal).min())
    attainable = _head_support_range(rig, skin, normal)
    lower, upper = max(back_plane, attainable[0]), min(back_plane + .08 * height, attainable[1])
    if lower > upper:
        raise ValueError('The fixed leg-press headrest cannot fit the mascot within the existing padding/neck limits.')
    head_plane = (lower + upper) / 2
    head_adjustment = abs(_head_on_plane(rig, skin, head_plane, normal))
    base = {bone.name: bone.matrix_basis.copy() for bone in pose}
    hips = {side: np.array(pose[cage.P + title + 'UpLeg'].head) for side, title in SIDES}
    lengths = {side: tuple(pose[cage.P + title + part].length for part in ('UpLeg', 'Leg')) for side, title in SIDES}
    sole_turn = Vector((0., 0., -1.)).rotation_difference(Vector(axis)).to_matrix()
    foot_turns = {title: sole_turn @ pose[cage.P + title + 'Foot'].bone.matrix_local.to_3x3() for _, title in SIDES}
    previous, worst_residual, adjustments = {}, 0., []
    for index, frame in enumerate(range(first, last + 1)):
        scene.frame_set(frame)
        for name, basis in base.items():
            pose[name].matrix_basis = basis
        bpy.context.view_layer.update()
        angle = 175. * (1. - progress[index]) + bottom * progress[index]
        parameters = []
        lo, hi = -float('inf'), float('inf')
        for side_index, (side, _) in enumerate(SIDES):
            up, low = lengths[side]
            distance_squared = up * up + low * low - 2 * up * low * np.cos(np.radians(angle))
            delta = centre + offsets[side_index] - hips[side]
            projection = float(delta @ axis)
            discriminant = projection * projection - float(delta @ delta) + distance_squared
            if discriminant < -1e-8:
                raise ValueError('The fixed rail cannot reach the source knee bend without changing the body or limb lengths.')
            parameters.append(-projection + np.sqrt(max(0., discriminant)))
            maximum = up * up + low * low - 2 * up * low * np.cos(np.radians(176.))
            span = np.sqrt(max(0., projection * projection - float(delta @ delta) + maximum))
            lo, hi = max(lo, -projection - span), min(hi, -projection + span)
        if lo > hi:
            raise ValueError('Both leg-press feet cannot share the same rigid rail.')
        parameter = float(np.clip(np.mean(parameters), lo, hi))
        targets = centre + parameter * axis + offsets
        adjustment = float(np.linalg.norm(ankles[index] - targets, axis=1).max())
        if adjustment > .05 * height:
            raise ValueError(f'The fixed leg-press trajectory exceeds the existing 5%-height source path correction limit '
                             f'at frame {frame}: {adjustment * 1000:.2f} mm.')
        adjustments.append(adjustment)
        for side_index, (_, title) in enumerate(SIDES):
            worst_residual = max(worst_residual, _leg_to(rig, title, Vector(targets[side_index]), foot_turns[title], poles[title]))
            pose[cage.P + title + 'ToeBase'].matrix_basis.identity()
        if worst_residual > .002:
            raise ValueError('The fixed leg-press rail requires limb stretch.')
        cage._key(rig, previous, frame, set(base))
        pose[cage.P + 'Hips'].keyframe_insert('location', frame=frame)
    scene.frame_set(first)
    pelvis = _mask(skin, ['Hips'])
    raise_m = max(0., .20 * height - float(skin_wrap.evaluated(skin)[0][pelvis, 2].min()))
    for frame in range(first, last + 1):
        scene.frame_set(frame)
        cage._move_root(rig, Vector((0., 0., raise_m)))
        pose[cage.P + 'Hips'].keyframe_insert('location', frame=frame)
    scene.frame_set(first)
    correction = {'applied': True, 'source_rail_deviation_mm': source_deviation * 1000,
        'max_ankle_path_adjustment_mm': max(adjustments) * 1000, 'ankle_path_limit_mm': .05 * height * 1000,
        'max_ankle_residual_mm': worst_residual * 1000, 'max_seat_settling_shift_mm': 0.,
        'max_body_pitch_adjustment_deg': 0., 'max_head_pitch_adjustment_deg': head_adjustment,
        'backrest': {'normal': normal.tolist(), 'plane': back_plane + raise_m * normal[2],
            'head_plane': head_plane + raise_m * normal[2], 'body_axis': body_axis.tolist()},
        'source_bottom_knees_deg': source_bottom, 'soft_extension_deg': 175.,
        'foot_span_m': float(np.linalg.norm(offsets[1] - offsets[0])),
        'foot_span_source': 'One fixed median source stance on the rigid footplate; source depth remains inferred.',
        'foot_orientation': 'rest plantar soles aligned to the rigid footplate normal',
        'source': 'reference-inferred rail and source lower knee bend; declared soft extension and eased source phase timing; not calibrated'}
    return correction, {'applied': True, 'max_ankle_residual_mm': worst_residual * 1000, 'world_raise_mm': raise_m * 1000,
        'method': 'One fixed supported body and direct leg-only rail IK; no per-frame torso/head search.'}




def measure(rig, first, last, phase_frames):
    scene, pose, world = bpy.context.scene, rig.pose.bones, rig.matrix_world
    samples = []
    for frame in range(first, last + 1):
        scene.frame_set(frame)
        points = {name: {side: list(world @ pose[cage.P + title + bone].head) for side, title in SIDES}
                  for name, bone in (('hip', 'UpLeg'), ('knee', 'Leg'), ('ankle', 'Foot'),
                                     ('shoulder', 'Arm'), ('wrist', 'Hand'))}
        points['head'] = list(world @ pose[cage.P + 'Head'].head)
        samples.append({'frame': frame, 'points_m': points})
    scene.frame_set(first)
    return summarize(samples, json.loads(rig['skin_fit'])['measured']['height'], phase_frames)
