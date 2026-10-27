"""Align the actual seated press trunk while preserving its floor marks."""
import math
import json

import bpy
from mathutils import Matrix, Vector

import mpfb_cage as cage


def correct(rig, skin, first, last):
    """A neutral seated trunk is a declared pose, not inferred photo depth.

    The fitted rest cage itself leans, so identity pose turns are insufficient.
    Correct its measured hip-mid to shoulder-mid axis, solve the fixed-length seated legs
    back to their original ankles, then restore world arm/head orientations.
    """
    import skin_wrap
    from exercise_supports import _leg_to, _mask
    scene, pose = bpy.context.scene, rig.pose.bones
    scene.frame_set(first)
    axis = ((pose[cage.P + 'LeftArm'].head + pose[cage.P + 'RightArm'].head)
            - (pose[cage.P + 'LeftUpLeg'].head + pose[cage.P + 'RightUpLeg'].head)) / 2
    if axis.z <= 0 or abs(axis.x) > .02:
        raise ValueError('The seated press needs a measurable upright, symmetric trunk axis.')
    angle = math.atan2(axis.y, axis.z)
    if abs(math.degrees(angle)) > 45.:
        raise ValueError('The seated torso cannot be corrected within the declared 45-degree fitted-rest bound.')
    twist = Matrix.Rotation(angle, 3, 'X')
    height = json.loads(rig['skin_fit'])['measured']['height']
    # Preserve the observed floor marks in X/Y while fitting the actual heel
    # and toe patches flat to Z=0. A single pointed toe is not a planted sole.
    base = {bone.name: bone.matrix_basis.copy() for bone in pose}
    floor_marks, floor_adjustments = {}, {}
    for side in ('Left', 'Right'):
        foot = pose[cage.P + side + 'Foot']
        matrix = foot.bone.matrix_local.to_3x3().to_4x4()
        matrix.translation = foot.head
        foot.matrix = matrix
        bpy.context.view_layer.update()
        toe = pose[cage.P + side + 'ToeBase']
        matrix = toe.bone.matrix_local.to_3x3().to_4x4(); matrix.translation = toe.head
        toe.matrix = matrix
        bpy.context.view_layer.update()
        points = skin_wrap.evaluated(skin)[0]
        dz = -float(points[_mask(skin, [side + 'Foot', side + 'ToeBase']), 2].min())
        if abs(dz) > height * .05:
            raise ValueError('A flat seated press sole needs more than 5% of character height in ankle correction.')
        goal = foot.matrix.copy(); goal.translation.z += dz
        floor_marks[side] = goal
        floor_adjustments[side.lower()] = dz * 1000
    for bone in pose:
        bone.matrix_basis = base[bone.name]
    bpy.context.view_layer.update()
    names = {cage.P + 'Hips', cage.P + 'Neck', cage.P + 'Head'} | {
        cage.P + side + part for side in ('Left', 'Right') for part in ('UpLeg', 'Leg', 'Foot', 'ToeBase', 'Arm', 'ForeArm', 'Hand')}
    previous, worst = {}, 0.
    for frame in range(first, last + 1):
        scene.frame_set(frame)
        feet = floor_marks
        poles = {side: pose[cage.P + side + 'Leg'].head - pose[cage.P + side + 'UpLeg'].head for side in ('Left', 'Right')}
        rotations = {name: pose[cage.P + name].matrix.to_3x3().copy() for name in
            ('Neck', 'Head', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightArm', 'RightForeArm', 'RightHand')}
        root = pose[cage.P + 'Hips']
        matrix = (twist @ root.matrix.to_3x3()).to_4x4()
        matrix.translation = root.head
        root.matrix = matrix
        bpy.context.view_layer.update()
        for side in ('Left', 'Right'):
            worst = max(worst, _leg_to(rig, side, feet[side].translation, feet[side].to_3x3(), poles[side]))
            toe = pose[cage.P + side + 'ToeBase']
            matrix = toe.bone.matrix_local.to_3x3().to_4x4(); matrix.translation = toe.head
            toe.matrix = matrix
            bpy.context.view_layer.update()
        for name, rotation in rotations.items():
            bone = pose[cage.P + name]
            matrix = rotation.to_4x4(); matrix.translation = bone.head
            bone.matrix = matrix
            bpy.context.view_layer.update()
        cage._key(rig, previous, frame, names)
        root.keyframe_insert('location', frame=frame)
    scene.frame_set(first)
    if worst > .002:
        raise ValueError('The seated press cannot keep its observed floor marks without stretching the legs.')
    return {'applied': True, 'fitted_trunk_pitch_correction_deg': math.degrees(angle),
        'max_ankle_residual_mm': worst * 1000,
        'floor_ankle_adjustments_mm': floor_adjustments,
        'method': 'Actual hip-mid to shoulder-mid axis made upright; observed floor marks and fixed leg lengths retained with flat heel/toe patches; photo depth is uncalibrated'}
