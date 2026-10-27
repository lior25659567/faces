"""Declared upright base for standing strength profiles, independent of Blender.

Monocular limb depth does not establish a crouched stance. The body and soles
use the fitted rest posture; each exercise's arm solver remains responsible
for its movement and the actual skin still has to pass floor/contact checks.
"""
import numpy as np

BODY = {'Hips', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head', 'LeftShoulder', 'RightShoulder'} | {
    side + part for side in ('Left', 'Right') for part in ('UpLeg', 'Leg', 'Foot', 'ToeBase')}


def upright(keys):
    """Return copies with a neutral stationary body, never mutate photo poses."""
    return [{name: np.eye(3) if name in BODY else np.array(value, copy=True)
             for name, value in key.items()} for key in keys]


def correct(rig, skin, first, last):
    """Align the actual torso vertically while keeping fixed soles and lengths.

    The mascot's fitted rest skeleton has a real forward trunk offset. Identity
    source rotations preserve that offset, so they alone are not an upright
    exercise pose. Correct the measured shoulder/hip line on the generated rig.
    """
    import bpy
    from mathutils import Vector
    import mpfb_cage as cage
    from exercise_supports import _leg_to
    import skin_wrap
    scene, pose = bpy.context.scene, rig.pose.bones
    previous, readings = {}, []
    # Authored isolation arms are relative to the torso and travel rigidly
    # with its correction. Preserve only the head's world gaze; bar/press
    # constraints rebuild their own absolute arm goals afterward.
    restored = ('Neck', 'Head')
    keyed = {cage.P + name for name in restored} | {cage.P + 'Hips'} | {
        cage.P + side + part for side in ('Left', 'Right') for part in ('UpLeg', 'Leg', 'Foot')}
    for frame in range(first, last + 1):
        scene.frame_set(frame)
        hips = (pose[cage.P + 'LeftUpLeg'].head + pose[cage.P + 'RightUpLeg'].head) / 2
        shoulders = (pose[cage.P + 'LeftArm'].head + pose[cage.P + 'RightArm'].head) / 2
        direction = shoulders - hips
        if direction.length < 1e-6:
            raise ValueError('An upright standing correction needs a measured torso.')
        before = np.degrees(direction.angle(Vector((0., 0., 1.))))
        turns = {name: pose[cage.P + name].matrix.to_3x3().copy() for name in restored}
        feet = {side: pose[cage.P + side + 'Foot'].matrix.copy() for side in ('Left', 'Right')}
        root = pose[cage.P + 'Hips']
        matrix = (direction.rotation_difference(Vector((0., 0., 1.))).to_matrix() @ root.matrix.to_3x3()).to_4x4()
        matrix.translation = root.matrix.translation
        root.matrix = matrix
        bpy.context.view_layer.update()
        moved = (pose[cage.P + 'LeftUpLeg'].head + pose[cage.P + 'RightUpLeg'].head) / 2
        cage._move_root(rig, hips - moved)
        # Stack the pelvis above the fixed soles. Keeping the rest hip's
        # forward offset would leave a straight-legged backward lean even
        # after the upper torso itself had been made vertical.
        ankle_mid = sum((goal.translation for goal in feet.values()), Vector()) / len(feet)
        target = Vector((ankle_mid.x, ankle_mid.y, hips.z))
        shift = target - hips
        heights = []
        for side, goal in feet.items():
            upper, lower = [pose[cage.P + side + name] for name in ('UpLeg', 'Leg')]
            moved_hip = upper.head + shift
            horizontal = Vector((goal.translation.x - moved_hip.x, goal.translation.y - moved_hip.y, 0.))
            reach = upper.length + lower.length - .0005
            if horizontal.length >= reach:
                raise ValueError('The fixed standing stance exceeds the fitted leg reach.')
            heights.append(goal.translation.z + np.sqrt(reach * reach - horizontal.length_squared)
                           - (upper.head.z - hips.z))
        target.z = min(heights)
        cage._move_root(rig, target - hips)
        for name in restored:
            bone = pose[cage.P + name]
            matrix = turns[name].to_4x4()
            matrix.translation = bone.head
            bone.matrix = matrix
            bpy.context.view_layer.update()
        residual = max(_leg_to(rig, side, goal.translation, goal.to_3x3()) for side, goal in feet.items())
        hips = (pose[cage.P + 'LeftUpLeg'].head + pose[cage.P + 'RightUpLeg'].head) / 2
        shoulders = (pose[cage.P + 'LeftArm'].head + pose[cage.P + 'RightArm'].head) / 2
        after = np.degrees((shoulders - hips).angle(Vector((0., 0., 1.))))
        readings.append({'frame': frame, 'before_lean_deg': float(before), 'after_lean_deg': float(after),
            'ankle_residual_mm': float(residual * 1000),
            'body_floor_min_mm': float(skin_wrap.evaluated(skin)[0][:, 2].min() * 1000)})
        cage._key(rig, previous, frame, keyed)
        root.keyframe_insert('location', frame=frame)
    scene.frame_set(first)
    return {'applied': True, 'frames': len(readings), 'per_frame': readings,
        'maximum_before_lean_deg': max(r['before_lean_deg'] for r in readings),
        'maximum_after_lean_deg': max(r['after_lean_deg'] for r in readings),
        'max_ankle_residual_mm': max(r['ankle_residual_mm'] for r in readings),
        'method': 'Actual fitted shoulder/hip line aligned to vertical and pelvis stacked over fixed soles; fixed ankle targets and limb lengths, authored body-relative arm paths and world head gaze retained; not calibrated photo depth.'}
