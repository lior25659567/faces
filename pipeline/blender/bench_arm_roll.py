"""A stable humerus frame for the explicit bench trajectory, in Blender.

Photographs establish joint positions but do not establish the upper arm's
axial twist. Re-aiming an inherited source pose retains that unobserved twist,
even after the elbow target changes. Use one declared mirrored roll from the
clavicle's fitted rest frame instead; do not alter the solved distal chain.
"""
import math

import bpy
from mathutils import Matrix, Vector

import mpfb_cage as cage


def orient_upperarms(rig, roll_degrees=30.):
    """Set a fixed mirrored press roll, preserving elbow, wrist and hand poses.

    The roll is an authored mascot posture, not a measurement of photo depth.
    Distal matrices are restored so this cannot change the solved bar contact
    or move an elbow to make a deformation check pass.
    """
    if not math.isfinite(roll_degrees) or abs(roll_degrees) > 45.:
        raise ValueError('The bench humerus roll must be finite and within 45 degrees.')
    largest = 0.
    for side, sign in (('Left', 1.), ('Right', -1.)):
        arm, forearm, hand = [rig.pose.bones[cage.P + side + part] for part in ('Arm', 'ForeArm', 'Hand')]
        forearm_matrix, hand_matrix = forearm.matrix.copy(), hand.matrix.copy()
        origin, original = arm.matrix.translation.copy(), arm.matrix.to_quaternion()
        axis = arm.tail - arm.head
        if axis.length <= 1e-8:
            raise ValueError('A bench humerus needs a finite nonzero upper arm.')
        axis.normalize()
        clavicle = arm.parent
        reference = (clavicle.matrix.to_3x3() @ clavicle.bone.matrix_local.to_3x3().inverted()
                     @ arm.bone.matrix_local.to_3x3())
        turn = ((reference @ Vector((0., 1., 0.))).rotation_difference(axis).to_matrix() @ reference)
        turn = Matrix.Rotation(sign * math.radians(roll_degrees), 3, axis) @ turn
        matrix = turn.to_4x4()
        matrix.translation = origin
        arm.matrix = matrix
        bpy.context.view_layer.update()
        forearm.matrix = forearm_matrix
        bpy.context.view_layer.update()
        hand.matrix = hand_matrix
        bpy.context.view_layer.update()
        largest = max(largest, math.degrees(original.rotation_difference(arm.matrix.to_quaternion()).angle))
    return {'roll_degrees': roll_degrees, 'max_change_degrees': largest,
            'method': 'fixed mirrored humerus roll from the fitted clavicle rest frame; solved distal transforms retained',
            'source_twist_calibrated': False}
