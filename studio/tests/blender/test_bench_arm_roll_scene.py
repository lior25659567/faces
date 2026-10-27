"""Replay the authored bench shoulder roll without moving solved distal poses.

Run inside Blender against an actual bench-press scene. This verifies that
different unobserved source twists converge to the same declared roll while
the solved elbow, forearm, wrist and hand transforms stay fixed.
"""
import sys
from pathlib import Path

try:
    import bpy
except ImportError:
    import pytest
    pytest.skip('Requires an actual built bench-press scene', allow_module_level=True)

import numpy as np
from mathutils import Matrix

sys.path[:0] = [str(Path(__file__).resolve().parents[3] / 'pipeline/blender')]
import bench_arm_roll
import mpfb_cage as cage


def main():
    scene = bpy.context.scene
    rig = next(obj for obj in scene.objects if obj.type == 'ARMATURE' and 'skin_fit' in obj)
    largest_distal_error = largest_source_error = 0.
    for frame in range(scene.frame_start, scene.frame_end + 1):
        scene.frame_set(frame)
        originals = {bone.name: bone.matrix_basis.copy() for bone in rig.pose.bones}
        distal = {cage.P + side + part: rig.pose.bones[cage.P + side + part].matrix.copy()
                  for side in ('Left', 'Right') for part in ('ForeArm', 'Hand')}
        bench_arm_roll.orient_upperarms(rig)
        canonical = {side: rig.pose.bones[cage.P + side + 'Arm'].matrix.copy() for side in ('Left', 'Right')}
        for name, matrix in distal.items():
            largest_distal_error = max(largest_distal_error, float(abs(np.array(rig.pose.bones[name].matrix) - matrix).max()))
        for source_roll in (-1.2, 1.4):
            for bone in rig.pose.bones:
                bone.matrix_basis = originals[bone.name]
            bpy.context.view_layer.update()
            for side in ('Left', 'Right'):
                arm = rig.pose.bones[cage.P + side + 'Arm']
                forearm, hand = [rig.pose.bones[cage.P + side + part] for part in ('ForeArm', 'Hand')]
                matrix = arm.matrix.copy()
                turn = Matrix.Rotation(source_roll, 3, arm.tail - arm.head) @ matrix.to_3x3()
                arm.matrix = turn.to_4x4()
                arm.matrix.translation = matrix.translation
                bpy.context.view_layer.update()
                forearm.matrix = distal[forearm.name]
                bpy.context.view_layer.update()
                hand.matrix = distal[hand.name]
                bpy.context.view_layer.update()
            bench_arm_roll.orient_upperarms(rig)
            for side, matrix in canonical.items():
                largest_source_error = max(largest_source_error,
                    float(abs(np.array(rig.pose.bones[cage.P + side + 'Arm'].matrix) - matrix).max()))
            for name, matrix in distal.items():
                largest_distal_error = max(largest_distal_error,
                    float(abs(np.array(rig.pose.bones[name].matrix) - matrix).max()))
    assert largest_distal_error <= .00002, largest_distal_error
    assert largest_source_error <= .00002, largest_source_error
    print('BENCH_ARM_ROLL_REPLAY_PASS', {'frames': scene.frame_end - scene.frame_start + 1,
          'max_distal_matrix_error': largest_distal_error,
          'max_inherited_source_twist_error': largest_source_error}, flush=True)


if __name__ == '__main__':
    main()
