"""Blender measurement bridge for the declared isolation form policies."""
import bpy
import mpfb_cage as cage
from isolation_range import summarize


def measure(rig, first, last, height, phases, exercise):
    scene, pose, world = bpy.context.scene, rig.pose.bones, rig.matrix_world
    original = scene.frame_current
    samples = []
    try:
        for frame in range(first, last + 1):
            scene.frame_set(frame)
            points = {name: {side.lower(): list(world @ pose[cage.P + side + bone].head)
                            for side in ('Left', 'Right')}
                      for name, bone in (('shoulders_m', 'Arm'), ('elbows_m', 'ForeArm'), ('wrists_m', 'Hand'),
                          ('hips_m', 'UpLeg'), ('knees_m', 'Leg'), ('ankles_m', 'Foot'))}
            samples.append({'frame': frame, **points, 'spine2_m': list(world @ pose[cage.P + 'Spine2'].head)})
        return summarize(samples, height, phases, exercise)
    finally:
        scene.frame_set(original)
