"""Independently replay a real declared squat preset's full keyed geometry.

blender --background RUN/scene.blend --python-exit-code 1 --python THIS -- --run RUN
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

try:
    import bpy
except ImportError:
    import pytest
    pytest.skip('Requires Blender and an actual squat preset scene', allow_module_level=True)

import numpy as np
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'pipeline/blender'), str(ROOT / 'studio/blender'), str(ROOT / 'studio/measure')]
import inspect_run
import skin_preflight
import squat_preset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    run = args.run
    scene_file = run / 'scene.blend'
    assert Path(bpy.data.filepath).resolve() == scene_file.resolve()
    squat_preset.validate_clip(ROOT / '3d-character/mixamo/squat.fbx')
    report = json.loads((run / 'frames/retarget.json').read_text())
    scene = bpy.context.scene
    rig, skin = inspect_run.find_rig_and_skin()
    bar = bpy.data.objects.get('barbell')
    assert bar is not None and bar.type == 'MESH'
    first, last = scene.frame_start, scene.frame_end
    assert (first, last, scene.render.fps) == (1, 69, 30)
    original_mesh = np.array([tuple(vertex.co) for vertex in skin.data.vertices])
    measured = squat_preset.measure(rig, skin, bar, first, last)
    movement = measured['squat_range']
    for metric in ('top_knee_extension_shortfall_deg', 'top_hip_extension_shortfall_deg',
                   'bottom_knee_flexion_excess_deg', 'bottom_hip_flexion_excess_deg',
                   'depth_shortfall_mm', 'pelvis_travel_shortfall_mm', 'bar_level_excess_deg'):
        assert movement[metric] == 0., (metric, movement[metric])
    assert movement['knee_asymmetry_deg'] <= 10., movement['knee_asymmetry_deg']
    assert movement['spine_deviation_deg'] <= 15., movement['spine_deviation_deg']
    assert movement['loop_position_error_mm'] <= 5., movement['loop_position_error_mm']
    assert measured['body_clearance']['measured_frames'] == 69
    assert measured['body_clearance']['max_penetration_mm'] <= 3., measured['body_clearance']
    for side, reading in measured['sides'].items():
        assert reading['frames_touching_within_3mm'] == 69, (side, reading)
        assert reading['nearest_gap_mm_max'] <= 3. and reading['deepest_inside_bar_mm'] <= 3., (side, reading)
    for patch, reading in measured['supports']['surfaces'].items():
        assert reading['frames'] == reading['measured_frames'] == 69
        assert reading['max_gap_mm'] <= 5. and reading['max_penetration_mm'] <= 3., (patch, reading)
    assert measured['supports']['planted_feet']['max_slide_mm'] <= 3.
    grips = json.loads(rig['hand_grip'])
    anchor, length_error = 0., 0.
    spans = []
    for frame in range(first, last + 1):
        scene.frame_set(frame)
        centre = np.array(bar.matrix_world.translation)
        axis = np.array((bar.matrix_world.to_3x3() @ Vector((1., 0., 0.))).normalized())
        contacts = []
        for side, title, _ in squat_preset.SIDES:
            arm, forearm, hand = [rig.pose.bones['mixamorig:' + title + name] for name in ('Arm', 'ForeArm', 'Hand')]
            turn = np.array(hand.matrix.to_3x3() @ hand.bone.matrix_local.to_3x3().inverted())
            palm = rig.matrix_world @ Vector(np.array(hand.head) + turn @ (np.array(grips[side]['B']) - grips[side]['wrist']))
            delta = np.array(palm) - centre
            anchor = max(anchor, float(np.linalg.norm(delta - axis * (delta @ axis))))
            contacts.append(palm)
            for bone, child in ((arm, forearm), (forearm, hand)):
                length_error = max(length_error, abs((child.head - bone.head).length - bone.bone.length))
        spans.append((contacts[1] - contacts[0]).length)
    assert anchor <= .002 and max(spans) - min(spans) <= .002, (anchor, max(spans) - min(spans))
    assert length_error <= .001, length_error
    arms = inspect_run.upper_limb_motion(rig, first, last, loop=True)
    arm_rows, arm_ready = skin_preflight.upper_limb_verdict(arms, first, last)
    assert arm_ready, arm_rows
    held = [obj for obj in inspect_run.rendered_meshes(skin) if obj.name != 'floor']
    checked_skin = skin_preflight.measure(rig, skin, first, last, 'Barbell Back Squat', held,
        measured['floor_contact']['box'], run / 'keyposes.json',
        reference_motion=report['reference_motion'], preview_views=report['preview_cameras'])
    assert checked_skin['ready'], checked_skin['rows']
    assert np.array_equal(original_mesh, np.array([tuple(vertex.co) for vertex in skin.data.vertices]))
    assert (run / 'animation.gif').read_bytes()[:6] in (b'GIF87a', b'GIF89a')
    summary = {key: value for key, value in movement.items() if key not in ('per_frame', 'torso_turn', 'leg_turns')}
    output = {'status': 'passed', 'frames': 69, 'scene_sha256': hashlib.sha256(scene_file.read_bytes()).hexdigest(),
              'rendered': False, 'geometry_unchanged': True, 'preset': squat_preset.reference_motion(),
              'range': summary, 'contact': measured['sides'], 'supports': measured['supports'],
              'bar_body_max_overlap_mm': measured['body_clearance']['max_penetration_mm'],
              'max_anchor_error_mm': anchor * 1000, 'arm_length_error_mm': length_error * 1000,
              'upper_limb_rows': arm_rows, 'skin_rows': checked_skin['rows']}
    (run / 'squat-preset-replay.json').write_text(json.dumps(output, indent=2, default=float) + '\n')
    print('SQUAT_PRESET_REPLAY_PASS ' + str(run / 'squat-preset-replay.json'), flush=True)


if __name__ == '__main__':
    main()
