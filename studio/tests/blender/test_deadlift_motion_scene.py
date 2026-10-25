"""Replay a real conventional deadlift's floor origin and straight-arm lockout.

blender --background RUN/scene.blend --python-exit-code 1 --python THIS -- --run RUN
Use --geometry-only for an explicitly unrendered fitter probe, never for release
approval. Saved JSON alone cannot establish the actual body or loaded equipment.
"""
import argparse
import json
import sys
from pathlib import Path
try:
    import bpy
except ImportError:
    import pytest
    pytest.skip('Requires Blender and an actual deadlift scene', allow_module_level=True)
import numpy as np
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'pipeline'), str(ROOT / 'pipeline/blender'), str(ROOT / 'studio/blender')]
import deadlift_range
import exercise_contacts
import inspect_run
import retarget
import row_bar
import skin_wrap
from row_motion import shaft_gaps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--geometry-only', action='store_true')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    run = args.run
    report_file = run / 'frames/retarget.json'
    if args.geometry_only and not report_file.is_file():
        report_file = run / 'report.json'
    report = json.loads(report_file.read_text())
    rig, skin = inspect_run.find_rig_and_skin()
    scene = bpy.context.scene
    assert Path(bpy.data.filepath).resolve() == (run / 'scene.blend').resolve()
    height = json.loads(rig['skin_fit'])['measured']['height']
    bar = bpy.data.objects['barbell']
    plate_radius = exercise_contacts.equipment.PLATE_DIAMETER / 2 * height / 1.75
    shaft_radius = exercise_contacts.equipment.BARBELL_BAR_DIAMETER / 2 * height / 1.75
    half = exercise_contacts.equipment.BARBELL_LENGTH / 2 * height / 1.75
    hand_masks = exercise_contacts._hand_masks(skin)
    samples, centres, worst_gap, worst_depth, length_error = [], [], 0., 0., 0.
    for frame in range(scene.frame_start, scene.frame_end + 1):
        scene.frame_set(frame)
        centre = np.array(bar.matrix_world.translation)
        centres.append(centre)
        along = np.array((bar.matrix_world.to_3x3() @ Vector((1., 0., 0.))).normalized())
        assert abs(along[2]) <= 1e-4, 'The deadlift shaft is tilted'
        points = skin_wrap.evaluated(skin)[0]
        assert points[:, 2].min() >= -.003, 'Body penetrates the floor'
        for side, title in [('left', 'Left'), ('right', 'Right')]:
            gaps = shaft_gaps(points[hand_masks[side]], centre, along, half, shaft_radius)
            worst_gap = max(worst_gap, float(abs(gaps).min()))
            worst_depth = max(worst_depth, max(0., float(-gaps.min())))
            for part in ('UpLeg', 'Leg', 'Arm', 'ForeArm'):
                bone = rig.pose.bones['mixamorig:' + title + part]
                length_error = max(length_error, abs(float((bone.tail - bone.head).length) - bone.bone.length))
        joints = {side: {name: np.array(rig.matrix_world @ rig.pose.bones['mixamorig:' + title + bone].head)
                         for name, bone in [('shoulder', 'Arm'), ('elbow', 'ForeArm'), ('wrist', 'Hand'),
                                            ('hip', 'UpLeg'), ('knee', 'Leg'), ('ankle', 'Foot')]}
                  for side, title in [('left', 'Left'), ('right', 'Right')]}
        samples.append(deadlift_range.measure_frame(frame, joints, centre))
        bounds = np.array([bar.matrix_world @ Vector(corner) for corner in bar.bound_box])
        assert bounds[:, 2].min() >= -.001, 'Loaded plates penetrate the floor'
        if frame == scene.frame_start:
            vertices = np.array([bar.matrix_world @ vertex.co for vertex in bar.data.vertices]) - centre
            radius = float(np.linalg.norm(vertices - np.outer(vertices @ along, along), axis=1).max())
            assert abs(radius - plate_radius) <= .001, 'The scene needs actual full-size loaded plates'
            assert abs(bounds[:, 2].min()) <= .001, 'The deadlift must start with its plates on the floor'
    assert worst_gap <= .003 and worst_depth <= .003, (worst_gap, worst_depth)
    assert length_error <= .00001, f'The motion stretches a limb by {length_error * 1000} mm'
    assert np.ptp(np.array(centres)[:, 1]) <= .002, 'The bar does not follow its vertical floor-pull path'
    measured = deadlift_range.summarize(samples, height, report['reference_motion']['phase_frames'])
    assert measured['ready'], measured
    feet = retarget.skin_contact(rig, skin, scene.frame_start, scene.frame_end)
    assert feet['penetration_mm'] <= 3., feet
    for reading in feet['sides'].values():
        assert reading['sole_slide_mm'] <= 5. and reading['sole_lift_mm'] <= 3., reading
        assert reading['initial_floor_gap_mm'] <= 3., reading
    body = row_bar.body_clearance(rig, skin, bar, scene.frame_start, scene.frame_end)
    assert body['measured_frames'] == scene.frame_end - scene.frame_start + 1
    assert body['max_penetration_mm'] <= 3., body
    if not args.geometry_only:
        assert (run / 'animation.gif').read_bytes()[:6] in (b'GIF87a', b'GIF89a')
        ours, checks = [json.loads((run / name).read_text()) for name in ('ours.json', 'checks.json')]
        assert checks['summary']['unmeasured'] == 0
        assert report['physical']['validation']['ready']
        for folder in (run / 'frames', run / 'frames/side'):
            assert len(list(folder.glob('frame_*.png'))) == len(samples), folder
        for still in ours['stills']:
            assert (run / still['file']).read_bytes()[:8] == b'\x89PNG\r\n\x1a\n'
    measured.pop('per_frame')
    result = {'mode': 'geometry-only' if args.geometry_only else 'completed-render', 'range': measured,
              'skin_contact': feet, 'max_hand_gap_mm': worst_gap * 1000, 'max_hand_penetration_mm': worst_depth * 1000,
              'posed_length_error_mm': length_error * 1000, 'body_clearance': body}
    (run / 'deadlift-validation.json').write_text(json.dumps(result, indent=2))
    print('DEADLIFT_REPLAY_PASS ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
