"""Verify actual exercise skin identity and folds without weakening geometry checks.

blender --background SCENE --python-exit-code 1 --python THIS -- --run RUN --out OUT
The source scene is not modified. --out receives the corrected scene and measured
fold/deformation report for subsequent independent render/replay validation.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
try:
    import bpy
except ImportError:
    import pytest
    pytest.skip('Requires Blender and the actual fitted mascot', allow_module_level=True)
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'pipeline/blender'), str(ROOT / 'studio/blender'), str(ROOT / 'studio/measure')]
import inspect_run as inspect
import mpfb_cage
import pose_cleanup
import retarget
import skin_wrap
from row_motion import shaft_gaps
from mathutils import Vector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--exercise')
    parser.add_argument('--fold-only', action='store_true', help='Verify skin identity and folds without a single-bar grip check.')
    parser.add_argument('--camera-prop-prefix', action='append', default=[],
                        help='Recompute the production preview cube on this scene, including every matching prop.')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    source = Path(bpy.data.filepath)
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    args.out.mkdir(parents=True, exist_ok=True)
    assert (args.out / 'scene.blend').resolve() != source.resolve()
    rig, skin = inspect.find_rig_and_skin()
    first, last = bpy.context.scene.frame_start, bpy.context.scene.frame_end
    original_data = skin.data
    loops = np.array([loop.vertex_index for loop in original_data.loops])
    protected_groups = {g.index for g in skin.vertex_groups if 'Head' in g.name or 'Hand' in g.name}
    protected = np.array([sum(g.weight for g in vertex.groups if g.group in protected_groups) > .5 for vertex in original_data.vertices])
    corners = protected[loops]
    held_pose = rig.data.pose_position
    rig.data.pose_position = 'REST'
    bpy.context.view_layer.update()
    rest_corners = skin_wrap.evaluated(skin)[0][loops].copy()
    authored_normals = np.array([n.vector[:] for n in original_data.corner_normals])
    rig.data.pose_position = held_pose
    before = {}
    for frame in (first, first + (last - first) // 2, last):
        bpy.context.scene.frame_set(frame)
        before[frame] = skin_wrap.evaluated(skin)[0][loops][corners].copy()
    report = pose_cleanup.apply(rig, skin, first, last, exercise=args.exercise)
    assert skin.data is not original_data
    assert report['vertices_after'] < report['vertices_before']
    assert report['uv_and_materials_preserved']
    assert report['max_weld_displacement_mm'] <= report['seam_tolerance_mm'] * 1.01
    assert report['max_bind_displacement_mm'] <= .01
    new_loops = np.array([loop.vertex_index for loop in skin.data.loops])
    rig.data.pose_position = 'REST'
    bpy.context.view_layer.update()
    after_rest = skin_wrap.evaluated(skin)[0][new_loops]
    assert np.linalg.norm(after_rest - rest_corners, axis=1).max() <= 1e-5
    normals = np.array([n.vector[:] for n in skin.data.corner_normals])
    assert np.array_equal(normals[corners], authored_normals[corners]), 'Authored face/hair normals changed'
    assert report['skin_weights_preserved']
    assert report['protected_custom_normals_preserved']
    rig.data.pose_position = held_pose
    face_hand_drift = 0.
    for frame, values in before.items():
        bpy.context.scene.frame_set(frame)
        now = skin_wrap.evaluated(skin)[0][new_loops][corners]
        face_hand_drift = max(face_hand_drift, float(np.linalg.norm(now - values, axis=1).max()))
    assert face_hand_drift <= 1e-5, face_hand_drift
    mesh = skin.data
    assert pose_cleanup.apply(rig, skin, first, last, exercise=args.exercise) == report
    assert skin.data is mesh, 'Repeated cleanup copied or changed the mesh again'
    yaw, *_ = inspect.studio_view(rig, 'left')
    if args.camera_prop_prefix:
        # A saved probe and the current production run can have different root
        # positions/supports. Derive the actual scene's skin+prop preview cube,
        # exactly as animate.py does, instead of borrowing another run's box.
        props = [o for o in inspect.rendered_meshes(skin) if o.name.startswith(tuple(args.camera_prop_prefix))]
        for prefix in args.camera_prop_prefix:
            assert any(o.name.startswith(prefix) for o in props), f'No visible camera prop matches {prefix}'
        low, high = inspect.gif_box(args.out / 'no-report', skin, first, last)
        for frame in range(first, last + 1):
            bpy.context.scene.frame_set(frame)
            for prop in props:
                points = np.array([prop.matrix_world @ Vector(corner) for corner in prop.bound_box])
                low, high = np.minimum(low, points.min(0)), np.maximum(high, points.max(0))
        centre = (low + high) / 2
        half = max(float(max(high - low)), .5) * .55
        box = np.array([centre - half, centre + half])
    else:
        box = inspect.gif_box(args.run, skin, first, last)
    cameras = inspect.gif_cameras(box, yaw)
    # A source's behind camera can expose an elbow fold that neither preview
    # angle sees. Recompute each comparison camera on this actual skin, using
    # the production run's recorded view/frame, and include it only on that
    # comparison frame, exactly as inspect_run does.
    reference_cameras = {}
    source_camera_report = args.run / 'ours.json'
    if source_camera_report.is_file():
        camera_report = json.loads(source_camera_report.read_text())
        side = camera_report.get('camera_side', 'left')
        for still in camera_report.get('stills') or []:
            frame = still['frame']
            assert first <= frame <= last, 'A comparison camera falls outside this clip.'
            azimuth, elevation, _ = inspect.camera_for(still['view'], side, yaw)
            camera = inspect.aim_camera(skin, frame, azimuth, elevation)
            reference_cameras.setdefault(frame, []).append(np.array(camera.location))
    folds = inspect.skin_folds(rig, skin, first, last,
                               lambda frame: cameras + reference_cameras.get(frame, []))
    defects = [fold for fold in folds['folds'] if fold['kind'] == 'defect']
    assert not defects, defects
    deformation = mpfb_cage.clip_deformation(rig, skin, first, last)
    assert deformation['crack_max_mm'] <= 1.
    bar = None if args.fold_only else bpy.data.objects['barbell']
    masks = {}
    for side, title in (('left', 'Left'), ('right', 'Right')):
        groups = {g.index for g in skin.vertex_groups if title + 'Hand' in g.name}
        masks[side] = np.array([sum(g.weight for g in vertex.groups if g.group in groups) > .5 for vertex in skin.data.vertices])
    height = json.loads(rig['skin_fit'])['measured']['height']
    worst_gap = worst_depth = 0.
    for frame in ([] if bar is None else range(first, last + 1)):
        bpy.context.scene.frame_set(frame)
        points = skin_wrap.evaluated(skin)[0]
        centre = np.array(bar.matrix_world.translation)
        along = np.array((bar.matrix_world.to_3x3() @ Vector((1, 0, 0))).normalized())
        for mask in masks.values():
            gaps = shaft_gaps(points[mask], centre, along, 2.2 / 2 * height / 1.75, .028 / 2 * height / 1.75)
            worst_gap = max(worst_gap, float(abs(gaps).min()))
            worst_depth = max(worst_depth, max(0., float(-gaps.min())))
    assert worst_gap <= .003 and worst_depth <= .003, (worst_gap, worst_depth)
    report.update(face_hand_drift_mm=face_hand_drift * 1000, visible_fold_defects=0,
                  hand_gap_mm=None if bar is None else worst_gap * 1000, hand_penetration_mm=None if bar is None else worst_depth * 1000,
                  crack_mm=deformation['crack_max_mm'])
    report['fold_camera_box'] = box.tolist()
    report['fold_camera_source'] = 'actual scene skin and props' if args.camera_prop_prefix else 'run report or skin fallback'
    report['comparison_camera_frames'] = sorted(reference_cameras)
    (args.out / 'cleanup-validation.json').write_text(json.dumps(report, indent=2))
    (args.out / 'cleanup-folds.json').write_text(json.dumps(folds, indent=2))
    bpy.ops.wm.save_as_mainfile(filepath=str(args.out / 'scene.blend'))
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_digest
    print('POSE_CLEANUP_PASS ' + json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
