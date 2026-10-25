"""Compare the render-free fold gate with a completed run's real Checks.

blender --background RUN/scene.blend --python-exit-code 1 --python THIS -- --run RUN
For the old pull-up negative control, add --require-view behind.
"""
import argparse
import json
import sys
from pathlib import Path
try:
    import bpy
except ImportError:
    import pytest
    pytest.skip('Requires Blender and a completed real animation', allow_module_level=True)
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'pipeline'), str(ROOT / 'pipeline/blender'), str(ROOT / 'studio/blender')]
import inspect_run
import skin_preflight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--require-view')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    run = args.run
    assert Path(bpy.data.filepath).resolve() == (run / 'scene.blend').resolve()
    report = json.loads((run / 'frames/retarget.json').read_text())
    checks = json.loads((run / 'checks.json').read_text())
    snapshot = json.loads((run / 'references.json').read_text())
    rig, skin = inspect_run.find_rig_and_skin()
    scene = bpy.context.scene
    held = [obj for obj in inspect_run.rendered_meshes(skin) if obj.name != 'floor']
    original_frame, original_camera = scene.frame_current, scene.camera
    camera_matrix = None if original_camera is None else np.array(original_camera.matrix_world)
    objects = set(obj.name for obj in bpy.data.objects)
    vertices = np.array([tuple(vertex.co) for vertex in skin.data.vertices])
    result = skin_preflight.measure(rig, skin, scene.frame_start, scene.frame_end, snapshot['exercise'], held,
                                    report['contact']['box'], run / 'keyposes.json',
                                    reference_motion=report.get('reference_motion'))
    expected = next(row for row in checks['rows'] if row['metric'] == 'folded_skin')
    assert result['measurement_error'] is None, result
    expected_motion = next(row for row in checks['rows'] if row['metric'] == 'smooth_motion')
    assert result['fold_rows'] == [expected], (expected, result['fold_rows'])
    assert result['motion_rows'] == [expected_motion], (expected_motion, result['motion_rows'])
    expected_ready = all(row['status'] != 'fail' and row['kind'] != 'not-measured' for row in [expected, expected_motion])
    assert result['ready'] is expected_ready
    assert result['measured_frames'] == scene.frame_end - scene.frame_start + 1
    assert scene.frame_current == original_frame and scene.camera == original_camera
    assert set(obj.name for obj in bpy.data.objects) == objects
    assert np.array_equal(vertices, np.array([tuple(vertex.co) for vertex in skin.data.vertices]))
    if original_camera is not None:
        assert np.allclose(camera_matrix, np.array(original_camera.matrix_world), atol=1e-9)
    if args.require_view:
        assert any(camera['view'] == args.require_view for camera in result['comparison_cameras'])
    for camera in result['comparison_cameras']:
        actual = inspect_run.aim_camera(skin, camera['frame'], camera['azimuth'], camera['elevation'])
        assert np.linalg.norm(np.array(actual.location) - camera['location']) < 1e-5
    missing = skin_preflight.measure(rig, skin, scene.frame_start, scene.frame_end, snapshot['exercise'], held,
                                     report['contact']['box'], run / 'missing-reference-dir/keyposes.json',
                                     reference_motion=report.get('reference_motion'))
    assert missing['ready'] is False and missing['rows'][0]['kind'] == 'not-measured'
    partial = skin_preflight.measure(rig, skin, scene.frame_start, scene.frame_end - 1, snapshot['exercise'], held,
                                     report['contact']['box'], run / 'keyposes.json',
                                     reference_motion=report.get('reference_motion'))
    assert partial['ready'] is False and partial['motion_rows'][0]['kind'] == 'not-measured'
    print('SKIN_PREFLIGHT_REPLAY_PASS ' + json.dumps({'exercise': snapshot['exercise'],
          'ready': result['ready'], 'row': expected, 'source_camera_frames': result['comparison_camera_frames'],
          'geometry_unchanged': True, 'missing_metadata_blocks': True, 'partial_clip_blocks': True}), flush=True)


if __name__ == '__main__':
    main()
