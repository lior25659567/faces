"""Replay the leg-only sled motion and actual stationary support geometry.

blender --background RUN/scene.blend --python-exit-code 1 --python THIS -- --run RUN --out REPORT
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

try:
    import bpy
except ImportError:
    import pytest
    pytest.skip('Requires the actual finished leg-press Blender scene', allow_module_level=True)

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'pipeline/blender'), str(ROOT / 'studio/measure')]
from sled_body import measure
from sled_motion import summarize
from physical_checks import rows_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    scene = bpy.context.scene
    rig = next(obj for obj in scene.objects if obj.type == 'ARMATURE' and 'skin_fit' in obj)
    report = json.loads((args.run / 'frames/retarget.json').read_text())
    actual = measure(rig, scene.frame_start, scene.frame_end, report['reference_motion']['phase_frames'])
    assert actual['max_supported_body_drift_mm'] <= 2., actual
    assert actual['max_hand_drift_mm'] <= 2., actual
    assert all(actual[key] == 0. for key in ('top_extension_shortfall_deg', 'bottom_flexion_shortfall_deg', 'bottom_overfold_deg'))
    assert actual['max_knee_asymmetry_deg'] <= 5., actual
    report['physical']['supports']['sled_motion'] = actual
    measured_rows = [row for row in rows_for(report) if row['metric'].startswith('sled_')]
    assert len(measured_rows) >= 7 and all(row['status'] == 'pass' for row in measured_rows), measured_rows
    assert (args.run / 'animation.gif').read_bytes()[:6] in (b'GIF87a', b'GIF89a')
    # Independently inspect every actual pad/sole/rail mesh, not the report's
    # stored maxima. This also verifies fixed pads and unchanged bone lengths.
    runpy.run_path(str(ROOT / 'studio/tests/blender/test_support_geometry.py'), run_name='__main__')
    old = json.loads((ROOT / 'studio/tests/fixtures/sled-original-motion.json').read_text())
    rejected = summarize(old['per_frame'], old['height_m'], old['phase_frames'])
    assert rejected['max_supported_body_drift_mm'] > 2. and rejected['max_hand_drift_mm'] > 2.
    result = {key: value for key, value in actual.items() if key != 'per_frame'}
    result.update(status='passed', rendered=True, original_rocking_rejected=True,
                  scene_sha256=hashlib.sha256((args.run / 'scene.blend').read_bytes()).hexdigest())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('SLED_MOTION_REPLAY_PASS ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
