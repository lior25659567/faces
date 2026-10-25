"""Replay source elbow bend and finite handle contact on a saved pushdown scene.

blender --background RUN/scene.blend --python-exit-code 1 --python THIS_FILE -- --run RUN --out REPORT.json
The source's inferred depth and the prop's width remain explicitly approximate.
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
    pytest.skip('Runs inside Blender with an actual pushdown scene', allow_module_level=True)

import numpy as np
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'pipeline/blender')]
import equipment
import skin_wrap
from exercise_contacts import _elbows, _upperarms, _hand_masks
from row_motion import shaft_gaps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    report_file = args.run / 'frames/retarget.json'
    if not report_file.exists():
        report_file = args.run / 'report.json'
    report = json.loads(report_file.read_text())
    contacts = report['physical']['contacts']
    source_rows = {row['frame']: row for row in contacts['source_elbow_goals']}
    source = {frame: row['elbows_deg'] for frame, row in source_rows.items()}
    source_limit = float(contacts.get('source_angle_tolerance_deg', 3.))
    rig = next(obj for obj in bpy.data.objects if obj.type == 'ARMATURE' and 'skin_fit' in obj)
    skin = next(obj for obj in bpy.data.objects if obj.type == 'MESH' and not obj.hide_render
                and any(mod.type == 'ARMATURE' and mod.object == rig for mod in obj.modifiers))
    handle = bpy.data.objects['cable_handle']
    half = float(handle['contact_half_length'])
    extent = np.array([vertex.co[:] for vertex in handle.data.vertices])[:, 0]
    assert abs(extent.min() + half) < 1e-6 and abs(extent.max() - half) < 1e-6, 'Contact proxy does not match the rendered handle'
    height = json.loads(rig['skin_fit'])['measured']['height']
    radius = equipment.BARBELL_BAR_DIAMETER / 2 * height / 1.75
    assert 2 * half < equipment.BARBELL_LENGTH * height / 1.75
    scene = bpy.context.scene
    first, last = scene.frame_start, scene.frame_end
    assert set(source) == set(range(first, last + 1)), 'Incomplete source bend record'
    rows, base_root, root_drift = [], None, 0.
    masks = _hand_masks(skin)
    for frame in range(first, last + 1):
        scene.frame_set(frame)
        angles = _elbows(rig.pose.bones)
        error = max(abs(angles[side] - source[frame][side]) for side in ('left', 'right'))
        assert error <= source_limit + .01, (frame, source[frame], angles, error)
        upper = None
        if 'upperarm_goal_deg' in source_rows[frame]:
            upper = _upperarms(rig)
            assert max(abs(value - source_rows[frame]['upperarm_goal_deg']) for value in upper.values()) <= source_limit + .01
        root = np.array(rig.pose.bones['mixamorig:Hips'].head)
        base_root = root if base_root is None else base_root
        root_drift = max(root_drift, float(np.linalg.norm(root - base_root) * 1000))
        assert root_drift < .1, 'Pushdown contact must keep the planted body fixed'
        points = skin_wrap.evaluated(skin)[0]
        centre = np.array(handle.matrix_world.translation)
        along = np.array((handle.matrix_world.to_3x3() @ Vector((1, 0, 0))).normalized())
        hands = {}
        for side, mask in masks.items():
            gaps = shaft_gaps(points[mask], centre, along, half, radius)
            gap, depth = float(abs(gaps).min() * 1000), max(0., float(-gaps.min() * 1000))
            assert gap <= 3. and depth <= 3., (frame, side, gap, depth)
            hands[side] = {'gap_mm': gap, 'penetration_mm': depth}
        rows.append({'frame': frame, 'source_elbows_deg': source[frame], 'actual_elbows_deg': angles,
                     'elbow_error_deg': error, 'source_upperarm_goal_deg': source_rows[frame].get('upperarm_goal_deg'),
                     'actual_upperarm_deg': upper, 'hands': hands})
    result = {'status': 'passed', 'frames': len(rows), 'rendered': False,
              'source': 'independent replay of final keyed skin, arm bones and finite short handle',
              'scene_sha256': hashlib.sha256(Path(bpy.data.filepath).read_bytes()).hexdigest(),
              'max_source_elbow_error_deg': max(row['elbow_error_deg'] for row in rows),
              'source_angle_tolerance_deg': source_limit,
              'root_drift_mm': root_drift, 'handle_length_character_units': 2 * half,
              'source_width_calibrated': False, 'per_frame': rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print('TRICEP_CONTACT_PASS ' + str(args.out), flush=True)


if __name__ == '__main__':
    main()
