"""Replay a real floor pull and reject bare-bar or floating-origin shortcuts.

blender --background scene.blend --python-exit-code 1 --python THIS_FILE -- --report report.json
The input scene must be a physically fitted complete conventional floor pull.
All counterfactual changes stay in this disposable process; no source assets or
saved scenes are overwritten.
"""
import argparse
import json
import sys
from pathlib import Path
try:
    import bpy
except ImportError:
    import pytest
    pytest.skip('Requires the actual fitted Blender character and loaded bar', allow_module_level=True)

import bmesh
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'pipeline/blender'), str(ROOT / 'studio/measure')]
from deadlift_range import measure_rig
from physical_checks import rows_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    report = json.loads(args.report.read_text())
    phases = report['reference_motion']['phase_frames']
    scene = bpy.context.scene
    rig = next(obj for obj in scene.objects if obj.type == 'ARMATURE' and 'skin_fit' in obj)
    bar = bpy.data.objects['barbell']
    height = json.loads(rig['skin_fit'])['measured']['height']
    first, last = scene.frame_start, scene.frame_end
    original = measure_rig(rig, bar, first, last, height, phases)
    assert original['ready'], original['failures']
    report['physical']['contacts']['deadlift_range'] = original
    report['physical'].setdefault('profile', {'family': 'hinge', 'holds': True})
    checks = [row for row in rows_for(report) if row['metric'].startswith('deadlift_')]
    assert len(checks) == 12 and all(row['status'] == 'pass' for row in checks), checks
    assert original['plate_floor']['missing_loaded_plates'] == 0
    # Shift the object's origin 50 mm upward without moving its rendered mesh.
    # Actual geometry must give identical shaft heights and plate floor proof.
    bar.data = bar.data.copy()
    for vertex in bar.data.vertices:
        vertex.co.z -= .05
    anchor = bpy.data.objects.new('test_origin_offset', None)
    scene.collection.objects.link(anchor)
    anchor.location.z = .05
    bar.parent = anchor
    bar.matrix_parent_inverse.identity()
    bpy.context.view_layer.update()
    shifted = measure_rig(rig, bar, first, last, height, phases)
    assert shifted['ready'], shifted['failures']
    assert abs(shifted['forward_bar_travel_mm'] - original['forward_bar_travel_mm']) < .001
    assert abs(shifted['plate_floor']['start_floor_gap_mm'] - original['plate_floor']['start_floor_gap_mm']) < .001
    assert abs(shifted['plate_floor']['max_floor_penetration_mm'] - original['plate_floor']['max_floor_penetration_mm']) < .001
    # A real 4 mm plate penetration must fail even with the centre above floor.
    anchor.location.z -= .004
    bpy.context.view_layer.update()
    sunk = measure_rig(rig, bar, first, last, height, phases)
    assert not sunk['ready'] and 'plate_floor_penetration' in sunk['failures']
    assert sunk['plate_floor']['max_floor_penetration_mm'] > 3.9
    anchor.location.z += .004
    bpy.context.view_layer.update()
    # Keep the object name, centre, sleeves and decorative rim bands. Removing
    # the two real weight-disc surfaces must still be detected as a bare bar.
    indices = {index for part in shifted['plate_mesh']['plates'].values() for index in part['vertex_indices']}
    mesh = bmesh.new()
    mesh.from_mesh(bar.data)
    mesh.verts.ensure_lookup_table()
    bmesh.ops.delete(mesh, geom=[mesh.verts[index] for index in indices], context='VERTS')
    mesh.to_mesh(bar.data)
    mesh.free()
    bpy.context.view_layer.update()
    bare = measure_rig(rig, bar, first, last, height, phases)
    assert not bare['ready'] and 'loaded_plate_geometry' in bare['failures']
    assert bare['plate_floor']['missing_loaded_plates'] == 2
    assert bare['plate_floor']['start_floor_gap_mm'] is None
    assert bare['plate_floor']['max_floor_penetration_mm'] is None
    print('DEADLIFT_RANGE_REPLAY_PASS ' + json.dumps({'frames': original['frames'],
        'minimum_elbow_deg': original['minimum_elbow_deg'],
        'forward_bar_travel_mm': original['forward_bar_travel_mm'],
        'plate_radius_mm': original['plate_floor']['radii_mm'],
        'actual_floor_penetration_mm': original['plate_floor']['max_floor_penetration_mm'],
        'origin_shift_preserves_measurement': True, 'bare_bar_rejected': True,
        'actual_penetration_rejected': True}), flush=True)


if __name__ == '__main__':
    main()
