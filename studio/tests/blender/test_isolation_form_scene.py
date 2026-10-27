"""Replay actual isolation form, closed grips and arm continuity from a saved scene.

blender --background RUN/scene.blend --python-exit-code 1 --python THIS_FILE -- --run RUN
Photo comparisons are preserved separately, including a partial pushdown source.
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
    pytest.skip('Requires Blender and an actual saved isolation scene', allow_module_level=True)

import numpy as np
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'pipeline/blender'), str(ROOT / 'studio/blender'), str(ROOT / 'studio/measure')]
import equipment
import exercise_contacts as contacts
import isolation_contacts
from isolation_range import CURL, RAISE, PUSHDOWN
from row_motion import shaft_gaps
import skin_wrap
from inspect_run import upper_limb_motion
from checks import _upper_limb_rows
from bench_feet import floor_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    run = args.run
    report = json.loads((run / 'frames/retarget.json').read_text())
    exercise = json.loads((run / 'references.json').read_text())['exercise']
    assert exercise in (CURL, RAISE, PUSHDOWN)
    rig = next(o for o in bpy.data.objects if o.type == 'ARMATURE' and 'skin_fit' in o)
    skin = next(o for o in bpy.data.objects if o.type == 'MESH' and not o.hide_render
        and any(m.type == 'ARMATURE' and m.object == rig for m in o.modifiers))
    scene = bpy.context.scene
    first, last = scene.frame_start, scene.frame_end
    height = json.loads(rig['skin_fit'])['measured']['height']
    form = isolation_contacts.measure(rig, first, last, height, report['reference_motion']['phase_frames'], exercise)
    for metric in ('body_lean_excess_deg', 'knee_extension_shortfall_deg', 'hip_extension_shortfall_deg'):
        assert form[metric] <= 1e-4, (metric, form[metric])
    assert form['body_drift_mm'] <= 2. and form['elbow_asymmetry_deg'] <= 5., form
    if exercise == CURL:
        assert form['upperarm_drift_deg'] <= 5.
        zero = ('upperarm_position_excess_deg', 'start_extension_shortfall_deg', 'end_flexion_excess_deg', 'end_overfold_deg')
    elif exercise == RAISE:
        assert form['elbow_bend_variation_deg'] <= 3.
        zero = ('start_elevation_excess_deg', 'end_elevation_shortfall_deg', 'end_elevation_excess_deg', 'elbow_bend_excess_deg')
    else:
        assert form['upperarm_drift_deg'] <= 12.
        zero = ('upperarm_position_excess_deg', 'start_bend_excess_deg', 'start_overfold_deg', 'end_extension_shortfall_deg')
    assert all(form[name] <= 1e-4 for name in zero), {name: form[name] for name in zero}
    arm = upper_limb_motion(rig, first, last, loop=True)
    arm_rows = _upper_limb_rows({'first': first, 'last': last, 'upper_limb_motion': arm})
    assert all(row['status'] == 'pass' for row in arm_rows), arm_rows
    masks = contacts._hand_masks(skin)
    separate = exercise != PUSHDOWN
    props = {side: bpy.data.objects['dumbbell_' + side] for side in masks} if separate else {'bar': bpy.data.objects['cable_handle']}
    max_gap = max_depth = 0.
    for frame in range(first, last + 1):
        scene.frame_set(frame)
        points = skin_wrap.evaluated(skin)[0]
        assert points[:, 2].min() >= -.003, (frame, float(points[:, 2].min()))
        for side, mask in masks.items():
            obj = props[side] if separate else props['bar']
            centre = np.array(obj.matrix_world.translation)
            axis = np.array((obj.matrix_world.to_3x3() @ Vector((1., 0., 0.))).normalized())
            half = equipment.DUMBBELL_HANDLE_LENGTH / 2 * height / 1.75 if separate else float(obj['contact_half_length'])
            gaps = shaft_gaps(points[mask], centre, axis, half, equipment.BARBELL_BAR_DIAMETER / 2 * height / 1.75)
            max_gap = max(max_gap, float(abs(gaps).min() * 1000))
            max_depth = max(max_depth, max(0., float(-gaps.min() * 1000)))
    assert max_gap <= 3. and max_depth <= 3., (max_gap, max_depth)
    feet = {}
    for side in ('Left', 'Right'):
        groups = {g.index for g in skin.vertex_groups if g.name in ('mixamorig:' + side + 'Foot', 'mixamorig:' + side + 'ToeBase')}
        feet[side.lower()] = np.array([sum(g.weight for g in v.groups if g.group in groups) > .5 for v in skin.data.vertices])
    def floor_frames():
        for frame in range(first, last + 1):
            scene.frame_set(frame)
            yield frame, skin_wrap.evaluated(skin)[0]
    floor = floor_report(np.array([v.co for v in skin.data.vertices]), feet, floor_frames())
    assert floor['planted_feet']['max_slide_mm'] <= 3., floor
    assert all(r['max_gap_mm'] <= 5. and r['max_penetration_mm'] <= 3. for r in floor['surfaces'].values()), floor
    if separate:
        heads = contacts.dumbbell_clearance(skin, props, first, last, height)
        assert heads['frames'] == last - first + 1 and heads['max_penetration_mm'] <= 3., heads
    result = {key: value for key, value in form.items() if key != 'per_frame'}
    result.update(status='passed', source='Independent actual saved-scene joint/skin/grip replay', rendered=False,
        scene_sha256=hashlib.sha256(Path(bpy.data.filepath).read_bytes()).hexdigest(),
        max_hand_gap_mm=max_gap, max_hand_penetration_mm=max_depth,
        planted_feet=floor,
        upper_limb_rows=arm_rows)
    (run / 'isolation-form-replay.json').write_text(json.dumps(result, indent=2, allow_nan=False, default=float))
    print('ISOLATION_FORM_PASS ' + json.dumps(result, allow_nan=False, default=float), flush=True)


if __name__ == '__main__':
    main()
