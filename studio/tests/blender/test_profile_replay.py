"""Independently replay a completed new profile's real skin/handles and outputs.

blender --background RUN/scene.blend --python-exit-code 1 --python THIS_FILE -- --run RUN
This requires a finished GIF, reference comparisons and a saved scene; it cannot
be satisfied by a synthetic geometry-only probe.
"""
import argparse
import json
import sys
from pathlib import Path
try:
    import bpy
except ImportError:
    import pytest
    pytest.skip('Requires Blender and a real completed run', allow_module_level=True)
import numpy as np
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'pipeline/blender'), str(ROOT / 'studio/measure')]
import exercise_contacts as contact
import skin_wrap
from row_motion import shaft_gaps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    run = args.run
    report = json.loads((run / 'frames/retarget.json').read_text())
    ours = json.loads((run / 'ours.json').read_text())
    checks = json.loads((run / 'checks.json').read_text())
    snapshot = json.loads((run / 'references.json').read_text())
    assert snapshot['exercise'] in contact.DUMBBELLS | {'Conventional Barbell Deadlift', 'Lat Pulldown (seated, to the front)'}
    assert (run / 'animation.gif').read_bytes()[:6] in (b'GIF89a', b'GIF87a')
    assert len((run / 'animation.gif').read_bytes()) > 1000
    assert len(ours['stills']) >= 2
    for still in ours['stills']:
        assert (run / still['file']).read_bytes()[:8] == b'\x89PNG\r\n\x1a\n'
    assert checks['summary']['unmeasured'] == 0, checks['summary']
    rig = next(o for o in bpy.data.objects if o.type == 'ARMATURE' and 'mixamorig:Hips' in o.pose.bones)
    skin = next(o for o in bpy.data.objects if o.type == 'MESH' and not o.hide_render and any(m.type == 'ARMATURE' and m.object == rig for m in o.modifiers))
    fit = json.loads(rig['skin_fit'])
    height = fit['measured']['height']
    masks = contact._hand_masks(skin)
    separate = snapshot['exercise'] in contact.DUMBBELLS
    props = {side: bpy.data.objects['dumbbell_' + side] for side in masks} if separate else {'bar': bpy.data.objects['barbell']}
    worst_gap = worst_depth = 0.
    first, last = bpy.context.scene.frame_start, bpy.context.scene.frame_end
    for frame in range(first, last + 1):
        bpy.context.scene.frame_set(frame)
        points = skin_wrap.evaluated(skin)[0]
        assert points[:, 2].min() >= -.005, 'Body penetrates the floor'
        for side in masks:
            obj = props[side] if separate else props['bar']
            centre = np.array(obj.matrix_world.translation)
            along = np.array((obj.matrix_world.to_3x3() @ Vector((1, 0, 0))).normalized())
            half = (contact.equipment.DUMBBELL_HANDLE_LENGTH if separate else contact.equipment.BARBELL_LENGTH) / 2 * height / 1.75
            gaps = shaft_gaps(points[masks[side]], centre, along, half, contact.equipment.BARBELL_BAR_DIAMETER / 2 * height / 1.75)
            worst_gap = max(worst_gap, float(abs(gaps).min()))
            worst_depth = max(worst_depth, max(0., float(-gaps.min())))
    assert worst_gap <= .003 and worst_depth <= .003, (worst_gap, worst_depth)
    if separate:
        heads = contact.dumbbell_clearance(skin, props, first, last, height)
        assert heads['frames'] == last - first + 1
        assert heads['max_penetration_mm'] <= 3., heads
    assert report['physical']['validation']['ready']
    print('PROFILE_REPLAY_PASS ' + json.dumps({'exercise': snapshot['exercise'], 'frames': last - first + 1,
        'max_hand_gap_mm': worst_gap * 1000, 'max_hand_penetration_mm': worst_depth * 1000,
        'summary': checks['summary']}), flush=True)


if __name__ == '__main__':
    main()
