"""Independently measure a real saved pulldown, including the old negative scene.

blender --background RUN/scene.blend --python-exit-code 1 --python THIS -- --run RUN
Add --expect-inverted for the old contact-only scene that pulls with raised elbows.
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
    pytest.skip('Requires an actual saved pulldown Blender scene', allow_module_level=True)

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'pipeline/blender')]
import pulldown_contacts
import numpy as np
from mathutils import Vector
import equipment
import exercise_contacts
import skin_wrap
from row_motion import shaft_gaps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--expect-inverted', action='store_true')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    report = json.loads((args.run / 'frames/retarget.json').read_text())
    phases = report['reference_motion']['phase_frames']
    scene = bpy.context.scene
    rig = next(o for o in scene.objects if o.type == 'ARMATURE' and 'mixamorig:Hips' in o.pose.bones)
    measured = pulldown_contacts.measure(rig, scene.objects['barbell'], scene.frame_start, scene.frame_end, phases)
    assert measured['measured_frames'] == scene.frame_end - scene.frame_start + 1
    if args.expect_inverted:
        assert measured['bottom_elbow_above_shoulder_mm'] > 100.
        assert measured['max_elbow_above_wrist_mm'] > 80.
        assert measured['bottom_upperarm_excess_deg'] > 20.
        if args.out:
            measured.update(status='original_inverted_branch_reproduced', rendered=False,
                scene_sha256=hashlib.sha256(Path(bpy.data.filepath).read_bytes()).hexdigest())
            args.out.write_text(json.dumps(measured, indent=2, allow_nan=False))
        print('PULLDOWN_OLD_INVERSION_REPRODUCED', measured['bottom_elbow_above_shoulder_mm'], flush=True)
        return
    for metric in ('max_elbow_above_wrist_mm', 'bottom_elbow_above_shoulder_mm',
                   'bottom_upperarm_excess_deg', 'top_extension_shortfall_deg', 'travel_shortfall_mm'):
        assert measured[metric] == 0., (metric, measured[metric])
    saved = report['physical']['contacts']['pulldown_range']
    for key in ('top_elbows_deg', 'bottom_upperarm_deg', 'bar_travel_mm', 'phase_frames'):
        assert saved[key] == measured[key], (key, saved[key], measured[key])
    checks = json.loads((args.run / 'checks.json').read_text())
    assert checks['summary']['fail'] == 0 and checks['summary']['unmeasured'] == 0
    assert (args.run / 'animation.gif').read_bytes()[:6] in (b'GIF87a', b'GIF89a')
    skin = next(obj for obj in scene.objects if obj.type == 'MESH' and not obj.hide_render
                and any(mod.type == 'ARMATURE' and mod.object == rig for mod in obj.modifiers))
    bar = scene.objects['barbell']
    height = json.loads(rig['skin_fit'])['measured']['height']
    unit = height / 1.75
    masks = exercise_contacts._hand_masks(skin)
    rest_lengths = {bone.name: bone.bone.length for bone in rig.pose.bones
                    if bone.name.endswith(('Arm', 'ForeArm'))}
    grips = json.loads(rig['hand_grip'])
    base_turns = {}
    rolls = {side: [] for side in masks}
    palms = {side: [] for side in masks}
    contacts = {side: [] for side in masks}
    for frame in range(scene.frame_start, scene.frame_end + 1):
        scene.frame_set(frame)
        points = skin_wrap.evaluated(skin)[0]
        assert points[:, 2].min() >= -.003, (frame, 'body below floor')
        for name, length in rest_lengths.items():
            assert abs(rig.pose.bones[name].length - length) <= .0001, (frame, name, 'stretched limb')
        along = np.array((bar.matrix_world.to_3x3() @ Vector((1, 0, 0))).normalized())
        for side, mask in masks.items():
            hand = rig.pose.bones['mixamorig:' + side.title() + 'Hand']
            turn = (rig.matrix_world @ hand.matrix).to_quaternion()
            base_turns.setdefault(side, turn.copy())
            angle = float(np.degrees(base_turns[side].rotation_difference(turn).angle))
            assert angle <= 25.001, (frame, side, 'grip roll', angle)
            rolls[side].append(angle)
            palm = float((exercise_contacts._turn(hand) @ np.array(grips[side]['n']))[1])
            assert palm > 0., (frame, side, 'supinated grip', palm)
            palms[side].append(palm)
            gaps = shaft_gaps(points[mask], np.array(bar.matrix_world.translation), along,
                               equipment.BARBELL_LENGTH / 2 * unit, equipment.BARBELL_BAR_DIAMETER / 2 * unit)
            gap, penetration = float(abs(gaps).min() * 1000), max(0., float(-gaps.min() * 1000))
            assert gap <= 3. and penetration <= 3., (frame, side, gap, penetration)
            contacts[side].append((gap, penetration))
    measured.update(status='passed', rendered=False,
        scene_sha256=hashlib.sha256(Path(bpy.data.filepath).read_bytes()).hexdigest(),
        grip_roll={side: {'frames': len(values), 'max_turn_deg': max(values),
                         'minimum_palm_forward_dot': min(palms[side])} for side, values in rolls.items()},
        contact={side: {'frames': len(values), 'max_gap_mm': max(v[0] for v in values),
                        'max_penetration_mm': max(v[1] for v in values)} for side, values in contacts.items()})
    destination = args.out or args.run / 'pulldown-form-replay.json'
    destination.write_text(json.dumps(measured, indent=2, allow_nan=False))
    print('PULLDOWN_FORM_SCENE_PASS ' + str(destination), flush=True)


if __name__ == '__main__':
    main()
