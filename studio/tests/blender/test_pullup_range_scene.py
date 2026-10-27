"""Replay full pull-up range and closed-hand contact from an actual saved scene.

blender --background SCENE.blend --python-exit-code 1 --python THIS_FILE -- --out REPORT.json
This does not claim that the source athlete's inferred depth is calibrated.
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
    pytest.skip('Runs inside Blender with an actual pull-up scene', allow_module_level=True)

import numpy as np
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'pipeline/blender')]
import equipment
import mpfb_cage
import skin_wrap
from exercise_contacts import pullup_range, _hand_masks, _turn
from row_motion import shaft_gaps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    scene = bpy.context.scene
    rig = next(obj for obj in bpy.data.objects if obj.type == 'ARMATURE' and 'skin_fit' in obj)
    skin = next(obj for obj in bpy.data.objects if obj.type == 'MESH' and not obj.hide_render
                and any(mod.type == 'ARMATURE' and mod.object == rig for mod in obj.modifiers))
    bar = next(obj for obj in bpy.data.objects if obj.name == 'barbell')
    first, last = scene.frame_start, scene.frame_end
    movement = pullup_range(rig, skin, bar, first, last)
    assert movement['frames'] == last - first + 1
    assert movement['chin_shortfall_mm'] == 0., movement
    assert movement['extension_shortfall_deg'] <= 5., movement
    pole = movement.get('elbow_pole') or {}
    assert pole.get('measured_frames') == movement['frames'], movement
    assert all(pole.get(key) == 0. for key in
               ('max_elbow_above_bar_mm', 'max_elbow_above_wrist_mm', 'peak_elbow_above_shoulder_mm')), pole
    motion = movement.get('motion_continuity') or {}
    assert motion.get('frames') == movement['frames'], motion
    assert motion.get('max_elbow_speed_arm_lengths_per_s', float('inf')) <= 6., motion
    assert motion.get('max_return_path_error_arm_lengths', float('inf')) <= .12, motion
    height = json.loads(rig['skin_fit'])['measured']['height']
    unit = height / 1.75
    radius = equipment.BARBELL_BAR_DIAMETER / 2 * unit
    half = equipment.BARBELL_LENGTH / 2 * unit
    masks = _hand_masks(skin)
    grips = json.loads(rig['hand_grip'])
    contacts = {side: [] for side in masks}
    hand_base, hand_roll = {}, {side: [] for side in masks}
    palm_forward = {side: [] for side in masks}
    base = None
    drift = rotation = 0.
    for frame in range(first, last + 1):
        scene.frame_set(frame)
        now = (bar.matrix_world.translation.copy(), bar.matrix_world.to_quaternion())
        base = base or now
        drift = max(drift, float((now[0] - base[0]).length) * 1000)
        rotation = max(rotation, float(np.degrees(now[1].rotation_difference(base[1]).angle)))
        points = skin_wrap.evaluated(skin)[0]
        axis = np.array((bar.matrix_world.to_3x3() @ Vector((1, 0, 0))).normalized())
        for side, mask in masks.items():
            title = side.title()
            turn = (rig.matrix_world @ rig.pose.bones[mpfb_cage.P + title + 'Hand'].matrix).to_quaternion()
            hand_base.setdefault(side, turn.copy())
            angle = float(np.degrees(hand_base[side].rotation_difference(turn).angle))
            assert angle <= 20.001, (frame, side, angle)
            hand_roll[side].append({'frame': frame, 'roll_deg': angle})
            # Overhand pull-ups keep the palms facing away from the athlete.
            # The fitted character's forward direction is canonical +Y.
            forward = float((_turn(rig.pose.bones[mpfb_cage.P + title + 'Hand']) @ np.array(grips[side]['n']))[1])
            assert forward > 0., (frame, side, 'inverted/supinated grip', forward)
            palm_forward[side].append(forward)
            gaps = shaft_gaps(points[mask], np.array(now[0]), axis, half, radius)
            gap, depth = float(abs(gaps).min() * 1000), max(0., float(-gaps.min() * 1000))
            assert gap <= 3. and depth <= 3., (frame, side, gap, depth)
            contacts[side].append({'frame': frame, 'gap_mm': gap, 'penetration_mm': depth})
    assert drift <= 1. and rotation <= .1, (drift, rotation)
    movement.update(status='passed', source='replayed actual keyed skin and fixed bar', rendered=False,
                    scene_sha256=hashlib.sha256(Path(bpy.data.filepath).read_bytes()).hexdigest(),
                    fixed_bar_drift_mm=drift, fixed_bar_rotation_deg=rotation,
                    grip_roll={side: {'frames': len(rows), 'max_turn_deg': max(row['roll_deg'] for row in rows),
                                     'minimum_palm_forward_dot': min(palm_forward[side]),
                                     'per_frame': rows} for side, rows in hand_roll.items()},
                    contact={side: {'frames': len(rows), 'max_gap_mm': max(r['gap_mm'] for r in rows),
                                    'max_penetration_mm': max(r['penetration_mm'] for r in rows)}
                             for side, rows in contacts.items()})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(movement, indent=2, allow_nan=False))
    print('PULLUP_RANGE_PASS ' + str(args.out), flush=True)


if __name__ == '__main__':
    main()
