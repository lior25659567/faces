"""Replay actual full-range overhead press geometry, props and supports.

blender --background RUN/scene.blend --python-exit-code 1 --python THIS -- --run RUN
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
    pytest.skip('Requires an actual overhead press scene', allow_module_level=True)

import numpy as np
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'pipeline/blender'), str(ROOT / 'studio/blender'), str(ROOT / 'studio/measure')]
import equipment
import exercise_contacts as contact
import inspect_run
import overhead_contacts
import skin_wrap
from bench_feet import floor_report
from checks import _upper_limb_rows
from row_motion import shaft_gaps


def selected(skin, suffixes):
    groups = {g.index for g in skin.vertex_groups if any(g.name.endswith(s) for s in suffixes)}
    mask = np.array([sum(g.weight for g in v.groups if g.group in groups) > .5 for v in skin.data.vertices])
    assert mask.any(), 'Missing actual region: ' + ', '.join(suffixes)
    return mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    report = json.loads((args.run / 'frames/retarget.json').read_text())
    snapshot = json.loads((args.run / 'references.json').read_text())
    checks = json.loads((args.run / 'checks.json').read_text())
    assert snapshot['exercise'] in overhead_contacts.EXERCISES
    assert checks['summary']['fail'] == checks['summary']['unmeasured'] == 0, checks['summary']
    assert (args.run / 'animation.gif').read_bytes()[:6] in (b'GIF89a', b'GIF87a')
    rig, skin = inspect_run.find_rig_and_skin()
    scene = bpy.context.scene
    first, last = scene.frame_start, scene.frame_end
    height = json.loads(rig['skin_fit'])['measured']['height']
    grips = json.loads(rig['hand_grip'])
    separate = snapshot['exercise'] == overhead_contacts.SEATED
    props = {side: scene.objects['dumbbell_' + side] for side in ('left', 'right')} if separate else {'bar': scene.objects['barbell']}
    measured = overhead_contacts.measure(rig, skin, props, separate, grips, first, last, height,
                                         report['reference_motion']['phase_frames'])
    for name in ('lockout_shortfall_deg', 'bottom_flexion_excess_deg', 'bottom_overfold_deg',
                 'elbow_above_wrist_mm', 'travel_shortfall_mm', 'peak_stack_excess_mm', 'lockout_head_shortfall_mm'):
        assert measured[name] <= 1e-5, (name, measured[name])
    assert measured['elbow_asymmetry_deg'] <= 5., measured
    assert measured['hip_drift_mm'] <= 3., measured
    assert measured['torso_lean_deg'] <= 15., measured
    hands = contact._hand_masks(skin)
    feet = {side.lower(): selected(skin, [side + 'Foot', side + 'ToeBase']) for side in ('Left', 'Right')}
    frames = []
    gap = penetration = length_error = back_gap = back_penetration = 0.
    radius = equipment.BARBELL_BAR_DIAMETER / 2 * height / 1.75
    rest = np.array([v.co[:] for v in skin.data.vertices])
    for frame in range(first, last + 1):
        scene.frame_set(frame)
        points = skin_wrap.evaluated(skin)[0]
        frames.append((frame, points.copy()))
        for side in ('left', 'right'):
            obj = props[side] if separate else props['bar']
            along = np.array((obj.matrix_world.to_3x3() @ Vector((1., 0., 0.))).normalized())
            half = (equipment.DUMBBELL_HANDLE_LENGTH if separate else equipment.BARBELL_LENGTH) / 2 * height / 1.75
            distances = shaft_gaps(points[hands[side]], np.array(obj.matrix_world.translation), along, half, radius)
            gap = max(gap, float(abs(distances).min()) * 1000)
            penetration = max(penetration, max(0., float(-distances.min())) * 1000)
        for side in ('Left', 'Right'):
            for part in ('Arm', 'ForeArm', 'Hand', 'UpLeg', 'Leg', 'Foot'):
                bone = rig.pose.bones['mixamorig:' + side + part]
                length_error = max(length_error, abs((bone.tail - bone.head).length - bone.bone.length) * 1000)
        if separate:
            pad = scene.objects['seat_back']
            plane = float(pad.location.y + pad.dimensions.y / 2)
            selected_back = points[selected(skin, ['Spine1', 'Spine2'])]
            on_pad = selected_back[(abs(selected_back[:, 0] - pad.location.x) <= pad.dimensions.x / 2)
                                   & (abs(selected_back[:, 2] - pad.location.z) <= pad.dimensions.z / 2)]
            assert len(on_pad), ('No body surface over the backrest', frame)
            back_gap = max(back_gap, float(abs(on_pad[:, 1] - plane).min()) * 1000)
            back_penetration = max(back_penetration, max(0., float(plane - on_pad[:, 1].min())) * 1000)
    assert gap <= 3. and penetration <= 3., (gap, penetration)
    assert length_error <= .5, length_error
    floor = floor_report(rest, feet, frames)
    assert all(v['max_gap_mm'] <= 5. and v['max_penetration_mm'] <= 3. for v in floor['surfaces'].values()), floor
    assert floor['planted_feet']['max_slide_mm'] <= 3., floor
    assert back_gap <= 5. and back_penetration <= 3., (back_gap, back_penetration)
    arms = inspect_run.upper_limb_motion(rig, first, last, loop=True)
    arm_rows = _upper_limb_rows({'first': first, 'last': last, 'upper_limb_motion': arms})
    assert all(row['status'] == 'pass' for row in arm_rows), arm_rows
    roll = {}
    for side in ('left', 'right'):
        start = np.asarray(arms['per_frame'][0]['turns_wxyz']['hand.' + side])
        roll[side] = max(float(np.degrees(2 * np.arccos(np.clip(abs(start @
            np.asarray(sample['turns_wxyz']['hand.' + side])), 0., 1.)))) for sample in arms['per_frame'])
    if not separate:
        assert max(roll.values()) <= 10.05, roll
    if separate:
        weights = contact.dumbbell_clearance(skin, props, first, last, height)
        assert weights['max_penetration_mm'] <= 3., weights
    else:
        bar = props['bar']
        points = np.array([v.co[:] for v in bar.data.vertices])
        for sign in (-1., 1.):
            plate = points[points[:, 0] * sign > .4 * height]
            assert len(plate) and np.max(np.linalg.norm(plate[:, 1:], axis=1)) >= .225 * height / 1.75 - .003
        from row_bar import body_clearance
        weights = body_clearance(rig, skin, bar, first, last, include_head=True)
        assert weights['max_penetration_mm'] <= 3., weights
    summary = {key: value for key, value in measured.items() if key != 'per_frame'}
    summary.update(status='passed', scene_sha256=hashlib.sha256(Path(bpy.data.filepath).read_bytes()).hexdigest(),
        rendered=False, max_hand_gap_mm=gap, max_hand_penetration_mm=penetration,
        max_bone_length_error_mm=length_error, back_gap_mm=back_gap, back_penetration_mm=back_penetration,
        planted_feet=floor['planted_feet'], hand_roll_from_start_deg=roll,
        arm_continuity_rows=arm_rows, summary=checks['summary'])
    (args.run / 'overhead-range-replay.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print('OVERHEAD_PRESS_REPLAY_PASS ' + json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
