"""Independently check the saved bench animation's actual mesh and movement.

blender --background RUN/scene.blend --python-exit-code 1 --python THIS -- --run RUN
"""
import argparse
import json
from pathlib import Path
import sys

try:
    import bpy
except ImportError:
    import pytest
    pytest.skip('Requires an actual bench-press Blender scene', allow_module_level=True)

import numpy as np
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'pipeline'), str(ROOT / 'pipeline/blender'),
               str(ROOT / 'studio/blender'), str(ROOT / 'studio/measure')]
import equipment
import inspect_run
import skin_wrap
from pullup_range import included
from row_motion import shaft_gaps


def selected(skin, suffixes):
    groups = {g.index for g in skin.vertex_groups if any(g.name.endswith(s) for s in suffixes)}
    mask = np.array([sum(g.weight for g in v.groups if g.group in groups) > .5 for v in skin.data.vertices])
    assert mask.any(), 'Missing actual skin region: ' + ','.join(suffixes)
    return mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    report = json.loads((args.run / 'frames/retarget.json').read_text())
    snapshot = json.loads((args.run / 'references.json').read_text())
    assert snapshot['exercise'] == 'Flat Barbell Bench Press'
    scene = bpy.context.scene
    rig, skin = inspect_run.find_rig_and_skin()
    pose = rig.pose.bones
    prefix = 'mixamorig:'
    height = json.loads(rig['skin_fit'])['measured']['height']
    bar = scene.objects['barbell']
    first, last = scene.frame_start, scene.frame_end
    phases = report['reference_motion']['phase_frames']
    assert phases['start'] == first and first < phases['end'] <= last
    radius = equipment.BARBELL_BAR_DIAMETER / 2 * height / 1.75
    half_length = equipment.BARBELL_LENGTH / 2 * height / 1.75
    rest = np.array([v.co[:] for v in skin.data.vertices])
    hands = {side: selected(skin, [side + 'Hand']) for side in ('Left', 'Right')}
    soles = {side: selected(skin, [side + 'Foot', side + 'ToeBase']) for side in ('Left', 'Right')}
    plantar = {side: mask & (rest[:, 2] <= float(rest[mask, 2].min()) + .002) for side, mask in soles.items()}
    for side, mask in plantar.items():
        assert mask.sum() >= 6, 'Missing full heel/toe patch for ' + side
    local = np.array([v.co[:] for v in bar.data.vertices])
    for sign in (-1., 1.):
        plate = local[local[:, 0] * sign > .4 * height]
        assert len(plate), 'Missing loaded plate'
        assert np.max(np.linalg.norm(plate[:, 1:], axis=1)) >= .225 * height / 1.75 - .003
    bars, longitudinal, upper, fixed, sole_origins = [], [], [], None, {}
    top_angles = bottom_angles = None
    max_gap = max_inside = max_tilt = max_asymmetry = max_body_drift = max_sole_gap = max_sole_drift = 0.
    for frame in range(first, last + 1):
        scene.frame_set(frame)
        points = skin_wrap.evaluated(skin)[0]
        actual = {name: {side: np.array(rig.matrix_world @ pose[prefix + side + bone].head)
                  for side in ('Left', 'Right')}
                  for name, bone in (('shoulders', 'Arm'), ('elbows', 'ForeArm'), ('wrists', 'Hand'))}
        hip = np.array(rig.matrix_world @ pose[prefix + 'Hips'].head)
        shoulder = np.mean(list(actual['shoulders'].values()), axis=0)
        if fixed is None:
            fixed = {'hip': hip.copy(), **{side: point.copy() for side, point in actual['shoulders'].items()}}
            axis = hip - shoulder; axis[2] = 0.; torso = np.linalg.norm(axis); axis /= torso
            start_shoulder = shoulder.copy()
        max_body_drift = max(max_body_drift, np.linalg.norm(hip - fixed['hip']),
            *(np.linalg.norm(point - fixed[side]) for side, point in actual['shoulders'].items()))
        centre = np.array(bar.matrix_world.translation)
        along = np.array((bar.matrix_world.to_3x3() @ Vector((1., 0., 0.))).normalized())
        assert abs(along[2]) <= 1e-5, 'Bench bar tilts'
        bars.append(centre); longitudinal.append(float((centre - start_shoulder) @ axis))
        angles = {}
        for side in ('Left', 'Right'):
            angles[side] = included(actual['shoulders'][side], actual['elbows'][side], actual['wrists'][side])
            fore = actual['wrists'][side] - actual['elbows'][side]
            tilt = np.degrees(np.arccos(np.clip(fore[2] / np.linalg.norm(fore), -1., 1.)))
            max_tilt = max(max_tilt, tilt)
            gaps = shaft_gaps(points[hands[side]], centre, along, half_length, radius)
            max_gap = max(max_gap, float(abs(gaps).min()))
            max_inside = max(max_inside, float(-gaps.min()))
            patch = points[plantar[side]]
            max_sole_gap = max(max_sole_gap, float(patch[:, 2].max()))
            assert float(patch[:, 2].min()) >= -.003, 'Sole penetrates floor'
            if side not in sole_origins:
                sole_origins[side] = patch.copy()
            max_sole_drift = max(max_sole_drift, float(np.linalg.norm(patch - sole_origins[side], axis=1).max()))
        max_asymmetry = max(max_asymmetry, abs(angles['Left'] - angles['Right']))
        if frame == first:
            top_angles = angles
        if frame == phases['end']:
            bottom_angles = angles
        for bone in pose:
            assert abs(bone.length - bone.bone.length) <= .0001, 'Limb stretching: ' + bone.name
    assert min(top_angles.values()) >= 175., top_angles
    assert 65. <= min(bottom_angles.values()) and max(bottom_angles.values()) <= 105., bottom_angles
    assert max_tilt <= 30., max_tilt
    assert max_asymmetry <= 5., max_asymmetry
    assert max_body_drift <= .002, max_body_drift
    assert max_gap <= .003 and max_inside <= .003, (max_gap, max_inside)
    assert max_sole_gap <= .005 and max_sole_drift <= .002, (max_sole_gap, max_sole_drift)
    low_index = phases['end'] - first
    assert bars[0][2] - bars[low_index][2] >= .12 * height
    assert .2 * torso <= longitudinal[low_index] <= .6 * torso
    assert abs(longitudinal[0]) <= .03 * height
    assert min(longitudinal) >= -.03 * height and max(longitudinal) <= .6 * torso
    # Include held frames in the descent; neither coordinate may reverse before bottom.
    assert max(np.diff(np.array(bars[:low_index + 1])[:, 2])) <= .003
    assert min(np.diff(longitudinal[:low_index + 1])) >= -.003
    assert min(np.diff(np.array(bars[low_index:])[:, 2])) >= -.003
    assert max(np.diff(longitudinal[low_index:])) <= .003
    motion = inspect_run.upper_limb_motion(rig, first, last, loop=True)
    assert motion['max_speed_heights_s'] <= 2., motion
    assert motion['max_turn_deg_s'] <= 720., motion
    print('BENCH_MOTION_REPLAY_PASS ' + json.dumps({'frames': last - first + 1,
        'top_elbows_deg': top_angles, 'bottom_elbows_deg': bottom_angles,
        'bar_travel_mm': (bars[0][2] - bars[low_index][2]) * 1000,
        'max_forearm_tilt_deg': max_tilt, 'max_elbow_asymmetry_deg': max_asymmetry,
        'max_hand_gap_mm': max_gap * 1000, 'max_hand_inside_mm': max_inside * 1000,
        'max_sole_gap_mm': max_sole_gap * 1000, 'max_sole_drift_mm': max_sole_drift * 1000,
        'upper_limb_speed_heights_s': motion['max_speed_heights_s'],
        'upper_limb_turn_deg_s': motion['max_turn_deg_s'], 'loaded_plates': True}, default=float), flush=True)


if __name__ == '__main__':
    main()
