"""Replay a real one-shot lunge from source contacts and actual evaluated skin.

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
    pytest.skip('Requires Blender and an actual saved walking-lunge scene', allow_module_level=True)

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'pipeline/blender'), str(ROOT / 'studio/blender'), str(ROOT / 'studio/measure')]
import inspect_run
import lunge_contacts
from lunge_motion import SIDES, samples_from, timeline
from lunge_range import summarize
from checks import _upper_limb_rows
import skin_wrap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    run, scene = args.run, bpy.context.scene
    report = json.loads((run / 'frames/retarget.json').read_text())
    reference = json.loads((run / 'keyposes.json').read_text())
    source = samples_from(reference)
    source_times = [sample['timeSeconds'] for sample in source]
    fps = scene.render.fps / scene.render.fps_base
    first, last = scene.frame_start, scene.frame_end
    times = [source_times[0] + min((frame - first) / fps, source_times[-1] - source_times[0])
             for frame in range(first, last + 1)]
    expected = timeline(source, times)
    assert report['reference_motion']['mode'] == 'sequence'
    assert report['reference_motion']['playback'] == 'once'
    assert expected[-1]['foot_states']['left'] == 'airborne', 'The observed ending is a swing-through, not a two-foot stand.'
    rig, skin = inspect_run.find_rig_and_skin()
    _, feet, patches, _ = lunge_contacts._rest_geometry(rig, skin)
    height = json.loads(rig['skin_fit'])['measured']['height']
    lengths = {side: [rig.data.bones['mixamorig:' + side.title() + part].length
                     for part in ('UpLeg', 'Leg')] for side in SIDES}
    current, readings = {}, []
    for index, frame in enumerate(range(first, last + 1)):
        scene.frame_set(frame)
        points = skin_wrap.evaluated(skin)[0]
        contacts = {name: points[selected].mean(axis=0).tolist() for name, selected in patches.items()}
        active = set(expected[index]['contacts'])
        current = {name: current[name] for name in current.keys() & active}
        # Initialize an observed contact episode from its actual skin position,
        # independently of any saved target or solver summary.
        for name in active - current.keys():
            current[name] = contacts[name].copy()
            current[name][2] = 0.
        world, pose = rig.matrix_world, rig.pose.bones
        readings.append({'frame': frame, 'time_s': times[index], 'foot_states': expected[index]['foot_states'],
            'contact_points_m': contacts, 'contact_anchors_m': {name: point.copy() for name, point in current.items()},
            'contact_floor': {name: {'min_z_m': float(points[mask, 2].min()), 'max_z_m': float(points[mask, 2].max())}
                              for name, mask in patches.items()},
            'foot_min_z_m': {side: float(points[mask, 2].min()) for side, mask in feet.items()},
            'hips_m': {side: list(world @ pose['mixamorig:' + side.title() + 'UpLeg'].head) for side in SIDES},
            'knees_m': {side: list(world @ pose['mixamorig:' + side.title() + 'Leg'].head) for side in SIDES},
            'ankles_m': {side: list(world @ pose['mixamorig:' + side.title() + 'Foot'].head) for side in SIDES},
            'leg_lengths_m': lengths, 'body_min_z_m': float(points[:, 2].min()),
            'root_m': np.mean([np.array(world @ pose['mixamorig:' + side.title() + 'UpLeg'].head) for side in SIDES], axis=0).tolist()})
    measured = summarize(readings, height, report['reference_motion']['phase_frames'], source_times, fps)
    limits = {'max_contact_gap_mm': 5., 'max_contact_penetration_mm': 3., 'max_anchor_drift_mm': 3.,
        'max_swing_clearance_shortfall_mm': 0., 'max_leg_length_error_mm': .1, 'max_backward_knee_mm': 3.,
        'max_body_floor_penetration_mm': 3., 'missing_forward_travel': 0., 'max_reverse_root_mm': 3.,
        'start_extension_shortfall_deg': 0., 'middle_knee_overfold_deg': 0., 'middle_knee_depth_shortfall_deg': 0.}
    for name, limit in limits.items():
        assert measured[name] <= limit, (name, measured[name], limit)
    arms = inspect_run.upper_limb_motion(rig, first, last, loop=False)
    arm_rows = _upper_limb_rows({'first': first, 'last': last, 'upper_limb_motion': arms})
    assert all(row['status'] == 'pass' for row in arm_rows), arm_rows
    gif = (run / 'animation.gif').read_bytes()
    assert gif[:6] in (b'GIF87a', b'GIF89a')
    assert b'NETSCAPE2.0' not in gif and b'ANIMEXTS1.0' not in gif, 'A walking step must not loop back to its start.'
    checks = json.loads((run / 'checks.json').read_text())
    assert checks['summary']['fail'] == 0 and checks['summary']['unmeasured'] == 0, checks['summary']
    measured.update(status='passed', rendered=False, source_inferred_depth=True,
        source_video_sha256='3d83e21334e0119fc831609c7581af02ec30642f1ebedf514d951ca25fd4d154',
        scene_sha256=hashlib.sha256(Path(bpy.data.filepath).read_bytes()).hexdigest(),
        arm_continuity_rows=arm_rows, gif_repeats=False)
    destination = args.out or run / 'lunge-contacts-replay.json'
    destination.write_text(json.dumps(measured, indent=2, allow_nan=False))
    print('LUNGE_CONTACT_SCENE_PASS ' + str(destination), flush=True)


if __name__ == '__main__':
    main()
