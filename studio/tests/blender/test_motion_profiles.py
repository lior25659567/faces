"""Probe actual reference-driven equipment/support geometry on the cached mascot.

blender --background --python-exit-code 1 --python THIS_FILE -- --run RUN --out OUT [--render]
The input must contain real source keyposes. Without --render this checks geometry
only. Reports preserve failed/unmeasured contacts instead of certifying every profile.
"""
import argparse
import json
import sys
from pathlib import Path
try:
    import bpy
except ImportError:
    import pytest
    pytest.skip('Runs inside Blender with the actual mascot', allow_module_level=True)

import numpy as np
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'pipeline/blender'), str(ROOT / 'studio/measure')]
import mpfb_cage
import retarget
import keypose_clip
from motion_profiles import profile_for
import exercise_contacts
import exercise_supports
from physical_checks import rows_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    keys = json.loads((args.run / 'keyposes.json').read_text())
    profile = profile_for(keys['exercise'])
    assert profile, 'No motion policy for this exercise'
    args.out.mkdir(parents=True, exist_ok=True)
    rest = retarget.load_clip(keypose_clip.REST_CLIP, turned='all')
    skin, rig, _, fit = mpfb_cage.wear(ROOT / '3d-character/3d cartoon boy.glb', args.out)
    leans = [v for v in mpfb_cage.rest_thigh_outlines(rig).values() if v is not None]
    clip = keypose_clip.clip_from_keyposes(rest, keys, fps=12, feet=profile['feet'],
        legs=keypose_clip.leg_rest(sum(leans) / len(leans) if leans else 0.))
    result = retarget.BoneRetarget(rig, skin, clip, exercise=keys['exercise'], feet=profile['feet']).run()
    scene = bpy.context.scene
    first, last = scene.frame_start, scene.frame_end
    props, support = exercise_supports.apply(rig, skin, first, last, keys['exercise'], reference=keys,
        reference_phase_frames=clip.reference_motion['phase_frames'])
    held, contacts = exercise_contacts.apply(rig, skin, first, last, keys['exercise'], pulley_anchor=(support or {}).get('pulley_anchor'))
    props += held
    result['physical'] = {'profile': profile, 'contacts': contacts, 'supports': support}
    result['contact'] = retarget.skin_contact(rig, skin, first, last)
    result['exercise'] = keys['exercise']
    result['source'] = str(args.run / 'keyposes.json')
    result['validation'] = {'rendered': args.render, 'rows': rows_for(result)}
    failures = [r for r in result['validation']['rows'] if r['status'] != 'pass']
    result['validation']['status'] = 'needs-review' if failures else 'physical-checks-passed'
    if profile['mode'] == 'hold':
        roots = []
        for frame in range(first, last + 1):
            scene.frame_set(frame)
            roots.append(np.array(rig.pose.bones[mpfb_cage.P + 'Hips'].location))
        assert np.max(np.linalg.norm(np.array(roots) - roots[0], axis=1)) < 1e-6, 'Static hold root jumps'
    retarget.write_report(args.out / 'report.json', result)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.out / 'scene.blend'))
    if args.render:
        low, high = [np.array(v) for v in result['contact']['box']]
        for frame in range(first, last + 1):
            scene.frame_set(frame)
            for prop in props:
                points = np.array([prop.matrix_world @ Vector(c) for c in prop.bound_box])
                low, high = np.minimum(low, points.min(0)), np.maximum(high, points.max(0))
        centre = (low + high) / 2
        extent = max(float(max(high - low)), .5) * 1.1
        mpfb_cage.preview_set(360, 360, samples=4)
        mpfb_cage.camera_on(centre, 35, 10, mpfb_cage.distance_for(extent))
        scene.render.filepath = str(args.out / 'frame_')
        bpy.ops.render.render(animation=True)
        retarget.frames_to_gif(args.out, args.out / 'animation.gif', scene.render.fps)
    print('PROFILE_RESULT ' + json.dumps({'exercise': keys['exercise'], **result['validation']}), flush=True)


if __name__ == '__main__':
    main()
