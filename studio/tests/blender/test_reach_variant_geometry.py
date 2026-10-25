"""Real cached-mascot rest/closed-hand invariants for the fixed reach variant.

blender --background --python-exit-code 1 --python THIS_FILE
No original/cache asset is saved or replaced. Reference-driven ROM and rendered
validation are separate from these exact mesh/bone/grip invariants.
"""
import hashlib
import json
from pathlib import Path
import sys
try:
    import bpy
except ImportError:
    import pytest
    pytest.skip('Requires Blender with the fitted mascot', allow_module_level=True)
import numpy as np
from mathutils import Matrix

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'pipeline/blender'))
import fitted_cache
import hand_rig
import mpfb_cage
import reach_variant
import skin_wrap


def coords(skin):
    return np.array([vertex.co[:] for vertex in skin.data.vertices])


def unpose(rig):
    for pb in rig.pose.bones:
        pb.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()


def close_hands(rig):
    grips = json.loads(rig['hand_grip'])
    for side in ('left', 'right'):
        for name, rotation in hand_rig.pose_basis(rig, side.title(), grips[side]['rotations']).items():
            rig.pose.bones[name].rotation_mode = 'QUATERNION'
            rig.pose.bones[name].rotation_quaternion = rotation
    bpy.context.view_layer.update()


def main():
    source = ROOT / '3d-character/3d cartoon boy.glb'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    cached = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in fitted_cache.cache_directory().glob('*') if p.is_file()}
    skin, rig, _, _ = mpfb_cage.wear(source, '/tmp/reach-variant-invariants')
    old = coords(skin)
    weights = [tuple((g.group, g.weight) for g in vertex.groups) for vertex in skin.data.vertices]
    old_grips = json.loads(rig['hand_grip'])
    head_groups = {g.index for g in skin.vertex_groups if g.name == 'mixamorig:Head'}
    face = np.array([sum(g.weight for g in vertex.groups if g.group in head_groups) > .05 for vertex in skin.data.vertices])
    hands = {}
    old_fingers = {}
    for side in ('left', 'right'):
        groups = {g.index for g in skin.vertex_groups if g.name.startswith('mixamorig:' + side.title() + 'Hand')}
        hands[side] = np.array([sum(g.weight for g in vertex.groups if g.group in groups) > .99 for vertex in skin.data.vertices])
        old_fingers[side] = {b.name: np.array(b.tail_local - b.head_local) for b in rig.data.bones if b.name.startswith('mixamorig:' + side.title() + 'Hand')}
    original_reach = reach_variant.measure_reach(rig, skin)
    assert min(s['head_shortfall_mm'] for s in original_reach['sides'].values()) > 60.
    close_hands(rig)
    old_closed = skin_wrap.evaluated(skin)[0]
    unpose(rig)
    report = reach_variant.apply_if_needed(rig, skin, 'Dumbbell Shoulder Press (seated overhead press)', 'auto')
    assert report['upper_scale'] == report['fore_scale'] == 1.8
    assert report['face_max_displacement_mm'] == 0.
    assert np.array_equal(coords(skin)[face], old[face]), 'Face vertices changed'
    assert [tuple((g.group, g.weight) for g in v.groups) for v in skin.data.vertices] == weights
    for side in ('left', 'right'):
        for name, direction in old_fingers[side].items():
            bone = rig.data.bones[name]
            assert np.linalg.norm(np.array(bone.tail_local - bone.head_local) - direction) < 1e-6, name
    new_grips = json.loads(rig['hand_grip'])
    for side in ('left', 'right'):
        for key in ('kb', 'hb', 'n', 'rotations'):
            assert new_grips[side][key] == old_grips[side][key], key
        before = np.array(old_grips[side]['B']) - old_grips[side]['wrist']
        after = np.array(new_grips[side]['B']) - new_grips[side]['wrist']
        assert np.linalg.norm(before - after) < 1e-9
    close_hands(rig)
    new_closed = skin_wrap.evaluated(skin)[0]
    errors = {}
    for side in ('left', 'right'):
        delta = np.array(report['sides'][side]['wrist_translation_m'])
        errors[side] = float(np.linalg.norm(new_closed[hands[side]] - old_closed[hands[side]] - delta, axis=1).max() * 1000)
        assert errors[side] < .01, errors
    unpose(rig)
    again = reach_variant.apply_if_needed(rig, skin, 'Pull-Up', 'auto')
    assert again == report, 'The selection wrapper compounded the variant'
    try:
        reach_variant.apply(rig, skin)
        raise AssertionError('Direct duplicate deformation was accepted')
    except ValueError:
        pass
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    for file, before in cached.items():
        assert (file.stat().st_size, file.stat().st_mtime_ns) == before, f'Cache modified: {file}'
    assert not (rig.animation_data and rig.animation_data.action)
    print('REACH_INVARIANTS_PASS ' + json.dumps({'face_delta_mm': 0., 'closed_hand_shape_error_mm': errors,
          'source_hash_unchanged': True, 'cache_files_unchanged': True, 'original': original_reach,
          'actual': reach_variant.measure_reach(rig, skin)}), flush=True)


if __name__ == '__main__':
    main()
