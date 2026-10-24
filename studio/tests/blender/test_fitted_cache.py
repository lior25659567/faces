"""Fast real-Blender library cache regression; no expensive fitting or external assets.

    blender --background --python-exit-code 1 --python studio/tests/blender/test_fitted_cache.py

This structural fixture covers library API behavior. The actual mascot cold/warm
fit and rendered-row validation additionally exercise real geometry and hand models.
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

try:
    import bpy
except ImportError:
    import pytest
    pytest.skip('Runs inside Blender', allow_module_level=True)
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'pipeline/blender'))
import fitted_cache


def main():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    scene = bpy.context.scene
    data = bpy.data.armatures.new('rig_data')
    rig = bpy.data.objects.new('rig', data)
    scene.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    names = ['Hips', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head']
    for side in ('Left', 'Right'):
        names += [side + name for name in ('Shoulder', 'Arm', 'ForeArm', 'UpLeg', 'Leg', 'Foot', 'ToeBase')]
        names += [side + 'Hand'] + [side + 'Hand' + finger + str(n)
                   for finger in ('Thumb', 'Index', 'Middle', 'Ring', 'Pinky') for n in (1, 2, 3)]
    assert len(names) == 52
    for n, name in enumerate(names):
        bone = data.edit_bones.new('mixamorig:' + name)
        bone.head, bone.tail = (0, 0, n * .01), (0, 0, n * .01 + .009)
    bpy.ops.object.mode_set(mode='OBJECT')
    rig['rig_profile'] = 'mpfb-mixamo'
    rig['skin_fit'] = json.dumps({'hands': {}, 'measured': {'height': 1}})
    bpy.ops.mesh.primitive_cube_add(size=1)
    skin = bpy.context.object; skin.name = 'skin'; skin.parent = rig
    group = skin.vertex_groups.new(name='mixamorig:Hips')
    group.add(list(range(len(skin.data.vertices))), 1, 'REPLACE')
    modifier = skin.modifiers.new('bind', 'ARMATURE'); modifier.object = rig
    skin['skin_bake'] = json.dumps({'vertices': len(skin.data.vertices)})
    material = bpy.data.materials.new('cache-test-material'); material.diffuse_color = (.2, .4, .6, 1)
    skin.data.materials.append(material)
    bpy.ops.mesh.primitive_cube_add(size=.5)
    cage = bpy.context.object; cage.name = 'cage'; cage.hide_render = cage.hide_viewport = True
    before = np.array([tuple(v.co) for v in skin.data.vertices])
    folder = Path(tempfile.mkdtemp(prefix='fitted-library-test-'))
    entry_folder = folder / 'test-key'
    try:
        fitted_cache._store(entry_folder, 'test-key', (skin, rig, cage), {'bake': {'vertices': len(before)}})
        entry = fitted_cache.read_manifest(entry_folder, 'test-key')
        original_names = entry['objects'][:]
        # Name collisions and a currently posed caller must not taint the stored rest fit.
        rig.pose.bones['mixamorig:LeftArm'].rotation_mode = 'XYZ'
        rig.pose.bones['mixamorig:LeftArm'].rotation_euler.x = .7
        scene.frame_set(37); scene.render.resolution_x = 321; scene.render.resolution_y = 287
        original_world, original_path = scene.world, bpy.data.filepath
        loaded_skin, loaded_rig, loaded_cage, report = fitted_cache._load(entry_folder, entry)
        # libraries.load mutates target.objects in-place; the manifest's strings must survive.
        assert entry['objects'] == original_names == ['skin', 'rig', 'cage']
        assert [loaded_skin.name, loaded_rig.name, loaded_cage.name] == original_names
        assert set(obj.name for obj in scene.objects) == set(original_names)
        assert loaded_skin.parent is loaded_rig
        assert loaded_skin.modifiers['bind'].object is loaded_rig
        assert scene.frame_current == 37 and (scene.render.resolution_x, scene.render.resolution_y) == (321, 287)
        assert bpy.data.filepath == original_path and scene.world is original_world
        assert report['bake']['vertices'] == len(before)
        np.testing.assert_allclose(np.array([tuple(v.co) for v in loaded_skin.data.vertices]), before)
        np.testing.assert_allclose(loaded_skin.data.materials[0].diffuse_color, (.2, .4, .6, 1), atol=1e-6)
        assert len(loaded_rig.data.bones) == 52
        assert loaded_cage.hide_render and loaded_cage.hide_viewport
        loaded_rig.pose.bones['mixamorig:LeftArm'].rotation_mode = 'XYZ'
        loaded_rig.pose.bones['mixamorig:LeftArm'].rotation_euler.x = .4
        try:
            fitted_cache._store(folder / 'posed', 'posed', (loaded_skin, loaded_rig, loaded_cage), report)
        except ValueError as error:
            assert 'Posed' in str(error)
        else:
            raise AssertionError('A posed character must not become a reusable rest entry')
        assert not (folder / 'posed').exists()
        print('test_fitted_cache: PASS (real library roundtrip, names, binding, scene state, posed rejection)')
    finally:
        shutil.rmtree(folder)


if __name__ == '__main__':
    main()
