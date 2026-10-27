"""Independently replay support contacts on a saved, actually built character.

blender --background scene.blend --python-exit-code 1 --python THIS_FILE
The scene must be a bench, plank, or leg-press result. This checks evaluated mesh
positions on every frame, rather than trusting the saved contact report.
"""
import json
import sys
from pathlib import Path

try:
    import bpy
except ImportError:
    import pytest
    pytest.skip('Requires an actual built Blender scene', allow_module_level=True)

import numpy as np
from mathutils import Vector
sys.path[:0] = [str(Path(__file__).resolve().parents[3] / 'pipeline/blender')]
import skin_wrap


def mask(skin, names):
    groups = {group.index for group in skin.vertex_groups
              if any(group.name.endswith(name) for name in names)}
    selected = np.array([sum(group.weight for group in vertex.groups if group.group in groups) > .5
                         for vertex in skin.data.vertices])
    assert selected.any(), 'No measurable skin vertices for ' + ', '.join(names)
    return selected


def near_plane(points, selected, plane, name, half=None, centre=None, below=False):
    sample = points[selected]
    if half is not None:
        sample = sample[np.all(abs(sample[:, :2] - centre) <= half - .001, axis=1)]
    assert len(sample), name + ' has no vertices within the finite support'
    gap = float(abs(sample[:, 2] - plane).min())
    depth = float((sample[:, 2] - plane).max() if below else (plane - sample[:, 2]).max())
    assert gap <= .005, f'{name} gap {gap * 1000:.3f} mm'
    assert depth <= .003, f'{name} penetration {depth * 1000:.3f} mm'
    return gap, max(0., depth)


def main():
    scene = bpy.context.scene
    rig = next(obj for obj in scene.objects if obj.type == 'ARMATURE' and 'skin_fit' in obj)
    skin = next(obj for obj in scene.objects if obj.type == 'MESH' and not obj.hide_render
                and any(mod.type == 'ARMATURE' and mod.object == rig for mod in obj.modifiers))
    is_sled = 'leg_press_footplate' in scene.objects
    is_bench = 'body_support_pad' in scene.objects and not is_sled
    soles = {side: mask(skin, [side + 'Foot', side + 'ToeBase']) for side in ('Left', 'Right')}
    rest_points = np.array([vertex.co for vertex in skin.data.vertices])
    plantar = {side: selected & (rest_points[:, 2] <= float(rest_points[selected, 2].min()) + .002)
               for side, selected in soles.items()}
    starts, path, readings, sole_slide = {}, [], [], 0.
    rest_lengths = {bone.name: bone.length for bone in rig.data.bones}
    pose_length_error = 0.
    fixed_supports = {}
    plantar_gap = 0.
    plate_turn = None
    for frame in range(scene.frame_start, scene.frame_end + 1):
        scene.frame_set(frame)
        points = skin_wrap.evaluated(skin)[0]
        for name in ('body_support_pad', 'leg_press_backrest', 'leg_press_headrest'):
            if name not in scene.objects:
                continue
            matrix = np.array(scene.objects[name].matrix_world)
            if name not in fixed_supports:
                fixed_supports[name] = matrix.copy()
            assert np.allclose(matrix, fixed_supports[name], atol=1e-6, rtol=0.), name + ' moved to follow the body'
        assert float(points[:, 2].min()) >= -.003, f'Floor penetration in frame {frame}'
        for bone in rig.pose.bones:
            assert bone.bone.length == rest_lengths[bone.name], 'Support solver changed the rest skeleton'
            error = abs(bone.length - bone.bone.length)
            pose_length_error = max(pose_length_error, error)
            # Float matrix/keyframe conversions accumulate micrometre error;
            # 0.1 mm is still far below the 3 mm contact tolerance.
            assert error <= .0001, 'Support solver stretched ' + bone.name
        if is_sled:
            assert 'leg_press_backrest' in scene.objects, 'A reclined leg press needs a fixed back/head support'
            assert 'leg_press_headrest' in scene.objects, 'The mascot needs its measured fixed headrest'
            for names, pad_name in ((['Spine1', 'Spine2'], 'leg_press_backrest'), (['Head'], 'leg_press_headrest')):
                backrest = scene.objects[pad_name]
                back_local = np.array([backrest.matrix_world.inverted() @ Vector(point) for point in points])
                back_bounds = np.array([point for point in backrest.bound_box])
                back_half = (back_bounds.max(0) - back_bounds.min(0))[:2] / 2
                readings.append(near_plane(back_local, mask(skin, names), float(back_bounds[:, 2].max()),
                    '/'.join(names) + ' backrest', back_half, np.zeros(2)))
            plate = scene.objects['leg_press_footplate']
            local = np.array([plate.matrix_world.inverted() @ Vector(point) for point in points])
            bounds = np.array([point for point in plate.bound_box])
            half = (bounds.max(0) - bounds.min(0))[:2] / 2
            plane = float(bounds[:, 2].min())
            quaternion = plate.matrix_world.to_quaternion()
            if plate_turn is None:
                plate_turn = quaternion.copy()
            assert abs(quaternion.rotation_difference(plate_turn).angle) <= 1e-5, 'Footplate changed its rail orientation'
            path.append(np.array(plate.matrix_world.translation))
            for side, selected in soles.items():
                readings.append(near_plane(local, selected, plane, side + ' footplate', half, np.zeros(2), below=True))
                patch = local[plantar[side]]
                assert len(patch) >= 6, 'No readable plantar heel/toe patch'
                plantar_gap = max(plantar_gap, float((plane - patch[:, 2]).max()))
                assert float((plane - patch[:, 2]).max()) <= .005, side + ' heel or toe lifted from the footplate'
                assert float((patch[:, 2] - plane).max()) <= .003, side + ' heel or toe penetrated the footplate'
                if side not in starts:
                    indices = np.flatnonzero(plantar[side])
                    assert len(indices) >= 3, 'No readable sole patch on the footplate'
                    starts[side] = (indices, local[indices].copy())
                indices, origin = starts[side]
                sole_slide = max(sole_slide, float(np.linalg.norm(local[indices] - origin, axis=1).max()))
        elif is_bench:
            pad = scene.objects['body_support_pad']
            corners = np.array([pad.matrix_world @ Vector(point) for point in pad.bound_box])
            low, high = corners.min(0), corners.max(0)
            for names in (['Hips'], ['Spine1', 'Spine2'], ['Head']):
                readings.append(near_plane(points, mask(skin, names), float(high[2]), '/'.join(names),
                    (high - low)[:2] / 2, (low + high)[:2] / 2))
            for side, selected in soles.items():
                patch = points[plantar[side]]
                assert len(patch) >= 6, 'No readable plantar heel/toe patch'
                plantar_gap = max(plantar_gap, float(abs(patch[:, 2]).max()))
                assert float(patch[:, 2].max()) <= .005, side + ' heel or toe lifted from the floor'
                assert float(patch[:, 2].min()) >= -.003, side + ' heel or toe penetrated the floor'
                if side not in starts:
                    indices = np.flatnonzero(selected & (points[:, 2] <= .002))
                    assert len(indices) >= 3, 'No readable sole patch on the floor'
                    starts[side] = (indices, points[indices].copy())
                indices, origin = starts[side]
                sole_slide = max(sole_slide, float(np.linalg.norm(points[indices] - origin, axis=1).max()))
        else:
            for side in ('Left', 'Right'):
                for name in ('ForeArm', 'ToeBase'):
                    selected = mask(skin, [side + name])
                    readings.append(near_plane(points, selected, 0., side + name))
                    key = side + name
                    if key not in starts:
                        starts[key] = points[selected].copy()
                    sole_slide = max(sole_slide, float(np.linalg.norm(points[selected] - starts[key], axis=1).max()))
    if is_sled:
        path = np.array(path)
        centre = path.mean(0)
        _, _, axes = np.linalg.svd(path - centre)
        offsets = path - centre
        residual = np.linalg.norm(offsets - np.outer(offsets @ axes[0], axes[0]), axis=1)
        assert residual.max() <= .001, 'Footplate did not follow one straight rail'
    assert sole_slide <= .002, f'Soles slid {sole_slide * 1000:.3f} mm on their support'
    print('SUPPORT_REPLAY_PASS ' + json.dumps({'frames': scene.frame_end - scene.frame_start + 1,
        'support': 'sled' if is_sled else 'bench' if is_bench else 'plank',
        'max_gap_mm': max(value[0] for value in readings) * 1000,
        'max_penetration_mm': max(value[1] for value in readings) * 1000,
        'sole_slide_mm': sole_slide * 1000,
        'max_plantar_patch_gap_mm': plantar_gap * 1000,
        'max_pose_length_error_mm': pose_length_error * 1000}), flush=True)


if __name__ == '__main__':
    main()
