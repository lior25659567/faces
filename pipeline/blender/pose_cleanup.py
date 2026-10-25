"""Rest-preserving local garment cleanup for reference-driven exercise poses.

The baked mascot has duplicate vertices at UV seams. Smoothing those independently
opens cracks; welding the coincident seam vertices first preserves the UV loops and
lets the corrective surface span the actual garment. The rest-bound modifier has
zero displacement in the bind pose and affects only the collar/shoulder region.

This is a Blender surface correction, not a change to a fold threshold, camera or
the original GLB/cache. Call it for a validated Studio blend/GIF profile only; a
skeletal-only export cannot silently discard its post-skin deformation.
"""
import json

import numpy as np

MARKER = 'studio_collar_surface_cleanup'
MODIFIER = 'Studio collar surface cleanup'
GROUP = 'studio_collar_surface_region'
SCHEMA = 1


def collar_weights(points, centres, height, protected=None):
    """Smooth local influence with no movement on the face, hands or distant skin."""
    points = np.asarray(points, dtype=float)
    centres = np.asarray(centres, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or centres.shape != (2, 3):
        raise ValueError('Collar cleanup needs 3D skin points and both clavicle roots.')
    if not np.isfinite(points).all() or not np.isfinite(centres).all() or not np.isfinite(height) or height <= 0:
        raise ValueError('Collar cleanup geometry must be finite and have positive height.')
    distance = np.linalg.norm(points[:, None, :] - centres[None, :, :], axis=2).min(axis=1) / height
    amount = np.clip((.20 - distance) / .10, 0., 1.)
    amount = amount * amount * (3. - 2. * amount)
    if protected is not None:
        protected = np.asarray(protected, dtype=bool)
        if protected.shape != (len(points),):
            raise ValueError('The protected skin mask does not match its points.')
        amount[protected] = 0.
    return amount


def arm_emphasis(weights, arm_share, factor=1.5):
    """Increase outer-arm smoothing while reducing inner-collar group influence.

    The smooth arm-share ramp counterbalances the stronger modifier with less
    group influence on the inner collar. This does not claim identical posed
    collar coordinates: the complete clip still needs actual fold/identity checks.
    """
    weights, arm_share = np.asarray(weights, dtype=float), np.asarray(arm_share, dtype=float)
    if weights.shape != arm_share.shape or not np.isfinite(weights).all() or not np.isfinite(arm_share).all():
        raise ValueError('Arm correction needs matching finite skin influences.')
    if not np.isfinite(factor) or factor < 1.:
        raise ValueError('Arm correction strength must be finite and at least one.')
    amount = np.clip((arm_share - .25) / .30, 0., 1.)
    amount = amount * amount * (3. - 2. * amount)
    return weights * (1. + (factor - 1.) * amount) / factor


def _positions(mesh, world):
    values = np.empty(len(mesh.vertices) * 3, dtype=float)
    mesh.vertices.foreach_get('co', values)
    values = values.reshape(-1, 3)
    transform = np.asarray(world, dtype=float)
    return values @ transform[:3, :3].T + transform[:3, 3]


def _max_nearest_distance(points, candidates):
    from mathutils import Vector
    from mathutils.kdtree import KDTree
    tree = KDTree(len(candidates))
    for index, point in enumerate(candidates):
        tree.insert(Vector(point), index)
    tree.balance()
    return max((tree.find(Vector(point))[2] for point in points), default=0.)


def apply(rig, skin, first=None, last=None, log=print, exercise=None):
    """Weld coincident seams and add the validated local rest-bound correction.

    Bone transforms, weights, UV loops and material assignments remain intact.
    The mesh is copied before mutation; the source file and cached library are
    untouched. Returns measurement/provenance for the run report.
    """
    import bmesh
    import bpy
    import skin_wrap

    name = str(exercise or '').casefold().replace('-', ' ')
    arm_factor = 1.5 if 'standing barbell overhead press' in name else 1.
    arm_sides = ('Left', 'Right')
    if 'dumbbell shoulder press' in name or 'lat pulldown' in name:
        arm_factor = 2.
    if 'lat pulldown' in name:
        arm_sides = ('Right',)
    iterations = 30 if 'dumbbell shoulder press' in name else 20

    if skin.get(MARKER):
        report = json.loads(skin[MARKER])
        modifier = skin.modifiers.get(MODIFIER)
        if report.get('schema') != SCHEMA or not modifier or not modifier.is_bind:
            raise ValueError('Existing collar cleanup metadata does not match a bound modifier.')
        if abs(modifier.factor - report.get('factor', -1.)) > 1e-6:
            raise ValueError('Existing collar cleanup strength differs from its recorded settings.')
        if modifier.iterations != report.get('iterations'):
            raise ValueError('Existing collar cleanup iterations differ from their recorded settings.')
        if exercise is not None and abs(modifier.factor - arm_factor) > 1e-6:
            raise ValueError('Existing collar cleanup belongs to a different exercise correction.')
        if exercise is not None and modifier.iterations != iterations:
            raise ValueError('Existing collar cleanup belongs to a different exercise refinement.')
        if exercise is not None and arm_factor > 1. and (report.get('arm_emphasis') or {}).get('sides') != [side.lower() for side in arm_sides]:
            raise ValueError('Existing collar cleanup has a different arm correction region.')
        return report
    if skin.data.shape_keys:
        raise ValueError('Collar seam cleanup does not support a mesh with existing shape keys.')
    if skin.modifiers.get(MODIFIER) or skin.vertex_groups.get(GROUP):
        raise ValueError('The collar cleanup names are already used by an unrelated modifier or vertex group.')
    armatures = [m for m in skin.modifiers if m.type == 'ARMATURE' and m.object == rig]
    if len(armatures) != 1:
        raise ValueError('Collar cleanup requires exactly one armature driving the skin.')

    original = skin.data
    mesh = original.copy()
    held_pose = rig.data.pose_position
    held_active = bpy.context.view_layer.objects.active
    held_frame = bpy.context.scene.frame_current
    group = modifier = None
    skin.data = mesh
    try:
        rig.data.pose_position = 'REST'
        bpy.context.view_layer.update()
        before = _positions(mesh, skin.matrix_world)
        height = float(np.ptp(before[:, 2]))
        if not np.isfinite(height) or height <= 0:
            raise ValueError('The skin has no finite bind-pose height.')
        seam_tolerance = height * 1e-5
        uv = {}
        for layer in mesh.uv_layers:
            values = np.empty(len(layer.data) * 2, dtype=float)
            layer.data.foreach_get('uv', values)
            uv[layer.name] = values
        material = np.array([p.material_index for p in mesh.polygons])
        deform_weights = [tuple((g.group, g.weight) for g in vertex.groups) for vertex in mesh.vertices]
        corner_normals = None
        encoded_normals = None
        if mesh.has_custom_normals:
            corner_normals = np.empty(len(mesh.corner_normals) * 3, dtype=float)
            mesh.corner_normals.foreach_get('vector', corner_normals)
            corner_normals = corner_normals.reshape(-1, 3)
            encoded = mesh.attributes.get('custom_normal')
            if encoded is not None and encoded.data_type == 'INT16_2D' and encoded.domain == 'CORNER':
                encoded_normals = np.empty(len(encoded.data) * 2, dtype=np.int32)
                encoded.data.foreach_get('value', encoded_normals)
                encoded_normals = encoded_normals.reshape(-1, 2)
        loop_vertices = np.array([loop.vertex_index for loop in mesh.loops])
        loop_points = before[loop_vertices]
        centres = np.array([rig.matrix_world @ rig.data.bones['mixamorig:' + side + 'Shoulder'].head_local
                            for side in ('Left', 'Right')])
        protect_groups = {g.index for g in skin.vertex_groups if 'Head' in g.name or 'Hand' in g.name}
        protected_before = np.array([sum(g.weight for g in v.groups if g.group in protect_groups) > .5
                                     for v in mesh.vertices])
        # Only join the surface that the correction can touch. A whole-mesh
        # weld changes normal-fan coordinate frames on the authored face/hands,
        # even when reinstalling their original custom corner normals. Include
        # a zero-weight halo so the correction's neighbourhood is continuous.
        distance = np.linalg.norm(before[:, None, :] - centres[None, :, :], axis=2).min(axis=1)
        weld_region = (distance < height * .25) & ~protected_before
        loops_before, faces_before = len(mesh.loops), len(mesh.polygons)
        bm = bmesh.new()
        try:
            bm.from_mesh(mesh)
            bm.verts.ensure_lookup_table()
            bmesh.ops.remove_doubles(bm, verts=[bm.verts[int(i)] for i in np.flatnonzero(weld_region)], dist=seam_tolerance)
            bm.to_mesh(mesh)
        finally:
            bm.free()
        mesh.update()
        if len(mesh.loops) != loops_before or len(mesh.polygons) != faces_before:
            raise ValueError('Seam welding changed faces or UV-loop topology.')
        if set(uv) != {layer.name for layer in mesh.uv_layers}:
            raise ValueError('Seam welding changed the UV layers.')
        for layer in mesh.uv_layers:
            values = np.empty(len(layer.data) * 2, dtype=float)
            layer.data.foreach_get('uv', values)
            if not np.array_equal(values, uv[layer.name]):
                raise ValueError('Seam welding changed the UV coordinates.')
        if not np.array_equal(material, [p.material_index for p in mesh.polygons]):
            raise ValueError('Seam welding changed material assignments.')
        after = _positions(mesh, skin.matrix_world)
        new_loop_vertices = np.array([loop.vertex_index for loop in mesh.loops])
        after_loops = after[new_loop_vertices]
        if np.linalg.norm(after_loops - loop_points, axis=1).max() > seam_tolerance * 1.01:
            raise ValueError('Seam welding reordered the surface corners.')
        joined_weights = [tuple((g.group, g.weight) for g in vertex.groups) for vertex in mesh.vertices]
        if any(deform_weights[a] != joined_weights[b] for a, b in zip(loop_vertices, new_loop_vertices)):
            raise ValueError('Seam welding changed authored skin weights.')
        # BMesh preserves corner UVs but changes the loop-normal coordinate
        # frames at a merged seam. Reinstall the original custom normals in the
        # unchanged loop order so face/hair shading retains its authored shape.
        if corner_normals is not None:
            mesh.normals_split_custom_set(corner_normals.tolist())
            if encoded_normals is not None:
                # Reuse authored encoded normals where the normal-fan frame
                # remains identical. Round-tripping through a float vector
                # would otherwise add quantisation error even on a face vertex
                # with completely unchanged geometry/topology.
                encoded = mesh.attributes.get('custom_normal')
                values = np.empty(len(encoded.data) * 2, dtype=np.int32)
                encoded.data.foreach_get('value', values)
                values = values.reshape(-1, 2)
                converted = values.copy()
                untouched = ~weld_region[loop_vertices]
                values[untouched] = encoded_normals[untouched]
                encoded.data.foreach_set('value', values.reshape(-1))
                mesh.update()
                normals_after = np.empty(len(mesh.corner_normals) * 3, dtype=float)
                mesh.corner_normals.foreach_get('vector', normals_after)
                same_frame = np.all(normals_after.reshape(-1, 3) == corner_normals, axis=1)
                protected_corners = protected_before[loop_vertices]
                if not np.all(same_frame[protected_corners]):
                    raise ValueError('Seam cleanup changed authored face or hand custom normals.')
                # A rest-space fitness variant may have already changed an
                # arm fan's geometry. Its old encoded normal is then no longer
                # valid after BMesh rebuilds the normal coordinate frame, even
                # on the unjoined side of the weld boundary. Keep the authored
                # vector re-encoded in that new frame instead. Protected face
                # and hand fans must still retain exact original vectors.
                invalid_frame = untouched & ~same_frame
                values[invalid_frame] = converted[invalid_frame]
                encoded.data.foreach_set('value', values.reshape(-1))
            mesh.update()
            checked_normals = np.empty(len(mesh.corner_normals) * 3, dtype=float)
            mesh.corner_normals.foreach_get('vector', checked_normals)
            checked_normals = checked_normals.reshape(-1, 3)
            if not np.array_equal(checked_normals[protected_before[loop_vertices]], corner_normals[protected_before[loop_vertices]]):
                raise ValueError('Seam cleanup changed authored face or hand custom normals.')
        rest_drift = _max_nearest_distance(before, after)
        if rest_drift > seam_tolerance * 1.01:
            raise ValueError('Seam welding moved the skin outside its seam tolerance.')
        protected = np.array([sum(g.weight for g in v.groups if g.group in protect_groups) > .5
                              for v in mesh.vertices])
        weights = collar_weights(after, centres, height, protected)
        if arm_factor > 1.:
            arm_groups = {g.index for g in skin.vertex_groups if g.name in {'mixamorig:' + side + 'Arm' for side in arm_sides}}
            arm_share = np.array([sum(g.weight for g in v.groups if g.group in arm_groups)
                                  for v in mesh.vertices])
            weights = arm_emphasis(weights, arm_share, arm_factor)
        group = skin.vertex_groups.new(name=GROUP)
        for index in np.flatnonzero(weights > 0):
            group.add([int(index)], float(weights[index]), 'REPLACE')
        modifier = skin.modifiers.new(MODIFIER, 'CORRECTIVE_SMOOTH')
        modifier.vertex_group = group.name
        modifier.rest_source = 'BIND'
        modifier.smooth_type = 'LENGTH_WEIGHTED'
        modifier.use_pin_boundary = False
        modifier.factor = arm_factor
        modifier.iterations = iterations
        bpy.context.view_layer.objects.active = skin
        bpy.context.view_layer.update()
        bpy.ops.object.correctivesmooth_bind(modifier=modifier.name)
        if not modifier.is_bind:
            raise ValueError('The collar surface correction could not bind in the rest pose.')
        rest_after = skin_wrap.evaluated(skin)[0]
        bound_drift = float(np.linalg.norm(rest_after - after, axis=1).max())
        if bound_drift > height * 1e-5:
            raise ValueError('The corrective surface changed the bind-pose character.')
        report = {'schema': SCHEMA, 'method': 'coincident seam weld and local rest-bound corrective surface',
            'vertices_before': len(before), 'vertices_after': len(after),
            'faces': faces_before, 'uv_loops': loops_before, 'uv_and_materials_preserved': True,
            'skin_weights_preserved': True,
            'protected_custom_normals_preserved': True,
            'seam_tolerance_mm': seam_tolerance * 1000, 'max_weld_displacement_mm': rest_drift * 1000,
            'max_bind_displacement_mm': bound_drift * 1000,
            'corrected_vertices': int(np.count_nonzero(weights)), 'protected_vertices': int(protected.sum()),
            'factor': modifier.factor, 'iterations': modifier.iterations,
            'arm_emphasis': {'factor': arm_factor, 'influence_ramp': [.25, .55],
                             'sides': [side.lower() for side in arm_sides],
                             'reason': 'outer-arm correction for overhead movement with reduced inner-collar influence'} if arm_factor > 1. else None,
            'export': 'post-skin Blender surface correction; skeletal-only exports must not discard it'}
        skin[MARKER] = json.dumps(report)
    except Exception:
        if modifier:
            skin.modifiers.remove(modifier)
        if group:
            skin.vertex_groups.remove(group)
        skin.data = original
        bpy.data.meshes.remove(mesh)
        raise
    finally:
        rig.data.pose_position = held_pose
        bpy.context.view_layer.objects.active = held_active
        bpy.context.scene.frame_set(held_frame)
        bpy.context.view_layer.update()
    log(f"surface: rest-preserving collar cleanup; welded {len(before) - len(after)} seam duplicates, "
        f"bind displacement {report['max_bind_displacement_mm']:.4f} mm")
    return report
