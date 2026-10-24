"""A row's one shared horizontal bar and exact two-handed grip, in Blender."""
import json
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

import equipment
import hand_rig
import mpfb_cage as cage
from row_motion import solve_shared_bar, shaft_gaps, reference_arms, reference_arm_at


def apply(rig, skin, first, last, targets=None, keyposes=None, log=print):
    """Constrain the reference-driven arms onto a bar; preserve the stable body pose.

    Derive arm targets directly from measured reference landmarks, with source elbow
    angles adapted to the cartoon's limb lengths. The level shared bar keeps its grip
    width, and a frontal body-surface guard prevents it entering the torso or thighs.
    Reference depth remains inferred; exact contact alone is not pose validation.
    """
    scene, pose = bpy.context.scene, rig.pose.bones
    fit = json.loads(rig['skin_fit'])
    grips = json.loads(rig.get('hand_grip', '{}'))
    if any(side not in grips for side in ('left', 'right')):
        raise ValueError('Row contact requires both fitted, closed-hand grip models.')
    sides = (('left', 'Left', -1), ('right', 'Right', 1))
    scene.frame_set(first)
    shoulders = {s: np.array(pose[cage.P + title + 'Arm'].head) for s, title, _ in sides}
    along = shoulders['right'] - shoulders['left']
    along[2] = 0
    along /= np.linalg.norm(along)
    shoulder_mid = (shoulders['left'] + shoulders['right']) / 2
    shoulder_width = np.linalg.norm(shoulders['right'] - shoulders['left'])
    lengths = {s: (pose[cage.P + title + 'Arm'].length, pose[cage.P + title + 'ForeArm'].length)
               for s, title, _ in sides}
    if not keyposes:
        raise ValueError('The row bar needs its original measured reference key poses.')
    source = reference_arms(json.loads(Path(keyposes).read_text()))
    from keypose_clip import timeline
    progress = timeline(scene.render.fps)
    progress = np.interp(np.linspace(0, len(progress) - 1, last - first + 1), np.arange(len(progress)), progress)
    path, widths = {}, []
    for frame, amount in zip(range(first, last + 1), progress):
        scene.frame_set(frame)
        points, hints, frame_shoulders = {}, {}, {}
        for side, title, sign in sides:
            shoulder = np.array(pose[cage.P + title + 'Arm'].head)
            frame_shoulders[side] = shoulder
            arm = reference_arm_at(source, side, amount, *lengths[side])
            wrist = shoulder + arm['wrist']
            turn = hand_rig.hand_turn(grips[side], sign, along, arm['forearm_direction'])
            points[side] = wrist + turn @ (np.array(grips[side]['B']) - grips[side]['wrist'])
            hints[side] = shoulder + arm['elbow']
        centre = (points['left'] + points['right']) / 2
        centre -= along * ((centre - shoulder_mid) @ along)
        path[frame] = (centre, hints, frame_shoulders)
        widths.append(abs((points['right'] - points['left']) @ along))
    ratio = None
    if targets and Path(targets).is_file():
        payload = json.loads(Path(targets).read_text())
        reading = (payload.get('widths') or {}).get('grip_ratio') or {}
        value = reading.get('value')
        if isinstance(value, (float, int)) and not isinstance(value, bool) and np.isfinite(value):
            ratio = float(value)
    requested = ratio if ratio is not None else float(np.median(widths) / shoulder_width)
    # The cartoon has different arm proportions. Keep adaptation visible in the report.
    chosen = float(np.clip(requested, .9, 1.5))
    half = chosen * shoulder_width / 2
    bar = equipment.barbell(fit['measured']['height'], plates=False)
    up = Vector((0, 0, 1))
    basis = Matrix((Vector(along), up.cross(Vector(along)), up)).transposed()
    bar.matrix_world = rig.matrix_world @ basis.to_4x4()
    fingers, curl = {}, {}
    for side, title, _ in sides:
        fingers[side] = hand_rig.hand_bones(title)[1:]
        curl.update(hand_rig.pose_basis(rig, title, grips[side]['rotations']))
    names = {cage.P + title + n for _, title, _ in sides for n in ('Arm', 'ForeArm', 'Hand')} | set(curl)
    # The torso and legs stay at their starting hinge. Build the guard once on
    # their evaluated surface before arm corrections can affect the geometry.
    scene.frame_set(first)
    guard = frontal_guard(rig, skin, along)
    previous, previous_elbows = {}, {}
    worst_bend = worst_shift = worst_contact = worst_length = worst_surface = 0.
    trace = []
    for frame in range(first, last + 1):
        scene.frame_set(frame)
        centre, hints, frame_shoulders = path[frame]
        centre, surface_shift = guard(centre)
        # Moving a target back toward the shoulder would re-enter the belly. If
        # this collision-free position is unreachable, stop rather than clip.
        adapted, solutions, shift = solve_shared_bar(frame_shoulders, centre, along, half, lengths, grips,
                                                     hints, previous_elbows, allow_adaptation=False)
        worst_surface = max(worst_surface, surface_shift)
        bar.location = rig.matrix_world @ Vector(adapted)
        bar.keyframe_insert('location', frame=frame)
        for side, title, sign in sides:
            arm, fore, hand = [pose[cage.P + title + name] for name in ('Arm', 'ForeArm', 'Hand')]
            solution = solutions[side]
            cage._aim_pose(arm, Vector(solution['elbow']) - arm.head)
            cage._aim_pose(fore, Vector(solution['wrist']) - fore.head)
            turn = Matrix(solution['turn'].tolist())
            matrix = (turn @ hand.bone.matrix_local.to_3x3()).to_4x4()
            matrix.translation = hand.head
            hand.matrix = matrix
            bpy.context.view_layer.update()
            for name in fingers[side]:
                pose[name].rotation_mode = 'QUATERNION'
                pose[name].rotation_quaternion = curl[name]
            bpy.context.view_layer.update()
            actual_turn = np.array(hand.matrix.to_3x3() @ hand.bone.matrix_local.to_3x3().inverted())
            actual_contact = np.array(hand.head) + actual_turn @ (np.array(grips[side]['B']) - grips[side]['wrist'])
            target_contact = adapted + sign * half * along
            worst_contact = max(worst_contact, float(np.linalg.norm(actual_contact - target_contact)))
            worst_length = max(worst_length, float(np.linalg.norm(np.array(hand.head) - solution['wrist'])))
            worst_bend = max(worst_bend, solution['bend_deg'])
            previous_elbows[side] = {'elbow': solution['elbow'], 'turn': solution['turn']}
        cage._key(rig, previous, frame, names)
        worst_shift = max(worst_shift, shift)
        trace.append({'frame': frame, 'bar': adapted.tolist(), 'path_adjustment_mm': round(shift * 1000, 3),
                      'surface_adjustment_mm': round(surface_shift * 1000, 3)})
    scene.frame_set(first)
    # A pose matrix update must not quietly invalidate the pure geometry's exact contact.
    if worst_contact > .002 or worst_length > .002:
        raise ValueError(f'Row hand contact failed after posing: {worst_contact * 1000:.2f} mm.')
    report = {'source': 'shared bar from direct measured arm landmarks; elbow angles adapted to character limb lengths',
              'requested_grip_ratio': requested, 'grip_ratio': chosen,
              'worst_wrist_bend_deg': worst_bend, 'max_path_adjustment_mm': worst_shift * 1000,
              'max_grip_anchor_error_mm': worst_contact * 1000,
              'max_surface_adjustment_mm': worst_surface * 1000,
              'frames': trace}
    report['contact'] = cage.bar_contact(rig, skin, bar, first, last)
    report['contact']['sides'] = side_contacts(rig, skin, bar, first, last)
    report['body_clearance'] = body_clearance(rig, skin, bar, first, last)
    # Combined nearest-gap metrics alone can hide one floating hand. Both sides must
    # independently touch the bar throughout the clip, with their own depth measured.
    for side, contact in report['contact']['sides'].items():
        if contact['frames_touching_within_3mm'] < contact['frames'] * .8:
            raise ValueError(f"The row's {side} hand does not maintain bar contact.")
    log(f"row: shared bar, grip {chosen:.2f}x shoulder width, anchor error {worst_contact * 1000:.3f} mm, "
        f"wrist bend {worst_bend:.1f} degrees, path adapted {worst_shift * 1000:.1f} mm")
    return bar, report



def side_contacts(rig, skin, bar, first, last):
    """Measure each rendered hand separately against the bar's finite central shaft."""
    import skin_wrap
    fit = json.loads(rig['skin_fit'])
    unit = fit['measured']['height'] / 1.75
    radius = equipment.BARBELL_BAR_DIAMETER / 2 * unit
    half_length = equipment.BARBELL_LENGTH / 2 * unit
    masks = {}
    for side, title in (('left', 'Left'), ('right', 'Right')):
        indices = {g.index for g in skin.vertex_groups if title + 'Hand' in g.name}
        masks[side] = np.array([any(g.group in indices and g.weight > .5 for g in v.groups)
                                for v in skin.data.vertices])
        if not masks[side].any():
            raise ValueError(f"No skin vertices can measure the row's {side} hand contact.")
    rows = {side: [] for side in masks}
    scene = bpy.context.scene
    for frame in range(first, last + 1):
        scene.frame_set(frame)
        points = skin_wrap.evaluated(skin)[0]
        centre = np.array(bar.matrix_world.translation)
        along = np.array((bar.matrix_world.to_3x3() @ Vector((1, 0, 0))).normalized())
        for side, mask in masks.items():
            gap = shaft_gaps(points[mask], centre, along, half_length, radius)
            rows[side].append({'surface': float(np.abs(gap).min()),
                               'inside': max(0., float(-gap.min()))})
    scene.frame_set(first)
    return {side: {'frames': len(gaps),
                   'frames_touching_within_3mm': sum(g['surface'] <= .003 for g in gaps),
                   'deepest_inside_bar_mm': max(g['inside'] for g in gaps) * 1000,
                   'nearest_gap_mm_max': max(g['surface'] for g in gaps) * 1000}
            for side, gaps in rows.items()}



def body_clearance(rig, skin, bar, first, last, include_head=False):
    """Sample the shaft against the evaluated torso/thigh surface, excluding hands.

    Surface overlap is a conservative candidate. An interior centre is only called
    confirmed when three ray-parity directions agree and its nearest normal points
    inward. Open boundaries and thin/disconnected skin can make other samples
    ambiguous, which are reported without inventing a passing collision verdict.
    """
    import skin_wrap
    from mathutils.bvhtree import BVHTree
    fit = json.loads(rig['skin_fit'])
    unit = fit['measured']['height'] / 1.75
    radius = equipment.BARBELL_BAR_DIAMETER / 2 * unit
    half_length = equipment.BARBELL_LENGTH / 2 * unit
    regions = ('Hips', 'Spine', 'UpLeg', 'Leg') + (('Neck', 'Head') if include_head else ())
    torso = {g.index for g in skin.vertex_groups if any(k in g.name for k in regions) and 'Hand' not in g.name}
    selected = np.array([sum(g.weight for g in v.groups if g.group in torso) > .5
                         for v in skin.data.vertices])
    skin.data.calc_loop_triangles()
    triangles = [tuple(tri.vertices) for tri in skin.data.loop_triangles
                 if all(selected[v] for v in tri.vertices)]
    if not triangles:
        return {'status': 'not_measured', 'reason': 'No torso or thigh surface triangles.'}
    directions = [Vector(v).normalized() for v in ((1,.173,.291), (.193,1,.271), (.137,.231,1))]
    def inside(tree, point):
        for direction in directions:
            origin, crossings = point.copy(), 0
            for _ in range(128):
                hit, _, _, _ = tree.ray_cast(origin, direction, 10 * unit)
                if hit is None:
                    break
                crossings += 1
                origin = hit + direction * (1e-5 * unit)
            if crossings == 128 or crossings % 2 != 1:
                return False
        return True
    rows, worst = [], None
    scene = bpy.context.scene
    samples = np.linspace(-half_length, half_length, 101)
    for frame in range(first, last + 1):
        scene.frame_set(frame)
        points = skin_wrap.evaluated(skin)[0]
        tree = BVHTree.FromPolygons([Vector(p) for p in points], triangles, all_triangles=True)
        centre = bar.matrix_world.translation
        along = (bar.matrix_world.to_3x3() @ Vector((1,0,0))).normalized()
        overlaps, confirmed = [], []
        for axial in samples:
            point = centre + along * float(axial)
            nearest, normal, _, distance = tree.find_nearest(point)
            if nearest is None:
                continue
            signed = (point - nearest).dot(normal)
            interior = signed < -1e-5 and inside(tree, point)
            depth = radius + distance if interior else max(0., radius - distance)
            if depth > 0:
                row = {'axial_mm': float(axial * 1000), 'depth_mm': float(depth * 1000),
                       'confirmed_inside': interior}
                overlaps.append(row)
                if interior:
                    confirmed.append(row)
                if worst is None or row['depth_mm'] > worst['depth_mm']:
                    worst = {'frame': frame, **row}
        if overlaps:
            rows.append({'frame': frame, 'max_overlap_mm': max(r['depth_mm'] for r in overlaps),
                         'confirmed_inside_mm': max((r['depth_mm'] for r in confirmed), default=0.),
                         'axial_extent_mm': [min(r['axial_mm'] for r in overlaps), max(r['axial_mm'] for r in overlaps)]})
    scene.frame_set(first)
    return {'status': 'overlap_candidates' if rows else 'clear_at_samples', 'worst': worst,
            'sample_spacing_mm': float((samples[1] - samples[0]) * 1000),
            'triangles': len(triangles), 'frames': rows}



def frontal_guard(rig, skin, along):
    """Return a bar-centre guard in front of the fixed torso and thighs.

    Ray-cast from the front onto the real posed body at shaft samples and vertical
    offsets across its circular section. The target moves forward only, preserving
    the measured rise and the grip width. Hands/arms are excluded so the bar is not
    repelled by the very hands that will grip it.
    """
    import skin_wrap
    from mathutils.bvhtree import BVHTree
    fit = json.loads(rig['skin_fit'])
    unit = fit['measured']['height'] / 1.75
    radius = equipment.BARBELL_BAR_DIAMETER / 2 * unit
    half_length = equipment.BARBELL_LENGTH / 2 * unit
    clearance = .003
    groups = {g.index for g in skin.vertex_groups if any(k in g.name for k in ('Hips','Spine','UpLeg','Leg'))}
    mask = np.array([sum(g.weight for g in v.groups if g.group in groups) > .5 for v in skin.data.vertices])
    skin.data.calc_loop_triangles()
    faces = [tuple(t.vertices) for t in skin.data.loop_triangles if all(mask[v] for v in t.vertices)]
    if not faces:
        raise ValueError('The row cannot determine its torso/thigh collision surface.')
    points = skin_wrap.evaluated(skin)[0]
    tree = BVHTree.FromPolygons([Vector(p) for p in points], faces, all_triangles=True)
    world = rig.matrix_world
    world_inverse = world.inverted()
    forward = np.cross([0.,0.,1.], along)
    forward /= np.linalg.norm(forward)
    fw = (world.to_3x3() @ Vector(forward)).normalized()
    axial = np.linspace(-half_length, half_length, int(2 * half_length / (radius / 2)) + 1)
    vertical = np.linspace(-radius, radius, 9)
    def guard(centre):
        centre = np.asarray(centre, dtype=float)
        required = float(centre @ forward)
        for dz in vertical:
            section = np.sqrt(max(0., radius * radius - dz * dz))
            for x in axial:
                sample = centre + along * x + np.array([0.,0.,dz])
                start = world @ Vector(sample + forward * (2 * unit))
                hit, _, _, _ = tree.ray_cast(start, -fw, 4 * unit)
                if hit is None:
                    continue
                surface = np.array(world_inverse @ hit)
                required = max(required, float(surface @ forward) + section + clearance)
        shift = max(0., required - float(centre @ forward))
        return centre + forward * shift, shift
    return guard
