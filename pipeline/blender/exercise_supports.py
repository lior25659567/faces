"""Body-aligned exercise supports and measured contact diagnostics, in Blender.

Equipment dimensions follow the posed mascot. Placement is measured from the
evaluated skin, not a detached decorative prop. Failed contact stays a failed check.
"""
import json
import bpy
import numpy as np
from mathutils import Matrix, Vector
import equipment
import mpfb_cage as cage
import skin_wrap

SEATED = {'Lat Pulldown (seated, to the front)', 'Dumbbell Shoulder Press (seated overhead press)'}
BENCH = 'Flat Barbell Bench Press'
HANG = 'Pull-Up'
PLANK = 'Front Plank (forearm plank)'
SLED = 'Leg Press (seated machine)'
CABLE = {'Lat Pulldown (seated, to the front)', 'Tricep Pushdown (cable, bar or rope attachment)'}


def _mask(skin, names):
    groups = {g.index for g in skin.vertex_groups if any(name in g.name for name in names)}
    result = np.array([sum(g.weight for g in vertex.groups if g.group in groups) > .5 for vertex in skin.data.vertices])
    if not result.any():
        raise ValueError('No measurable skin surface for ' + ', '.join(names))
    return result


def _pad(name, top, width, length, thickness, material):
    return equipment._pad(name, (top[0], top[1], top[2] - thickness / 2), (width, length, thickness), material)


def apply(rig, skin, first, last, exercise):
    scene = bpy.context.scene
    height = json.loads(rig['skin_fit'])['measured']['height']
    scene.frame_set(first)
    points = skin_wrap.evaluated(skin)[0]
    metal, rubber, _ = equipment._materials()
    props, requirements = [], []
    report = {'frames': last - first + 1, 'surfaces': {}}
    supported = exercise in SEATED | {BENCH, HANG, PLANK, SLED}
    if not supported and exercise not in CABLE:
        return [], None
    pelvis = _mask(skin, ['Hips'])
    torso = _mask(skin, ['Spine', 'Hips'])
    # Preserve pelvis position on supports. Free-foot retargeting follows the
    # ankles by default, which otherwise moves a seat or a bench through the rider.
    if exercise in SEATED | {BENCH, SLED, HANG}:
        previous = {}
        pb = rig.pose.bones[cage.P + 'Hips']
        if exercise == HANG:
            def anchor():
                return sum((rig.pose.bones[cage.P + side + 'Hand'].head for side in ('Left', 'Right')), Vector()) / 2
        else:
            def anchor():
                return sum((rig.pose.bones[cage.P + side + 'UpLeg'].head for side in ('Left', 'Right')), Vector()) / 2
        origin = anchor().copy()
        if exercise == HANG:
            origin.z += height * .10
        for frame in range(first, last + 1):
            scene.frame_set(frame)
            cage._move_root(rig, origin - anchor())
            pb.keyframe_insert('location', frame=frame)
        scene.frame_set(first)
        points = skin_wrap.evaluated(skin)[0]
    if exercise in SEATED | {BENCH, SLED}:
        selected = points[pelvis]
        top = np.array([np.median(selected[:, 0]), np.median(selected[:, 1]), selected[:, 2].min()])
        width = max(float(np.ptp(selected[:, 0])) * .9, height * .18)
        length = height * (.65 if exercise == BENCH else .22)
        if exercise == BENCH:
            # Bench long axis follows the torso's actual horizontal axis.
            upper = points[torso]
            top[1] = (upper[:, 1].min() + upper[:, 1].max()) / 2
            length = max(length, float(np.ptp(upper[:, 1])) * 1.1)
        props.append(_pad('body_support_pad', top, width, length, height * .035, rubber))
        for sign in (-1, 1):
            base = (top[0], top[1] + sign * length * .3, height * .015)
            end = (base[0], base[1], max(height * .02, top[2] - height * .035))
            props.append(equipment.capsule('support_leg', base, end, height * .018, metal))
            props.append(_pad('support_base', (base[0], base[1], height * .02), width * 1.2, height * .09, height * .02, metal))
        requirements.append(('pelvis', pelvis, float(top[2]), top[:2].copy(), np.array([width, length]) / 2))
        if exercise == BENCH:
            requirements.append(('back', _mask(skin, ['Spine1', 'Spine2']), float(top[2]), top[:2].copy(), np.array([width, length]) / 2))
        elif exercise in SEATED:
            upper = points[_mask(skin, ['Spine1', 'Spine2'])]
            low, high = upper.min(0), upper.max(0)
            centre = ((low + high) / 2).tolist()
            centre[1] = float(low[1] - height * .015)
            props.append(equipment._pad('seat_back', centre,
                (width, height * .03, max(float(high[2] - low[2]), height * .2)), rubber))
            report['backrest_plane_y'] = float(low[1])
    # Pull-up posts are built against the final fixed bar in exercise_contacts.
    if exercise == PLANK:
        for side in ('Left', 'Right'):
            requirements.append((side.lower() + '_forearm', _mask(skin, [side + 'ForeArm']), 0., None, None))
            requirements.append((side.lower() + '_toe', _mask(skin, [side + 'ToeBase']), 0., None, None))
    if exercise in CABLE:
        shoulders = np.array([rig.matrix_world @ rig.pose.bones[cage.P + side + 'Arm'].head for side in ('Left', 'Right')])
        pulley = shoulders.mean(0) + np.array([0., .45 * height, .45 * height])
        props.append(equipment.capsule('cable_tower_post', (pulley[0], pulley[1], .02 * height), pulley, .02 * height, metal))
        report['pulley_anchor'] = pulley.tolist()
    rows = {name: [] for name, *_ in requirements}
    for frame in range(first, last + 1):
        scene.frame_set(frame)
        points = skin_wrap.evaluated(skin)[0]
        for name, mask, plane, centre, half in requirements:
            surface = points[mask]
            if centre is not None:
                surface = surface[np.all(abs(surface[:, :2] - centre) <= half, axis=1)]
            if not len(surface):
                rows[name].append(None)
            else:
                # Distance to the plane only counts within the finite support pad.
                rows[name].append((float(abs(surface[:, 2] - plane).min()), max(0., float(plane - surface[:, 2].min()))))
    for name, values in rows.items():
        measured = [v for v in values if v is not None]
        report['surfaces'][name] = {'frames': len(values), 'measured_frames': len(measured),
            'max_gap_mm': max((v[0] * 1000 for v in measured), default=None),
            'max_penetration_mm': max((v[1] * 1000 for v in measured), default=None)}
    if exercise == SLED:
        sled, measured = moving_sled(rig, skin, first, last, height, metal, rubber)
        props.extend(sled)
        report['surfaces'].update(measured['surfaces'])
        report['rail_path'] = measured['rail_path']
    scene.frame_set(first)
    return props, report


def moving_sled(rig, skin, first, last, height, metal, rubber):
    """A finite footplate on straight rails; report incompatible curved reference paths.

    Rails follow the measured travel axis. An inferred oblique angle is explicitly
    not claimed to be a calibrated 45-degree machine.
    """
    scene = bpy.context.scene
    feet = {side: _mask(skin, [title + 'Foot', title + 'ToeBase']) for side, title in (('left', 'Left'), ('right', 'Right'))}
    positions = []
    for frame in range(first, last + 1):
        scene.frame_set(frame)
        positions.append(np.mean([np.array(rig.matrix_world @ rig.pose.bones[cage.P + side + 'Foot'].head) for side in ('Left', 'Right')], axis=0))
    positions = np.array(positions)
    centre = positions.mean(0)
    _, singular, directions = np.linalg.svd(positions - centre)
    if singular[0] < height * .01:
        raise ValueError('The leg-press references provide no readable sled travel.')
    axis = directions[0]
    if axis[2] < 0:
        axis = -axis
    distance = (positions - centre) @ axis
    line = centre + distance[:, None] * axis
    residual = np.linalg.norm(positions - line, axis=1)
    scene.frame_set(first)
    points = skin_wrap.evaluated(skin)[0]
    offset = max(float(((points[mask] - positions[0]) @ axis).max()) for mask in feet.values())
    line += axis * offset
    lateral = np.array([1., 0., 0.])
    lateral -= axis * (lateral @ axis)
    lateral /= np.linalg.norm(lateral)
    tangent = np.cross(axis, lateral)
    width, length, thickness = height * .5, height * .35, height * .025
    plate = equipment._pad('leg_press_footplate', (0, 0, 0), (width, length, thickness), rubber)
    rotation = Matrix((Vector(lateral), Vector(tangent), Vector(axis))).transposed().to_4x4()
    values = {side: [] for side in feet}
    for frame, target in zip(range(first, last + 1), line):
        scene.frame_set(frame)
        # The skin touches the near face, whose normal points toward the rider.
        matrix = rotation.copy()
        matrix.translation = Vector(target + axis * thickness / 2)
        plate.matrix_world = matrix
        plate.rotation_mode = 'QUATERNION'
        plate.keyframe_insert('location', frame=frame)
        plate.keyframe_insert('rotation_quaternion', frame=frame)
        points = skin_wrap.evaluated(skin)[0]
        for side, mask in feet.items():
            local = points[mask] - target
            within = (abs(local @ lateral) <= width / 2) & (abs(local @ tangent) <= length / 2)
            gaps = local[within] @ axis
            values[side].append(None if not len(gaps) else (float(abs(gaps).min()), max(0., float(gaps.max()))))
    rails = []
    start = line[int(np.argmin(distance))] - axis * height * .1
    end = line[int(np.argmax(distance))] + axis * height * .1
    for sign in (-1, 1):
        offset = lateral * sign * width * .6
        rails.append(equipment.capsule('leg_press_rail', start + offset, end + offset, height * .018, metal))
        for point in (start + offset, end + offset):
            rails.append(equipment.capsule('leg_press_rail_stand', (point[0], point[1], 0.), point, height * .015, metal))
    surfaces = {}
    for side, rows in values.items():
        measured = [row for row in rows if row is not None]
        surfaces['footplate_' + side] = {'frames': len(rows), 'measured_frames': len(measured),
            'max_gap_mm': max((r[0] * 1000 for r in measured), default=None),
            'max_penetration_mm': max((r[1] * 1000 for r in measured), default=None)}
    return [plate, *rails], {'surfaces': surfaces, 'rail_path': {'max_deviation_mm': float(residual.max() * 1000),
        'incline_deg': float(np.degrees(np.arcsin(axis[2]))), 'source': 'reference-inferred travel axis; not calibrated'}}
