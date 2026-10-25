"""A straight-arm conventional floor pull on fixed, flat character feet.

The source photos establish the exercise, not calibrated 3D depths. This fitter
uses explicit floor-pull and neutral-lockout constraints instead of reproducing
the detector's bent elbows or its translated standing pelvis. Limb lengths and
the original mesh are never changed here; an unsuitable short-arm fit is refused.
"""
import json
import math
import numpy as np

SIDES = (('left', 'Left', -1), ('right', 'Right', 1))


def _rotation(degrees):
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])


def _angle(a, b):
    return float(np.degrees(np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1., 1.))))


def knee_position(hip, ankle, upper, lower):
    """An exact fixed-length leg with a forward knee; never stretch a target."""
    delta = np.asarray(ankle) - hip
    distance = float(np.linalg.norm(delta))
    if not abs(upper - lower) + 1e-7 < distance < upper + lower - 1e-7:
        raise ValueError('The conventional deadlift stance is outside the fixed leg reach.')
    axis = delta / distance
    pole = np.array([0., 1., 0.]) - axis * axis[1]
    pole /= np.linalg.norm(pole)
    reach = (upper * upper + distance * distance - lower * lower) / (2 * distance)
    radius = math.sqrt(max(0., upper * upper - reach * reach))
    return hip + axis * reach + pole * radius


def plan(rest, grips, height, progress, feet=None, grip_ratio=1.45, start_lean_deg=55.):
    """Plan rig-space body and bar positions using the fitted closed-hand models.

    ``rest`` contains each side's hip/shoulder/ankle and fixed leg/arm lengths.
    ``progress`` is the existing reference clock, including its holds and return.
    The bar starts at the loaded plate radius, remains over the same foot region,
    and reaches a simultaneous neutral standing pose with straight lifting arms.
    """
    from row_motion import solve_shared_bar
    if not np.isfinite(height) or height <= 0 or not 40. <= start_lean_deg <= 65.:
        raise ValueError('The floor pull needs a positive height and a bounded conventional hinge.')
    values = np.asarray(progress, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
        raise ValueError('The floor pull needs finite progress values from zero to one.')
    feet = {side: np.array((feet or {}).get(side, rest[side]['ankle']), dtype=float) for side, _, _ in SIDES}
    hip_rest = np.mean([rest[s]['hip'] for s, _, _ in SIDES], axis=0)
    shoulder_rest = np.mean([rest[s]['shoulder'] for s, _, _ in SIDES], axis=0)
    torso = shoulder_rest - hip_rest
    rest_lean = math.degrees(math.atan2(torso[1], torso[2]))
    shoulder_width = abs(float(np.asarray(rest['right']['shoulder'])[0] - np.asarray(rest['left']['shoulder'])[0]))
    half_grip = float(grip_ratio) * shoulder_width / 2
    along = np.array([1., 0., 0.])
    ankle_mid = np.mean(list(feet.values()), axis=0)
    bar_y = float(ankle_mid[1] + .11 * height)
    bar_floor = .225 * height / 1.75
    neutral = _rotation(rest_lean)
    neutral_hips = {side: neutral @ (np.asarray(rest[side]['hip']) - hip_rest) for side, _, _ in SIDES}
    top_y = float(ankle_mid[1])
    top_heights = []
    for side, _, _ in SIDES:
        upper, lower = rest[side]['leg_lengths']
        reach = math.sqrt(upper * upper + lower * lower - 2 * upper * lower * math.cos(math.radians(175.)))
        offset = feet[side] - neutral_hips[side]
        radial = (offset[0] - ankle_mid[0]) ** 2 + (offset[1] - top_y) ** 2
        if radial >= reach * reach:
            raise ValueError('The conventional stance cannot reach a neutral hip lockout.')
        top_heights.append(float(offset[2] + math.sqrt(reach * reach - radial)))
    top_hip = np.array([ankle_mid[0], top_y, min(top_heights)])
    lengths = {side: tuple(rest[side]['arm_lengths']) for side, _, _ in SIDES}
    preferences = {side: {'elbow_deg': 178., 'min_elbow_deg': 175.} for side, _, _ in SIDES}

    def body(lean, hip):
        rotation = _rotation(rest_lean - lean)
        hips = {side: hip + rotation @ (np.asarray(rest[side]['hip']) - hip_rest) for side, _, _ in SIDES}
        shoulders = {side: hip + rotation @ (np.asarray(rest[side]['shoulder']) - hip_rest) for side, _, _ in SIDES}
        knees = {side: knee_position(hips[side], feet[side], *rest[side]['leg_lengths']) for side, _, _ in SIDES}
        angles = {side: _angle(hips[side] - knees[side], feet[side] - knees[side]) for side, _, _ in SIDES}
        if min(angles.values()) < 80.:
            raise ValueError('The short-arm character would need a deep squat instead of a conventional floor pull.')
        return rotation, hips, shoulders, knees, angles

    def arms(shoulders, centre, previous=None):
        hints = {side: shoulders[side] + np.array([0., 0., -lengths[side][0]]) for side, _, _ in SIDES}
        return solve_shared_bar(shoulders, centre, along, half_grip, lengths, grips, hints,
                                previous=previous, allow_adaptation=False, preferences=preferences)[1]

    _, _, top_shoulders, _, _ = body(0., top_hip)
    total_reach = np.mean([sum(lengths[side]) + np.linalg.norm(np.asarray(grips[side]['B']) - grips[side]['wrist'])
                          for side, _, _ in SIDES])
    lateral_reach = half_grip - shoulder_width / 2
    sagittal_reach_squared = total_reach * total_reach - lateral_reach * lateral_reach
    if sagittal_reach_squared <= (.11 * height) ** 2:
        raise ValueError('The conventional outside-knee grip is beyond the fixed arm reach.')
    nominal_top = float(np.mean(list(top_shoulders.values()), axis=0)[2]
                        - math.sqrt(sagittal_reach_squared - (.11 * height) ** 2))
    end_candidates = []
    for delta in np.arange(-.025, .0251, .0005) * height:
        centre = np.array([ankle_mid[0], bar_y, nominal_top + delta])
        try:
            solutions = arms(top_shoulders, centre)
            end_candidates.append((sum(value['cost'] for value in solutions.values()), centre, solutions))
        except ValueError:
            pass
    if not end_candidates:
        raise ValueError('The character cannot hold a shared deadlift bar with straight arms at neutral lockout; use a suitable fitness fit.')
    _, top_bar, _ = min(end_candidates, key=lambda value: value[0])
    if top_bar[2] - bar_floor < .10 * height:
        raise ValueError('The fitted character has insufficient conventional floor-pull travel.')
    top_drop = float(np.mean(list(top_shoulders.values()), axis=0)[2] - top_bar[2])
    top_side_reach = math.sqrt(sagittal_reach_squared - (.11 * height) ** 2)
    previous, poses = None, []
    for fraction in values:
        lean = float(start_lean_deg * (1 - fraction))
        rotation = _rotation(rest_lean - lean)
        shoulder_offset = rotation @ (shoulder_rest - hip_rest)
        shoulder_y = float((bar_y + .03 * height) * (1 - fraction) + top_y * fraction)
        centre = np.array([ankle_mid[0], bar_y, bar_floor + fraction * (top_bar[2] - bar_floor)])
        drop = top_drop + math.sqrt(max(0., sagittal_reach_squared - (shoulder_y - bar_y) ** 2)) - top_side_reach
        nominal = float(centre[2] + drop - shoulder_offset[2])
        found = []
        offsets = [0.] + [sign * distance for distance in np.arange(.0005, .0101, .0005) * height for sign in (1, -1)]
        for offset in offsets if fraction < 1 - 1e-9 else [top_hip[2] - nominal]:
            hip = np.array([ankle_mid[0], shoulder_y - shoulder_offset[1], nominal + offset])
            if hip[2] > top_hip[2] + 1e-6:
                continue
            try:
                rotation, hips, shoulders, knees, angles = body(lean, hip)
                if fraction < 1e-9 and (min(knee[2] for knee in knees.values()) - bar_floor < .02 * height
                                       or max(angles.values()) > 165.):
                    continue
                solutions = arms(shoulders, centre, previous)
            except ValueError:
                continue
            found.append((sum(value['cost'] for value in solutions.values()) + (offset / (.002 * height)) ** 2,
                          hip, rotation, hips, shoulders, knees, angles, solutions))
            # The nominal height and a small exact-contact correction suffice;
            # keep the search bounded instead of rebuilding an arbitrary pose.
            if len(found) >= 3:
                break
        if not found:
            raise ValueError('The character cannot perform the straight-arm conventional floor pull with planted feet; use a suitable fitness fit.')
        _, hip, rotation, hips, shoulders, knees, angles, previous = min(found, key=lambda value: value[0])
        poses.append({'progress': float(fraction), 'hip': hip, 'rotation': rotation, 'hips': hips,
                      'shoulders': shoulders, 'knees': knees, 'knees_deg': angles, 'arms': previous,
                      'bar': centre, 'along': along, 'height_adjustment_mm': float((hip[2] - nominal) * 1000)})
    return {'poses': poses, 'feet': feet, 'grip_ratio': grip_ratio, 'plate_radius_m': bar_floor,
            'bar_y_from_ankles_m': bar_y - ankle_mid[1], 'top_bar_m': top_bar,
            'start_torso_lean_deg': start_lean_deg, 'rest_torso_lean_deg': rest_lean}


def apply(rig, skin, first, last, reference=None):
    """Re-key the body before equipment contacts; return the explicit bar path."""
    import bpy
    from mathutils import Matrix, Vector
    import mpfb_cage as cage
    import hand_rig
    import retarget
    import skin_wrap
    from keypose_clip import timeline
    scene, pose = bpy.context.scene, rig.pose.bones
    height = float(json.loads(rig['skin_fit'])['measured']['height'])
    grips = json.loads(rig['hand_grip'])
    rest_lengths = {bone.name: float(bone.length) for bone in rig.data.bones}
    rest_vertices = np.array([tuple(vertex.co) for vertex in skin.data.vertices])
    source_feet = None
    source_hips = {}
    for frame in range(first, last + 1):
        scene.frame_set(frame)
        source_hips[frame] = np.mean([np.array(pose[cage.P + title + 'UpLeg'].head) for _, title, _ in SIDES], axis=0)
        if frame == first:
            source_feet = {side: list(pose[cage.P + title + 'Foot'].head) for side, title, _ in SIDES}
    scene.frame_set(first)
    for bone in pose:
        bone.matrix_basis = Matrix.Identity(4)
        bone.rotation_mode = 'QUATERNION'
    bpy.context.view_layer.update()
    points = skin_wrap.evaluated(skin)[0]
    feet, rest = {}, {}
    for side, title, _ in SIDES:
        groups = {group.index for group in skin.vertex_groups if title + 'Foot' in group.name or title + 'ToeBase' in group.name}
        mask = np.array([sum(group.weight for group in vertex.groups if group.group in groups) > .5 for vertex in skin.data.vertices])
        if not mask.any():
            raise ValueError('The conventional floor pull cannot measure both flat feet.')
        ankle = np.array(rig.data.bones[cage.P + title + 'Foot'].head_local)
        ankle[2] -= float(points[mask, 2].min())
        feet[side] = ankle
        rest[side] = {'hip': np.array(rig.data.bones[cage.P + title + 'UpLeg'].head_local),
                      'shoulder': np.array(rig.data.bones[cage.P + title + 'Arm'].head_local),
                      'ankle': ankle,
                      'leg_lengths': [rig.data.bones[cage.P + title + part].length for part in ('UpLeg', 'Leg')],
                      'arm_lengths': [rig.data.bones[cage.P + title + part].length for part in ('Arm', 'ForeArm')]}
    progress = timeline(float(scene.render.fps) / scene.render.fps_base)
    if len(progress) != last - first + 1:
        raise ValueError('The conventional floor pull does not match the recorded reference clock.')
    fitted = plan(rest, grips, height, progress, feet=feet)
    previous, maximum_leg_error, posed_length_drift, path, adaptations = {}, 0., 0., {}, []
    for frame, entry in zip(range(first, last + 1), fitted['poses']):
        scene.frame_set(frame)
        for bone in pose:
            bone.matrix_basis = Matrix.Identity(4)
        rotation = Matrix(entry['rotation'].tolist()).to_4x4()
        root = pose[cage.P + 'Hips']
        root.matrix = rotation @ root.bone.matrix_local
        bpy.context.view_layer.update()
        hip_mid = sum((pose[cage.P + title + 'UpLeg'].head for _, title, _ in SIDES), Vector()) / 2
        cage._move_root(rig, Vector(entry['hip']) - hip_mid)
        for side, title, _ in SIDES:
            upper, lower, foot = [pose[cage.P + title + part] for part in ('UpLeg', 'Leg', 'Foot')]
            cage._aim_pose(upper, Vector(entry['knees'][side]) - upper.head)
            cage._aim_pose(lower, Vector(feet[side]) - lower.head)
            foot_matrix = foot.bone.matrix_local.copy()
            foot_matrix.translation = foot.head
            foot.matrix = foot_matrix
            bpy.context.view_layer.update()
            toe = pose[cage.P + title + 'ToeBase']
            toe_matrix = toe.bone.matrix_local.copy()
            toe_matrix.translation = toe.head
            toe.matrix = toe_matrix
            bpy.context.view_layer.update()
            maximum_leg_error = max(maximum_leg_error, float((foot.head - Vector(feet[side])).length))
            upper, lower, hand = [pose[cage.P + title + part] for part in ('Arm', 'ForeArm', 'Hand')]
            solution = entry['arms'][side]
            cage._aim_pose(upper, Vector(solution['elbow']) - upper.head)
            cage._aim_pose(lower, Vector(solution['wrist']) - lower.head)
            hand_matrix = (Matrix(solution['turn'].tolist()) @ hand.bone.matrix_local.to_3x3()).to_4x4()
            hand_matrix.translation = hand.head
            hand.matrix = hand_matrix
            for name, quaternion in hand_rig.pose_basis(rig, title, grips[side]['rotations']).items():
                pose[name].rotation_quaternion = quaternion
        bpy.context.view_layer.update()
        for _, title, _ in SIDES:
            for part in ('UpLeg', 'Leg', 'Arm', 'ForeArm'):
                bone = pose[cage.P + title + part]
                posed_length_drift = max(posed_length_drift,
                                         abs(float((bone.tail - bone.head).length) - rest_lengths[bone.name]))
        cage._key(rig, previous, frame)
        root.keyframe_insert('location', frame=frame)
        path[frame] = {'centre': entry['bar'].tolist(), 'along': entry['along'].tolist()}
        adaptations.append({'frame': frame, 'root_shift_mm': ((entry['hip'] - source_hips[frame]) * 1000).tolist(),
                            'height_fit_mm': entry['height_adjustment_mm'],
                            'torso_lean_deg': fitted['start_torso_lean_deg'] * (1 - entry['progress'])})
    scene.frame_set(first)
    drift = max(abs(float(bone.length) - rest_lengths[bone.name]) for bone in rig.data.bones)
    vertex_drift = float(np.linalg.norm(np.array([tuple(vertex.co) for vertex in skin.data.vertices]) - rest_vertices, axis=1).max())
    if maximum_leg_error > .001 or drift > 1e-8 or vertex_drift > 1e-12 or posed_length_drift > 1e-5:
        raise ValueError('The conventional floor pull failed its fixed-length foot constraints.')
    return {'method': 'conventional loaded floor pull with straight arms, flat fixed feet and neutral hip lockout',
            'reference_depth': 'uncalibrated; explicit movement constraints replace inferred elbow/pelvis drift',
            'grip_ratio': fitted['grip_ratio'], 'source_grip_ratio': None,
            'stance_source': 'flat character bind feet; source width is not metrically calibrated',
            'start_torso_lean_deg': fitted['start_torso_lean_deg'], 'bar_path': path,
            'plate_radius_m': fitted['plate_radius_m'], 'bar_y_from_ankles_m': fitted['bar_y_from_ankles_m'],
            'ankle_residual_mm': maximum_leg_error * 1000, 'rest_length_drift_mm': drift * 1000,
            'posed_length_drift_mm': posed_length_drift * 1000,
            'rest_mesh_drift_mm': vertex_drift * 1000, 'ankle_anchors_m': {side: value.tolist() for side, value in feet.items()},
            'original_ankle_anchors_m': source_feet,
            'adaptations': adaptations, 'contact': retarget.skin_contact(rig, skin, first, last)}
