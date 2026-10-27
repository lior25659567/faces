"""Declared back-squat clip with fitted closed-hand back-bar equipment.

The body movement comes from the repository's pinned 3D clip. The arm hold is an
authored mascot posture, not a reconstruction of uncertain photo depth. This
module never changes the separate photograph-correction mode.
"""
import hashlib
import json
import math
from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

import hand_rig
import mpfb_cage as cage
import retarget
import row_bar
import skin_wrap
from row_motion import solve_shared_bar
from squat_range import PHASE_FRAMES, SPINE, summarize

PRESET_ID = 'squat-v1'
CLIP_SHA256 = 'e4f720f5d9b9657c2f7b4c52c516bdf3c17ec6d4fcfe5b9b527cb962d4157705'
PROFILE = {'exercise': 'Barbell Back Squat', 'family': 'squat', 'feet': 'planted',
           'holds': True, 'overhead': False, 'mode': 'cycle', 'motion_preset': PRESET_ID}
SIDES = (('left', 'Left', -1), ('right', 'Right', 1))


def validate_clip(path):
    path = Path(path)
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != CLIP_SHA256:
        raise ValueError('The squat preset requires its pinned, reviewed 3D motion clip.')
    return path


def reference_motion():
    return {'source': 'preset', 'mode': 'cycle', 'preset': PRESET_ID,
            'source_sha256': CLIP_SHA256, 'phase_frames': dict(PHASE_FRAMES),
            'explicit_phases': [], 'photos_context_only': True,
            'description': 'Pinned 3D squat body motion with an authored fitted back-bar grip; photographs are context only.'}


def _range(rig, skin, bar, first, last):
    # Use the same fitted-axis measurements as the final inspector. The normal
    # animator has not loaded the inspector's Blender helpers at this stage.
    inspector_path = str(Path(__file__).resolve().parents[2] / 'studio/blender')
    if inspector_path not in sys.path:
        sys.path.insert(0, inspector_path)
    import inspect_run
    scene, pose, world = bpy.context.scene, rig.pose.bones, rig.matrix_world
    _, _, side_normal, _, torso = inspect_run.studio_view(rig, 'left')
    legs = inspect_run.leg_turns(rig, side_normal, skin)
    samples = []
    for frame in range(first, last + 1):
        scene.frame_set(frame)
        fields = {label: {side: list(world @ pose[cage.P + title + part].head)
                          for side, title, _ in SIDES}
                  for label, part in (('hips_m', 'UpLeg'), ('knees_m', 'Leg'),
                                      ('ankles_m', 'Foot'), ('shoulders_m', 'Arm'))}
        axis = (bar.matrix_world.to_3x3() @ Vector((1., 0., 0.))).normalized()
        samples.append({'frame': frame, **fields, 'bar_m': list(bar.matrix_world.translation),
                        'bar_axis_m': list(axis),
                        'spine_m': {name: list(world @ pose[cage.P + name].head) for name in SPINE}})
    height = json.loads(rig['skin_fit'])['measured']['height']
    scene.frame_set(first)
    return summarize(samples, height, PHASE_FRAMES, torso, legs)


def measure(rig, skin, bar, first, last):
    """Replay actual keyed skin, supports and bilateral range without posing it."""
    from bench_feet import floor_report
    scene = bpy.context.scene
    height = json.loads(rig['skin_fit'])['measured']['height']
    hands = row_bar.side_contacts(rig, skin, bar, first, last)
    sides = {side: {'frames': reading['frames'],
                   'frames_touching_within_3mm': reading['frames_touching_within_3mm'],
                   'nearest_gap_mm_max': reading['nearest_gap_mm_max'],
                   'deepest_inside_bar_mm': reading['deepest_inside_bar_mm']}
             for side, reading in hands.items()}
    # The same complete rest-defined heel/toe patches as the bench gate.
    rest = np.array([vertex.co for vertex in skin.data.vertices])
    groups = {side: {group.index for group in skin.vertex_groups
                    if group.name in (cage.P + title + 'Foot', cage.P + title + 'ToeBase')}
              for side, title, _ in SIDES}
    masks = {side: np.array([sum(g.weight for g in vertex.groups if g.group in indices) > .5
                            for vertex in skin.data.vertices]) for side, indices in groups.items()}
    def frames():
        for frame in range(first, last + 1):
            scene.frame_set(frame)
            yield frame, skin_wrap.evaluated(skin)[0]
    floor = floor_report(rest, masks, frames())
    clearance = row_bar.body_clearance(rig, skin, bar, first, last, include_head=True)
    result = {'frames': last - first + 1, 'equipment': 'bar', 'sides': sides,
              'body_clearance': clearance, 'squat_range': _range(rig, skin, bar, first, last),
              'supports': floor, 'floor_contact': retarget.skin_contact(rig, skin, first, last),
              'preset': reference_motion()}
    scene.frame_set(first)
    return result


def _plant_feet(rig, skin, first, last):
    """Keep the captured toe-out directions on flat, fixed sole marks."""
    from exercise_supports import _leg_to
    scene, pose = bpy.context.scene, rig.pose.bones
    scene.frame_set(first)
    base = {bone.name: bone.matrix_basis.copy() for bone in pose}
    goals, adjustments = {}, {}
    inverse = rig.matrix_world.inverted()
    for side, title, _ in SIDES:
        foot, toe = [pose[cage.P + title + part] for part in ('Foot', 'ToeBase')]
        direction = toe.head - foot.head
        rest = toe.bone.head_local - foot.bone.head_local
        yaw = math.atan2(direction.y, direction.x) - math.atan2(rest.y, rest.x)
        rotation = Matrix.Rotation(yaw, 3, 'Z') @ foot.bone.matrix_local.to_3x3()
        matrix = rotation.to_4x4()
        matrix.translation = foot.head
        foot.matrix = matrix
        toe.rotation_mode = 'QUATERNION'
        toe.rotation_quaternion = (1., 0., 0., 0.)
        bpy.context.view_layer.update()
        indices = {group.index for group in skin.vertex_groups
                   if group.name in (cage.P + title + 'Foot', cage.P + title + 'ToeBase')}
        mask = np.array([sum(group.weight for group in vertex.groups if group.group in indices) > .5
                         for vertex in skin.data.vertices])
        low = float(skin_wrap.evaluated(skin)[0][mask, 2].min())
        delta = inverse.to_3x3() @ Vector((0., 0., -low))
        target = matrix.translation + delta
        goals[title] = (target, rotation)
        adjustments[side] = {'floor_settling_mm': -low * 1000,
                             'toe_out_yaw_deg': math.degrees(yaw)}
    for bone in pose:
        bone.matrix_basis = base[bone.name]
    bpy.context.view_layer.update()
    names = {cage.P + title + part for _, title, _ in SIDES for part in ('UpLeg', 'Leg', 'Foot', 'ToeBase')}
    previous, residual = {}, 0.
    for frame in range(first, last + 1):
        scene.frame_set(frame)
        for _, title, _ in SIDES:
            target, rotation = goals[title]
            residual = max(residual, _leg_to(rig, title, target, rotation))
            toe = pose[cage.P + title + 'ToeBase']
            toe.rotation_mode = 'QUATERNION'
            toe.rotation_quaternion = (1., 0., 0., 0.)
            bpy.context.view_layer.update()
        cage._key(rig, previous, frame, names)
    if residual > .002:
        raise ValueError('The squat cannot keep flat fixed soles without changing the leg lengths.')
    scene.frame_set(first)
    return {'method': 'fixed first-frame floor marks and toe-out yaw, flat fitted soles, exact two-bone knees',
            'ankle_residual_mm': residual * 1000, 'sides': adjustments}


def apply(rig, skin, first, last):
    """Replace the air-squat arm capture with one steady, closed back-bar hold."""
    scene, pose = bpy.context.scene, rig.pose.bones
    if first != 1 or last != 69 or scene.render.fps / scene.render.fps_base != 30:
        raise ValueError('The squat preset preserves all 69 captured frames at 30 fps.')
    planted = _plant_feet(rig, skin, first, last)
    scene.frame_set(first)
    # Captured air-squat clavicles narrow and protract the shoulders. They are
    # not the authored back-bar posture. Keep neutral fitted clavicles with the
    # original chest movement and solve the actual closed palms onto the bar.
    for _, title, _ in SIDES:
        shoulder = pose[cage.P + title + 'Shoulder']
        shoulder.rotation_mode = 'QUATERNION'
        shoulder.rotation_quaternion = (1., 0., 0., 0.)
    bpy.context.view_layer.update()
    bar, mounted = cage.mount_barbell(rig, skin)
    local = rig.matrix_world.inverted() @ bar.matrix_world
    centre = np.array(local.translation)
    along = np.array((local.to_3x3() @ Vector((1., 0., 0.))).normalized())
    shoulders = {side: np.array(pose[cage.P + title + 'Arm'].head) for side, title, _ in SIDES}
    width = float(np.linalg.norm(shoulders['right'] - shoulders['left']))
    half = width * 1.5 / 2
    lengths = {side: tuple(pose[cage.P + title + part].length for part in ('Arm', 'ForeArm'))
               for side, title, _ in SIDES}
    grips = json.loads(rig['hand_grip'])
    hints = {side: shoulders[side] + np.array([0., -.04, -.10])
             for side, _, _ in SIDES}
    preferences = {side: {'elbow_below_wrist': True, 'elbow_clearance_m': .005,
                          'forearm_vertical_fraction': .8} for side, _, _ in SIDES}
    try:
        _, solutions, _ = solve_shared_bar(shoulders, centre, along, half, lengths, grips, hints,
            allow_adaptation=False, preferences=preferences)
    except ValueError as error:
        raise ValueError('These character proportions cannot hold the declared squat back-bar grip without wrist strain or changing limb lengths.') from error
    names, curl = set(), {}
    for side, title, _ in SIDES:
        curl.update(hand_rig.pose_basis(rig, title, grips[side]['rotations']))
        names.update(cage.P + title + part for part in ('Shoulder', 'Arm', 'ForeArm', 'Hand'))
        arm, forearm, hand = [pose[cage.P + title + part] for part in ('Arm', 'ForeArm', 'Hand')]
        solution = solutions[side]
        cage._aim_pose(arm, Vector(solution['elbow']) - arm.head)
        cage._aim_pose(forearm, Vector(solution['wrist']) - forearm.head)
        matrix = (Matrix(solution['turn'].tolist()) @ hand.bone.matrix_local.to_3x3()).to_4x4()
        matrix.translation = hand.head
        hand.matrix = matrix
        bpy.context.view_layer.update()
    for name, value in curl.items():
        pose[name].rotation_mode = 'QUATERNION'
        pose[name].rotation_quaternion = value
    bpy.context.view_layer.update()
    from bench_arm_roll import orient_upperarms
    axial = orient_upperarms(rig, roll_degrees=0.)
    names.update(curl)
    basis = {name: pose[name].matrix_basis.copy() for name in names}
    previous = {}
    worst_anchor = 0.
    for frame in range(first, last + 1):
        scene.frame_set(frame)
        for name, value in basis.items():
            pose[name].matrix_basis = value
        bpy.context.view_layer.update()
        cage._key(rig, previous, frame, names)
        local = rig.matrix_world.inverted() @ bar.matrix_world
        shaft = np.array(local.translation)
        axis = np.array((local.to_3x3() @ Vector((1., 0., 0.))).normalized())
        for side, title, sign in SIDES:
            hand = pose[cage.P + title + 'Hand']
            turn = np.array(hand.matrix.to_3x3() @ hand.bone.matrix_local.to_3x3().inverted())
            anchor = np.array(hand.head) + turn @ (np.array(grips[side]['B']) - grips[side]['wrist'])
            worst_anchor = max(worst_anchor, float(np.linalg.norm(anchor - (shaft + sign * half * axis))))
    result = measure(rig, skin, bar, first, last)
    result.update(max_grip_anchor_error_mm=worst_anchor * 1000,
                  worst_wrist_bend_deg=max(solution['bend_deg'] for solution in solutions.values()),
                  requested_grip_ratio=1.5, grip_ratio=1.5,
                  grip_source='declared 1.5-times fitted joint-shoulder width; no calibrated photograph width',
                  back_bar=mounted, arm_roll={**axial, 'source_twist_calibrated': False},
                  foot_staging=planted,
                  path_adjustments=[{'frame': frame, 'adjustment_mm': 0.} for frame in range(first, last + 1)])
    if worst_anchor > .002:
        raise ValueError('The fixed squat hold did not preserve both closed palms on the shaft.')
    return bar, result
