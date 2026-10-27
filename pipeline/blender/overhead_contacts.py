"""Fitted full-range overhead presses on a fixed-length mascot arm chain."""
import bpy
import numpy as np
from mathutils import Matrix, Vector

import mpfb_cage as cage
import skin_wrap
from row_motion import solve_contact, solve_shared_bar
from overhead_range import progress_at, summarize
from keypose_clip import slerp

STANDING = 'Standing Barbell Overhead Press (Shoulder Press)'
SEATED = 'Dumbbell Shoulder Press (seated overhead press)'
EXERCISES = {STANDING, SEATED}
SIDES = (('left', 'Left', -1), ('right', 'Right', 1))


def prepare(rig, skin, first, last, phase_frames, height, grips, lengths, separate=False):
    progress_at(first, first, last, phase_frames)
    scene, pose = bpy.context.scene, rig.pose.bones
    scene.frame_set(first)
    shoulders = {side: np.array(pose[cage.P + title + 'Arm'].head) for side, title, _ in SIDES}
    origin = (shoulders['left'] + shoulders['right']) / 2
    along = shoulders['right'] - shoulders['left']; along[2] = 0.; along /= np.linalg.norm(along)
    forward = np.cross([0., 0., 1.], along)
    width = float(np.linalg.norm(shoulders['right'] - shoulders['left']))
    half = .85 * width
    bottom = {}
    bottom_preferences = {side: {'elbow_deg': 90., 'elbow_below_wrist': True,
        'forearm_vertical_fraction': .95, 'minimum_forearm_vertical_fraction': .5,
        'min_elbow_y': float(shoulders[side][1] - .015 * height)} for side, _, _ in SIDES}
    hints = {side: shoulders[side] + sign * along * .06 * height + forward * .12 * height
        - np.array([0., 0., .12 * height]) for side, _, sign in SIDES}
    if separate:
        for side, _, sign in SIDES:
            point = origin + sign * along * .23 * height + forward * .10 * height
            point[2] = shoulders[side][2] + .16 * height
            bottom[side] = point
        lower = {side: solve_contact(shoulders[side], bottom[side], along, sign, *lengths[side],
            grips[side], hints[side], preference=bottom_preferences[side]) for side, _, sign in SIDES}
    else:
        centre = origin + forward * .20 * height
        centre[2] = origin[2] + .06 * height
        _, lower, _ = solve_shared_bar(shoulders, centre, along, half, lengths, grips, hints,
            allow_adaptation=False, preferences=bottom_preferences)
        bottom = {side: centre + sign * half * along for side, _, sign in SIDES}
    hand_turns = {side: solution['turn'].copy() for side, solution in lower.items()}
    preferences = {side: {'elbow_deg': 178., 'min_elbow_deg': 175., 'elbow_below_wrist': True,
        'forearm_vertical_fraction': .98, 'reference_hand_turn': hand_turns[side], 'max_hand_turn_deg': 35.}
        for side, _, _ in SIDES}
    # Clavicle elevation changes the shoulder origins during an overhead lift.
    # Place lockout against its actual peak frame rather than the lower pose.
    scene.frame_set(phase_frames['end'])
    shoulders = {side: np.array(pose[cage.P + title + 'Arm'].head) for side, title, _ in SIDES}
    origin = (shoulders['left'] + shoulders['right']) / 2
    top, top_hand_turns = {}, {}
    if separate:
        for side, _, sign in SIDES:
            chosen = None
            for outward in (.035, .05, .02, .065, .08):
                for front in (.04, .06, .02, .08):
                    for z in np.arange(shoulders[side][2] + sum(lengths[side]) + .09 * height,
                                       shoulders[side][2] + .22 * height, -.0005 * height):
                        candidate = shoulders[side] + sign * along * outward * height + forward * front * height
                        candidate[2] = z
                        try:
                            trial = solve_contact(shoulders[side], candidate, along, sign, *lengths[side], grips[side],
                                hints[side], preference=preferences[side])
                        except ValueError:
                            continue
                        chosen = candidate
                        top_hand_turns[side] = trial['turn'].copy()
                        break
                    if chosen is not None: break
                if chosen is not None: break
            if chosen is None:
                raise ValueError('The seated press cannot reach a real lockout without changing its arm lengths.')
            top[side] = chosen
    else:
        chosen = None
        for front in (.04, .06, .02, .08):
            for lateral in (0., .002, -.002, .004, -.004, .006, -.006):
                for z in np.arange(origin[2] + max(sum(v) for v in lengths.values()) + .09 * height,
                                   origin[2] + .22 * height, -.0005 * height):
                    centre = origin + forward * front * height + along * lateral * height
                    centre[2] = z
                    try:
                        solve_shared_bar(shoulders, centre, along, half, lengths, grips, hints,
                            allow_adaptation=False, preferences=preferences)
                    except ValueError:
                        continue
                    chosen = centre
                    break
                if chosen is not None: break
            if chosen is not None: break
        if chosen is None:
            raise ValueError('The standing press cannot reach bilateral lockout at its fixed grip.')
        top = {side: chosen + sign * half * along for side, _, sign in SIDES}
    scene.frame_set(first)
    return {'bottom': bottom, 'top': top, 'along': along, 'forward': forward, 'half_grip': half,
        'hand_turns': hand_turns, 'top_hand_turns': top_hand_turns,
        'first': first, 'last': last, 'phase_frames': dict(phase_frames),
        'separate': separate, 'clearance_shifts': [], 'path_hand_turns': {},
        'motion_hand_roll_limit_deg': 5. if separate else 10.,
        'method': 'declared fitted shoulder-level press to stacked overhead lockout; constant limb lengths and bounded overhand grip; photo depth is uncalibrated'}


def solve(plan, frame, shoulders, lengths, grips, previous, height, guard=None):
    p = progress_at(frame, plan['first'], plan['last'], plan['phase_frames'])
    along, forward = plan['along'], plan['forward']
    goals, hints, preferences = {}, {}, {}
    for side, _, sign in SIDES:
        goals[side] = plan['bottom'][side] * (1. - p) + plan['top'][side] * p
        # The loaded bar clears the front of the mascot's face on its way up;
        # dumbbells stay outside the face rather than crossing through it.
        if not plan['separate']:
            goals[side] += forward * .11 * height * np.sin(np.pi * p)
        hints[side] = shoulders[side] + sign * along * .06 * height + forward * .10 * height
        hints[side][2] += (-.12 + .39 * p) * height
        reference_turn = plan['path_hand_turns'].get(side, plan['hand_turns'][side])
        if plan['separate'] and side in plan['path_hand_turns']:
            reference_turn = slerp(reference_turn, plan['top_hand_turns'][side], p)
        preferences[side] = {'elbow_deg': 90. + 88. * p, 'elbow_below_wrist': True,
            'forearm_vertical_fraction': .95, 'minimum_forearm_vertical_fraction': .3,
            'min_elbow_y': float(shoulders[side][1] - .015 * height),
            'reference_hand_turn': reference_turn,
            'max_hand_turn_deg': plan['motion_hand_roll_limit_deg'] if side in plan['path_hand_turns'] else 35.}
        if p >= 1. - 1e-8:
            preferences[side]['min_elbow_deg'] = 175.
    history = previous
    if plan['separate']:
        solved = {side: solve_contact(shoulders[side], goals[side], along, sign, *lengths[side],
            grips[side], hints[side], previous=(history.get(side) or {}).get('elbow'),
            previous_turn=(history.get(side) or {}).get('turn'), preference=preferences[side])
            for side, _, sign in SIDES}
        if abs(solved['left']['elbow_deg'] - solved['right']['elbow_deg']) > 3.:
            goal = float(np.mean([v['elbow_deg'] for v in solved.values()]))
            trial = {side: solve_contact(shoulders[side], goals[side], along, sign, *lengths[side],
                grips[side], hints[side], previous=(history.get(side) or {}).get('elbow'),
                previous_turn=(history.get(side) or {}).get('turn'),
                preference={**preferences[side], 'elbow_deg': goal}) for side, _, sign in SIDES}
            if abs(trial['left']['elbow_deg'] - trial['right']['elbow_deg']) < abs(solved['left']['elbow_deg'] - solved['right']['elbow_deg']):
                solved = trial
        if frame == plan['first']:
            plan['path_hand_turns'] = {side: value['turn'].copy() for side, value in solved.items()}
        return goals, solved, 0.
    centre = (goals['left'] + goals['right']) / 2
    if guard is not None:
        centre, shift = guard(centre)
        plan['clearance_shifts'].append({'frame': frame, 'forward_mm': float(shift * 1000)})
    try:
        _, solved, adjustment = solve_shared_bar(shoulders, centre, along, plan['half_grip'], lengths,
            grips, hints, history, allow_adaptation=False, preferences=preferences)
    except ValueError as error:
        raise ValueError(f'The overhead press cannot preserve a full-range closed grip at frame {frame}.') from error
    if abs(solved['left']['elbow_deg'] - solved['right']['elbow_deg']) > 3.:
        for goal in (float(np.mean([v['elbow_deg'] for v in solved.values()])),
                     min(v['elbow_deg'] for v in solved.values()), max(v['elbow_deg'] for v in solved.values())):
            coupled = {side: {**value, 'elbow_deg': goal} for side, value in preferences.items()}
            _, trial, _ = solve_shared_bar(shoulders, centre, along, plan['half_grip'], lengths,
                grips, hints, history, allow_adaptation=False, preferences=coupled)
            if abs(trial['left']['elbow_deg'] - trial['right']['elbow_deg']) < abs(solved['left']['elbow_deg'] - solved['right']['elbow_deg']):
                solved = trial
    if frame == plan['first']:
        plan['path_hand_turns'] = {side: value['turn'].copy() for side, value in solved.items()}
    return centre, solved, adjustment


def apply_solution(rig, side, solution):
    title = dict((s, t) for s, t, _ in SIDES)[side]
    arm, forearm, hand = [rig.pose.bones[cage.P + title + part] for part in ('Arm', 'ForeArm', 'Hand')]
    cage._aim_pose(arm, Vector(solution['elbow']) - arm.head)
    cage._aim_pose(forearm, Vector(solution['wrist']) - forearm.head)
    matrix = (Matrix(solution['turn'].tolist()) @ hand.bone.matrix_local.to_3x3()).to_4x4()
    matrix.translation = hand.head
    hand.matrix = matrix
    bpy.context.view_layer.update()


def measure(rig, skin, props, separate, grips, first, last, height, phases):
    scene, pose, world = bpy.context.scene, rig.pose.bones, rig.matrix_world
    groups = {g.index for g in skin.vertex_groups if g.name == cage.P + 'Head'}
    mask = np.array([sum(g.weight for g in vertex.groups if g.group in groups) > .5 for vertex in skin.data.vertices])
    samples = []
    for frame in range(first, last + 1):
        scene.frame_set(frame)
        points = {label: {side: list(world @ pose[cage.P + title + bone].head) for side, title, _ in SIDES}
            for label, bone in (('shoulders_m', 'Arm'), ('hips_m', 'UpLeg'), ('elbows_m', 'ForeArm'), ('wrists_m', 'Hand'))}
        handles = {}
        for side, title, _ in SIDES:
            hand = pose[cage.P + title + 'Hand']
            turn = np.array(hand.matrix.to_3x3() @ hand.bone.matrix_local.to_3x3().inverted())
            anchor = np.array(hand.head) + turn @ (np.array(grips[side]['B']) - grips[side]['wrist'])
            handles[side] = list(world @ Vector(anchor))
        samples.append({'frame': frame, **points, 'handles_m': handles,
            'hip_m': list(world @ pose[cage.P + 'Hips'].head),
            'spine_m': list(world @ pose[cage.P + 'Spine2'].head),
            'head_m': list(world @ pose[cage.P + 'Head'].head),
            'head_top_m': float(skin_wrap.evaluated(skin)[0][mask, 2].max())})
    scene.frame_set(first)
    return summarize(samples, height, phases)
