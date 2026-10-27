"""Blender bridge for the explicit, reference-timed flat bench correction."""
import json
import bpy
import numpy as np

import bench_press
import mpfb_cage as cage
from row_motion import solve_shared_bar

SIDES = (('left', 'Left', -1), ('right', 'Right', 1))


def prepare(rig, skin, first, last, height, grips, lengths, guard, phase_frames):
    scene, pose = bpy.context.scene, rig.pose.bones
    scene.frame_set(first)
    if phase_frames is None:
        raise ValueError('The bench correction requires the recorded reference phase frames.')
    bench_press.progress_at(first, first, last, phase_frames)
    shoulders = {s: np.array(pose[cage.P + t + 'Arm'].head) for s, t, _ in SIDES}
    origin = (shoulders['left'] + shoulders['right']) / 2
    along = shoulders['right'] - shoulders['left']; along[2] = 0.
    width = float(np.linalg.norm(along)); along /= width
    body = np.array(pose[cage.P + 'Hips'].head) - origin; body[2] = 0.
    torso_length = float(np.linalg.norm(body)); body /= torso_length
    # This is a fitted medium-grip silhouette for the mascot's long-arm variant,
    # not a width measurement recovered from an oblique photograph.
    half = width
    bottom = origin + body * .5 * torso_length
    bottom[2] = origin[2] + .03 * height
    bottom, bottom_guard_shift = guard(bottom)
    hints = {s: shoulders[s] + sign * along * .12 * height + body * .5 * torso_length
             - np.array([0., 0., .02 * height]) for s, _, sign in SIDES}
    lower_preferences = {s: {'elbow_deg': 90., 'elbow_below_wrist': True,
        'forearm_vertical_fraction': .98, 'minimum_forearm_vertical_fraction': float(np.cos(np.radians(29.5))),
        'max_elbow_z': float(origin[2] + .02 * height), 'elbow_forward_axis': body,
        'minimum_elbow_forward': float(shoulders[s] @ body)} for s, _, _ in SIDES}
    _, lower_solutions, _ = solve_shared_bar(shoulders, bottom, along, half, lengths, grips, hints,
        allow_adaptation=False, preferences=lower_preferences)
    hand_turns = {side: solution['turn'] for side, solution in lower_solutions.items()}
    top = origin + body * .015 * height
    start_z = origin[2] + max(sum(v) for v in lengths.values()) + .09 * height
    preferences = {s: {'elbow_deg': 178., 'min_elbow_deg': 175., 'elbow_below_wrist': True,
                      'forearm_vertical_fraction': .98,
                      'reference_hand_turn': hand_turns[s], 'max_hand_turn_deg': 30.} for s, _, _ in SIDES}
    chosen, lateral_shift = None, 0.
    for shift in (0., .002, -.002, .004, -.004, .006, -.006, .008, -.008, .010, -.010):
        for z in np.arange(start_z, bottom[2] + .12 * height, -.0005 * height):
            top = origin + body * .015 * height + along * shift * height
            top[2] = z
            try:
                _, trial, _ = solve_shared_bar(shoulders, top, along, half, lengths, grips, hints,
                    allow_adaptation=False, preferences=preferences)
            except ValueError:
                continue
            chosen, lateral_shift = top.copy(), shift * height
            break
        if chosen is not None:
            break
    if chosen is None:
        raise ValueError('The bench profile cannot reach a real bilateral lockout without changing limb lengths.')
    return {'top': chosen, 'bottom': bottom, 'along': along, 'body': body, 'torso_length': torso_length,
        'half_grip': half, 'phase_frames': dict(phase_frames), 'first': first, 'last': last,
        'origin': origin, 'bottom_guard_shift': bottom_guard_shift, 'hand_turns': hand_turns,
        'lockout_lateral_fit_mm': float(lateral_shift * 1000),
        'method': 'explicit fixed-grip shoulder-lockout to lower-chest trajectory, using recorded reference phase timing; source depth is uncalibrated'}


def solve(plan, frame, shoulders, lengths, grips, previous, height):
    p = bench_press.progress_at(frame, plan['first'], plan['last'], plan['phase_frames'])
    centre = bench_press.path_at(plan['top'], plan['bottom'], p)
    hints = {}
    for side, _, sign in SIDES:
        bottom = shoulders[side] + sign * plan['along'] * .10 * height + plan['body'] * .5 * plan['torso_length']
        bottom[2] = plan['origin'][2] - .04 * height
        target = plan['top'] + sign * plan['half_grip'] * plan['along']
        top = shoulders[side] + (target - shoulders[side]) * lengths[side][0] / sum(lengths[side])
        hints[side] = top * (1. - p) + bottom * p
    preferences = {s: {'elbow_deg': 178. - 100. * p,
        'elbow_below_wrist': True, 'forearm_vertical_fraction': .98,
        'minimum_forearm_vertical_fraction': float(np.cos(np.radians(29.5))),
        'reference_hand_turn': plan['hand_turns'][s], 'max_hand_turn_deg': 30.,
        'elbow_forward_axis': plan['body'],
        'minimum_elbow_forward': float(shoulders[s] @ plan['body'] - .020 * (1. - p) * height),
        'max_elbow_z': float(plan['origin'][2] + (.22 - .20 * p) * height)} for s, _, _ in SIDES}
    if frame == plan['first']:
        for preference in preferences.values():
            preference['min_elbow_deg'] = 175.
    elbow_history = {side: value['elbow'] if isinstance(value, dict) else value for side, value in previous.items()}
    _, solutions, adjustment = solve_shared_bar(shoulders, centre, plan['along'], plan['half_grip'],
        lengths, grips, hints, elbow_history, allow_adaptation=False, preferences=preferences)
    # The fitted left/right forearms differ slightly in rest length. Couple the
    # elbow objective near extension so independent hand-roll searches cannot
    # turn that small rest asymmetry into visibly different arm bends.
    if abs(solutions['left']['elbow_deg'] - solutions['right']['elbow_deg']) > 3.:
        goals = [solution['elbow_deg'] for solution in solutions.values()]
        for goal in (float(np.mean(goals)), min(goals), max(goals)):
            coupled = {side: {**value, 'elbow_deg': goal} for side, value in preferences.items()}
            _, trial, _ = solve_shared_bar(shoulders, centre, plan['along'], plan['half_grip'],
                lengths, grips, hints, elbow_history, allow_adaptation=False, preferences=coupled)
            if abs(trial['left']['elbow_deg'] - trial['right']['elbow_deg']) < abs(solutions['left']['elbow_deg'] - solutions['right']['elbow_deg']):
                solutions = trial
    return centre, solutions, adjustment


def measure(rig, bar, first, last, height, phase_frames):
    from pullup_range import included
    scene, pose, world = bpy.context.scene, rig.pose.bones, rig.matrix_world
    samples = []
    for frame in range(first, last + 1):
        scene.frame_set(frame)
        points = {label: {s: list(world @ pose[cage.P + t + bone].head) for s, t, _ in SIDES}
                  for label, bone in (('shoulders_m', 'Arm'), ('elbows_m', 'ForeArm'), ('wrists_m', 'Hand'))}
        samples.append({'frame': frame, 'bar_m': list(bar.matrix_world.translation),
            'hip_m': list(world @ pose[cage.P + 'Hips'].head), **points,
            'elbows_deg': {s: included(np.array(points['shoulders_m'][s]), np.array(points['elbows_m'][s]),
                                     np.array(points['wrists_m'][s])) for s, _, _ in SIDES}})
    scene.frame_set(first)
    return bench_press.summarize(samples, height, phase_frames)
