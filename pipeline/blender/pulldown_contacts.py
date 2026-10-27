"""Constrain a front pulldown to the downward elbow branch."""
import bpy
import numpy as np

import mpfb_cage as cage
from row_motion import solve_shared_bar
from pulldown_range import progress_at, summarize

SIDES = (('left', 'Left', -1), ('right', 'Right', 1))


def prepare(rig, first, last, phase_frames, shoulders, along, half_grip, lengths, grips, height, guard):
    progress_at(first, first, last, phase_frames)
    origin = (shoulders['left'] + shoulders['right']) / 2
    forward = np.cross([0., 0., 1.], along)
    bottom = origin + forward * .12 * height
    bottom[2] = origin[2] - .03 * height
    bottom, shift = guard(bottom)
    hints = {side: shoulders[side] + sign * along * .08 * height + forward * .06 * height
             - np.array([0., 0., .14 * height]) for side, _, sign in SIDES}
    preferences = {side: {'elbow_deg': 80., 'elbow_below_wrist': True,
        'elbow_clearance_m': .005 * height, 'forearm_vertical_fraction': .9,
        'minimum_forearm_vertical_fraction': .1,
        'max_elbow_z': float(shoulders[side][2] - .02 * height),
        'min_elbow_y': float(shoulders[side][1] - .015 * height)} for side, _, _ in SIDES}
    _, lower, _ = solve_shared_bar(shoulders, bottom, along, half_grip, lengths, grips, hints,
        allow_adaptation=False, preferences=preferences)
    hand_turns = {side: solution['turn'] for side, solution in lower.items()}
    preferences = {side: {'elbow_deg': 178., 'min_elbow_deg': 175., 'elbow_below_wrist': True,
        'forearm_vertical_fraction': .98, 'reference_hand_turn': hand_turns[side], 'max_hand_turn_deg': 35.}
        for side, _, _ in SIDES}
    top = None
    for forward_shift in (.08, .10, .12, .14, .06):
        for z in np.arange(origin[2] + max(sum(v) for v in lengths.values()) + .09 * height,
                           origin[2] + .20 * height, -.0005 * height):
            candidate = origin + forward * forward_shift * height
            candidate[2] = z
            try:
                solve_shared_bar(shoulders, candidate, along, half_grip, lengths, grips, hints,
                                 allow_adaptation=False, preferences=preferences)
            except ValueError:
                continue
            top = candidate
            break
        if top is not None:
            break
    if top is None:
        raise ValueError('The front pulldown cannot reach bilateral overhead extension at its fixed grip.')
    return {'top': top, 'bottom': bottom, 'along': along.copy(), 'half_grip': half_grip,
        'hand_turns': hand_turns, 'first_hand_turns': {}, 'max_hand_roll_deg': 0.,
        'bottom_guard_shift_mm': shift * 1000, 'clearance_path_shifts': [],
        'method': 'declared fitted overhead-extension to upper-chest path; uncalibrated photo depth is not treated as a mechanical target'}


def solve(rig, plan, frame, first, last, phase_frames, shoulders, lengths, grips, previous, height, guard):
    p = progress_at(frame, first, last, phase_frames)
    along, half_grip = plan['along'], plan['half_grip']
    centre = plan['top'] * (1. - p) + plan['bottom'] * p
    # The mascot's large face needs a forward bow between the overhead and
    # upper-chest endpoints. This is part of the disclosed mechanical path,
    # rather than an unreported change to a supposedly calibrated source path.
    centre, clearance_shift = guard(centre)
    plan['clearance_path_shifts'].append({'frame': frame, 'forward_mm': float(clearance_shift * 1000)})
    forward = np.cross([0., 0., 1.], along)
    hints, preferences = {}, {}
    for side, title, sign in SIDES:
        hints[side] = (shoulders[side] + sign * along * .06 * height + forward * .04 * height
                       + np.array([0., 0., (.16 - .30 * p) * height]))
        # Intermediate monocular elbow angles can select a different depth
        # branch even while the hands stay on the same shaft. Fixed lengths,
        # the bar path and one forward elbow pole determine the bend instead.
        preferences[side] = {'elbow_below_wrist': True, 'elbow_clearance_m': .005 * height,
            'forearm_vertical_fraction': .9, 'minimum_forearm_vertical_fraction': .1,
            'max_elbow_z': float(shoulders[side][2] + (.30 - .32 * p) * height),
            'min_elbow_y': float(shoulders[side][1] - .015 * height),
            'reference_hand_turn': plan['first_hand_turns'].get(side, plan['hand_turns'][side]),
            'max_hand_turn_deg': max(.5, 25. * np.sqrt(p)) if side in plan['first_hand_turns'] else 35.}
        if frame == first:
            preferences[side]['min_elbow_deg'] = 175.
    # Each movement position has the same solution on the pull and return.
    # A prior-frame objective otherwise creates two different arm paths.
    _, solutions, adjustment = solve_shared_bar(shoulders, centre, along, half_grip, lengths, grips, hints,
        {}, allow_adaptation=False, preferences=preferences)
    for side, solution in solutions.items():
        plan['first_hand_turns'].setdefault(side, solution['turn'].copy())
        change = plan['first_hand_turns'][side].T @ solution['turn']
        plan['max_hand_roll_deg'] = max(plan['max_hand_roll_deg'],
            float(np.degrees(np.arccos(np.clip((np.trace(change) - 1.) / 2., -1., 1.)))))
    return centre, solutions, adjustment


def measure(rig, bar, first, last, phase_frames):
    scene, pose, world = bpy.context.scene, rig.pose.bones, rig.matrix_world
    samples = []
    for frame in range(first, last + 1):
        scene.frame_set(frame)
        points = {label: {s: list(world @ pose[cage.P + t + bone].head) for s, t, _ in SIDES}
                  for label, bone in (('shoulders_m', 'Arm'), ('elbows_m', 'ForeArm'), ('wrists_m', 'Hand'))}
        samples.append({'frame': frame, 'bar_m': list(bar.matrix_world.translation),
            'hip_m': list(world @ pose[cage.P + 'Hips'].head), **points})
    scene.frame_set(first)
    import json
    height = json.loads(rig['skin_fit'])['measured']['height']
    return summarize(samples, height, phase_frames)
