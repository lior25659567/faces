"""Declared complete pushdown endpoint while retaining photo disagreements."""
import numpy as np

from row_motion import solve_shared_bar
from keypose_clip import timeline


def solve(frame, first, last, phase_frames, shoulders, centre, along, half, lengths, grips, hints, previous, height, guard, fps):
    end = phase_frames.get('end') if isinstance(phase_frames, dict) else None
    if type(end) is not int or not first < end <= last:
        raise ValueError('A complete pushdown requires its recorded start and extension frames.')
    # Match the same generated forward/hold/return clock as the clip. The
    # generic pushdown starts at 90 degrees and extends rather than lowering.
    parameters = timeline(fps, 'Tricep Pushdown (cable, bar or rope attachment)')
    if len(parameters) != last - first + 1:
        raise ValueError('The pushdown correction requires the exact generated phase clock.')
    fraction = parameters[frame - first]
    goal = 90. + 85. * fraction
    preferences = {side: {'elbow_deg': goal, 'upperarm_deg': 12.,
                         'torso_down': np.array([0., 0., -1.])} for side in shoulders}
    if frame == end:
        for preference in preferences.values():
            preference['min_elbow_deg'] = 175.
    origin = np.mean(list(shoulders.values()), axis=0)
    toward = origin - centre
    toward -= along * (toward @ along)
    forward = np.cross([0., 0., 1.], along)
    vertical = np.array([0., 0., 1.])
    def attempt(target):
        solutions = solve_shared_bar(shoulders, target, along, half, lengths, grips, hints,
            previous, allow_adaptation=False, preferences=preferences)[1]
        if any(abs(value['elbow_deg'] - goal) > 3. for value in solutions.values()):
            raise ValueError('The fitted handle cannot preserve the declared complete pushdown range.')
        return solutions
    from motion_geometry import collision_safe_target
    return collision_safe_target(centre, (toward, forward, -forward, vertical, -vertical,
        forward - vertical, forward + vertical), guard, attempt)
