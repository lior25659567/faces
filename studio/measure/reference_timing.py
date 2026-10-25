"""Use actual generated pose times instead of choosing frames that fit the targets.

Captured clips still use metric-based selection. Generated reference motion records
its key times, so searching for the closest angle would judge a different instant.
"""


def recorded_phase_frames(per_frame, reference_motion, skip=()):
    """The validated start/middle/end map, or None for a non-generated clip.

    Invalid generated metadata is an error, not permission to fall back to a
    better-looking frame. A source key that jumps must be repaired before judging.
    """
    if not isinstance(reference_motion, dict) or reference_motion.get('source') != 'keyposes':
        return None
    mode = reference_motion.get('mode', 'cycle')
    if mode not in ('cycle', 'hold', 'step'):
        raise ValueError('The generated motion has an unknown timing mode.')
    provided = reference_motion.get('phase_frames')
    if not isinstance(provided, dict):
        raise ValueError('The generated motion has no recorded reference pose times.')
    available = {row.get('frame') for row in per_frame if type(row.get('frame')) is int}
    blocked = set(skip)
    selected = {}
    for phase in ('start', 'middle', 'end'):
        frame = provided.get(phase)
        if type(frame) is not int or frame not in available:
            raise ValueError(f'The generated {phase} pose time is missing or outside the measured clip.')
        if frame in blocked:
            raise ValueError(f'The generated {phase} pose jumps at frame {frame}; repair the motion before comparing it.')
        selected[phase] = frame
    start, middle, end = (selected[phase] for phase in ('start', 'middle', 'end'))
    if mode == 'hold':
        if len({start, middle, end}) != 1:
            raise ValueError('A static hold must be compared at one recorded pose, without an invented cycle.')
    elif not start <= middle <= end or start == end:
        raise ValueError('The generated reference pose times are not in movement order.')
    return selected
