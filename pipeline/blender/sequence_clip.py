"""Chronological pose samples and contact-derived root travel, without Blender.

This path never invents a return stroke or closes a walking sequence. A root
translation can be recovered from hip-relative landmarks only while a visibly
planted foot connects adjacent samples. An unobserved handoff is a hard boundary:
do not pin an occluded foot, reset the root, or substitute an in-place cycle.
Frontal single-camera depth remains inferred even when this continuity check passes.
"""
import math

import numpy as np

from keypose_clip import DOWN, bone_turns, character_frame, pose_at

SIDES = ('left', 'right')
CONTACT_STATES = ('flat', 'airborne', 'uncertain')


class SequenceError(ValueError):
    def __init__(self, message, report=None):
        super().__init__(message)
        self.report = report or {'ready': False, 'blockers': [message]}


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _samples(keyposes):
    if (keyposes.get('motion') or {}).get('mode') != 'sequence':
        raise SequenceError('A chronological sequence with source timestamps is required.')
    sequence = keyposes.get('sequence') or {}
    samples = sequence.get('samples') or []
    if len(samples) < 2:
        raise SequenceError('At least two readable chronological motion samples are required.')
    previous_time = -math.inf
    for index, sample in enumerate(samples):
        if sample.get('sequenceIndex') != index:
            raise SequenceError('Motion samples must retain every source sequence index in chronological order.')
        time = sample.get('timeSeconds')
        if not _finite(time) or time < 0 or time <= previous_time:
            raise SequenceError('Motion sample timestamps must be finite and strictly increasing.')
        previous_time = time
        if sample.get('error') or not isinstance(sample.get('world'), dict):
            raise SequenceError(f"Motion sample {index} has no readable body landmarks.")
        if any(not isinstance(point, (list, tuple)) or len(point) != 3 or not all(_finite(v) for v in point)
               for point in sample['world'].values()):
            raise SequenceError(f"Motion sample {index} has invalid body landmarks.")
        states = CONTACT_STATES + ('heel', 'toe') if sequence.get('contactModel') == 'named_heel_toe' else CONTACT_STATES
        if any((sample.get('footContacts') or {}).get(side) not in states for side in SIDES):
            raise SequenceError(f"Motion sample {index} needs observed per-foot contact states.")
        for name in ('left_hip', 'right_hip') + tuple(f'{s}_{j}' for s in SIDES for j in ('ankle', 'heel', 'foot_index')):
            point = sample['world'].get(name)
            if not isinstance(point, (list, tuple)) or len(point) != 3 or not all(_finite(v) for v in point):
                raise SequenceError(f"Motion sample {index} has invalid {name.replace('_', ' ')} landmarks.")
    return samples


def sequence_bodies(samples):
    """All hip-relative landmarks in one fixed character frame, preserving asymmetry."""
    first_world = {name: np.asarray(point, dtype=float) for name, point in samples[0]['world'].items()}
    try:
        rows, _ = character_frame(first_world)
    except ValueError as error:
        raise SequenceError(str(error)) from error
    out = []
    for sample in samples:
        world = {name: np.asarray(point, dtype=float) for name, point in sample['world'].items()}
        origin = (world['left_hip'] + world['right_hip']) / 2
        out.append({name: rows @ (point - origin) for name, point in world.items()})
    return out


def visible_support(sample, min_visibility=.8):
    """Only explicitly observed flat support with a readable ankle can constrain travel."""
    observed = sample.get('observedSupport')
    allowed = set(SIDES) if observed == 'both' else {observed} if observed in SIDES else set()
    return {side for side in allowed if sample['footContacts'][side] == 'flat'
            and _finite((sample.get('visibility') or {}).get(f'{side}_ankle'))
            and sample['visibility'][f'{side}_ankle'] >= min_visibility}


def contact_roots(points, supports, *, initial=None, tolerance=.003):
    """Recover translation from shared planted points; report every unobservable interval.

    ``points`` are hip-relative per-side foot positions in one fixed frame. The
    tolerance has the same units as those points. No point is used across an
    airborne, uncertain, low-visibility, or unobserved contact state.
    """
    if len(points) != len(supports) or not points:
        raise SequenceError('Foot points and support samples must match.')
    if not _finite(tolerance) or tolerance < 0:
        raise SequenceError('The contact tolerance must be finite and nonnegative.')
    roots = [np.zeros(3) if initial is None else np.asarray(initial, dtype=float)]
    blockers, intervals = [], []
    if not supports[0]:
        blockers.append('Sample 0 has no visible planted foot to establish the initial root.')
        roots[0] = None
    for index in range(1, len(points)):
        shared = sorted(supports[index - 1] & supports[index])
        interval = {'fromIndex': index - 1, 'toIndex': index, 'visibleSupport': shared}
        if not shared:
            blockers.append(f'Samples {index - 1}→{index} have no shared visible planted foot; '
                            'forward root travel across this handoff is unobserved.')
            roots.append(None)
        elif roots[-1] is None:
            roots.append(None)
        else:
            candidates = [roots[-1] + points[index - 1][s] - points[index][s] for s in shared]
            root = np.mean(candidates, axis=0)
            residual = max(float(np.linalg.norm(candidate - root)) for candidate in candidates)
            interval['contactResidual'] = residual
            if residual > tolerance:
                blockers.append(f'Samples {index - 1}→{index} have incompatible visible foot anchors '
                                f'({residual:.6f} versus tolerance {tolerance:.6f}).')
                roots.append(None)
            else:
                roots.append(root)
        intervals.append(interval)
    return {'ready': not blockers, 'blockers': blockers,
            'roots': [root.tolist() if root is not None else None for root in roots], 'intervals': intervals}


def inspect_sequence(keyposes, min_visibility=.8, tolerance=.003):
    samples = _samples(keyposes)
    if (keyposes.get('sequence') or {}).get('contactModel') == 'named_heel_toe':
        anchors, intervals, blockers = [], [], []
        for index, sample in enumerate(samples):
            points = sample.get('observedContactPoints')
            if not isinstance(points, dict) or set(points) != set(SIDES):
                raise SequenceError(f'Motion sample {index} needs observed heel/toe contact identities for both feet.')
            active = set()
            for side in SIDES:
                named = points[side]
                if (not isinstance(named, list) or len(set(named)) != len(named)
                        or any(name not in ('heel', 'toe') for name in named)):
                    raise SequenceError(f'Motion sample {index} has invalid named contact points.')
                state = sample['footContacts'][side]
                expected = {'flat': {'heel', 'toe'}, 'heel': {'heel'}, 'toe': {'toe'},
                            'airborne': set(), 'uncertain': set()}[state]
                if set(named) != expected:
                    raise SequenceError(f'Motion sample {index} contact points disagree with its observed foot state.')
                if named and sample.get('observedSupport') not in (side, 'both'):
                    raise SequenceError(f'Motion sample {index} anchors an unobserved supporting foot.')
                active.update(side + '.' + name for name in named)
            if not active:
                blockers.append(f'Motion sample {index} has no observed supporting heel or toe.')
            if anchors:
                shared = sorted(active & anchors[-1])
                intervals.append({'fromIndex': index - 1, 'toIndex': index, 'observedContactPoints': shared})
                if not shared:
                    blockers.append(f'The handoff from samples {index - 1} to {index} has no shared observed contact point.')
            anchors.append(active)
        return {'ready': not blockers, 'input_ready': not blockers, 'blockers': blockers,
            'mode': 'sequence', 'playback': 'once', 'contactModel': 'named_heel_toe',
            'sampleCount': len(samples), 'sourceTimes': [s['timeSeconds'] for s in samples],
            'observedContacts': [sorted(points) for points in anchors], 'intervals': intervals,
            'rootFitRequired': True, 'sourceInferredDepth': True, 'calibrated': False,
            'completeClosedCycle': False,
            'note': 'Observed support identity is connected. Model-space root fitting and actual skin contacts must still pass before rendering.'}
    bodies = sequence_bodies(samples)
    supports = [visible_support(sample, min_visibility) for sample in samples]
    points = [{s: (body[f'{s}_heel'] + body[f'{s}_foot_index']) / 2 for s in SIDES} for body in bodies]
    result = contact_roots(points, supports, tolerance=tolerance)
    return {**result, 'mode': 'sequence', 'playback': 'once', 'sampleCount': len(samples),
            'sourceTimes': [sample['timeSeconds'] for sample in samples],
            'sourceInferredDepth': True, 'calibrated': False,
            'completeClosedCycle': bool((keyposes.get('sequence') or {}).get('completeClosedCycle'))}


def _leg_frame(pose, rest, legs):
    """Leg joints about the pelvis; do not ground each frame or lock both feet."""
    mid = (rest['LeftUpLeg'] + rest['RightUpLeg']) / 2
    points = {}
    for side in ('Left', 'Right'):
        hip = pose['Hips'] @ (rest[f'{side}UpLeg'] - mid)
        knee = hip + pose[f'{side}UpLeg'] @ legs * np.linalg.norm(rest[f'{side}Leg'] - rest[f'{side}UpLeg'])
        ankle = knee + pose[f'{side}Leg'] @ legs * np.linalg.norm(rest[f'{side}Foot'] - rest[f'{side}Leg'])
        toe = ankle + pose[f'{side}Foot'] @ (rest[f'{side}ToeBase'] - rest[f'{side}Foot'])
        points.update({f'{side}UpLeg': hip, f'{side}Leg': knee, f'{side}Foot': ankle, f'{side}ToeBase': toe})
    return points


def motion_from_sequence(keyposes, rest, fps=30, legs=DOWN, contact_tolerance=.003):
    """Pure adapter for a future Blender Clip builder; raises on ambiguous travel.

    ``rest`` is the same character-frame rest-joint dictionary used by
    keypose_clip. Returned turns/frames run once in source-time order. Their
    contact-derived root must still pass rendered sole/ground and stepping checks
    after retargeting. Passing this source continuity check is not automatic-build
    approval or metric camera calibration.
    """
    if not _finite(fps) or fps <= 0 or fps > 120:
        raise SequenceError('Sequence frame rate must be between 0 and 120 fps.')
    report = inspect_sequence(keyposes, tolerance=contact_tolerance)
    if not report['ready']:
        raise SequenceError(' '.join(report['blockers']), report)
    samples = _samples(keyposes)
    bodies = sequence_bodies(samples)
    try:
        keys = [bone_turns(body, body, rest, square=False, legs=legs) for body in bodies]
    except (KeyError, ValueError) as error:
        raise SequenceError('The chronological sequence has missing or invalid body directions.') from error
    joint_keys = [_leg_frame(pose, rest, legs) for pose in keys]
    foot_keys = [{s: points[f'{s.title()}ToeBase'] for s in SIDES} for points in joint_keys]
    supports = [visible_support(sample) for sample in samples]
    initial = (rest['LeftUpLeg'] + rest['RightUpLeg']) / 2
    roots = contact_roots(foot_keys, supports, initial=initial, tolerance=contact_tolerance)
    if not roots['ready']:
        raise SequenceError(' '.join(roots['blockers']), {**report, **roots})
    origin = samples[0]['timeSeconds']
    marks = [sample['timeSeconds'] - origin for sample in samples]
    frame_times = [min(index / fps, marks[-1]) for index in range(math.ceil(marks[-1] * fps) + 1)]
    turns, frames = [], []
    for time in frame_times:
        interval = min(next((i for i in range(len(marks) - 1) if time <= marks[i + 1]), len(marks) - 2), len(marks) - 2)
        pose = pose_at(keys, marks, time)
        points = _leg_frame(pose, rest, legs)
        shared = supports[interval] & supports[interval + 1]
        candidates = [roots['roots'][interval] + foot_keys[interval][s] - points[f'{s.title()}ToeBase'] for s in shared]
        root = np.mean(candidates, axis=0)
        if any(np.linalg.norm(candidate - root) > contact_tolerance for candidate in candidates):
            raise SequenceError('Interpolated feet cannot both retain their visible planted anchors.')
        turns.append(pose)
        frames.append({name: point + root for name, point in points.items()})
    return {'turns': turns, 'frames': frames, 'fps': float(fps), 'first': 1,
            'reference_motion': {'source': 'keyposes', 'mode': 'sequence', 'playback': 'once',
              'phase_frames': {sample['source'].get('phase'): 1 + int(round((sample['timeSeconds'] - origin) * fps))
                               for sample in samples if not sample.get('motionOnly') and (sample.get('source') or {}).get('phase')},
              'sample_frames': [1 + int(round(mark * fps)) for mark in marks],
              'source_times': [sample['timeSeconds'] for sample in samples],
              'sourceInferredDepth': True, 'calibrated': False},
            'report': report}


def clip_from_sequence(rest, keyposes, fps=16, label='keyposes.json', legs=DOWN):
    """Blender Clip for one observed chronological step, without a return cycle."""
    import retarget
    from mathutils import Matrix, Vector
    report = inspect_sequence(keyposes)
    if not report['ready']:
        raise SequenceError(' '.join(report['blockers']), report)
    if report.get('contactModel') != 'named_heel_toe':
        raise SequenceError('Walking rendering requires the reviewed named heel/toe contact model.')
    if not _finite(fps) or not 0 < fps <= 120:
        raise SequenceError('Sequence playback needs a finite positive frame rate up to 120 fps.')
    samples = _samples(keyposes)
    rows = np.asarray(retarget.facing(rest.rest)[0])
    to_world = np.column_stack([-rows[0], -rows[1], rows[2]])
    rest_points = {name: to_world.T @ np.array(point) for name, point in rest.rest.items()}
    bodies = sequence_bodies(samples)
    keys = [bone_turns(body, body, rest_points, square=False, legs=legs) for body in bodies]
    from motion_profiles import prepare_key_turns
    keys = prepare_key_turns(keyposes.get('exercise'), keys)
    bones = sorted(keys[0])
    if any(name not in rest.rest_turns for name in bones):
        raise SequenceError('The pinned rest skeleton lacks the required walking bones.')
    origin = samples[0]['timeSeconds']
    marks = [sample['timeSeconds'] - origin for sample in samples]
    times = [min(index / fps, marks[-1]) for index in range(math.ceil(marks[-1] * fps) + 1)]
    poses = [pose_at(keys, marks, time) for time in times]
    pelvis = (rest_points['LeftUpLeg'] + rest_points['RightUpLeg']) / 2
    frames = [{name: Vector((to_world @ (point + pelvis)).tolist())
               for name, point in _leg_frame(pose, rest_points, legs).items()} for pose in poses]
    turns = [{name: Matrix((to_world @ pose[name] @ to_world.T).tolist()).to_quaternion() @ rest.rest_turns[name]
              for name in bones} for pose in poses]
    phases = {}
    for sample in samples:
        phase = sample.get('phase') or (sample.get('source') or {}).get('phase')
        if phase in ('start', 'middle', 'end') and not sample.get('motionOnly'):
            phases[phase] = min(range(len(times)), key=lambda i: abs(times[i] - (sample['timeSeconds'] - origin))) + 1
    if set(phases) != {'start', 'middle', 'end'}:
        raise SequenceError('The chronological step needs its three actual observed comparison anchors.')
    clip = retarget.Clip(path=str(label), fps=float(fps), first=1, rest=dict(rest.rest), frames=frames,
        bones=dict(rest.bones), rest_turns={name: rest.rest_turns[name] for name in bones}, turns=turns,
        notes=['Observed chronological step; one-shot playback; frontal depth remains inferred.'])
    clip.reference_motion = {'source': 'keyposes', 'mode': 'sequence', 'playback': 'once',
        'phase_frames': phases, 'sample_frames': [min(range(len(times)), key=lambda i: abs(times[i] - mark)) + 1 for mark in marks],
        'source_times': [sample['timeSeconds'] for sample in samples],
        'contactModel': 'named_heel_toe', 'sourceInferredDepth': True, 'calibrated': False,
        'observed_contacts': [sample['observedContactPoints'] for sample in samples]}
    return clip
