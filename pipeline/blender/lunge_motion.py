"""Observed one-shot lunge timing and fixed-length contact geometry.

The frontal video supplies chronology and observed support identities. Its
monocular depth is approximate; the fitted step is not a calibrated recording.
No reverse stroke, hidden contact or final two-foot standing pose is invented.
"""
import math
import numpy as np

SIDES = ('left', 'right')
POINTS = tuple(f'{side}.{point}' for side in SIDES for point in ('heel', 'toe'))


def samples_from(reference):
    sequence = (reference or {}).get('sequence') or {}
    samples = sequence.get('samples')
    if sequence.get('contactModel') != 'named_heel_toe' or not isinstance(samples, list) or len(samples) < 3:
        raise ValueError('The lunge needs reviewed chronological heel/toe contact samples.')
    times = []
    for index, sample in enumerate(samples):
        time = sample.get('timeSeconds')
        if (sample.get('sequenceIndex') != index or isinstance(time, bool)
                or not isinstance(time, (int, float)) or not np.isfinite(time)
                or (times and time <= times[-1])):
            raise ValueError('Every lunge sample must retain its original ordered timestamp.')
        named_contacts(sample)
        times.append(float(time))
    for before, after in zip(samples, samples[1:]):
        if not named_contacts(before) & named_contacts(after):
            raise ValueError('The lunge has an unobserved support handoff; root travel cannot be fitted.')
    return samples


def named_contacts(sample):
    declared = sample.get('observedContactPoints') or {}
    states = sample.get('footContacts') or {}
    result = set()
    allowed = {'flat': {'heel', 'toe'}, 'heel': {'heel'}, 'toe': {'toe'}, 'airborne': set(), 'uncertain': set()}
    for side in SIDES:
        if states.get(side) not in allowed or set(declared.get(side, [])) != allowed[states[side]]:
            raise ValueError('Heel/toe anchors must match the visibly reviewed foot contact state.')
        result.update(f'{side}.{point}' for point in declared[side])
    return result


def timeline(samples, frame_times):
    """Keep only shared observed anchors between source samples."""
    times = np.asarray([sample['timeSeconds'] for sample in samples], dtype=float)
    rows = []
    for time in frame_times:
        if not np.isfinite(time) or time < times[0] - 1e-6 or time > times[-1] + 1e-6:
            raise ValueError('The lunge cannot extrapolate outside its observed recording.')
        nearest = int(np.argmin(abs(times - time)))
        if abs(times[nearest] - time) <= 1e-6:
            contacts = named_contacts(samples[nearest])
            states = dict(samples[nearest]['footContacts'])
        else:
            index = max(0, min(len(times) - 2, int(np.searchsorted(times, time) - 1)))
            a, b = samples[index:index + 2]
            contacts = named_contacts(a) & named_contacts(b)
            states = {}
            for side in SIDES:
                points = {name.split('.')[1] for name in contacts if name.startswith(side + '.')}
                states[side] = ('flat' if points == {'heel', 'toe'} else next(iter(points)) if points
                    else 'airborne' if a['footContacts'][side] == b['footContacts'][side] == 'airborne' else 'uncertain')
        if not contacts:
            raise ValueError('An interpolated lunge frame has no observed support.')
        rows.append({'time_s': float(time), 'contacts': sorted(contacts), 'foot_states': states})
    return rows


def foot_pitch(samples, frame_times, side):
    """Declared support postures; photo depth does not establish foot pitch."""
    times = [sample['timeSeconds'] for sample in samples]
    values = []
    for index, sample in enumerate(samples):
        state = sample['footContacts'][side]
        if state == 'toe':
            value = -15.
        elif state == 'heel':
            value = 15.
        elif state == 'uncertain' and index and samples[index - 1]['footContacts'][side] == 'toe':
            value = -10.
        elif state == 'uncertain' and index + 1 < len(samples) and samples[index + 1]['footContacts'][side] == 'flat':
            value = 10.
        else:
            value = 0.
        values.append(value)
    return np.interp(frame_times, times, values)


def descent_progress(samples, frame_times):
    """Use perspective-normalized visible hip height to time the down/up fit."""
    heights = []
    for sample in samples:
        pixels = sample.get('pixels') or {}
        try:
            hips = np.mean([pixels[side + '_hip'] for side in SIDES], axis=0)
            shoulders = np.mean([pixels[side + '_shoulder'] for side in SIDES], axis=0)
            support = named_contacts(sample)
            ground = np.mean([pixels[name.replace('.toe', '_foot_index').replace('.heel', '_heel')][1]
                              for name in support])
            torso = np.linalg.norm(hips - shoulders)
            if torso <= 1e-6 or not np.isfinite([*hips, *shoulders, ground]).all():
                raise ValueError()
            heights.append(float((ground - hips[1]) / torso))
        except (KeyError, TypeError, ValueError):
            raise ValueError('The lunge needs readable hip/shoulder and observed foot pixels.') from None
    middle = next((index for index, sample in enumerate(samples)
                   if sample.get('phase') == 'middle' or (sample.get('source') or {}).get('phase') == 'middle'),
                  int(np.argmin(heights)))
    if middle in (0, len(samples) - 1) or heights[0] - heights[middle] <= .1:
        raise ValueError('The observed sequence needs a real lowered position and subsequent rise.')
    progress = np.clip((heights[0] - np.asarray(heights)) / (heights[0] - heights[middle]), 0., 1.)
    progress[:middle + 1] = np.maximum.accumulate(progress[:middle + 1])
    progress[middle:] = np.maximum.accumulate(progress[middle:][::-1])[::-1]
    progress[0], progress[middle] = 0., 1.
    return np.interp(frame_times, [sample['timeSeconds'] for sample in samples], progress)


def contact_fit(points, supports, preferred_side=None):
    """Fit an approximate root while retaining each observed contact episode.

    New foot placement uses the source's current fitted geometry while another
    observed support anchors the root. Named points survive flat/toe changes.
    The actual Blender leg solve must still satisfy the floor and slip checks.
    """
    if not points or len(points) != len(supports):
        raise ValueError('Root fitting needs corresponding observed supports and foot positions.')
    previous, anchors, roots, snapshots = set(), {}, [], []
    for index, (frame, active) in enumerate(zip(points, supports)):
        active = set(active)
        if not active or not active <= set(POINTS) or any(np.asarray(frame[p]).shape != (3,)
                or not np.isfinite(frame[p]).all() for p in POINTS):
            raise ValueError('Root fitting requires finite named foot points and visible support.')
        shared = active & previous
        if index and not shared:
            raise ValueError('Root travel cannot cross an unobserved heel/toe handoff.')
        selected = {p for p in shared if p.startswith(str(preferred_side) + '.')} or shared
        shift = np.mean([anchors[p] - frame[p] for p in selected], axis=0) if selected else np.zeros(3)
        kept = {point: anchors[point] for point in shared}
        for point in sorted(active - shared):
            side = point.split('.')[0]
            related = next((p for p in kept if p.startswith(side + '.')), None)
            target = (kept[related] + frame[point] - frame[related] if related else frame[point] + shift).copy()
            target[2] = 0.
            kept[point] = target
        anchors = kept
        selected = {p for p in active if p.startswith(str(preferred_side) + '.')} or active
        shift = np.mean([anchors[p] - frame[p] for p in selected], axis=0)
        roots.append(shift)
        snapshots.append({point: value.copy() for point, value in anchors.items()})
        previous = active
    return np.asarray(roots), snapshots


def knee_position(hip, ankle, upper, lower, pole=(0., 1., 0.)):
    hip, ankle = np.asarray(hip, dtype=float), np.asarray(ankle, dtype=float)
    delta = ankle - hip
    distance = float(np.linalg.norm(delta))
    if not abs(upper - lower) + 1e-7 < distance < upper + lower - 1e-7:
        raise ValueError('The lunge target is outside the unchanged leg reach.')
    axis = delta / distance
    bend = np.asarray(pole, dtype=float) - axis * (np.asarray(pole) @ axis)
    if np.linalg.norm(bend) <= 1e-9:
        raise ValueError('The lunge needs an observed forward knee direction.')
    bend /= np.linalg.norm(bend)
    reach = (upper * upper + distance * distance - lower * lower) / (2 * distance)
    radius = math.sqrt(max(0., upper * upper - reach * reach))
    return hip + axis * reach + bend * radius


def root_height(root_xy, hip_offsets, ankles, lengths, desired, minimum_angle=75., maximum_angle=175.):
    low, high = -math.inf, math.inf
    for side, ankle in ankles.items():
        upper, lower = lengths[side]
        lateral = np.asarray(root_xy) + np.asarray(hip_offsets[side])[:2] - np.asarray(ankle)[:2]
        span = float(lateral @ lateral)
        d_min = upper * upper + lower * lower - 2 * upper * lower * math.cos(math.radians(minimum_angle))
        d_max = upper * upper + lower * lower - 2 * upper * lower * math.cos(math.radians(maximum_angle))
        if span >= d_max:
            raise ValueError('The observed step is too wide for the fixed mascot legs.')
        offset = float(ankle[2] - hip_offsets[side][2])
        low = max(low, offset + math.sqrt(max(0., d_min - span)))
        high = min(high, offset + math.sqrt(d_max - span))
    if low > high:
        raise ValueError('The fixed-length legs cannot share the observed lunge supports.')
    return float(np.clip(desired, low + 1e-5, high - 1e-5)), low, high


def eased(time, start, end):
    if not start < end:
        raise ValueError('A lunge phase needs distinct observed timestamps.')
    t = float(np.clip((time - start) / (end - start), 0., 1.))
    return t * t * t * (10. - 15. * t + 6. * t * t)


def knee_clock(time, start, middle, end):
    """A smooth mechanical bend over the actual observed one-shot phases."""
    return (175. - 75. * eased(time, start, middle) if time <= middle
            else 100. + 70. * eased(time, middle, end))


def root_from_knee(root_xy, hip_offset, ankle, lengths, degrees):
    upper, lower = lengths
    span = np.asarray(root_xy) + np.asarray(hip_offset)[:2] - np.asarray(ankle)[:2]
    radius2 = upper * upper + lower * lower - 2 * upper * lower * math.cos(math.radians(degrees))
    vertical2 = radius2 - float(span @ span)
    if vertical2 <= 0:
        raise ValueError('The observed lunge step is too wide for its bounded knee angle.')
    return float(ankle[2] - hip_offset[2] + math.sqrt(vertical2))


def ankle_from_knee(hip, ankle_xy, lengths, degrees):
    upper, lower = lengths
    span = np.asarray(ankle_xy) - np.asarray(hip)[:2]
    radius2 = upper * upper + lower * lower - 2 * upper * lower * math.cos(math.radians(degrees))
    vertical2 = radius2 - float(span @ span)
    if vertical2 <= 0:
        raise ValueError('The observed swing is outside the fixed leg reach at this knee angle.')
    return float(hip[2] - math.sqrt(vertical2))


def carrying_arms(shoulders, lengths, grip_ratio, elbow_ratio):
    """Hold the source's visible frontal widths with fixed, downward arms."""
    shoulders = {side: np.asarray(shoulders[side], dtype=float) for side in SIDES}
    width = float(np.linalg.norm(shoulders['right'] - shoulders['left']))
    centre = np.mean(list(shoulders.values()), axis=0)
    if width <= 0 or not np.isfinite([grip_ratio, elbow_ratio]).all() or not 1. <= elbow_ratio < grip_ratio:
        raise ValueError('The carrying pose needs readable outward elbow/wrist widths.')
    result = {}
    for side, sign in (('left', -1), ('right', 1)):
        upper, fore = lengths[side]
        elbow = shoulders[side].copy()
        elbow[0] = centre[0] + sign * elbow_ratio * width / 2
        dx = float(elbow[0] - shoulders[side][0])
        wrist = elbow.copy()
        wrist[0] = centre[0] + sign * grip_ratio * width / 2
        lower_dx = float(wrist[0] - elbow[0])
        if abs(dx) >= upper or abs(lower_dx) >= fore:
            raise ValueError('The source carrying width is outside the unchanged arm reach.')
        elbow[2] -= math.sqrt(upper * upper - dx * dx)
        wrist[2] = elbow[2] - math.sqrt(fore * fore - lower_dx * lower_dx)
        result[side] = {'elbow': elbow, 'wrist': wrist}
    return result
