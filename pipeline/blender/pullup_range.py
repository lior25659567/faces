"""Pull-up range measurements on the rendered character, independent of Blender.

The lower-face region is a geometric chin proxy on this fitted mascot. It is
not a calibrated anatomical landmark or a measurement of the source athlete.
"""
import numpy as np


def continuous_pole(shoulders, bar, along, width, fraction, lengths, height):
    """One down-and-forward elbow branch for both strokes of a strict pull-up.

    The source pictures calibrate width, not sagittal depth. This declared depth
    policy keeps elbows in front of the torso while they descend continuously;
    it cannot turn an inferred source depth into an observed measurement.
    """
    fraction = float(fraction)
    if not np.isfinite(fraction) or not 0. <= fraction <= 1.:
        raise ValueError('The pull-up pole needs a finite movement fraction.')
    bar, along = np.asarray(bar), np.asarray(along)
    shoulders = {side: np.asarray(value) for side, value in shoulders.items()}
    origin = (shoulders['left'] + shoulders['right']) / 2
    span = float(np.linalg.norm(shoulders['right'] - shoulders['left']))
    if not np.isfinite([span, width, height]).all() or min(span, width, height) <= 0:
        raise ValueError('The pull-up pole needs positive finite widths and height.')
    hints, preferences = {}, {}
    for side, sign in (('left', -1), ('right', 1)):
        upper = float(lengths[side][0])
        hint = origin + sign * width * span / 2 * along
        hint[1] = shoulders[side][1] + .08 * height
        hint[2] = shoulders[side][2] + upper * (1. - 1.65 * fraction)
        hints[side] = hint
        preferences[side] = {
            'elbow_lateral': float(origin @ along) + sign * width * span / 2,
            'lateral_axis': along, 'lateral_scale': span * .015,
            'max_elbow_z': float(shoulders[side][2] + (bar[2] - shoulders[side][2]) * (1. - fraction) - .005 * height),
            'min_elbow_y': float(shoulders[side][1] + height * (.035 + .10 * np.sqrt(fraction) * (1. - fraction))),
            'elbow_below_wrist': True, 'elbow_clearance_m': .005 * height,
            'forearm_vertical_fraction': .9,
        }
    return hints, preferences


def motion_summary(samples, fps, phase_frames):
    """Measure branch jumps and a different lowering path from actual joints.

    Offsets are relative to the moving shoulders, so body travel cannot conceal
    a snapping elbow. Return-path error compares the same body heights, without
    assuming equal ascent and descent speed.
    """
    if (isinstance(fps, bool) or not isinstance(fps, (int, float)) or not np.isfinite(fps) or fps <= 0
            or not isinstance(samples, list) or len(samples) < 3 or not isinstance(phase_frames, dict)):
        raise ValueError('Pull-up motion requires complete joint samples, fps and recorded phases.')
    frames = [sample.get('frame') for sample in samples]
    first, last = frames[0], frames[-1]
    if (type(first) is not int or type(last) is not int or frames != list(range(first, last + 1))
            or phase_frames.get('start') != first or type(phase_frames.get('end')) is not int
            or not first < phase_frames['end'] < last):
        raise ValueError('Pull-up motion requires every frame and its recorded ascent/return boundary.')
    hips = np.array([sample.get('hip_height_m') for sample in samples], dtype=float)
    if not np.isfinite(hips).all() or np.ptp(hips) <= 1e-8:
        raise ValueError('Pull-up motion requires a measured body rise.')
    top = phase_frames['end'] - first
    if hips[top] < hips.max() - 1e-5:
        raise ValueError('The recorded pull-up top must be the measured highest body position.')
    speed, retrace = 0., 0.
    for side in ('left', 'right'):
        try:
            offsets = np.array([sample['elbow_offsets_m'][side] for sample in samples], dtype=float)
        except (KeyError, TypeError):
            raise ValueError('Every pull-up frame needs both shoulder-relative elbow positions.') from None
        if offsets.shape != (len(samples), 3) or not np.isfinite(offsets).all():
            raise ValueError('Every pull-up frame needs finite three-dimensional elbow offsets.')
        lengths = np.linalg.norm(offsets, axis=1)
        upper = float(np.median(lengths))
        if upper <= 1e-8 or np.max(abs(lengths - upper)) > upper * .01:
            raise ValueError('The measured pull-up upper arm must keep its fixed length.')
        speed = max(speed, float(np.linalg.norm(np.diff(np.vstack([offsets, offsets[0]]), axis=0), axis=1).max() * fps / upper))
        # Interpolate each stroke independently at the same body heights;
        # repeated top-hold heights share one interpolation anchor.
        up_x, up_i = np.unique(hips[:top + 1], return_index=True)
        down_x, down_i = np.unique(hips[top:], return_index=True)
        low, high = max(up_x[0], down_x[0]), min(up_x[-1], down_x[-1])
        if len(up_x) < 2 or len(down_x) < 2 or high - low <= 1e-8:
            raise ValueError('The pull-up must contain both a measurable pull and return.')
        common = np.unique(np.r_[up_x[(up_x >= low) & (up_x <= high)], down_x[(down_x >= low) & (down_x <= high)]])
        up = np.column_stack([np.interp(common, up_x, offsets[:top + 1][up_i, axis]) for axis in range(3)])
        down = np.column_stack([np.interp(common, down_x, offsets[top:][down_i, axis]) for axis in range(3)])
        retrace = max(retrace, float(np.linalg.norm(up - down, axis=1).max() / upper))
    return {'frames': len(samples), 'measured_frames': len(samples), 'fps': float(fps),
            'phase_frames': dict(phase_frames), 'max_elbow_speed_arm_lengths_per_s': speed,
            'max_return_path_error_arm_lengths': retrace,
            'limits': {'max_elbow_speed_arm_lengths_per_s': 6., 'max_return_path_error_arm_lengths': .12},
            'method': 'Measured shoulder-relative elbow path, including loop seam; compare pull and lowering at the same body height.'}


def grip_candidates(requested, tolerance=0., step=.005):
    """Nearest-first grip ratios inside an explicit source tolerance."""
    for value in (requested, tolerance, step):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError('The source grip, tolerance and search step must be finite numbers.')
    if requested <= 0 or tolerance < 0 or step <= 0:
        raise ValueError('The source grip and step must be positive; tolerance cannot be negative.')
    radii = list(np.arange(step, tolerance + 1e-10, step))
    if tolerance > 0 and (not radii or tolerance - radii[-1] > 1e-10):
        radii.append(tolerance)
    return [float(requested)] + [float(requested + sign * distance)
        for distance in radii for sign in (1., -1.) if requested + sign * distance > 0]


def elbow_width_timeline(source_widths, phase_frames, fractions, first=1):
    """Interpolate photographed elbow widths at their recorded source positions.

    ``fractions`` follows the source body's upward travel, including the hold
    and return. It avoids moving the elbows during an unchanged top pose.
    Estimated phases without an observed width are not invented measurements.
    """
    return reference_timeline(source_widths, phase_frames, fractions, 'elbow_width_ratio', first)


def reference_timeline(source_widths, phase_frames, fractions, metric, first=1):
    """Source values at observed movement positions, without certifying depth."""
    if source_widths is None:
        return None
    fractions = np.asarray(fractions, dtype=float)
    if fractions.ndim != 1 or not len(fractions) or not np.isfinite(fractions).all():
        raise ValueError('Reference pose values require a finite movement timeline.')
    if not isinstance(source_widths, dict) or not isinstance(phase_frames, dict):
        raise ValueError('Reference pose values require their recorded phase frames.')
    anchors = []
    for phase, record in source_widths.items():
        if isinstance(record, dict) and record.get('estimated'):
            continue
        record = record.get(metric, record) if isinstance(record, dict) else None
        value = record.get('value') if isinstance(record, dict) else None
        if value is None or ('image' in record and record['image'] is None):
            continue
        frame = phase_frames.get(phase)
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0
                or (metric == 'elbow_width_ratio' and value == 0)
                or type(frame) is not int or not first <= frame < first + len(fractions)):
            raise ValueError('Every observed reference goal needs a finite value and recorded frame in this motion.')
        anchors.append((float(fractions[frame - first]), float(value)))
    anchors.sort()
    if len(anchors) < 2 or any(a[0] >= b[0] for a, b in zip(anchors, anchors[1:])):
        raise ValueError('Reference goals need two distinct photographed movement positions.')
    return np.interp(fractions, [a[0] for a in anchors], [a[1] for a in anchors])


def chin_region(points, head_mask):
    """Return lower central/front head vertices, selected once in the rest pose.

    Fitted characters face +Y. The front 40% and central 20% of the rigidly
    head-owned surface exclude the neck, ears and hair. The lowest 2 mm band
    (at 1 m character height) follows the underside of the lower face.
    """
    points = np.asarray(points, dtype=float)
    head_mask = np.asarray(head_mask, dtype=bool)
    if (points.ndim != 2 or points.shape[1] != 3 or head_mask.shape != (len(points),)
            or not np.isfinite(points).all() or not head_mask.any()):
        raise ValueError('The chin proxy needs a finite rest mesh and head surface.')
    indices = np.where(head_mask)[0]
    head = points[indices]
    low, high = head.min(axis=0), head.max(axis=0)
    extent = high - low
    if min(extent) <= 1e-8:
        raise ValueError('Cannot locate the lower face on a degenerate head.')
    centre = (low[0] + high[0]) / 2
    region = indices[(abs(head[:, 0] - centre) <= .1 * extent[0])
                     & (head[:, 1] >= high[1] - .4 * extent[1])
                     & (head[:, 2] < (low[2] + high[2]) / 2)]
    if not len(region):
        raise ValueError('Cannot locate a lower central/front head surface.')
    region = region[points[region, 2] <= points[region, 2].min() + .008 * extent[2]]
    return region


def included(a, joint, b):
    a, b = np.asarray(a) - joint, np.asarray(b) - joint
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if not np.isfinite(denominator) or denominator <= 1e-12:
        raise ValueError('An elbow range measurement needs two nonzero segments.')
    return float(np.degrees(np.arccos(np.clip(a @ b / denominator, -1., 1.))))


def summarize(samples):
    """Require simultaneous extension at the bottom and chin clearance at top."""
    if not samples:
        raise ValueError('A pull-up range requires the complete motion samples.')
    for sample in samples:
        values = [sample.get('chin_clearance_mm'), *(sample.get('elbows_deg') or {}).values()]
        if (len(values) != 3 or set(sample.get('elbows_deg') or {}) != {'left', 'right'}
                or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v) for v in values)
                or any(not 0. <= a <= 180. for a in sample['elbows_deg'].values())):
            raise ValueError('Every pull-up frame needs finite chin and both elbow measurements.')
    top = max(samples, key=lambda sample: sample['chin_clearance_mm'])
    bottom = max(samples, key=lambda sample: min(sample['elbows_deg'].values()))
    result = {'frames': len(samples), 'top_frame': top['frame'], 'bottom_frame': bottom['frame'],
            'chin_clearance_mm': top['chin_clearance_mm'],
            'chin_shortfall_mm': max(0., -top['chin_clearance_mm']),
            'bottom_elbows_deg': bottom['elbows_deg'],
            'extension_shortfall_deg': max(0., 180. - min(bottom['elbows_deg'].values())),
            'minimum_required_elbow_included_deg': 175., 'per_frame': samples}
    if all('elbow_bar_clearance_mm' in sample for sample in samples):
        result['elbow_pole'] = pole_summary(samples)
    return result


def pole_summary(samples, peak_frame=None):
    """Reject the inverted elbow branch even when chin and grip checks pass."""
    if not samples:
        raise ValueError('The elbow pole requires every frame of the movement.')
    fields = ('elbow_bar_clearance_mm', 'elbow_wrist_clearance_mm', 'elbow_shoulder_clearance_mm')
    for sample in samples:
        for field in fields:
            values = sample.get(field)
            if (not isinstance(values, dict) or set(values) != {'left', 'right'}
                    or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value)
                           for value in values.values())):
                raise ValueError('Every elbow pole sample needs finite bar, wrist and shoulder clearances for both arms.')
        height = sample.get('hip_height_m')
        if isinstance(height, bool) or not isinstance(height, (int, float)) or not np.isfinite(height):
            raise ValueError('The pull-up peak requires the measured hip height.')
    if peak_frame is None:
        peak = max(samples, key=lambda sample: sample['hip_height_m'])
    else:
        selected = [sample for sample in samples if sample['frame'] == peak_frame]
        if len(selected) != 1:
            raise ValueError('The recorded peak must belong to this movement timeline.')
        peak = selected[0]
    bar = min(value for sample in samples for value in sample['elbow_bar_clearance_mm'].values())
    wrist = min(value for sample in samples for value in sample['elbow_wrist_clearance_mm'].values())
    shoulder = min(peak['elbow_shoulder_clearance_mm'].values())
    return {'frames': len(samples), 'measured_frames': len(samples), 'peak_frame': peak['frame'],
            'minimum_bar_clearance_mm': bar, 'minimum_wrist_clearance_mm': wrist,
            'peak_minimum_shoulder_clearance_mm': shoulder,
            'max_elbow_above_bar_mm': max(0., -bar), 'max_elbow_above_wrist_mm': max(0., -wrist),
            'peak_elbow_above_shoulder_mm': max(0., -shoulder)}
