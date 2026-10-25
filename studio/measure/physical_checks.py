"""Strict checks of the extra measured equipment/body support reports."""
import math
from functools import lru_cache
import importlib.util
from pathlib import Path


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


@lru_cache(maxsize=1)
def _deadlift_rules():
    # run_check.py runs outside Blender with only studio/measure on sys.path.
    # Load the same numpy-only rules that measured the generated rig, avoiding
    # a second definition of the mechanical range limits.
    path = Path(__file__).resolve().parents[2] / 'pipeline/blender/deadlift_range.py'
    spec = importlib.util.spec_from_file_location('_studio_deadlift_range', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _deadlift_measurements(report, contacts):
    """Recheck complete samples and recorded phases rather than trusting ready."""
    movement = (contacts or {}).get('deadlift_range') or {}
    if not isinstance(movement, dict):
        return None, None, None
    count = report.get('frames')
    first, last = movement.get('first_frame'), movement.get('last_frame')
    expected = (report.get('reference_motion') or {}).get('phase_frames')
    phases = movement.get('phase_frames')
    coverage = (type(count) is int and count > 0 and (contacts or {}).get('frames') == count
        and movement.get('frames') == count and type(first) is int and type(last) is int
        and last - first + 1 == count and isinstance(expected, dict) and phases == expected
        and all(type(value) is int and first <= value <= last for value in expected.values())
        and type(expected.get('start')) is int and type(expected.get('end')) is int
        and expected['start'] < expected['end']
        and finite(movement.get('height_m')) and movement['height_m'] > 0
        and not movement.get('measurement_error'))
    samples = movement.get('per_frame')
    if (not coverage or not isinstance(samples, list) or len(samples) != count
            or not all(isinstance(sample, dict) for sample in samples)
            or [sample.get('frame') for sample in samples] != list(range(first, last + 1))):
        return None, None, None
    rules = _deadlift_rules()
    try:
        checked = rules.summarize(samples, movement['height_m'], phases)
    except (ValueError, TypeError, KeyError):
        return None, None, None
    floor = movement.get('plate_floor') or {}
    if not isinstance(floor, dict):
        return checked, None, None
    floor_samples = floor.get('per_frame')
    floor_coverage = (floor.get('frames') == count and floor.get('first_frame') == first
        and floor.get('last_frame') == last and floor.get('start_frame') == phases['start']
        and all(type(floor.get(name)) is int for name in ('frames', 'first_frame', 'last_frame', 'start_frame'))
        and isinstance(floor_samples, list))
    if not floor_coverage:
        return checked, None, None
    try:
        plate_floor = rules.summarize_plate_floor(floor_samples, phases['start'], first, last)
        expected_radius = .225 * movement['height_m'] / 1.75 * 1000
        missing = max(2 - sum(abs(reading['radius_mm'] - expected_radius) <= 3.
            for reading in sample['plates'].values()) for sample in floor_samples)
    except (ValueError, TypeError, KeyError):
        return checked, None, None
    return checked, plate_floor, missing


def rows_for(report):
    physical = report.get('physical')
    if not physical:
        return []
    rows = []
    def row(metric, value, limit, complete=True, note=None, unit='mm'):
        measured = finite(value) and value >= 0 and complete
        rows.append({'group': 'body', 'metric': metric, 'target': limit, 'ours': value if finite(value) else None,
            'status': ('pass' if value <= limit else 'fail') if measured else 'info',
            'kind': None if measured else 'not-measured', 'frame': None, 'unit': unit,
            'note': note if measured else 'Required physical contact was not measured for every frame.', 'uncertain_why': None})
    contacts = physical.get('contacts')
    profile = physical.get('profile') or {}
    if profile.get('holds') and not contacts:
        row('equipment_contact', None, 3., False)
    if contacts:
        count = contacts.get('frames')
        complete = type(count) is int and count > 0 and count == report.get('frames')
        for side in ('left', 'right'):
            side_read = (contacts.get('sides') or {}).get(side) or {}
            measured = complete and side_read.get('frames') == count
            row('equipment_gap_' + side, side_read.get('nearest_gap_mm_max'), 3., measured)
            row('equipment_penetration_' + side, side_read.get('deepest_inside_bar_mm'), 3., measured)
        row('equipment_anchor_error', contacts.get('max_grip_anchor_error_mm'), 2., complete)
        if contacts.get('path_adjustments'):
            values = [r.get('adjustment_mm') for r in contacts['path_adjustments']]
            row('equipment_path_adaptation', max(values) if all(finite(v) for v in values) else None,
                20., complete and len(values) == count, 'Reach adaptation changes the photographed equipment path.')
        if contacts.get('equipment') != 'dumbbells':
            clearance = contacts.get('body_clearance') or {}
            first, last = clearance.get('first_frame'), clearance.get('last_frame')
            coverage = (complete and type(clearance.get('measured_frames')) is int and
                        clearance['measured_frames'] == count and type(first) is int and type(last) is int and
                        last - first + 1 == count and clearance.get('status') in ('clear_at_samples', 'overlap_candidates'))
            row('equipment_body_overlap', clearance.get('max_penetration_mm'), 3., coverage,
                'Conservative shaft/body surface samples; overlap candidates need visual review.')
        heads = contacts.get('head_body_clearance')
        if contacts.get('equipment') == 'dumbbells':
            row('weight_head_body_overlap', (heads or {}).get('max_penetration_mm'), 3.,
                complete and (heads or {}).get('frames') == count,
                'Finite convex weight-head/body surface samples; inspect the rendered silhouette too.')
        if profile.get('overhead'):
            row('overhead_shortfall', contacts.get('overhead_shortfall_mm'), 0., complete,
                'Both handles must reach above the head; a secure grip alone does not establish the required movement range.')
    if profile.get('family') == 'hanging':
        count = (contacts or {}).get('frames')
        complete = type(count) is int and count > 0 and count == report.get('frames')
        movement = (contacts or {}).get('pullup_range') or {}
        measured = complete and movement.get('frames') == count and len(movement.get('per_frame') or []) == count
        row('pullup_chin_shortfall', movement.get('chin_shortfall_mm'), 0., measured,
            'The lower-face geometric proxy must clear the top of the fixed bar at the top of the rep. Inspect the rendered chin too.')
        row('pullup_extension_shortfall', movement.get('extension_shortfall_deg'), 5., measured,
            'Both elbows must reach at least 175 degrees together at the bottom; a secure grip alone is incomplete movement.', 'deg')
        row('pullup_bar_drift', (contacts or {}).get('fixed_bar_drift_mm'), 1., complete,
            'The body moves below a fixed pull-up bar.')
        row('pullup_bar_rotation', (contacts or {}).get('fixed_bar_rotation_deg'), .1, complete, unit='deg')
        adaptation = (contacts or {}).get('hang_extension') or {}
        row('pullup_hang_body_adaptation', adaptation.get('max_body_drop_mm'), 20., complete,
            'Declared whole-body lowering completes the dead hang without stretching the arms or moving the bar.')
        pole = movement.get('elbow_pole') or {}
        pole_complete = measured and pole.get('frames') == count and pole.get('measured_frames') == count
        row('pullup_elbow_above_bar', pole.get('max_elbow_above_bar_mm'), 0., pole_complete,
            'Both elbows must remain below the fixed bar throughout the pull-up.')
        row('pullup_elbow_above_wrist', pole.get('max_elbow_above_wrist_mm'), 0., pole_complete,
            'The forearms must point upward to the fixed grip; an inverted elbow branch is rejected.')
        row('pullup_peak_elbow_above_shoulder', pole.get('peak_elbow_above_shoulder_mm'), 0., pole_complete,
            'At the highest body position the elbows must lie below their shoulders, as in the source pull-up.')
    if profile.get('family') == 'hinge':
        movement, floor, missing_plates = _deadlift_measurements(report, contacts)
        movement = movement or {}
        limits = movement.get('limits') or {}
        complete = bool(movement)
        def deadlift_row(metric, value, limit, unit='deg', measured=complete, note=None):
            row('deadlift_' + metric, value, limit, measured, note, unit)
            if not measured:
                rows[-1]['note'] = ('Both loaded plate meshes and their actual floor distances must be measured in every frame, '
                    'including the recorded source start.' if metric.startswith('plate_') or metric == 'missing_loaded_plates' else
                    'Deadlift range requires every generated frame and the same recorded source start/end phases.')
        deadlift_row('elbow_extension_shortfall', movement.get('elbow_extension_shortfall_deg'), 0.,
            note='Both lifting elbows must remain at least 175 degrees throughout the rep.')
        start_hips, start_knees = movement.get('start_hips_deg'), movement.get('start_knees_deg')
        deadlift_row('start_hip_excess', max(0., max(start_hips.values()) - limits['maximum_start_hip_deg']) if start_hips else None, 0.)
        deadlift_row('start_knee_excess', max(0., max(start_knees.values()) - limits['maximum_start_knee_deg']) if start_knees else None, 0.)
        deadlift_row('hip_lockout_shortfall', movement.get('hip_lockout_shortfall_deg'), 0.)
        deadlift_row('knee_lockout_shortfall', movement.get('knee_lockout_shortfall_deg'), 0.)
        deadlift_row('lockout_torso_lean', movement.get('lockout_torso_lean_deg'), 10.,
            note='Hip and knee lockout and upright torso must occur at the recorded end phase.')
        for metric, value, required in (
                ('start_bar_height_shortfall', 'start_bar_below_knees_mm', 'minimum_start_bar_below_knees_mm'),
                ('grip_outside_knees_shortfall', 'start_grip_outside_knees_mm', 'minimum_start_grip_outside_knees_mm'),
                ('bar_travel_shortfall', 'forward_bar_travel_mm', 'minimum_forward_bar_travel_mm')):
            deadlift_row(metric, max(0., limits[required] - movement[value]) if complete else None, 0., 'mm')
        deadlift_row('missing_loaded_plates', missing_plates, 0., 'plates', floor is not None,
            'Both full-size weight-disc meshes must be present; a bare bar or decorative rim does not establish a floor pull.')
        plate_complete = floor is not None and floor['missing_loaded_plates'] == 0
        deadlift_row('plate_start_floor_gap', (floor or {}).get('start_floor_gap_mm'), 5., 'mm', plate_complete,
            'Both actual lowest plate vertices must reach the floor at the recorded start phase.')
        deadlift_row('plate_floor_penetration', (floor or {}).get('max_floor_penetration_mm'), 3., 'mm', plate_complete,
            'Actual evaluated plate vertices must stay above the floor throughout every frame.')
    supports = physical.get('supports') or {}
    required = {'seated': ('pelvis',), 'supine': ('pelvis', 'back', 'head'),
        'hold': ('left_forearm', 'right_forearm', 'left_toe', 'right_toe'),
        'sled': ('pelvis', 'back', 'head', 'footplate_left', 'footplate_right',
                 'footplate_left_heel', 'footplate_left_toe', 'footplate_right_heel', 'footplate_right_toe')}.get(profile.get('family'), ())
    for name in required:
        if name not in (supports.get('surfaces') or {}):
            row('support_gap_' + name, None, 5., False)
    for name, reading in (supports.get('surfaces') or {}).items():
        count = reading.get('frames')
        complete = type(count) is int and count > 0 and count == report.get('frames') and reading.get('measured_frames') == count
        row('support_gap_' + name, reading.get('max_gap_mm'), 5., complete)
        row('support_penetration_' + name, reading.get('max_penetration_mm'), 3., complete)
    rail_path = supports.get('rail_path') or {}
    if rail_path or profile.get('family') == 'sled':
        row('sled_rail_deviation', rail_path.get('max_deviation_mm'), 10.,
            rail_path.get('frames') == report.get('frames') and type(rail_path.get('frames')) is int and rail_path['frames'] > 0)
    correction = supports.get('correction') or {}
    if correction.get('applied') is False:
        row('support_geometry', None, 0., False)
        rows[-1]['note'] = correction.get('reason') or 'The body could not satisfy the required physical support geometry.'
    if supports.get('limitation'):
        row('support_motion', None, 0., False, supports['limitation'])
        rows[-1]['note'] = supports['limitation']
    return rows
