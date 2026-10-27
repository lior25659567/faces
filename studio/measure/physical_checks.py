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


@lru_cache(maxsize=1)
def _bench_rules():
    path = Path(__file__).resolve().parents[2] / 'pipeline/blender/bench_press.py'
    spec = importlib.util.spec_from_file_location('_studio_bench_press', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bench_measurements(report, contacts):
    """Judge the keyed bar and limbs, including the intervening frames."""
    movement = (contacts or {}).get('bench_range') or {}
    if not isinstance(movement, dict):
        return None
    count = report.get('frames')
    first, last = movement.get('first_frame'), movement.get('last_frame')
    phases = (report.get('reference_motion') or {}).get('phase_frames')
    samples = movement.get('per_frame')
    if not (type(count) is int and count > 1 and (contacts or {}).get('frames') == count
            and movement.get('frames') == count and type(first) is int and type(last) is int
            and last - first + 1 == count and isinstance(phases, dict) and phases == movement.get('phase_frames')
            and all(type(value) is int and first <= value <= last for value in phases.values())
            and phases.get('start') == first and type(phases.get('end')) is int and first < phases['end']
            and finite(movement.get('height_m')) and movement['height_m'] > 0
            and not movement.get('measurement_error') and isinstance(samples, list) and len(samples) == count
            and all(isinstance(sample, dict) for sample in samples)
            and [sample.get('frame') for sample in samples] == list(range(first, last + 1))):
        return None
    try:
        return _bench_rules().summarize(samples, movement['height_m'], phases)
    except (ValueError, TypeError, KeyError):
        return None


@lru_cache(maxsize=1)
def _pullup_rules():
    path = Path(__file__).resolve().parents[2] / 'pipeline/blender/pullup_range.py'
    spec = importlib.util.spec_from_file_location('_studio_pullup_range', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pullup_motion(report, contacts):
    movement = (contacts or {}).get('pullup_range') or {}
    samples = movement.get('per_frame') if isinstance(movement, dict) else None
    count = report.get('frames')
    if not (type(count) is int and count > 2 and (contacts or {}).get('frames') == count
            and movement.get('frames') == count and isinstance(samples, list) and len(samples) == count
            and all(isinstance(sample, dict) for sample in samples)):
        return None
    try:
        return _pullup_rules().motion_summary(samples, report.get('fps'),
            (report.get('reference_motion') or {}).get('phase_frames'))
    except (ValueError, TypeError, KeyError):
        return None


@lru_cache(maxsize=1)
def _pulldown_rules():
    path = Path(__file__).resolve().parents[2] / 'pipeline/blender/pulldown_range.py'
    spec = importlib.util.spec_from_file_location('_studio_pulldown_range', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pulldown_measurements(report, contacts):
    movement = (contacts or {}).get('pulldown_range') or {}
    if not isinstance(movement, dict):
        return None
    count = report.get('frames')
    first, last = movement.get('first_frame'), movement.get('last_frame')
    phases = (report.get('reference_motion') or {}).get('phase_frames')
    samples = movement.get('per_frame')
    if not (type(count) is int and count > 1 and (contacts or {}).get('frames') == count
            and movement.get('frames') == count and movement.get('measured_frames') == count
            and type(first) is int and type(last) is int and last - first + 1 == count
            and isinstance(phases, dict) and phases == movement.get('phase_frames')
            and all(type(value) is int and first <= value <= last for value in phases.values())
            and phases.get('start') == first and type(phases.get('end')) is int and first < phases['end']
            and finite(movement.get('height_m')) and movement['height_m'] > 0
            and not movement.get('measurement_error') and isinstance(samples, list) and len(samples) == count
            and all(isinstance(sample, dict) for sample in samples)
            and [sample.get('frame') for sample in samples] == list(range(first, last + 1))):
        return None
    try:
        return _pulldown_rules().summarize(samples, movement['height_m'], phases)
    except (ValueError, TypeError, KeyError):
        return None


@lru_cache(maxsize=4)
def _technique_rules(module_name):
    path = Path(__file__).resolve().parents[2] / 'pipeline/blender' / (module_name + '.py')
    spec = importlib.util.spec_from_file_location('_studio_' + module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _technique_measurements(report, contacts, key, module_name, *rule_args):
    movement = (contacts or {}).get(key) or {}
    if not isinstance(movement, dict):
        return None
    count = report.get('frames')
    first, last = movement.get('first_frame'), movement.get('last_frame')
    phases = (report.get('reference_motion') or {}).get('phase_frames')
    samples = movement.get('per_frame')
    if not (type(count) is int and count > 1 and (contacts or {}).get('frames') == count
            and movement.get('frames') == count and movement.get('measured_frames') == count
            and type(first) is int and type(last) is int and last - first + 1 == count
            and isinstance(phases, dict) and phases == movement.get('phase_frames')
            and all(type(value) is int and first <= value <= last for value in phases.values())
            and phases.get('start') == first and type(phases.get('end')) is int and first < phases['end']
            and finite(movement.get('height_m')) and movement['height_m'] > 0
            and not movement.get('measurement_error') and isinstance(samples, list) and len(samples) == count
            and all(isinstance(sample, dict) for sample in samples)
            and [sample.get('frame') for sample in samples] == list(range(first, last + 1))):
        return None
    try:
        return _technique_rules(module_name).summarize(samples, movement['height_m'], phases, *rule_args)
    except (ValueError, TypeError, KeyError):
        return None


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


def rows_for(report, exercise=None):
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
    exercise = exercise or profile.get('exercise')
    if exercise == 'Walking Lunge (dumbbells optional)' or physical.get('lunge') is not None:
        raw = physical.get('lunge') or {}
        container = {'frames': report.get('frames'), 'lunge_range': raw}
        movement = _technique_measurements(report, container, 'lunge_range', 'lunge_range',
            raw.get('source_times'), raw.get('fps')) or {}
        for metric, limit, unit, note in (
                ('max_contact_gap_mm', 5., 'mm', 'Every observed active heel or toe support point reaches the floor.'),
                ('max_contact_penetration_mm', 3., 'mm', 'The observed heel and toe support patches do not sink into the floor.'),
                ('max_anchor_drift_mm', 3., 'mm', 'Each planted support point remains on its fitted contact anchor.'),
                ('max_swing_clearance_shortfall_mm', 0., 'mm', 'The airborne foot clears the floor between observed supports.'),
                ('max_leg_length_error_mm', .1, 'mm', 'Both legs keep their fixed fitted lengths during the step.'),
                ('max_backward_knee_mm', 3., 'mm', 'Both knees bend forward in the observed direction of travel.'),
                ('max_body_floor_penetration_mm', 3., 'mm', 'The complete evaluated body remains above the floor.'),
                ('missing_forward_travel', 0., 'steps', 'The observed walking step advances rather than becoming an in-place repetition.'),
                ('max_reverse_root_mm', 3., 'mm', 'The one-way step does not invent a backwards return.'),
                ('start_extension_shortfall_deg', 0., 'deg', 'The observed starting position has extended knees.'),
                ('middle_knee_overfold_deg', 0., 'deg', 'The leading knee does not fold beyond the fitted lunge range.'),
                ('middle_knee_depth_shortfall_deg', 0., 'deg', 'The leading knee reaches the observed lowered step position.')):
            row('lunge_' + metric.removesuffix('_deg').removesuffix('_mm'), movement.get(metric), limit, bool(movement), note, unit)
            if not movement:
                rows[-1]['note'] = 'Walking-lunge checks require the full chronological one-way sequence and measured heel/toe contacts; depth inferred from the video remains uncertain.'
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
        row('pullup_hang_raise_adaptation', adaptation.get('max_body_raise_mm'), 20., complete,
            'Any extra body rise to clear the fixed bar must be declared and bounded; the grip and limb lengths stay fixed.')
        pole = movement.get('elbow_pole') or {}
        pole_complete = measured and pole.get('frames') == count and pole.get('measured_frames') == count
        row('pullup_elbow_above_bar', pole.get('max_elbow_above_bar_mm'), 0., pole_complete,
            'Both elbows must remain below the fixed bar throughout the pull-up.')
        row('pullup_elbow_above_wrist', pole.get('max_elbow_above_wrist_mm'), 0., pole_complete,
            'The forearms must point upward to the fixed grip; an inverted elbow branch is rejected.')
        row('pullup_peak_elbow_above_shoulder', pole.get('peak_elbow_above_shoulder_mm'), 0., pole_complete,
            'At the highest body position the elbows must lie below their shoulders, as in the source pull-up.')
        continuity = _pullup_motion(report, contacts) or {}
        row('pullup_elbow_path_speed', continuity.get('max_elbow_speed_arm_lengths_per_s'), 6., bool(continuity),
            'Both shoulder-relative elbow paths must remain continuous throughout the pull, lowering and loop seam.', 'arm lengths/s')
        row('pullup_return_path_error', continuity.get('max_return_path_error_arm_lengths'), .12, bool(continuity),
            'The elbows must follow the same controlled path on the pull and lowering at the same body height.', 'arm lengths')
    is_pulldown = (exercise == 'Lat Pulldown (seated, to the front)'
        or 'pulldown_range' in (contacts or {})
        or profile.get('overhead') and profile.get('family') == 'seated' and (contacts or {}).get('equipment') == 'bar')
    if is_pulldown:
        movement = _pulldown_measurements(report, contacts) or {}
        for metric, value, unit, note in (
                ('elbow_above_wrist', 'max_elbow_above_wrist_mm', 'mm', 'The forearms point upward to the handle throughout the pulldown.'),
                ('bottom_elbow_above_shoulder', 'bottom_elbow_above_shoulder_mm', 'mm', 'At the recorded chest position both elbows must be below their shoulders.'),
                ('bottom_upperarm_excess', 'bottom_upperarm_excess_deg', 'deg', 'The elbows pull down beside the torso rather than remaining above the body.'),
                ('top_extension_shortfall', 'top_extension_shortfall_deg', 'deg', 'Both elbows extend to at least 175 degrees at the top of the cable return.'),
                ('travel_shortfall', 'travel_shortfall_mm', 'mm', 'A full pulldown requires meaningful bar travel from overhead toward the upper chest.')):
            row('pulldown_' + metric, movement.get(value), 0., bool(movement), note, unit)
            if not movement:
                rows[-1]['note'] = 'Pulldown technique requires measured bar, torso and both arm joints for every frame at the recorded source phases.'
    is_press = (exercise in ('Standing Barbell Overhead Press (Shoulder Press)',
        'Dumbbell Shoulder Press (seated overhead press)') or 'overhead_range' in (contacts or {}))
    if is_press:
        movement = _technique_measurements(report, contacts, 'overhead_range', 'overhead_range') or {}
        for metric, limit, unit, note in (
                ('lockout_shortfall_deg', 0., 'deg', 'Both elbows must reach at least 175 degrees together at the recorded overhead lockout.'),
                ('bottom_flexion_excess_deg', 0., 'deg', 'Both arms return to a bent lower press position.'),
                ('bottom_overfold_deg', 0., 'deg', 'The lower position must not overfold the elbows below 65 degrees.'),
                ('elbow_above_wrist_mm', 0., 'mm', 'The forearms point up to the grips throughout the press.'),
                ('travel_shortfall_mm', 0., 'mm', 'Both handles must complete a meaningful overhead stroke.'),
                ('peak_stack_excess_mm', 0., 'mm', 'The wrists stack over the shoulders at lockout.'),
                ('lockout_head_shortfall_mm', 0., 'mm', 'Both handles clear the actual top of the head at lockout.'),
                ('torso_lean_deg', 15. if exercise == 'Dumbbell Shoulder Press (seated overhead press)' else 10., 'deg', 'The anatomical shoulder-to-hip torso stays upright rather than reclining through the press.'),
                ('elbow_asymmetry_deg', 5., 'deg', 'Both arms press together.'),
                ('hip_drift_mm', 3., 'mm', 'The supported or planted body stays stable during the press.')):
            row('press_' + metric.removesuffix('_deg').removesuffix('_mm'), movement.get(metric), limit, bool(movement), note, unit)
            if not movement:
                rows[-1]['note'] = 'Overhead press range requires both handles and measured bilateral joints for every recorded movement frame.'
    if profile.get('family') == 'squat' or 'squat_range' in (contacts or {}):
        raw = (contacts or {}).get('squat_range') or {}
        movement = _technique_measurements(report, contacts, 'squat_range', 'squat_range',
            raw.get('torso_turn'), raw.get('leg_turns')) or {}
        for metric, limit, unit, note in (
                ('top_knee_extension_shortfall_deg', 0., 'deg', 'Both knees extend at the beginning and end of the squat.'),
                ('top_hip_extension_shortfall_deg', 0., 'deg', 'Both hips extend at the beginning and end of the squat.'),
                ('bottom_knee_flexion_excess_deg', 0., 'deg', 'Both knees bend sufficiently at the recorded bottom.'),
                ('bottom_hip_flexion_excess_deg', 0., 'deg', 'Both hips reach the recorded squat bottom.'),
                ('depth_shortfall_mm', 0., 'mm', 'The pelvis reaches knee height at the bottom.'),
                ('pelvis_travel_shortfall_mm', 0., 'mm', 'The squat includes meaningful vertical pelvis travel.'),
                ('knee_asymmetry_deg', 10., 'deg', 'The legs bend together throughout the squat.'),
                ('bar_level_excess_deg', 0., 'deg', 'The back bar stays level throughout the squat.'),
                ('spine_deviation_deg', 15., 'deg', 'The squat preserves its neutral spine shape.'),
                ('loop_position_error_mm', 5., 'mm', 'The full squat returns smoothly to its starting position.')):
            row('squat_' + metric.removesuffix('_deg').removesuffix('_mm'), movement.get(metric), limit, bool(movement), note, unit)
            if not movement:
                rows[-1]['note'] = 'The squat preset requires all 69 pinned frames and measured bilateral joints, back bar and fitted rest axes.'
    isolation = exercise or ((contacts or {}).get('isolation_range') or {}).get('exercise')
    if isolation in ('Dumbbell Bicep Curl (standing, both arms)', 'Dumbbell Lateral Raise',
                     'Tricep Pushdown (cable, bar or rope attachment)'):
        movement = _technique_measurements(report, contacts, 'isolation_range', 'isolation_range', isolation) or {}
        specs = [
            ('body_lean_excess_deg', 0., 'deg', 'Standing isolation exercises keep an upright torso.'),
            ('knee_extension_shortfall_deg', 0., 'deg', 'Both knees remain upright rather than holding a crouch.'),
            ('hip_extension_shortfall_deg', 0., 'deg', 'Both hips remain extended in the standing posture.'),
            ('body_drift_mm', 2., 'mm', 'The body stays stable while the arms move.'),
            ('elbow_asymmetry_deg', 5., 'deg', 'Both arms move together.')]
        if isolation == 'Dumbbell Lateral Raise':
            specs.extend([
                ('start_elevation_excess_deg', 0., 'deg', 'The raise starts with the upper arms beside the body.'),
                ('end_elevation_shortfall_deg', 0., 'deg', 'Both arms reach shoulder height.'),
                ('end_elevation_excess_deg', 0., 'deg', 'The raise stops at shoulder height rather than going overhead.'),
                ('elbow_bend_excess_deg', 0., 'deg', 'The elbows keep a slight bend throughout the raise.'),
                ('elbow_bend_variation_deg', 3., 'deg', 'The elbows keep their bend while the shoulders raise the arms.')])
        else:
            specs.extend([
                ('upperarm_drift_deg', 5. if isolation.startswith('Dumbbell') else 12., 'deg', 'The upper arms stay beside the torso while the elbows bend.'),
                ('upperarm_position_excess_deg', 0., 'deg', 'The upper arms remain close to the body.')])
            if isolation.startswith('Dumbbell'):
                specs.extend([
                    ('start_extension_shortfall_deg', 0., 'deg', 'The curl starts with both elbows extended.'),
                    ('end_flexion_excess_deg', 0., 'deg', 'Both elbows complete the curl.'),
                    ('end_overfold_deg', 0., 'deg', 'The elbows do not fold beyond the declared curl range.')])
            else:
                specs.extend([
                    ('start_bend_excess_deg', 0., 'deg', 'The pushdown starts with bent elbows.'),
                    ('start_overfold_deg', 0., 'deg', 'The pushdown starts without overfolding the elbows.'),
                    ('end_extension_shortfall_deg', 0., 'deg', 'Both elbows reach full extension at the end of the pushdown.')])
        for metric, limit, unit, note in specs:
            row('isolation_' + metric.removesuffix('_deg').removesuffix('_mm'), movement.get(metric), limit, bool(movement), note, unit)
            if not movement:
                rows[-1]['note'] = 'Isolation technique requires measured bilateral body and arm joints for every recorded frame.'
    if profile.get('family') == 'supine':
        movement = _bench_measurements(report, contacts) or {}
        for metric, limit, unit, note in (
                ('lockout_shortfall_deg', 0., 'deg', 'Both elbows must reach at least 175 degrees at the recorded lockout.'),
                ('bottom_flexion_excess_deg', 0., 'deg', 'Both elbows must bend to at most 105 degrees at the lower-chest position.'),
                ('bottom_overfold_deg', 0., 'deg', 'The lower-chest position must not overfold either elbow below 65 degrees.'),
                ('travel_shortfall_mm', 0., 'mm', 'The bar must complete a meaningful vertical press, not remain beside the face.'),
                ('forearm_tilt_excess_deg', 0., 'deg', 'The wrists must stay above their elbows with forearms within 30 degrees of vertical.'),
                ('bar_torso_path_excess_mm', 0., 'mm', 'The shaft must remain in the shoulder-to-chest corridor rather than moving over the neck or abdomen.'),
                ('bottom_chest_shortfall_mm', 0., 'mm', 'The bottom must reach the lower chest, not remain over the shoulders or neck.'),
                ('lockout_shoulder_offset_mm', 0., 'mm', 'The completed press must return the bar over the shoulder line.'),
                ('stroke_reversal_mm', 3., 'mm', 'Each stroke must follow one press path without wandering or reversing direction.'),
                ('stationary_body_drift_mm', 2., 'mm', 'The supported torso and pelvis stay stationary during the press.'),
                ('elbow_asymmetry_deg', 5., 'deg', 'Both arms must press together.')):
            row('bench_' + metric.removesuffix('_deg').removesuffix('_mm'), movement.get(metric), limit, bool(movement), note, unit)
            if not movement:
                rows[-1]['note'] = 'Bench movement requires complete bar and bilateral arm measurements at the recorded source phases and every intervening frame.'
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
    if profile.get('family') == 'sled':
        movement = _technique_measurements(report, supports, 'sled_motion', 'sled_motion') or {}
        for metric, limit, unit, note in (
                ('max_supported_body_drift_mm', 2., 'mm', 'The supported torso, head and shoulders remain stable while only the legs press.'),
                ('max_hand_drift_mm', 2., 'mm', 'The hands stay on their fixed support positions.'),
                ('top_extension_shortfall_deg', 0., 'deg', 'Both knees reach the extended sled position together.'),
                ('bottom_flexion_shortfall_deg', 0., 'deg', 'Both knees reach the lowered sled position.'),
                ('bottom_overfold_deg', 0., 'deg', 'The knees do not fold beyond the supported sled range.'),
                ('max_knee_asymmetry_deg', 5., 'deg', 'Both legs press together.')):
            row('sled_' + metric.removesuffix('_deg').removesuffix('_mm'), movement.get(metric), limit, bool(movement), note, unit)
            if not movement:
                rows[-1]['note'] = 'Leg press movement requires measured body and bilateral leg joints throughout every recorded frame.'
    required = {'seated': ('pelvis',), 'supine': ('pelvis', 'back', 'head',
            'foot_left_heel', 'foot_left_toe', 'foot_right_heel', 'foot_right_toe'),
        'hold': ('left_forearm', 'right_forearm', 'left_toe', 'right_toe'),
        'sled': ('pelvis', 'back', 'head', 'footplate_left', 'footplate_right',
                 'footplate_left_heel', 'footplate_left_toe', 'footplate_right_heel', 'footplate_right_toe')}.get(profile.get('family'), ())
    if exercise == 'Dumbbell Shoulder Press (seated overhead press)':
        required = ('pelvis', 'back', 'foot_left_heel', 'foot_left_toe', 'foot_right_heel', 'foot_right_toe')
    if profile.get('family') == 'squat':
        required = ('foot_left_heel', 'foot_left_toe', 'foot_right_heel', 'foot_right_toe')
    for name in required:
        if name not in (supports.get('surfaces') or {}):
            row('support_gap_' + name, None, 5., False)
    for name, reading in (supports.get('surfaces') or {}).items():
        count = reading.get('frames')
        complete = type(count) is int and count > 0 and count == report.get('frames') and reading.get('measured_frames') == count
        row('support_gap_' + name, reading.get('max_gap_mm'), 5., complete)
        row('support_penetration_' + name, reading.get('max_penetration_mm'), 3., complete)
    if profile.get('family') in ('supine', 'squat') or exercise == 'Dumbbell Shoulder Press (seated overhead press)':
        feet = supports.get('planted_feet') or {}
        count = report.get('frames')
        complete = (type(count) is int and count > 0 and feet.get('frames') == count
            and feet.get('measured_frames') == count and type(feet.get('first_frame')) is int
            and type(feet.get('last_frame')) is int and feet['last_frame'] - feet['first_frame'] + 1 == count)
        foot_metric = 'squat_foot_slide' if profile.get('family') == 'squat' else 'bench_foot_slide' if profile.get('family') == 'supine' else 'press_foot_slide'
        row(foot_metric, feet.get('max_slide_mm'), 3., complete,
            'Both heel and toe patches stay planted throughout the press.')
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
