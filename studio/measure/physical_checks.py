"""Strict checks of the extra measured equipment/body support reports."""
import math


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def rows_for(report):
    physical = report.get('physical')
    if not physical:
        return []
    rows = []
    def row(metric, value, limit, complete=True, note=None):
        measured = finite(value) and value >= 0 and complete
        rows.append({'group': 'body', 'metric': metric, 'target': limit, 'ours': value if finite(value) else None,
            'status': ('pass' if value <= limit else 'fail') if measured else 'info',
            'kind': None if measured else 'not-measured', 'frame': None, 'unit': 'mm',
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
        clearance = contacts.get('body_clearance')
        if clearance:
            row('equipment_body_overlap', (clearance.get('worst') or {}).get('depth_mm', 0.), 3.,
                clearance.get('status') in ('clear_at_samples', 'overlap_candidates'),
                'Conservative shaft/body surface samples; overlap candidates need visual review.')
        heads = contacts.get('head_body_clearance')
        if contacts.get('equipment') == 'dumbbells':
            row('weight_head_body_overlap', (heads or {}).get('max_penetration_mm'), 3.,
                complete and (heads or {}).get('frames') == count,
                'Finite convex weight-head/body surface samples; inspect the rendered silhouette too.')
        if profile.get('overhead'):
            row('overhead_shortfall', contacts.get('overhead_shortfall_mm'), 0., complete,
                'Both handles must reach above the head; a secure grip alone does not establish the required movement range.')
    supports = physical.get('supports') or {}
    required = {'seated': ('pelvis',), 'supine': ('pelvis', 'back'),
        'hold': ('left_forearm', 'right_forearm', 'left_toe', 'right_toe'),
        'sled': ('pelvis', 'footplate_left', 'footplate_right')}.get(profile.get('family'), ())
    for name in required:
        if name not in (supports.get('surfaces') or {}):
            row('support_gap_' + name, None, 5., False)
    for name, reading in (supports.get('surfaces') or {}).items():
        count = reading.get('frames')
        complete = type(count) is int and count > 0 and count == report.get('frames') and reading.get('measured_frames') == count
        row('support_gap_' + name, reading.get('max_gap_mm'), 5., complete)
        row('support_penetration_' + name, reading.get('max_penetration_mm'), 3., complete)
    if supports.get('rail_path'):
        row('sled_rail_deviation', supports['rail_path'].get('max_deviation_mm'), 10.)
    if supports.get('limitation'):
        row('support_motion', None, 0., False, supports['limitation'])
        rows[-1]['note'] = supports['limitation']
    return rows
