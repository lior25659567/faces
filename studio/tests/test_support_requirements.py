"""A leg press or bench cannot pass with unsupported back/head geometry."""
from copy import deepcopy
from physical_checks import rows_for

SLED_FEET = ['pelvis', 'footplate_left', 'footplate_right', 'footplate_left_heel',
             'footplate_left_toe', 'footplate_right_heel', 'footplate_right_toe']


def report_for(family, names):
    return {'frames': 8, 'physical': {'profile': {'family': family}, 'supports': {'surfaces': {
        name: {'frames': 8, 'measured_frames': 8, 'max_gap_mm': .2, 'max_penetration_mm': .1}
        for name in names}}}}


def test_leg_press_requires_back_and_head_even_when_seat_and_footplate_pass():
    report = report_for('sled', SLED_FEET)
    report['physical']['supports']['rail_path'] = {'frames': 8, 'max_deviation_mm': .1}
    rows = rows_for(report)
    missing = {row['metric'] for row in rows if row['kind'] == 'not-measured'}
    assert missing == {'support_gap_back', 'support_gap_head'}


def test_every_frame_of_a_leg_press_backrest_must_be_measured():
    report = report_for('sled', SLED_FEET + ['back', 'head'])
    report['physical']['supports']['rail_path'] = {'frames': 8, 'max_deviation_mm': .1}
    assert all(row['status'] == 'pass' for row in rows_for(report))
    for name in ('back', 'head'):
        incomplete = deepcopy(report)
        incomplete['physical']['supports']['surfaces'][name]['measured_frames'] = 7
        rows = rows_for(incomplete)
        assert any(row['metric'] == 'support_gap_' + name and row['kind'] == 'not-measured' for row in rows)
        incomplete = deepcopy(report)
        incomplete['physical']['supports']['surfaces'][name]['max_gap_mm'] = 6.
        assert any(row['metric'] == 'support_gap_' + name and row['status'] == 'fail' for row in rows_for(incomplete))


def test_leg_press_cannot_pass_without_a_complete_measured_rail_path():
    report = report_for('sled', SLED_FEET + ['back', 'head'])
    for rail in (None, {'frames': 7, 'max_deviation_mm': 0.}, {'max_deviation_mm': 0.}):
        if rail is not None:
            report['physical']['supports']['rail_path'] = rail
        row = next(row for row in rows_for(report) if row['metric'] == 'sled_rail_deviation')
        assert row['kind'] == 'not-measured'


def test_nearest_toe_contact_cannot_certify_a_flat_foot_on_the_plate():
    report = report_for('sled', ['pelvis', 'back', 'head', 'footplate_left', 'footplate_right'])
    report['physical']['supports']['rail_path'] = {'frames': 8, 'max_deviation_mm': 0.}
    missing = {row['metric'] for row in rows_for(report) if row['kind'] == 'not-measured'}
    assert missing == {'support_gap_footplate_left_heel', 'support_gap_footplate_left_toe',
        'support_gap_footplate_right_heel', 'support_gap_footplate_right_toe'}


def test_bench_requires_skull_support_in_addition_to_back_and_pelvis():
    rows = rows_for(report_for('supine', ['pelvis', 'back']))
    assert any(row['metric'] == 'support_gap_head' and row['kind'] == 'not-measured' for row in rows)


def test_a_rejected_body_correction_cannot_pass_with_stale_good_contact_numbers():
    report = report_for('supine', ['pelvis', 'back', 'head'])
    report['physical']['supports']['correction'] = {'applied': False, 'reason': 'No feasible joint pose.'}
    rows = rows_for(report)
    row = next(row for row in rows if row['metric'] == 'support_geometry')
    assert row['status'] != 'pass'
    assert row['note'] == 'No feasible joint pose.'
