"""A missing or partial shaft/body sweep cannot be certified as zero overlap."""
import pytest

from physical_checks import rows_for


def report():
    return {'frames': 32, 'physical': {'profile': {'holds': True}, 'contacts': {
        'equipment': 'bar', 'frames': 32, 'max_grip_anchor_error_mm': .01,
        'sides': {side: {'frames': 32, 'nearest_gap_mm_max': .1, 'deepest_inside_bar_mm': .5}
                  for side in ('left', 'right')},
        'body_clearance': {'status': 'clear_at_samples', 'measured_frames': 32,
                           'first_frame': 1, 'last_frame': 32, 'max_penetration_mm': 0., 'worst': None},
    }}}


def clearance_row(value):
    return next(row for row in rows_for(value) if row['metric'] == 'equipment_body_overlap')


def test_complete_explicit_zero_clearance_passes_without_an_overlap_candidate():
    row = clearance_row(report())
    assert row['ours'] == 0.
    assert row['status'] == 'pass'


@pytest.mark.parametrize('clearance', [None, {}, {'status': 'clear_at_samples'},
    {'status': 'clear_at_samples', 'worst': None},
    {'status': 'clear_at_samples', 'worst': {'depth_mm': 0.}}])
def test_absent_or_legacy_clearance_cannot_default_to_zero(clearance):
    value = report()
    value['physical']['contacts']['body_clearance'] = clearance
    row = clearance_row(value)
    assert row['kind'] == 'not-measured'
    assert row['status'] != 'pass'


@pytest.mark.parametrize('key,value', [('measured_frames', 31), ('measured_frames', True),
                                     ('first_frame', 2), ('last_frame', 31), ('last_frame', True),
                                     ('status', 'not_measured')])
def test_incomplete_or_invalid_sweep_coverage_does_not_pass(key, value):
    source = report()
    source['physical']['contacts']['body_clearance'][key] = value
    assert clearance_row(source)['kind'] == 'not-measured'


@pytest.mark.parametrize('value', [None, True, float('nan'), float('inf'), -1., '0'])
def test_clearance_depth_must_be_an_explicit_finite_nonnegative_measurement(value):
    source = report()
    source['physical']['contacts']['body_clearance']['max_penetration_mm'] = value
    assert clearance_row(source)['kind'] == 'not-measured'


def test_a_complete_overlap_above_the_existing_three_mm_limit_fails():
    value = report()
    value['physical']['contacts']['body_clearance'].update(status='overlap_candidates', max_penetration_mm=3.01)
    row = clearance_row(value)
    assert row['status'] == 'fail' and row['target'] == 3.


def test_dumbbells_require_their_weight_head_sweep_instead_of_a_continuous_bar_sweep():
    value = report()
    contacts = value['physical']['contacts']
    contacts['equipment'] = 'dumbbells'
    contacts.pop('body_clearance')
    contacts['head_body_clearance'] = {'frames': 32, 'max_penetration_mm': 0.}
    rows = rows_for(value)
    assert not any(row['metric'] == 'equipment_body_overlap' for row in rows)
    assert next(row for row in rows if row['metric'] == 'weight_head_body_overlap')['status'] == 'pass'
    contacts.pop('head_body_clearance')
    assert next(row for row in rows_for(value) if row['metric'] == 'weight_head_body_overlap')['kind'] == 'not-measured'
