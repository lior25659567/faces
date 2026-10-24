"""Independent row checks may not substitute unrelated inferred render depth."""
from checks import _phase_rows, _body_rows
from build_targets import TOLERANCE


def test_oblique_row_compares_actual_rig_and_discloses_depth():
    targets = {'exercise': 'Barbell Bent-Over Row', 'equipment': 'barbell', 'widths': {},
               'tolerance': TOLERANCE, 'phases': {'start': {
                   'torso_lean': {'value': 50, 'method': '3d', 'source': 'inferred_3d',
                                  'uncertain': False, 'cross': None}}}}
    ours = {'phase_frames': {'start': 1}, 'per_frame': [{'frame': 1, '3d': {'torso_lean': 50}}],
            'picture': {'start': {'ok': True, '3d': {'torso_lean': 5}}}}
    row = next(r for r in _phase_rows(targets, ours, targets['exercise']) if r['metric'] == 'torso_lean')
    assert row['ours'] == 50 and row['status'] == 'pass'
    assert 'inferred' in row['note']


def test_row_with_no_contact_report_cannot_claim_contact_pass():
    row = next(r for r in _body_rows({}, 'Barbell Bent-Over Row') if r['metric'] == 'hands_on_bar')
    assert row['status'] == 'info' and row['kind'] == 'not-measured'


def test_one_contacting_hand_cannot_hide_a_floating_hand():
    hands={'frames':81, 'deepest_inside_bar_mm':0, 'frames_touching_within_3mm':81,
           'sides': {side: {'frames':81,'deepest_inside_bar_mm':0,'frames_touching_within_3mm':count}
                     for side,count in [('left',81),('right',0)]}}
    rows=_body_rows({'grip':{'contact':hands}},'Barbell Bent-Over Row')
    row=next(r for r in rows if r['metric']=='hands_on_bar')
    assert row['status']=='fail'
    del hands['sides']['right']
    row=next(r for r in _body_rows({'grip':{'contact':hands}},'Barbell Bent-Over Row') if r['metric']=='hands_on_bar')
    assert row['kind']=='not-measured'
