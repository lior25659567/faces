"""Different photographed phases must not share a fixed elbow-width target."""
from copy import deepcopy
import json

import pytest

from conftest import ROOT
from build_targets import build_targets
from checks import _width_rows

RAW = json.loads((ROOT / 'studio/tests/fixtures/squat-measurements.json').read_text())
EXERCISE = 'Pull-Up'


def inputs():
    images = [{'id': name, 'phase': phase, 'order': index, 'view': 'front'}
              for index, (name, phase) in enumerate([('start-photo', 'start'), ('end-photo', 'end')])]
    measured = {name: deepcopy(RAW['15.png']) for name in ('start-photo', 'end-photo')}
    for name, width in [('start-photo', 1.616), ('end-photo', 2.048)]:
        for method in ('flat', '3d'):
            measured[name][method]['elbow_width_ratio'] = width
    ours = {'phase_frames': {'start': 1, 'middle': 2, 'end': 3},
            'per_frame': [{'frame': frame, 'flat': {'grip_ratio': 1.87, 'elbow_width_ratio': width}}
                          for frame, width in [(1, 1.198), (2, 1.8), (3, 2.048)]]}
    return images, measured, ours


def elbows(targets, ours, exercise=EXERCISE):
    return {row['group']: row for row in _width_rows(targets, ours, exercise) if row['metric'] == 'elbow_width_ratio'}


def test_start_and_end_use_their_own_photographed_elbow_spread():
    images, measured, ours = inputs()
    targets = build_targets(EXERCISE, images, measured)
    assert targets['phase_widths']['start']['elbow_width_ratio']['image'] == 'start-photo'
    assert targets['phase_widths']['end']['elbow_width_ratio']['image'] == 'end-photo'
    rows = elbows(targets, ours)
    assert set(rows) == {'start', 'middle', 'end'}
    assert rows['start']['target'] == pytest.approx(1.616)
    assert rows['start']['ours'] == pytest.approx(1.198)
    assert rows['start']['status'] == 'fail'  # Fixing the End target does not hide the real Start error.
    assert rows['end']['target'] == pytest.approx(2.048)
    assert rows['end']['status'] == 'pass'
    assert rows['middle']['kind'] == 'not-judged'
    assert 'estimated' in rows['middle']['note']


def test_missing_end_width_never_reuses_the_start_photo_as_its_target():
    images, measured, ours = inputs()
    measured['end-photo']['flat']['elbow_width_ratio'] = None
    measured['end-photo']['3d']['elbow_width_ratio'] = None
    rows = elbows(build_targets(EXERCISE, images, measured), ours)
    assert rows['end']['target'] is None
    assert rows['end']['kind'] == 'not-judged'
    assert rows['start']['status'] == 'fail'


def test_missing_character_measurement_is_not_measured_at_that_phase():
    images, measured, ours = inputs()
    ours['per_frame'] = ours['per_frame'][:-1]
    row = elbows(build_targets(EXERCISE, images, measured), ours)['end']
    assert row['target'] == pytest.approx(2.048)
    assert row['ours'] is None
    assert row['kind'] == 'not-measured'
    assert row['frame'] == 3


def test_uncertain_phase_width_keeps_its_source_uncertainty():
    images, measured, ours = inputs()
    measured['end-photo']['3d']['elbow_width_ratio'] = 3.
    row = elbows(build_targets(EXERCISE, images, measured), ours)['end']
    assert row['status'] == 'uncertain'
    assert row['target'] == pytest.approx(2.048)
    assert row['uncertain_why'] == 'coach'


def test_global_grip_target_is_preserved_while_elbow_spread_changes():
    images, measured, ours = inputs()
    measured['end-photo']['flat']['grip_ratio'] = 2.8
    targets = build_targets(EXERCISE, images, measured)
    assert targets['widths']['grip_ratio']['image'] == 'start-photo'
    assert all(set(reading) <= {'elbow_width_ratio'} for reading in targets['phase_widths'].values())
    rows = _width_rows(targets, ours, EXERCISE)
    assert [row['group'] for row in rows if row['metric'] == 'grip_ratio'] == ['all positions']


def test_legacy_squat_width_policy_stays_global_even_if_phase_metadata_is_present():
    images = [{'id': name, 'phase': phase, 'order': index, 'view': view}
              for index, (name, phase, view) in enumerate([('15.png', 'start', 'side'),
                ('16.png', 'middle', 'side'), ('17.png', 'end', 'side'), ('behind-thumb.png', 'start', 'behind')])]
    targets = build_targets('Barbell Back Squat', images, RAW)
    assert 'phase_widths' not in targets
    targets['phase_widths'] = {'start': {'elbow_width_ratio': {'value': 999.}}}
    ours = inputs()[2]
    rows = elbows(targets, ours, 'Barbell Back Squat')
    assert set(rows) == {'all positions'}
    assert rows['all positions']['target'] != 999.
