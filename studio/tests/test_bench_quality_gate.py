"""The original visually incorrect bench must fail even with a good contact report."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from physical_checks import rows_for


def measured_report():
    movement = json.loads((Path(__file__).with_name('fixtures') / 'bench-original-range.json').read_text())
    return {'frames': movement['frames'], 'reference_motion': {'phase_frames': movement['phase_frames']},
        'physical': {'profile': {'family': 'supine'},
            'contacts': {'frames': movement['frames'], 'bench_range': movement}}}


def movement_rows(report):
    return {row['metric']: row for row in rows_for(report)
        if row['metric'].startswith('bench_') and row['metric'] != 'bench_foot_slide'}


def test_actual_original_bench_fails_lockout_forearm_and_bar_path_checks():
    rows = movement_rows(measured_report())
    assert rows['bench_lockout_shortfall']['ours'] > 48.
    assert rows['bench_stroke_reversal']['ours'] > 40.
    for metric in ('bench_lockout_shortfall', 'bench_forearm_tilt_excess',
                   'bench_bar_torso_path_excess', 'bench_stroke_reversal', 'bench_lockout_shoulder_offset'):
        assert rows[metric]['status'] == 'fail'


def test_cached_zero_failure_summary_cannot_hide_the_measured_original_defect():
    report = measured_report()
    movement = report['physical']['contacts']['bench_range']
    movement.update(lockout_shortfall_deg=0., stroke_reversal_mm=0., ready=True)
    rows = movement_rows(report)
    assert rows['bench_lockout_shortfall']['status'] == 'fail'
    assert rows['bench_stroke_reversal']['status'] == 'fail'


@pytest.mark.parametrize('defect', ['missing', 'duplicate', 'phase', 'nonfinite', 'forged_angle'])
def test_incomplete_or_inconsistent_raw_geometry_cannot_pass(defect):
    report = measured_report()
    movement = report['physical']['contacts']['bench_range']
    if defect == 'missing':
        movement['per_frame'].pop(10)
    elif defect == 'duplicate':
        movement['per_frame'][10] = deepcopy(movement['per_frame'][9])
    elif defect == 'phase':
        movement['phase_frames'] = {**movement['phase_frames'], 'end': 18}
    elif defect == 'nonfinite':
        movement['per_frame'][10]['bar_m'][2] = float('nan')
    else:
        movement['per_frame'][0]['elbows_deg']['left'] = 179.
    rows = movement_rows(report)
    assert rows and all(row['kind'] == 'not-measured' for row in rows.values())


def test_bench_requires_both_heel_and_toe_patches_and_all_frame_foot_drift():
    report = measured_report()
    report['physical']['supports'] = {'surfaces': {
        name: {'frames': report['frames'], 'measured_frames': report['frames'],
               'max_gap_mm': 0., 'max_penetration_mm': 0.}
        for name in ('pelvis', 'back', 'head', 'foot_left_toe', 'foot_right_toe')}}
    missing = {row['metric'] for row in rows_for(report) if row['kind'] == 'not-measured'}
    assert {'support_gap_foot_left_heel', 'support_gap_foot_right_heel', 'bench_foot_slide'} <= missing
