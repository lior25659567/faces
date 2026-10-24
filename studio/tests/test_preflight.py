"""Preflight tests using committed measurements, without MediaPipe or private photographs."""
from copy import deepcopy
import json

import pytest

from conftest import ROOT
from build_targets import build_targets, to_pipeline_targets
from preflight import plan_for
import run_read

RAW = json.loads((ROOT / 'studio/tests/fixtures/squat-measurements.json').read_text())


def inputs(include_front=True):
    names = [('15.png', 'start'), ('16.png', 'middle'), ('17.png', 'end')]
    if include_front:
        names.append(('behind-thumb.png', 'start'))
    images = [dict(id=name, path=name, file=name, phase=phase, order=n,
                   view='behind' if name.startswith('behind') else 'side', viewSetBy='user')
              for n, (name, phase) in enumerate(names)]
    return {'exercise': 'Barbell Back Squat', 'equipment': 'barbell', 'images': images}


def plan(snapshot, clip='squat.fbx'):
    targets = build_targets(snapshot['exercise'], snapshot['images'], RAW)
    return plan_for(snapshot, targets, to_pipeline_targets(targets), run_read.key_positions(targets, RAW), clip)


def test_squat_missing_widths_stops_before_render():
    result = plan(inputs(False))
    assert not result['ready']
    assert 'front or rear' in result['blockers'][0]


def test_complete_squat_has_correction_and_exposes_uncertainty():
    result = plan(inputs())
    assert result['ready']
    assert result['mode'] == 'Corrected reference poses'
    assert any('disagree' in w for w in result['warnings'])


@pytest.mark.parametrize('value', [None, float('nan'), float('inf'), True])
def test_invalid_target_cannot_enable_solver(value):
    targets = build_targets('Barbell Back Squat', inputs()['images'], RAW)
    targets['phases']['end']['torso_lean']['value'] = value
    assert to_pipeline_targets(targets) is None
    result = plan_for(inputs(), targets, None, {}, 'squat.fbx')
    assert not result['ready']
    assert any('end position' in b and 'torso lean' in b for b in result['blockers'])


def test_row_uses_reference_motion_even_with_clip_and_explains_missing_bar():
    snap = inputs(False)
    snap['exercise'] = 'Barbell Bent-Over Row'
    result = plan(snap, 'row.fbx')
    assert result['ready']
    assert result['mode'] == 'Image key poses'
    assert not any('does not correct' in w for w in result['warnings'])
    assert any('hand contact' in w for w in result['warnings'])
    assert plan(snap, None)['mode'] == 'Image key poses'


def test_run_read_writes_blockers_and_removes_stale_solver_targets(tmp_path, monkeypatch):
    snap = inputs(False)
    (tmp_path / 'references.json').write_text(json.dumps(snap))
    (tmp_path / 'run.json').write_text(json.dumps({'clip': 'squat.fbx'}))
    (tmp_path / 'pipeline-targets.json').write_text('{"stale": true}')
    monkeypatch.setattr(run_read, 'read_cached', lambda path, reader, cache_dir: deepcopy(RAW[path.name]))
    assert run_read.main(tmp_path) == 3
    assert not (tmp_path / 'pipeline-targets.json').exists()
    assert not json.loads((tmp_path / 'preflight.json').read_text())['ready']
    assert (tmp_path / 'measurements.json').exists()


def test_run_read_complete_preserves_pose_outputs(tmp_path, monkeypatch):
    (tmp_path / 'references.json').write_text(json.dumps(inputs()))
    (tmp_path / 'run.json').write_text(json.dumps({'clip': 'squat.fbx', 'readOnly': True}))
    monkeypatch.setattr(run_read, 'read_cached', lambda path, reader, cache_dir: deepcopy(RAW[path.name]))
    assert run_read.main(tmp_path) == 0
    assert json.loads((tmp_path / 'preflight.json').read_text())['ready']
    assert json.loads((tmp_path / 'pipeline-targets.json').read_text())['phases']['bottom']['knee_included'] == 47
    assert len(json.loads((tmp_path / 'keyposes.json').read_text())['phases']) == 3


def test_run_read_too_few_poses_reports_preflight(tmp_path, monkeypatch):
    snap = inputs(False)
    snap['images'] = snap['images'][:1]
    (tmp_path / 'references.json').write_text(json.dumps(snap))
    monkeypatch.setattr(run_read, 'read_cached', lambda path, reader, cache_dir: deepcopy(RAW[path.name]))
    assert run_read.main(tmp_path) == 3
    result = json.loads((tmp_path / 'preflight.json').read_text())
    assert not result['ready']
    assert 'missing: end' in result['blockers'][0]
