"""Row oblique-input semantics, without running or falsifying a photo detector."""
from copy import deepcopy
import json

import pytest

from conftest import ROOT
from build_targets import ROW_EXERCISE, TooFewPositions, build_targets
from preflight import plan_for
from run_read import key_positions
import run_read

RAW = json.loads((ROOT / 'studio/tests/fixtures/squat-measurements.json').read_text())


def fixture(view='three-quarter'):
    images, readings = [], {}
    for n, (phase, source) in enumerate((('start', '15.png'), ('end', '17.png'))):
        name = phase + '.jpg'
        images.append({'id': phase, 'path': name, 'file': name, 'original': name,
                       'phase': phase, 'order': n, 'view': view, 'viewSetBy': 'user'})
        result = deepcopy(RAW[source])
        result.update(view=view, facing=None if view == 'three-quarter' else 'right')
        result['pixels'] = {name: point[:2] for name, point in result['world'].items()}
        result['visibility'] = {name: 0.9 for name in result['world']}
        readings[phase] = result
    return {'exercise': ROW_EXERCISE, 'equipment': 'barbell', 'images': images}, readings


def test_oblique_row_uses_inferred_3d_without_retagging_picture():
    snap, readings = fixture()
    targets = build_targets(ROW_EXERCISE, snap['images'], readings)
    for phase in ('start', 'end'):
        target = targets['phases'][phase]['hip_included']
        assert target['method'] == '3d'
        assert target['source'] == 'inferred_3d'
        assert target['value'] == readings[phase]['3d']['hip_included']
        assert target['cross'] is None
        assert targets['images'][phase]['view'] == 'three-quarter'
        assert 'back_rounding' not in targets['phases'][phase]
    keys = key_positions(targets, readings)
    assert all(entry['view'] == 'three-quarter' and entry['facing'] is None for entry in keys['phases'])
    assert keys['phases'][0]['visibility'] == readings['start']['visibility']
    plan = plan_for(snap, targets, None, keys)
    assert plan['ready']
    assert any('estimated 3D depth' in w for w in plan['warnings'])


def test_side_row_preserves_projected_targets():
    snap, readings = fixture('side')
    targets = build_targets(ROW_EXERCISE, snap['images'], readings)
    assert targets['phases']['start']['hip_included']['method'] == 'flat'
    assert targets['phases']['start']['hip_included']['value'] == readings['start']['flat']['hip_included']


def test_oblique_photos_do_not_enable_squat_correction():
    snap, readings = fixture()
    with pytest.raises(TooFewPositions, match='side-view'):
        build_targets('Barbell Back Squat', snap['images'], readings)


@pytest.mark.parametrize('damage', ['missing_wrist', 'nonfinite', 'missing_phase', 'hip_overlap'])
def test_row_preflight_blocks_unusable_geometry(damage):
    snap, readings = fixture()
    targets = build_targets(ROW_EXERCISE, snap['images'], readings)
    keys = key_positions(targets, readings)
    if damage == 'missing_wrist':
        del keys['phases'][1]['world']['left_wrist']
    elif damage == 'nonfinite':
        keys['phases'][1]['world']['left_wrist'] = [1, float('nan'), 0]
    elif damage == 'missing_phase':
        keys['phases'] = keys['phases'][:1]
    else:
        keys['phases'][1]['world']['left_hip'] = keys['phases'][1]['world']['right_hip'][:]
    plan = plan_for(snap, targets, None, keys)
    assert not plan['ready']
    assert plan['blockers']


def test_mixed_side_and_oblique_do_not_average_different_projection_methods():
    snap, readings = fixture()
    snap['images'][0]['view'] = readings['start']['view'] = 'side'
    readings['start']['facing'] = 'right'
    targets = build_targets(ROW_EXERCISE, snap['images'], readings)
    assert targets['phases']['middle']['estimated']
    assert 'hip_included' not in targets['phases']['middle']
    plan = plan_for(snap, targets, None, key_positions(targets, readings))
    assert not plan['ready']
    assert any("same view" in b and "tagged as photographed" in b for b in plan['blockers'])


def test_run_read_writes_actual_view_and_method_for_oblique_row(tmp_path, monkeypatch):
    snap, readings = fixture()
    (tmp_path / 'references.json').write_text(json.dumps(snap))
    monkeypatch.setattr(run_read, 'read_cached', lambda path, reader, cache_dir: deepcopy(readings[path.stem]))
    assert run_read.main(tmp_path) == 0
    targets = json.loads((tmp_path / 'targets.json').read_text())
    assert targets['phases']['start']['torso_lean']['method'] == '3d'
    keys = json.loads((tmp_path / 'keyposes.json').read_text())
    assert keys['phases'][0]['view'] == 'three-quarter'
    assert keys['phases'][0]['facing'] is None
    assert json.loads((tmp_path / 'preflight.json').read_text())['ready']


def test_middle_cannot_change_projection_midway():
    snap, readings = fixture()
    targets = build_targets(ROW_EXERCISE, snap['images'], readings)
    keys = key_positions(targets, readings)
    middle = deepcopy(keys['phases'][0])
    middle.update(phase='middle', view='side')
    keys['phases'].insert(1, middle)
    plan = plan_for(snap, targets, None, keys)
    assert not plan['ready']
    assert any('Middle' in b and 'same view' in b for b in plan['blockers'])
