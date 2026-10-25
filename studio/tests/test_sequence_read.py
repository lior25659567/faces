"""Sequence images stay readable without becoming duplicate comparison targets."""
from copy import deepcopy
import json

from conftest import ROOT
from build_targets import build_targets
from read_pose import NoBody
import run_read

MANIFEST = json.loads((ROOT / 'reference/exercise-lunge/sequence.json').read_text())
RAW = json.loads((ROOT / 'studio/tests/fixtures/squat-measurements.json').read_text())['15.png']
EXERCISE = MANIFEST['exercise']


def inputs():
    anchors = {0: 'start', 3: 'middle', 8: 'end'}
    images, measured = [], {}
    for frame in MANIFEST['frames']:
        index = frame['sequenceIndex']
        source = {**frame, 'phase': anchors.get(index, 'middle'), 'timeSeconds': frame['sourceTimestampSeconds'],
                  'motionOnly': index not in anchors, 'sequenceInfo': {'completeClosedCycle': False, 'calibrated': False}}
        image = dict(id=frame['id'], file=frame['file'], path=frame['file'], order=index,
                     phase=source['phase'], view='front', motionOnly=source['motionOnly'], source=source)
        result = deepcopy(RAW)
        result['3d']['knee_included'] = 170 - index * 3
        images.append(image)
        measured[image['id']] = result
    return images, measured


def test_motion_samples_do_not_take_phases_or_widths_from_comparison_anchors():
    images, measured = inputs()
    targets = build_targets(EXERCISE, images, measured)
    assert targets['motion']['mode'] == 'sequence'
    for phase, index in [('start', 0), ('middle', 3), ('end', 8)]:
        reading = targets['phases'][phase]['knee_included']
        assert reading['image'] == f'lunge-{index:02}'
        assert reading['value'] == 170 - index * 3
        assert reading['method'] == '3d' and reading['uncertain']
    assert targets['images']['lunge-01'] == {'used': False, 'why': 'Motion sample', 'view': 'front'}
    assert targets['widths']['grip_ratio']['image'] == 'lunge-00'
    assert not targets['phases']['middle'].get('estimated')


def test_keyposes_keep_every_chronological_sample_and_uncertain_contact_annotation():
    images, measured = inputs()
    keys = run_read.key_positions(build_targets(EXERCISE, images, measured), measured, images)
    assert len(keys['phases']) == 3
    samples = keys['sequence']['samples']
    assert len(samples) == 12
    assert [sample['sequenceIndex'] for sample in samples] == list(range(12))
    assert [sample['timeSeconds'] for sample in samples] == [frame['sourceTimestampSeconds'] for frame in MANIFEST['frames']]
    assert samples[3]['footContacts']['left'] == 'uncertain'
    assert samples[8]['footContacts']['right'] == 'uncertain'
    assert samples[10]['footContacts']['right'] == 'airborne'
    assert keys['sequence']['calibrated'] is False
    assert keys['sequence']['completeClosedCycle'] is False
    assert keys['sequence']['sourceInferredDepth'] is True


def test_unreadable_motion_sample_is_preserved_as_a_missing_position_not_silently_dropped():
    images, measured = inputs()
    measured['lunge-02'] = NoBody('occluded')
    targets = build_targets(EXERCISE, images, measured)
    keys = run_read.key_positions(targets, measured, images)
    assert len(keys['sequence']['samples']) == 12
    assert keys['sequence']['samples'][2]['sequenceIndex'] == 2
    assert 'error' in keys['sequence']['samples'][2]
    assert 'world' not in keys['sequence']['samples'][2]


def test_run_read_reads_motion_samples_but_does_not_fall_back_to_a_cyclic_existing_clip(tmp_path, monkeypatch):
    images, measured = inputs()
    old_context = {'id': 'old-context', 'path': '/missing/old.jpg', 'phase': 'start', 'contextOnly': True}
    snapshot = {'exercise': EXERCISE, 'equipment': 'dumbbells', 'images': [old_context, *images]}
    (tmp_path / 'references.json').write_text(json.dumps(snapshot))
    (tmp_path / 'run.json').write_text(json.dumps({'clip': 'lunge.fbx'}))
    calls = []

    def read(path, reader, cache_dir):
        calls.append(path.stem)
        return deepcopy(measured[path.stem])

    monkeypatch.setattr(run_read, 'read_cached', read)
    assert run_read.main(tmp_path) == 3
    assert calls == [image['id'] for image in images]
    assert len(json.loads((tmp_path / 'measurements.json').read_text())) == 12
    assert len(json.loads((tmp_path / 'keyposes.json').read_text())['sequence']['samples']) == 12
    plan = json.loads((tmp_path / 'preflight.json').read_text())
    assert plan['ready'] is False
    assert plan['mode'] == 'Chronological image sequence'
    assert any('forward root-motion' in blocker for blocker in plan['blockers'])
    assert not any('does not correct' in warning for warning in plan['warnings'])
    assert (tmp_path / 'sequence-continuity.json').exists()


def test_motion_only_in_source_also_excludes_an_image_from_comparisons():
    images, measured = inputs()
    for image in images:
        image.pop('motionOnly')
    targets = build_targets(EXERCISE, images, measured)
    assert targets['phases']['middle']['knee_included']['image'] == 'lunge-03'
    assert targets['images']['lunge-01']['why'] == 'Motion sample'
