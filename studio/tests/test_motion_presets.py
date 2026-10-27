"""Preset mode is explicit; it does not weaken photograph reconstruction checks."""
import json
from pathlib import Path

import pytest

import run_read
from motion_presets import PRESETS, ROOT, movement_images, preset_for, preset_plan, preset_targets
from preflight import plan_for


def test_a_pinned_preset_does_not_require_fabricated_side_photographs():
    snapshot = {'exercise': 'Barbell Back Squat', 'motionPreset': 'squat-v1', 'equipment': 'barbell', 'images': []}
    plan = plan_for(snapshot, {}, None, None)
    assert plan['ready'] and plan['mode'] == 'Movement preset'
    assert plan['motionPreset'] == 'squat-v1'
    assert 'context' in plan['warnings'][0]


def test_unknown_cross_exercise_and_changed_clips_are_rejected(tmp_path):
    for name, identifier in [('Pull-Up', 'squat-v1'), ('Barbell Back Squat', '../clip')]:
        with pytest.raises(ValueError, match='not available'):
            preset_for(name, identifier)
    clip = tmp_path / PRESETS['squat-v1']['clip']
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b'changed clip')
    with pytest.raises(ValueError, match='changed'):
        preset_for('Barbell Back Squat', 'squat-v1', root=tmp_path)


def test_preset_read_keeps_tags_and_does_not_read_context_images(tmp_path, monkeypatch):
    snapshot = {'exercise': 'Barbell Back Squat', 'motionPreset': 'squat-v1', 'equipment': 'barbell',
                'images': [{'id': 'context', 'view': 'three-quarter', 'contextOnly': False, 'path': 'not-a-body.jpg'}]}
    source = json.dumps(snapshot)
    (tmp_path / 'references.json').write_text(source)
    (tmp_path / 'pipeline-targets.json').write_text('{"stale":true}')
    monkeypatch.setattr(run_read, 'read_cached', lambda *args: pytest.fail('Preset context must not drive inferred motion.'))
    assert run_read.main(tmp_path) == 0
    assert (tmp_path / 'references.json').read_text() == source
    assert not (tmp_path / 'pipeline-targets.json').exists()
    targets = json.loads((tmp_path / 'targets.json').read_text())
    assert targets['motion']['source'] == 'preset'
    assert targets['images']['context']['used'] is False
    assert json.loads((tmp_path / 'measurements.json').read_text()) == {}
    assert json.loads((tmp_path / 'keyposes.json').read_text())['phases'] == []


def test_preset_has_recorded_timing_and_explicit_non_photo_targets():
    preset = preset_for('Barbell Back Squat', 'squat-v1')
    targets = preset_targets({'exercise': preset['exercise'], 'images': []}, preset)
    assert targets['motion']['phase_frames'] == {'start': 1, 'middle': 19, 'end': 38}
    assert targets['phases']['end']['knee_included']['source'] == 'declared-preset-range'
    assert preset_plan({'exercise': 'Barbell Back Squat', 'motionPreset': 'unknown'})['ready'] is False


def selected_walking_snapshot():
    source = json.loads((ROOT / 'reference/exercise-lunge/sources.json').read_text())
    sequence = json.loads((ROOT / 'reference/exercise-lunge/one-step/sequence.json').read_text())
    bundle = {'id': f"wger-{source['exerciseId']}-video-{source['videoId']}-one-step-v1", 'commit': source['sha256']}
    images = [{'id': 'old-sequence-start', 'phase': 'start', 'view': 'front', 'contextOnly': False,
               'source': {'catalogId': 'previous-sequence'}}]
    for frame in sequence['frames']:
        images.append({'id': frame['file'], 'phase': frame.get('phase') or 'middle', 'view': 'front', 'contextOnly': False,
            'path': str(ROOT / 'reference/exercise-lunge/one-step' / frame['file']),
            'source': {'catalogId': bundle['id'], 'commit': bundle['commit'],
                       'sequenceIndex': frame['sequenceIndex'], 'sha256': frame['sha256']}})
    return {'exercise': sequence['exercise'], 'referenceBundle': bundle, 'images': images}


def test_selected_dense_chronology_interprets_old_images_as_context_without_retagging_them():
    snapshot = selected_walking_snapshot()
    original = json.dumps(snapshot)
    images = movement_images(snapshot)
    assert images[0]['contextOnly'] is True
    assert len([image for image in images if not image['contextOnly']]) == 26
    assert json.dumps(snapshot) == original


@pytest.mark.parametrize('edit', ['phase', 'view', 'hash', 'bundle'])
def test_selected_source_and_manual_tag_changes_cannot_silently_change_chronology(edit):
    snapshot = selected_walking_snapshot()
    if edit == 'phase': snapshot['images'][1]['phase'] = 'end'
    elif edit == 'view': snapshot['images'][1]['view'] = 'side'
    elif edit == 'hash': snapshot['images'][1]['source']['sha256'] = '0' * 64
    else: snapshot['referenceBundle']['commit'] = '0' * 64
    with pytest.raises(ValueError):
        movement_images(snapshot)


def test_invalid_sequence_selection_removes_stale_inputs_before_stopping(tmp_path):
    snapshot = selected_walking_snapshot()
    snapshot['referenceBundle']['commit'] = '0' * 64
    (tmp_path / 'references.json').write_text(json.dumps(snapshot))
    (tmp_path / 'keyposes.json').write_text('{"stale":true}')
    (tmp_path / 'pipeline-targets.json').write_text('{"stale":true}')
    assert run_read.main(tmp_path) == 3
    assert not (tmp_path / 'keyposes.json').exists()
    assert not (tmp_path / 'pipeline-targets.json').exists()
    assert json.loads((tmp_path / 'preflight.json').read_text())['ready'] is False


def test_a_source_hash_label_does_not_certify_replaced_image_bytes(tmp_path):
    snapshot = selected_walking_snapshot()
    changed = tmp_path / 'changed.jpg'
    changed.write_bytes(b'not the reviewed source frame')
    snapshot['images'][1]['path'] = str(changed)
    with pytest.raises(ValueError, match='missing or changed'):
        movement_images(snapshot)
