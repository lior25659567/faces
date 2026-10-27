"""Explicit pinned movement sources, separate from reconstructing photographs."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PRESETS = json.loads(Path(__file__).with_name('motion-presets.json').read_text())['presets']


def preset_for(exercise, identifier, root=ROOT):
    if identifier is None:
        return None
    preset = PRESETS.get(identifier) if isinstance(identifier, str) else None
    if not preset or preset['exercise'] != exercise:
        raise ValueError('That movement preset is not available for this exercise.')
    clip = Path(root) / preset['clip']
    if not clip.is_file():
        raise ValueError('The pinned movement clip is missing.')
    if hashlib.sha256(clip.read_bytes()).hexdigest() != preset['sha256']:
        raise ValueError('The movement clip changed. Restore the pinned clip before using this preset.')
    return {'id': identifier, **preset}


def preset_targets(snapshot, preset):
    from build_targets import TOLERANCE
    return {'exercise': snapshot['exercise'], 'equipment': snapshot.get('equipment'),
        'phases': {phase: {'estimated': True, 'knee_included': {'value': angle, 'method': 'flat',
            'source': 'declared-preset-range', 'uncertain': False}}
            for phase, angle in (('start', 180.), ('middle', 100.), ('end', 45.))},
        'widths': {}, 'tolerance': dict(TOLERANCE), 'camera': {'side': 'left'},
        'images': {image['id']: {'view': image.get('view'), 'used': False,
            'why': 'Context for the declared movement preset; not a calibrated motion target.'}
            for image in snapshot.get('images', []) if not image.get('deleted')},
        'motion': {'source': 'preset', 'preset': preset['id'], 'mode': preset['mode'],
            'phase_frames': preset['phase_frames'], 'sourceDepthCalibrated': False,
            'description': preset['source']}}


def preset_plan(snapshot):
    try:
        preset = preset_for(snapshot['exercise'], snapshot.get('motionPreset'))
    except ValueError as error:
        return {'ready': False, 'mode': 'Movement preset', 'blockers': [str(error)], 'warnings': []}
    if preset is None:
        return None
    return {'ready': True, 'mode': 'Movement preset', 'motionPreset': preset['id'],
        'blockers': [], 'warnings': [preset['source'],
            'Every build still has to pass movement range, bar contact, floor clearance and visible skin checks.']}


def movement_images(snapshot):
    """Interpret an explicit source selection without changing stored photo tags."""
    images = snapshot.get('images') or []
    selection = snapshot.get('referenceBundle')
    if selection is None:
        return images
    source = json.loads((ROOT / 'reference/exercise-lunge/sources.json').read_text())
    sequence = json.loads((ROOT / 'reference/exercise-lunge/one-step/sequence.json').read_text())
    identifier = f"wger-{source['exerciseId']}-video-{source['videoId']}-one-step-v1"
    if (snapshot.get('exercise') != sequence['exercise'] or not isinstance(selection, dict)
            or selection.get('id') != identifier or selection.get('commit') != source['sha256']):
        raise ValueError('The selected chronological movement source is not the reviewed one-step bundle.')
    matched, result = 0, []
    for image in images:
        provenance = image.get('source') or {}
        chosen = provenance.get('catalogId') == identifier and provenance.get('commit') == source['sha256']
        if chosen and not image.get('deleted'):
            index = provenance.get('sequenceIndex')
            if type(index) is not int or not 0 <= index < len(sequence['frames']):
                raise ValueError('A selected walking frame has an invalid source index.')
            frame = sequence['frames'][index]
            if (provenance.get('sha256') != frame['sha256'] or image.get('phase') != (frame.get('phase') or 'middle')
                    or image.get('view') != 'front'):
                raise ValueError('A reviewed walking frame or its tags changed. Review that edit before rebuilding the selected sequence.')
            file = Path(image.get('path') or '')
            if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != frame['sha256']:
                raise ValueError('A reviewed walking image is missing or changed. Retrieve its pinned source frame before rebuilding.')
            matched += 1
        result.append(image if chosen else {**image, 'contextOnly': True})
    if not matched:
        raise ValueError('The selected walking reference bundle is missing; retrieve its reviewed frames before building.')
    return result
