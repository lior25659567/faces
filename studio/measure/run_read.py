"""Read a run's reference images and write measurements.json, targets.json, pipeline-targets.json, and
keyposes.json (the key side pictures' landmarks, which an exercise without a Mixamo clip is animated from)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_targets import PHASES, SOLVER_EXERCISES, TooFewPositions, build_targets, to_pipeline_targets  # noqa: E402
from read_pose import NoBody, read_pose  # noqa: E402
from preflight import plan_for  # noqa: E402
from pose_cache import read_cached  # noqa: E402
from motion_presets import movement_images, preset_for, preset_plan, preset_targets  # noqa: E402

KEY_FIELDS = ("view", "facing", "camera_side", "size", "world", "pixels", "visibility")


def key_positions(targets, measured, images=None):
    """keyposes.json's content: {"exercise", "phases": [{"phase", "image", "view", "facing", "camera_side",
    "size", "world", "pixels"}]} for the side pictures the targets were read from (`view` as the targets took
    it: the user's tag, else the detector's), in start, middle, end order
    (pipeline/blender/keypose_clip.py animates between them). A position the targets only estimated (no middle
    picture) is left out; None without a start and an end that carry world landmarks."""
    phases = []
    seen_images = set()
    for phase in PHASES:
        readings = targets["phases"].get(phase) or {}
        image = next((r.get("image") for r in readings.values() if isinstance(r, dict) and r.get("image")), None)
        result = measured.get(image) if image else None
        if (targets.get('motion') or {}).get('mode') == 'hold' and image in seen_images:
            continue
        if not isinstance(result, dict) or not result.get("world"):
            continue
        entry = {"phase": phase, "image": image, **{k: result.get(k) for k in KEY_FIELDS}}
        # the view the targets used: the user's tag when set (a picture the detector called three-quarter can be
        # set to Side), else the detector's
        entry["view"] = (targets.get("images", {}).get(image) or {}).get("view") or entry["view"]
        phases.append(entry)
        seen_images.add(image)
    hold = (targets.get('motion') or {}).get('mode') == 'hold'
    if not (hold and len(phases) == 1) and {"start", "end"} - {p["phase"] for p in phases}:
        return None
    result = {"exercise": targets["exercise"], "phases": phases}
    if targets.get('motion'):
        result['motion'] = targets['motion']
    if (targets.get('motion') or {}).get('mode') == 'sequence':
        samples = []
        sequence_info = None
        for image in images or []:
            if image.get('contextOnly') or image.get('deleted'):
                continue
            source = image.get('source') or {}
            if not isinstance(source.get('sequenceInfo'), dict):
                continue
            sequence_info = sequence_info or source['sequenceInfo']
            reading = measured.get(image['id'])
            entry = {'image': image['id'], 'source': source,
                     'phase': image.get('phase'),
                     **{key: source.get(key) for key in ('sequenceIndex', 'timeSeconds', 'sourceFrameIndex',
                       'role', 'leadLeg', 'observedSupport', 'footContacts', 'observedContactPoints', 'contactNote', 'motionOnly')}}
            if isinstance(reading, dict) and reading.get('world'):
                entry.update({key: reading.get(key) for key in KEY_FIELDS})
                entry['view'] = image.get('view') or entry['view']
            else:
                entry['error'] = 'The motion sample has no readable body landmarks.'
            samples.append(entry)
        result['sequence'] = {**(sequence_info or {}), 'samples': samples,
                              'sourceInferredDepth': True, 'calibrated': False}
    return result


def main(run_dir):
    run_dir = Path(run_dir)
    snap = json.loads((run_dir / "references.json").read_text())
    # Remove derived inputs even when the selected source itself is invalid.
    for name in ('targets.json', 'pipeline-targets.json', 'keyposes.json', 'preflight.json', 'read-problems.json', 'sequence-continuity.json'):
        (run_dir / name).unlink(missing_ok=True)
    try:
        images = movement_images(snap)
    except ValueError as error:
        (run_dir / 'preflight.json').write_text(json.dumps({'ready': False, 'mode': 'Chronological image sequence',
            'blockers': [str(error)], 'warnings': []}, indent=2))
        (run_dir / 'measurements.json').write_text('{}\n')
        print(str(error), file=sys.stderr)
        return 3
    if snap.get('exercise') == 'Walking Lunge (dumbbells optional)' and (snap.get('motion') or {}).get('contactModel') == 'named_heel_toe':
        # New one-shot snapshots retain their source-video timestamp name.
        # Normalize only the derived read input; never rewrite user metadata.
        images = [{**image, 'source': {**image.get('source', {}),
            'sequenceInfo': {**snap['motion'], **((image.get('source') or {}).get('sequenceInfo') or {})},
            'timeSeconds': (image.get('source') or {}).get('timeSeconds', (image.get('source') or {}).get('sourceTimestampSeconds'))}}
            for image in images]
    # These are derived from this read. Never leave stale correction inputs after a failed retry.
    run_file = run_dir / "run.json"
    run = json.loads(run_file.read_text()) if run_file.exists() else {}
    if snap.get('motionPreset') is not None:
        plan = preset_plan(snap)
        (run_dir / 'preflight.json').write_text(json.dumps(plan, indent=2))
        (run_dir / 'measurements.json').write_text('{}\n')
        if not plan['ready']:
            print(' '.join(plan['blockers']), file=sys.stderr)
            return 3
        preset = preset_for(snap['exercise'], snap['motionPreset'])
        targets = preset_targets(snap, preset)
        (run_dir / 'targets.json').write_text(json.dumps(targets, indent=2))
        (run_dir / 'keyposes.json').write_text(json.dumps({'exercise': snap['exercise'],
            'motionPreset': preset['id'], 'phases': [], 'motion': targets['motion']}, indent=2))
        print(f"Movement source: {preset['label']}. Photographs remain context; every rendered build is checked.")
        return 0
    cache_dir = run_dir.parent / ".pose-cache"
    measured, out = {}, {}
    for n, img in enumerate(images, 1):
        if img.get("deleted") or img.get("contextOnly"):
            continue
        print(f"image {n}/{len(images)}", flush=True)
        try:
            measured[img["id"]] = out[img["id"]] = read_cached(Path(img["path"]), read_pose, cache_dir)
        except NoBody as e:
            measured[img["id"]] = e
            out[img["id"]] = {"error": "No body found in this image", "crop": None}
        except Exception as e:  # one unreadable image never stops the run
            print(f"{img.get('original') or img['id']}: {e!r}", file=sys.stderr, flush=True)
            measured[img["id"]] = e
            out[img["id"]] = {"error": "Could not read this image"}
    (run_dir / "measurements.json").write_text(json.dumps(out, indent=2))
    try:
        targets = build_targets(snap["exercise"], images, measured, snap.get("equipment"))
    except TooFewPositions as e:
        # this run's own reasons per picture, so the page never shows an older run's beside them
        (run_dir / "read-problems.json").write_text(json.dumps(
            {"exercise": snap["exercise"], "message": str(e), "images": e.images}, indent=2))
        (run_dir / "preflight.json").write_text(json.dumps(
            {"ready": False, "mode": "Reference check", "blockers": [str(e)], "warnings": []}, indent=2))
        print(str(e), file=sys.stderr)   # the last line: the run's failure message names each picture
        return 3
    # the exercise's equipment word (the Studio's, from the run's snapshot): for an exercise added in the Studio, the
    # checks go by it (studio/measure/kit.py)
    targets["equipment"] = snap.get("equipment")
    (run_dir / "targets.json").write_text(json.dumps(targets, indent=2))
    pt = None
    if snap["exercise"] in SOLVER_EXERCISES:
        pt = to_pipeline_targets(targets)
        if pt is not None:
            (run_dir / "pipeline-targets.json").write_text(json.dumps(pt, indent=2))
    keys = key_positions(targets, measured, images)
    if keys is not None:
        (run_dir / "keyposes.json").write_text(json.dumps(keys, indent=1))
    sequence = (targets.get('motion') or {}).get('mode') == 'sequence'
    # A chronological stepping sequence must never fall back to the old cyclic
    # FBX or the two-endpoint forward/reverse image path.
    plan = plan_for(snap, targets, pt, keys, clip=None if sequence else run.get("clip"))
    if sequence:
        plan['mode'] = 'Chronological image sequence'
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'pipeline/blender'))
        from sequence_clip import SequenceError, inspect_sequence
        try:
            continuity = inspect_sequence(keys or {})
        except SequenceError as error:
            continuity = error.report
        (run_dir / 'sequence-continuity.json').write_text(json.dumps(continuity, indent=2))
        if snap['exercise'] == 'Walking Lunge (dumbbells optional)' and continuity.get('contactModel') == 'named_heel_toe' and continuity.get('input_ready'):
            plan['blockers'] = []
            plan['ready'] = True
            plan['warnings'].append('One observed step, not a closed loop. Frontal depth remains inferred; actual heel/toe contacts and root motion must pass before rendering.')
        else:
            plan['blockers'].append('This walking sequence needs observed heel/toe handoffs and forward root-motion validation before rendering.')
            plan['blockers'].extend(continuity.get('blockers') or [])
            plan['ready'] = False
    if snap.get('referenceBundle') is not None:
        plan['referenceBundle'] = snap['referenceBundle']
    (run_dir / "preflight.json").write_text(json.dumps(plan, indent=2))
    if not plan["ready"]:
        print(" ".join(plan["blockers"]), file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
