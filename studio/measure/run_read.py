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

KEY_FIELDS = ("view", "facing", "camera_side", "size", "world", "pixels")


def key_positions(targets, measured):
    """keyposes.json's content: {"exercise", "phases": [{"phase", "image", "view", "facing", "camera_side",
    "size", "world", "pixels"}]} for the side pictures the targets were read from (`view` as the targets took
    it: the user's tag, else the detector's), in start, middle, end order
    (pipeline/blender/keypose_clip.py animates between them). A position the targets only estimated (no middle
    picture) is left out; None without a start and an end that carry world landmarks."""
    phases = []
    for phase in PHASES:
        readings = targets["phases"].get(phase) or {}
        image = next((r.get("image") for r in readings.values() if isinstance(r, dict) and r.get("image")), None)
        result = measured.get(image) if image else None
        if not isinstance(result, dict) or not result.get("world"):
            continue
        entry = {"phase": phase, "image": image, **{k: result.get(k) for k in KEY_FIELDS}}
        # the view the targets used: the user's tag when set (a picture the detector called three-quarter can be
        # set to Side), else the detector's
        entry["view"] = (targets.get("images", {}).get(image) or {}).get("view") or entry["view"]
        phases.append(entry)
    if {"start", "end"} - {p["phase"] for p in phases}:
        return None
    return {"exercise": targets["exercise"], "phases": phases}


def main(run_dir):
    run_dir = Path(run_dir)
    snap = json.loads((run_dir / "references.json").read_text())
    images = snap["images"]
    # These are derived from this read. Never leave stale correction inputs after a failed retry.
    for name in ("targets.json", "pipeline-targets.json", "keyposes.json", "preflight.json", "read-problems.json"):
        (run_dir / name).unlink(missing_ok=True)
    run_file = run_dir / "run.json"
    run = json.loads(run_file.read_text()) if run_file.exists() else {}
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
    keys = key_positions(targets, measured)
    if keys is not None:
        (run_dir / "keyposes.json").write_text(json.dumps(keys, indent=1))
    plan = plan_for(snap, targets, pt, keys, clip=run.get("clip"))
    (run_dir / "preflight.json").write_text(json.dumps(plan, indent=2))
    if not plan["ready"]:
        print(" ".join(plan["blockers"]), file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
