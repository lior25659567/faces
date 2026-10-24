"""Describe what the selected animation path can actually do before starting Blender."""
import math

from build_targets import FORM_KEYS, PHASES, ROW_EXERCISE, SOLVER_EXERCISES, WIDTH_METRICS
from kit import picture_refusal, unheld

LABELS = {"knee_included": "knee bend", "hip_included": "hip bend", "torso_lean": "torso lean",
          "elbow_included": "elbow bend", "upperarm_vs_torso": "upper-arm angle"}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


ROW_JOINTS = ("nose", "left_ear", "right_ear") + tuple(
    f"{side}_{joint}" for side in ("left", "right")
    for joint in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle", "heel", "foot_index"))


def _point(value, dimensions):
    return isinstance(value, (list, tuple)) and len(value) >= dimensions and all(finite(v) for v in value[:dimensions])


def row_key_problems(keys):
    """Catch unusable geometry before Blender; keep oblique directions and tags truthful."""
    phases = {entry.get("phase"): entry for entry in (keys or {}).get("phases", [])}
    problems = []
    start_view = (phases.get("start") or {}).get("view")
    end_view = (phases.get("end") or {}).get("view")
    if start_view in ("side", "three-quarter") and end_view in ("side", "three-quarter") and start_view != end_view:
        problems.append("Use the same view for the row's Start and End: both side or both three-quarter. "
                        "Keep each picture tagged as photographed; choose a matching reference instead of changing its tag.")
    middle_view = (phases.get("middle") or {}).get("view")
    if middle_view and start_view in ("side", "three-quarter") and middle_view != start_view:
        problems.append("Use the same view for the row's Middle as its Start and End; choose a matching reference.")
    for phase in ("start", "middle", "end"):
        entry = phases.get(phase)
        if phase == "middle" and not entry:
            continue
        if not entry:
            problems.append(f"The row needs a readable {phase} position from a side or three-quarter image.")
            continue
        world = entry.get("world") or {}
        missing = [name.replace("_", " ") for name in ROW_JOINTS if not _point(world.get(name), 3)]
        if missing:
            problems.append(f"The row's {phase} position has unreadable 3D landmarks: {', '.join(missing)}.")
        elif sum((world["left_hip"][n] - world["right_hip"][n]) ** 2 for n in (0, 2)) < 1e-12:
            problems.append(f"The row's {phase} hip landmarks overlap; use a clearer picture.")
        if entry.get("view") not in ("side", "three-quarter"):
            problems.append(f"The row's {phase} needs a side or three-quarter view, tagged as photographed.")
        if entry.get("camera_side") not in ("left", "right"):
            problems.append(f"The row's {phase} camera side could not be determined.")
        if entry.get("view") == "side":
            pixels = entry.get("pixels") or {}
            missing_pixels = [name.replace("_", " ") for name in ROW_JOINTS if not _point(pixels.get(name), 2)]
            if missing_pixels:
                problems.append(f"The row's {phase} image is missing readable pixel landmarks; check the references again.")
    return problems


def plan_for(snapshot, targets, pipeline_targets, keys, clip=None):
    exercise = snapshot["exercise"]
    solver = exercise in SOLVER_EXERCISES
    row = exercise == ROW_EXERCISE
    if row:
        clip = None
    mode = "Corrected reference poses" if solver else "Existing motion clip" if clip else "Image key poses"
    blockers, warnings = [], []
    if row:
        blockers.extend(row_key_problems(keys))
        warnings.append("The visible arm drives both sides of this bilateral row. The hidden arm's estimated depth is not used independently.")
        for phase in ("start", "end"):
            missing = [LABELS[k] for k in FORM_KEYS
                       if not finite((targets["phases"].get(phase, {}).get(k) or {}).get("value"))]
            if missing:
                blockers.append(f"The row's {phase} position is missing: {', '.join(missing)}. Use a clearer whole-body reference.")
        leans = [(targets["phases"].get(p, {}).get("torso_lean") or {}).get("value") for p in ("start", "end")]
        if all(finite(v) for v in leans) and abs(leans[1] - leans[0]) > 10:
            warnings.append(f"The row holds the starting torso hinge. These references differ by {abs(leans[1] - leans[0]):.0f} "
                            "degrees of torso lean, so the held torso will differ from the End reference.")
        if any(entry.get("view") == "three-quarter" for entry in (keys or {}).get("phases", [])):
            warnings.append("Three-quarter row references use estimated 3D depth. Their projected side angles are not "
                            "used as measurements. Check the rendered body and bar path against the photographs.")
    if solver and pipeline_targets is None:
        widths = targets.get("widths", {})
        if any(not finite((widths.get(k) or {}).get("value")) for k in WIDTH_METRICS):
            blockers.append("Add a front or rear reference with both shoulders, elbows and hands visible, "
                            "then tag its view. Squat correction needs the grip and elbow widths.")
        for phase in PHASES:
            missing = [LABELS[k] for k in FORM_KEYS
                       if not finite((targets["phases"].get(phase, {}).get(k) or {}).get("value"))]
            if missing:
                blockers.append(f"The {phase} position is missing: {', '.join(missing)}. "
                                "Use a clear side image with the whole body visible.")
        if not blockers:
            blockers.append("These references could not produce squat correction targets. Check their view and position tags.")
    if not clip:
        refusal = picture_refusal(exercise, snapshot.get("equipment"))
        if refusal:
            blockers.append(refusal)
        if keys is None and not row:
            blockers.append("A readable side image is needed for both Start and End before motion can be built.")
    elif not solver:
        warnings.append("This exercise uses an existing motion clip. Your pictures are compared with it, "
                        "but changing them does not correct the clip's movement.")
    held = unheld(exercise, snapshot.get("equipment"))
    if held:
        warnings.append(f"This exercise does not yet build its {held} or solve hand contact with it. "
                        "Rebuilding the same pictures will not fix that limitation.")
    if targets["phases"].get("middle", {}).get("estimated"):
        warnings.append("The middle position is estimated. Add an image from the same view at the middle of the movement "
                        "if the path between Start and End looks wrong.")
    uncertain = []
    for phase in PHASES:
        for metric, reading in targets["phases"].get(phase, {}).items():
            if isinstance(reading, dict) and reading.get("uncertain"):
                uncertain.append(f"{phase} {LABELS.get(metric, metric.replace('_', ' '))}")
    if uncertain:
        warnings.append("The image and estimated 3D pose disagree on " + ", ".join(uncertain) +
                        ". Check the detected joints and use a clear, level side view.")
    return {"ready": not blockers, "mode": mode, "blockers": blockers, "warnings": warnings}
