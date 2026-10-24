"""Describe what the selected animation path can actually do before starting Blender."""
import math

from build_targets import FORM_KEYS, PHASES, SOLVER_EXERCISES, WIDTH_METRICS
from kit import picture_refusal, unheld

LABELS = {"knee_included": "knee bend", "hip_included": "hip bend", "torso_lean": "torso lean",
          "elbow_included": "elbow bend", "upperarm_vs_torso": "upper-arm angle"}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def plan_for(snapshot, targets, pipeline_targets, keys, clip=None):
    exercise = snapshot["exercise"]
    solver = exercise in SOLVER_EXERCISES
    if exercise == "Barbell Bent-Over Row":
        clip = None
    mode = "Corrected reference poses" if solver else "Existing motion clip" if clip else "Image key poses"
    blockers, warnings = [], []
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
        if keys is None:
            blockers.append("A readable side image is needed for both Start and End before motion can be built.")
    elif not solver:
        warnings.append("This exercise uses an existing motion clip. Your pictures are compared with it, "
                        "but changing them does not correct the clip's movement.")
    held = unheld(exercise, snapshot.get("equipment"))
    if held:
        warnings.append(f"This exercise does not yet build its {held} or solve hand contact with it. "
                        "Rebuilding the same pictures will not fix that limitation.")
    if targets["phases"].get("middle", {}).get("estimated"):
        warnings.append("The middle position is estimated. Add a side image at the middle of the movement "
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
