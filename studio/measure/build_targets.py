"""Turn a run's measured reference images into the exercise's targets."""
import math

from kit import has_depth
from posture import classify_back

ANGLE_METRICS = ("knee_included", "hip_included", "torso_lean", "elbow_included", "upperarm_vs_torso")
WIDTH_METRICS = ("grip_ratio", "elbow_width_ratio")
# the coach's posture in each side view (studio/measure/posture.py). back_rounding and neck_vs_torso are
# judged; head_pitch is shown only: its coach reading rests on an assumed ear->nose offset
POSTURE_METRICS = ("back_rounding", "neck_vs_torso", "head_pitch")
# the bottom's depth, read at the end position only and only for an exercise whose kit row says so (kit.has_depth: the
# back squat): the seat's lowest point below the top of the knee, in thigh lengths, as a judge sees it (judged, task
# 22; posture.seat_depth) and the MediaPipe hip -> knee depth (shown only: our rig's line is not his landmarks')
DEPTH_METRICS = ("seat_below_knee", "hip_below_knee")
# metric -> (method that is the value, whether the other method is a cross-check)
RULES = {
    "knee_included": ("flat", True),
    "hip_included": ("flat", True),
    "torso_lean": ("flat", False),
    "elbow_included": ("3d", False),
    "upperarm_vs_torso": ("3d", False),
    "hip_below_knee": ("flat", False),
    "seat_below_knee": ("flat", False),
    "grip_ratio": ("flat", True),
    "elbow_width_ratio": ("flat", True),
    "back_rounding": ("flat", False),
    "neck_vs_torso": ("flat", True),
    "head_pitch": ("flat", False),
}
# neck_vs_torso in degrees, wider than the joint angles: the coach's ear is MediaPipe's landmark, ours the
# outer ear found on the skin (studio/blender/inspect_run.py ear_vertices)
# back_rounding: the coach's silhouette reading, only classified (posture.classify_back); spine_curl:
# degrees our spine may curl or arch from its rest pose and still count as straight (checks.py)
# seat_below_knee: thigh lengths. The coach's 17.png reads 0.646; our bottoms from the hips over the knees (0.507) to
# the old deep one (0.589) span 0.08 - a tenth of a thigh is about 4 cm on him, 8 degrees of hip angle on ours
TOLERANCE = {"angles_deg": 5, "ratios": 0.12, "hip_below_knee": 0.10, "seat_below_knee": 0.10, "back_rounding": 0.04,
             "neck_vs_torso": 10, "spine_curl": 10}
# a reading is uncertain when flat and 3D differ by more than twice the check's tolerance
CROSS_LIMIT = {"angle": 2 * TOLERANCE["angles_deg"], "ratio": 2 * TOLERANCE["ratios"],
               "neck": 2 * TOLERANCE["neck_vs_torso"]}
SOLVER_EXERCISES = ("Barbell Back Squat",)
ROW_EXERCISE = "Barbell Bent-Over Row"
HOLD_EXERCISE = "Front Plank (forearm plank)"
SEQUENCE_EXERCISE = "Walking Lunge (dumbbells optional)"
REFERENCE_EXERCISES = ("Conventional Barbell Deadlift", "Flat Barbell Bench Press",
    "Standing Barbell Overhead Press (Shoulder Press)", "Pull-Up", "Lat Pulldown (seated, to the front)",
    "Dumbbell Bicep Curl (standing, both arms)", "Dumbbell Lateral Raise",
    "Dumbbell Shoulder Press (seated overhead press)", HOLD_EXERCISE,
    "Tricep Pushdown (cable, bar or rope attachment)", "Leg Press (seated machine)")
PHASES = ("start", "middle", "end")
SIDE_VIEWS = ("side",)
FRONT_VIEWS = ("front", "behind", "frontal")
FORM_KEYS = ("knee_included", "hip_included", "torso_lean", "elbow_included", "upperarm_vs_torso")
# the coach's posture the solver aims the head at (pipeline/blender/mpfb_cage.py `_place_head`), per position
SOLVER_POSTURE = ("neck_vs_torso", "head_pitch")


class TooFewPositions(Exception):
    """Too few side positions to build targets. `images` holds each picture's reasons (as targets.json's
    `images` would), so a failed read can still show them beside the pictures."""

    def __init__(self, message, images=None):
        super().__init__(message)
        self.images = images or {}


VIEW_WORDS = {"front": "front", "behind": "behind", "frontal": "front or behind", "three-quarter": "three-quarter"}


def _kind(metric):
    if metric in WIDTH_METRICS:
        return "ratio"
    return "neck" if metric == "neck_vs_torso" else "angle"


def _reading(result, metric, image_id):
    method, has_cross = RULES[metric]
    value = result[method].get(metric)
    if value is None:
        return None
    cross = None
    uncertain = False
    if has_cross:
        other = "3d" if method == "flat" else "flat"
        cross = result[other].get(metric)
        if cross is not None:
            uncertain = abs(value - cross) > CROSS_LIMIT[_kind(metric)]
    return {"value": value, "method": method, "cross": cross, "uncertain": uncertain, "image": image_id}


def _oblique_reading(result, metric, image_id):
    """An inferred 3D angle, never the foreshortened angle of an oblique photograph.

    The flat value is not a cross-check here: it measures a different projection. Keep
    the source explicit so downstream reports can distinguish an inferred target.
    """
    value = (result.get("3d") or {}).get(metric)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        return None
    return {"value": value, "method": "3d", "cross": None, "uncertain": False,
            "image": image_id, "source": "inferred_3d"}


def _posture_reading(result, metric, image_id):
    """Like _reading, but a posture the picture cannot show (value None) is kept with no value, so the
    check can say the coach's picture does not show it; None only when the image was measured before
    posture was read (no key). back_rounding carries `seen` (the share of the 10-90% stretch of the
    back in view), `span` (the unbroken stretch read) and the coach's back classified from them
    (posture.classify_back: `state`, `cause`, `curl`, `excess`, `run`)."""
    if metric not in result["flat"]:
        return None
    reading = _reading(result, metric, image_id) or {"value": None, "method": RULES[metric][0], "cross": None,
                                                     "uncertain": False, "image": image_id}
    if metric == "back_rounding":
        flat = result["flat"]
        reading["seen"], reading["span"] = flat.get("back_seen"), flat.get("back_span")
        state = classify_back(reading["value"], reading["seen"], reading["span"], flat.get("back_why"),
                              TOLERANCE["back_rounding"])
        reading.update({k: state[k] for k in ("state", "cause", "curl", "excess", "run")})
    return reading


def _why_not(meta, result):
    if meta.get("contextOnly"):
        return "Context only"
    if isinstance(result, Exception):
        from read_pose import NoBody
        return "No body found in this image" if isinstance(result, NoBody) else "Could not read this image"
    if meta.get('motionOnly') or (meta.get('source') or {}).get('motionOnly'):
        return 'Motion sample'
    return None


def _problem(meta, result, view):
    """Plain words for a picture that gave no side position, naming it, for the read step's failure
    message; None when the user set it that way on purpose (context only, or a view they picked)."""
    name = meta.get("original") or meta.get("file") or meta["id"]
    if meta.get("contextOnly") or meta.get('motionOnly') or (meta.get('source') or {}).get('motionOnly'):
        return None
    if isinstance(result, Exception):
        from read_pose import NoBody
        return f"{name}: no body found" if isinstance(result, NoBody) else f"{name}: could not be read"
    if view in SIDE_VIEWS:
        return f"{name} is tagged {str(meta.get('phase')).capitalize()}, but another picture already gives that position"
    if meta.get("viewSetBy") == "user":
        return None
    if not view:
        return f"{name}: its view isn't known — set it to Side if it is one"
    return f"{name} was read as a {VIEW_WORDS.get(view, view)} view — set it to Side if it isn't"


def _mean(a, b):
    def half(u, v):
        return None if u is None or v is None else (u + v) / 2

    out = {}
    for k in a:
        if k not in b:
            continue
        x, y = a[k], b[k]
        if x["method"] != y["method"]:
            continue  # a projected side angle and inferred 3D angle are not interchangeable
        out[k] = {"value": half(x["value"], y["value"]), "method": x["method"], "cross": half(x["cross"], y["cross"]),
                  "uncertain": x["uncertain"] or y["uncertain"], "image": None}
    return out


def build_targets(exercise, images, measured, equipment=None):
    """The targets; `equipment`: the Studio's equipment word, for an exercise added there (kit.py)."""
    images = sorted((m for m in images if not m.get("deleted")), key=lambda m: m.get("order", 0))
    sequence = exercise == SEQUENCE_EXERCISE and any(
        isinstance((m.get('source') or {}).get('sequenceInfo'), dict) for m in images if not m.get('contextOnly'))
    depth = DEPTH_METRICS if has_depth(exercise, equipment) else ()
    info, phases, widths, phase_widths = {}, {}, {}, {}
    camera = None
    problems = []   # pictures that gave no side position, in plain words, for a failed read
    for m in images:
        i = m["id"]
        result = measured.get(i)
        if result is None:
            result = Exception("not measured")
        why = _why_not(m, result)
        view = m.get("view") or (None if isinstance(result, Exception) else result.get("view"))
        oblique_row = (exercise == ROW_EXERCISE and view == 'three-quarter') or ((exercise in REFERENCE_EXERCISES or sequence) and view in ('three-quarter', 'front', 'behind', 'frontal'))
        phase_view = view in SIDE_VIEWS or oblique_row
        if why or not phase_view or m.get("phase") in phases:
            problem = _problem(m, result, view)
            if problem:
                problems.append(problem)
        if why:
            info[i] = {"used": False, "why": why, "view": view}
            continue
        if phase_view:
            phase = m["phase"]
            if phase in phases:
                info[i] = {"used": False, "why": f"Another image already gives the {phase} position", "view": view}
                continue
            reader = _oblique_reading if oblique_row else _reading
            reads = {k: reader(result, k, i) for k in ANGLE_METRICS + (depth if phase == "end" else ())}
            if view in FRONT_VIEWS and (exercise in REFERENCE_EXERCISES or sequence) and not widths:
                widths.update({k: v for k in WIDTH_METRICS if (v := _reading(result, k, i)) is not None})
            if view in FRONT_VIEWS and exercise in REFERENCE_EXERCISES:
                # A pull-up/press changes elbow spread across its photographed
                # positions. That width belongs to this exact phase; grip span
                # remains a shared target, as does the legacy squat policy.
                elbow_width = _reading(result, 'elbow_width_ratio', i)
                phase_widths[phase] = {'elbow_width_ratio': elbow_width} if elbow_width else {}
            if oblique_row and exercise != ROW_EXERCISE:
                for reading in reads.values():
                    if reading:
                        reading['uncertain'] = True
            # the seat's depth kept without a value where his picture doesn't show it, as a posture is
            seat = ("seat_below_knee",) if phase == "end" and "seat_below_knee" in depth else ()
            # The silhouette and face direction of an oblique view do not give side posture.
            if not oblique_row:
                reads.update({k: _posture_reading(result, k, i) for k in POSTURE_METRICS + seat})
            phases[phase] = {k: v for k, v in reads.items() if v is not None}
            camera = camera or {"side": result["camera_side"], "facing": result["facing"]}
            info[i] = {"used": True, "why": None, "view": view}
        elif view in FRONT_VIEWS:
            if widths:
                info[i] = {"used": False, "why": "Another image already gives the widths", "view": view}
                continue
            reads = {k: _reading(result, k, i) for k in WIDTH_METRICS}
            widths.update({k: v for k, v in reads.items() if v is not None})
            info[i] = {"used": True, "why": None, "view": view}
        else:
            info[i] = {"used": False, "why": "This view gives no target (only side and front/behind views do)",
                       "view": view}
    source_phases = list(phases)
    hold = exercise == HOLD_EXERCISE
    if hold and len(phases) == 1:
        # One photographed hold, evaluated at two times. No new pose is inferred.
        pose = next(iter(phases.values()))
        phases = {'start': pose, 'end': pose}
    missing = [p for p in ("start", "end") if p not in phases]
    if missing:
        named = f" {'; '.join(problems)}." if problems else ""
        views = "side or three-quarter images" if exercise in (ROW_EXERCISE,) + REFERENCE_EXERCISES else "chronological sequence images" if sequence else "side-view images"
        raise TooFewPositions(f"Need a start and an end position from {views} with a body found; "
                              f"missing: {' and '.join(missing)}.{named}", info)
    if "middle" not in phases:
        mid = _mean(phases["start"], phases["end"])
        mid["estimated"] = True
        phases["middle"] = mid
    result = {"exercise": exercise, "phases": {p: phases[p] for p in PHASES}, "widths": widths,
              "tolerance": dict(TOLERANCE), "images": info, "camera": camera}
    if phase_widths:
        result['phase_widths'] = phase_widths
    if exercise in REFERENCE_EXERCISES or sequence:
        result['motion'] = {'mode': 'sequence' if sequence else 'hold' if hold else 'cycle', 'sourcePhases': source_phases,
                            'approximateDepth': any(i.get('used') and i.get('view') != 'side' for i in info.values())}
    return result


def _solver_posture(targets, names):
    """{phase: {"neck_vs_torso", "head_pitch"}} for the solver (one decimal), when every position has both
    with a value; else {} (the solver then keeps its neutral nod rather than aim the head at some
    positions only). The back is never aimed at: the solver keeps ours straight, judged from the rig."""
    out = {}
    for src, dst in names.items():
        ph = targets["phases"].get(src, {})
        values = {k: (ph.get(k) or {}).get("value") for k in SOLVER_POSTURE}
        if any(not isinstance(v, (int, float)) or isinstance(v, bool) or v != v for v in values.values()):
            return {}
        out[dst] = {k: round(float(v), 1) for k, v in values.items()}
    return out


def to_pipeline_targets(targets):
    names = {"start": "standing", "middle": "mid", "end": "bottom"}
    out_phases = {}
    for src, dst in names.items():
        ph = targets["phases"].get(src, {})
        if any(not isinstance((ph.get(k) or {}).get("value"), (int, float))
               or isinstance(ph[k]["value"], bool) or not math.isfinite(ph[k]["value"]) for k in FORM_KEYS):
            return None
        out_phases[dst] = {k: int(round(ph[k]["value"])) for k in FORM_KEYS}
    w = targets.get("widths", {})
    if any(not isinstance((w.get(k) or {}).get("value"), (int, float))
           or isinstance(w[k]["value"], bool) or not math.isfinite(w[k]["value"]) for k in WIDTH_METRICS):
        return None
    measure = {}
    posture = _solver_posture(targets, names)
    for dst, values in posture.items():
        out_phases[dst].update(values)
    for k in FORM_KEYS + (SOLVER_POSTURE if posture else ()):
        methods = {targets["phases"][src][k]["method"] for src in names}
        if len(methods) != 1:
            raise ValueError(f"{k} was measured with different methods in different phases: {sorted(methods)}")
        measure[k] = methods.pop()
    out = {"exercise": targets["exercise"], "phases": out_phases, "measure": measure,
           "arms_all_phases": {k: round(w[k]["value"], 2) for k in WIDTH_METRICS},
           "tolerance": {"angles_deg": 5, "ratios": 0.12}}
    side = (targets.get("camera") or {}).get("side")
    if side in ("left", "right"):     # the arm the Studio judges the upper arm and elbow on (its side camera's)
        out["camera_side"] = side
    return out
