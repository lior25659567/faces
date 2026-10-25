"""Compare the animated character with the coach's targets, and check the body itself.

Every status is computed from the run's own measurements. Never show a pass that is not one.
"""
from build_targets import (ANGLE_METRICS, DEPTH_METRICS, FRONT_VIEWS, PHASES, POSTURE_METRICS, REFERENCE_EXERCISES, RULES, SOLVER_EXERCISES,
                           TOLERANCE, WIDTH_METRICS)
from feasibility import closest_arm_pose
from kit import has_depth, kit_of, unheld, unheld_note
from picture import PICTURE_METRICS, REPORTED_METRICS, SEEN_MIN, picture_value
from posture import NEUTRAL_EAR_NOSE_DEG, classify_back

ARM_FEASIBILITY = ("Barbell Back Squat",)
ARM_METRICS = ("elbow_included", "upperarm_vs_torso", "grip_ratio", "elbow_width_ratio")
STATUSES = ("pass", "fail", "limited", "uncertain", "info")
SEVERITY = {"fail": 4, "limited": 3, "uncertain": 2, "pass": 1, "info": 0}
# Every info row says why it has no verdict (`kind`; judged rows have none):
#   not-measured: it would be judged, but the character wasn't measured there (ours is null)
#   not-judged:   the reason is on the coach's side (unreadable, no picture, estimated) or the row can't
#                 be judged by design (folds hidden from every view)
#   shown-only:   never judged (face direction, skin stretch, the reading that found a moment)
NOT_MEASURED, NOT_JUDGED, SHOWN_ONLY = "not-measured", "not-judged", "shown-only"
# Metrics only ever shown: without a reading their row is still "shown-only", never "not-measured", so it
# doesn't hold an exercise back (summary `unmeasured`) and its chip agrees with the header (task 18).
SHOWN_ONLY_METRICS = ("head_pitch", "skin_stretch")
# Exercises whose character holds a bar (pipeline/blender/animate.py: the form-target solver and
# MPFB_BARBELL_ON_BACK): their hands_on_bar row is there even when the contact wasn't measured.
BAR_EXERCISES = SOLVER_EXERCISES + ("Barbell Bent-Over Row",)
NOT_IN_COACH_PICTURE = "Not judged: not measured in the coach's picture"

CRACK_MAX_MM = 1.0
FOOT_SLIDE_MAX_MM = 5.0
FEET_IN_FLOOR_MAX_MM = 5.0
BAR_DEPTH_MAX_MM = 5.0
BAR_TOUCH_MIN = 0.80


LENGTH_UNITS = {"hip_below_knee": "torso", "seat_below_knee": "thigh"}     # depths: in torso or thigh lengths


def _unit(metric):
    if metric in WIDTH_METRICS:
        return "x"
    return LENGTH_UNITS.get(metric, "deg")


def _tol(metric, tolerance):
    """The metric's tolerance from the targets', else today's (a targets.json read before a metric existed)."""
    key = "ratios" if metric in WIDTH_METRICS else metric if metric in LENGTH_UNITS else "angles_deg"
    return tolerance.get(key, TOLERANCE[key])


def _fmt(metric, v):
    return f"{v:.2f}×" if metric in WIDTH_METRICS else (f"{v:.2f}" if metric in LENGTH_UNITS else f"{v:.0f}°")


def _disagree_note(metric, t):
    if t.get('source') == 'inferred_3d' and t.get('cross') is None:
        return 'The reference depth is inferred from one photograph; a calibrated side view is needed to judge this angle.'
    other = "3d" if t["method"] == "flat" else "flat"
    names = {"flat": "flat", "3d": "in 3D"}
    return (f"Two readings of the coach disagree: {_fmt(metric, t['value'])} {names[t['method']]}, "
            f"{_fmt(metric, t['cross'])} {names[other]}")


def _num(v):
    """A real, finite number or None (NaN, inf, text and missing all mean "not measured")."""
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or v in (float("inf"), float("-inf")):
        return None
    return v


def _row(group, metric, target, ours, status, frame=None, note=None, unit=None, kind=None, uncertain_why=None):
    """One check row. An uncertain row says whose two readings disagree (`uncertain_why`): the coach's (flat
    and 3-D, the default) or ours (our picture and our rig, task 19); None on every other row."""
    if (status == "info") != (kind is not None):
        raise ValueError(f"{group} {metric}: an info row needs a kind, a judged row none")
    return {"group": group, "metric": metric, "target": _num(target), "ours": _num(ours), "unit": unit or _unit(metric),
            "status": status, "frame": frame, "note": note, "kind": kind,
            "uncertain_why": (uncertain_why or "coach") if status == "uncertain" else None}


def _info(group, metric, target, ours, kind, note, frame=None, unit=None):
    return _row(group, metric, target, ours, "info", frame, note, unit, kind)


def _ours_value(ours, frame, method, metric):
    for f in ours.get("per_frame") or []:
        if f.get("frame") == frame:
            return _num((f.get(method) or {}).get(metric))
    return None


def _want(targets, phase):
    p, w = targets["phases"][phase], targets["widths"]
    try:
        return {"upperarm_vs_torso": p["upperarm_vs_torso"]["value"], "elbow_included": p["elbow_included"]["value"],
                "grip_ratio": w["grip_ratio"]["value"], "elbow_width_ratio": w["elbow_width_ratio"]["value"]}
    except KeyError:
        return None


HUMAN_ARM = {"shoulder_half_width": 0.19, "upper_arm": 0.30, "forearm": 0.27}


def human_arm_like(arm):
    """Human proportions scaled to this character's shoulder width."""
    k = arm["shoulder_half_width"] / HUMAN_ARM["shoulder_half_width"]
    return {n: v * k for n, v in HUMAN_ARM.items()}


def _fmt_arm(metric, v):
    return f"{v:.2f}×" if metric in WIDTH_METRICS else f"{v:.0f}°"


def _judge(exercise, arm, want, metric, target, mine, tol):
    """-> (status, note, miss). 'limited' only when a human-proportioned arm meets the target, this
    character's cannot, and we got as close as it can."""
    miss = abs(mine - target)
    if miss <= tol:
        return "pass", None, miss
    off = (f"Off by {miss:.2f}×" if metric in WIDTH_METRICS else
           f"Off by {_amount(miss, tol, 2, f' {LENGTH_UNITS[metric]} lengths')}" if metric in LENGTH_UNITS else
           f"Off by {miss:.1f}°")
    if exercise in ARM_FEASIBILITY and metric in ARM_METRICS and arm and want:
        try:
            char = closest_arm_pose(arm, want)[metric]
            human = closest_arm_pose(human_arm_like(arm), want)[metric]
        except (KeyError, TypeError, ZeroDivisionError):
            return "fail", off, miss
        if abs(human - target) <= tol:
            if abs(char - target) > tol and abs(mine - char) <= tol:
                return ("limited", f"This character's arms reach at most {_fmt_arm(metric, char)}; "
                        f"a human-proportioned arm reaches {_fmt_arm(metric, target)}.", miss)
        elif abs(char - target) > tol:
            # The coach's own arm numbers don't fit one real arm (a human arm misses this one too), so no
            # character could meet it without breaking the others. That is the coach's reading — arms
            # foreshortened and part-hidden in a side photo — not a fault in ours: uncertain, not a fail.
            return ("uncertain", f"The coach's arm numbers don't all fit together in the arm model "
                    f"(a human arm reaches {_fmt_arm(metric, human)}): this reading can't be met without "
                    f"breaking the others, so it's read from the coach's picture, not judged on the character.",
                    miss)
    return "fail", off, miss


NOT_ON_CHARACTER = "Not measured on the character"


def _missing(group, metric, target, unit=None, frame=None):
    return _info(group, metric, target, None, NOT_MEASURED, NOT_ON_CHARACTER, frame, unit)


NO_PICTURE_HERE = "no picture of ours at this position"


def _joined(*parts):
    """Note sentences joined: "Off by 6.0°. Read from the picture" (no doubled full stop)."""
    parts = [p for p in parts if p]
    if not parts:
        return None
    out = parts[0]
    for p in parts[1:]:
        out += (" " if out.endswith(".") else ". ") + p
    return out


NEAR_LINE = 1.0         # degrees: a picture-rig gap this close to 2 x the tolerance is said to be near the line
ARMS_UNREAD = "the detector doesn't read the cartoon's arms reliably"
ARM_ANGLES = ("elbow_included", "upperarm_vs_torso")


def _knee_unseen(summary):
    """The picture's camera-side knee visibility when MediaPipe barely sees it (below picture.SEEN_MIN), else
    None: the hip angle and the knee angle run to that landmark."""
    v = ((summary or {}).get("visibility") or {}).get("knee") if isinstance(summary, dict) else None
    return v if _num(v) is not None and v < SEEN_MIN else None


def _picture_read(ours, phase, metric, method, rig, tol, target):
    """(ours, note, doubt) for one row, with ours read from the picture as the coach is (task 19,
    studio/measure/picture.py). Without a picture reading in ours.json (a run from before it), the rig's
    value and no note. REPORTED_METRICS: the rig's, the picture's reading in the note (with why it is not
    trusted: the cartoon's arms; a knee landmark barely seen). PICTURE_METRICS: the picture's value when
    MediaPipe read our picture, else the rig's, saying why. The rig is the cross-check: more than 2 x the
    tolerance apart (the coach's own rule for two readings that disagree), the row is uncertain (`doubt`
    True) only when the two could disagree about the verdict - they lie either side of the coach's value, or
    one of them is within the tolerance; both off by more than it on the same side, it is judged (a fail) with
    both numbers. A gap within NEAR_LINE of that line is said to be near it."""
    block = ours.get("picture")
    if not isinstance(block, dict) or metric not in PICTURE_METRICS + REPORTED_METRICS:
        return rig, None, False
    summary = block.get(phase)
    seen = picture_value(summary, method, metric)
    knee = _knee_unseen(summary)
    if metric in REPORTED_METRICS:
        if seen is None:
            return rig, None, False
        why = ARMS_UNREAD if metric in ARM_ANGLES else (
            f"its knee landmark is seen at {knee:.2f}" if metric == "knee_included" and knee is not None else None)
        return rig, f"The picture of ours reads {seen:.0f}° (shown only{'; ' + why if why else ''})", False
    if seen is None:
        why = (summary.get("why") if isinstance(summary, dict) else None) or NO_PICTURE_HERE
        if rig is None:         # neither reading: the row keeps the picture's reason
            return None, f"{NOT_ON_CHARACTER}: {why}", False
        return rig, f"Read on the rig: {why}", False
    knee_note = f"; the picture's knee landmark is seen at {knee:.2f}" if metric == "hip_included" and knee is not None else ""
    return _cross(seen, rig, tol, target, "the picture", knee_note)


def _degrees(v):
    return f"{v:.0f}°"


def _cross(seen, rig, tol, target, what="the picture", extra="", other="the rig", fmt=_degrees, near_line=NEAR_LINE):
    """(ours, note, doubt): `seen` (our picture's reading) judged with the rig's `rig` as its cross-check (task 19's
    rule, _picture_read): more than 2 x the tolerance apart, uncertain only when the two could disagree about the
    verdict, both off the same side a fail with both numbers. `what` names the reading in the notes, `other` the
    cross-check (written by `fmt`); a gap within `near_line` of the line is said to be near it."""
    if rig is None:
        return seen, f"Read from {what}, as the coach's; {other} wasn't measured here{extra}", False
    gap, line = abs(seen - rig), 2 * tol
    near = abs(gap - line) <= near_line
    if gap > line:
        both = f"{what[0].upper() + what[1:]} reads {fmt(seen)}, {other} reads {fmt(rig)}{extra}"
        off_s, off_r = (None, None) if target is None else (seen - target, rig - target)
        if target is None or off_s * off_r < 0 or abs(off_s) <= tol or abs(off_r) <= tol:
            return seen, _joined(both, "Close to the uncertain line; a later run may be judged" if near else None), True
        return seen, both, False        # both off on the same side: judged, a fail
    note = f"Read from {what}, as the coach's; {other} reads {fmt(rig)}{extra}"
    return seen, _joined(note, "Close to the uncertain line; a later run may read uncertain" if near else None), False


DEPTH_SHOWN = ("Shown only: the rig's thigh line against the coach's MediaPipe hip and knee, which do not sit alike; the "
               "depth is judged on the seat against the top of the knee")
LEG_ROWS = ("knee_included", "hip_included", "hip_below_knee")     # counted from the thigh's outline in the bind pose
SEAT_READ, SEAT_CROSS = "our picture", "the 3D skin"
SEAT_NEAR = 0.02        # thigh lengths: a picture-skin gap this close to 2 x the tolerance is said to be near the line
SEAT_ABOUT = ("The seat's lowest point against the top of whichever knee shows (the highest point of either leg within "
              "0.15 thigh lengths of the knee), in thigh lengths, read alike on both pictures; his thicker thigh and "
              "seat read a little deeper than ours at the same pose")
PREDATES = "This run predates the new depth reading; run it again"


def _seat_read(ours, phase, tol, target):
    """(ours, note, doubt) for the depth as a judge sees it (task 22): our side still's silhouette read as the coach's
    (ours.json `seat_below_knee` {phase: {"picture", "skin", "why", "skin_why"}}, posture.seat_depth), the 3D skin's
    reading of the same points at that frame the cross-check. Without a picture reading the skin's, saying why;
    without either, none, keeping the reasons; a run from before the reading (no block), none, saying so."""
    block = ours.get("seat_below_knee")
    block = block.get(phase) if isinstance(block, dict) else None
    if not isinstance(block, dict):
        return None, PREDATES, False
    seen, skin = _num(block.get("picture")), _num(block.get("skin"))
    if seen is None:
        why = block.get("why") or NO_PICTURE_HERE
        if skin is None:
            return None, f"{NOT_ON_CHARACTER}: {why}" + (f"; the 3D skin: {block['skin_why']}"
                                                         if block.get("skin_why") else ""), False
        return skin, f"Read on the 3D skin: {why}", False
    return _cross(seen, skin, tol, target, SEAT_READ, other=SEAT_CROSS, fmt=lambda v: f"{v:.2f}", near_line=SEAT_NEAR)


def _phase_rows(targets, ours, exercise):
    rows, tol, arm = [], targets["tolerance"], ours.get("arm")
    depth_judged = has_depth(exercise, targets.get("equipment"))
    for phase in PHASES:
        pt = targets["phases"].get(phase) or {}
        if not pt:
            continue
        frame = (ours.get("phase_frames") or {}).get(phase)
        # depth is read at the bottom (end) only, and only for an exercise whose kit row says so (kit.has_depth); every
        # angle at every position. A reading the coach's picture didn't give is a not-judged row, never a row that
        # silently isn't there.
        depth = DEPTH_METRICS if depth_judged and (phase == "end" or any(m in pt for m in DEPTH_METRICS)) else ()
        for metric in ANGLE_METRICS + depth:
            t = pt.get(metric)
            method = t.get("method") if isinstance(t, dict) else RULES[metric][0]
            rig = None if frame is None else _ours_value(ours, frame, method, metric)
            target = t.get("value") if isinstance(t, dict) else None
            if metric == "seat_below_knee":
                mine, said, doubt = _seat_read(ours, phase, _tol(metric, tol), _num(target))
                deeper = None if mine is None or _num(target) is None else mine - target
                if deeper is not None and deeper > _tol(metric, tol) / 2:   # the deep side said, the verdict unchanged
                    said = _joined(f"Deeper than the coach by {deeper:.2f} thigh lengths", said)
                if said != PREDATES and not (said or "").startswith(NOT_ON_CHARACTER):
                    said = _joined(said, SEAT_ABOUT)
            elif isinstance(t, dict) and t.get("source") == "inferred_3d":
                # An oblique photo's inferred depth cannot be checked against a new estimate
                # from a side render. Compare with the actual rig and disclose the reference limit.
                mine, said, doubt = rig, "Reference angle inferred from three-quarter depth; compared with the 3D rig", False
            else:
                mine, said, doubt = _picture_read(ours, phase, metric, method, rig, _tol(metric, tol), _num(target))
            if metric == "seat_below_knee" and not isinstance(t, dict):     # targets read before it existed
                rows.append(_info(phase, metric, None, mine, NOT_MEASURED, PREDATES, frame, _unit(metric)))
            elif not isinstance(t, dict) or _num(t.get("value")) is None:
                rows.append(_info(phase, metric, None, mine, NOT_JUDGED, NOT_IN_COACH_PICTURE, frame))
            elif metric == "hip_below_knee":
                rows.append(_info(phase, metric, t["value"], rig, SHOWN_ONLY, DEPTH_SHOWN, frame))
            elif phase == "middle" and metric == ours.get("phase_driver") and rig is not None:
                # the middle frame was found by matching this metric there, on the rig's reading: it can't be
                # judged, and the rig's value is the one shown (the picture's beside it)
                seen = picture_value((ours.get("picture") or {}).get(phase), method, metric) \
                    if isinstance(ours.get("picture"), dict) else None
                note = "Used to find this moment" + ("" if seen is None else f"; the picture reads {seen:.0f}°")
                rows.append(_info(phase, metric, t["value"], rig, SHOWN_ONLY, note, frame))
            elif pt.get("estimated"):
                note = "Estimated between the other two positions"
                skipped = ((ours.get('picture') or {}).get(phase) or {}).get('skipped')
                if skipped == 'estimated-reference-position' and metric in PICTURE_METRICS:
                    mine = rig
                    note += ". Shown on the rig; there is no reference picture at this position."
                rows.append(_info(phase, metric, t["value"], mine, NOT_JUDGED, note, frame))
            elif mine is None:      # a gap on our side, whether or not the coach's readings agree (and why, if known)
                known = said if said == PREDATES or (said or "").startswith(NOT_ON_CHARACTER) else NOT_ON_CHARACTER
                rows.append(_info(phase, metric, t["value"], None, NOT_MEASURED, known, frame))
            elif t["uncertain"]:
                rows.append(_row(phase, metric, t["value"], mine, "uncertain", frame,
                                 _joined(_disagree_note(metric, t), said)))
            elif doubt:
                rows.append(_row(phase, metric, t["value"], mine, "uncertain", frame, said, uncertain_why="ours"))
            else:
                st, note, _ = _judge(exercise, arm, _want(targets, phase), metric, t["value"], mine, _tol(metric, tol))
                if exercise == "Barbell Bent-Over Row" and phase == "end" and metric == "torso_lean":
                    said = _joined(said, "The row holds the starting torso hinge")
                rows.append(_row(phase, metric, t["value"], mine, st, frame, _joined(note, said)))
    warning = (ours.get("legs_at_rest") or {}).get("warning") if isinstance(ours.get("legs_at_rest"), dict) else None
    if warning:     # the bind pose's legs, which every leg reading of ours is counted from, may read off (task 21)
        for r in rows:
            if r["metric"] in LEG_ROWS:
                r["note"] = _joined(r["note"], warning)
    return rows


def _phase_elbow_width_rows(targets, ours, exercise):
    """Compare each photographed elbow spread with that phase's own width."""
    rows, metric = [], 'elbow_width_ratio'
    phase_widths = targets.get('phase_widths')
    phase_widths = phase_widths if isinstance(phase_widths, dict) else {}
    for phase in PHASES:
        if (targets['phases'].get(phase) or {}).get('estimated'):
            rows.append(_info(phase, metric, None, None, NOT_JUDGED, 'Not judged: this position is estimated'))
            continue
        target = (phase_widths.get(phase) or {}).get(metric)
        if not isinstance(target, dict) or _num(target.get('value')) is None:
            rows.append(_info(phase, metric, None, None, NOT_JUDGED,
                              'Not judged: no elbow-width measurement in this position\'s front or behind picture'))
            continue
        frame = (ours.get('phase_frames') or {}).get(phase)
        mine = None if frame is None else _ours_value(ours, frame, target['method'], metric)
        if mine is None:
            rows.append(_missing(phase, metric, target['value'], frame=frame))
        elif target.get('uncertain'):
            rows.append(_row(phase, metric, target['value'], mine, 'uncertain', frame, _disagree_note(metric, target)))
        else:
            status, note, _ = _judge(exercise, ours.get('arm'), None, metric, target['value'], mine,
                                     _tol(metric, targets['tolerance']))
            rows.append(_row(phase, metric, target['value'], mine, status, frame, note))
    return rows


def _width_rows(targets, ours, exercise):
    rows, tol, arm = [], targets["tolerance"], ours.get("arm")
    front = any(i.get("used") and i.get("view") in FRONT_VIEWS for i in (targets.get("images") or {}).values()
                if isinstance(i, dict))
    for metric in WIDTH_METRICS:
        if metric == 'elbow_width_ratio' and exercise in REFERENCE_EXERCISES and 'phase_widths' in targets:
            rows.extend(_phase_elbow_width_rows(targets, ours, exercise))
            continue
        t = (targets.get("widths") or {}).get(metric)
        if not isinstance(t, dict) or _num(t.get("value")) is None:
            why = "not measured in the front or behind picture" if front or t is not None else "no front or behind picture"
            rows.append(_info("all positions", metric, None, None, NOT_JUDGED, f"Not judged: {why}"))
            continue
        cands = []
        for phase in PHASES:
            frame = (ours.get("phase_frames") or {}).get(phase)
            mine = None if frame is None else _ours_value(ours, frame, t["method"], metric)
            if mine is None:
                continue
            if targets["phases"].get(phase, {}).get("estimated"):
                continue  # an estimated position has no real arm reading to be judged against
            st, note, miss = _judge(exercise, arm, _want(targets, phase), metric, t["value"], mine, _tol(metric, tol))
            cands.append((SEVERITY[st], miss, st, note, mine, frame))
        if not cands:
            rows.append(_missing("all positions", metric, t["value"]))   # ours wasn't read at any real position
            continue
        if t["uncertain"]:      # not judged, but ours is shown: the position furthest from the coach
            _, _, _, _, mine, frame = max(cands, key=lambda c: c[1])
            rows.append(_row("all positions", metric, t["value"], mine, "uncertain", frame, _disagree_note(metric, t)))
            continue
        _, _, st, note, mine, frame = max(cands)  # worst position is the one reported
        if len(cands) < len(PHASES):
            partial = f"Judged at {len(cands)} of {len(PHASES)} positions"
            note = f"{note}. {partial}" if note else partial
        rows.append(_row("all positions", metric, t["value"], mine, st, frame, note))
    return rows


def _body_rows(report, exercise, equipment=None):
    rows = []
    contact = report.get("contact") or {}
    deform = report.get("deformation") or {}

    crack = _num(deform.get("crack_max_mm"))
    if crack is None:
        rows.append(_missing("body", "cracks", CRACK_MAX_MM, "mm"))
    else:
        rows.append(_row("body", "cracks", CRACK_MAX_MM, crack, "pass" if crack <= CRACK_MAX_MM else "fail",
                         deform.get("deepest_frame"), None, "mm"))
    for side in ("left", "right"):
        s = (contact.get("sides") or {}).get(side) or {}
        v = _num(s.get("sole_slide_mm"))
        name = f"foot_slide_{side}"
        if report.get('feet') == 'free':
            rows.append(_info('body', name, None, v, NOT_JUDGED, 'This movement does not lock its feet to the floor.', unit='mm'))
        elif v is None:
            rows.append(_missing("body", name, FOOT_SLIDE_MAX_MM, "mm"))
        else:
            rows.append(_row("body", name, FOOT_SLIDE_MAX_MM, v, "pass" if v <= FOOT_SLIDE_MAX_MM else "fail",
                             s.get("slide_frame"), None, "mm"))
        if report.get('feet') == 'planted' and 'initial_floor_gap_mm' in s:
            gap = _num(s.get('initial_floor_gap_mm'))
            rows.append(_missing('body', 'foot_floor_gap_' + side, 5., 'mm') if gap is None else
                        _row('body', 'foot_floor_gap_' + side, 5., gap, 'pass' if gap <= 5. else 'fail', unit='mm'))
    pen = _num(contact.get("penetration_mm"))
    if pen is None:
        rows.append(_missing("body", "feet_in_floor", FEET_IN_FLOOR_MAX_MM, "mm"))
    else:
        rows.append(_row("body", "feet_in_floor", FEET_IN_FLOOR_MAX_MM, pen,
                         "pass" if pen <= FEET_IN_FLOOR_MAX_MM else "fail", contact.get("penetration_frame"), None, "mm"))
    hands = ((report.get("form") or {}).get("contact") or (report.get("grip") or {}).get("contact"))
    if hands and exercise == "Barbell Bent-Over Row":
        # A combined minimum can hide a floating hand. A row needs independent
        # finite-shaft measurements on both hands for the same complete animation.
        sides = hands.get("sides") or {}
        n = _num(hands.get("frames"))
        reads = [sides.get(side) or {} for side in ("left", "right")]
        if not n or any(r.get("frames") != n or _num(r.get("deepest_inside_bar_mm")) is None
                        or _num(r.get("frames_touching_within_3mm")) is None for r in reads):
            hands = None
        else:
            hands = {**hands, "deepest_inside_bar_mm": max(r["deepest_inside_bar_mm"] for r in reads),
                     "frames_touching_within_3mm": min(r["frames_touching_within_3mm"] for r in reads)}
    if not hands and exercise in BAR_EXERCISES:
        rows.append(_missing("body", "hands_on_bar", BAR_DEPTH_MAX_MM, "mm"))
    elif hands and not (_num(hands.get("deepest_inside_bar_mm")) is not None
                      and _num(hands.get("frames_touching_within_3mm")) is not None and _num(hands.get("frames"))):
        rows.append(_missing("body", "hands_on_bar", BAR_DEPTH_MAX_MM, "mm"))
    elif hands:
        depth, touch, n = hands["deepest_inside_bar_mm"], hands["frames_touching_within_3mm"], hands["frames"]
        share = touch / n
        ok = depth <= BAR_DEPTH_MAX_MM and share >= BAR_TOUCH_MIN
        rows.append(_row("body", "hands_on_bar", BAR_DEPTH_MAX_MM, depth, "pass" if ok else "fail", None,
                         f"Touching the bar in {touch} of {n} frames", "mm"))
    # the bar, dumbbells or handle the hands should hold, when the animation path doesn't build it (kit.py: every
    # exercise but the back squat): not measured, so the exercise can never show green with empty hands (task 9 review)
    thing = unheld(exercise, equipment)
    if thing:
        rows.append(_info("body", "equipment", None, None, NOT_MEASURED, unheld_note(thing)))
    growth = _num((deform.get("deepest") or {}).get("max_growth_pct"))
    if growth is not None:
        rows.append(_info("body", "skin_stretch", None, growth, SHOWN_ONLY, "Measured only, no pass or fail",
                          deform.get("deepest_frame"), "%"))
    else:       # shown only, read or not: never the amber "Not measured" the header doesn't count
        rows.append(_info("body", "skin_stretch", None, None, SHOWN_ONLY, NOT_ON_CHARACTER, None, "%"))
    return rows


JUMP_FRAMES_SHOWN = 8


def _records(v):
    return isinstance(v, list) and all(isinstance(e, dict) for e in v)


def _motion_events(jumps, steps):
    """[(frame, mm or None)] to name: every step (where the pose snaps, frame = the later of the two), plus
    the jump frames of each run of consecutive jump frames that no step enters (a step already names it)."""
    def jump_mm(j):
        return max((v for v in (_num(j.get("hips_mm")), _num(j.get("knees_mm"))) if v is not None), default=None)

    events = {s.get("frame"): _num(s.get("mm")) for s in steps}
    at = {j.get("frame"): jump_mm(j) for j in jumps}
    for f in at:
        first = f
        while isinstance(first, int) and first - 1 in at:
            first -= 1
        if first not in events:
            events.setdefault(f, at[f])
    return sorted(events.items(), key=lambda e: (e[0] is None, e[0] if isinstance(e[0], int) else 0))


def _frame_list(frames):
    if len(frames) > JUMP_FRAMES_SHOWN:
        return ", ".join(frames[:JUMP_FRAMES_SHOWN]) + " …"
    return frames[0] if len(frames) == 1 else ", ".join(frames[:-1]) + " and " + frames[-1]


def _motion_rows(ours):
    """Smooth motion (ours.json `jumps` and `steps`, written by studio/blender/inspect_run.py from
    metrics.jump_frames / jump_steps): pass only when both were measured and are empty; fail when the
    pose snaps (a step) or pops (a jump); otherwise not measured."""
    jumps, steps = ours.get("jumps"), ours.get("steps")
    if not _records(jumps):
        return [_missing("body", "smooth_motion", 0, "frames")]
    measured_steps = _records(steps)
    events = _motion_events(jumps, steps if measured_steps else [])
    if not events:
        if not measured_steps:
            return [_missing("body", "smooth_motion", 0, "frames")]
        return [_row("body", "smooth_motion", 0, 0, "pass", None, None, "frames")]
    frames = [str(f) for f, _ in events]
    sizes = [mm for _, mm in events if mm is not None]
    note = ("Jump on frame " if len(frames) == 1 else "Jumps on frames ") + _frame_list(frames)
    if sizes:
        note += f" ({max(sizes):.0f} mm)" if len({round(v) for v in sizes}) == 1 else f" (up to {max(sizes):.0f} mm)"
    worst = max(events, key=lambda e: e[1] or 0)[0]
    return [_row("body", "smooth_motion", 0, len(events), "fail", worst, note, "frames")]


# Skin turned over (ours.json `folds`, from studio/blender/inspect_run.py): one record per patch over
# the limit per frame, of kind "crease" (inside a hinge bent past 60 degrees, see creases.py:
# normal in deep bends, never fails), "defect" (a camera the user looks at sees at least the limit's
# area of it: fails) or "hidden" (no such camera sees that much: the row can't pass, so it is info).
FOLDED_SKIN_MAX_MM2 = 100.0     # for a character 1.0 m tall; scaled by (height / 1 m)²
FOLD_KINDS = ("crease", "defect", "hidden")
# the joint at a bone's head (where it meets its parent): inspect_run names each fold by the bone, of
# those carrying it, whose head is nearest the fold's centre. Skin only turns over where bones' weights
# blend, so a fold is always beside one of these joints.
_JOINT_AT = {"Arm": "shoulder", "ForeArm": "elbow", "Hand": "wrist", "UpLeg": "hip", "Leg": "knee",
             "Foot": "ankle", "ToeBase": "toes", "Toe_End": "toes", "Shoulder": "collarbone"}
_MIDDLE_AT = {"Hips": "the crotch", "Spine": "the waist", "Spine1": "the belly", "Spine2": "the upper torso",
              "Neck": "the neckline", "Head": "the neckline"}


def fold_limit(height):
    """The largest fold allowed (mm²) on a character this tall (m); None when the height is unknown."""
    h = _num(height)
    return round(FOLDED_SKIN_MAX_MM2 * h * h, 1) if h is not None and h > 0 else None


def fold_place(bone):
    """Plain words for where a fold is, from the bone named in ours.json `folds`."""
    if not isinstance(bone, str) or not bone:
        return "an unnamed place"
    name = bone.split(":")[-1]
    if name in _MIDDLE_AT:
        return _MIDDLE_AT[name]
    for side in ("Left", "Right"):
        if name.startswith(side):
            part = name[len(side):]
            if part in _JOINT_AT:
                return f"the {side.lower()} {_JOINT_AT[part]}"
            if part.startswith("Hand"):
                return f"the {side.lower()} hand"
    return bone


def _when(frames, first, last):
    frames = sorted(frames)
    total = last - first + 1 if isinstance(first, int) and isinstance(last, int) and last >= first else None
    if total and len(frames) == total:
        return "every frame"
    if len(frames) == 1:
        return f"frame {frames[0]}"
    if frames[-1] - frames[0] + 1 == len(frames):
        return f"frames {frames[0]}–{frames[-1]}"
    return f"{len(frames)} of {total} frames" if total else f"{len(frames)} frames"


def _and(parts):
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def _by_place(records):
    """[(place, largest area, frames)], largest first."""
    places = {}
    for frame, bone, area, *_ in records:
        p = places.setdefault(fold_place(bone), [0.0, set()])
        p[0] = max(p[0], area)
        p[1].add(frame)
    return sorted(((name, a, fr) for name, (a, fr) in places.items()), key=lambda p: -p[1])


def _sized_places(records, ours):
    return _and([f"{name} (up to {a:,.0f} mm², {_when(fr, ours.get('first'), ours.get('last'))})"
                 for name, a, fr in _by_place(records)])


def _joint_places(records):
    """Places with the two sides of a joint put together: "the knees", "the left elbow", "the hips"."""
    joints = {}
    for name, a, _ in _by_place(records):
        words = name.split(" ")
        side, joint = (words[1], " ".join(words[2:])) if words[1:2] in (["left"], ["right"]) else (None, name)
        j = joints.setdefault(joint, [a, set()])
        j[1].add(side)
    out = []
    for joint, (_, sides) in sorted(joints.items(), key=lambda kv: -kv[1][0]):
        if None in sides:
            out.append(joint)
        elif "both" in sides or sides == {"left", "right"}:
            out.append(f"the {joint if joint.endswith('s') else joint + 's'}")
        else:
            out.append(f"the {sides.pop()} {joint}")
    return _and(out)


def _fold_rows(ours):
    """Skin turned over: fail when a fold the user can see is over the limit, naming every such place
    and how big; info when the only folds over it are hidden from every view; pass when only creases
    of deep bends (or nothing) are, showing the crease the cameras see most; not measured without
    readable folds and a character height."""
    limit = fold_limit(ours.get("height_m"))
    folds = ours.get("folds")
    target = limit if limit is not None else FOLDED_SKIN_MAX_MM2
    if limit is None or not _records(folds):
        return [_missing("body", "folded_skin", target, "mm2")]
    kinds = {k: [] for k in FOLD_KINDS}
    unreadable = False
    for f in folds:
        frame, area, seen, kind = f.get("frame"), _num(f.get("area_mm2")), _num(f.get("visible_mm2")), f.get("kind")
        if (area is None or seen is None or kind not in kinds or isinstance(frame, bool)
                or not isinstance(frame, int)):
            unreadable = True
            continue
        kinds[kind].append((frame, f.get("bone"), area, seen))
    creases = f"creases at {_joint_places(kinds['crease'])}, which are normal in deep bends" if kinds["crease"] else None
    hidden = f"folds hidden from every view at {_joint_places(kinds['hidden'])}" if kinds["hidden"] else None
    if kinds["defect"]:
        frame, _, area, _ = max(kinds["defect"], key=lambda r: r[2])
        note = "Skin folds over at " + _sized_places(kinds["defect"], ours)
        extra = [x for x in (creases, hidden) if x]
        if extra:
            note += "; plus " + ", and ".join(extra)
        return [_row("body", "folded_skin", limit, area, "fail", frame, note, "mm2")]
    if unreadable:
        return [_missing("body", "folded_skin", limit, "mm2")]
    if kinds["hidden"]:
        frame, _, area, _ = max(kinds["hidden"], key=lambda r: r[2])
        note = "Folds hidden from every view at " + _sized_places(kinds["hidden"], ours)
        if creases:
            note += "; plus " + creases
        return [_info("body", "folded_skin", limit, area, NOT_JUDGED, note, frame, "mm2")]
    if not creases:
        return [_row("body", "folded_skin", limit, 0, "pass", None, None, "mm2")]
    # a crease is not judged; the one the cameras see most is shown, so the pass isn't a blank 0
    frame, bone, _, seen = max(kinds["crease"], key=lambda r: r[3])
    note = f"Creases only, at {_joint_places(kinds['crease'])}, which are normal in deep bends; " + (
        f"the most seen is {seen:,.0f} mm² at {fold_place(bone)}" if seen > 0 else "none is seen")
    return [_row("body", "folded_skin", limit, seen, "pass", frame if seen > 0 else None, note, "mm2")]


# Posture (studio/measure/posture.py; targets from build_targets), per position.
# neck_vs_torso: the coach's own value is the target, judged within its tolerance. Both are read the same
#   way: hip joints -> shoulder joints against shoulder joints -> the ears (the coach's MediaPipe ears, ours
#   found on the skin: inspect_run.ear_vertices), absolute; our torso line counted from the rig's upright
#   bind pose, as all our torso-based angles are (metrics.rest_turn, task 18).
# head_pitch: shown, never judged (the coach's reading rests on an assumed ear->nose offset).
# back_rounding: our back is its spine's curl from its rest pose (ours.json flat spine_curl, degrees; within
#   the spine_curl tolerance it is straight). The coach's back is only classified from his silhouette
#   (posture.classify_back): straight -> ours must be straight; rounded/arched -> ours must curl the same
#   way about as much; unreadable or can't tell -> ours is judged against straight when the exercise is
#   in STRAIGHT_BACK (its form notes ask for a straight back), otherwise not judged, with the plain reason.
# The exercises whose form notes (reference/exercises/form-guidelines.json) clearly ask for a neutral,
# unrounded back, with the cue quoted verbatim (tests/test_checks.py checks the quotes). When the coach's
# back can't be read, ours is judged against straight only for these. Left out on purpose: Flat Barbell
# Bench Press (a deliberate arch: "natural lumbar arch - about a hand's thickness of gap") and Standing
# Barbell Overhead Press ("chest lifted by arching the upper back").
STRAIGHT_BACK = {
    "Barbell Back Squat": "SPINE: natural standing lordotic arch, neither rounded nor over-arched.",
    "Conventional Barbell Deadlift": ("Lower back is neutral throughout: the normal inward curve, with no rounding and "
                                      "no exaggerated arch."),
    "Barbell Bent-Over Row": "Lower back stays neutral with its natural standing arch",
    "Pull-Up": "TORSO: hanging vertically, spine long and neutral, chest lifted, abdominals braced",
    "Dumbbell Bicep Curl (standing, both arms)": "Spine neutral with its natural lumbar curve",
    "Tricep Pushdown (cable, bar or rope attachment)": "Spine neutral with a maintained natural lumbar arch",
    "Dumbbell Lateral Raise": ("Torso erect and vertical, abdominals and core braced ('bracing'), ribcage stacked "
                               "over the pelvis, natural lumbar curve."),
    "Dumbbell Shoulder Press (seated overhead press)": "Spine neutral with its natural lumbar curve and no excess arch.",
    "Walking Lunge (dumbbells optional)": ("Back straight, spine neutral, chest lifted, shoulders down and back, "
                                           "abdominals tight."),
    "Leg Press (seated machine)": "with no gap behind the lower back and no rounding",
    "Front Plank (forearm plank)": ("Pelvis neutral: no sag or arch in the lower back, and no hiking of the hips up "
                                    "into a tent."),
    "Lat Pulldown (seated, to the front)": ("Abdominals braced ('stiffened') to stabilize the spine, with the natural "
                                            "lumbar curve kept and no arching of the low back."),
}
JUDGED_STRAIGHT = "judged against a straight back, as the exercise's form notes require"


def straight_back_rule(exercise):
    """The quoted form-notes cue asking this exercise for a straight back (STRAIGHT_BACK), or None."""
    return STRAIGHT_BACK.get(exercise)


def _amount(v, tol, digits, unit):
    """|v| to `digits`, one more digit when rounding would show a miss as no bigger than the tolerance."""
    text = f"{abs(v):.{digits}f}"
    if abs(v) > tol and float(text) <= tol:
        text = f"{abs(v):.{digits + 1}f}"
    return text + unit


def _neck_judged(t, mine, tol):
    diff = mine - t["value"]
    if abs(diff) <= tol:
        return "pass", None
    word = "forward" if diff > 0 else "back"
    return "fail", f"Head {_amount(diff, tol, 0, '°')} further {word} than the coach"


def _pitch_note(target, mine):
    note = (f"Shown, not judged: the coach's reading assumes a level gaze puts the nose "
            f"{NEUTRAL_EAR_NOSE_DEG:.0f}° below the ear")
    diff = round(mine - target)
    if diff == 0:
        return note + ". Ours looks the same way as the coach"
    return note + f". Ours looks {abs(diff)}° further {'up' if diff > 0 else 'down'} than the coach"


def _curl_words(curl, tol, about=False):
    """"rounds 18°", "arches about 12°", "stays straight" for a spine curl (degrees)."""
    if abs(curl) <= tol:
        return "stays straight"
    return f"{'rounds' if curl > 0 else 'arches'} {'about ' if about else ''}{_amount(curl, tol, 0, '°')}"


def _unreadable(cause, t):
    """(why not judged, in plain words; the same as a clause after "The coach's back ...")."""
    seen, run = _num(t.get("seen")), _num(t.get("run"))
    if cause == "level":
        return "he's lying level", "can't be read in these photos (he's lying level)"
    if cause == "broken" and run is not None:
        return (f"the arms break his back up — the longest unbroken stretch in view is {run:.0%}",
                "isn't visible in one piece in these photos (the arms break it up)")
    if cause == "cant-tell":
        return ("his back reads neither clearly straight nor clearly rounded",
                "reads neither clearly straight nor clearly rounded in these photos")
    if seen is not None:
        return (f"only {seen:.0%} of his back is in view — the arms hide the rest",
                f"isn't visible in these photos (the arms hide {1 - seen:.0%})")
    return "his back can't be read in this photo", "can't be read in these photos"


def _back_row(phase, t, pt, ours, frame, tol, rule):
    """The back row: our spine's curl against the coach's classified back (see "Posture" above)."""
    mine = None if frame is None else _ours_value(ours, frame, "flat", "spine_curl")
    state = t.get("state")
    if state is None:       # targets written before the classification: classify the reading now
        c = classify_back(_num(t.get("value")), _num(t.get("seen")), t.get("span"))
        state, t = c["state"], {**t, **{k: c[k] for k in ("cause", "curl", "run")}}
    cause = t.get("cause") if state == "unreadable" else state
    if pt.get("estimated"):
        return _info(phase, "back_rounding", None, mine, NOT_JUDGED, "Estimated between the other two positions",
                     frame, "deg")
    if cause == "no-seen":
        return _info(phase, "back_rounding", None, mine, NOT_JUDGED,
                     "Not judged: the coach's reading doesn't say how much of his back was in view", frame, "deg")
    if state in ("rounded", "arched") and _num(t.get("curl")) is not None:
        target = t["curl"]
        if mine is None:
            return _missing(phase, "back_rounding", target, "deg", frame)
        note = f"His back {_curl_words(mine, tol)}; the coach's {_curl_words(target, tol, about=True)}"
        return _row(phase, "back_rounding", target, mine, "pass" if abs(mine - target) <= tol else "fail", frame,
                    note, "deg")
    if state == "straight":
        if mine is None:
            return _missing(phase, "back_rounding", 0, "deg", frame)
        if abs(mine) <= tol:
            return _row(phase, "back_rounding", 0, mine, "pass", frame, "Straight like the coach's", "deg")
        return _row(phase, "back_rounding", 0, mine, "fail", frame,
                    f"His back {_curl_words(mine, tol)} where the coach keeps it straight", "deg")
    reason, clause = _unreadable(cause, t)
    if not rule:
        return _info(phase, "back_rounding", None, mine, NOT_JUDGED, f"Not judged: {reason}", frame, "deg")
    if mine is None:
        return _missing(phase, "back_rounding", 0, "deg", frame)
    note = f"The coach's back {clause}; {JUDGED_STRAIGHT}"
    if abs(mine) <= tol:
        return _row(phase, "back_rounding", 0, mine, "pass", frame, note, "deg")
    return _row(phase, "back_rounding", 0, mine, "fail", frame, f"His back {_curl_words(mine, tol)}. {note}", "deg")


def _posture_rows(targets, ours, exercise):
    rows, tol = [], {**TOLERANCE, **targets.get("tolerance", {})}
    rule = straight_back_rule(exercise)
    for phase in PHASES:
        pt = targets["phases"].get(phase, {})
        frame = (ours.get("phase_frames") or {}).get(phase)
        for metric in POSTURE_METRICS:
            t = pt.get(metric)
            if not isinstance(t, dict):
                continue
            if metric == "back_rounding":
                rows.append(_back_row(phase, t, pt, ours, frame, tol["spine_curl"], rule))
                continue
            target = _num(t.get("value"))
            mine = None if frame is None else _ours_value(ours, frame, t.get("method", "flat"), metric)
            if target is None:
                rows.append(_info(phase, metric, None, mine, NOT_JUDGED, "Not measured in the coach's picture", frame))
            elif pt.get("estimated"):
                rows.append(_info(phase, metric, target, mine, NOT_JUDGED, "Estimated between the other two positions",
                                  frame))
            elif metric == "head_pitch":    # shown only, read or not: never counted as not measured
                rows.append(_info(phase, metric, target, mine, SHOWN_ONLY,
                                  NOT_ON_CHARACTER if mine is None else _pitch_note(target, mine), frame))
            elif mine is None:      # a gap on our side, whether or not the coach's readings agree
                rows.append(_missing(phase, metric, target, frame=frame))
            elif t.get("uncertain") and _num(t.get("cross")) is not None:
                rows.append(_row(phase, metric, target, mine, "uncertain", frame, _disagree_note(metric, t)))
            else:
                status, note = _neck_judged(t, mine, tol[metric])
                rows.append(_row(phase, metric, target, mine, status, frame, note))
    return rows


def unmeasured(rows):
    """Rows that would be judged but the character wasn't measured there: any one keeps the exercise from
    passing (the page shows "N not measured"). Exactly the rows whose chip says "Not measured": shown-only
    metrics (face direction, skin stretch) are never of that kind, read or not (task 18)."""
    return sum(1 for r in rows if r.get("kind") == NOT_MEASURED)


def build_checks(targets, ours, report, exercise):
    from physical_checks import rows_for
    rows = (_phase_rows(targets, ours, exercise) + _width_rows(targets, ours, exercise)
            + _posture_rows(targets, ours, exercise)
            + _body_rows(report, exercise, targets.get("equipment")) + _fold_rows(ours) + _motion_rows(ours)
            + rows_for(report))
    capabilities = kit_of(exercise)
    if capabilities.get('reference_motion'):
        physical = report.get('physical') or {}
        if capabilities.get('holds') and not physical.get('contacts'):
            rows.append(_missing('body', 'equipment_contact', 3., 'mm'))
        if (capabilities.get('support_built') or exercise == 'Front Plank (forearm plank)') and not physical.get('supports'):
            rows.append(_missing('body', 'body_support_contact', 5., 'mm'))
        if not physical.get('profile'):
            rows.append(_missing('body', 'physical_profile', None, ''))
    summary = {s: sum(1 for r in rows if r["status"] == s) for s in STATUSES}
    summary["unmeasured"] = unmeasured(rows)
    return {"summary": summary, "rows": rows}
