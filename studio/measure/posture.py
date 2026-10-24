"""Posture in a side view, measured the same way on the coach's photo and on our character.

numpy + math only, so it runs under Python 3.10 and Blender's 3.13. read_pose.py feeds it the coach's
MediaPipe landmarks and silhouette; studio/blender/inspect_run.py our rig's bones. Points are (x, y) in
the picture, y running down; the 3D forms take any axes.

The coach's back is read off his silhouette (back_profile) and only CLASSIFIED (classify_back:
straight, rounded, arched, can't tell, unreadable), counted from a human neutral on the same stretch
(back-calibration.json). Our back is judged from the rig instead (spine_bend: the side-seen bends of
the spine's bones from the rest pose), so the cartoon's build and clothes never enter it.

back_rounding  The back's silhouette edge (the side away from where the person faces) between the
               shoulders and the hips, read on lines square to the hip->shoulder chord from 10% to
               90% of the way up (the shoulder blade's and the buttocks' curves left out). Each
               line runs from the chord out to where the silhouette ends (a gap under GAP is a hole
               in the mask, not the outside). The edge's largest deviation from the straight line
               through the first and last of those edge points, signed (+ = bowed outward, a
               rounded/flexed back; - = drawn in, an arched one), over the torso length (shoulder
               midpoint to hip midpoint). Lines whose edge point lies within an arm's radius of its
               segments (shoulder->elbow, elbow->wrist, wrist->hand; ARM_RADIUS unless a segment
               carries its own), or on a mask of the arms alone when there is one (our character's,
               rendered), read the arm's outline, not the back: they are left out. `seen` is the
               share of the 10-90% stretch actually kept. The deviation is read on the longest
               unbroken stretch of kept lines (gaps up to MAX_GAP crossed; `span`, `run`) - never
               across an arm - and not at all when that is shorter than MIN_SPAN (`why` "hidden"),
               nor on a torso within 15 degrees of level (`why` "level"). A neutral human back is
               an S (the waist's hollow, the shoulder blades' curve), so a whole back reads about
               -0.096, not 0 (back-calibration.json).
neck_vs_torso  The angle between the torso line (hip midpoint -> shoulder midpoint) and the neck line
               (shoulder midpoint -> the ears: the coach's MediaPipe ear, ours found on the skin).
               0 = the head in line with the back, + = forward of it. None within 15 degrees of
               level (a plank: the picture's left/right gives "forward" no sign). Our torso line is
               counted from the rig's upright bind pose (`torso_turn`, metrics.rest_turn; task 18).
head_pitch     The face's direction against the horizontal, + = looking up.
seat_below_knee  The squat's depth the way a judge sees it (task 22): how far the seat's lowest point sits below
               the top of the knee, over the thigh's length (hip -> knee in the picture), + = the seat lower (deep).
               The seat: the silhouette's lowest point in the columns from the hip's back SEAT_BEHIND thigh
               lengths, walking down from the hip's height through the body (on both bodies it lies under the
               hip, on the shorts). The knee's top: the highest point of the silhouette in the columns within
               KNEE_WINDOW of the knee's, walking up from the knee's height through the body - the top of whichever
               knee shows, the far one where it rides higher - a column whose body reaches KNEE_TORSO above the
               knee being the torso, not a knee. One function for the coach's photo (MediaPipe's person mask and
               landmarks) and our side still (the mask render and our projected joints). Build enters it: the
               coach's thicker thigh and seat read a little deeper than our cartoon's at the same pose.
"""
import json
import math
from pathlib import Path

import numpy as np

SKIP = 0.10             # of the torso at each end: the shoulder blade's and the buttocks' curves
ARM_RADIUS = 0.25       # torso lengths: the coach's arm outline lies up to 0.21 from MediaPipe's arm
                        # segments (15-17.png; shoulder and elbow landmarks sit inside the deltoid and
                        # the elbow's point), so 0.25 leaves it all out
ARM_MASK_NEAR = 0.02    # torso lengths: an edge point this near the arms' own mask is on their outline
GAP = 0.03              # torso lengths: background narrower than this along a line is a hole, not the outside
MIN_SPAN = 0.15         # torso lengths: the shortest stretch of back worth reading a curve on
MAX_GAP = 0.06          # torso lengths: a break in the kept lines up to this (a touch of an arm and its
                        # ARM_MASK_NEAR margins) is crossed; a longer one ends the stretch
LEVEL_DEG = 15.0        # a torso within this of level (a plank) has no left/right back or neck side
MIN_SIDE = math.sin(math.radians(LEVEL_DEG))
STEP = 0.5              # pixels between the points read along each line
REACH = 1.5             # torso lengths: how far out along a line the edge is looked for
NEUTRAL_EAR_NOSE_DEG = 12.0   # a level-gazing head's ear->nose line points about this far below level
FACING = {"right": (1.0, 0.0), "left": (-1.0, 0.0)}
METRICS = ("back_rounding", "back_seen", "back_span", "back_why", "neck_vs_torso", "head_pitch", "spine_curl",
           "seat_below_knee")
SEAT_BEHIND = 0.8       # thigh lengths behind the hip's column: where the seat's lowest point is looked for
KNEE_WINDOW = 0.15      # thigh lengths either side of the knee's column: where a knee's top is looked for (either
                        # knee's: the coach's reads 0.636 / 0.646 / 0.661 at 0.10 / 0.15 / 0.20)
KNEE_TORSO = 1.0        # thigh lengths above the knee: a column whose body reaches this high is the torso, not a knee

# Classifying the coach's back (classify_back). The calibration: an MPFB human's back read by back_profile
# with its arms clear, at spine curls of -30..+45 degrees (back-calibration.json). Worst case over where
# a stretch sits, a 30-degree curl (3 x our 10-degree "straight" band) reads beyond BACK_TOL of the neutral
# only with 87.5% of the back in view (60%: even 45 degrees reads 0.028; 75%: +30 reads 0.023); the whole
# back cannot tell 15 degrees from straight (0.031). Below BACK_MIN_SEEN the coach's back is unreadable.
BACK_MIN_SEEN = 0.875
BACK_TOL = 0.04         # torso lengths from the neutral: within it straight; past twice it rounded or arched
_CAL = json.loads((Path(__file__).resolve().parent / "back-calibration.json").read_text())
_CAL_S = np.asarray(_CAL["s"], dtype=float)
_CAL_D = {float(curl): np.asarray(d, dtype=float) for curl, d in _CAL["profiles"].items()}


def _seg_distance(p, a, b):
    ab = b - a
    k = 0.0 if ab @ ab < 1e-12 else min(max(float((p - a) @ ab / (ab @ ab)), 0.0), 1.0)
    return float(np.linalg.norm(p - (a + k * ab)))


def _dilate(mask, r):
    """True within r pixels (a square) of a True pixel."""
    rows = mask.copy()
    for k in range(1, r + 1):
        rows[k:, :] |= mask[:-k, :]
        rows[:-k, :] |= mask[k:, :]
    out = rows.copy()
    for k in range(1, r + 1):
        out[:, k:] |= rows[:, :-k]
        out[:, :-k] |= rows[:, k:]
    return out


def _coverage(mask, x, y):
    """The mask at a point between pixel centres, bilinear; 0 outside the picture."""
    height, width = mask.shape
    x0, y0 = math.floor(x), math.floor(y)
    fx, fy = x - x0, y - y0
    total = 0.0
    for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xi, yi = x0 + dx, y0 + dy
        if w > 0 and 0 <= xi < width and 0 <= yi < height:
            total += w * float(mask[yi, xi])
    return total


def _edge(mask, start, direction, reach, gap):
    """Distance (px) from `start` to the silhouette's edge along `direction` (where the coverage, read
    bilinearly, falls through 0.5), or None when `start` is not inside it. Background runs no longer
    than `gap` px are crossed."""
    last = None             # (distance, coverage) of the last point inside
    crossing, outside, d = None, 0.0, 0.0
    while d <= reach:
        c = _coverage(mask, *(start + d * direction))
        if c >= 0.5:
            last, crossing, outside = (d, c), None, 0.0
        elif last is None:
            return None     # the line starts outside the body
        else:
            if crossing is None:    # where it fell through 0.5, between the last point inside and this one
                crossing = last[0] + STEP * (last[1] - 0.5) / max(last[1] - c, 1e-9)
            outside += STEP
            if outside > gap:
                break
        d += STEP
    return crossing if crossing is not None else last[0]


def seat_depth(mask, hip, knee, facing, window=KNEE_WINDOW):
    """{"depth": thigh lengths or None, "seat": (x, y) or None, "knee_top": (x, y) or None, "why": None or plain
    words} (see "seat_below_knee" above). mask: 2D, mask[y, x] 0-1 (0.5 and up is the body); hip, knee: (x, y) in the
    picture, y down (the coach's camera-side MediaPipe landmarks; our rig's joints projected); facing: "right" or
    "left", the way the body faces in the picture (the seat is behind the hip); window: thigh lengths either side of
    the knee's column a knee's top is looked for (KNEE_WINDOW; others only to see how much it matters)."""
    none = {"depth": None, "seat": None, "knee_top": None, "why": None}
    if facing not in FACING:
        return {**none, "why": "not a side view: which way the body faces is not known"}
    hip, knee = np.asarray(hip, dtype=float)[:2], np.asarray(knee, dtype=float)[:2]
    length = float(np.linalg.norm(knee - hip))
    if length < 1e-6:
        return {**none, "why": "no thigh: the hip and knee are on one point"}
    body = np.asarray(mask, dtype=float) >= 0.5
    height, width = body.shape
    seat, top = None, None
    row = int(hip[1])
    if 0 <= row < height:
        lo, hi = sorted((hip[0], hip[0] - FACING[facing][0] * SEAT_BEHIND * length))
        for x in range(max(0, int(lo)), min(width, int(hi) + 1)):
            column = body[:, x]
            if not column[row]:
                continue
            y = row
            while y + 1 < height and column[y + 1]:
                y += 1
            if seat is None or y > seat[1]:
                seat = (x, y)
    if seat is None:
        return {**none, "why": "no seat: no body under the hip behind it"}
    limit = int(knee[1] - KNEE_TORSO * length)
    first = max(limit, 0)
    for x in range(int(knee[0] - window * length), int(knee[0] + window * length) + 1):
        if not 0 <= x < width or int(knee[1]) < first:
            continue
        column = body[:, x]
        y = min(int(knee[1]), height - 1)
        if not column[y]:                       # the knee's row outside the body: from the nearest body above it
            above = np.nonzero(column[first:y])[0]
            if not len(above):
                continue
            y = first + int(above[-1])
        while y - 1 >= first and column[y - 1]:
            y -= 1
        if y == first:                          # the body reaches KNEE_TORSO up (or the picture's top): the torso
            continue
        if top is None or y < top[1]:
            top = (x, y)
    if top is None:
        return {**none, "seat": seat, "why": "no knee top: the body over the knee reaches up into the torso"}
    return {"depth": (seat[1] - top[1]) / length, "seat": seat, "knee_top": top, "why": None}


def back_profile(mask, shoulder, hip, facing, arms=(), arm_radius=ARM_RADIUS, arm_mask=None, span=None):
    """{"rounding": torso lengths or None, "seen": 0-1, "run": 0-1, "span": (lo, hi) or None, "why": None |
    "hidden" | "level", "samples": [...]}.

    mask: 2D, mask[y, x] True (or a coverage of 0.5 and up, of 0-1) on the body. shoulder, hip: the
    (x, y) midpoints. facing: "right" or "left", the side of the picture the face is on. arms: [(a, b)]
    or [(a, b, radius)] segments in pixels; arm_radius (torso lengths) for those without a radius of
    their own. arm_mask: the arms alone on the same picture (coverage 0-1); an edge point within
    ARM_MASK_NEAR of it is left out too. span: (lo, hi), read only that stretch of the torso (within the
    10-90% one; e.g. the stretch the coach's picture shows, to compare like with like).
    Each sample: {"s" (0 hip - 1 shoulder), "d" (torso lengths from the chord), "edge" (x, y), "clear"
    (how far the edge lies outside the nearest arm's radius, torso lengths; inf without arms), "on_arm"
    (on the arms' own mask), "kept"}. None without a facing or a torso."""
    if facing not in FACING:
        return None
    shoulder, hip = np.asarray(shoulder, dtype=float)[:2], np.asarray(hip, dtype=float)[:2]
    length = float(np.linalg.norm(shoulder - hip))
    if length < 1e-6:
        return None
    t = (shoulder - hip) / length
    n = np.array([-t[1], t[0]])
    side = float(n @ np.array(FACING[facing]))
    if abs(side) < MIN_SIDE:
        return {"rounding": None, "seen": 0.0, "run": 0.0, "span": None, "samples": [], "why": "level"}
    n = n if side < 0 else -n                       # toward the back
    mask = np.asarray(mask, dtype=float)
    arms = [(np.asarray(arm[0], dtype=float)[:2], np.asarray(arm[1], dtype=float)[:2],
             arm[2] if len(arm) > 2 else arm_radius) for arm in arms]
    covered = None
    if arm_mask is not None:
        covered = _dilate(np.asarray(arm_mask, dtype=float) >= 0.5, max(1, int(round(ARM_MASK_NEAR * length))))
    lo, hi = (SKIP, 1 - SKIP) if span is None else (max(SKIP, float(span[0])), min(1 - SKIP, float(span[1])))
    if hi <= lo:
        return {"rounding": None, "seen": 0.0, "run": 0.0, "span": None, "samples": [], "why": "hidden"}
    count = max(int(round((hi - lo) * length)), 2) + 1     # about one line per pixel of chord
    samples = []
    for s in np.linspace(lo, hi, count):
        start = hip + s * (shoulder - hip)
        d = _edge(mask, start, n, REACH * length, GAP * length)
        if d is None:
            continue
        edge = start + d * n
        # how far the edge lies outside the nearest arm's radius (torso lengths); inside it, the arm's outline
        clear = min((_seg_distance(edge, a, b) / length - r for a, b, r in arms), default=float("inf"))
        xi, yi = math.floor(edge[0] + 0.5), math.floor(edge[1] + 0.5)
        on_arm = bool(covered is not None and 0 <= yi < covered.shape[0] and 0 <= xi < covered.shape[1]
                      and covered[yi, xi])
        samples.append({"s": float(s), "d": d / length, "edge": (float(edge[0]), float(edge[1])),
                        "clear": clear, "on_arm": on_arm, "kept": clear >= 0 and not on_arm})
    return _read_back(samples, lo, hi, count)


def _deviation(s, d):
    """The largest deviation (signed) of d(s) from the straight line through its two ends."""
    s, d = np.asarray(s, dtype=float), np.asarray(d, dtype=float)
    dev = d - (d[0] + (d[-1] - d[0]) * (s - s[0]) / (s[-1] - s[0]))
    return float(dev[int(np.argmax(np.abs(dev)))])


def _read_back(samples, lo, hi, count):
    kept = [p for p in samples if p["kept"]]
    step = (hi - lo) / (count - 1)
    seen = min(1.0, len(kept) * step / (1 - 2 * SKIP))
    runs = []                                   # unbroken stretches of kept lines
    for p in kept:
        if runs and p["s"] - runs[-1][-1]["s"] <= MAX_GAP + 1e-9:
            runs[-1].append(p)
        else:
            runs.append([p])
    run = max(runs, key=lambda r: r[-1]["s"] - r[0]["s"], default=[])
    length = run[-1]["s"] - run[0]["s"] if run else 0.0
    out = {"rounding": None, "seen": seen, "run": length / (1 - 2 * SKIP), "span": None, "samples": samples,
           "why": "hidden"}
    if len(run) >= 3 and length >= MIN_SPAN:
        out.update(rounding=_deviation([p["s"] for p in run], [p["d"] for p in run]),
                   span=(run[0]["s"], run[-1]["s"]), why=None)
    elif run:
        out["span"] = (run[0]["s"], run[-1]["s"])
    return out


def neutral_rounding(span):
    """What a neutral human back reads on that stretch (the calibration's curl 0), torso lengths."""
    return _calibrated(0.0, span)


def _calibrated(curl, span):
    lo, hi = span
    inside = (_CAL_S >= lo - 1e-9) & (_CAL_S <= hi + 1e-9)
    if inside.sum() < 3:
        return None
    return _deviation(_CAL_S[inside], _CAL_D[curl][inside])


def curl_from_rounding(rounding, span):
    """The spine curl (degrees, + rounded) a back reading on that stretch means on the calibration
    human: interpolated between its curls, carried on straight past the ends. None off the grid."""
    pairs = sorted((_calibrated(c, span), c) for c in _CAL_D)
    if any(r is None for r, _ in pairs):
        return None
    xs, ys = [r for r, _ in pairs], [c for _, c in pairs]
    if rounding <= xs[0] or rounding >= xs[-1]:
        (x0, y0), (x1, y1) = (pairs[0], pairs[1]) if rounding <= xs[0] else (pairs[-2], pairs[-1])
        return float(y0 + (rounding - x0) * (y1 - y0) / (x1 - x0)) if abs(x1 - x0) > 1e-12 else float(y0)
    return float(np.interp(rounding, xs, ys))


def classify_back(rounding, seen, span, why=None, tol=BACK_TOL):
    """The coach's back from his silhouette reading: {"state": "straight" | "rounded" | "arched" |
    "cant-tell" | "unreadable", "cause" (why unreadable: "level", "no-seen", "hidden" - less than
    BACK_MIN_SEEN of it in view - or "broken" - in view, but no unbroken stretch that long), "curl"
    (degrees, rounded/arched only), "excess" (torso lengths past the neutral), "seen", "run"}. Counted
    from neutral_rounding on the same stretch: within tol straight, past 2 x tol rounded or arched, in
    between "cant-tell"."""
    run = None if span is None else (span[1] - span[0]) / (1 - 2 * SKIP)
    out = {"state": "unreadable", "cause": None, "curl": None, "excess": None, "seen": seen, "run": run}
    if why == "level":
        out["cause"] = "level"
    elif seen is None:
        out["cause"] = "no-seen"
    elif rounding is None or span is None or seen < BACK_MIN_SEEN:
        out["cause"] = "hidden"
    elif run < BACK_MIN_SEEN:
        out["cause"] = "broken"
    else:
        neutral = neutral_rounding(span)
        if neutral is None:
            out["cause"] = "hidden"
            return out
        excess = rounding - neutral
        out["excess"] = excess
        if abs(excess) <= tol:
            out["state"] = "straight"
        elif abs(excess) > 2 * tol:
            out["state"] = "rounded" if excess > 0 else "arched"
            out["curl"] = curl_from_rounding(rounding, span)
        else:
            out["state"] = "cant-tell"
    return out


def spine_bend(points, forward, up, normal=None):
    """Degrees the chain of points up the spine (hips first) bends in all, seen on the plane square to
    `normal`: the sum of the signed bends between consecutive segments, + when the upper one tips toward
    `forward` (curling, rounding), - away (arching)."""
    pts = [np.asarray(p, dtype=float) for p in points]
    flat = _flatten([b - a for a, b in zip(pts, pts[1:])] + [forward, up], normal)
    segments, forward, up = flat[:-2], flat[-2], flat[-1]
    up = up / np.linalg.norm(up)
    forward = forward - (forward @ up) * up
    forward = forward / np.linalg.norm(forward)
    tilt = [math.degrees(math.atan2(float(v @ forward), float(v @ up))) for v in segments]
    return float(sum((b - a + 180.0) % 360.0 - 180.0 for a, b in zip(tilt, tilt[1:])))


def back_rounding(mask, shoulder, hip, facing, arms=(), arm_radius=ARM_RADIUS, arm_mask=None, span=None):
    """The back's rounding in torso lengths (+ rounded, - arched), or None (see back_profile)."""
    profile = back_profile(mask, shoulder, hip, facing, arms, arm_radius, arm_mask, span)
    return None if profile is None else profile["rounding"]


def _flatten(vectors, normal):
    vectors = [np.asarray(v, dtype=float) for v in vectors]
    if normal is None:
        return vectors
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    return [v - (v @ n) * n for v in vectors]


def neck_vs_torso(hip, shoulder, head, forward, normal=None, up=None, torso_turn=None):
    """Degrees between the torso line (hip -> shoulder) and the neck line (shoulder -> head), + when the
    head is forward of the torso line (toward `forward`, the body's or face's front). With `normal`
    (3D) everything is first laid on the plane square to it, e.g. the body's side plane. None when
    either line has no length, or when the torso lies within LEVEL_DEG of level (`up` given: of square
    to it; otherwise of `forward`, a level front): there "forward of the back" has no sign.
    torso_turn (3D, a 3x3 rotation: metrics.rest_turn): our rig's torso line is turned by it first, as
    every torso-based metric of ours is (task 18); the neck line still runs from the shoulder joints."""
    torso = np.asarray(shoulder, float) - np.asarray(hip, float)
    if torso_turn is not None:
        torso = np.asarray(torso_turn, dtype=float) @ torso
    torso, neck, forward, level = _flatten([torso,
                                            np.asarray(head, float) - np.asarray(shoulder, float), forward,
                                            forward if up is None else up], normal)
    lt, ln = float(np.linalg.norm(torso)), float(np.linalg.norm(neck))
    if lt < 1e-9 or ln < 1e-9:
        return None
    along = abs(float(torso @ level)) / (lt * max(float(np.linalg.norm(level)), 1e-12))
    if (up is None and along > math.cos(math.radians(LEVEL_DEG))) or \
            (up is not None and along < math.sin(math.radians(LEVEL_DEG))):
        return None
    angle = math.degrees(math.acos(max(-1.0, min(1.0, float(torso @ neck) / (lt * ln)))))
    across = neck - (neck @ torso) / (lt * lt) * torso
    return angle if float(across @ forward) >= 0 else -angle


def head_pitch(face, up, forward, normal=None, offset=0.0):
    """Degrees the face points above level (+ up, - down), from the direction it points (`face`), the
    up direction and the body's front, all laid on the plane square to `normal` when given; `offset`
    is added (the coach's ear->nose line sits NEUTRAL_EAR_NOSE_DEG below a level gaze). None when the
    face has no direction on that plane."""
    face, up, forward = _flatten([face, up, forward], normal)
    up = up / np.linalg.norm(up)
    forward = forward - (forward @ up) * up
    forward = forward / np.linalg.norm(forward)
    if float(np.linalg.norm(face)) < 1e-9:
        return None
    return math.degrees(math.atan2(float(face @ up), float(face @ forward))) + offset
