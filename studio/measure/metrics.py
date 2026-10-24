"""Body-angle metrics shared by the photo reader and the Blender rig reader.

numpy + math only, so it runs under Python 3.10 and Blender's 3.13.
Definitions: 3d-character/reference-squat/targets.json.
"""
import math

import numpy as np

from posture import METRICS as POSTURE_METRICS


def _angle(a, b, c):
    """Angle at b between a and c, degrees (180 = straight)."""
    u, v = a - b, c - b
    size = np.linalg.norm(u) * np.linalg.norm(v)
    if size < 1e-12:  # a limb pointing straight along the view shows no bend
        return 180.0
    cos = float(u @ v) / size
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def _between(u, v):
    """Angle between two directions, degrees."""
    size = np.linalg.norm(u) * np.linalg.norm(v)
    if size < 1e-12:
        return 180.0
    return math.degrees(math.acos(max(-1.0, min(1.0, float(u @ v) / size))))


def _flat(v, normal):
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    return v - (v @ n) * n


def rest_lean(rest_joints, up, normal, front):
    """Degrees the torso line (hip joints' midpoint -> shoulder joints' midpoint) of a rest pose leans
    toward `front` (+) or back (-), seen on the plane square to `normal` (the side camera's axis)."""
    hip = (np.asarray(rest_joints["hip.left"], float) + np.asarray(rest_joints["hip.right"], float)) / 2
    shoulder = (np.asarray(rest_joints["shoulder.left"], float) + np.asarray(rest_joints["shoulder.right"], float)) / 2
    t, u, f = (_flat(np.asarray(v, dtype=float), normal) for v in (shoulder - hip, up, front))
    u, f = u / np.linalg.norm(u), f - (f @ u) / (u @ u) * u
    return math.degrees(math.atan2(float(t @ f) / np.linalg.norm(f), float(t @ u)))


def rest_turn(rest_joints, up, normal):
    """The turn, about `normal` (the side camera's axis), that stands a rest pose's torso line (hip joints'
    midpoint -> shoulder joints' midpoint, seen on the plane square to `normal`) upright: a 3x3 rotation for
    measure's `torso_turn`.

    Why (task 18): a person's MediaPipe shoulder and hip landmarks read about 0 degrees of lean standing
    tall, so the coach's lean, hip angle, upper-arm angle and neck angle are all counted from upright. A
    rig's joints need not be: the cartoon's shoulder joints sit 76 mm behind its hip joints in its upright
    bind pose (25 mm from the back of a 146 mm chest), so its joint line reads 15 degrees back standing tall
    and, matched number for number, the coach's 7-degree start cost a visible 22-degree bow. Ours is
    therefore counted from the rig's own bind pose, a neutral upright stance."""
    hip = (np.asarray(rest_joints["hip.left"], float) + np.asarray(rest_joints["hip.right"], float)) / 2
    shoulder = (np.asarray(rest_joints["shoulder.left"], float) + np.asarray(rest_joints["shoulder.right"], float)) / 2
    return _turn_onto(shoulder - hip, up, normal)


def _turn_onto(v, target, normal):
    """The 3x3 turn about `normal` that brings `v` onto `target`, both seen on the plane square to `normal`."""
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    t, u = _flat(np.asarray(v, dtype=float), n), _flat(np.asarray(target, dtype=float), n)
    a = math.atan2(float(n @ np.cross(t, u)), float(t @ u))       # turns t onto u about n
    k = np.array([[0.0, -n[2], n[1]], [n[2], 0.0, -n[0]], [-n[1], n[0], 0.0]])
    return np.eye(3) + math.sin(a) * k + (1.0 - math.cos(a)) * (k @ k)


LEG_LINES = {"thigh": ("hip", "knee"), "shin": ("knee", "ankle")}   # each limb's joint line, from -> to


def rest_leg_turns(rest_joints, normal, outlines):
    """{"thigh.left", "thigh.right", "shin.left", "shin.right"}: 3x3 rotations for measure's `leg_turns`. Each
    side's thigh line (hip -> knee joints) is turned, about `normal` (the side camera's axis), onto that thigh's
    skin outline as the rest (bind) pose shows it (`outlines` {"left", "right"}: limb_outline's direction; None:
    that side is read on its joints, no turn); the shin turns with its thigh, so the knee keeps its joints' reading.

    Why (task 20): the coach's hip, knee and depth are read off MediaPipe's landmarks, which run down the middle of
    the leg as the picture shows it. A rig's joints need not: the cartoon's hip joint sits toward the front of the
    thigh's top, so its thigh joint line slopes 10.6 degrees back in its bind pose while the thigh's own skin
    leans back 4.5 (MediaPipe on its bind render: about 4). Read on the joints, a squat at the coach's depth looked
    deeper than his; counted as straight down from the bind pose, it looked shallower. The skin outline is what the
    eye compares. The shin is not turned onto its own outline: the cartoon's is all knee and heel (its middle reads
    +0.5 degrees, knee to ankle -8), and MediaPipe reads the bind knee straight (178.4), as the joints do (180)."""
    out = {}
    for s in ("left", "right"):
        a, b = (np.asarray(rest_joints[f"{k}.{s}"], float) for k in LEG_LINES["thigh"])
        seen = outlines.get(s)
        r = np.eye(3) if seen is None else _turn_onto(b - a, seen, normal)
        out[f"thigh.{s}"], out[f"shin.{s}"] = r, r.copy()
    return out


OUTLINE_SPAN = (0.30, 0.85)     # of the way from the hip joint to the knee: the thigh between the seat's and the knee's
OUTLINE_STEP = 0.05             # bulges, cut every 5% of its length
OUTLINE_CUTS = 4                # the fewest cuts an outline is fitted through


def limb_outline(points, edges, a, b, normal, span=OUTLINE_SPAN, step=OUTLINE_STEP, edge=None):
    """The unit direction (in the plane square to `normal`) of a limb's skin outline: its own skin (`points`, an
    (N, 3) array, and its `edges`, (M, 2) indices) cut square to the joint line a -> b every `step` of the way over
    `span`; each cut's ring (where the edges cross it) seen on the plane, its front and back; the line fitted through
    their midpoints - or, given `edge` (a direction), through the ring's side toward it (the thigh's front: its top
    at the bottom of a squat, task 21). Edges, not vertices: a coarse mesh (the cartoon's shin: rings 11 mm apart)
    still gives every cut its ring. None when fewer than OUTLINE_CUTS cuts find skin."""
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    pts, edges = np.asarray(points, dtype=float), np.asarray(edges, dtype=int).reshape(-1, 2)
    full = b - a
    d = _flat(full, n)
    length = float(np.linalg.norm(d))
    if length < 1e-9 or not len(edges):
        return None
    u = d / length
    across = np.cross(n, u)
    toward = 0.0 if edge is None else float(np.sign(across @ _flat(np.asarray(edge, dtype=float), n)))
    along = (pts - a) @ full / float(full @ full)       # 0 at a, 1 at b, square to the joint line
    s0, s1 = along[edges[:, 0]], along[edges[:, 1]]
    p0, p1 = pts[edges[:, 0]], pts[edges[:, 1]]
    mids = []
    for t in np.arange(span[0], span[1] + 1e-9, step):
        x = (s0 - t) * (s1 - t) < 0
        if x.sum() < 3:
            continue
        w = (t - s0[x]) / (s1[x] - s0[x])
        c = (p0[x] + w[:, None] * (p1[x] - p0[x]) - a) @ across
        mids.append((t * length, (c.min() + c.max()) / 2 if not toward else (c.max() if toward > 0 else c.min())))
    if len(mids) < OUTLINE_CUTS:
        return None
    mids = np.array(mids)
    line = u + np.polyfit(mids[:, 0], mids[:, 1], 1)[0] * across
    return line / np.linalg.norm(line)


def limb_slope(v, up, normal, front):
    """Degrees a downward limb line `v` points toward `front` (+) or back (-) of straight down, seen on the plane
    square to `normal`."""
    n = np.asarray(normal, dtype=float)
    u = _flat(np.asarray(up, dtype=float), n)
    u = u / np.linalg.norm(u)
    f = _flat(np.asarray(front, dtype=float), n)
    f = f - (f @ u) * u
    f = f / np.linalg.norm(f)
    v = _flat(np.asarray(v, dtype=float), n)
    return math.degrees(math.atan2(float(v @ f), float(-(v @ u))))


def rest_leg_slopes(rest_joints, up, normal, front):
    """{"thigh.left", ...}: degrees each limb's rest joint line points toward `front` (+) or back (-) of straight
    down, seen on the plane square to `normal` (the cartoon's: -10.6 each)."""
    return {f"{limb}.{s}": limb_slope(np.asarray(rest_joints[f"{b}.{s}"], float) - np.asarray(rest_joints[f"{a}.{s}"], float),
                                      up, normal, front)
            for limb, (a, b) in LEG_LINES.items() for s in ("left", "right")}


BIND_LEAN_MAX = 25.0    # degrees: a bind pose's torso line leaning further than this is not standing upright
CHEST_TILT_MAX = 30.0   # degrees: ... nor is one whose chest (the Spine bone's head to the Neck's) tips this far
# (the cartoon's own: torso line -15.4, chest -5.0; its single spine segments zig-zag, -57 to +45, so the chest
# is read as one line, not bone by bone; a stock MPFB human: +1.8 and about -4)


def bind_pose_warning(lean, chest):
    """Plain words when a rig's bind pose does not stand upright, else None: `lean` its torso line's rest lean
    (rest_lean, side camera), `chest` its chest's rest tilt (None: not read). Every torso angle of ours is
    counted from the bind pose (task 18), which assumes it stands tall. The one rule: the Studio's stills step
    and the solver (pipeline/blender/mpfb_cage.py) both say it."""
    parts = [f"its torso line leans {lean:+.1f} degrees"] if abs(lean) > BIND_LEAN_MAX else []
    if chest is not None and abs(chest) > CHEST_TILT_MAX:
        parts.append(f"its chest tips {chest:+.1f} degrees")
    if not parts:
        return None
    return ("WARNING: the rig's bind pose does not stand upright (" + "; ".join(parts) + "); every torso angle "
            "of ours is counted from it, so they may read off")


BIND_THIGH_MAX = 5.0    # degrees (the Studio's angle tolerance): how far MediaPipe's reading of the bind pose's thigh
# may lie from the thigh's skin outline before the legs' readings are said to be in doubt (the cartoon's: 0.6;
# a stock MPFB human's: 0.3)


def picture_slope(top, bottom, facing):
    """Degrees a downward limb line in a picture (`top` -> `bottom`, (x, y) with y down) points toward where the body
    faces (+) or back (-) of straight down; None without a facing ("right" or "left": not a side view)."""
    if facing not in ("right", "left"):
        return None
    ahead = (bottom[0] - top[0]) * (1.0 if facing == "right" else -1.0)
    return math.degrees(math.atan2(ahead, bottom[1] - top[1]))


def bind_legs_warning(outlines, mediapipe, side):
    """Plain words when the legs, counted from the thigh's skin outline in the bind pose (rest_leg_turns), may read
    off; None otherwise (task 21). `outlines` {"thigh.left", "thigh.right"}: each outline's slope, + forward of
    straight down (None: not found, that leg is read on its joints); `mediapipe` its reading of a render of the
    bind pose ({"thigh_slope": the camera side's hip -> knee landmarks' slope, picture_slope}; None or {"why"}: not
    read, nothing to compare with). Said when a thigh has no outline or the camera side's two readings lie more than
    BIND_THIGH_MAX apart."""
    missing = [s for s in ("left", "right") if outlines.get(f"thigh.{s}") is None]
    if missing:
        return (f"The {' and '.join(missing)} thigh's outline in the bind pose was not found: "
                + ("that leg is" if len(missing) == 1 else "those legs are") + " read on the rig's joints")
    if not isinstance(mediapipe, dict) or "why" in mediapipe:
        return None
    seen, ours = mediapipe.get("thigh_slope"), outlines[f"thigh.{side}"]
    if isinstance(seen, bool) or not isinstance(seen, (int, float)) or not math.isfinite(seen):
        return None
    if abs(seen - ours) <= BIND_THIGH_MAX:
        return None
    return (f"The bind pose's thigh reads {ours:+.1f} degrees from straight down by its skin outline but {seen:+.1f} "
            f"by MediaPipe on its render: the legs, counted from the outline, may read off")


def measure(joints, side, up, view_normal, torso_turn=None, leg_turns=None):
    """Metrics from joints keyed "shoulder.left", "knee.right", ...

    With view_normal every point is first projected onto the plane perpendicular
    to it ("flat"); without it the 3D points are used. `up` is the world up axis.

    torso_turn (3x3, rest_turn): our rig's torso lines - each side's hip -> shoulder and the midpoints' -
    are turned by it before the angles that use them: torso_lean, hip_included (torso against the thigh)
    and upperarm_vs_torso (the upper arm against the torso, from the shoulder joint). The joints
    themselves stay where they are. None (the coach's landmarks): as measured.

    leg_turns ({"thigh.left": 3x3, ...}, rest_leg_turns): likewise our rig's thigh (hip -> knee) and shin
    (knee -> ankle) lines, turned before the measures that use them: knee_included, hip_included and
    hip_below_knee (the thigh's rise, so a thigh level as its skin looks reads 0) and thigh_level (the same
    rise as an angle above the horizontal, + = the knee higher). None: as measured.
    """
    p = {k: np.asarray(v, dtype=float) for k, v in joints.items()}
    up = np.asarray(up, dtype=float)
    if view_normal is not None:
        n = np.asarray(view_normal, dtype=float)
        n = n / np.linalg.norm(n)
        p = {k: v - (v @ n) * n for k, v in p.items()}
        up = up - (up @ n) * n
    up = up / np.linalg.norm(up)
    turn = np.eye(3) if torso_turn is None else np.asarray(torso_turn, dtype=float)

    def j(name, s=side):
        return p[f"{name}.{s}"]

    shoulder_mid = (p["shoulder.left"] + p["shoulder.right"]) / 2
    hip_mid = (p["hip.left"] + p["hip.right"]) / 2
    torso = shoulder_mid - hip_mid
    torso_len = float(np.linalg.norm(torso))
    shoulder_width = float(np.linalg.norm(p["shoulder.left"] - p["shoulder.right"]))
    side_torso = turn @ (j("shoulder") - j("hip"))      # this side's hip -> shoulder, turned
    legs = leg_turns or {}

    def limb(name):
        a, b = LEG_LINES[name]
        v = j(b) - j(a)
        r = legs.get(f"{name}.{side}")
        return v if r is None else np.asarray(r, dtype=float) @ v

    thigh, shin = limb("thigh"), limb("shin")           # this side's hip -> knee and knee -> ankle, turned

    def width_ratio(name):
        if shoulder_width < 1e-9:  # shoulders edge-on to the view: no scale to compare with
            return float("nan")
        return float(np.linalg.norm(p[f"{name}.left"] - p[f"{name}.right"])) / shoulder_width

    return {
        "knee_included": _between(-thigh, shin),
        "hip_included": _between(side_torso, thigh),
        "elbow_included": _angle(j("shoulder"), j("elbow"), j("wrist")),
        "upperarm_vs_torso": _between(j("elbow") - j("shoulder"), -side_torso),
        "torso_lean": _between(turn @ torso, up),
        "hip_below_knee": float(thigh @ up) / torso_len,
        "thigh_level": math.degrees(math.atan2(float(thigh @ up), float(np.linalg.norm(thigh - (thigh @ up) * up)))),
        "grip_ratio": width_ratio("wrist"),
        "elbow_width_ratio": width_ratio("elbow"),
    }


WIDTH_METRICS = ("grip_ratio", "elbow_width_ratio")


def _finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _tolerance_of(metric, tolerance):
    if metric in WIDTH_METRICS:
        return tolerance["ratios"]
    return tolerance["hip_below_knee"] if metric == "hip_below_knee" else tolerance["angles_deg"]


JUMP_LIMIT = 2      # tolerances: the spec's "two readings disagree" limit (10 deg, 0.24)
JUMP_WINDOW = 2     # frames each side: two-frame jumps every other frame still read as jumps


def _steady(series, tol):
    """The (frame, value) readings within JUMP_LIMIT tolerances of their neighbourhood's median."""
    values = [v for _, v in series]
    keep = []
    for i, (frame, v) in enumerate(series):
        around = values[max(0, i - JUMP_WINDOW):i + JUMP_WINDOW + 1]
        if abs(v - float(np.median(around))) <= JUMP_LIMIT * tol:
            keep.append((frame, v))
    return keep


def _driver(per_frame, targets):
    """(metric, start target, end target, tolerance, [(frame, reading)]) of the driving metric, among those
    the character has readings for (by the target's method): metrics whose start and end targets are both
    certain come first, then the one whose targets are furthest apart in tolerance units. Posture
    (posture.METRICS) is judged at the positions, never used to find them. Raises ValueError (plain
    words) when there is none."""
    phases, tolerance = targets["phases"], targets["tolerance"]
    best = None
    for metric, start in phases.get("start", {}).items():
        if metric in POSTURE_METRICS:
            continue
        end = phases.get("end", {}).get(metric)
        if not (isinstance(start, dict) and isinstance(end, dict)
                and _finite(start.get("value")) and _finite(end.get("value"))):
            continue
        method = start["method"]
        series = [(f["frame"], (f.get(method) or {}).get(metric)) for f in per_frame]
        series = [(frame, v) for frame, v in series if _finite(v)]
        if not series:
            continue
        tol = _tolerance_of(metric, tolerance)
        certain = not (start.get("uncertain") or end.get("uncertain"))
        rank = (certain, abs(start["value"] - end["value"]) / tol)
        if best is None or rank > best[0]:
            best = (rank, metric, start["value"], end["value"], tol, series)
    if best is None:
        raise ValueError("Could not find the start, middle and end on our character: no position the "
                         "coach was measured in is measured on the character.")
    return best[1:]


POSITION_JUMP = 0.20    # torso lengths: 2 x the spec's hip-vs-knee tolerance (0.10)
POSITION_WINDOW = 3     # frames each side: a run of three popped frames is still outvoted 4 to 3
SEGMENT_MAX = 10        # frames: a stretch between two steps at most this long can be a displaced block
TRACKED = ("hip.mid", "knee.left", "knee.right")


def _point(j, key):
    if key == "hip.mid":
        return (np.asarray(j["hip.left"], dtype=float) + np.asarray(j["hip.right"], dtype=float)) / 2
    return np.asarray(j[key], dtype=float)


def _tracks(joints_by_frame):
    order = sorted(joints_by_frame)
    return order, {k: np.array([_point(joints_by_frame[f], k) for f in order]) for k in TRACKED}


def _off_median(joints_by_frame, keep=None):
    """{frame: (hip midpoint's distance, the worse knee's distance)} from the component-wise median of
    the frame's 7-frame window (clipped at the clip's ends). With `keep`, the median is taken over the
    window's frames in `keep` only, and the frame maps to None when none of them is."""
    order, tracks = _tracks(joints_by_frame)
    out = {}
    for i, f in enumerate(order):
        window = range(max(0, i - POSITION_WINDOW), min(len(order), i + POSITION_WINDOW + 1))
        rows = [w for w in window if keep is None or order[w] in keep]
        if not rows:
            out[f] = None
            continue

        def off(key):
            return float(np.linalg.norm(tracks[key][i] - np.median(tracks[key][rows], axis=0)))

        out[f] = (off("hip.mid"), max(off("knee.left"), off("knee.right")))
    return out


def _torso_length(joints_by_frame):
    """The clip's torso length (hip midpoint to shoulder midpoint), the median over its frames."""
    return float(np.median([np.linalg.norm(
        (np.asarray(j["shoulder.left"], dtype=float) + np.asarray(j["shoulder.right"], dtype=float)) / 2
        - _point(j, "hip.mid")) for j in joints_by_frame.values()]))


def jump_steps(joints_by_frame):
    """[{"frame", "mm"}]: the frames whose hip midpoint or either knee moved more than POSITION_JUMP torso
    lengths since the frame before (frame = the later of the two; mm = the largest of those moves, to 0.1).
    A discontinuity, however long the stretch it opens: the step rule a window median cannot follow."""
    if len(joints_by_frame) < 2:
        return []
    limit = POSITION_JUMP * _torso_length(joints_by_frame)
    order, tracks = _tracks(joints_by_frame)
    out = []
    for i in range(1, len(order)):
        size = max(float(np.linalg.norm(t[i] - t[i - 1])) for t in tracks.values())
        if size > limit:
            out.append({"frame": order[i], "mm": round(size * 1000, 1)})
    return out


def _displaced_blocks(joints_by_frame, bad, limit):
    """Frames of each stretch between two consecutive steps (at most SEGMENT_MAX frames long) whose pose
    sits, at the median, more than `limit` off the straight line between the nearest good frames either
    side (frames in `bad` and blocks already found are not good). That last test keeps the good frame
    between two single-frame pops (also bounded by steps) out."""
    order, tracks = _tracks(joints_by_frame)
    index = {f: i for i, f in enumerate(order)}
    steps = [s["frame"] for s in jump_steps(joints_by_frame)]
    bad, found = set(bad), set()
    for s1, s2 in zip(steps, steps[1:]):
        block = [f for f in order if s1 <= f < s2]
        if not block or len(block) > SEGMENT_MAX:
            continue
        before = [f for f in order if f < s1 and f not in bad | found]
        after = [f for f in order if f >= s2 and f not in bad | found]
        if not before or not after:
            continue
        b, a = before[-1], after[0]
        offs = []
        for f in block:
            t = (f - b) / (a - b)
            offs.append(max(float(np.linalg.norm(tr[index[f]] - ((1 - t) * tr[index[b]] + t * tr[index[a]])))
                            for tr in tracks.values()))
        if float(np.median(offs)) > limit:
            found.update(block)
    return found


def jump_frames(per_frame, targets, joints_by_frame=None):
    """Frames where the pose pops: a defect of the animation to report, never a position to pick.

    Any rule makes a frame a jump: the driving metric's reading is more than JUMP_LIMIT tolerances from
    the median of the five frames around it; or (given joints_by_frame, {frame: {"hip.left": xyz, ...}})
    the hip midpoint or either knee is more than POSITION_JUMP torso lengths from the median of its
    7-frame window; or the frame is in a displaced block between two steps (jump_steps). The position
    rules catch what angles cannot (the whole body dropping), the window catches runs of up to three,
    the steps catch longer blocks that a window median follows.
    """
    _, _, _, tol, series = _driver(per_frame, targets)
    steady = {frame for frame, _ in _steady(series, tol)}
    frames = {frame for frame, _ in series if frame not in steady}
    if joints_by_frame:
        limit = POSITION_JUMP * _torso_length(joints_by_frame)
        frames |= {f for f, (hips, knees) in _off_median(joints_by_frame).items() if max(hips, knees) > limit}
        frames |= _displaced_blocks(joints_by_frame, frames, limit)
    return sorted(frames)


def jump_sizes(joints_by_frame, frames):
    """How far each jump frame's hip midpoint and worse knee sit from the median of its 7-frame window's
    steady frames (the frames not in `frames`), in mm to 0.1: robust, so a popped neighbour cannot
    inflate it. null when the whole window jumps: no number is guessed. joints_by_frame in metres."""
    jumps = set(frames)
    off = _off_median(joints_by_frame, keep={f for f in joints_by_frame if f not in jumps})
    return [{"frame": f,
             "hips_mm": None if off[f] is None else round(off[f][0] * 1000, 1),
             "knees_mm": None if off[f] is None else round(off[f][1] * 1000, 1)} for f in sorted(jumps)]


def phase_driver(per_frame, targets):
    """The metric pick_phase_frames times the positions by (see _driver). ValueError when there is none."""
    return _driver(per_frame, targets)[0]


def pick_phase_frames(per_frame, targets, skip=()):
    """The animation frames of the coach's start, middle and end positions.

    per_frame: [{"frame", "flat": {metric: v}, "3d": {metric: v}}] (ours.json); targets: targets.json.
    The driving metric: see _driver. Start and end are its turning points, not matches: when its start
    target is above its end target, start = the frame where it is largest and end = where it is smallest
    (the other way round otherwise), ties to the earlier frame. Matching them would make the driver's own
    start and end checks pass by construction. Middle is the frame between those two whose reading is
    nearest the middle target (halfway between start and end when the middle has no value for it): it is
    matched, so checks.py does not judge the driver there. Frames in `skip` (jump_frames with the
    joints) and the driver's own jump frames are never picked; when that leaves no frame, ValueError
    (plain words). Also ValueError when no metric can drive.
    """
    metric, start_value, end_value, tol, series = _driver(per_frame, targets)
    steady, skip = {frame for frame, _ in _steady(series, tol)}, set(skip)
    series = [(frame, v) for frame, v in series if frame in steady and frame not in skip]
    if not series:
        raise ValueError("Every frame of the animation jumps; no position can be measured.")

    def highest(frames):
        return min(frames, key=lambda fv: (-fv[1], fv[0]))[0]

    def lowest(frames):
        return min(frames, key=lambda fv: (fv[1], fv[0]))[0]

    def nearest(value, frames):
        return min(frames, key=lambda fv: (abs(fv[1] - value), fv[0]))[0]

    if start_value >= end_value:
        first, last = highest(series), lowest(series)
    else:
        first, last = lowest(series), highest(series)
    middle = (targets["phases"].get("middle") or {}).get(metric)
    middle_value = middle["value"] if isinstance(middle, dict) and _finite(middle.get("value")) \
        else (start_value + end_value) / 2
    low, high = sorted((first, last))
    between = [(frame, v) for frame, v in series if low <= frame <= high]
    return {"start": first, "middle": nearest(middle_value, between), "end": last}
