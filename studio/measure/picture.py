"""Our character read from its picture, as the coach is read from his (task 19).

The coach's angles are MediaPipe's landmarks on his photo (read_pose.py). Ours were read off the rig's
joints, which on this cartoon do not sit where MediaPipe puts a body's shoulders and hips: its shoulder
joints are at the back of the chest beside the hands, MediaPipe's shoulders near the base of the neck, and
as he squats MediaPipe's shoulders slide back toward the bar and its hips ride up into the shorts. Counted
from the bind pose (task 18) the rig line read the coach's 18 / 22 degrees at the middle and bottom while
the coach's own detector, run on our stills, read 12 / 10 - and the user's eye saw him upright too. So the
torso-based rows (PICTURE_METRICS) are judged on our picture, read the coach's way; the rig is their
cross-check. The other angles are read and shown (REPORTED_METRICS), but stay rig-primary: on the cartoon's
short legs in baggy shorts MediaPipe puts the bottom position's knee on the shin.

How (studio/blender/inspect_run.py, the stills step; the solver's picture loop uses the same code): our side
view is rendered at each position's frame in each of FRAMINGS - the still itself, and the framing's centre
moved by 1.5% of the frame left, right, up and down - and read_pose reads each. `summarize` keeps the
framings whose shoulders and hips MediaPipe sees clearly (each of the four landmarks' visibility at least
SEEN_MIN) and takes the median of each angle over them; at least MIN_SEEN framings must be seen, else the
picture gives no reading and the rows fall back to the rig, saying why.

- Several framings: on our renders one MediaPipe reading moves by up to 1.5 degrees (lean) and 2.5 (hip)
  when the framing moves by 1.5% or the pixels change invisibly (task 19, exp2), and the bottom still itself
  sat at the edge of that spread (lean 10.5 against a median of 11.5). The coach's photo is one fixed
  picture; ours can be rendered again, so its reading need not hang on one render's noise.
- SEEN_MIN 0.5: MediaPipe's own threshold for drawing a landmark as visible (mediapipe drawing_utils
  _VISIBILITY_THRESHOLD). The coach's shoulders and hips read 1.00 on all three side photos, ours at least
  0.999; the knees read 0.03-0.74 on the coach and 0.04-0.74 on ours, so they are not gated (a knee gate
  would refuse the coach's own targets).
"""
import math
import statistics

PICTURE_METRICS = ("torso_lean", "hip_included")
REPORTED_METRICS = ("knee_included", "elbow_included", "upperarm_vs_torso")
TORSO_LANDMARKS = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
SEEN_MIN = 0.5
MIN_SEEN = 3
# (across, up, scale): the framing's centre moved by these fractions of the framed extent, its size scaled
FRAMINGS = ((0.0, 0.0, 1.0), (0.015, 0.0, 1.0), (-0.015, 0.0, 1.0), (0.0, 0.015, 1.0), (0.0, -0.015, 1.0))
NO_BODY = "MediaPipe found no body in our picture"


def _finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def torso_visibility(result):
    """The least visibility of the shoulders and hips in one read_pose result; None without one."""
    vis = (result or {}).get("visibility") if isinstance(result, dict) else None
    if not isinstance(vis, dict):
        return None
    values = [vis.get(n) for n in TORSO_LANDMARKS]
    return min(values) if all(_finite(v) for v in values) else None


def _median(values):
    values = [v for v in values if _finite(v)]
    return round(float(statistics.median(values)), 3) if values else None


def summarize(results):
    """One position's picture reading from the read_pose results of its framings (a result with an
    "error" key: no body found). {"ok", "reads", "seen", "flat", "3d", "spread", "visibility", "why"}:
    `flat`/`3d` the median of each angle over the framings seen (None when not ok), `spread` the least and
    largest of each picture metric (flat), `visibility` the median torso (least of the shoulders and hips)
    and camera-side knee visibility over every framing read."""
    results = list(results or [])
    found = [r for r in results if isinstance(r, dict) and "error" not in r]
    seen = [r for r in found if (torso_visibility(r) or 0.0) >= SEEN_MIN]
    ok = len(seen) >= MIN_SEEN
    knees = [((r.get("visibility") or {}).get(f"{r.get('camera_side')}_knee")) for r in found]
    out = {"ok": ok, "reads": len(results), "seen": len(seen), "flat": None, "3d": None, "spread": None,
           "visibility": {"torso": _median([torso_visibility(r) for r in found]), "knee": _median(knees)},
           "why": None}
    if not found:
        out["why"] = NO_BODY
        return out
    if not ok:
        out["why"] = f"MediaPipe saw our shoulders and hips clearly in {len(seen)} of {len(results)} framings"
        return out
    metrics = PICTURE_METRICS + REPORTED_METRICS
    for method in ("flat", "3d"):
        out[method] = {m: _median([(r.get(method) or {}).get(m) for r in seen]) for m in metrics}
    out["spread"] = {}
    for m in PICTURE_METRICS:
        vals = [v for v in ((r.get("flat") or {}).get(m) for r in seen) if _finite(v)]
        out["spread"][m] = [round(min(vals), 3), round(max(vals), 3)] if vals else None
    return out


def picture_value(summary, method, metric):
    """The picture's reading of a metric (`method` "flat" or "3d"), or None when the picture gave none."""
    if not isinstance(summary, dict) or not summary.get("ok"):
        return None
    v = (summary.get(method) or {}).get(metric)
    return float(v) if _finite(v) else None
