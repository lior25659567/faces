"""What an exercise holds and what holds it up, and what of that the animation path builds: exercise-kit.json, the
one table (read here, by pipeline/blender/animate.py through this module, and by studio/server/store.js).

- `picture_refusal`: the plain sentence the picture path (no Mixamo clip; pipeline/blender/keypose_clip.py) refuses
  an exercise with - its feet step (the walking lunge), or it needs a support the path doesn't build (a bench, a
  seat, a pull-up bar, the leg press's sled) - else None. A scene without them would be physically wrong.
- `unheld`: the equipment the hands should hold that the path doesn't build ("bar", "dumbbells", ...), else None:
  checks.py's `equipment` body row says so, not measured, so the exercise can never show green with empty hands.
- `has_depth`: whether the exercise's depth at the bottom is read and judged (build_targets.py, checks.py; task 22):
  the back squat only. An exercise added in the Studio never, whatever its equipment.

An exercise added in the Studio is not in the table; it goes by the equipment word picked when it was added.
numpy-free, so Blender's Python and the Studio's both import it.
"""
import json
from pathlib import Path

KIT_PATH = Path(__file__).resolve().with_name("exercise-kit.json")
_table = None


def load():
    global _table
    if _table is None:
        _table = json.loads(KIT_PATH.read_text())
    return _table


def kit_of(exercise, equipment=None):
    """The exercise's row ({"holds", "support", "steps", "built", "clip", "depth"}, absent keys empty): by name, else by
    the Studio's equipment word, else nothing known."""
    table = load()
    row = table["exercises"].get(exercise)
    if row is None:
        row = table["by_equipment"].get(equipment or "", {})
    return {"holds": row.get("holds"), "support": row.get("support"), "steps": bool(row.get("steps")),
            "built": bool(row.get("built")), "clip": row.get("clip"), "depth": bool(row.get("depth")),
            "support_built": bool(row.get('supportBuilt')), 'reference_motion': bool(row.get('referenceMotion'))}


def picture_refusal(exercise, equipment=None):
    """The picture path's refusal for this exercise, in plain words, or None when it can animate it."""
    row, refusals = kit_of(exercise, equipment), load()["refusals"]
    if row["steps"]:
        return refusals["steps"].format(clip=row["clip"] or "its clip")
    if row["support"] and not row['support_built']:
        return refusals["support"].format(support=row["support"])
    return None


def unheld(exercise, equipment=None):
    """What the hands should hold that the animation path doesn't build, or None."""
    row = kit_of(exercise, equipment)
    return None if row["built"] else row["holds"]


def has_depth(exercise, equipment=None):
    """Whether the exercise's depth at the bottom is read and judged (its row's `depth`)."""
    return kit_of(exercise, equipment)["depth"]


def unheld_note(thing):
    """The equipment row's note for `thing` (unheld's answer)."""
    return load()["unheld"].format(holds=thing)
