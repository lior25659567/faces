#!/usr/bin/env python
"""Draw the house mascot standing still, front and back, as the body the muscle map paints on.

The app was showing which muscles a movement works on a flat grey mannequin. It read as a medical
diagram dropped into an app that otherwise looks like one consistent thing, so this draws the same
character the exercise demonstrations use, in the same style, from the same seed, and the
highlighting goes on top of it.

The division of labour here is the whole point, and it is the opposite of asking the image model
to draw anatomy. FLUX is reliable at a character, an outfit, a pose and a style, and unreliable at
specific geometry — that is the finding recorded at the top of exercise_art.py, where eight of ten
drawings got the joint angles wrong. Where a muscle *is* has to be right every time, so the model
is never asked. It draws one neutral body; Apple Vision measures where that body's shoulders, hips
and neck actually landed; and the muscle shapes are placed against those measurements by
`regions.py`. The model supplies the look and nothing that has to be correct.

The pose is deliberately an A-pose with the arms held away from the sides. Arms hanging against
the body hide the lats and the obliques behind a forearm, and a muscle map whose highlight is
behind an arm is a muscle map that cannot show a row.

Two calls, one per side, and never for a picture already on disk.

    pipeline/.venv/bin/python pipeline/muscle_figure.py --dry-run
    pipeline/.venv/bin/python pipeline/muscle_figure.py
    pipeline/.venv/bin/python pipeline/muscle_figure.py --redraw --only back
"""

import argparse
import base64
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

import apple_vision
import exercise_art
import restyle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "reference" / "muscle-figure"
RAW = OUT / "raw"
INDEX = OUT / "index.json"

SIZE = (768, 1152)
# The same seed the demonstrations use, for the same reason: it is what keeps the character one
# character. A muscle map drawn from a different seed would be a stranger's body beside the
# demonstration of the movement, on the same screen, two tabs apart.
SEED = exercise_art.SEED

# Arms out at roughly 35 degrees, palms forward, feet apart: a reference stance, not a pose. The
# numbers are spelled out because "A-pose" means nothing to the model and "arms slightly away from
# the body" came back with the arms still touching the ribs.
POSES = {
    "front": (
        "seen from directly in front, facing the camera straight on",
        "standing still and symmetrical in a neutral reference stance, facing the camera: both arms held straight "
        "down and OUT to the sides at about thirty-five degrees from the body so there is clear empty space between "
        "each arm and the ribs and nothing overlaps the torso, elbows straight, palms open and facing forward, "
        "fingers relaxed and together, both feet flat on the floor about shoulder-width apart with the toes pointing "
        "straight forward, knees straight, hips level, shoulders level and relaxed down, spine upright, head level "
        "and facing straight ahead",
    ),
    # The pose to generate a model for RIGGING from, and the reason it exists is worth writing
    # down. The arms-down drawing produced a Tripo mesh with the arm FUSED to the torso at the
    # armpit — the silhouette only separates into arm, torso, arm at 58% of the figure's height
    # while the shoulder joint sits at 63%, so a five-centimetre band of shared geometry has to
    # stretch across the whole opening when the arm lifts. It comes out as a fan of long thin
    # triangles under the arm and no amount of weighting repairs it, because the geometry to
    # stretch simply is not there.
    #
    # A T-pose fixes it upstream: arms straight out horizontally leaves the armpit open, so the
    # arm is its own tube of mesh from the shoulder down. This is why every character meant for
    # animation is modelled this way.
    "tpose": (
        "seen from directly in front, facing the camera straight on",
        "standing in a symmetrical T-pose, the reference stance used for 3D character rigging: both arms raised "
        "STRAIGHT OUT SIDEWAYS and held exactly horizontal, level with the shoulders and parallel with the floor, "
        "reaching their full length out to the left and right so there is a wide clear gap of empty background "
        "under each armpit and the arms do not touch or overlap the body anywhere, elbows completely straight, "
        "palms open and facing straight forward with the fingers extended and together, legs straight and slightly "
        "apart with both feet flat on the floor about shoulder-width apart and the toes pointing forward, a clear "
        "gap of empty background between the inner thighs, hips level, spine upright, head level and facing "
        "straight ahead at the camera",
    ),
    "trainer-front": (
        "seen from directly in front, facing the camera straight on",
        "standing in a symmetrical T-pose, the reference stance used for 3D character rigging: both arms raised "
        "STRAIGHT OUT SIDEWAYS and held exactly horizontal, level with the shoulders and parallel with the floor, "
        "reaching their full length out to the left and right so there is a wide clear gap of empty background "
        "under each armpit and the arms do not touch or overlap the body anywhere, elbows completely straight, "
        "palms open and facing straight forward with the fingers extended and together, legs straight and apart "
        "with both feet flat on the floor a little wider than shoulder-width and the toes pointing forward, a "
        "clear gap of empty background between the inner thighs, hips level, spine upright, head level and facing "
        "straight ahead at the camera",
    ),
    # An A-pose, and preferred over the T-pose for anything that will be animated.
    #
    # A rest pose sits at one end of the movement's range. From a T-pose the shoulder has to
    # rotate a full 90 degrees to bring the arms down, and that is where the mesh crumples. At
    # 45 degrees it is 45 either way — half the rotation, roughly half the deformation — while
    # still leaving the armpit open, which is the whole reason the T-pose was drawn.
    "trainer-apose": (
        "seen from directly in front, facing the camera straight on",
        "standing in a symmetrical A-pose, the reference stance used for 3D character rigging: both arms completely "
        "straight and angled DOWNWARD and outward so each arm points down and away from the shoulder toward the "
        "floor at roughly forty-five degrees, the hands level with the hips and held well out to the sides away "
        "from the thighs, forming a wide letter A — the arms are NOT horizontal, NOT raised, and NOT hanging "
        "vertically, but exactly halfway between straight out sideways and hanging straight down, with a clear "
        "wide triangle of empty background between each arm and the ribs, "
        "elbows completely straight, palms open and facing forward with the fingers extended and together, legs "
        "straight and apart with both feet flat on the floor a little wider than shoulder-width and the toes "
        "pointing forward, a clear gap of empty background between the inner thighs, hips level, spine upright, "
        "head level and facing straight ahead at the camera",
    ),
    "trainer-back": (
        "seen from directly behind, the back of the head toward the camera",
        "standing in a symmetrical T-pose seen from directly behind with the whole bare back toward the camera and "
        "the face completely hidden: both arms raised STRAIGHT OUT SIDEWAYS and held exactly horizontal, level "
        "with the shoulders and parallel with the floor, reaching their full length out to the left and right so "
        "there is a wide clear gap of empty background under each armpit, elbows completely straight, palms facing "
        "backward toward the camera, legs straight and apart with both feet flat on the floor a little wider than "
        "shoulder-width and the heels toward the camera, a clear gap of empty background between the inner "
        "thighs, hips level, spine upright, head level and facing away",
    ),
    "back": (
        "seen from directly behind, the back of the head toward the camera",
        "standing still and symmetrical in a neutral reference stance, seen from directly behind with the whole back "
        "toward the camera and the face completely hidden: both arms held straight down and OUT to the sides at "
        "about thirty-five degrees from the body so there is clear empty space between each arm and the ribs and "
        "nothing overlaps the back, elbows straight, palms facing backward toward the camera, fingers relaxed and "
        "together, both feet flat on the floor about shoulder-width apart with the heels toward the camera, knees "
        "straight, hips level, shoulders level and relaxed down, spine upright, head level and facing away",
    ),
}

# The vest has to go. A black tank top hides the lats, the traps and the whole of the back, which
# are exactly the muscles the back view exists to show. Skin-coloured athletic wear keeps the
# character dressed and the body readable, and it is the one place this file departs from
# restyle.OUTFITS.
WEARING = ("plain fitted light-grey sleeveless athletic top and plain fitted light-grey shorts, both close to the "
           "body with no folds, no logos and no printed text")

# The demonstrator for the exercise animations: an adult, and bare above the waist.
#
# Both parts earn their place. An adult reads as somebody showing you how a lift is done, where
# the child mascot reads as the user's own avatar — and these are deliberately different
# characters, since the whole point of `GENERIC` is that the demonstrator is nobody. Bare skin is
# what lets the muscle map sit on the body itself: painted over a vest, the chest and the whole
# back are hidden behind cloth, which is most of what a lifter wants to see.
#
# "Athletic, not a bodybuilder" is the one thing to hold on to. Muscle groups have to be legible
# enough that a highlight lands on something, and exaggerated mass reads as an aspiration rather
# than as a diagram.
TRAINER = (
    "The character is a generic invented adult fitness demonstrator of no particular identity and is not a portrait "
    "of any real person: an adult with an athletic, trim, naturally muscular build with clearly readable muscle "
    "groups — chest, shoulders, arms, abdominals, back and legs all visible in their natural shapes but NOT "
    "bodybuilder-sized, NOT bulging and NOT exaggerated — plain friendly features, a smooth rounded head, short "
    f"simple dark brown hair, medium tan skin ({restyle.SKIN_SHADES[4]}), no facial hair, no glasses, no jewelry, no "
    "tattoos and no printed text or logos anywhere, wearing a plain light grey sleeveless athletic tank top with "
    "wide armholes that leave the whole shoulder and the whole arm bare, and plain mid-grey gym shorts of an "
    "ordinary loose cut that end just above the knee — not tight, not cycling shorts, with a simple waistband and "
    "no folds, no logos and no printed text anywhere. "
)


def prompt_for(side):
    camera, pose = POSES[side]
    if side.startswith("trainer"):
        generic = TRAINER
    else:
        generic = exercise_art.GENERIC.replace(
            f"wearing {restyle.OUTFITS['tank-shorts']}. ", f"wearing {WEARING}. ")
    return (
        f"A cute 3D avatar character standing in a neutral full-body reference pose. "
        + restyle.STYLE
        + generic
        + f"Position: {pose}. "
        + "The one thing that must be exactly right: the character stands still and symmetrical with both arms held "
          "away from the sides so nothing overlaps the torso. "
        + exercise_art.FRAMING.format(camera=camera)
        + exercise_art.BACKDROP
    )


def draw(side):
    fields = {"prompt": prompt_for(side), "width": SIZE[0], "height": SIZE[1], "seed": SEED}
    result, problem = restyle.cloudflare(
        restyle.MODEL, files={key: (None, str(value)) for key, value in fields.items()})
    if problem == restyle.ALLOWANCE_USED:
        sys.exit(problem)
    if not result:
        return None
    return Image.open(io.BytesIO(base64.b64decode(result["image"]))).convert("RGB")


def measure(picture):
    """Where the body actually landed, so the muscle shapes can be placed against it.

    Reported in the cut-out picture's own pixels and as fractions of it, because the app scales the
    picture to whatever room it has and the shapes have to scale with it. A joint Vision could not
    find is absent rather than guessed, and `regions.py` refuses to place anything that depends on
    a missing one — a highlight in the wrong place is worse than a highlight missing."""
    _, joints = apple_vision.body(picture)
    width, height = picture.size
    return {
        "size": [width, height],
        "joints": {name: [round(x, 1), round(y, 1)] for name, (x, y) in sorted(joints.items())},
        "normalised": {name: [round(x / width, 4), round(y / height, 4)]
                       for name, (x, y) in sorted(joints.items())},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", action="append", choices=sorted(POSES), help="just this side (repeatable)")
    parser.add_argument("--dry-run", action="store_true", help="print the prompts and draw nothing")
    parser.add_argument("--redraw", action="store_true", help="draw again even where a picture exists")
    parser.add_argument("--remeasure", action="store_true", help="redo the cut-out and the joints, no calls")
    args = parser.parse_args()

    sides = args.only or sorted(POSES)
    if args.dry_run:
        for side in sides:
            print(f"--- {side}  [{SIZE}, seed {SEED}]\n{prompt_for(side)}\n")
        print(f"{len(sides)} prompts, 0 calls.")
        return

    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    index = json.loads(INDEX.read_text()) if INDEX.exists() else {"sides": {}}
    calls = 0

    for side in sides:
        cut_path, raw_path = OUT / f"{side}.png", RAW / f"{side}.png"
        if raw_path.exists() and not args.redraw:
            picture = Image.open(raw_path).convert("RGB")
            print(f"{side}: using the saved drawing.")
        else:
            if args.remeasure:
                print(f"{side}: nothing saved; --remeasure draws nothing.")
                continue
            print(f"{side}: drawing {SIZE}.")
            calls += 1
            picture = draw(side)
            if picture is None:
                print(f"{side}: Cloudflare would not draw it.", file=sys.stderr)
                continue
            picture.save(raw_path, "PNG")

        # One person, no equipment, so anything that is not the figure is a stray.
        cut, cut_measure = exercise_art.cut_out(picture, solo=True)
        if cut is None:
            print(f"{side}: not cut out — {cut_measure}", file=sys.stderr)
            continue
        cut.save(cut_path, "PNG")
        index["sides"][side] = {
            "file": cut_path.name,
            "raw": f"{RAW.name}/{raw_path.name}",
            "cut": cut_measure,
            "body": measure(cut),
            "prompt": prompt_for(side),
            # Nobody has looked at it yet. The same gate the demonstrations have, for the same
            # reason: a back view that came back as a front view is a muscle map that highlights
            # the chest when it means the lats.
            "checked": index.get("sides", {}).get(side, {}).get("checked"),
        }
        found = index["sides"][side]["body"]["joints"]
        print(f"  -> {cut_path.name} {cut.width}x{cut.height}, joints: {', '.join(sorted(found)) or 'none'}")

    index["note"] = ("The house mascot standing still, the body the muscle map paints on. One drawing per side, "
                     "cut out to a transparent background, with the joints Apple Vision found in each so the "
                     "muscle shapes can be anchored to the body rather than guessed. Written by "
                     "pipeline/muscle_figure.py.")
    index["seed"] = SEED
    index["model"] = restyle.MODEL
    index["updated_at"] = datetime.now(timezone.utc).isoformat()
    INDEX.write_text(json.dumps(index, indent=2) + "\n")
    print(f"\n{calls} Workers AI calls. Index: {INDEX}")


if __name__ == "__main__":
    main()
