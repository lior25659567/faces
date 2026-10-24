#!/usr/bin/env python
"""Draw one person's avatar once per expression the home screen needs.

The avatar is a flat PNG. Nothing can change its face after it is drawn, so an avatar whose face
follows how training is going has to be several pictures, drawn in advance, and chosen between at
display time. This produces that set.

What makes them one character and not several people is that everything except a single clause is
held still: the same photo, the same `features()` description (asked for once and cached on disk,
so a re-run costs nothing), the same `STYLE`, the same framing and backdrop, and above all the same
`seed_for()` seed. The seed is the whole trick. Cloudflare's image model redraws close to the same
picture for the same seed and prompt, and a recognisably different person for a different seed, so
holding it still is what turns five drawings into five moods of one face rather than five
strangers.

Four expressions, for five moods
--------------------------------
`Sources/Faces/Model/Mood.swift` has five cases. They map onto four faces:

    energised (trained today)        -> pleased
    ready     (today still open)     -> neutral
    resting   (a deliberate rest day)-> resting
    sleepy    (nothing for a while)  -> sleepy
    new       (nothing recorded yet) -> neutral

**There is no disappointed or sad face, and there must never be one.** The avatar is a drawing of
the person who is using the app. A sad version of someone's own face, shown back to them because
they did not train, is the precise thing this app refuses to do — `Mood`'s own comment says the
opinion has to be earned and that it never scolds. So the register runs from sleepy to pleased and
stops there. The lowest state in the app is "it's been a few days", and the face that carries it is
asleep, not unhappy: a resting face is an invitation to wake up, where a frown is an accusation.

`new` shares the neutral face deliberately, and not only to save a drawing. Nothing has been
recorded, so the avatar has no grounds for any opinion at all, and neutral is what the absence of
an opinion looks like. It happens to be the right engineering answer too: someone who has just
installed the app has exactly one picture of themselves, the one their scan drew, and that picture
is already the neutral face (see below). The mood every new person starts in therefore needs no
variants to look correct.

`ready` shares it for the same reason: today is still open, nothing has happened either way.

The neutral face is the avatar that already exists
--------------------------------------------------
`STYLE` ends with "a gentle closed-mouth smile", so every avatar `restyle.py` has ever drawn is
already wearing the neutral expression. The neutral clause below restates that smile word for word
rather than inventing a different calm face, which is what lets `MoodAvatar.swift` fall back to a
scan's single avatar for `ready` and `new` and have the face actually be right, instead of merely
being the best available.

The one clause that fights the style
------------------------------------
`STYLE` is reused verbatim, because it is the house look and new art that does not reuse it does
not belong to the same app. But it names an expression, and three of the four expressions here
contradict it. The clause is therefore placed last, immediately before the framing, and says out
loud which part of the style it replaces; a later, more specific instruction is what the image
model weighs most heavily. Whether that is enough is a question about the model, not about this
code, and the answer is in the report for the run that generated the first set — if a sleepy face
comes back smiling, the fix is to substitute the phrase inside `STYLE` rather than to argue with
it, and that is a change to the house look that wants deciding on purpose.

Cost and re-runs
----------------
Every drawing is a real call against the free daily allowance. Nothing is ever generated twice: a
PNG already on disk is skipped, and the description is cached beside the pictures. An exhausted
allowance stops the run with what it has, and the next run picks up the expressions that are still
missing. The number of calls actually spent is printed at the end.

Needs CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN in the environment.

    pipeline/.venv/bin/python pipeline/mood_faces.py input/me.jpg out/moods --outfit tee-pants
"""

import argparse
import base64
import io
import json
import sys
from pathlib import Path

from PIL import Image, ImageOps, ImageStat

import restyle
from apple_vision import faces
from restyle import (
    ALLOWANCE_USED,
    CLOSE_UP_SIZE,
    DARKEST_FACE,
    DARKEST_PHOTO,
    FRAMING,
    MODEL,
    OUTFITS,
    PHOTO_LIMITS,
    SIZES,
    TOO_DARK,
    as_input,
    cloudflare,
    describe,
    face_close_up,
    features,
    seed_for,
)

# The faces, in the order a contact sheet should show them: calmest to brightest. The keys are the
# file-name suffixes and are also `AvatarExpression`'s raw values in MoodAvatar.swift; renaming one
# here renames it there, and orphans every PNG already written under the old name.
#
# Each clause describes eyes, brows and mouth separately. Asking for a mood by name ("a happy
# face") leaves the model to decide which features carry it, and across a seeded set that decision
# is the thing most likely to come out the same every time; naming the features is what actually
# moves them.
EXPRESSIONS = {
    # Every clause is written as something to draw rather than as something to avoid. Image models
    # read a negation as the thing it names, so "a mouth with no smile" risks a smile and "not
    # unhappy" risks the one face this set must never contain.
    "sleepy": (
        "both eyes fully closed with soft heavy eyelids and visible lower lashes, eyebrows low and "
        "relaxed, a small relaxed closed mouth, the head tipped very slightly to one side, "
        "peacefully asleep and content"
    ),
    "resting": (
        "eyes softly half-closed and looking a little downwards, eyebrows level and relaxed, a "
        "small calm closed-mouth smile, the face completely at ease"
    ),
    "neutral": (
        "eyes open and looking straight into the camera, eyebrows level, a gentle closed-mouth "
        "smile"
    ),
    "pleased": (
        "a wide open happy smile showing the upper teeth, eyes crinkled into bright happy curves, "
        "eyebrows raised, cheeks lifted and rosy"
    ),
}

# `Mood`'s cases and the face each one wears. Kept here as well as in MoodAvatar.swift because this
# file decides which pictures are worth the calls, and that decision is this mapping.
MOODS = {
    "energised": "pleased",
    "ready": "neutral",
    "resting": "resting",
    "sleepy": "sleepy",
    "new": "neutral",
}

# Said as a replacement rather than as an addition, because `STYLE` has already asked for a gentle
# closed-mouth smile a few sentences earlier and three of the four expressions disagree with it.
# Naming the phrase being replaced is the strongest form of the instruction available without
# editing the house style itself.
EXPRESSION = (
    "Expression: {clause}. This expression replaces the gentle closed-mouth smile named in the "
    "style above; everything else about the character stays exactly as described. "
)

# How many Cloudflare calls this run actually spent, printed at the end. The person who owns this
# allowance has asked more than once for it to be spent carefully, and a script that quietly makes
# four calls where it could have made none is not something they should have to discover from a
# dashboard.
SPENT = {"description": 0, "drawing": 0}


def clause_for(expression):
    """The sentence that replaces the style's smile, for one expression."""
    return EXPRESSION.format(clause=EXPRESSIONS[expression])


def moods_for(expression):
    """Which of `Mood`'s cases wear this face. Two of them share one, and it is drawn once."""
    return [mood for mood, worn in MOODS.items() if worn == expression]


def prompt_for(facts, body=None, outfit=None, close_up=True, *, expression):
    """`restyle.prompt_for()`, with the expression clause inserted before the framing.

    Built by rewriting the real prompt rather than by assembling a parallel one, so that the two
    cannot drift: a clause added to `restyle.prompt_for` appears here the same day, in the same
    order, and the only text this file is responsible for is the one sentence it inserts.

    A missing anchor raises instead of returning the prompt unchanged. An unchanged prompt would
    draw a perfectly good avatar wearing the wrong face and save it under the right name, which is
    the silent wrong answer this project does not allow — the set would look generated and be
    identical, and nobody would know why the feature did nothing.
    """
    base = restyle.prompt_for(facts, body, outfit, close_up)
    framing = FRAMING["body" if outfit else "portrait"]
    if framing not in base:
        raise RuntimeError(
            "restyle.prompt_for no longer contains FRAMING verbatim, so there is nowhere to put "
            "the expression clause. Fix the anchor here before generating anything."
        )
    return base.replace(framing, clause_for(expression) + framing, 1)


def look_at(path):
    """The photo and a close-up of its face, or exit if it is too dark to read features from.

    `restyle.main`'s gate, repeated for the same reason it exists there: it runs before any call,
    so a photo nothing could have drawn a likeness from costs nothing at all. Here it matters more
    than usual, because the mistake it prevents would be made four times over.
    """
    photo = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    found = faces(photo)
    grey = photo.convert("L")
    region = grey.crop(tuple(round(v) for v in found[0])) if found else grey
    lit = ImageStat.Stat(region).mean[0]
    if lit < (DARKEST_FACE if found else DARKEST_PHOTO):
        print(f"The photo is too dark: brightness {lit:.0f} of 255.", file=sys.stderr)
        sys.exit(TOO_DARK)
    if not found:
        print("No face found in the photo, so the avatars are drawn without a close-up of it.")
    return photo, (face_close_up(photo, found[0]) if found else None)


def seen_from(photo, close_up, cache):
    """The vision model's description of the person, from `cache` if it has been asked before.

    Cached because the description is the same for every expression — it is about the person, not
    the face they are pulling — and because it is a call. Without the cache, re-running to fill in
    one missing expression would pay for the description again, and the project's own rule is that
    nothing is ever generated twice.

    A cache that will not parse is reported and ignored rather than crashing the run: the
    description is recoverable for the price of one call, and the alternative is a script that
    cannot be run at all until a file is deleted by hand.
    """
    if cache.exists():
        try:
            return json.loads(cache.read_text())
        except ValueError:
            print(f"{cache} is not readable JSON; asking for the description again.", file=sys.stderr)
    seen = describe(photo, close_up)
    SPENT["description"] += 1
    if seen:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(seen, indent=2, sort_keys=True) + "\n")
    return seen


def draw(photo, close_up, prompt, size, seed):
    """One drawing. Returns (the picture, None), or (None, the exit status to stop on).

    `restyle.main`'s retry at the smaller input size, kept because the reason for it has not
    changed: Workers AI documents its inputs as smaller than it accepts, and a refusal can be about
    the size of what was sent.
    """
    width, height = size
    result = problem = None
    for limit in PHOTO_LIMITS:
        pictures = [as_input(photo, limit)] + ([as_input(close_up, CLOSE_UP_SIZE)] if close_up else [])
        result, problem = cloudflare(
            MODEL,
            data={"prompt": prompt, "width": width, "height": height, "seed": str(seed)},
            files={f"input_image_{index}": (f"{index}.png", picture, "image/png")
                   for index, picture in enumerate(pictures)},
        )
        SPENT["drawing"] += 1
        if result or problem == ALLOWANCE_USED:
            break
    if not result:
        return None, problem
    return Image.open(io.BytesIO(base64.b64decode(result["image"]))).convert("RGB"), None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("photo", type=Path, help="photo of the person")
    parser.add_argument("out", type=Path, help="directory for the set of avatars")
    parser.add_argument("--outfit", choices=OUTFITS,
                        help="whole-body avatars in this gym outfit. Use the SAME outfit the "
                             "person's scan used, or the figure changes shape with its mood")
    parser.add_argument("--name", help="the character's name, steady across their repeat photos "
                                       "(default: the photo's file name); fixes the seed")
    parser.add_argument("--only", nargs="+", choices=sorted(EXPRESSIONS), metavar="EXPRESSION",
                        help=f"draw only these expressions ({', '.join(EXPRESSIONS)})")
    parser.add_argument("--mood", nargs="+", choices=sorted(MOODS), metavar="MOOD",
                        help=f"draw whichever faces these moods need ({', '.join(MOODS)}); two "
                             "moods can share a face and it is drawn once")
    args = parser.parse_args()

    name = args.name or args.photo.stem
    asked = set(args.only or []) | {MOODS[mood] for mood in (args.mood or [])}
    # Put back into the canonical order whatever order they were asked in, so that a contact sheet
    # made from the directory listing always reads calmest to brightest.
    wanted = [expression for expression in EXPRESSIONS if expression in asked] or list(EXPRESSIONS)

    args.out.mkdir(parents=True, exist_ok=True)
    paths = {expression: args.out / f"{name}-{expression}.png" for expression in wanted}
    missing = [expression for expression in wanted if not paths[expression].exists()]
    for expression in wanted:
        if expression not in missing:
            print(f"{paths[expression]} is already there; not drawing it again.")
    if not missing:
        print("Every expression asked for is already on disk. Nothing to do, nothing spent.")
        return

    photo, close_up = look_at(args.photo)
    seen = seen_from(photo, close_up, args.out / f"{name}-seen.json")
    if not seen:
        # Four drawings from the photo alone would be four avatars of a generic person wearing four
        # expressions, which is worse than none: the whole point of the set is that it is the same
        # recognisable person each time, and without the facts it is not recognisably anyone.
        print("The description failed, so there is nothing to keep the likeness steady across the "
              "set. Refusing to draw rather than spending the allowance on four strangers.",
              file=sys.stderr)
        sys.exit(restyle.REFUSED)

    facts, body = features(seen, args.outfit)
    print(f"Features: {facts}" + (f"\nBody: {body}" if body else ""))

    seed = seed_for(name)
    size = SIZES["body" if args.outfit else "portrait"]
    stopped = None
    for expression in missing:
        prompt = prompt_for(facts, body, args.outfit, bool(close_up), expression=expression)
        picture, problem = draw(photo, close_up, prompt, size, seed)
        if not picture:
            stopped = problem
            print(f"Stopped at {expression}; {len(missing) - missing.index(expression)} "
                  "expression(s) not drawn. Run this again to finish the set.", file=sys.stderr)
            break
        picture.save(paths[expression], "PNG")
        print(f"{expression} (for {', '.join(moods_for(expression))}): {paths[expression]} "
              f"({picture.width} x {picture.height})")

    print(f"Cloudflare calls this run: {SPENT['description']} description, {SPENT['drawing']} drawing.")
    if stopped:
        sys.exit(stopped)


if __name__ == "__main__":
    main()
