#!/usr/bin/env python
"""Turn a photo of a person into a cute 3D avatar that keeps every feature of theirs.

Three steps. Only the last two use the network: two calls to Cloudflare Workers AI, both inside
its free daily allowance.
  1. Find the face, on this computer, with Apple's Vision framework (apple_vision.py).
  2. Describe: a vision model writes down the face shape, skin tone, hairstyle, facial hair,
     glasses, jewelry, clothes and body size.
  3. Draw: FLUX.2 [klein] draws the avatar from the photo, a close-up of the face, and those facts.

The facts are what keep the likeness. Asked only for a cute avatar, the model draws a generic one.

Without --outfit the avatar is a portrait from the waist up, in the person's own clothes. With
--outfit it is their whole body, in that gym outfit.

Needs CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN in the environment (the dev server passes
them in from .env). RESTYLE_MODEL swaps the image model.

    pipeline/.venv/bin/python pipeline/restyle.py input/me.jpg input/me-3d.png
    pipeline/.venv/bin/python pipeline/restyle.py input/me.jpg input/me-3d.png --outfit tee-pants
"""

import argparse
import base64
import dataclasses
import hashlib
import io
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
from PIL import Image, ImageOps, ImageStat

import apple_vision
import body_measure
from apple_vision import faces

API = os.environ.get("CLOUDFLARE_API_BASE", "https://api.cloudflare.com/client/v4")
MODEL = os.environ.get("RESTYLE_MODEL", "@cf/black-forest-labs/flux-2-klein-4b")
VISION = "@cf/google/gemma-4-26b-a4b-it"
# Workers AI documents input images as smaller than 512 x 512, but takes twice that in practice,
# and at the smaller size a face in a whole-body photo is a few dozen pixels wide.
PHOTO_LIMITS = (1008, 504)  # the second is tried if the first is ever refused
CLOSE_UP_SIZE = 504
SIZES = {"portrait": (864, 1152), "body": (768, 1360)}

CLOSE_UP = 1.5  # side of the face close-up, in face-box heights
# Mean brightness of the face, out of 255, below which its features can't be read. Faces in the
# test photos measured 85 to 170; a photo taken in a dark room, 19.
DARKEST_FACE, DARKEST_PHOTO = 35, 25

# Exit statuses the dev server turns into its own messages.
ALLOWANCE_USED = 4
REFUSED = 5
TOO_DARK = 6

# The gym outfits of a whole-body avatar. The dev server lists the same names in frames-api.js.
OUTFITS = {
    "tee-pants": "a plain grey T-shirt, black jogger pants and white trainers",
    "tank-shorts": "a plain black athletic tank top, black running shorts and white trainers",
}

# The Monk Skin Tone scale (Google, free to use), lightest to darkest. The vision model picks a shade
# and the drawing gets its color. Skin measured off the photo was too unreliable for this: a sunlit
# face measured several shades lighter than it is, a backlit one several shades darker.
SKIN_SHADES = ["#f6ede4", "#f3e7db", "#f7ead0", "#eadaba", "#d7bd96", "#a07e56", "#825c43", "#604134", "#3a312a", "#292420"]

# Body sizes as the vision model names them, and how the image model has to be told: mild words
# such as "plus-size with a full figure" were ignored, and everyone came out slim.
#
# None of these says anything about shoulder width any more. They used to — "athletic" promised
# broad shoulders — and that claim is an impression read off a photo, while shoulder width is
# something body_measure.shape_phrase actually measures. When the two disagreed the prompt said
# both: a slim man the vision model had called athletic was described as having broad shoulders
# twice over, and came back drawn like a bodybuilder. Breadth is the measurement's to state now.
BODIES = {
    "slim": "a slim, slender body with thin arms and legs and very little muscle",
    "average": "an average body",
    "athletic": "an athletic body with clearly visible muscle",
    "stocky": "a stocky, solid body with a thick waist",
    "heavyset": "a heavyset body with a big round belly, a thick waist and heavy arms and legs",
    "plus-size": "a plus-size body that is wide and round, with a full round belly, wide hips, thick thighs and full arms",
}

ASK = """You are helping make a cute 3D cartoon avatar that must match this person exactly. Image 1 is their photo; image 2, if there is one, is a close-up of their face.
If there are several people, describe only the one shown in image 2, or else the one whose face is largest.
Describe only what you can see, precisely. Answer with one JSON object only, with these keys:
"age": "child", "teen", "adult" or "senior".
"skin": their skin tone in plain words, e.g. "fair pink", "light warm beige", "medium tan", "warm brown", "deep brown".
"skin_shade": the closest shade on the Monk Skin Tone scale, as a number from 1 (lightest) to 10 (darkest). Warm, orange or red lighting (sunset, red carpets, indoor bulbs, a camera flash) makes skin read lighter and more saturated than it is, and so does bright sun; shade it darker than your first instinct whenever the lighting around the face is warm or bright. If the whites of the eyes are visible, use them as a neutral reference: skin much darker than the eye whites is a high number even if the lighting looks warm or golden.
"face": the face shape and what stands out in it, e.g. "long oval face with high cheekbones and a narrow chin", "round face with full cheeks", "square face with a strong jaw".
"hair": the hair on their head: length, texture (straight, wavy, curly, coily, twists, locs or braids), the exact style (e.g. "short skin fade", "side part swept to the right", "bob with straight bangs", "high ponytail", "big round afro", "messy tousled curls"), the hairline if it is receding or high, and its color. If it is braided or tied, say how many braids, ponytails or buns there are and where they sit (e.g. "one braid over the right shoulder, the rest loose"). If they are bald or the head is shaved, say so plainly. If a head covering hides the hair, say "hidden".
"head_covering": e.g. "black baseball cap worn forwards", "dusty pink hijab", or "none".
"eyes": eye color and shape, e.g. "dark brown almond-shaped eyes", "blue hooded eyes".
"eyebrows": thickness, shape and color.
"facial_hair": e.g. "none", "light stubble", "thick black handlebar mustache", "full bushy white beard", with its exact color, which can differ from the hair (e.g. a reddish-brown beard with dark brown hair), and its length. Mention even light stubble.
"glasses": e.g. "none", "black rectangular glasses", "round tortoiseshell glasses".
"jewelry": visible earrings, piercings, necklaces or watches, or "none".
"makeup": e.g. "none", "red lipstick", "winged eyeliner".
"top": what they wear on the upper body: garment type, color, pattern and collar. If there is a logo or printed text, describe only its shape, color and position (e.g. "a small red rectangular logo on the left chest"); never name the brand or quote the exact words, even if you recognize them.
"bottom": what they wear on the lower body, or "not visible". Describe any logo as above, never by brand name or exact words.
"shoes": their shoes, or "not visible". Describe any logo as above, never by brand name or exact words.
"build": their body size, as one of "slim", "average", "athletic", "stocky", "heavyset", "plus-size", or "not visible". Say "slim" for a thin, narrow body with little muscle, even when it looks lean, toned or in shape; say "athletic" only when muscle is plainly visible, such as defined arms or shoulders, and not merely because someone is slim and flat-stomached. Judge the body you can see, not how fit the person seems. Be precise and honest: this avatar must have their real body.
Do not mention gender, ethnicity, mood, objects they hold, or the background. Do not guess anything you cannot see."""

# What went wrong before this prompt, so nobody reintroduces it:
#   - asked only for a cute avatar, the model drew a generic one: hair on a bald head, light skin
#     for a dark-skinned man, a child's face, neat hair for messy hair;
#   - naming optional features in general ("the same glasses, earrings and head covering") made it
#     add them to people who have none, so only what a person has is named;
#   - a chest-up framing left out tops it found hard to draw, so a portrait shows the waist up;
#   - sending reference/style.png as a second image made it copy that character.
STYLE = (
    "Style: a polished, cute 3D cartoon avatar, like a premium emoji character: a big rounded head, large expressive eyes "
    "with clear colored irises and soft highlights, a small simple nose, smooth soft matte skin, stylized sculpted hair that "
    "keeps its real shape, volume and texture, a gentle closed-mouth smile, soft even studio lighting. "
)
FRAMING = {
    "portrait": (
        "Framing: from the waist up, centered, facing the camera, head upright and level, eyes looking into the camera, "
        "arms relaxed, with their clothes clearly visible and empty space above and beside the head. "
    ),
    "body": (
        "Framing: the whole body from head to feet, standing straight and facing the camera, arms relaxed at the sides, "
        "with a slightly larger head. "
    ),
}
BACKDROP = "Background: a plain pastel teal studio backdrop, with nothing in their hands and no other people or objects."


def present(value):
    """Whether a described feature is there at all."""
    return bool(value) and str(value).strip().lower() not in ("none", "not visible", "hidden", "n/a", "no", "unknown")


def as_input(picture, limit):
    """The picture as PNG bytes, shrunk to fit the limit."""
    picture = picture.copy()
    picture.thumbnail((limit, limit), Image.LANCZOS)
    buffer = io.BytesIO()
    picture.save(buffer, "PNG")
    return buffer.getvalue()


def face_close_up(photo, box):
    """The face as a square close-up, for the model to read its features from."""
    x0, y0, x1, y1 = box
    side = min(CLOSE_UP * (y1 - y0), photo.width, photo.height)
    left = min(max((x0 + x1) / 2 - side / 2, 0), photo.width - side)
    top = min(max((y0 + y1) / 2 - side / 2, 0), photo.height - side)
    crop = photo.crop(tuple(round(value) for value in (left, top, left + side, top + side)))
    return crop.resize((CLOSE_UP_SIZE, CLOSE_UP_SIZE), Image.LANCZOS)


def cloudflare(model, **request):
    """One Workers AI call. Returns (its result, None), or (None, the exit status the dev server explains)."""
    try:
        account, token = os.environ["CLOUDFLARE_ACCOUNT_ID"], os.environ["CLOUDFLARE_API_TOKEN"]
    except KeyError as missing:
        print(f"{missing.args[0]} is not set.", file=sys.stderr)
        return None, REFUSED

    response = requests.post(
        f"{API}/accounts/{account}/ai/run/{model}",
        headers={"Authorization": f"Bearer {token}"},
        timeout=180,
        **request,
    )
    try:
        body = response.json()
    except ValueError:
        body = {}

    if not response.ok or not body.get("success", True):
        reasons = "; ".join(error.get("message", "") for error in body.get("errors", [])) or response.text[:300]
        print(f"Cloudflare answered {response.status_code} for {model}: {reasons}", file=sys.stderr)
        return None, ALLOWANCE_USED if response.status_code == 429 or "allocation" in reasons.lower() else REFUSED
    return body["result"], None


def describe(photo, close_up):
    """The vision model's description of the person, as a dict. {} if it could not give one; an
    exhausted allowance stops the restyle, since the drawing would be refused next anyway."""
    pictures = [photo] + ([close_up] if close_up else [])
    shown = [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(as_input(picture, 768)).decode()}}
        for picture in pictures
    ]
    result, problem = cloudflare(
        VISION,
        json={
            "messages": [{"role": "user", "content": [{"type": "text", "text": ASK}, *shown]}],
            "max_tokens": 700,
            "temperature": 0,  # as close to deterministic as the model allows: a progress photo and
            # an earlier one of the same person should get back close to the same description, not
            # a different word for the same earrings each time, which a seeded draw would then follow
            "chat_template_kwargs": {"enable_thinking": False},  # it is a reasoning model; the answer alone is enough
        },
    )
    if problem == ALLOWANCE_USED:
        sys.exit(problem)
    if not result:
        return {}
    text = result.get("response") or result.get("choices", [{}])[0].get("message", {}).get("content") or ""
    found = re.search(r"\{.*\}", text, re.S) if isinstance(text, str) else None
    try:
        return json.loads(found.group(0)) if found else {}
    except ValueError:
        return {}


def features(seen, outfit=None):
    """The person's features as one sentence for the drawing, and their body (whole-body avatars
    only): the vision model's overall impression, as one of BODIES.

    The measured ratios used to be appended here as a sentence about the taper. They are not any
    more — see the long note above `measure` in body_measure.py, which is where the reasoning
    lives. In short: over 32 photos that sentence came out the same for a shredded body and a
    severely obese one, so it told the drawing nothing it did not already have from the word, and
    told it something false at the heavy end. The ratios are still measured and still saved; they
    are evidence for the person, not adjectives for the prompt."""
    parts = [{"senior": "an older adult with a mature face", "teen": "a teenager", "child": "a child"}.get(
        str(seen.get("age", "")).lower(), "an adult with a grown-up face")]
    if present(seen.get("face")):
        parts.append(str(seen["face"]))

    skin = str(seen.get("skin", "")) if present(seen.get("skin")) else ""
    shade = str(seen.get("skin_shade", "")).strip()
    if shade.isdigit() and 1 <= int(shade) <= len(SKIN_SHADES):
        skin = f"{skin} ({SKIN_SHADES[int(shade) - 1]})".strip()
    if skin:
        parts.append(f"{skin} skin")

    if present(seen.get("hair")):
        parts.append(f"hair: {seen['hair']}")
    if present(seen.get("head_covering")):
        parts.append(f"wearing {seen['head_covering']}")

    for key in ("eyes", "eyebrows"):
        if present(seen.get(key)):
            described = str(seen[key])
            parts.append(described if key in described.lower() else f"{described} {key}")
    facial_hair = seen.get("facial_hair")
    if present(facial_hair):
        stubble = "stubble" in str(facial_hair).lower()
        parts.append(f"{facial_hair}, drawn as a clearly visible shadow of fine stubble texture over the jaw, chin and upper lip" if stubble else str(facial_hair))
    for key in ("glasses", "jewelry", "makeup"):
        if present(seen.get(key)):
            parts.append(str(seen[key]))
    if not outfit and present(seen.get("top")):
        parts.append(f"wearing {seen['top']}")

    build = str(seen.get("build", "")).lower()
    body = next((phrase for word, phrase in BODIES.items() if word in build), None) if outfit else None
    return "; ".join(parts), body


def prompt_for(facts, body=None, outfit=None, close_up=True):
    """What the image model is asked to draw."""
    mode = "body" if outfit else "portrait"
    return (
        "Turn the person in image 1 into a cute 3D avatar character" + (f" with their real body: {body}" if body else "") + ". "
        + ("Image 2 is a close-up of their face. " if close_up else "")
        + STYLE
        + (f"It must be unmistakably this person, with exactly these features and nothing added: {facts}. " if facts else
           "It must be unmistakably this person, with their own hairstyle, hair color, skin tone, eye color and face. ")
        + (f"Keep their body exactly this size and shape: {body}. " if body else "")
        + (f"Change only their clothes, to {OUTFITS[outfit]}. " if outfit else "")
        + FRAMING[mode]
        + BACKDROP
    )


def seed_for(name):
    """A number from a character's name, steady for as long as the name is: Cloudflare's image
    model draws a recognizably different picture for a different seed, and close to the same
    picture again for the same one (checked on 3 repeats, see faces-progress-accuracy memory).
    Reusing it for everyone's repeat photos keeps the pose, crop and rendering choices steady from
    one of their visits to the next, so a later photo differs from an earlier one mostly where
    they actually do, which is what makes two avatars worth comparing as progress."""
    return int(hashlib.sha256(name.encode()).hexdigest(), 16) % (2**31)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("photo", type=Path, help="photo of the person")
    parser.add_argument("out", type=Path, help="where to save the avatar still (PNG)")
    parser.add_argument("--outfit", choices=OUTFITS, help="whole-body avatar in this gym outfit (default: a portrait in their own clothes)")
    parser.add_argument("--name", help="the character's name, steady across their repeat photos (default: the photo's file name); fixes the render's seed")
    args = parser.parse_args()

    photo = ImageOps.exif_transpose(Image.open(args.photo)).convert("RGB")
    found = faces(photo)
    lit = ImageStat.Stat(photo.convert("L").crop(tuple(round(v) for v in found[0])) if found else photo.convert("L")).mean[0]
    if lit < (DARKEST_FACE if found else DARKEST_PHOTO):
        print(f"The photo is too dark: brightness {lit:.0f} of 255.", file=sys.stderr)
        sys.exit(TOO_DARK)  # before any call, so it costs nothing
    close_up = face_close_up(photo, found[0]) if found else None
    if not found:
        print("No face found in the photo, so the avatar is drawn without a close-up of it.")

    measured = None
    if args.outfit and found:
        mask, joints = apple_vision.body(photo)
        measured = body_measure.measure(mask, joints, found[0])
        if not measured:
            print("Could not measure a body shape from the photo (loose clothing, or a pose Vision can't read); using the description's impression alone.")

    seen = describe(photo, close_up)
    if not seen:
        print("The description failed, so the avatar is drawn from the photo alone.", file=sys.stderr)
    facts, body = features(seen, args.outfit)
    print(f"Features: {facts}" + (f"\nBody: {body}" if body else ""))

    width, height = SIZES["body" if args.outfit else "portrait"]
    seed = seed_for(args.name or args.photo.stem)
    for limit in PHOTO_LIMITS:
        pictures = [as_input(photo, limit)] + ([as_input(close_up, CLOSE_UP_SIZE)] if close_up else [])
        result, problem = cloudflare(
            MODEL,
            data={"prompt": prompt_for(facts, body, args.outfit, bool(close_up)), "width": width, "height": height, "seed": str(seed)},
            files={f"input_image_{index}": (f"{index}.png", picture, "image/png") for index, picture in enumerate(pictures)},
        )
        if result or problem == ALLOWANCE_USED:
            break
    if not result:
        sys.exit(problem)

    still = Image.open(io.BytesIO(base64.b64decode(result["image"]))).convert("RGB")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    still.save(args.out, "PNG")

    # The measured numbers, not the avatar, are what is fit to compare photo to photo: see the
    # "Body shape, measured" section of the README. A person scanning on different days gets one
    # of these files per scan; an app wanting a progress chart reads this, not the picture.
    measurements = args.out.with_suffix(".json")
    if measured:
        measurements.write_text(json.dumps({"measured_at": datetime.now(timezone.utc).isoformat(), **dataclasses.asdict(measured)}, indent=2) + "\n")
    else:
        measurements.unlink(missing_ok=True)

    print(f"{args.photo} -> {args.out} ({still.width} x {still.height})")


if __name__ == "__main__":
    main()
