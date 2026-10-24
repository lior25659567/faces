"""Build and measure a set of whole-body photos, to calibrate the body-shape bands against.

Run with --vision to also record what the vision model calls each body, which costs one further
call per photo and is what any claim about the bands earning their place has to be argued from.

`body_measure.shape_phrase` turns three measured ratios into words, and the words have thresholds
in them. Those thresholds were set from four photos, which is not a calibration — it is an
anecdote with arithmetic on top. This makes a real set: a known build and a known sex for each
photo, every one taken under the conditions the app asks for, so the bands can be chosen from
where the groups actually separate rather than from where they looked about right.

The photos are generated rather than collected, which is a limit worth stating plainly: a model's
idea of "stocky" is not a measurement of a stocky person, and no set built this way can tell us
about real bodies it has never seen. What it can do is show whether a threshold separates bodies
the model drew as different, and whether the same threshold works for men and women — which is
the specific thing the current bands do not know.

Re-runnable: photos already on disk are kept, so a second run costs nothing and only measures.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import io, base64
from PIL import Image, ImageOps

import apple_vision
import body_measure
import restyle

OUT = Path(__file__).parent.parent / "reference/bodies"

# The capture conditions the Scan screen asks for. Without the gap at the arms the measurement
# refuses the photo, which is the whole reason it is spelled out this heavily.
POSE = ("standing straight and facing the camera, both arms held well away from the sides with a "
        "wide clear gap of empty background between each arm and the torso, the whole body from "
        "head to feet in frame, plain pale grey studio background, flat even lighting, "
        "full-length photograph")

# The body-fat and musculature axis, from one end to the other. These are the words the image
# model is given, not the words the vision model answers with (restyle.py's BODIES): the point is
# to produce bodies that really differ, then see what the measurement makes of them.
BUILDS = {
    "underweight": "severely underweight and bony, with visible collarbones and ribs and very "
                   "thin arms and legs",
    "slim": "slim and thin, with narrow shoulders, thin arms and thin legs",
    "lean": "lean and toned with low body fat and light muscle definition, not bulky",
    "athletic": "athletic and fit with clearly defined muscles and broad shoulders, like a "
                "regular gym-goer",
    "muscular": "heavily muscular like a competitive bodybuilder, with very broad shoulders, "
                "large arms and a thick chest",
    "shredded": "extremely lean and shredded with very low body fat, visible six-pack abs and "
                "sharply defined muscles",
    "average": "an ordinary average build, neither thin nor heavy, with no visible muscle",
    "stocky": "short and stocky, solid and thick-set through the chest and waist",
    "overweight": "noticeably overweight, with a soft round belly and a thick waist",
    "obese": "severely obese and very large, with a very big round belly, wide hips and heavy "
             "arms and legs",
}

WEAR = {
    "man": "a fitted plain t-shirt and fitted shorts",
    "woman": "a fitted athletic tank top and fitted shorts",
}

# Varied so the set is not one face measured six times: skin tone and age change the face box,
# which is the scale every ratio is expressed against.
# Twelve different people, so a build is not measured twelve times on one face. The variation is
# deliberate and not decoration: skin tone changes how cleanly Vision separates a person from the
# background, hair changes the face box that every ratio is scaled against, and both are places
# where a measurement can be quietly worse for some people than others.
PEOPLE = [
    ("man", "a Black man in his twenties with dark brown skin and a short afro"),
    ("man", "a white man in his forties with fair skin and short grey hair"),
    ("man", "an East Asian man in his thirties with light tan skin and straight black hair"),
    ("man", "a South Asian man in his fifties with brown skin, black and grey hair and a beard"),
    ("man", "a Latino man in his twenties with olive skin and dark wavy hair"),
    ("man", "a bald white man in his fifties with ruddy fair skin and a thick red beard"),
    ("woman", "a white woman in her twenties with fair skin and blonde hair tied back"),
    ("woman", "a Black woman in her thirties with deep brown skin and long box braids"),
    ("woman", "an East Asian woman in her forties with light skin and a straight black bob"),
    ("woman", "a South Asian woman in her twenties with brown skin and long dark hair tied back"),
    ("woman", "a Latina woman in her thirties with olive skin and curly brown hair"),
    ("woman", "a white woman in her sixties with pale skin and short grey hair"),
]

# A second group holding the build still and moving only skin tone, to see whether the
# segmentation — and so the measurement — works as well on dark skin as on light. Vision has to
# separate a person from the background, and that is a contrast problem before it is anything
# else; a measurement that quietly degrades for darker skin would be the kind of unfairness that
# never shows up unless it is looked for.
SKIN_TONES = [
    "very pale white skin", "light beige skin", "medium tan skin",
    "warm brown skin", "deep brown skin", "very dark brown-black skin",
]

def generate(name, prompt):
    path = OUT / f"{name}.png"
    if path.exists():
        return path, False
    result, problem = restyle.cloudflare(
        restyle.MODEL,
        data={"prompt": prompt, "width": 768, "height": 1360,
              "seed": str(restyle.seed_for(name))})
    if not result:
        print(f"  {name}: generation failed ({problem})", file=sys.stderr)
        return None, False
    Image.open(io.BytesIO(base64.b64decode(result["image"]))).convert("RGB").save(path)
    return path, True


def measure(path):
    photo = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    faces = apple_vision.faces(photo)
    if not faces:
        return None, "no face"
    mask, joints = apple_vision.body(photo)
    build = body_measure.measure(mask, joints, faces[0])
    if build is None:
        # body_measure decides what it can work with; saying why here is only for the report
        if "leftShoulder" not in joints or "rightShoulder" not in joints:
            return None, "no shoulders"
        if not any(j in joints for j in ("leftHip", "rightHip", "root")):
            return None, "no hip line"
        return None, "arms too close"
    return build, None


def vision_word(path):
    """What the vision model calls this body — the word the prompt actually uses.

    Without it the central question cannot be answered: the measured phrase is appended to this
    word, so what matters is not whether the ratios separate builds but whether they separate
    builds the WORD gets wrong. A measurement that fires exactly when the word already says
    "obese" adds nothing to the drawing; one that corrects the word is the whole point. Costs one
    vision call per photo, so it is opt-in.
    """
    photo = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    found = apple_vision.faces(photo)
    close_up = restyle.face_close_up(photo, found[0]) if found else None
    seen = restyle.describe(photo, close_up)
    return seen.get("build")


def main():
    # Every build once as a man and once as a woman — the pairing is the point, since the open
    # question is whether one threshold can serve both — with the faces rotated so the set is not
    # a single person measured twelve times. Twelve photos is a dozen calls against the user's
    # daily allowance; more can be added at the boundaries if these do not separate cleanly.
    men = [person for person in PEOPLE if person[0] == "man"]
    women = [person for person in PEOPLE if person[0] == "woman"]
    wanted = []
    # Every build, once as a man and once as a woman, rotating through the faces.
    for index, build in enumerate(BUILDS):
        for group in (men, women):
            sex, who = group[index % len(group)]
            wanted.append((f"{sex}-{build}", sex, who, build))
    # And the build held still while the skin tone moves.
    for index, tone in enumerate(SKIN_TONES):
        for sex, hair in (("man", "short dark hair"), ("woman", "dark hair tied back")):
            who = f"a {sex} in their thirties with {tone} and {hair}"
            wanted.append((f"tone-{index}-{sex}", sex, who, "average"))

    print(f"{len(wanted)} photos wanted in {OUT}")
    rows, made = [], 0
    for index, (name, sex, who, build) in enumerate(wanted, 1):
        prompt = f"{who}, {BUILDS[build]}, wearing {WEAR[sex]}, {POSE}"
        path, fresh = generate(name, prompt)
        made += fresh
        if fresh:
            time.sleep(1)  # the allowance is the user's, and a burst invites a 429
        if path is None:
            continue
        measured, why = measure(path)
        row = {"name": name, "sex": sex, "build": build, "who": who,
               "file": str(path.relative_to(OUT.parent.parent))}
        if "--vision" in sys.argv:
            row["vision_build"] = vision_word(path)
        if measured is None:
            row["refused"] = why
        else:
            row.update(waist_to_shoulder=measured.waist_to_shoulder,
                       hip_to_shoulder=measured.hip_to_shoulder,
                       waist_to_hip=measured.waist_to_hip,
                       shoulder_to_head=measured.shoulder_to_head)
        rows.append(row)
        state = why if measured is None else f"s/head {measured.shoulder_to_head:.2f}"
        print(f"  [{index}/{len(wanted)}] {name:28} {state}")

    (OUT / "measured.json").write_text(json.dumps(rows, indent=2) + "\n")
    usable = [r for r in rows if "refused" not in r]
    print(f"\n{made} generated, {len(usable)}/{len(rows)} measured")
    print(f"wrote {OUT / 'measured.json'}")


if __name__ == "__main__":
    main()
