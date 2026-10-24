"""Write the golden vectors the Swift port of the prompt pipeline is tested against.

The iOS app builds the same prompt this pipeline does, and it has to build it *exactly* the same:
the image model's seed comes from the person's name, so the same name and the same prompt draw the
same character, and that steadiness is the only reason two avatars months apart are worth putting
side by side. A prompt that differs by a word is a different character, and the comparison is void.

So the Swift tests do not assert against figures someone typed out by hand, or against whatever
the port itself produced on the day it was written. They assert against this file, and this file
is written by running the Python. Regenerate it with:

    pipeline/.venv/bin/python pipeline/prompt_golden.py

and run the Swift tests, which will fail loudly if the port and the pipeline have drifted apart.
Anything added to restyle.py that changes a prompt — a new feature key, a reworded STYLE, a
different body phrase — needs a case here, or the port can drift on it unnoticed.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import body_measure
import restyle

OUT = Path(__file__).parent.parent.parent / "faces-ios/Tests/Fixtures/prompt-golden.json"

# A measured build to sharpen the body word with, for the cases that have one.
MEASURED = body_measure.Build(
    waist_to_shoulder=0.661, hip_to_shoulder=0.776, waist_to_hip=0.852, shoulder_to_head=3.87
)

# (name, what the vision model said, outfit, whether a measurement was taken)
CASES = [
    # everything present, portrait: the ordinary case, and the one that pins clause order
    ("portrait-full", {
        "age": "adult", "skin": "warm brown", "skin_shade": "7",
        "face": "oval face with high cheekbones", "hair": "short black twists",
        "head_covering": "none", "eyes": "dark brown almond-shaped eyes",
        "eyebrows": "thick dark brown", "facial_hair": "light stubble", "glasses": "none",
        "jewelry": "small gold stud in the left ear", "makeup": "none",
        "top": "a white crew-neck t-shirt", "bottom": "olive cargo pants",
        "shoes": "white trainers", "build": "slim",
    }, None, False),
    # whole body with a measured taper, a head covering hiding the hair, and a beard
    ("body-with-measurement", {
        "age": "senior", "skin": "fair pink", "skin_shade": "2",
        "face": "round face with full cheeks", "hair": "hidden",
        "head_covering": "black baseball cap worn forwards", "eyes": "blue hooded eyes",
        "eyebrows": "thin grey", "facial_hair": "full bushy white beard",
        "glasses": "round tortoiseshell glasses", "jewelry": "none", "makeup": "none",
        "top": "a navy polo shirt", "bottom": "khaki trousers", "shoes": "brown shoes",
        "build": "heavyset",
    }, "tee-pants", True),
    # the model answering vaguely: an out-of-range shade, empty strings, and its words for absence
    ("body-sparse", {
        "age": "teen", "skin": "", "skin_shade": "11",
        "face": "not visible", "hair": "bob with straight bangs, light brown",
        "head_covering": "n/a", "eyes": "green eyes", "eyebrows": "", "facial_hair": "none",
        "glasses": "", "jewelry": "unknown", "makeup": "red lipstick",
        "top": "a yellow puffer jacket", "bottom": "blue jeans", "shoes": "not visible",
        "build": "athletic",
    }, "tank-shorts", False),
    # a whole-body avatar whose build the model could not read: the prompt then has no body
    # clause at all, while still being framed and dressed as a whole body
    ("body-without-build", {
        "age": "adult", "skin": "medium tan", "skin_shade": "5",
        "face": "square face with a strong jaw", "hair": "short skin fade, black",
        "head_covering": "none", "eyes": "brown eyes", "eyebrows": "straight black",
        "facial_hair": "none", "glasses": "none", "jewelry": "none", "makeup": "none",
        "top": "a grey hoodie", "bottom": "not visible", "shoes": "not visible",
        "build": "not visible",
    }, "tee-pants", True),
    # a build word the model qualified, which the substring match has to resolve the same way
    # Python's dictionary order resolves it
    ("body-two-build-words", {
        "age": "adult", "skin": "deep brown", "skin_shade": "9",
        "face": "long oval face", "hair": "short locs", "head_covering": "none",
        "eyes": "dark brown eyes", "eyebrows": "thick black", "facial_hair": "short black beard",
        "glasses": "none", "jewelry": "none", "makeup": "none", "top": "a black t-shirt",
        "bottom": "grey joggers", "shoes": "black trainers",
        "build": "athletic but stocky",
    }, "tee-pants", False),
    # the description failing entirely: the prompt still has to name what must survive
    ("nothing-seen", {}, None, False),
]

# Ratios either side of shape_phrase's band edges, including the edges themselves, since a
# strict-versus-non-strict comparison there is invisible until someone lands exactly on one.
SHAPES = [
    (0.55, 0.80), (0.77, 0.87), (0.78, 0.88), (0.90, 1.00), (0.95, 1.05), (1.20, 1.10),
    (0.779, 0.879), (0.781, 0.881), (0.949, 1.049), (0.951, 1.051),
]

# Names chosen for the shapes of their bytes, not their meaning: empty, multi-byte, long, and
# one that is only a combining character, since the seed hashes UTF-8 and Swift's String is not
# a byte array.
NAMES = ["p11", "lean-man", "Lior Baum", "", "emoji 🙂", "a" * 200, "x", "Ωmega", "é", "🙂🙂🙂"]


def main():
    out = {
        "ask": restyle.ASK,
        # every shade, so that changing one hex in the port fails a test rather than quietly
        # colouring somebody the wrong tone
        "skinShades": restyle.SKIN_SHADES,
        "outfits": restyle.OUTFITS,
        # in order: the body words are matched by substring, so which one wins when a build names
        # two of them depends on the order they are tried in
        "bodyWords": list(restyle.BODIES.keys()),
        "bodies": restyle.BODIES,
        "sizes": {name: list(size) for name, size in restyle.SIZES.items()},
        "seeds": {name: restyle.seed_for(name) for name in NAMES},
        # No shapePhrases any more: the measured sentence was withdrawn from the prompt
        # entirely. Its absence is itself the thing to pin, so the Swift port cannot quietly
        # reintroduce one. See the note above `measure` in body_measure.py.
        "shapePhrases": [],
        "measurementUsed": {
            "waist_to_shoulder": MEASURED.waist_to_shoulder,
            "hip_to_shoulder": MEASURED.hip_to_shoulder,
            "waist_to_hip": MEASURED.waist_to_hip,
            "shoulder_to_head": MEASURED.shoulder_to_head,
        },
        "cases": [],
    }

    for name, seen, outfit, measured in CASES:
        facts, body = restyle.features(seen, outfit)
        out["cases"].append({
            "name": name, "seen": seen, "outfit": outfit, "measured": measured,
            "facts": facts, "body": body,
            "promptWithCloseUp": restyle.prompt_for(facts, body, outfit, True),
            "promptWithoutCloseUp": restyle.prompt_for(facts, body, outfit, False),
        })

    # The expression clauses, so the Swift port of the mood faces is held to the same standard
    # as the rest of the prompt: identical text, or the phone draws a different character from
    # the one the Mac drew for the same person.
    import mood_faces
    out["expressions"] = mood_faces.EXPRESSIONS
    out["expressionTemplate"] = mood_faces.EXPRESSION
    out["moodExpressions"] = mood_faces.MOODS
    out["expressionPrompts"] = {
        expression: mood_faces.prompt_for(
            out["cases"][1]["facts"], out["cases"][1]["body"], "tee-pants", True,
            expression=expression)
        for expression in mood_faces.EXPRESSIONS
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {OUT}")
    print(f"  {len(out['cases'])} cases, {len(out['seeds'])} seeds, "
          f"{len(out['shapePhrases'])} shape phrases, {len(out['skinShades'])} skin shades")


if __name__ == "__main__":
    main()
