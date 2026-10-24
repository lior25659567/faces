#!/usr/bin/env python
"""Draw a generic avatar demonstrating each exercise, cut out so the background is transparent.

The app shows one of these beside a movement's name while somebody is working out, so they have to
read as the same character family as the person's own avatar. That is why `restyle.STYLE` is
imported and reused verbatim rather than paraphrased here, and why the prompt is assembled in the
same order `restyle.prompt_for` uses: lead sentence, style, who the character is, the position, the
framing, the backdrop. A second house look would be obvious the moment a demonstration sat next to
an avatar on the same screen.

The figure is deliberately nobody. This is a diagram of a movement, not a picture of the user, so
the character is one invented mascot described identically for every exercise, drawn with a single
seed (see SEED) so the twelve pictures are the same character twelve times rather than twelve
strangers.

The positions are condensed from reference/exercises/form-guidelines.json, whose entries come from
StrongLifts, ACE and NASM. They are condensed rather than quoted because each entry runs to several
hundred words of joint angles, and a prompt that long is paraphrased by the image model instead of
followed. Every `Demo` below records which field it was condensed from, so a reviewer can check a
pose against its source. Getting this wrong is worse than showing nothing: somebody will copy the
picture.

Two steps, only the first of which costs anything:
  1. Draw: FLUX draws the character in the position. One Workers AI call per exercise, at most
     BUDGET of them, and never for a picture already on disk.
  2. Cut out: the flat backdrop is keyed out and Apple Vision's person segmentation
     (apple_vision.body) is folded in, giving a PNG with a real alpha channel. This step is free,
     so the raw drawing is kept beside the cut-out and --recut reworks the alpha without redrawing.

What the first run actually produced, because it decides whether this is worth continuing. Twelve
images were drawn, the whole budget. The house look came through on all of them and the cut-outs
are clean: the bar, the plates, the dumbbells and the bench all survive, which was the thing most
at risk. The form did not. Two of the ten that could be cut out show the movement correctly — the
overhead press and the curl — and the other eight show a neighbouring position or a named mistake:
a partial squat, a dead hang instead of the top of a pull-up, a high plank instead of a forearm
plank, a lateral raise with the arms overhead. Bending the prompt around the one fact that matters
(see `Demo.key`) did not fix the lateral raise on a second attempt.

The pattern is that this model places a character, an outfit and a piece of equipment reliably and
specific joint angles unreliably, and no wording tried here changed that.

Thirteen positions have now been drawn and judged, and the three that came out right sort cleanly
from the ten that did not:

    correct   curl (top), curl (bottom), overhead press (lockout)
    wrong     squat, deadlift, bench press x2, bent-over row, pull-up, plank, lunge,
              lateral raise, overhead press (racked)

Every correct one is a character STANDING UPRIGHT, FACING THE CAMERA, SYMMETRICAL, with the
movement happening only at the elbows or the shoulders. Every wrong one asks for something else:
lying on a bench, hinged at the hips, hanging from a bar, held horizontal, seated, or carrying a
joint angle the body's own orientation has to be inferred from. The bench press is the clearest
case — one attempt put the barbell through the character's head, and the pair faced opposite ways,
so they would have flipped when cross-faded.

So the next calls are worth spending on the standing symmetrical movements still undrawn, and are
not worth spending on the lying, hinged or hanging ones. Those need a different source of
pictures, which is what reference/exercises/ANIMATION.md is about. Anyone picking this up
should expect to spend calls on repeats rather than on coverage, and should treat `--review` as the
part that does the work: every drawing is judged against the form notes by a person before the app
can show it, because a wrong demonstration gets copied.

Needs CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN in the environment, the same two
restyle.py needs.

    pipeline/.venv/bin/python pipeline/exercise_art.py --dry-run
    pipeline/.venv/bin/python pipeline/exercise_art.py --limit 2
    pipeline/.venv/bin/python pipeline/exercise_art.py
    pipeline/.venv/bin/python pipeline/exercise_art.py --recut
"""

import argparse
import base64
import dataclasses
import io
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

import apple_vision
import restyle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GUIDELINES = ROOT / "reference" / "exercises" / "form-guidelines.json"
ART = ROOT / "reference" / "exercise-art"
RAW = ART / "raw"
INDEX = ART / "index.json"
# The app looks art up by the exercise's name, so the names here have to be the catalogue's names
# exactly. Checked against the sibling checkout when it is there; absence is reported, not assumed
# to be agreement.
CATALOGUE = ROOT.parent / "faces-ios" / "Sources" / "Faces" / "Model" / "ExerciseCatalogue.swift"

# The hard ceiling on Workers AI calls in one run. Every generated image is a real call against the
# user's daily allowance, and this script exists to be re-run, so it refuses to draw more than the
# twelve pictures it is for however it is invoked.
BUDGET = 12

# Canvas shapes, in the same spirit as restyle.SIZES: a standing press needs height, a plank needs
# width, and a square canvas spent on either wastes half of it on backdrop that is about to become
# transparent anyway.
SHAPES = {"tall": (768, 1152), "square": (1024, 1024), "wide": (1152, 768)}

# One seed for the whole set, from restyle.seed_for. Its docstring records that the image model
# draws close to the same picture again for the same seed and a recognizably different one for a
# different seed; here the thing that must stay the same from picture to picture is the character,
# so the whole family shares a seed and only the prompt's position changes.
CHARACTER = "generic exercise demonstrator"
SEED = restyle.seed_for(CHARACTER)

# Who the character is. Written once and repeated verbatim for every exercise, for the same reason
# restyle sends the vision model's facts with every draw: asked for a cute character and nothing
# else, the model invents a different one each time.
#
# The skin is the middle of restyle.SKIN_SHADES and the outfit is one of restyle.OUTFITS, so the
# demonstrator is dressed out of the same wardrobe as a whole-body avatar. The build is
# BODIES["average"] rather than "athletic": visible muscle tells a reader nothing about a movement,
# and the note above BODIES records that the model drifts slim regardless, which for a diagram is
# harmless. "Gender-neutral" is deliberate — a demonstration that reads as a man implies the
# movement is a man's.
GENERIC = (
    "The character is a generic invented mascot of no particular identity and is not a portrait of any real "
    "person: a gender-neutral adult with plain friendly features, a smooth rounded head, short simple dark brown "
    f"hair, medium tan skin ({restyle.SKIN_SHADES[4]}), {restyle.BODIES['average']}, no facial hair, no glasses, "
    f"no jewelry, no tattoos and no printed text or logos anywhere, wearing {restyle.OUTFITS['tank-shorts']}. "
)

FRAMING = (
    "Framing: the whole body from head to feet and all of the equipment inside the frame, {camera}, "
    "centered with clear empty space on every side, nothing cropped or touching the edges of the picture. "
)

# The backdrop is named by hex and asked to be flat and shadowless because the cut-out keys it out
# by colour: a gradient, a floor line or a contact shadow all survive the key and arrive in the app
# as grey smears around the character's feet. It is the pastel teal of restyle.BACKDROP, and unlike
# that line it does not forbid objects, since the equipment is the point.
BACKDROP_RGB = (167, 218, 213)
BACKDROP = (
    "Background: one completely flat, even, uniform pastel teal (#a7dad5) filling the entire frame edge to edge, "
    "with no gradient, no vignette, no floor, no horizon line, no cast shadow or contact shadow anywhere, no gym, "
    "no walls, no props, no other people and no text, numbers or logos."
)


@dataclasses.dataclass(frozen=True)
class Pose:
    """One drawn position of a movement.

    A movement has two positions its source actually documents — `start_position` and
    `end_position` — and the app shows them as a pair so the reader sees where the rep begins and
    where it ends. There is deliberately no third drawing between them: no entry in
    form-guidelines.json states the joint angles halfway through a rep, so a middle frame would be
    this script inventing a position and presenting it as instruction. Two documented positions
    alternating is a demonstration; three positions where one is made up is a lie with a picture
    on it.

    `label` names the position in two or three words, for the control that switches between them.
    `cue` is the one line a person reads under the picture, which is not the prompt: the prompt is
    written for an image model that needs every angle spelled out, and the cue is written for
    somebody holding a barbell.
    """

    phase: str
    label: str
    cue: str
    key: str
    pose: str


@dataclasses.dataclass(frozen=True)
class Demo:
    """One movement's drawings.

    `phase` names the field of form-guidelines.json that `pose` was condensed from, so the
    condensation can be audited against its source. Where the two positions of a lift differ in how
    much they show, the drawn one is the position a reader is most likely to get wrong: the bottom
    of a squat rather than standing with a bar on the back, the top of a curl rather than arms
    hanging by the thighs.

    `Pose.key` is the one piece of geometry that makes a picture a demonstration of this movement
    rather than of a neighbouring one, and it exists because the first lateral raise drawn here
    came back with the arms overhead. The instruction was in `pose`, correctly, but it was one
    clause among twenty and the model averaged it away. `key` is repeated on its own immediately
    before and after `pose`, where a short emphatic sentence is actually followed.

    The shape, the camera and the kit are shared by every position of one movement. A squat seen
    from the side at the top and from the front at the bottom would read as two different
    exercises, and the whole point of drawing both is that they are the same rep.
    """

    shape: str
    camera: str
    kit: str
    poses: tuple


# The twelve drawn, and after them the two deliberately left out. Both lists together have to cover
# form-guidelines.json exactly; `check_names` enforces it, so adding a fifteenth movement to the
# guidelines makes this script refuse rather than quietly ignore it.
DEMOS = {
    "Barbell Back Squat": Demo(
        shape="square",
        camera="seen from the side and slightly in front",
        kit="a loaded Olympic barbell lying across the upper back behind the neck, resting on the muscle of the upper "
            "back and clear of the neck bones, with a round weight plate on each end; no squat rack",
        poses=(
        Pose(
            phase="start_position",
            label="Standing, bar racked",
            cue="Stand tall with the bar on your upper back, feet shoulder-width, toes turned out a little.",
            key="Standing fully erect: hips and knees locked at ~180 degrees, shins vertical, bar in a vertical line over the MIDFOOT.",
            pose="Standing fully erect, bar racked across the upper back. Heels directly under the shoulder joints, shoulder-width, NOT hip-width; toes out about 30 degrees; whole foot flat. Bar centred on the back and in a vertical line over the MIDFOOT: low-bar across the scapular spine between traps and rear deltoids, or high-bar on top of the traps. Hands just outside the shoulders. Elbows pointed down and BEHIND the torso. Torso vertical, chest raised, upper back arched, shoulder blades squeezed. Natural lordotic arch. Hips locked ~180 degrees, knees locked ~180 degrees, ankles neutral ~90 degrees, shins vertical. Head in line with the torso, gaze straight ahead and slightly down.",
        ),
        Pose(
            phase="end_position",
            label="Hips below knees",
            cue="Sit down until your hip crease is lower than the top of your kneecaps.",
            key="the hips are BELOW the knees at the bottom of the squat, the shins angled forward and the torso leaning "
                "forward",
            pose="at the bottom of the rep: the hip crease is lower than the top of the kneecaps so the hips are below the "
                 "knees, the shins are inclined forward, the knees are pushed out to the sides and track over and slightly "
                 "past the toes, the torso leans about forty degrees forward of vertical with the chest up and the lower "
                 "back holding its natural inward arch, the hands grip the bar just outside the shoulders, the whole foot "
                 "is flat on the floor with the feet shoulder-width apart and the toes turned out about thirty degrees, "
                 "and the head is in line with the torso with the gaze straight ahead and slightly down",
        ),
        ),
    ),
    "Conventional Barbell Deadlift": Demo(
        shape="square",
        camera="seen from the side",
        kit="a loaded Olympic barbell resting on the floor with a round weight plate on each end",
        poses=(
        Pose(
            phase="start_position",
            label="Bar on the floor",
            cue="Crouch over the bar with straight arms, hips higher than your knees, shins touching the bar.",
            key="the bar is still resting on the floor and the character is crouched over it with the arms hanging "
                "straight, the hips higher than the knees but well below the shoulders and the back flat",
            pose="crouched over the bar at the start of the lift: the bar is on the floor over the middle of the feet with "
                 "the shins lightly touching it, the feet are hip-width apart with the toes turned out slightly and flat "
                 "on the floor, the hands grip the bar just outside the shins at about shoulder width with both palms "
                 "facing back, the arms are completely straight and hang vertically with the shoulders slightly in front "
                 "of the bar, the hips are higher than the knees but well below the shoulders so it looks like a half "
                 "squat rather than a deep squat, the torso is inclined about thirty degrees above horizontal with the "
                 "chest lifted and the lower back holding its normal inward curve, and the head is in line with the spine "
                 "with the gaze on the floor a short way ahead",
        ),
        Pose(
            phase="end_position",
            label="Standing at lockout",
            cue="Stand up straight with the bar against your upper thighs, shoulders back, no leaning back.",
            key="Hips and knees are both straightened to 180 degrees with the bar at the hip crease, and the torso does NOT lean back behind vertical.",
            pose="Full lockout, standing tall. Hips and knees both fully straightened to 180 degrees but not hyperextended. Torso vertical, lower back holding its natural arch, chest up. Bar resting against the upper thighs at the hip crease. Arms hanging straight and perfectly vertical from the shoulders, elbows straight. Shoulders pulled back and down, not shrugged up toward the ears, and the torso does NOT lean back behind vertical. Head neutral and in line with the torso, gaze forward. Feet unchanged: same hip-width stance, toes still out 15 degrees, whole foot flat. The bar has travelled in a straight vertical line from over the midfoot to the hip crease.",
        ),
        ),
    ),
    "Flat Barbell Bench Press": Demo(
        shape="wide",
        # Straight side profile at bench height, not "side and slightly above". Three drawings
        # from the raised angle came back with the body skewed across the bench, once with the
        # barbell passing through the character's head — the model loses which way the body lies
        # as soon as the camera is off the horizontal.
        camera="seen from exactly the side at bench height, the character in flat profile with "
               "the bench drawn edge-on as a horizontal slab",
        kit="a flat weight bench and a loaded Olympic barbell with a round weight plate on each end; no rack uprights "
            "and no spotter",
        poses=(
        Pose(
            phase="start_position",
            label="Bar above shoulders",
            cue="Lie flat, squeeze your shoulder blades down, and hold the bar straight over your shoulders.",
            key="Lying on the bench with the elbows fully extended to ~180 degrees and the bar directly above the shoulder joints, not touching the chest.",
            pose="Lying supine on a flat bench, bar at arm's length above the shoulders, just unracked. Five points of contact: head lightly on the bench, upper back and shoulder blades pressed into it, buttocks on the bench, both feet flat. Feet flat, heels down, under or slightly behind the knees, about shoulder-width heel to heel, toes out roughly 30 degrees. Shoulder blades pinched together, pulled back and DOWN into the bench. Natural lumbar arch, about a hand's thickness of gap under the lower back, chest up. Eyes under the bar, neck neutral. Bar directly above the shoulder joints, arms vertical, elbows fully extended to ~180 degrees, wrists stacked straight over the elbows.",
        ),
        Pose(
            phase="end_position",
            label="Bar on mid-chest",
            cue="Touch the middle of your chest with your elbows tucked at about 75 degrees.",
            key="the bar is touching the middle of the chest, the forearms vertical and both feet flat on the floor",
            pose="at the bottom of the rep: lying face up along the bench with the bar touching the middle of the chest, "
                 "the elbows tucked to about seventy-five degrees from the torso rather than flared straight out and "
                 "sitting directly under the bar, the forearms vertical, the shoulder blades pinched together and pulled "
                 "down into the bench, the chest pushed up with a small natural arch in the lower back, and five points of "
                 "contact kept: the head resting lightly on the bench, the upper back on the bench, the buttocks on the "
                 "bench and both feet flat on the floor about shoulder-width apart with the heels down",
        ),
        ),
    ),
    "Standing Barbell Overhead Press (Shoulder Press)": Demo(
        shape="tall",
        camera="seen from the side and slightly in front",
        kit="a loaded Olympic barbell with a round weight plate on each end; no rack",
        poses=(
        Pose(
            phase="start_position",
            label="Bar on shoulders",
            cue="Stand tall with the bar on your front shoulders and your elbows tucked under it.",
            key="Standing with the bar resting across the FRONT DELTOIDS just below the collarbones, elbows almost directly UNDER the bar, hips and knees locked.",
            pose="Standing erect, bar resting across the front deltoids just below the collarbones. Heels about hip-width apart, flat on the floor, toes slightly out, roughly 15 degrees. Hands just outside the shoulders, a NARROW grip. Elbows almost directly UNDER the bar from the side, angled roughly 45 degrees in toward the midline from the front. Forearms vertical to the floor from both side and front; upper arms NOT horizontal. Wrists almost straight, knuckles rotated back about 75 degrees. Chest lifted by an arched upper back, lower back neutral, hips pushed slightly forward so the torso leans back a small amount. Hips and knees fully locked straight. Head neutral, gaze straight ahead.",
        ),
        Pose(
            phase="end_position",
            label="Locked out overhead",
            cue="Press straight up and finish with the bar over your shoulders, arms beside your ears.",
            key="the arms are locked completely straight overhead with the bar directly above the shoulders and the "
                "whole body in one straight vertical line",
            pose="at full lockout overhead: the bar is in a vertical line directly above the shoulder joints and above the "
                 "middle of the feet, the elbows are locked straight with the arms vertical and the upper arms beside the "
                 "ears, the shoulders are shrugged slightly up at the top, the whole body forms one straight vertical line "
                 "from the middle of the feet through the hips and shoulders to the bar with no backward lean, the hips and "
                 "knees are locked straight, the feet are hip-width apart and flat with the toes turned slightly out, and "
                 "the head is neutral and has come forward under the bar with the gaze straight ahead",
        ),
        ),
    ),
    "Barbell Bent-Over Row": Demo(
        shape="wide",
        camera="seen from the side",
        kit="a loaded Olympic barbell with a round weight plate on each end",
        poses=(
        Pose(
            phase="start_position",
            label="Bar on the floor",
            cue="Torso flat like a table, arms hanging straight down, bar over the middle of your feet.",
            key="The TORSO is HORIZONTAL and the straight arms hang vertically to the bar on the floor.",
            pose="Bent over a loaded barbell resting on the floor. Feet flat in a medium stance, wider than a deadlift and narrower than a squat, toes turned out about 30 degrees in line with the knees. The bar sits over the midfoot. Torso HORIZONTAL, parallel to the floor. Hips set HIGH, higher than a deadlift setup. Knees unlocked, slightly bent and held back out of the bar's path, pushed out to the sides. Lower back neutral with its natural arch, chest raised. Arms hang straight down, perfectly vertical, elbows fully extended, shoulder blades slightly spread. Double overhand grip. Head in line with the spine, gaze down and slightly forward.",
        ),
        Pose(
            phase="end_position",
            label="Bar at lower chest",
            cue="Pull the bar to the bottom of your chest and send your elbows back past your sides.",
            key="the torso is HORIZONTAL, parallel to the floor, and the bar has been pulled all the way up to touch the "
                "lower chest",
            pose="at the top of the pull: the torso is horizontal and parallel to the floor, the bar has travelled straight "
                 "up to touch the lower chest at the bottom of the breastbone, the elbows are pulled back behind the torso "
                 "and tucked in to about seventy-five degrees rather than flared out sideways, the shoulder blades are "
                 "squeezed together, the hips are set high, the knees are unlocked and only slightly bent and held back out "
                 "of the bar's path, the feet are flat and a little wider than hip-width with the toes turned out about "
                 "thirty degrees, the lower back holds its natural standing arch, and the head is in line with the spine "
                 "with the gaze down and slightly forward",
        ),
        ),
    ),
    "Pull-Up": Demo(
        shape="tall",
        camera="seen from the front and slightly to one side",
        kit="a plain fixed horizontal metal pull-up bar mounted high overhead, drawn as a short straight section with "
            "both ends inside the picture, with the feet well clear of the floor",
        poses=(
        Pose(
            phase="start_position",
            label="Full dead hang",
            cue="Hang with your arms straight and your shoulders pulled down away from your ears.",
            key="The arms are fully extended overhead at roughly 180 degrees in a full dead hang.",
            pose="Hanging at full stretch from a fixed overhead bar, feet clear of the floor. Hands slightly wider than shoulder-width, pronated with the palms facing away, thumbs wrapped. Arms fully extended overhead, elbows at roughly 180 degrees. Shoulder blades DEPRESSED and packed down the back, away from the ears, not shrugged. Torso hanging vertically, spine long and neutral, chest lifted, no sag in the lower back. Legs squeezed together and still, either straight and hanging directly below the body or knees bent with the ANKLES CROSSED behind. Head neutral, neck in line with the spine, gaze forward or slightly up at the bar. One rigid vertical line.",
        ),
        Pose(
            phase="end_position",
            label="Chin over the bar",
            cue="Chin clears the bar with the chest up - the body rises, the neck does not reach.",
            key="the character hangs from the bar with the feet off the floor and the arms bent, the chin above the bar",
            pose="at the top of the rep: the chin has cleared the top of the bar, the hands grip the bar slightly wider "
                 "than shoulder-width with both palms facing away from the body and the thumbs wrapped around, the elbows "
                 "are driven down and back past the ribs, the shoulder blades are squeezed together and pulled down away "
                 "from the ears, the chest is lifted toward the bar, the torso hangs vertically with no swing and no arch, "
                 "the legs are squeezed together with the knees bent and the ankles crossed behind, and the head is neutral "
                 "with the neck in line with the spine",
        ),
        ),
    ),
    "Dumbbell Bicep Curl (standing, both arms)": Demo(
        shape="tall",
        camera="seen from the front and slightly to one side",
        kit="one dumbbell in each hand",
        poses=(
        Pose(
            phase="start_position",
            label="Dumbbells at the thighs",
            cue="Stand tall with the dumbbells beside your thighs, palms forward, arms straight.",
            key="The elbows are fully extended at about 170-180 degrees, dumbbells hanging beside the thighs.",
            pose="Standing tall, feet flat and hip-width apart, about 15-25 cm between the insteps, toes pointing straight forward, knees soft at about 5-10 degrees of flexion. Pelvis neutral, ribcage stacked over the pelvis, small natural lumbar arch. Dumbbells hang at arm's length beside the thighs, handles roughly level with the mid-thigh and 2-5 cm clear of the leg. Elbows fully extended at about 170-180 degrees, upper arms vertical and pinned alongside the ribs, the elbow points facing straight back. Shoulders depressed and retracted, collarbones wide, traps long. Wrists straight, forearms supinated so the palms face forward. Head stacked over the shoulders, chin level, gaze straight ahead.",
        ),
        Pose(
            phase="end_position",
            label="Dumbbells at chest height",
            cue="Curl to chest height and keep your elbows still, right under your shoulders.",
            key="the dumbbells are at chest height with the elbows pinned against the sides and the upper arms still "
                "vertical",
            pose="at the top of the curl: both elbows are bent so the dumbbells finish at upper-chest height just in front "
                 "of the collarbones, the upper arms are still vertical and pinned alongside the ribs with the point of "
                 "each elbow directly under its shoulder and not travelled forward, the forearms angle up and slightly "
                 "inward with the palms facing the chest and the wrists straight and in line with the forearms, the "
                 "shoulders are pulled down and back with the neck long, the torso is upright with no lean back and no hip "
                 "thrust, the feet are flat and hip-width apart with the toes pointing forward and the knees soft, and the "
                 "gaze is level and forward",
        ),
        ),
    ),
    "Dumbbell Lateral Raise": Demo(
        shape="square",
        camera="seen from the front",
        kit="one light dumbbell in each hand",
        poses=(
        Pose(
            phase="start_position",
            label="Dumbbells at the thighs",
            cue="Stand tall with the dumbbells hanging beside your thighs, palms facing in, arms straight.",
            key="Arms hang straight down beside the thighs with the palms facing in toward the legs.",
            pose="Standing. Feet slightly wider than hip-width, about 25-35 cm apart, toes forward; knees slightly bent, weight even through both feet. Torso erect and vertical, ribcage stacked over the pelvis, natural lumbar curve. Shoulder blades pulled down and back. Arms hang straight down beside the thighs, elbows either fully extended or holding a slight 10-20 degree bend, dumbbells 5-10 cm clear of the outer thighs. Neutral grip, palms facing in toward the body, handles across the mid-palm, hands directly below the shoulder joints and about hip-width apart. Wrists straight. Head in line with the spine, chin level, gaze straight ahead.",
        ),
        Pose(
            phase="end_position",
            label="Arms at shoulder height",
            cue="Raise your arms out to the sides until they are level with your shoulders, then stop.",
            key="both arms are EXACTLY HORIZONTAL, straight out to the sides at shoulder height so the body makes a "
                "level T; the hands are no higher than the shoulders and the arms are NOT overhead and NOT raised above "
                "the head",
            pose="at the top of the raise: both arms are raised out to the sides until they are level with the shoulders "
                 "and roughly parallel with the floor so the body makes a wide T, each elbow holds the same slight bend of "
                 "about fifteen degrees it had at the start and sits a few degrees ahead of the forearm, the dumbbells are "
                 "level with the tops of the shoulders with the thumb side of each a touch higher than the little-finger "
                 "side, the palms face down and in, the wrists are straight and in line with the forearms, the shoulders "
                 "stay pulled down with the traps not shrugged up toward the ears, the torso is erect and vertical with no "
                 "lean back and no sideways sway, and the feet are flat and a little wider than hip-width with the toes "
                 "forward and the knees slightly bent",
        ),
        ),
    ),
    "Dumbbell Shoulder Press (seated overhead press)": Demo(
        shape="tall",
        camera="seen from the side and slightly in front",
        kit="an upright weight bench with a tall back support, and one dumbbell in each hand",
        poses=(
        Pose(
            phase="start_position",
            label="Dumbbells at the shoulders",
            cue="Sit back against the pad, dumbbells at shoulder height, elbows pointing down and forward.",
            key="Elbows are bent to about 90 degrees and point down and FORWARD, 30-45 degrees in front of the body.",
            pose="Seated upright on a bench with a back pad; head, shoulders and glutes all touching the pad. Feet flat on the floor hip- to shoulder-width apart, shins near vertical. Dumbbells at shoulder level, handles roughly level with the top of the shoulders and ears, just outside the shoulder line, hands shoulder-width or slightly wider. Pronated grip, palms facing forward. Elbows flexed to about 90 degrees, pointing down and forward, roughly 30-45 degrees in front of the frontal plane. Forearms near vertical under the dumbbells. Wrists straight. Shoulder blades pulled down and back. Spine neutral with its natural lumbar curve. Head against the pad, chin level, gaze straight ahead.",
        ),
        Pose(
            phase="end_position",
            label="Arms locked overhead",
            cue="Press straight up until your arms are locked, the weights right above your shoulders.",
            key="the character is SEATED on a bench with a back support and the arms are locked straight overhead",
            pose="at full lockout overhead: seated upright with the head, shoulders and buttocks all touching the back pad "
                 "and both feet planted flat on the floor about hip-width apart, the dumbbells pressed overhead until the "
                 "elbows are fully straight and the dumbbells sit directly above the shoulder joints roughly over the ears "
                 "rather than out in front of the face, the arms near vertical with the upper arms close to in line with "
                 "the torso, the palms facing forward, the wrists straight and stacked directly over the elbows, the "
                 "shoulders not jammed up into the ears, the lower back not arched off the pad and the ribcage down, and "
                 "the chin level with the gaze straight ahead",
        ),
        ),
    ),
    "Walking Lunge (dumbbells optional)": Demo(
        shape="wide",
        camera="seen from the side",
        kit="one dumbbell hanging in each hand",
        poses=(
        Pose(
            phase="start_position",
            label="Split stance landing",
            cue="Stand tall, then take one long step forward and land with your front foot flat.",
            key="The feet are a full stride apart in a split stance, trailing leg extended behind with its heel lifted.",
            pose="Standing tall, then a long step forward with the lead leg \u2014 about one full stride, 60-90 cm \u2014 caught at the moment of landing. Split stance: lead foot flat on the floor, toes straight ahead, directly in line with the hip; lead shin near vertical. Trailing leg extended behind, heel lifted, weight on the ball of that foot. Feet about hip-width apart side to side. Torso vertical, spine neutral, chest lifted, shoulders down and back. Arms straight down by the sides, elbows extended, neutral grip with palms facing each other, dumbbells hanging 5-10 cm clear of the outer thighs. Head in line with the spine, chin level, gaze straight ahead.",
        ),
        Pose(
            phase="end_position",
            label="Bottom of the lunge",
            cue="Lower your hips straight down until both knees are bent to about 90 degrees.",
            key="both knees are bent to ninety degrees in a long split stance, the front thigh parallel to the floor and "
                "the back knee just above the floor",
            pose="at the bottom of the lunge in a long split stance: both knees are bent to about ninety degrees, the front "
                 "thigh is roughly parallel with the floor with the front shin near vertical and the front knee stacked "
                 "directly above the ankle rather than pushed out past the toes, the trailing knee hovers a few "
                 "centimetres above the floor directly under the trailing hip with the rear heel lifted and the toes "
                 "tucked under so the ball of that foot carries the weight, the torso stays upright and vertical with the "
                 "chest lifted and the hips square to the direction of travel, the arms hang straight down with a dumbbell "
                 "beside each thigh and the palms facing each other, and the head is neutral with the gaze forward at the "
                 "horizon rather than down at the feet",
        ),
        ),
    ),
    "Front Plank (forearm plank)": Demo(
        shape="wide",
        camera="seen from the side",
        # The guidelines call a mat optional, and BACKDROP forbids a floor because a floor line
        # survives the colour key and arrives in the app as a grey bar under the character.
        kit="no equipment at all",
        # The plank is a hold and not a rep, so its two frames are not two ends of a travel. The
        # source names them anyway — "for animation, the moving frames are the lift-off and the
        # controlled lower" — which is what makes a second drawing of this one sourced rather
        # than invented, and what the pair has to show is the body leaving the floor.
        poses=(
            Pose(
                phase="start_position",
                label="Lying face down",
                cue="Lie face down with your elbows under your shoulders and your toes tucked, before you lift anything.",
                key="the body is still RESTING ON THE FLOOR, face down, with the forearms flat and the elbows directly "
                    "under the shoulders",
                pose="lying face down on the floor before the lift: the whole torso, hips and thighs are still resting "
                     "on the floor, the elbows are bent and tucked close to the sides of the ribs and placed directly "
                     "under the shoulders, the forearms are flat on the floor and roughly parallel about shoulder-width "
                     "apart with the palms down and the fingers pointing forward, the legs are extended straight behind "
                     "with the feet about hip-width apart, the ankles are bent so the toes are pulled toward the shins "
                     "with the toe pads planted and the heels pointing up, the knees are straight, and the head is in "
                     "line with the spine with the neck long and the gaze down at the floor just in front of the hands",
            ),
            Pose(
                phase="end_position",
                label="Holding the plank",
                cue="Lift until your body is one straight line on your forearms and toes, hips neither up nor sagging.",
                key="the whole body is held off the floor in one straight line, resting only on the forearms and the "
                    "toes, with the hips neither sagging down nor raised up",
                pose="holding the plank: only the forearms and the toes touch the floor, and the body is held as one "
                     "straight stiff line from the crown of the head through the shoulders, hips, knees and ankles, the "
                     "shoulders stacked directly over the elbows with the upper arms vertical and the elbows bent to "
                     "about ninety degrees and tucked in near the ribs, the forearms flat on the floor and parallel to "
                     "each other with the palms down, the pelvis neutral with no sag in the lower back and no hips "
                     "hiked up into a tent, the glutes and thighs engaged with the knees fully straight, the ankles "
                     "bent so the weight is on the toe pads and the heels point up, and the head in line with the "
                     "spine with the gaze down and slightly forward at the floor",
            ),
        ),
    ),
    "Lat Pulldown (seated, to the front)": Demo(
        shape="square",
        camera="seen from the side and slightly in front",
        kit="a simple seated lat pulldown machine: a seat with a padded bar across the top of the thighs, an upright "
            "frame, a single cable running up to a pulley overhead and a long straight bar attachment on the end of it",
        poses=(
        Pose(
            phase="start_position",
            label="Arms straight overhead",
            cue="Thighs under the pad, reach up to the bar, shoulders pulled down before you pull.",
            key="The arms are straight overhead at about 170-180 degrees, upper arms framing the head.",
            pose="Seated in a lat pulldown machine, the thigh pad firmly against the top of the thighs, feet flat on the floor about hip-width apart, knees bent to roughly 90 degrees. Hands reach up to grip a long bar overhand, a little wider than shoulder-width. Arms extended overhead, elbows nearly straight at about 170-180 degrees, upper arms framing the head. Shoulder blades depressed and retracted, pulled down and back out of the shrug. Torso leaning back slightly from vertical, no more than about 30 degrees, chest lifted toward the bar, spine long with its natural lumbar curve. Head in line with the spine, chin level, gaze up and forward at the bar.",
        ),
        Pose(
            phase="end_position",
            label="Bar at upper chest",
            cue="Bring the bar to the top of your chest, elbows driving down, and stop there.",
            key="the character is SEATED and the bar has been pulled down in front of the face to collarbone height, "
                "with the elbows driven down toward the sides",
            pose="at the bottom of the pull: seated with the thighs pinned under the thigh pad and both feet flat on the "
                 "floor about hip-width apart with the knees bent to about ninety degrees, the bar pulled down in front of "
                 "the face to collarbone height a few centimetres from the chest, the hands gripping it a little wider "
                 "than shoulder-width with both palms facing forward, the elbows driven down toward the floor and in "
                 "toward the sides of the torso with the forearms close to vertical and in line with the cable, the "
                 "shoulder blades pulled down and together so the chest is open and the shoulders are nowhere near the "
                 "ears, the wrists straight, the torso leaning back only slightly from vertical with the lower back not "
                 "arched, and the head in line with the spine with the gaze up and forward at the bar",
        ),
        ),
    ),
}

# The two of the fourteen left undrawn, and why. Both are cable or selectorized machines whose form
# is defined by a rig the image model has no reliable idea of, and a demonstration on an invented
# machine teaches the wrong thing about a real one. Lat Pulldown is drawn anyway as the one probe of
# the machine case, so the next person has evidence rather than this guess; if it comes out well
# these two are the obvious candidates for the next twelve-image budget.
SKIPPED = {
    "Tricep Pushdown (cable, bar or rope attachment)":
        "a cable stack and attachment the image model renders inconsistently; the movement is defined by the rig",
    "Leg Press (seated machine)":
        "a selectorized sled whose seat, plate and rails decide the joint angles; an invented machine would show "
        "invented form",
}

# --- Cutting the figure out ------------------------------------------------------------------
#
# Keyed by colour rather than by Apple Vision's person mask alone, which is the one judgement in
# this file worth spelling out. Vision segments PEOPLE: on a squat it returns the lifter and throws
# the barbell and both plates away, which for a demonstration of a barbell lift is a worse picture
# than no picture. The backdrop here is a flat colour this script asked for, so everything that is
# not that colour is kept — figure, bar, plates, bench, machine. Vision is still run and folded in,
# because it is the check that the key did not eat part of the figure and the rescue for any limb
# whose colour lands near the backdrop; `measure` in the index records how much of each cut-out
# Vision alone would have discarded.

# How far from the backdrop colour a pixel has to be, as a distance in RGB out of a possible 441,
# before it counts as part of the picture. The nearest thing the character wears to pastel teal is
# the mid-tan skin, 84 away, so this has to sit comfortably below that and above the model's own
# unevenness in a nominally flat backdrop.
KEY_TOLERANCE = 46
# BACKDROP asks for no cast shadow and does not get one: the squat, the deadlift and the bench
# press all came back with a soft contact shadow pooled under the feet, far enough from the
# backdrop colour to survive KEY_TOLERANCE and large enough to survive MIN_PIECE_SHARE, and it
# arrived in the app as a grey smear floating under the character.
#
# A shadow on a flat backdrop is that backdrop dimmed, so it is keyed by direction instead of by
# distance: a pixel is backdrop-in-shadow when it is nearly a scalar multiple of the backdrop
# colour and that scalar is a dimming rather than a different colour.
#
# Both halves of the window are load-bearing, measured against this backdrop. Tan skin (residual
# 77) and white trainers (49) are saved by the residual: they are not teal at any brightness. The
# grey weight plates are not — a neutral grey leaves a residual of only 14 — and are saved solely
# by the lower end of the scale, which they sit under at 0.46, as does the black vest at 0.10.
# Lowering that bound to chase the last of a shadow would start dissolving the plates.
SHADE_RESIDUAL, SHADE_RANGE = 22, (0.62, 1.10)
# A piece of picture smaller than this share of the frame is a speck of backdrop noise, not a
# dumbbell. Weight plates, benches and machine frames all land far above it.
MIN_PIECE_SHARE = 0.004
# A backdrop-coloured island smaller than this share of the frame, fully enclosed by the figure, is
# compression noise and is filled in. Real enclosed gaps — the triangle between an arm and the
# torso at the top of a lateral raise — are larger than this and are left transparent.
MIN_HOLE_SHARE = 0.0004
# The outline is shrunk by a pixel before it is softened, because the model's own edge pixels are a
# blend of figure and backdrop and keeping them puts a pale teal rim around the character, which is
# plainly visible on the app's dark cards.
CHOKE, FEATHER = 1, 0.8
TRIM_MARGIN = 8
# The share of the picture's outer frame that has to be the backdrop colour before the key is
# trusted at all. The whole method assumes the model drew the backdrop it was asked for, and when
# it does not the key keeps everything and writes a fully opaque rectangle — which is what happened
# to the lat pulldown, where the model invented a white room instead. Ten of the eleven drawings
# measured 100% here and that one measured 62%, so anything under this is the model having ignored
# BACKDROP, and the honest answer is no picture rather than an opaque one.
MIN_BACKDROP_BORDER = 0.90


def slug(name):
    """A file name from an exercise name. Names are the identity, so this has to stay reversible by
    eye: the index maps the real name to the file and nothing reads the file name back."""
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", name.lower())).strip("-")


def short(phase):
    """"start" or "finish" from the guidelines' own field name. The app's JSON and the file names
    both read better for it, and `phase` keeps the field name so a pose is still traceable to the
    paragraph it was condensed from."""
    return {"start_position": "start", "end_position": "finish"}[phase]


def key_of(name, pose):
    """How one drawn position is identified in the index.

    A movement is no longer one row: it is one row per position, so the key has to carry both.
    The exercise name is kept intact inside it rather than slugged, because everything else in
    this project looks art up by the name the catalogue spells."""
    return f"{name}|{short(pose.phase)}"


def guidelines():
    """The form notes, by exercise name."""
    if not GUIDELINES.exists():
        raise SystemExit(f"{GUIDELINES} is missing; the poses below were condensed from it and cannot be checked.")
    return {entry["name"]: entry for entry in json.loads(GUIDELINES.read_text())["exercises"]}


def check_names(notes):
    """Refuse to draw if the names here have drifted from the two lists that have to agree.

    A picture filed under a name the app does not use is invisible to the app, and a pose condensed
    from an entry that has since been rewritten is a form diagram with no source. Both are silent
    wrong answers, so both stop the run."""
    mine = set(DEMOS) | set(SKIPPED)
    if missing := sorted(set(notes) - mine):
        raise SystemExit("form-guidelines.json has movements this script does not account for; add them to DEMOS or "
                         "SKIPPED:\n  " + "\n  ".join(missing))
    if extra := sorted(mine - set(notes)):
        raise SystemExit("These names are not in form-guidelines.json, so their poses have no source:\n  " + "\n  ".join(extra))

    if not CATALOGUE.exists():
        print(f"{CATALOGUE} is not in this checkout, so the names were not checked against the app's catalogue.")
        return
    catalogue = set(re.findall(r'LibraryExercise\(name: "([^"]+)"', CATALOGUE.read_text()))
    if unknown := sorted(set(DEMOS) - catalogue):
        raise SystemExit("The app's ExerciseCatalogue has no movement by these names, so it would never find their "
                         "art:\n  " + "\n  ".join(unknown))


def prompt_for(name, demo, pose):
    """What the image model is asked to draw, assembled in the order restyle.prompt_for uses: lead
    sentence, style, who the character is, the position, the framing, the backdrop. `pose.key`
    brackets the position for the reason given on `Demo`.

    Every position of one movement shares the shape, the camera and the kit, so the two drawings
    differ only where the body is. That is what lets the app cross-fade between them: a changed
    camera between frames reads as a cut to a different exercise, not as a rep."""
    return (
        f"A cute 3D avatar character demonstrating the exercise \"{name}\" with correct form: {pose.key}. "
        + restyle.STYLE
        + GENERIC
        + f"Equipment: {demo.kit}. "
        + f"Position: {pose.pose}. "
        + f"The one thing that must be exactly right: {pose.key}. "
        + FRAMING.format(camera=demo.camera)
        + BACKDROP
    )


def draw(name, demo, pose):
    """The drawing, as a PIL image. Exits on an exhausted allowance, the way restyle.describe does:
    every later call in the run would be refused too, and a half-finished set is worth more than a
    screenful of identical failures."""
    width, height = SHAPES[demo.shape]
    # The fields go in `files` with no file name rather than in `data` or `json`, which is the only
    # way to make requests send a multipart body: the model refuses a JSON one outright ("required
    # properties at '/' are 'multipart'"), and restyle.py never hit that because it always has an
    # input image to attach and so always sends multipart anyway.
    fields = {"prompt": prompt_for(name, demo, pose), "width": width, "height": height, "seed": SEED}
    result, problem = restyle.cloudflare(
        restyle.MODEL,
        files={key: (None, str(value)) for key, value in fields.items()},
    )
    if problem == restyle.ALLOWANCE_USED:
        sys.exit(problem)
    if not result:
        return None
    return Image.open(io.BytesIO(base64.b64decode(result["image"]))).convert("RGB")


def person_mask(picture):
    """Where Apple Vision sees a person in the drawing, or None if it sees none.

    None is a real answer, not an empty one: apple-vision.swift only writes the mask file when the
    segmentation returned a buffer, so a stylized drawing it cannot read raises instead of handing
    back an empty array. Treating that as "no person here" would cut the whole figure away."""
    try:
        mask, _ = apple_vision.body(picture)
    except (OSError, ValueError, subprocess.CalledProcessError) as problem:
        print(f"    Vision could not segment the drawing: {problem}")
        return None
    return mask if mask is not None and mask.any() else None


# A cast shadow pooled on the floor is neutral — its channels sit within a few points of each
# other — while skin is warm and a shoe is dark. Those three facts separate them, but only low
# down: a grey tank top is just as neutral, so the test is confined to the bottom of the figure
# where there is nothing but legs, feet and floor.
SHADOW_NEUTRAL, SHADOW_LIGHT, SHADOW_ZONE = 30, 130, 0.25


def floor_shadow(rgb, solid):
    """Where a contact shadow is lying on the ground, as a mask.

    The colour key upstream cannot catch this one. It removes the backdrop DIMMED — a shadow
    falling on the teal — whereas the model drew an actual grey floor patch, a different colour
    entirely, far enough from the backdrop to read as a real object. Nor does keeping the largest
    piece help: a contact shadow starts at the feet, so it is joined to the figure and they are
    one piece.
    """
    rows = np.where(solid.any(axis=1))[0]
    if not len(rows):
        return np.zeros_like(solid)
    top, bottom = rows[0], rows[-1]
    zone = np.zeros_like(solid)
    zone[int(bottom - (bottom - top) * SHADOW_ZONE):bottom + 1] = True
    neutral = (rgb.max(axis=2) - rgb.min(axis=2)) < SHADOW_NEUTRAL
    light = rgb.mean(axis=2) > SHADOW_LIGHT
    return zone & neutral & light & solid


def cut_out(picture, box=None, solo=False):
    """The drawing with a real alpha channel, and what the cut-out did, as a dict for the index.

    `box` forces the crop instead of letting the picture trim to its own contents, and it is how
    the positions of one movement end up able to cross-fade into each other. Every position is
    drawn on the same canvas with the same camera, so the character stands in the same place in
    all of them — and then trimming each to its own bounding box moves it, because a raised
    barbell makes the box taller and a hanging dumbbell does not. Cropping every position of one
    movement to the union of their boxes keeps the body still and lets the limbs move, which is
    the whole difference between a demonstration and a jump cut.

    Returns (None, why) rather than a picture when the drawing's border is not the backdrop that
    was asked for, because the key has nothing to key against and would hand back the whole
    rectangle."""
    rgb = np.asarray(picture, dtype=np.float32)
    height, width, _ = rgb.shape
    frame = height * width
    back = np.array(BACKDROP_RGB, dtype=np.float32)
    distance = np.linalg.norm(rgb - back, axis=2)

    border = np.concatenate([distance[0], distance[-1], distance[:, 0], distance[:, -1]])
    edges = float((border <= KEY_TOLERANCE).mean())
    if edges < MIN_BACKDROP_BORDER:
        return None, (f"only {edges:.0%} of the picture's border is the pastel teal backdrop the prompt asked for, "
                      f"so there is nothing to key out and the cut-out would be an opaque rectangle")

    dimming = (rgb @ back) / float(back @ back)
    shadow = ((np.linalg.norm(rgb - dimming[..., None] * back, axis=2) <= SHADE_RESIDUAL)
              & (dimming >= SHADE_RANGE[0]) & (dimming <= SHADE_RANGE[1]))

    kept = (distance > KEY_TOLERANCE) & ~shadow
    pieces, found = ndimage.label(kept)
    sizes = np.bincount(pieces.ravel(), minlength=found + 1)
    sizes[0] = 0
    # The largest piece is the character. Everything else is kept only if it is big enough to be a
    # real object, so a stray wisp of shadow goes and a weight plate stays.
    # `solo` keeps the largest piece and nothing else, for a drawing that is one person with no
    # props. The size rule below exists to save a dumbbell or a weight plate from being binned as
    # noise, and with nothing to save it becomes a liability: the trainer's back view came back
    # with a grey floor shadow covering 2.1% of the frame, comfortably over the threshold, and it
    # was kept as though it were equipment. It also escapes the shadow key above, which looks for
    # the backdrop colour dimmed — this was neutral grey, a different colour entirely.
    wanted = ({int(sizes.argmax())} if solo
              else {int(sizes.argmax())} | {index for index in range(1, found + 1)
                                            if sizes[index] >= MIN_PIECE_SHARE * frame})
    specks = int(sizes.sum() - sizes[list(wanted)].sum())
    solid = np.isin(pieces, list(wanted))

    holes, found = ndimage.label(~solid & ndimage.binary_fill_holes(solid))
    sizes = np.bincount(holes.ravel(), minlength=found + 1)
    solid |= np.isin(holes, [index for index in range(1, found + 1) if sizes[index] < MIN_HOLE_SHARE * frame])

    if solo:
        solid &= ~floor_shadow(rgb, solid)

    vision = person_mask(picture)
    if vision is not None and vision.shape == solid.shape:
        missed = float((solid & ~vision).sum() / max(solid.sum(), 1))
        solid |= vision
    else:
        missed = None

    choked = ndimage.binary_erosion(solid, iterations=CHOKE, border_value=1)
    alpha = np.clip((ndimage.gaussian_filter(choked.astype(np.float32), FEATHER) - 0.25) / 0.5, 0, 1)

    # Un-mix the backdrop out of the half-transparent edge pixels. Without this the rim keeps the
    # teal it was blended with and shows as a halo wherever the app puts the picture on a dark card.
    edge = (alpha > 0) & (alpha < 1)
    share = alpha[edge][:, None]
    rgb[edge] = np.clip((rgb[edge] - (1 - share) * back) / share, 0, 255)

    cut = Image.fromarray(np.dstack([rgb, alpha * 255]).round().clip(0, 255).astype(np.uint8), "RGBA")
    found = Image.fromarray((alpha > 0.5).astype(np.uint8) * 255).getbbox()
    crop = box or found
    if crop:
        left, top, right, bottom = crop
        cut = cut.crop((max(left - TRIM_MARGIN, 0), max(top - TRIM_MARGIN, 0),
                        min(right + TRIM_MARGIN, width), min(bottom + TRIM_MARGIN, height)))

    return cut, {
        "backdrop_border": round(edges, 4),
        # Where the picture's contents sat on the canvas it was drawn on, which is what lets a
        # later pass crop the positions of one movement to a shared box.
        "box": list(found) if found else None,
        "drawn": [width, height],
        "size": [cut.width, cut.height],
        "figure_share": round(float((alpha > 0.5).sum() / frame), 4),
        "vision_share": None if vision is None else round(float(vision.sum() / frame), 4),
        "missed_by_vision": None if missed is None else round(missed, 4),
        "pieces_kept": len(wanted),
        "specks_dropped_share": round(specks / frame, 5),
        "soft_edge_share": round(float(edge.sum() / frame), 4),
    }


def align(entries):
    """Re-crop every position of a movement to one shared box, so they cross-fade instead of jumping.

    Costs nothing: it works from the saved drawings. Runs over whatever is on disk rather than
    only over what this run drew, because a run that adds a second position to a movement drawn
    weeks ago has to move that older picture too — it is the one that was trimmed on its own.

    A movement with one position is left alone. There is nothing to line it up with, and its own
    bounding box is the tightest honest crop of it."""
    byMovement = {}
    for key in entries:
        byMovement.setdefault(entries[key]["exercise"], []).append(key)
    for exercise, keys in byMovement.items():
        existing = json.loads(INDEX.read_text()).get("exercises", {}) if INDEX.exists() else {}
        keys = sorted({*keys, *(k for k, e in existing.items() if e.get("exercise") == exercise)})
        if len(keys) < 2:
            continue
        rows = {k: entries.get(k) or existing.get(k) for k in keys}
        boxes = [row["measure"]["box"] for row in rows.values() if row and row["measure"].get("box")]
        if len(boxes) < 2:
            print(f"{exercise}: positions cannot be aligned, one of them predates the box being recorded. "
                  f"Re-run with --recut.")
            continue
        union = (min(b[0] for b in boxes), min(b[1] for b in boxes),
                 max(b[2] for b in boxes), max(b[3] for b in boxes))
        for key, row in rows.items():
            if not row:
                continue
            raw_path = ROOT / "reference" / "exercise-art" / row["raw"]
            if not raw_path.exists():
                continue
            cut, measure = cut_out(Image.open(raw_path).convert("RGB"), box=union)
            if cut is None:
                continue
            cut.save(ART / row["file"], "PNG")
            row["measure"] = {**measure, "aligned_to": list(union)}
            entries[key] = row
        print(f"{exercise}: {len(rows)} positions aligned to one box {union}, "
              f"{union[2] - union[0]}x{union[3] - union[1]}.")


def write_index(entries, refused):
    """The index the app reads, merged into whatever is already there so a part-finished run keeps
    the pictures an earlier one made.

    An exercise that could not be cut out leaves `exercises` entirely rather than sitting there
    with a null file, so every name in `exercises` means there is a real picture on disk."""
    existing = json.loads(INDEX.read_text()) if INDEX.exists() else {}
    art = {name: entry for name, entry in {**existing.get("exercises", {}), **entries}.items() if name not in refused}
    blocked = {**existing.get("not_cut_out", {}), **refused}
    INDEX.write_text(json.dumps({
        "note": "Generic avatar demonstrations, one per exercise, cut out to a transparent background. Keys are "
                "exercise names exactly as Sources/Faces/Model/ExerciseCatalogue.swift spells them. Written by "
                "pipeline/exercise_art.py; poses condensed from reference/exercises/form-guidelines.json.",
        "show_only": "Entries whose form_checked.verdict is \"correct\". A null verdict means nobody has compared the "
                     "picture with its form notes yet, and \"wrong\" means somebody has and it does not show the "
                     "movement. Both are worse than showing no picture, because a demonstration gets copied.",
        "style": "pipeline/restyle.py STYLE, reused verbatim",
        "model": restyle.MODEL,
        "seed": SEED,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "not_drawn": SKIPPED,
        # Computed rather than written down, so a movement missing because a run stopped early is
        # never mistaken for one nobody wanted. Re-running the script draws these.
        "not_yet_drawn": [key_of(name, pose) for name, demo in DEMOS.items() for pose in demo.poses
                          if key_of(name, pose) not in art and key_of(name, pose) not in blocked],
        "not_cut_out": blocked,
        "exercises": dict(sorted(art.items())),
    }, indent=2) + "\n")


def review(verdicts):
    """Record a person's judgement of whether a drawing shows the movement, and nothing else.

    This gate exists because the first two lateral raises drawn here had the arms about forty-five
    degrees above horizontal, which is a front raise, not a lateral raise — and both were otherwise
    good pictures that would have gone straight into the app. The model's confidence says nothing
    about whether the form is right, so a picture only becomes visible to the app once somebody has
    held it against reference/exercises/form-guidelines.json and said so. Costs no calls."""
    if not INDEX.exists():
        raise SystemExit(f"{INDEX} does not exist yet, so there is nothing to review.")
    index = json.loads(INDEX.read_text())
    for name, verdict, note in verdicts:
        if name not in index.get("exercises", {}):
            raise SystemExit(f"{name!r} has not been drawn, so there is no picture to judge.")
        if verdict not in ("correct", "wrong"):
            raise SystemExit(f"A verdict is \"correct\" or \"wrong\", not {verdict!r}.")
        index["exercises"][name]["form_checked"] = {
            "verdict": verdict, "note": note, "at": datetime.now(timezone.utc).isoformat(),
        }
        print(f"{name}: {verdict} — {note}")
    index["updated_at"] = datetime.now(timezone.utc).isoformat()
    INDEX.write_text(json.dumps(index, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", action="append", metavar="NAME", help="just this exercise, by its exact name (repeatable)")
    parser.add_argument("--limit", type=int, metavar="N", help="draw at most N pictures this run")
    parser.add_argument("--dry-run", action="store_true", help="print the prompts and draw nothing")
    parser.add_argument("--recut", action="store_true", help="redo the alpha from the saved drawings, with no calls")
    parser.add_argument("--redraw", action="store_true", help="draw again even where a picture already exists")
    parser.add_argument("--review", nargs=3, action="append", metavar=("NAME", "correct|wrong", "NOTE"),
                        help="record that somebody checked a drawing against its form notes (repeatable, no calls)")
    args = parser.parse_args()

    notes = guidelines()
    check_names(notes)

    if args.review:
        review(args.review)
        return

    wanted = list(DEMOS)
    if args.only:
        if unknown := [name for name in args.only if name not in DEMOS]:
            raise SystemExit("Not a movement this script draws:\n  " + "\n  ".join(unknown)
                             + "\nIt draws:\n  " + "\n  ".join(DEMOS))
        wanted = [name for name in wanted if name in args.only]

    if args.dry_run:
        for name in wanted:
            demo = DEMOS[name]
            for pose in demo.poses:
                print(f"--- {name}  [{pose.phase}, {demo.shape} {SHAPES[demo.shape]}, seed {SEED}]")
                print(prompt_for(name, demo, pose) + "\n")
        print(f"{sum(len(DEMOS[n].poses) for n in wanted)} prompts across {len(wanted)} movements, 0 calls.")
        return

    ART.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    judged = json.loads(INDEX.read_text()).get("exercises", {}) if INDEX.exists() else {}
    entries, refused, calls = {}, {}, 0

    for name, demo, pose in [(name, DEMOS[name], pose) for name in wanted for pose in DEMOS[name].poses]:
        # The finishing position keeps the bare name, because two pictures already shipped under
        # it and were judged correct under it; renaming them would throw that judgement away and
        # cost two calls to get back. Only the newer positions carry a suffix.
        stem = slug(name) if pose.phase == "end_position" else f"{slug(name)}-{short(pose.phase)}"
        label = f"{name} [{short(pose.phase)}]"
        cut_path, raw_path = ART / f"{stem}.png", RAW / f"{stem}.png"
        have = cut_path.exists() and raw_path.exists() and not args.redraw and not args.recut
        if have:
            print(f"{label}: already drawn, skipping ({cut_path.name}).")
            continue

        if raw_path.exists() and not args.redraw:
            picture = Image.open(raw_path).convert("RGB")
            print(f"{label}: cutting out the saved drawing.")
            # A recut changes the alpha channel, not the pose, so a verdict somebody reached about
            # whether the movement is right still holds. A redraw does not reach here.
            checked = judged.get(key_of(name, pose), {}).get("form_checked")
        else:
            checked = None
            if args.recut:
                # --recut reworks saved drawings and must never reach the network: without this it
                # fell through and tried to draw the one movement that had no saved drawing.
                print(f"{label}: nothing saved to cut out; --recut draws nothing.")
                continue
            if args.limit is not None and calls >= args.limit:
                print(f"{label}: not drawn, --limit {args.limit} reached.")
                continue
            if calls >= BUDGET:
                print(f"{label}: not drawn, the {BUDGET}-image budget is spent.")
                continue
            print(f"{label}: drawing ({demo.shape} {SHAPES[demo.shape]}).")
            calls += 1
            picture = draw(name, demo, pose)
            if picture is None:
                print(f"{label}: Cloudflare would not draw it; left undrawn.", file=sys.stderr)
                continue
            picture.save(raw_path, "PNG")

        cut, measure = cut_out(picture)
        if cut is None:
            # The drawing is kept — it was paid for and a person should see what came back — but no
            # cut-out is written and the app is told the exercise has no picture rather than handed
            # an opaque one.
            print(f"{label}: not cut out — {measure}", file=sys.stderr)
            cut_path.unlink(missing_ok=True)
            refused[key_of(name, pose)] = {"raw": f"{RAW.name}/{raw_path.name}", "reason": measure}
            continue

        cut.save(cut_path, "PNG")
        entries[key_of(name, pose)] = {
            "exercise": name,
            "file": cut_path.name,
            "raw": f"{RAW.name}/{raw_path.name}",
            "phase": pose.phase,
            "label": pose.label,
            "cue": pose.cue,
            "source": notes[name].get("source_url"),
            # A fresh drawing is unjudged, and a redrawn one throws away any judgement of the
            # picture it replaced, which was a judgement of a different pose.
            "form_checked": checked,
            "measure": measure,
            "prompt": prompt_for(name, demo, pose),
        }
        print(f"  -> {cut_path.name} {cut.width}x{cut.height}, figure {measure['figure_share']:.0%} of the frame, "
              + ("Vision saw no person" if measure["missed_by_vision"] is None
                 else f"{measure['missed_by_vision']:.0%} of it outside Vision's person mask"))

    align(entries)

    if entries or refused:
        write_index(entries, refused)
        print(f"\n{len(entries)} written, {len(refused)} refused, {calls} Workers AI calls. Index: {INDEX}")
    else:
        print(f"\nNothing to do, {calls} Workers AI calls.")


if __name__ == "__main__":
    main()
