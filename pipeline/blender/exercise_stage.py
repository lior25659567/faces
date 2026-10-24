#!/usr/bin/env python
"""Per-exercise staging: what equipment each movement gets, where it sits, and where the camera
stands so the movement is legible.

    blender --background --python pipeline/blender/exercise_stage.py -- \
        --model "3d-character/3d cartoon boy.glb" --tiles /tmp/tiles

Why this exists. Before this module every one of the fourteen exercises rendered as the same
empty-handed figure photographed from the front: a bench press and an overhead press were the
same picture. Four of the seven body orientations are also unreadable from the front — supine
shows the soles of the feet, bent over shows the top of the head, and hanging and front support
are similarly lost. `STAGING` below is the differentiation: one prop set, one camera and one
reason per exercise, in one table a reader can audit against the brief.

## Props follow the body, never the reverse

Nothing here is a fixed world coordinate. Every prop is placed from the character's own posed
skeleton: the bar goes where the hands actually are, the bench pad goes where the back actually
is, the foot plate goes where the feet actually are. The character's proportions are measured
off whatever mesh is loaded (`fit_model.measure`), so a number tuned against one mesh would be
wrong for the next one, and the spec asks for this to hold across all fourteen.

## Read the POSED RIG, not the solved skeleton

This is the one subtlety worth stating up front, because it is counter-intuitive and it is
measurable. The rendered figure has the solved skeleton's SHAPE but not its proportions:

  * The rig's bone lengths come from `fit_model.measure`'s reading of the actual mesh, not from
    `pose_model.PROPORTIONS`, so solved and posed joints differ by up to 0.13 (13% of the
    character's height) on `3d cartoon boy.glb` — a hip 8 cm out, in human terms. The solved
    pelvis of a standing figure is at 0.491; this character's is at 0.426, because it is a
    mascot with a quarter of its height in its head and correspondingly short legs.
  * `animate.with_clavicles` moves the solved shoulders after the solve, which changes the
    shoulder-to-elbow distance the solver produced while the upper-arm bone keeps its own
    length, so the posed elbow lands up to 5 cm off the solved elbow.

A prop placed from `skeleton.at("hand.left")` would therefore hang in the air next to the hand.
Everything below reads `rig.pose.bones[...]` (`.head`/`.tail` are the POSED positions, in
armature space) through `joints_of()`. `skeletons` is still taken as an argument and still used
— for the orientation of each phase, and to decide which phase defines a contact — but never as
a source of world coordinates.

## Contact at both keyframes

A prop the body rests on has to be in contact at the start AND the finish, and those are two
different poses. Each entry names the phase its placement is read from (`contact`), and the
contact sheet renders that same phase (`showcase`) so the frame a human judges is the frame the
contact was computed for. Where the other phase breaks contact, the entry's `notes` says so.

## The floor is at zero, and the figure is on it

The first version of this module moved the FLOOR to meet the figure, because `animate` never
translated the rig's root and so the plane the figure stood on was wherever its rest pose had
left it — 0.426 up, in every pose of every exercise. That is fixed at the cause now
(`animate.ground_rig`), so the floor and the backdrop sit at z = 0 and the figure is grounded
where `pose_model.ground` says it belongs. Nothing here computes a ground plane any more, and
every prop that stands on the floor stands at zero like a real one.

Two consequences worth naming, because they are what the workaround was hiding:

  * A supine bench press rests on a BENCH, not on the ground, and the form notes have no idea a
    bench exists — `ground` lies the figure flat on the floor. `support_height` reads the pad's
    own height off `equipment`'s constant and hands it to `animate.ground_rig`, so the body
    rests on the pad and the pad stands on the floor. Before, the only two outcomes available
    were a body sunk through the pad or a bench sunk through the floor.
  * A cable machine's attachment is wherever the user has pulled it to, which is why
    `equipment.lat_pulldown` and `equipment.cable_tower` now take `drop=`. Both are solved for
    this figure's own grip height, so the bar meets the hands AND the machine stands on the
    floor. Before, those two were mutually exclusive and both tiles were unshippable.

`sink` survives for one prop, the pull-up bar, and it is now worth almost nothing: the solver
hangs a figure with its hands at `pose_model.HANGING_BAR` = 1.25 of standing height, and
`equipment.PULLUP_BAR_HEIGHT` is 2.2 m = 1.257 of this character's height, so bringing the bar
to the hands leaves its feet about 6 mm under the floor. The measured depth is printed per
exercise rather than asserted.

## Camera

The azimuth/elevation of every exercise is the brief's own choice, with the brief's own reason
kept beside it as a string so the table can be audited against the spec. Azimuth is degrees
around the figure's vertical axis: 0 puts the camera straight in front of the figure (which
faces +Y), +90 puts it at the figure's LEFT side, -90 at its right. Elevation lifts it.

The lens maths is `make_character.stage`'s, deliberately unchanged — a 60 mm lens on a 36 mm
vertical sensor — but what it is backed off from is now the CHARACTER and nothing else.

That is the second thing this module got wrong. Every camera used to retreat until its own set
fitted the frame, which meant the distance was set by the equipment: a 2.2 m barbell or a 2.2 m
lat tower pushed it out, a small dumbbell let it come close, and the character's share of the
frame ran from 80% down to 27% across the fourteen tiles. A reviewer looking at the contact
sheet said the obvious thing — that it is not the same person in every tile — and they were
right: a 2.93x size range is not a series, it is fourteen photographs of different people.

So there is now ONE distance, `BASE_FRAMING` times the distance that holds the character's own
unposed height, identical in all fourteen. Equipment is allowed to run out of frame, because a
barbell cut off at both edges still reads as a barbell while a shrunken lifter does not read as
the same lifter. `framing` stays as a per-exercise multiplier but is constrained to within 15%
of 1.0 and checked at runtime, and each entry that uses one says why in its `notes`.

The backdrop needed fixing rather than keeping: `make_character.stage` puts it at y = -1.8,
square to a camera at +Y, so any other azimuth would have seen it edge-on and the render would
have come back as the inside of a 12-unit plane seen on its edge — i.e. nothing. It now orbits
with the camera and stays square to it. The three area lights orbit with the camera too, which
is not strictly what the brief asked for, but it is the same light rig (positions read off the
objects `make_character.stage` built, not copied as numbers) and keeping it fixed would have lit
the side views from behind.

## Known problems in the body, found while staging and NOT fixed here

`pose_model` and `fit_model` belong to other tasks and were not touched, and `poses.json` is
another task's to edit. These are what the fourteen staged frames show, reported because they
are visible in the contact sheet and a reader will otherwise think the staging did it:

  1. `seated` now seats the figure — it was fixed in `pose_model` while this pass was running,
     and three tiles changed under it for the better. What it leaves is a proportions problem
     rather than a pose one: the solver seats an adult's hip at 0.246 of standing height, which
     is the 0.43 m that `equipment`'s 0.45 m seats are built for, while this mascot's short
     shins put its seated hip at 0.209. Every seat in the library is therefore too tall for it
     by about a tenth of its height. Staging measures the figure's own sitting surface and
     moves the machine to it (`seat_surface`, `_seat_base`), which is the core rule — props
     follow the body — applied to the one contact that defines a seated exercise.
  2. The Pull-Up's FINISH puts the hands at CHEST height (`shoulder from_vertical 0` with the
     elbows bent 50), and `ground` anchors a hanging figure by its hands — so the finish renders
     as a figure floating in mid-air holding a bar across its ribs. The entry shows the start
     instead and says so.
  3. The Lat Pulldown's FINISH records no shoulder angle at all, so the arms fall back to the
     rest pose's hanging and the solved bar height is the hip. Same remedy, same note.
  4. The Walking Lunge's START records no angles whatsoever — it is a figure standing to
     attention — and its finish bends BOTH knees to 90 with no hip angle, which is a kneeling
     lunge rather than a stride. The finish is staged and reads as a lunge; the start would
     have been indistinguishable from the bicep curl's.
  5. The Front Plank's `front_support` does not put the forearms and the toes in one plane. The
     forearms are on the mat and the toes hang about a tenth of a body height above it.
  6. `lying_supine` grounds the back and the feet in the SAME plane, so lifting the back onto a
     0.45 m bench lifts the feet with it and the bench press's legs stick out horizontally in
     mid-air. A bench press is performed with the feet planted; no angle in the phase says so.
"""

import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import equipment  # noqa: E402
import make_character  # noqa: E402

# Where on the hand bone a grip sits.
#
# `equipment.hold`'s own documented anchor is the bone's HEAD, and `make_character.BONES` runs
# "hand.left" from the wrist (head) to the hand (tail) — so an offset of zero puts a dumbbell at
# the wrist, which is what its author flagged and never got to test inside an animation. It is
# visibly wrong in a render: the bar sits across the wrists with both palms past it, about 5 cm
# (5% of the character's height) out of the hand. Half way along the hand bone is the palm, so
# every grip below is taken there.
PALM = 0.5

# Mirrored from `equipment.py`'s own expressions, which keep these as local literals rather than
# module constants. Each one is checked at runtime against the built mesh by `_check_contact`,
# so if equipment.py's geometry moves, staging says so instead of silently misplacing a prop.
#
# The lat pulldown's and cable tower's cable drops and pulley insets USED to be mirrored here
# too. They are `equipment.LAT_BAR_DROP`, `LAT_PULLEY_INSET`, `CABLE_DROP` and
# `CABLE_PULLEY_INSET` now, promoted out of those function bodies when the `drop=` parameter
# went in, so there is one fewer copy of a number to rot.
LAT_SEAT_X = -0.35           # equipment.lat_pulldown: seat_x
LEG_PRESS_RAIL_BASE = (0.0, -0.18, 0.06)   # equipment.leg_press: rail_base, in metres
LEG_PRESS_PLATE_OFFSET = (0.02, 0.0, 0.05)  # equipment.leg_press: plate_centre - rail_top
LEG_PRESS_TILT = 45.0        # equipment.leg_press: tilt

# The one camera distance, as a multiple of the distance that exactly holds the character's own
# unposed height with `make_character.stage`'s 1.06 margin. 1.25 leaves the standing figure at
# about three quarters of the frame height — which is where the tiles a reviewer liked already
# were — with enough air above it for the one pose that reaches higher than the character is
# tall (a pull-up hangs its hands at 1.25 of standing height).
BASE_FRAMING = 1.25

# How far a per-exercise `framing` may stray from 1.0. The point of a shared distance is that
# the character is the same size in every tile, and a multiplier is a hole in that; 15% is small
# enough that the figure still reads as the same figure and is checked rather than trusted.
#
# Raising this number would undo Finding 1 of this task — a reviewer looked at the first contact
# sheet and said the character was not the same person, and they were right, because every
# camera had been backed off to hold its own equipment. One entry is past the tolerance and it
# carries `framing_exception` saying why; that is the mechanism for the next one, not this
# constant.
FRAMING_TOLERANCE = 0.15

# The overhead lamp that puts a shadow under the figure, and how hard it works.
#
# The house style is flat pastel with almost no shadow ON the figure — most of the light in
# `make_character.stage` is a bright even world, and the three lamps only shape it. That is the
# right look and it has one cost: nothing underneath the figure says it is standing on the
# floor. It did not matter while the floor was being moved up to meet the figure; now that the
# grounding is real (`animate.ground_rig`) it is the only cue a viewer has, and without it a
# lunge and a pull-up are the same picture — a boy in the air.
#
# So: one more lamp, straight above, dim. Its job is the contact patch, not modelling. An area
# light's shadow is sharp where the two surfaces meet and blurs out with distance, which is
# exactly the shape of a grounding cue, so the size is chosen for that softness and the energy
# for an effect that reads at a glance and does not survive a second look. Added here rather
# than in `make_character.stage` so the one-off renders elsewhere in the project keep the look
# they have.
SHADOW_LIGHT = "shadow"
SHADOW_RECEIVERS = "shadow_receivers"
SHADOW_SIZE = 0.55            # Blender units of panel: soft, but not so soft it has no edge
SHADOW_ENERGY = 220.0        # watts; it lights the floor only, so this never reaches the figure
SHADOW_ELEVATION = 52.0      # degrees above the horizon
SHADOW_AZIMUTH = 55.0        # degrees round from the camera, so the shadow falls to one side
SHADOW_REACH = 2.6           # how far from the figure it hangs          # above the figure's own centre

# The surface each support prop offers the body, as a height above the floor in metres, read
# from `equipment`'s own constants rather than copied. `support_height` turns this into the
# `rest_at` that `animate.ground_rig` grounds the figure onto.
SUPPORT_TOP = {
    "flat_bench": lambda: equipment.BENCH_HEIGHT,
    "floor_mat": lambda: equipment.FLOOR_MAT_THICKNESS,
}


# --- the table ---------------------------------------------------------------------------------
#
# One entry per name in reference/exercises/poses.json, spelled exactly as that file spells it.
#
#   equipment  props and how each one attaches (see `_MOUNTS` for what each mount means)
#   azimuth    degrees around the figure: 0 = straight in front, +90 = its left side
#   elevation  degrees above the horizon
#   framing    multiplier on the shared camera distance, within FRAMING_TOLERANCE of 1.0
#   framing_exception
#              the reason an entry is allowed past that tolerance. Required for one, and the
#              guard refuses the multiplier without it, so an exception cannot be made quietly
#   airborne   this body does not rest on anything (a pull-up hangs), so the check that its
#              lowest surface is on the floor would be wrong to make
#   reason     why THIS angle, from the brief, kept here so the choice is auditable
#   contact    the phase a resting prop's placement is read from
#   showcase   the phase the verification frame renders (kept equal to `contact`, so the frame a
#              human judges is the frame the contact was computed for)
#   notes      anything about contact or placement a reader would otherwise have to infer

STAGING = {
    "Barbell Back Squat": {
        "equipment": [{"prop": "barbell", "mount": "upper_back"}],
        "azimuth": 35, "elevation": 10, "framing": 1.0,
        "reason": "3/4 front: depth and knee track both read",
        "contact": "finish", "showcase": "finish",
        "notes": "Bar rests across the traps, behind the neck, parented to the spine bone so it "
                 "stays on the back through the descent. Placed from the shoulder midpoint and "
                 "the torso's own back direction, not from a fixed height. Contact is the same "
                 "in both phases (the bar never leaves the back); the finish is read because "
                 "that is the frame shown. The bar runs well outside the frame at this "
                 "distance, which is the deliberate trade: 2.2 m of barbell against a 1.75 m "
                 "lifter cannot share a frame with a character the same size as in the other "
                 "thirteen tiles, and a cropped bar still reads as a bar.",
    },
    "Conventional Barbell Deadlift": {
        "equipment": [{"prop": "barbell", "mount": "both_hands"}],
        "azimuth": 90, "elevation": 8, "framing": 1.0,
        "reason": "side: the hip hinge and the bar path are a side view",
        "contact": "start", "showcase": "start",
        "notes": "Bar in both hands, read from the START (bent over, bar on the floor) because "
                 "that is the phase where contact with the ground is defined: at the finish the "
                 "bar is at the hips and touches nothing. Seen from the side the bar points at "
                 "the camera, so its length costs nothing and only the near plate is wide — "
                 "which is why this one needs no framing multiplier even though it is loaded.",
    },
    "Flat Barbell Bench Press": {
        "equipment": [{"prop": "flat_bench", "mount": "under_body", "support": True,
                       "span": ("head", "hip.left")},
                      {"prop": "barbell", "mount": "both_hands"}],
        "azimuth": 90, "elevation": 34, "framing": 1.0,
        "reason": "side, raised: the press path and elbow angle; a front camera shows the soles",
        "contact": "finish", "showcase": "finish",
        "notes": "The only exercise whose body is not supported by the floor. `support: True` "
                 "makes the bench stand on the floor like a real one and grounds the FIGURE on "
                 "the pad instead (`support_height` -> `animate.ground_rig`), so the back lies "
                 "on the pad at the pad's own measured height. Either phase defines that "
                 "contact identically — the spine does not move between them, only the arms — "
                 "and the finish is read because it is the frame shown, and because the start "
                 "is the lockout, where the bar is a metre above the chest. The feet still hang "
                 "clear of the floor: `lying_supine` grounds the back and the feet in the same "
                 "plane, so lifting the back onto a 0.45 m pad lifts the feet with it. A bench "
                 "press is performed with the feet planted; the form notes do not say so in any "
                 "quantity the solver can apply, and inventing a hip angle here is not staging's "
                 "call.",
    },
    "Standing Barbell Overhead Press (Shoulder Press)": {
        "equipment": [{"prop": "barbell", "mount": "both_hands"}],
        "azimuth": 30, "elevation": 8, "framing": 1.0,
        "reason": "3/4 front: the bar passing the face",
        "contact": "start", "showcase": "start",
        "notes": "Start is shown: the racked position is the one a reader has to recognise, and "
                 "the finish is a lockout with the bar overhead and out of frame at this "
                 "distance. `shoulder flexion 45` now swings the humerus FORWARD rather than "
                 "out to the side, which is the correction made to poses.json after the first "
                 "contact sheet; the render is the check on it.",
    },
    "Barbell Bent-Over Row": {
        "equipment": [{"prop": "barbell", "mount": "both_hands"}],
        "azimuth": 50, "elevation": 10, "framing": 1.0,
        "reason": "3/4 side: the torso angle is what people get wrong, and at 50 degrees the "
                  "hinge still reads while the near plate stops covering the lifter",
        "contact": "finish", "showcase": "finish",
        "notes": "The one azimuth in this table that is not the brief's own, changed on the "
                 "brief author's instruction after the pure side view was rendered and rejected. "
                 "At 90 the barbell points straight down the lens, so the near plate — 0.45 m "
                 "across and 0.63 closer to the camera than the lifter, which perspective "
                 "enlarges by about 1.4 — covered both legs completely: the tile was a head, a "
                 "back, a pair of shorts and a black disc. Swept 35/45/50/55/65/75 and chose by "
                 "looking. Below about 55 the plate clears the legs; by 65 it is back over them; "
                 "at 35 the lean starts to flatten out. 50 keeps the horizontal torso "
                 "unmistakable, shows both feet planted, and puts both plates and the whole bar "
                 "path in frame.\n"
                 "Finish is shown. `shoulder from_vertical 75` now tilts the humerus BACK along "
                 "the torso instead of abducting it sideways — the correction made to poses.json "
                 "after the first contact sheet reported a wide T. At the start the hands hang "
                 "low enough that the plates would be below the foot plane, which is a second "
                 "reason to read the finish.",
    },
    "Pull-Up": {
        "equipment": [{"prop": "pullup_bar", "mount": "bar_to_hands", "sink": True}],
        "azimuth": 0, "elevation": 5, "framing": 1.20, "airborne": True,
        "framing_exception":
            "The only exercise in the table at anything other than the shared distance, and the "
            "only one allowed past FRAMING_TOLERANCE. The arithmetic: the body rises 0.425 of "
            "its own height between the keys (box 0.199..1.250 hanging, 0.624..1.636 at the "
            "top), so the movement spans 1.437 against a frame that holds 1.324 — it overflows "
            "by 8.5% wherever the camera is pointed, and no framing choice at the shared "
            "distance holds it. Measured, not tuned.\n"
            "1.20 is the value, and how it was arrived at is the point. The first ruling was "
            "1.27, taken from a measurement made while the centre clamp in `hold_centre` was "
            "still firing and pushing the whole overflow onto one end. Once the movement fits "
            "the clamp goes quiet, the camera centres on the union, and the requirement drops; "
            "swept again afterwards, the rep is clipped at 1.10 (fit 1.05) and at 1.15 (1.00, "
            "touching), holds at 1.20 (0.96) and holds at 1.27 (0.90). The minimum is about "
            "1.17, so 1.20 keeps a real margin where 1.17 is the bare edge, and it cuts the "
            "consistency break from 21% to 17%. The general lesson, which cost this file four "
            "rounds to learn: measure the thing AFTER the change, not before it. "
            "1.27 makes the character 21% smaller in this one tile of fourteen, which is a "
            "consistency cost visible on a contact sheet and invisible to somebody watching one "
            "demonstration in an app; the alternative loses the chin clearing the bar, which is "
            "the instant the exercise exists to show, in the view that actually matters. Ruled "
            "the lesser cost by the brief's author on the measurement above. That it is the ONLY "
            "exception is what makes it defensible: see FRAMING_TOLERANCE.",
        "reason": "front: grip width and chin clearing the bar",
        "contact": "start", "showcase": "start",
        "notes": "The bar goes where the hands are, per the core rule, and with the root "
                 "translation fixed that is overhead: `pose_model.ground` anchors a hanging "
                 "figure by its hands at 1.25 of standing height and the prop's own bar is at "
                 "2.2 m = 1.257 of this character's height, so the two agree to within a "
                 "centimetre and the uprights stand on the floor. `sink` is kept only because "
                 "the agreement is not exact; the measured depth is printed per run and is "
                 "about 2 cm.\n"
                 "The START is shown — the dead hang, `shoulder flexion 180` with the elbows "
                 "straight — and not the finish the brief's reason names. Both were rendered. "
                 "The finish records `shoulder from_vertical 0` with the elbows bent 50, which "
                 "puts the hands at CHEST height; `ground` then hangs the figure by those hands, "
                 "so the tile is a figure floating in mid-air gripping a bar across its ribs. "
                 "That is not a chin over a bar and nothing in the staging can make it one. The "
                 "dead hang shows the grip width the brief asks for, and shows it better.",
    },
    "Lat Pulldown (seated, to the front)": {
        # equipment.lat_pulldown builds its seat pad centred at local x = -0.35 m with the mast
        # further along -X, so its user faces the machine's own local -X, hence `facing: -1`.
        "equipment": [{"prop": "lat_pulldown", "mount": "ground_under",
                       "anchor": (-0.35, 0.0, 0.0), "anchor_name": "seat",
                       "joint": "pelvis", "facing": -1, "drop_to_grip": True,
                       # equipment.lat_pulldown: the seat pad's top is its own LAT_SEAT_HEIGHT
                       "seat_top": 0.45}],
        "azimuth": 35, "elevation": 8, "framing": 1.0,
        "reason": "3/4 front: bar path to the chest",
        "contact": "start", "showcase": "start",
        "notes": "Anchored by the SEAT under the hips, on the floor, AND by the bar onto the "
                 "hands — both, now that `equipment.lat_pulldown` takes `drop=`. The machine's "
                 "seat-to-bar distance used to be fixed at a seated adult's proportions, which "
                 "forced a choice between a floating bar and a mast buried through the lifter's "
                 "chest; `drop_to_grip` solves the cable length for this figure's own grip "
                 "height instead, so the mast stands at zero and the bar is at the hands. The "
                 "The machine's vertical placement is measured, not assumed: `seat_top` puts "
                 "its seat pad against the body's own underside at the hips, found by a ray "
                 "cast (`seat_surface`). That is not cosmetic. This mascot's seated hip lands "
                 "at 0.209 of its height where an adult's is at 0.246, so a 0.45 m seat left on "
                 "the floor sits ABOVE its hips: the first render after the pose fix put the "
                 "seat pad at its waist and the thigh pad across its chest. Seated properly the "
                 "machine sinks 0.130, which also brings its 2.2 m mast inside the frame.\n"
                 "The START is shown, and it is the one place this table disagrees with its own "
                 "rule that the frame shown is the frame the contact was computed for. Both "
                 "were rendered. At the finish `poses.json` records no shoulder angle, so the "
                 "arms hang at the sides and the solved bar height is the HIP: a machine with "
                 "its bar at the lifter's waist, which nobody would name as a lat pulldown. At "
                 "the start the arms are overhead on `shoulder flexion 180` and the picture is "
                 "unmistakable. The cost is a measured 0.198 gap between the bar and the grip, "
                 "because the mast stands 0.25 m in front of the seat while overhead hands are "
                 "directly above the shoulders; at the finish that gap is 0.002.",
    },
    "Dumbbell Bicep Curl (standing, both arms)": {
        "equipment": [{"prop": "dumbbell", "kg": 10, "mount": "each_hand"}],
        "azimuth": 35, "elevation": 8, "framing": 1.0,
        "reason": "3/4 front: both arms visible, elbow stays pinned",
        "contact": "finish", "showcase": "finish",
        "notes": "A dumbbell per hand, in the palm, with the handle across the hand "
                 "(perpendicular to the forearm) rather than along it.",
    },
    "Dumbbell Lateral Raise": {
        "equipment": [{"prop": "dumbbell", "kg": 6, "mount": "each_hand"}],
        "azimuth": 0, "elevation": 5, "framing": 1.0,
        "reason": "front: the arc out to the sides is the whole movement",
        "contact": "finish", "showcase": "finish",
        "notes": "6 kg rather than the library's 10 kg default, because a lateral raise is a "
                 "light-dumbbell movement and `equipment.dumbbell` scales its heads by weight. "
                 "At the top of the arc the forearm points sideways, so the handle turns "
                 "front-to-back — which is how a dumbbell is actually held there. This is the "
                 "widest the figure itself ever gets, arms out to 1.1 units against a 1.32-unit "
                 "frame, and it is what sets the floor under `BASE_FRAMING`.",
    },
    "Dumbbell Shoulder Press (seated overhead press)": {
        # equipment.incline_bench centres its seat pad at local x = -0.0525 m (its own
        # -INCLINE_SEAT_LENGTH * 0.15) with the back pad hinged off the seat's rear edge toward
        # -X, so its user faces +X; the pad's top is at BENCH_HEIGHT like every other seat here.
        "equipment": [{"prop": "incline_bench", "degrees": 78, "mount": "ground_under",
                       "anchor": (-0.0525, 0.0, 0.0), "anchor_name": "seat",
                       "joint": "pelvis", "facing": 1, "seat_top": 0.45},
                      {"prop": "dumbbell", "kg": 10, "mount": "each_hand"}],
        "azimuth": 0, "elevation": 8, "framing": 1.0,
        "reason": "front: symmetry of the press",
        "contact": "finish", "showcase": "finish",
        "notes": "A bench, now that there is something to put one under. This entry read 'no "
                 "bench, deliberately' for as long as the `seated` orientation left the hip "
                 "unflexed and a pad under the pelvis would have cut through both thighs; "
                 "`pose_model` seats the figure properly as of this pass, and a seated overhead "
                 "press performed on thin air was the most obviously wrong tile on the sheet. "
                 "Upright at 78 degrees, which is how an adjustable bench is set for pressing. "
                 "The FINISH "
                 "is shown, both rendered: the finish is the overhead lockout, two dumbbells "
                 "straight up above the head, which nobody can mistake for anything else, while "
                 "the start's `shoulder flexion 37.5` with the elbows at 90 holds them in front "
                 "of the chest and reads as a man holding two dumbbells. (An earlier pass chose "
                 "the start on the grounds that the finish 'returns the hands to the hips'; that "
                 "was true of the pose data as it stood then and is not true of it now.)",
    },
    "Walking Lunge (dumbbells optional)": {
        "equipment": [{"prop": "dumbbell", "kg": 10, "mount": "each_hand"}],
        "azimuth": -90, "elevation": 8, "framing": 1.0,
        "reason": "side: both 90-degree angles at the bottom",
        "contact": "finish", "showcase": "finish",
        "notes": "Azimuth -90 (the figure's RIGHT side) rather than +90: the right leg is the "
                 "one that steps forward, so from this side the lead leg is nearest the camera "
                 "instead of hidden behind the trailing one. Dumbbells are 'optional' in the "
                 "movement's own name; they are staged because loaded is the common form and an "
                 "empty-handed lunge is what this task exists to stop. The stride was the one "
                 "case flagged as possibly needing more room than a shared distance gives; "
                 "measured, the figure spans 0.74 of a 1.32-unit frame at the bottom, so it "
                 "keeps the shared distance like everything else.",
    },
    "Front Plank (forearm plank)": {
        "equipment": [{"prop": "floor_mat", "mount": "under_body", "support": True,
                       "span": ("head", "ankle.left")}],
        "azimuth": 90, "elevation": 12, "framing": 1.0,
        "reason": "side: the straight line from heel to head is the cue",
        "contact": "finish", "showcase": "finish",
        "notes": "Mat under the body at the FINISH, the `front_support` phase, because that is "
                 "the phase where contact is defined — the start is `lying_prone` and rests on "
                 "the whole front of the body instead of on the forearms and toes. The mat lies "
                 "on the floor and the figure is grounded on the mat's own 15 mm top, the same "
                 "`support` mechanism the bench uses. The toes sit above it: the solved front "
                 "support does not put forearms and toes in one plane.",
    },
    "Tricep Pushdown (cable, bar or rope attachment)": {
        "equipment": [{"prop": "cable_tower", "attachment": "bar", "mount": "floor_cable",
                       "drop_to_grip": True, "clear_body": True}],
        "azimuth": 60, "elevation": 8, "framing": 1.10,
        "reason": "3/4 side: the elbow pinned to the ribs",
        "contact": "start", "showcase": "start",
        "notes": "Anchored by the attachment bar onto the hands at the START (elbows up, the "
                 "top of the pushdown), which is where a cable is taut, with the tower standing "
                 "on the floor: `drop_to_grip` solves the cable length for the grip height "
                 "instead of sinking the column until its fixed 0.55 m cable reaches. Measured "
                 "gap between the bar and the grip: 0.000.\n"
                 "The only `framing` multiplier in the table, and the reason is that this "
                 "exercise IS its machine. At the shared distance the pulley head is just off "
                 "the top of the frame, so the cable enters the picture from nowhere and the "
                 "tower reads as a pole; 1.10 is the smallest multiplier that holds the whole "
                 "tower (measured — `fits-at` says 1.1), and it costs this tile a head-bone "
                 "measure of 12.40% against the set's 12.37-13.73%, which is to say nothing. "
                 "The column sits directly under its own pulley, so it would stand inside the "
                 "lifter; it is slid forward along the figure's own facing by exactly enough to "
                 "clear the figure's measured silhouette, less the pulley head's own reach, "
                 "which is what lets the bar sit exactly at the hands rather than a cable's "
                 "width in front of them. The column still crosses the lifter's near arm from "
                 "this azimuth; that is the brief's angle and it is the one thing here I would "
                 "still change if the angle were mine.",
    },
    "Leg Press (seated machine)": {
        # equipment.leg_press puts its seat at rail_base + (-0.12, 0.16, 0.04) m, i.e. local
        # (-0.12, -0.02, 0.10); its sled rises along local +X, so the user faces +X.
        "equipment": [{"prop": "leg_press", "mount": "ground_under",
                       "anchor": (-0.12, -0.02, 0.10), "anchor_name": "seat",
                       # equipment.leg_press: the seat's centre is at local z 0.10 and its pad
                       # is 0.09 thick, so the surface a rider sits on is at 0.145.
                       "joint": "pelvis", "facing": 1, "seat_top": 0.145}],
        "azimuth": 90, "elevation": 10, "framing": 1.0,
        "reason": "side: the sled travel",
        "contact": "start", "showcase": "start",
        "notes": "Anchored by the SEAT under the hips at the START, the knees-bent phase, with "
                 "`seat_top` putting the pad against the rider's own underside — the same "
                 "measured seating the lat pulldown gets, and here it lifts the machine 0.040 "
                 "rather than sinking it, because this prop's seat is LOWER than the rider sits "
                 "where the pulldown's is higher. With `pose_model` seating the figure, this "
                 "tile now reads as a person in a leg press rather than a person kneeling "
                 "beside one.\n"
                 "The brief names the foot plate as a both-keyframe contact. It is still not "
                 "met and it is now clear why. Anchoring the plate to the feet instead of the "
                 "seat buries the machine (the plate sits 0.46 above its own rail base and the "
                 "rider's feet are at 0.06), and sliding the plate DOWN the rails to the feet "
                 "was built, measured and taken out again: this prop's rails are offset 0.18 m "
                 "to one side of its seat and rise at 45 degrees from behind it, so the line "
                 "they trace never passes near a seated rider's feet — the closest the plate "
                 "gets is 0.205 of a body height away, against the shins, which renders as a "
                 "slab through both legs. `equipment.leg_press`'s own docstring records that "
                 "experiment. The feet rest on the machine's base frame, which is at least "
                 "something the machine owns.",
    },
}


def support_height(name, body):
    """The height the body rests at: the floor, or a support prop's own top surface.

    Thirteen of the fourteen are the floor. A bench press rests on a bench, and nothing in the
    form notes knows that: `pose_model.ground` lies a `lying_supine` figure flat on the ground,
    so a bench staged under it had to be sunk below the floor to meet it. Asking the prop how
    tall it is instead, and grounding the figure onto that, puts the body on the pad and the pad
    on the floor.

    Called by `animate` before any frame is posed, so it must not need a posed rig — it reads
    `equipment`'s own constants and the character's unposed height, nothing else.
    """
    unit = rest_height(body) / 1.75
    highest = 0.0
    for spec in staging_for(name)["equipment"]:
        if not spec.get("support"):
            continue
        top = SUPPORT_TOP.get(spec["prop"])
        if top is None:
            raise ValueError(f"{name!r} marks {spec['prop']!r} as the body's support, but there "
                             f"is no resting surface recorded for it in SUPPORT_TOP "
                             f"({sorted(SUPPORT_TOP)})")
        highest = max(highest, top() * unit)
    return highest


# --- reading the posed body ---------------------------------------------------------------------

def _joint_bones():
    """joint name -> (bone, which end), inverted from `make_character.BONES`.

    A joint is a bone's head or another bone's tail, and both report the same posed position, so
    whichever is found first is as good as the other.
    """
    table = {}
    for bone, (head, tail) in make_character.BONES.items():
        table.setdefault(head, (bone, "head"))
        table[tail] = (bone, "tail")
    return table


def joints_of(rig):
    """Every joint's WORLD position, read from the posed armature.

    Not from the solved skeleton: see "Read the POSED RIG" in the module docstring. `pose_bone`'s
    `.head`/`.tail` are the evaluated positions in armature space, so one multiply by the rig's
    own world matrix is the whole conversion.
    """
    bpy.context.view_layer.update()
    out = {}
    for joint, (bone, end) in _joint_bones().items():
        if bone not in rig.pose.bones:
            continue
        posed = rig.pose.bones[bone]
        out[joint] = rig.matrix_world @ (posed.head if end == "head" else posed.tail)
    return out


def body_axes(joints):
    """The figure's own (lateral, forward, up) unit axes, measured from the posed skeleton.

    Every placement below is expressed in these, which is what lets one table serve a standing
    squat and a horizontal plank without a special case: "behind the neck" is -forward whichever
    way the torso is pointing, and "under the back" is -forward when the figure is supine just as
    much as it is when the figure is upright.

    `forward` is derived rather than measured because no joint pair points out of the chest:
    up x lateral is +Y for a figure standing with its shoulders along +X, which is the direction
    `pose_model` has the character facing.
    """
    lateral = (joints["shoulder.right"] - joints["shoulder.left"])
    up = (joints["chest"] - joints["pelvis"])
    if lateral.length < 1e-9 or up.length < 1e-9:
        raise ValueError("the posed rig has a zero-length torso or shoulder span")
    up = up.normalized()
    lateral = (lateral - up * lateral.dot(up))      # drop any component along the spine
    if lateral.length < 1e-9:
        raise ValueError("shoulder span is parallel to the spine; cannot derive a body frame")
    lateral = lateral.normalized()
    return lateral, up.cross(lateral), up


def grip(joints, side):
    """The middle of the palm: `PALM` of the way along the hand bone from the wrist."""
    wrist = joints[f"wrist.{side}"]
    tip = joints[f"hand.{side}"]
    return wrist + (tip - wrist) * PALM


def basis_from(long_axis, up_hint):
    """A rotation whose local X is `long_axis` and whose local Z leans toward `up_hint`.

    Every bar-shaped prop in `equipment` is built along its own local X (the barbell, the
    dumbbell handle, the pull-up bar, the lat bar, the cable attachment), so aiming local X is
    all that is ever needed; `up_hint` only settles the roll about that axis, which on a round
    bar is invisible but matters for the dumbbell's hex heads and the lat bar's ends.
    """
    x = Vector(long_axis).normalized()
    z = Vector(up_hint) - x * Vector(up_hint).dot(x)
    if z.length < 1e-6:                             # up_hint parallel to the bar: pick anything
        z = Vector((0.0, 0.0, 1.0)) - x * x.z
        if z.length < 1e-6:
            z = Vector((0.0, 1.0, 0.0))
    z.normalize()
    return Matrix((x, z.cross(x), z)).transposed()  # columns are the local axes


def body_middle(obj):
    """The centre of an object's posed bounding box."""
    corners = world_bbox(obj)
    return Vector((
        (max(c.x for c in corners) + min(c.x for c in corners)) / 2,
        (max(c.y for c in corners) + min(c.y for c in corners)) / 2,
        (max(c.z for c in corners) + min(c.z for c in corners)) / 2,
    ))


def world_bbox(obj):
    """An object's world-space bounding box corners, as it is actually posed."""
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    return [evaluated.matrix_world @ Vector(corner) for corner in evaluated.bound_box]


def rest_height(body):
    """The character's height UNPOSED, measured off its own undeformed mesh.

    Two things need this and neither may read the posed figure: the size every prop is built at
    (`equipment`'s functions all take the character's height as their first argument and derive
    every real-world dimension from it) and the shared camera distance.

    `object.bound_box` is not the undeformed mesh — Blender evaluates it through the armature
    modifier, so a plank's `bound_box` is 0.44 units tall and a bench press's is 0.25, not the
    character's 1.0. An earlier version of this module read it anyway and the consequences were
    invisible on the standing exercises and absurd on the rest: the plank got a half-size mat,
    the bench press a bench a third too short and a barbell 0.55 units long instead of 1.31, and
    the camera came in to half its distance, all without an error. `body.data.vertices` IS the
    undeformed mesh — the modifier is evaluated downstream of it — so that is what gets measured,
    once per object, cached on the object because 46,000 vertices is not free.
    """
    if "rest_height" not in body:
        zs = [(body.matrix_world @ vertex.co).z for vertex in body.data.vertices]
        body["rest_height"] = (max(zs) - min(zs)) if zs else 0.0
    return float(body["rest_height"])


# --- attaching ----------------------------------------------------------------------------------

def hold_at(obj, rig, bone, point, basis):
    """Parent `obj` to `bone` so that it lands at world `point` with world orientation `basis`.

    `equipment.hold` does the parenting and cancels Blender's bone-parent-at-the-TAIL quirk, so
    its `offset` and `rotation` are measured from the bone's head in the bone's own axes. This
    inverts that: the caller says where in the world the prop goes, and the bone's current posed
    matrix converts it. Expressing placements in world space is what lets them be derived from
    the body's own measured frame instead of from per-bone magic numbers, and because the result
    is still a bone parent the prop follows the bone through the whole animation.
    """
    posed = rig.pose.bones[bone]
    rotation = (rig.matrix_world @ posed.matrix).to_3x3()
    head = rig.matrix_world @ posed.head
    inverse = rotation.inverted()
    offset = inverse @ (Vector(point) - head)
    local = (inverse @ basis).to_euler()
    equipment.hold(obj, rig, bone, offset=tuple(offset), rotation=tuple(local))
    bpy.context.view_layer.update()
    landed = obj.matrix_world.translation
    if (landed - Vector(point)).length > 1e-4:
        print(f"  WARNING {obj.name}: asked for {tuple(round(v, 4) for v in point)}, "
              f"landed at {tuple(round(v, 4) for v in landed)}")
    return obj


def _check_contact(obj, local_point, label, tolerance=0.05):
    """Warn if `local_point` is nowhere near `obj`'s own surface.

    The contact points for the lat pulldown, the cable tower and the leg press are re-derived
    here from expressions that live as local literals inside `equipment.py`, so they can silently
    rot if that file is edited. This is the guard: the point a prop is anchored by has to be ON
    the prop. Tolerance is in Blender units, generous enough to allow for the half-thickness of
    whatever part the point is meant to be the centre of.
    """
    point = Vector(local_point)
    # `closest_point_on_mesh` reads the surface, not the vertices: the bar these anchors sit on
    # is a cylinder whose only vertices are at its two ends, so a nearest-vertex test would
    # report half a bar's length and cry wolf on every correctly placed prop.
    hit, landed, _, _ = obj.closest_point_on_mesh(point)
    nearest = (landed - point).length if hit else None
    if nearest is None or nearest > tolerance:
        print(f"  WARNING {obj.name}: {label} anchor {tuple(round(v, 3) for v in point)} is "
              f"{nearest:.3f} from its nearest surface — equipment.py's geometry may have moved")
    return obj


def _yaw_to(direction):
    """The Z rotation that turns a prop's local +X onto `direction`'s horizontal part."""
    return math.atan2(direction.y, direction.x)


def _mount_both_hands(obj, rig, joints, axes, spec, unit, body=None):
    """A bar held in both hands: centred between the two palms, along the line joining them."""
    left, right = grip(joints, "left"), grip(joints, "right")
    span = right - left
    if span.length < 1e-6:
        span = axes[0]
    centre = (left + right) / 2
    hold_at(obj, rig, "hand.left", centre, basis_from(span, axes[2]))
    return {"sink": False}


def _mount_each_hand(obj, rig, joints, axes, spec, unit, body=None):
    """Not used directly — `stage_exercise` builds one prop per hand and calls `_mount_in_hand`."""
    raise AssertionError("each_hand is expanded by stage_exercise")


def _mount_in_hand(obj, rig, joints, axes, side, unit):
    """One dumbbell in one palm, handle across the hand rather than along the forearm.

    A dumbbell's handle is perpendicular to the forearm, and which perpendicular is set by the
    arm's own position: at the top of a curl it is roughly lateral, at the top of a lateral raise
    the arm IS lateral and the handle turns front-to-back. Both fall out of one cross product of
    the forearm against the body's up axis, so there is no per-exercise special case.
    """
    _, _, up = axes
    forearm = joints[f"wrist.{side}"] - joints[f"elbow.{side}"]
    across = forearm.cross(up)
    if across.length < 1e-6:                        # forearm vertical: fall back to lateral
        across = axes[0]
    hold_at(obj, rig, f"hand.{side}", grip(joints, side), basis_from(across, up))
    return {"sink": False}


def back_surface(body, point, forward, reach=0.6):
    """Where the figure's own back is, directly behind `point`.

    Found by firing a ray at the body from well behind it, along `forward`, through `point`: the
    first thing it hits is the skin on the back at exactly that height. Returns None if it misses
    (which would mean `point` is not over the torso at all).

    This replaces a fraction-of-torso-depth guess, and the reason is this character's head. The
    first version put the squat bar a third of a torso depth behind the shoulder line, which is
    where a trap is on an adult and is INSIDE the skull on a mascot whose head is a quarter of
    its height: the bar rendered through the hair. The mesh knows where its own back is at any
    height, and unlike a fraction it keeps knowing when the model changes.
    """
    evaluated = body.evaluated_get(bpy.context.evaluated_depsgraph_get())
    to_local = body.matrix_world.inverted()
    origin = to_local @ (Vector(point) - Vector(forward) * reach)
    direction = (to_local.to_3x3() @ Vector(forward)).normalized()
    hit, location, _, _ = evaluated.ray_cast(origin, direction, distance=reach * 2)
    return (body.matrix_world @ location) if hit else None


def seat_surface(body, point, up, reach=0.6):
    """Where the figure's own underside is, directly below `point` — what it sits ON.

    The same ray cast `back_surface` does for the squat bar, fired upward from under the hips
    instead of backward from behind the shoulders, and for the same reason: a fraction of a
    torso is a guess and the mesh knows.

    It exists because the `seated` orientation now genuinely seats this figure, and that turned
    a pose problem into a proportions problem. The solver sits an adult's hip at 0.246 of
    standing height, which is the 0.43 m that `equipment`'s 0.45 m seats are built for. This
    character is a mascot with a quarter of its height in its head and correspondingly short
    shins, so its seated hip lands at 0.209 — 0.37 m in human terms. A 0.45 m seat under it is
    not a seat, it is a pad through its waist, which is exactly what the first render after the
    pose fix showed on the lat pulldown.
    """
    return back_surface(body, point, up, reach=reach)


def _seat_base(spec, joints, axes, body, unit):
    """How far a seated machine has to move vertically to put its seat under this figure.

    Zero for a machine that simply stands on the floor. For one whose entry names a `seat_top`,
    the machine is placed so that surface meets the body's own underside at the hips — props
    follow the body, which is this task's first rule, and a seat is as literal a contact as the
    rule ever gets. The result is usually negative (the machine sinks, because gym seats are
    taller than this character sits) and whatever that buries is hidden by the floor.
    """
    if spec.get("seat_top") is None:
        return 0.0
    _, _, up = axes
    hips = joints[spec.get("joint", "pelvis")]
    surface = seat_surface(body, hips, up)
    if surface is None:
        print(f"  WARNING {spec['prop']}: no body surface found under the hips; standing it on "
              f"the floor instead of seating it")
        return 0.0
    return surface.z - spec["seat_top"] * unit


def _mount_upper_back(obj, rig, joints, axes, spec, unit, body=None):
    """A barbell racked across the back, behind and below the base of the neck.

    Parented to the spine (not to a hand) because a back squat's bar is carried by the trunk — if
    it followed a hand it would ride down with the arms as they hang.

    The height is a little below the shoulder line, measured down the figure's own spine, which
    both keeps the bar clear of this character's very large head and is where a bar actually sits
    on a squatter: across the shelf of muscle below the neck, not on the neck itself. The depth
    is not guessed at all — see `back_surface`.
    """
    lateral, forward, up = axes
    shoulders = (joints["shoulder.left"] + joints["shoulder.right"]) / 2
    depth = max((joints["chest"] - joints["pelvis"]).length, 1e-6)
    along_spine = shoulders - up * (depth * 0.22)
    bar_radius = (equipment.BARBELL_BAR_DIAMETER / 2) * unit
    skin = back_surface(body, along_spine, forward)
    if skin is None:                                # not over the torso: fall back to a fraction
        print("  WARNING upper_back: no back surface found behind the shoulders; "
              "falling back to a third of a torso depth")
        point = along_spine - forward * (depth * 0.33)
    else:
        point = skin - forward * bar_radius         # resting ON the back, not inside it
    hold_at(obj, rig, "spine", point, basis_from(lateral, up))
    return {"sink": False}


def _mount_under_body(obj, rig, joints, axes, spec, unit, body=None):
    """A pad or a mat the body rests on: its top surface meets the body's own lowest surface.

    The lowest surface, not the lowest BONE: a forearm's bone line is a few centimetres above the
    skin that actually touches the mat, and a bench placed to the spine would have the back
    sunk into the pad. The posed mesh knows where its own skin is, so ask it.

    Laid along the span the entry names (head-to-hip for a bench, head-to-ankle for a mat) so it
    runs with the body rather than across it, and centred on that span.
    """
    lateral, forward, up = axes
    first, second = spec.get("span", ("head", "hip.left"))
    along = joints[second] - joints[first]
    if along.length < 1e-6:
        along = up
    centre = (joints[first] + joints[second]) / 2
    underside = min(corner.z for corner in world_bbox(body))

    local = [Vector(corner) for corner in obj.bound_box]
    top = max(corner.z for corner in local)         # the pad's own top, in its local space
    flat = Vector((along.x, along.y, 0.0))
    if flat.length < 1e-6:
        flat = Vector((lateral.x, lateral.y, 0.0))
    # `flat_bench` is built long on X, `floor_mat` long on Y, so the yaw that lines a prop up
    # with the body depends on which; read it off the prop's own bounding box rather than
    # hard-coding a name.
    long_is_y = (max(c.y for c in local) - min(c.y for c in local)) > \
                (max(c.x for c in local) - min(c.x for c in local))
    yaw = _yaw_to(flat) - (math.pi / 2 if long_is_y else 0.0)

    # On the floor, at zero, like a real one. The figure was already grounded onto this prop's
    # own top surface by `animate.ground_rig` (`support_height` told it how high that is), so
    # the two should meet without either being moved to suit the other. That is worth checking
    # rather than assuming: the whole reason this module used to sink the bench below the floor
    # was that nobody had measured the gap.
    base = 0.0 if spec.get("support") else underside - top
    equipment.stand(obj, location=(centre.x, centre.y, base), rotation=(0, 0, yaw))
    if spec.get("support"):
        bpy.context.view_layer.update()
        gap = underside - max(corner.z for corner in world_bbox(obj))
        if abs(gap) > 0.01:
            print(f"  WARNING {obj.name}: the body's underside is {gap:+.4f} from the pad's top "
                  f"— `support_height` and this prop's own geometry disagree")
    return {"sink": False}


def _mount_bar_to_hands(obj, rig, joints, axes, spec, unit, body=None):
    """A floor-standing frame whose own bar is brought to the hands.

    One prop uses this now, the pull-up bar, and it costs almost nothing: `pose_model.ground`
    anchors a hanging figure with its hands at `HANGING_BAR` = 1.25 of standing height and this
    prop's bar is at `PULLUP_BAR_HEIGHT` = 2.2 m, which on a 0.9995-unit character is 1.257 — so
    the bar meets the hands with its uprights within a centimetre of standing on the floor. The
    depth it ends up below zero is measured and returned rather than waved at, because this used
    to be two metres of buried upright and the number is the thing that says it no longer is.
    """
    lateral, forward, up = axes
    left, right = grip(joints, "left"), grip(joints, "right")
    span = right - left
    centre = (left + right) / 2
    name = spec["prop"]
    if name != "pullup_bar":
        raise ValueError(f"bar_to_hands stages the pull-up bar; {name!r} should stand on the "
                         f"floor and solve its own cable `drop=` for the grip instead")
    local = Vector((0.0, 0.0, equipment.PULLUP_BAR_HEIGHT * unit))
    _check_contact(obj, local, f"{name} bar")

    # Yaw first: the prop's bar is built along its local X, and it has to end up along the line
    # joining the two hands. With the yaw known, the anchor's own offset rotates with it, which
    # is why the translation is computed after and not before.
    if span.length < 1e-6:
        span = lateral
    yaw = _yaw_to(span)
    turn = Matrix.Rotation(yaw, 3, "Z")
    equipment.stand(obj, location=tuple(centre - turn @ local), rotation=(0, 0, yaw))
    bpy.context.view_layer.update()
    depth = -min(corner.z for corner in world_bbox(obj))
    return {"sink": bool(spec.get("sink")), "depth_below_floor": depth}


def _mount_floor_cable(obj, rig, joints, axes, spec, unit, body=None):
    """A cable tower standing on the floor with its attachment already at the hands' height.

    The height is not this mount's problem: `_drop_to_grip` has already solved the prop's own
    `drop=` for this figure's grip, so standing the tower at zero puts the attachment exactly
    where the hands are. What is left is the horizontal placement, and that is a genuine
    conflict — `equipment.cable_tower`'s pulley sits directly above its own column, so the bar
    it hangs cannot be anywhere the column is not, and the hands on a pushdown are barely a
    hand's width in front of the belly. Placed honestly, the column stands inside the lifter.

    So the machine slides forward along the figure's own facing by exactly enough to clear the
    figure's measured silhouette, which leaves the bar a few centimetres in front of the grip
    rather than inside the ribcage. Measured off the body, not a constant.
    """
    lateral, forward, up = axes
    left, right = grip(joints, "left"), grip(joints, "right")
    span = right - left
    if span.length < 1e-6:
        span = lateral
    centre = (left + right) / 2
    # The attachment hangs `CABLE_TOWER_REACH` off the column's axis, on the side the user is
    # on, and `_drop_to_grip` has already set its height — so this is the prop's own attachment
    # point in its own local space, and the placement below is just "put that point there".
    hang = equipment.CABLE_TOWER_REACH * unit
    attach = Vector((0.0, -hang, centre.z))
    _check_contact(obj, attach, f"{spec['prop']} attachment")

    if spec.get("clear_body") and body is not None:
        middle = body_middle(body)
        reach = max((Vector(c) - middle).dot(forward) for c in world_bbox(body))
        clearance = spec.get("clearance", 0.05) * (unit * 1.75)
        # What has to clear the body is the COLUMN, which stands `hang` further forward than the
        # bar does. Taking that off the push is what lets the bar sit at the hands rather than a
        # cable's reach in front of them.
        push = max(0.0, reach + clearance - hang - (centre - middle).dot(forward))
        centre = centre + forward * push
    yaw = _yaw_to(span)
    turn = Matrix.Rotation(yaw, 3, "Z")
    placed = centre - turn @ Vector((attach.x, attach.y, 0.0))
    equipment.stand(obj, location=(placed.x, placed.y, 0.0), rotation=(0, 0, yaw))
    return {"sink": False}


def _mount_ground_under(obj, rig, joints, axes, spec, unit, body=None):
    """A floor-standing machine, stood on the figure's own ground with a named part of itself
    under a named joint — the lat pulldown's seat under the hips, say.

    Both machines that use it stand at zero, on the same floor as the figure, and put a named
    part of themselves under a named joint. Where the same prop also has to reach the hands,
    `drop_to_grip` solves its cable length for that instead of asking this mount to compromise
    — which is what the first version of this module had to do, and why its lat pulldown either
    floated its bar or drove its mast through the lifter's chest.
    """
    lateral, forward, up = axes
    _check_contact(obj, Vector(spec["anchor"]) * unit,
                   f"{spec['prop']} {spec.get('anchor_name', 'anchor')}", tolerance=0.12)
    origin, turn = _machine_frame(spec, joints, axes, unit)
    base = spec.get("base_z", 0.0)                  # the floor, unless a seat says otherwise
    equipment.stand(obj, location=tuple(origin), rotation=(0, 0, turn.to_euler().z))
    result = {"sink": bool(spec.get("sink")) or base < -1e-6}
    if abs(base) > 1e-6:
        result["seat_drop"] = base
    return result


def _machine_frame(spec, joints, axes, unit):
    """Where a `ground_under` machine's own origin lands, and which way it is turned.

    Pulled out so that the mount and anything that has to run BEFORE the prop is built read one
    piece of placement arithmetic rather than two copies of it that can drift apart. The leg
    press's abandoned sled solver was the second copy that made the point.
    """
    lateral, forward, up = axes
    anchor = Vector(spec["anchor"]) * unit
    target = joints[spec.get("joint", "pelvis")]
    facing = forward if spec.get("facing", 1) > 0 else -forward
    turn = Matrix.Rotation(_yaw_to(facing), 3, "Z")
    placed = turn @ Vector((anchor.x, anchor.y, 0.0))
    origin = Vector((target.x - placed.x, target.y - placed.y, spec.get("base_z", 0.0)))
    return origin, turn


def _mount_plate_to_feet(obj, rig, joints, axes, spec, unit, body=None):
    """The leg press, brought up the rails until its foot plate is at the feet."""
    lateral, forward, up = axes
    feet = (joints["foot.left"] + joints["foot.right"]) / 2
    tilt = math.radians(LEG_PRESS_TILT)
    rail = Vector(LEG_PRESS_RAIL_BASE) * unit
    top = rail + Vector((math.cos(tilt), 0.0, math.sin(tilt))) * \
        (equipment.LEG_PRESS_RAIL_LENGTH * unit)
    local = top + Vector(LEG_PRESS_PLATE_OFFSET) * unit
    _check_contact(obj, local, "leg_press foot plate")

    # The machine's sled runs up its own local +X, so its user faces that way; turn it so that
    # direction is the figure's own forward.
    yaw = _yaw_to(forward)
    turn = Matrix.Rotation(yaw, 3, "Z")
    equipment.stand(obj, location=tuple(feet - turn @ local), rotation=(0, 0, yaw))
    return {"sink": bool(spec.get("sink"))}


def _drop_to_grip(spec, joints, unit):
    """The cable length, in metres, that puts a machine's attachment at this figure's own grip.

    Both cable props hang their attachment a fixed distance below a pulley whose height is a
    property of the machine, and both stand on the floor at zero, so the whole of the placement
    is this one number: pulley height minus the hands' height. Returned in metres because that
    is what `equipment`'s builders take.

    It is solved rather than looked up because the hands are wherever the pose puts them, and
    the pose is different in every exercise and both phases of each. A fixed drop is what made
    these two the worst tiles on the first contact sheet.
    """
    grip_z = (grip(joints, "left").z + grip(joints, "right").z) / 2 - spec.get("base_z", 0.0)
    pulleys = {
        "lat_pulldown": equipment.LAT_COLUMN_HEIGHT - equipment.LAT_PULLEY_INSET,
        "cable_tower": equipment.CABLE_TOWER_HEIGHT - equipment.CABLE_PULLEY_INSET,
    }
    if spec["prop"] not in pulleys:
        raise ValueError(f"{spec['prop']!r} has no pulley height recorded, so there is no cable "
                         f"drop to solve; `drop_to_grip` applies to {sorted(pulleys)}")
    drop = pulleys[spec["prop"]] - grip_z / unit
    if drop <= 0.0:
        raise ValueError(f"{spec['prop']!r}: the grip is at {grip_z / unit:.3f} m, at or above "
                         f"its pulley at {pulleys[spec['prop']]:.3f} m — the attachment cannot "
                         f"hang up to it. Stage this exercise from the other phase, or use a "
                         f"taller machine.")
    return drop


_MOUNTS = {
    "both_hands": _mount_both_hands,
    "each_hand": _mount_each_hand,
    "upper_back": _mount_upper_back,
    "under_body": _mount_under_body,
    "bar_to_hands": _mount_bar_to_hands,
    "floor_cable": _mount_floor_cable,
    "ground_under": _mount_ground_under,
    "plate_to_feet": _mount_plate_to_feet,
}

_BUILDERS = {
    "barbell": lambda height, spec: equipment.barbell(height, plates=spec.get("plates", True)),
    "dumbbell": lambda height, spec: equipment.dumbbell(height, kg=spec.get("kg", 10)),
    "flat_bench": lambda height, spec: equipment.flat_bench(height),
    "incline_bench": lambda height, spec: equipment.incline_bench(
        height, degrees=spec.get("degrees", 75)),
    "pullup_bar": lambda height, spec: equipment.pullup_bar(height),
    "lat_pulldown": lambda height, spec: equipment.lat_pulldown(
        height, drop=spec.get("drop")),
    "cable_tower": lambda height, spec: equipment.cable_tower(
        height, attachment=spec.get("attachment", "bar"), drop=spec.get("drop")),
    "leg_press": lambda height, spec: equipment.leg_press(height),
    "squat_rack": lambda height, spec: equipment.squat_rack(height),
    "floor_mat": lambda height, spec: equipment.floor_mat(height),
}


# --- the set ------------------------------------------------------------------------------------

LENS = 60.0          # mm, as make_character.stage
SENSOR = 36.0        # mm of vertical sensor, as make_character.stage
MARGIN = 1.06        # as make_character.stage


def _home(obj):
    """The object's position the first time staging saw it, remembered on the object itself.

    The lights and the backdrop are moved on every call, so their original positions have to come
    from somewhere other than their current ones. Reading them off the objects `make_character`
    built, once, and stashing them as a custom property keeps `make_character.stage`'s own light
    rig as the single source of those numbers — copying them into this file as literals would
    have meant two places to change.
    """
    if "stage_home" not in obj:
        obj["stage_home"] = tuple(obj.location)
    return Vector(tuple(obj["stage_home"]))


def camera_distance(subject, framing=1.0):
    """`make_character.stage`'s own framing maths, on the CHARACTER and on nothing else.

    Vertical field of view from the lens and the sensor, then far enough back to hold the
    subject's own height with a 1.06 margin, times `BASE_FRAMING`. Two things about that:

      * The height is the UNPOSED one (`rest_height`). It is the same in every pose, so the
        character is the same size in all fourteen frames.
      * Nothing about the EQUIPMENT enters this. That was the bug: the first version let each
        camera retreat until its own props fitted, which handed the distance to whatever was
        widest in the scene and ran the character's share of the frame from 80% down to 27%
        across the set. `framing` survives for the cases where a movement genuinely needs more
        room than the character's own height, and `stage_exercise` refuses one further than
        `FRAMING_TOLERANCE` from 1.0.
    """
    extent = rest_height(subject)
    fov = 2 * math.atan((SENSOR / 2) / LENS)
    base = ((extent * MARGIN / 2) / math.tan(fov / 2)) * BASE_FRAMING
    return base * framing, extent, fov


def orbit(azimuth, elevation):
    """The unit vector from the figure's centre to the camera.

    Azimuth 0 is straight in front of the figure, which faces +Y, so 0 gives +Y and
    `make_character.stage`'s own camera position; +90 swings round to -X, the figure's left.
    """
    a, e = math.radians(azimuth), math.radians(elevation)
    return Vector((-math.sin(a) * math.cos(e), math.cos(a) * math.cos(e), math.sin(e)))


def camera_axes(azimuth, elevation):
    """The camera's own right and up vectors in world space, for a given orbit.

    Built from the same `to_track_quat` call `place_camera` uses, rather than from a cross
    product that would have to agree with it by hand.
    """
    offset = orbit(azimuth, elevation)
    basis = (-offset).to_track_quat("-Z", "Y").to_matrix()
    return basis.col[0], basis.col[1]


def hold_centre(union, judged, azimuth, elevation, distance, fov):
    """Where to point the camera: at the whole movement, without losing the frame being judged.

    Centring on the union of both phases is what lets one staged camera hold a rep. It has one
    failure mode and this is the guard for it: when the union is LARGER than the frame can hold
    at the shared distance, centring on it pushes both ends out equally, and if the union is big
    because of something other than the movement, the result is a still that is worse without an
    animation that is better. The front plank is exactly that case — `pose_model.ground` slides
    its body 0.579 along the floor between `lying_prone` and `front_support` because the two
    orientations rest on different sets of joints, so its union is 1.59 long where the figure is
    1.01, and framing for it pushed the plank half out of the tile.

    So: aim at the union's centre, but never further than the point at which the phase this
    frame actually shows would leave the frame. When the union fits, the clamp never fires and
    this returns the union's own centre. When it does not, the judged phase stays whole and the
    overflow is pushed entirely onto the other end of the movement, where `fit_corners` measures
    it and the caller prints it.

    The clamp is orthographic — it ignores the perspective divide — which makes it slightly
    conservative. The fit check afterwards is not, and reports the truth.
    """
    right, up = camera_axes(azimuth, elevation)
    half_h = distance * math.tan(fov / 2)
    scene = bpy.context.scene
    half_w = half_h * (scene.render.resolution_x / max(scene.render.resolution_y, 1))
    centre = Vector((
        (max(c.x for c in union) + min(c.x for c in union)) / 2,
        (max(c.y for c in union) + min(c.y for c in union)) / 2,
        (max(c.z for c in union) + min(c.z for c in union)) / 2,
    ))
    for axis, half in ((right, half_w), (up, half_h)):
        here = centre.dot(axis)
        lo = min(c.dot(axis) for c in judged)
        hi = max(c.dot(axis) for c in judged)
        if hi - lo > 2 * half:                      # the judged phase alone does not fit
            wanted = (lo + hi) / 2
        else:
            wanted = min(max(here, hi - half), lo + half)
        centre = centre + axis * (wanted - here)
    return centre


def place_camera(camera, centre, azimuth, elevation, distance):
    offset = orbit(azimuth, elevation) * distance
    camera.location = centre + offset
    camera.rotation_mode = "QUATERNION"
    camera.rotation_quaternion = (-offset).to_track_quat("-Z", "Y")
    camera.data.lens = LENS
    camera.data.sensor_fit = "VERTICAL"
    camera.data.sensor_height = SENSOR
    bpy.context.scene.camera = camera
    return camera


def place_set(centre, azimuth, props):
    """Floor, backdrop and lights, turned to suit this exercise's camera.

    Two fixes over `make_character.stage`, which only ever had to serve one camera:

      * the backdrop orbits with the camera. It is a 12-unit plane standing at y = -1.8 facing
        the default camera at +Y; from any other azimuth it would have been seen edge-on and the
        render would have come back as an empty frame. It also backs off far enough to stay
        behind the equipment, which can reach further from the figure than the figure does.
      * the lights orbit with the camera, keeping the same rig relative to the view. Fixed
        lights would have lit every side and 3/4 view from behind.
      * a fourth lamp, `SHADOW_LIGHT`, which exists to put a shadow under the figure and is
        light-linked so that it illuminates the FLOOR and nothing else. See `_place_shadow`.

    The floor stays at z = 0, where a floor belongs. An earlier version moved it to meet a
    figure that was never translated onto it; that is fixed in `animate.ground_rig` now, and
    this function no longer takes a ground height at all.
    """
    a = math.radians(azimuth)
    away = Vector((math.sin(a), -math.cos(a), 0.0))

    if "floor" in bpy.data.objects:
        floor = bpy.data.objects["floor"]
        floor.location = (centre.x, centre.y, 0.0)

    if "backdrop" in bpy.data.objects:
        backdrop = bpy.data.objects["backdrop"]
        home = _home(backdrop)
        reach = home.length                         # 1.8 in make_character.stage
        for prop in props:
            for corner in world_bbox(prop):
                reach = max(reach, (corner - centre).dot(away) + 0.4)
        backdrop.location = (centre.x + away.x * reach, centre.y + away.y * reach, 0.0)
        # The plane is created with a 90 degree turn about X, which points its normal along -Y;
        # turning it a further `a` about Z keeps that normal on `away`, square to the camera.
        backdrop.rotation_euler = (math.pi / 2, 0.0, a)

    turn = Matrix.Rotation(a, 3, "Z")
    base = Vector((centre.x, centre.y, 0.0))
    for name in ("key", "fill", "rim"):
        if name not in bpy.data.objects:
            continue
        light = bpy.data.objects[name]
        home = _home(light)
        light.location = base + turn @ home
        # make_character.stage aims its lamps at (0, 0, 0.8) — a figure's chest, for a figure
        # standing at the origin with its feet on the floor. The same point, moved with the set.
        target = base + turn @ Vector((0.0, 0.0, 0.8))
        direction = target - Vector(light.location)
        light.rotation_mode = "QUATERNION"
        light.rotation_quaternion = direction.to_track_quat("-Z", "Y")
        # An area light is geometry to a camera ray, and these are two-metre panels barely more
        # than a metre from the subject — at any framing that pulls back they drift into shot as
        # a bright rectangle across the backdrop, which is what the first wide tiles showed.
        # They still light the scene; they just stop being in it.
        light.visible_camera = False

    _place_shadow(centre, azimuth)


def _place_shadow(centre, azimuth):
    """The lamp whose only job is the shadow under the figure.

    Three things about it, each of which was found by rendering the obvious version first:

      * **It is not overhead.** Straight above is where a contact shadow belongs and it is
        invisible from these cameras: they sit 5 to 12 degrees above the horizon, so the floor
        directly beneath the figure is foreshortened to nothing and hidden behind the figure's
        own feet. A 400 W lamp straight above lit the entire floor and produced no visible
        shadow whatsoever. Nor is it on the camera's own azimuth, which throws the shadow
        straight back along the view axis where the figure stands on top of it. It is `SHADOW_
        AZIMUTH` round from the camera, so the shadow lies out to one side where it can be
        seen, which is also where the existing key light's own faint shadow already fell.
      * **It lights the set and nothing else**, through Blender's light linking: the floor and
        the backdrop are its only receivers, while everything still BLOCKS it, which is the
        asymmetry this needs. So the lamp can be as strong as the shadow wants without touching
        the shading on the character — and the house style is flat pastel with almost no shadow
        ON the figure, which a lamp bright enough to ground it would have wrecked. The backdrop
        is in there with the floor so that the two lift together: with the floor alone it came
        up a visible step brighter than the wall behind it and the join turned into a hard
        line, which is a different kind of wrong.
      * **It is aimed at the feet, not the chest**, for the same reason: the subject is the
        floor.

    The shadow is the absence of this lamp's contribution, so what a viewer reads is the
    contact patch against a floor lifted a little above its ambient value, not a dark smear.
    """
    floor = bpy.data.objects.get("floor")
    if floor is None:
        return None
    shadow = bpy.data.objects.get(SHADOW_LIGHT)
    if shadow is None:
        # Built through the data API rather than `bpy.ops.object.light_add`, which is not free
        # of side effects: the operator leaves the new light as the active object and drops the
        # armature out of pose mode behind the caller's back. Nothing downstream happened to
        # depend on that, which is the only reason it was not already a bug.
        lamp = bpy.data.lights.new(SHADOW_LIGHT, type="AREA")
        lamp.energy = SHADOW_ENERGY
        lamp.size = SHADOW_SIZE
        shadow = bpy.data.objects.new(SHADOW_LIGHT, lamp)
        bpy.context.scene.collection.objects.link(shadow)
        shadow.visible_camera = False
    receivers = bpy.data.collections.get(SHADOW_RECEIVERS)
    if receivers is None:
        # Deliberately not linked into the scene collection: a light-linking collection is a
        # membership list, not a place objects live, and the floor already lives in the scene.
        receivers = bpy.data.collections.new(SHADOW_RECEIVERS)
    for surface in (floor, bpy.data.objects.get("backdrop")):
        if surface is not None and surface.name not in receivers.objects:
            receivers.objects.link(surface)
    shadow.light_linking.receiver_collection = receivers

    direction = orbit(azimuth + SHADOW_AZIMUTH, SHADOW_ELEVATION)
    foot = Vector((centre.x, centre.y, 0.0))
    shadow.location = foot + direction * SHADOW_REACH
    shadow.rotation_mode = "QUATERNION"
    shadow.rotation_quaternion = (foot - Vector(shadow.location)).to_track_quat("-Z", "Y")
    return shadow


def fit_corners(camera, corners, fov, floor=None):
    """How much of the frame a set of world points uses, and whether they spill out of it.

    The point of this task is a render a human judges by eye, and the specific failure it is
    replacing is a check that passed while the picture was useless. So: project every point into
    the camera's own axes and report the worst excursion as a fraction of the half-frame. Over
    1.0 is clipped, and it is printed loudly rather than quietly fixed, because something that
    does not fit is a staging decision to revisit, not a number to nudge.

    Takes points rather than objects because the thing that has to fit is no longer a single
    posed frame: it is the whole movement, which means corners gathered from more than one pose
    and, in the verification sweep, from every frame of a rep.
    """
    # `matrix_world` is lazy: assigning `camera.location` marks it dirty but does not recompute
    # it, so without this the whole check silently measures the camera's PREVIOUS position. That
    # is precisely the class of bug this check exists to catch, so it does not get to have one.
    bpy.context.view_layer.update()
    matrix = camera.matrix_world.inverted()
    scene = bpy.context.scene
    aspect = scene.render.resolution_x / max(scene.render.resolution_y, 1)
    half_v = math.tan(fov / 2)
    worst_x = worst_y = 0.0
    behind = False
    if True:
        for corner in corners:
            if floor is not None and corner.z < floor:
                # Below the floor plane, so the floor hides it. A prop deliberately sunk to
                # bring its bar to the hands would otherwise demand a frame big enough to show
                # geometry nobody can see.
                corner = Vector((corner.x, corner.y, floor))
            local = matrix @ corner
            depth = -local.z
            if depth <= 1e-6:
                # Behind the lens, or in it. Not "off the edge of the measurement" — a camera
                # standing inside the barbell is as clipped as it gets, and an early version
                # that skipped these corners reported a perfect fit for exactly that.
                behind = True
                continue
            worst_x = max(worst_x, abs(local.x) / (depth * half_v * aspect))
            worst_y = max(worst_y, abs(local.y) / (depth * half_v))
    return {"x": worst_x, "y": worst_y, "behind": behind,
            "clipped": behind or max(worst_x, worst_y) > 1.0}


def fit_check(camera, objects, fov, floor=None):
    """`fit_corners` over whole objects, as they are posed right now."""
    corners = []
    for obj in objects:
        corners.extend(world_bbox(obj))
    return fit_corners(camera, corners, fov, floor)


def suggest_framing(camera, objects, centre, azimuth, elevation, base, fov, floor):
    """The smallest multiple of the shared camera distance that holds the whole staged scene.

    Printed beside every tile so the `framing` numbers in the table are measured rather than
    guessed, and so a prop that does not fit shows up as a number instead of as a cropped render
    nobody looked at. It is a diagnostic, not a correction: the table keeps whatever was declared
    for it, because an automatic pull-back would quietly hide a staging decision worth revisiting
    (a near plate that fills the lens is a reason to move the camera, not to zoom out).
    """
    home_location = camera.location.copy()
    home_rotation = camera.rotation_quaternion.copy()
    answer = None
    step = 0.05
    for index in range(120):
        multiplier = 0.6 + index * step
        place_camera(camera, centre, azimuth, elevation, base * multiplier)
        if not fit_check(camera, objects, fov, floor)["clipped"]:
            answer = multiplier
            break
    camera.location = home_location
    camera.rotation_quaternion = home_rotation
    return answer


# --- the entry point ----------------------------------------------------------------------------

# Each exercise's phases, as `pose_model` reads them, cached because the union below asks for
# them once per exercise and re-reading a JSON file fourteen times is silly.
_PHASES = {}


def phases_of(name):
    """This exercise's phase dictionaries, by name, from `reference/exercises/poses.json`."""
    import pose_model                                # lazy: Blender-side callers only
    if name not in _PHASES:
        _PHASES[name] = pose_model.phases_of(pose_model.load(), name)
    return _PHASES[name]


# One pelvis height per exercise, for the orientations whose contact is a seat and not the
# floor. Cached because it costs a pose to measure and it is a property of the exercise.
_PELVIS_ANCHOR = {}


def pelvis_anchor(name, rig, body):
    """The height to pin this exercise's pelvis at, or None to let each frame find the floor.

    Only for the orientations `pose_model.PINNED_BY` anchors at the pelvis — `seated` and
    `lying_supine`, the two whose contact with the world is a seat or a bench that the solver
    cannot see. For everything else the floor is a real contact and deriving the height per
    frame is right, which is why this returns None rather than a number for them.

    The height itself comes from the exercise's own CONTACT phase, grounded the ordinary way:
    for the three seated exercises in this table that means the feet resting on the floor, which
    is a real contact and does set a seated person's hip height. It is then held for the whole
    rep. What it must not do is be re-derived per frame — see `animate.ground_rig`.

    **The honest limit of this**, recorded beside the pose layer's gap #16: it works because
    this character's feet reach the floor while it sits. If a seated figure's feet ever leave
    the floor — a shorter character, a taller machine — then nothing in the data supplies its
    height at all, because no source gives a seat height and `ground` knows only the floor. At
    that point the number below is measuring a contact that does not exist, and the answer is a
    seat height in the data rather than a better guess here.
    """
    import pose_model
    entry = staging_for(name)
    phases = phases_of(name)
    contact = entry.get("contact", entry["showcase"])
    orientation = phases[contact].get("orientation", "standing")
    if "pelvis" not in pose_model.PINNED_BY.get(orientation, ()):
        return None
    if name not in _PELVIS_ANCHOR:
        import animate
        animate.aim_bones(rig, animate.with_clavicles(pose_model.solve(phases[contact])))
        animate.ground_rig(rig, body, orientation, rest_at=support_height(name, body))
        _PELVIS_ANCHOR[name] = joints_of(rig)["pelvis"].z
    return _PELVIS_ANCHOR[name]


def pose_phase(rig, body, name, which):
    """Put the rig into one solved phase of this exercise, grounded, without keyframing it.

    Everything about posing is `animate`'s — the same shortest-arc swing and the same root
    grounding the rendered animation uses, called rather than copied — so a pose measured here
    is the pose that will be rendered. The skeleton is re-solved rather than taken from the
    caller's `skeletons` because `animate.with_clavicles` MUTATES what it is given: applied
    twice to one Skeleton it moves the shoulders twice, and the union below has to pose each
    phase after the caller has already posed one of them.
    """
    import animate                                   # lazy: animate imports this module
    import pose_model
    phases = phases_of(name)
    anchor = pelvis_anchor(name, rig, body)          # measured before this pose, if it applies
    skeleton = animate.with_clavicles(pose_model.solve(phases[which]))
    animate.aim_bones(rig, skeleton)
    animate.ground_rig(rig, body, phases[which].get("orientation", "standing"),
                       rest_at=support_height(name, body), pelvis_at=anchor)
    return skeleton


def movement_union(rig, body, props, name, phases, showcase):
    """Every corner of the figure — and of whatever it is holding — at every phase of the rep.

    This is what the camera has to contain, and getting it wrong is the last thing that was
    wrong with this module. The camera used to be framed on the figure as posed in ONE phase,
    which was harmless while the armature's root never moved and became a real fault the moment
    it did: the body now travels between the two keyframes, so a camera centred on the start
    loses the finish. A pull-up rises 0.425 of a body height between its keys and left the top
    of the frame entirely by frame 17 of its own rendered rep.

    So the box is gathered across both solved phases, by posing the rig into each and reading
    the evaluated mesh. The rig is left in `showcase` afterwards, which is the pose every prop
    was placed from and the one the still frame renders.

    Returns two sets of corners: the FIGURE's, and the figure's plus whatever it is holding.

    The camera is centred on the first of those, and that is a deliberate choice against the
    obvious reading of "the movement". A barbell at the top of an overhead press is 2.2 m wide
    with its plates a third of a body above the lifter's head, and Finding 1 of this task
    already settled that equipment may run out of frame so that the character does not have to
    shrink. Centring on the figure-plus-barbell honours the letter of "hold the movement" by
    pushing the lifter down into the bottom third of the frame to make room for two plates — it
    trades the one thing a reviewer actually noticed (the character changing size and position
    between tiles) for geometry that was already allowed to leave. So: the figure sets the
    frame, the held props are measured and reported beside it, and nothing is hidden either way.
    """
    held = [prop for prop in props if prop.parent is rig]
    figure, carried = [], []
    for which in phases:
        pose_phase(rig, body, name, which)
        figure.extend(world_bbox(body))
        for obj in held:
            carried.extend(world_bbox(obj))
    pose_phase(rig, body, name, showcase)
    return figure, figure + carried


def staging_for(name):
    """This exercise's entry, or a clear error naming every exercise that has one.

    An unknown name is an error, not a fallback. A silent fallback to a bare front view is the
    exact failure this table exists to replace, so it would be the worst possible default — the
    caller would get a render, it would look like all thirteen others, and nothing would say so.
    """
    try:
        return STAGING[name]
    except KeyError:
        raise ValueError(
            f"no staging for exercise {name!r}. The table covers: " +
            "; ".join(sorted(STAGING)) +
            ". Add an entry rather than letting it fall back to a bare front view.") from None


def stage_exercise(name, rig, body, skeletons, phase=None):
    """Build, attach and aim everything this exercise needs. Returns what was created.

    `skeletons` is {"start": Skeleton, "finish": Skeleton} from `pose_model.solve`, used for the
    phase names and their orientations — not for world coordinates, which come from the posed rig
    (module docstring). `phase` overrides which phase the placement is read from, for a caller
    that wants to stage a frame other than the table's own `showcase`; the rig must already be
    posed in that phase, because that is what every placement is measured from.

    An unknown name raises (see `staging_for`) rather than falling back to a bare front view.
    """
    entry = staging_for(name)
    showcase = phase or entry["showcase"]
    if skeletons and showcase not in skeletons:
        raise ValueError(f"{name!r} staged for phase {showcase!r}, which is not in "
                         f"{sorted(skeletons)}")
    framing = entry.get("framing", 1.0)
    if abs(framing - 1.0) > FRAMING_TOLERANCE + 1e-9 and not entry.get("framing_exception"):
        raise ValueError(
            f"{name!r} asks for framing {framing}, which is further than "
            f"{FRAMING_TOLERANCE:.0%} from the shared camera distance. The character has to be "
            f"the same size in every tile — that is what this number exists to protect — so "
            f"either find an azimuth that holds the movement at the shared distance, or let the "
            f"equipment run out of frame. If the conflict is real and measured, an entry may "
            f"carry `framing_exception` with the reason and the arithmetic; exactly one in this "
            f"table does, and what makes it defensible is that it is the only one.")

    joints = joints_of(rig)
    axes = body_axes(joints)
    height = rest_height(body)             # unposed: see `rest_height` for why it has to be
    unit = height / 1.75                   # equipment's own conversion, from its first argument

    props, sunk, measured = [], [], {}
    for spec in entry["equipment"]:
        builder = _BUILDERS.get(spec["prop"])
        if builder is None:
            raise ValueError(f"{name!r} asks for prop {spec['prop']!r}, which equipment.py "
                             f"does not build. Known: {sorted(_BUILDERS)}")
        if spec["mount"] == "each_hand":
            for side in ("left", "right"):
                obj = builder(height, spec)
                obj.name = f"{spec['prop']}.{side}"
                _mount_in_hand(obj, rig, joints, axes, side, unit)
                props.append(obj)
            continue
        # A cable machine's drop has to be known before the prop is built, because it is part of
        # the mesh. Solved from the hands, so the attachment lands on them with the machine's
        # own feet still on the floor.
        if spec.get("seat_top") is not None:
            spec = dict(spec, base_z=_seat_base(spec, joints, axes, body, unit))
            measured["seat_base"] = spec["base_z"]
        if spec.get("drop_to_grip"):
            spec = dict(spec, drop=_drop_to_grip(spec, joints, unit))
            measured["drop"] = spec["drop"]
        obj = builder(height, spec)
        mount = _MOUNTS.get(spec["mount"])
        if mount is None:
            raise ValueError(f"{name!r} asks for mount {spec['mount']!r}; known: "
                             f"{sorted(_MOUNTS)}")
        result = mount(obj, rig, joints, axes, spec, unit, body=body)
        props.append(obj)
        measured.update({k: v for k, v in result.items() if k != "sink"})
        if result.get("sink"):
            sunk.append(obj)
        if spec.get("drop_to_grip"):
            measured["grip_gap"] = _grip_gap(obj, spec, joints, unit)

    bpy.context.view_layer.update()

    # Where the figure is actually resting, and how far its own lowest surface is from it. Not a
    # ground plane to move the set to — the set is at zero — but the check that the figure
    # arrived there: `animate.ground_rig` is supposed to have put it on this plane, and a tile
    # with the feet 20% of a body above the floor is the failure this replaced.
    rest_at = support_height(name, body)
    under = min(c.z for c in world_bbox(body)) - rest_at
    if abs(under) > 0.01 and not entry.get("airborne"):
        print(f"  WARNING {name}: the figure's lowest surface is {under:+.4f} from the plane it "
              f"rests on ({rest_at:+.4f}) — was the rig grounded before staging?")

    # What the camera has to hold: the figure across the WHOLE movement, not as posed in the
    # one phase this frame happens to show. See `movement_union`.
    moving = [w for w in ("start", "finish") if w in (skeletons or {})] or [showcase]
    union, union_carried = movement_union(rig, body, props, name, moving, showcase)
    span = Vector((
        max(c.x for c in union) - min(c.x for c in union),
        max(c.y for c in union) - min(c.y for c in union),
        max(c.z for c in union) - min(c.z for c in union),
    ))

    distance, extent, fov = camera_distance(body, framing)
    centre = hold_centre(union, world_bbox(body), entry["azimuth"], entry["elevation"],
                         distance, fov)
    camera = bpy.data.objects.get("camera")
    if camera is None:
        bpy.ops.object.camera_add(location=(0, 0, 0))
        camera = bpy.context.object
        camera.name = "camera"
    place_set(centre, entry["azimuth"], props)
    base = distance / max(framing, 1e-6)
    place_camera(camera, centre, entry["azimuth"], entry["elevation"], distance)

    # Two separate questions, and conflating them is how the first version ended up letting a
    # barbell decide how big the lifter was.
    #
    #   `body_fit`  does the CHARACTER fit? This one matters. A cropped head or a foot off the
    #               bottom edge is a staging fault and is printed loudly.
    #   `fit`       does the whole scene fit? Informational. Equipment is expected to spill out
    #               of frame at a shared distance, and `suggest` says what it would have cost to
    #               hold it — which is the number that used to be applied, for comparison.
    # `body_fit` is now the movement, not the frame: it asks whether the figure fits at BOTH
    # ends of the rep. That is the question a still cannot ask and the one that matters, since
    # every one of these fourteen is staged once and then animated.
    body_fit = fit_corners(camera, union, fov, 0.0)
    visible = [body] + [p for p in props if p not in sunk]
    scene_corners = list(union_carried)
    for prop in visible:
        if prop is not body and prop.parent is not rig:
            scene_corners.extend(world_bbox(prop))
    fit = fit_corners(camera, scene_corners, fov, 0.0)
    fit["suggest"] = suggest_framing(camera, visible, centre, entry["azimuth"],
                                     entry["elevation"], base, fov, 0.0)
    place_camera(camera, centre, entry["azimuth"], entry["elevation"], distance)
    if body_fit["clipped"]:
        # Reported, never corrected. Pulling the camera back would hold the movement at the
        # cost of the character changing size between tiles, and that trade has already been
        # made the other way round on purpose — it is the whole of Finding 1.
        print(f"  CLIPPED {name}: the FIGURE's own movement does not fit at the shared "
              f"distance — frame use x={body_fit['x']:.2f} y={body_fit['y']:.2f}; holding it "
              f"would need framing {max(body_fit['x'], body_fit['y']):.2f}")

    return {
        "name": name,
        "props": props,
        "sunk": sunk,
        "camera": camera,
        "centre": centre,
        "rest_at": rest_at,
        "resting_gap": under,
        "union": (centre, span),
        "azimuth": entry["azimuth"],
        "elevation": entry["elevation"],
        "framing": framing,
        "distance": distance,
        "reason": entry["reason"],
        "contact": entry["contact"],
        "showcase": showcase,
        "notes": entry["notes"],
        "unit": unit,
        "fit": fit,
        "body_fit": body_fit,
        "measured": measured,
    }


def _grip_gap(obj, spec, joints, unit):
    """How far a cable machine's attachment ended up from the hands it was solved for.

    The point of `drop_to_grip` is that the bar meets the grip, and the point of this is that
    nobody has to take that on trust. Measured on the built mesh after placement, so it catches
    a wrong pulley constant, a prop whose geometry moved, and the clearance push the cable tower
    needs to stay out of the lifter — which shows up here as a real gap, because it is one.
    """
    bpy.context.view_layer.update()
    centre = (grip(joints, "left") + grip(joints, "right")) / 2
    local = obj.matrix_world.inverted() @ centre
    hit, landed, _, _ = obj.closest_point_on_mesh(local)
    return (obj.matrix_world @ landed - centre).length if hit else None


def clear_props(staged):
    """Remove the props a previous `stage_exercise` made, so one session can stage many."""
    for obj in staged.get("props", []):
        if obj.name in bpy.data.objects:
            bpy.data.objects.remove(obj, do_unlink=True)


# --- verification -------------------------------------------------------------------------------

def _pose_without_keys(rig, skeleton, body, orientation, rest_at=0.0):
    """Pose and ground the rig from a solved skeleton without touching the animation.

    `animate.pose` keyframes every bone, which is right for a rep and wrong here: the contact
    sheet is fourteen unrelated single frames in one session, and an action left behind on the
    rig would drive the next one back to whatever was keyed. Everything else is `animate`'s own
    — the same shortest-arc swing and the same root grounding the rendered animation uses, both
    called rather than copied — so a tile is posed exactly as frame 1 of the real run would be,
    which is the point of verifying with it.
    """
    import animate                                  # lazy: animate imports this module
    animate.aim_bones(rig, skeleton)
    return animate.ground_rig(rig, body, orientation, rest_at=rest_at)


def sweep_rep(rig, body, props, camera, fov, name, showcase):
    """Run the containment check on EVERY frame of the real rep, not just on its two keys.

    The two keyframes are not the whole movement. Between them the bones interpolate on a Bezier
    ease, and an interpolated pose is not obliged to stay inside the box its endpoints make — an
    arm swinging through an arc can reach further than it does at either end of the swing. A
    check that only looks at the keys is therefore a check that can pass while the render is
    cropped, which is the exact class of failure this module keeps finding in itself.

    So this builds the actual animation `animate.py` would build — the same timeline, the same
    ease, through `animate`'s own functions — scrubs it frame by frame, and reports the worst
    excursion and the frame it happened on. It then clears the action and restores the showcase
    pose, because the contact sheet is fourteen unrelated single frames in one session and an
    action left on the rig would drive the next one.
    """
    import animate
    import pose_model
    phases = phases_of(name)
    if not {"start", "finish"} <= set(phases):
        return None
    timeline = [(1, "start"), (1 + animate.HOLD, "start"),
                (1 + animate.HOLD + animate.HALF_REP, "finish"),
                (1 + 2 * animate.HOLD + animate.HALF_REP, "finish")]
    rest_at = support_height(name, body)
    anchor = pelvis_anchor(name, rig, body)
    scene = bpy.context.scene
    for frame, which in timeline:
        animate.pose(rig, animate.with_clavicles(pose_model.solve(phases[which])), frame,
                     body=body, orientation=phases[which].get("orientation", "standing"),
                     rest_at=rest_at, pelvis_at=anchor)
    for curve in animate.fcurves_of(rig.animation_data.action):
        for point in curve.keyframe_points:
            point.interpolation = "BEZIER"
            point.easing = "EASE_IN_OUT"

    held = [prop for prop in props if prop.parent is rig]
    worst = {"x": 0.0, "y": 0.0, "frame": None, "clipped": False, "carried": 0.0}
    for frame in range(1, timeline[-1][0] + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        corners = world_bbox(body)
        carried = list(corners)
        for obj in held:
            carried.extend(world_bbox(obj))
        fit = fit_corners(camera, corners, fov, 0.0)
        with_props = fit_corners(camera, carried, fov, 0.0)
        if max(fit["x"], fit["y"]) > max(worst["x"], worst["y"]):
            worst = {"x": fit["x"], "y": fit["y"], "frame": frame,
                     "clipped": fit["clipped"], "carried": worst.get("carried", 0.0)}
        worst["carried"] = max(worst.get("carried", 0.0),
                               with_props["x"], with_props["y"])
    worst["frames"] = timeline[-1][0]

    rig.animation_data_clear()
    scene.frame_set(1)
    pose_phase(rig, body, name, showcase)
    return worst


def silhouette_share(camera, body, fov):
    """How much of the frame the FIGURE fills, vertically and horizontally, as a fraction.

    The number the contact sheet is judged on, and the number that was wrong: a reviewer
    measured the figure's pixel height in each of the first fourteen tiles and found it ran from
    80% of the tile down to 27%, a 2.93x range, because each camera had been backed off to hold
    its own equipment. Measured here from the posed mesh's own vertices rather than from its
    bounding box, because a bounding box corner is not a point on the silhouette and a bent-over
    figure's box is mostly air.
    """
    bpy.context.view_layer.update()
    evaluated = body.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        matrix = camera.matrix_world.inverted() @ evaluated.matrix_world
        scene = bpy.context.scene
        aspect = scene.render.resolution_x / max(scene.render.resolution_y, 1)
        half_v = math.tan(fov / 2)
        lo_x = lo_y = 1e9
        hi_x = hi_y = -1e9
        for vertex in mesh.vertices:
            local = matrix @ vertex.co
            depth = -local.z
            if depth <= 1e-6:
                continue
            x = local.x / (depth * half_v * aspect)
            y = local.y / (depth * half_v)
            lo_x, hi_x = min(lo_x, x), max(hi_x, x)
            lo_y, hi_y = min(lo_y, y), max(hi_y, y)
    finally:
        evaluated.to_mesh_clear()
    return {"height": (hi_y - lo_y) / 2, "width": (hi_x - lo_x) / 2,
            "top": hi_y / 2 + 0.5, "bottom": lo_y / 2 + 0.5}


def head_share(camera, rig, fov):
    """The projected length of the head bone as a fraction of frame height.

    A second opinion on "is it the same character", independent of the pose. The silhouette
    share cannot be equal across fourteen tiles — a plank seen from the side is a horizontal
    body and its silhouette is genuinely short — but a rigid bone's apparent length can only
    change through foreshortening and perspective, so if this is constant the camera really is
    the same distance away and the character really is the same size.
    """
    bpy.context.view_layer.update()
    if "head" not in rig.pose.bones:
        return None
    bone = rig.pose.bones["head"]
    matrix = camera.matrix_world.inverted() @ rig.matrix_world
    half_v = math.tan(fov / 2)
    points = []
    for end in (bone.head, bone.tail):
        local = matrix @ end
        depth = -local.z
        if depth <= 1e-6:
            return None
        points.append(Vector((local.x / (depth * half_v), local.y / (depth * half_v))))
    return (points[1] - points[0]).length / 2


def main():
    import pose_model                                # noqa: E402  (inside main: Blender only)
    import animate                                   # noqa: E402

    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="render one staged frame per exercise")
    parser.add_argument("--model", default=None, help="a character mesh to rig, as animate.py")
    parser.add_argument("--tiles", required=True, help="directory for one PNG per exercise")
    parser.add_argument("--resolution", type=int, default=520)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--only", default=None, help="substring: stage just the matching ones")
    parser.add_argument("--azimuth", type=float, default=None,
                        help="override every staged exercise's camera azimuth, for choosing one "
                             "by looking at a sweep instead of by arguing about it")
    parser.add_argument("--phase", default=None, choices=("start", "finish"),
                        help="stage every exercise from this phase instead of the table's own "
                             "`showcase`, to compare the two ends of a movement")
    parser.add_argument("--measure-only", action="store_true",
                        help="stage and measure every exercise but render nothing, for checking "
                             "the framing numbers without waiting for Cycles")
    parser.add_argument("--palm", type=float, default=None,
                        help="override PALM, to compare equipment.hold's own wrist anchor (0.0) "
                             "against the palm (0.5) in a render")
    args = parser.parse_args(argv)

    global PALM
    if args.palm is not None:
        PALM = args.palm
        print(f"PALM = {PALM} (grip position along the hand bone)")

    out = Path(args.tiles)
    out.mkdir(parents=True, exist_ok=True)

    data = pose_model.load()
    if args.model:
        import fit_model
        body = fit_model.load(args.model)
        fit_model.decimate(body)
        marks = fit_model.measure(body, out / "silhouette.png")
        joints = fit_model.joints_from(marks)
        rig = fit_model.build_rig(joints)
        fit_model.weight(body, rig, joints)
        skeleton = None
    else:
        make_character.clear()
        skeleton = make_character.rest_skeleton()
        body = make_character.build_body(skeleton)
        rig = make_character.build_rig(skeleton)
        make_character.bind(body, rig)
    make_character.stage(skeleton, body)

    scene = bpy.context.scene
    make_character.use_cycles(scene, samples=args.samples)
    scene.render.resolution_x = scene.render.resolution_y = args.resolution
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False

    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="POSE")

    rows = []
    for entry in data["exercises"]:
        name = entry["name"]
        if args.only and args.only.lower() not in name.lower():
            continue
        phases = pose_model.phases_of(data, name)
        skeletons = {which: pose_model.solve(phases[which])
                     for which in ("start", "finish") if which in phases}
        if args.azimuth is not None:
            # Diagnostic only: the table is the record, and a sweep that quietly rewrote it
            # would be the silent substitution the brief forbids. It prints what it did.
            print(f"  --azimuth {args.azimuth} overriding the table's "
                  f"{STAGING[name]['azimuth']} for this run only")
            STAGING[name]["azimuth"] = args.azimuth
        which = args.phase or STAGING[name]["showcase"]
        _pose_without_keys(rig, animate.with_clavicles(skeletons[which]), body,
                                   phases[which].get("orientation", "standing"),
                                   rest_at=support_height(name, body))
        staged = stage_exercise(name, rig, body, skeletons, phase=args.phase)
        _, _, fov = camera_distance(body)
        shape = silhouette_share(staged["camera"], body, fov)
        head = head_share(staged["camera"], rig, fov)
        rep = sweep_rep(rig, body, staged["props"], staged["camera"], fov, name, which)

        path = out / (name.split("(")[0].strip().lower().replace(" ", "-").replace("/", "-")
                      + ".png")
        if not args.measure_only:
            scene.render.filepath = str(path)
            bpy.ops.render.render(write_still=True)
        staged["rep"] = rep
        rows.append((name, staged, shape, head, path))
        extras = {key: (round(value, 4) if isinstance(value, float) else value)
                  for key, value in staged["measured"].items()}
        print(f"{name}\n    phase={staged['showcase']} "
              f"orientation={phases[which].get('orientation')} "
              f"az={staged['azimuth']} el={staged['elevation']} "
              f"framing={staged['framing']} distance={staged['distance']:.4f} "
              f"root-z {rig.location.z:+.4f} "
              f"rest_at={staged['rest_at']:+.4f} gap={staged['resting_gap']:+.4f}\n"
              f"    FIGURE height {shape['height']:.1%} width {shape['width']:.1%} "
              f"(top {shape['top']:.2f} bottom {shape['bottom']:.2f}) "
              f"head-bone {head:.2%} | body-fit x={staged['body_fit']['x']:.2f} "
              f"y={staged['body_fit']['y']:.2f} | scene-fit "
              f"x={staged['fit']['x']:.2f} y={staged['fit']['y']:.2f} "
              f"(would need {staged['fit']['suggest']})\n"
              f"    REP over {rep['frames']} frames: figure worst x={rep['x']:.2f} "
              f"y={rep['y']:.2f} at frame {rep['frame']}; with what it holds "
              f"{rep['carried']:.2f}"
              f"{'  <-- FIGURE LEAVES THE FRAME' if rep['clipped'] else ''}\n"
              f"    measured={extras} "
              f"props={[p.name for p in staged['props']]} -> {path.name}")
        clear_props(staged)

    print(f"\n{len(rows)} tiles in {out}")
    for name, staged, shape, head, path in rows:
        print(f"  {path.name}|{name}")
    heights = [shape["height"] for _, _, shape, _, _ in rows]
    heads = [head for _, _, _, head, _ in rows if head]
    distances = sorted({round(staged["distance"], 6) for _, staged, _, _, _ in rows})
    print(f"\ncamera distance: {distances} — one number for all fourteen is the whole point of "
          f"BASE_FRAMING; more than one entry here means a `framing` multiplier is in use")
    print(f"\nfigure height as a share of the frame: {min(heights):.1%} .. {max(heights):.1%} "
          f"(ratio {max(heights) / max(min(heights), 1e-9):.2f}x)")
    print(f"head bone, same measure: {min(heads):.2%} .. {max(heads):.2%} "
          f"(ratio {max(heads) / max(min(heads), 1e-9):.2f}x) — "
          f"the pose-independent check that it is one character")
    for name, staged, shape, head, path in sorted(rows, key=lambda r: -r[2]["height"]):
        print(f"  {shape['height']:6.1%}  head {head:5.2%}  {name}")
    print("\nthe whole rep, every frame, against the shared distance "
          "(over 1.00 means part of the figure leaves the frame somewhere in the movement):")
    for name, staged, shape, head, path in sorted(
            rows, key=lambda r: -max(r[1]["rep"]["x"], r[1]["rep"]["y"]) if r[1]["rep"] else 0):
        rep = staged["rep"]
        if rep is None:
            continue
        flag = "CLIPPED" if rep["clipped"] else "holds  "
        print(f"  {flag}  figure x={rep['x']:.2f} y={rep['y']:.2f} at frame "
              f"{rep['frame']:2d} (would need framing {max(rep['x'], rep['y']):.2f}); "
              f"holding its equipment too {rep['carried']:.2f}  {name}")


if __name__ == "__main__":
    main()
