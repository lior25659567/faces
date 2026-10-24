#!/usr/bin/env python
"""Turn the form notes' joint angles into a posed skeleton, with no Blender in sight.

This is the middle of three layers, and the only one that can be checked without 3D assets:

    reference/exercises/poses.json     what the source says, in anatomical terms
    pipeline/pose_model.py             this: anatomy -> a skeleton with real joint positions
    pipeline/blender/build_rig.py      that skeleton -> bone rotations on an actual rig

Keeping the middle layer free of Blender is deliberate. Blender cannot be scripted into a
screenshot in this environment, the rig does not exist yet, and the thing most likely to be
wrong is not the rendering — it is whether "knee flexed to roughly 60-70 degrees of included
angle" produces a deep squat or a standing figure. `pose_preview.py` draws what this returns as
a stick figure, so that question is answered by looking, before anybody models a character.

## Conventions, which are the whole file

Axes are Blender's: +X right, +Y forward (the direction the character faces), +Z up. Angles are
degrees.

A positive rotation about +X swings a downward-pointing bone FORWARD, toward +Y. Worth deriving
rather than trusting: with axis X and v = (0,0,-1), the cross product X x v is (0,+1,0), so
Rodrigues sends the bone toward +Y as the angle grows. The first draft of this file asserted the
opposite in a comment and then followed its own wrong comment through four joints — hips, knees,
elbows and shoulders all inverted — which the stick figures showed as a bench press with its legs
in the air. That is exactly what `pose_preview.py` is for.

So: hip flexion and shoulder flexion take POSITIVE rotations, because both carry the limb
forward. Knee flexion takes a NEGATIVE one, because the shin swings backward and a knee bends
only that way. Elbow flexion is positive, carrying the hand forward and up as in a curl.

The rest pose is a standing figure, arms hanging, palms facing in, legs straight, feet forward.
Every angle in `poses.json` is a departure from that.

The two measures that are opposites, repeated here because confusing them is the failure mode
this whole pipeline is built to avoid:

    included angle   the angle BETWEEN the two bones. 180 is straight. 60 is deeply bent.
    flexion          how far from straight. 0 is straight. 60 is a third bent.

`flexion_from` converts either into the one number the solver uses, which is always flexion.

## Nothing is dropped

Every angle in a phase is either applied to the figure or refused out loud. `solve` declares the
`(joint, measure)` pairs it knows how to apply in `APPLIES`, and raises `ValueError` naming every
entry of the phase it did not use — including every entry still marked `ambiguous`.

That rule is here because the opposite held for a long time, quietly. `_lookup` selects entries by
measure, so an angle whose pair no lookup asked for was not refused, it was invisible. Forty of
this file's eighty-five angles went that way, twenty-one of them non-zero: every foot splay in the
file, both of the plank's ninety-degree elbows, the entire range of movement of the tricep
pushdown. `animate.py` runs a pre-flight loop over `solve` for exactly this purpose and passed all
of it. An angle worth sourcing is worth applying or refusing; being ignored is the one outcome the
rest of this docstring rules out.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POSES = ROOT / "reference" / "exercises" / "poses.json"


# --- proportions -------------------------------------------------------------------------------
#
# Fractions of standing height, from the same anthropometric habit the body measurement side of
# this project uses: a real distribution rather than a drawing. These are ANSUR-ish adult means
# rounded to three places, and they are only proportions — the rig they drive will have its own,
# and the solver is written so that swapping them changes the figure and nothing else.

@dataclass(frozen=True)
class Proportions:
    pelvis: float = 0.530       # height of the hip joints
    spine: float = 0.180        # pelvis to the base of the neck
    neck: float = 0.055
    head: float = 0.130
    shoulder_half: float = 0.105  # centre line to shoulder joint
    hip_half: float = 0.055
    upper_arm: float = 0.186
    forearm: float = 0.146
    hand: float = 0.108
    thigh: float = 0.245
    shin: float = 0.246
    foot: float = 0.150


PROPORTIONS = Proportions()


# --- the angles --------------------------------------------------------------------------------

def flexion_from(measure: str, degrees: float) -> float:
    """One number the solver understands, whichever way the source expressed it.

    Raises on `ambiguous` rather than guessing. The extraction records that value when the
    source gives a figure without saying which convention it used, and the honest response is to
    stop: a 60 that might mean "nearly straight" or "nearly folded" cannot be drawn either way.

    This raise is not what protected `solve`, and believing that it was is how twelve ambiguous
    angles reached the renders: `_lookup` filters on `measure`, so an `ambiguous` entry never
    arrived here at all. `_unapplied` refuses them now. This stays for direct callers.
    """
    if measure == "flexion":
        return degrees
    if measure == "included":
        return 180.0 - degrees
    if measure == "ambiguous":
        raise ValueError("the source did not say whether this is an included angle or flexion; "
                         "it cannot be posed without deciding, and deciding is not this code's job")
    raise ValueError(f"{measure!r} is not a bend; it is handled elsewhere in `solve`")


# How the body as a whole sits, before any joint moves. Expressed as a rotation about +X applied
# to everything, because every one of these is a pitch: lying on your back is standing rotated
# backward through a right angle.
#
# `front_support` is the plank, and it is NOT the same as lying prone: the body is held clear of
# the floor on the forearms and toes, which the joint angles then describe. The orientation only
# sets which way "up the body" points.
ORIENTATIONS = {
    "standing": 0.0,
    "seated": 0.0,          # the hips and knees carry it; the torso is still upright
    "bent_over": 0.0,       # the spine angle carries it, so the root stays upright
    "hanging": 0.0,
    "lying_supine": 90.0,   # on the back, face up
    "lying_prone": -90.0,   # face down
    "front_support": -90.0,  # face down, and held off the floor — see `RESTS_ON`
}
# The three lying pitches were the other way round until a measurement caught them: `oriented` a
# forward vector by -90 and the figure faces the FLOOR, so `lying_supine: -90.0` was drawing a
# lifter face down on the bench. Nothing in the data could show it, because every lying exercise
# is symmetrical top to bottom until a limb bends — and then the bench press's elbows carried the
# hands down THROUGH the bench, which is what it did. The names, the comment above and the
# geometry now agree.


# What an orientation word says about the FIGURE, beyond which way it is pitched.
#
# `seated` is the one that carries geometry, because sitting IS a hip folded to about a right angle
# with the thigh forward and the shin down. That is the definition of the word, not an inference
# about any particular exercise, so recording it here asserts nothing a source would have to state
# — the same standing as the floor invariant in `ground`.
#
# It is here because three exercises in this file say `seated`, stated a knee angle, stated no hip
# angle at all, and were drawn KNEELING: with the thigh left hanging, bending the knee swings the
# shin backward instead of dropping it from a raised knee. The lat pulldown hung off its bar on its
# knees with the thigh pad through its hips; the leg press knelt on its own sled. Nothing in the
# data was missing by the data's own standards — every angle was applied and every check passed —
# and the figure was still wrong, because the joint that aims the limb was described by an
# ORIENTATION WORD rather than by a sentence, and nothing was reading the word.
#
# `lying_supine` deliberately carries none, and the distinction is the point. A bench press plants
# its feet on the floor with the knees bent, but that is the lift's setup rather than what the word
# means: a floor press, a crunch and a supine stretch are all `lying_supine` with their legs flat
# out. The bench press's own prose states the foot plant ("both feet flat on the floor… positioned
# under or slightly behind the knees") and gives no angle for it, so it stays a data gap in that
# entry instead of being quietly supplied from here.
#
# Everything here is a DEFAULT for a joint the data leaves silent. Anything an exercise states for
# itself wins — the lat pulldown's and leg press's own knee angles still do.
POSTURE = {
    "seated": {
        # The thigh as a segment tilt, read exactly like `hip/from_horizontal`: 0 is horizontal,
        # pointing forward out of the hip.
        "hip_from_horizontal": 0.0,
        # And the shin dropped from it, read like `knee/flexion`: 90 takes a horizontal thigh to a
        # vertical shin, which is a chair.
        "knee_flexion": 90.0,
    },
}


@dataclass
class Joint:
    """One point of the skeleton, with where it ended up and what rotated to put it there."""
    name: str
    position: tuple[float, float, float]
    parent: str | None = None
    # The bone's direction after posing, kept so the Blender layer can build a rotation from it
    # rather than re-deriving one from positions and inheriting the ambiguity of a zero-length
    # bone.
    direction: tuple[float, float, float] | None = None


@dataclass
class Skeleton:
    joints: dict[str, Joint] = field(default_factory=dict)

    def add(self, name, position, parent=None, direction=None):
        self.joints[name] = Joint(name, tuple(position), parent, direction)
        return self.joints[name]

    def at(self, name):
        return self.joints[name].position

    @property
    def bones(self):
        """Every parent-child pair, which is what a stick figure draws and what a rig keyframes."""
        return [(j.parent, j.name) for j in self.joints.values() if j.parent]


# --- maths -------------------------------------------------------------------------------------

def _rotate(vector, axis, degrees):
    """Rodrigues, because importing numpy into a file Blender will also run is a needless risk —
    Blender ships its own Python and numpy is not guaranteed to be importable in it."""
    angle = math.radians(degrees)
    x, y, z = vector
    ax, ay, az = axis
    length = math.sqrt(ax * ax + ay * ay + az * az)
    ax, ay, az = ax / length, ay / length, az / length
    cos, sin = math.cos(angle), math.sin(angle)
    dot = ax * x + ay * y + az * z
    return (
        x * cos + (ay * z - az * y) * sin + ax * dot * (1 - cos),
        y * cos + (az * x - ax * z) * sin + ay * dot * (1 - cos),
        z * cos + (ax * y - ay * x) * sin + az * dot * (1 - cos),
    )


def _step(origin, direction, length):
    return (origin[0] + direction[0] * length,
            origin[1] + direction[1] * length,
            origin[2] + direction[2] * length)


X, Y, Z = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
DOWN = (0.0, 0.0, -1.0)


# --- the solver --------------------------------------------------------------------------------

def _lookup(angles, joint, side, measures):
    """The one angle matching this joint and side, or None.

    `both` matches either side, and a side-specific entry wins over it, so an exercise that
    describes one arm in detail and the other in passing poses correctly.

    Two entries can describe the same joint on purpose. The squat's finish torso is recorded
    twice, once for a low-bar squat and once for high-bar, and the entries' own notes say the two
    must never be averaged; the pushdown's start elbow is recorded twice because the source
    offers a higher alternative start. Those are choices about the exercise, so one entry of each
    group carries `"preferred": true` in the pose file and that is the one posed. This used to
    return whichever came first in the file, which decided the squat's bar position by where
    somebody happened to paste a line.
    """
    specific = [a for a in angles
                if a["joint"] == joint and a["side"] == side and a["measure"] in measures]
    shared = [a for a in angles
              if a["joint"] == joint and a["side"] == "both" and a["measure"] in measures]
    for found in (specific, shared):
        if not found:
            continue
        if len(found) == 1:
            return found[0]
        chosen = [a for a in found if a.get("preferred")]
        if len(chosen) == 1:
            return chosen[0]
        listed = ", ".join(f"{a['measure']} {a['degrees']}" for a in found)
        if not chosen:
            raise ValueError(f"the {joint} ({side}) is described {len(found)} times in this phase "
                             f"({listed}) and none of them is marked \"preferred\" in the pose "
                             f"file; which one the figure takes is a decision about the exercise, "
                             f"not one this code can make from file order")
        raise ValueError(f"the {joint} ({side}) has {len(chosen)} entries marked \"preferred\" "
                         f"({listed}); exactly one entry of a group can be the one to pose")
    return None


def _bend(angles, joint, side, default=0.0):
    """How far this joint is bent, in degrees from straight."""
    found = _lookup(angles, joint, side, {"included", "flexion"})
    if not found:
        return default
    return flexion_from(found["measure"], found["degrees"])


def _from_vertical(found):
    """A segment's tilt away from vertical, whichever end of the right angle the source measured.

    Both spellings are in the file for the same kind of quantity — the squat's torso is "40-50
    degrees from vertical" and the deadlift's is "roughly 20-45 degrees above horizontal" — and
    they are the same statement with 90 subtracted. Kept in one function so that the spine, the
    humerus, the femur and the forearm cannot drift into disagreeing about which spelling means
    what.

    `direction` says which WAY it leans, because the degrees are a magnitude and the file holds
    both: the squat, the deadlift and the row all hinge forward over the weight, and the lat
    pulldown is the single entry that leans back ("The torso leans back slightly from vertical").
    A word rather than a minus sign on `degrees`, because a negative number in a field called
    "degrees from vertical" reads as a typo to the next person to edit it. Absent means forward,
    so every entry written before the field existed keeps exactly the meaning it had.
    """
    direction = found.get("direction", "forward")
    if direction not in ("forward", "backward"):
        raise ValueError(f"{direction!r} is not a direction for a segment tilt; it leans "
                         f"'forward' or 'backward', and leaving it out means forward")
    away = -1.0 if direction == "backward" else 1.0
    if found["measure"] == "from_vertical":
        return away * found["degrees"]
    if found["measure"] == "from_horizontal":
        return away * (90.0 - found["degrees"])
    raise ValueError(f"{found['measure']!r} is not a segment tilt; it is handled elsewhere in "
                     f"`solve`")


def ground(skeleton, orientation):
    """Put the figure back on the floor.

    Every joint is built downward from the pelvis, and the pelvis is at a fixed height, so a
    bending knee lifts the foot instead of lowering the hips. Squats floated. The fix is to
    solve freely and then translate the whole skeleton so whatever is touching the ground is at
    zero — the feet standing, the back lying down, the forearms and toes in a plank.

    A hanging pull-up is the exception and is anchored by the hands instead, since nothing of it
    touches the floor and grounding the feet would push the bar down to head height.

    `RESTS_ON` is the other exception: a plank is held clear of the floor on its forearms and toes
    while a prone lie is not, and since the two have the same shape and the same pitch, this is the
    only place that difference can live.

    Two separate questions live in this function and they must not be answered by one table: what
    holds the figure UP (`RESTS_ON`, per orientation) and where it sits in the GROUND PLANE
    (`PINNED_BY`, which for the plank's two orientations names the same contacts so the body pivots
    about them instead of walking).

    Whatever the anchor decides, nothing is left below the floor: see the clamp at the end.
    """
    if orientation == "hanging":
        hands = [skeleton.at(n)[2] for n in ("hand.left", "hand.right") if n in skeleton.joints]
        lift = HANGING_BAR - max(hands) if hands else 0.0
        slide = 0.0
    else:
        resting = [joint for name, joint in skeleton.joints.items()
                   if name.split(".")[0] in RESTS_ON.get(orientation, ())]
        lift = -min(joint.position[2] for joint in (resting or skeleton.joints.values()))
        # And horizontally, over whatever is carrying the weight. The chain is built downward
        # from the pelvis, so flexing a hip walks the feet forward instead of sending the hips
        # back, and a squat drawn without this leans out of its own frame. The rig will not have
        # this problem — Blender anchors a foot with IK — but the preview is what the angles get
        # judged on, and a figure toppling forward is unreadable.
        standing_on = [j.position for name, j in skeleton.joints.items()
                       if name.split(".")[0] in PINNED_BY.get(orientation, ())]
        if not standing_on:
            standing_on = [j.position for name, j in skeleton.joints.items()
                           if j.position[2] + lift < CONTACT and name.startswith(("foot", "hand"))]
        if not standing_on:
            standing_on = [j.position for j in skeleton.joints.values()
                           if j.position[2] + lift < CONTACT]
        slide = -sum(p[1] for p in standing_on) / len(standing_on) if standing_on else 0.0
    if lift or slide:
        for joint in skeleton.joints.values():
            x, y, z = joint.position
            joint.position = (x, y + slide, z + lift)

    # And then the floor, which is not an anchoring question at all. Anchoring says which parts
    # CARRY the figure; this says that no part of it is underground, which is true of every
    # exercise ever performed and so asserts nothing about any of them. It invents no number,
    # which is what makes it usable where the sources are silent.
    #
    # It fires once in this set, on the plank's prone start: with no shoulder abduction recorded
    # for that phase, the 90-degree elbows drive the forearms into the mat, and six joints sat as
    # much as 0.254 below the floor — a seventh of a body height. The clamp lifts the figure until
    # the lowest of them rests on it. That does not answer the missing angle, it stops the figure
    # being drawn through the ground while the angle is missing; the gap is recorded in the pose
    # file's own note on that phase and in the task report.
    sunk = min(joint.position[2] for joint in skeleton.joints.values())
    if sunk < 0.0:
        for joint in skeleton.joints.values():
            x, y, z = joint.position
            joint.position = (x, y, z - sunk)
    return skeleton


# What each orientation RESTS on, for the cases where "whatever is lowest" is the wrong answer.
#
# A plank and a prone lie have the same pitch, and that is correct — both are face down and
# horizontal. What differs is the anchor. `front_support` hangs the body between its forearms and
# its toes with everything in between in the air; `lying_prone` puts the body's own surface down,
# chest and thighs, and the arms come along. Grounding both on their lowest joint made them the
# same picture, and that is how the plank ended up with no movement in it: its two phases differed
# by a word that did nothing, while the module docstring above claimed the opposite.
#
# Anything not named here grounds on its lowest joint, which is right for a figure standing on its
# feet or lying on its back.
RESTS_ON = {
    "front_support": ("elbow", "wrist", "hand", "foot"),
    "lying_prone": ("pelvis", "chest", "neck", "head", "shoulder", "hip", "knee", "ankle"),
}
# The toes are left out of the prone set deliberately. A `foot` joint here is the tip of a 0.150
# segment held square to the shin, so a prone figure with dorsiflexed ankles — toes tucked, which
# is what the plank's source describes — hangs that tip 0.150 below a shin that is itself flat on
# the mat. Anchoring on it stands the body on its toes and lifts the trunk off the floor, which
# collapsed the plank's lift-off from 0.186 to 0.036 the moment the orientation signs were put
# right. A real toe pad is level with the underside of the shin; this skeleton has no way to say
# that, so the trunk and the legs carry the prone figure and the foot is simply not consulted.


# What stays put in the GROUND PLANE, which is a different question from what holds the figure up.
#
# `ground` answers both, and conflating them made a plank walk. The plank's two phases anchor
# vertically on different joints, correctly — a prone body rests on its whole front, a front
# support hangs between its forearms and toes — and that same per-orientation choice was also
# deciding where each phase sat ALONG the mat, so the figure scooted 0.220 between two keyframes of
# a movement whose whole point is that nothing travels. No camera can hold that, because it is the
# body moving and not the shot.
#
# What does not slide when you push up into a plank is what you are pushing against: the forearms
# and the toes. Of those two, only the TOES can be pinned here, and the measurement says why. The
# legs are identical in both phases, so the toes sit in the same place relative to the body in
# both; the forearms do not, because the start's shoulder abduction is a recorded data gap and its
# elbows end up beside the ribs where the finish has them under the shoulders. Pinning a joint
# whose placement is known to be wrong drags the whole body to compensate for it.
#
# Measured, travel between the two keyframes, by landmark:
#
#     pin set                    toe     wrist    elbow   pelvis    head
#     forearms and toes        239mm      92mm     54mm    240mm   239mm
#     toes only                  0mm     332mm    186mm      0mm     0mm
#     forearm contact only     386mm      54mm    200mm    386mm   386mm
#     `RESTS_ON` (per phase)   the 0.220 scoot this whole table exists to kill
#
# Toes only is the one where the BODY does not travel at all. The 332 mm left on the wrist is the
# arm gap moving, not the figure walking, and it is confined to the two limbs the data is honest
# about being unsure of.
#
# Anything not named here keeps the general rule below: sit over whatever feet or hands are near
# the floor. For an upright figure that IS the contact set, and it must stay that way — the walking
# lunge genuinely travels, stepping from both feet onto its lead foot, and a global pin would
# freeze a step its own source describes.
# The same question asked of every other orientation that does not travel, because a drift check
# written against a file with three un-pinned orientations would have had to be loosened until it
# could not see them. A seated figure's contact with the world is the pelvis on the seat; a supine
# one's is the back on the bench; a bent-over one's is the feet on the floor, which is also its own
# source's "WHAT STAYS STILL: the torso angle, the hips, the knees, the feet".
#
# `standing` is deliberately absent. Two standing exercises genuinely travel — the squat's hips go
# back as it descends and the walking lunge steps — and they say so in `poses.json`, in a `travels`
# field quoting the sentence that says it. The rule is in the data so that the check in
# `pose_audit.py` does not need editing when the exercise list does.
PINNED_BY = {
    "lying_prone": ("foot",),
    "front_support": ("foot",),
    "seated": ("pelvis",),
    "lying_supine": ("pelvis",),
    "bent_over": ("foot",),
}


# How close to the floor a joint has to be to count as touching it.
CONTACT = 0.06


# Where a pull-up bar hangs, as a fraction of standing height. Only used to place a figure that
# touches nothing, so the drawing has the bar where a bar would be.
HANGING_BAR = 1.25


# Every `(joint, measure)` pair `solve` knows how to put on a figure, and the list it is checked
# against. A pair here is a promise that some lookup below asks for it and does something with
# the answer; a pair missing from here is refused rather than skipped.
#
# It is written out by hand rather than collected from the lookups automatically, because the
# point of it is to be read: a reviewer comparing this list against `reference/exercises/poses.json`
# can see what the solver cannot yet express, which is a question about the data, and nothing
# derived from the code would answer it.
APPLIES = frozenset({
    # Segment tilts. Not joint bends — they say where a whole segment points.
    ("spine", "from_vertical"), ("spine", "from_horizontal"),
    ("shoulder", "from_vertical"), ("shoulder", "from_horizontal"),
    ("forearm", "from_vertical"), ("forearm", "from_horizontal"),
    ("hip", "from_vertical"), ("hip", "from_horizontal"),
    # Bends, in either of the two opposite conventions `flexion_from` reconciles.
    ("spine", "flexion"),
    ("neck", "flexion"),
    ("shoulder", "flexion"), ("shoulder", "abduction"), ("shoulder", "rotation"),
    ("elbow", "included"), ("elbow", "flexion"),
    ("wrist", "included"), ("wrist", "flexion"),
    ("hip", "included"), ("hip", "flexion"),
    ("hip", "splay"),
    ("knee", "included"), ("knee", "flexion"),
    ("ankle", "included"), ("ankle", "flexion"),
    # Turns about a long axis rather than bends across one.
    ("wrist", "rotation"),
    ("foot", "splay"),
})


def _unapplied(angles):
    """Everything in a phase that `solve` did not put on the figure, described in full.

    The hole this closes: `_lookup` selects by `(joint, measure)`, so an angle whose pair no
    lookup asks for was never refused — it was simply never seen. Across the fourteen exercises
    and both phases that was forty of eighty-five angles, and twenty-one of those were non-zero
    and silently wrong, which is why the leg press used to start with straight legs and the plank
    used to hold a straight arm.

    Of the other nineteen, one was the squat's deliberate high-bar alternative and eighteen were
    zero — and the zeros were the more dangerous half, because a zero equals the rest pose, so a
    missing handler for one looks exactly like a correct one until somebody edits the number.
    They are refused too. `pipeline/pose_audit.py` re-measures all of this against the file.
    """
    problems = []
    for angle in angles:
        joint, measure, side = angle["joint"], angle["measure"], angle["side"]
        source = angle.get("source", "")
        if measure == "ambiguous":
            problems.append(f"{joint} ({side}) {angle['degrees']} is still `ambiguous`: the "
                            f"source did not say whether it is an included angle or flexion, and "
                            f"deciding is not this code's job — {source!r}")
        elif (joint, measure) not in APPLIES:
            problems.append(f"{joint} ({side}) {measure} {angle['degrees']}: `solve` has no "
                            f"handler for ({joint}, {measure}) — {source!r}")
    return problems


def solve(pose, proportions=PROPORTIONS):
    """A posed skeleton from one phase of one exercise.

    Anything the source did not state is left at its rest value rather than invented, which is
    why a sparsely described movement comes out looking half-neutral. That is the correct
    outcome: it is visibly incomplete, which sends somebody back to the notes, where a
    confidently wrong pose would not.

    Raises `ValueError` if the phase states an angle this solver cannot apply — see `APPLIES` and
    `_unapplied`. Not stated and stated-but-unusable are different situations and only the first
    one is allowed to be quiet.
    """
    p = proportions
    angles = pose.get("angles", [])
    skeleton = Skeleton()

    pitch = ORIENTATIONS.get(pose.get("orientation", "standing"), 0.0)
    posture = POSTURE.get(pose.get("orientation", "standing"), {})

    def oriented(vector):
        return _rotate(vector, X, pitch) if pitch else vector

    # The spine. `from_vertical` is a global tilt and not a joint bend, so it rotates the whole
    # torso rather than accumulating down a chain of vertebrae this skeleton does not have.
    lean = _lookup(angles, "spine", "both", {"from_vertical", "from_horizontal"})
    tilt = _from_vertical(lean) if lean else 0.0

    # The spine's own SHAPE, which is a different quantity from the lean above: a deadlift with a
    # thirty-degree lean and a flat back, and the same deadlift rounded over, differ here and
    # nowhere else. The sources talk about it constantly ("no sag", "no rounding") and every value
    # in the file is currently zero, so this handler exists for whoever edits those numbers.
    #
    # One segment cannot hold a curve, so the rounding is applied as a single bend at the chest,
    # which is where a rounded back is most visible, and the pelvis-to-chest segment keeps the
    # lean. Deliberately the same axis and the same sign as the lean, so the two can never
    # disagree about which way forward is: whatever a 45-degree lean does, 45 degrees of rounding
    # does more of. That is also why the sign correction below is one change and not three.
    bow = _lookup(angles, "spine", "both", {"flexion"})
    rounded = bow["degrees"] if bow else 0.0

    # And the head on top of it. Anatomically the nod belongs to the cervical spine, and this
    # skeleton has exactly one cervical segment, so it lands on neck-to-head.
    nod = _lookup(angles, "neck", "both", {"flexion"})
    nodded = nod["degrees"] if nod else 0.0

    pelvis = (0.0, 0.0, p.pelvis)
    skeleton.add("pelvis", pelvis)

    # Negated, all three of them. A positive rotation about +X carries a DOWNWARD bone forward,
    # which is what the top of this file derives and what every arm and leg here relies on — but
    # the spine points UP, so the same positive rotation carried it BACKWARD. Every hinging figure
    # in the set leaned away from its own weight: the squat finished with its chest behind its
    # hips, the deadlift started with its chest behind its heels, and the bent-over row rowed
    # while tipping backwards.
    #
    # The hip angle cannot find this, and that is the trap worth remembering. `hip` is the angle
    # BETWEEN the torso and the femur, so negating the torso's rotation and negating the femur's
    # `- tilt` produce exactly the same number — the squat solves to 56.7 against a stated 55 and
    # the deadlift to 99.6 against a stated 100 either way, and a figure leaning backwards scores
    # as well as one leaning forwards. Doing both at once scores 38.4 and 22.5 and is the only
    # variant the number rejects.
    #
    # What tells them apart is where the chest actually ends up. `chest.y - pelvis.y` must be
    # POSITIVE for a forward hinge, because +Y is the way the figure faces; it was -0.127 for the
    # squat finish, -0.152 for the deadlift start and -0.180 for the row. That sign is the check
    # to keep, and `pose_audit.py` cannot keep it — it accounts for angles, not for directions —
    # so it belongs to the stick figures and to whoever reads them.
    spine_dir = oriented(_rotate((0.0, 0.0, 1.0), X, -tilt))
    chest = _step(pelvis, spine_dir, p.spine)
    skeleton.add("chest", chest, "pelvis", spine_dir)
    upper_spine = oriented(_rotate((0.0, 0.0, 1.0), X, -(tilt + rounded)))
    neck = _step(chest, upper_spine, p.neck)
    skeleton.add("neck", neck, "chest", upper_spine)
    head_dir = oriented(_rotate((0.0, 0.0, 1.0), X, -(tilt + rounded + nodded)))
    skeleton.add("head", _step(neck, head_dir, p.head), "neck", head_dir)

    # The body's own axes. Every orientation in `ORIENTATIONS` is a pitch about X, so `across` is
    # always X exactly; it is written this way because the day an orientation rolls or turns the
    # figure, this line is already right and the ones that use it do not need revisiting.
    across = oriented((1.0, 0.0, 0.0))
    facing = oriented((0.0, 1.0, 0.0))
    # The axis the figure stands on: what a turn "seen from above" turns about. Only the hip's
    # splay uses it — the foot's splay turns about its own leg, which for a standing figure is
    # this same line and for a lying one is not.
    upright = oriented((0.0, 0.0, 1.0))

    for side, sign in (("left", -1.0), ("right", 1.0)):
        # --- arm
        shoulder = (chest[0] + across[0] * p.shoulder_half * sign,
                    chest[1] + across[1] * p.shoulder_half * sign,
                    chest[2] + across[2] * p.shoulder_half * sign)
        skeleton.add(f"shoulder.{side}", shoulder, "chest")

        upper = oriented(DOWN)
        # The frame the arm below the shoulder bends in. It starts as the body's and is carried
        # through every rotation the upper arm takes, so the elbow and the wrist always bend about
        # an axis that is square to the humerus instead of one that can line up with it.
        arm_across = across
        hang = _lookup(angles, "shoulder", side, {"from_vertical", "from_horizontal"})
        if hang:
            # The humerus as a segment tilt, measured against gravity rather than against the
            # torso, which is exactly what makes it a different quantity from the flexion below:
            # the pushdown's "upper arms vertical and pinned to the sides" is the defining
            # constraint of that lift and says nothing about how far the shoulder has moved.
            #
            # NOT passed through `oriented`, and that is the whole point of the measure. The
            # plank's "upper arms close to vertical" is a statement about the floor, not about the
            # body's own axis, and a front-support figure's axis lies along the floor — orienting
            # this would lay the upper arms down flat and draw a plank with nothing holding it up.
            upper = _rotate(DOWN, X, _from_vertical(hang))
        spread = _lookup(angles, "shoulder", side, {"abduction"})
        if spread:
            # Abduction swings the arm out sideways, away from the midline on whichever side it
            # belongs to — hence the sign. "Elbows 75 degrees from the torso" lands here, not on
            # the elbow, which is the single most common misreading of these notes.
            # Negated, and the sign was wrong until a 3D render caught it. A positive rotation
            # about the forward axis carries a downward bone toward -X, so the right arm — whose
            # shoulder sits at +X — was swung across the body and out the other side. The arms
            # crossed.
            #
            # The stick-figure preview could not have found this and did not: at exactly ninety
            # degrees a crossed arm and an abducted one are both horizontal, and a front view
            # flattens the difference away. Worth remembering about that preview — it catches a
            # limb bending the wrong way, not a limb on the wrong side.
            #
            # About the body's own forward axis, not the world's +Y. That distinction was invisible
            # while every abducting exercise stood up — and silent, not wrong-looking, on the one
            # that lies down: the bench press's arm lies ALONG the world +Y when the figure is
            # supine, a rotation about an axis parallel to the bone changes nothing, and its only
            # shoulder angle moved the figure exactly 0.000000 m. Rotating about `facing` swings
            # the arm in the figure's own frontal plane whichever way the figure is lying.
            #
            # And it is taken BEFORE the flexion below, for the reason the leg's version spells
            # out: both start from the hanging arm, so the second one has to be taken in the frame
            # the first leaves behind or it turns the bone about its own length.
            turn = -spread["degrees"] * sign
            upper = _rotate(upper, facing, turn)
            # The elbow is a hinge fixed to the humerus: abduct the arm and the hinge abducts with
            # it. So the arm's own across-axis travels through the same rotation, and the forearm
            # below bends about that rather than about the body's. Bending about the body's axis
            # is why the lateral raise's "soft bend" vanished at the top of the raise — at ninety
            # degrees of abduction the upper arm lies along the body's across-axis, and the bend
            # had nothing to turn about.
            arm_across = _rotate(arm_across, facing, turn)
        raise_ = _lookup(angles, "shoulder", side, {"flexion"})
        if raise_:
            # Shoulder flexion lifts the arm forward, so the hand travels toward +Y. Taken about
            # the arm's own across-axis rather than about X, so that it composes with the tilt and
            # the abduction above instead of overwriting them; with neither of those, it is the
            # same rotation this line has always done.
            upper = _rotate(upper, arm_across, raise_["degrees"])
        roll = _lookup(angles, "shoulder", side, {"rotation"})
        if roll:
            # The humerus turned about its own length. It moves nothing by itself — a bone is a
            # line and rolling a line leaves it where it was — so it is the one rotation here whose
            # entire effect is on the joint BELOW it: it turns the elbow's hinge, and so decides
            # which plane the forearm swings in.
            #
            # That is why the plank needed it. Lying prone with the upper arms along the ribs, the
            # elbow's hinge lies in the floor plane, so a 90-degree elbow drives the forearm
            # straight down into the mat at every abduction between 0 and 120 — which is what the
            # arithmetic in the task report shows. Rolling the humerus brings the hinge upright and
            # the forearm comes to rest flat on the floor instead, with the elbow still at 90.
            #
            # Taken against `sign` like the other two, so a positive roll turns both forearms away
            # from the midline rather than both the same way across the body.
            arm_across = _rotate(arm_across, upper, roll["degrees"] * sign)
        elbow = _step(shoulder, upper, p.upper_arm)
        skeleton.add(f"elbow.{side}", elbow, f"shoulder.{side}", upper)

        # The forearm bends in the plane of the upper arm, about the axis across the body.
        # The forearm can also be stated outright instead of through the elbow, and the overhead
        # press is where that matters: "FOREARMS: vertical to the floor" is the entry's own master
        # cue and it fixes where the bar sits without saying one word about the elbow.
        #
        # Measured from the UPWARD vertical, unlike the humerus's and the femur's tilts, which are
        # measured from the hanging one. Each is measured from the direction its own sources call
        # that segment's vertical: an arm or a thigh called vertical is hanging, and a forearm
        # called vertical is holding something up — nobody describes a forearm resting by the hip
        # as vertical, they say the arm is straight. So 0 here is the hand ABOVE the elbow, which
        # is where a racked bar is, and 180 would be the hand below it. The sign matches the
        # spine's: positive tips the hand forward. Not re-oriented, for the same reason the other
        # segment tilts are not — it is a statement about the floor.
        fore = _rotate(upper, arm_across, _bend(angles, "elbow", side))
        held = _lookup(angles, "forearm", side, {"from_vertical", "from_horizontal"})
        if held:
            fore = _rotate((0.0, 0.0, 1.0), X, -_from_vertical(held))
        wrist = _step(elbow, fore, p.forearm)
        skeleton.add(f"wrist.{side}", wrist, f"elbow.{side}", fore)

        # The hand carries on along the forearm unless the wrist has something to say. Both of
        # the wrist's measures turn about `across`, the same axis the elbow bends about, because
        # the hand folds in the plane of the arm.
        #
        # `rotation` is one entry in the whole file, the overhead press's "knuckles rotated back
        # about 75 degrees", and the axis is a reading of the source rather than a fact from it,
        # so: back is extension, the direction opposite to flexion, hence the minus. The rival
        # reading is an axial twist of the forearm, and this layer cannot express that at all —
        # a hand here is one bone and its direction is all that leaves this file, so a turn about
        # its own axis would change nothing and quietly discard the 75 degrees, which is the
        # failure this file is now built to refuse. The entry's own note argues the same way: it
        # "suggests the hand tipped back relative to the forearm rather than an axial twist".
        twist = _lookup(angles, "wrist", side, {"rotation"})
        knuckles = twist["degrees"] if twist else 0.0
        hand_dir = _rotate(fore, arm_across, _bend(angles, "wrist", side) - knuckles)
        skeleton.add(f"hand.{side}", _step(wrist, hand_dir, p.hand), f"wrist.{side}", hand_dir)

        # --- leg
        hip = (pelvis[0] + across[0] * p.hip_half * sign,
               pelvis[1] + across[1] * p.hip_half * sign,
               pelvis[2] + across[2] * p.hip_half * sign)
        skeleton.add(f"hip.{side}", hip, "pelvis")

        # Hip flexion brings the thigh toward the chest. Measured against the torso, so the
        # torso's own lean is added back in: hinging forward at the hips with the thigh staying
        # vertical is a large hip flexion and a motionless leg, and the deadlift start depends
        # on getting exactly that right.
        #
        # When the source states nothing, the thigh stays vertical rather than inheriting the
        # lean. Those are different claims and the bent-over row is where the difference shows:
        # it states a horizontal torso and no hip angle anywhere, and inheriting the lean swung
        # its legs straight out behind it — a superman, floating, with nothing under it. That is
        # not a neutral default, it is an anatomically impossible assertion, because a standing
        # figure that leans without hinging at the hip falls over. Absent data leaves a joint at
        # rest, and rest for a thigh is vertical.
        thigh = oriented(DOWN)
        leg_across = across
        stated_hip = _lookup(angles, "hip", side, {"included", "flexion"})
        if stated_hip:
            thigh = _rotate(thigh, leg_across,
                            flexion_from(stated_hip["measure"], stated_hip["degrees"]) - tilt)
        femur = _lookup(angles, "hip", side, {"from_horizontal", "from_vertical"})
        if not femur and not stated_hip and "hip_from_horizontal" in posture:
            # Nothing stated anywhere for this hip, and the orientation word has an opinion. Taken
            # as a segment tilt rather than as flexion, because "seated" describes where the thigh
            # POINTS — horizontal, out of the hip — and not how far the hip has folded relative to
            # a torso that may be leaning back against a pad, which is exactly the lat pulldown.
            thigh = _rotate(DOWN, X, 90.0 - posture["hip_from_horizontal"])
        if femur:
            # The femur as a segment tilt, the same kind of statement as the spine's lean. The
            # lunge's "front thigh is roughly parallel with the floor" says where the thigh
            # points, not how far the hip has flexed, and that entry's own note is explicit that
            # the source never states a hip angle. Measured against gravity, so it REPLACES the
            # torso-relative flexion above rather than adding to it, and it is not re-oriented,
            # for the same reason the humerus's tilt is not.
            thigh = _rotate(DOWN, X, _from_vertical(femur))
        # Knees out, which is the squat's defining cue at the bottom and the only place in the
        # file that states it. `splay` and not `abduction`, and the source is what decides that:
        # it gives the number by equating it with the feet — "thighs in line with the 30-degree
        # foot splay" — and a foot splay is a turn about the body's upright, seen from above.
        # Abduction is a turn in the frontal plane, and taken that way this same 30 came out as
        # five degrees of visible knees-out, because at the bottom of a squat the thigh is nearly
        # horizontal and nearly parallel to the axis abduction turns about. The same degeneracy as
        # the bench press's arm, in the other limb.
        #
        # AFTER the flexion, for the same reason and in the opposite order to the arm's abduction:
        # a hanging thigh lies ALONG the upright, so splaying it first would be a roll about its
        # own length and would change nothing at all. A transverse turn has to wait until the limb
        # has left the vertical.
        #
        # Negated against `sign` exactly as the foot's splay is, so each thigh turns away from the
        # midline, and the knee's hinge is carried with it so the shin and the foot follow the
        # thigh out.
        knees_out = _lookup(angles, "hip", side, {"splay"})
        if knees_out:
            turn = -knees_out["degrees"] * sign
            thigh = _rotate(thigh, upright, turn)
            leg_across = _rotate(leg_across, upright, turn)

        knee = _step(hip, thigh, p.thigh)
        skeleton.add(f"knee.{side}", knee, f"hip.{side}", thigh)

        # The knee only bends one way: the shin swings backward, never forward, which is the
        # one negative sign in here.
        shin = _rotate(thigh, leg_across,
                       -_bend(angles, "knee", side, posture.get("knee_flexion", 0.0)))
        ankle = _step(knee, shin, p.shin)
        skeleton.add(f"ankle.{side}", ankle, f"knee.{side}", shin)

        # The foot sits at a right angle to the shin unless the source says otherwise; an ankle
        # "neutral ~90 degrees" is that right angle, which is zero departure from rest.
        ankle_angle = _lookup(angles, "ankle", side, {"included", "flexion"})
        lift = 0.0
        if ankle_angle:
            lift = (90.0 - ankle_angle["degrees"] if ankle_angle["measure"] == "included"
                    else ankle_angle["degrees"])
        foot = _rotate(shin, leg_across, 90.0 + lift)

        # Toes out. Eight non-zero splays across five exercises used to land nowhere at all, so
        # no figure in the set turned its feet out, which is the first thing a lifter would
        # notice about the squat.
        #
        # The axis is the LEG'S OWN, pointing up the shin, because splaying the toes out is the
        # whole leg turning in its hip socket. For a figure standing on a flat floor that axis is
        # the world's +Z and this is exactly "rotate the foot outward about the vertical"; it stops
        # being so the moment a shin is not vertical, and the axis has to follow the leg or the
        # turn leaks away into the wrong plane. The deadlift's inclined shins lost a third of their
        # splay that way, and the walking lunge's trailing leg — shin back, foot pointing straight
        # down the body's upright — would have lost all of it, silently, the first time anybody
        # gave that phase a foot angle.
        #
        # Negated against `sign` for the same reason abduction is: a positive turn about the
        # upright axis carries a forward-pointing foot toward -X, so the left foot — whose hip sits
        # at -X — needs the positive one to splay away from the midline.
        #
        # Measured from STRAIGHT AHEAD, which is what every one of its sources says ("toes turned
        # out approximately 30 degrees", "toes pointing straight forward"), so when the hip has
        # already turned the whole leg out the ankle only has to supply the remainder. Without
        # that subtraction the squat's bottom position counted the same rotation twice — the
        # source's one sentence, "thighs in line with the 30-degree foot splay", is recorded once
        # on the hip and once on the foot — and the toes came out 58 degrees from forward instead
        # of 30.
        #
        # The two compose exactly when they are equal, which is the case the sources describe. At
        # unequal values the subtraction is close rather than exact, because a turn about the
        # upright and a turn about an inclined shin are not the same rotation; that is worth
        # knowing before anybody writes a foot splay that disagrees with its own hip.
        splay = _lookup(angles, "foot", side, {"splay"})
        if splay:
            already = knees_out["degrees"] if knees_out else 0.0
            up_the_leg = (-shin[0], -shin[1], -shin[2])
            foot = _rotate(foot, up_the_leg, -(splay["degrees"] - already) * sign)
        skeleton.add(f"foot.{side}", _step(ankle, foot, p.foot), f"ankle.{side}", foot)

    # Only now, with every lookup made, is it known what went unused. Checked on the way out
    # rather than on the way in so that this reads as one list of what the solver can apply
    # (`APPLIES`) against one list of what the phase asked for, with no second copy of the
    # matching rules to fall out of step with `_lookup`.
    problems = _unapplied(angles)
    if problems:
        raise ValueError(
            f"{len(problems)} of this phase's {len(angles)} angles were not applied to the "
            f"figure:\n" + "\n".join(f"  - {problem}" for problem in problems) +
            f"\n  Add a handler in `solve` and list the pair in `APPLIES`, or correct the entry "
            f"in {POSES.name}. A zero is no excuse for a missing handler: it matches the rest "
            f"pose today and stops matching it the moment somebody edits the number.")

    return ground(skeleton, pose.get("orientation", "standing"))


def sides_are_sane(skeleton, tolerance=0.02):
    """Each hand on the same side of the midline as its own shoulder, or a description of which
    is not.

    Exists because the abduction sign was inverted and nothing noticed: the arms crossed over
    the chest, and a front-on stick figure drew that as the same horizontal line a correct raise
    draws. This is the check that would have caught it, so it runs in the preview now.
    """
    wrong = []
    for side, expect in (("left", -1.0), ("right", 1.0)):
        shoulder = skeleton.at(f"shoulder.{side}")[0]
        hand = skeleton.at(f"hand.{side}")[0]
        if shoulder * expect < 0:
            wrong.append(f"the {side} shoulder is on the wrong side of the body")
        if hand * expect < -tolerance:
            wrong.append(f"the {side} hand has crossed to the other side of the body "
                         f"(x={hand:+.3f}, shoulder x={shoulder:+.3f})")
    return wrong


def load(path=POSES):
    if not Path(path).exists():
        raise SystemExit(f"{path} does not exist yet — run the extraction that writes it first.")
    return json.loads(Path(path).read_text())


def phases_of(data, exercise):
    for entry in data["exercises"]:
        if entry["name"] == exercise:
            return {phase["phase"]: phase for phase in entry["phases"]}
    raise SystemExit(f"{exercise!r} is not in the pose file.")


def _self_check():
    """Solve every phase of every exercise in the pose file, and run `sides_are_sane` on each.

    Importing this module proves nothing about a file whose entire output is angles, and until
    now running it proved nothing either — it had no entry point at all, so the one check it
    ships, `sides_are_sane`, was called from nowhere in the project despite its own docstring
    saying it runs. `pipeline/pose_audit.py` is the thorough version, which also accounts for
    every angle individually; this is the thirty-line version that belongs to the module, so
    `python3 pipeline/pose_model.py` is worth typing.
    """
    data = load()
    failures = []
    for entry in data["exercises"]:
        for phase in entry["phases"]:
            where = f"{entry['name']} [{phase['phase']}]"
            try:
                skeleton = solve(phase)
            except ValueError as problem:
                failures.append(f"{where}: {problem}")
                continue
            for complaint in sides_are_sane(skeleton):
                failures.append(f"{where}: {complaint}")
            print(f"  ok  {where}: {len(phase['angles'])} angles, "
                  f"{len(skeleton.joints)} joints")
    if failures:
        print(f"\n{len(failures)} failure(s):")
        for failure in failures:
            print(f"  FAIL  {failure}")
        raise SystemExit(1)
    print(f"\nall {len(data['exercises'])} exercises solved, both phases, sides sane.")


if __name__ == "__main__":
    _self_check()
