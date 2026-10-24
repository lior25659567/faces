#!/usr/bin/env python
"""Build the fitness character's BODY as a quad mesh made to bend, inside Blender.

    blender --background --python pipeline/blender/character.py -- --out /tmp/sheet.png

Why this file exists. The character we had came out of image-to-3D: 52,431 faces, every one of
them a triangle, 35,071 non-manifold edges, no edge loop anywhere and no finger geometry at all.
A night of rigging work went into making it fold and the conclusion was that the surface itself
was never built to fold — you cannot put a crease in triangle soup, you can only hide it. So the
surface is built here instead, and it is built the way a modeller builds one:

  * by LOFTING. Every part of this body is a tube of rings. A ring is N vertices on a cross
    section; consecutive rings are bridged into a band of quads. Nothing else makes a quad —
    there is no triangulation step to come back and undo, because no face is ever created from
    anything but four corners of a (ring, column) lattice. 100% quads is a property of the
    construction, not a cleanup pass.

  * with the loops WHERE THE BEND IS. Three or more rings packed into each joint's bend zone and
    sparse rings down the straight middle of a bone. That is the single thing that lets a surface
    fold: with one ring at a joint the fold has to happen inside one band of quads and the band
    collapses; with three the crease is shared and the silhouette stays round.

  * with the holes CUT BY NOT BUILDING THEM. An arm socket is a rectangle of the torso's
    (ring, column) lattice whose faces are never created, so its boundary is known exactly —
    2*(rings + columns) vertices, in order, by arithmetic. Nothing is deleted, nothing is walked,
    nothing has to be found. That is also why the limb ring that bridges onto it has exactly the
    matching vertex count: the count is chosen from the socket's arithmetic.

  * with the weights ASSIGNED AT BIRTH. `Build.ring` takes the bone weights for the ring it is
    creating. A ring is created because it belongs somewhere, so the thing that knows where it
    belongs is the line that makes it. There is no distance heuristic in this file, no nearest
    segment, no heat map, and nothing in `fit_model.py` is needed to use it.

The bone names are `make_character.BONES`'s, so `pose_model`, `exercise_stage` and `animate` drive
this character with no changes: they pose bones by name and the names are the same.

This stage delivers a BODY. No face, no hair, no materials beyond a flat skin tone to render the
silhouette against. It is judged on topology and proportion.
"""

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import bmesh
import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import make_character  # noqa: E402  — BONES, PARENTS, material(), use_cycles(), the palette


# --- proportions -------------------------------------------------------------------------------
#
# Everything that decides how big a part of the body is lives in here, as a FRACTION of total
# height. Nothing in the geometry below is allowed a dimension of its own: the app measures a real
# body and will want to push those measurements in, so a number baked into a ring position would
# be a number that cannot be measured.
#
# The defaults are a stylised figure, not an anatomical one, and they are NOT the old character's.
# Measured on the old one, as fractions of height: arm (shoulder to hand) 31.6%, leg (hip to
# ankle) 36.1%, hand 5.1%. The arm-to-leg ratio of 0.88 is why the arms read as too long — the
# arm is normal and the LEG is short, so the hands hang past the hem of the shorts with almost no
# leg below them. A stylised figure runs nearer 0.79. These defaults are 0.38 / 0.47 = 0.81, and
# the hand is 10% rather than 5.1%, which is the other half of the same complaint: a hand at half
# length cannot read as a hand, only as a blob on the end of an arm.
@dataclass
class Proportions:
    """The body as fractions of its own height. `height` is the only unit-carrying number."""

    height: float = 1.0                 # heel to crown, in Blender units (metres)

    # The vertical budget. These four are the whole height and `validate` says so.
    #
    # The hip joint is 2cm lower than stage 1 had it (0.455) and the knee with it, and the torso
    # takes the 2cm. Not a taste call: with the joint at 0.455 the crotch hung 6cm below it, so
    # every ring of thigh that fitted inside the hip's bend zone sat ABOVE the hole it was bridged
    # to — the thigh folded back up inside the pelvis and came out again, and that fold was the
    # "hem of a pair of shorts". The house drawing has the knee at about 0.20 and the crotch at
    # about 0.37, so the move is also toward the drawing, not away from it.
    leg_f: float = 0.435                # floor to hip joint
    # Stage 2b raised the chest joint 0.5cm and took it out of the neck, so the head did not move:
    # measured off silhouettes, the old character's shoulders run flat at 0.73-0.74 of its height
    # right up to its neck, where ours sloped away from 0.71; that slope was what made our neck
    # read as a stalk, not its thickness — the old one's neck is the NARROWER of the two.
    torso_f: float = 0.255              # hip joint to the base of the neck (chest joint)
    neck_f: float = 0.050               # chest joint to the base of the skull (neck joint)
    head_f: float = 0.260               # base of the skull to the crown

    # How the leg divides. ankle_f + shin_f + thigh_f == leg_f.
    thigh_f: float = 0.215
    shin_f: float = 0.167
    ankle_f: float = 0.053              # ankle joint height off the floor

    # The arm, shoulder joint to fingertip: upper_arm_f + forearm_f + hand_f.
    upper_arm_f: float = 0.137
    forearm_f: float = 0.128
    hand_f: float = 0.095               # wrist to fingertip

    # Widths and depths.
    shoulder_half_f: float = 0.120      # centre line to shoulder joint (shoulder width = 2x)
    hip_half_f: float = 0.058
    shoulder_drop_f: float = 0.010      # shoulder joint sits this far below the chest joint
    chest_half_f: float = 0.100         # half width at the chest
    chest_depth_f: float = 0.086        # half depth (front to back) at the chest
    waist_half_f: float = 0.084
    waist_depth_f: float = 0.074
    pelvis_half_f: float = 0.094        # the bottom of the trunk, which the thighs leave
    pelvis_depth_f: float = 0.072
    seat_f: float = 0.024               # how far the seat bulges out BEHIND the pelvis's depth
    chest_set_f: float = 0.018          # and how far the chest ring sits in front of the hips'
    neck_r_f: float = 0.047
    head_r_f: float = 0.130             # head_f / 2 — a sphere, so they are the same number
    head_depth_f: float = 0.143         # heads are deeper than they are wide

    # Limb thicknesses, as radii.
    shoulder_r_f: float = 0.047
    bicep_r_f: float = 0.044
    elbow_r_f: float = 0.035
    forearm_r_f: float = 0.038
    wrist_r_f: float = 0.023
    hip_r_f: float = 0.058
    thigh_r_f: float = 0.053
    knee_r_f: float = 0.042
    calf_r_f: float = 0.049
    ankle_r_f: float = 0.028

    # The hand, which is the part the user named first. A hand is a palm slab plus five digits;
    # the finger widths below are what makes the distal cap's seven columns read as four fingers
    # with three narrow webs rather than as seven equal teeth.
    palm_len_f: float = 0.055           # wrist to knuckles
    palm_knuckle_half_f: float = 0.030
    palm_knuckle_depth_f: float = 0.009
    finger_web_f: float = 0.004         # the gap between two fingers at the knuckle
    finger_r_f: float = 0.0068
    finger_len_f: tuple = (0.045, 0.047, 0.043, 0.036)   # index, middle, ring, pinky
    thumb_len_f: float = 0.038
    thumb_r_f: float = 0.0105

    # The foot: an ankle column with a toe box forward of it.
    foot_f: float = 0.100               # ankle to toe tip, forward
    heel_f: float = 0.042               # how far the heel sticks out behind the ankle
    instep_f: float = 0.046             # how far the ankle column reaches forward on its own
    foot_half_f: float = 0.038
    toe_half_f: float = 0.034           # half the width of the foot where the toes leave it
    toe_depth_f: float = 0.026          # the forefoot's thickness scale (see FOOT_HEIGHT)
    sole_z_f: float = 0.002             # the sole plane: how high the underside sits off the floor
    arch_f: float = 0.016               # how far the inside of the midfoot lifts, heel to ball
    toe_lift_f: float = 0.006           # twice how far the crease under the toes lifts off it

    # The A-pose. 45 degrees below horizontal at the shoulder is the brief's number and it is not
    # cosmetic: a T-posed rest makes every arms-down pose a 90-degree shoulder rotation, which is
    # where the old model's shoulder tore. The legs splay so that a hand is not within a
    # centimetre of a thigh — nothing in this file weights by distance, but the renders and the
    # exports are read by things that do.
    arm_drop_deg: float = 45.0          # below horizontal
    leg_splay_deg: float = 5.0          # outward from vertical
    hip_rise_f: float = 0.020           # the hip saddle: how far the crotch ring lifts at the side
    crotch_drop_f: float = 0.016        # how far the bottom ring dips on the centre line
    crotch_depth_f: float = 0.014       # and how far the crotch vertex sits below that

    # --- absolute dimensions ---------------------------------------------------------------
    def __getattr__(self, name):
        """`p.thigh` is `p.thigh_f * p.height`, for every fraction on the dataclass.

        Written as one accessor rather than thirty properties so that adding a proportion is
        adding one line. A tuple of fractions (finger lengths) scales element-wise.
        """
        field_name = f"{name}_f"
        if name.startswith("_") or field_name == name:
            raise AttributeError(name)
        try:
            value = object.__getattribute__(self, field_name)
        except AttributeError:
            raise AttributeError(name) from None
        height = object.__getattribute__(self, "height")
        if isinstance(value, tuple):
            return tuple(v * height for v in value)
        return value * height

    def validate(self):
        """The budgets have to add up, and a typo in one fraction is otherwise invisible.

        Caught a real one: with the head at 0.20 and the legs at 0.47 an earlier draft left the
        torso and neck summing to 0.345, so the figure was 1.015 tall and every fraction reported
        afterwards was wrong by 1.5% — small enough to pass a glance at the render.
        """
        problems = []
        total = self.leg_f + self.torso_f + self.neck_f + self.head_f
        if abs(total - 1.0) > 1e-6:
            problems.append(f"leg+torso+neck+head = {total:.6f}, not 1.0")
        leg = self.thigh_f + self.shin_f + self.ankle_f
        if abs(leg - self.leg_f) > 1e-6:
            problems.append(f"thigh+shin+ankle = {leg:.6f}, not leg_f = {self.leg_f}")
        if self.head_f and abs(self.head_r_f * 2 - self.head_f) > 1e-6:
            problems.append(f"head radius {self.head_r_f} is not half of head_f {self.head_f}")
        if problems:
            raise SystemExit("proportions do not add up:\n  " + "\n  ".join(problems))
        return self

    @property
    def arm_f(self):
        return self.upper_arm_f + self.forearm_f + self.hand_f

    def fractions(self):
        """The numbers this stage gets judged on, as a dict, so they can be printed and compared."""
        return {
            "head (crown to chin)": self.head_f,
            "neck": self.neck_f,
            "torso (hip to neck base)": self.torso_f,
            "leg (hip to floor)": self.leg_f,
            "arm (shoulder to fingertip)": self.arm_f,
            "hand (wrist to fingertip)": self.hand_f,
            "shoulder width": self.shoulder_half_f * 2,
            "arm / leg": self.arm_f / self.leg_f,
        }

    # --- the joints the geometry is built around -------------------------------------------
    def joints(self):
        """Every joint `make_character.BONES` names, in the A-pose, derived from the fractions.

        The armature is built from this same dict, which is the point: a bone cannot end up
        somewhere the surface was not built around, because there is only one source for both.
        """
        at = {}
        at["pelvis"] = Vector((0.0, 0.0, self.leg))
        at["chest"] = Vector((0.0, 0.0, self.leg + self.torso))
        at["neck"] = at["chest"] + Vector((0.0, 0.0, self.neck))
        at["head"] = at["neck"] + Vector((0.0, 0.0, self.head_r))
        drop = math.radians(self.arm_drop_deg)
        splay = math.radians(self.leg_splay_deg)
        for side, sign in (("left", -1.0), ("right", 1.0)):
            arm_dir = Vector((sign * math.cos(drop), 0.0, -math.sin(drop)))
            shoulder = Vector((sign * self.shoulder_half, 0.0,
                               at["chest"].z - self.shoulder_drop))
            at[f"shoulder.{side}"] = shoulder
            at[f"elbow.{side}"] = shoulder + arm_dir * self.upper_arm
            at[f"wrist.{side}"] = at[f"elbow.{side}"] + arm_dir * self.forearm
            at[f"hand.{side}"] = at[f"wrist.{side}"] + arm_dir * self.hand

            leg_dir = Vector((sign * math.sin(splay), 0.0, -math.cos(splay)))
            hip = Vector((sign * self.hip_half, 0.0, self.leg))
            at[f"hip.{side}"] = hip
            at[f"knee.{side}"] = hip + leg_dir * self.thigh
            at[f"ankle.{side}"] = at[f"knee.{side}"] + leg_dir * self.shin
            at[f"foot.{side}"] = at[f"ankle.{side}"] + Vector((0.0, self.foot, -self.ankle * 0.55))
        return at


# --- the lattice -------------------------------------------------------------------------------
#
# Ring sizes. Every one of these is decided by the hole it has to meet, not by taste:
#
#   TORSO = 20   a 3 x 5 rectangle of its lattice is an arm socket, and 2*(3+5) = 16 = ARM.
#                Its bottom ring splits in two around one crotch vertex: an 11-vertex arc plus
#                that vertex is 12 = LEG, on each side.
#   ARM = 16     the wrist carries straight on into the palm, so the palm's cross section is the
#                arm's. 16 = 2*(7+1), which is a seven-column strip across the knuckles: four
#                finger columns and three webs.
#   LEG = 12     2*(4+2) caps the sole as a 4 x 2 grid; a 3 x 3 rectangle of the lower lattice is
#                the socket the front of the foot leaves, and 2*(3+3) = 12 = TOE. (Stage 1's was
#                2 x 3, a socket only two bands tall, so the top of the foot started 4cm below
#                the ankle and the shin met it in a notch: the "peg with a sausage" foot.)
#   NECK = 12    2*(3+3) is the 3 x 3 hole in the middle of the torso's 5 x 5 shoulder cap, and
#                the same hole in the middle of the head's 5 x 5 bottom face.
TORSO, ARM, LEG, NECK, TOE, THUMB, FINGER = 20, 16, 12, 12, 12, 8, 4
HEAD_CELLS = 7          # cells per cube-sphere face; odd, so a 3 x 3 hole can be centred
# Seven rather than five. The chin hole is always the middle 3 x 3 of a face, so on a 5 x 5 face
# it is 60% of the head's width — a neck as wide as a jaw, and at 26% of height the head is big
# enough that the flare showed. On a 7 x 7 face the same hole is 43%, which is a neck.
# Four by four, not three by five. Both give a 16-vertex boundary, but 3 x 5 is a slot 61mm tall
# and 110mm wide on a figure whose arm is 94mm across: the bridge onto it had to flare in one
# direction and pinch in the other, and at the front corner of each armpit the band's quads
# crossed over themselves. 4 x 4 is 78 x 88mm — the shape of the thing going into it.
SOCKET_RINGS, SOCKET_COLS = 4, 4

# Where the rings go. These are topology, not proportion: a parameter in bone units (0 at the
# first joint, 1 at the next) saying where a cross section sits. Clusters of three-plus inside a
# bend zone, singles down the straight middle. This table is the whole of requirement 3.
ARM_RINGS = [
    # Six rings over the first quarter of the humerus, not four. The first two are mostly the
    # socket's own outline (see SHOULDER_FLOW) and are too far off the arm's axis for the loop
    # audit to certify as cross sections, which is correct — they are a deltoid, not a tube. Four
    # certifiable loops still land inside the bend zone, and the extra pair is what gives the
    # armpit room to twist gradually instead of crossing over itself in one band.
    0.035, 0.08, 0.125, 0.17, 0.22, 0.27,   # shoulder
    0.35, 0.50, 0.64,                       # the straight middle of the humerus
    0.76, 0.85, 0.92, 1.00, 1.08, 1.15, 1.24,   # elbow: seven loops across the bend
    1.38, 1.52, 1.66,                       # the straight middle of the forearm
    1.76, 1.84, 1.90, 1.95, 2.00,           # wrist: five loops up to the joint itself
]
# Six, not five, and the first one earns its place: the thumb socket breaks the loop walk on the
# three rings it spans, so with five rings the whole palm side of the wrist was uncertifiable and
# the wrist reported four loops with not one of them past the joint.
PALM_RINGS = [0.06, 0.16, 0.32, 0.52, 0.76, 1.00]   # wrist to knuckles, in palm-length units
LEG_RINGS = [
    # The first three are DRAPED (see `drape`): their nominal positions only set their radii and
    # the centres `flow_out` shapes them about. The fourth is the first plain ring, and it has to
    # be below the lowest point of the hole — the crotch — or a drape has nowhere to go.
    0.10, 0.16, 0.22, 0.28,                 # hip
    0.40, 0.55, 0.70,
    0.82, 0.90, 0.96, 1.00, 1.05, 1.11, 1.19,   # knee
    1.32, 1.48, 1.64,
    1.76, 1.86, 1.93, 2.00,                 # ankle, from above
]
# In TORSO units, not in fractions of height: 0 is the hip joint and 1 is the chest joint, so
# these survive a change to the vertical budget. They were absolute heights until the head went
# from 20% to 26% of the figure and every one of them had to be retyped by hand — which is the
# definition of a hard-coded constant in the geometry, however it is spelled.
#
# The bottom ring sits just under the hip joint (it was 3cm under it), so the hole the legs come
# out of is above every leg ring in the bend zone — see `drape`. The rings from 0.391 up are
# stage 1's, re-expressed for the longer torso so that they stay at the same height under the
# chest joint: the arm socket and the shoulder did not move.
#
# Stage 2b: these are now the rings' SHAPE coordinates — where each one reads its width, depth
# and weights off the profiles in `torso` — and only the first four are also its height. The arm
# sockets' six rings are placed from the SHOULDER JOINT (SOCKET_LEVELS), and the three between are
# spaced evenly; see `torso_heights`. Squaring the shoulders meant raising the shoulder joint 2.3cm
# against the torso, and an armpit is the one place in this mesh where a centimetre matters: the
# arm's first ring sits within half a millimetre of the socket's corners. Tied to the joint, the
# socket moves with it and the junction stays the shape it was tuned at.
TORSO_RINGS = [-0.032, 0.052, 0.148, 0.260, 0.391, 0.519, 0.646,
               0.766, 0.858, 0.943, 1.014, 1.050, 1.078]
SOCKET_FROM = 7          # the socket spans TORSO_RINGS[7] .. [11]
# Rings 7..12, in shoulder radii above (+) or below (-) the shoulder joint: exactly where stage 1's
# socket rings sat relative to its own joint.
SOCKET_LEVELS = (-0.819, -0.330, 0.122, 0.500, 0.691, 0.840)
# How the hip's lift and the thigh's share of the skin fade upward over the lowest torso rings.
HIP_SADDLE = (1.0, 0.52, 0.18)
HIP_SHARE = (0.55, 0.26, 0.08)
# From the shoulder cap's hole to the chin, evenly. They bunched toward the chin when the neck
# was 4cm long; at 2cm that left two rings 2mm apart, and a 32 degree nod stretched the edge
# between them 2.8 times.
NECK_RINGS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
# The hole the neck leaves the shoulder yoke through: half width and half depth in neck radii.
NECK_HOLE = (1.26, 1.32)
# Task 5c: the neck met the shoulders in a CORNER, and the trunk met the yoke in another. Stage
# 2b's flatter yoke left a lid on a can: the trunk's top ring stood straight up off the socket
# (85 degrees between faces), the yoke ran across flat (6 degrees), and the neck stood straight up
# off it (73 degrees at its first ring, whose band to the hole was dead level). Subdivided, the
# neck base turned 50 degrees in 3mm — a fillet of radius 3.7mm on a body a metre tall. Both
# corners are softened by moving rings that already exist — nothing is added, so no count and no
# weight changes. Measured: neck base 73.7 -> 30.9 degrees (shared with the next ring), trunk top
# 84.5 -> 44.9, subdivided neck fillet 3.7 -> ~10mm; on the wrapped boy the worst skin stretch at
# arms-down 26.1 -> 15.3mm, and on the man the region fit no longer has to drop its pull here.
#
# What this does NOT do on its own is make the boy's vest strap lie flush with the arms down. That
# flap is mostly the vest and the body under it taking DIFFERENT weights — the vest from the arm
# root's socket ring (55% upper arm), the body from the trunk's top ring beside it (26-38%) — and
# sliding 10-18mm apart when the arm turns. See .sdd/task-5-report.md, task 5c.
#
# TRUNK_ROUND  how far the trunk's top ring is pulled in toward the neck hole, as a fraction of the
#              yoke's width at that column. Splits the 85 degree edge into two of about 40.
# NECK_FILLET  where the neck's first rings sit on a quarter-ellipse from the hole (90: level, the
#              yoke's tangent) to the ring after them (0: the neck's own upright line). Even steps
#              of 30 degrees, so the turn is shared by three rings instead of taken by one.
TRUNK_ROUND = 0.25
NECK_FILLET = (60.0, 30.0)
# How much of the bone ABOVE a joint the first rings below it keep. Without these the weights
# step from one bone to the next across a single five-millimetre edge, and a five-millimetre edge
# asked to absorb a 42-degree wrist stretched by a factor of three.
SHOULDER_HANDOVER = (0.48, 0.38, 0.28, 0.18, 0.10, 0.04)
# The smallest step, in shoulder radii, that a draped column of the armpit may take between the
# socket and the first plain ring (see `arm`). How far the socket reaches down the arm depends on
# the arm's angle — measured: -13mm at 10 degrees, +2 at 30, +13 at 45, +23 at 60 — so a fixed
# count of draped rings that is right at 45 squeezes the bottom column into 1.4mm steps at 60,
# and those folded in the bend test. `skin_wrap` builds this body at whatever angle a model's
# arms hang, so the count is worked out per build: at 45 it is four.
SHOULDER_DRAPE_STEP = 0.064
# The hip's is per COLUMN, not per ring: this many thigh rings hand over evenly from whatever the
# hole above each column carries to all-thigh — see `drape`. Stage 1 had one spine share per ring
# (0.35, 0.18, 0.07) over three rings.
HIP_HANDOVER = 5
# How much of the pelvis's opening the thigh's first rings keep: 0 is the hole's own outline,
# 1 is a circle. See `leg`.
HIP_FLOW = (0.30, 0.62, 0.86)
SHOULDER_FLOW = (0.15, 0.38, 0.58, 0.76, 0.90)
ANKLE_HANDOVER = (0.45, 0.22, 0.08)
WRIST_HANDOVER = (0.45, 0.22, 0.08)
# The neck, ring by ring, with the head taking a share well below the skull. A big head on a
# short neck nods through a 32-degree arc whose radius is twice the neck's own length, so the
# skin has to come from somewhere; spread down the neck it is a stylised stretch, and confined to
# the top ring it is a tear.
NECK_WEIGHTS = ({"spine": 0.34, "neck": 0.44, "head": 0.22},
                {"spine": 0.18, "neck": 0.46, "head": 0.36},
                {"spine": 0.06, "neck": 0.44, "head": 0.50},
                {"neck": 0.36, "head": 0.64},
                {"neck": 0.24, "head": 0.76},
                {"neck": 0.14, "head": 0.86})
FINGER_RINGS = [0.18, 0.42, 0.68, 0.92]
THUMB_RINGS = [0.30, 0.56, 0.82, 1.0]
# The front of the foot, socket (0) to the line where the toes leave it (1), in units of the
# distance between the two. The last three rings turn the foot's D section into a flat strip of
# five cells, and five toes leave those cells.
TOE_RINGS = [0.18, 0.40, 0.62, 0.80, 0.92, 1.0]
# The toes, outermost first: each one's share of the width of the strip at the toe line, and its
# length beyond it as a fraction of `foot`. The big toe is the widest and the longest, which is
# also what slants the line of the toe tips back toward the outside of the foot.
TOE_SHARES = (0.15, 0.16, 0.18, 0.21, 0.30)
TOE_LENGTHS = (0.075, 0.085, 0.095, 0.105, 0.12)
TOE_STUB_RINGS = (0.45, 0.85)
# Below the ankle, in shin-length units. Five, not four: the socket the foot leaves is three bands
# tall and still has to leave the top one (0.04) whole, because that ring is one of the ankle's
# certified loops.
SOLE_RINGS = [-0.04, -0.09, -0.14, -0.19, -0.25]
# How the back and the front of those rings grow, as a fraction of the way from the ankle's own
# radius to `heel` and to `instep`, against u (0 at the ankle, 1 on the floor). The heel bulges
# out behind the Achilles line and rounds back under; the instep keeps going.
HEEL = [(0.0, 0.0), (0.35, 0.80), (0.60, 1.0), (0.80, 0.96), (1.0, 0.72)]
INSTEP = [(0.0, 0.0), (0.40, 0.75), (0.70, 0.97), (1.0, 1.0)]
# Along the front of the foot, socket (0) to toe line (1): half width as a multiple of
# `foot_half`, THICKNESS (top above underside) as a multiple of `toe_depth`, and how far the
# underside sits off the floor as a multiple of `sole_z`. The ball is the widest and lowest point.
# Thickness and not height off the floor: as a height, the toe lift ate it, and the first try's
# last ring came out 2.6mm thick — a pointed slipper with a 0.3mm edge.
FOOT_WIDTH = [(0.0, 0.74), (0.18, 0.82), (0.40, 0.90), (0.62, 0.95), (0.80, 0.95),
              (0.92, 0.92)]                  # and at the toe line itself, `toe_half`
FOOT_HEIGHT = [(0.18, 1.48), (0.40, 1.32), (0.62, 1.12), (0.80, 0.96), (0.92, 0.84),
               (1.0, 0.74)]
FOOT_LOW = [(0.0, 1.0), (1.0, 1.0)]     # one sole plane, heel to ball: see `sole_z`


def blend(a, b, w):
    """Weights for a ring that sits part way across a joint. w = 0 is all `a`, 1 is all `b`."""
    out = {}
    if 1.0 - w > 1e-9:
        out[a] = 1.0 - w
    if w > 1e-9:
        out[b] = w
    return out


# --- cross sections ----------------------------------------------------------------------------

def ellipse(n, rx, ry, phase=0.5, power=2.0):
    """n points on an ellipse, or a superellipse when `power` > 2 (a slightly boxy torso).

    `phase` of 0.5 puts the first point half a step round, so point 0 of a 16-ring sits in the
    +u +v quadrant — which is where point 0 of the palm's rectangle sits. The palm is the same
    tube as the forearm, so the two parametrisations have to agree or the wrist twists.
    """
    pts = []
    for j in range(n):
        t = 2 * math.pi * (j + phase) / n
        c, s = math.cos(t), math.sin(t)
        if power != 2.0:
            e = 2.0 / power
            c = math.copysign(abs(c) ** e, c)
            s = math.copysign(abs(s) ** e, s)
        pts.append((rx * c, ry * s))
    return pts


def slab(rows, half_depth):
    """A flat cross section: `rows` across the top, the same back along the bottom.

    `rows` is a list of u positions, from +u to -u. With eight of them this is the boundary of a
    7 x 1 grid, and those seven cells across the knuckles are the four fingers and three webs.
    """
    return ([(u, half_depth) for u in rows]
            + [(u, -half_depth) for u in reversed(rows)])


def knuckle_rows(p):
    """The eight u positions across the knuckles: four finger columns, three webs between.

    Unequal on purpose. Seven equal columns would give a hand with seven identical teeth; the
    fingers want most of the width and the webs want a crack.
    """
    width = p.palm_knuckle_half * 2
    web = p.finger_web
    finger = (width - 3 * web) / 4
    rows, u = [], width / 2
    for i in range(4):
        rows.append(u)
        u -= finger
        rows.append(u)
        u -= web
    return rows[:8]


def lerp_section(a, b, w):
    return [(ax + (bx - ax) * w, ay + (by - ay) * w) for (ax, ay), (bx, by) in zip(a, b)]


def profile(table, t):
    """Piecewise-linear lookup over [(t, value), ...], for radii down a limb."""
    if t <= table[0][0]:
        return table[0][1]
    for (t0, v0), (t1, v1) in zip(table, table[1:]):
        if t <= t1:
            return v0 + (v1 - v0) * (t - t0) / (t1 - t0)
    return table[-1][1]


def frame(axis, first, second):
    """An orthonormal (e1, e2) across a limb, from two world references rather than one.

    Every ring on a limb uses the same two references, so the frame cannot creep round the limb
    and put a twist in the tube — which is what happens if each ring is framed off the last one.

    Both axes are named rather than one being `axis.cross(e1)`, and that is a fix and not a
    flourish. A cross product does not mirror: the left leg's `axis` is the right leg's mirrored,
    so `axis.cross(e1)` came out as MINUS the mirror of the right leg's second axis. The two legs
    were then built in frames of opposite handedness, and since the sole's rings were framed on
    the world axes instead, the band joining the right leg to its own foot was bridged vertex-for
    -vertex between two rings wound in opposite directions — a bow tie through the ankle, on one
    side only. It cost nothing in the face counts and it would have shown up as a mangled right
    ankle in a render. Naming both axes in world terms makes left and right exact mirrors.
    """
    d = Vector(axis).normalized()
    e1 = Vector(first) - d * Vector(first).dot(d)
    if e1.length < 1e-6:
        raise SystemExit(f"the frame's first reference {tuple(first)} is along the limb")
    e1.normalize()
    e2 = Vector(second) - d * Vector(second).dot(d)
    e2 = e2 - e1 * e2.dot(e1)
    if e2.length < 1e-6:
        raise SystemExit(f"the frame's two references {tuple(first)}, {tuple(second)} agree")
    return e1, e2.normalized()


# --- the builder -------------------------------------------------------------------------------

class Build:
    """A bmesh plus the bone weights of every vertex in it, recorded as the vertex is made."""

    def __init__(self):
        self.bm = bmesh.new()
        self.weights = {}
        self.loops = []            # (label, [verts]) — for reporting, never for verification
        self.orphans = set()       # vertices a socket swallows; see `socket` and `object`

    # -- vertices and faces
    def vert(self, co, w):
        v = self.bm.verts.new(Vector(co))
        total = sum(w.values())
        self.weights[v] = {b: x / total for b, x in w.items()} if total else {}
        return v

    def ring(self, centre, e1, e2, section, w, label=None):
        verts = [self.vert(Vector(centre) + Vector(e1) * a + Vector(e2) * b, w)
                 for a, b in section]
        self.loops.append((label or "", verts))
        return verts

    def face(self, verts):
        if len(set(verts)) != 4:
            raise SystemExit(f"a face with a repeated corner: {verts}")
        return self.bm.faces.new(verts)

    def band(self, a, b, skip=()):
        """Bridge two equal-length rings into a band of quads. `skip` omits columns (a socket)."""
        if len(a) != len(b):
            raise SystemExit(f"cannot bridge {len(a)} vertices to {len(b)}")
        n = len(a)
        for j in range(n):
            if j in skip:
                continue
            self.face([a[j], a[(j + 1) % n], b[(j + 1) % n], b[j]])

    def tube(self, rings, skip=()):
        for i, (a, b) in enumerate(zip(rings, rings[1:])):
            self.band(a, b, skip={j for (ri, j) in skip if ri == i})

    def mix(self, vert, extra):
        """Add bone shares to a vertex, scaling what it already owns to make room.

        Used where two parts of the body meet and the handover has to happen over a few
        centimetres of skin rather than across one edge. A shoulder is the case that matters: the
        socket's own ring at 55% arm against a neighbouring ring at 100% spine puts the entire
        handover into one band of quads, and that band is the hard seam where an arm looks like it
        steps out of the shirt. Spread over the ring beyond it as well, the deltoid bends instead.
        """
        share = sum(extra.values())
        if share >= 1.0:
            raise SystemExit(f"a mix of {share} leaves nothing of the original weights")
        out = {bone: value * (1.0 - share) for bone, value in self.weights[vert].items()}
        for bone, value in extra.items():
            out[bone] = out.get(bone, 0.0) + value
        self.weights[vert] = {b: v for b, v in out.items() if v > 1e-6}

    def socket(self, rings, i0, nrings, j0, ncols, label=""):
        """Reserve a rectangle of a tube's lattice as a hole for a limb.

        Returns the set of (band, column) cells not to create and the hole's boundary as an
        ordered loop, 2*(nrings + ncols) long — which is where every ring size in this file comes
        from. A ring is built whole, so the lattice points strictly inside the rectangle end up
        belonging to nothing; they are recorded here and removed, by name, in `object`. Recording
        them rather than sweeping up loose vertices at the end is the difference between "this
        vertex is unused because a socket swallowed it" and "something upstream forgot a face".
        """
        n = len(rings[0])
        skip = {(i0 + di, (j0 + dj) % n) for di in range(nrings) for dj in range(ncols)}
        loop = rect_loop(lambda x, y: rings[x][y % n], i0, i0 + nrings, j0, j0 + ncols)
        for di in range(1, nrings):
            for dj in range(1, ncols):
                self.orphans.add(rings[i0 + di][(j0 + dj) % n])
        self.loops.append((label, loop))
        return skip, loop

    def match_order(self, loop, ring):
        """Indices into `ring`, in the order that pairs it with `loop` at least cost.

        Split out of `match` because the hip needs the pairing itself and not just the reordered
        ring: the first rings of the thigh are reshaped toward the vertex of the pelvis each one
        answers to, and that is a question about which vertex meets which.
        """
        n = len(ring)
        best, chosen = None, None
        for flipped in (False, True):
            base = list(range(n))[::-1] if flipped else list(range(n))
            for off in range(n):
                order = base[off:] + base[:off]
                cost = sum((a.co - ring[j].co).length_squared for a, j in zip(loop, order))
                if best is None or cost < best:
                    best, chosen = cost, order
        return chosen

    def match(self, loop, ring):
        """Rotate and if need be reverse `ring` so corresponding vertices are nearest.

        A bridge between two loops only has to decide which vertex meets which, and getting it
        wrong by one puts a half-turn of twist in a shoulder. Rather than reason about the order
        each loop happens to have been built in, every offset and both directions are tried and
        the closest wins. It is 32 dot products and it cannot be wrong by one.
        """
        return [ring[j] for j in self.match_order(loop, ring)]

    # -- caps
    def grid_cap(self, loop, p, q, dome=0.0, out=(0, 0, 1), weights=None, hole=None,
                 skip_cells=(), label=""):
        """Fill a 2*(p+q) loop with a p x q grid of quads, so a tube end closes without an n-gon.

        A pole cap — a fan to one centre vertex — is the usual way to close a tube and it is
        exactly what this mesh may not have: it is a ring of triangles, and under Catmull-Clark a
        pole is where a rounded end goes pointy. A grid has no pole. Interior vertices come from a
        Coons blend of the boundary, so the cap follows whatever shape the ring is, plus `dome`
        along `out` to round it off.

        `hole` is a (x0, x1, y0, y1) rectangle of cells not created; its boundary comes back as an
        ordered loop, which is how the neck leaves the shoulders and how the head takes the neck.
        `skip_cells` is the same trick one cell at a time — the four finger columns of a knuckle.
        """
        if len(loop) != 2 * (p + q):
            raise SystemExit(f"a {p}x{q} cap needs {2 * (p + q)} boundary vertices, got {len(loop)}")
        g = {}
        for x in range(p + 1):
            g[(x, 0)] = loop[x]
            g[(x, q)] = loop[2 * p + q - x]
        for y in range(q + 1):
            g[(p, y)] = loop[p + y]
            g[(0, y)] = loop[(2 * (p + q) - y) % (2 * (p + q))]
        out = Vector(out).normalized()
        skip_cells = set(skip_cells)
        cells = [(x, y) for x in range(p) for y in range(q)
                 if (x, y) not in skip_cells
                 and not (hole and hole[0] <= x < hole[1] and hole[2] <= y < hole[3])]
        # Only the lattice points some surviving cell actually uses get made. Creating the whole
        # lattice and then skipping cells leaves the middle of a hole behind as loose vertices —
        # four of them inside the neck hole, invisible in a render and fatal to a manifold count.
        needed = {(x + dx, y + dy) for (x, y) in cells for dx in (0, 1) for dy in (0, 1)}
        for (x, y) in sorted(needed):
            if (x, y) in g:
                continue
            u, v = x / p, y / q
            # Coons: the two pairs of boundary curves blended, minus the bilinear corners.
            co = ((1 - v) * g[(x, 0)].co + v * g[(x, q)].co
                  + (1 - u) * g[(0, y)].co + u * g[(p, y)].co
                  - ((1 - u) * (1 - v) * g[(0, 0)].co + u * (1 - v) * g[(p, 0)].co
                     + (1 - u) * v * g[(0, q)].co + u * v * g[(p, q)].co))
            co = co + out * dome * math.sin(math.pi * u) * math.sin(math.pi * v)
            g[(x, y)] = self.vert(co, weights or {})
        for (x, y) in cells:
            self.face([g[(x, y)], g[(x + 1, y)], g[(x + 1, y + 1)], g[(x, y + 1)]])
        if hole:
            ring = rect_loop(lambda x, y: g[(x, y)], *hole)
            self.loops.append((label, ring))
            return g, ring
        return g, None

    # -- output
    def object(self, name, props):
        bm = self.bm
        loose = {v for v in bm.verts if not v.link_faces}
        if loose != self.orphans:
            extra = sorted(tuple(round(c, 4) for c in v.co) for v in loose - self.orphans)
            missing = len(self.orphans - loose)
            raise SystemExit(
                f"{len(extra)} vertices belong to no face and no socket accounts for them "
                f"(first few: {extra[:6]}); {missing} socket interiors unexpectedly kept a face")
        if self.orphans:
            bmesh.ops.delete(bm, geom=list(self.orphans), context="VERTS")
            for v in self.orphans:
                self.weights.pop(v, None)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
        bm.verts.index_update()
        by_index = {v.index: w for v, w in self.weights.items()}
        mesh = bpy.data.meshes.new(name)
        bm.to_mesh(mesh)
        mesh.update()
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.collection.objects.link(obj)
        groups = {}
        for bone in make_character.BONES:
            groups[bone] = obj.vertex_groups.new(name=bone)
        unweighted = 0
        for index in range(len(mesh.vertices)):
            w = by_index.get(index) or {}
            if not w:
                unweighted += 1
            for bone, value in w.items():
                if bone not in groups:
                    raise SystemExit(f"{bone!r} is not a bone in make_character.BONES")
                groups[bone].add([index], value, "REPLACE")
        if unweighted:
            raise SystemExit(f"{unweighted} vertices were created without a bone")
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.shade_smooth()
        return obj


def flow_out(b, loop, rings, order, axis, centres, radii, blends):
    """Reshape a limb's first rings toward the hole it grows out of.

    A limb bridged from a hole in the body onto a round ring puts every bit of the difference
    between the two shapes into one band of quads. At the hip that band was the hard seam that
    made the figure look like it was wearing a nappy. At the shoulder it was worse than cosmetic:
    the arm's first ring sits at the height of the socket's lower edge, and once the two shapes
    disagree enough the band's quads cross over themselves — four BOW TIES in the armpit of the
    rest mesh, which `topology` now counts and refuses.

    So each early ring keeps the hole's own outline, scaled to that ring's radius, and relaxes
    into a circle over the next few. `order` is the pairing from `Build.match_order`, so each
    ring vertex is shaped toward the vertex of the hole it actually answers to.
    """
    for k, relax in enumerate(blends):
        centre = centres[k]
        perps = []
        for vert in loop:
            d = vert.co - centre
            perps.append(d - axis * d.dot(axis))
        average = sum(v.length for v in perps) / len(perps)
        scale = radii[k] / max(average, 1e-9)
        for i, j in enumerate(order):
            vert = rings[k][j]
            shaped = centre + perps[i] * scale
            vert.co = shaped.lerp(vert.co, relax)


def drape(b, loop, rings, order, origin, axis, count, bone=None, handover=0):
    """Hang a limb's first `count` rings off the hole below it, so the skin only ever goes DOWN.

    `flow_out` shapes those rings across the limb and leaves each one flat, at a single height
    along the axis. The hip's hole is not flat — it is a saddle, high at the outside of the hip and
    low at the crotch — so a flat ring that is below the crotch is a long way below the outside of
    the hip, and a flat ring that is anywhere near the outside of the hip is ABOVE the crotch. Stage
    1's first two thigh rings were above the whole hole: the thigh rose up inside the pelvis and
    came back out, a fold in the rest mesh that rendered as the hem of a pair of shorts. No face
    in it crossed itself, so the topology count was clean, and two of the four loops the audit
    certified at the hip were inside that fold.

    So each column of each draped ring is placed between the hole vertex it answers to and the
    first flat ring, at an even fraction of the way down. Every column then descends in `count +
    1` equal steps from the hole to the first flat ring — a fold-back is impossible by
    construction, and `inward_faces` measures it on the finished mesh anyway. Moves are along the
    axis only, so the cross-section `flow_out` made is kept.

    The WEIGHTS are handed over column by column too, from whatever the hole vertex above carries
    to all-`bone`, in even steps over the first `handover` rings — more rings than the drape
    moves. Stage 1 gave every vertex of a ring the same share, so the back of the hip, where the
    hole is mostly spine, stepped from 19% thigh to 65% across one edge; once the fold that had
    been hiding slack there was gone, a 60 degree flexion stretched that edge 2.2 times. Handing
    over per column inside the drape alone only moved the jump two rings down and folded a face
    there: the back of a hip has to lengthen by about r x 60 degrees, and that wants more than
    four centimetres of skin to come from.

    `handover=0` moves positions only. The shoulder uses it that way: its weights already hand
    over from collarbone to arm across six rings (SHOULDER_HANDOVER) and the torso round the
    socket, and its bend was tuned on those.
    """
    if count <= 0 and handover <= 0:
        return
    offsets = [(v.co - origin).dot(axis) for v in loop]
    first_flat = sum((v.co - origin).dot(axis) for v in rings[count]) / len(rings[count])
    if first_flat <= max(offsets) + 1e-4:
        raise SystemExit(
            f"the first flat leg ring ({first_flat:.4f} down the axis) is not below the lowest "
            f"point of the hole it hangs from ({max(offsets):.4f}); a drape would fold back")
    for k in range(count):
        share = (k + 1) / (count + 1)
        for i, j in enumerate(order):
            vert = rings[k][j]
            now = (vert.co - origin).dot(axis)
            want = offsets[i] + (first_flat - offsets[i]) * share
            vert.co = vert.co + axis * (want - now)
    for k in range(handover):
        share = (k + 1) / (handover + 1)
        for i, j in enumerate(order):
            above = b.weights[loop[i]]
            weights = {name: value * (1.0 - share) for name, value in above.items()}
            weights[bone] = weights.get(bone, 0.0) + share
            b.weights[rings[k][j]] = weights


def rect_loop(get, x0, x1, y0, y1):
    """The boundary of a lattice rectangle, in order, as a cycle. 2*((x1-x0) + (y1-y0)) long."""
    loop = [get(x, y0) for x in range(x0, x1)]
    loop += [get(x1, y) for y in range(y0, y1)]
    loop += [get(x, y1) for x in range(x1, x0, -1)]
    loop += [get(x0, y) for y in range(y1, y0, -1)]
    return loop


# --- the body ----------------------------------------------------------------------------------

def torso_heights(p, at):
    """The height of every torso ring: the hips by the torso's own length, the arm sockets by the
    shoulder joint, and the three between spaced evenly. See SOCKET_LEVELS."""
    lower = [p.leg + t * p.torso for t in TORSO_RINGS[:4]]
    shoulder = at["shoulder.right"].z
    upper = [shoulder + k * p.shoulder_r for k in SOCKET_LEVELS]
    gap = len(TORSO_RINGS) - len(lower) - len(upper)
    middle = [lower[-1] + (upper[0] - lower[-1]) * (n + 1) / (gap + 1) for n in range(gap)]
    heights = lower + middle + upper
    if any(b <= a for a, b in zip(heights, heights[1:])):
        raise SystemExit(f"torso rings out of order: {[round(h, 4) for h in heights]} — the "
                         "shoulder joint is too low for the hips under it")
    return heights


def torso(b, p, at):
    """The trunk: a tube of 20-rings from the crotch to the top of the shoulders.

    The bottom ring is a saddle, not a circle on a plane: it lifts by `hip_rise` at the sides and
    drops to the crotch front and back. That is what the top of a leg looks like, and it also
    puts the ring the thigh grows out of level with the hip joint instead of a hand's width below
    it, which is the difference between three loops near the hip and three loops under it.
    """
    rings, sections = [], []
    for t in TORSO_RINGS:                                    # 0 at the hip, 1 at the chest
        # Breakpoints at 0.36 and 0.77 rather than stage 1's 0.33 and 0.75: those were in units of
        # the shorter torso, and these put the waist and the chest at the same HEIGHT as before.
        half = profile([(-0.2, p.pelvis_half), (0.0, p.pelvis_half), (0.36, p.waist_half),
                        (0.77, p.chest_half), (1.0, p.chest_half * 0.92)], t)
        depth = profile([(-0.2, p.pelvis_depth), (0.0, p.pelvis_depth), (0.36, p.waist_depth),
                         (0.77, p.chest_depth), (1.0, p.chest_depth * 0.95)], t)
        # Below the arm sockets the front and the back are shaped separately: the seat bulges
        # BEHIND and nothing else moves. Stage 1 slid the whole ring backward at the seat instead,
        # which pulled the belly in by the same amount — a body whose front went concave just
        # above the hips and then stuck out again at the pelvis, as a lip over the thighs. A ring
        # that only grows backward cannot do that.
        seat = profile([(-0.2, p.seat * 0.45), (0.10, p.seat), (0.30, p.seat * 0.55),
                        (0.50, p.seat * 0.10), (0.62, 0.0)], t)
        # The chest is still SLID forward, as stage 1 did it, and that is not an oversight. The
        # arm sockets are cut out of these rings, and sliding the ring is what puts the socket a
        # centimetre and a half in front of the shoulder joint. Bulging only the front half
        # instead (the first try) left the socket on the joint's line, and the arm's first ring
        # then crossed the socket's lower corners: two bow-tie quads in the base cage and four in
        # the subdivided one, at the armpit.
        shift = profile([(0.54, 0.0), (0.816, p.chest_set), (1.0, p.chest_set * 0.5)], t)
        sections.append([(a, c * (depth if c > 0 else depth + seat) + shift)
                         for a, c in ellipse(TORSO, half, 1.0, phase=0.0, power=2.0)])
    for i, (t, section, z) in enumerate(zip(TORSO_RINGS, sections, torso_heights(p, at))):
        if i < len(HIP_SADDLE):
            # The hip saddle, per column. The bottom ring lifts by `hip_rise` at +-X and drops to
            # the crotch on the centre line, and the two rings above it carry a fading share of
            # the same lift, which is what a pelvis looks like from the side.
            #
            # The fade is not decoration. With the lift on the bottom ring alone, its lateral
            # vertices rose to within 4mm of the ring above — a seam of near-zero-length edges
            # between a ring that is 43% thigh and a ring that is 100% spine, and a 60 degree hip
            # flexion stretched those 4mm edges by a factor of six. Spreading the lift spreads the
            # gap; spreading the thigh's share over the same three rings spreads the weight.
            verts = []
            for j, (a, c) in enumerate(section):
                theta = 2 * math.pi * j / TORSO
                lateral = abs(math.cos(theta)) ** 1.4
                rise = p.hip_rise * HIP_SADDLE[i] * lateral
                drop = p.crotch_drop * HIP_SADDLE[i]
                share = HIP_SHARE[i]
                near = share * lateral + share * 0.18 * (1 - lateral)
                far = share * 0.18 * (1 - lateral)
                if math.cos(theta) > 1e-9:
                    mine, theirs = "thigh.right", "thigh.left"
                elif math.cos(theta) < -1e-9:
                    mine, theirs = "thigh.left", "thigh.right"
                else:
                    # the centre line, front and back: both legs pull on it, so both own it
                    mine, theirs = "thigh.right", "thigh.left"
                    near = far = share * 0.45
                w = {"spine": 1.0 - near - far, mine: near, theirs: far}
                verts.append(b.vert((a, c, z - drop + rise), w))
            b.loops.append((f"pelvis{i}", verts))
            rings.append(verts)
            continue
        w = {"spine": 1.0} if t < 0.78 else blend("spine", "neck", min(0.45, (t - 0.78) * 1.4))
        rings.append(b.ring((0, 0, z), (1, 0, 0), (0, 1, 0), section, w, label=f"torso{i}"))

    skip, sockets = set(), {}
    # Where each socket starts, in lattice columns. The left one is the RIGHT one's mirror image
    # rather than the same arithmetic about column 10, and with an odd number of socket columns
    # those are not the same thing: five cells centred on cell 0 reach from column -2 to column 3,
    # whose mirror is column 7 to column 12 — not 8 to 13. One column of each shoulder was
    # weighted as an armpit on one side of the body and as a chest on the other, a 0.65 weight
    # difference between two vertices that are mirror images to within 1e-7 of a millimetre.
    right_start = -(SOCKET_COLS // 2)
    for side, c0 in (("right", right_start),
                     ("left", TORSO // 2 - right_start - SOCKET_COLS)):
        cells, sockets[side] = b.socket(rings, SOCKET_FROM, SOCKET_RINGS, c0, SOCKET_COLS,
                                        label=f"socket.{side}")
        skip |= cells
        # One lattice ring beyond the socket, given a share of the arm so the handover from torso
        # to limb happens across three rings of skin instead of one band of quads.
        for i in range(SOCKET_FROM - 1, SOCKET_FROM + SOCKET_RINGS + 2):
            for j in range(c0 - 1, c0 + SOCKET_COLS + 2):
                if not 0 <= i < len(rings):
                    continue
                if SOCKET_FROM <= i <= SOCKET_FROM + SOCKET_RINGS and c0 <= j <= c0 + SOCKET_COLS:
                    continue                      # the socket and its own boundary
                b.mix(rings[i][j % TORSO],
                      {f"clavicle.{side}": 0.15, f"upper_arm.{side}": 0.20})
    b.tube(rings, skip=skip)

    # Round the top edge of the trunk (TRUNK_ROUND): each vertex of the top ring moves part of the
    # way toward the neck hole's outline at its own angle, level, so the edge leans in instead of
    # standing straight up off the ring below it. The ring below is the arm socket's top edge and
    # does not move; this ring is in no socket and no limb is bridged to it.
    hole_x, hole_y = (p.neck_r * f for f in NECK_HOLE)
    for v in rings[-1]:
        angle = math.atan2(v.co.y, v.co.x)
        c, s = math.cos(angle), math.sin(angle)
        reach = hole_x * hole_y / math.sqrt((hole_y * c) ** 2 + (hole_x * s) ** 2)
        target = Vector((reach * c, reach * s, v.co.z))
        v.co = v.co.lerp(target, TRUNK_ROUND)

    # The shoulder yoke: a 5 x 5 grid over the top ring with its middle 3 x 3 left out for the
    # neck. The grid's corners land on the ring's +X, +Y, -X and -Y vertices, so the lattice sits
    # diagonally on the body: the +X corner is (0, 0) and the -X corner is (5, 5), which makes
    # x + y the way across the shoulders. Which half of the yoke follows which collarbone is then
    # index arithmetic and not a measurement of where a vertex happens to have ended up.
    # A lower dome than stage 1's 0.014: with the shoulders raised, a high yoke left the neck 1cm
    # long between it and the chin, and the old character's is about 2.
    grid, neck_hole = b.grid_cap(rings[-1], 5, 5, dome=p.height * 0.007, out=(0, 0, 1),
                                 weights={"spine": 0.44, "neck": 0.33, "head": 0.23},
                                 hole=(1, 4, 1, 4),
                                 label="neck hole")
    # The yoke's outer edge follows the collarbone, and that share is set on the TOP RING by
    # column angle rather than on the cap's lattice. The lattice version was wrong twice over:
    # every interior point of a 5 x 5 cap with a 3 x 3 hole in it is on the hole's boundary, so
    # the clause could only ever fire on the four points its own bounds happened to miss — and
    # the cap's lattice is diagonal, so those four are not a mirror-symmetric set. It gave two
    # vertices that are mirror images to 1e-7 a 0.5 difference in weight.
    # The hole the neck leaves through is placed, not inherited. Its twelve vertices come out of
    # the cap's Coons blend, and on a lattice whose corners sit on the ring's own +X/+Y/-X/-Y that
    # blend bunches them towards the corners: 3mm edges beside 30mm ones and a quad of the yoke
    # bent 143 degrees, which Catmull-Clark then turned into a crossed face. Re-placed as an even
    # ring — in the order and direction the hole already runs, and on a grid of angles that is
    # symmetric about the centre line so the two shoulders stay mirror images.
    #
    # Task 5c: on a grid with its CORNERS ON THE AXES. Stages 2-2b offset it by half a step (15,
    # 45, 75...) "so the set mirrors" — and the set of positions did mirror, so `mirror_error`
    # passed, but the hole's four lattice corners (vertices 0, 3, 6, 9: diagonally in from the
    # cap's corners, which are the top ring's +X/+Y/-X/-Y vertices) landed 15 degrees round from
    # where the lattice puts them. Which hole vertex joins which top-ring vertex then did NOT
    # mirror: the corner cell behind the neck took the hole vertex at -76 degrees instead of -90,
    # the cell beside it was sheared a third of the way round the neck into a dart (122.8 degrees
    # across one diagonal, 143 once the trunk's top edge was rounded), and the band up to the
    # neck's first ring, which sits at 0, 30, 60..., had a 15 degree twist in it. On the axes,
    # the cells mirror, the dart is gone and every neck column runs straight up out of the hole.
    angles = [math.atan2(v.co.y, v.co.x) for v in neck_hole]
    step = 2 * math.pi / len(neck_hole)
    forward = ((angles[1] - angles[0] + math.pi) % (2 * math.pi)) - math.pi
    step = step if forward > 0 else -step
    base = round(angles[0] / abs(step)) * abs(step)
    corner = (base / (math.pi / 2)) % 1.0
    if min(corner, 1.0 - corner) > 1e-6:
        raise SystemExit(f"the neck hole's first vertex came out at {math.degrees(angles[0]):.1f} "
                         "degrees: its lattice corners would not sit on the axes")
    hole_z = sum(v.co.z for v in neck_hole) / len(neck_hole)
    for i, v in enumerate(neck_hole):
        angle = base + step * i
        v.co = Vector((math.cos(angle) * hole_x, math.sin(angle) * hole_y, hole_z))

    for j, v in enumerate(rings[-1]):
        theta = 2 * math.pi * j / TORSO
        lateral = abs(math.cos(theta)) ** 1.5
        if lateral < 1e-6:
            continue
        side = "clavicle.right" if math.cos(theta) > 0 else "clavicle.left"
        b.mix(v, {side: 0.40 * lateral})
    return rings, sockets, neck_hole


def neck_and_head(b, p, at, hole):
    """A 12-ring neck out of the shoulder cap and into a hole in the bottom of a quad sphere."""
    top = max(v.co.z for v in hole)
    joint = at["neck"].z                       # the base of the skull: where the head turns
    ceiling = joint + p.head_r * 0.06          # the last ring, just inside the chin hole
    hole_x, hole_y = (p.neck_r * f for f in NECK_HOLE)

    def upright(t):
        """The neck's own line at t: (half width, half depth, height)."""
        r = p.neck_r * profile([(0.0, 1.08), (0.5, 1.0), (1.0, 1.02)], t)
        return r, r * 1.10, top + t * (ceiling - top)

    # The first rings sit on a quarter-ellipse from the hole to the first upright ring
    # (NECK_FILLET). Every point here is on an ellipse at the same parameter as the hole's and
    # the upright ring's, so the fillet ring is one too, and mirror-symmetric like them.
    end_x, end_y, end_z = upright(NECK_RINGS[len(NECK_FILLET)])
    rings = []
    for k, t in enumerate(NECK_RINGS):
        rx, ry, z = upright(t)
        if k < len(NECK_FILLET):
            phi = math.radians(NECK_FILLET[k])
            out, up = 1.0 - math.cos(phi), 1.0 - math.sin(phi)
            rx = end_x + (hole_x - end_x) * out
            ry = end_y + (hole_y - end_y) * out
            z = top + (end_z - top) * up
        w = dict(NECK_WEIGHTS[min(k, len(NECK_WEIGHTS) - 1)])
        rings.append(b.ring((0, 0, z), (1, 0, 0), (0, 1, 0),
                            ellipse(NECK, rx, ry, phase=0.0), w, label=f"neck{k}"))
    b.band(hole, b.match(hole, rings[0]))
    b.tube(rings)

    # The head: a cube sphere, so there is no pole at the crown and no triangle anywhere.
    k = HEAD_CELLS
    centre = at["head"]
    lattice = {}
    hole_from = (k - 3) // 2                 # the 3 x 3 chin hole, centred whatever k is

    def head_vert(key):
        if key in lattice:
            return lattice[key]
        d = Vector(key).normalized()
        co = centre + Vector((d.x * p.head_r, d.y * p.head_depth, d.z * p.head_r))
        lattice[key] = b.vert(co, {"head": 1.0})
        return lattice[key]

    def key_of(face, i, j):
        u, v = -1 + 2 * i / k, -1 + 2 * j / k
        return {"+z": (u, v, 1.0), "-z": (u, v, -1.0), "+x": (1.0, u, v), "-x": (-1.0, u, v),
                "+y": (u, 1.0, v), "-y": (u, -1.0, v)}[face]

    for face in ("+z", "-z", "+x", "-x", "+y", "-y"):
        for i in range(k):
            for j in range(k):
                # the middle 3 x 3 of the bottom face is the chin hole, and its four innermost
                # lattice points are never asked for, so they are never made
                if face == "-z" and hole_from <= i < hole_from + 3 and hole_from <= j < hole_from + 3:
                    continue
                b.face([head_vert(key_of(face, i, j)), head_vert(key_of(face, i + 1, j)),
                        head_vert(key_of(face, i + 1, j + 1)), head_vert(key_of(face, i, j + 1))])
    chin = rect_loop(lambda x, y: head_vert(key_of("-z", x, y)),
                     hole_from, hole_from + 3, hole_from, hole_from + 3)
    b.loops.append(("chin hole", chin))
    for v in chin:
        b.weights[v] = {"neck": 0.3, "head": 0.7}
    b.band(chin, b.match(chin, rings[-1]))


def arm(b, p, at, side, socket):
    """Shoulder to fingertip as ONE tube, because a wrist is not a join.

    The palm's cross section is the forearm's 16-ring flattened; the knuckles are that same ring
    with its two long sides spread into the seven columns the fingers come out of. Carrying one
    tube all the way means the only bridge in the whole arm is the one onto the torso's socket,
    and a bridge is the one place a loft can go wrong.
    """
    sign = 1.0 if side == "right" else -1.0
    shoulder, elbow, wrist = at[f"shoulder.{side}"], at[f"elbow.{side}"], at[f"wrist.{side}"]
    axis = (elbow - shoulder).normalized()
    e1, e2 = frame(axis, (0, 1, 0), (sign, 0, 0))   # across the hand's width, then its thickness
    upper, fore = p.upper_arm, p.forearm
    # A limb that is a cylinder reads as a mannequin however good its topology is: these are the
    # bumps that make it read as an arm — a bicep, the swell of the forearm just past the elbow,
    # and a wrist thin enough that the hand is clearly a separate thing on the end of it.
    #
    # There is deliberately NO bump at t = 0. The deltoid is the one piece of shaping this arm
    # does not get, because the first rings are the socket's own outline (SHOULDER_FLOW) and
    # growing them pushes the armpit into the socket's lower corner: a 3mm shoulder bump put two
    # crossed quads into the rest mesh and two more into the subdivided one. The deltoid belongs
    # to the torso side of that junction and can come from a shape key later.
    radii = [(0.0, p.shoulder_r), (0.35, p.bicep_r), (0.62, p.bicep_r * 0.92),
             (0.8, p.elbow_r * 1.06), (1.0, p.elbow_r), (1.22, p.forearm_r),
             (1.55, p.forearm_r * 0.84), (2.0, p.wrist_r)]
    rings = []
    for t in ARM_RINGS:
        if t <= 1.0:
            centre = shoulder + axis * (t * upper)
            w = ({"upper_arm." + side: 1.0} if t < 0.78
                 else blend(f"upper_arm.{side}", f"forearm.{side}", (t - 0.78) * 2.27))
        else:
            centre = elbow + axis * ((t - 1.0) * fore)
            w = (blend(f"upper_arm.{side}", f"forearm.{side}", min(1.0, 0.5 + (t - 1.0) * 2.27))
                 if t < 1.22 else {f"forearm.{side}": 1.0})
            if t > 1.86:
                w = blend(f"forearm.{side}", f"hand.{side}", (t - 1.86) * 2.8)
        r = profile(radii, t)
        rings.append(b.ring(centre, e1, e2, ellipse(ARM, r, r * 1.06), w, label=f"arm.{side}{t}"))
    order = b.match_order(socket, rings[0])
    b.band(socket, [rings[0][j] for j in order])
    flow_out(b, socket, rings, order, axis,
             [shoulder + axis * (ARM_RINGS[k] * upper) for k in range(len(SHOULDER_FLOW))],
             [profile(radii, ARM_RINGS[k]) for k in range(len(SHOULDER_FLOW))], SHOULDER_FLOW)
    # The armpit had the hip's fold. The socket is a patch of the torso's side wall and the arm
    # leaves it at 45 degrees, so the socket's lower edge sits further DOWN the arm than the arm's
    # first three rings: under the arm the skin ran out to the socket's lower edge and back up to
    # the first ring before setting off along the arm. Seven faces each side pointed into the arm,
    # unchanged since stage 1, and an armpit close-up rendered clean the whole time. Same cure as
    # the hip: hang the first rings between the socket and the first plain ring, column by column.
    reach = max((v.co - shoulder).dot(axis) for v in socket)
    step = p.shoulder_r * SHOULDER_DRAPE_STEP
    count = next(k for k, t in enumerate(ARM_RINGS) if t * upper >= reach + step * (k + 1))
    drape(b, socket, rings, order, shoulder, axis, count)
    for v in socket:
        b.weights[v] = blend(f"clavicle.{side}", f"upper_arm.{side}", 0.55)
    for k, share in enumerate(SHOULDER_HANDOVER):
        for v in rings[k]:
            b.mix(v, {f"clavicle.{side}": share})

    # the palm: the same ring, flattened and spread into knuckle columns
    knuckles = slab(knuckle_rows(p), p.palm_knuckle_depth)
    palm = []
    for s in PALM_RINGS:
        centre = wrist + axis * (s * p.palm_len)
        # The palm's first ring IS the forearm's, and it flattens from there. Started from a
        # palm width of its own, the hand stepped from a 27mm-deep wrist to a 15mm-deep palm over
        # 3mm of length, and the hands read as flippers clipped onto the arms.
        half = p.wrist_r + (p.palm_knuckle_half - p.wrist_r) * s
        depth = p.wrist_r * 1.06 + (p.palm_knuckle_depth - p.wrist_r * 1.06) * s
        section = lerp_section(ellipse(ARM, half, depth), knuckles, s ** 1.5)
        palm.append(b.ring(centre, e1, e2, section, {f"hand.{side}": 1.0},
                           label=f"palm.{side}{s}"))
    for k, share in enumerate(WRIST_HANDOVER):
        for v in palm[k]:
            b.mix(v, {f"forearm.{side}": share})
    # The thumb leaves the radial edge of the palm, straddling column 0 — which is the +u +v
    # corner of the flattened section, so half the socket is on the palm and half on the back of
    # the hand. That is where a thumb grows from; a thumb off the palm face alone reads as a
    # fifth finger pointing the wrong way.
    combined = rings + palm
    cells, thumb_loop = b.socket(combined, len(rings) + 1, 2, -1, 2, label=f"thumb socket.{side}")
    b.tube(combined, skip=cells)

    # four fingers off the knuckle strip, and they are columns 0, 2, 4, 6 of it: the webs are
    # columns 1, 3, 5 and are what keeps four holes four holes instead of one mitt.
    cap, _ = b.grid_cap(palm[-1], 7, 1, weights={f"hand.{side}": 1.0},
                        skip_cells=((0, 0), (2, 0), (4, 0), (6, 0)), label=f"knuckles.{side}")
    for n, (col, length) in enumerate(zip((0, 2, 4, 6), p.finger_len)):
        root = [cap[(col, 0)], cap[(col + 1, 0)], cap[(col + 1, 1)], cap[(col, 1)]]
        base = sum((v.co for v in root), Vector()) / 4
        splay = e1 * (0.10 * (1.5 - n) * p.finger_r) * (1.0 if side == "right" else 1.0)
        fr = p.finger_r
        f_rings = []
        for s in FINGER_RINGS:
            r = fr * profile([(0.0, 1.0), (0.5, 0.96), (1.0, 0.84)], s)
            # a shade deeper than wide, as a finger is at the knuckle, and rounder at the tip
            f_rings.append(b.ring(base + axis * (s * length) + splay * s, e1, e2,
                                  ellipse(FINGER, r * 0.95, r * profile(
                                      [(0.0, 1.18), (1.0, 1.0)], s), phase=0.5),
                                  {f"hand.{side}": 1.0},
                                  label=f"finger{n}.{side}{s}"))
        b.band(root, b.match(root, f_rings[0]))
        b.tube(f_rings)
        b.face(f_rings[-1])          # a four-vertex ring closes with one quad

    # the thumb, off the palm's radial side, swung forward and down toward the fingers
    thumb_axis = (e1 * 0.78 + e2 * 0.34 + axis * 0.52).normalized()
    t1, t2 = frame(thumb_axis, axis, e2)
    base = sum((v.co for v in thumb_loop), Vector()) / len(thumb_loop)
    t_rings = []
    for s in THUMB_RINGS:
        r = p.thumb_r * profile([(0.0, 1.0), (0.5, 0.95), (1.0, 0.82)], s)
        t_rings.append(b.ring(base + thumb_axis * (s * p.thumb_len), t1, t2,
                              ellipse(THUMB, r, r, phase=0.5), {f"hand.{side}": 1.0},
                              label=f"thumb.{side}{s}"))
    b.band(thumb_loop, b.match(thumb_loop, t_rings[0]))
    b.tube(t_rings)
    b.grid_cap(t_rings[-1], 2, 2, dome=p.thumb_r * 0.55, out=thumb_axis,
               weights={f"hand.{side}": 1.0})
    return rings, palm


def leg(b, p, at, side, hip_loop):
    """Hip to toe. The thigh grows out of the saddle the torso's bottom ring makes."""
    hip, knee, ankle = at[f"hip.{side}"], at[f"knee.{side}"], at[f"ankle.{side}"]
    axis = (knee - hip).normalized()
    sign = 1.0 if side == "right" else -1.0
    e1, e2 = frame(axis, (sign, 0, 0), (0, 1, 0))   # outward, then forward
    thigh, shin = p.thigh, p.shin
    # and the same for the leg: the quadriceps, a knee, a calf high on the shin, a thin ankle
    radii = [(0.0, p.hip_r), (0.26, p.thigh_r), (0.60, p.thigh_r * 0.90),
             (0.85, p.knee_r * 1.08), (1.0, p.knee_r), (1.22, p.calf_r),
             (1.55, p.calf_r * 0.82), (1.85, p.ankle_r * 1.12), (2.0, p.ankle_r)]
    rings = []
    for t in LEG_RINGS:
        if t <= 1.0:
            centre = hip + axis * (t * thigh)
            w = ({f"thigh.{side}": 1.0} if t < 0.78
                 else blend(f"thigh.{side}", f"shin.{side}", (t - 0.76) * 2.08))
        else:
            centre = knee + axis * ((t - 1.0) * shin)
            w = (blend(f"thigh.{side}", f"shin.{side}", min(1.0, 0.5 + (t - 1.0) * 2.08))
                 if t < 1.24 else {f"shin.{side}": 1.0})
            if t > 1.88:
                w = blend(f"shin.{side}", f"foot.{side}", (t - 1.88) * 3.0)
        r = profile(radii, t)
        rings.append(b.ring(centre, e1, e2, ellipse(LEG, r, r * 1.10), w, label=f"leg.{side}{t}"))
    order = b.match_order(hip_loop, rings[0])
    b.band(hip_loop, [rings[0][j] for j in order])

    # The top of the thigh is not a circle. It is the pelvis's own opening, shrunk, relaxing into
    # a circle over the next few rings — so the thigh FLOWS OUT of the hip instead of being a
    # cylinder socketed into it. Bridging a saddle-shaped hole straight onto a round ring put all
    # the slack in two quads on the centre line and none at the sides, and that uneven band is
    # the hard seam across the hip: the thing that made the figure look like it was wearing a
    # nappy. Each ring keeps the hole's outline in proportion to how close to the hole it is.
    flow_out(b, hip_loop, rings, order, axis,
             [hip + axis * (LEG_RINGS[k] * thigh) for k in range(len(HIP_FLOW))],
             [profile(radii, LEG_RINGS[k]) for k in range(len(HIP_FLOW))], HIP_FLOW)
    # and hang them below the hole, positions and weights both, so the skin only goes down
    drape(b, hip_loop, rings, order, hip, axis, len(HIP_FLOW), f"thigh.{side}", HIP_HANDOVER)

    # Below the ankle the rings stop being a leg and start being the back half of a foot: a heel
    # that bulges out behind the line of the Achilles and rounds back under, an instep that keeps
    # reaching forward, and a flat bottom ring ON the sole plane. Front and back are shaped
    # separately, because a heel and an instep are not the same shape mirrored.
    sole = []
    for k, t in enumerate(SOLE_RINGS):
        u = -t / abs(SOLE_RINGS[-1])                       # 0 at the ankle, 1 at the sole
        centre_z = ankle.z + (p.sole_z - ankle.z) * u ** 1.25
        centre = Vector((ankle.x, ankle.y, centre_z))
        half = p.ankle_r + (p.foot_half * 0.86 - p.ankle_r) * profile(
            [(0.0, 0.0), (0.5, 0.8), (1.0, 1.0)], u)
        back = p.ankle_r * 1.10 + (p.heel - p.ankle_r * 1.10) * profile(HEEL, u)
        front = p.ankle_r * 1.10 + (p.instep - p.ankle_r * 1.10) * profile(INSTEP, u)
        section = [(a, c * (front if c > 0 else back)) for a, c in ellipse(LEG, half, 1.0)]
        sole.append(b.ring(centre, e1, e2, section, {f"foot.{side}": 1.0},
                           label=f"sole.{side}{t}"))
    for k, share in enumerate(ANKLE_HANDOVER):
        for v in sole[k]:
            b.mix(v, {f"shin.{side}": share})

    # The front of the foot leaves the FRONT of the three lowest bands. With twelve columns at
    # fifteen degrees plus thirty per step, cells 1, 2 and 3 span 45 to 135 degrees — centred on
    # +Y, which is the way the figure faces. Three bands tall rather than stage 1's two, so the top
    # of the foot starts at the instep and slopes down to the toes instead of starting 4cm under
    # the ankle and meeting the shin in a notch.
    combined = rings + sole
    cells, toe_loop = b.socket(combined, len(rings) + 1, 3, 1, 3, label=f"toe socket.{side}")
    b.tube(combined, skip=cells)
    sole_grid, _ = b.grid_cap(sole[-1], 4, 2, dome=0.0, out=(0, 0, -1),
                              weights={f"foot.{side}": 1.0})

    # The front of the foot: a D in cross section — flat underneath, domed on top — that widens to
    # the ball and then flattens into a strip for the toes. Twelve around, phase half a step,
    # which puts four vertices along the top and four along the bottom: the same layout as the
    # 3 x 3 socket it leaves, so the first ring can flow out of the socket's own outline.
    base = sum((v.co for v in toe_loop), Vector()) / len(toe_loop)
    t_axis = Vector((0.0, 1.0, 0.0))
    t1, t2 = frame(t_axis, (sign, 0, 0), (0, 0, 1))     # outward, then up — for either foot
    reach = p.foot * (1.0 - TOE_LENGTHS[-1]) - (base.y - ankle.y)
    t_rings = []
    for s in TOE_RINGS:
        width = profile([(t, p.foot_half * w) for t, w in FOOT_WIDTH] + [(1.0, p.toe_half)], s)
        # the crease under the toe line lifts off the floor and the toes come back down to it
        low = p.sole_z * profile(FOOT_LOW, s) + p.toe_lift * profile([(0.80, 0.0), (1.0, 0.5)], s)
        thickness = p.toe_depth * profile(FOOT_HEIGHT, s)
        top, under = thickness * 0.62, thickness * 0.38
        section = [(a, c * (top if c > 0 else under))
                   for a, c in ellipse(TOE, width, 1.0, phase=0.5, power=2.6)]
        # Toward the toe line the D becomes a strip: six vertices across the top, six across the
        # bottom, so its five cells are the five toes' roots. Vertex j of the D keeps its place in
        # the order (j = 0 is the outer side, 5 the inner, 6..11 back along the bottom), so the
        # blend moves every vertex a short way and twists nothing — the palm does the same thing
        # on its way to the knuckles.
        u = [width]
        for share in TOE_SHARES:
            u.append(u[-1] - 2 * width * share)
        strip = [(x, top) for x in u] + [(x, -under) for x in reversed(u)]
        section = lerp_section(section, strip, profile([(0.62, 0.0), (0.92, 0.6), (1.0, 1.0)], s))
        centre = Vector((base.x, base.y + s * reach, low + under))
        t_rings.append(b.ring(centre, t1, t2, section, {f"foot.{side}": 1.0},
                              label=f"toe.{side}{s}"))
    order = b.match_order(toe_loop, t_rings[0])
    b.band(toe_loop, [t_rings[0][j] for j in order])
    # and the first ring keeps half the socket's own outline, so the foot flows out of the instep
    # rather than being a second shape pushed into a hole in the first
    first = sum((v.co for v in t_rings[0]), Vector()) / TOE
    spread = sum(((v.co - first) - t_axis * (v.co - first).dot(t_axis)).length
                 for v in t_rings[0]) / TOE
    flow_out(b, toe_loop, t_rings, order, t_axis, [first], [spread], (0.5,))
    b.tube(t_rings)

    # Five toes off the strip's five cells. Neighbouring toes SHARE the edge between their
    # cells, the way the two legs share the crotch: each takes one face on it, so it is manifold,
    # and the two faces leave it toward their own toe — which is the crease between two toes.
    # The fingers have webs between them because fingers spread; toes do not.
    last = t_rings[-1]
    toe_tips = []
    for n, (length, share) in enumerate(zip(TOE_LENGTHS, TOE_SHARES)):
        root = [last[n], last[n + 1], last[TOE - 2 - n], last[TOE - 1 - n]]
        middle = sum((v.co for v in root), Vector()) / 4
        across = abs((root[0].co - root[1].co).dot(t1))
        tall = abs((root[0].co - root[3].co).dot(t2))
        rings_n = []
        for k, f in enumerate(TOE_STUB_RINGS):
            shrink = profile([(0.0, 0.97), (1.0, 0.86)], f)
            centre = middle + t_axis * (f * length * p.foot) - t2 * (tall * 0.10 * f)
            rings_n.append(b.ring(centre, t1, t2,
                                  ellipse(4, across * 0.5 * shrink, tall * 0.5 * shrink, phase=0.5),
                                  {f"foot.{side}": 1.0}, label=f"toe{n}.{side}{f}"))
        b.band(root, b.match(root, rings_n[0]))
        b.tube(rings_n)
        b.face(rings_n[-1])
        toe_tips += rings_n[-1]

    # The arch. A foot rests on the heel and the ball and the span between lifts — on the INSIDE
    # of the foot, not down the middle, which is where stage 1 put it. Measured along the foot
    # from the back of the heel to the toe tip, and across it from the outer edge to the inner.
    #
    # Every vertex of the foot low enough to be part of the sole's edge takes a share, fading
    # with height: lifting only the underside's own vertices, as the first version here did,
    # left the walls beside them where they were, and subdivision averaged a 13mm lift down to 2.
    heel_y = min(v.co.y for v in sole[-1])
    tip_y = max(v.co.y for v in toe_tips)
    floor_z = p.sole_z
    reach_up = p.toe_depth
    lower = set(sole[-1]) | set(sole_grid.values()) | {v for ring in sole[-3:] for v in ring}
    lower |= {v for ring in t_rings for v in ring}
    for v in lower:
        along = (v.co.y - heel_y) / (tip_y - heel_y)
        inner = (base.x - v.co.x) * sign / p.foot_half
        rise = math.sin(math.pi * min(1.0, max(0.0, (along - 0.20) / 0.48)))
        rise *= min(1.0, max(0.0, (inner + 0.30) / 0.80))
        rise *= max(0.0, 1.0 - (v.co.z - floor_z) / reach_up)
        v.co.z += p.arch * rise


def build_body(p, name="body"):
    p.validate()
    at = p.joints()
    b = Build()
    rings, sockets, neck_hole = torso(b, p, at)
    neck_and_head(b, p, at, neck_hole)
    for side in ("left", "right"):
        arm(b, p, at, side, sockets[side])

    # The crotch: the torso's bottom ring is cut in two by one vertex on the centre line, and
    # each half plus that vertex is a 12-loop for a leg. The two legs SHARE those three
    # vertices, which is the whole reason this is one surface and not two tubes pushed into a
    # hole: the edges either side of the crotch vertex get one face from each leg.
    bottom = rings[0]
    front, back = TORSO // 4, 3 * TORSO // 4
    crotch = b.vert((0.0, 0.0, bottom[front].co.z - p.crotch_depth),
                    {"spine": 0.5, "thigh.left": 0.25, "thigh.right": 0.25})
    for side in ("right", "left"):
        if side == "right":
            arc = [bottom[(front - i) % TORSO] for i in range(front - back + TORSO + 1)]
        else:
            arc = [bottom[(front + i) % TORSO] for i in range(back - front + 1)]
        loop = arc + [crotch]
        b.loops.append((f"hip.{side}", loop))
        leg(b, p, at, side, loop)
    return b.object(name, p), at


# --- the armature ------------------------------------------------------------------------------

def build_rig(at, name="rig"):
    """`make_character.BONES` and `PARENTS`, at this body's own joints. Same names, so the pose
    solver, the staging and the USDZ export drive this character with no changes at all."""
    bpy.ops.object.armature_add(location=(0, 0, 0))
    rig = bpy.context.object
    rig.name = name
    bpy.ops.object.mode_set(mode="EDIT")
    bones = rig.data.edit_bones
    for bone in list(bones):
        bones.remove(bone)
    for bone_name, (head, tail) in make_character.BONES.items():
        bone = bones.new(bone_name)
        bone.head = at[head]
        bone.tail = at[tail]
        if (bone.tail - bone.head).length < 1e-5:
            raise SystemExit(f"bone {bone_name!r} has no length")
    for child, parent in make_character.PARENTS.items():
        bones[child].parent = bones[parent]
        bones[child].use_connect = False
    bpy.ops.object.mode_set(mode="OBJECT")
    return rig


def bind(body, rig):
    bpy.ops.object.select_all(action="DESELECT")
    body.select_set(True)
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.parent_set(type="ARMATURE_NAME")


def skin(obj):
    """The house skin, from `make_character`'s own palette and its own shader helper.

    Not a new look: the same sampled linear colour and the same subsurface scattering the drawn
    avatars and the old renders were made with, so a render of this body sits beside them as the
    same character family rather than as a grey proxy that has to be imagined in colour.
    """
    # make_character.build_body's own numbers for skin, not a near miss of them: stage 1 had
    # 0.62 and 0.35, which is a slightly drier, flatter skin than the character it sits beside.
    material = make_character.material("skin", make_character.SKIN,
                                       roughness=0.52, subsurface=0.55)
    obj.data.materials.clear()
    obj.data.materials.append(material)
    return material


def subsurf(obj, levels=2):
    mod = obj.modifiers.new("subsurf", "SUBSURF")
    mod.subdivision_type = "CATMULL_CLARK"
    mod.levels = levels
    mod.render_levels = levels
    return mod


# --- verification ------------------------------------------------------------------------------
#
# Six times tonight a check passed while the thing it was protecting was broken, because the check
# measured something next to the real property. The loop count is the one most exposed to that:
# "there are rings near the elbow" is not "three rings cross the elbow", and a ring of the TORSO
# near the hip winds round the thigh's axis just as happily as a ring of the thigh does. So the
# count below is deliberately built to under-report rather than over-report, and every claim it
# makes is one of these three, in order of strength:
#
#   1. A closed EDGE LOOP, found by Blender's own loop rule — at a four-valence vertex carry on
#      through the edge that shares no face with the one you came in on. If the walk meets an
#      irregular vertex it fails and the loop is not counted at all. Nothing is inferred from the
#      order rings were built in; `Build.loops` is never read by any of this.
#   2. It ENCIRCLES THE LIMB: winding number one about the joint's own axis, and no vertex further
#      than `1.45 * r` from that axis, where r is the limb's radius there from `Proportions`. The
#      second half is what rejects a torso ring at the hip — it winds, but it is twice as far out.
#   3. They form a CHAIN: between two consecutive loops there are exactly len(loop) quads with two
#      vertices in each. Three loops in a zone that are not joined by bands of quads would not
#      make a surface that folds; this is the part that makes "three loops cross the bend" mean
#      what it sounds like, and it is reported separately so you can see it held.
#
# And then, because topology can satisfy all three and still deform badly, `deformation` bends
# every joint and measures what happens to the skin.

JOINT_BONES = {
    # joint -> (bone above, bone below, which Proportions radius is the limb's, which of the two
    # bones the cross sections are built around).
    #
    # The last field matters and getting it wrong would be exactly the sort of nearly-right
    # measurement this file is trying not to make. At every joint but one, both bones are
    # collinear in the A-pose and the rings are perpendicular to both. The ankle is the exception:
    # `foot` points FORWARD, and the rings that fold at an ankle are the horizontal ones round the
    # ankle column, so the axis there is the shin's. Measured against the foot bone instead, the
    # ankle reports two loops and the real five are invisible.
    "neck": ("neck", "head", "neck_r", "below"),
    "shoulder.{side}": ("clavicle.{side}", "upper_arm.{side}", "shoulder_r", "below"),
    "elbow.{side}": ("upper_arm.{side}", "forearm.{side}", "elbow_r", "below"),
    "wrist.{side}": ("forearm.{side}", "hand.{side}", "wrist_r", "below"),
    "hip.{side}": ("spine", "thigh.{side}", "hip_r", "below"),
    "knee.{side}": ("thigh.{side}", "shin.{side}", "knee_r", "below"),
    "ankle.{side}": ("shin.{side}", "foot.{side}", "ankle_r", "above"),
}
ZONE = 0.30         # the bend zone is this fraction of the SHORTER of the two bones at a joint


def crossed(quad):
    """Does this quad's outline cross itself — is it a bow tie rather than a surface?

    Four points in a cycle, tested against their own Newell normal. Take the four corner
    triangles of the cycle and look at the signs of their areas: all four agreeing is a convex
    quad, exactly one disagreeing is a concave but perfectly good one, and two or three
    disagreeing means the outline crosses and there is no surface there at all.

    This is a property of the CYCLE. The first version of this test compared the normals of the
    two triangles either side of one diagonal, which is a property of which vertex the face
    happens to start at: it reported four bow ties in a mesh that has none, missed their mirror
    images on the other side of the body, and — in the deformation test, where the same code ran
    on a posed copy — produced a folded face on one shoulder and not the other for a pose that is
    exactly mirror-symmetric. Two of the three defects it reported were itself.
    """
    normal = Vector((0.0, 0.0, 0.0))
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        normal += a.cross(b)
    if normal.length < 1e-14:
        return True
    signs = [(quad[(i + 1) % 4] - quad[i]).cross(quad[(i + 2) % 4] - quad[(i + 1) % 4]).dot(normal)
             for i in range(4)]
    if min(abs(x) for x in signs) < 1e-18:
        return True
    return sum(1 for x in signs if x < 0) in (2, 3)


def opposite_edge(edge, vert):
    """Blender's loop rule at one vertex, or None where the loop has to stop."""
    if len(vert.link_edges) != 4:
        return None
    faces = set(edge.link_faces)
    out = [e for e in vert.link_edges if e is not edge and not faces & set(e.link_faces)]
    return out[0] if len(out) == 1 else None


def edge_loop(edge, limit=600):
    """The closed edge loop through `edge` as an ordered vertex list, or None if it is not one."""
    first, v = edge.verts[0], edge.verts[1]
    verts, cur = [first, v], edge
    for _ in range(limit):
        nxt = opposite_edge(cur, v)
        if nxt is None:
            return None
        if nxt is edge:
            return verts[:-1] if verts[-1] is first else None
        v = nxt.other_vert(v)
        if v in verts and v is not first:
            return None
        verts.append(v)
        cur = nxt
    return None


def winding(verts, origin, axis):
    """How many times an ordered loop goes round the line (origin, axis). One, for a cross section."""
    axis = axis.normalized()
    u = axis.cross(Vector((0.0, 0.0, 1.0)))
    if u.length < 1e-6:
        u = axis.cross(Vector((1.0, 0.0, 0.0)))
    u.normalize()
    w = axis.cross(u)
    total, previous = 0.0, None
    for v in list(verts) + [verts[0]]:
        d = v.co - origin
        d = d - axis * d.dot(axis)
        angle = math.atan2(d.dot(w), d.dot(u))
        if previous is not None:
            step = angle - previous
            while step > math.pi:
                step -= 2 * math.pi
            while step < -math.pi:
                step += 2 * math.pi
            total += step
        previous = angle
    return total / (2 * math.pi)


def quads_between(a, b):
    """Faces with two vertices in loop `a` and two in loop `b` — the band that joins them."""
    sa, sb = set(a), set(b)
    count = 0
    for face in {f for v in a for f in v.link_faces}:
        vs = set(face.verts)
        if len(vs & sa) == 2 and len(vs & sb) == 2 and len(vs) == 4:
            count += 1
    return count


def joint_loops(bm, origin, axis, half, radius):
    """Every loop that satisfies claims 1 and 2 above, as [(offset along the axis, verts)]."""
    axis = axis.normalized()
    found, seen = [], set()
    for edge in bm.edges:
        mid = (edge.verts[0].co + edge.verts[1].co) / 2
        d = mid - origin
        along = d.dot(axis)
        if abs(along) > half:
            continue
        if (d - axis * along).length > radius:
            continue
        direction = (edge.verts[1].co - edge.verts[0].co)
        if direction.length < 1e-9 or abs(direction.normalized().dot(axis)) > 0.6:
            continue                       # a rung, not a ring
        loop = edge_loop(edge)
        if loop is None:
            continue
        key = frozenset(v.index for v in loop)
        if key in seen:
            continue
        seen.add(key)
        offsets = [(v.co - origin).dot(axis) for v in loop]
        if max(abs(o) for o in offsets) > half:
            continue                       # wanders out of the bend zone
        if max((v.co - origin - axis * o).length for v, o in zip(loop, offsets)) > radius:
            continue                       # too far out to be this limb's cross section
        if abs(abs(winding(loop, origin, axis)) - 1.0) > 0.03:
            continue                       # does not encircle the limb
        found.append((sum(offsets) / len(offsets), loop))
    found.sort()
    return found


def chains(found):
    """The longest run of loops joined by full bands of quads, as (length, [band sizes])."""
    best, bands = 1, []
    run, run_bands = 1, []
    for (_, a), (_, b) in zip(found, found[1:]):
        n = quads_between(a, b)
        if n == len(a) == len(b):
            run += 1
            run_bands.append(n)
            if run > best:
                best, bands = run, list(run_bands)
        else:
            run, run_bands = 1, []
    return best, bands


def loop_report(obj, props, at):
    """Loop counts per joint, measured off the mesh — not off the table that built it."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.index_update()
    bones = make_character.BONES
    rows = {}
    for pattern, (above, below, radius_name, axis_from) in JOINT_BONES.items():
        for side in ("left", "right"):
            joint = pattern.format(side=side)
            if joint not in at:
                continue
            a, b = above.format(side=side), below.format(side=side)
            length_a = (Vector(at[bones[a][1]]) - Vector(at[bones[a][0]])).length
            length_b = (Vector(at[bones[b][1]]) - Vector(at[bones[b][0]])).length
            half = ZONE * min(length_a, length_b)
            radius = getattr(props, radius_name) * 1.45
            spine = bones[a if axis_from == "above" else b]
            axis = (Vector(at[spine[1]]) - Vector(at[spine[0]])).normalized()
            found = joint_loops(bm, Vector(at[joint]), axis, half, radius)
            run, bands = chains(found)
            rows[joint] = {
                "loops": len(found),
                "chain": run,
                "bands": bands,
                "zone": half,
                "radius": radius,
                "offsets": [round(o, 4) for o, _ in found],
                "proximal": sum(1 for o, _ in found if o < -1e-6),
                "distal": sum(1 for o, _ in found if o > 1e-6),
            }
            if pattern == "neck":
                break
    bm.free()
    return rows


def inward_faces(obj, props, at, threshold=-0.25):
    """Faces at a joint, at REST, whose normal points into the limb: skin folded back on itself.

    Added in stage 2 because stage 1's hip had exactly this and nothing measured it. The thigh's
    first two rings sat above the hole they were bridged to, so the skin went down to the pelvis,
    back UP inside it and down again: a fold with no crossed quad in it, no non-manifold edge and
    no triangle, which every count passed and the render showed as a hem. On a surface that only
    goes down a limb, the outward normal always has a component away from the limb's axis; on the
    band that doubles back it points toward it.

    Measured off the finished mesh, in the same zone and radius as `loop_report`, with Blender's
    own face normals — nothing from the build is consulted. `threshold` is the dot product of the
    face normal with the outward direction from the axis below which a face counts.

    Only the joint's OWN skin counts: a face whose corners' summed weight is largest for one of
    the joint's two bones, and — on a paired limb — that is nearer this limb's axis than its
    mirror image's. The first version counted every face in the zone and reported seven folds at
    the right hip that were all the LEFT thigh's inner face, which faces the right leg because it
    is the other side of the crotch, and three at each shoulder that were the chest wall under
    the arm, which faces the arm because that is what an armpit is. Ownership by weight alone was
    the second version, and it let the crotch faces back in as soon as the drape gave them more
    spine than thigh; nearness to the other leg's axis is the property that actually says whose
    skin a face is.

    And only faces whose normal lies ACROSS the limb (|normal . axis| <= 0.6). Skin that folds
    back runs along the limb, so its normal is across it, pointing in; a wall the limb grows out
    of faces along the limb instead. Added in stage 2b, when the shoulder drape had removed the
    four faces of the armpit's fold and the three that remained were the chest wall under the
    socket (normal . axis 0.83), kept in by weight because the blending round the socket gives
    that wall an arm majority. Checked on three meshes before trusting it: it still finds stage
    1's hip fold, and the armpit's four with the drape switched off.
    """
    obj.data.update()
    bones = make_character.BONES
    names = {g.index: g.name for g in obj.vertex_groups}

    def owner(poly):
        total = {}
        for index in poly.vertices:
            for g in obj.data.vertices[index].groups:
                total[names[g.group]] = total.get(names[g.group], 0.0) + g.weight
        return max(total, key=total.get) if total else None

    rows = {}
    for pattern, (above, below, radius_name, axis_from) in JOINT_BONES.items():
        for side in ("left", "right"):
            joint = pattern.format(side=side)
            if joint not in at:
                continue
            a, b = above.format(side=side), below.format(side=side)
            length = min((Vector(at[bones[a][1]]) - Vector(at[bones[a][0]])).length,
                         (Vector(at[bones[b][1]]) - Vector(at[bones[b][0]])).length)
            half = ZONE * length
            radius = getattr(props, radius_name) * 1.45
            spine = bones[a if axis_from == "above" else b]
            axis = (Vector(at[spine[1]]) - Vector(at[spine[0]])).normalized()
            origin = Vector(at[joint])
            mirror_origin = Vector((-origin.x, origin.y, origin.z))
            mirror_axis = Vector((-axis.x, axis.y, axis.z))
            count, worst = 0, 1.0
            for poly in obj.data.polygons:
                d = Vector(poly.center) - origin
                along = d.dot(axis)
                radial = d - axis * along
                if abs(along) > half or radial.length > radius or radial.length < 1e-6:
                    continue
                if owner(poly) not in (a, b):
                    continue
                if abs(Vector(poly.normal).dot(axis)) > 0.6:
                    continue                       # a wall the limb leaves, not the limb's skin
                if "{side}" in pattern:
                    m = Vector(poly.center) - mirror_origin
                    if (m - mirror_axis * m.dot(mirror_axis)).length < radial.length:
                        continue                   # the other limb's skin, facing this one
                dot = Vector(poly.normal).dot(radial.normalized())
                worst = min(worst, dot)
                if dot < threshold:
                    count += 1
            rows[joint] = {"inward": count, "worst": round(worst, 3)}
            if pattern == "neck":
                break
    return rows


def mirror_error(obj):
    """How far the mesh is from being its own mirror image in x, in position and in weights.

    Here because the deformation test disagreed between a left shoulder and a right one, and the
    only honest way to tell a real asymmetry from a float-level tie in a fold test is to measure
    the symmetry itself. It also catches the frame bug in `frame`'s docstring directly: a bow tie
    through one ankle is an asymmetry of millimetres, which this reports and a face count does not.
    """
    verts = obj.data.vertices
    index = {}
    for v in verts:
        index.setdefault((round(-v.co.x, 5), round(v.co.y, 5), round(v.co.z, 5)), []).append(v.index)
    worst_position, worst_weight, unmatched, worst_at = 0.0, 0.0, 0, None
    flip = {"left": "right", "right": "left"}
    for v in verts:
        key = (round(v.co.x, 5), round(v.co.y, 5), round(v.co.z, 5))
        partners = index.get(key)
        if not partners:
            unmatched += 1
            if unmatched <= 6:
                print(f"  no mirror for vertex {v.index} at {tuple(round(c, 5) for c in v.co)}")
            continue
        other = verts[partners[0]]
        worst_position = max(worst_position,
                             (Vector((-other.co.x, other.co.y, other.co.z)) - v.co).length)
        mine = {obj.vertex_groups[g.group].name: g.weight for g in v.groups if g.weight > 1e-6}
        theirs = {}
        for g in other.groups:
            if g.weight <= 1e-6:
                continue
            name = obj.vertex_groups[g.group].name
            if "." in name:
                stem, side = name.rsplit(".", 1)
                name = f"{stem}.{flip.get(side, side)}"
            theirs[name] = g.weight
        for bone in set(mine) | set(theirs):
            gap = abs(mine.get(bone, 0.0) - theirs.get(bone, 0.0))
            if gap > worst_weight:
                worst_weight = gap
                worst_at = (v.index, other.index, tuple(round(c, 4) for c in v.co), bone,
                            dict(sorted(mine.items())), dict(sorted(theirs.items())))
    return {"unmatched": unmatched, "position": worst_position, "weight": worst_weight,
            "where": worst_at}


def topology(obj, label=""):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    faces = {}
    for face in bm.faces:
        faces[len(face.verts)] = faces.get(len(face.verts), 0) + 1
    bow_ties, worst_bend, thinnest = 0, 0.0, 1e9
    for face in bm.faces:
        if len(face.verts) != 4:
            continue
        q = [v.co for v in face.verts]
        thinnest = min(thinnest, min((q[(i + 1) % 4] - q[i]).length for i in range(4)))
        if crossed(q):
            bow_ties += 1
        flat = (q[1] - q[0]).cross(q[2] - q[0])
        other = (q[3] - q[2]).cross(q[0] - q[2])
        if flat.length > 1e-12 and other.length > 1e-12:
            dot = flat.normalized().dot(other.normalized())
            worst_bend = max(worst_bend, math.degrees(math.acos(max(-1.0, min(1.0, dot)))))
    shells, seen = 0, set()
    for vert in bm.verts:
        if vert in seen:
            continue
        shells += 1
        stack = [vert]
        while stack:
            v = stack.pop()
            if v in seen:
                continue
            seen.add(v)
            for e in v.link_edges:
                stack.append(e.other_vert(v))
    out = {
        "label": label,
        "verts": len(bm.verts),
        "edges": len(bm.edges),
        "faces": len(bm.faces),
        "quads": faces.get(4, 0),
        "tris": faces.get(3, 0),
        "ngons": sum(n for size, n in faces.items() if size > 4),
        "non_manifold_edges": sum(1 for e in bm.edges if len(e.link_faces) != 2),
        "boundary_edges": sum(1 for e in bm.edges if len(e.link_faces) == 1),
        "wire_edges": sum(1 for e in bm.edges if not e.link_faces),
        "loose_verts": sum(1 for v in bm.verts if not v.link_faces),
        "irregular_verts": sum(1 for v in bm.verts if len(v.link_edges) != 4),
        "bow_ties": bow_ties,
        "shortest_edge": round(thinnest, 5),
        "worst_quad_bend_deg": round(worst_bend, 1),
        "shells": shells,
        "euler": len(bm.verts) - len(bm.edges) + len(bm.faces),
    }
    bm.free()
    return out


def subdivided_topology(obj):
    """The same counts on the Catmull-Clark result, because that is what gets rendered."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = bpy.data.meshes.new_from_object(evaluated, depsgraph=depsgraph)
    holder = bpy.data.objects.new("subdivided", mesh)
    out = topology(holder, "subdivided")
    bpy.data.objects.remove(holder)
    bpy.data.meshes.remove(mesh)
    return out


# A bend per joint for the deformation test: degrees about the WORLD X axis, which is the lateral
# axis, so every one of these is a flexion in the sagittal plane — the body folding the way a
# squat or a curl folds it. Signed so the fold goes the anatomical way: hips and shoulders
# forward, the knee back, the toe up.
#
# About the world axis and not the pose bone's own X, which is what the first version did. A
# bone's local X depends on its roll, and roll is mirrored between left and right, so that test
# flexed one hip and extended the other: the legs went into the splits and the crotch was
# measured being torn apart by the test rather than by the mesh. The hip's stretch read 6.1 for a
# pose the app will never ask for.
TEST_BENDS = {
    "head": 32.0, "upper_arm.{side}": 55.0, "forearm.{side}": 95.0, "hand.{side}": 42.0,
    "thigh.{side}": 60.0, "shin.{side}": -95.0, "foot.{side}": 30.0,
}
TEST_AXIS = Vector((1.0, 0.0, 0.0))


def deformation(body, rig, props, at):
    """Bend every joint and measure what the skin does. The check topology cannot make.

    Three loops at a joint are supposed to mean the fold is shared out. If it is not — if one
    band of quads takes the whole crease — the band collapses: its edges go to nearly nothing and
    its faces turn inside out. So the numbers are per joint, on the deformed base cage:

      squash   the smallest ratio of a deformed edge to its rest length, over edges in the zone.
               One band taking a 95-degree fold on its own reads about 0.1 here, and a face that
               has folded through itself reads negative on the normal test below.
      stretch  the largest such ratio, on the outside of the same crease.
      folded   quads that have folded THROUGH themselves: their two halves' normals, which agreed
               at rest, now oppose. This is rotation-free by construction — a face that merely
               swung 95 degrees with its bone cannot score — and it is the pinch itself.
      area     the smallest deformed-to-rest face area ratio in the zone. A band taking a fold
               alone collapses, and a collapse is an area going to nothing.
    """
    import numpy as np
    from mathutils import Quaternion

    subsurfs = [m for m in body.modifiers if m.type == "SUBSURF" and m.show_viewport]
    for mod in subsurfs:
        mod.show_viewport = False
    rest = np.array([v.co[:] for v in body.data.vertices], dtype=np.float64)
    for name, angle in TEST_BENDS.items():
        for side in ("left", "right"):
            bone = name.format(side=side)
            if bone in rig.pose.bones:
                pbone = rig.pose.bones[bone]
                pbone.rotation_mode = "QUATERNION"
                # the world axis expressed in this bone's own rest frame, so the fold happens in
                # the sagittal plane whichever side of the body the bone is on
                local = pbone.bone.matrix_local.to_3x3().inverted() @ TEST_AXIS
                pbone.rotation_quaternion = Quaternion(local.normalized(), math.radians(angle))
            if "{side}" not in name:
                break
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = body.evaluated_get(depsgraph)
    mesh = bpy.data.meshes.new_from_object(evaluated, depsgraph=depsgraph)
    posed = np.array([v.co[:] for v in mesh.vertices], dtype=np.float64)
    if len(posed) != len(rest):
        raise SystemExit("the evaluated mesh is not the base cage — turn the subsurf off first")

    bm = bmesh.new()
    bm.from_mesh(body.data)
    bm.verts.index_update()
    bm.faces.index_update()
    bones = make_character.BONES
    rows = {}
    for pattern, (above, below, radius_name, axis_from) in JOINT_BONES.items():
        for side in ("left", "right"):
            joint = pattern.format(side=side)
            if joint not in at:
                continue
            a, b = above.format(side=side), below.format(side=side)
            length = min((Vector(at[bones[a][1]]) - Vector(at[bones[a][0]])).length,
                         (Vector(at[bones[b][1]]) - Vector(at[bones[b][0]])).length)
            half = ZONE * length
            origin = Vector(at[joint])
            spine = bones[a if axis_from == "above" else b]
            axis = (Vector(at[spine[1]]) - Vector(at[spine[0]])).normalized()
            radius = getattr(props, radius_name) * 1.6
            # A band round the limb, not a ball round the joint. The ball was the first version
            # and it reported the neck as having no edges at all: the neck's own radius is bigger
            # than 1.6 times a bone that short, so every vertex of it fell outside the ball and
            # `squash` came back as its own starting value. A check that measures nothing reports
            # perfection.
            near = {v.index for v in bm.verts
                    if abs((v.co - origin).dot(axis)) <= half * 1.2
                    and (v.co - origin - axis * (v.co - origin).dot(axis)).length <= radius}
            if not near:
                raise SystemExit(f"the bend zone at {joint} contains no vertices")
            squash, stretch, worst = 1e9, 0.0, None
            for edge in bm.edges:
                i, j = edge.verts[0].index, edge.verts[1].index
                if i not in near or j not in near:
                    continue
                r = np.linalg.norm(rest[i] - rest[j])
                if r < 1e-9:
                    continue
                ratio = np.linalg.norm(posed[i] - posed[j]) / r
                squash = min(squash, ratio)
                if ratio > stretch:
                    stretch, worst = ratio, (i, j)
            folded, area_ratio = 0, 1e9
            for face in bm.faces:
                idx = [v.index for v in face.verts]
                if len(idx) != 4 or not all(i in near for i in idx):
                    continue
                before = [Vector(rest[i]) for i in idx]
                after = [Vector(posed[i]) for i in idx]
                rest_area = sum(((before[(i + 1) % 4] - before[i])
                                 .cross(before[(i + 2) % 4] - before[i])).length
                                for i in range(4)) / 4
                posed_area = sum(((after[(i + 1) % 4] - after[i])
                                  .cross(after[(i + 2) % 4] - after[i])).length
                                 for i in range(4)) / 4
                if rest_area > 1e-12:
                    area_ratio = min(area_ratio, posed_area / rest_area)
                if crossed(after) and not crossed(before):
                    folded += 1
            where = ""
            if worst:
                groups = []
                for index in worst:
                    w = {body.vertex_groups[g.group].name: round(g.weight, 2)
                         for g in body.data.vertices[index].groups if g.weight > 0.01}
                    groups.append(f"{tuple(round(c, 3) for c in rest[index])} {w}")
                where = " -> ".join(groups)
            rows[joint] = {"squash": round(squash, 3), "stretch": round(stretch, 3),
                           "folded": folded, "area": round(area_ratio, 3), "verts": len(near),
                           "worst": where}
            if pattern == "neck":
                break
    bm.free()
    bpy.data.meshes.remove(mesh)
    for mod in subsurfs:
        mod.show_viewport = True
    return rows


def rest_pose(rig):
    from mathutils import Quaternion
    for pbone in rig.pose.bones:
        pbone.rotation_mode = "QUATERNION"
        pbone.rotation_quaternion = Quaternion()
    bpy.context.view_layer.update()


# --- looking at it -----------------------------------------------------------------------------

def stage(props, subject=None):
    """The house stage — `make_character.stage` itself, called, not copied.

    Stage 1 rebuilt it here "in spirit", with the backdrop 40cm further back and the lamps aimed
    at a different height, so its renders and the old character's were lit differently and could
    not be compared directly. Calling the function is the only way to be sure the two are lit the
    same: the backdrop, the floor, the world and the three lamps, to the watt. It also makes a
    front camera framed on `subject`; the sheet uses its own cameras and ignores that one.

    `stage(props)` with no figure in the scene still works, because `skin_wrap.stage` calls it
    that way: make_character's function only needs a subject to frame its camera, so a box of
    the figure's height stands in for one and is removed again. Nothing it builds depends on it.
    """
    subject = subject or bpy.data.objects.get("body")
    if subject is not None:
        return make_character.stage(None, subject)
    stand_in = bpy.data.objects.new("stage stand-in", bpy.data.meshes.new("stage stand-in"))
    stand_in.data.from_pydata([(0.0, 0.0, 0.0), (0.0, 0.0, props.height)], [], [])
    bpy.context.collection.objects.link(stand_in)
    camera = make_character.stage(None, stand_in)
    mesh = stand_in.data
    bpy.data.objects.remove(stand_in)
    bpy.data.meshes.remove(mesh)
    return camera


def wire_copy(body, props):
    """The base cage as solid wire, to be rendered over the smooth surface.

    A wireframe is the only honest picture of topology: a smooth render of a bad cage and a smooth
    render of a good one look the same until something bends. The modifier is on a COPY with no
    subsurf, so what you see is the cage itself and not the cage of the subdivided mesh.
    """
    mesh = body.data.copy()
    wire = bpy.data.objects.new("cage", mesh)
    bpy.context.collection.objects.link(wire)
    wire.matrix_world = body.matrix_world.copy()
    for mod in list(wire.modifiers):
        wire.modifiers.remove(mod)
    mod = wire.modifiers.new("wire", "WIREFRAME")
    mod.thickness = props.height * 0.0032
    mod.use_replace = True
    mat = make_character.material("cage", (0.012, 0.014, 0.020, 1.0), roughness=0.5)
    mesh.materials.clear()
    mesh.materials.append(mat)
    for other in body.modifiers:
        if other.type == "ARMATURE":
            copy = wire.modifiers.new("armature", "ARMATURE")
            copy.object = other.object
    wire.parent = body.parent
    return wire


def frame_camera(name, eye, target, extent, lens=62.0):
    bpy.ops.object.camera_add(location=eye)
    camera = bpy.context.object
    camera.name = name
    camera.data.lens = lens
    camera.data.sensor_fit = "VERTICAL"
    camera.data.sensor_height = 36.0
    fov = 2 * math.atan((36.0 / 2) / lens)
    distance = (extent / 2) / math.tan(fov / 2)
    direction = (Vector(target) - Vector(eye)).normalized()
    camera.location = Vector(target) - direction * distance
    camera.rotation_mode = "QUATERNION"
    camera.rotation_quaternion = direction.to_track_quat("-Z", "Y")
    return camera


def render_to(path, camera, show, resolution=(620, 900), samples=40):
    scene = bpy.context.scene
    scene.camera = camera
    scene.render.resolution_x, scene.render.resolution_y = resolution
    scene.render.filepath = str(path)
    scene.render.image_settings.file_format = "PNG"
    for obj in bpy.data.objects:
        if obj.type == "MESH" and obj.name in ("body", "cage"):
            obj.hide_render = obj.name not in show
    make_character.use_cycles(scene, samples=samples)
    bpy.ops.render.render(write_still=True)
    return str(path)


OLD_CHARACTER = Path(__file__).resolve().parent.parent.parent / "3d-character" / "3d cartoon boy.glb"
# The views the comparison renders, the same for both characters: (name, eye).
COMPARE_VIEWS = (("front", (0.0, 4.0, 0.5)), ("three-quarter", (2.6, 2.6, 0.62)),
                 ("side", (4.0, 0.0, 0.5)))


def compare_with_old(out_dir, props, samples=40, panel=(620, 900)):
    """The OLD character and this one, each rendered alone on the same stage, by the same cameras.

    Stage 1's "beside" panel pasted a crop of an old animation frame next to this body, scaled by
    pixel measurements: a different render, at a different size, under different light. This loads
    the old model itself (`fit_model.load`, which turns it to face +Y) and renders it under
    `stage`, then builds this body and renders it under the same `stage` with the same three
    cameras at the same distance, framed on the same height. Both are 1.0 tall (the old one
    measures 0.9995), so nothing is scaled. Each is rendered at the origin on its own, rather
    than the two side by side in one frame, so that neither stands nearer the key light.

    `fit_model.load` empties the scene, so this destroys whatever was built before it and leaves
    THIS body, freshly built, in the scene afterwards. Returns {(who, view): path}.
    """
    import fit_model
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = Vector((0.0, 0.0, props.height * 0.5))
    extent = props.height * 1.08
    paths = {}
    for who in ("old", "new"):
        if who == "old":
            subject = fit_model.load(OLD_CHARACTER)
            subject.name = "old"
        else:
            subject = build(props)[0]
        stage(props, subject)
        for name, eye in COMPARE_VIEWS:
            eye = Vector(eye) * props.height
            camera = frame_camera(f"{who}-{name}", eye, target, extent)
            paths[(who, name)] = render_to(out_dir / f"compare-{who}-{name}.png", camera,
                                           {"body"}, panel, samples)
    return paths


def montage(paths, columns, out):
    """Stitch the panels into one sheet. Blender's own image API, so there is nothing to install."""
    import numpy as np
    images = [bpy.data.images.load(p) for p in paths]
    width, height = images[0].size
    rows = (len(images) + columns - 1) // columns
    canvas = np.zeros((rows * height, columns * width, 4), dtype=np.float32)
    canvas[..., 3] = 1.0
    for i, image in enumerate(images):
        pixels = np.array(image.pixels[:], dtype=np.float32).reshape(image.size[1], image.size[0], 4)
        row, column = divmod(i, columns)
        bottom = (rows - 1 - row) * height          # Blender images count up from the bottom
        canvas[bottom:bottom + pixels.shape[0], column * width:column * width + pixels.shape[1]] = pixels
    # A BYTE image. Stage 1 made this a float buffer, and Blender treats float pixels as linear
    # and encodes them to sRGB on save — so the 8-bit sRGB panels were encoded twice and the whole
    # sheet came out washed: a skin pixel of 0.694 in the panel was 0.851 in the sheet. Reading
    # the sheet back through Blender undoes the same conversion and says they match, which is
    # why it went unnoticed; an independent PNG decoder is what shows it.
    sheet = bpy.data.images.new("sheet", columns * width, rows * height, alpha=True,
                                float_buffer=False)
    sheet.pixels = canvas.reshape(-1).tolist()
    sheet.filepath_raw = str(out)
    sheet.file_format = "PNG"
    sheet.save()
    return str(out)


def sheet(body, rig, props, at, out, samples=40, panel=(620, 900)):
    """Front, side, three-quarter, the cage in wire, a hand, a foot, the body bent.

    Renders the panels into `<out stem>-panels/` and returns ({name: path}, deformation rows).
    The comparison with the old character is `compare_with_old`, run after this because loading
    the old model empties the scene; `main` puts the two together into one sheet.
    """
    out = Path(out)
    tmp = out.parent / f"{out.stem}-panels"
    tmp.mkdir(parents=True, exist_ok=True)
    centre = Vector((0.0, 0.0, props.height * 0.5))
    tall = props.height * 1.08
    wide_tall = props.height * 1.14
    hand = Vector(at["wrist.right"]) + (Vector(at["hand.right"]) - Vector(at["wrist.right"])) * 0.5
    hand_extent = props.hand * 2.3
    foot = Vector(at["ankle.right"]) * 0.40 + Vector(at["foot.right"]) * 0.60
    foot_extent = props.foot * 2.4
    wire = wire_copy(body, props)
    panels = {}

    for name, eye, target, extent, show, in (
            ("front", (0, 4, props.height * 0.5), centre, tall, {"body"}),
            ("side", (4, 0, props.height * 0.5), centre, tall, {"body"}),
            ("three-quarter", (2.6, 2.6, props.height * 0.62), centre, wide_tall, {"body"}),
            ("cage", (1.9, 3.2, props.height * 0.60), centre, wide_tall, {"body", "cage"}),
            # square on to the back of the hand: the hand's own plane is spanned by the arm
            # axis and the frame's second axis, so the camera stands off along that normal
            ("hand", (Vector(at["hand.right"]) + Vector((0.62, 0.42, 0.62))), hand,
             hand_extent, {"body", "cage"}),
            ("foot", (Vector(at["foot.right"]) + Vector((0.72, 0.50, 0.30))), foot,
             foot_extent, {"body", "cage"})):
        if name in ("hand", "foot"):
            # the cage's wire is 3mm on a body a metre tall, which is most of a finger: thinner,
            # or the close-up shows scaffolding and no hand
            wire.modifiers["wire"].thickness = props.height * 0.0011
        camera = frame_camera(name, Vector(eye), Vector(target), extent)
        panels[name] = render_to(tmp / f"{name}.png", camera, show, panel, samples)

    # and the same body bent, because a cage that looks right standing still is only half the claim
    deform = deformation(body, rig, props, at)
    camera = frame_camera("bent", Vector((2.4, 2.8, props.height * 0.55)), centre,
                          props.height * 1.25)
    panels["bent"] = render_to(tmp / "bent.png", camera, {"body"}, panel, samples)
    rest_pose(rig)
    bpy.data.objects.remove(wire)
    return panels, deform


# --- main --------------------------------------------------------------------------------------

def build(props=None, levels=2):
    """The whole character: mesh, armature, binding, subsurf. Everything downstream needs."""
    props = (props or Proportions()).validate()
    make_character.clear()
    body, at = build_body(props)
    skin(body)
    rig = build_rig(at)
    bind(body, rig)
    subsurf(body, levels)
    return body, rig, props, at


def report(body, rig, props, at, loops, deform, cage, subdiv, inward=None):
    lines = []

    def say(text=""):
        lines.append(text)
        print(text)

    say("")
    say("=== proportions, as fractions of height ===")
    for name, value in props.fractions().items():
        say(f"  {name:32s} {value:.3f}")
    say(f"  mirror symmetry: worst position error {cage['mirror']['position']:.2e}, "
        f"worst weight error {cage['mirror']['weight']:.2e}, "
        f"vertices with no mirror {cage['mirror']['unmatched']}")
    if cage["mirror"]["weight"] > 1e-6:
        say(f"    worst at {cage['mirror']['where']}")
    low = min(v.co.z for v in body.data.vertices)
    high = max(v.co.z for v in body.data.vertices)
    arm = (Vector(at["hand.right"]) - Vector(at["shoulder.right"])).length
    drop = math.degrees(math.asin((at["shoulder.right"].z - at["hand.right"].z) / arm))
    say(f"  measured cage height            {high - low:.4f} (floor {low:+.4f}, crown {high:.4f})")
    say(f"  A-pose arm angle below horizontal {drop:.2f} deg   (asked for {props.arm_drop_deg})")
    say(f"  leg splay from vertical           {props.leg_splay_deg:.2f} deg")

    say("")
    say("=== topology ===")
    for row in (cage, subdiv):
        say(f"  {row['label']:12s} faces {row['faces']:6d}  quads {row['quads']:6d}  "
            f"tris {row['tris']}  ngons {row['ngons']}")
        say(f"  {'':12s} verts {row['verts']:6d}  edges {row['edges']:6d}  "
            f"shells {row['shells']}  Euler {row['euler']}")
        say(f"  {'':12s} bow-tie quads {row['bow_ties']}  "
            f"worst quad bend {row['worst_quad_bend_deg']} deg  "
            f"shortest edge {row['shortest_edge']}")
        say(f"  {'':12s} non-manifold edges {row['non_manifold_edges']}  "
            f"boundary {row['boundary_edges']}  wire {row['wire_edges']}  "
            f"loose verts {row['loose_verts']}  irregular verts {row['irregular_verts']}")

    say("")
    say("=== edge loops crossing each joint's bend zone ===")
    say("  (a loop is counted only if the valence-4 walk closes it, it winds once about the")
    say("   limb axis, and every vertex of it is inside the zone and within the radius shown)")
    for joint in sorted(loops):
        row = loops[joint]
        say(f"  {joint:16s} loops {row['loops']}  chained {row['chain']}  "
            f"bands {row['bands']}  "
            f"proximal/distal {row['proximal']}/{row['distal']}  "
            f"zone +-{row['zone']:.4f}  radius {row['radius']:.4f}")
        say(f"  {'':16s} offsets along the axis: {row['offsets']}")

    if inward is not None:
        say("")
        say("=== skin folded back on itself at rest (faces whose normal points into the limb) ===")
        for joint in sorted(inward):
            row = inward[joint]
            say(f"  {joint:16s} inward-facing faces {row['inward']}  "
                f"worst normal . outward {row['worst']:+.3f}")

    say("")
    say("=== the same joints, bent ===")
    for joint in sorted(deform):
        row = deform[joint]
        say(f"  {joint:16s} squash {row['squash']:.3f}  stretch {row['stretch']:.3f}  "
            f"min area {row['area']:.3f}  folded-through faces {row['folded']}")
        if row['stretch'] > 1.8 or row['folded']:
            say(f"  {'':16s} worst-stretched edge: {row['worst']}")

    say("")
    say("=== vertex groups ===")
    counts = {}
    for vert in body.data.vertices:
        for g in vert.groups:
            name = body.vertex_groups[g.group].name
            counts[name] = counts.get(name, 0) + 1
    say(f"  bones in make_character.BONES: {len(make_character.BONES)}; "
        f"groups on the mesh: {len(body.vertex_groups)}; used: {len(counts)}")
    for name in sorted(counts):
        say(f"  {name:18s} {counts[name]:5d} vertices")
    missing = [n for n in make_character.BONES if n not in counts]
    if missing:
        say(f"  bones with NO vertices: {missing}")
    return "\n".join(lines)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="build and check the character body")
    parser.add_argument("--out", default=None, help="render the verification sheet here")
    parser.add_argument("--save", default=None, help="write the .blend here")
    parser.add_argument("--levels", type=int, default=2, help="Catmull-Clark levels")
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--height", type=float, default=1.0)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args(argv)

    props = Proportions(height=args.height)
    body, rig, props, at = build(props, levels=args.levels)
    cage = topology(body, "base cage")
    cage["mirror"] = mirror_error(body)
    subdiv = subdivided_topology(body)
    loops = loop_report(body, props, at)
    inward = inward_faces(body, props, at)

    if args.out and not args.no_render:
        stage(props, body)
        panels, deform = sheet(body, rig, props, at, args.out, samples=args.samples)
    else:
        deform = deformation(body, rig, props, at)
        rest_pose(rig)
        panels = None
    text = report(body, rig, props, at, loops, deform, cage, subdiv, inward)
    if args.save:
        bpy.ops.wm.save_as_mainfile(filepath=args.save)
        print("saved", args.save)
    if panels:
        # Last, because loading the old model empties the scene (and the .blend is saved above).
        out = Path(args.out)
        tmp = out.parent / f"{out.stem}-panels"
        compare = compare_with_old(tmp, props, samples=args.samples)
        views = [name for name, _ in COMPARE_VIEWS]
        side_by_side = montage([compare[("old", v)] for v in views]
                               + [compare[("new", v)] for v in views], 3,
                               out.parent / f"{out.stem}-comparison.png")
        path = montage([panels[n] for n in ("front", "side", "three-quarter", "cage", "hand",
                                            "foot")]
                       + [compare[("old", "front")], compare[("new", "front")], panels["bent"]],
                       3, out)
        print(f"\nsheet: {path}\ncomparison: {side_by_side}")
        out.with_suffix(".txt").write_text(
            text + f"\n\nsheet: {path}\ncomparison: {side_by_side}\n")


if __name__ == "__main__":
    main()
