#!/usr/bin/env python
"""A library of gym equipment props, built from primitives, to stand or sit beside the mascot.

    blender --background --python pipeline/blender/equipment.py -- --still /tmp/equipment.png

Why this exists. `make_character.build_body(equipment="dumbbells")` joins a pair of dumbbells
into the procedural character's own mesh, which only works because that character IS a mesh
this script owns outright. The real character — an imported model, fitted by `fit_model.py` — is
somebody else's mesh, and gluing a dumbbell onto a wrist by joining geometry is not an option
there, nor is it a sane way to reuse one barbell across fourteen exercises. So equipment lives
here as its own set of objects, built once per prop and either parented to a posed bone
(`hold`) or dropped on the floor (`stand`), same as a prop department hands an actor a cup
rather than sculpting it onto their hand.

## Scale

The character this stands next to is not modelled in metres — it is a cartoon with an oversized
head, and its measured height in Blender units is about 0.93. Every function below takes that
height as its first argument and converts with `unit = height / 1.75`, treating the character as
a 1.75 m adult no matter what units its mesh actually uses. A dimension below written in metres
is multiplied by `unit`, nothing else, so a prop sized against a 0.93-unit character and a prop
sized against a to-scale 1.75-unit import land at the same real-world size relative to each body.

## The real-world dimensions this module was built from

Two reference points were handed down in the brief and are load-bearing — get these two wrong
and every barbell and bench in the render is wrong with them:

    Olympic barbell        2.2 m long, 28 mm bar, 450 mm plates (standard spec)
    Flat bench pad         1.22 m long x 0.45 m high x 0.30 m wide (standard spec)
    Pull-up bar            about 2.2 m off the floor (typical doorway/tower bar)

Everything else is either a direct standard spec (noted "spec") or a reasoned estimate anchored
to one (noted "rationale"), because no two gyms' leg press machines share a part number:

    BARBELL_BAR_DIAMETER    = 0.028 m   men's Olympic bar, spec
    BARBELL_SLEEVE_DIAMETER = 0.050 m   where plates load, standard Olympic bar spec
    BARBELL_SLEEVE_LENGTH   = 0.415 m   usable loading length each side, spec
    PLATE_DIAMETER          = 0.450 m   full-size "45" plate, spec (brief)
    PLATE_THICKNESS         = 0.028 m   a 20 kg cast plate this diameter is about this thick, spec
    COLLAR_DIAMETER         = 0.090 m   rationale: a stop proud of the 50 mm sleeve, not a product
    COLLAR_THICKNESS        = 0.025 m   rationale, as above

    DUMBBELL_HANDLE_DIAMETER = 0.028 m  rationale: matches the barbell's knurled bar stock
    DUMBBELL_HANDLE_LENGTH   = 0.130 m  rationale: a one-hand grip, typical hex-dumbbell spec
    DUMBBELL_HEAD_RADIUS_10KG   = 0.050 m  rationale: a 10 kg rubber hex dumbbell's head, spec-ish
    DUMBBELL_HEAD_THICKNESS_10KG = 0.035 m rationale, as above
        Other weights scale these two by (kg / 10) ** (1/3) — volume scaling at roughly constant
        density, which is how a dumbbell set actually grows: a 40 kg head is not four times as
        wide, it is a bit less than 1.6x as wide and proportionately as thick. This is the shape
        of the scaling law, not a specific manufacturer's numbers.

    BENCH_HEIGHT       = 0.45 m   pad height off the floor, spec (brief)
    BENCH_PAD_LENGTH   = 1.22 m   spec (brief)
    BENCH_PAD_WIDTH    = 0.30 m   spec (brief)
    INCLINE_SEAT_LENGTH = 0.35 m  rationale: shorter than a flat pad, it only carries the hips
    INCLINE_BACK_LENGTH = 0.60 m  rationale: long enough to support the spine to the shoulders
    INCLINE_PAD_WIDTH   = 0.30 m  rationale: matches the flat bench pad's width

    PULLUP_BAR_HEIGHT = 2.2 m    spec (brief)
    PULLUP_BAR_SPAN   = 1.2 m    rationale: a comfortable wide-grip width on a fixed bar
    PULLUP_BAR_DIAMETER = 0.032 m rationale: pull-up bars commonly run thicker than a barbell

    LAT_SEAT_HEIGHT    = 0.45 m  rationale: matches a flat bench's seat height
    LAT_COLUMN_HEIGHT  = 2.2 m   rationale: matches the pull-up bar reference for a cable mast
    LAT_BAR_WIDTH      = 1.10 m  rationale: a standard wide lat bar's grip span
    CABLE_DIAMETER     = 0.008 m rationale: a thin steel cable, visually distinct from the bar stock

    CABLE_TOWER_HEIGHT  = 2.0 m  rationale: a functional-trainer column, shorter than a lat mast
    CABLE_BAR_LENGTH    = 0.46 m rationale: a standard straight short bar attachment
    CABLE_ROPE_LENGTH   = 0.60 m rationale: a standard rope attachment's hanging length

    LEG_PRESS_RAIL_LENGTH = 1.20 m  rationale: sled travel on a 45-degree leg press
    LEG_PRESS_PAD_SIZE  = (0.40, 0.55) m  rationale: width x length of the reclined back pad
    LEG_PRESS_PLATE_SIZE = (0.46, 0.36) m rationale: a typical foot-plate footprint

    SQUAT_RACK_HEIGHT   = 2.1 m  rationale: a power-rack upright, just under the pull-up height
    SQUAT_RACK_SPAN     = 1.0 m  rationale: wide enough to clear a loaded barbell's sleeves
    SQUAT_RACK_HOOK_HEIGHT = 1.3 m  rationale: about 0.74 of a 1.75 m lifter's height, roughly
        shoulder height, which is where a J-hook sits for an un-rack before a squat

    FLOOR_MAT_LENGTH    = 1.73 m  spec: a standard consumer exercise mat, 68 in
    FLOOR_MAT_WIDTH     = 0.61 m  spec: as above, 24 in
    FLOOR_MAT_THICKNESS = 0.015 m spec: a mid-weight mat, roughly 15 mm

## Anchors

Every function documents its own origin, but the convention is the same one `make_character`
already uses for the floor and the camera: for anything that stands on the ground, local Z = 0
is the plane where it touches the floor, and X/Y = 0 is the centre of its footprint (or, where a
prop has an obvious centre line — the pull-up bar's two uprights, the squat rack's two uprights —
the midpoint between the parts). `stand()` then becomes a plain assignment with nothing to
correct for. The two bar-shaped props a hand holds, `dumbbell` and `barbell`, use the other
sensible anchor instead: the centre of the bar, because that is the point `hold()` needs to line
up with a hand bone.

## Materials

The style brief allows three: a mid-grey matte metal, a near-black matte rubber, and one accent.
Picked, like `make_character`'s own palette, as linear RGB converted from an sRGB value with
gamma 2.2 (`(s / 255) ** 2.2`), because Blender's shader wants linear and a value pasted straight
from a colour picker renders washed out:

    METAL   sRGB (150, 150, 155) -- mid grey, a hint cool, for bars, frames, pulleys, uprights
    RUBBER  sRGB  (40,  38,  40) -- near-black, for plates, pads, mats, grips
    ACCENT  sRGB (230, 140,  80) -- one warm pastel, for the floor mat and small weight-colour
            details, the same role a coloured end-cap plays on real dumbbells

## What is NOT modelled

No branding, no cable pulleys' internal sheaves, no knurl texture (a bump map would need UVs
this module does not bother with; the "knurled-looking" handle in the brief is read from its
slightly bevelled, slightly rough-shaded cylinder, not from geometry that actually grips). The
leg press and cable tower are reduced to the parts that make them legible in silhouette — rails,
pads, a column, a pulley — not a mechanically accurate carriage, because the verification render
is a contact sheet to check proportion, not a machine-correct replica.
"""

import argparse
import math
import sys
import tempfile
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import make_character  # noqa: E402

# --- palette -------------------------------------------------------------------------------

METAL = (0.311, 0.311, 0.334, 1.0)
RUBBER = (0.0171, 0.0157, 0.0171, 1.0)
ACCENT = (0.797, 0.268, 0.078, 1.0)

# --- real-world dimensions, metres (see the module docstring for sources/rationale) ---------

BARBELL_LENGTH = 2.2
BARBELL_BAR_DIAMETER = 0.028
BARBELL_SLEEVE_DIAMETER = 0.050
BARBELL_SLEEVE_LENGTH = 0.415
PLATE_DIAMETER = 0.450
PLATE_THICKNESS = 0.028
COLLAR_DIAMETER = 0.090
COLLAR_THICKNESS = 0.025

DUMBBELL_HANDLE_DIAMETER = 0.028
DUMBBELL_HANDLE_LENGTH = 0.130
DUMBBELL_HEAD_RADIUS_10KG = 0.050
DUMBBELL_HEAD_THICKNESS_10KG = 0.035

BENCH_HEIGHT = 0.45
BENCH_PAD_LENGTH = 1.22
BENCH_PAD_WIDTH = 0.30
BENCH_LEG_THICKNESS = 0.045

INCLINE_SEAT_LENGTH = 0.35
INCLINE_BACK_LENGTH = 0.60
INCLINE_PAD_WIDTH = 0.30

PULLUP_BAR_HEIGHT = 2.2
PULLUP_BAR_SPAN = 1.2
PULLUP_BAR_DIAMETER = 0.032
PULLUP_UPRIGHT_DIAMETER = 0.06

LAT_SEAT_HEIGHT = 0.45
LAT_SEAT_SIZE = (0.36, 0.32)
LAT_THIGH_PAD_HEIGHT = 0.62
LAT_COLUMN_HEIGHT = 2.2
LAT_BAR_WIDTH = 1.10
# A real gym cable is 6-8 mm of steel in a nylon sheath. At this character's scale that is
# 0.0046 Blender units — about one pixel in a 520-pixel tile — so a dimensionally honest cable
# is an invisible cable, and a cable machine whose cable cannot be seen is a pole. Thickened to
# the same "legibility over fidelity" rule this module already applies to the leg press's tilt
# plane and its rail separation, and for the same reason: the prop exists to be recognised.
CABLE_DIAMETER = 0.018
# How far below the top of the mast the pulley's centre sits, and how much cable hangs below it
# by default. Both were local literals inside `lat_pulldown` until a caller needed to know them:
# `exercise_stage` has to put the bar where a particular figure's hands are, which means solving
# for the drop, which means reading the inset. A caller that mirrors a literal out of a function
# body is a caller waiting for that body to be edited, so they are constants here instead.
LAT_PULLEY_INSET = 0.02
LAT_BAR_DROP = 0.45
# How far from the column's own axis the cable hangs. It was zero, which put the cable exactly
# down the centre line of a column eleven times its diameter — so the one part of the machine
# that says "cable" was inside the mast in every render ever made of it, and the prop read as a
# pole with a bar stuck through it. A real cable leaves the pulley at the rim, so these are the
# pulley radii: the cable hangs in free air on the side the user is on, and the attachment hangs
# under it, where a hand can be put on it.
# Far enough that the cable hangs clear of the 0.09 m column rather than grazing it: a pulley
# head on a real machine projects from the mast, and the cable drops from the head's outer end.
# At the pulley radius (the first attempt) the cable's own centre line cleared the column by
# 8 mm and was still hidden behind it from any camera that was not exactly side-on.
LAT_CABLE_REACH = 0.14       # toward the seat, the machine's own +X

CABLE_TOWER_HEIGHT = 2.0
CABLE_BAR_LENGTH = 0.46
CABLE_ROPE_LENGTH = 0.60
CABLE_ROPE_SPLAY = 0.22
CABLE_PULLEY_INSET = 0.02
CABLE_DROP = 0.55
CABLE_TOWER_REACH = 0.14     # toward the user, the tower's own -Y; see LAT_CABLE_REACH

LEG_PRESS_RAIL_LENGTH = 1.20
LEG_PRESS_BACK_PAD = (0.40, 0.55)
LEG_PRESS_FOOT_PLATE = (0.46, 0.36)

SQUAT_RACK_HEIGHT = 2.1
SQUAT_RACK_SPAN = 1.0
SQUAT_RACK_HOOK_HEIGHT = 1.3
SQUAT_RACK_UPRIGHT = 0.06

FLOOR_MAT_LENGTH = 1.73
FLOOR_MAT_WIDTH = 0.61
FLOOR_MAT_THICKNESS = 0.015

COLUMN_DIAMETER = 0.09  # shared by the lat pulldown and cable tower's upright


# --- materials, primitives and joining -----------------------------------------------------

def _materials():
    """The three materials this whole library is allowed, fetched once and reused.

    Calling `make_character.material()` twice for the same name would hand Blender a second
    datablock named "equip_metal.001", silently wrong in a way that only shows up as a colour
    that does not match between two props in the same render. Fetching an existing one first,
    the same guard `bpy.data.materials.get` offers any caller, avoids it.
    """
    metal = bpy.data.materials.get("equip_metal") or \
        make_character.material("equip_metal", METAL, roughness=0.38)
    rubber = bpy.data.materials.get("equip_rubber") or \
        make_character.material("equip_rubber", RUBBER, roughness=0.88)
    accent = bpy.data.materials.get("equip_accent") or \
        make_character.material("equip_accent", ACCENT, roughness=0.55)
    return metal, rubber, accent


def _origin(name):
    """An empty mesh at the world origin, built only to be the target of a later join.

    `bpy.ops.object.join()` bakes every selected object's geometry into whichever object is
    ACTIVE at the time, in that object's own local space — so the active object's transform is
    where the merged result's origin ends up. Every part below is built at an absolute world
    coordinate chosen so that the documented anchor point already sits at world (0, 0, 0), which
    only survives the join if the active object itself has an untouched identity transform. An
    ordinary part risks carrying its own rotation or an off-centre `location` into that role; an
    empty object made the same way `make_character.skinned_body` makes one — a fresh mesh
    datablock linked into the scene, never moved — has no geometry to contribute and an identity
    transform by construction, so it fixes the anchor exactly where the docstring says it is.
    """
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def _join(name, parts):
    """Select every part, make the first (the anchor) active, and merge them into one object."""
    bpy.ops.object.select_all(action="DESELECT")
    for part in parts:
        part.select_set(True)
    bpy.context.view_layer.objects.active = parts[0]
    bpy.ops.object.join()
    out = bpy.context.object
    out.name = name
    return out


def _pad(name, centre, size, mat, rotation=(0.0, 0.0, 0.0), bevel=None, segments=2):
    """A rounded slab: a cube, bevelled on every edge and shaded smooth.

    `make_character` only hands this module rods and spheres (`capsule`, `ball`), and neither
    draws a bench pad, a foot plate or a mat. A cube with a Bevel modifier applied and then
    shaded smooth reads the same soft, no-hard-corners way those two primitives do, which is the
    one rule the brief's style section sets. `size` is the full (x, y, z) extent in Blender units
    already — every caller below has already multiplied its real-world metres by `unit` before
    this is called — matching how a caller thinks about a pad's length/width/height.

    `bevel` left at its default scales with the pad itself — 15% of its thinnest dimension —
    rather than defaulting to one fixed Blender-unit number. A fixed default was this module's
    second scale bug: every pad in the library but `floor_mat` (which passed its own scaled
    bevel) used the same flat bevel width regardless of `unit`, so a prop built for a half-scale
    procedural character and the same prop built for a true-to-life imported one would carry
    IDENTICAL edge rounding in Blender units — proportionally twice as chunky on the smaller
    figure. Scaling it off `size` keeps the rounding looking like the same material at any `unit`.
    """
    if bevel is None:
        bevel = min(size) * 0.15
    # `size=1.0` on the primitive gives a cube with edge length 1.0 on every axis already, so
    # the scale that turns it into a `size[i]`-metre slab is `size[i]` itself. A first version
    # of this divided by 2 here, reasoning as if the primitive's HALF-extent were 1.0 rather
    # than its full edge — every pad, pad foot and mat in this module rendered at exactly half
    # its intended length until a measured floor mat came back 0.305 x 0.865 m against a
    # 0.61 x 1.73 m target and the ratio gave the bug away.
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=centre, rotation=rotation)
    obj = bpy.context.object
    obj.name = name
    obj.scale = (size[0], size[1], size[2])
    # Applied before the bevel, because the Bevel modifier's `width` is a world-space distance
    # measured along the mesh's own local axes; left un-applied, the object's non-uniform scale
    # would stretch that width differently on each axis and the "rounded" edge would come out
    # rounded on one side of the pad and knife-sharp on the other.
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    mod = obj.modifiers.new("bevel", "BEVEL")
    mod.width = bevel
    mod.segments = segments
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier="bevel")
    obj.data.materials.append(mat)
    bpy.ops.object.shade_smooth()
    return obj


def _disc(name, centre, radius, thickness, mat, hex=False, axis="Y"):
    """A flat disc or hex prism — a weight plate, a collar, a pulley wheel.

    Built with its flat faces on +Y and -Y (the axis a barbell's sleeve runs along in this
    module, since the bar is built lying along X and the plate's own thickness runs along the
    bar), then reoriented for the few callers that want it upright instead. `vertices=6` for the
    hex option is not a stand-in for a round plate at low resolution — Blender's cylinder
    primitive with six sides over this small a radius already reads as "hex plate" at a glance,
    which is exactly the shape a real hex dumbbell or hex plate is named for.
    """
    bpy.ops.mesh.primitive_cylinder_add(vertices=6 if hex else 24, radius=radius, depth=thickness,
                                        location=centre, rotation=(math.pi / 2, 0, 0))
    obj = bpy.context.object
    obj.name = name
    if axis == "Z":
        obj.rotation_euler = (0.0, 0.0, 0.0)
    elif axis == "X":
        obj.rotation_euler = (0.0, math.pi / 2, 0.0)
    obj.data.materials.append(mat)
    bpy.ops.object.shade_smooth()
    return obj


def capsule(name, start, end, radius, mat, segments=16, flush=True):
    """`make_character.capsule`, re-exported under this module's default segment count, and with
    its rounded end caps pulled flush to `start`/`end` by default.

    Two differences from the original, both because this module's callers care about hitting a
    real-world length and the character's limbs never had to. First, the segment count: the
    original defaults to 24, tuned for the character's own limbs, and a barbell bar at 24
    segments is indistinguishable from one at 12 in a contact-sheet render — every spare segment
    on every rod in a fourteen-part library costs face budget this module does not have against
    the brief's ~2000-face ceiling per prop.

    Second, and the one that actually changes a measurement: a capsule is a cylinder plus a ball
    on each end, which means its total reach along its own axis is the distance from `start` to
    `end` PLUS one more `radius` at each tip — the ball sits centred ON the endpoint and bulges
    past it. On the character that overshoot is invisible, folded into a joint that is already
    meant to look like a rounded ball. On a barbell it is not: the first render of this module's
    barbell, built with the sleeve literally ending at the bar's intended 2.2 m tip, measured
    2.25 m, because that sleeve's own rounded cap pushed its visible end another 25 mm past
    where the collar and the module's own real-world dimension table said the bar should stop.
    Insetting `start` and `end` by `radius` before handing them to the underlying primitive — so
    the ball's surface, not its centre, lands on the coordinate the caller asked for — makes
    every rod in this file flush to the length its caller specified, same as the brief's own
    reference dimensions assume. `flush=False` restores the original behaviour, kept for a
    caller that wants the overshoot (there is none in this module, but there was no reason to
    make it impossible).
    """
    if flush:
        start_v, end_v = Vector(start), Vector(end)
        axis = end_v - start_v
        length = axis.length
        if length > 1e-9:
            inset = min(radius, length * 0.475)
            direction = axis / length
            start = tuple(start_v + direction * inset)
            end = tuple(end_v - direction * inset)
    return make_character.capsule(name, start, end, radius, mat, segments=segments)


# --- props -----------------------------------------------------------------------------------

def dumbbell(height, kg=10):
    """A hex dumbbell: knurled-looking steel handle, two hex heads. Origin: the centre of the
    handle, matching `barbell`'s convention, with the handle's long axis along X.

    Head size scales with weight by (kg / 10) ** (1/3) off the 10 kg anchor in the module
    docstring — volume scaling at roughly constant density, which is the shape of how a real
    dumbbell set grows (see the docstring for why this is a rationale and not a spec)."""
    unit = height / 1.75
    metal, rubber, accent = _materials()
    scale = (kg / 10.0) ** (1.0 / 3.0)
    head_radius = DUMBBELL_HEAD_RADIUS_10KG * scale * unit
    head_thickness = DUMBBELL_HEAD_THICKNESS_10KG * scale * unit
    handle_radius = (DUMBBELL_HANDLE_DIAMETER / 2) * unit
    half_handle = (DUMBBELL_HANDLE_LENGTH / 2) * unit

    parts = [_origin("dumbbell")]
    parts.append(capsule("handle", (-half_handle, 0, 0), (half_handle, 0, 0), handle_radius,
                         metal, segments=10))
    for sign in (-1, 1):
        head_centre = (sign * (half_handle + head_thickness / 2), 0, 0)
        parts.append(_disc(f"head.{'l' if sign < 0 else 'r'}", head_centre, head_radius,
                           head_thickness, rubber, hex=True, axis="X"))
        # A small accent cap on the outer face only, the cartoon's equivalent of the colour-coded
        # end plate a commercial rubber hex set uses to mark its weight at a glance.
        cap_centre = (sign * (half_handle + head_thickness - 0.004 * unit), 0, 0)
        parts.append(_disc(f"cap.{'l' if sign < 0 else 'r'}", cap_centre, head_radius * 0.55,
                           0.006 * unit, accent, hex=True, axis="X"))
    return _join("dumbbell", parts)


def barbell(height, plates=True):
    """An Olympic barbell: bar, inner collars, and (unless `plates=False`) a pair of plates per
    side. Origin: the centre of the bar, as named in the brief. Built lying along X."""
    unit = height / 1.75
    metal, rubber, accent = _materials()
    bar_radius = (BARBELL_BAR_DIAMETER / 2) * unit
    half_bar = (BARBELL_LENGTH / 2) * unit
    sleeve_radius = (BARBELL_SLEEVE_DIAMETER / 2) * unit
    sleeve_length = BARBELL_SLEEVE_LENGTH * unit
    collar_radius = (COLLAR_DIAMETER / 2) * unit
    collar_thickness = COLLAR_THICKNESS * unit
    plate_radius = (PLATE_DIAMETER / 2) * unit
    plate_thickness = PLATE_THICKNESS * unit

    parts = [_origin("barbell")]
    parts.append(capsule("bar", (-half_bar, 0, 0), (half_bar, 0, 0), bar_radius, metal,
                         segments=12))
    # The sleeve is the fatter stub the plates actually load onto; built as its own cylinder
    # rather than just thickening the whole bar, because the knurled grip in the centre is
    # meant to stay the thin 28 mm bar the brief specifies.
    for sign in (-1, 1):
        sleeve_start = sign * (half_bar - sleeve_length)
        sleeve_end = sign * half_bar
        parts.append(capsule(f"sleeve.{'l' if sign < 0 else 'r'}", (sleeve_start, 0, 0),
                             (sleeve_end, 0, 0), sleeve_radius, metal, segments=12))
        collar_centre = (sign * (half_bar - sleeve_length - collar_thickness / 2), 0, 0)
        parts.append(_disc(f"collar.{'l' if sign < 0 else 'r'}", collar_centre, collar_radius,
                           collar_thickness, metal, hex=False, axis="X"))
        if plates:
            plate_centre = (sign * (half_bar - sleeve_length / 2), 0, 0)
            parts.append(_disc(f"plate.{'l' if sign < 0 else 'r'}", plate_centre, plate_radius,
                               plate_thickness, rubber, hex=False, axis="X"))
            # A thin accent band around the plate's rim, the cartoon's version of the painted
            # ring a gym marks a plate's weight class with.
            band_centre = plate_centre
            parts.append(_disc(f"band.{'l' if sign < 0 else 'r'}", band_centre,
                               plate_radius * 1.0, plate_thickness * 0.18, accent,
                               hex=False, axis="X"))
    return _join("barbell", parts)


def flat_bench(height):
    """A flat bench: pad, base column, two feet. Origin: floor level, centred under the pad."""
    unit = height / 1.75
    metal, rubber, _ = _materials()
    pad_length = BENCH_PAD_LENGTH * unit
    pad_width = BENCH_PAD_WIDTH * unit
    bench_height = BENCH_HEIGHT * unit
    leg = BENCH_LEG_THICKNESS * unit

    parts = [_origin("flat_bench")]
    pad_thickness = 0.08 * unit
    parts.append(_pad("pad", (0, 0, bench_height - pad_thickness / 2),
                      (pad_length, pad_width, pad_thickness), rubber))
    # One central column rather than four legs: this is a stand-in for a bench, not a parts
    # catalogue, and a single rounded post reads just as clearly in silhouette from across a
    # contact sheet while costing a fraction of the faces.
    parts.append(capsule("column", (0, 0, 0), (0, 0, bench_height - pad_thickness), leg / 2,
                         metal, segments=10))
    for sign in (-1, 1):
        foot_centre = (sign * pad_length * 0.32, 0, leg * 0.3)
        parts.append(_pad(f"foot.{'l' if sign < 0 else 'r'}", foot_centre,
                          (leg * 2.2, pad_width * 0.7, leg * 0.6), metal))
    return _join("flat_bench", parts)


def incline_bench(height, degrees=75):
    """An adjustable incline bench: seat pad, angled back pad, upright frame. Origin: floor
    level, centred under the seat. `degrees` is the back pad's angle off the floor (0 = flat,
    90 = vertical), the usual way an adjustable bench's incline setting is described."""
    unit = height / 1.75
    metal, rubber, _ = _materials()
    seat_length = INCLINE_SEAT_LENGTH * unit
    back_length = INCLINE_BACK_LENGTH * unit
    pad_width = INCLINE_PAD_WIDTH * unit
    bench_height = BENCH_HEIGHT * unit
    pad_thickness = 0.08 * unit
    tilt = math.radians(degrees)

    parts = [_origin("incline_bench")]
    seat_centre_x = -seat_length * 0.15
    parts.append(_pad("seat", (seat_centre_x, 0, bench_height - pad_thickness / 2),
                      (seat_length, pad_width, pad_thickness), rubber))
    # The back pad hinges from the rear edge of the seat and tilts up from there, so its own
    # centre sits half its length away from that hinge, lifted and pushed back by the tilt's
    # sine and cosine rather than guessed — the geometry a hinge actually describes.
    hinge = Vector((seat_centre_x - seat_length / 2, 0, bench_height))
    back_dir = Vector((-math.cos(tilt), 0, math.sin(tilt)))
    back_centre = hinge + back_dir * (back_length / 2)
    # A rotation about Y maps the X-Z plane to itself, so with `pad_width` on local X and
    # `back_length` on local Z, both start inside that plane and stay there, swinging together
    # as a single rigid rectangle — the backrest's actual width stays visible to a camera
    # looking down -Y instead of turning edge-on to it. Thickness, left on local Y, becomes the
    # pad's only extent along the camera's own view axis, which is what keeps its broad face
    # turned toward the camera at any recline angle rather than swinging away from it (the same
    # reasoning `leg_press` uses for its own back pad and foot plate, and for the same reason:
    # a version that put thickness on the swept axis instead rendered as a thin dark sliver).
    #
    # `_pad` builds its long axis (here `back_length`) along local Z before any rotation, and a
    # rotation about Y by angle phi swings that local Z axis to (sin(phi), 0, cos(phi)) in world
    # space (checked against Blender's own convention, not assumed). Setting
    # phi = tilt - 90 degrees makes that resolve to (-cos(tilt), 0, sin(tilt)), which is
    # `back_dir` above — the one angle that makes the pad's own orientation agree with the
    # direction already used to place its centre, rather than mirrored across the hinge. An
    # earlier version got this sign backwards (`90 - tilt`) and the backrest rendered mirrored
    # off at a diagonal, pointing away from where its own centre had been placed.
    parts.append(_pad("back", tuple(back_centre), (pad_width, pad_thickness, back_length),
                      rubber, rotation=(0.0, tilt - math.radians(90), 0.0)))
    parts.append(capsule("frame.front", (seat_centre_x + seat_length * 0.3, 0, 0),
                         (seat_centre_x + seat_length * 0.3, 0, bench_height - pad_thickness),
                         0.025 * unit, metal, segments=10))
    frame_back = hinge
    parts.append(capsule("frame.back", (frame_back.x, 0, 0), tuple(frame_back),
                         0.025 * unit, metal, segments=10))
    return _join("incline_bench", parts)


def pullup_bar(height):
    """A free-standing pull-up bar: horizontal bar on two uprights, each with a small base foot.
    Origin: floor level, midway between the two uprights."""
    unit = height / 1.75
    metal, _, _ = _materials()
    bar_height = PULLUP_BAR_HEIGHT * unit
    span = PULLUP_BAR_SPAN * unit
    bar_radius = (PULLUP_BAR_DIAMETER / 2) * unit
    upright_radius = (PULLUP_UPRIGHT_DIAMETER / 2) * unit

    parts = [_origin("pullup_bar")]
    parts.append(capsule("bar", (-span / 2, 0, bar_height), (span / 2, 0, bar_height), bar_radius,
                         metal, segments=12))
    for sign in (-1, 1):
        x = sign * span / 2
        parts.append(capsule(f"upright.{'l' if sign < 0 else 'r'}", (x, 0, 0), (x, 0, bar_height),
                             upright_radius, metal, segments=10))
        parts.append(_pad(f"foot.{'l' if sign < 0 else 'r'}", (x, 0, upright_radius * 0.6),
                          (upright_radius * 5, upright_radius * 5, upright_radius * 1.2), metal))
    return _join("pullup_bar", parts)


def lat_pulldown(height, drop=None):
    """A seated lat pulldown: seat, thigh pad, upright column, top pulley, wide bar on a cable.
    Origin: floor level, centred under the seat, facing +Y like the character.

    `drop` is how much cable hangs below the pulley, in metres, defaulting to `LAT_BAR_DROP`.
    It exists because the bar's height is the one dimension of this machine that is not a
    property of the machine: a real pulldown's cable pays out, so the bar is wherever the user's
    hands have taken it. With the drop fixed, a caller staging this prop on a particular figure
    had to choose between standing the machine on the floor (and leaving the bar floating above
    the hands) or putting the bar in the hands (and burying the mast) — which is exactly the
    choice the first staging pass had to make, and it made it badly in both directions. Solve
    `LAT_COLUMN_HEIGHT - LAT_PULLEY_INSET - drop` for the hand height and both are satisfied.
    """
    unit = height / 1.75
    metal, rubber, _ = _materials()
    seat_height = LAT_SEAT_HEIGHT * unit
    seat_size = tuple(v * unit for v in LAT_SEAT_SIZE)
    column_height = LAT_COLUMN_HEIGHT * unit
    bar_width = LAT_BAR_WIDTH * unit
    cable_radius = (CABLE_DIAMETER / 2) * unit
    column_radius = (COLUMN_DIAMETER / 2) * unit

    parts = [_origin("lat_pulldown")]
    seat_x = -0.35 * unit
    seat_pad_thickness = 0.10 * unit
    parts.append(_pad("seat", (seat_x, 0, seat_height - seat_pad_thickness / 2),
                      (seat_size[0], seat_size[1], seat_pad_thickness), rubber))
    # A seat with nothing under it reads as a pad floating in mid-air next to the machine, not
    # part of it — a first render did exactly that. A single support post, the same stand-in
    # `flat_bench` uses for its own legs, is enough to read as "this pad is mounted to the
    # machine" without modelling the actual welded frame a real one uses.
    parts.append(capsule("seat_post", (seat_x, 0, 0), (seat_x, 0, seat_height - seat_pad_thickness),
                         0.025 * unit, metal, segments=10))
    thigh_z = LAT_THIGH_PAD_HEIGHT * unit
    parts.append(_pad("thigh_pad", (seat_x, seat_size[1] * 0.1, thigh_z),
                      (seat_size[0] * 0.9, 0.12 * unit, 0.10 * unit), rubber))
    column_x = seat_x - seat_size[0] * 0.7
    parts.append(capsule("column", (column_x, 0, 0), (column_x, 0, column_height), column_radius,
                         metal, segments=12))
    # A floor foot under the column, the same stand-in `pullup_bar` uses for its own uprights —
    # round 1 of review did not flag the column itself as floating, but it has the identical
    # "mast with nothing visibly anchoring it" problem the seat had, so it gets the same fix.
    parts.append(_pad("column_foot", (column_x, 0, column_radius * 0.6),
                      (column_radius * 5, column_radius * 5, column_radius * 1.2), metal))

    pulley_centre = (column_x, 0, column_height - LAT_PULLEY_INSET * unit)
    # A bare disc at the top of the mast reads as a wheel floating next to the column, not a
    # pulley the cable actually runs through — review's finding 2. A housing — a box the pulley
    # sits inside, open only on the underside the cable drops from — is what makes it read as
    # the mechanism the cable comes FROM rather than a coincidental circle nearby. Sized a
    # little larger than the pulley disc on every side so the disc reads as housed, not
    # intersecting a box of the same size.
    # A head that reaches out over the seat rather than a box sitting on the mast's own axis,
    # so the cable visibly drops from its outer end instead of from somewhere inside the mast.
    cable_x = column_x + LAT_CABLE_REACH * unit
    housing_size = 0.07 * unit * 2.3
    parts.append(_pad("pulley_housing",
                      ((column_x + cable_x) / 2, 0, pulley_centre[2]),
                      (abs(cable_x - column_x) + housing_size * 0.7, housing_size * 0.8,
                       housing_size * 0.7), metal))
    parts.append(_disc("pulley", pulley_centre, 0.07 * unit, 0.035 * unit, metal, axis="Y"))
    # The bar hangs `drop` below the pulley. The default is a modest cable's length rather than
    # all the way down at hand height, because an unstaged prop has nobody gripping it — enough
    # cable to read as "hanging", not a guess at mid-rep. A caller staging it on a figure passes
    # the drop that puts the bar in that figure's hands.
    bar_drop = (LAT_BAR_DROP if drop is None else drop) * unit
    if not 0.0 <= bar_drop <= column_height - LAT_PULLEY_INSET * unit:
        raise ValueError(
            f"lat_pulldown drop {bar_drop / unit:.3f} m does not fit between the pulley at "
            f"{(column_height - LAT_PULLEY_INSET * unit) / unit:.3f} m and the floor; the bar "
            f"would hang above its own pulley or below the ground")
    bar_z = column_height - LAT_PULLEY_INSET * unit - bar_drop
    # Off the head's outer end on the seat's side, not down the middle of the mast — see
    # LAT_CABLE_REACH. The pulley's own plane is X-Z (it is built with its axis along Y), so +X
    # is the side a cable can actually leave it on, and +X is also where the user sits.
    parts.append(capsule("cable", (cable_x, 0, pulley_centre[2]), (cable_x, 0, bar_z),
                         cable_radius, metal, segments=6))
    # Along local Y, across the user, not along local X. This machine's own long axis is X —
    # the seat sits at x = -0.35 with the mast further along -X, so its user faces -X — and the
    # bar was built along that same axis, which put a 1.10 m lat bar running front-to-back over
    # the user's head instead of across both hands. Nothing in the prop's own verification sheet
    # could show it: the tile is rendered from one fixed camera and a bar pointing at the lens
    # looks like a bar seen end-on. It showed the moment the machine was turned to face a posed
    # figure. A lat bar is gripped with the hands wide apart either side of the head, so it is
    # perpendicular to the direction the user faces.
    parts.append(capsule("bar", (cable_x, -bar_width / 2, bar_z),
                         (cable_x, bar_width / 2, bar_z), 0.014 * unit, metal, segments=10))
    return _join("lat_pulldown", parts)


def cable_tower(height, attachment="bar", drop=None):
    """A single cable-tower column with a high pulley and a cable, ending in either a straight
    bar (`attachment="bar"`) or a rope's two hanging tails (`attachment="rope"`). Origin: floor
    level, at the base of the column.

    `drop` is the length of cable below the pulley, in metres, defaulting to `CABLE_DROP`. Same
    reason as `lat_pulldown`'s: on a cable machine the attachment is wherever the user has
    pulled it to, so a staged tower solves this for the figure's own hand height and then stands
    on the floor like a real one, instead of being sunk until its fixed cable reaches.
    """
    unit = height / 1.75
    metal, rubber, _ = _materials()
    column_height = CABLE_TOWER_HEIGHT * unit
    column_radius = (COLUMN_DIAMETER / 2) * unit
    cable_radius = (CABLE_DIAMETER / 2) * unit

    parts = [_origin("cable_tower")]
    parts.append(capsule("column", (0, 0, 0), (0, 0, column_height), column_radius, metal,
                         segments=12))
    pulley_centre = (0, 0, column_height - CABLE_PULLEY_INSET * unit)
    # Axis along X, so the wheel's own plane is Y-Z and the cable can leave its rim along -Y,
    # which is the side this tower's user stands on (the attachment bar runs along local X,
    # across both hands, so the user faces the column along Y).
    #
    # Housed, for the reason round 1 of review gave for housing the LAT tower's pulley and which
    # applies here word for word: a bare disc barely wider than the column it sits on reads as a
    # rounded pole top, not as the mechanism a cable comes from. The lat tower got the fix and
    # this one did not, because this one's single-camera verification tile never showed a cable
    # at all — see CABLE_TOWER_REACH. Same box, turned a quarter turn because this pulley's
    # plane is Y-Z where the lat tower's is X-Z.
    housing = 0.06 * unit * 2.3
    reach = CABLE_TOWER_REACH * unit
    parts.append(_pad("pulley_housing", (0, -reach / 2, pulley_centre[2]),
                      (housing * 0.8, reach + housing * 0.6, housing * 0.7), metal))
    parts.append(_disc("pulley", pulley_centre, 0.06 * unit, 0.03 * unit, metal, axis="X"))
    cable_drop = (CABLE_DROP if drop is None else drop) * unit
    if not 0.0 <= cable_drop <= column_height - CABLE_PULLEY_INSET * unit:
        raise ValueError(
            f"cable_tower drop {cable_drop / unit:.3f} m does not fit between the pulley at "
            f"{(column_height - CABLE_PULLEY_INSET * unit) / unit:.3f} m and the floor")
    attach_z = column_height - CABLE_PULLEY_INSET * unit - cable_drop
    cable_y = -CABLE_TOWER_REACH * unit
    parts.append(capsule("cable", (0, cable_y, pulley_centre[2]), (0, cable_y, attach_z),
                         cable_radius, metal, segments=6))

    if attachment == "rope":
        rope_length = CABLE_ROPE_LENGTH * unit
        splay = CABLE_ROPE_SPLAY * unit
        for sign in (-1, 1):
            tail_end = (sign * splay / 2, cable_y, attach_z - rope_length)
            parts.append(capsule(f"rope.{'l' if sign < 0 else 'r'}", (0, cable_y, attach_z),
                                 tail_end, 0.010 * unit, rubber, segments=6))
    else:
        bar_length = CABLE_BAR_LENGTH * unit
        parts.append(capsule("bar", (-bar_length / 2, cable_y, attach_z),
                             (bar_length / 2, cable_y, attach_z),
                             0.013 * unit, metal, segments=10))
    return _join("cable_tower", parts)


def leg_press(height):
    """A simplified 45-degree leg press: a floor base, two sled rails, a low seat and a
    reclined back pad behind it, and a foot plate at the top of the rails' travel. Origin:
    floor level, centred under the base frame, facing +Y.

    Round 1 of review flagged this prop as unrecognisable — a flat plate on a stick, reading as
    a shovel — for two compounding reasons, both now fixed.

    First, the tilt's own plane. Built tilting in the X-Z plane — sideways as the camera sees it
    — rather than reclining away from the camera in depth. An earlier version did the physically
    obvious thing and tilted the rails back in Y, away from the viewer, and it rendered looking
    like a flat table: the one camera this module is asked to reuse (`make_character.stage`)
    looks straight down -Y, so a recline built in that same axis is seen almost edge-on,
    foreshortened into a sliver. Turning the whole machine side-on instead — which is,
    incidentally, the angle a real leg press is usually photographed from, for the same reason —
    puts the 45-degree angle in the plane the camera can actually show.

    Second, the two rails' own separation. A sled runs on two parallel rails, and a first fix of
    this prop separated them along Y — lateral, in the user's own frame, but DEPTH from the
    camera's — so they collapsed into what looked like one rod from the side, the same
    foreshortening problem as the tilt, just applied to the wrong dimension this time. `rail_sep`
    below separates them along X instead: the axis `make_character.stage`'s camera does not look
    down (it looks down -Y), so two rails offset by a constant X shift render as two distinct,
    parallel diagonal lines rather than one. Everything that bridges the two rails — the base
    frame, the seat, the back pad, the foot plate — is built wide enough in X to visibly span
    the gap between them, the way a real cross-member would.

    Reduced on purpose to the parts that make the machine legible — see "What is NOT modelled"
    in the module docstring; there is no carriage/sled mechanism, no weight horn, no seat belt.

    A `travel=` parameter was fitted here and taken out again, by the same reasoning that gave
    `lat_pulldown` and `cable_tower` their `drop=`: a sled moves, so where the foot plate sits
    is a fact about the rider rather than about the machine. It does not work, and the reason is
    this prop's own deliberate geometry. Its rails are offset 0.18 m to one side of the seat and
    rise at 45 degrees from a base behind it (see the note above on why they are built that way
    — it is so they read from the one camera this module's own contact sheet uses), so the line
    they trace does not pass near a seated rider's feet at all. Solving for the travel that puts
    the plate closest to the feet leaves it 0.205 of a body height away, up against the rider's
    SHINS, which renders as a slab through both legs — measurably and visibly worse than leaving
    it at the top of the rails. Fixing it properly means mounting the plate off the rails the
    way a real sled carries it, which is a redesign of the prop, not a parameter."""
    unit = height / 1.75
    metal, rubber, _ = _materials()
    rail_length = LEG_PRESS_RAIL_LENGTH * unit
    back_w, back_l = (v * unit for v in LEG_PRESS_BACK_PAD)
    plate_w, plate_l = (v * unit for v in LEG_PRESS_FOOT_PLATE)
    tilt = math.radians(45)
    rail_sep = 0.34 * unit
    frame_thickness = 0.06 * unit

    parts = [_origin("leg_press")]
    # A floor-level base, so the rails read as bolted to something rather than floating — an
    # earlier version started the rails in mid-air with nothing under them at all.
    parts.append(_pad("base_frame", (0, -0.05 * unit, frame_thickness / 2),
                      (rail_sep + 0.18 * unit, 0.55 * unit, frame_thickness), metal))

    rail_base = Vector((0, -0.18 * unit, frame_thickness))
    rail_dir = Vector((math.cos(tilt), 0, math.sin(tilt)))
    rail_top = rail_base + rail_dir * rail_length
    for sign in (-1, 1):
        offset = Vector((sign * rail_sep / 2, 0, 0))
        parts.append(capsule(f"rail.{'l' if sign < 0 else 'r'}", tuple(rail_base + offset),
                             tuple(rail_top + offset), 0.035 * unit, metal, segments=10))

    # The seat, low and between the rails, roughly where the hips actually land on a 45-degree
    # machine — close to the floor, well behind where the rails themselves start.
    seat_centre = rail_base + Vector((-0.12 * unit, 0.16 * unit, 0.04 * unit))
    parts.append(_pad("seat", tuple(seat_centre),
                      (rail_sep + 0.10 * unit, 0.32 * unit, 0.09 * unit), rubber))

    # The back pad hinges from the rear of the seat and reclines up from there — same hinge
    # geometry `incline_bench` uses, with `back_dir` carrying the sine/cosine rather than a
    # guessed offset.
    back_dir = Vector((-math.cos(tilt), 0, math.sin(tilt)))
    hinge = seat_centre + Vector((-0.06 * unit, -0.14 * unit, 0.05 * unit))
    back_centre = hinge + back_dir * (back_l / 2)
    # Width on X and length on Z, both swept together by the Y rotation, rather than width on
    # the fixed Y axis the way a non-reclining pad would use — the reason is the camera, not the
    # pad. A rotation about Y maps the X-Z plane to itself, so with width and length both
    # starting in that plane, they stay in it and swing together as a rigid rectangle; thickness,
    # left on Y, is the pad's only extent along the camera's own viewing axis, which is what
    # keeps the pad's broad face turned toward the camera at any recline angle instead of
    # swinging edge-on toward it (checked by rendering the alternative, which came out as a thin
    # dark sliver). `back_w` here is set wide enough to visibly bridge both rails, same reasoning
    # as the base frame and the foot plate.
    parts.append(_pad("back_pad", tuple(back_centre),
                      (max(back_w, rail_sep + 0.10 * unit), 0.09 * unit, back_l),
                      rubber, rotation=(0.0, tilt - math.pi / 2, 0.0)))

    # The foot plate at the top of rail travel: width on X, thickness on Y, same reasoning and
    # same axis pattern as the back pad, so its wide face stays turned toward the camera rather
    # than swinging to present its edge. (A real foot plate's own flat face actually points back
    # down the rails, not at the viewer; this module's camera is fixed and straight-on, and the
    # brief's own bar is legibility in the verification render over mechanical fidelity here —
    # see "What is NOT modelled".) `-tilt` angles it to roughly mirror the rails' own incline
    # rather than sitting flat, which is what makes it read as mounted at the end of the travel
    # instead of just floating above it.
    plate_centre = rail_top + Vector((0.02 * unit, 0, 0.05 * unit))
    parts.append(_pad("foot_plate", tuple(plate_centre),
                      (max(plate_w, rail_sep + 0.14 * unit), 0.06 * unit, plate_l),
                      metal, rotation=(0.0, -tilt, 0.0)))
    return _join("leg_press", parts)


def squat_rack(height):
    """A squat rack reduced to its two uprights and their J-hooks, no crossbar (per the brief —
    this is a squat stand, not a full power cage). Origin: floor level, midway between the two
    uprights."""
    unit = height / 1.75
    metal, _, accent = _materials()
    rack_height = SQUAT_RACK_HEIGHT * unit
    span = SQUAT_RACK_SPAN * unit
    upright_radius = (SQUAT_RACK_UPRIGHT / 2) * unit
    hook_height = SQUAT_RACK_HOOK_HEIGHT * unit

    parts = [_origin("squat_rack")]
    for sign in (-1, 1):
        x = sign * span / 2
        side = "l" if sign < 0 else "r"
        parts.append(capsule(f"upright.{side}", (x, 0, 0), (x, 0, rack_height), upright_radius,
                             metal, segments=10))
        parts.append(_pad(f"foot.{side}", (x, 0, upright_radius * 0.6),
                          (upright_radius * 5, upright_radius * 5, upright_radius * 1.2), metal))
        # A J-hook reaches forward (toward +Y, the way the character faces) and up, the shape a
        # hook needs to catch a bar from below and hold it from sliding off toward the lifter.
        hook_out = capsule(f"hook.{side}.out", (x, 0, hook_height), (x, 0.09 * unit, hook_height),
                           0.016 * unit, accent, segments=8)
        hook_up = capsule(f"hook.{side}.up", (x, 0.09 * unit, hook_height),
                          (x, 0.09 * unit, hook_height + 0.07 * unit), 0.016 * unit, accent,
                          segments=8)
        parts += [hook_out, hook_up]
    return _join("squat_rack", parts)


def floor_mat(height):
    """A thin rounded exercise mat. Origin: floor level, centred on the mat, long axis along Y
    so a figure lying on it (a plank, a sit-up) is oriented the way the character already
    faces."""
    unit = height / 1.75
    _, _, accent = _materials()
    length = FLOOR_MAT_LENGTH * unit
    width = FLOOR_MAT_WIDTH * unit
    thickness = FLOOR_MAT_THICKNESS * unit
    mat = _pad("floor_mat", (0, 0, thickness / 2), (width, length, thickness), accent,
              bevel=thickness * 0.4, segments=2)
    return mat


# --- attachment --------------------------------------------------------------------------------

def hold(obj, rig, bone_name, offset=(0, 0, 0), rotation=(0, 0, 0)):
    """Parent `obj` to a posed armature bone so it follows that bone through an animation, with
    `offset=(0, 0, 0)` meaning "at the bone's head" rather than Blender's own default.

    Blender's bone parenting (`parent_type="BONE"`) places the child at the bone's TAIL, not its
    head — documented behaviour, and a well-known trap, because every other intuition about
    parenting says a child sits where its parent IS, and a bone's own head is where you would
    point to if asked where it "is". A bone-parented child's local space has the bone's TAIL at
    its origin, with the child's own local Y axis running back up the bone toward the head (the
    same convention an edit-bone's local space already uses internally — Y runs head-to-tail, so
    the inverse, tail-to-head, is -Y). Shifting the child by the bone's length along -Y before
    applying the caller's own `offset` undoes exactly that displacement, so an `offset` of zero
    reads, as documented, as "at the head".

    Which real joint that is depends on the bone, not on this function — `make_character.BONES`
    defines "hand.left"/"hand.right" as running from the WRIST to the hand (confirmed against a
    bound rig, not assumed), so `offset=(0, 0, 0)` on a hand bone lands a prop at the wrist, which
    is in fact a reasonable default grip centre for a dumbbell, not the fingertips. A caller that
    wants the prop nearer the tail instead — further into the palm — slides it back `offset`'s
    own length along local Y, same as any other fine placement; this function only fixes the
    well-known head/tail trap, it does not also guess where on the bone a prop should sit.
    """
    obj.parent = rig
    obj.parent_type = "BONE"
    obj.parent_bone = bone_name
    obj.matrix_parent_inverse = Matrix.Identity(4)
    length = rig.data.bones[bone_name].length
    obj.location = (offset[0], offset[1] - length, offset[2])
    obj.rotation_euler = rotation
    return obj


def stand(obj, location=(0, 0, 0), rotation=(0, 0, 0)):
    """Place floor-standing gear with no bone and no parent.

    Every floor prop in this module is built with local Z = 0 already at the plane where it
    meets the floor (see "Anchors" in the module docstring), so standing one up is nothing more
    than setting its transform — there is no offset left to compensate for, which is the point
    of having picked that anchor in the first place.
    """
    obj.location = location
    obj.rotation_euler = rotation
    return obj


# --- verification ------------------------------------------------------------------------------

PROPS = [
    ("dumbbell", lambda h: dumbbell(h, kg=10)),
    ("barbell", lambda h: barbell(h, plates=True)),
    ("barbell_empty", lambda h: barbell(h, plates=False)),
    ("flat_bench", flat_bench),
    ("incline_bench", lambda h: incline_bench(h, degrees=75)),
    ("pullup_bar", pullup_bar),
    ("lat_pulldown", lat_pulldown),
    ("cable_tower_bar", lambda h: cable_tower(h, attachment="bar")),
    ("cable_tower_rope", lambda h: cable_tower(h, attachment="rope")),
    ("leg_press", leg_press),
    ("squat_rack", squat_rack),
    ("floor_mat", floor_mat),
]

DISPLAY_NAMES = {
    "dumbbell": "Dumbbell (10 kg)",
    "barbell": "Barbell (loaded)",
    "barbell_empty": "Barbell (empty)",
    "flat_bench": "Flat Bench",
    "incline_bench": "Incline Bench (75 deg)",
    "pullup_bar": "Pull-up Bar",
    "lat_pulldown": "Lat Pulldown",
    "cable_tower_bar": "Cable Tower (bar)",
    "cable_tower_rope": "Cable Tower (rope)",
    "leg_press": "Leg Press",
    "squat_rack": "Squat Rack",
    "floor_mat": "Floor Mat",
}

# Props a hand holds rather than the floor, shown in the contact sheet at roughly hand height
# next to the reference figure instead of resting on the ground at their own documented anchor
# (the centre of the bar). "Is this the right length for the person holding it" is judged better
# roughly where the thing is actually held than lying on the floor with its centre buried in it.
HELD_PROPS = {"dumbbell", "barbell", "barbell_empty"}


#  The character's measured height, for this module's own verification renders only.
#
# Not derived here, because two derivations have now each been wrong in a different way. A first
# version reused `make_character.stage`'s cheap camera-framing estimate (topmost skeleton joint
# plus a fixed multiple of `HEAD_RADIUS`) and got 1.14 units. A second version built the actual
# procedural mascot (`make_character.build_body`) and measured its real `bound_box`, getting
# 1.05 — closer, but still wrong, because that mascot is a stand-in for development and not the
# mesh this library actually has to sit next to. The number that matters is the real, imported,
# fitted character's own height, and the person running this pipeline measured that one directly
# off the actual mesh (46,261 verts, z from 0.0000 to 0.9995, single mesh, world transform
# applied) rather than off a proxy — a measurement this module has no way to reproduce itself,
# since it does not import `fit_model` or touch the real character mesh. 0.9995 is taken on that
# authority, not re-derived. The library's own functions are unaffected either way: every one of
# them takes `height` as an argument and this constant only supplies the default for the
# contact sheet this file renders on its own.
CHARACTER_HEIGHT = 0.9995


def character_height():
    """The reference height this module's own verification renders are built at. See
    `CHARACTER_HEIGHT` for where the number comes from. Kept as a function, not just the
    constant, so a caller verifying against a different build of the character can monkeypatch
    or re-point this one place rather than every call site."""
    return CHARACTER_HEIGHT


# A pale neutral blue-grey, deliberately outside this module's three prop materials (mid-grey
# metal, near-black rubber, warm accent) so a reader can never mistake the scale reference for a
# piece of equipment that ships with the library.
REFERENCE = (0.70, 0.76, 0.80, 1.0)


def reference_figure(height):
    """A plain capsule standing exactly `height` Blender units tall, for scale only — never
    joined to a prop and never shipped as part of the equipment library.

    Round 1 of review (finding 3): a wide shot of every prop laid out at once left each one a
    few percent of the frame, so the question a contact sheet exists to answer — "is this
    barbell the right size next to the person who lifts it" — could not actually be answered by
    looking at it; a reader had to go open the individual debug renders instead. `main()` now
    stands one of these beside every prop's own tile, built at radius 0.08 * unit, a generic
    standing girth, so the scale relationship is visible without reading the dimension table.
    """
    unit = height / 1.75
    radius = 0.08 * unit
    mat = bpy.data.materials.get("equip_reference") or \
        make_character.material("equip_reference", REFERENCE, roughness=0.6)
    return capsule("reference_figure", (0, 0, radius), (0, 0, height - radius), radius, mat,
                   segments=16)


def _label(text, unit, location):
    """A flat 3D text object at `location`, oriented to read correctly from `make_character.
    stage`'s camera — derived from first principles and checked against a render, not assumed.

    A freshly added Blender text object reads along its own local +X with its readable face's
    normal on local +Z. This module's camera sits at positive Y looking down -Y, and — checked
    directly, the same way `frame_camera`'s own docstring reports it — that camera's local +X
    maps to WORLD -X, so the render mirrors world X left-to-right. A naive 90-degree spin to
    stand the text upright renders it backwards: tried, and "ABC-123" came back as mirrored,
    reversed-order glyphs. The matrix below is instead built from where each local axis needs to
    land in world space: local X (reading direction) to world -X, so reading order runs toward
    screen-right; local Y (the axis letters stand up along) to world +Z; local Z (the readable
    face's own normal) to world +Y, facing the camera. Checked to have a positive determinant
    first — so the glyphs are rotated, not mirror-flipped — then confirmed by an actual render
    of "ABC-123" coming back correctly before being trusted here.

    One more trap, caught only because the floor mat's tile came back with everything crammed
    into one corner: `obj.bound_box` does not update the instant `.data.body` and `.data.size`
    are set — it still reports the stale, default "Text" placeholder's box until Blender
    re-evaluates the curve, which does not happen until something forces it to. `frame_camera`
    reads `bound_box` immediately after this function returns, so without the explicit update
    below it was framing every tile against a phantom box sized for the word "Text" at size 1.0
    rather than this label's actual string and size — usually close enough not to be obvious,
    but wildly wrong once `floor_mat`'s real content was small enough for the phantom box to
    dominate the framing instead of just padding it.
    """
    bpy.ops.object.text_add(location=(0, 0, 0))
    obj = bpy.context.object
    obj.data.body = text
    obj.data.size = 0.16 * unit
    obj.data.align_x = "CENTER"
    obj.data.align_y = "CENTER"
    obj.data.extrude = 0.003 * unit
    mat = bpy.data.materials.get("equip_label") or \
        make_character.material("equip_label", (0.05, 0.05, 0.05, 1.0), roughness=0.5)
    obj.data.materials.append(mat)
    rot = Matrix((Vector((-1, 0, 0)), Vector((0, 0, 1)), Vector((0, 1, 0)))).transposed().to_4x4()
    obj.matrix_world = rot
    obj.location = location
    bpy.context.view_layer.update()
    return obj


def render_tile(name, factory, height, path, resolution, samples):
    """Build one prop plus its scale reference and name label, frame a camera tightly on just
    that group, render it to `path`, and report the PROP's own face count and bounding box (the
    reference figure and the label are scaffolding, not part of what gets measured).

    Each tile clears and rebuilds the whole scene — backdrop, floor, lights, materials — rather
    than sharing one scene across all twelve, which is what the original single wide shot did
    (round 1's finding 3). Sharing a scene is what made every prop a few percent of one frame;
    building and framing one prop at a time is what makes each one fill its own tile instead.
    """
    make_character.clear()
    unit = height / 1.75
    obj = factory(height)
    lift = height * 0.45 if name in HELD_PROPS else 0.0
    stand(obj, location=(0, 0, lift))

    faces = len(obj.data.polygons)
    bbox = bounding_box_metres(obj, unit)

    prop_corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    prop_max_x = max(c.x for c in prop_corners)
    prop_min_x = min(c.x for c in prop_corners)
    gap = 0.22 * unit
    figure = reference_figure(height)
    figure.location = (prop_max_x + gap + 0.08 * unit, 0.0, figure.location.z)

    mid_x = (prop_min_x + figure.location.x) / 2
    label = _label(DISPLAY_NAMES.get(name, name), unit, (mid_x, 0.35 * unit, 0.04 * unit))

    skeleton = make_character.rest_skeleton()
    camera = make_character.stage(skeleton, subject=obj)
    scene = bpy.context.scene
    scene.render.resolution_x, scene.render.resolution_y = resolution
    frame_camera(camera, [obj, figure, label], margin=1.18)

    make_character.use_cycles(scene, samples=samples)
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    return faces, bbox


def composite_grid(tile_paths, columns, tile_resolution, out_path):
    """Stitch rendered tile images into one grid PNG, left to right then top to bottom, with no
    external image library — `bpy.data.images` reads and writes PNGs directly, and `numpy`
    (bundled with Blender, checked before relying on it here) does the pixel-block copying.

    Blender's own `Image.pixels` buffer is bottom-up (row 0 of the flat array is the BOTTOM row
    of the image, the usual OpenGL convention) — checked by writing a two-colour test image and
    looking at which half came out on top, not assumed, because getting this backwards would
    silently flip every tile. Each loaded tile is flipped once into ordinary top-down order to
    make the grid math easy to read, and the whole canvas is flipped back once at the very end
    before it is handed to Blender's own save.
    """
    tile_w, tile_h = tile_resolution
    rows = math.ceil(len(tile_paths) / columns)
    canvas = np.zeros((rows * tile_h, columns * tile_w, 4), dtype=np.float32)
    canvas[:, :, 3] = 1.0

    for i, path in enumerate(tile_paths):
        img = bpy.data.images.load(path)
        tile = np.array(img.pixels[:], dtype=np.float32).reshape(tile_h, tile_w, 4)
        tile = np.flipud(tile)  # Blender's bottom-up -> ordinary top-down, for this function only
        bpy.data.images.remove(img)
        row, col = divmod(i, columns)
        y0, x0 = row * tile_h, col * tile_w
        canvas[y0:y0 + tile_h, x0:x0 + tile_w] = tile

    canvas = np.flipud(canvas)  # back to Blender's bottom-up convention before saving
    out = bpy.data.images.new("contact_sheet", columns * tile_w, rows * tile_h)
    out.pixels = canvas.reshape(-1).tolist()
    out.filepath_raw = out_path
    out.file_format = "PNG"
    out.save()


def bounding_box_metres(obj, unit):
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    xs, ys, zs = ([c[i] for c in corners] for i in range(3))
    return tuple((max(v) - min(v)) / unit for v in (xs, ys, zs))


def frame_camera(camera, objects, margin=1.15):
    """Point `camera` at the combined bounding box of `objects`, backing off far enough that
    both the width AND the height fit, not just the height — and far enough, too, that whatever
    part of the group sits closest to the camera does not loom larger than the margin intended.

    `make_character.stage` picks a camera distance from vertical extent alone, which is the
    right call for the one subject it was written for — a standing figure, taller than it is
    wide. A contact sheet of a dozen props laid out in a grid is the opposite shape, wider than
    it is tall, and framing it by height alone would crop the outer columns clean off. So this
    checks both constraints — the distance needed to fit the height, and the distance needed to
    fit the width given the camera's horizontal field of view, which is derived from the vertical
    one and the render's aspect ratio because `sensor_fit` is VERTICAL — and backs off to
    whichever is larger.

    That much was true of the first version of this function, and it still under-framed
    `floor_mat`'s own tile: the mat is long enough in Y (it is built running front-to-back, not
    side-to-side) that its near edge, and this function's own name label beside it, sit well
    closer to the camera than the group's Y-centre — the one distance this function's original
    width/height check never looked at. Perspective makes anything closer than the distance a
    shot was framed for loom larger than planned, which is exactly what rendered: the mat and
    its label crowded the bottom of the tile, cut off, while the group's own bounding box said
    there should have been margin to spare. The fix backs the camera off by however far the
    nearest point of the group sits closer than the group's own Y-centre, which restores that
    nearest point to roughly the distance the width/height margin was actually computed for.
    """
    corners = []
    for obj in objects:
        corners += [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    xs = [c.x for c in corners]
    ys = [c.y for c in corners]
    zs = [c.z for c in corners]
    centre = Vector(((max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2, (max(zs) + min(zs)) / 2))
    width, height = max(xs) - min(xs), max(zs) - min(zs)

    render = bpy.context.scene.render
    aspect = render.resolution_x / render.resolution_y
    vfov = 2 * math.atan((camera.data.sensor_height / 2) / camera.data.lens)
    hfov = 2 * math.atan(math.tan(vfov / 2) * aspect)
    distance_for_height = (height * margin / 2) / math.tan(vfov / 2)
    distance_for_width = (width * margin / 2) / math.tan(hfov / 2)
    distance = max(distance_for_height, distance_for_width)
    # The camera sits in +Y and looks toward -Y, so the point of the group nearest the camera is
    # the one with the largest Y — restore it to roughly `distance` away, not `distance` minus
    # however far it already sat toward the camera from the group's own Y-centre.
    near_offset = max(0.0, max(ys) - centre.y)
    distance += near_offset

    camera.location = (centre.x, centre.y + distance, centre.z)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--still", default=None, help="render a contact sheet of every prop here")
    parser.add_argument("--samples", type=int, default=24,
                        help="Cycles samples per tile (twelve tiles are rendered, so this is "
                             "kept modest by default)")
    args = parser.parse_args(argv)

    height = character_height()
    unit = height / 1.75
    print(f"character_height={height:.4f} blender units (unit={unit:.4f} per metre)")

    tile_resolution = (400, 400)
    columns = 4

    if args.still:
        with tempfile.TemporaryDirectory(prefix="equipment_tiles_") as tmp:
            tile_paths = []
            for name, factory in PROPS:
                path = str(Path(tmp) / f"{name}.png")
                faces, bbox = render_tile(name, factory, height, path, tile_resolution,
                                          args.samples)
                print(f"{name}: {faces} faces, bbox (m) = "
                      f"{bbox[0]:.3f} x {bbox[1]:.3f} x {bbox[2]:.3f}")
                tile_paths.append(path)
            composite_grid(tile_paths, columns, tile_resolution, args.still)
        print("rendered", args.still)
    else:
        # No render requested: still measure and report every prop, at the tile pipeline's own
        # resolution, just without the (slow) final render or composite step.
        make_character.clear()
        for name, factory in PROPS:
            obj = factory(height)
            faces = len(obj.data.polygons)
            bbox = bounding_box_metres(obj, unit)
            print(f"{name}: {faces} faces, bbox (m) = "
                  f"{bbox[0]:.3f} x {bbox[1]:.3f} x {bbox[2]:.3f}")
            make_character.clear()


if __name__ == "__main__":
    main()
