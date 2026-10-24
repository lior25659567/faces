#!/usr/bin/env python
"""Work out where each muscle sits on the drawn figure, by measuring the figure.

The alternative was to hand-write coordinates against the picture by eye, which is how the first
flat-mannequin version was built and is why its lats were two vertical bars. This measures
instead: it walks the cut-out's alpha channel row by row, finds where the torso ends and the arms
begin, where the legs separate, where the waist is narrowest, and places the muscle shapes against
those landmarks. Redrawing the character with a different seed or a different build moves every
shape with it, because nothing here is a number somebody typed while looking at one picture.

Costs no Workers AI calls. It reads reference/muscle-figure/*.png and writes the polygons the app
fills, in fractions of the picture so they scale with it.

    pipeline/.venv/bin/python pipeline/muscle_regions.py
    pipeline/.venv/bin/python pipeline/muscle_regions.py --preview "Dumbbell Bicep Curl (standing, both arms)"
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIGURES = ROOT / "reference" / "muscle-figure"
MUSCLE_MAP = ROOT / "reference" / "exercises" / "muscle-map.json"
OUT = FIGURES / "regions.json"
APP = ROOT.parent / "faces-ios" / "App" / "Resources"

def spindle(fraction, pinch):
    """How wide a muscle is this far along itself, as a share of its widest.

    A muscle is a spindle: narrow where it starts, full in the belly, narrow where it inserts.
    Every shape here began as a band with straight sides, and blurring a straight-sided band does
    not make it a muscle — it makes a blurred rectangle, which is what the first version in the
    app looked like, a panel sewn onto the vest. Pinching the ends is the whole difference.

    `pinch` is how much of that to apply: 1.0 for a belly like the biceps, less for something
    that really is a sheet, like the abs.
    """
    import math
    return 1.0 - pinch * (1.0 - math.sin(math.pi * min(max(fraction, 0.0), 1.0)) ** 0.55)


def ramp(inset, fraction):
    """An inset that is either fixed or runs from one value at the top of a shape to another at
    the bottom."""
    if isinstance(inset, (tuple, list)):
        return inset[0] + (inset[1] - inset[0]) * fraction
    return inset


# A row has to be this wide to count as a limb rather than as a stray pixel of anti-aliasing that
# survived the cut-out.
MIN_RUN = 6


def runs(row):
    """The separate pieces of figure in one row, as (left, right) pairs."""
    pieces, found = ndimage.label(row)
    out = []
    for index in range(1, found + 1):
        xs = np.where(pieces == index)[0]
        if len(xs) >= MIN_RUN:
            out.append((int(xs.min()), int(xs.max())))
    return out


def landmarks(mask):
    """Where this particular body's parts are, in pixels.

    Everything downstream is expressed against these, so this is the one place that looks at the
    picture. `torso` is the middle piece of each row, which is the body where the arms have
    cleared it and the body-plus-arms where they have not; `arms` is the outer pieces, present
    only on the rows where they are separate. Above the row where the arms clear, the torso run
    includes them, which is why the chest and the back are inset from it rather than drawn to its
    edge."""
    height, width = mask.shape
    filled = mask.any(axis=1)
    top, bottom = int(filled.argmax()), int(height - 1 - filled[::-1].argmax())
    span = bottom - top

    # A piece is an arm when it sits out near the edge of the frame and body when it does not.
    # Classifying by position and not by "outermost piece" is what separates an arm from the far
    # leg: below the hips every row has two pieces near the middle and up to two more outside
    # them, and the first version of this took the outer leg for a forearm and ran the highlight
    # down the thigh. The hand breaks into several pieces where the fingers separate, so each
    # arm is the union of everything on its side rather than a single piece.
    OUTER = 0.30
    torso, arms = {}, {}
    for y in range(top, bottom + 1):
        pieces = runs(mask[y])
        if not pieces:
            continue
        left_arm = [p for p in pieces if (p[0] + p[1]) / 2 < width * OUTER]
        right_arm = [p for p in pieces if (p[0] + p[1]) / 2 > width * (1 - OUTER)]
        body = [p for p in pieces if p not in left_arm and p not in right_arm]
        if body:
            torso[y] = (min(p[0] for p in body), max(p[1] for p in body))
        elif pieces:
            torso[y] = (min(p[0] for p in pieces), max(p[1] for p in pieces))
        # An arm is a piece outside the body, so there has to be a body between them. Without
        # that test the feet qualify: the toes splay wider than the legs, so the last few rows of
        # the picture are two pieces near the edges and nothing in the middle, and the forearm
        # highlight ran all the way down to the floor.
        if left_arm and right_arm and body:
            arms[y] = ((min(p[0] for p in left_arm), max(p[1] for p in left_arm)),
                       (min(p[0] for p in right_arm), max(p[1] for p in right_arm)))

    # The first row where the arms are clear of the body, and the row where the legs separate.
    split_arms = next((y for y in sorted(arms) if y > top + span * 0.25), None)
    legs = next((y for y in sorted(torso)
                 if y > top + span * 0.5 and len(runs(mask[y])) >= 2
                 and len([p for p in runs(mask[y]) if p[0] > width * 0.2 and p[1] < width * 0.8]) >= 2), None)
    # The waist: the narrowest torso row between the arms clearing and the legs separating.
    between = [y for y in torso if split_arms and legs and split_arms < y < legs]
    waist = min(between, key=lambda y: torso[y][1] - torso[y][0]) if between else None
    # The neck: the narrowest row in the top fifth, where the head meets the shoulders.
    upper = [y for y in torso if top + span * 0.08 < y < top + span * 0.28]
    neck = min(upper, key=lambda y: torso[y][1] - torso[y][0]) if upper else None

    return {"top": top, "bottom": bottom, "span": span, "width": width, "height": height,
            "torso": torso, "arms": arms, "neck": neck, "waist": waist,
            "arms_clear": split_arms, "legs_split": legs}


class Figure:
    """One side of the body, with the measurements taken and the helpers the shapes are built from."""

    def __init__(self, mask):
        self.m = landmarks(mask)
        # Filled in by `build` as soon as it has worked the landmark out, because `arm` needs it
        # to narrow the upper arm into the shoulder.
        self.shoulder = 0.29

    # Fractions of the whole figure's height, so a shape is written as "from the shoulders to the
    # waist" rather than as a pixel number.
    def y(self, fraction):
        return self.m["top"] + self.m["span"] * fraction

    def at(self, y):
        """The torso's left and right edge at this height, falling back to the nearest measured
        row so a shape never collapses because one row was empty."""
        y = int(round(y))
        rows = self.m["torso"]
        if y in rows:
            return rows[y]
        nearest = min(rows, key=lambda row: abs(row - y))
        return rows[nearest]

    def arm(self, y, side):
        """The left or right arm's (left, right) at this height.

        Above the row where the arms come clear of the ribs there is no separate arm to measure,
        and the upper arm is exactly where the biceps and the triceps are. So that stretch is
        carved off the outside of the torso run instead: at shoulder height the run spans deltoid
        to deltoid, and its outer fifth is the arm. Guessing is acceptable here and nowhere else
        — the alternative is no bicep on a bicep curl."""
        y = int(round(y))
        rows = self.m["arms"]
        if rows:
            nearest = min(rows, key=lambda row: abs(row - y))
            if abs(nearest - y) <= self.m["span"] * 0.03:
                return rows[nearest][0 if side == "left" else 1]
        # Above that, the arm is still merged with the ribs and has to be inferred. Carving a
        # fixed share off the outside of the torso run was the first attempt and it put the
        # bicep on the chest: at shoulder height that run spans deltoid to deltoid, so a fifth
        # of it reaches well inside the armpit. Instead the arm's own measured width at the row
        # where it comes clear is carried upward and narrowed into the shoulder, which is what
        # an upper arm does.
        left, right = self.at(y)
        if right - left <= 0 or not rows:
            return None
        first = min(rows)
        measured = rows[first][0 if side == "left" else 1]
        thickness = measured[1] - measured[0]
        ceiling = self.y(self.shoulder)
        if first <= ceiling:
            share = 1.0
        else:
            share = 0.58 + 0.42 * max(0.0, min(1.0, (y - ceiling) / (first - ceiling)))
        thickness *= share
        return (left, left + thickness) if side == "left" else (right - thickness, right)

    def norm(self, points):
        return [[round(x / self.m["width"], 4), round(y / self.m["height"], 4)] for x, y in points]

    # --- the shapes ----------------------------------------------------------------------------

    def band(self, y0, y1, inset_left=0.0, inset_right=0.0, steps=19, pinch=0.55):
        """A shape that follows the torso's own outline between two heights.

        An inset may be a pair, which ramps it from the top of the shape to the bottom. That is
        what gives the trapezius its kite: wide across the shoulders and narrowing to the spine.
        Written as straight bands, the traps, the rhomboids and the lower back stacked into one
        vertical bar which, crossed by the lats, drew a large letter T on the back of every
        figure in the set.

        Following the outline rather than drawing a rectangle is what makes the lats taper with
        the waist and the chest widen with the ribcage. The insets are fractions of the torso's
        width at each height, so a shape stays proportionally inside the body as the body narrows.
        """
        left, right = [], []
        for step in range(steps + 1):
            fraction = step / steps
            y = y0 + (y1 - y0) * fraction
            l, r = self.at(y)
            width = r - l
            inner = l + width * ramp(inset_left, fraction)
            outer = r - width * ramp(inset_right, fraction)
            middle, half = (inner + outer) / 2, (outer - inner) / 2 * spindle(fraction, pinch)
            left.append((middle - half, y))
            right.append((middle + half, y))
        return left + right[::-1]

    def half(self, y0, y1, side, inset_outer=0.04, inset_inner=0.52, steps=19, pinch=0.55):
        """One side of the torso — a pectoral, a lat, an oblique."""
        # The inner inset is given as "how far across the body the shape reaches", which reads
        # the right way round when writing a lat; the band wants the complement. A ramped pair
        # has to be flipped end for end, not subtracted as a number.
        def flip(inset):
            if isinstance(inset, (tuple, list)):
                return tuple(1 - value for value in inset)
            return 1 - inset

        if side == "left":
            return self.band(y0, y1, inset_left=inset_outer, inset_right=flip(inset_inner),
                             steps=steps, pinch=pinch)
        return self.band(y0, y1, inset_left=flip(inset_inner), inset_right=inset_outer,
                         steps=steps, pinch=pinch)

    def limb(self, y0, y1, side, inset=0.12, steps=15, pinch=0.6):
        """A shape down one arm, following where the arm actually is."""
        left, right = [], []
        for step in range(steps + 1):
            fraction = step / steps
            y = y0 + (y1 - y0) * fraction
            found = self.arm(y, side)
            if found is None:
                continue
            l, r = found
            width = r - l
            middle, half = (l + r) / 2, (width / 2 - width * inset) * spindle(fraction, pinch)
            left.append((middle - half, y))
            right.append((middle + half, y))
        if len(left) < 2:
            return []
        return left + right[::-1]

    def leg(self, y0, y1, side, inset=0.10, steps=15, pinch=0.6):
        """A shape down one leg. Below the split the torso row is one leg, so the legs are read
        from the row's own pieces rather than from `torso`."""
        left, right = [], []
        for step in range(steps + 1):
            fraction = step / steps
            y = int(round(y0 + (y1 - y0) * fraction))
            pieces = [p for p in self._row(y) if self._is_leg(p)]
            if len(pieces) < 2:
                continue
            piece = pieces[0] if side == "left" else pieces[-1]
            width = piece[1] - piece[0]
            middle = (piece[0] + piece[1]) / 2
            half = (width / 2 - width * inset) * spindle(fraction, pinch)
            left.append((middle - half, y))
            right.append((middle + half, y))
        if len(left) < 2:
            return []
        return left + right[::-1]

    def _row(self, y):
        return self._rows.get(int(y), [])

    def _is_leg(self, piece):
        middle = self.m["width"] / 2
        return abs((piece[0] + piece[1]) / 2 - middle) < self.m["width"] * 0.33


def build(side, mask):
    figure = Figure(mask)
    height, width = mask.shape
    figure._rows = {y: runs(mask[y]) for y in range(mask.shape[0])}

    m = figure.m
    # The landmarks as fractions, so the shapes below read as anatomy rather than as arithmetic.
    shoulder = (m["neck"] - m["top"]) / m["span"] + 0.02 if m["neck"] else 0.21
    clear = (m["arms_clear"] - m["top"]) / m["span"] if m["arms_clear"] else 0.41
    waist = (m["waist"] - m["top"]) / m["span"] if m["waist"] else 0.46
    crotch = (m["legs_split"] - m["top"]) / m["span"] if m["legs_split"] else 0.64
    # The elbow is where the arm has been clear of the body for a little way; the hand is where
    # the arm pieces stop.
    # The elbow sits at the waist when the arms hang, which is the landmark used here. The
    # narrowest point of the free arm was tried first and it finds the WRIST — a forearm tapers
    # all the way down — which put the whole upper arm and most of the forearm inside the bicep
    # and left the highlight looking like a sleeve.
    arm_rows = sorted(m["arms"])
    hand = (arm_rows[-1] - m["top"]) / m["span"] if arm_rows else 0.68
    elbow = waist
    knee = crotch + (1 - crotch) * 0.45

    figure.shoulder = shoulder
    f, s = figure, side
    regions = {}

    def put(name, *rings):
        """One muscle, as one closed ring or as a left-and-right pair.

        Kept as separate rings rather than one flat list because the app fills what it is given:
        two arms written end to end as a single ring fill as one shape joined across the chest.
        An empty ring — a limb that could not be measured — drops out, and a muscle left with
        nothing is simply absent, which the app draws as no highlight rather than as a guess."""
        kept = [f.norm(ring) for ring in rings if ring and len(ring) >= 3]
        if kept:
            regions[name] = kept

    if s == "front":
        put("chest", f.half(f.y(shoulder + 0.012), f.y(shoulder + 0.105), "left",
                            inset_outer=0.13, inset_inner=0.485, pinch=0.72),
            f.half(f.y(shoulder + 0.012), f.y(shoulder + 0.105), "right",
                   inset_outer=0.13, inset_inner=0.485, pinch=0.72))
        put("abs", f.band(f.y(shoulder + 0.10), f.y(waist + 0.035),
                          inset_left=(0.36, 0.33), inset_right=(0.36, 0.33), pinch=0.28))
        put("obliques", f.band(f.y(shoulder + 0.125), f.y(waist + 0.025),
                               inset_left=(0.17, 0.21), inset_right=(0.70, 0.68), pinch=0.5),
            f.band(f.y(shoulder + 0.125), f.y(waist + 0.025),
                   inset_left=(0.70, 0.68), inset_right=(0.17, 0.21), pinch=0.5))
        put("serratus", f.band(f.y(shoulder + 0.08), f.y(shoulder + 0.14), inset_left=0.08, inset_right=0.78),
            f.band(f.y(shoulder + 0.08), f.y(shoulder + 0.14), inset_left=0.78, inset_right=0.08))
        put("front-delts", f.band(f.y(shoulder + 0.002), f.y(shoulder + 0.062), inset_left=0.01, inset_right=0.86),
            f.band(f.y(shoulder + 0.002), f.y(shoulder + 0.062), inset_left=0.86, inset_right=0.01))
        put("biceps", f.limb(f.y(shoulder + 0.03), f.y(elbow), "left"),
            f.limb(f.y(shoulder + 0.03), f.y(elbow), "right"))
        put("quads", f.leg(f.y(crotch + 0.01), f.y(knee), "left"), f.leg(f.y(crotch + 0.01), f.y(knee), "right"))
        put("adductors", f.leg(f.y(crotch + 0.01), f.y(crotch + (knee - crotch) * 0.8), "left", inset=0.55),
            f.leg(f.y(crotch + 0.01), f.y(crotch + (knee - crotch) * 0.8), "right", inset=0.55))
    else:
        # The kite: across the shoulders at the top, down to the spine at the bottom.
        put("traps", f.band(f.y(shoulder - 0.015), f.y(shoulder + 0.105),
                            inset_left=(0.12, 0.40), inset_right=(0.12, 0.40)))
        # Between the shoulder blades: wider than it is tall, and sitting under the traps.
        put("rhomboids", f.band(f.y(shoulder + 0.055), f.y(shoulder + 0.115),
                                inset_left=(0.22, 0.30), inset_right=(0.22, 0.30)))
        put("lats", f.half(f.y(shoulder + 0.055), f.y(waist + 0.005), "left",
                           inset_outer=0.08, inset_inner=(0.47, 0.37), pinch=0.42),
            f.half(f.y(shoulder + 0.055), f.y(waist + 0.005), "right",
                   inset_outer=0.08, inset_inner=(0.47, 0.37), pinch=0.42))
        # Narrow, and widening a little into the pelvis rather than running as a straight bar.
        put("lower-back", f.band(f.y(waist - 0.055), f.y(waist + 0.045),
                                 inset_left=(0.33, 0.26), inset_right=(0.33, 0.26)))
        put("glutes", f.band(f.y(waist + 0.045), f.y(crotch - 0.005),
                             inset_left=0.11, inset_right=0.52, pinch=0.78),
            f.band(f.y(waist + 0.045), f.y(crotch - 0.005),
                   inset_left=0.52, inset_right=0.11, pinch=0.78))
        put("hamstrings", f.leg(f.y(crotch + 0.01), f.y(knee), "left"), f.leg(f.y(crotch + 0.01), f.y(knee), "right"))
        put("calves", f.leg(f.y(knee + 0.02), f.y(knee + (1 - knee) * 0.62), "left"),
            f.leg(f.y(knee + 0.02), f.y(knee + (1 - knee) * 0.62), "right"))
        put("triceps", f.limb(f.y(shoulder + 0.03), f.y(elbow), "left"),
            f.limb(f.y(shoulder + 0.03), f.y(elbow), "right"))
        put("rear-delts", f.band(f.y(shoulder + 0.002), f.y(shoulder + 0.062), inset_left=0.01, inset_right=0.86),
            f.band(f.y(shoulder + 0.002), f.y(shoulder + 0.062), inset_left=0.86, inset_right=0.01))

    # On both sides: visible from either, and a reader looking at the back for a row should still
    # see that their grip is working.
    put("forearms", f.limb(f.y(elbow), f.y(hand - 0.015), "left"), f.limb(f.y(elbow), f.y(hand - 0.015), "right"))
    put("side-delts", f.band(f.y(shoulder - 0.012), f.y(shoulder + 0.05), inset_left=0.0, inset_right=0.90),
            f.band(f.y(shoulder - 0.012), f.y(shoulder + 0.05), inset_left=0.90, inset_right=0.0))
    put("neck", f.band(f.y(shoulder - 0.075), f.y(shoulder - 0.005), inset_left=0.30, inset_right=0.30))

    return regions, {"shoulder": round(shoulder, 4), "arms_clear": round(clear, 4),
                     "waist": round(waist, 4), "crotch": round(crotch, 4),
                     "elbow": round(elbow, 4), "hand": round(hand, 4), "knee": round(knee, 4)}


def preview(exercise, regions, out):
    work = json.loads(MUSCLE_MAP.read_text()).get(exercise)
    if not work:
        raise SystemExit(f"{exercise!r} is not in {MUSCLE_MAP}")
    tiles = []
    for side in ("front", "back"):
        base = Image.open(FIGURES / f"{side}.png").convert("RGBA")
        card = Image.new("RGBA", base.size, (255, 255, 255, 255))
        card.alpha_composite(base)
        for muscle in work["prime"] + work["supporting"]:
            rings = regions[side].get(muscle)
            if not rings:
                continue
            alpha = 158 if muscle in work["prime"] else 66
            sheet = Image.new("RGBA", base.size, (0, 0, 0, 0))
            for ring in rings:
                ImageDraw.Draw(sheet).polygon(
                    [(x * base.width, y * base.height) for x, y in ring], fill=(0, 110, 255, alpha))
            # Only over the body, never over the backdrop.
            sheet.putalpha(Image.fromarray(
                (np.array(sheet)[..., 3] * (np.array(base)[..., 3] > 128)).astype(np.uint8)))
            card = Image.alpha_composite(card, sheet)
        tiles.append(card.convert("RGB"))
    strip = Image.new("RGB", (sum(t.width for t in tiles) + 20, max(t.height for t in tiles)), "white")
    x = 0
    for t in tiles:
        strip.paste(t, (x, 0)); x += t.width + 20
    strip.save(out)
    print(f"{out}  prime={','.join(work['prime'])}  supporting={','.join(work['supporting'])}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", metavar="EXERCISE", help="render one exercise's highlights to a PNG")
    parser.add_argument("--out", default="/tmp/muscle-preview.png")
    args = parser.parse_args()

    regions, marks, sizes = {}, {}, {}
    for side in ("front", "back"):
        mask = np.array(Image.open(FIGURES / f"{side}.png").convert("RGBA"))[..., 3] > 128
        sizes[side] = [int(mask.shape[1]), int(mask.shape[0])]
        regions[side], marks[side] = build(side, mask)
        print(f"{side}: {len(regions[side])} regions  {marks[side]}")

    OUT.write_text(json.dumps({
        "note": "Muscle shapes in fractions of each side's picture, measured from the drawing by "
                "pipeline/muscle_regions.py. Fill these over reference/muscle-figure/<side>.png, clipped to that "
                "picture's own alpha so nothing paints on the backdrop.",
        "landmarks": marks,
        # The app needs each side's shape to lay the polygons over the picture: the two drawings
        # came back a few pixels apart, and a shared aspect ratio would slide every highlight.
        "size": sizes,
        "sides": regions,
    }, indent=2) + "\n")

    if args.preview:
        preview(args.preview, regions, args.out)


if __name__ == "__main__":
    main()
