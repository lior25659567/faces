"""Body proportions measured from a whole-body photo, for the avatar's body shape and for progress
tracking: the same numbers, read again from a later photo, are real data a person can compare.

Apple Vision gives two readings of the person (apple_vision.body): a silhouette mask, and the
joints it is confident of (shoulders, hips, neck, pelvis). The joints alone are not what this
needs: they mark the skeleton, which is close to the same width on everyone of a given frame size,
not the soft tissue that actually changes with fitness. What they are good for is height: they say
where the shoulder, waist and hip rows are, steadily, regardless of the clothes or the pose.

The widths themselves come from the silhouette at those rows — but only across the torso, and only
where the torso can actually be told apart from the arms hanging beside it. That last condition is
the one this module is strictest about, because it is the one that silently produces wrong numbers
rather than no numbers. Measured on a photo of a slim man in a fitted t-shirt, the gap between his
arm and his waist was two pixels wide; re-encoding the same photo closed it, and the waist row then
read 202 pixels instead of 150 — a 35% error with nothing to show it had happened. So a row counts
only when a detached run of silhouette is visible on *both* sides of the torso, which is the
physical condition the reading depends on, tested rather than assumed.

The waist and hip widths are expressed as a share of shoulder width, which gives shape; shoulder
width is also kept as a share of the person's own head width, which gives size. Both are a person's
own measurements compared to another of their own, rather than to an absolute scale that shifts
with how far they stood from the camera.

What this still cannot see: clothing. A photo of a woman in a puffer jacket measured a waist 23%
wider than her shoulders, steady across every variation of the photo — a stable, confident
measurement of a jacket. Fitted clothing is a condition of the measurement meaning anything, and
nothing here can verify it; the capture guidance has to ask for it.
"""

from dataclasses import dataclass

import numpy as np

# Where the waist can be, as a share of the way down from the shoulders to the hips: a torso's
# narrowest point is usually in the lower half, above the hip joints themselves.
WAIST_BAND = (0.45, 0.85)
MIN_ROWS = 3  # waist rows with a clean reading needed to trust the measurement

# Runs of silhouette shorter than this are speckle in the segmentation, not an arm.
MIN_RUN = 4
# Share of the waist band's readable rows whose width must be the torso's own. A row where an arm
# has merged into the torso does not read too wide and get ignored by the min below — it hides
# whatever that row's true width was, which may be the narrowest on the body. So a band that is
# mostly merged cannot be measured at all, however confident a number it would produce.
CLEAN_SHARE = 0.8
# How much wider than the row above a row may read and still be the same torso. A body's outline
# moves about a pixel per row; an arm merging into it adds tens at once.
MAX_JUMP = 0.04


@dataclass
class Build:
    waist_to_shoulder: float  # 1.0: waist as wide as the shoulders. A V taper is well under 1.
    hip_to_shoulder: float | None  # 1.0: hips as wide as the shoulders. None when the hips
    # themselves could not be read, which is commoner than it sounds: the arms hang beside the
    # hips in the stance most people take, and a hand resting there hides the hip line while
    # leaving the waist perfectly visible. Measured on 95 whole-body photographs, four had a
    # waist 88-90% readable and no hip width at all. Refusing those outright threw away the one
    # number the app actually charts to protect a secondary one, so the hip ratios are allowed
    # to be absent instead.
    waist_to_hip: float | None  # the standard waist-to-hip ratio, from the two above. Compare
    # it only with the same person's own earlier reading: across 32 calibration photos it told
    # men from women with almost no overlap (0.773-0.849 against 0.721-0.780), which makes it a
    # cleaner sex classifier than the shoulder measure withdrawn for being one. Never threshold
    # it, and never show it against a population reference.
    shoulder_to_head: float  # frame size: shoulder width as a multiple of the person's own head
    # width, which does not change with fitness, so this is comparable photo to photo even as the
    # shape ratios above hold steady — two people, or the same person months apart, can have the
    # same taper at very different overall sizes.


def _runs(pixels):
    """Start and end of each run of silhouette along this row, ignoring speckle."""
    found, start = [], None
    for x, lit in enumerate(pixels):
        if lit and start is None:
            start = x
        elif not lit and start is not None:
            if x - start >= MIN_RUN:
                found.append((start, x - 1))
            start = None
    if start is not None and len(pixels) - start >= MIN_RUN:
        found.append((start, len(pixels) - 1))
    return found


def _torso_width(mask, row, center_x):
    """Width of the run of silhouette containing the body's centerline, whatever it contains.
    None if the centerline is not on the body, or the run reaches the edge of the picture — where
    a gap may have been cropped away, so nothing about its edge can be relied on."""
    pixels = mask[row]
    center = int(round(center_x))
    if not (0 <= center < len(pixels)):
        return None
    runs = _runs(pixels)
    torso = next(((a, b) for a, b in runs if a <= center <= b), None)
    if torso is None:
        return None
    left, right = torso
    if left == 0 or right == len(pixels) - 1:
        return None
    return float(right - left)


def _arms_detached(mask, row, center_x):
    """Whether a separate run of silhouette is visible beyond the torso on both sides: the arms,
    held clear of the body, which is what makes the torso run end at the torso's own edge."""
    pixels = mask[row]
    center = int(round(center_x))
    runs = _runs(pixels)
    torso = next(((a, b) for a, b in runs if a <= center <= b), None)
    if torso is None:
        return False
    left, right = torso
    return any(end < left for _, end in runs) and any(start > right for start, _ in runs)


def _trusted_profile(mask, center_x, top, bottom):
    """{row: torso width} for the rows down the body whose width really is the torso's.

    Scanned downward, because the two ways to know are different. Where the arms are held clear,
    a detached run beyond the torso proves the edge is the torso's own. Where the arms have ended
    above a row — hands raised, or a short arm in frame — there is no gap to find and no arm to
    confuse it with either, and demanding one would reject a perfectly clean reading; what settles
    those rows is continuity, since a torso's outline moves about a pixel per row while an arm
    merging into it adds tens at once. So a row with no visible gap is trusted only while its
    width still continues the last trusted row above it, and a jump breaks the chain until the
    arms come clearly into view again.
    """
    profile, last = {}, None
    for row in range(max(0, top), bottom + 1):
        width = _torso_width(mask, row, center_x)
        if width is None:
            last = None
            continue
        if _arms_detached(mask, row, center_x):
            profile[row] = width
            last = width
        elif last is not None and width <= last * (1 + MAX_JUMP):
            profile[row] = width
            last = width
        else:
            last = None
    return profile


def _width_near(profile, row, reach):
    """The widest trusted width within `reach` rows of `row` either way: a joint's own row can
    land on a fold in the shirt or a slight turn of the shoulders, so nearby rows are tried too.
    Only trusted rows count, so this cannot widen a measurement by reaching for the one nearby
    row where an arm had merged into the body."""
    widths = [profile[r] for r in range(row - reach, row + reach + 1) if r in profile]
    return max(widths) if widths else None


# There is no shape_phrase any more, and this is the record of why, because the next person to
# reach for one will reach for the same two ideas that were tried here.
#
# It turned the measured ratios into words appended to the vision model's own word for the body
# ("a slim, slender body ...", "a heavyset body ..."). Two attempts, both withdrawn:
#
#   1. Shoulder width over head width, to say whether someone was narrow or broadly built. A
#      calibration set of 32 photos (calibrate_bodies.py, reference/bodies) showed the men at
#      3.11-3.85 and the women at 2.69-3.37 with no overlap, so the threshold was sorting by sex,
#      not by build — it would have called a stocky woman narrow-framed. Worse, darkening only a
#      person's pixels in one photo, the body untouched, moved the number 11%, because Vision's
#      face box shrinks on a darker or lower-contrast face. It is sensitive to the lighting in
#      the room and to the person's skin, which is everything you do not want in a body measure.
#
#   2. The waist and hip taper, which is what remained. Over the same 32 photos the whole clause
#      produced TWO distinct sentences, and all 32 began "a pronounced V-taper: the waist
#      noticeably narrower than the shoulders" — a severely obese man at 0.771 described exactly
#      as a shredded one at 0.566. Redundant where it agreed with the vision word, and wrong
#      where it mattered.
#
# A cut at waist/shoulder 0.67 does separate the overweight and obese photos from the other 28
# with no errors. It was found, tested and refused: the zero-error window is bounded by single
# images on both sides, it is 0.0144 from the nearest ordinary body against redraw noise of
# 0.0219, and placing it pooled gives the same answer as placing it on the men alone — so it is
# set by male bodies and would describe heavier-than-average men sooner than equally heavy women,
# in an app that cannot know which it is looking at. The gap it exploits is the spacing between
# two words a text-to-image model was given, not a boundary between bodies.
#
# What the photos did show, and the reason the ratios below are still worth measuring: across the
# eight identity prompts that appear twice, once leaner and once heavier, waist/shoulder rose in
# 8 of 8 and hip/shoulder in 8 of 8. The measurement tracks what it is supposed to track. It is
# describing it in words, against a threshold, for a stranger, that it cannot do.
#
# AND THEN REAL BODIES SAID SOMETHING DIFFERENT, which is why this note does not end above.
# ANSUR II (reference/ansur, 6,068 soldiers measured with a tape) holds waist breadth over
# shoulder breadth — the same frontal widths this module reads off a silhouette. Real people:
# median 0.784 for men and 0.814 for women, middle 90% from 0.672 to 0.939. All 32 generated
# photos fell between 0.541 and 0.771, which is at or below the real 25th percentile. The image
# model draws everybody far more V-tapered than people are, so "the clause produced two sentences
# across 32 bodies" was a fact about the generator and not about the threshold. On real bodies
# 0.78 sits almost exactly at the median and would divide them roughly in half.
#
# Two things follow, and the second is why nothing has been put back. First, the sex objection is
# weaker than it looked: in ANSUR the sexes overlap across 0.495 of the range and the best single
# cut tells them apart only 67% of the time, against the near-perfect separation the generated
# set implied. Second, and decisive: ANSUR measures a tape at the navel, this module measures the
# narrowest row it can find between the shoulders and the hips, and nobody has measured the
# offset between those two numbers on a single real person. Until some real photographs are
# measured, there is no way to know which part of the 0.54-0.77 the photos gave is the generator
# and which is this code reading low by construction. That measurement, not more generated
# people, is what the next person should go and get.


def measure(mask, joints, face_box):
    """Build from a person mask, the joints Vision found (see apple_vision.body), and their face
    box (see apple_vision.faces), for a stable sense of scale.

    None when the photo cannot support the measurement: the shoulders or the hip line missing — a
    close-up portrait, a side-on pose, loose clothing that hides the body's outline from Vision's
    pose model as much as from this measurement — or arms held too close to the body for the torso
    to be told apart from them down most of the waist.

    The hip line is taken from whichever of the hip joints Vision was sure of, and from the pelvis
    joint when it was sure of neither hip. That is not a softening of the standard: all three are
    joints the pose model reported, not a position inferred from the others. It matters because of
    who was being turned away. On a calibration set of twelve photos the three refusals were all
    the heaviest bodies, and every one of them failed the same way — the shoulders found, the hips
    lost under a t-shirt draping over the waist. Requiring both hips meant a fitness app refusing
    to measure exactly the people most likely to be tracking a change, so where the model still
    reports a pelvis, that is enough to place the line.
    """
    if "leftShoulder" not in joints or "rightShoulder" not in joints:
        return None
    hips = [joints[name] for name in ("leftHip", "rightHip") if name in joints]
    if not hips and "root" in joints:
        hips = [joints["root"]]
    if not hips:
        return None
    head_width = face_box[2] - face_box[0]
    (lsx, lsy), (rsx, rsy) = joints["leftShoulder"], joints["rightShoulder"]
    shoulder_y = (lsy + rsy) / 2
    hip_y = sum(y for _, y in hips) / len(hips)
    # The centreline is the body's, so it is averaged over the shoulders and whatever marks the
    # hip line: with only one hip found, that one hip would pull the centre off to its own side.
    center_x = ((lsx + rsx) / 2 + sum(x for x, _ in hips) / len(hips)) / 2
    reach = max(1, round(abs(hip_y - shoulder_y) * 0.03))

    # The arms leave the body at the shoulders, so there is no gap beside them to look for there
    # and the deltoids are part of shoulder width anyway: that row is read as it comes. Everything
    # below it has to earn its reading, which is what the trusted profile decides.
    shoulder_width = max(
        (w for w in (_torso_width(mask, r, center_x)
                     for r in range(round(shoulder_y) - reach, round(shoulder_y) + reach + 1))
         if w is not None),
        default=None,
    )
    profile = _trusted_profile(mask, center_x, round(shoulder_y), round(hip_y) + reach)
    hip_width = _width_near(profile, round(hip_y), reach)

    top, bottom = (int(shoulder_y + edge * (hip_y - shoulder_y)) for edge in WAIST_BAND)
    band = range(top, max(top + 1, bottom))
    # Readable at all, versus readable as the torso's own width: the ratio of the two says whether
    # the arms were held far enough from the body for the waist to mean anything.
    readable = [r for r in band if _torso_width(mask, r, center_x) is not None]
    waist_rows = [profile[r] for r in readable if r in profile]

    # The hips are allowed to be missing; the shoulders and the waist are not, because without
    # them there is no measurement at all rather than a smaller one.
    if not shoulder_width or head_width <= 0:
        return None
    if len(waist_rows) < MIN_ROWS or len(waist_rows) < CLEAN_SHARE * len(readable):
        return None
    waist_width = min(waist_rows)

    return Build(
        waist_to_shoulder=waist_width / shoulder_width,
        hip_to_shoulder=hip_width / shoulder_width if hip_width else None,
        waist_to_hip=waist_width / hip_width if hip_width else None,
        shoulder_to_head=shoulder_width / head_width,
    )
