"""What real bodies measure, from the one source that is both real and free to use.

Everything calibrated so far came from photos a text-to-image model drew to order, and the
reviewer's objection to that was exactly right: the spread between a model's build words is not
the spread between people. This is the correction. ANSUR II is the 2012 US Army anthropometric
survey — 4,082 men and 1,986 women, 93 measurements each, taken with calipers and a tape by
people whose job it was. It is a US Government work and in the public domain, so it can sit in
this repository with no licence and no one's photograph in it.

It happens to hold the exact quantity this project measures. `waistbreadth` and
`biacromialbreadth` are FRONTAL WIDTHS, the same thing a silhouette gives, so
waistbreadth/biacromialbreadth is directly comparable to Build.waist_to_shoulder. It also holds
`waistdepth`, which is what a single photo cannot see, and `stature`, which turns a width into
the waist-to-height ratio the literature actually validates.

Its limits, which matter: these are serving soldiers. They are fitter, younger and far less
heavy than the general population, so the upper tail here understates the public's. And a tape
around a waist is not a silhouette read off a photo — ANSUR measures at the navel, this project
measures the narrowest row it can find, so the two are not expected to agree in level even when
they agree in shape.

    pipeline/.venv/bin/python pipeline/ansur_reference.py
"""

import csv
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).parent.parent / "reference/ansur"
OUT = HERE / "summary.json"


def load(name):
    rows = []
    for record in csv.DictReader(open(HERE / name, encoding="latin-1")):
        try:
            waist = float(record["waistbreadth"])
            shoulder = float(record["biacromialbreadth"])
            rows.append({
                "waist_to_shoulder": waist / shoulder,
                "hip_to_shoulder": float(record["hipbreadth"]) / shoulder,
                "waist_to_height": float(record["waistcircumference"]) / float(record["stature"]),
                "waist_breadth": waist,
                "waist_depth": float(record["waistdepth"]),
                "waist_circumference": float(record["waistcircumference"]),
            })
        except (ValueError, KeyError, ZeroDivisionError):
            continue
    return rows


def percentile(values, p):
    ordered = sorted(values)
    return ordered[int(p / 100 * (len(ordered) - 1))]


def ellipse(breadth, depth):
    """Ramanujan's perimeter, the standard way to get a circumference from a width and a depth."""
    a, b = breadth / 2, depth / 2
    return math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))


def main():
    men, women = load("ANSUR_II_MALE_Public.csv"), load("ANSUR_II_FEMALE_Public.csv")
    both = men + women
    summary = {"men": len(men), "women": len(women)}

    for label, group in (("men", men), ("women", women)):
        values = [r["waist_to_shoulder"] for r in group]
        summary[f"{label}_waist_to_shoulder"] = {
            str(p): round(percentile(values, p), 4) for p in (5, 25, 50, 75, 95)}

    # How well a single frontal width stands in for a circumference, which is the whole question
    # behind ever showing a person a number in centimetres.
    typical_depth = statistics.median(r["waist_depth"] / r["waist_breadth"] for r in both)
    with_depth = [abs(ellipse(r["waist_breadth"], r["waist_depth"]) - r["waist_circumference"])
                  / r["waist_circumference"] for r in both]
    width_only = [abs(ellipse(r["waist_breadth"], r["waist_breadth"] * typical_depth)
                      - r["waist_circumference"]) / r["waist_circumference"] for r in both]
    summary["circumference_from_width"] = {
        "assumed_depth_over_width": round(typical_depth, 4),
        "median_error_with_measured_depth": round(statistics.median(with_depth), 4),
        "median_error_width_only": round(statistics.median(width_only), 4),
        "p90_error_width_only": round(percentile(width_only, 90), 4),
    }
    summary["waist_to_height_over_half"] = {
        label: round(sum(1 for r in g if r["waist_to_height"] >= 0.5) / len(g), 3)
        for label, g in (("men", men), ("women", women))}

    OUT.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
