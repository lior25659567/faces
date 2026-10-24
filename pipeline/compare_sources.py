"""Three sources, one ratio, and the question of which one this code actually agrees with.

waist breadth over shoulder breadth can be had three ways now:

  - reference/ansur       6,068 real people measured with a tape. Real bodies, not photographs.
  - reference/bodies      32 photographs a text-to-image model drew to order, measured by this
                          pipeline.
  - reference/open-photos openly licensed photographs of real people, measured by this pipeline.

The first two disagree sharply: ANSUR puts the middle 90% at 0.672-0.939 and every generated
photo fell between 0.541 and 0.771. Two explanations, with opposite consequences:

  - the image model draws people far more V-tapered than people are, and the measurement is fine;
  - or the measurement reads low by construction — ANSUR puts a tape round the navel, this finds
    the narrowest row between shoulders and hips, and narrowest is narrower than navel.

The third source separates them, because it is real bodies read by this pipeline. If those land
with ANSUR, the generator was the problem and the original thresholds were never wrong. If they
land with the generated photos, this code reads low and every threshold needs an offset before it
means anything.

    pipeline/.venv/bin/python pipeline/compare_sources.py
"""

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).parent.parent


def percentile(values, p):
    ordered = sorted(values)
    return ordered[int(p / 100 * (len(ordered) - 1))]


def describe(label, values, note=""):
    if len(values) < 2:
        print(f"{label:22} {len(values)} rows — not enough to say anything")
        return
    print(f"{label:22} n={len(values):5}  "
          + "  ".join(f"p{p}={percentile(values, p):.3f}" for p in (5, 25, 50, 75, 95)))
    if note:
        print(f"{'':22} {note}")


def main():
    ansur = json.loads((ROOT / "reference/ansur/summary.json").read_text())
    generated = [r["waist_to_shoulder"] for r in
                 json.loads((ROOT / "reference/bodies/measured.json").read_text())
                 if "refused" not in r]
    openly, note = [], ""
    path = ROOT / "reference/open-photos/measured.json"
    if path.exists():
        found = json.loads(path.read_text())
        if isinstance(found, dict):
            openly = [r["waist_to_shoulder"] for r in found.get("measured", [])]
            note = found.get("note", "")
        else:
            openly = [r["waist_to_shoulder"] for r in found]

    print("waist breadth / shoulder breadth\n")
    for sex in ("men", "women"):
        band = ansur[f"{sex}_waist_to_shoulder"]
        print(f"{'ANSUR ' + sex:22} n={ansur[sex]:5}  "
              + "  ".join(f"p{p}={band[str(p)]:.3f}" for p in (5, 25, 50, 75, 95)))
    print()
    describe("generated photos", generated, "drawn to order, measured by this pipeline")
    describe("open photographs", openly, "real people, measured by this pipeline")

    if len(openly) >= 5:
        middle = statistics.median(openly)
        to_ansur = abs(middle - (ansur["men_waist_to_shoulder"]["50"]
                                 + ansur["women_waist_to_shoulder"]["50"]) / 2)
        to_generated = abs(middle - statistics.median(generated))
        print(f"\nreal photographs sit at {middle:.3f}")
        print(f"  {to_ansur:.3f} from the real-body median, "
              f"{to_generated:.3f} from the generated median")
        if to_ansur < to_generated:
            print("  -> they agree with real bodies: the generator was the problem, and the\n"
                  "     original thresholds were measuring against unrepresentative photographs.")
        else:
            print("  -> they agree with the generated photos: this pipeline reads lower than a\n"
                  "     tape does, and any threshold borrowed from ANSUR needs that offset first.")
    else:
        print("\nNot enough open photographs to judge, and that is itself the finding.")
        if note:
            print("  " + note.replace(". ", ".\n  "))


if __name__ == "__main__":
    main()
