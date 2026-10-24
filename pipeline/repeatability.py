"""How much the measurement moves when the body has not.

Every claim about whether a change is real rests on this number, and it was missing: a reviewer
reading the repository found no repeat measurement of any body anywhere and correctly treated a
change threshold derived from one as unfounded. So it lives here now, and it is re-runnable.

Each photo is measured again under changes that alter no part of the body — a JPEG re-encode, a
few pixels cropped, a slight resize, a brightness shift. Whatever the reading does across those
is noise, and a real change has to clear it before anyone is told their body changed.

    pipeline/.venv/bin/python pipeline/repeatability.py
"""

import io
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from PIL import Image, ImageEnhance, ImageOps

import apple_vision
import body_measure

BODIES = Path(__file__).parent.parent / "reference/bodies"
OUT = BODIES / "repeatability.json"


def variants(photo):
    width, height = photo.size
    yield "as shot", photo
    for quality in (92, 78):
        buffer = io.BytesIO()
        photo.save(buffer, "JPEG", quality=quality)
        yield f"jpeg{quality}", Image.open(buffer).convert("RGB")
    for factor in (0.9, 1.1):
        yield f"bright{factor}", ImageEnhance.Brightness(photo).enhance(factor)
    for scale in (0.95, 1.04):
        yield f"scale{scale}", photo.resize(
            (int(width * scale), int(height * scale)), Image.LANCZOS)
    yield "crop3", photo.crop((3, 3, width - 3, height - 3))


def main():
    rows = []
    for path in sorted(BODIES.glob("*.png")):
        photo = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        readings = []
        for _, shot in variants(photo):
            faces = apple_vision.faces(shot)
            if not faces:
                continue
            mask, joints = apple_vision.body(shot)
            build = body_measure.measure(mask, joints, faces[0])
            if build:
                readings.append(build.waist_to_shoulder)
        if len(readings) < 3:
            print(f"  {path.stem:20} too few readings to judge")
            continue
        middle = statistics.median(readings)
        spread = max(readings) - min(readings)
        rows.append({"name": path.stem, "readings": len(readings), "median": middle,
                     "spread": spread, "percent": 100 * spread / middle})
        print(f"  {path.stem:20} {len(readings):2} readings  median {middle:.3f}  "
              f"spread {spread:.4f}  {100 * spread / middle:4.1f}%")

    worst = max(r["percent"] for r in rows)
    typical = statistics.median(r["percent"] for r in rows)
    summary = {"photos": len(rows), "worst_percent": worst, "median_percent": typical,
               "rows": rows}
    OUT.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\n{len(rows)} photos: worst {worst:.1f}%, median {typical:.1f}%")
    print("A reported change smaller than the worst figure is not evidence of anything.")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
