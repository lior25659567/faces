"""Measure openly licensed photographs of real people, and keep only the measurements.

Everything calibrated so far came from pictures a text-to-image model drew, and real bodies
turned out to sit nowhere near them: ANSUR II puts the middle 90% of people at 0.672-0.939 on
waist over shoulder, while all 32 generated photos fell between 0.541 and 0.771. So the numbers
this project needs have to come off photographs of actual people.

Those are harder to come by honestly than they look. The academic pose datasets — MPII,
COCO-WholeBody, OCHuman, CrowdPose — are almost all research-and-non-commercial, which an App
Store app cannot use. Stock libraries license the picture but commonly forbid using it to build
an algorithm, and a photograph of an identifiable person is personal data however it is licensed.

What this does instead: it searches Openverse, which is Creative Commons' own index, for images
under licences that permit commercial use; it measures them on this machine; and it writes down
ONLY the ratios, with the source URL, the licence and the creator so every row can be traced.
No photograph is copied into this repository and nothing identifying anyone is kept. The numbers
are measurements of a shape, which is a fact about the picture rather than a piece of it.

WHAT IT FOUND, so that nobody spends another afternoon on it: of 684 freely licensed photographs
from Wikimedia Commons, ZERO were usable. 232 had no detectable face at all — "full length
portrait" matches scanned book pages and group photographs more often than it matches a person;
176 lost the hip line to clothing, which on Commons means period dress, long coats and gowns;
148 had the arms against the torso or a draping garment over the waist; 77 would not decode; 51
were not tall enough to hold a standing body.

The conclusion is not that the licence filter is too strict. It is that this measurement needs a
pose almost nobody is photographed in — standing square to the camera, whole body in frame, in
fitted clothes, with daylight visible between each arm and the waist. That is a scan, and people
do not take scans for the public domain. No amount of searching fixes it, and a larger or more
permissive corpus would not either.

Where the missing numbers have to come from instead:
  - what real bodies measure: ANSUR II, already in reference/ansur. 6,068 people, public domain.
  - what this pipeline reads off a real person in the scan pose: only photographs taken for that
    purpose. The app's own users produce exactly that, with their consent, and no tape measure
    is needed — comparing the distribution this pipeline reads against ANSUR's is enough to
    recover the offset between "narrowest row" and "tape at the navel".

    pipeline/.venv/bin/python pipeline/open_photos.py
"""

import io
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from PIL import Image, ImageOps

import apple_vision
import body_measure

OUT = Path(__file__).parent.parent / "reference/open-photos/measured.json"
API = "https://commons.wikimedia.org/w/api.php"
AGENT = "faces-body-calibration/1.0 (body-proportion research; one request per second)"

# Licences that allow commercial use. Anything NonCommercial or NoDerivatives is skipped, and so
# is anything whose licence cannot be read, because "probably fine" is not a licence.
ALLOWED = ("cc0", "pdm", "public domain", "cc by", "cc-by", "cc by-sa", "cc-by-sa",
           "attribution", "no restrictions")
FORBIDDEN = ("nc", "noncommercial", "nd", "noderiv", "fair use", "non-free")

QUERIES = [
    "full length portrait photograph standing",
    "studio portrait full length standing",
    "full-length portrait standing facing front",
    "whole body photograph person standing",
    "full length photograph man standing",
    "full length photograph woman standing",
    "standing portrait plain background full length",
]


def call(**params):
    params.setdefault("format", "json")
    request = urllib.request.Request(
        API + "?" + urllib.parse.urlencode(params), headers={"User-Agent": AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def usable_licence(text):
    low = (text or "").lower()
    if any(bad in low.split() or bad in low for bad in FORBIDDEN):
        return False
    return any(good in low for good in ALLOWED)


def main():
    seen, rows, tried = set(), [], 0
    refused = {}
    for query in QUERIES:
        for offset in (0, 50, 100):
            try:
                found = call(action="query", list="search", srsearch=query,
                             srnamespace=6, srlimit=50, sroffset=offset)
            except Exception as error:
                print(f"  search failed ({query!r} +{offset}): {error}")
                continue
            titles = [r["title"] for r in found.get("query", {}).get("search", [])
                      if r["title"] not in seen]
            seen.update(titles)
            for batch in range(0, len(titles), 10):
                chunk = titles[batch:batch + 10]
                try:
                    info = call(action="query", titles="|".join(chunk), prop="imageinfo",
                                iiprop="url|extmetadata", iiurlwidth=900)
                except Exception:
                    continue
                for page in info.get("query", {}).get("pages", {}).values():
                    image = (page.get("imageinfo") or [{}])[0]
                    meta = image.get("extmetadata", {})
                    licence = (meta.get("LicenseShortName", {}) or {}).get("value", "")
                    if not usable_licence(licence):
                        refused["licence not usable"] = refused.get("licence not usable", 0) + 1
                        continue
                    url = image.get("thumburl") or image.get("url")
                    if not url:
                        continue
                    tried += 1
                    try:
                        request = urllib.request.Request(url, headers={"User-Agent": AGENT})
                        with urllib.request.urlopen(request, timeout=60) as response:
                            photo = ImageOps.exif_transpose(
                                Image.open(io.BytesIO(response.read()))).convert("RGB")
                    except Exception:
                        refused["unreadable"] = refused.get("unreadable", 0) + 1
                        continue
                    if photo.height < photo.width or min(photo.size) < 300:
                        refused["not a tall photo"] = refused.get("not a tall photo", 0) + 1
                        continue
                    faces = apple_vision.faces(photo)
                    if not faces:
                        refused["no face"] = refused.get("no face", 0) + 1
                        continue
                    mask, joints = apple_vision.body(photo)
                    build = body_measure.measure(mask, joints, faces[0])
                    if build is None:
                        why = ("no hip line" if not any(
                            j in joints for j in ("leftHip", "rightHip", "root"))
                            else "arms too close, or clothing hides the torso")
                        refused[why] = refused.get(why, 0) + 1
                        continue
                    rows.append({
                        "waist_to_shoulder": build.waist_to_shoulder,
                        "hip_to_shoulder": build.hip_to_shoulder,
                        "waist_to_hip": build.waist_to_hip,
                        "shoulder_to_head": build.shoulder_to_head,
                        "title": page.get("title"),
                        "source": (meta.get("DescriptionUrl") or {}).get("value")
                                  or image.get("descriptionurl"),
                        "licence": licence,
                        "creator": (meta.get("Artist") or {}).get("value", "")[:120],
                    })
                    print(f"  [{len(rows):3}] w/s {build.waist_to_shoulder:.3f}  "
                          f"{licence[:14]:16} {str(page.get('title'))[5:52]}")
                time.sleep(1)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\ntried {tried} photographs, measured {len(rows)}")
    for why, count in sorted(refused.items(), key=lambda kv: -kv[1]):
        print(f"  {why}: {count}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
