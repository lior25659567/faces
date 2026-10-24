#!/usr/bin/env python
"""Turn one character still into the frames <FaceFollow /> needs.

Renders the head at a grid of angles with LivePortrait (local, no API) and writes
src/faces/<name>/{base.webp, atlas.webp, manifest.json}.

    npm run frames -- input/me.png
    npm run frames -- input/me.png --name me --steps 7 --yaw 18 --pitch 12
"""

import argparse
import hashlib
import itertools
import json
import os
import sys
import time
from pathlib import Path

# LivePortrait's 3D grid_sample has no MPS kernel, so that op falls back to the CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "pipeline" / "vendor" / "LivePortrait"
sys.path.insert(0, str(VENDOR))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.config.inference_config import InferenceConfig  # noqa: E402
from src.live_portrait_wrapper import LivePortraitWrapper  # noqa: E402
from src.utils.camera import get_rotation_matrix  # noqa: E402
from src.utils.crop import paste_back, prepare_paste_back  # noqa: E402
from src.utils.io import resize_to_limit  # noqa: E402

from apple_vision import faces  # noqa: E402

CROP_SIZE = 512  # LivePortrait's working resolution
MAX_TEXTURE = 4096  # safe single-image size on phones
GUTTER = 16  # px between frames in the atlas, filled with each frame's own edge: WebP smears a few px across a hard seam
NO_FACE = 3  # exit status the dev server turns into its own message


# Restoring the still's detail, see detail_restorer(). Each pair is (full detail up to here, none from here).
FLOW_MISS = (10, 30)  # how far a frame may be from what the flow predicts, per channel out of 255
FLOW_DRIFT = (1.0, 3.0)  # how far following the flow there and back may land from the start, in pixels

# The part of the head box the atlas keeps, see moving_box().
MOVED = 12  # an area counts as moving once it differs from the straight-ahead frame by this much, out of 255
FEATHER = 0.08  # still pixels kept around the moving part, as a share of it: <FaceFollow /> fades its outer 6%

# Distance in RGB between neighboring backdrop stops, see backdrop_stops().
# A studio backdrop's own gradient moves about 5 from one stop to the next; clothing jumps by 100 or more.
BACKDROP_JUMP = 40

# LivePortrait's own axes, found by rendering the extremes and looking at them.
# The grid is defined in screen space: u = -1 looks at screen-left, v = -1 looks up.
YAW_SIGN = -1
PITCH_SIGN = 1
EYE_X_SIGN = 1
EYE_Y_SIGN = -1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="front-facing character still")
    parser.add_argument("--name", help="output folder name (default: the source file name)")
    parser.add_argument("--out", type=Path, help="output folder (default: src/faces/<name>)")
    parser.add_argument("--steps", type=int, default=7, help="frames across, left to right, odd (default: 7). The atlas has a fixed size, so more frames means smaller, softer ones")
    parser.add_argument("--rows", type=int, help="frames down, odd (default: as many as give the same turn per frame as across, 5 for the default angles)")
    parser.add_argument("--yaw", type=float, default=18, help="head turn left/right at the edges, degrees")
    parser.add_argument("--pitch", type=float, default=12, help="head tilt up/down at the edges, degrees")
    parser.add_argument("--eyes-x", type=float, default=8, help="extra eye travel left/right at the edges (over about 12 smears large cartoon eyes)")
    parser.add_argument("--eyes-y", type=float, default=18, help="extra eye travel up/down at the edges")
    parser.add_argument("--scale", type=float, default=2.5, help="width of the animated square, in face widths (1.8 to 3.2)")
    parser.add_argument("--face", help="skip face finding: x0,y0,x1,y1 in pixels of the source, forehead to chin and cheek to cheek")
    parser.add_argument("--cell", type=int, help="longest side of one frame in the atlas, px (default: as large as a 4096 px atlas allows)")
    parser.add_argument("--max-dim", type=int, default=1280, help="the still is shrunk to this before rendering")
    parser.add_argument("--quality", type=int, default=82, help="WebP quality of the atlas")
    args = parser.parse_args()
    if args.steps < 3 or args.steps % 2 == 0:
        parser.error("--steps must be odd and at least 3, so there is a straight-ahead frame")
    if args.rows is None:
        # The head tilts less than it turns, so it needs fewer frames down than across.
        even_turn = args.steps * args.pitch / args.yaw if args.yaw else args.steps
        args.rows = max(3, 2 * round((even_turn - 1) / 2) + 1)
    if args.rows < 3 or args.rows % 2 == 0:
        parser.error("--rows must be odd and at least 3, so there is a straight-ahead frame")
    if not args.source.is_file():
        parser.error(f"no such file: {args.source}")
    if args.face:
        try:
            args.face = [float(value) for value in args.face.split(",")]
            assert len(args.face) == 4
        except (ValueError, AssertionError):
            parser.error("--face takes four numbers: x0,y0,x1,y1")
    return args


def load_still(path, max_dim):
    """The still as RGB, shrunk to LivePortrait's limit, and how much it was shrunk by."""
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    still = np.ascontiguousarray(resize_to_limit(np.asarray(image), max_dim=max_dim, division=2))
    return still, still.shape[1] / image.width


def head_crop(still, box, scale):
    """The square LivePortrait animates, and the matrix that maps it back onto the still.

    LivePortrait frames this square from facial landmarks, but its landmark model collapses on
    stylized faces. On its own example photos the square comes out `scale` face-box widths wide,
    centered on the box horizontally and 41% of the way down it, so the box alone reproduces the
    framing. The head is assumed to be upright.
    """
    x0, y0, x1, y1 = box
    side = scale * (x1 - x0)
    left = (x0 + x1) / 2 - side / 2
    top = y0 + 0.41 * (y1 - y0) - side / 2
    crop_to_still = np.array([[side / CROP_SIZE, 0, left], [0, side / CROP_SIZE, top], [0, 0, 1]], dtype=np.float32)
    # In a close-up the square reaches past the still. The still's edge is stretched out to fill it:
    # left black, the fill shows as a dark line along the edge whenever the head moves away from it.
    crop = cv2.warpAffine(
        still,
        np.linalg.inv(crop_to_still)[:2],
        (CROP_SIZE, CROP_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return crop, crop_to_still


def shift_eyeballs(exp, x, y):
    """Move the two eyeball keypoints (11, 15) and let the lids follow.

    Constants are LivePortrait's own, from GradioPipeline.update_delta_new_eyeball_direction.
    """
    exp = exp.clone()
    left, right = (0.0007, 0.001) if x > 0 else (0.001, 0.0007)
    exp[0, 11, 0] += x * left
    exp[0, 15, 0] += x * right
    exp[0, 11, 1] += y * -0.001
    exp[0, 15, 1] += y * -0.001
    blink = -y / 2.0
    exp[0, 11, 1] += blink * -0.001
    exp[0, 13, 1] += blink * 0.0003
    exp[0, 15, 1] += blink * -0.001
    exp[0, 16, 1] += blink * 0.0003
    return exp


def each_cell(grid):
    """The grid's cells in order, with a progress bar in a terminal and plain `frame i/n` lines
    when piped, which is what the dev server reads."""
    if sys.stderr.isatty():
        yield from tqdm(grid, desc="frames", unit="frame")
        return
    for index, cell in enumerate(grid, 1):
        yield cell
        print(f"frame {index}/{len(grid)}", flush=True)


def head_box(mask):
    """Bounding box of every pixel the paste-back mask lets LivePortrait change."""
    ys, xs = np.nonzero(mask.max(axis=2) > 0)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def detail_restorer(still, neutral):
    """A function that gives a rendered frame back the fine detail of the still.

    LivePortrait renders the head at 512 px, softer than the still. What it lost is known for the
    straight-ahead frame: the still minus that frame. Optical flow says where each part of the
    straight-ahead frame went in another frame, so the lost detail is moved along and added back.

    Nothing is added where the flow cannot be trusted, and that part stays as soft as LivePortrait
    drew it. That is wherever the frame differs from what the flow predicts, and wherever following
    the flow there and back does not return to the start: skin that was hidden in the straight-ahead
    frame, such as the neck under a lifted chin, has nowhere to come from, and would otherwise
    inherit the outline of whatever covered it.
    """
    detail = still.astype(np.float32) - neutral
    neutral_gray = cv2.cvtColor(neutral, cv2.COLOR_RGB2GRAY)
    finder = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    ys, xs = np.mgrid[0 : still.shape[0], 0 : still.shape[1]].astype(np.float32)

    def within(value, limits):
        """1 up to the first limit, falling to 0 at the second."""
        full, none = limits
        return np.clip((none - value) / (none - full), 0, 1)

    def restore(frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        back = finder.calc(frame_gray, neutral_gray, None)  # from each pixel of the frame to where it was
        forth = finder.calc(neutral_gray, frame_gray, None)
        source = (xs + back[..., 0], ys + back[..., 1])

        predicted = cv2.remap(neutral, *source, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        miss = cv2.absdiff(predicted, frame).max(axis=2).astype(np.float32)
        returned = cv2.remap(forth, *source, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        drift = np.linalg.norm(back + returned, axis=2)
        trust = np.minimum(within(miss, FLOW_MISS), within(drift, FLOW_DRIFT))
        trust = cv2.GaussianBlur(cv2.erode(trust, np.ones((5, 5), np.uint8)), (0, 0), 2)

        moved = cv2.remap(detail, *source, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return np.clip(frame + moved * trust[..., None], 0, 255).astype(np.uint8)

    return restore


def moving_box(base, frames, limit):
    """Box around the head: the largest area that differs between the straight-ahead frame and
    the given ones, with a margin for the page's feathered edge, kept inside `limit`.

    The paste-back mask covers a square much larger than the head, most of it backdrop and
    shoulders. Leaving those out spends the atlas's pixels on the head. A collar or necklace
    that shifts a little on its own, away from the head, is left as it is in the base.
    """
    left, top, right, bottom = limit
    size = max(3, max(right - left, bottom - top) // 32) | 1
    moved = np.zeros(base.shape[:2], dtype=np.float32)
    for frame in frames:
        change = cv2.absdiff(frame, base).max(axis=2).astype(np.float32)
        moved = np.maximum(moved, cv2.blur(change, (size, size)))
    areas = (moved[top:bottom, left:right] > MOVED).astype(np.uint8)
    count, _, boxes, _ = cv2.connectedComponentsWithStats(areas)
    if count < 2:
        return limit
    x, y, wide, tall, _ = boxes[1 + np.argmax(boxes[1:, cv2.CC_STAT_AREA])]
    pad_x, pad_y = FEATHER * wide, FEATHER * tall
    return (
        int(max(left, left + x - pad_x)),
        int(max(top, top + y - pad_y)),
        int(min(right, left + x + wide + pad_x)),
        int(min(bottom, top + y + tall + pad_y)),
    )


def backdrop_stops(still, count=12):
    """The backdrop's color down the still, as evenly spaced gradient stops from top to bottom.

    Sampled at the left and right edges, so a page can continue the backdrop past the still.
    The median keeps a hand or sleeve that touches one edge from tinting a stop. A stop that jumps
    away from the one above it is not backdrop but something in front of it, such as shoulders
    that reach both edges of a close-up, so it repeats the color above.
    """
    edge = max(2, still.shape[1] // 50)
    edges = np.concatenate([still[:, :edge], still[:, -edge:]], axis=1)
    stops = []
    for band in np.array_split(edges, count):
        color = np.median(band.reshape(-1, 3), axis=0)
        if stops and np.linalg.norm(color - stops[-1]) > BACKDROP_JUMP:
            color = stops[-1]
        stops.append(color)
    return ["#%02x%02x%02x" % tuple(int(channel) for channel in color) for color in stops]


@torch.no_grad()
def main():
    args = parse_args()
    name = args.name or args.source.stem
    out = args.out or ROOT / "src" / "faces" / name
    started = time.time()

    still, shrink = load_still(args.source, args.max_dim)
    height, width = still.shape[:2]
    if args.face:
        box = np.array(args.face) * shrink
    else:
        found = faces(Image.fromarray(still))
        box = np.array(found[0]) if found else None
        if box is None:
            print("No face found. Pass its box yourself: --face x0,y0,x1,y1 (pixels, forehead to chin, cheek to cheek).", file=sys.stderr)
            sys.exit(NO_FACE)
    crop, crop_to_still = head_crop(still, box, args.scale)

    wrapper = LivePortraitWrapper(inference_cfg=InferenceConfig())
    source = wrapper.prepare_source(cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA))
    info = wrapper.get_kp_info(source)
    features = wrapper.extract_feature_3d(source)
    kp_source = wrapper.transform_keypoint(info)
    translation = info["t"].clone()
    translation[..., 2] = 0  # depth shift is unused, as in transform_keypoint
    mask = prepare_paste_back(wrapper.inference_cfg.mask_crop, crop_to_still, dsize=(width, height))

    def render(u, v):
        """Full still with the head looking at (u, v), each in [-1, 1]."""
        rotation = get_rotation_matrix(
            info["pitch"] + PITCH_SIGN * v * args.pitch,
            info["yaw"] + YAW_SIGN * u * args.yaw,
            info["roll"],
        )
        exp = shift_eyeballs(info["exp"], EYE_X_SIGN * u * args.eyes_x, EYE_Y_SIGN * v * args.eyes_y)
        kp_target = info["scale"] * (info["kp"] @ rotation + exp) + translation
        kp_target = wrapper.stitching(kp_source, kp_target)
        decoded = wrapper.warp_decode(features, kp_source, kp_target)
        return paste_back(wrapper.parse_output(decoded["out"])[0], crop_to_still, still, mask)

    reach = head_box(mask)
    changed = np.s_[reach[1] : reach[3], reach[0] : reach[2]]
    restore = detail_restorer(still[changed], render(0, 0)[changed])

    def pose(u, v):
        """render(u, v) with the still's own sharpness."""
        frame = render(u, v)
        frame[changed] = restore(frame[changed])
        return frame

    # The straight-ahead frame and the eight furthest ones come first: between them they show
    # how much of the still ever moves, which decides what the atlas has to hold.
    cols, rows = args.steps, args.rows
    across, down = np.linspace(-1, 1, cols), np.linspace(-1, 1, rows)
    grid = [(row, col) for row in range(rows) for col in range(cols)]
    first = [(row, col) for row in (rows // 2, 0, rows - 1) for col in (cols // 2, 0, cols - 1)]
    cells = each_cell(first + [cell for cell in grid if cell not in first])
    early = {(row, col): pose(across[col], down[row]) for row, col in itertools.islice(cells, len(first))}

    # The base is the straight-ahead frame, not the raw still, so nothing shifts when the atlas takes over.
    base = early[rows // 2, cols // 2]
    x0, y0, x1, y1 = moving_box(base, early.values(), reach)
    if args.cell:
        fit = min(1.0, args.cell / max(x1 - x0, y1 - y0))
    else:
        room = [(MAX_TEXTURE - (count - 1) * GUTTER) // count for count in (cols, rows)]
        fit = min(1.0, room[0] / (x1 - x0), room[1] / (y1 - y0))
    cell = (int((x1 - x0) * fit), int((y1 - y0) * fit))
    size = (cols * cell[0] + (cols - 1) * GUTTER, rows * cell[1] + (rows - 1) * GUTTER)
    if max(size) > MAX_TEXTURE:
        print(f"warning: the atlas is {size[0]} x {size[1]} px, over the {MAX_TEXTURE} px some phones allow; lower --cell, --steps or --rows")

    atlas = Image.new("RGB", size)
    edge = GUTTER // 2

    def keep(frame, row, col):
        """Put a frame's head in its place in the atlas, with its edge repeated into half the gutter around it."""
        head = np.asarray(Image.fromarray(frame[y0:y1, x0:x1]).resize(cell, Image.LANCZOS))
        padded = np.pad(head, ((edge, edge), (edge, edge), (0, 0)), mode="edge")
        atlas.paste(Image.fromarray(padded), (col * (cell[0] + GUTTER) - edge, row * (cell[1] + GUTTER) - edge))

    for (row, col), frame in early.items():
        keep(frame, row, col)
    del early
    for row, col in cells:
        keep(pose(across[col], down[row]), row, col)

    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(base).save(out / "base.webp", quality=90, method=6)
    atlas.save(out / "atlas.webp", quality=args.quality, method=6)
    manifest = {
        "width": width,
        "height": height,
        "head": {
            "x": x0 / width,
            "y": y0 / height,
            "width": (x1 - x0) / width,
            "height": (y1 - y0) / height,
        },
        "cols": cols,
        "rows": rows,
        # The space between frames in the atlas, as a share of one frame.
        "gutter": {"x": GUTTER / cell[0], "y": GUTTER / cell[1]},
        "backdrop": backdrop_stops(still),
        # Changes whenever the frames do, so pages can tell a re-render from a cached image.
        "version": hashlib.sha1((out / "atlas.webp").read_bytes()).hexdigest()[:8],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    size_kb = sum((out / f).stat().st_size for f in ("base.webp", "atlas.webp")) // 1024
    print(f"{cols * rows} frames in {time.time() - started:.0f}s -> {out} ({size_kb} KB)")


if __name__ == "__main__":
    main()
