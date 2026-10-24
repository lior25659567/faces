"""Read a coach's body angles from one reference image with MediaPipe Pose.

CLI: python studio/measure/read_pose.py IMG [IMG...]  -> one JSON object per line.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

from metrics import measure
from posture import FACING, NEUTRAL_EAR_NOSE_DEG, back_profile, head_pitch, neck_vs_torso, seat_depth

MODEL = Path(__file__).resolve().parents[1] / "models" / "pose_landmarker_heavy.task"

NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye",
    "right_eye_outer", "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_pinky",
    "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb", "left_hip",
    "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel",
    "right_heel", "left_foot_index", "right_foot_index",
]
JOINTS = ("shoulder", "elbow", "wrist", "hip", "knee", "ankle")
UP = np.array([0.0, -1.0, 0.0])          # image and world landmarks both have y down
VIEW_NORMAL = np.array([0.0, 0.0, 1.0])  # flat = the image plane
CROP_MARGIN = 0.12                       # of the body box's longer side, on every side
POSTURE_KEYS = ("back_rounding", "back_seen", "back_span", "back_why", "neck_vs_torso", "head_pitch", "seat_below_knee")

_detector = None


class NoBody(Exception):
    """No person was found in the image."""


def _get_detector():
    global _detector
    if _detector is None:
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision
        _detector = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=mpt.BaseOptions(model_asset_path=str(MODEL)), num_poses=1,
            output_segmentation_masks=True))
    return _detector


def _clean(metrics):
    return {k: (None if math.isnan(v) else round(v, 3)) for k, v in metrics.items()}


def body_crop(points, size, margin=CROP_MARGIN):
    """The box around the body's landmarks, grown by `margin` of its longer side on every side, as
    [x0, y0, x1, y1] fractions of the image, clamped to it. The longer side, so a lying body (plank)
    keeps its back and head. None when there are no points or the body is wholly outside the image."""
    if not points:
        return None
    w, h = size
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    pad = margin * max(max(xs) - min(xs), max(ys) - min(ys))
    x0 = min(max((min(xs) - pad) / w, 0.0), 1.0)
    y0 = min(max((min(ys) - pad) / h, 0.0), 1.0)
    x1 = min(max((max(xs) + pad) / w, 0.0), 1.0)
    y1 = min(max((max(ys) + pad) / h, 0.0), 1.0)
    if x1 <= x0 or y1 <= y0:
        return None
    return [round(float(v), 4) for v in (x0, y0, x1, y1)]


def _head_points(pix):
    """MediaPipe has no point on the top of the head, so the box would end at the eyes. Add the four
    extremes of a circle around the ears, with radius 0.7 x nose-to-shoulders (about the head's size
    in any view, and still right for a lying body)."""
    ear = (np.asarray(pix["left_ear"][:2]) + np.asarray(pix["right_ear"][:2])) / 2
    shoulders = (np.asarray(pix["left_shoulder"][:2]) + np.asarray(pix["right_shoulder"][:2])) / 2
    r = 0.7 * float(np.linalg.norm(np.asarray(pix["nose"][:2]) - shoulders))
    return [(ear[0] - r, ear[1]), (ear[0] + r, ear[1]), (ear[0], ear[1] - r), (ear[0], ear[1] + r)]


def _round(v):
    return None if v is None or not math.isfinite(v) else round(float(v), 3)


def _posture(pix, world, mask, side, facing):
    """({"back_rounding", "back_seen", "back_span", "back_why", "neck_vs_torso", "head_pitch"} on the
    picture, the neck in 3D) by studio/measure/posture.py; all None off a side view (no facing). Flat: the
    silhouette `mask` (MediaPipe's person mask, 0-1 per pixel) with the arms left out within
    posture.ARM_RADIUS of their landmark segments (shoulder -> elbow -> wrist -> index fingertip), the
    camera-side ear, and the face's direction from that ear to the nose plus NEUTRAL_EAR_NOSE_DEG; the seat's lowest
    point below the top of the knee, by the camera-side hip and knee (posture.seat_depth: the depth as a judge sees
    it, task 22). 3D: the neck on
    the body's side plane (square to the hip line) from the ears' midpoint, as the shoulders' midpoint
    is: MediaPipe's world depth puts the near ear about 11 cm toward the camera from the shoulders."""
    flat = dict.fromkeys(POSTURE_KEYS)
    if facing not in FACING:
        return flat, None

    def at(points, name):
        return np.asarray(points[name], dtype=float)

    def mid(points, name, dims):
        return (at(points, f"left_{name}")[:dims] + at(points, f"right_{name}")[:dims]) / 2

    shoulder, hip = mid(pix, "shoulder", 2), mid(pix, "hip", 2)
    forward = np.array(FACING[facing])
    if mask is not None:
        arms = [(at(pix, f"{s}_{a}")[:2], at(pix, f"{s}_{b}")[:2]) for s in ("left", "right")
                for a, b in (("shoulder", "elbow"), ("elbow", "wrist"), ("wrist", "index"))]
        profile = back_profile(mask, shoulder, hip, facing, arms)
        if profile is not None:
            flat["back_rounding"], flat["back_seen"] = profile["rounding"], profile["seen"]
            flat["back_span"] = profile["span"] and [round(float(v), 3) for v in profile["span"]]
            flat["back_why"] = profile["why"]
        near_hip, near_knee = at(pix, f"{side}_hip")[:2], at(pix, f"{side}_knee")[:2]
        flat["seat_below_knee"] = seat_depth(mask, near_hip, near_knee, facing)["depth"]
    ear = at(pix, f"{side}_ear")[:2]
    flat["neck_vs_torso"] = neck_vs_torso(hip, shoulder, ear, forward, up=UP[:2])
    flat["head_pitch"] = head_pitch(at(pix, "nose")[:2] - ear, (0.0, -1.0), forward, offset=NEUTRAL_EAR_NOSE_DEG)
    ears = mid(world, "ear", 3)
    cross = neck_vs_torso(mid(world, "hip", 3), mid(world, "shoulder", 3), ears, at(world, "nose") - ears,
                          normal=at(world, "left_hip") - at(world, "right_hip"), up=UP)
    return {k: v if k in ("back_span", "back_why") else _round(v) for k, v in flat.items()}, _round(cross)


def _posture_or_none(pix, world, masks, size, side, facing, name):
    """_posture on the detector's person mask (`masks`), or every posture key None (said on stderr) when
    it fails: one posture error never costs the image its other measurements."""
    try:
        mask = np.asarray(masks[0].numpy_view(), dtype=float).reshape(size[1], size[0]) if masks else None
        return _posture(pix, world, mask, side, facing)
    except Exception as e:      # noqa: BLE001 - reported, and the checks then say "not measured"
        print(f"{name}: posture not measured: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        return dict.fromkeys(POSTURE_KEYS), None


def _joints(points):
    return {f"{j}.{s}": np.asarray(points[f"{s}_{j}"], dtype=float) for j in JOINTS for s in ("left", "right")}


def load_rgb(path):
    """Any picture PIL can open (PNG in any mode, JPEG, WebP...) as an upright 8-bit RGB array: turned
    the way its EXIF orientation says, as a browser shows it, so the crop and landmarks line up with
    the picture on the page."""
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        return np.ascontiguousarray(np.asarray(ImageOps.exif_transpose(im).convert("RGB"), dtype=np.uint8))


def read_pose(path):
    import mediapipe as mp
    path = Path(path)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=load_rgb(path))
    result = _get_detector().detect(img)
    if not result.pose_landmarks:
        raise NoBody(f"no person found in {path.name}")
    lm, wl = result.pose_landmarks[0], result.pose_world_landmarks[0]
    w, h = img.width, img.height
    pix = {n: (lm[i].x * w, lm[i].y * h, 0.0) for i, n in enumerate(NAMES)}
    world = {n: (wl[i].x, wl[i].y, wl[i].z) for i, n in enumerate(NAMES)}
    vis = {n: lm[i].visibility for i, n in enumerate(NAMES)}

    side = "left" if vis["left_knee"] + vis["left_elbow"] >= vis["right_knee"] + vis["right_elbow"] else "right"
    flat_j, world_j = _joints(pix), _joints(world)

    sh_mid = (flat_j["shoulder.left"] + flat_j["shoulder.right"]) / 2
    hip_mid = (flat_j["hip.left"] + flat_j["hip.right"]) / 2
    span = float(np.linalg.norm(flat_j["shoulder.left"] - flat_j["shoulder.right"])
                 / np.linalg.norm(sh_mid - hip_mid))
    view = "side" if span < 0.35 else "frontal" if span > 0.55 else "three-quarter"
    facing = None
    if view == "side":
        facing = "right" if pix["nose"][0] > sh_mid[0] else "left"

    flat, posture3d = _posture_or_none(pix, world, getattr(result, "segmentation_masks", None), (w, h), side,
                                       facing, path.name)

    return {
        "image": path.name,
        "size": [w, h],
        "view": view,
        "shoulder_span": round(span, 3),
        "camera_side": side,
        "facing": facing,
        "crop": body_crop([p[:2] for p in pix.values()] + _head_points(pix), (w, h)),
        "flat": {**_clean(measure(flat_j, side, UP, VIEW_NORMAL)), **flat},
        "3d": {**_clean(measure(world_j, side, UP, None)), "neck_vs_torso": posture3d},
        "world": {n: [round(float(c), 5) for c in world[n]] for n in NAMES},
        # the picture's own landmarks (pixels, y down): what the flat readings are measured on. The key positions
        # an exercise without a clip is animated from take their side plane from these (task 9,
        # pipeline/blender/keypose_clip.py): in a side picture the world landmarks' depth is the noisy guess
        "pixels": {n: [round(float(pix[n][0]), 2), round(float(pix[n][1]), 2)] for n in NAMES},
        # how surely each joint's landmark is seen (0-1; picture.py gates our own picture on it, task 19)
        "visibility": {f"{s}_{j}": round(float(vis[f"{s}_{j}"]), 3) for j in JOINTS for s in ("left", "right")},
    }


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        try:
            print(json.dumps(read_pose(Path(arg))))
        except NoBody as e:
            print(json.dumps({"image": Path(arg).name, "error": str(e)}))
