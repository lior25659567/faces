"""A side picture's key position as studio/measure/read_pose.py would read it, built from known side-seen angles:
for pipeline/blender/keypose_clip.py's tests (studio/tests/test_keypose_clip.py, and in Blender
studio/tests/blender/test_keypose.py). numpy only."""
import math

import numpy as np


def sagittal(lean, hip, knee):
    """(forward, up) directions of the trunk, thigh and shin for a side-seen lean, hip and knee angle."""
    rad = math.radians
    trunk = (math.sin(rad(lean)), math.cos(rad(lean)))
    thigh = (math.sin(rad(lean + hip)), math.cos(rad(lean + hip)))
    shin = (math.sin(rad(lean + hip + 180 - knee)), math.cos(rad(lean + hip + 180 - knee)))
    return trunk, thigh, shin


def mp_pose(lean, hip, knee, facing="right", face_down=12.0, foot_pitch=0.0, phase="start", arms="down", px_per_m=400.0):
    """A side picture's key position as read_pose gives it: MediaPipe world landmarks (x right, y down, z depth,
    metres, hips at the origin) and the picture's own landmarks (pixels), for a person facing `facing` with the
    given side-seen angles. Arms hang straight down, or with `arms="forward"` reach straight ahead at shoulder
    height (an air squat's); the ear->nose line points `face_down` below level. The picture has `px_per_m` pixels
    per metre."""
    s = 1.0 if facing == "right" else -1.0
    trunk, thigh, shin = sagittal(lean, hip, knee)

    def at(fwd, up, lat):                      # person's forward / up / left -> MediaPipe world
        return np.array([s * fwd, -up, s * lat])
    hip_mid = np.zeros(3)
    sh_mid = hip_mid + 0.5 * at(*trunk, 0.0)
    pts = {}
    for side, lat in (("left", 1.0), ("right", -1.0)):
        h = hip_mid + at(0.0, 0.0, 0.1 * lat)
        k = h + 0.42 * at(*thigh, 0.0)
        a = k + 0.40 * at(*shin, 0.0)
        pitch = math.radians(foot_pitch)
        heel = a + at(-0.05, -0.06, 0.0)
        toe = heel + 0.2 * at(math.cos(pitch), -math.sin(pitch), 0.0)
        sh = sh_mid + at(0.0, 0.0, 0.18 * lat)
        reach = at(1.0, 0.0, 0.0) if arms == "forward" else at(0.0, -1.0, 0.0)
        pts.update({f"{side}_hip": h, f"{side}_knee": k, f"{side}_ankle": a, f"{side}_heel": heel,
                    f"{side}_foot_index": toe, f"{side}_shoulder": sh, f"{side}_elbow": sh + 0.28 * reach,
                    f"{side}_wrist": sh + 0.53 * reach, f"{side}_index": sh + 0.62 * reach,
                    f"{side}_ear": sh_mid + at(0.0, 0.22, 0.07 * lat)})
    ears = (pts["left_ear"] + pts["right_ear"]) / 2
    d = math.radians(face_down)
    pts["nose"] = ears + 0.1 * at(math.cos(d), -math.sin(d), 0.0)
    world = {n: [float(c) for c in p] for n, p in pts.items()}
    pixels = {n: [500.0 + px_per_m * p[0], 500.0 + px_per_m * p[1]] for n, p in pts.items()}
    return {"phase": phase, "view": "side", "facing": facing, "camera_side": facing, "world": world,
            "pixels": pixels}
