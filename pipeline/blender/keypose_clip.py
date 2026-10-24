#!/usr/bin/env python
"""Motion from the reference pictures' key positions, for an exercise that has no Mixamo clip.

    blender --background --python pipeline/blender/animate.py -- --mpfb --exercise NAME \
        --wrap-skin "3d-character/3d cartoon boy.glb" --keyposes RUN/keyposes.json --out RUN/frames

The Exercise Studio's read step writes RUN/keyposes.json (studio/measure/run_read.py): for each key side
picture, in start, middle, end order, MediaPipe's WORLD landmarks (metres, hips at the origin, x right, y DOWN,
z depth) and the picture's own landmarks (pixels). `clip_from_keyposes` turns them into a `retarget.Clip` - the
same thing `retarget.load_clip` reads out of a Mixamo FBX - so it enters `animate.run_mpfb` where a clip does
and everything after it is unchanged: `retarget.BoneRetarget` (bone for bone, the neck's share, grounding, the
planted feet locked), and for a solver exercise the form targets, the grip and the picture loop.

## The rest it is measured against

`rest` is the Mixamo Y Bot's rest pose read from 3d-character/mixamo/squat.fbx (every Mixamo FBX shares it):
a T-pose facing -Y, its left +X, up +Z. A Mixamo clip's frames are that skeleton's bones turned from this rest,
and `BoneRetarget` carries each bone's turn from rest onto the rig's own rest. So each key position becomes,
per bone, a TURN FROM REST, worked out in the character's frame (person's left -> +X, up -> +Z, facing -Y):

  * limbs: the shortest turn taking the bone's rest direction onto its landmark direction, on top of its
    parent's turn (thigh on the pelvis, shin on the thigh, upper arm on the chest, forearm on the upper arm), so
    a bone's roll is carried down the chain rather than invented per bone (an elbow or a knee twisted against
    the bone above it is a candy-wrapper fold in the skin); hands keep the forearm's turn. The legs' rest
    direction is not the Y Bot's (its rest knee is bent 4.4 degrees, ankle 3 cm behind the knee, while the rig's
    is straight; counted from it every knee came out 4.4 degrees too straight - the Blender test read 94.4 for 90)
    but the rig's own thigh as the Studio reads it: its skin outline in the bind pose (`leg_rest`; the cartoon's
    leans back 4.7 degrees, task 20), the shin straight on from it, so a picture's thigh puts our thigh's skin
    where the picture's is and the knee bends as the picture's; a straight, vertical leg when not given;
  * pelvis and chest: a full frame each - the hips' or the shoulders' left-right line and the hip-mid ->
    shoulder-mid line. Both stand on that one trunk line, so the back stays straight (the Studio wants a
    straight back for 12 of the 14 exercises and judges ours on the spine bones' bends); the two spine bones in
    between share equally what differs between pelvis and chest (the shoulders' turn and tilt on the hips).
    Counted from UPRIGHT, not from the Y Bot's rest line, which leans back 6.7 degrees: a person standing tall
    reads 0 degrees in MediaPipe, and the Studio counts ours from the rig's bind pose (task 18);
  * head: the face's direction (ears -> nose, raised by the 12 degrees a level gaze's ear -> nose line points
    below level - studio/measure/posture.py NEUTRAL_EAR_NOSE_DEG), the neck half way between chest and head.
    BoneRetarget then gives the neck and head only `NECK_SHARE` (40%) of their turns, exactly as it does a
    Mixamo clip's;
  * feet: heel -> toe. A heel -> toe line within FLAT_FOOT_DEG of level is a foot flat on the floor and is
    made level (MediaPipe read the coach's flat feet 9 degrees toe-down). Planted feet keep the first
    position's foot in every frame, and its stance in every position: a planted foot neither turns (its toes
    would slide) nor moves (the ankle lock would pull the legs apart: the squat's bottom knee read 55 for 47).

## The coach's depth is noisy: the side plane comes from the picture

MediaPipe's world landmarks are a 3D lift of one picture, and in a side picture the depth axis IS the body's
left-right axis - the guess it makes least well. On the squat's pictures the 3D knee disagrees with the flat
one by up to 17 degrees, the hip line reads 15 degrees off square and the standing ankle sits 15 cm out from a
straight knee. The Studio judges knee, hip and lean on the picture ("flat"), so in a side picture the key
position is built the way the Studio reads it:

  * the frame is square to the picture (the picture plane is the side plane, as the Studio assumes); the hips'
    3D line only says which side is left, and a picture whose left and right labels disagree with which way
    the person faces is swapped back;
  * every landmark's side-plane position (forward/back, up/down) is the PICTURE's, scaled to metres; only its
    left-right offset comes from the 3D depth - so the side-seen angles of the trunk, legs and feet are exactly
    the flat readings the Studio judges, and the stance width and knees-out survive;
  * left-right is made symmetric (every exercise here is): each left/right pair's offsets from the midline are
    averaged, every midline point put on it. On the squat's pictures the depth put the near ankle on the far
    side at the bottom and the stance and knees lopsided; what both sides agree on (stance, knees out) stays;
  * a knee is a hinge: its left-right offset from the hip -> ankle line may tip the leg's bend plane at most
    KNEE_OUT_MAX out or KNEE_IN_MAX in (a nearly straight leg has almost no bend to tip, so depth noise there
    no longer bends it sideways: the coach's standing knees read 51-54 degrees inward, held to 10);
  * the pelvis, chest and head are square (their left-right lines run straight into the picture, along the
    depth);
  * arms are the seen arm's 3D landmarks, the hidden arm its mirror image: the Studio judges elbows and upper
    arms in 3D, because in a side picture the arm folds toward the camera (the coach's flat elbow reads 6
    degrees, his 3D one 90), and MediaPipe's guess at the arm it cannot see stuck straight out sideways at the
    squat's middle and end.

Off a side view (no facing known), everything comes from the 3D landmarks, the frame from the hips' left-right
line made level.

## Timing: a real lift

Start -> middle -> end over 1.5 s, held 0.2 s, back end -> middle -> start over 1.0 s (the lift reverses along
one path); each stroke eased in and out (smoothstep), every bone's rotation slerped between neighbouring
positions. The middle sits along the stroke where its share of the bones' turning puts it. Without a middle
picture the stroke runs start -> end. At 30 fps: 81 frames, 45 down, 6 held, 30 up (the last leads into the first:
the rep loops, so no frame is shown twice).

The numpy core (everything above `clip_from_keyposes`) runs without Blender and is tested by
studio/tests/test_keypose_clip.py; the whole path by studio/tests/blender/test_keypose.py.
"""
import math
from pathlib import Path

import numpy as np

FPS = 30
DESCENT_S, HOLD_S, ASCENT_S = 1.5, 0.2, 1.0
PHASES = ("start", "middle", "end")
REST_CLIP = Path(__file__).resolve().parents[2] / "3d-character" / "mixamo" / "squat.fbx"

UP = np.array([0.0, -1.0, 0.0])          # MediaPipe world landmarks: y is down
FACING_X = {"right": 1.0, "left": -1.0}  # read_pose's "facing": the way the nose points along the picture's x
X, Z = np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
FORWARD = np.array([0.0, -1.0, 0.0])     # the character's front
DOWN = -Z                                # a straight, vertical leg: what thigh and shin are counted from by default


def leg_rest(lean):
    """The legs' rest direction for a rig whose thigh's skin outline points `lean` degrees forward (+) of straight
    down in its bind pose (mpfb_cage.rest_thigh_outlines; the cartoon's -4.7, leaning back), character frame."""
    a = math.radians(lean)
    return math.cos(a) * DOWN + math.sin(a) * FORWARD

KNEE_OUT_MAX = 45.0       # degrees a leg's bend plane may tip outward from the side plane (knees out)
KNEE_IN_MAX = 10.0        # ... and inward
FLAT_FOOT_DEG = 15.0      # a heel -> toe line within this of level is a foot flat on the floor
NEUTRAL_EAR_NOSE_DEG = 12.0   # a level gaze's ear -> nose line points this far below level (posture.py)
NECK_FOLLOW = 0.5         # the neck's part of the head's turn on the chest (BoneRetarget then shares each 40%)
# landmarks whose picture and 3D positions set the picture's scale (the trunk and legs: what the Studio judges)
FIT = tuple(f"{s}_{j}" for s in ("left", "right") for j in ("hip", "shoulder", "knee", "ankle", "heel",
                                                             "foot_index"))
FEET = ("LeftFoot", "RightFoot", "LeftToeBase", "RightToeBase")
ARM = ("shoulder", "elbow", "wrist", "pinky", "index", "thumb")      # MediaPipe's arm landmarks, after left_/right_
STANCE = ("ankle", "heel", "foot_index")                            # ... and the foot's


# --- rotations -------------------------------------------------------------------------------------

def _unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("a direction of zero length")
    return v / n


def _level(v):
    return np.array([v[0], v[1], 0.0])


def _rotation(axis, angle):
    """Rodrigues: the turn by `angle` radians about the unit `axis`."""
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


def swing(a, b):
    """The shortest turn taking direction a onto direction b (no roll about either)."""
    a, b = _unit(a), _unit(b)
    cross, dot = np.cross(a, b), float(a @ b)
    size = float(np.linalg.norm(cross))
    if size < 1e-12:
        if dot > 0:
            return np.eye(3)
        other = np.cross(a, X)
        other = other if np.linalg.norm(other) > 1e-6 else np.cross(a, Z)
        return _rotation(_unit(other), math.pi)
    return _rotation(cross / size, math.atan2(size, dot))


def _frame(primary, secondary):
    p = _unit(primary)
    s = _unit(secondary - (secondary @ p) * p)
    return np.column_stack([p, s, np.cross(p, s)])


def frame_turn(primary0, secondary0, primary1, secondary1):
    """The turn taking primary0 onto primary1 exactly, and secondary0 as near secondary1 as that allows."""
    return _frame(primary1, secondary1) @ _frame(primary0, secondary0).T


def _quat(r):
    """(w, x, y, z) of a rotation matrix."""
    t = np.trace(r)
    if t > 0:
        s = 2.0 * math.sqrt(t + 1.0)
        q = [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * math.sqrt(1.0 + r[i, i] - r[j, j] - r[k, k])
        q = [0.0] * 4
        q[0] = (r[k, j] - r[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (r[j, i] + r[i, j]) / s
        q[1 + k] = (r[k, i] + r[i, k]) / s
    return np.array(q) / np.linalg.norm(q)


def _matrix(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def slerp(r0, r1, t):
    """The rotation t of the way from r0 to r1 along the shorter arc."""
    q0, q1 = _quat(r0), _quat(r1)
    if q0 @ q1 < 0.0:
        q1 = -q1
    theta = math.acos(min(1.0, float(q0 @ q1)))
    if theta < 1e-9:
        return _matrix(q0)
    q = (math.sin((1.0 - t) * theta) * q0 + math.sin(t * theta) * q1) / math.sin(theta)
    return _matrix(q / np.linalg.norm(q))


def _turned_by(r):
    """Radians a rotation turns through."""
    return math.acos(max(-1.0, min(1.0, (float(np.trace(r)) - 1.0) / 2.0)))


# --- one key position --------------------------------------------------------------------------------

def _points(d):
    return {n: np.asarray(v, dtype=float) for n, v in d.items()}


def _swap_sides(points):
    out = {}
    for n, p in points.items():
        if n.startswith("left_"):
            out["right_" + n[5:]] = p
        elif n.startswith("right_"):
            out["left_" + n[6:]] = p
        else:
            out[n] = p
    return out


def character_frame(world, facing=None):
    """(rows, swapped): rows are the character's X (the person's left), Y (back) and Z (up) in MediaPipe's
    world axes, so `rows @ p` is p in the character's frame. In a side picture (`facing` "left"/"right") the
    frame is square to the picture and `swapped` says the hips' 3D line put the left hip on the right; off one,
    the hips' left-right line made level."""
    left = world["left_hip"] - world["right_hip"]
    left = left - (left @ UP) * UP
    if facing in FACING_X:
        square = np.cross(UP, np.array([FACING_X[facing], 0.0, 0.0]))
        return np.array([square, np.cross(UP, square), UP]), bool(left @ square < 0.0)
    left = _unit(left) if np.linalg.norm(left) > 1e-9 else None
    if left is None:
        raise ValueError("the hips' left and right landmarks coincide: no left-right line to stand the figure on")
    return np.array([left, np.cross(UP, left), UP]), False


def _hinge_knee(body, side):
    """The knee's left-right offset from the hip -> ankle line held to a bend plane tipped at most KNEE_OUT_MAX
    out / KNEE_IN_MAX in from the side plane. Only X (left-right) moves: the side-seen angles stay."""
    h, k, a = body[f"{side}_hip"], body[f"{side}_knee"], body[f"{side}_ankle"]
    out = 1.0 if side == "left" else -1.0
    line = a[1:] - h[1:]
    t = float((k[1:] - h[1:]) @ line) / max(float(line @ line), 1e-12)
    off = float(np.linalg.norm(k[1:] - h[1:] - t * line))        # the knee's side-seen bend off the line
    on_line = h[0] + t * (a[0] - h[0])
    tilt = math.degrees(math.atan2((k[0] - on_line) * out, off))
    kept = min(max(tilt, -KNEE_IN_MAX), KNEE_OUT_MAX)
    info = {"tilt_deg": round(tilt, 1), "clamped": kept != tilt}
    if kept == tilt:
        return body, info
    knee = k.copy()
    knee[0] = on_line + out * off * math.tan(math.radians(kept))
    return {**body, f"{side}_knee": knee}, info


def _symmetric(points):
    """Left-right symmetric: each left/right pair's offsets from the midline averaged (the left one's and the
    right one's mirrored), every midline point on it. Only X (left-right) moves."""
    out = {}
    for n, p in points.items():
        q = p.copy()
        if n.startswith("left_") and "right_" + n[5:] in points:
            q[0] = (p[0] - points["right_" + n[5:]][0]) / 2
        elif n.startswith("right_") and "left_" + n[6:] in points:
            q[0] = (p[0] - points["left_" + n[6:]][0]) / 2
        elif not n.startswith(("left_", "right_")):
            q[0] = 0.0
        out[n] = q
    return out


def _mirrored_arm(points, seen):
    """`points` with the arm the camera did not see replaced by the seen one mirrored (X negated)."""
    hidden = "left" if seen == "right" else "right"
    out = dict(points)
    for j in ARM:
        if f"{seen}_{j}" in points:
            out[f"{hidden}_{j}"] = points[f"{seen}_{j}"] * np.array([-1.0, 1.0, 1.0])
    return out


def stance_of(body):
    """{foot point: its left-right offset}: where a position's feet stand, for `key_points(..., stance=)`."""
    return {f"{s}_{j}": float(body[f"{s}_{j}"][0]) for s in ("left", "right") for j in STANCE if f"{s}_{j}" in body}


def _with_stance(points, stance):
    out = dict(points)
    for n, x in (stance or {}).items():
        if n in out:
            out[n] = out[n].copy()
            out[n][0] = x
    return out


def key_points(entry, stance=None):
    """(body, arms, found) for one keyposes.json position, in the character's frame, hips' midpoint at the
    origin: `body` for the trunk, legs, feet and head (in a side picture its side plane from the picture), `arms`
    the 3D landmarks (in a side picture the camera's side's arm on both sides). `found` says how it was read.

    In a side picture left-right is MediaPipe's depth guess, and on the squat's pictures it put the near ankle
    on the far side at the bottom, the stance and knees lopsided, and the hidden arm straight out sideways at
    the middle and end. So a side picture's key position is made left-right symmetric (every exercise here is):
    each pair's left-right offsets averaged (what both sides agree on - the stance, knees out - survives) and
    the hidden arm replaced by the seen one mirrored.

    `stance` (stance_of another position) puts the feet's left-right offsets there instead: planted feet stand
    where the first position put them (`key_poses`)."""
    world = _points(entry["world"])
    pixels = _points(entry["pixels"]) if entry.get("pixels") else None
    side_view = entry.get("view", "side") == "side"
    facing = entry.get("facing") if side_view else None
    if side_view and facing not in FACING_X and pixels:
        # set to Side by the user where the detector saw three-quarter (it gives no facing then): the way the nose
        # points from the shoulders' midpoint along the picture, read_pose's own rule
        mid = (pixels["left_shoulder"][0] + pixels["right_shoulder"][0]) / 2
        facing = "right" if pixels["nose"][0] > mid else "left"
    facing = facing if facing in FACING_X else None
    rows, swapped = character_frame(world, facing)
    seen = entry.get("camera_side") if entry.get("camera_side") in ("left", "right") else None
    if swapped:
        world, pixels = _swap_sides(world), (pixels and _swap_sides(pixels))
        seen = seen and ("left" if seen == "right" else "right")
    origin = (world["left_hip"] + world["right_hip"]) / 2
    solid = {n: rows @ (p - origin) for n, p in world.items()}
    found = {"swapped_sides": swapped, "square": facing is not None, "facing": facing, "side_plane": "3d",
             "knees": {}, "mirrored_arm": None}
    if facing is None:
        return _with_stance(solid, stance), solid, found
    arms = solid
    if seen:
        arms = _mirrored_arm(solid, seen)
        found["mirrored_arm"] = "left" if seen == "right" else "right"
    body = dict(solid)
    if pixels:
        sign = FACING_X[facing]
        flat = {n: np.array([-sign * p[0], -p[1]]) for n, p in pixels.items()}     # (back, up) in pixels
        centre = (flat["left_hip"] + flat["right_hip"]) / 2
        flat = {n: p - centre for n, p in flat.items()}
        fit = [n for n in FIT if n in flat and n in solid]
        scale = sum(float(flat[n] @ solid[n][1:]) for n in fit) / max(sum(float(flat[n] @ flat[n]) for n in fit),
                                                                       1e-12)
        if scale > 0.0:
            body = {n: (np.array([p[0], *(scale * flat[n])]) if n in flat else p) for n, p in solid.items()}
            found.update(side_plane="picture", metres_per_pixel=scale)
    body = _with_stance(_symmetric(body), stance)
    for side in ("left", "right"):
        body, found["knees"][side] = _hinge_knee(body, side)
    return body, arms, found


def bone_turns(body, arms, rest, square=False, legs=DOWN):
    """{Mixamo bone: its turn from rest} (3x3, character frame) for one key position. `rest`: the Y Bot's rest
    joints {point: position} in the character frame (Clip.rest's names); `square`: a side picture, whose hips',
    shoulders' and ears' left-right lines are taken square to it (they run along the depth); `legs`: the
    legs' rest direction (`leg_rest`)."""
    hip_mid = (body["left_hip"] + body["right_hip"]) / 2
    trunk = (body["left_shoulder"] + body["right_shoulder"]) / 2 - hip_mid
    lat_hips = X if square else body["left_hip"] - body["right_hip"]
    lat_chest = X if square else body["left_shoulder"] - body["right_shoulder"]
    pelvis = frame_turn(Z, _level(rest["LeftUpLeg"] - rest["RightUpLeg"]), trunk, lat_hips)
    chest = frame_turn(Z, _level(rest["LeftArm"] - rest["RightArm"]), trunk, lat_chest)
    turns = {"Hips": pelvis, "Spine": slerp(pelvis, chest, 1 / 3), "Spine1": slerp(pelvis, chest, 2 / 3),
             "Spine2": chest}

    face = _unit(body["nose"] - (body["left_ear"] + body["right_ear"]) / 2)
    lat_head = X if square else body["left_ear"] - body["right_ear"]
    raise_axis = -_unit(lat_head - (lat_head @ face) * face)
    gaze = _rotation(raise_axis, math.radians(NEUTRAL_EAR_NOSE_DEG)) @ face
    head = frame_turn(FORWARD, X, gaze, lat_head)
    turns.update(Head=head, Neck=slerp(chest, head, NECK_FOLLOW))

    for side, s in (("left", "Left"), ("right", "Right")):
        upper = swing(chest @ (rest[f"{s}ForeArm"] - rest[f"{s}Arm"]),
                      arms[f"{side}_elbow"] - arms[f"{side}_shoulder"]) @ chest
        fore = swing(upper @ (rest[f"{s}Hand"] - rest[f"{s}ForeArm"]),
                     arms[f"{side}_wrist"] - arms[f"{side}_elbow"]) @ upper
        thigh = swing(pelvis @ legs, body[f"{side}_knee"] - body[f"{side}_hip"]) @ pelvis
        shin = swing(thigh @ legs, body[f"{side}_ankle"] - body[f"{side}_knee"]) @ thigh
        sole = body[f"{side}_foot_index"] - body[f"{side}_heel"]
        pitch = math.degrees(math.atan2(sole[2], float(np.linalg.norm(sole[:2]))))
        foot = swing(_level(rest[f"{s}ToeBase"] - rest[f"{s}Foot"]),
                     _level(sole) if abs(pitch) <= FLAT_FOOT_DEG else sole)
        turns.update({f"{s}Shoulder": chest, f"{s}Arm": upper, f"{s}ForeArm": fore, f"{s}Hand": fore,
                      f"{s}UpLeg": thigh, f"{s}Leg": shin, f"{s}Foot": foot, f"{s}ToeBase": foot})
    return turns


def key_pose(entry, rest):
    """bone_turns of one keyposes.json position."""
    body, arms, found = key_points(entry)
    return bone_turns(body, arms, rest, square=found["square"])


def side_angles(body, side):
    """(lean, hip, knee) of `body` seen on its side plane (Y-Z), on `side`: what the key position shows the
    Studio, for the log."""
    def yz(n):
        return np.array([-body[n][1], body[n][2]])

    def included(a, b, c):
        u, v = a - b, c - b
        return math.degrees(math.acos(max(-1.0, min(1.0, float(u @ v) / float(np.linalg.norm(u) * np.linalg.norm(v))))))
    hip = (yz("left_hip") + yz("right_hip")) / 2
    trunk = (yz("left_shoulder") + yz("right_shoulder")) / 2 - hip
    return (math.degrees(math.atan2(trunk[0], trunk[1])),
            included(trunk + yz(f"{side}_hip"), yz(f"{side}_hip"), yz(f"{side}_knee")),
            included(yz(f"{side}_hip"), yz(f"{side}_knee"), yz(f"{side}_ankle")))


def key_poses(keyposes, rest, notes=None, feet="planted", legs=DOWN):
    """[(phase, bone_turns)] in start, middle, end order (the first entry for each phase; middle optional).
    ValueError in plain words without a start and an end. `notes` (a list) gets one line per position. "planted"
    feet stand where the start put them in every position (`stance_of`): a planted foot does not move, and a
    stance read narrower later (the squat's: 17 cm out standing, 9 at the bottom) had the ankle lock pull the legs
    apart and straighten the bottom's knee by 7 degrees."""
    by_phase = {}
    for entry in keyposes.get("phases") or []:
        phase = entry.get("phase")
        if phase in PHASES and phase not in by_phase and entry.get("world"):
            by_phase[phase] = entry
    if "start" not in by_phase or "end" not in by_phase:
        raise ValueError("Need a start and an end position from side pictures to move between; found "
                         + (", ".join(p for p in PHASES if p in by_phase) or "none"))
    out, stance = [], None
    for phase in PHASES:
        if phase not in by_phase:
            continue
        entry = by_phase[phase]
        body, arms, found = key_points(entry, stance=stance)
        if feet == "planted" and stance is None:
            stance = stance_of(body)
        out.append((phase, bone_turns(body, arms, rest, square=found["square"], legs=legs)))
        if notes is not None:
            side = entry.get("camera_side") if entry.get("camera_side") in ("left", "right") else "left"
            lean, hip, knee = side_angles(body, side)
            knees = ", ".join(f"{s} {k['tilt_deg']:+.0f}{' (held to the hinge limit)' if k['clamped'] else ''}"
                              for s, k in found["knees"].items())
            notes.append(f"{phase} ({entry.get('image') or 'no picture id'}): side plane from the "
                         f"{found['side_plane']}{', left and right swapped back' if found['swapped_sides'] else ''}; "
                         f"lean {lean:.1f}, hip {hip:.1f}, knee {knee:.1f} ({side} side)"
                         + (f"; knees tipped out (deg): {knees}" if knees else ""))
    return out


# --- the rep ----------------------------------------------------------------------------------------

def _smoothstep(t):
    return t * t * (3.0 - 2.0 * t)


def timeline(fps=FPS):
    """The path parameter per frame: 0 at the start position, 1 at the end; down over DESCENT_S, held HOLD_S,
    back over ASCENT_S, each stroke eased in and out. The rep loops (the GIF, the app), so the way up stops one frame
    short of the start: the next frame is the first one, not a repeat of it."""
    down, hold, up = (int(round(s * fps)) for s in (DESCENT_S, HOLD_S, ASCENT_S))
    return ([_smoothstep(i / down) for i in range(down + 1)] + [1.0] * hold
            + [_smoothstep(1.0 - i / up) for i in range(1, up)])


def path_marks(keys):
    """Where along the path (0-1) each key position sits: by how far the bones turn between neighbours."""
    gaps = [sum(_turned_by(a[n].T @ b[n]) for n in a) for a, b in zip(keys, keys[1:])]
    total = sum(gaps)
    if total < 1e-9:
        return [i / (len(keys) - 1) for i in range(len(keys))]
    marks, run = [0.0], 0.0
    for gap in gaps:
        run += gap
        marks.append(run / total)
    marks[-1] = 1.0
    return marks


def pose_at(keys, marks, s):
    """Every bone's turn at path parameter s: slerped between the two key positions either side of it."""
    for j in range(len(keys) - 1):
        if s <= marks[j + 1] or j == len(keys) - 2:
            span = marks[j + 1] - marks[j]
            t = 0.0 if span < 1e-12 else min(max((s - marks[j]) / span, 0.0), 1.0)
            return {n: slerp(keys[j][n], keys[j + 1][n], t) for n in keys[j]}
    raise ValueError("no key positions")


def motion(keys, fps=FPS, feet="planted"):
    """Every frame's bone turns. "planted" feet keep the first position's foot throughout."""
    if feet == "planted":
        keys = [{**k, **{b: keys[0][b] for b in FEET}} for k in keys]
    marks = path_marks(keys)
    return [pose_at(keys, marks, s) for s in timeline(fps)]


def leg_points(pose, rest, legs=DOWN):
    """The legs' joints ({UpLeg, Leg, Foot, ToeBase} each side, Clip.frames' names) for one frame's turns, the
    two ankles' midpoint kept where it is at rest: the hips' path BoneRetarget follows (with planted feet it then
    locks the ankles anyway)."""
    mid = (rest["LeftUpLeg"] + rest["RightUpLeg"]) / 2
    out = {}
    for s in ("Left", "Right"):
        hip = mid + pose["Hips"] @ (rest[f"{s}UpLeg"] - mid)
        knee = hip + pose[f"{s}UpLeg"] @ legs * float(np.linalg.norm(rest[f"{s}Leg"] - rest[f"{s}UpLeg"]))
        ankle = knee + pose[f"{s}Leg"] @ legs * float(np.linalg.norm(rest[f"{s}Foot"] - rest[f"{s}Leg"]))
        toe = ankle + pose[f"{s}Foot"] @ (rest[f"{s}ToeBase"] - rest[f"{s}Foot"])
        out.update({f"{s}UpLeg": hip, f"{s}Leg": knee, f"{s}Foot": ankle, f"{s}ToeBase": toe})
    shift = (rest["LeftFoot"] + rest["RightFoot"]) / 2 - (out["LeftFoot"] + out["RightFoot"]) / 2
    return {n: p + shift for n, p in out.items()}


# --- the clip (Blender) ------------------------------------------------------------------------------

def clip_from_keyposes(rest, keyposes, fps=FPS, feet="planted", label="keyposes.json", legs=DOWN):
    """A `retarget.Clip` moving between the key positions, on `rest`'s skeleton (a Clip read with
    `retarget.load_clip(REST_CLIP, turned="all")`): `turns` for the bones a key position sets (the rest keep
    their parent's turn in BoneRetarget), `frames` the legs' joints; `legs` the legs' rest direction (`leg_rest`).
    ValueError (plain words) without a start and an end position."""
    import retarget
    from mathutils import Matrix, Vector
    rows = np.array(retarget.facing(rest.rest)[0])          # the source's right, forward, up, in its world
    to_world = np.column_stack([-rows[0], -rows[1], rows[2]])   # character (left, back, up) -> source world
    rest_points = {n: to_world.T @ np.array(p) for n, p in rest.rest.items()}
    notes = []
    keys = key_poses(keyposes, rest_points, notes, feet=feet, legs=legs)
    bones = sorted(keys[0][1])
    lacking = [b for b in bones if b not in rest.rest_turns]
    if lacking:
        raise ValueError(f"the rest clip has no {lacking} bones (load it with turned='all')")
    key_turns = [turns for _, turns in keys]
    if keyposes.get("exercise") == "Barbell Bent-Over Row":
        from row_motion import stabilize_body
        key_turns = stabilize_body(key_turns)
        notes.append("row: starting body hinge held throughout; arms retain reference motion")
    poses = motion(key_turns, fps=fps, feet=feet)

    def in_world(r):
        return Matrix((to_world @ r @ to_world.T).tolist()).to_quaternion()
    turns = [{b: in_world(pose[b]) @ rest.rest_turns[b] for b in bones} for pose in poses]
    frames = [{n: Vector((to_world @ p).tolist()) for n, p in leg_points(pose, rest_points, legs).items()}
              for pose in poses]
    notes.append(f"key positions {', '.join(p for p, _ in keys)}; {len(poses)} frames at {fps} fps "
                 f"({DESCENT_S:g} s down, {HOLD_S:g} s held, {ASCENT_S:g} s up), feet {feet}")
    return retarget.Clip(path=str(label), fps=float(fps), first=1, rest=dict(rest.rest), frames=frames,
                         bones=dict(rest.bones), notes=notes, rest_turns={b: rest.rest_turns[b] for b in bones},
                         turns=turns)
