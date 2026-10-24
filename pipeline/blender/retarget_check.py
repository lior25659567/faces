#!/usr/bin/env python
"""Prove the retarget before any real Mixamo file exists. Two halves, run in this order:

    # 1. inside Blender: build synthetic Mixamo clips and render the SOURCE as a stick figure
    blender --background --python pipeline/blender/retarget_check.py -- make --out DIR

    # 2. retarget each clip with the real entry point (see RUNS below for the exact commands),
    #    each into DIR/runs/<name>/ — `animate.py --mixamo ... --out DIR/runs/<name>`

    # 3. plain Python: check every expectation and build the side-by-side GIFs
    python3 pipeline/blender/retarget_check.py verify --out DIR --gifs 3d-character/output/retarget

## Why synthetic clips, and why these

`new 3d-character/new man model.glb` carries a complete Mixamo skeleton (`mixamorig:` names, 55
bones). Its skin is broken and is thrown away; the skeleton is posed here into poses that are
ASYMMETRIC on purpose, because a mirrored or swapped retarget is indistinguishable from a correct
one in any symmetric pose — an air squat, a T-pose, a curl with both arms. Each pose moves exactly
one thing in one direction:

    left arm forward      only the LEFT upper arm, toward the character's front
    right knee bent       only the RIGHT shin, swung back (the foot lifts behind)
    left arm out          only the LEFT arm, out to the character's left side
    twist left            the chest turned 30 degrees toward the character's left
    lean right            the spine bent 20 degrees toward the character's right
    turn left             the whole body turned 45 degrees to its left (the root's roll)

The expectations in `verify` are written in OUR world directly — forward is +Y, the character's
left is -X, up is +Z — and NOT derived from `retarget.facing`. The source's own frame is asserted
from its geometry when it is loaded (toes point +X, the `Left*` bones sit at +Y). So the test
states independently which way is forward and which side is left on both skeletons, and checks
that the code in between agrees.

The same clip is also written three more ways, to exercise what a real download varies:
    asym-turned     the armature turned to face -Y (as a real Mixamo import does), its bones
                    renamed `mixamorig1:`, and scaled 100x — must give the same figure
    asym-mirrored   every Left and Right name swapped — must be REFUSED
    squat           a two-rep squat with the source's feet exactly planted and no name prefix —
                    the foot-lock test: our legs are proportioned differently (thigh 1.76x the
                    shin, against 1.17x on the source), so copied directions alone make the feet
                    skate, and the lock must stop it
"""

import json
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
GLB = ROOT / "new 3d-character" / "new man model.glb"

try:
    import bpy
    from mathutils import Matrix, Quaternion, Vector
    sys.path.insert(0, str(HERE.parent))
    sys.path.insert(0, str(HERE))
    import animate                                        # noqa: E402  (for `aim`)
except ImportError:                                       # `verify` runs in plain Python
    bpy = None

FPS = 30
HOLD, MOVE = 6, 8


# --- the source skeleton -------------------------------------------------------------------------

def _source_axes():
    # The test source's anatomical frame, read off its rest pose by inspection and ASSERTED in
    # `load_source`: its toes point +X, its Left* bones sit at +Y, its head is up +Z. Written out
    # rather than computed so that `retarget.facing` is checked against something it did not make.
    return Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((0, 0, 1))


def load_source():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(GLB))
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    for obj in list(bpy.data.objects):
        if obj is not arm:
            bpy.data.objects.remove(obj, do_unlink=True)
    fwd, left, up = _source_axes()
    bones = arm.data.bones
    head = lambda n: bones[f"mixamorig:{n}"].head_local              # noqa: E731
    assert (head("LeftToeBase") - head("LeftFoot")).dot(fwd) > 0.03, "source toes are not +X"
    assert (head("RightToeBase") - head("RightFoot")).dot(fwd) > 0.03, "source toes are not +X"
    assert (head("LeftUpLeg") - head("RightUpLeg")).dot(left) > 0.05, "source Left* is not +Y"
    assert (head("LeftArm") - head("RightArm")).dot(left) > 0.1, "source Left* is not +Y"
    assert (head("Head") - head("Hips")).dot(up) > 0.2, "source head is not +Z"
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
    scene = bpy.context.scene
    scene.render.fps, scene.render.fps_base = FPS, 1.0
    return arm


def _pb(arm, name):
    return arm.pose.bones[f"mixamorig:{name}"]


def _update():
    bpy.context.view_layer.update()


def reset(arm):
    for pb in arm.pose.bones:
        pb.rotation_quaternion = Quaternion()
        pb.location = (0.0, 0.0, 0.0)
    _update()


def aim(arm, name, direction):
    animate.aim(_pb(arm, name), Vector(direction))
    _update()


def turn(arm, name, axis, degrees):
    """Rotate a bone about a WORLD axis through its own head, on top of how it stands now."""
    pb = _pb(arm, name)
    m = pb.matrix.copy()
    t = (Quaternion(Vector(axis), math.radians(degrees)).to_matrix() @ m.to_3x3()).to_4x4()
    t.translation = m.translation
    pb.matrix = t
    _update()


def head_of(arm, name):
    return arm.matrix_world @ _pb(arm, name).head


def arms_hanging(arm):
    fwd, left, up = _source_axes()
    for side, out in (("Left", left), ("Right", -left)):
        hang = (-up * 0.94 + out * 0.34).normalized()
        for bone in ("Arm", "ForeArm", "Hand"):
            aim(arm, side + bone, hang)


def pose_named(arm, name):
    fwd, left, up = _source_axes()
    reset(arm)
    if name == "twist left":
        turn(arm, "Spine", up, 30.0)          # +30 about up: the chest turns to the figure's left
    if name == "lean right":
        turn(arm, "Spine", fwd, 20.0)         # +20 about forward carries up toward -left = right
    if name == "turn left":
        turn(arm, "Hips", up, 45.0)
    arms_hanging(arm)
    if name == "left arm forward":
        for bone in ("LeftArm", "LeftForeArm", "LeftHand"):
            aim(arm, bone, fwd)
    if name == "left arm out":
        for bone in ("LeftArm", "LeftForeArm", "LeftHand"):
            aim(arm, bone, left)
    if name == "right knee bent":
        aim(arm, "RightLeg", -fwd)            # shin straight back, thigh left hanging


ASYM = ["neutral", "left arm forward", "neutral", "right knee bent", "neutral", "left arm out",
        "neutral", "twist left", "neutral", "lean right", "neutral", "turn left", "neutral"]


def key_all(arm, frame):
    for pb in arm.pose.bones:
        pb.keyframe_insert("rotation_quaternion", frame=frame)
        pb.keyframe_insert("location", frame=frame)


def make_asym(arm):
    """The asymmetric clip. Returns [(pose, the middle frame of its hold)]."""
    frame, held = 1, []
    for name in ASYM:
        pose_named(arm, name)
        key_all(arm, frame)
        key_all(arm, frame + HOLD - 1)
        held.append((name, frame + HOLD // 2))
        frame += HOLD - 1 + MOVE
    bpy.context.scene.frame_start, bpy.context.scene.frame_end = 1, frame - MOVE
    return held


def _two_bone(hip, target, l1, l2, pole):
    to = target - hip
    d = min(to.length, l1 + l2 - 1e-6)
    u = to.normalized()
    v = (pole - u * pole.dot(u)).normalized()
    a = math.acos(max(-1.0, min(1.0, (l1 * l1 + d * d - l2 * l2) / (2 * l1 * d))))
    return hip + (u * math.cos(a) + v * math.sin(a)) * l1


def make_squat(arm):
    """Two reps of a squat with the source's own feet exactly planted (solved per frame)."""
    fwd, left, up = _source_axes()
    reset(arm)
    ankles = {s: head_of(arm, f"{s}Foot").copy() for s in ("Left", "Right")}
    foot_dir = {s: (_pb(arm, f"{s}Foot").tail - _pb(arm, f"{s}Foot").head).normalized()
                for s in ("Left", "Right")}
    lengths = {s: ((head_of(arm, f"{s}Leg") - head_of(arm, f"{s}UpLeg")).length,
                   (head_of(arm, f"{s}Foot") - head_of(arm, f"{s}Leg")).length)
               for s in ("Left", "Right")}
    rep, still = 36, 6
    depths = [0.0] * still
    for _ in range(2):
        depths += [(1 - math.cos(2 * math.pi * k / rep)) / 2 for k in range(rep)]
    depths += [0.0] * still
    for i, depth in enumerate(depths):
        reset(arm)
        hips = _pb(arm, "Hips")
        m = hips.matrix.copy()
        m.translation = m.translation + (-fwd * 0.09 - up * 0.14) * depth
        hips.matrix = m
        _update()
        turn(arm, "Hips", left, 25.0 * depth)     # +about left tips the top of the pelvis forward
        turn(arm, "Spine", left, 15.0 * depth)
        lift = min(1.0, depth * 2.0)
        for side, out in (("Left", left), ("Right", -left)):
            hang = (-up * 0.94 + out * 0.34).normalized()
            reach = (hang * (1 - lift) + fwd * lift).normalized()
            for bone in ("Arm", "ForeArm", "Hand"):
                aim(arm, side + bone, reach)
            hip = head_of(arm, f"{side}UpLeg")
            l1, l2 = lengths[side]
            knee = _two_bone(hip, ankles[side], l1, l2, fwd + out * 0.35)
            aim(arm, f"{side}UpLeg", knee - hip)
            aim(arm, f"{side}Leg", ankles[side] - knee)
            aim(arm, f"{side}Foot", foot_dir[side])
        key_all(arm, i + 1)
    bpy.context.scene.frame_start, bpy.context.scene.frame_end = 1, len(depths)
    # The source must itself be planted, or the lock test proves nothing.
    travel = 0.0
    for f in range(1, len(depths) + 1):
        bpy.context.scene.frame_set(f)
        for s in ("Left", "Right"):
            travel = max(travel, (head_of(arm, f"{s}Foot") - ankles[s]).length,
                         (head_of(arm, f"{s}ToeBase") - (ankles[s] + (
                             _pb(arm, f"{s}ToeBase").bone.head_local -
                             _pb(arm, f"{s}Foot").bone.head_local))).length)
    print(f"squat source: {len(depths)} frames, source ankle/toe travel max {travel * 1000:.3f} mm")
    return travel


# --- writing the variants ------------------------------------------------------------------------

def export_fbx(arm, path):
    bpy.ops.object.select_all(action="DESELECT")
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.export_scene.fbx(
        filepath=str(path), use_selection=True, object_types={"ARMATURE"},
        add_leaf_bones=False, bake_anim=True, bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=False, bake_anim_use_all_actions=False,
        bake_anim_force_startend_keying=True, bake_anim_step=1.0,
        bake_anim_simplify_factor=0.0)
    print(f"wrote {path}")


def rename_bones(arm, rename):
    """Rename every bone; Blender rewrites the action's F-curve paths to match."""
    for bone in list(arm.data.bones):
        bone.name = f"__tmp__{bone.name}"
    for bone in list(arm.data.bones):
        bone.name = rename(bone.name[len("__tmp__"):])


# --- rendering the source ------------------------------------------------------------------------

SOURCE_STICKS = [
    ("Hips", "Spine"), ("Spine", "Spine1"), ("Spine1", "Spine2"), ("Spine2", "Neck"),
    ("Neck", "Head"),
    ("LeftShoulder", "LeftArm"), ("LeftArm", "LeftForeArm"), ("LeftForeArm", "LeftHand"),
    ("LeftHand", "LeftHandMiddle1"), ("Hips", "LeftUpLeg"), ("LeftUpLeg", "LeftLeg"),
    ("LeftLeg", "LeftFoot"), ("LeftFoot", "LeftToeBase"), ("LeftToeBase", "LeftToe_End"),
    ("Spine2", "LeftShoulder"),
]
SOURCE_STICKS += [(a.replace("Left", "Right"), b.replace("Left", "Right"))
                  for a, b in SOURCE_STICKS if "Left" in a + b]

COLOURS = {"Left": (0.05, 0.25, 0.95, 1.0), "Right": (0.95, 0.12, 0.08, 1.0),
           "": (0.55, 0.55, 0.58, 1.0)}


def _material(name, colour):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = colour
    return mat


def stick_figure(arm):
    """Cylinders bone-parented to the source armature, coloured by the source's own side names."""
    import exercise_stage
    _update()
    mats = {k: _material(f"stick{k}", c) for k, c in COLOURS.items()}
    for a, b in SOURCE_STICKS:
        start, end = head_of(arm, a), head_of(arm, b)
        span = end - start
        if span.length < 1e-6:
            continue
        side = "Left" if "Left" in a + b else ("Right" if "Right" in a + b else "")
        radius = 0.011 if side else 0.014
        bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=span.length, vertices=10)
        stick = bpy.context.object
        stick.data.materials.append(mats[side])
        basis = span.normalized().to_track_quat("Z", "Y").to_matrix()
        exercise_stage.hold_at(stick, arm, f"mixamorig:{a}", (start + end) / 2, basis)
        for point in (start, end):
            bpy.ops.mesh.primitive_uv_sphere_add(radius=radius * 1.25, segments=12, ring_count=6)
            ball = bpy.context.object
            ball.data.materials.append(mats[side])
            exercise_stage.hold_at(ball, arm, f"mixamorig:{a}", point, Matrix.Identity(3))
    fwd, left, up = _source_axes()
    head = head_of(arm, "Head")
    for name, at, radius, colour in (("skull", head + up * 0.05, 0.06, COLOURS[""]),
                                     ("nose", head + up * 0.045 + fwd * 0.06, 0.018,
                                      (0.95, 0.75, 0.1, 1.0))):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, segments=16, ring_count=8)
        ball = bpy.context.object
        ball.data.materials.append(_material(name, colour))
        exercise_stage.hold_at(ball, arm, "mixamorig:Head", at, Matrix.Identity(3))


def render_source(arm, out, frames, azimuth=30.0, elevation=8.0):
    """The source from ITS own front (toes), at the same azimuth `retarget.frame_clip` uses."""
    import exercise_stage
    fwd, left, up = _source_axes()
    stick_figure(arm)
    bpy.ops.mesh.primitive_plane_add(size=6, location=(0, 0, min(
        head_of(arm, f"{s}ToeBase").z for s in ("Left", "Right")) - 0.012))
    bpy.context.object.data.materials.append(_material("floor", (0.72, 0.78, 0.77, 1.0)))
    a, e = math.radians(azimuth), math.radians(elevation)
    orbit = fwd * math.cos(a) * math.cos(e) + left * math.sin(a) * math.cos(e) + up * math.sin(e)
    centre = (head_of(arm, "Head") + head_of(arm, "LeftFoot")) / 2
    lens, sensor = exercise_stage.LENS, exercise_stage.SENSOR
    fov = 2 * math.atan((sensor / 2) / lens)
    distance = (1.0 * 1.06 / 2) / math.tan(fov / 2) * 1.25
    bpy.ops.object.camera_add(location=centre + orbit * distance)
    camera = bpy.context.object
    camera.rotation_mode = "QUATERNION"
    camera.rotation_quaternion = (-orbit).to_track_quat("-Z", "Y")
    camera.data.lens, camera.data.sensor_fit, camera.data.sensor_height = lens, "VERTICAL", sensor
    scene = bpy.context.scene
    scene.camera = camera
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "MATERIAL"
    scene.render.resolution_x, scene.render.resolution_y = 360, 480
    scene.render.image_settings.file_format = "PNG"
    scene.world = scene.world or bpy.data.worlds.new("world")
    scene.world.color = (0.74, 0.83, 0.82)
    out.mkdir(parents=True, exist_ok=True)
    for f in range(1, frames + 1):
        scene.frame_set(f)
        scene.render.filepath = str(out / f"frame_{f:04d}.png")
        bpy.ops.render.render(write_still=True)
    print(f"source frames -> {out}")


def make(out):
    out = Path(out)
    sources = out / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    expect = {}

    arm = load_source()
    held = make_asym(arm)
    expect["asym"] = {"held": held, "frames": bpy.context.scene.frame_end}
    export_fbx(arm, sources / "asym.fbx")
    render_source(arm, out / "source-frames" / "asym", bpy.context.scene.frame_end)

    arm = load_source()
    make_asym(arm)
    rename_bones(arm, lambda n: n.replace("mixamorig:", "mixamorig1:"))
    arm.rotation_mode = "XYZ"
    arm.rotation_euler = (0.0, 0.0, math.radians(-90.0))   # faces -Y, as a Mixamo import does
    arm.scale = (100.0, 100.0, 100.0)
    _update()
    export_fbx(arm, sources / "asym-turned.fbx")

    arm = load_source()
    make_asym(arm)
    rename_bones(arm, lambda n: n.replace("Left", "@").replace("Right", "Left").replace("@", "Right"))
    export_fbx(arm, sources / "asym-mirrored.fbx")

    arm = load_source()
    travel = make_squat(arm)
    expect["squat"] = {"frames": bpy.context.scene.frame_end, "source_travel_mm": travel * 1000}
    render_source(arm, out / "source-frames" / "squat", bpy.context.scene.frame_end)
    arm = load_source()
    make_squat(arm)
    rename_bones(arm, lambda n: n.replace("mixamorig:", ""))
    export_fbx(arm, sources / "squat.fbx")

    (sources / "expect.json").write_text(json.dumps(expect, indent=1))
    print(f"expectations -> {sources / 'expect.json'}")


# --- verifying (plain Python) ----------------------------------------------------------------------

def _v(p):
    return tuple(float(c) for c in p)


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _norm(a):
    n = math.sqrt(sum(c * c for c in a))
    return tuple(c / n for c in a) if n else a


def _angle(a, b):
    a, b = _norm(a), _norm(b)
    return math.degrees(math.acos(max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b))))))


def _yaw(a):
    return math.degrees(math.atan2(a[1], a[0]))


# OUR world, written out: the figure faces +Y, its own left is -X, up is +Z.
FORWARD, LEFT, UP = (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)
DOWN = (0.0, 0.0, -1.0)


def _checks(name, j, neutral, lock):
    """[(description, passed, measured)] for one held pose, against OUR axes as written above."""
    seg = lambda a, b: _sub(_v(j[b]), _v(j[a]))                       # noqa: E731
    nseg = lambda a, b: _sub(_v(neutral[b]), _v(neutral[a]))         # noqa: E731
    out = []

    def check(text, ok, measured):
        out.append((text, bool(ok), measured))

    def hanging(side):
        a = _angle(seg(f"shoulder.{side}", f"elbow.{side}"), DOWN)
        check(f"{side} upper arm hangs (within 30 deg of down)", a < 30, f"{a:.1f} deg")

    if name == "neutral":
        hanging("left"), hanging("right")
        for side in ("left", "right"):
            a = _angle(seg(f"hip.{side}", f"knee.{side}"), DOWN)
            check(f"{side} thigh vertical (within 12 deg)", a < 12, f"{a:.1f} deg")
        toes = _sub(_v(j["foot.left"]), _v(j["ankle.left"]))
        a = _angle((toes[0], toes[1], 0.0), FORWARD)
        check("toes point forward (+Y) within 25 deg", a < 25, f"{a:.1f} deg")
        check("left hand is on -X (the character's left)", j["hand.left"][0] < j["pelvis"][0] - 0.05,
              f"x {j['hand.left'][0]:+.3f}")
    elif name == "left arm forward":
        a = _angle(seg("shoulder.left", "elbow.left"), FORWARD)
        check("LEFT upper arm points forward (+Y) within 12 deg", a < 12, f"{a:.1f} deg")
        hanging("right")
        check("left elbow is on -X side", j["elbow.left"][0] < j["pelvis"][0], f"x {j['elbow.left'][0]:+.3f}")
    elif name == "right knee bent":
        a = _angle(seg("knee.right", "ankle.right"), (0.0, -1.0, 0.0))
        check("RIGHT shin points backward (-Y) within 12 deg", a < 12, f"{a:.1f} deg")
        a = _angle(seg("hip.right", "knee.right"), DOWN)
        check("right thigh still hangs (within 12 deg)", a < 12, f"{a:.1f} deg")
        a = _angle(seg("knee.left", "ankle.left"), DOWN)
        check("LEFT shin straight down (within 12 deg)", a < 12, f"{a:.1f} deg")
        check("right foot lock released, left held", lock["right"] == 0 and lock["left"] == 1,
              f"lock {lock}")
        hanging("left"), hanging("right")
    elif name == "left arm out":
        a = _angle(seg("shoulder.left", "elbow.left"), LEFT)
        check("LEFT upper arm points to the character's left (-X) within 12 deg", a < 12, f"{a:.1f} deg")
        hanging("right")
    elif name == "twist left":
        now = _yaw(seg("shoulder.left", "shoulder.right"))
        was = _yaw(nseg("shoulder.left", "shoulder.right"))
        turn = (now - was + 180) % 360 - 180
        check("shoulder line turned +30 about +Z (toward the left) within 8", abs(turn - 30) < 8,
              f"{turn:+.1f} deg")
    elif name == "lean right":
        s = seg("pelvis", "neck")
        lean = math.degrees(math.atan2(s[0], s[2]))
        was = nseg("pelvis", "neck")
        lean0 = math.degrees(math.atan2(was[0], was[2]))
        check("spine leans toward +X (the character's right) by ~20 (within 8)",
              abs(lean - lean0 - 20) < 8, f"{lean - lean0:+.1f} deg")
    elif name == "turn left":
        now = _yaw(seg("hip.left", "hip.right"))
        was = _yaw(nseg("hip.left", "hip.right"))
        turn = (now - was + 180) % 360 - 180
        check("hip line turned +45 about +Z (toward the left) within 5", abs(turn - 45) < 5,
              f"{turn:+.1f} deg")
        toes = _sub(_v(j["foot.left"]), _v(j["ankle.left"]))
        yaw = (_yaw(toes) - _yaw(FORWARD) + 180) % 360 - 180
        check("left toes turned +45 (within 12)", abs(yaw - 45) < 12, f"{yaw:+.1f} deg")
    return out


def verify(out, gifs=None):
    out = Path(out)
    expect = json.loads((out / "sources" / "expect.json").read_text())
    runs = out / "runs"
    failures, lines = 0, []

    def say(text=""):
        lines.append(text)
        print(text)

    def load(name):
        path = runs / name / "retarget.json"
        return json.loads(path.read_text()) if path.exists() else None

    # 1. The asymmetric poses, on every variant that ran.
    reference = None
    for name in ("asym", "asym-turned"):
        result = load(name)
        if result is None:
            say(f"[{name}] not run")
            continue
        frames = {f["frame"]: f for f in result["per_frame"]}
        say(f"\n[{name}] {result['clip']}: facing {result['facing']}, scale {result['scale']}, "
            f"feet {result['feet']}")
        neutral = frames[expect["asym"]["held"][0][1]]["joints"]
        for pose, frame in expect["asym"]["held"]:
            for text, ok, measured in _checks(pose, frames[frame]["joints"], neutral,
                                              frames[frame]["lock"]):
                failures += not ok
                say(f"  {'PASS' if ok else 'FAIL'}  frame {frame:3d} {pose:17s} {text}: {measured}")
        if name == "asym":
            reference = frames
        elif reference is not None:
            worst, where = 0.0, None
            for f, data in frames.items():
                for k, p in data["joints"].items():
                    if k == "head":
                        continue      # the head's tip is the importer's guess at a tail: see below
                    d = math.dist(p, reference[f]["joints"][k])
                    if d > worst:
                        worst, where = d, (f, k)
            ok = worst < 0.001
            failures += not ok
            say(f"  {'PASS' if ok else 'FAIL'}  every joint of every frame matches the plain FBX "
                f"within 1 mm: worst {worst * 1000:.4f} mm at {where}")

    # 2. The mirrored skeleton must have been refused.
    log = runs / "asym-mirrored" / "log.txt"
    if log.exists():
        refused = "MIRRORED SKELETON" in log.read_text()
        failures += not refused
        say(f"\n[asym-mirrored]\n  {'PASS' if refused else 'FAIL'}  the mirrored skeleton was "
            f"refused")

    # 3. The squat: feet before and after the lock.
    for name in ("squat-nolock", "squat"):
        result = load(name)
        if result is None:
            continue
        c = result["contact"]
        say(f"\n[{name}] feet {result['feet']}, lock {'on' if result['foot_lock'] else 'OFF'}: "
            f"penetration {c['penetration_mm']} mm (frame {c['penetration_frame']})")
        for side, m in c["sides"].items():
            say(f"  {side:5s} sole slide {m['sole_slide_mm']:7.2f} mm (frame {m['slide_frame']}), "
                f"sole lift {m['sole_lift_mm']:6.2f} mm, ankle {m['ankle_slide_mm']:7.2f} mm, "
                f"toe {m['toe_slide_mm']:7.2f} mm, over {m['planted_frames']} planted frames")
        planted = [m["planted_frames"] for m in c["sides"].values()]
        ok = all(n == expect["squat"]["frames"] for n in planted)
        failures += not ok
        say(f"  {'PASS' if ok else 'FAIL'}  feet '{result['feet']}' found both feet planted in "
            f"every frame of a squat whose source feet never move: {planted} of "
            f"{expect['squat']['frames']}")
        if result["foot_lock"]:
            slide = max(m["sole_slide_mm"] for m in c["sides"].values())
            ok = slide < 5.0 and c["penetration_mm"] < 5.0
            failures += not ok
            say(f"  {'PASS' if ok else 'FAIL'}  slide {slide:.2f} mm and penetration "
                f"{c['penetration_mm']} mm both under 5 mm")
        say(f"  played-back joints vs posed in the loop: "
            f"{result.get('replay_vs_posed_max', float('nan')) * 1000:.4f} mm")
        if "form_check" in result:
            for phase, found in result["form_check"].items():
                say(f"  form {phase}: closest frame {found['frame']}, flagged {found['flagged']}")

    say(f"\n{'ALL PASS' if failures == 0 else f'{failures} FAILURE(S)'}")
    (out / "verify.txt").write_text("\n".join(lines) + "\n")

    if gifs:
        side_by_side(out, Path(gifs))
    return failures


def side_by_side(out, gifs):
    """Source stick figure | retargeted character, one GIF per clip, at the clip's own rate."""
    gifs.mkdir(parents=True, exist_ok=True)
    pairs = [("asym", "asym", "test-asymmetric-left-right-facing.gif"),
             ("squat", "squat-nolock", "test-synthetic-squat-foot-lock-OFF.gif"),
             ("squat", "squat", "test-synthetic-squat-foot-lock-on.gif")]
    for source, run, name in pairs:
        left, right = out / "source-frames" / source, out / "runs" / run
        if not (left.exists() and right.exists()):
            continue
        target = gifs / name
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
            "-i", str(left / "frame_%04d.png"), "-framerate", str(FPS),
            "-i", str(right / "frame_%04d.png"), "-filter_complex",
            "[0][1]hstack=inputs=2,split[a][b];[a]palettegen=stats_mode=diff[p];"
            "[b][p]paletteuse=dither=bayer", "-loop", "0", str(target)], check=True)
        print(f"gif -> {target}")


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("make", "verify"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--gifs", default=None)
    args = parser.parse_args(argv)
    if args.mode == "make":
        if bpy is None:
            raise SystemExit("`make` runs inside Blender")
        make(args.out)
    else:
        sys.exit(1 if verify(args.out, args.gifs) else 0)


if __name__ == "__main__":
    main()
