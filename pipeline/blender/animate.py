#!/usr/bin/env python
"""Pose the rigged mascot from the form notes and render the rep. Runs inside Blender.

    blender --background --python pipeline/blender/animate.py -- \
        --exercise "Dumbbell Lateral Raise" --out /tmp/raise

Builds the character (make_character.py), poses the rig from reference/exercises/poses.json at
each documented position, and writes a PNG per frame.

The rig's bones are named for `pose_model`'s joints, so there is no mapping table and nothing to
keep in step: a bone is posed by pointing it at the same place the solved skeleton points it. The
stick-figure preview and the render are therefore reading the same numbers, and a pose that looks
right in `pose_preview.py` cannot arrive here bent differently.

Interpolation between the two documented positions is a Bezier ease, not invented geometry: the
source gives where a rep starts and where it ends, and a lift reverses along one path. Where the
notes name more keyframes than two — the lat pulldown, the lunge — this says so rather than
pretending two is the whole movement.

Three ways to get a character, one flag each:

    (none)              the built-in procedural mascot (`make_character`)
    --model PATH        rig a supplied mesh by measuring it and weighting it (`fit_model`)
    --wrap-skin PATH    wear a supplied mesh over the clean cage (`character` + `skin_wrap`):
                        the cage is rigged and posed, the mesh follows it by Surface Deform

The third is `wear_skin` below. Everything after it — solving, posing, grounding, staging,
rendering — is the same code as the other two, driven by bone name.

Or, with --mpfb (together with --wrap-skin and --mixamo), the skin is worn over an MPFB human
instead of `character.py`'s cage — Mixamo's own skeleton, MakeHuman's artist weights — and the clip
is put on it bone for bone (`run_mpfb`, `mpfb_cage.py`, `retarget.BoneRetarget`).

And two ways to get the MOTION:

    (none)              the two sourced positions from poses.json, Bezier-eased (the timeline in
                        `main`)
    --mixamo PATH       a captured clip, one key per captured frame, retargeted by direction
                        through the same `aim` / `aim_bones` / `ground_rig` below, every frame
                        grounded and the planted feet locked (`retarget.py`). `--fast` renders a
                        Workbench preview and `--gif` stitches it.

and, on the --mpfb path only, a third for an exercise with no clip:

    --keyposes JSON     the reference pictures' key positions (the Exercise Studio's keyposes.json),
                        made into a clip by `keypose_clip.py` and put on the rig exactly as a Mixamo
                        clip is (`keypose_clip_from`).
"""

import argparse
import math
import sys
import tempfile
from pathlib import Path

import bpy
from mathutils import Vector, Quaternion

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import pose_model  # noqa: E402
import make_character  # noqa: E402
import fit_model  # noqa: E402
import exercise_stage  # noqa: E402

FPS = 24
HALF_REP = 15          # frames from one end of the movement to the other
HOLD = 3               # beats held at each end, so the position can be read

MORE_KEYFRAMES = {
    "Lat Pulldown (seated, to the front)":
        "arms overhead, scapular depression initiating the pull, elbows driving down to the "
        "chest-level finish, then the controlled return",
    "Walking Lunge (dumbbells optional)":
        "tall stance, step-and-touchdown, 90/90 bottom, drive-up with the rear leg swinging through",
    "Front Plank (forearm plank)":
        "the lift-off and the controlled lower; the hold itself barely animates",
}


def aim(bone, direction):
    """Swing a bone onto a direction by the SHORTEST arc, keeping whatever roll it had.

    The first version built an absolute orientation with `to_track_quat`, which aligns the bone
    and then picks a roll out of the air. Raising an arm from hanging to horizontal came with an
    arbitrary twist of up to a right angle about the arm's own length.

    On the procedural character that was invisible — its limbs are capsules, and a capsule looks
    identical however it is spun. On a real mesh it tears: the shoulder vertices are shared with
    the torso, so a twisting arm smears them into a flat fan. That is exactly what the render
    showed, and it is why weighting was the wrong thing to blame.

    `rotation_difference` gives the minimal rotation between two directions — no roll invented —
    and composing it with the bone's current matrix preserves the roll the rest pose had.
    """
    target = Vector(direction)
    if target.length < 1e-9:
        return
    bone.rotation_mode = "QUATERNION"
    current = bone.matrix.to_3x3()
    now = (current @ Vector((0.0, 1.0, 0.0)))
    if now.length < 1e-9:
        return
    swing = now.normalized().rotation_difference(target.normalized())
    turned = (swing.to_matrix() @ current).to_4x4()
    turned.translation = bone.matrix.translation
    bone.matrix = turned


# How much of the arm's swing the collarbone takes with it.
#
# A shoulder is not one hinge. Raising an arm rotates the scapula and clavicle as well, and rigs
# copy that because it is also what stops the mesh tearing: ask a single joint to turn through a
# right angle and the skin spanning it has nowhere to go, which is the fan that appeared on the
# shoulders whenever the arms came down. Split across two joints, neither turns far enough to
# open up.
#
# A third is the usual figure and it is not tuned to this model — a larger share starts to look
# like a shrug, a smaller one leaves the tear.
CLAVICLE_FOLLOW = 0.34


def with_clavicles(skeleton):
    """The solved skeleton with each collarbone turned part of the way toward its arm.

    Done here rather than in `pose_model` on purpose. The solver's job is to say what the form
    notes describe, and the notes say nothing about collarbones; this is a property of the rig
    that makes the skin behave, so it belongs with the rigging.
    """
    for side in ("left", "right"):
        chest = Vector(skeleton.at("chest"))
        shoulder = Vector(skeleton.at(f"shoulder.{side}"))
        elbow = Vector(skeleton.at(f"elbow.{side}"))
        rest = (shoulder - chest)
        arm = (elbow - shoulder)
        if rest.length < 1e-9 or arm.length < 1e-9:
            continue
        swing = rest.normalized().rotation_difference(arm.normalized())
        partial = Quaternion().slerp(swing, CLAVICLE_FOLLOW)
        skeleton.joints[f"shoulder.{side}"].position = tuple(chest + (partial @ rest))
    return skeleton


def aim_bones(rig, skeleton):
    """Swing every bone onto its solved direction. No keyframes, no translation.

    Split out of `pose` because the contact sheet in `exercise_stage` needs the same aiming
    without an action left behind on the rig, and two copies of this loop had already drifted
    apart once.
    """
    bpy.context.view_layer.update()
    for name, (head, tail) in make_character.BONES.items():
        if name not in rig.pose.bones:
            continue
        if head not in skeleton.joints or tail not in skeleton.joints:
            continue
        a, b = skeleton.at(head), skeleton.at(tail)
        direction = Vector((b[0] - a[0], b[1] - a[1], b[2] - a[2]))
        if direction.length < 1e-9:
            continue
        aim(rig.pose.bones[name], direction)
        bpy.context.view_layer.update()


def ground_rig(rig, body, orientation, rest_at=0.0, pelvis_at=None):
    """Translate the rig so the figure stands where `pose_model.ground` put it. Returns the move.

    This is the last step of posing and it used to be missing, which made every earlier render
    of this pipeline wrong in the same way. `aim_bones` only swings bones: a bone keeps its rest
    length and the armature's root never moves, so the pelvis sat at one fixed height — 0.4264
    on `3d cartoon boy.glb` — in every pose of every exercise. `pose_model.solve` ends with a
    call to `ground`, which translates the solved figure so that whatever carries it is on the
    floor (the feet standing, the back lying down, the forearms and toes in a plank, the hands
    on the bar when it hangs from one), and that translation was then thrown away at the last
    step. A squat at the bottom of the rep floated its feet a quarter of a body above the floor,
    a supine bench press lay in mid-air at hip height, and a pull-up stood on the ground holding
    a waist-high bar. Staging worked around it by moving the FLOOR to meet the figure, which
    cannot be right for two phases at once and put a bench's feet below its own floor.

    `ground` is asked to do the work rather than being re-implemented here: a `Skeleton` is
    built from the POSED rig's own joints and passed to it, so the rule about which part of the
    body carries which orientation stays in one place — the solver's. What comes back is a pure
    translation, so the pelvis's share of it is the whole of it.

    Two corrections on top, both of them about the difference between a stick figure and a mesh:

      * `ground` works in joints and a render is judged on skin. It stands the figure on its
        lowest JOINT, and the sole of a foot is 0.06 below the ankle joint on this model, so a
        figure grounded by the joint sinks its feet into the floor by 6% of its own height. The
        clamp below is `ground`'s own closing clamp ("nothing is left below the floor") measured
        on the mesh instead: it lifts the figure until its lowest surface rests ON the plane it
        is resting on. It fires on all 26 of the 28 phases in this set that rest on anything —
        the other two are the pull-up's — so on this character it is not a safety net but the
        contact itself: the skin is below the plane in every one of them before it runs.
      * `rest_at` is the height of that plane, and it is not always zero. A supine bench press
        rests on a bench, not on the ground, and nothing in the form notes knows that a bench
        exists; `exercise_stage.support_height` reads the pad's own height off the prop and
        passes it here, so the body lies on the pad while the pad stands on the floor. Without
        it the only two possibilities are a body through the pad or a bench through the floor,
        which is precisely the pair the first staging pass was choosing between.

    `hanging` is left alone by the clamp: it is anchored by the hands and its feet are meant to
    be off the ground.

    **`pelvis_at` pins the height instead of deriving it, and some orientations need that.**
    `ground` answers "how far down until something touches the floor", and for a figure standing,
    bending or lying on the ground that is the whole truth. For a figure on a SEAT it is not a
    question the floor can answer, and asking it anyway produces an answer that moves: the leg
    press's rider is held up by its ankles at the start, and when the legs extend the ankles rise
    above the hands, so the anchor silently changes limb and the hips drop 173 mm through the
    seat. Measured, both phases: lowest joint `ankle.left` +0.281 then `hand.left` +0.333, which
    is the whole of the drop. The solved skeleton's pelvis does not move at all between those two
    keyframes — it is +0.260 in both — so every millimetre of that sink was introduced here.

    `pose_model.PINNED_BY` already names the orientations whose contact with the world is the
    pelvis rather than the floor (`seated`, `lying_supine`), so the caller reads that table and
    hands the height down once for the whole rep instead of letting each frame re-derive it from
    whichever limb happens to hang lowest. The floor clamp still runs underneath as a safety and
    is not expected to fire.
    """
    if rig.parent is not None:
        raise ValueError(f"{rig.name} is parented to {rig.parent.name}; this moves the rig in "
                         "world space and would need the parent's transform taken out first")
    joints = exercise_stage.joints_of(rig)
    if "pelvis" not in joints:
        raise ValueError("the rig has no pelvis joint to anchor")
    anchored = pose_model.Skeleton()
    for name, position in joints.items():
        anchored.add(name, tuple(position))
    pose_model.ground(anchored, orientation)
    move = Vector(anchored.at("pelvis")) - joints["pelvis"]
    if pelvis_at is not None:
        # Keep `ground`'s horizontal answer, which is sound in every orientation, and replace
        # only its vertical one. See `pelvis_at` above.
        move.z = pelvis_at - joints["pelvis"].z

    rig.location = Vector(rig.location) + move
    bpy.context.view_layer.update()
    if orientation != "hanging" and body is not None:
        under = min((body.evaluated_get(bpy.context.evaluated_depsgraph_get()).matrix_world
                     @ Vector(corner)).z
                    for corner in body.evaluated_get(
                        bpy.context.evaluated_depsgraph_get()).bound_box)
        if under < rest_at:
            lift = rest_at - under
            rig.location.z += lift
            move.z += lift
            bpy.context.view_layer.update()

    # The check this function exists to make: did the rig actually move? `location` is in the
    # object's parent space and `matrix_world` is lazy, so an assignment that silently did
    # nothing would look exactly like a successful one from here.
    landed = exercise_stage.joints_of(rig)["pelvis"]
    asked = joints["pelvis"] + move
    if (landed - asked).length > 1e-4:
        print(f"  WARNING ground_rig: asked for a pelvis at "
              f"{tuple(round(v, 4) for v in asked)}, landed at "
              f"{tuple(round(v, 4) for v in landed)}")
    return move


def pose(rig, skeleton, frame, body=None, orientation="standing", rest_at=0.0, pelvis_at=None):
    """Set every bone from the solved skeleton, put the figure on the ground, and key both."""
    aim_bones(rig, skeleton)
    ground_rig(rig, body, orientation, rest_at=rest_at, pelvis_at=pelvis_at)
    for name in make_character.BONES:
        if name in rig.pose.bones:
            rig.pose.bones[name].keyframe_insert("rotation_quaternion", frame=frame)
    # The root translation is keyed too, and it has to be: it differs between the two phases of
    # one exercise (a squat's pelvis drops, a pull-up's whole body rises), so a single untracked
    # offset would be right for one end of the rep and wrong for the other. Keyed here, the
    # Bezier ease that `main` puts on the rotations carries the body's rise and fall with them.
    rig.keyframe_insert("location", frame=frame)


def wear_skin(path, scratch):
    """Build the clean cage at a supplied mesh's proportions and wear the mesh over it.

    Returns `(skin, rig, cage)`, and the three are NOT interchangeable, which is the whole reason
    this is not just another way to make `body`:

      * `rig` is the house armature (`make_character.BONES` names, `PARENTS` hierarchy), built by
        `character.build_rig` at the fitted cage's own joints. Everything downstream poses it by
        bone name, so `aim_bones`, `ground_rig`, `exercise_stage` and `equipment.hold` drive it
        unchanged.
      * `cage` is the clean quad body `character.py` builds, Armature-bound to `rig`. It is the
        deformation DRIVER, not the surface anyone sees, so it is hidden from render here.
      * `skin` is the supplied mesh with its own materials and textures, bound to the cage by
        Surface Deform. It has no Armature modifier. This is what the caller renders, measures
        and exports, so it is what `main` hands on as `body` — every measurement
        `exercise_stage` makes of "the body" (its rest height off `data.vertices`, its posed box
        off the evaluated object) is a measurement of what the viewer actually sees.

    **The skin is object-parented to the rig, and it has to be.** Surface Deform follows what
    happens to the cage's MESH — the Armature modifier bending it — and ignores what happens to
    the cage's OBJECT transform after the bind. `ground_rig` stands the figure on the floor by
    translating the rig object, and the cage, parented to the rig, goes with it; the skin did
    not. Measured on the first integration run: the rig moved 141 mm back and 47 mm down to
    ground the Dumbbell Lateral Raise, the skin stayed exactly where it was imported, and every
    dumbbell — placed on the rig's hand bones — hung 143 mm from the visible hand, in the cage's
    invisible one. The bone rotations still came through, so the arms rose and fell on cue and
    nothing looked broken until the props were checked against the skin rather than the rig.
    Parented, the rig's translation carries skin and cage together and the bind's relative
    transform never changes; `_check_skin_follows` proves it on every run.

    Order is forced: `fit_cage_to` first (`character.build` clears the scene), then the skin is
    loaded beside the cage, the cage is pulled onto it region by region, and only then bound —
    `region_fit` changes the cage's rest shape, and Surface Deform binds to whatever shape the
    cage has at bind time.

    A failed bind is an exception, not a value. `skin_wrap.wrap` already raises on it; it is
    asserted again here because the failure mode is silent — an unbound Surface Deform does
    nothing at all, so the skin would render in its imported T-pose in every frame of every
    exercise, standing perfectly still while the dumbbells swung round it.
    """
    import skin_wrap            # lazy: the other two paths must not depend on character.py
    cage, rig, _ = skin_wrap.fit_cage_to(path, scratch=scratch)
    skin = skin_wrap.load_skin(path)
    skin_wrap.region_fit(cage, skin)
    modifier = skin_wrap.wrap(cage, rig, skin)
    if not modifier.is_bound:
        raise SystemExit(f"Surface Deform did not bind {skin.name!r} to {cage.name!r}; refusing "
                         f"to animate a skin that would stand still in every frame")
    bpy.context.view_layer.update()
    world = skin.matrix_world.copy()
    skin.parent = rig
    skin.matrix_parent_inverse = rig.matrix_world.inverted()
    bpy.context.view_layer.update()
    if max(abs(a - b) for row_a, row_b in zip(skin.matrix_world, world)
           for a, b in zip(row_a, row_b)) > 1e-6:
        raise SystemExit("parenting the skin to the rig moved it; the bind would no longer match")
    _check_skin_follows(skin, rig)
    cage.hide_render = True
    print(f"wear_skin: {Path(path).name} -> skin {len(skin.data.vertices)} verts on cage "
          f"{len(cage.data.vertices)} verts, bound={modifier.is_bound}")
    skin_wrap.bake_to_skin(cage, rig, skin)   # Surface Deform -> bone weights, so the .usdz is skeletal
    return skin, rig, cage


def _box_min(obj):
    corners = exercise_stage.world_bbox(obj)
    return Vector([min(c[i] for c in corners) for i in range(3)])


def _check_skin_follows(skin, rig, step=0.25):
    """Move the rig object and confirm the skin moved with it by exactly as much; then put it back.

    The test that would have caught the skin being left behind by `ground_rig` (see `wear_skin`):
    a rig translation the skin does not follow is invisible in every check that reads the rig,
    and only shows when something placed on the rig is compared against the skin.
    """
    home = rig.location.copy()
    before = _box_min(skin)
    rig.location = home + Vector((0.0, step, step))
    bpy.context.view_layer.update()
    after = _box_min(skin)
    rig.location = home
    bpy.context.view_layer.update()
    moved = after - before
    if (moved - Vector((0.0, step, step))).length > 1e-4:
        raise SystemExit(f"the skin does not follow the rig: the rig moved (0, {step}, {step}) "
                         f"and the skin's box moved {tuple(round(v, 4) for v in moved)}")


def fcurves_of(action):
    """Every F-curve in an action, across both Blender action APIs.

    Blender 4.4 introduced slotted actions and 5.x dropped `action.fcurves`, so the curves now
    live under layers -> strips -> channelbags. Reading both keeps this working on whatever
    version somebody has installed rather than on the one it was written against.
    """
    if hasattr(action, "fcurves"):
        yield from action.fcurves
        return
    for layer in getattr(action, "layers", []):
        for strip in getattr(layer, "strips", []):
            for bag in getattr(strip, "channelbags", []):
                yield from bag.fcurves


def export_usdz(path, body, rig):
    """Write the rigged, animated character as a .usdz the app can play.

    A real model rather than a sheet of pre-rendered pictures. The frames were 1.4 MB for ONE
    exercise and fixed at whatever angle and frame rate they were rendered at; a model carries
    the character once and every later movement is a short clip of joint rotations on top of it.
    It also plays at the device's own refresh rate and can be turned, which a flipbook cannot.

    USD because that is what Apple's scene loaders read natively. The lights and the backdrop are
    deliberately left out: the app lights its own scene, and a baked-in floor would sit in the
    middle of a screen that already has a background of its own.

    **What a `--wrap-skin` body exports as, measured (task 7).** Blender's USD writer turns a
    mesh with any modifier other than Subdivision into a mesh with per-frame `points` samples,
    and it skins to the skeleton only a mesh with an Armature modifier. A wrapped skin has a
    Surface Deform modifier and no Armature modifier, so it comes out as a vertex-animated mesh
    (50 MB for one 22-frame rep, against 7 MB skinned), the skeleton and its SkelAnimation come
    out beside it under no SkelRoot, and NOTHING binds the two. SceneKit loads that file as a
    still figure: the skin has no skinner and the point samples are never played. Surface
    Deform cannot be exported as skeletal animation; the export needs a different last step
    (the report names it). `read_back_usdz` says which kind of file was written, every time.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    for name in ("backdrop", "floor", "key", "fill", "rim",
                 exercise_stage.SHADOW_LIGHT):
        if name in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)

    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    body.select_set(True)
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig

    bpy.ops.wm.usd_export(
        filepath=str(path),
        selected_objects_only=True,
        export_animation=True,
        export_materials=True,
        export_armatures=True,
        export_meshes=True,
        # Textures travel inside the .usdz rather than as files beside it, which is the whole
        # point of the format: one file for the app to carry. The parameter names here are
        # Blender 5.2's — an earlier guess at `export_textures` and `export_skinweights` was
        # rejected outright, so they are taken from the operator's own property list.
        export_textures_mode="NEW",
        relative_paths=False,
        convert_orientation=True,
        export_global_forward_selection="NEGATIVE_Z",
        export_global_up_selection="Y",
    )
    size = path.stat().st_size / 1024 if path.exists() else 0
    print(f"exported {path} ({size:.0f} KB)")
    read_back_usdz(path)


def read_back_usdz(path):
    """Open the written file and say what it actually carries — not what the call was asked for.

    The export call returning is not evidence of an animation: a file can hold a perfectly good
    skeleton and SkelAnimation that drive nothing at all, which is exactly what a wrapped skin
    produces (see `export_usdz`). So this asks UsdSkel itself which meshes each skeleton skins,
    and lists every other mesh that moves only by baked point samples. Returns the findings, or
    None when this Blender has no `pxr` module to read with.
    """
    try:
        from pxr import Usd, UsdGeom, UsdSkel
    except ImportError:
        print("  read-back skipped: no pxr module in this Blender")
        return None
    stage = Usd.Stage.Open(str(path))
    predicate = Usd.TraverseInstanceProxies()
    animations = [str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdSkel.Animation)
                  and UsdSkel.Animation(p).GetRotationsAttr().GetNumTimeSamples() > 1]
    skinned = []
    for prim in stage.Traverse():
        if prim.IsA(UsdSkel.Root):
            cache = UsdSkel.Cache()
            root = UsdSkel.Root(prim)
            cache.Populate(root, predicate)
            for binding in cache.ComputeSkelBindings(root, predicate):
                skinned += [str(t.GetPrim().GetPath()) for t in binding.GetSkinningTargets()]
    baked = [(str(p.GetPath()), UsdGeom.Mesh(p).GetPointsAttr().GetNumTimeSamples())
             for p in stage.Traverse() if p.IsA(UsdGeom.Mesh) and str(p.GetPath()) not in skinned
             and UsdGeom.Mesh(p).GetPointsAttr().GetNumTimeSamples() > 1]
    found = {"animated_skeletons": animations, "skinned_meshes": skinned, "point_cached_meshes": baked}
    print(f"  read back: skeleton animations {len(animations)}, meshes skinned to a skeleton "
          f"{len(skinned)}, meshes moved only by baked point samples {len(baked)}")
    if baked:
        print(f"  WARNING usdz NOT SKELETAL: {[name for name, _ in baked]} move by "
              f"{max(n for _, n in baked)} baked point samples and are bound to no skeleton"
              f"{' (the skeleton animation in the file drives nothing)' if animations and not skinned else ''}. "
              f"SceneKit plays a file like this as a still figure.")
    return found


def run_mpfb(args):
    """`--mpfb`: the skin worn over an MPFB human instead of `character.py`'s cage, and the clip put
    on it bone for bone. See `mpfb_cage.py` (the cage, the bake) and `retarget.BoneRetarget` (the
    motion). Everything else here — pose_model's two phases, `exercise_stage`'s staging — is the
    house rig's and is not used: the clip is the motion, and the preview is EEVEE (`mpfb_cage`'s
    `preview_set`), a 3/4 view with the exercise's prop and, with --side-view, the figure's left
    side beside it. Writes OUT/retarget.json with the contact and deformation numbers.

    With --keyposes instead of --mixamo the clip is made from the reference pictures' key positions
    (`keypose_clip_from`); it enters here as the Mixamo clip does, so everything after it is the same."""
    import json
    import mpfb_cage
    retarget = retarget_module()
    if not args.wrap_skin or bool(args.mixamo) == bool(args.keyposes):
        raise SystemExit("--mpfb needs --wrap-skin (the skin) and one motion: --mixamo (a clip) or --keyposes "
                         "(the reference pictures' key positions)")
    clip = (retarget.load_clip(args.mixamo, turned="all") if args.mixamo
            else keypose_clip_from(args.keyposes, args.exercise, args.feet, quiet=True))   # refused here, before building
    workspace = Path(args.out) if args.out else Path(tempfile.gettempdir())
    workspace.mkdir(parents=True, exist_ok=True)
    skin, rig, cage, built = mpfb_cage.wear(args.wrap_skin, workspace)
    _check_skin_follows(skin, rig)
    if args.keyposes:
        # the legs counted from the rig's own thigh as the Studio reads it - its skin outline in the bind pose
        # (task 20) - so a picture's thigh puts ours where the picture's is
        leans = [v for v in mpfb_cage.rest_thigh_outlines(rig).values() if v is not None]
        lean = sum(leans) / len(leans) if leans else 0.0
        print(f"retarget: the key positions' legs counted from the thighs' skin outline at rest ({lean:+.1f} degrees "
              f"forward of straight down)", flush=True)
        clip = keypose_clip_from(args.keyposes, args.exercise, args.feet, lean=lean)
    driver = retarget.BoneRetarget(rig, skin, clip, exercise=args.exercise, feet=args.feet,
                                   foot_lock=not args.no_foot_lock)
    result = driver.run()
    scene = bpy.context.scene
    first, last = scene.frame_start, scene.frame_end
    result["contact"] = retarget.skin_contact(rig, skin, first, last)
    result["cage"] = built

    held = []
    if args.form_targets:
        # The poses corrected toward a measured reference, at the clip's own timing
        # (`mpfb_cage`, "matching a reference"); the bar's height and the grip are chosen there.
        # (with the coach's neck and face when the Studio wrote them into the targets: `read_form_targets`)
        targets = mpfb_cage.read_form_targets(args.form_targets)
        # the Studio's own targets beside the file (a Studio run): its positions, and our picture read as it
        # reads it, the lean and hip angle then aimed at that picture (task 19)
        studio = mpfb_cage.read_studio_targets(args.form_targets)
        plan, legs = mpfb_cage.match_legs_torso(rig, targets, first, last, skin=skin)
        bar, arms = mpfb_cage.match_arms(rig, skin, first, last, plan, targets, studio=studio)
        held.append(bar)
        picture, arms_after = None, None
        if studio is not None:
            picture = mpfb_cage.match_picture(rig, skin, first, last, targets, studio)
            # the picture moved the legs and torso: the arms read where the Studio will now pick its positions
            arms_after = mpfb_cage.studio_arm_rows(rig, first, last, studio, chosen_at=arms.get("positions"))
        result["form"] = {"targets": args.form_targets, "legs_torso": legs, "arms": arms, "picture": picture,
                          "arms_at_studio_positions": arms_after,
                          "contact": mpfb_cage.bar_contact(rig, skin, bar, first, last)}
        result["form_report"] = mpfb_cage.form_report(rig, first, last, targets, skin=skin)
        # the legs moved: the feet are measured again on the skin
        result["contact"] = retarget.skin_contact(rig, skin, first, last)
        print(f"mpfb: form matched: {result['form']}")
        for row in result["form_report"]["table"]:
            print(f"  {row['phase']:8s} f{row['frame']:<3d} {row['metric']:30s} target {str(row['target']):>24s} "
                  f"ours {row['ours']:>7} {'PASS' if row['pass'] else 'FAIL'}")
    elif args.exercise in MPFB_BARBELL_ON_BACK:
        scene.frame_set(first)
        bar, where = mpfb_cage.mount_barbell(rig, skin)
        held.append(bar)
        result["barbell"] = where
        result["grip"] = mpfb_cage.grip_bar(rig, bar, first, last)
        result["grip"]["contact"] = mpfb_cage.bar_contact(rig, skin, bar, first, last)
        print(f"mpfb: hands on the bar: {result['grip']}")
    # measured on the animation as it will play: after the grip re-keyed the arms and fingers
    result["deformation"] = mpfb_cage.clip_deformation(rig, skin, first, last)
    c, d = result["contact"], result["deformation"]
    print(f"mpfb: contact penetration {c['penetration_mm']} mm | "
          + " | ".join(f"{s} sole slide {m['sole_slide_mm']} mm lift {m['sole_lift_mm']} mm"
                       for s, m in c["sides"].items())
          + f" | crack max {d['crack_max_mm']:.4f} mm over {d['coincident_unconnected_pairs']} pairs"
          f" | deepest frame {d['deepest_frame']}: max edge growth "
          f"{d['deepest']['max_growth_mm']:.1f} mm, {d['deepest']['edges_over_1pct']} edges > 1%")
    scene.frame_set(first)
    if args.out or args.save:
        report = Path(args.out) if args.out else Path(args.save).parent
        report.mkdir(parents=True, exist_ok=True)
        retarget.write_report(report / "retarget.json", result)
    if args.out:
        import numpy as np
        out = Path(args.out)
        mpfb_cage.preview_set(args.width, args.height)
        low, high = np.array(c["box"][0]), np.array(c["box"][1])
        centre = (low + high) / 2
        distance = mpfb_cage.distance_for(max(high[2] - low[2], 0.5) * 1.05)
        mpfb_cage.camera_on(centre, 35, 10, distance)
        scene.render.filepath = str(out / "frame_")
        bpy.ops.render.render(animation=True)
        side = None
        if args.side_view:
            side = out / "side"
            side.mkdir(parents=True, exist_ok=True)
            # the bar stays in: it is unloaded (`mpfb_cage.mount_barbell`), so end-on it is the
            # small circle the reference shows, not a plate across the body
            mpfb_cage.camera_on(centre, 90, 5, distance)
            scene.render.filepath = str(side / "frame_")
            bpy.ops.render.render(animation=True)
        if args.gif:
            retarget.frames_to_gif(out, args.gif, scene.render.fps, beside=side)
    if args.save:
        bpy.ops.wm.save_as_mainfile(filepath=args.save)
    if args.usdz:
        bpy.context.view_layer.objects.active = rig
        export_usdz(Path(args.usdz), skin, rig)


def keypose_clip_from(path, exercise, feet=None, lean=None, quiet=False):
    """`--keyposes`: a clip moving between the reference pictures' key positions (keypose_clip.py), for an
    exercise with no Mixamo clip, on the Mixamo rest every clip shares (read from keypose_clip.REST_CLIP). It is
    a `retarget.Clip` like the one `load_clip` reads, so everything after it in `run_mpfb` is unchanged. The
    planted feet follow the exercise's foot mode (`retarget.EXERCISES`, planted unless listed). `lean`: the rig's
    thigh skin outline at rest, degrees forward of straight down (keypose_clip.leg_rest; None: a straight leg).
    `quiet`: no "read" line (the Studio counts frames from it; run_mpfb builds the clip again once the rig, and
    its thighs, exist)."""
    import json
    import keypose_clip
    retarget = retarget_module()
    # refused before anything is built, in the Studio's words (studio/measure/kit.py, one table): feet that step,
    # or a bench / seat / pull-up bar / sled this path doesn't make. The lunge used to build the cage for two
    # minutes and then die in BoneRetarget on feet 'auto' (task 9 review).
    kit = _kit()
    refusal = kit.picture_refusal(exercise)
    mode = feet or retarget.EXERCISES.get(exercise, {}).get("feet", "planted")
    if refusal is None and mode not in ("planted", "free"):
        refusal = kit.load()["refusals"]["steps"].format(clip=kit.kit_of(exercise)["clip"] or "its clip")
    if refusal:
        raise SystemExit(refusal)
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"{path} does not exist")
    if not keypose_clip.REST_CLIP.exists():
        raise SystemExit(f"--keyposes needs the Mixamo rest pose from {keypose_clip.REST_CLIP} (any Mixamo FBX "
                         f"shares it); it is missing")
    rest = retarget.load_clip(keypose_clip.REST_CLIP, turned="all")
    try:
        legs = keypose_clip.DOWN if lean is None else keypose_clip.leg_rest(lean)
        clip = keypose_clip.clip_from_keyposes(rest, json.loads(path.read_text()), feet=mode, label=str(path),
                                               legs=legs)
    except ValueError as problem:
        raise SystemExit(f"No motion for this exercise: {problem}") from None
    # the same line `load_clip` prints, so the Exercise Studio reads this clip's frame count (server/steps.js)
    if quiet:
        return clip
    print(f"retarget: read {path.name}: {len(clip.frames)} frames at {clip.fps:g} fps (the pictures' key "
          f"positions)" + "".join(f"\n  note: {n}" for n in clip.notes), flush=True)
    return clip


def _kit():
    """studio/measure/kit.py (exercise-kit.json: what each exercise holds and stands on), imported on first use."""
    path = str(HERE.parents[1] / "studio" / "measure")
    if path not in sys.path:
        sys.path.insert(0, path)
    import kit
    return kit


# Exercises whose bar rides on the upper back (`mpfb_cage.mount_barbell`) in the --mpfb path. (The Studio's
# studio/measure/exercise-kit.json marks the same one "built"; every other held bar or dumbbell is not.)
MPFB_BARBELL_ON_BACK = ("Barbell Back Squat",)


def retarget_module():
    """`retarget`, imported on first use: it imports this module, so not at the top of it."""
    import retarget
    return retarget


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exercise", default=None,
                        help="required, except with --mixamo, where leaving it out runs the clip "
                             "with no staging, no holds and no form check (for test clips)")
    parser.add_argument("--out", default=None, help="directory for the frames")
    parser.add_argument("--width", type=int, default=None, help="540, or 360 with --fast")
    parser.add_argument("--height", type=int, default=None, help="720, or 480 with --fast")
    parser.add_argument("--save", default=None)
    parser.add_argument("--mixamo", default=None, metavar="PATH",
                        help="drive the rig from a captured clip (Mixamo FBX) instead of the "
                             "two-keyframe pose_model timeline; see retarget.py")
    parser.add_argument("--feet", default=None, choices=("planted", "auto", "free"),
                        help="with --mixamo: override the exercise's foot-lock mode")
    parser.add_argument("--no-foot-lock", action="store_true",
                        help="with --mixamo: ground every frame but let the feet slide (to "
                             "measure what the lock is fixing)")
    parser.add_argument("--no-holds", action="store_true",
                        help="with --mixamo: keep every region from the capture, ignoring the "
                             "exercise's held regions (retarget.EXERCISES)")
    parser.add_argument("--fast", action="store_true",
                        help="Workbench preview at 360x480 instead of Cycles: for judging motion")
    parser.add_argument("--gif", default=None, metavar="PATH",
                        help="stitch the rendered frames into this GIF (needs --out and ffmpeg)")
    parser.add_argument("--side-view", action="store_true",
                        help="also render every frame from the figure's left side (azimuth 90, "
                             "same centre and distance) into OUT/side; --gif puts the two side "
                             "by side")
    parser.add_argument("--mark-sides", action="store_true",
                        help="debug: blue balls on the character's LEFT hand and foot, red on the "
                             "right")
    parser.add_argument("--model", default=None,
                        help="a character mesh to rig and animate instead of the built-in one")
    parser.add_argument("--wrap-skin", default=None, metavar="PATH",
                        help="a character mesh to wear over the clean cage (skin_wrap) instead "
                             "of rigging it directly")
    parser.add_argument("--mpfb", action="store_true",
                        help="with --wrap-skin and --mixamo: wear the skin over an MPFB human (Mixamo "
                             "skeleton, artist weights; mpfb_cage.py) and drive it bone for bone "
                             "(retarget.BoneRetarget); EEVEE preview at 560x560")
    parser.add_argument("--keyposes", default=None, metavar="JSON",
                        help="with --mpfb and --wrap-skin, instead of --mixamo: no clip - move between the "
                             "reference pictures' key positions (the Exercise Studio's keyposes.json; "
                             "keypose_clip.py), start -> middle -> end in 1.5 s, held 0.2 s, back in 1.0 s")
    parser.add_argument("--form-targets", default=None, metavar="JSON",
                        help="with --mpfb: correct the clip's poses toward measured reference form "
                             "(e.g. 3d-character/reference-squat/targets.json), keeping its timing")
    parser.add_argument("--bare-cage", action="store_true",
                        help="with --wrap-skin: animate the clean cage fitted to that skin, without "
                             "wearing the skin (fast; for checking motion)")
    parser.add_argument("--dumbbells", action="store_true",
                        help="put a dumbbell in each hand")
    parser.add_argument("--usdz", default=None, metavar="PATH",
                        help="export the rigged, animated character as .usdz for the app")
    parser.add_argument("--transparent", action="store_true",
                        help="no backdrop, alpha behind the figure, for compositing in the app")
    args = parser.parse_args(argv)
    if not args.exercise and not args.mixamo:
        parser.error("--exercise is required (it may be left out only with --mixamo)")
    if args.keyposes and not args.mpfb:
        parser.error("--keyposes works on the --mpfb path (with --wrap-skin)")
    if args.mpfb:
        args.width = args.width or 560
        args.height = args.height or 560
        run_mpfb(args)
        return
    args.width = args.width or (360 if args.fast else 540)
    args.height = args.height or (480 if args.fast else 720)

    # A captured clip is read FIRST, into plain numbers: every way of building the character below
    # empties the scene (`make_character.clear`), and the clip's armature would go with it.
    clip = None
    if args.mixamo:
        import retarget
        clip = retarget.load_clip(args.mixamo)

    phases = {}
    if args.exercise:
        data = pose_model.load()
        phases = pose_model.phases_of(data, args.exercise)
        if "start" not in phases or "finish" not in phases:
            raise SystemExit(f"{args.exercise!r} needs both a start and a finish position; it has "
                             f"{sorted(phases)}")
        if args.exercise in MORE_KEYFRAMES and clip is None:
            print(f"NOTE: the notes for this movement name more positions than the two used "
                  f"here —\n      {MORE_KEYFRAMES[args.exercise]}")

    # Refuse rather than guess. `pose_model.solve` raises on an angle the source stated without
    # saying which convention it used, and that is the correct outcome: a demonstration posed
    # from a coin-flip is the failure this whole pipeline exists to avoid.
    for name, phase in phases.items():
        try:
            pose_model.solve(phase)
        except ValueError as problem:
            raise SystemExit(f"{args.exercise} [{name}]: {problem}\n"
                             f"Resolve it in reference/exercises/poses.json and run again.")

    if args.model and args.wrap_skin:
        raise SystemExit("--model and --wrap-skin are two different ways to build the character; "
                         "give one of them")
    if args.wrap_skin:
        # A supplied mesh worn over the clean cage. `body` is the SKIN from here on — the thing
        # rendered, measured and exported — and the cage stays out of everything except the
        # deformation. See `wear_skin`.
        workspace = Path(args.out) if args.out else Path(tempfile.gettempdir())
        workspace.mkdir(parents=True, exist_ok=True)
        if args.bare_cage:
            # The clean cage at the skin's proportions and in its bind pose, WITHOUT the skin:
            # no Surface Deform, no bake. The same rig and the same bone lengths the worn skin
            # would get, so a retarget proven on it is proven for the skin's skeleton; it is for
            # tests, where the skin's weights are another question and the bake is minutes of
            # work that changes nothing about where a bone points.
            import skin_wrap
            body, rig, _ = skin_wrap.fit_cage_to(args.wrap_skin, scratch=workspace)
            cage = body
        else:
            body, rig, cage = wear_skin(args.wrap_skin, workspace)
        skeleton = None
    elif args.model:
        # A supplied mesh: rig it by measuring it. Only the bone DIRECTIONS come from the pose
        # solver, so the model's own proportions are kept — a character with a huge head and
        # short legs stays that character while its joints follow the form notes.
        body = fit_model.load(args.model)
        fit_model.decimate(body)
        # The silhouette is working material, not output, so it goes beside the frames when
        # there are frames and into a temporary file when the run only exports a model.
        workspace = Path(args.out) if args.out else Path(tempfile.gettempdir())
        workspace.mkdir(parents=True, exist_ok=True)
        marks = fit_model.measure(body, workspace / "silhouette.png")
        joints = fit_model.joints_from(marks)
        rig = fit_model.build_rig(joints)
        fit_model.describe(fit_model.weight(body, rig, joints))
        skeleton = None
    else:
        make_character.clear()
        skeleton = make_character.rest_skeleton()
        body = make_character.build_body(
            skeleton, equipment="dumbbells" if args.dumbbells else None)
        rig = make_character.build_rig(skeleton)
        make_character.bind(body, rig)
    make_character.stage(skeleton, body)
    if args.transparent:
        # The app puts this on its own card, the way the drawn art is cut out. The lights stay;
        # only the backdrop and floor go, so the figure keeps the same studio shading.
        for name in ("backdrop", "floor"):
            if name in bpy.data.objects:
                bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)

    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="POSE")

    # One direction only: start, hold, finish, hold. The player bounces it back, so the rep
    # returns along the path it came down — which is what a lift does — and the loop is seamless
    # by construction rather than by the last frame happening to match the first. It also halves
    # the sheet, which matters on a screen already carrying a 3D backdrop.
    timeline = [(1, "start"), (1 + HOLD, "start"),
                (1 + HOLD + HALF_REP, "finish"), (1 + 2 * HOLD + HALF_REP, "finish")]
    # Where this exercise's body is supported: the floor for thirteen of the fourteen, the
    # bench pad's own measured height for a bench press. Asked once, before the loop, because it
    # is a property of the exercise and not of the frame.
    rest_at = exercise_stage.support_height(args.exercise, body) if args.exercise else 0.0
    # One height for the whole rep where the figure's contact is a seat rather than the floor,
    # measured off its own contact phase. See `ground_rig`'s `pelvis_at`.
    pelvis_at = exercise_stage.pelvis_anchor(args.exercise, rig, body) if args.exercise else None
    scene = bpy.context.scene
    retargeted = None
    if clip is not None:
        # A captured performance instead of the two documented positions: one key per captured
        # frame, grounded every frame and with the planted feet locked. See retarget.py.
        orientations = {phase.get("orientation", "standing") for phase in phases.values()}
        orientation = phases["start"].get("orientation", "standing") if phases else "standing"
        if len(orientations) > 1:
            print(f"NOTE: {args.exercise} changes orientation between its phases "
                  f"({sorted(orientations)}); the clip is grounded as {orientation!r} throughout")
        driver = retarget.Retarget(rig, body, clip, exercise=args.exercise,
                                   orientation=orientation, rest_at=rest_at, pelvis_at=pelvis_at,
                                   feet=args.feet, foot_lock=not args.no_foot_lock,
                                   holds={} if args.no_holds else None)
        retargeted = driver.run()
        metrics, replay = retarget.contact_metrics(rig, body, driver.plan, rest_at=rest_at,
                                                   first_frame=scene.frame_start)
        retargeted["contact"] = metrics
        drift = max(max((Vector(a) - Vector(b)).length for a, b in
                        zip(frame["joints"].values(), joints.values()))
                    for frame, joints in zip(retargeted["per_frame"], replay))
        retargeted["replay_vs_posed_max"] = drift
        print(f"retarget contact (played back): penetration {metrics['penetration_mm']} mm | "
              + " | ".join(f"{s} sole slide {m['sole_slide_mm']} mm, lift {m['sole_lift_mm']} mm "
                           f"over {m['planted_frames']} planted frames"
                           for s, m in metrics["sides"].items())
              + f" | played-back joints vs posed: {drift * 1000:.3f} mm")
        kept = retarget.fidelity(clip, driver.rotation, replay)
        retargeted["fidelity_vs_capture"] = kept
        print("retarget fidelity (max |our angle - the capture's own|, whole clip): " +
              ", ".join(f"{k} {v[0]:.1f}" for k, v in kept.items()))
        if args.exercise:
            check = retarget.form_check(args.exercise, replay, driver.holds,
                                        first_frame=scene.frame_start)
            for found in check.values():
                source = retarget.source_angles(clip, driver.rotation,
                                                found["frame"] - scene.frame_start)
                for row in found["rows"]:
                    row["capture"] = round(source[row["angle"]], 1)
            retarget.print_form(check, args.exercise)
            retargeted["form_check"] = check
        timeline = [(found["frame"], phase)
                    for phase, found in sorted(retargeted.get("form_check", {}).items(),
                                               key=lambda item: item[1]["frame"])]
    else:
        for frame, which in timeline:
            pose(rig, with_clavicles(pose_model.solve(phases[which])), frame, body=body,
                 orientation=phases[which].get("orientation", "standing"), rest_at=rest_at,
                 pelvis_at=pelvis_at)

        scene.frame_start, scene.frame_end = 1, timeline[-1][0]
        scene.render.fps = FPS
        for curve in fcurves_of(rig.animation_data.action):
            for point in curve.keyframe_points:
                point.interpolation = "BEZIER"
                point.easing = "EASE_IN_OUT"

    # Per-exercise staging: this movement's equipment, where that equipment sits on this body,
    # and a camera angle that makes THIS movement legible. Without it all fourteen render as the
    # same empty-handed figure seen from the front, and four of the seven orientations are not
    # readable from the front at all.
    #
    # It runs here, after the keyframes, and not beside `make_character.stage` above, for two
    # reasons. Every prop is positioned from the POSED rig — the bar goes where the hands
    # actually are — so there has to be a pose to read; and it has to be the right pose, so the
    # frame is set to whichever phase that exercise's table says defines its contact (the bench
    # under a supine back, the plates on the floor at the bottom of a deadlift) before the
    # measurement is taken. Held props are bone-parented, so they follow the rig for the whole
    # rep from that one placement.
    #
    # An exercise with no entry in the table raises rather than rendering a bare front view,
    # which is the failure this replaces. `--dumbbells` above is now redundant for any exercise
    # whose entry already hands it a pair.
    # The first frame each phase is held at. `timeline` names each phase twice (it holds a beat
    # at each end), and a dict comprehension would silently keep the second one, so this keeps
    # the one the phase arrives on.
    # The render settings go in BEFORE the staging call, and the order matters now. Staging
    # frames the camera on the whole movement and checks what fits, and "what fits" depends on
    # the frame's aspect ratio — so with these set afterwards it was measuring against
    # Blender's 1920x1080 default instead of against the frame actually being rendered.
    if args.fast:
        retarget_module().use_workbench(scene, args.width, args.height)
    else:
        make_character.use_cycles(scene, samples=64)
    scene.render.resolution_x, scene.render.resolution_y = args.width, args.height
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA" if args.transparent else "RGB"
    scene.render.film_transparent = args.transparent

    phase_frames = {}
    for frame, which in timeline:
        phase_frames.setdefault(which, frame)
    if args.exercise:
        staging = exercise_stage.staging_for(args.exercise)["contact"]
        scene.frame_set(phase_frames[staging])
        staged = exercise_stage.stage_exercise(
            args.exercise, rig, body,
            {which: pose_model.solve(phase) for which, phase in phases.items()})
        print(f"staged {args.exercise}: {[prop.name for prop in staged['props']]} | "
              f"camera azimuth {staged['azimuth']} elevation {staged['elevation']} "
              f"({staged['reason']}) | body resting at {staged['rest_at']:+.4f} | "
              f"placed from the {staged['contact']} phase, frame {phase_frames[staging]}")
        view = {"centre": staged["centre"], "elevation": staged["elevation"],
                "distance": staged["distance"], "props": staged["props"]}
        if retargeted is not None:
            # The staged camera holds the two SOLVED phases; a capture moves through its own
            # range, so the same camera (azimuth, elevation, distance) is re-centred on that.
            box = retargeted["contact"]["box"]
            view["centre"] = (Vector(box[0]) + Vector(box[1])) / 2
            exercise_stage.place_camera(staged["camera"], view["centre"], staged["azimuth"],
                                        staged["elevation"], staged["distance"])
    else:
        _, centre, distance = retarget_module().frame_clip(body, retargeted["contact"]["box"])
        view = {"centre": centre, "elevation": 8.0, "distance": distance, "props": []}
    if args.mark_sides:
        retarget_module().mark_sides(rig)
    scene.frame_set(scene.frame_start)
    if retargeted is not None and (args.out or args.save):
        report = Path(args.out) if args.out else Path(args.save).parent
        report.mkdir(parents=True, exist_ok=True)
        retarget_module().write_report(report / "retarget.json", retargeted)

    # Render, then save, then export — in that order, because `export_usdz` DELETES the backdrop,
    # the floor and every lamp before it writes (the app lights its own scene). It used to run
    # first, so a run given both `--usdz` and `--out` rendered its frames in the dark with no set,
    # and `--save` was skipped outright whenever there were no frames.
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        scene.render.filepath = str(out / "frame_")
        bpy.ops.render.render(animation=True)
        print(f"{args.exercise or args.mixamo}: frames {scene.frame_start}..{scene.frame_end} at "
              f"{scene.render.fps}fps -> {out}")
        side = None
        if args.side_view:
            # The figure's own left side, at the same centre, height and distance: the view a
            # coach judges a squat, a hinge or a press from.
            #
            # The props the figure HOLDS are left out of this pass. Seen square from the side a
            # barbell points straight at the camera and its near plate is a disc the size of the
            # lifter's torso, standing between the camera and exactly the hip, knee and back line
            # this view exists to show. The staged view beside it carries the equipment.
            side = out / "side"
            side.mkdir(parents=True, exist_ok=True)
            held = [prop for prop in view["props"] if prop.parent is rig and not prop.hide_render]
            for prop in held:
                prop.hide_render = True
            exercise_stage.place_set(view["centre"], 90.0, view["props"])
            exercise_stage.place_camera(scene.camera, view["centre"], 90.0, view["elevation"],
                                        view["distance"])
            scene.render.filepath = str(side / "frame_")
            bpy.ops.render.render(animation=True)
            for prop in held:
                prop.hide_render = False
            print(f"side view -> {side} (held props hidden: {[p.name for p in held]})")
        if args.gif:
            retarget_module().frames_to_gif(out, args.gif, scene.render.fps, beside=side)

    if args.save:
        bpy.ops.wm.save_as_mainfile(filepath=args.save)

    if args.usdz:
        export_usdz(Path(args.usdz), body, rig)


if __name__ == "__main__":
    main()
