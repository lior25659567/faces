"""Replay a completed reference-driven row and assert its actual keyed geometry.

Run inside Blender, without rebuilding the fitted character::

    blender --background RUN/scene.blend --python-exit-code 1 \\
        --python studio/tests/blender/test_row_contact.py -- --run RUN

Requires the real run's references, keyposes, report, animation and both camera frame
sets. This is a regression check of saved scene data, not a synthetic replacement
for the reference photographs or a claim that inferred reference depth is exact.
"""
import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

try:
    import bpy
except ImportError:
    import pytest
    pytest.skip("Replay test runs inside Blender", allow_module_level=True)
import numpy as np
from mathutils import Vector

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "pipeline" / "blender"))
sys.path.insert(0, str(REPO / "studio" / "blender"))
import equipment  # noqa: E402
import inspect_run  # noqa: E402
import keypose_clip  # noqa: E402
import mpfb_cage  # noqa: E402
import retarget  # noqa: E402
import row_bar  # noqa: E402

P = "mixamorig:"


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def read_json(path):
    assert path.is_file(), f"Missing run input/output: {path}"
    return json.loads(path.read_text())


def png(path):
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", f"Not a PNG: {path}"
    width, height = struct.unpack(">II", data[16:24])
    assert width >= 64 and height >= 64, (path, width, height)
    assert len(data) > 1000, f"Empty/trivial rendered image: {path}"
    return hashlib.sha256(data).hexdigest()


def included_angle(shoulder, elbow, wrist):
    upper = np.asarray(shoulder, dtype=float) - elbow
    forearm = np.asarray(wrist, dtype=float) - elbow
    norm = np.linalg.norm(upper) * np.linalg.norm(forearm)
    assert norm > 1e-10, "Coincident joints cannot establish an elbow angle"
    return math.degrees(math.acos(float(np.clip(upper @ forearm / norm, -1., 1.))))


def quaternion_gap(a, b):
    # Antipodal quaternion representations describe the same orientation.
    return math.degrees(2 * math.acos(min(1., abs(a.dot(b)))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--scene", type=Path, help="Intermediate pose scene for geometry-only iteration")
    parser.add_argument("--report", type=Path, help="Report belonging to that intermediate scene")
    parser.add_argument("--geometry-only", action="store_true", help="Check geometry without claiming rendered-output validation")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    run = args.run.resolve()
    snapshot = read_json(run / "references.json")
    keys = read_json(run / "keyposes.json")
    assert snapshot["exercise"] == keys["exercise"] == "Barbell Bent-Over Row"
    images = {image["id"]: image for image in snapshot["images"] if not image.get("contextOnly")}
    for phase in ("start", "end"):
        entry = next((row for row in keys["phases"] if row["phase"] == phase), None)
        assert entry and entry.get("image") in images, f"No source photograph for {phase}"
        assert Path(images[entry["image"]]["path"]).is_file(), f"Missing {phase} source photograph"

    scene_file = args.scene.resolve() if args.scene else run / "scene.blend"
    assert scene_file.is_file(), f"Missing completed scene: {scene_file}"
    if not bpy.data.filepath or Path(bpy.data.filepath).resolve() != scene_file:
        bpy.ops.wm.open_mainfile(filepath=str(scene_file))
    scene = bpy.context.scene
    rig, skin = inspect_run.find_rig_and_skin()
    bar = bpy.data.objects.get("barbell")
    assert bar is not None and bar.type == "MESH", "Scene has no built row bar"
    assert bar.animation_data and bar.animation_data.action, "Bar has no keyed path"
    report = read_json(args.report if args.report else run / "frames" / "retarget.json")
    row_report = report.get("row")
    assert isinstance(row_report, dict), "Run has no row-solver report"
    for name in ("max_grip_anchor_error_mm", "worst_wrist_bend_deg", "grip_ratio"):
        assert finite(row_report.get(name)), f"Missing finite row report field: {name}"
    assert row_report["max_grip_anchor_error_mm"] <= 2
    assert row_report["worst_wrist_bend_deg"] <= 25
    first, last = scene.frame_start, scene.frame_end
    count = last - first + 1
    assert count > 2, "No meaningful animated repetition"
    fit, grips = json.loads(rig["skin_fit"]), json.loads(rig["hand_grip"])
    half_length = equipment.BARBELL_LENGTH / 2 * fit["measured"]["height"] / 1.75
    timeline = keypose_clip.timeline(float(scene.render.fps) / scene.render.fps_base)
    assert len(timeline) == count, "Row frame count differs from its reference clock"
    contracted_index = int(np.argmax(timeline))
    assert contracted_index > 0 and timeline[contracted_index] >= .999
    endpoint_elbows = {}
    source_elbows = {}
    for phase in ("start", "end"):
        entry = next(row for row in keys["phases"] if row["phase"] == phase)
        side = entry.get("camera_side")
        assert side in ("left", "right"), f"No measured camera side for {phase}"
        points = entry["world"]
        source_elbows[phase] = included_angle(*(np.array(points[f"{side}_{joint}"])
                                                for joint in ("shoulder", "elbow", "wrist")))
    centres, widths = [], []
    hand_locations = {side: [] for side in ("left", "right")}
    hand_rotations = {side: [] for side in hand_locations}
    worst_anchor, worst_level, worst_symmetry = 0., 0., 0.
    initial_hinge = None
    hinge_drift = 0.
    for frame in range(first, last + 1):
        scene.frame_set(frame)
        centre = np.array(bar.matrix_world.translation)
        along = np.array((bar.matrix_world.to_3x3() @ Vector((1, 0, 0))).normalized())
        centres.append(centre)
        worst_level = max(worst_level, abs(float(along[2])))
        if frame in (first, first + contracted_index):
            phase = "start" if frame == first else "end"
            endpoint_elbows[phase] = {}
            for side, title in (("left", "Left"), ("right", "Right")):
                joints = [np.array(rig.matrix_world @ rig.pose.bones[P + title + bone].head)
                          for bone in ("Arm", "ForeArm", "Hand")]
                endpoint_elbows[phase][side] = included_angle(*joints)
        contacts = []
        for side, title in (("left", "Left"), ("right", "Right")):
            hand = rig.pose.bones[P + title + "Hand"]
            turn = hand.matrix.to_3x3() @ hand.bone.matrix_local.to_3x3().inverted()
            local = np.array(hand.head) + np.array(turn) @ (np.array(grips[side]["B"]) - grips[side]["wrist"])
            point = np.array(rig.matrix_world @ Vector(local))
            delta = point - centre
            axial = float(delta @ along)
            radial = float(np.linalg.norm(delta - axial * along))
            assert abs(axial) < half_length, f"{side} anchor misses finite shaft at frame {frame}"
            worst_anchor = max(worst_anchor, radial)
            contacts.append(point)
            hand_locations[side].append(np.array(rig.matrix_world @ hand.head))
            hand_rotations[side].append((rig.matrix_world @ hand.matrix).to_quaternion().normalized())
        widths.append(float(np.linalg.norm(contacts[1] - contacts[0])))
        worst_symmetry = max(worst_symmetry, float(np.linalg.norm((contacts[0] + contacts[1]) / 2 - centre)))
        hinge = (rig.matrix_world @ rig.pose.bones[P + "Hips"].matrix).to_quaternion().normalized()
        if initial_hinge is None:
            initial_hinge = hinge
        hinge_drift = max(hinge_drift, quaternion_gap(initial_hinge, hinge))
    assert worst_anchor <= .002, f"Replayed anchor leaves bar by {worst_anchor * 1000:.3f} mm"
    assert worst_level <= 1e-4, f"Bar is tilted: axis.z={worst_level}"
    assert worst_symmetry <= .002, f"Hands aren't symmetric about bar: {worst_symmetry * 1000:.3f} mm"
    assert max(widths) - min(widths) <= .002, "Grip width changes across the rep"
    assert hinge_drift <= 1., f"Row hinge drifts {hinge_drift:.2f} degrees"
    # The clock, not the observed bar extremum, determines the contracted reference
    # frame. Otherwise an inverted/static motion could choose its own passing endpoint.
    bar_rise = float(centres[contracted_index][2] - centres[0][2])
    # The mascot's oversized head contributes height but does not contribute arm
    # travel. Normalize the minimum visible pull to its measured arm-chain length
    # rather than total height: a short-arm cartoon can preserve the observed elbow
    # flexion with less vertical bar travel than a proportionate human.
    arm_reach = float(np.mean([sum(rig.pose.bones[P + title + bone].length
                                  for bone in ("Arm", "ForeArm")) for title in ("Left", "Right")]))
    minimum_rise = .15 * arm_reach
    assert bar_rise >= minimum_rise, f"The row does not pull up: {bar_rise * 1000:.2f} mm; need {minimum_rise * 1000:.2f} mm"
    for side in ("left", "right"):
        start_angle, end_angle = endpoint_elbows["start"][side], endpoint_elbows["end"][side]
        assert abs(start_angle - source_elbows["start"]) <= 20., f"{side} start elbow {start_angle:.1f} vs reference {source_elbows['start']:.1f}"
        assert abs(end_angle - source_elbows["end"]) <= 20., f"{side} contracted elbow {end_angle:.1f} vs reference {source_elbows['end']:.1f}"
        assert start_angle - end_angle >= 25., f"{side} row barely flexes: {start_angle:.1f} -> {end_angle:.1f}"


    # Test the seam as well as ordinary adjacent frames: a repeated GIF must not pop
    # on the return from last frame to first. The final frame intentionally leads into
    # the first, so exact duplicate endpoint positions are not required.
    jumps = {}
    rotations = {}
    for name, points in {"bar": centres, **hand_locations}.items():
        distances = [float(np.linalg.norm(b - a)) for a, b in zip(points, points[1:] + points[:1])]
        jumps[name] = {"max_mm": max(distances) * 1000, "seam_mm": distances[-1] * 1000}
        assert max(distances) <= .020, f"{name} snaps between frames: {max(distances) * 1000:.2f} mm"
        assert distances[-1] <= .010, f"{name} loop seam snaps: {distances[-1] * 1000:.2f} mm"
    for side, turns in hand_rotations.items():
        angles = [quaternion_gap(a, b) for a, b in zip(turns, turns[1:] + turns[:1])]
        rotations[side] = {"max_deg": max(angles), "seam_deg": angles[-1]}
        assert max(angles) <= 30., f"{side} hand flips between frames: {max(angles):.2f} degrees"
        assert angles[-1] <= 15., f"{side} hand loop seam rotates {angles[-1]:.2f} degrees"

    # Recompute actual evaluated skin measurements from the scene. Stored JSON alone
    # cannot prove that either hand or foot in the saved animation is in contact.
    sides = row_bar.side_contacts(rig, skin, bar, first, last)
    for side in ("left", "right"):
        saved = (row_report.get("contact", {}).get("sides") or {}).get(side)
        assert isinstance(saved, dict), f"Missing separate {side} hand contact report"
        contact = sides[side]
        assert contact["frames"] == count
        assert contact["frames_touching_within_3mm"] >= count * .8, f"{side} hand floats: {contact}"
        assert contact["deepest_inside_bar_mm"] <= 5., f"{side} hand penetrates bar: {contact}"
    feet = retarget.skin_contact(rig, skin, first, last)
    assert feet["penetration_mm"] <= 5., f"Skin below the floor: {feet['penetration_mm']} mm"
    for side, contact in feet["sides"].items():
        assert contact["sole_vertices"] > 0, f"No measured {side} sole vertices"
        assert contact["sole_slide_mm"] <= 5., f"{side} foot slides: {contact}"
    body = row_bar.body_clearance(rig, skin, bar, first, last)
    assert body.get("status") != "not_measured", "No body-clearance evidence"
    assert not any(r.get("confirmed_inside_mm", 0) > 5 for r in body.get("frames", [])), "Bar penetrates torso/thigh"

    if args.geometry_only:
        print("test_row_contact: GEOMETRY ONLY PASS (rendered outputs not checked)")
        return

    # The whole finite bar must be visible, not just the skin used by older framing.
    from bpy_extras.object_utils import world_to_camera_view
    box = np.array(report["preview_box"], dtype=float)
    centre = box.mean(axis=0)
    distance = mpfb_cage.distance_for(max(box[1, 2] - box[0, 2], .5) * 1.05)
    for azimuth, elevation in ((35, 10), (90, 5)):
        camera = mpfb_cage.camera_on(centre, azimuth, elevation, distance)
        for frame in range(first, last + 1):
            scene.frame_set(frame)
            for corner in bar.bound_box:
                uv = world_to_camera_view(scene, camera, bar.matrix_world @ Vector(corner))
                assert uv.z > 0 and 0 <= uv.x <= 1 and 0 <= uv.y <= 1, ("Cropped bar", azimuth, frame, tuple(uv))

    hashes = []
    for folder in (run / "frames", run / "frames" / "side"):
        files = sorted(folder.glob("frame_*.png"))
        assert len(files) == count, f"Incomplete camera renders: {folder}: {len(files)} / {count}"
        camera_hashes = [png(files[index]) for index in (0, count // 2, count - 1)]
        assert len(set(camera_hashes)) > 1, f"Rendered {folder.name} camera does not move"
        # A valid PNG can still be a blank image: check the actual loaded pixels.
        image = bpy.data.images.load(str(files[count // 2]), check_existing=False)
        pixels = np.array(image.pixels[:], dtype=float).reshape(-1, image.channels)[:, :3]
        assert float(pixels.std()) > .01, f"Blank camera render: {files[count // 2]}"
        bpy.data.images.remove(image)
        hashes.extend(camera_hashes)
    gif = run / "animation.gif"
    assert gif.is_file() and gif.stat().st_size > 1000, "Missing/trivial GIF"
    assert gif.read_bytes()[:6] in (b"GIF87a", b"GIF89a"), "Animation isn't a GIF"
    result = {"status": "passed", "frames": count, "source": "replayed saved scene with real reference inputs",
              "anchor_error_mm": worst_anchor * 1000, "hinge_drift_deg": hinge_drift,
              "grip_width_range_mm": (max(widths) - min(widths)) * 1000,
              "bar_rise_mm": bar_rise * 1000, "minimum_bar_rise_mm": minimum_rise * 1000,
              "contracted_frame": first + contracted_index,
              "reference_elbows_deg": source_elbows, "actual_elbows_deg": endpoint_elbows,
              "position_jumps": jumps, "hand_rotation_jumps": rotations, "hand_contact": sides,
              "feet": feet, "render_sample_sha256": hashes}
    (run / "row-validation.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    scene.frame_set(first)
    print("test_row_contact: PASS " + json.dumps({k: result[k] for k in ("frames", "anchor_error_mm", "hinge_drift_deg")}))


if __name__ == "__main__":
    main()
