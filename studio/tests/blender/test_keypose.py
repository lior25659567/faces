"""animate.py --mpfb --keyposes on the cartoon skin, end to end: an exercise with no Mixamo clip moved between two
key positions read from side pictures - a standing air squat and its bottom (knee 90, hip 90, lean 20) - written
as read_pose writes them (MediaPipe world landmarks and the picture's own), the world ones with the depth noise the
coach's pictures showed (a straight knee bent sideways, both knees' depth off, the hidden arm sticking out). At
the deepest frame ours reads, from the Studio's side camera (inspect_run's own measures): the knee within 5 degrees
of 90, the lean (counted from the bind pose, as the Studio counts it) within 5 of 20, and the hip within 5 of 90 -
the Studio's own rig reading, its thigh counted from the thigh's skin outline in the bind pose (task 20: the
cartoon's thigh joint line slopes 10.6 degrees back standing, its skin 4.7; read joint for joint the hip read 101
here), and the key positions' legs counted from that same outline (animate.py, keypose_clip.leg_rest). Our
picture's readings are printed (the Studio judges hip and lean on
it; for the squat the solver's picture loop aims at it, for an exercise without a clip nothing does). Also: the
descent takes longer than the ascent, the rep loops without repeating a frame (81), and both soles slide and sink
at most 5 mm (retarget.json). Runs inside Blender (no
other Blender running), about 4 minutes:

    /Applications/Blender.app/Contents/MacOS/Blender --background --python-exit-code 1 \
        --python studio/tests/blender/test_keypose.py

Any failed assert exits 1. The run folder (frames, retarget.json, keyposes.json) is left in the system temp
dir (printed) so the frames can be looked at.
"""
import json
import sys
import tempfile
from pathlib import Path

try:
    import bpy  # noqa: F401
except ImportError:  # collected by pytest outside Blender: this one runs inside Blender only
    import pytest
    pytest.skip("runs inside Blender (see the module docstring)", allow_module_level=True)
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "studio" / "tests"))
sys.path.insert(0, str(REPO / "studio" / "blender"))
sys.path.insert(0, str(REPO / "pipeline" / "blender"))
from synthetic_pose import mp_pose  # noqa: E402
import inspect_run  # noqa: E402
import animate  # noqa: E402

TOL_DEG = 5.0
TOL_MM = 5.0


def noisy(entry, knee_in=0.0):
    """MediaPipe's 3D lift of a side picture, as on the coach's: depth (z) off at the knees and ankles, its own
    side-plane guess off at a knee, and the arm the camera cannot see (his left) straight out sideways. The
    picture's landmarks stay exact."""
    w = entry["world"]
    for name, (dx, dy, dz) in {"left_knee": (0.03, -0.03, -knee_in), "right_knee": (0.0, 0.02, -0.05),
                               "left_ankle": (0.0, 0.0, 0.12), "right_ankle": (0.0, 0.0, 0.03)}.items():
        w[name] = [a + b for a, b in zip(w[name], (dx, dy, dz))]
    sh = w["left_shoulder"]
    w["left_elbow"], w["left_wrist"] = [sh[0], sh[1], sh[2] + 0.28], [sh[0], sh[1], sh[2] + 0.53]
    return entry


HIP_PENDING = None      # task 20: the thigh counted from its skin outline, by the Studio and the key positions alike


def main():
    run = Path(tempfile.mkdtemp(prefix="keypose-test-"))
    keyposes = {"exercise": "Key pose test", "phases": [
        noisy(mp_pose(0, 180, 178, phase="start", arms="forward"), knee_in=0.06),
        noisy(mp_pose(20, 90, 90, phase="end", arms="forward"))]}
    (run / "keyposes.json").write_text(json.dumps(keyposes, indent=1))
    print(f"test_keypose: run folder {run}", flush=True)

    sys.argv = ["blender", "--", "--mpfb", "--exercise", "Key pose test",
                "--wrap-skin", str(REPO / "3d-character" / "3d cartoon boy.glb"),
                "--keyposes", str(run / "keyposes.json"), "--out", str(run / "frames"),
                "--width", "320", "--height", "320", "--save", str(run / "scene.blend")]
    animate.main()

    scene = bpy.context.scene
    first, last = scene.frame_start, scene.frame_end
    assert (last - first + 1, round(scene.render.fps)) == (81, 30), (first, last, scene.render.fps)
    rig, skin = inspect_run.find_rig_and_skin()
    yaw, _, side_normal, behind_normal, turn = inspect_run.studio_view(rig, "right")
    per_frame, joints = inspect_run.measure_frames(rig, first, last, "right", side_normal, behind_normal, turn)
    hips = {f: float((j["hip.left"][2] + j["hip.right"][2]) / 2) for f, j in joints.items()}
    deepest = min(hips, key=hips.get)
    rig_flat = next(p["flat"] for p in per_frame if p["frame"] == deepest)
    knee, hip, lean = rig_flat["knee_included"], rig_flat["hip_included"], rig_flat["torso_lean"]
    print(f"test_keypose: deepest frame {deepest}: knee {knee:.1f}, lean {lean:.1f}, hip {hip:.1f} (the Studio's rig "
          f"readings); hips {hips[first]:.3f} -> {hips[deepest]:.3f} m", flush=True)
    side_camera = inspect_run.camera_for("side", "right", yaw)[:2]
    picture = inspect_run.picture_at(skin, {"start": first, "end": deepest}, *side_camera, run / "picture")
    for phase in ("start", "end"):
        print(f"test_keypose: our picture at the {phase}: {picture[phase]['flat']} ({picture[phase]['seen']} of "
              f"{picture[phase]['reads']} framings seen)", flush=True)
    for name, got, want in (("knee", knee, 90.0), ("lean", lean, 20.0)):
        assert abs(got - want) <= TOL_DEG, (name, got, want)
    if HIP_PENDING:
        print(f"test_keypose: SKIPPED the hip assertion (pending): reads {hip:.1f}, want 90 +- {TOL_DEG:g}; "
              f"{HIP_PENDING}", flush=True)
    else:
        assert abs(hip - 90.0) <= TOL_DEG, ("hip", hip, 90.0)

    bottom = [f for f, h in hips.items() if h <= hips[deepest] + 0.001]
    down, up = min(bottom) - first, last - max(bottom)
    print(f"test_keypose: {down} frames down, held {len(bottom)}, {up} up", flush=True)
    assert down > up, (down, up)

    contact = json.loads((run / "frames" / "retarget.json").read_text())["contact"]
    slides = {s: m["sole_slide_mm"] for s, m in contact["sides"].items()}
    print(f"test_keypose: soles slide {slides} mm, penetration {contact['penetration_mm']} mm", flush=True)
    assert all(v <= TOL_MM for v in slides.values()), slides
    assert contact["penetration_mm"] <= TOL_MM, contact["penetration_mm"]
    assert len(list((run / "frames").glob("frame_*.png"))) == 81
    print("test_keypose: PASS", flush=True)


main()
