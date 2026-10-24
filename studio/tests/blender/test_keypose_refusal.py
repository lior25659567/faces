"""animate.py --mpfb --keyposes refuses, in the Studio's plain words and BEFORE the character is built, an exercise
the picture path can't animate (studio/measure/kit.py): the walking lunge (its feet step) and one that needs a
support the path doesn't build (the bench press's bench). Before, the lunge built the cage for two minutes and then
died in BoneRetarget on feet 'auto' (task 9 review). Runs inside Blender (no other Blender running), seconds:

    /Applications/Blender.app/Contents/MacOS/Blender --background --python-exit-code 1 \
        --python studio/tests/blender/test_keypose_refusal.py
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

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "studio" / "tests"))
sys.path.insert(0, str(REPO / "pipeline" / "blender"))
from synthetic_pose import mp_pose  # noqa: E402
import animate  # noqa: E402
import mpfb_cage  # noqa: E402

WANT = {
    "Walking Lunge (dumbbells optional)": "No motion for this exercise: its feet step, which the picture path can't "
                                          "animate yet. Add lunge.fbx to 3d-character/mixamo (its README says how).",
    "Flat Barbell Bench Press": "No motion for this exercise: it needs a bench the picture path doesn't build yet.",
}


def reached(*args, **kwargs):
    raise AssertionError("the character was being built: the refusal came too late")


def main():
    mpfb_cage.wear = reached
    folder = Path(tempfile.mkdtemp(prefix="keypose-refusal-"))
    (folder / "keyposes.json").write_text(json.dumps({"phases": [mp_pose(0, 180, 178),
                                                                 mp_pose(20, 90, 90, phase="end")]}))
    for exercise, want in WANT.items():
        sys.argv = ["blender", "--", "--mpfb", "--exercise", exercise,
                    "--wrap-skin", str(REPO / "3d-character" / "3d cartoon boy.glb"),
                    "--keyposes", str(folder / "keyposes.json"), "--out", str(folder / "frames")]
        try:
            animate.main()
        except SystemExit as refused:
            assert str(refused) == want, (exercise, str(refused))
            print(f"test_keypose_refusal: {exercise}: {refused}", flush=True)
            continue
        raise AssertionError(f"{exercise}: animated, want the refusal {want!r}")
    print("test_keypose_refusal: PASS", flush=True)


main()
