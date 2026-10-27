"""Replay actual keyed arm continuity without rendering or fitting again.

blender --background SCENE --python-exit-code 1 --python THIS_FILE -- --out REPORT --expect clear|snap
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

try:
    import bpy
except ImportError:
    import pytest
    pytest.skip('Requires Blender and an actual keyed character scene', allow_module_level=True)

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'studio/blender'), str(ROOT / 'studio/measure')]
import inspect_run
from checks import _upper_limb_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--expect', choices=('clear', 'snap'), required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    scene = bpy.context.scene
    rig = next(obj for obj in bpy.data.objects if obj.type == 'ARMATURE' and 'skin_fit' in obj)
    measured = inspect_run.upper_limb_motion(rig, scene.frame_start, scene.frame_end, loop=True)
    rows = _upper_limb_rows({'first': scene.frame_start, 'last': scene.frame_end, 'upper_limb_motion': measured})
    assert all(row['kind'] is None for row in rows), rows
    failed = any(row['status'] == 'fail' for row in rows)
    assert failed == (args.expect == 'snap'), rows
    measured.update(rows=rows, expected=args.expect, status='passed', rendered=False,
                    scene_sha256=hashlib.sha256(Path(bpy.data.filepath).read_bytes()).hexdigest())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(measured, indent=2, allow_nan=False))
    print('UPPER_LIMB_REPLAY_PASS ' + str(args.out), flush=True)


if __name__ == '__main__':
    main()
