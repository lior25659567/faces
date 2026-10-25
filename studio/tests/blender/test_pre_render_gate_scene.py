"""Exercise the real CLI's pre-render stop, with render/GIF calls guarded.

The negative case injects only the recorded verdict from a real failed pull-up;
it does not alter the production classifier or claim the plank has that defect.
All output belongs to the caller's isolated --directory.
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path
try:
    import bpy
except ImportError:
    import pytest
    pytest.skip('Requires Blender and a real checked reference set', allow_module_level=True)

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'pipeline'), str(ROOT / 'pipeline/blender'), str(ROOT / 'studio/blender')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('positive', 'negative'), required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--negative-verdict', type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    args.directory.mkdir(parents=True, exist_ok=True)
    import animate
    import retarget
    import skin_preflight
    original_measure, original_gif = skin_preflight.measure, retarget.frames_to_gif
    original_temp = animate.tempfile.gettempdir
    render_module = bpy.ops.render
    original_render = render_module.render
    had_render_attribute = 'render' in vars(bpy.ops)
    old_render_attribute = vars(bpy.ops).get('render')
    counts = {'measure': 0, 'render': 0, 'gif': 0}
    failed = json.loads(args.negative_verdict.read_text())
    negative_row = next(row for row in failed['rows'] if row['metric'] == 'folded_skin')
    assert negative_row['status'] == 'fail' and negative_row['ours'] >= 300.

    def measure(*values, **options):
        counts['measure'] += 1
        if args.case == 'positive':
            return original_measure(*values, **options)
        return {'ready': False, 'rows': [copy.deepcopy(negative_row)], 'fold_rows': [copy.deepcopy(negative_row)],
                'motion_rows': [], 'validation_fixture': {'injected_verdict_only': True,
                'actual_negative_source': str(args.negative_verdict)}}

    def render(*values, **options):
        counts['render'] += 1
        raise AssertionError('The pre-render integration test must never spend time rendering.')

    def gif(*values, **options):
        counts['gif'] += 1
        raise AssertionError('The pre-render integration test must never encode a GIF.')

    skin_preflight.measure = measure
    retarget.frames_to_gif = gif
    render_module.render = render
    bpy.ops.render = render_module
    assert bpy.ops.render.render is render
    animate.tempfile.gettempdir = lambda: str(args.directory)
    argv = [sys.argv[0], '--', '--mpfb', '--fast', '--side-view', '--exercise', 'Front Plank (forearm plank)',
            '--wrap-skin', str(ROOT / '3d-character/3d cartoon boy.glb'), '--keyposes', str(args.source / 'keyposes.json'),
            '--save', str(args.directory / 'scene.blend')]
    if args.case == 'negative':
        argv += ['--out', str(args.directory / 'frames'), '--gif', str(args.directory / 'animation.gif')]
    held_argv, sys.argv = sys.argv, argv
    started, exit_message = time.monotonic(), None
    try:
        try:
            animate.main()
        except SystemExit as error:
            exit_message = str(error)
    finally:
        sys.argv = held_argv
        skin_preflight.measure, retarget.frames_to_gif = original_measure, original_gif
        animate.tempfile.gettempdir = original_temp
        render_module.render = original_render
        if had_render_attribute:
            bpy.ops.render = old_render_attribute
        else:
            del bpy.ops.render
    assert counts == {'measure': 1, 'render': 0, 'gif': 0}, counts
    assert (args.directory / 'scene.blend').is_file()
    assert not (args.directory / 'animation.gif').exists()
    assert not list(args.directory.rglob('frame_*.png'))
    report_file = args.directory / ('frames/retarget.json' if args.case == 'negative' else 'retarget.json')
    report = json.loads(report_file.read_text())
    if args.case == 'positive':
        assert exit_message is None and report['skin_preflight']['ready'] is True
        assert report['pre_render_validation']['ready'] is True
        assert not (args.directory / 'checks.json').exists()
    else:
        assert exit_message and 'before rendering' in exit_message, exit_message
        checks = json.loads((args.directory / 'checks.json').read_text())
        assert negative_row in checks['rows'] and checks['summary']['fail'] >= 1
        assert report['skin_preflight']['ready'] is False
        assert report['pre_render_validation']['ready'] is False
    result = {'case': args.case, 'seconds': round(time.monotonic() - started, 3), 'source': str(args.source),
              'output_directory': str(args.directory), 'calls': counts, 'scene_saved': True, 'report_saved': True,
              'checks_saved': (args.directory / 'checks.json').is_file(), 'frames_written': 0, 'gif_written': False,
              'exit_message': exit_message, 'positive_uses_actual_measurement': args.case == 'positive',
              'negative_injects_actual_failed_verdict_only': args.case == 'negative'}
    (args.directory / 'integration-result.json').write_text(json.dumps(result, indent=2))
    print('PRE_RENDER_GATE_INTEGRATION_PASS ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
