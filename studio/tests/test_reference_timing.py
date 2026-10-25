"""Generated reference comparisons use source instants, never best-fit rig angles."""
import importlib.util
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location('timing_keypose_clip', Path(__file__).resolve().parents[2] / 'pipeline/blender/keypose_clip.py')
kc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kc)


def turn(degrees):
    a = np.radians(degrees)
    c, s = np.cos(a), np.sin(a)
    return {'Arm': np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])}


def test_row_records_real_first_and_contracted_frames_even_if_middle_angular_distance_differs():
    result = kc.reference_timing({'exercise': 'Barbell Bent-Over Row'}, ['start', 'middle', 'end'], [turn(0), turn(1), turn(90)])
    assert result['phase_frames']['start'] == 1
    assert result['phase_frames']['end'] == 46
    assert result['phase_frames']['middle'] in (23, 24)
    assert result['explicit_phases'] == ['start', 'middle', 'end']


def test_other_profiles_record_the_actual_nonuniform_middle_mark():
    result = kc.reference_timing({'exercise': 'Another exercise'}, ['start', 'middle', 'end'], [turn(0), turn(1), turn(90)], fps=12)
    assert result['phase_frames'] == {'start': 1, 'middle': 2, 'end': 19}


def test_static_hold_has_one_source_instant_and_preserves_output_origin():
    result = kc.reference_timing({'motion': {'mode': 'hold'}}, ['start'], [turn(0)], fps=12, first=7)
    assert result == {'source': 'keyposes', 'mode': 'hold', 'phase_frames': {'start': 7, 'middle': 7, 'end': 7}, 'explicit_phases': ['start']}
