import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'pipeline/blender'))
from row_motion import stabilize_body


def test_row_holds_body_but_preserves_arm_motion():
    a = {n: np.eye(3) for n in ('Hips', 'Spine', 'Head', 'LeftUpLeg', 'LeftArm', 'RightForeArm')}
    b = {n: np.array([[1,0,0],[0,0,-1],[0,1,0]], dtype=float) for n in a}
    result = stabilize_body([a, b])
    for n in ('Hips', 'Spine', 'Head', 'LeftUpLeg'):
        np.testing.assert_allclose(result[1][n], a[n])
    np.testing.assert_allclose(result[1]['LeftArm'], b['LeftArm'])
    assert not np.allclose(result[1]['LeftArm'], a['LeftArm'])


def test_empty_motion_has_no_invented_pose():
    assert stabilize_body([]) == []


def test_body_matrices_are_copied():
    keys = [{'Hips': np.eye(3)}, {'Hips': np.eye(3) * 2}]
    output = stabilize_body(keys)
    output[1]['Hips'][0, 0] = 7
    assert keys[0]['Hips'][0, 0] == 1
