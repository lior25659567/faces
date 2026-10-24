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

from row_motion import solve_contact, solve_shared_bar


def grip(sign=1):
    return {'wrist': [0., 0., 0.], 'B': [0., .04, 0.],
            'kb': [float(sign), 0., 0.], 'hb': [0., 1., 0.]}


def test_row_arm_has_exact_lengths_and_grip_contact():
    shoulder, point = np.array([-.2, 0., 1.]), np.array([-.22, .2, .5])
    model = grip()
    result = solve_contact(shoulder, point, [1,0,0], -1, .3, .3, model, [-.2,.05,.75])
    assert np.linalg.norm(result['elbow'] - shoulder) == pytest.approx(.3)
    assert np.linalg.norm(result['wrist'] - result['elbow']) == pytest.approx(.3)
    np.testing.assert_allclose(result['wrist'] + result['turn'] @ np.array(model['B']), point, atol=1e-10)
    assert result['bend_deg'] <= 25
    assert np.linalg.det(result['turn']) == pytest.approx(1)


def test_unreachable_row_fails_instead_of_stretching_arms():
    with pytest.raises(ValueError, match='cannot be reached'):
        solve_contact([0,0,1], [0,0,-1], [1,0,0], -1, .2, .2, grip(), [0,0,.8])


def test_shared_bar_uses_one_centre_and_level_fixed_grip():
    shoulders = {'left': np.array([-.2,0,1]), 'right': np.array([.2,0,1])}
    models = {'left': grip(), 'right': grip(-1)}
    lengths = {side: (.3,.3) for side in shoulders}
    hints = {'left': np.array([-.22,.05,.75]), 'right': np.array([.22,.05,.75])}
    centre, results, shift = solve_shared_bar(shoulders, [0,.2,.5], [1,0,0], .22, lengths, models, hints)
    contacts = {side: r['wrist'] + r['turn'] @ np.array(models[side]['B']) for side,r in results.items()}
    np.testing.assert_allclose((contacts['left'] + contacts['right']) / 2, centre, atol=1e-10)
    np.testing.assert_allclose(contacts['right'] - contacts['left'], [.44,0,0], atol=1e-10)
    assert shift == pytest.approx(0)


def test_shared_bar_reports_reach_adaptation_without_stretching():
    shoulders = {'left': np.array([-.2,0,1]), 'right': np.array([.2,0,1])}
    models = {'left': grip(), 'right': grip(-1)}
    lengths = {side: (.3,.3) for side in shoulders}
    hints = {'left': np.array([-.22,.05,.75]), 'right': np.array([.22,.05,.75])}
    wanted = np.array([0.,.25,.35])
    centre, results, shift = solve_shared_bar(shoulders, wanted, [1,0,0], .22, lengths, models, hints)
    assert shift > .01
    assert shift == pytest.approx(np.linalg.norm(centre - wanted))
    for side, result in results.items():
        assert np.linalg.norm(result['elbow'] - shoulders[side]) == pytest.approx(.3)
        assert np.linalg.norm(result['wrist'] - result['elbow']) == pytest.approx(.3)


def test_row_grip_rotation_preserves_arbitrary_horizontal_bar_axis():
    axis = np.array([1.,1.,0.]) / np.sqrt(2)
    model = grip()
    point = np.array([.15,.2,.5])
    result = solve_contact([0,0,1], point, axis, -1, .3,.3, model, [0,.05,.75])
    np.testing.assert_allclose(result['turn'] @ model['kb'], axis, atol=1e-9)
    np.testing.assert_allclose(result['wrist'] + result['turn'] @ np.array(model['B']), point, atol=1e-9)

from row_motion import shaft_gaps


def test_finite_shaft_does_not_count_contact_beyond_its_end():
    points = np.array([[0,.1,0], [2,.1,0], [1.1,0,0]])
    gaps = shaft_gaps(points, [0,0,0], [1,0,0], 1, .1)
    assert gaps[0] == pytest.approx(0)
    assert gaps[1] > .9  # infinite-cylinder distance would misleadingly be zero
    assert gaps[2] == pytest.approx(0)  # actual capsule end cap


def test_finite_shaft_distance_is_rotation_and_translation_invariant():
    centre = np.array([3.,4.,5.])
    axis = np.array([0.,1.,0.])
    points = centre + np.array([[.1,0,0], [0,2,0]])
    gaps = shaft_gaps(points, centre, axis, 1, .1)
    np.testing.assert_allclose(gaps, [0,.9], atol=1e-9)


def test_measured_row_direct_path_rises_and_preserves_visible_elbow_bend():
    import json
    from row_motion import reference_arms, reference_arm_at
    fixture = json.loads((Path(__file__).with_name('fixtures') / 'row-measurements.json').read_text())['measurements']
    keyposes = {'phases': [{**fixture[name], 'phase': phase, 'view': 'three-quarter'}
                           for name, phase in (('start.jpg','start'),('end.jpg','end'))]}
    path = reference_arms(keyposes)
    for side in ('left','right'):
        start = reference_arm_at(path, side, 0, .105, .082)
        end = reference_arm_at(path, side, 1, .105, .082)
        assert start['elbow_deg'] == pytest.approx(172.46, abs=.1)
        assert end['elbow_deg'] == pytest.approx(107.85, abs=.1)
        assert end['wrist'][2] > start['wrist'][2] + .04
        start_reach = np.linalg.norm(start['wrist'])
        end_reach = np.linalg.norm(end['wrist'])
        assert start_reach > end_reach
        # Recover the angle from the character's own unequal arm lengths.
        angle = np.degrees(np.arccos((.105**2 + .082**2 - end_reach**2) / (2*.105*.082)))
        assert angle == pytest.approx(end['elbow_deg'])


def test_row_body_cannot_inherit_one_lower_shoulder_from_depth_noise():
    from row_motion import symmetric_body
    points={'left_shoulder':np.array([.15,-.3,.5]),'right_shoulder':np.array([-.10,-.2,.4]),
            'left_hip':np.array([.08,0.,0.]),'right_hip':np.array([-.08,0.,0.]),'nose':np.array([.1,-.4,.6])}
    body=symmetric_body(points)
    np.testing.assert_allclose(body['left_shoulder'][1:], body['right_shoulder'][1:])
    assert body['left_shoulder'][0] == -body['right_shoulder'][0]
    assert body['nose'][0] == 0
    keys=[{'LeftShoulder':np.eye(3),'LeftArm':np.eye(3)},
          {'LeftShoulder':np.eye(3)*2,'LeftArm':np.eye(3)*2}]
    np.testing.assert_allclose(stabilize_body(keys)[1]['LeftShoulder'],np.eye(3))


def test_collision_safe_target_cannot_be_adapted_back_into_body():
    shoulders = {'left': np.array([-.2,0,1]), 'right': np.array([.2,0,1])}
    models = {'left': grip(), 'right': grip(-1)}
    lengths = {side: (.3,.3) for side in shoulders}
    hints = {'left': np.array([-.22,.05,.75]), 'right': np.array([.22,.05,.75])}
    # A target the old inward adaptation could reach may violate a clearance guard.
    # The collision-safe solve must refuse it when exact reach is impossible.
    with pytest.raises(ValueError, match='outside'):
        solve_shared_bar(shoulders, [0.,.25,.35], [1,0,0], .22,
                         lengths, models, hints, allow_adaptation=False)
