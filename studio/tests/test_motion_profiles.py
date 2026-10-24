"""Motion policies and static-hold semantics without Blender."""
import importlib.util
import sys
from pathlib import Path
from copy import deepcopy
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'pipeline/blender'))
from motion_profiles import profile_for, prepare_key_turns
from physical_checks import rows_for
from build_targets import build_targets
from run_read import key_positions


def test_isolation_preserves_body_but_not_arms_and_never_mutates_inputs():
    keys = [{name: np.eye(3) * value for name in ('Hips', 'Spine2', 'LeftLeg', 'LeftArm')} for value in (1, 2)]
    before = deepcopy(keys)
    poses = prepare_key_turns('Dumbbell Bicep Curl (standing, both arms)', keys)
    assert np.array_equal(poses[1]['Hips'], keys[0]['Hips'])
    assert np.array_equal(poses[1]['LeftArm'], keys[1]['LeftArm'])
    assert all(np.array_equal(k[name], original[name]) for k, original in zip(keys, before) for name in k)
    hinge = prepare_key_turns('Conventional Barbell Deadlift', keys)
    assert np.array_equal(hinge[1]['LeftLeg'], keys[1]['LeftLeg'])
    assert np.array_equal(hinge[1]['Hips'], keys[1]['Hips'])


def test_static_hold_is_one_pose_and_repeats_without_motion():
    import keypose_clip
    keys = [{'Hips': np.eye(3)}, {'Hips': np.eye(3) * 2}]
    fixed = prepare_key_turns('Front Plank (forearm plank)', keys)
    assert len(fixed) == 1
    frames = keypose_clip.motion(fixed, fps=12, feet='free')
    assert len(frames) == 12
    assert all(np.array_equal(f['Hips'], fixed[0]['Hips']) for f in frames)
    assert profile_for('Walking Lunge (dumbbells optional)') is None


@pytest.mark.parametrize('invalid', [None, float('nan'), float('inf'), True, '0'])
def test_invalid_contact_measurements_cannot_pass(invalid):
    report = {'physical': {'profile': {'holds': True}, 'contacts': {'frames': 2,
        'sides': {'left': {'frames': 2, 'nearest_gap_mm_max': invalid}, 'right': {}}}}}
    rows = rows_for(report)
    assert any(r['metric'] == 'equipment_gap_left' and r['kind'] == 'not-measured' for r in rows)
    assert not any(r['status'] == 'pass' for r in rows)


def test_missing_required_contacts_and_incomplete_support_do_not_certify_success():
    rows = rows_for({'physical': {'profile': {'holds': True, 'family': 'seated'},
        'supports': {'surfaces': {'pelvis': {'frames': 12, 'measured_frames': 11, 'max_gap_mm': 0., 'max_penetration_mm': 0.}}}}})
    assert rows and all(row['kind'] == 'not-measured' for row in rows)


def test_a_seat_alone_does_not_certify_a_leg_press_without_footplate_measurements():
    rows = rows_for({'frames': 1, 'physical': {'profile': {'family': 'sled'},
        'supports': {'surfaces': {'pelvis': {'frames': 1, 'measured_frames': 1, 'max_gap_mm': 0., 'max_penetration_mm': 0.}}}}})
    assert any(r['metric'] == 'support_gap_footplate_left' and r['kind'] == 'not-measured' for r in rows)
    assert any(r['metric'] == 'support_gap_footplate_right' and r['kind'] == 'not-measured' for r in rows)


def test_oblique_sources_remain_uncertain_and_static_hold_keeps_one_source():
    import json
    raw = json.loads((ROOT / 'studio/tests/fixtures/squat-measurements.json').read_text())
    image = {'id': 'hold', 'phase': 'start', 'view': 'three-quarter', 'viewSetBy': 'user'}
    measured = {'hold': raw['15.png']}
    targets = build_targets('Front Plank (forearm plank)', [image], measured)
    assert targets['motion']['mode'] == 'hold'
    assert targets['motion']['sourcePhases'] == ['start']
    assert targets['motion']['approximateDepth']
    assert all(r['uncertain'] for r in targets['phases']['start'].values() if isinstance(r, dict))
    keys = key_positions(targets, measured)
    assert len(keys['phases']) == 1
    assert keys['motion']['mode'] == 'hold'


def test_a_frontal_pair_can_drive_an_uncertain_preview_without_becoming_a_side_view():
    import json
    raw = json.loads((ROOT / 'studio/tests/fixtures/squat-measurements.json').read_text())
    images = [{'id': name, 'phase': phase, 'view': 'front', 'viewSetBy': 'user', 'order': i}
              for i, (name, phase) in enumerate([('15.png', 'start'), ('17.png', 'end')])]
    targets = build_targets('Pull-Up', images, raw)
    assert all(info['view'] == 'front' for info in targets['images'].values())
    assert targets['motion']['approximateDepth']
    assert targets['phases']['end']['elbow_included']['uncertain']


def test_static_hold_preflight_rejects_setup_and_motion_keys():
    from preflight import plan_for
    targets = {'phases': {}, 'motion': {'mode': 'hold'}}
    keys = {'phases': [{'phase': 'start', 'view': 'side'}, {'phase': 'end', 'view': 'side'}]}
    result = plan_for({'exercise': 'Front Plank (forearm plank)'}, targets, None, keys)
    assert not result['ready']
    assert any('one active' in b for b in result['blockers'])


def test_weight_heads_have_finite_end_caps_and_hexagonal_sides():
    from motion_geometry import hex_head_depth
    points = np.array([[.5, 0., 0.], [.5 + .06, 0., 0.], [.5, .3, 0.]])
    depth = hex_head_depth(points, .5, radius=.1, thickness=.1)
    assert depth[0] == pytest.approx(.05)
    assert np.array_equal(depth[1:], [0., 0.])


def test_an_overhead_grip_does_not_certify_an_incomplete_overhead_pose():
    report = {'frames': 1, 'physical': {'profile': {'overhead': True}, 'contacts': {'frames': 1, 'overhead_shortfall_mm': 50.}}}
    rows = rows_for(report)
    assert any(r['metric'] == 'overhead_shortfall' and r['status'] == 'fail' for r in rows)
    assert profile_for('Lat Pulldown (seated, to the front)')['overhead']


def test_high_pulldown_arms_share_elevation_with_clavicles_without_rotating_the_measured_arms():
    def around_y(angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
    key = {name: np.eye(3) for name in ('Hips', 'Spine', 'Spine1', 'Spine2', 'LeftShoulder', 'RightShoulder')}
    key.update(LeftArm=around_y(-np.pi / 2), RightArm=around_y(np.pi / 2))
    pose = prepare_key_turns('Lat Pulldown (seated, to the front)', [key])[0]
    assert np.array_equal(pose['LeftArm'], key['LeftArm'])
    assert np.array_equal(pose['RightArm'], key['RightArm'])
    assert (pose['LeftShoulder'] @ np.array([1., 0., 0.]))[2] > 0
    assert (pose['RightShoulder'] @ np.array([-1., 0., 0.]))[2] > 0
    assert np.array_equal(pose['Spine2'], key['Spine2'])
