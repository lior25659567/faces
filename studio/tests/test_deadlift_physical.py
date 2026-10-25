"""Measured mechanical range and loaded-mesh proof are mandatory for a floor pull."""
import copy
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'pipeline/blender'))
from deadlift_range import mesh_geometry, summarize, summarize_plate_floor
from physical_checks import rows_for


def floor_samples():
    return [{'frame': frame, 'mesh_vertex_count': 600, 'mesh_polygon_count': 500,
        'plates': {side: {'radius_mm': .225 / 1.75 * 1000, 'vertex_count': 96, 'polygon_count': 74,
                         'lowest_vertex_index': index, 'lowest_world_z_mm': bottom}
                   for side, index in [('left', 20), ('right', 220)]}}
        for frame, bottom in [(1, 0.), (2, 100.), (3, 230.)]]


def report():
    samples = [{'frame': frame, 'elbows_deg': {'left': 178., 'right': 178.},
        'knees_deg': {'left': knee, 'right': knee}, 'hips_deg': {'left': hip, 'right': hip},
        'torso_lean_deg': lean, 'bar_height_m': bar, 'bar_below_knees_m': .06,
        'grip_outside_knees_m': .04}
        for frame, knee, hip, lean, bar in [(1, 120., 100., 50., .128), (2, 145., 140., 20., .23), (3, 175., 178., 0., .36)]]
    phases = {'start': 1, 'middle': 2, 'end': 3}
    movement = summarize(samples, 1., phases)
    movement.update({'height_m': 1., 'first_frame': 1, 'last_frame': 3, 'phase_frames': phases,
        'plate_floor': summarize_plate_floor(floor_samples(), 1, 1, 3)})
    return {'frames': 3, 'reference_motion': {'phase_frames': phases}, 'physical': {
        'profile': {'family': 'hinge'}, 'contacts': {'frames': 3, 'deadlift_range': movement}}}


def deadlift_rows(value):
    return {row['metric']: row for row in rows_for(value) if row['metric'].startswith('deadlift_')}


def movement(value):
    return value['physical']['contacts']['deadlift_range']


def test_complete_mechanical_range_and_real_plate_proof_pass_all_twelve_gates():
    rows = deadlift_rows(report())
    assert len(rows) == 12
    assert all(row['status'] == 'pass' for row in rows.values())
    assert rows['deadlift_lockout_torso_lean']['target'] == 10.
    assert rows['deadlift_plate_start_floor_gap']['target'] == 5.
    assert rows['deadlift_plate_floor_penetration']['target'] == 3.
    assert rows['deadlift_missing_loaded_plates']['unit'] == 'plates'


@pytest.mark.parametrize('field,side,value,frame,metric', [
    ('elbows_deg', 'left', 120., 2, 'elbow_extension_shortfall'),
    ('hips_deg', 'right', 170., 1, 'start_hip_excess'),
    ('knees_deg', 'left', 175., 1, 'start_knee_excess'),
    ('hips_deg', 'left', 145., 3, 'hip_lockout_shortfall'),
    ('knees_deg', 'right', 155., 3, 'knee_lockout_shortfall'),
    ('torso_lean_deg', None, 20., 3, 'lockout_torso_lean'),
    ('bar_below_knees_m', None, -.01, 1, 'start_bar_height_shortfall'),
    ('grip_outside_knees_m', None, -.01, 1, 'grip_outside_knees_shortfall'),
    ('bar_height_m', None, .15, 3, 'bar_travel_shortfall')])
def test_bad_range_is_recomputed_even_when_saved_ready_and_summary_claim_success(field, side, value, frame, metric):
    value_report = report()
    reading = movement(value_report)
    sample = reading['per_frame'][frame - 1]
    if side:
        sample[field][side] = value
    else:
        sample[field] = value
    reading['ready'] = True
    reading['limits']['minimum_elbow_deg'] = 0.
    assert deadlift_rows(value_report)['deadlift_' + metric]['status'] == 'fail'


@pytest.mark.parametrize('change', ['missing', 'partial', 'duplicate', 'wrong_phase', 'missing_source', 'bad_sample', 'nan'])
def test_missing_or_partial_range_cannot_be_certified(change):
    value = report()
    reading = movement(value)
    if change == 'missing':
        del value['physical']['contacts']['deadlift_range']
    elif change == 'partial':
        reading['per_frame'].pop()
    elif change == 'duplicate':
        reading['per_frame'][1]['frame'] = 1
    elif change == 'wrong_phase':
        reading['phase_frames'] = {'start': 1, 'end': 2}
    elif change == 'missing_source':
        del value['reference_motion']
    elif change == 'bad_sample':
        reading['per_frame'][1] = None
    else:
        reading['per_frame'][1]['elbows_deg']['left'] = float('nan')
    rows = deadlift_rows(value)
    assert len(rows) == 12
    assert all(row['kind'] == 'not-measured' for row in rows.values())


def test_a_later_best_lockout_cannot_replace_the_recorded_source_end():
    value = report()
    value['reference_motion']['phase_frames'] = {'start': 1, 'end': 2}
    movement(value)['phase_frames'] = {'start': 1, 'end': 2}
    rows = deadlift_rows(value)
    assert rows['deadlift_hip_lockout_shortfall']['status'] == 'fail'
    assert rows['deadlift_knee_lockout_shortfall']['status'] == 'fail'


@pytest.mark.parametrize('change', ['missing', 'partial', 'wrong_start', 'invalid_vertex', 'invalid_surface'])
def test_unknown_plate_geometry_or_incomplete_floor_replay_cannot_pass(change):
    value = report()
    floor = movement(value)['plate_floor']
    if change == 'missing':
        del movement(value)['plate_floor']
    elif change == 'partial':
        floor['per_frame'].pop()
    elif change == 'wrong_start':
        floor['start_frame'] = 2
    elif change == 'invalid_vertex':
        floor['per_frame'][0]['plates']['left']['lowest_vertex_index'] = None
    else:
        floor['per_frame'][0]['plates']['left']['polygon_count'] = 0
    rows = deadlift_rows(value)
    assert all(rows[name]['kind'] == 'not-measured' for name in
        ('deadlift_missing_loaded_plates', 'deadlift_plate_start_floor_gap', 'deadlift_plate_floor_penetration'))


def test_loaded_plates_must_both_exist_at_every_frame_and_have_full_size_radii():
    value = report()
    del movement(value)['plate_floor']['per_frame'][1]['plates']['right']
    assert deadlift_rows(value)['deadlift_missing_loaded_plates']['status'] == 'fail'
    value = report()
    movement(value)['plate_floor']['per_frame'][1]['plates']['right']['radius_mm'] = 10.
    assert deadlift_rows(value)['deadlift_missing_loaded_plates']['status'] == 'fail'


def test_plate_floor_checks_use_actual_vertices_and_recorded_start_not_saved_scalars():
    value = report()
    floor = movement(value)['plate_floor']
    floor['start_floor_gap_mm'] = 0.
    floor['max_floor_penetration_mm'] = 0.
    floor['per_frame'][0]['plates']['right']['lowest_world_z_mm'] = 6.
    floor['per_frame'][1]['plates']['left']['lowest_world_z_mm'] = -4.
    rows = deadlift_rows(value)
    assert rows['deadlift_plate_start_floor_gap']['ours'] == 6.
    assert rows['deadlift_plate_start_floor_gap']['status'] == 'fail'
    assert rows['deadlift_plate_floor_penetration']['ours'] == 4.
    assert rows['deadlift_plate_floor_penetration']['status'] == 'fail'


def joined_bar(loaded=('left', 'right'), radius=.225 / 1.75, bands=False):
    points, edges, faces = [], [], []
    def append(vertices, polygons):
        offset = len(points)
        points.extend(vertices)
        for polygon in polygons:
            mapped = [int(index) + offset for index in polygon]
            faces.append(mapped)
            edges.extend([(mapped[i], mapped[(i + 1) % len(mapped)]) for i in range(len(mapped))])
    half, small = 1.1 / 1.75, .014 / 1.75
    append([(x, y, z) for x in (-half, half) for y in (-small, small) for z in (-small, small)],
           [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)])
    for side, sign in [('left', -1), ('right', 1)]:
        for thickness in ([.028 / 1.75] if side in loaded else []) + ([.028 / 1.75 * .18] if bands else []):
            centre = sign * (1.1 - .415 / 2) / 1.75
            ring = [(centre + x, radius * np.cos(a), radius * np.sin(a))
                    for x in (-thickness / 2, thickness / 2) for a in np.linspace(0., 2 * np.pi, 24, endpoint=False)]
            append(ring, [tuple(range(24)), tuple(range(24, 48))] +
                   [(i, (i + 1) % 24, (i + 1) % 24 + 24, i + 24) for i in range(24)])
    return np.array(points), np.array(edges, dtype=int), faces


def test_real_disc_geometry_is_required_and_thin_decorative_bands_do_not_count():
    points, edges, faces = joined_bar()
    actual = mesh_geometry(points, edges, 1., faces=faces)
    assert set(actual['plates']) == {'left', 'right'}
    assert actual['plates']['left']['radius_m'] == pytest.approx(.225 / 1.75)
    points, edges, faces = joined_bar(loaded=(), bands=True)
    assert mesh_geometry(points, edges, 1., faces=faces)['plates'] == {}
    points, edges, faces = joined_bar(loaded=('left',))
    assert set(mesh_geometry(points, edges, 1., faces=faces)['plates']) == {'left'}
    points, edges, faces = joined_bar(radius=.05)
    assert mesh_geometry(points, edges, 1., faces=faces)['plates'] == {}


def test_shifted_mesh_origin_cannot_fake_the_geometric_bar_centre_or_loaded_discs():
    points, edges, faces = joined_bar()
    original = mesh_geometry(points, edges, 1., faces=faces)
    offset = np.array([2., -.5, .4])
    shifted = mesh_geometry(points + offset, edges, 1., faces=faces)
    assert shifted['shaft_centre_local_m'] == pytest.approx(np.array(original['shaft_centre_local_m']) + offset)
    assert set(shifted['plates']) == {'left', 'right'}
    assert shifted['plates']['left']['radius_m'] == pytest.approx(original['plates']['left']['radius_m'])
    with pytest.raises(ValueError):
        mesh_geometry(points, edges, 1.)
