"""Mechanical deadlift range on the generated rig, independent of source confidence.

Reference depth uncertainty cannot excuse bent lifting arms or an incomplete
lockout on a rig whose world-space joints are known. Hip flexion is measured in
the body's sagittal plane, so an outward foot stance is not counted as a hinge.
This module does not certify spinal loading or anatomical safety.
"""
import numpy as np


MIN_ELBOW_DEG = 175.
MAX_START_HIP_DEG = 150.
MAX_START_KNEE_DEG = 165.
MIN_LOCKOUT_HIP_DEG = 170.
MIN_LOCKOUT_KNEE_DEG = 170.
MAX_LOCKOUT_TORSO_DEG = 10.
MIN_START_BAR_BELOW_KNEE_HEIGHT = .02
MIN_START_GRIP_OUTSIDE_KNEES_HEIGHT = .005
MIN_FORWARD_BAR_TRAVEL_HEIGHT = .10


def _point(value):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError('Deadlift measurements require finite 3D joint positions.')
    return result


def _included(a, joint, b):
    a, b = a - joint, b - joint
    scale = float(np.linalg.norm(a) * np.linalg.norm(b))
    if scale <= 1e-12:
        raise ValueError('A deadlift angle needs two nonzero bone segments.')
    return float(np.degrees(np.arccos(np.clip(a @ b / scale, -1., 1.))))


def measure_frame(frame, joints, bar_centre):
    """Measure one frame from left/right shoulder, elbow, wrist, hip, knee, ankle.

    Inputs are actual world-space rig joint positions in metres. The bar centre
    is measured on the keyed equipment, not inferred from wrist height.
    """
    if isinstance(frame, bool) or not isinstance(frame, int):
        raise ValueError('A deadlift frame number must be an integer.')
    if set(joints) != {'left', 'right'}:
        raise ValueError('Deadlift range needs joints on both sides.')
    names = ('shoulder', 'elbow', 'wrist', 'hip', 'knee', 'ankle')
    points = {side: {name: _point(joints[side][name]) for name in names} for side in joints}
    bar = _point(bar_centre)
    lateral = points['right']['hip'] - points['left']['hip']
    lateral[2] = 0.
    width = float(np.linalg.norm(lateral))
    if width <= 1e-8:
        raise ValueError('Cannot measure the deadlift sagittal plane on overlapping hips.')
    forward = np.cross(lateral / width, [0., 0., 1.])

    def sagittal(value):
        return np.array([float(value @ forward), float(value[2])])

    elbows, knees, hips = {}, {}, {}
    for side, p in points.items():
        elbows[side] = _included(p['shoulder'], p['elbow'], p['wrist'])
        knees[side] = _included(p['hip'], p['knee'], p['ankle'])
        hips[side] = _included(sagittal(p['shoulder']), sagittal(p['hip']), sagittal(p['knee']))
    torso = (points['left']['shoulder'] + points['right']['shoulder']
             - points['left']['hip'] - points['right']['hip']) / 2
    sagittal_torso = sagittal(torso)
    length = float(np.linalg.norm(sagittal_torso))
    if length <= 1e-8:
        raise ValueError('The deadlift torso has no sagittal extent.')
    return {'frame': frame, 'elbows_deg': elbows, 'knees_deg': knees, 'hips_deg': hips,
            'torso_lean_deg': float(np.degrees(np.arccos(np.clip(torso[2] / length, -1., 1.)))),
            'bar_height_m': float(bar[2]),
            'bar_below_knees_m': float(min(p['knee'][2] for p in points.values()) - bar[2]),
            'grip_outside_knees_m': float(min(sign * ((points[side]['wrist'] - points[side]['knee']) @ (lateral / width))
                                                 for side, sign in (('left', -1), ('right', 1))))}


def summarize(samples, height, phase_frames):
    """Require straight arms throughout, a low start and simultaneous lockout.

    The start/end frames are the recorded source phases. No best-looking frame
    is selected to hide an endpoint error. Samples must cover every frame.
    """
    if isinstance(height, bool) or not isinstance(height, (int, float)) or not np.isfinite(height) or height <= 0:
        raise ValueError('Deadlift range requires a finite positive character height.')
    if not samples:
        raise ValueError('Deadlift range requires all motion frames.')
    for sample in samples:
        frame = sample.get('frame')
        if isinstance(frame, bool) or not isinstance(frame, int):
            raise ValueError('Every deadlift sample needs an integer frame number.')
        for key in ('elbows_deg', 'knees_deg', 'hips_deg'):
            angles = sample.get(key) or {}
            if set(angles) != {'left', 'right'} or any(isinstance(a, bool) or not isinstance(a, (int, float))
                    or not np.isfinite(a) or not 0. <= a <= 180. for a in angles.values()):
                raise ValueError('Every deadlift frame needs finite angles on both sides.')
        for key in ('torso_lean_deg', 'bar_height_m', 'bar_below_knees_m', 'grip_outside_knees_m'):
            value = sample.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError('Every deadlift frame needs finite torso and actual bar measurements.')
        if not 0. <= sample['torso_lean_deg'] <= 180.:
            raise ValueError('A deadlift torso angle must be between zero and 180 degrees.')
    frames = [s['frame'] for s in samples]
    if frames != list(range(frames[0], frames[-1] + 1)):
        raise ValueError('Deadlift range requires one ordered sample for every frame.')
    if not isinstance(phase_frames, dict):
        raise ValueError('Deadlift range needs recorded start and end phase frames.')
    start_frame, end_frame = phase_frames.get('start'), phase_frames.get('end')
    if any(isinstance(f, bool) or not isinstance(f, int) or f not in frames for f in (start_frame, end_frame)) or start_frame >= end_frame:
        raise ValueError('Deadlift start/end phase frames must exist in forward order.')
    start, end = samples[start_frame - frames[0]], samples[end_frame - frames[0]]
    worst = min(samples, key=lambda s: min(s['elbows_deg'].values()))
    minimum_elbow = min(worst['elbows_deg'].values())
    bar_travel = end['bar_height_m'] - start['bar_height_m']
    failures = []
    for name, passes in (
            ('arms_straight_throughout', minimum_elbow >= MIN_ELBOW_DEG),
            ('start_hip_hinge', max(start['hips_deg'].values()) <= MAX_START_HIP_DEG),
            ('start_knee_bend', max(start['knees_deg'].values()) <= MAX_START_KNEE_DEG),
            ('hip_lockout', min(end['hips_deg'].values()) >= MIN_LOCKOUT_HIP_DEG),
            ('knee_lockout', min(end['knees_deg'].values()) >= MIN_LOCKOUT_KNEE_DEG),
            ('upright_lockout', end['torso_lean_deg'] <= MAX_LOCKOUT_TORSO_DEG),
            ('start_bar_below_knees', start['bar_below_knees_m'] >= height * MIN_START_BAR_BELOW_KNEE_HEIGHT),
            ('conventional_grip_outside_knees', start['grip_outside_knees_m'] >= height * MIN_START_GRIP_OUTSIDE_KNEES_HEIGHT),
            ('forward_bar_travel', bar_travel >= height * MIN_FORWARD_BAR_TRAVEL_HEIGHT)):
        if not passes:
            failures.append(name)
    return {'frames': len(samples), 'start_frame': start_frame, 'lockout_frame': end_frame,
            'minimum_elbow_deg': minimum_elbow, 'worst_elbow_frame': worst['frame'],
            'minimum_elbows_deg': {side: min(s['elbows_deg'][side] for s in samples) for side in ('left', 'right')},
            'elbow_extension_shortfall_deg': max(0., MIN_ELBOW_DEG - minimum_elbow),
            'lockout_hips_deg': end['hips_deg'], 'lockout_knees_deg': end['knees_deg'],
            'start_hips_deg': start['hips_deg'], 'start_knees_deg': start['knees_deg'],
            'lockout_torso_lean_deg': end['torso_lean_deg'],
            'hip_lockout_shortfall_deg': max(0., MIN_LOCKOUT_HIP_DEG - min(end['hips_deg'].values())),
            'knee_lockout_shortfall_deg': max(0., MIN_LOCKOUT_KNEE_DEG - min(end['knees_deg'].values())),
            'start_bar_below_knees_mm': start['bar_below_knees_m'] * 1000.,
            'start_grip_outside_knees_mm': start['grip_outside_knees_m'] * 1000.,
            'forward_bar_travel_mm': bar_travel * 1000.,
            'limits': {'minimum_elbow_deg': MIN_ELBOW_DEG, 'minimum_lockout_hip_deg': MIN_LOCKOUT_HIP_DEG,
                       'maximum_start_hip_deg': MAX_START_HIP_DEG, 'maximum_start_knee_deg': MAX_START_KNEE_DEG,
                       'minimum_lockout_knee_deg': MIN_LOCKOUT_KNEE_DEG, 'maximum_lockout_torso_lean_deg': MAX_LOCKOUT_TORSO_DEG,
                       'minimum_start_bar_below_knees_mm': height * MIN_START_BAR_BELOW_KNEE_HEIGHT * 1000.,
                       'minimum_start_grip_outside_knees_mm': height * MIN_START_GRIP_OUTSIDE_KNEES_HEIGHT * 1000.,
                       'minimum_forward_bar_travel_mm': height * MIN_FORWARD_BAR_TRAVEL_HEIGHT * 1000.},
            'ready': not failures, 'failures': failures, 'per_frame': samples}


def mesh_geometry(points, edges, height, *, faces=None, plate_radius=None, plate_thickness=None,
                  bar_half_length=None, sleeve_length=None):
    """Locate actual disconnected weight discs and shaft in a joined bar mesh.

    Object names and materials are not evidence of loaded plates. Components
    must have the standard full-size disc radius and a real plate thickness;
    thin decorative rim bands and the sleeves do not count as weight plates.
    Coordinates are local to the library's X-aligned bar construction.
    """
    points = np.asarray(points, dtype=float)
    edges = np.asarray(edges)
    if points.ndim != 2 or points.shape[1:] != (3,) or not len(points) or not np.isfinite(points).all():
        raise ValueError('A loaded bar needs a finite evaluated mesh.')
    if edges.ndim != 2 or edges.shape[1:] != (2,) or not np.issubdtype(edges.dtype, np.integer) or not len(edges):
        raise ValueError('Loaded plate geometry needs its actual mesh connectivity.')
    if edges.min() < 0 or edges.max() >= len(points):
        raise ValueError('A loaded bar has invalid mesh connectivity.')
    if not isinstance(faces, list) or not faces:
        raise ValueError('Loaded plates require evaluated polygon surfaces, not only wire vertices.')
    for face in faces:
        indices = np.asarray(face)
        if (indices.ndim != 1 or len(indices) < 3 or not np.issubdtype(indices.dtype, np.integer)
                or indices.min() < 0 or indices.max() >= len(points)):
            raise ValueError('A loaded bar has invalid polygon surfaces.')
    if isinstance(height, bool) or not isinstance(height, (int, float)) or not np.isfinite(height) or height <= 0:
        raise ValueError('Loaded plate geometry needs a finite positive character height.')
    unit = height / 1.75
    radius = .225 * unit if plate_radius is None else plate_radius
    thickness = .028 * unit if plate_thickness is None else plate_thickness
    half = 1.1 * unit if bar_half_length is None else bar_half_length
    sleeve = .415 * unit if sleeve_length is None else sleeve_length
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool)
               and np.isfinite(value) and value > 0 for value in (height, radius, thickness, half, sleeve)):
        raise ValueError('A loaded bar needs positive measured equipment dimensions.')
    parent = list(range(len(points)))
    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index
    for a, b in edges:
        a, b = root(int(a)), root(int(b))
        parent[b] = a
    components = {}
    for index in range(len(points)):
        components.setdefault(root(index), []).append(index)
    polygon_counts = {}
    for face in faces:
        component = root(int(face[0]))
        if any(root(int(index)) != component for index in face):
            raise ValueError('A loaded bar polygon has disconnected mesh edges.')
        polygon_counts[component] = polygon_counts.get(component, 0) + 1
    shaft, shaft_span, plates, parts = None, 0., {}, []
    for component, indices in components.items():
        part = points[indices]
        low, high = part.min(0), part.max(0)
        centre = (low + high) / 2
        axial = float(high[0] - low[0])
        radial = float(np.linalg.norm(part[:, 1:] - centre[1:], axis=1).max())
        polygon_count = polygon_counts.get(component, 0)
        if axial > shaft_span and axial >= half * 1.6 and polygon_count >= 4:
            shaft, shaft_span = centre, axial
        parts.append((indices, centre, axial, radial, polygon_count))
    if shaft is None:
        raise ValueError('The actual joined bar mesh has no measurable central shaft.')
    for indices, centre, axial, radial, polygon_count in parts:
        axial_offset = centre[0] - shaft[0]
        if (len(indices) < 16 or polygon_count < 8 or abs(radial - radius) > .003 or axial < thickness * .5
                or axial > thickness * 2 or abs(axial_offset) < half - sleeve):
            continue
        side = 'left' if axial_offset < 0 else 'right'
        candidate = {'vertex_indices': indices, 'radius_m': radial,
                     'polygon_count': polygon_count, 'axial_thickness_m': axial, 'centre_local_m': centre.tolist()}
        if side not in plates or axial > plates[side]['axial_thickness_m']:
            plates[side] = candidate
    return {'shaft_centre_local_m': shaft.tolist(), 'plates': plates,
            'expected_plate_radius_m': float(radius), 'mesh_vertex_count': len(points), 'mesh_polygon_count': len(faces)}


def summarize_plate_floor(samples, start_frame, first, last):
    """Floor contact from both loaded plate meshes, including every motion frame."""
    if (not isinstance(samples, list) or any(type(value) is not int for value in (first, last)) or first > last
            or any(not isinstance(sample, dict) or type(sample.get('frame')) is not int for sample in samples)
            or [sample.get('frame') for sample in samples] != list(range(first, last + 1))):
        raise ValueError('Loaded plate floor proof needs every ordered motion frame.')
    if type(start_frame) is not int or not first <= start_frame <= last:
        raise ValueError('Loaded plate floor proof needs the recorded start frame.')
    worst_missing, minimum, start_gaps = 0, [], None
    radii = {side: [] for side in ('left', 'right')}
    for sample in samples:
        plates = sample.get('plates')
        if (not isinstance(plates, dict) or not set(plates) <= {'left', 'right'}
                or type(sample.get('mesh_vertex_count')) is not int or sample['mesh_vertex_count'] <= 0
                or type(sample.get('mesh_polygon_count')) is not int or sample['mesh_polygon_count'] <= 0):
            raise ValueError('Every loaded plate sample needs an evaluated bar mesh.')
        worst_missing = max(worst_missing, 2 - len(plates))
        for side, reading in plates.items():
            if not isinstance(reading, dict):
                raise ValueError('A loaded plate needs its measured mesh geometry.')
            radius, z = reading.get('radius_mm'), reading.get('lowest_world_z_mm')
            if (type(reading.get('vertex_count')) is not int or not 16 <= reading['vertex_count'] <= sample['mesh_vertex_count']
                    or type(reading.get('polygon_count')) is not int or not 8 <= reading['polygon_count'] <= sample['mesh_polygon_count']
                    or type(reading.get('lowest_vertex_index')) is not int
                    or not 0 <= reading['lowest_vertex_index'] < sample['mesh_vertex_count']
                    or isinstance(radius, bool) or not isinstance(radius, (int, float)) or not np.isfinite(radius) or radius <= 0
                    or isinstance(z, bool) or not isinstance(z, (int, float)) or not np.isfinite(z)):
                raise ValueError('Loaded plate samples need finite radii and actual lowest mesh vertices.')
            radii[side].append(radius)
            minimum.append(z)
        if sample['frame'] == start_frame and set(plates) == {'left', 'right'}:
            start_gaps = {side: max(0., reading['lowest_world_z_mm']) for side, reading in plates.items()}
    return {'frames': len(samples), 'first_frame': first, 'last_frame': last,
            'start_frame': start_frame, 'missing_loaded_plates': worst_missing,
            'start_floor_gaps_mm': start_gaps,
            'start_floor_gap_mm': max(start_gaps.values()) if start_gaps else None,
            'max_floor_penetration_mm': max(0., -min(minimum)) if worst_missing == 0 and minimum else None,
            'radii_mm': {side: {'minimum': min(values), 'maximum': max(values)} if values else None
                         for side, values in radii.items()},
            'method': 'evaluated full-size disc mesh components; actual world-space lowest plate vertex in every frame',
            'per_frame': samples}


def measure_rig(rig, bar, first, last, height, phase_frames):
    """Blender adapter: actual world joints, keyed shaft and loaded plate meshes.

    Range uses the recorded source start/end frames. It never searches for a
    better endpoint. Plate floor checks use evaluated vertices, so object origin
    placement or nominal centre-minus-radius cannot substitute for real contact.
    """
    import bpy
    import equipment
    if any(type(value) is not int for value in (first, last)) or first > last:
        raise ValueError('Deadlift replay needs an ordered integer frame range.')
    if bar is None or bar.type != 'MESH':
        raise ValueError('Deadlift replay needs the actual loaded bar mesh.')
    scene = bpy.context.scene
    mapping = {'shoulder': 'Arm', 'elbow': 'ForeArm', 'wrist': 'Hand',
               'hip': 'UpLeg', 'knee': 'Leg', 'ankle': 'Foot'}
    samples, floor_samples, first_geometry = [], [], None
    try:
        for frame in range(first, last + 1):
            scene.frame_set(frame)
            joints = {side: {name: np.array(rig.matrix_world @ rig.pose.bones['mixamorig:' + title + suffix].head)
                             for name, suffix in mapping.items()}
                      for side, title in (('left', 'Left'), ('right', 'Right'))}
            evaluated = bar.evaluated_get(bpy.context.evaluated_depsgraph_get())
            mesh = evaluated.to_mesh()
            try:
                points = np.array([vertex.co for vertex in mesh.vertices])
                geometry = mesh_geometry(points, np.array([edge.vertices[:] for edge in mesh.edges]), height,
                    faces=[list(polygon.vertices) for polygon in mesh.polygons],
                    plate_radius=equipment.PLATE_DIAMETER / 2 * height / 1.75,
                    plate_thickness=equipment.PLATE_THICKNESS * height / 1.75,
                    bar_half_length=equipment.BARBELL_LENGTH / 2 * height / 1.75,
                    sleeve_length=equipment.BARBELL_SLEEVE_LENGTH * height / 1.75)
                matrix = np.array(evaluated.matrix_world)
                world = points @ matrix[:3, :3].T + matrix[:3, 3]
                centre = np.asarray(geometry['shaft_centre_local_m']) @ matrix[:3, :3].T + matrix[:3, 3]
                samples.append(measure_frame(frame, joints, centre))
                plates = {}
                along = matrix[:3, 0] / np.linalg.norm(matrix[:3, 0])
                for side, part in geometry['plates'].items():
                    indices = part['vertex_indices']
                    actual = world[indices]
                    middle = np.asarray(part['centre_local_m']) @ matrix[:3, :3].T + matrix[:3, 3]
                    radial = actual - middle
                    radial -= np.outer(radial @ along, along)
                    lowest = int(np.argmin(actual[:, 2]))
                    plates[side] = {'radius_mm': float(np.linalg.norm(radial, axis=1).max()) * 1000,
                                    'vertex_count': len(indices), 'lowest_vertex_index': indices[lowest],
                                    'polygon_count': part['polygon_count'],
                                    'lowest_world_z_mm': float(actual[lowest, 2]) * 1000}
                floor_samples.append({'frame': frame, 'mesh_vertex_count': len(points),
                                      'mesh_polygon_count': len(mesh.polygons), 'plates': plates})
                if first_geometry is None:
                    first_geometry = geometry
            finally:
                evaluated.to_mesh_clear()
        result = summarize(samples, height, phase_frames)
        result.update({'first_frame': first, 'last_frame': last, 'height_m': float(height),
                       'phase_frames': dict(phase_frames),
                       'plate_floor': summarize_plate_floor(floor_samples, phase_frames['start'], first, last),
                       'plate_mesh': first_geometry})
        floor = result['plate_floor']
        expected_radius_mm = equipment.PLATE_DIAMETER / 2 * height / 1.75 * 1000
        correct_radii = all(abs(reading['radius_mm'] - expected_radius_mm) <= 3.
                            for sample in floor_samples for reading in sample['plates'].values())
        if floor['missing_loaded_plates'] or not correct_radii:
            result['failures'].append('loaded_plate_geometry')
        if floor['start_floor_gap_mm'] is None or floor['start_floor_gap_mm'] > 5.:
            result['failures'].append('start_plate_floor_contact')
        if floor['max_floor_penetration_mm'] is None or floor['max_floor_penetration_mm'] > 3.:
            result['failures'].append('plate_floor_penetration')
        result['limits'].update({'maximum_start_plate_floor_gap_mm': 5.,
            'maximum_plate_floor_penetration_mm': 3., 'plate_radius_tolerance_mm': 3.})
        result['ready'] = not result['failures']
        return result
    finally:
        scene.frame_set(first)
