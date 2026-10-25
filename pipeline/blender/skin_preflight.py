"""Measure movement and visible skin from the planned cameras before rendering.

The fold detector, visibility calculation and verdict are the Studio inspector's
existing ones. No new classifier or tolerance lives here. The caller stages the
real preview set first; this helper reads its geometry without rendering or
creating cameras, lights or meshes.
"""
import json
import math
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def preview_bounds(contact_box, prop_corners=()):
    """The renderer's universal padded cube, including props at every frame."""
    from preview_camera_policy import padded_box
    box = np.asarray(contact_box, dtype=float)
    if box.shape != (2, 3) or not np.isfinite(box).all() or (box[1] < box[0]).any():
        raise ValueError('Visible skin checks need finite ordered character bounds.')
    low, high = box.copy()
    for corners in prop_corners:
        points = np.asarray(corners, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
            raise ValueError('Visible skin checks need every held prop bound.')
        low, high = np.minimum(low, points.min(axis=0)), np.maximum(high, points.max(axis=0))
    return padded_box(low, high)


def source_camera_plan(snapshot, measurements, targets, phase_frames, yaw, camera_for, effective_view):
    """Every source comparison view at its recorded frame; no best-fit selection."""
    side = ((targets or {}).get('camera') or {}).get('side')
    if side not in ('left', 'right') and ((targets or {}).get('motion') or {}).get('approximateDepth'):
        side = 'left'
    if side not in ('left', 'right'):
        raise ValueError('The reference camera side is missing.')
    images = [image for image in (snapshot.get('images') or []) if not image.get('contextOnly')
              and not image.get('deleted') and not image.get('motionOnly')
              and not (image.get('source') or {}).get('motionOnly')]
    if not images:
        raise ValueError('There are no active reference comparison cameras to check.')
    result = []
    for image in images:
        phase = image.get('phase')
        if phase not in phase_frames:
            raise ValueError('An active reference image has no recorded comparison frame.')
        measured_side = (measurements.get(image['id']) or {}).get('camera_side')
        measured_side = measured_side if measured_side in ('left', 'right') else side
        azimuth, elevation, view = camera_for(effective_view(image, measurements), measured_side, yaw)
        result.append({'image': image['id'], 'phase': phase, 'frame': phase_frames[phase],
                       'view': view, 'azimuth': float(azimuth), 'elevation': float(elevation)})
    return result


def fold_verdict(measured, first, last, error=None):
    """Preserve the exact inspector row, including hidden info and missing data."""
    from checks import _fold_rows
    ours = {'first': first, 'last': last}
    if error is None and isinstance(measured, dict):
        ours.update(measured)
    rows = _fold_rows(ours)
    ready = measured is not None and error is None and bool(rows) and all(
        row['status'] != 'fail' and row.get('kind') != 'not-measured' for row in rows)
    return rows, ready


def motion_verdict(measured, error=None):
    """Use the unchanged inspector jump/step verdict, including missing coverage."""
    from checks import _motion_rows
    rows = _motion_rows(measured if error is None and isinstance(measured, dict) else {})
    ready = measured is not None and error is None and bool(rows) and all(
        row['status'] != 'fail' and row.get('kind') != 'not-measured' for row in rows)
    return rows, ready


def measure(rig, skin, first, last, exercise, held, contact_box, keyposes_path, side_view=True,
            reference_motion=None, preview_views=None):
    """Return exact planned-view folds and motion rows without changing the scene.

    Pass the generated clip's ``reference_motion`` during a fresh build. Replays
    may read it from the saved retarget report. Missing source-camera coverage or
    unreadable fold measurements block rendering instead of becoming a pass.
    ``preview_cameras`` contains world azimuth/elevation pairs for the renderer.
    """
    import bpy
    from mathutils import Vector
    for path in (ROOT / 'pipeline', ROOT / 'pipeline/blender', ROOT / 'studio/blender', ROOT / 'studio/measure'):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import inspect_run as inspector
    import mpfb_cage as cage
    from preview_camera_policy import views_for
    from reference_timing import recorded_phase_frames
    scene = bpy.context.scene
    original_frame, original_subframe, original_camera = scene.frame_current, scene.frame_subframe, scene.camera
    result = {'first': first, 'last': last, 'preview_box': None, 'preview_cameras': [],
              'preview_camera_space': 'world', 'preview_camera_positions': [], 'comparison_camera_frames': [],
              'comparison_cameras': [], 'height_m': None, 'folds': None, 'rows': [], 'fold_rows': [],
              'motion_rows': [], 'jumps': None, 'steps': None, 'ready': False,
              'measured_frames': 0, 'measurement_error': None}
    measured = None
    motion = None
    try:
        if type(first) is not int or type(last) is not int or first > last or first != scene.frame_start or last != scene.frame_end:
            raise ValueError('Visible skin checks require the entire valid frame range.')
        root = Path(keyposes_path).resolve().parent
        def read(name):
            value = json.loads((root / name).read_text())
            if not isinstance(value, dict):
                raise ValueError('The reference camera metadata is unreadable.')
            return value
        snapshot, targets, measurements = [read(name) for name in ('references.json', 'targets.json', 'measurements.json')]
        if reference_motion is None:
            reference_motion = read('frames/retarget.json').get('reference_motion')
        phases = recorded_phase_frames([{'frame': frame} for frame in range(first, last + 1)], reference_motion)
        if phases is None:
            raise ValueError('The generated animation has no recorded reference comparison times.')
        corners = []
        for frame in range(first, last + 1):
            scene.frame_set(frame)
            for prop in held or []:
                corners.append(np.array([prop.matrix_world @ Vector(corner) for corner in prop.bound_box]))
        box = preview_bounds(contact_box, corners)
        views = tuple(preview_views) if preview_views is not None else views_for(exercise, side_view=side_view)
        # Validate explicit world angles using the renderer's report parser.
        from preview_camera_policy import angles_from_report
        views = angles_from_report({'preview_cameras': [list(view) for view in views], 'preview_camera_space': 'world'})
        positions = inspector.gif_cameras(box, 0., views=views)
        result.update(preview_box=box, preview_cameras=[list(view) for view in views],
                      preview_camera_positions=[position.tolist() for position in positions], phase_frames=phases)
        side = (targets.get('camera') or {}).get('side')
        if side not in ('left', 'right') and (targets.get('motion') or {}).get('approximateDepth'):
            side = 'left'
        if side not in ('left', 'right'):
            raise ValueError('The reference camera side is missing.')
        yaw, _, side_normal, behind_normal, turn = inspector.studio_view(rig, side)
        per_frame, joints = inspector.measure_frames(rig, first, last, side, side_normal, behind_normal, turn)
        jump_set = inspector.jump_frames(per_frame, targets, joints)
        motion = {'jumps': inspector.jump_sizes(joints, jump_set), 'steps': inspector.jump_steps(joints)}
        result.update(motion)
        comparisons = source_camera_plan(snapshot, measurements, targets, phases, yaw,
                                         inspector.camera_for, inspector.effective_view)
        by_frame, points_at = {}, {}
        for camera in comparisons:
            frame, azimuth, elevation = camera['frame'], camera['azimuth'], camera['elevation']
            if frame not in points_at:
                points_at[frame] = inspector.figure_points(skin, frame)
            points = points_at[frame]
            low, high = points.min(axis=0), points.max(axis=0)
            angle = math.radians(azimuth)
            across = np.array([math.cos(angle), math.sin(angle), 0.])
            extent = max(float(high[2] - low[2]), float(np.ptp(points @ across))) * inspector.FRAME_MARGIN
            centre = (low + high) / 2
            position = centre + inspector.view_direction(azimuth, elevation) * cage.distance_for(extent)
            camera['location'] = position.tolist()
            by_frame.setdefault(frame, []).append(position)
        result.update(comparison_cameras=comparisons, comparison_camera_frames=sorted(by_frame))
        measured = inspector.skin_folds(rig, skin, first, last,
                                       lambda frame: positions + by_frame.get(frame, []))
        if not isinstance(measured, dict):
            raise ValueError('The animated skin topology cannot be measured against its rest pose.')
        result.update(height_m=measured.get('height_m'), folds=measured.get('folds'), measured_frames=last - first + 1)
    except Exception as error:
        result['measurement_error'] = f'{type(error).__name__}: {error}'
        measured = None
    finally:
        scene.frame_set(original_frame, subframe=original_subframe)
        scene.camera = original_camera
    rows, ready = fold_verdict(measured, first, last, error=result['measurement_error'])
    # A later fold/camera failure must not erase a completed movement reading.
    motion_rows, motion_ready = motion_verdict(motion, error=result['measurement_error'] if motion is None else None)
    result.update(fold_rows=rows, motion_rows=motion_rows, rows=rows + motion_rows,
                  fold_ready=ready, motion_ready=motion_ready, ready=ready and motion_ready)
    return result
