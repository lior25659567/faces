"""Measure the whole heel/toe support patches of a bench press, without Blender.

A nearest-point sole test can pass with only the toe touching. These patches
are selected once on the fitted rest mesh and then followed through the rep.
"""
import numpy as np


def plantar_patches(rest, feet):
    rest = np.asarray(rest, dtype=float)
    if rest.ndim != 2 or rest.shape[1] != 3 or not np.isfinite(rest).all():
        raise ValueError('Bench feet require a finite fitted rest mesh.')
    patches = {}
    for side in ('left', 'right'):
        mask = np.asarray(feet.get(side), dtype=bool)
        if mask.shape != (len(rest),) or not mask.any():
            raise ValueError('Both feet need measurable sole surfaces.')
        sole = mask & (rest[:, 2] <= rest[mask, 2].min() + .002)
        low, high = rest[sole, 1].min(), rest[sole, 1].max()
        if high - low <= 1e-6:
            raise ValueError('A bench foot needs distinct heel and toe patches.')
        for region, selected in (('heel', rest[:, 1] <= low + (high - low) * .3),
                                 ('toe', rest[:, 1] >= high - (high - low) * .3)):
            patch = sole & selected
            if patch.sum() < 3:
                raise ValueError('A bench heel or toe has too few measurable vertices.')
            patches[f'foot_{side}_{region}'] = patch
    return patches


def floor_report(rest, feet, frames):
    patches = plantar_patches(rest, feet)
    values, starts = {name: [] for name in patches}, {}
    slides = []
    frame_ids = []
    for frame, points in frames:
        points = np.asarray(points, dtype=float)
        if points.shape != np.asarray(rest).shape or not np.isfinite(points).all():
            raise ValueError('Every bench frame needs finite evaluated sole geometry.')
        if type(frame) is not int or (frame_ids and frame != frame_ids[-1] + 1):
            raise ValueError('Bench floor measurements require every consecutive frame.')
        frame_ids.append(frame)
        for name, selected in patches.items():
            sample = points[selected]
            starts.setdefault(name, sample.copy())
            values[name].append((float(abs(sample[:, 2]).max()), max(0., -float(sample[:, 2].min()))))
            slides.append(float(np.linalg.norm(sample - starts[name], axis=1).max()))
    if not frame_ids:
        raise ValueError('Bench floor measurements require the complete movement.')
    count = len(frame_ids)
    return {'surfaces': {name: {'frames': count, 'measured_frames': count,
                'max_gap_mm': max(row[0] for row in rows) * 1000,
                'max_penetration_mm': max(row[1] for row in rows) * 1000}
            for name, rows in values.items()},
            'planted_feet': {'frames': count, 'measured_frames': count,
                'first_frame': frame_ids[0], 'last_frame': frame_ids[-1],
                'max_slide_mm': max(slides) * 1000,
                'method': 'fitted rest heel/toe plantar patches, full evaluated surface on every frame'}}
