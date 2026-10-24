"""Numpy-only finite weight-head contact geometry."""
import numpy as np


def hex_head_depth(points, centre_x, radius, thickness):
    """Positive penetration into a six-sided prism whose cylinder axis is X.

    Blender's six cylinder vertices rotate from its Z axis onto X. The six face
    normals are midway between those vertices. Points beyond the finite end caps
    cannot count as contact, regardless of their distance from the infinite axis.
    """
    points = np.asarray(points, dtype=float)
    angles = (np.arange(6) + .5) * np.pi / 3
    normals = np.column_stack([np.sin(angles), -np.cos(angles)])
    sides = radius * np.cos(np.pi / 6) - (points[:, 1:] @ normals.T).max(axis=1)
    ends = thickness / 2 - abs(points[:, 0] - centre_x)
    return np.maximum(0., np.minimum(sides, ends))
