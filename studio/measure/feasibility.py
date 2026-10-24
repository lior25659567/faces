"""What the coach's arm targets allow on THIS character's arms (pure geometry).

Faithful port of the squat report's arm search: bar on the back, so seen from behind the forearm is
vertical, the wrist is level with the shoulder (+1 cm), the torso is upright. The shoulder joint sits
at +/- shoulder_half_width. We scan upper-arm directions and keep the pose with the lowest cost,
where cost = squared misses in tolerance units (5 deg, 0.12 of a shoulder width, 2 cm behind the bar line).
"""
import numpy as np

ANGLE_TOL = 5.0
RATIO_TOL = 0.12
WRIST_DEPTH_TOL = 0.02   # metres: wrist 3 cm behind the shoulder, the bar-on-back line
WRIST_DEPTH = -0.03
WRIST_ABOVE_SHOULDER = 0.01


def _angle(a, b):
    cos = (a * b).sum(-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def closest_arm_pose(arm, want):
    xs, l1, l2 = arm["shoulder_half_width"], arm["upper_arm"], arm["forearm"]
    sw = 2 * xs
    shoulder = np.array([xs, 0.0, 0.0])
    th, ph = np.meshgrid(np.radians(np.arange(0, 181, 2)), np.radians(np.arange(-90, 91, 3)), indexing="ij")
    # upper arm direction: polar angle from "down", azimuth 0 lateral, +90 back
    d = np.stack([np.sin(th) * np.cos(ph), -np.sin(th) * np.sin(ph), -np.cos(th)], -1)
    elbow = shoulder + l1 * d
    fz = shoulder[2] + WRIST_ABOVE_SHOULDER - elbow[..., 2]
    ok = np.abs(fz) <= l2
    fy = np.sqrt(np.clip(l2 ** 2 - fz ** 2, 0, None))
    best = None
    for sgn in (-1, 1):
        wrist = elbow + np.stack([np.zeros_like(fz), sgn * fy, fz], -1)
        upper = _angle(elbow - shoulder, np.array([0, 0, -1.0]))
        incl = _angle(shoulder - elbow, wrist - elbow)
        grip, ew = 2 * wrist[..., 0] / sw, 2 * elbow[..., 0] / sw
        cost = (((upper - want["upperarm_vs_torso"]) / ANGLE_TOL) ** 2
                + ((incl - want["elbow_included"]) / ANGLE_TOL) ** 2
                + ((grip - want["grip_ratio"]) / RATIO_TOL) ** 2
                + ((ew - want["elbow_width_ratio"]) / RATIO_TOL) ** 2
                + ((wrist[..., 1] - shoulder[1] - WRIST_DEPTH) / WRIST_DEPTH_TOL) ** 2)
        cost = np.where(ok, cost, np.inf)
        i = np.unravel_index(np.argmin(cost), cost.shape)
        if best is None or cost[i] < best[0]:
            best = (cost[i], upper[i], incl[i], grip[i], ew[i])
    _, upper, incl, grip, ew = best
    return {"upperarm_vs_torso": float(upper), "elbow_included": float(incl),
            "grip_ratio": float(grip), "elbow_width_ratio": float(ew)}
