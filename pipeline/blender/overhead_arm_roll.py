"""Authored anatomical upper-arm roll for the fitted seated press."""
from bench_arm_roll import orient_upperarms as orient


def orient_upperarms(rig):
    # This is the same clavicle-rest construction used for the fitted bench,
    # preserving actual elbow/wrist/hand transforms. Photo twist is unknown.
    return orient(rig, 30.)
