"""Declared upperarm orientation for the front pulldown's solved joint path.

Photo-derived joint positions do not determine humerus axial twist. Preserve
the solved elbow, wrist and grip while using one mirrored anatomical frame.
"""
from bench_arm_roll import orient_upperarms as _orient_upperarms


def orient_upperarms(rig):
    report = _orient_upperarms(rig, roll_degrees=-15.)
    report['method'] = ('fixed mirrored humerus roll from the fitted clavicle rest frame '
                        'for a front pulldown; solved elbow, wrist and grip retained')
    return report
