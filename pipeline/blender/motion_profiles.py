"""Reference motion policies. A static hold is never converted into a repetition.

This module is numpy-only so Studio can validate the policy before loading Blender.
The photographs still provide the joint directions; policies only preserve regions
that should remain fixed for that movement.
"""
import numpy as np

PROFILES = {
    'Conventional Barbell Deadlift': ('hinge', 'planted'),
    'Flat Barbell Bench Press': ('supine', 'free'),
    'Standing Barbell Overhead Press (Shoulder Press)': ('standing', 'planted'),
    'Pull-Up': ('hanging', 'free'),
    'Lat Pulldown (seated, to the front)': ('seated', 'free'),
    'Dumbbell Bicep Curl (standing, both arms)': ('standing', 'planted'),
    'Dumbbell Lateral Raise': ('standing', 'planted'),
    'Dumbbell Shoulder Press (seated overhead press)': ('seated', 'free'),
    'Front Plank (forearm plank)': ('hold', 'free'),
    'Tricep Pushdown (cable, bar or rope attachment)': ('standing', 'planted'),
    'Leg Press (seated machine)': ('sled', 'free'),
}


def profile_for(exercise):
    row = PROFILES.get(exercise)
    return {'family': row[0], 'feet': row[1], 'mode': 'hold' if row[0] == 'hold' else 'cycle',
            'holds': row[0] not in ('hold', 'sled'),
            'overhead': exercise in ('Standing Barbell Overhead Press (Shoulder Press)', 'Dumbbell Shoulder Press (seated overhead press)')} if row else None


def prepare_key_turns(exercise, keys):
    """Hold the starting lower body on an isolation exercise; keep measured arm motion.

    Hinge/sled motion changes the legs and is never held. Supine/seated/hanging
    anchoring is performed against actual body/support geometry after retargeting.
    """
    profile = profile_for(exercise)
    if not keys or not profile:
        return keys
    family = profile['family']
    if family == 'hold':
        return [{name: np.array(value, copy=True) for name, value in keys[0].items()}]
    if family == 'hinge':
        # Keep a neutral neck through the hinge. Monocular facial depth should
        # not counter-rotate the head against the trunk.
        poses = [{name: np.array(value, copy=True) for name, value in key.items()} for key in keys]
        for pose in poses:
            for name in ('Neck', 'Head'):
                if name in pose:
                    pose[name] = pose['Spine2'].copy()
        return poses
    if family not in ('standing', 'seated', 'supine', 'sled'):
        return keys
    fixed = {'Hips', 'Spine', 'Spine1', 'Spine2'}
    if family != 'sled':
        fixed.update(side + name for side in ('Left', 'Right')
                     for name in ('UpLeg', 'Leg', 'Foot', 'ToeBase'))
    poses = [{name: np.array(keys[0][name] if name in fixed else value, copy=True)
              for name, value in key.items()} for key in keys]
    if exercise == 'Lat Pulldown (seated, to the front)':
        # Share high-arm elevation with the clavicle instead of crushing the
        # whole turn into the shoulder socket. Keep each arm's measured WORLD
        # direction: the local basis compensates in BoneRetarget.
        for pose in poses:
            chest = pose['Spine2']
            for side, sign in (('Left', 1), ('Right', -1)):
                direction = chest.T @ pose[side + 'Arm'] @ np.array([sign, 0., 0.])
                elevation = np.degrees(np.arcsin(np.clip(direction[2], -1., 1.)))
                angle = np.radians(-sign * np.clip((elevation - 30.) / 3., 0., 20.))
                c, s = np.cos(angle), np.sin(angle)
                turn = np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
                pose[side + 'Shoulder'] = chest @ turn
    return poses
