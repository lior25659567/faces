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
    'Walking Lunge (dumbbells optional)': ('stepping', 'free'),
}


def profile_for(exercise):
    row = PROFILES.get(exercise)
    return {'exercise': exercise, 'family': row[0], 'feet': row[1],
            'mode': 'hold' if row[0] == 'hold' else 'sequence' if row[0] == 'stepping' else 'cycle',
            'holds': row[0] not in ('hold', 'sled'),
            'overhead': exercise in ('Standing Barbell Overhead Press (Shoulder Press)', 'Dumbbell Shoulder Press (seated overhead press)',
                                     'Lat Pulldown (seated, to the front)')} if row else None


def prepare_key_turns(exercise, keys):
    """Hold the starting lower body on an isolation exercise; keep measured arm motion.

    Hinge/sled motion changes the legs and is never held. Supine/seated/hanging
    anchoring is performed against actual body/support geometry after retargeting.
    """
    profile = profile_for(exercise)
    if not keys or not profile:
        return keys
    family = profile['family']
    if family == 'stepping':
        # A frontal depth lift does not establish pelvic roll or axial trunk
        # twist. The observed step keeps an upright carrying posture. Preserve
        # the chronological asymmetric leg/arm directions, never mirror them.
        neutral = {'Hips', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head', 'LeftShoulder', 'RightShoulder'}
        return [{name: np.eye(3) if name in neutral else np.array(value, copy=True)
                 for name, value in key.items()} for key in keys]
    from isolation_poses import EXERCISES, prepare
    if exercise in EXERCISES:
        return prepare(exercise, keys)
    if exercise == 'Standing Barbell Overhead Press (Shoulder Press)':
        from standing_posture import upright
        keys = upright(keys)
    if exercise == 'Dumbbell Shoulder Press (seated overhead press)':
        # Keep the observed seated legs, but do not turn noisy photo depth into
        # a deeply reclined torso for an upright shoulder press. Support staging
        # fits the actual seat/back/feet after this declared neutral body pose.
        upper_body = {'Hips', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head', 'LeftShoulder', 'RightShoulder'}
        keys = [{name: np.eye(3) if name in upper_body else np.array(value, copy=True)
                 for name, value in key.items()} for key in keys]
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
    if family not in ('standing', 'seated', 'supine', 'sled', 'hanging'):
        return keys
    # A strict pull-up carries its starting body/leg pose upward. Uncalibrated
    # depth must not add a head nod, torso sway or changing crossed-leg pose.
    # Clavicles still share the measured arm elevation below.
    fixed = {'Hips', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head'} if family == 'hanging' else {'Hips', 'Spine', 'Spine1', 'Spine2', 'LeftShoulder', 'RightShoulder'}
    if family != 'sled':
        fixed.update(side + name for side in ('Left', 'Right')
                     for name in ('UpLeg', 'Leg', 'Foot', 'ToeBase'))
    poses = [{name: np.array(keys[0][name] if name in fixed else value, copy=True)
              for name, value in key.items()} for key in keys]
    if profile.get('overhead') or family == 'hanging':
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
