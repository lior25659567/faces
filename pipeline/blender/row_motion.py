"""Keep the body hinge stable in a reference-driven bent-over row."""
def stabilize_body(keys):
    """Hold the starting hinge and lower body while allowing the row's arms to move.

    Matrices are absolute turns, so holding the trunk alone would leave the head or
    thighs drifting. Preserve every non-arm turn, including neck and feet.
    """
    if not keys:
        return []
    def arm(name):
        return name.startswith(('LeftArm', 'RightArm', 'LeftForeArm', 'RightForeArm',
                                'LeftHand', 'RightHand', 'LeftShoulder', 'RightShoulder'))
    return [{name: value if arm(name) else keys[0][name].copy() for name, value in key.items()}
            for key in keys]
