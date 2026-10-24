from pytest import approx

from feasibility import closest_arm_pose

CARTOON = {"shoulder_half_width": 0.117, "upper_arm": 0.105, "forearm": 0.124}
HUMAN = {"shoulder_half_width": 0.19, "upper_arm": 0.30, "forearm": 0.27}
WANT = {"upperarm_vs_torso": 52, "elbow_included": 82, "grip_ratio": 1.87, "elbow_width_ratio": 1.98}

def test_port_is_pinned_to_the_prototype():
    # armfeas.py: shoulder x 0.117 bottom -> upperarm 60.0, elbow 73.1, grip 1.78
    r = closest_arm_pose(CARTOON, WANT)
    assert r["upperarm_vs_torso"] == approx(60.0, abs=0.1)
    assert r["elbow_included"] == approx(73.1, abs=0.1)
    assert r["grip_ratio"] == approx(1.776, abs=0.01)
    assert r["elbow_width_ratio"] == approx(1.776, abs=0.01)
    start = {"upperarm_vs_torso": 41.3, "elbow_included": 90.5, "grip_ratio": 1.866, "elbow_width_ratio": 1.974}
    assert closest_arm_pose(CARTOON, start)["grip_ratio"] == approx(1.717, abs=0.01)


def test_human_arm_with_these_targets_is_pinned():
    # a human arm also misses the coach's 1.87 with all four cues together
    assert closest_arm_pose(HUMAN, WANT)["grip_ratio"] == approx(2.166, abs=0.01)


def test_cartoon_cannot_reach_the_coaches_elbow_width():
    assert closest_arm_pose(CARTOON, WANT)["elbow_width_ratio"] < 1.98 - 0.12
