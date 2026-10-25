"""Recorded key times are judged directly, including invalid/missing/jumping keys."""
from copy import deepcopy
import pytest
from reference_timing import recorded_phase_frames


def metadata(**changes):
    return {'source': 'keyposes', 'mode': 'cycle', 'phase_frames': {'start': 1, 'middle': 23, 'end': 46}, **changes}


def test_actual_source_times_win_over_a_better_angle_match():
    frames = [{'frame': n, '3d': {'elbow_included': 170 if n == 13 else 160}} for n in range(1, 82)]
    record = metadata()
    before = deepcopy(record)
    assert recorded_phase_frames(frames, record) == {'start': 1, 'middle': 23, 'end': 46}
    assert record == before


def test_captured_clips_keep_the_existing_selection_path():
    assert recorded_phase_frames([{'frame': 1}], None) is None
    assert recorded_phase_frames([{'frame': 1}], {'source': 'mixamo'}) is None


@pytest.mark.parametrize('bad', [None, 0, 82, 1.5, True, '1'])
def test_invalid_generated_time_cannot_fall_back_to_a_fitting_pose(bad):
    record = metadata(phase_frames={'start': bad, 'middle': 23, 'end': 46})
    with pytest.raises(ValueError, match='missing or outside'):
        recorded_phase_frames([{'frame': n} for n in range(1, 82)], record)


def test_jumping_source_key_is_an_error_instead_of_selecting_another_frame():
    with pytest.raises(ValueError, match='jumps at frame 1'):
        recorded_phase_frames([{'frame': n} for n in range(1, 82)], metadata(), skip={1})


def test_hold_uses_one_real_pose_without_fabricating_motion():
    record = metadata(mode='hold', phase_frames={'start': 1, 'middle': 1, 'end': 1})
    assert recorded_phase_frames([{'frame': n} for n in range(1, 13)], record) == record['phase_frames']
    record['phase_frames']['end'] = 12
    with pytest.raises(ValueError, match='static hold'):
        recorded_phase_frames([{'frame': n} for n in range(1, 13)], record)


def test_reversed_source_timing_does_not_silently_reverse_the_exercise():
    with pytest.raises(ValueError, match='movement order'):
        recorded_phase_frames([{'frame': n} for n in range(1, 82)], metadata(phase_frames={'start': 46, 'middle': 23, 'end': 1}))
