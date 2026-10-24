"""studio/measure/kit.py over studio/measure/exercise-kit.json: what each exercise holds and what holds it up, what
the animation path builds of it, and the plain refusals of the picture path (task 9 review). The same JSON is read
by studio/server/store.js (its tests pin the same sentences)."""
import json

import pytest

from conftest import ROOT
import kit

POSES = json.loads((ROOT / "reference" / "exercises" / "poses.json").read_text())
LUNGE = ("No motion for this exercise: its feet step, which the picture path can't animate yet. "
         "Add lunge.fbx to 3d-character/mixamo (its README says how).")


def test_the_table_has_the_14_exercises_and_nothing_else():
    assert set(kit.load()["exercises"]) == {e["name"] for e in POSES["exercises"]}


def test_the_lunge_is_refused_because_its_feet_step():
    assert kit.picture_refusal("Walking Lunge (dumbbells optional)") == LUNGE


@pytest.mark.parametrize("name,support", [
    ("Flat Barbell Bench Press", "bench"),
    ("Leg Press (seated machine)", "leg-press sled"),
    ("Pull-Up", "pull-up bar")])
def test_a_support_the_path_does_not_build_is_refused(name, support):
    assert kit.picture_refusal(name) == (f"No motion for this exercise: it needs a {support} the picture path "
                                         "doesn't build yet.")


@pytest.mark.parametrize("name", ["Barbell Bent-Over Row", "Barbell Back Squat", "Conventional Barbell Deadlift", "Front Plank (forearm plank)",
                                  "Dumbbell Lateral Raise", "Tricep Pushdown (cable, bar or rope attachment)"])
def test_standing_or_lying_exercises_move(name):
    assert kit.picture_refusal(name) is None


@pytest.mark.parametrize("name,thing", [
    ("Standing Barbell Overhead Press (Shoulder Press)", "bar"),
    ("Tricep Pushdown (cable, bar or rope attachment)", "cable handle")])
def test_equipment_held_but_not_built_is_named(name, thing):
    assert kit.unheld(name) == thing


@pytest.mark.parametrize("name", ["Barbell Bent-Over Row", "Barbell Back Squat", "Front Plank (forearm plank)",
                                  "Walking Lunge (dumbbells optional)"])
def test_nothing_unheld_when_the_bar_is_built_or_nothing_is_held(name):
    # the back squat's bar is built on the --mpfb path (animate.MPFB_BARBELL_ON_BACK); the plank holds nothing; the
    # lunge's dumbbells are optional
    assert kit.unheld(name) is None


@pytest.mark.parametrize('name', ['Conventional Barbell Deadlift', 'Dumbbell Bicep Curl (standing, both arms)', 'Dumbbell Lateral Raise',
    'Dumbbell Shoulder Press (seated overhead press)', 'Lat Pulldown (seated, to the front)'])
def test_new_profiles_build_their_equipment_and_required_support(name):
    assert kit.unheld(name) is None
    assert kit.picture_refusal(name) is None


def test_an_exercise_of_our_own_goes_by_its_equipment():
    # the Studio's "Add exercise" equipment: the name is not in the table
    assert kit.unheld("Back Squat from images", "barbell") == "bar"
    assert kit.picture_refusal("Back Squat from images", "barbell") is None
    assert kit.unheld("Hammer Curl", "dumbbells") == "dumbbells"
    assert kit.picture_refusal("Incline Press", "bench") == ("No motion for this exercise: it needs a bench the "
                                                            "picture path doesn't build yet.")
    assert kit.picture_refusal("Hack Squat", "machine") == ("No motion for this exercise: it needs a machine the "
                                                           "picture path doesn't build yet.")
    for word in ("bodyweight", "other", None):
        assert kit.unheld("Burpee", word) is None and kit.picture_refusal("Burpee", word) is None


def test_the_table_s_name_wins_over_an_equipment_word():
    assert kit.unheld("Barbell Back Squat", "barbell") is None


def test_only_the_squat_has_its_depth_judged():
    """Task 22: the depth rows (the seat against the top of the knee, the MediaPipe hip -> knee line) are read and
    shown only for an exercise whose row says `depth`: the back squat. Not the deadlift's or the press's end
    position, and not an exercise added in the Studio, whatever its equipment."""
    assert kit.has_depth("Barbell Back Squat") is True
    assert [e["name"] for e in POSES["exercises"] if kit.has_depth(e["name"])] == ["Barbell Back Squat"]
    assert kit.has_depth("My Own Squat", "barbell") is False and kit.has_depth("My Own Squat") is False
    assert kit.kit_of("Barbell Back Squat")["depth"] is True
