"""The cue token -> VLM question step."""
from brain.cue_phrasing import cue_question, is_perceptual


def test_formula_tokens_become_noun_phrases():
    """`Detect(X)` asked verbatim makes the model parse our notation before looking."""
    assert cue_question("Detect(TrafficCone)") == "a traffic cone"
    assert cue_question("Detect(StopSign)") == "a stop sign"
    assert cue_question("Detect(Intersection)") == "an intersection"      # article agrees
    assert cue_question("Detect(RedBrickBuildingWithRedDoors)") == \
        "a red brick building with red doors"


def test_spatial_suffix_is_spelled_out_not_left_in_camelcase():
    """The half the mission turns on.

    "turn left if the bus stand is on your left" generates `Detect(BusStandLeft)`, and
    inside CamelCase the LEFT is the part a model is most likely to skip. Split out, it
    becomes a clause the question actually asks about.
    """
    assert cue_question("Detect(BusStandLeft)") == "a bus stand on your left"
    assert cue_question("Detect(ConeOnRight)") == "a cone on your right"
    assert cue_question("Detect(BridgeAbove)") == "a bridge above you"
    assert cue_question("Detect(ParkingLotBothSides)") == "a parking lot on both sides of you"


def test_plain_english_is_left_alone():
    """The generator often writes a good phrase already; rewriting it is a second chance
    to get it wrong."""
    assert cue_question("the gate is shut") == "the gate is shut"
    assert cue_question("a traffic cone is in the intersection") == \
        "a traffic cone is in the intersection"


def test_maneuver_macros_are_not_perceptual():
    """`Bearing(Left) completed` says what the robot DID, not what is in frame.

    Asking a camera to confirm it is a category error that answers NO forever, which
    presents as a step that times out rather than fails -- the failure mode that reads as
    a planner bug.
    """
    assert not is_perceptual("Bearing(Left) completed")
    assert not is_perceptual("traverse")
    assert is_perceptual("Detect(TrafficCone)")
    assert is_perceptual("the gate is shut")


def test_empty_and_malformed_do_not_crash():
    assert cue_question("") == ""
    assert cue_question(None) == ""
    assert cue_question("Detect()") == "Detect()"
