"""Instance-scoped prohibitions: "turn at the cone-marked intersection, but not the 2nd".

WHAT HAD NO REPRESENTATION. `cue_ordinal` has always counted instances for ADVANCEMENT
("the 2nd bench") while nothing counted them for PROHIBITION, so an instance-scoped ban
could only be written by over-generalising it: the plan says `forbid_modes: ['junction']`
and the formula `G(not Phi_Junc)`, each forbidding every junction on a route the same
mission says to traverse. Every test here pairs a positive with a case that must NOT fire,
because a veto that triggers on everything is worse than no veto -- it would strand any
mission that has to pass through the forbidden instance.
"""
import pytest
from brain.trigger_policy import InstanceForbidTracker, branch_turns


def _turn(dirn):
    return {"vlm_cue": "traffic cone is present",
            "sub_plan": [{"transition_cue": f"Bearing({dirn}) completed"}]}


STRAIGHT = {"vlm_cue": "default",
            "sub_plan": [{"transition_cue": "Bearing(Straight) completed"}]}


def test_second_instance_is_the_one_forbidden():
    t = InstanceForbidTracker([{"mode": "junction", "ordinal": 2}])
    for c, l in [(0, "path"), (63, "junction"), (1, "path"),
                 (60, "junction"), (2, "path"), (59, "junction")]:
        t.observe(c, l)
    assert t.forbidden == {60}, "must forbid the 2nd junction only"


def test_reentering_the_same_region_is_not_a_second_instance():
    """Vehicles clip a junction boundary and the region stream reads 63 -> 63; counting
    that as two arrivals would forbid the FIRST junction instead of the second."""
    t = InstanceForbidTracker([{"mode": "junction", "ordinal": 2}])
    for c, l in [(63, "junction"), (63, "junction"), (63, "junction"), (1, "path")]:
        t.observe(c, l)
    assert t.forbidden == set(), "no second junction has been entered yet"


def test_a_mode_that_is_not_named_is_never_forbidden():
    t = InstanceForbidTracker([{"mode": "junction", "ordinal": 2}])
    for c, l in [(0, "path"), (1, "path"), (2, "path"), (3, "path")]:
        t.observe(c, l)
    assert t.forbidden == set()


@pytest.mark.parametrize("d", ["Right", "Left"])
def test_turning_branches_are_detected(d):
    assert branch_turns(_turn(d)) is True


def test_straight_is_not_a_turn():
    """The prohibition is on TURNING at the Nth instance. Passing through must stay legal,
    or a route that continues past the forbidden junction can never be driven."""
    assert branch_turns(STRAIGHT) is False


def test_branch_with_no_sub_plan_is_not_a_turn():
    assert branch_turns({"vlm_cue": "default"}) is False


def test_the_pair_that_makes_the_mission_unfakeable():
    """Same instruction, cone at the 2nd vs the 3rd junction, opposite required behaviour.

    A policy that always turns on a cone is correct on the world where the cone sits at a
    permitted junction while being wrong about the constraint entirely. Only the pair
    separates "followed the prohibition" from "turned when it saw a cone".
    """
    route = [(0, "path"), (63, "junction"), (1, "path"), (60, "junction"),
             (2, "path"), (59, "junction")]
    a = InstanceForbidTracker([{"mode": "junction", "ordinal": 2}])
    for c, l in route:
        a.observe(c, l)
    # cone at J60 (the 2nd): a turning branch there is vetoed
    assert 60 in a.forbidden and branch_turns(_turn("Right"))
    # cone at J59 (the 3rd): nothing forbids it, the same branch must fire
    assert 59 not in a.forbidden
