"""Cue grounding: two silent execution failure modes, as regressions.

Both pass every structural validator, the STL syntax gate, AND the offline simulator,
and only fail on the robot. That is the signature of a vocabulary the validators do not
cover.
"""
import os

import pytest

from nl_planner.cue_vocab import (CARLA_GT_VOCAB as V, check_cue,
                                  families_matching, validate_plan_cues)

WS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def test_unanswerable_cue_is_rejected():
    """Bug 1. The generator wrote a cue no publisher answers.

    `Detect(Junction)` with gt_cue_node publishing only `Detect(Intersection)` /
    `intersection`: step 0 never completes and the cue times out with no sightings while
    the vehicle sits in the junction it was asked to detect. It does not fail, it TIMES
    OUT, which is why nothing upstream notices.
    """
    # DERIVED, NOT HARDCODED. A hardcoded example (e.g. Detect(Bench), Detect(Fountain))
    # goes stale when that referent becomes a real family, and the test then keeps
    # passing while asserting nothing.
    #
    # So build a cue that cannot collide with the vocabulary by construction, and assert
    # that it really does not -- the same discipline the repo applies to spawn-pose
    # defaults: a constant that goes stale when the vocabulary moves is worse than none.
    unanswerable = "Detect(Zzyzx)"
    assert not families_matching(unanswerable, V), (
        f"{unanswerable} now matches {families_matching(unanswerable, V)} -- pick "
        f"another token; the point is a cue NOTHING answers")
    ok, why = check_cue(unanswerable, V)
    assert not ok and "no answerable predicate" in why
    # Fountain is a REAL family (spawnable prop).
    assert check_cue("Detect(Fountain)", V)[0]
    # and the known-problematic cues are answerable
    for fixed in ("Detect(Bench)", "way ahead is blocked", "Detect(StopSign)"):
        assert check_cue(fixed, V)[0], f"{fixed} should be answerable"


def test_ambiguous_cue_is_rejected():
    """Bug 2, and the more dangerous one because it does not stall -- it acts.

    `traffic cone is present in the intersection` matches BOTH the cone and the junction
    predicates. brain takes one, gets the junction answer (true at every junction), and the
    branch fires unconditionally -- the robot takes option 1 whether or not the cone is
    there.
    """
    for cue in ("traffic cone is present in the intersection",
                "Detect(TrafficCone) inside intersection"):
        ok, why = check_cue(cue, V)
        assert not ok, cue
        assert "matches 2 predicates" in why


@pytest.mark.parametrize("cue", ["Detect(TrafficCone)", "cone", "Detect(Junction)",
                                 "intersection"])
def test_a_cue_naming_one_predicate_passes(cue):
    assert check_cue(cue, V)[0], cue


def test_trajectory_cues_need_no_vocabulary():
    """`Bearing(Right)` is answered by the heading change, not by a publisher."""
    assert check_cue("Bearing(Right) completed", V)[0]


def test_a_vlm_deployment_is_not_constrained():
    """cue_source=vlm answers free text; inventing a vocabulary there would be wrong."""
    assert check_cue("stop where you can see the bench and the fountain", None)[0]


def test_it_catches_the_ambiguity_in_a_BRANCH_cue():
    """Where it actually bites: a step cue advances early, a branch cue goes the wrong way."""
    plan = {"steps": [{"step": 0, "transition_cue": None, "branches": [
        {"vlm_cue": "traffic cone is present in the intersection", "sub_plan": []},
        {"vlm_cue": "default", "sub_plan": []}]}]}
    bad = validate_plan_cues(plan, V)
    assert len(bad) == 1 and "branch" in bad[0]


def test_the_vocabulary_does_not_drift_from_the_runtime_that_answers_it():
    """This file being wrong the SAME way as the runtime would hide the bug it exists for.

    Now compares against `carla_gt_bridge.cue_answers`, which is the single
    implementation both the ROS node and the offline twin delegate to.

    Also asserts the ORDER, because order is semantics here: brain takes the first
    matching key, so object keys must precede place keys or a cone cue is answered by
    the junction predicate.
    """
    import sys
    sys.path.insert(0, os.path.join(WS, "carla_gt_bridge"))
    from carla_gt_bridge.cue_answers import answer_keys

    # Ask the runtime for EVERY family it can answer, not just the ones a particular
    # world happens to populate. answer_keys omits a family whose boolean is None -- that
    # is deliberate (an unanswerable cue must time out, not read as a confident False) --
    # so comparing against a bare call would let a family drift in unnoticed the moment
    # it is not supplied.
    from carla_gt_bridge.cue_answers import LANDMARK_SPELLINGS, LANDMARK_ALIAS
    every = {LANDMARK_ALIAS.get(f, f): True for f in LANDMARK_SPELLINGS}
    runtime = list(answer_keys(at_junction=True, cone=True, landmarks=every))
    mine = [k for keys in V.values() for k in keys]
    assert set(mine) == set(runtime), (
        f"vocab drift: only here {sorted(set(mine) - set(runtime))}, "
        f"only in the runtime {sorted(set(runtime) - set(mine))}")
    last_object = max(runtime.index(k) for k in V["cone"])
    first_place = min(runtime.index(k) for k in V["junction"])
    assert last_object < first_place, (
        "the runtime must answer object keys BEFORE place keys; brain takes the first "
        "match, so a cone cue would otherwise be answered by the junction predicate")


def test_the_vocabulary_gate_follows_the_sensor_not_the_language():
    """Open set under the VLM, closed set under the ground-truth publisher.

    `check_cue`'s docstring scopes this gate to the deployment -- "under
    cue_source=vlm a model answers free text and nothing here applies, pass vocab=None".
    Reading CARLA_GT_VOCAB unconditionally would, with a VLM answering, reject a cue it
    can obviously handle and burn a retry, enforcing a constraint the runtime does not
    have.

    The closed set stays correct for cue_source=topic: there the publisher's keys ARE
    the answerable set and a cue outside it times out silently.
    """
    for cue in ("Detect(Gnome)", "a red fire hydrant", "Detect(Zzyzx)"):
        assert not check_cue(cue, V)[0], f"{cue} should fail the closed (topic) vocabulary"
        assert check_cue(cue, None)[0], f"{cue} should PASS open-set (vlm)"


def test_stl_ablation_selects_the_vocabulary_from_cue_source():
    """The wiring, not just the helper -- a gate that is silently always-on is the bug.

    Asserted on the source because the alternative is running a full generation with an
    API key. The pairing that matters is `CUE_SOURCE` -> `_vocab` -> `validate_plan_cues`.
    """
    src = open(os.path.join(WS, "nl_planner", "scripts", "stl_ablation.py")).read()
    assert 'os.environ.get("CUE_SOURCE"' in src, \
        "stl_ablation does not consult CUE_SOURCE -- the gate is unconditional again"
    assert "_vocab = None if _cue_source == \"vlm\" else CARLA_GT_VOCAB" in src
    assert "validate_plan_cues(plan, _vocab)" in src, \
        "the gate still passes CARLA_GT_VOCAB directly, ignoring the sensor"
