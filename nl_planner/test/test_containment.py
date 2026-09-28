"""Tests for L(T) subseteq L(phi).

THE TEST THAT MATTERS IS THE NEGATIVE ONE. A containment check that returns True for
everything is worse than no check, because it launders an unverified claim. Every positive
case here is paired with a mutation that must be REJECTED.
"""
from __future__ import annotations

import copy

from nl_planner.containment import check_tree, leaf_paths, word_of


def _linear(modes, cues=None):
    """Branch-free tree: modes[i] is step i's goal, cues[i] its transition_cue."""
    cues = cues or [None] * len(modes)
    prev = "path"
    steps = []
    for i, (m, c) in enumerate(zip(modes, cues)):
        steps.append({"step": i, "start_mode": prev, "goal_mode": m,
                      "transition_cue": c, "description": f"s{i}"})
        prev = m
    return {"steps": steps}


def _branching():
    """path -> junction, then decide: cone => left, otherwise => right."""
    return {"steps": [
        {"step": 0, "start_mode": "path", "goal_mode": "junction",
         "transition_cue": None, "description": "reach the junction",
         "branches": [
             {"vlm_cue": "Detect(Cone) is visible", "sub_plan": [
                 {"step": 0, "start_mode": "junction", "goal_mode": "path",
                  "transition_cue": "Bearing(Left) completed", "description": "left"}]},
             {"vlm_cue": "default", "sub_plan": [
                 {"step": 0, "start_mode": "junction", "goal_mode": "path",
                  "transition_cue": "Bearing(Right) completed", "description": "right"}]},
         ]},
    ]}


# --------------------------------------------------------------------------- #
# word construction                                                            #
# --------------------------------------------------------------------------- #

def test_leaf_paths_enumerates_one_path_per_branch():
    assert len(leaf_paths(_branching())) == 2
    assert len(leaf_paths(_linear(["junction", "path"]))) == 1


def test_branch_cue_reaches_the_word():
    """The decision atom lives on the branch, not the step. If it does not reach the word,
    every decision formula is unsatisfiable and the check reports false violations."""
    paths = {p.name: word_of(p) for p in leaf_paths(_branching())}
    cone = next(w for n, w in paths.items() if "Cone" in n)
    assert any("detect:cone" in p.cues for p in cone)
    other = next(w for n, w in paths.items() if "Cone" not in n)
    assert not any("detect:cone" in p.cues for p in other)


def test_stutter_collapse_merges_same_mode_positions():
    """X-free LTL is stutter-invariant, so `junction -> junction` is one position."""
    t = _linear(["junction", "junction", "path"],
                [None, "Detect(Cone) seen", "Bearing(Left) completed"])
    w = word_of(leaf_paths(t)[0])
    assert [p.mode for p in w] == ["junction", "path"]
    assert "detect:cone" in w[0].cues        # the merged position keeps both cues


# --------------------------------------------------------------------------- #
# containment: each positive paired with a rejected mutation                   #
# --------------------------------------------------------------------------- #

def test_reach_goal_satisfied_and_its_mutation_rejected():
    ok = _linear(["junction"])
    f = r"\mathbf{F}(\text{Detect}(\text{Junction}))"
    assert check_tree(ok, f).contained is True
    bad = _linear(["path"])                  # never reaches a junction
    assert check_tree(bad, f).contained is False


def test_global_prohibition_rejects_the_plan_that_violates_it():
    f = r"\mathbf{G}(\lnot\text{Detect}(\text{Junction}))"
    assert check_tree(_linear(["path", "path"]), f).contained is True
    assert check_tree(_linear(["path", "junction"]), f).contained is False


def test_until_requires_the_hold_to_survive_the_run_up():
    f = (r"\Phi_{Road} \mathbf{U} \text{Detect}(\text{Cone})")
    good = _linear(["path", "path"], [None, "Detect(Cone) seen"])
    assert check_tree(good, f).contained is True
    # leaves the road before the cue -- the half of U that the tree cannot express
    bad = _linear(["path", "junction", "path"],
                  [None, None, "Detect(Cone) seen"])
    assert check_tree(bad, f).contained is False


def test_branching_plan_is_checked_on_every_leaf_not_just_one():
    """A violation on ONE branch must fail the whole plan. This is the property that makes
    the check worth anything: the tree enumerates paths, so a formula can be satisfied by
    the branch that happens to be listed first and violated by its sibling."""
    f = (r"\text{Detect}(\text{Junction}) \land "
         r"((\text{Detect}(\text{Cone}) \land \mathbf{F}\text{Bearing}(Left)) \lor "
         r"(\lnot\text{Detect}(\text{Cone}) \land \mathbf{F}\text{Bearing}(Right)))")
    assert check_tree(_branching(), f).contained is True

    swapped = copy.deepcopy(_branching())    # cone branch now turns RIGHT
    swapped["steps"][0]["branches"][0]["sub_plan"][0]["transition_cue"] = \
        "Bearing(Right) completed"
    rep = check_tree(swapped, f)
    assert rep.contained is False
    assert rep.n_violating == 1              # exactly the mutated leaf, not both
    assert "Cone" in rep.summary()


def test_unparseable_formula_reports_rather_than_crashes():
    rep = check_tree(_linear(["path"]), r"\text{Detect}(")
    assert rep.parsed is False and rep.contained is None and rep.error


def test_missing_formula_is_not_silently_contained():
    """A plan with no formula must not report as verified."""
    rep = check_tree(_linear(["path"]))
    assert rep.contained is None


def test_cue_ordinal_expands_into_an_alternation():
    """`cue_ordinal: N` on one step means N sightings, and the formula writes it that way.

    Without expanding, every ordinal plan fails its own formula: the plans are fine but a
    one-position word model cannot represent the count.
    """
    t = {"steps": [{"step": 0, "start_mode": "path", "goal_mode": "path",
                    "transition_cue": "Detect(Bench)", "cue_ordinal": 2,
                    "description": "stop at the second bench"}]}
    w = word_of(leaf_paths(t)[0])
    assert [sorted(p.cues) for p in w] == [["detect:bench"], [], ["detect:bench"]]

    f = (r"\mathbf{F}(\text{Detect}(Bench) \land "
         r"\mathbf{F}(\lnot\text{Detect}(Bench) \land \mathbf{F}\text{Detect}(Bench)))")
    assert check_tree(t, f).contained is True

    # and the FIRST bench must not satisfy a second-bench formula
    t1 = {"steps": [{"step": 0, "start_mode": "path", "goal_mode": "path",
                     "transition_cue": "Detect(Bench)", "cue_ordinal": 1,
                     "description": "stop at the first bench"}]}
    assert check_tree(t1, f).contained is False
