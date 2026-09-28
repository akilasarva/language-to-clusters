"""`repair_medium_detect_cue`: the unanswerable cues caused by the plan encoding itself.

Most unanswerable cues on Touchdown name a referent this deployment has no family for --
Touchdown is real NYC street view and the missions are ABOUT awnings, scaffolding and
crosswalks. Those are not generation errors and this repair must not touch them.

The repairable ones fall in two shapes, both deterministic, and occur on
`transition_cue`, not branch `vlm_cue`.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from nl_planner.cue_vocab import CARLA_GT_VOCAB, validate_plan_cues
from nl_planner.taxonomy import repair_medium_detect_cue


def _plan(**st):
    base = dict(step=0, description="d", start_mode="path", goal_mode="path",
                transition_cue=None, trigger="landmark")
    base.update(st)
    return {"plan_name": "t", "description": "d", "steps": [base]}


# --------------------------------------------------------------------------- #
# shape 1 -- seeing the medium you are entering                                #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("cue,start,goal", [
    ("Detect(Path)", "Other", "path"),
    ("Detect(Path)", "passage", "path"),
    ("Detect(Path)", "junction", "path"),
    ("Detect(Path)", "Edge: Along", "path"),
    ("Detect(Passage)", "path", "passage"),
    ("Detect(Passage)", "junction", "passage"),
    ("Detect(OpenSpace)", "path", "Open Space"),
    ("Detect(AlongEdge)", "path", "Edge: Along"),   # token-set match, either spelling
])
def test_entering_the_medium_becomes_a_traversal(cue, start, goal):
    p = _plan(transition_cue=cue, start_mode=start, goal_mode=goal)
    did = repair_medium_detect_cue(p)
    assert did, f"{cue} on {start}->{goal} was not repaired"
    st = p["steps"][0]
    assert st["transition_cue"] is None and st["trigger"] == "traverse"
    assert st["goal_mode"] == goal, "the repair moved WHERE the step ends"


@pytest.mark.parametrize("cue,start,goal", [
    # No cluster transition: `traverse` would complete instantly or never. Left for the
    # gate; the repair deliberately declines.
    ("Detect(Passage)", "path", "path"),
    ("Detect(Path)", "path", "path"),
    # A place you APPROACH, not a medium you are inside -- `Detect(Junction)` is
    # answerable, and `repair_place_detect_to_traverse` owns the place-noun case.
    ("Detect(Junction)", "path", "junction"),
    # Not the medium the step is entering.
    ("Detect(Passage)", "path", "junction"),
    # A real object: answerable, nothing to repair.
    ("Detect(Bench)", "path", "junction"),
    # No family exists. Untouchable here -- inventing one would turn an
    # honest "unanswerable" into a confident wrong answer.
    ("Detect(Scaffolding)", "path", "junction"),
    ("Detect(GreenArchedAwning)", "path", "path"),
])
def test_declines_when_the_conversion_would_not_be_sound(cue, start, goal):
    p = _plan(transition_cue=cue, start_mode=start, goal_mode=goal)
    assert repair_medium_detect_cue(p) == []
    assert p["steps"][0]["transition_cue"] == cue


# --------------------------------------------------------------------------- #
# shape 2 -- a manoeuvre written as an object                                  #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("cue,bearing", [("Detect(RightTurn)", "Bearing(Right) completed"),
                                         ("Detect(LeftTurn)", "Bearing(Left) completed"),
                                         ("detect( straight turn )",
                                          "Bearing(Straight) completed")])
def test_a_turn_becomes_a_bearing(cue, bearing):
    """This shape sits on a step whose start_mode == goal_mode, so there is no cluster
    transition to traverse to. `Bearing(...)` is resolved from the trajectory,
    which is why `cue_vocab._TRAJECTORY_CUE` exempts it -- answerable by construction."""
    p = _plan(transition_cue=cue, start_mode="path", goal_mode="path")
    assert repair_medium_detect_cue(p)
    st = p["steps"][0]
    assert st["transition_cue"] == bearing and st["trigger"] == "topology"
    assert validate_plan_cues(p, CARLA_GT_VOCAB) == []


# --------------------------------------------------------------------------- #
# scope                                                                        #
# --------------------------------------------------------------------------- #

def test_branch_cues_are_untouched():
    """A branch cue is a DECISION. `repair_prose_decision_cue` owns the decision step's
    own cue."""
    p = _plan(transition_cue="Detect(RightTurn)", branches=[
        {"vlm_cue": "Detect(RightTurn)", "sub_plan": []},
        {"vlm_cue": "default", "sub_plan": []}])
    assert repair_medium_detect_cue(p) == []
    assert p["steps"][0]["transition_cue"] == "Detect(RightTurn)"


def test_recurses_into_sub_plans():
    p = _plan(transition_cue=None, branches=[
        {"vlm_cue": "default",
         "sub_plan": [dict(step=1, description="d", start_mode="junction",
                           goal_mode="path", transition_cue="Detect(Path)",
                           trigger="landmark")]}])
    did = repair_medium_detect_cue(p)
    assert did and "br0/" in did[0]
    assert p["steps"][0]["branches"][0]["sub_plan"][0]["trigger"] == "traverse"


def test_it_is_wired_into_the_robot_path():
    """A repair that exists and is not called is not a fix."""
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "nl_planner", "pipeline.py")).read()
    assert "repair_medium_detect_cue(_d)" in src
