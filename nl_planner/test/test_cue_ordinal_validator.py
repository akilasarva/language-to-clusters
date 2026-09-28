"""The count lives in the step index OR in cue_ordinal, never both.

The model often unrolls correctly and stamps `cue_ordinal` anyway. Such a plan passes
modes, transitions and cues, and drives past its target -- asked for the 3rd
intersection, a plan with three junction steps whose last carries `cue_ordinal: 3` waits
for three MORE.
"""
from nl_planner.taxonomy import validate_cue_ordinal


def _step(goal, ordinal=None, start="path"):
    return {"step": 0, "description": "", "start_mode": start,
            "goal_mode": goal, "cue_ordinal": ordinal}


def _unrolled_third(last_ordinal=None):
    """`path->junction` three times, the pedestrian encoding of 'the third intersection'."""
    return {"steps": [_step("junction"), _step("path", start="junction"),
                      _step("junction"), _step("path", start="junction"),
                      _step("junction", last_ordinal)]}


def test_unrolled_traversal_with_no_ordinal_is_the_correct_shape():
    assert validate_cue_ordinal(_unrolled_third()) == []


def test_unrolled_traversal_that_also_sets_cue_ordinal_is_rejected():
    errs = validate_cue_ordinal(_unrolled_third(3))
    assert len(errs) == 1
    assert "double-count" in errs[0]
    assert "cue_ordinal=3" in errs[0]


def test_landmark_counting_is_untouched():
    """'stop at the 2nd bench' -- ONE path->path step carrying the ordinal.

    The robot never changes mode, so there is no step index to count with and
    `cue_ordinal` is the only encoding available. Rejecting this would break the shape
    generator.md Examples 4 and 5 explicitly teach.
    """
    plan = {"steps": [_step("path", 2)]}
    assert validate_cue_ordinal(plan) == []


def test_a_single_junction_step_with_an_ordinal_is_not_a_double_count():
    """Wrong for a different reason -- one step describes ONE junction -- but not this
    validator's business. `validate_plan_transitions` owns that."""
    assert validate_cue_ordinal({"steps": [_step("junction", 3)]}) == []


def test_it_looks_inside_branches():
    plan = {"steps": [
        _step("junction"), _step("path", start="junction"),
        {"step": 2, "description": "", "start_mode": "path", "goal_mode": "path",
         "cue_ordinal": None,
         "branches": [{"vlm_cue": "a cone", "sub_plan": [_step("junction", 2)]},
                      {"vlm_cue": "default", "sub_plan": [_step("junction")]}]}]}
    errs = validate_cue_ordinal(plan)
    assert len(errs) == 1 and "branch" in errs[0]
