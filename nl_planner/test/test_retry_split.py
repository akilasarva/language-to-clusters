"""A malformed formula must cost a formula retry, not a whole plan.

Every plan-side validator runs BEFORE the syntax gate, so when the formula is rejected
the plan in hand has already passed all of them. With a single shared budget, formula
repairs consume the retries the PLAN needs, and plans fail by running out of budget
rather than on any gate.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from nl_planner.pipeline import (DEFAULT_MAX_ATTEMPTS, DEFAULT_MAX_FORMULA_ATTEMPTS,
                                 PlanGenerationError, generate_plan)
from nl_planner.schemas import FilteredCommand, GeneratorOutput, NavPlan, PlanStep
from nl_planner.taxonomy import load_taxonomy

TAX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "config", "cluster_map.livox1.yaml")


def _plan() -> NavPlan:
    return NavPlan(
        plan_name="t", description="a legal one-step plan",
        steps=[PlanStep(step=0, description="walk the path", start_mode="path",
                        goal_mode="path", transition_cue="Detect(Bench)",
                        trigger="landmark")],
    )


class _Run:
    def __init__(self, out): self.output = out


class _Gen:
    """Returns a valid plan every time, with a formula we control."""

    def __init__(self, formula: str):
        self.formula, self.calls = formula, 0

    def run_sync(self, _msg):
        self.calls += 1
        return _Run(GeneratorOutput(
            filtered_command=FilteredCommand(original="go", filtered="go"),
            json_plan=_plan(), stl_formula=self.formula))


class _Bundle:
    def __init__(self, gen):
        self.generator, self.tripartite_verifier, self.model_id = gen, None, "stub"


def test_a_bad_formula_does_not_consume_the_plan_budget():
    """The loop must make more calls than `max_attempts` before giving up.

    With a permanently malformed formula and a permanently valid plan, a shared-budget
    loop would stop after `max_attempts` calls. The split adds the formula budget on top, so the
    call count proves the plan budget was not being charged.
    """
    # max_attempts=1 is the discriminator: the default budgets happen to coincide at 3,
    # so with both at their defaults a charged and an uncharged loop make the same number
    # of calls and the test would pass against the bug it exists to catch.
    gen = _Gen(r"\Phi_{NotAMacro} && bad")           # fails Rule 2 and Rule 5, always
    with pytest.raises(PlanGenerationError):
        generate_plan("go", taxonomy=load_taxonomy(TAX), agents=_Bundle(gen),
                      max_attempts=1, verify_tripartite=False)
    assert gen.calls > 1, (
        f"only {gen.calls} generator call(s) with max_attempts=1: a formula rejection is "
        f"still being charged to the plan budget"
    )
    assert gen.calls == DEFAULT_MAX_FORMULA_ATTEMPTS + 1


def test_a_valid_formula_still_accepts_on_the_first_call():
    """The split must not cost anything on the happy path."""
    gen = _Gen(r"\mathbf{G}\Phi_{Path}")
    res = generate_plan("go", taxonomy=load_taxonomy(TAX), agents=_Bundle(gen),
                        verify_tripartite=False)
    assert res.accepted and gen.calls == 1


def test_the_ordinal_repair_is_on_by_default():
    """The cue_ordinal repair is on by default.

    Tree plans frequently produce the double-counted ordinal; linear plans rarely do,
    so the repair mainly affects tree plans.
    """
    import inspect
    from nl_planner.pipeline import generate_plan
    sig = inspect.signature(generate_plan)
    assert sig.parameters["repair_ordinals"].default is True


def test_the_repair_leaves_a_landmark_ordinal_alone():
    """A landmark count lives on ONE extent-mode step and must survive the repair.

    The whole distinction `validate_cue_ordinal` draws is that N of a POINT mode means N
    steps, while "the 2nd bench" is one `path -> path` step carrying cue_ordinal: 2.
    Stripping the second would break landmark counting to fix junction counting.
    """
    from nl_planner.taxonomy import repair_cue_ordinal, validate_cue_ordinal
    plan = {"steps": [{"step": 0, "goal_mode": "junction", "cue_ordinal": None},
                      {"step": 1, "goal_mode": "junction", "cue_ordinal": 2},
                      {"step": 2, "goal_mode": "path", "cue_ordinal": 3}]}
    fixed = repair_cue_ordinal(plan)
    assert len(fixed) == 1 and "steps[1]" in fixed[0]
    assert plan["steps"][1]["cue_ordinal"] is None      # the double count, removed
    assert plan["steps"][2]["cue_ordinal"] == 3         # the landmark count, kept
    assert validate_cue_ordinal(plan) == []
