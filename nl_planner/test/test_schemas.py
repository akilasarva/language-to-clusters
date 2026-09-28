"""Pydantic contract round-trip + validator tests.

Run with: pytest nl_planner/test/
"""

from __future__ import annotations

import pytest

from nl_planner.schemas import (
    DEFAULT_BRANCH_CUE,
    MAX_BRANCH_DEPTH,
    Branch,
    FilteredCommand,
    GeneratorOutput,
    NavPlan,
    PlanStep,
)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _linear_step(step: int, start: str = "Road: On", goal: str = "Road: On",
                 cue: str | None = None) -> PlanStep:
    return PlanStep(
        step=step, description=f"step {step}",
        start_mode=start, goal_mode=goal, transition_cue=cue,
    )


def _decision_step(step: int, branches: list[Branch],
                   mode: str = "Bridge: Enter") -> PlanStep:
    return PlanStep(
        step=step, description=f"decision @ {mode}",
        start_mode=mode, goal_mode=mode, transition_cue="decision point",
        branches=branches,
    )


def _default_branch(sub_plan: list[PlanStep] | None = None) -> Branch:
    return Branch(vlm_cue=DEFAULT_BRANCH_CUE, sub_plan=sub_plan or [_linear_step(0)])


# --------------------------------------------------------------------------- #
# Linear plan round-trip                                                       #
# --------------------------------------------------------------------------- #

def test_linear_plan_round_trip():
    plan = NavPlan(
        plan_name="x",
        description="y",
        steps=[_linear_step(0), _linear_step(1, cue="Detect(StopSign)")],
    )
    blob = plan.model_dump_json()
    again = NavPlan.model_validate_json(blob)
    assert again == plan


# --------------------------------------------------------------------------- #
# Branch invariants                                                            #
# --------------------------------------------------------------------------- #

def test_branches_require_exactly_one_default():
    with pytest.raises(Exception):
        _decision_step(
            0,
            branches=[
                Branch(vlm_cue="blocked", sub_plan=[_linear_step(0)]),
                Branch(vlm_cue="also blocked", sub_plan=[_linear_step(0)]),
            ],
        )
    with pytest.raises(Exception):
        _decision_step(
            0,
            branches=[
                _default_branch(),
                _default_branch(),  # two defaults — illegal
            ],
        )


def test_branches_require_at_least_two_entries():
    with pytest.raises(Exception):
        _decision_step(0, branches=[_default_branch()])


def test_valid_decision_step_round_trip():
    step = _decision_step(
        0,
        branches=[
            Branch(vlm_cue="bridge is blocked", sub_plan=[_linear_step(0)]),
            _default_branch(),
        ],
    )
    plan = NavPlan(plan_name="b", description="b", steps=[step])
    again = NavPlan.model_validate_json(plan.model_dump_json())
    assert again == plan


# --------------------------------------------------------------------------- #
# Depth cap                                                                    #
# --------------------------------------------------------------------------- #

def test_branch_depth_cap_enforced():
    # Build a chain of decision steps exceeding MAX_BRANCH_DEPTH.
    def nested(depth: int) -> PlanStep:
        if depth == 0:
            return _linear_step(0)
        return _decision_step(
            0,
            branches=[
                Branch(vlm_cue="dive", sub_plan=[nested(depth - 1)]),
                _default_branch(),
            ],
        )

    # Just-OK depth.
    NavPlan(plan_name="ok", description="d", steps=[nested(MAX_BRANCH_DEPTH)])

    # Too-deep depth must fail.
    with pytest.raises(Exception):
        NavPlan(
            plan_name="bad", description="d",
            steps=[nested(MAX_BRANCH_DEPTH + 1)],
        )


# --------------------------------------------------------------------------- #
# Generator output                                                             #
# --------------------------------------------------------------------------- #

def test_generator_output_round_trip():
    out = GeneratorOutput(
        filtered_command=FilteredCommand(original="go", filtered="go"),
        json_plan=NavPlan(plan_name="x", description="x", steps=[_linear_step(0)]),
        stl_formula=r"\Phi_{Road}",
    )
    again = GeneratorOutput.model_validate_json(out.model_dump_json())
    assert again == out


# --------------------------------------------------------------------------- #
# Decision steps must be terminal (unreachable-step guard)                      #
# --------------------------------------------------------------------------- #

def test_decision_step_must_be_last_in_root():
    """Steps after a decision on the trunk can never run — reject the plan.

    PlanNavigator.descend() replaces the active sub_plan, so the trunk is gone
    once a branch is taken. Without this validator such a plan would validate fine
    and then report COMPLETE having silently skipped its tail.
    """
    with pytest.raises(ValueError, match="can never run"):
        NavPlan(
            plan_name="x", description="y",
            steps=[
                _linear_step(0),
                _decision_step(1, [
                    Branch(vlm_cue="bridge is flooded", sub_plan=[_linear_step(0)]),
                    _default_branch(),
                ]),
                _linear_step(2),          # <- unreachable
            ],
        )


def test_decision_step_must_be_last_inside_a_branch():
    """The same rule applies one level down, not just at the root."""
    inner = _decision_step(0, [
        Branch(vlm_cue="obstacle", sub_plan=[_linear_step(0)]),
        _default_branch(),
    ])
    with pytest.raises(ValueError, match="can never run"):
        NavPlan(
            plan_name="x", description="y",
            steps=[
                _decision_step(0, [
                    Branch(vlm_cue="side road", sub_plan=[inner, _linear_step(1)]),
                    _default_branch(),
                ]),
            ],
        )


def test_terminal_decision_is_accepted():
    """The legal shape: continuation lives INSIDE the branch that earns it."""
    plan = NavPlan(
        plan_name="bridge or stop", description="y",
        steps=[
            _linear_step(0),
            _decision_step(1, [
                # flooded -> the mission ends here, no continuation
                Branch(vlm_cue="bridge is flooded", sub_plan=[_linear_step(0)]),
                # crossed -> continue to the bench
                _default_branch([_linear_step(0), _linear_step(1, cue="Detect(Bench)")]),
            ]),
        ],
    )
    assert plan.steps[-1].branches is not None


def test_linear_plan_unaffected_by_terminal_rule():
    NavPlan(plan_name="x", description="y",
            steps=[_linear_step(0), _linear_step(1), _linear_step(2)])


# --------------------------------------------------------------------------- #
# `Phi_X U cue` — the dwell step                                               #
# --------------------------------------------------------------------------- #

def test_hold_mode_without_until_is_rejected():
    """Pins: a hold nothing can terminate must not be constructible.

    `hold_mode` alone asks the executor to stay in a mode forever. There is no cue
    to end the step, so the run dies on the tick budget and the log reads
    "timed out at step N" — indistinguishable from a routing failure, which is
    where the debugging would go. Rejecting it at the contract is the only place
    the cause is still visible.
    """
    with pytest.raises(ValueError) as e:
        PlanStep(step=0, description="hold forever",
                 start_mode="Road: On", goal_mode="Road: On",
                 hold_mode="Road: On")
    assert "hold_mode and until" in str(e.value)


def test_until_without_hold_mode_is_rejected():
    """Pins: a plan must not be able to CLAIM an invariant it is not getting.

    `until` on its own is the shape brain has always executed — a cue ends the
    step and nothing watches where the robot went in between. Accepting it would
    let a plan read as "stay on the walkway until the plaza" while executing
    "wander anywhere until the plaza", which is the exact pair of behaviours
    `InvariantMonitor` exists to separate.
    """
    with pytest.raises(ValueError) as e:
        PlanStep(step=0, description="until only",
                 start_mode="Road: On", goal_mode="Road: On",
                 until="Detect(Plaza)")
    assert "hold_mode and until" in str(e.value)


def test_a_dwell_step_may_not_also_branch():
    """Pins: descending a branch must not be able to abandon an invariant mid-hold.

    `PlanNavigator.descend()` REPLACES the active sub_plan, so a decision taken on
    a dwell step throws away the step that was holding the mode. The invariant
    would stop applying with nothing in the record saying it ever did — a plan
    that appears to constrain the robot and does not. There is also no destination
    on a dwell step to fork AT.
    """
    with pytest.raises(ValueError) as e:
        PlanStep(step=0, description="hold and decide",
                 start_mode="Road: On", goal_mode="Road: On",
                 hold_mode="Road: On", until="Detect(Plaza)",
                 branches=[
                     Branch(vlm_cue="flooded", sub_plan=[_linear_step(0)]),
                     _default_branch(),
                 ])
    assert "hold_mode" in str(e.value) and "branches" in str(e.value)


def test_a_dwell_step_may_start_and_end_in_the_same_mode():
    """Pins the legality of the shape, at the schema level, in both directions.

    A dwell step has no destination, so `goal_mode == start_mode` is what it
    should say. This is the same exemption the pure-decision step gets, and the
    reason it has to be written down twice (here and in
    `taxonomy.validate_plan_transitions`) is that a plan passing one validator and
    failing the other is a plan the generator cannot fix from the feedback.
    """
    step = PlanStep(step=0, description="hold the road",
                    start_mode="Road: On", goal_mode="Road: On",
                    hold_mode="Road: On", until="Detect(Intersection)")
    assert step.hold_mode == "Road: On" and step.until == "Detect(Intersection)"
    NavPlan(plan_name="x", description="y", steps=[step])


def test_steps_written_before_the_operator_existed_are_untouched():
    """Pins the no-op: neither field set is the state of most plans.

    A validator that fires on `hold_mode=None, until=None` would reject every plan
    without a dwell step.
    """
    step = _linear_step(0)
    assert step.hold_mode is None and step.until is None


# --------------------------------------------------------------------------- #
# `[]~X` — the plan-level negative constraint                                  #
# --------------------------------------------------------------------------- #

def test_forbid_modes_is_plan_level_not_step_level():
    """Pins WHERE the negative constraint lives, which is the whole design claim.

    "never enter the plaza" is a property of the mission, not of step 3. Hung off
    a step it would stop applying the instant that step advanced — the robot could
    satisfy it for the leg where it was never going near the plaza anyway and
    drive straight through on the next, with every check green.
    """
    plan = NavPlan(plan_name="x", description="y",
                   steps=[_linear_step(0), _linear_step(1)],
                   forbid_modes=["Open Space"])
    assert plan.forbid_modes == ["Open Space"]
    assert not hasattr(plan.steps[0], "forbid_modes")


def test_forbid_modes_defaults_to_none_so_old_plans_are_unchanged():
    """Pins: absence of the field must mean 'no constraint', never 'forbid nothing
    was checked'. Plans without the field must keep validating."""
    plan = NavPlan(plan_name="x", description="y", steps=[_linear_step(0)])
    assert plan.forbid_modes is None
