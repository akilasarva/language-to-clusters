"""Pydantic contract shared by generator, verifiers, executor, and brain.

This module is the single source of truth for what the LLM is allowed to emit,
and what the executor expects to consume. All validation that does NOT depend
on a per-environment taxonomy lives here as ``model_validator``s. Taxonomy-
dependent checks (semantic mode must exist in the YAML) live in
``pipeline.py`` so the retry loop can feed the failure back into the
generator.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


# --------------------------------------------------------------------------- #
# Soft caps                                                                   #
# --------------------------------------------------------------------------- #

#: Maximum branch nesting depth before the schema validator rejects the plan.
#: Documented to Prompt 1 so the LLM should never produce deeper trees.
#:
#: The cap is a soft one; the real limit is the generator's reliability, which
#: degrades with depth, so it is set high enough that deep plans are limited by
#: the model rather than refused by this validator.
MAX_BRANCH_DEPTH = 5

#: Sentinel for the always-present fallback branch at every decision point.
DEFAULT_BRANCH_CUE = "default"

#: How a step decides it is finished. This is the discriminant that answers
#: "which cluster gets used when" — it selects whether the cluster is the
#: evidence or merely a guard.
#:
#: ``traverse``  "go down the road" — the CLUSTER is the evidence. Advance on
#:               sustained membership in the strict accept set. Never degrades.
#: ``landmark``  "pass the blue building" — a Detect(...) predicate is the
#:               evidence and the cluster is a permissive guard. The step may be
#:               satisfied on a coarser cluster (mode_meta.accept_degraded), so a
#:               missed `along_edge` does not strand the plan.
#: ``topology``  "turn right at the intersection" — a Bearing(...) completion is
#:               the evidence. The cluster is logged as supporting evidence only,
#:               which is what lets `Intersection: In` work in an environment
#:               where junction is not perception-backed.
TRIGGERS = ("traverse", "landmark", "topology")


def infer_trigger(transition_cue: str | None) -> str:
    """Derive a step's trigger from its cue when the plan does not carry one.

    Kept deterministic and on the Python side on purpose: the trigger follows
    mechanically from which predicate the cue uses, so there is nothing for the
    LLM to get wrong, and plans generated before ``trigger`` existed (including
    the hand-written brain/plan.json) keep working unchanged.
    """
    cue = (transition_cue or "").strip()
    if not cue:
        return "traverse"
    low = cue.lower()
    if "detect(" in low:
        return "landmark"
    if "bearing(" in low:
        return "topology"
    # Free-text cue with no predicate ("a light brown bench", "decision point").
    # It is still a visual landmark check, so treat it as one.
    return "landmark"


# --------------------------------------------------------------------------- #
# Generator-side primitives                                                   #
# --------------------------------------------------------------------------- #

class FilteredCommand(BaseModel):
    """Original English plus the cleaned-up version the LLM actually planned for."""

    original: str = Field(..., description="The verbatim mission text the user issued.")
    filtered: str = Field(
        ...,
        description=(
            "Same intent with transient features removed (bikes, pedestrians, "
            "construction, cones, etc.). Keep stable landmarks."
        ),
    )


class PlanStep(BaseModel):
    """One step in the (possibly branch-shaped) navigation plan.

    Linear steps have ``branches=None``. Decision-point steps carry a list of
    ``Branch``es; each branch's ``sub_plan`` is itself a linear sequence of
    ``PlanStep``s that may end in another branch step (up to
    ``MAX_BRANCH_DEPTH``).

    A step with ``hold_mode`` set is a **dwell step** — "hold mode X until the
    cue fires", the JSON spelling of ``Phi_X U cue``. It is the third shape,
    and it differs from both of the above in one way that every consumer has to
    know about: **a dwell step has no destination.** Nothing is being travelled
    to, so ``StepTargeter`` must not be asked to ground a target region for it
    (there is no answer, and asking produces a plausible-looking wrong one), and
    ``goal_mode == start_mode`` is LEGAL rather than the "you never left"
    mistake ``taxonomy.validate_plan_transitions`` rejects for point modes.
    """

    step: int = Field(..., ge=0, description="0-based index within the containing sub_plan.")
    description: str = Field(..., description="One-line human-readable summary.")
    start_mode: str = Field(
        ...,
        description=(
            "Semantic mode the step starts in, e.g. 'Road: On' or "
            "'Intersection: Approach/Enter'. Must appear in the taxonomy YAML."
        ),
    )
    goal_mode: str = Field(
        ...,
        description=(
            "Semantic mode the step ends in. For a pure decision step "
            "(branches != None and the robot stays put), set goal_mode == start_mode. "
            "Same for a dwell step (hold_mode != None): it has no destination, so "
            "goal_mode == start_mode is the correct encoding, not a mistake."
        ),
    )
    transition_cue: Optional[str] = Field(
        None,
        description=(
            "Free-text visual cue brain_controller's VLM polls for on arrival "
            "at goal_mode. Use null for cue-free arrival-only advancement."
        ),
    )
    notes: Optional[str] = Field(None, description="Optional commentary for humans.")
    trigger: Optional[str] = Field(
        None,
        description=(
            "How this step decides it is done: 'traverse' (the cluster is the "
            "evidence), 'landmark' (a Detect(...) cue is the evidence and the "
            "cluster only permits), or 'topology' (a Bearing(...) completion is "
            "the evidence). Leave null to have it inferred from transition_cue."
        ),
    )
    cue_ordinal: Optional[int] = Field(
        None,
        ge=1,
        description=(
            "Which SIGHTING of transition_cue satisfies this step, 1-based. Use "
            "2 for 'the 2nd bench'. Distinct sightings are separated by the cue "
            "going out of view, so one bench held in frame counts once. Null "
            "means the first sighting (equivalent to 1)."
        ),
    )
    hold_mode: Optional[str] = Field(
        None,
        description=(
            "The mode that must HOLD for the whole duration of this step — the "
            "left half of 'Phi_X U cue'. Setting it makes this a DWELL step: it "
            "has no destination, so goal_mode may legally equal start_mode, and "
            "the executor must not ground a target region for it. Must appear in "
            "the taxonomy YAML. Requires ``until``."
        ),
    )
    until: Optional[str] = Field(
        None,
        description=(
            "The cue that TERMINATES a dwell step — the right half of "
            "'Phi_X U cue'. Same free-text/predicate vocabulary as "
            "transition_cue; the cue oracle answers it. Requires ``hold_mode``."
        ),
    )
    branches: Optional[List["Branch"]] = Field(
        None,
        description=(
            "Present only at decision points. Exactly one branch must have "
            "vlm_cue == 'default' as the fallback."
        ),
    )

    @model_validator(mode="after")
    def _check_until(self) -> "PlanStep":
        """``hold_mode`` and ``until`` are the two halves of ONE operator.

        ``Phi_X U cue`` is not two independent decorations on a step: the mode is
        what must hold, the cue is what ends the holding. Half of it is not a
        weaker version of the whole, it is a different and silently wrong step.

        ``hold_mode`` alone is a step nothing can terminate — the executor would
        hold heading forever and the run would die on the tick budget, which
        reads in the log exactly like a routing failure.

        ``until`` alone is the shape brain has ALWAYS executed (a cue ends the
        step, nothing watches where the robot went). Accepting it would let a
        plan claim an invariant it is not getting, which is the one failure
        ``InvariantMonitor`` exists to make visible.

        ``branches`` is rejected on the same step because a dwell step has no
        destination to fork at: ``PlanNavigator.descend()`` replaces the sub_plan
        at the decision, so the invariant would be abandoned mid-hold with no
        record that it ever applied.
        """
        if (self.hold_mode is None) != (self.until is None):
            raise ValueError(
                "hold_mode and until are the two halves of 'Phi_X U cue' and must "
                f"be set together; got hold_mode={self.hold_mode!r}, "
                f"until={self.until!r}. A hold with no cue never terminates; a cue "
                f"with no hold is just transition_cue."
            )
        if self.hold_mode is not None and self.branches is not None:
            raise ValueError(
                f"step {self.description!r} sets hold_mode={self.hold_mode!r} and "
                f"also carries branches. A dwell step has no destination to fork "
                f"at, and descending a branch abandons the invariant mid-hold. "
                f"Put the decision in a following step."
            )
        return self

    @model_validator(mode="after")
    def _check_trigger(self) -> "PlanStep":
        if self.trigger is not None and self.trigger not in TRIGGERS:
            raise ValueError(
                f"trigger must be one of {TRIGGERS}; got {self.trigger!r}"
            )
        return self

    @model_validator(mode="after")
    def _check_branches(self) -> "PlanStep":
        if self.branches is None:
            return self
        if len(self.branches) < 2:
            raise ValueError(
                "branches must have >= 2 entries (at least one real choice plus default)"
            )
        defaults = [b for b in self.branches if b.vlm_cue.strip().lower() == DEFAULT_BRANCH_CUE]
        if len(defaults) != 1:
            raise ValueError(
                f"branches must contain exactly one entry with vlm_cue == "
                f"'{DEFAULT_BRANCH_CUE}'; got {len(defaults)}"
            )
        return self


class Branch(BaseModel):
    """One choice at a decision point.

    The executor calls a VLM at the decision step, scoring each ``vlm_cue``
    against the latest camera image, and follows the winning branch's
    ``sub_plan``.

    A branch is TERMINAL: when its ``sub_plan`` runs out the mission is complete.
    It does not rejoin the list it forked from — see
    ``_assert_terminal_decisions``. There is deliberately no ``rejoin`` flag yet,
    because the missions seen so far want independent termination ("if it is
    flooded you cannot cross, and you end there") and a flag nobody needs is a
    flag the generator will misuse. Add one when a real mission needs a shared
    continuation badly enough that duplicating it into each branch is the worse
    option.
    """

    vlm_cue: str = Field(
        ...,
        description=(
            "Short visual description (or the literal string 'default'). The "
            "executor will ask the VLM 'which of these is best visible?' and "
            "pick this branch on a positive match."
        ),
    )
    sub_plan: List[PlanStep] = Field(
        ...,
        min_length=1,
        description="Linear continuation taken when this branch is chosen.",
    )


PlanStep.model_rebuild()


# --------------------------------------------------------------------------- #
# Top-level container                                                         #
# --------------------------------------------------------------------------- #

class ForbidInstance(BaseModel):
    """One instance-scoped prohibition: the ``ordinal``-th region of ``mode``.

    `ordinal` is 1-based and counts DISTINCT arrivals, the same rule `cue_ordinal` uses
    for advancement — re-entering the region you are already in is not a second instance.
    """

    mode: str = Field(..., description="A mode from the taxonomy, e.g. 'junction'.")
    ordinal: int = Field(..., ge=1, description="1-based: 2 means the second one entered.")


class NavPlan(BaseModel):
    """Tree-shaped navigation plan. ``steps`` is the root linear segment."""

    plan_name: str = Field(..., description="Short title for logs / UIs.")
    description: str = Field(..., description="One-paragraph summary of the mission.")
    steps: List[PlanStep] = Field(..., min_length=1)
    forbid_modes: Optional[List[str]] = Field(
        None,
        description=(
            "Modes the robot must NEVER enter, for the whole plan — the JSON "
            "spelling of 'always not X'. PLAN-LEVEL on purpose: 'do not drive "
            "through the plaza' is not a property of step 3, it is a property of "
            "the mission, and hanging it off a step would silently stop applying "
            "the moment that step advanced. Each entry must appear in the "
            "taxonomy YAML. Null and [] both mean 'no negative constraint', which "
            "is what every plan written before this field said."
        ),
    )
    forbid_instances: Optional[List[ForbidInstance]] = Field(
        None,
        description=(
            "Prohibitions scoped to ONE INSTANCE of a mode, e.g. 'do not turn at the "
            "SECOND intersection'. Distinct from forbid_modes, which bans a mode "
            "outright: 'not the second intersection' has no mode-level spelling, and "
            "writing forbid_modes: ['junction'] for it forbids every junction on a "
            "route the same plan traverses — a self-contradiction that passes every "
            "structural validator. Resolved at RUN TIME by counting arrivals, because "
            "a plan names modes rather than ids (which is what makes it portable), and "
            "on a branching plan which instance is Nth depends on the branch taken. "
            "Null and [] both mean 'no instance-scoped constraint'."
        ),
    )
    #: The formula this plan was generated alongside, carried so the EXECUTOR can compile
    #: it. Optional and defaulted, so every existing plan and every recorded JSON stays
    #: valid without migration.
    #:
    #: WHY IT IS HERE AND NOT DERIVED. The plan's own constraint fields are sometimes wrong
    #: or absent while the formula compiles to the right monitor. Without this field such
    #: constraints are unreachable on the robot: executor_node could not pass `stl=` to
    #: to_brain_tree, and the constraint would be silently dropped at the dispatch boundary.
    stl_formula: Optional[str] = Field(
        default=None,
        description="STL formula for this plan. Compiled to extra monitors at "
                    "materialization; never overrides a constraint the plan states.",
    )
    require_modes: Optional[List[str]] = Field(
        None,
        description=(
            "Modes the robot must NEVER LEAVE, for the whole plan — 'always X', "
            "the positive mirror of forbid_modes. 'Stay on the walkway the entire "
            "way' is the canonical case. PLAN-LEVEL for the same reason: it binds "
            "every branch, and a step-scoped version would switch itself off at the "
            "first fork. Distinct from PlanStep.hold_mode, which is scoped to ONE "
            "step and requires a paired `until`; this one has no terminator because "
            "it holds for the mission. Null and [] both mean 'no invariant'."
        ),
    )

    @model_validator(mode="after")
    def _check_depth(self) -> "NavPlan":
        for step in self.steps:
            _assert_depth(step, depth=1)
        return self

    @model_validator(mode="after")
    def _check_decisions_are_terminal(self) -> "NavPlan":
        _assert_terminal_decisions(self.steps, where="steps")
        return self


def _assert_depth(step: PlanStep, *, depth: int) -> None:
    if step.branches is None:
        return
    if depth > MAX_BRANCH_DEPTH:
        raise ValueError(
            f"branch nesting depth exceeds MAX_BRANCH_DEPTH={MAX_BRANCH_DEPTH}"
        )
    for branch in step.branches:
        for sub_step in branch.sub_plan:
            _assert_depth(sub_step, depth=depth + 1)


def _assert_terminal_decisions(steps: List[PlanStep], *, where: str) -> None:
    """A decision step must be LAST in its list. Steps after it are unreachable.

    ``brain.plan_navigator.PlanNavigator.descend()`` REPLACES the active sub_plan
    with the chosen branch's, so a branch is a one-way door: it never returns to
    the list it forked from. Anything written after the decision in that list is
    silently never executed — the plan reports COMPLETE having skipped it, with
    no error and no warning. That is the worst shape of bug this contract can
    let through, so it is rejected here instead.

    This is not a limitation being papered over; it is the intended semantics.
    Branches TERMINATE INDEPENDENTLY. "Cross the bridge unless it is flooded,
    then stop at the second bench" means the bench follows only the crossing —
    a robot that could not cross ends at the bridge. Writing the bench inside
    the ``default`` branch says exactly that. Writing it on the trunk would say
    "both outcomes lead to the bench", which is both a different mission and one
    the executor cannot run.

    So: put every continuation inside the branch that earns it. When a
    continuation genuinely IS common to all branches, duplicate it into each —
    and see the note on ``Branch`` about why there is no rejoin flag yet.
    """
    for i, step in enumerate(steps):
        if step.branches is None:
            continue
        if i != len(steps) - 1:
            orphans = [s.description for s in steps[i + 1:]]
            raise ValueError(
                f"decision step {i} ({step.description!r}) in {where} is followed "
                f"by {len(orphans)} more step(s): {orphans}. A branch never "
                f"returns to the list it forked from, so those steps can never "
                f"run. Move each one INTO the branch(es) it belongs to — if it "
                f"should happen only when the bridge is crossed, it goes in the "
                f"'default' branch; if it should happen either way, put a copy "
                f"in every branch."
            )
        for b_i, branch in enumerate(step.branches):
            _assert_terminal_decisions(
                branch.sub_plan,
                where=f"{where}[{i}].branches[{b_i}] ({branch.vlm_cue!r})",
            )


# --------------------------------------------------------------------------- #
# Agent outputs                                                               #
# --------------------------------------------------------------------------- #

class GeneratorOutput(BaseModel):
    """What the generator Agent returns in one call."""

    filtered_command: FilteredCommand
    json_plan: NavPlan
    stl_formula: str = Field(
        ...,
        description=(
            "Single STL formula (LaTeX-ish) covering ALL branches. The "
            "tripartite verifier checks it against the plan tree and the "
            "filtered English command."
        ),
    )


class SyntaxVerdict(BaseModel):
    """Output of the STL syntax-only verifier (Prompt 2a)."""

    ok: bool = Field(..., description="True iff the STL is well-formed.")
    error: Optional[str] = Field(
        None,
        description=(
            "Concrete rule(s) violated, copied verbatim into the retry feedback. "
            "Required iff ok is False."
        ),
    )


class TripartiteVerdict(BaseModel):
    """Output of the X<->Y<->Z verifier (Prompt 2b)."""

    english_stl_aligned: bool
    stl_json_aligned: bool
    english_json_aligned: bool
    notes: str = Field(
        ...,
        description="Concrete differences keyed by which pair disagreed.",
    )

    @property
    def ok(self) -> bool:
        return (
            self.english_stl_aligned
            and self.stl_json_aligned
            and self.english_json_aligned
        )


# --------------------------------------------------------------------------- #
# Errors                                                                      #
# --------------------------------------------------------------------------- #

class PlanGenerationError(RuntimeError):
    """Raised by ``pipeline.generate_plan`` when retries are exhausted."""


__all__ = [
    "MAX_BRANCH_DEPTH",
    "DEFAULT_BRANCH_CUE",
    "TRIGGERS",
    "infer_trigger",
    "FilteredCommand",
    "PlanStep",
    "Branch",
    "NavPlan",
    "GeneratorOutput",
    "SyntaxVerdict",
    "TripartiteVerdict",
    "PlanGenerationError",
]
