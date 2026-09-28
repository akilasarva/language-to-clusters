"""Generate -> verify -> revise loop, orchestrated outside of pydantic-ai.

Why a manual loop instead of pydantic-ai's ``result_validators`` /
``ModelRetry``? Because the verifiers are *separate* LLM agents (Prompts 2a
and 2b), and each attempt's ``(prompt, output, verdict)`` trace is logged
so prompt-engineering failures can be debugged offline.
"""

from __future__ import annotations

import os

import json
import time
from dataclasses import dataclass, field
from typing import Any

from .schemas import (
    GeneratorOutput,
    NavPlan,
    PlanGenerationError,
    SyntaxVerdict,
    TripartiteVerdict,
)
from .stl_syntax import quick_syntax_check
from .taxonomy import (
    ClusterTaxonomy,
    validate_plan_modes,
    validate_plan_transitions,
    validate_decision_cues,
    validate_cue_ordinal,
    repair_cue_ordinal,
    repair_medium_detect_cue,
    repair_prose_decision_cue,
    repair_place_detect_to_traverse,
)


# --------------------------------------------------------------------------- #
# Trace dataclasses                                                            #
# --------------------------------------------------------------------------- #

#: `cue_vocab=` default. A plain `None` default could not be distinguished from a
#: caller DELIBERATELY asking for the open set, and the two must not be confused:
#: one is "resolve it from the deployment", the other is "there is no closed set".
_VOCAB_FROM_DEPLOYMENT = "__from_deployment__"


@dataclass
class Attempt:
    """One generator+verifier round-trip, captured for debugging."""

    iteration: int
    generator_input: str
    generator_output: GeneratorOutput | None = None
    generator_error: str | None = None
    cue_repairs: list[str] | None = None
    syntax: SyntaxVerdict | None = None
    tripartite: TripartiteVerdict | None = None
    taxonomy_missing: list[str] = field(default_factory=list)
    bad_transitions: list[str] = field(default_factory=list)
    bad_cues: list[str] = field(default_factory=list)
    bad_ordinals: list[str] = field(default_factory=list)
    #: cues this deployment cannot answer (see `cue_vocab`)
    bad_vocab: list[str] = field(default_factory=list)
    #: deterministic cue_ordinal fixes applied before validation (opt-in)
    ordinal_repairs: list[str] = field(default_factory=list)
    accepted: bool = False
    failure_reason: str | None = None


@dataclass
class PipelineResult:
    """End-to-end output of ``generate_plan``."""

    accepted: bool
    output: GeneratorOutput | None
    attempts: list[Attempt]
    elapsed_seconds: float


# --------------------------------------------------------------------------- #
# Prompt rendering                                                             #
# --------------------------------------------------------------------------- #

def _render_generator_msg(
    english: str,
    modes: list[str],
    feedback: str | None,
) -> str:
    parts = [
        "User mission:",
        english.strip(),
        "",
        "Available semantic modes (use ONLY these for start_mode / goal_mode):",
        *(f"- {m}" for m in modes),
    ]
    if feedback:
        parts += [
            "",
            "Verifier feedback (your previous attempt failed — repair every issue below):",
            feedback.strip(),
        ]
    return "\n".join(parts)


def _render_tripartite_msg(english: str, gen: GeneratorOutput) -> str:
    return (
        "[X] Original English Command\n"
        f"{english.strip()}\n\n"
        "[X] Filtered Command\n"
        f"{gen.filtered_command.filtered.strip()}\n\n"
        "[Y] STL Formula\n"
        f"{gen.stl_formula.strip()}\n\n"
        "[Z] JSON Plan\n"
        f"{gen.json_plan.model_dump_json(indent=2)}\n"
    )


def _feedback_for_taxonomy(missing: list[str], taxonomy: ClusterTaxonomy) -> str:
    legal = taxonomy.modes_for_prompt()
    return (
        f"TAXONOMY FAIL: the following start_mode / goal_mode values are NOT in "
        f"the per-environment taxonomy and must be replaced: {missing}.\n"
        f"Legal modes: {legal}"
    )


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #

#: Single source of truth for the generator retry cap (cli.py and
#: config/nl_planner.yaml should agree with it). Each retry costs ~2 LLM calls
#: (generator + tripartite), so 3 attempts is ~6 calls worst case.
DEFAULT_MAX_ATTEMPTS = 3

#: Formula-side rejections get their OWN budget. Every plan-side validator runs BEFORE
#: the syntax gate, so a formula rejection discards a plan that had already passed all of
#: them; with a shared counter, formula repairs consume the retries the PLAN needs.
#: Separating the counters lets a malformed formula cost a formula retry instead of a
#: whole plan.
DEFAULT_MAX_FORMULA_ATTEMPTS = 2


def generate_plan(
    english: str,
    *,
    taxonomy: ClusterTaxonomy,
    agents=None,
    model_id: str | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    max_formula_attempts: int = DEFAULT_MAX_FORMULA_ATTEMPTS,
    repair_ordinals: bool = True,
    verify_syntax: bool = True,
    verify_tripartite: bool = True,
    cue_vocab: dict[str, tuple[str, ...]] | None | str = _VOCAB_FROM_DEPLOYMENT,
) -> PipelineResult:
    """Run the generate/verify/retry loop for one English mission.

    Args:
        english: free-text mission, e.g. ``"cross the bridge, take the long road if blocked"``.
        taxonomy: legal semantic modes for the current environment.
        agents: pre-built ``AgentBundle`` (if None, one is built from ``model_id``).
        model_id: ``"provider:model"`` string forwarded to ``build_agents``.
        max_attempts: hard cap on PLAN-side retries; defaults to
            ``DEFAULT_MAX_ATTEMPTS`` (3 keeps p99 latency bounded).
        max_formula_attempts: additional retries reserved for formula-side
            rejections (syntax, tripartite). These do not consume the plan
            budget, because by the time either fires the plan has already
            passed every plan-side validator.
        verify_syntax / verify_tripartite: gate off either verifier (debug only).
        cue_vocab: which cues this deployment can answer. Defaults to
            ``cue_vocab.vocab_for_deployment()``, which follows ``CUE_SOURCE``
            exactly as ``stl_ablation.run_one`` does. Pass ``None`` for a
            free-text (VLM) deployment or a translation-only caller; pass a
            dict to supply a non-CARLA vocabulary.

    Returns:
        A ``PipelineResult`` whose ``output`` is the accepted ``GeneratorOutput``.
        Raises ``PlanGenerationError`` if no attempt is accepted within
        ``max_attempts``.
    """
    if agents is None:
        from .agents import build_agents, DEFAULT_MODEL_ID
        agents = build_agents(model_id or DEFAULT_MODEL_ID)

    # RESOLVED ONCE, AND LOGGED: a gate that is silently off is
    # indistinguishable from one that passes.
    from .cue_vocab import validate_plan_cues, vocab_for_deployment
    _vocab = (vocab_for_deployment() if cue_vocab == _VOCAB_FROM_DEPLOYMENT
              else cue_vocab)

    started = time.monotonic()
    attempts: list[Attempt] = []
    feedback: str | None = None

    plan_tries = 0            # rejections attributable to the PLAN
    formula_tries = 0         # rejections attributable to the FORMULA
    i = 0

    while plan_tries < max_attempts and formula_tries <= max_formula_attempts:
        i += 1
        user_msg = _render_generator_msg(english, taxonomy.modes_for_prompt(), feedback)
        attempt = Attempt(iteration=i, generator_input=user_msg)

        try:
            gen_result = agents.generator.run_sync(user_msg)
            gen: GeneratorOutput = gen_result.output  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 - one LLM call, keep going
            attempt.generator_error = repr(exc)
            attempt.failure_reason = f"generator exception: {exc!r}"
            attempts.append(attempt)
            feedback = (
                "GENERATOR EXCEPTION: your previous output failed to parse against "
                f"the schema or to satisfy the in-schema validators ({exc!s}). "
                "Re-emit, paying particular attention to: at-most-one 'default' "
                "branch per decision step, branch nesting <= 3, every step has "
                "integer step >=0."
            )
            plan_tries += 1
            continue
        attempt.generator_output = gen

        missing = validate_plan_modes(gen.json_plan, taxonomy)
        attempt.taxonomy_missing = missing
        if missing:
            attempt.failure_reason = f"taxonomy fail: {missing}"
            attempts.append(attempt)
            feedback = _feedback_for_taxonomy(missing, taxonomy)
            plan_tries += 1
            continue

        # Modes all exist — but do they CHAIN? A plan can name only legal modes and
        # still describe a route no map affords (e.g. `junction -> junction` where no
        # junction is adjacent to another): schema valid, modes valid, STL syntax valid,
        # and not executable on the map. Same class of bug as an unreachable step, so it
        # is caught and fed back the same way rather than discovered at drive time.
        bad_transitions = validate_plan_transitions(gen.json_plan)
        attempt.bad_transitions = bad_transitions
        if bad_transitions:
            attempt.failure_reason = f"transition fail: {bad_transitions}"
            attempts.append(attempt)
            feedback = ("TRANSITION FAIL: the plan names legal modes but the steps do "
                        "not chain into a route.\n" + "\n".join(f"- {e}" for e in bad_transitions))
            plan_tries += 1
            continue

        # A branching step that also carries a prose landmark cue advances on
        # something nothing can answer, so it never advances at all (e.g.
        # `transition_cue='decision point'`: the branch resolves correctly and the robot
        # then drives past its target until the cue budget ends the run).
        # REPAIR BEFORE VALIDATING, exactly as `repair_cue_ordinal` does below. The
        # validator's own message says the fix ("set transition_cue to null and let the
        # branches decide"), there is one repair, and it needs no model -- so spending a
        # retry re-rolling an otherwise-correct plan to delete one field is waste.
        if repair_ordinals:
            import json as _j
            from .schemas import NavPlan as _NP
            _d = _j.loads(gen.json_plan.model_dump_json())
            _fixed = repair_prose_decision_cue(_d)
            _fixed += repair_place_detect_to_traverse(_d)
            _fixed += repair_medium_detect_cue(_d)
            if _fixed:
                gen.json_plan = _NP(**_d)
                attempt.cue_repairs = _fixed
        bad_cues = validate_decision_cues(gen.json_plan)
        attempt.bad_cues = bad_cues
        if bad_cues:
            attempt.failure_reason = f"cue fail: {bad_cues}"
            attempts.append(attempt)
            feedback = ("CUE FAIL: a step that branches also asks its own question, "
                        "in prose nothing can answer.\n"
                        + "\n".join(f"- {e}" for e in bad_cues))
            plan_tries += 1
            continue

        # The count is in the step index OR in cue_ordinal, never both. A plan that
        # double-counts passes every other validator and drives past its target.
        # Deterministic repair before the retry (default ON). `validate_cue_ordinal`
        # calls the extra ordinal "redundant", which means there is exactly one fix and
        # no model is needed to choose it; a retry would re-roll the whole plan, risking
        # the parts that were already correct. Linear plans rarely produce the
        # double-counted ordinal; tree plans produce it often enough that, unrepaired,
        # it is a major cause of exhausted retry budgets.
        if repair_ordinals:
            import json as _json
            from .schemas import NavPlan as _NavPlan
            as_dict = _json.loads(gen.json_plan.model_dump_json())
            repaired = repair_cue_ordinal(as_dict)
            if repaired:
                gen.json_plan = _NavPlan(**as_dict)
                attempt.ordinal_repairs = repaired

        bad_ordinals = validate_cue_ordinal(gen.json_plan)
        attempt.bad_ordinals = bad_ordinals
        if bad_ordinals:
            attempt.failure_reason = f"ordinal fail: {bad_ordinals}"
            attempts.append(attempt)
            feedback = ("ORDINAL FAIL: the plan counts the same thing twice.\n"
                        + "\n".join(f"- {e}" for e in bad_ordinals))
            plan_tries += 1
            continue

        # CUES THIS DEPLOYMENT CANNOT ANSWER. Mirrors the check in
        # `stl_ablation.run_one`; both paths must enforce it, since planner_node.py
        # uses this one.
        #
        # An unanswerable cue does not FAIL at runtime, it HANGS: e.g. `Detect(Junction)`
        # leaves step 0 never completing (see cue_vocab.py). Retrying generation is
        # cheap; a deadlocked robot is not.
        #
        # THE FEEDBACK NAMES THE ANSWERABLE SET, because the prompt does not: v5 tells the
        # model "any concrete visual landmark works", which is true of a VLM and false of
        # this publisher. Until that is reconciled the retry is where the model finds out.
        bad_vocab = validate_plan_cues(gen.json_plan, _vocab)
        attempt.bad_vocab = bad_vocab
        if bad_vocab:
            attempt.failure_reason = f"cue vocab: {bad_vocab}"
            attempts.append(attempt)
            feedback = ("CUE VOCABULARY FAIL: this deployment cannot answer these cues. "
                        "A cue it cannot answer does not fail -- the step waits forever.\n"
                        + "\n".join(f"- {e}" for e in bad_vocab)
                        + (f"\n\nIt answers exactly: {sorted(_vocab)}" if _vocab else ""))
            plan_tries += 1
            continue

        if verify_syntax:
            # Deterministic regex-based syntax check. Catches the failure
            # modes seen with the former LLM agent (unbalanced parens,
            # mode-names leaking into predicate args, invented \Phi_{...}
            # macros, ASCII operators, malformed \text{...} wrappers). It's
            # much faster than an LLM round-trip and cannot hallucinate.
            # The previous LLM-based syntax verifier rejected valid formulas
            # like `Detect(Wall)` (a single-word CamelCase identifier), so it
            # is no longer used for syntax. The
            # tripartite verifier still uses an LLM (it's a semantic check,
            # not a syntactic one).
            # `modes=` additionally rejects a predicate argument that names one
            # of THIS environment's modes (`Detect(Path)` against a taxonomy with
            # a `path` mode). Well-formed, unanswerable, and the step deadlocks.
            # ALLOW_EMPTY_STL, HONOURED HERE TOO. `quick_syntax_check("")` returns
            # `Rule 0: empty STL formula`, while generator.md says "AN EMPTY FORMULA IS
            # CORRECT AND EXPECTED". The same flag is honoured in stl_ablation.run_one;
            # the two paths must agree.
            #
            # Without it, e.g. on "turn right at any intersection with a cone, but never
            # turn at the second", a model forced to produce a NON-empty formula tends to
            # put the prohibition in the formula as an invented predicate such as
            # `G(!(AtJunction_2 & TurnRight))`, which is not a \Phi_{} macro, so the
            # allow-list never sees it and it compiles to no monitor -- instead of in
            # `forbid_instances`, the field that actually executes.
            _allow_empty = os.environ.get("ALLOW_EMPTY_STL") == "1"
            quick_ok, quick_err = (
                (True, None) if (_allow_empty and not (gen.stl_formula or "").strip())
                else quick_syntax_check(gen.stl_formula,
                                        modes=taxonomy.modes_for_prompt())
            )
            attempt.syntax = SyntaxVerdict(ok=quick_ok, error=quick_err)
            if not quick_ok:
                attempt.failure_reason = f"syntax fail: {quick_err}"
                attempts.append(attempt)
                feedback = (
                    "SYNTAX FAIL (deterministic check; the rule numbers refer "
                    "to verify_syntax.md): "
                    f"{quick_err}\nFix the STL formula and retry."
                )
                formula_tries += 1
                continue

        if verify_tripartite:
            try:
                tri_result = agents.tripartite_verifier.run_sync(
                    _render_tripartite_msg(english, gen)
                )
                tri: TripartiteVerdict = tri_result.output  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001
                attempt.failure_reason = f"tripartite verifier exception: {exc!r}"
                attempts.append(attempt)
                feedback = f"TRIPARTITE VERIFIER EXCEPTION: {exc!s}"
                formula_tries += 1
                continue
            attempt.tripartite = tri
            if not tri.ok:
                attempt.failure_reason = f"tripartite fail: {tri.notes}"
                attempts.append(attempt)
                feedback = (
                    f"TRIPARTITE FAIL "
                    f"(english_stl={tri.english_stl_aligned}, "
                    f"stl_json={tri.stl_json_aligned}, "
                    f"english_json={tri.english_json_aligned}): {tri.notes}"
                )
                formula_tries += 1
                continue

        attempt.accepted = True
        attempts.append(attempt)
        return PipelineResult(
            accepted=True,
            output=gen,
            attempts=attempts,
            elapsed_seconds=round(time.monotonic() - started, 3),
        )

    raise PlanGenerationError(
        f"plan generation did not converge within {plan_tries} plan attempt(s) "
        f"and {formula_tries} formula attempt(s). "
        f"Last failure: {attempts[-1].failure_reason if attempts else '(no attempts logged)'}"
    )


# --------------------------------------------------------------------------- #
# Trace serialization (CLI + tests)                                            #
# --------------------------------------------------------------------------- #

def attempt_to_dict(attempt: Attempt) -> dict[str, Any]:
    return {
        "iteration": attempt.iteration,
        "generator_input": attempt.generator_input,
        "generator_output": (
            attempt.generator_output.model_dump() if attempt.generator_output else None
        ),
        "generator_error": attempt.generator_error,
        "syntax": attempt.syntax.model_dump() if attempt.syntax else None,
        "tripartite": attempt.tripartite.model_dump() if attempt.tripartite else None,
        "taxonomy_missing": attempt.taxonomy_missing,
        "bad_transitions": attempt.bad_transitions,
        "bad_cues": attempt.bad_cues,
        "bad_vocab": attempt.bad_vocab,
        "accepted": attempt.accepted,
        "failure_reason": attempt.failure_reason,
    }


def result_to_dict(result: PipelineResult) -> dict[str, Any]:
    return {
        "accepted": result.accepted,
        "output": result.output.model_dump() if result.output else None,
        "attempts": [attempt_to_dict(a) for a in result.attempts],
        "elapsed_seconds": result.elapsed_seconds,
    }


__all__ = [
    "Attempt",
    "PipelineResult",
    "generate_plan",
    "attempt_to_dict",
    "result_to_dict",
]
