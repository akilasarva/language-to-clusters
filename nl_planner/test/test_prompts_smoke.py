"""Smoke test: all known prompts load and contain non-trivial content."""

from __future__ import annotations

import pytest

from nl_planner.prompts import KNOWN_PROMPTS, list_prompts, load_prompt


@pytest.mark.parametrize("name", KNOWN_PROMPTS)
def test_known_prompts_load(name):
    body = load_prompt(name)
    assert len(body) > 200, f"prompt {name!r} suspiciously short: {len(body)} bytes"
    assert "Role" in body or "role" in body.lower(), (
        f"prompt {name!r} doesn't look like a prompt"
    )


def test_unknown_prompt_raises():
    with pytest.raises(FileNotFoundError):
        load_prompt("does_not_exist")


def test_list_prompts_finds_each_known_one():
    found = set(list_prompts())
    missing = set(KNOWN_PROMPTS) - found
    assert not missing, f"missing prompt files on disk: {missing}"


# --------------------------------------------------------------------------- #
# STL syntax contract — keeps generator.md and verify_syntax.md in lockstep   #
# --------------------------------------------------------------------------- #
#
# The generator and verifier prompts can drift: e.g. generator examples using
# `\text{Detect}(\text{X})` while the verifier only lists bare `Detect(X)` makes
# every branching plan fail with "syntax fail: offending substring \text{Detect}".
# These tests pin the contract so it can't silently regress.

GENERATOR_REQUIRED = [
    # Cheatsheet must announce both bare and \text{}-wrapped predicate forms.
    "STL Syntax Cheatsheet",
    "\\text{Detect}(\\text{Stop Sign})",
    "Detect(StopSign)",
    # Operator allow-list must be explicit so the LLM doesn't hallucinate
    # bare `U` / `F` / `&&`.
    "\\mathbf{U}",
    "\\mathbf{F}",
    "\\land",
    "\\lor",
    "\\lnot",
    # Macro allow-list must be CLOSED with an explicit warning against
    # invented macros (e.g. \Phi_{GoForward}, \Phi_{TurnRight}), which the
    # LLM tends to invent.
    "DO NOT invent new macros",
    # Cross-vocabulary boundary: mode names (with spaces / colons) must NEVER
    # appear inside Detect(...) / Bearing(...) arguments, a common failure mode.
    "Predicate Args vs Mode Names",
    "Common Generator Mistakes",
    "Detect(Open Space)",  # cited as the canonical wrong form
    # --- goal_mode selection (the specificity rule) ---
    # The LLM must emit the FINEST mode and must NOT try to be safe by
    # downgrading; the taxonomy owns the coarser fallbacks. Without this the
    # generator says "Road: On" for "pass the blue building" and the
    # along_edge/landmark distinction is lost before it reaches the robot.
    "Emit the FINEST Match",
    "the taxonomy already accepts",   # the "don't downgrade" instruction
    "Along Wall",
    # --- trigger typology ---
    "`traverse`",
    "`landmark`",
    "`topology`",
    "Bearing(Right) completed",
    # --- counting: topology unroll vs landmark ordinal ---
    # These are different operations and conflating them produces a plan that
    # stalls at the first bench waiting for a mode change that never happens.
    "cue_ordinal",
    "do NOT unroll",
]

VERIFY_SYNTAX_REQUIRED = [
    # Numbered rules — keep them findable.
    "## 1. Balanced Brackets",
    "## 2. Allowed Macros",
    "## 3. Allowed Predicates",
    "## 4. Allowed Temporal Operators",
    "## 5. Allowed Boolean Operators",
    # All four predicate-styling shapes must be explicitly accepted.
    "Detect(StopSign)",
    "\\text{Detect}(StopSign)",
    "Detect(\\text{Stop Sign})",
    "\\text{Detect}(\\text{Stop Sign})",
    # Phrases inside \text{...} (spaces) must be explicitly permitted —
    # e.g. "Intersecting Road".
    "\\text{Intersecting Road}",
    # Each section must have at least one valid + invalid example to anchor
    # few-shot behavior.
    "# Valid Examples",
    "# Invalid Examples",
    # An anti-false-positive section so the verifier doesn't reject
    # Detect(\text{Wall}) etc.
    "Notes for the Verifier",
    "Detect(\\text{Wall})",
    # Structural argument forms must be named A and B (decision procedure).
    "Form A",
    "Form B",
]

VERIFY_TRIPARTITE_REQUIRED = [
    # The tripartite verifier must explicitly state that styling-only
    # predicate differences are NOT alignment failures.
    "Predicate Equivalence",
    "\\text{Detect}(\\text{StopSign})",
    "style-only",
    # The verifier's generic "are counts unrolled?" rule would reject a CORRECT
    # cue_ordinal plan (one step for "the 2nd bench") as a missing unroll. It
    # must know the two kinds of count apart, or every ordinal mission fails
    # verification — the same false-positive class as prompt drift.
    "## Counting",
    "cue_ordinal",
    # It must also not flag the generator's deliberate specificity (emitting
    # "Along Wall" rather than the safer "Road: On") as over-committing.
    "## Mode Specificity",
]


@pytest.mark.parametrize("anchor", GENERATOR_REQUIRED)
def test_generator_prompt_contract(anchor):
    body = load_prompt("generator")
    assert anchor in body, (
        f"generator.md is missing required anchor {anchor!r}. "
        f"This likely means generator.md and verify_syntax.md have drifted "
        f"out of sync. See test_prompts_smoke.py for context."
    )


@pytest.mark.parametrize("anchor", VERIFY_SYNTAX_REQUIRED)
def test_verify_syntax_prompt_contract(anchor):
    body = load_prompt("verify_syntax")
    assert anchor in body, (
        f"verify_syntax.md is missing required anchor {anchor!r}. "
        f"The syntax verifier must accept every form the generator produces."
    )


@pytest.mark.parametrize("anchor", VERIFY_TRIPARTITE_REQUIRED)
def test_verify_tripartite_prompt_contract(anchor):
    body = load_prompt("verify_tripartite")
    assert anchor in body, (
        f"verify_tripartite.md is missing required anchor {anchor!r}. "
        f"This guard prevents the tripartite verifier from rejecting plans "
        f"on stylistic-only predicate differences."
    )


def test_generator_and_syntax_verifier_use_same_predicate_examples():
    """Both prompts cite at least one identical predicate token.

    This is a coarse cross-prompt sanity check: if both prompts agree on
    e.g. ``Detect(StopSign)`` as a representative example, the LLM agents
    see the same canonical form when reading their system prompts at
    agent-build time.
    """
    gen = load_prompt("generator")
    ver = load_prompt("verify_syntax")
    shared_examples = [
        "Detect(StopSign)",
        "\\text{Detect}(\\text{Stop Sign})",
        "\\mathbf{F}",
        "\\mathbf{U}",
        "\\land",
        "\\lor",
        "\\Phi_{Bridge}",
    ]
    for ex in shared_examples:
        assert ex in gen, f"generator.md lacks shared example {ex!r}"
        assert ex in ver, f"verify_syntax.md lacks shared example {ex!r}"


# --------------------------------------------------------------------------- #
# The in-context examples must themselves be valid                            #
# --------------------------------------------------------------------------- #

def _generator_json_examples():
    """Every ```json fenced block in generator.md, parsed."""
    import json
    import re

    body = load_prompt("generator")
    blocks = re.findall(r"```json\n(.*?)\n```", body, re.S)
    return [(i, json.loads(b)) for i, b in enumerate(blocks)]


def test_generator_has_json_examples():
    examples = _generator_json_examples()
    assert len(examples) >= 3, (
        f"expected at least 3 worked JSON examples in generator.md, found "
        f"{len(examples)} — few-shot coverage has regressed"
    )


def test_every_in_context_example_validates_against_the_real_schema():
    """An example the schema rejects teaches the LLM to emit invalid plans.

    Same class of bug as prompt drift: a prompt showing a form the rest of the
    pipeline will not accept makes every generation fail.
    """
    from nl_planner.schemas import GeneratorOutput

    for i, obj in _generator_json_examples():
        GeneratorOutput.model_validate(obj)   # must not raise


def test_every_in_context_example_passes_the_real_syntax_checker():
    """The STL in each example must survive quick_syntax_check.

    If an example's formula would be rejected, the generator is being taught to
    produce something the pipeline rejects on attempt 1 — burning retries and
    possibly never converging.
    """
    from nl_planner.schemas import GeneratorOutput
    from nl_planner.stl_syntax import quick_syntax_check

    for i, obj in _generator_json_examples():
        gen = GeneratorOutput.model_validate(obj)
        ok, err = quick_syntax_check(gen.stl_formula)
        assert ok, f"generator.md example {i} has invalid STL: {err}"


def test_ordinal_example_uses_cue_ordinal_not_unrolled_steps():
    """Pins the distinction that Constraint 2(b) teaches.

    "the 2nd bench" must be ONE step carrying cue_ordinal=2 — not two steps.
    Two steps would make brain wait for a mode transition at the first bench
    that never occurs, stalling the plan.
    """
    from nl_planner.schemas import GeneratorOutput

    ordinal_steps = []
    for _, obj in _generator_json_examples():
        gen = GeneratorOutput.model_validate(obj)
        for step in gen.json_plan.steps:
            if step.cue_ordinal is not None:
                ordinal_steps.append(step)

    assert ordinal_steps, (
        "no in-context example demonstrates cue_ordinal — the LLM has nothing to "
        "imitate for 'the 2nd bench' and will unroll it into separate steps"
    )
    for step in ordinal_steps:
        assert step.cue_ordinal >= 2
        assert (step.trigger or "landmark") == "landmark", (
            "cue_ordinal is only meaningful on a landmark step"
        )
        assert step.start_mode == step.goal_mode, (
            "an ordinal-counting step must not change mode — the robot passes the "
            "landmark without a traversal-state change"
        )


def test_trigger_values_in_examples_are_legal():
    from nl_planner.schemas import TRIGGERS, GeneratorOutput

    for _, obj in _generator_json_examples():
        gen = GeneratorOutput.model_validate(obj)
        for step in gen.json_plan.steps:
            if step.trigger is not None:
                assert step.trigger in TRIGGERS


STL_MARKERS = ("\\Phi", "\\mathbf", "\\land", "\\lor", "\\lnot", "STL",
               "Signal Temporal Logic", "stl_formula")


def test_the_ablated_prompt_teaches_no_stl_in_what_the_model_actually_sees():
    """The `none` arm must not be taught the thing it is meant never to have heard of.

    Guards against STL leaking into the no-STL prompt: a doc header naming the macro
    vocabulary, an ordinal rule stated as an STL formula, a clause telling the model to
    express branches "using the macros above" (macros this prompt does not contain), or
    formula-style guidance after Example 4.

    Asserted against `load_prompt`, not the file, because the header is stripped at load
    and the file legitimately keeps its provenance note.
    """
    from nl_planner.prompts import load_prompt
    body = load_prompt("generator_nostl")
    found = sorted({m for m in STL_MARKERS if m in body})
    assert not found, f"generator_nostl.md still teaches STL: {found}"


def test_the_doc_header_never_reaches_the_model():
    from nl_planner.prompts import load_prompt
    from nl_planner.prompts import _strip_doc_header
    assert not load_prompt("generator_nostl").lstrip().startswith("<!--")
    assert _strip_doc_header("<!-- note -->\n\n# Role\nx").startswith("# Role")
    # a comment that is not a header is left alone -- bodies use them deliberately
    assert _strip_doc_header("# Role\n<!-- keep -->\n") == "# Role\n<!-- keep -->\n"


def test_the_full_prompt_still_does_teach_stl():
    """Guard against 'fixing' the leak by gutting the wrong file."""
    from nl_planner.prompts import load_prompt
    assert "\\Phi" in load_prompt("generator")


def test_the_flat_prompt_is_the_no_branch_control():
    """The structure-control arm.

    `full` and `none` both have full plan-tree structure, so neither can say whether
    STRUCTURE buys anything. This prompt is the same as the JSON arm minus exactly one
    construct.
    """
    from nl_planner.prompts import load_prompt
    flat = load_prompt("generator_flat")
    assert "# Branches (conditional plans)" not in flat
    assert "SINGLE LINEAR SEQUENCE" in flat
    assert not any(m in flat for m in STL_MARKERS), "the control must not teach STL either"
    # but it must still teach everything that is NOT the construct under test
    for kept in ("# Available Semantic Modes", "`trigger` — what actually ends the step",
                 "cue_ordinal", "# Plan-Level Constraints"):
        assert kept in flat, f"flat prompt lost non-branch teaching: {kept}"
