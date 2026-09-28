"""Every in-context example must pass the gates the model's output has to pass.

Examples are what a model imitates, so an example that breaks the prompt's own rules
teaches the model to break them. Known cases in `generator.md` (recorded in KNOWN_BAD):

  * Examples 1 and 2 use `\\Big(`, which the BRACKET DISCIPLINE section bans outright.
  * Example 1 uses `Detect(Inside Intersection)` — a bare predicate argument containing a
    space, listed under "what you must NEVER do" — and it cues on the medium the robot is
    already inside (mistake 5: a step that can never advance).
  * Example 2 sets `transition_cue: "decision point"` on a branching step, which mistake 6
    forbids and `repair_prose_decision_cue()` exists to strip back out.

Each is mechanically checkable against the validators already in the tree. A rule the
examples contradict is a rule that does not apply.
"""
from __future__ import annotations

import json
import os
import re

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = os.path.join(os.path.dirname(HERE), "nl_planner", "prompts")

#: Prompts whose examples are held to the rules. The ablation arms are deliberately
#: degraded and are exempt; these are the ones that generate production plans.
CHECKED = ("generator.md", "generator_v3.md", "generator_v4.md", "generator_v5.md")

#: KNOWN DEFECTS IN THE PROMPTS, recorded rather than hidden. xfail(strict) means the
#: suite stays green AND reports the moment one is fixed, so this list cannot rot into a
#: list of things nobody repaired.
#:
#: `generator_v3.md` is kept FROZEN so results generated from it stay reproducible; its
#: remaining defects are fixed in v4.
KNOWN_BAD: dict[tuple[str, str], str] = {
    ("test_example_formula_avoids_the_brackets_the_prompt_bans", "generator.md#1"):
        r"Example 1 uses \Big(, banned at generator.md:228",
    ("test_example_formula_avoids_the_brackets_the_prompt_bans", "generator.md#2"):
        r"Example 2 uses \Big(, banned at generator.md:228",
    ("test_example_branching_step_has_no_prose_cue", "generator.md#2"):
        "Example 2 sets transition_cue='decision point', banned by mistake 6",
    ("test_example_predicate_args_are_well_formed", "generator.md#1"):
        "Detect(Inside Intersection) — bare arg with a space, banned at generator.md:262",
    ("test_example_predicate_args_are_well_formed", "generator.md#2"):
        "Detect(End of Detour) / Detect(End of Bridge) — bare args with spaces",
    ("test_example_predicate_args_are_well_formed", "generator_v3.md#2"):
        "inherited from generator.md Example 2; fixed in v4, v3 frozen mid-campaign",
    # Found by the two checks below: examples that model a plan the pipeline itself
    # rejects, or a cue the deployment cannot answer.
    ("test_example_passes_the_validators_that_actually_gate", "generator.md#2"):
        "Example 2's car-on-road modes are absent from the combined taxonomy",
    ("test_example_passes_the_validators_that_actually_gate", "generator.md#3"):
        "Example 4's car-on-road modes are absent from the combined taxonomy",
    ("test_example_passes_the_validators_that_actually_gate", "generator_v3.md#3"):
        "inherited from generator.md; fixed in v4, v3 frozen mid-campaign",
    ("test_example_cues_are_answerable", "generator.md#2"):
        "Detect(End of Detour) / Detect(End of Bridge) -- no deployment publishes either",
    ("test_example_cues_are_answerable", "generator.md#3"):
        "Example 4 cues on landmarks this deployment does not answer",
    ("test_example_cues_are_answerable", "generator.md#4"):
        "Example 5 cues Detect(Alley) / Detect(StoneWall) / Detect(Doorway)",
    ("test_example_cues_are_answerable", "generator_v3.md#2"):
        "inherited from generator.md; fixed in v4, v3 frozen mid-campaign",
    ("test_example_cues_are_answerable", "generator_v3.md#3"):
        "inherited from generator.md; fixed in v4, v3 frozen mid-campaign",
    ("test_example_cues_are_answerable", "generator_v3.md#4"):
        "inherited from generator.md; fixed in v4, v3 frozen mid-campaign",
}


def _params(test: str):
    """Parametrize entries for `test`, with recorded defects marked xfail(strict)."""
    out = []
    for name, doc in ALL:
        why = KNOWN_BAD.get((test, name))
        marks = [pytest.mark.xfail(strict=True,
                                   reason=f"known defect: {why}")] if why else []
        out.append(pytest.param(name, doc, marks=marks, id=name))
    return out

_BLOCK = re.compile(r"```json\n(.*?)\n```", re.DOTALL)

#: THE GENERIC MEDIUM, not every noun that maps to a traversal concept. The distinction is
#: the prompt's own: a bridge IS a passage, and `Detect(Bridge)` is explicitly endorsed --
#: "the mode is the shape, the predicate is the thing" -- because you APPROACH a bridge and
#: can see it coming. What can never change is the generic name of the surface you are
#: already on, which is what mistake 5 describes as deadlocking.
_MEDIUM = frozenset({
    "path", "roadon", "road", "onpath", "openspace", "spaceopen", "open", "passage",
    "corridor", "alongwall", "wallalong", "alongedge", "edgealong", "walkway",
    "sidewalk", "trail", "lane", "pavement",
})


def _examples(fname: str) -> list[tuple[str, dict]]:
    path = os.path.join(PROMPTS, fname)
    if not os.path.exists(path):
        return []
    out = []
    for i, m in enumerate(_BLOCK.finditer(open(path).read()), 1):
        try:
            out.append((f"{fname}#{i}", json.loads(m.group(1))))
        except json.JSONDecodeError as exc:
            pytest.fail(f"{fname} example {i} is not valid JSON: {exc}")
    return out


ALL = [e for f in CHECKED for e in _examples(f)]
assert ALL, "no ```json examples found in any checked prompt"


def _walk(steps):
    for st in steps or []:
        yield st
        for b in (st.get("branches") or []):
            yield from _walk(b.get("sub_plan"))


@pytest.mark.parametrize("name,doc", ALL, ids=[n for n, _ in ALL])
def test_example_validates_as_generator_output(name, doc):
    """The schema the framework injects must accept the example verbatim."""
    from nl_planner.schemas import GeneratorOutput
    GeneratorOutput.model_validate(doc)


@pytest.mark.parametrize("name,doc", ALL, ids=[n for n, _ in ALL])
def test_example_formula_passes_the_syntax_gate(name, doc):
    """An example formula the verifier would reject teaches a rejected formula.

    The empty formula is exempt: `quick_syntax_check` fails it by Rule 0 while the prompt
    calls it correct and expected. That conflict is the subject of ALLOW_EMPTY_STL and is
    not this test's business.
    """
    from nl_planner.stl_syntax import quick_syntax_check
    stl = (doc.get("stl_formula") or "").strip()
    if not stl:
        return
    ok, err = quick_syntax_check(stl)
    assert ok, f"{name}: the prompt's own example would be rejected — {err}"


@pytest.mark.parametrize("name,doc", _params("test_example_formula_avoids_the_brackets_the_prompt_bans"))
def test_example_formula_avoids_the_brackets_the_prompt_bans(name, doc):
    r"""`\Big(` / `\left(` are banned for making bracket counting harder."""
    stl = doc.get("stl_formula") or ""
    bad = [t for t in (r"\Big", r"\big", r"\left", r"\right") if t in stl]
    assert not bad, f"{name}: uses {bad}, which the prompt forbids"


@pytest.mark.parametrize("name,doc", _params("test_example_branching_step_has_no_prose_cue"))
def test_example_branching_step_has_no_prose_cue(name, doc):
    """Mistake 6: a branching step's own cue is unanswerable, so the step never completes."""
    from nl_planner.taxonomy import repair_prose_decision_cue
    plan = doc.get("json_plan") or {}
    repaired = repair_prose_decision_cue(json.loads(json.dumps(plan)))
    assert not repaired, f"{name}: a repair function would have to fix this example — {repaired}"


@pytest.mark.parametrize("name,doc", _params("test_example_predicate_args_are_well_formed"))
def test_example_predicate_args_are_well_formed(name, doc):
    r"""A bare `Detect(...)` argument may not contain spaces, colons or punctuation.

    `generator.md` Example 1 shipped `Detect(Inside Intersection)`, which the same file
    lists under "what you must NEVER do".
    """
    bad = []
    for st in _walk((doc.get("json_plan") or {}).get("steps")):
        cue = st.get("transition_cue") or ""
        for m in re.finditer(r"\b(?:Detect|Bearing)\(([^)]*)\)", cue):
            arg = m.group(1)
            if not re.fullmatch(r"[A-Z][A-Za-z0-9]*", arg):
                bad.append(cue)
    assert not bad, f"{name}: malformed predicate argument(s) {bad}"


@pytest.mark.parametrize("name,doc", ALL, ids=[n for n, _ in ALL])
def test_example_never_cues_on_the_medium_it_travels_in(name, doc):
    """Mistake 5: `Detect(Path)` can never change, so the step can never advance."""
    bad = []
    for st in _walk((doc.get("json_plan") or {}).get("steps")):
        cue = st.get("transition_cue") or ""
        for m in re.finditer(r"\bDetect\(([^)]*)\)", cue):
            flat = re.sub(r"[^a-z]", "", m.group(1).lower())
            if flat in _MEDIUM:
                bad.append(cue)
    assert not bad, f"{name}: cues on the medium it is already inside — {bad}"


def test_v4_demonstrates_the_plan_level_constraint_fields():
    """The fields with no example at all were the fields nobody ever set.

    `generator.md` describes `forbid_modes` / `require_modes` in prose and demonstrates
    neither in its examples, and fields without an example tend never to be set. This
    asserts the demonstration exists.
    """
    ex = [d for n, d in _examples("generator_v4.md")]
    assert ex, "generator_v4.md has no examples"
    plans = [d.get("json_plan") or {} for d in ex]
    assert any(p.get("forbid_modes") for p in plans), "no example sets forbid_modes"
    assert any(p.get("require_modes") for p in plans), "no example sets require_modes"


def test_v4_demonstrates_the_empty_formula():
    """A rule the model does not follow from prose alone needs a worked example."""
    ex = [d for n, d in _examples("generator_v4.md")]
    assert any(not (d.get("stl_formula") or "").strip() for d in ex), \
        "no example shows the empty formula the decision procedure asks for"


@pytest.mark.parametrize("name,doc", _params("test_example_passes_the_validators_that_actually_gate"))
def test_example_passes_the_validators_that_actually_gate(name, doc):
    """Schema and syntax are two of six gates. These are the other four.

    `run_one` rejects a plan on `validate_plan_modes`, `validate_plan_transitions`,
    `validate_decision_cues` and `validate_plan_cues`. An example that models a shape the
    pipeline itself
    refuses teaches the model to be rejected.

    Modes are checked against the COMBINED taxonomy, which is what the harness injects;
    an example naming a mode outside it would not materialise.
    """
    from nl_planner.taxonomy import (load_taxonomy, validate_decision_cues,
                                     validate_plan_modes, validate_plan_transitions)
    tax_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))),
                            "bev_pipeline", "config", "cluster_map.combined.yaml")
    plan = doc.get("json_plan") or {}
    from nl_planner.schemas import NavPlan
    nav = NavPlan.model_validate(plan)
    assert not validate_plan_transitions(nav), \
        f"{name}: illegal transition — {validate_plan_transitions(nav)}"
    assert not validate_decision_cues(nav), \
        f"{name}: bad decision cue — {validate_decision_cues(nav)}"
    if os.path.exists(tax_path):
        bad = validate_plan_modes(nav, load_taxonomy(tax_path))
        assert not bad, f"{name}: names a mode the combined taxonomy lacks — {bad}"


@pytest.mark.parametrize("name,doc", _params("test_example_cues_are_answerable"))
def test_example_cues_are_answerable(name, doc):
    """A cue no deployment publishes does not fail — it TIMES OUT.

    See `cue_vocab.py`. An example
    teaching `Detect(Plaza)` or `Detect(Bridge)` teaches a plan that stalls at step 0 while
    every other part of it is correct. Advisory mode makes it survivable at generation
    time; it does not make it survivable on the robot.

    `CARLA_GT_VOCAB` IS ONE DEPLOYMENT'S ANSWER SET, not a universal rule -- a campus rig
    publishes different predicates. It is the right one to hold the examples to because it
    is the deployment the corpora in this tree score against, but a prompt written for a
    different rig would be held to that rig's vocabulary instead.
    """
    from nl_planner.cue_vocab import CARLA_GT_VOCAB, validate_plan_cues
    from nl_planner.schemas import NavPlan
    bad = validate_plan_cues(NavPlan.model_validate(doc.get("json_plan") or {}),
                             CARLA_GT_VOCAB)
    assert not bad, f"{name}: unanswerable cue — {bad}"


def test_invented_macro_with_punctuation_is_rejected_by_a_gate():
    r"""A `\Phi_{...}` body containing punctuation passes BOTH gates and then crashes.

    E.g. `\mathbf{G}(\lnot\Phi_{Junc\_Turn,\text{ordinal}=2})`. If `_MACRO_RE` only
    matched `[A-Za-z0-9_\\]+`, a body with a comma or braces would not be seen as a macro
    at all and the closed allow-list would never reject it: `quick_syntax_check` returns
    True, `validate_stl_modes` returns [], and `compile_monitors(parse(...))` raises
    ParseError. The plan scores VALID and the monitor does not exist.

    `_MACRO_RE` therefore matches to the closing brace (allowing one level of nesting, which is
    what a `\text{...}` inside a body looks like), so an invented body reaches the closed
    allow-list and is rejected where it is written. The legitimate macros still pass.
    """
    from nl_planner.stl_syntax import quick_syntax_check
    bad = r"\mathbf{G}(\lnot\Phi_{Junc\_Turn,\text{ordinal}=2})"
    ok, _ = quick_syntax_check(bad)
    assert not ok, "an invented macro body with punctuation should be rejected"


@pytest.mark.parametrize("formula", [
    r"\mathbf{G}\Phi_{Path}",
    r"\Phi_{Junc\_Turn} \land \mathbf{F}\Phi_{Open\_Space}",
    r"\Phi_{Int\_Turn}",
    r"\mathbf{G}(\lnot\Phi_{Passage})",
])
def test_legitimate_macros_still_pass_after_the_regex_widening(formula):
    """The guard on the fix above: widening `_MACRO_RE` must not start rejecting real ones."""
    from nl_planner.stl_syntax import quick_syntax_check
    ok, err = quick_syntax_check(formula)
    assert ok, f"{formula} should pass but was rejected: {err}"
