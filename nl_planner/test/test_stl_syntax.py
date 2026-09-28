"""Unit tests for nl_planner.stl_syntax.quick_syntax_check.

The check is regex-based and deterministic. The failure modes below are ones
observed from the generator in live planner_node runs.
"""
from __future__ import annotations

import pytest

from nl_planner.stl_syntax import quick_syntax_check


# --------------------------------------------------------------------------- #
# Valid formulas (should pass)                                                 #
# --------------------------------------------------------------------------- #

VALID_FORMULAS = [
    # Bare predicate + macro
    "Detect(StopSign) \\land \\mathbf{F}\\Phi_{Int_Turn}",
    # Same with escaped underscore
    "Detect(StopSign) \\land \\mathbf{F}\\Phi_{Int\\_Turn}",
    # \text{}-wrapped name and arg with spaces
    "\\text{Detect}(\\text{Stop Sign}) \\land \\mathbf{F}\\Phi_{Bridge}",
    # Branch (\lor) with mixed styles
    (
        "(\\text{Detect}(\\text{BlockedBridge}) \\land \\mathbf{F}\\Phi_{LongRoad}) "
        "\\lor (\\lnot \\text{Detect}(\\text{BlockedBridge}) \\land \\mathbf{F}\\Phi_{Bridge})"
    ),
    # \text{} wrapper around argument only
    "Detect(\\text{Wall}) \\lor Detect(\\text{Open Space})",
    # \text{} wrapper around argument with punctuation
    "Detect(\\text{End of Bridge})",
    # Display math wrapper
    "$$\\Phi_{Road} \\ \\mathbf{U} \\ \\text{Detect}(\\text{End of Bridge})$$",
    # Multiple temporal layers
    "\\Phi_{Road} \\ \\mathbf{U} \\ \\big( \\text{Detect}(\\text{StopSign}) \\land \\mathbf{F}\\Phi_{Int\\_Turn} \\big)",
    # Bearing predicate
    "Bearing(Right) \\land \\mathbf{F}\\Phi_{Road}",
]


@pytest.mark.parametrize("stl", VALID_FORMULAS)
def test_valid_formulas_pass(stl: str):
    ok, err = quick_syntax_check(stl)
    assert ok, f"expected pass but got error: {err}\nformula was: {stl}"
    assert err is None


# --------------------------------------------------------------------------- #
# Invalid formulas — each maps to a specific observed failure mode            #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("stl,expect_rule", [
    # Empty
    ("", "Rule 0"),
    ("   ", "Rule 0"),
    # Unbalanced
    ("Detect(Bridge", "Rule 1"),
    ("Detect(Bridge))", "Rule 1"),
    ("\\Phi_{Int_Pass", "Rule 1"),  # missing }
    # ASCII boolean
    ("Detect(Bridge) && \\Phi_{Road}", "Rule 5"),
    ("Detect(Bridge) || Detect(Wall)", "Rule 5"),
    # ASCII temporal — bare F before predicate
    ("F Detect(Bridge)", "Rule 5"),
    # Invented macro
    ("\\Phi_{GoForward} \\land Detect(Bridge)", "Rule 2"),
    ("\\Phi_{TurnRight} \\lor \\Phi_{Bridge}", "Rule 2"),
    # Mode-vocab leak (THE failure mode that prompted the programmatic check)
    ("Detect(Open Space) \\land \\Phi_{Road}", "Rule 3"),
    ("Bearing(Hard Left) \\land \\Phi_{Road}", "Rule 3"),
    # Quoted arg
    ('Detect("OpenSpace") \\land \\Phi_{Road}', "Rule 3"),
    # Punctuation in bare arg
    ("Detect(Open-Space) \\land \\Phi_{Road}", "Rule 3"),
    # Empty predicate arg
    ("Detect() \\land \\Phi_{Road}", "Rule 3"),
])
def test_invalid_formulas_caught(stl: str, expect_rule: str):
    ok, err = quick_syntax_check(stl)
    assert not ok, f"expected fail but got pass for {stl!r}"
    assert err is not None
    assert expect_rule in err, (
        f"expected error to mention {expect_rule!r} for {stl!r}; got {err!r}"
    )


# --------------------------------------------------------------------------- #
# Targeted behavior tests                                                      #
# --------------------------------------------------------------------------- #

def test_camel_case_argument_with_digits_is_form_a():
    # Argument like 'Intersection1' is a single CamelCase token with a digit.
    ok, err = quick_syntax_check("Detect(Intersection1)")
    assert ok, err


def test_form_b_with_basic_punctuation_inside_text():
    # Periods, dashes, commas, colons, slashes are allowed inside \text{...}.
    ok, err = quick_syntax_check("Detect(\\text{End of Bridge, mile 1})")
    assert ok, err


def test_form_b_forbids_latex_command_inside_text():
    ok, err = quick_syntax_check("Detect(\\text{\\Phi_{Road}})")
    assert not ok
    assert "Rule 3" in err


def test_suggestion_mentions_camel_case_alternative():
    # The error message should propose a concrete fix.
    ok, err = quick_syntax_check("Detect(Open Space)")
    assert not ok
    # Suggestion should at least include 'OpenSpace' or the \text{} alternative.
    assert "OpenSpace" in err or "\\text{Open Space}" in err


def test_letter_inside_camel_case_does_not_trip_temporal_check():
    # "F" inside "EndOfBridge" should NOT be flagged as a bare temporal.
    ok, err = quick_syntax_check(
        "Detect(EndOfBridge) \\land \\mathbf{F}\\Phi_{Bridge}"
    )
    assert ok, err


def test_letter_inside_text_wrapper_does_not_trip_temporal_check():
    # "U" inside \text{Underpass} should NOT be flagged.
    ok, err = quick_syntax_check(
        "Detect(\\text{Underpass}) \\land \\mathbf{F}\\Phi_{Road}"
    )
    assert ok, err


def test_strip_display_math_wrapper():
    # The check should not be confused by `$$...$$` wrappers.
    ok, err = quick_syntax_check("$$Detect(Bridge) \\land \\Phi_{Bridge}$$")
    assert ok, err


def test_error_message_quotes_offending_substring():
    # Errors should contain the actual offending substring for the LLM to
    # repair on retry.
    ok, err = quick_syntax_check("Detect(Open Space) \\land \\Phi_{Road}")
    assert not ok
    assert "Open Space" in err


# --------------------------------------------------------------------------- #
# Rule 3b — mode names leaking into predicate arguments                       #
#                                                                              #
# E.g. `Detect(Path)` against the CARLA `{junction, path}` taxonomy. It is      #
# valid Form A CamelCase, so Rule 3 passes it, but at drive time the cue never  #
# reaches sighting 1 — a step that can never advance.                          #
# --------------------------------------------------------------------------- #

CARLA_MODES = ["junction", "path"]
PED_MODES = ["Along Wall", "Intersection: In", "Open Space", "Passage", "Road: On"]


def test_mode_name_as_form_a_argument_is_rejected():
    ok, err = quick_syntax_check("Detect(Path)", modes=CARLA_MODES)
    assert not ok
    assert "Rule 3b" in err
    assert "'path'" in err


def test_mode_name_as_form_b_argument_is_rejected():
    # Form B spelling of the same leak — collapsing catches both.
    ok, err = quick_syntax_check(
        "Detect(\\text{Open Space})", modes=PED_MODES
    )
    assert not ok
    assert "Rule 3b" in err


def test_mode_name_with_a_colon_collapses_and_is_rejected():
    # `Road: On` -> `roadon`; the arg must collapse the same way to match.
    ok, err = quick_syntax_check("Detect(RoadOn)", modes=PED_MODES)
    assert not ok
    assert "Rule 3b" in err


def test_point_mode_stays_detectable():
    """`Detect(Junction)` must pass even though `junction` IS a mode.

    The rule is narrowed to EXTENT_MODES. `cue_oracle` (`missions.py`) answers
    "junction"/"intersection", so the formula drives; rejecting it would burn
    every retry on a correct formula. You APPROACH a junction; you are always
    inside a path.
    """
    ok, err = quick_syntax_check("Detect(Junction)", modes=CARLA_MODES)
    assert ok, err
    ok, err = quick_syntax_check(
        "\\text{Detect}(\\text{Junction})", modes=CARLA_MODES
    )
    assert ok, err
    # Same for the point modes in the pedestrian vocabulary.
    ok, err = quick_syntax_check(
        "Detect(\\text{Intersection: In})", modes=PED_MODES
    )
    assert ok, err


def test_mode_leak_error_does_not_propose_a_respelling():
    # The whole point of a separate rule. `_suggest_fix("Path")` is "Path", so
    # routing this through Rule 3 would feed back "rewrite `Path` as `Path`",
    # the generator would re-emit it, and every retry would burn on a fixed
    # point — a deadlock traded for a generate-fail.
    ok, err = quick_syntax_check("Detect(Path)", modes=CARLA_MODES)
    assert not ok
    assert "NOT a spelling problem" in err
    # It must name the constructs that actually fix it.
    assert "goal_mode" in err
    assert "traverse" in err


def test_mode_check_is_exact_not_substring():
    # `Detect(Intersection)` must survive a taxonomy holding `Intersection: In`.
    # It is a live cue in the hand-written CARLA missions and the oracle answers
    # it; a prefix/substring match would break every one of them.
    ok, err = quick_syntax_check("Detect(Intersection)", modes=PED_MODES)
    assert ok, err


def test_mode_check_is_off_by_default():
    # The vocabulary is per-environment: `Detect(OpenSpace)` is the Form A
    # rewrite generator.md TEACHES, and is only a leak where `Open Space` is a
    # mode. Callers without a taxonomy get no mode check.
    ok, err = quick_syntax_check("Detect(Path)")
    assert ok, err
    ok, err = quick_syntax_check("Detect(\\text{Open Space})")
    assert ok, err


def test_mode_check_does_not_touch_macros():
    # `Path` and `Passage` are legal \Phi macro bodies in the pedestrian
    # vocabulary. Rule 3b applies to predicate ARGUMENTS only.
    ok, err = quick_syntax_check(
        "\\Phi_{Path} \\land \\mathbf{F}\\Phi_{Passage}", modes=PED_MODES
    )
    assert ok, err


def test_non_mode_landmark_still_passes_with_modes_given():
    ok, err = quick_syntax_check(
        "Detect(Cone) \\land \\mathbf{F}\\Phi_{Int_Turn}", modes=CARLA_MODES
    )
    assert ok, err
    ok, err = quick_syntax_check("Bearing(Right)", modes=CARLA_MODES)
    assert ok, err


def test_mode_leak_caught_on_a_capitalised_taxonomy():
    """The campus vocabulary spells it `Path`. Rule 3b must still fire."""
    ok, err = quick_syntax_check(
        "Detect(Path)", modes=["Covered", "Edge: Along", "Junction", "Path"]
    )
    assert not ok
    assert "Rule 3b" in err


# --------------------------------------------------------------------------- #
# stl_compile — formula to monitors                                            #
#                                                                              #
# Most of a generated formula is redundant reach-goals that restate the JSON   #
# plan; these tests pin the two shapes that are NOT redundant.                 #
# --------------------------------------------------------------------------- #

def test_until_yields_a_hold_the_plan_tree_cannot_express():
    """`path -> junction` and "without leaving the path" are the SAME tree.

    The step records a destination; nothing records that a mode had to hold on the way.
    This is one of only two shapes where the formula carries more than the plan.
    """
    from nl_planner.stl_compile import compile_monitors, parse
    ast, _ = parse(r"(\Phi_{Path}) \ \mathbf{U} \ (\text{Detect}(\text{Junction}))")
    s = compile_monitors(ast)
    assert s.holds == [{"hold_mode": "path", "until": "Detect(Junction)"}]
    assert s.novel == 1


def test_global_negation_yields_a_forbidden_mode():
    from nl_planner.stl_compile import compile_monitors, parse
    for f in (r"\mathbf{G}(\lnot \Phi_{Open\_Space})", r"\Box \lnot \Phi_{Open\_Space}"):
        s = compile_monitors(parse(f)[0])
        assert s.forbid_modes == ["open_space"], f
        assert s.novel == 1


def test_reach_goals_are_counted_as_REDUNDANT_not_as_monitors():
    """`F Phi_X` restates a step that already exists.

    Merging it would double-count the same requirement and let the compiler look like it
    is doing work it is not. It is counted separately so the redundancy is reportable.
    """
    from nl_planner.stl_compile import compile_monitors, parse
    s = compile_monitors(parse(r"\Phi_{Path} \land \mathbf{F}\Phi_{Junc\_Pass}")[0])
    assert s.novel == 0
    assert s.reach == ["mode:junction"]


def test_trailing_prose_is_ignored_not_a_parse_error():
    """`Bearing(Right) completed` is how the generator spells a topology cue.

    Feeding "completed" to the grammar as a term would fail such formulas with a bogus
    "missing )", understating compile coverage.
    """
    from nl_planner.stl_compile import parse
    ast, junk = parse(r"\mathbf{F}(\text{Bearing}(\text{Right})\ \text{completed})")
    assert "completed" in junk
    assert ast.kind == "F"


def test_an_unparseable_formula_reports_rather_than_compiling_nothing():
    """Silence and 'nothing to compile' must not look the same."""
    from nl_planner.stl_compile import coverage
    c = coverage(r"\Phi_{Path} \land \land")
    assert c["parsed"] is False and c["error"]


def test_every_real_generated_formula_parses():
    """Regression on a corpus of real generated formulas: all must parse."""
    import json, os
    from nl_planner.stl_compile import coverage
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "reports", "eval_hard_postvalidators.json")
    if not os.path.exists(p):
        pytest.skip("corpus report not present")
    rows = json.load(open(p))
    rows = rows["rows"] if isinstance(rows, dict) and "rows" in rows else rows
    forms = [r["stl_formula"] for r in rows if r.get("stl_formula")]
    bad = [f for f in forms if not coverage(f)["parsed"]]
    assert not bad, f"{len(bad)}/{len(forms)} failed to parse: {bad[:2]}"
