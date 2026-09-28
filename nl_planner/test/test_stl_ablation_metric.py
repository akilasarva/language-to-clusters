import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
_WS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
from stl_ablation import captured_constraints


def test_constraint_stated_in_both_representations_counts_once():
    """The metric must not credit STL for restating what the JSON already holds.

    `captured_constraints` unions plan fields with compiled-formula output. If the two
    sources keyed their entries differently, a model that wrote BOTH `forbid_modes` and
    `G ¬Φ_Passage` would score 2 where a JSON-only plan scored 1 — crediting STL for
    redundancy. The key prefixes are chosen to collide so the set dedupes.
    """
    stl = r"\mathbf{G} \lnot \Phi_{Passage}"
    assert captured_constraints({"forbid_modes": ["passage"]}, stl) == ["forbid:passage"]
    assert captured_constraints({"forbid_modes": ["passage"]}, "") == ["forbid:passage"]
    assert captured_constraints({}, stl) == ["forbid:passage"]


def test_dropped_constraint_is_visible_as_empty():
    """The failure this metric exists to catch: a valid plan that silently drops the rule."""
    assert captured_constraints({"steps": [{"goal_mode": "path"}]}, "") == []


def test_formula_naming_an_absent_mode_is_caught():
    """An STL constraint over a mode the taxonomy lacks compiles to a dead monitor.

    Worse than a rejected plan: the accept set resolves to nothing, so the rule
    never fires while still looking enforced. The formula path must be validated
    just as `forbid_modes` is, or the comparison between them is asymmetric.
    """
    from nl_planner.taxonomy import load_taxonomy
    from nl_planner.stl_compile import validate_stl_modes
    tax = load_taxonomy(_WS + "/carla_gt_bridge/config/"
                        "cluster_map.carla_town01.yaml")          # {junction, path}
    assert validate_stl_modes(r"\mathbf{G} \lnot \Phi_{Passage}", tax) == ["passage"]
    assert validate_stl_modes(r"\mathbf{G} \Phi_{Path}", tax) == []


def test_require_modes_is_mode_validated_like_forbid_modes():
    from nl_planner.taxonomy import load_taxonomy, validate_plan_modes
    from nl_planner.schemas import NavPlan
    tax = load_taxonomy(_WS + "/carla_gt_bridge/config/"
                        "cluster_map.carla_town01.yaml")
    plan = NavPlan(plan_name="t", description="d", require_modes=["no_such_mode"],
                   steps=[dict(step=1, description="go", start_mode="path",
                               goal_mode="junction", transition_cue="Detect(Junction)",
                               trigger="traverse")])
    assert validate_plan_modes(plan, tax) == ["no_such_mode"]


def test_one_collapse_rule_for_the_gate_and_for_resolve():
    """The formula gate and `resolve` must agree on what counts as the same mode.

    If `validate_stl_modes` collapsed case and punctuation while `resolve` matched
    exactly, `G \\Phi_{Path}` against a taxonomy spelling it `Path` would pass
    validation and then raise TaxonomyError during materialization -- a crash where
    the whole point of validating first is to get a retryable failure the generator
    can fix.
    """
    from nl_planner.taxonomy import load_taxonomy, TaxonomyError
    from nl_planner.stl_compile import validate_stl_modes
    tax = load_taxonomy(_WS + "/bev_pipeline/config/"
                        "cluster_map.combined.yaml")              # spells it `Path`
    assert validate_stl_modes(r"\mathbf{G} \Phi_{Path}", tax) == []
    assert tax.resolve("path") == tax.resolve("Path")             # gate agrees with resolve
    assert tax.canonical_mode("spaceopen") == "Space: Open"
    with pytest.raises(TaxonomyError):
        tax.resolve("pth")                                        # a real typo still fails


def _tax(rel):
    from nl_planner.taxonomy import load_taxonomy
    return load_taxonomy(_WS + "/" + rel)


def test_a_formula_can_constrain_a_car_taxonomy():
    """MACRO_MODE's values are pedestrian spellings; the taxonomy must override them.

    Without a taxonomy, `\\Phi_{Road}` resolves to `path` regardless of the environment,
    so on a map spelling its modes differently the constraint lands in `unapplied`. With
    the taxonomy supplied, the macro resolves into that environment's own vocabulary.
    """
    from nl_planner.stl_compile import parse, compile_monitors
    car = _tax("carla_gt_bridge/config/cluster_map.carla_town01.yaml")   # {junction, path}
    campus = _tax("bev_pipeline/config/cluster_map.combined.yaml")       # {Path, Passage, ...}
    road = parse(r"\mathbf{G} \Phi_{Road}")[0]
    assert compile_monitors(road, car).require_modes == ["path"]
    assert compile_monitors(road, campus).require_modes == ["Path"]


def test_a_macro_names_its_own_mode_without_a_translation_entry():
    """`Passage` is not a MACRO_MODE key on the campus spelling; the macro name IS the mode."""
    from nl_planner.stl_compile import parse, compile_monitors
    campus = _tax("bev_pipeline/config/cluster_map.combined.yaml")
    spec = compile_monitors(parse(r"\mathbf{G} \lnot \Phi_{Passage}")[0], campus)
    assert spec.forbid_modes == ["Passage"]


def test_a_mode_the_taxonomy_lacks_stays_unapplied_rather_than_wrong():
    """Honest failure: town01 has no `passage`, so the constraint must NOT be invented."""
    from nl_planner.stl_compile import parse, compile_monitors
    car = _tax("carla_gt_bridge/config/cluster_map.carla_town01.yaml")
    spec = compile_monitors(parse(r"\mathbf{G} \lnot \Phi_{Passage}")[0], car)
    assert spec.forbid_modes == []
    assert spec.unhandled, "an unresolvable constraint must be reported, not dropped silently"


def test_omitting_the_taxonomy_preserves_the_old_behaviour():
    from nl_planner.stl_compile import parse, compile_monitors
    assert compile_monitors(parse(r"\mathbf{G} \lnot \Phi_{Passage}")[0]).forbid_modes == ["passage"]
