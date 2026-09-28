"""Tests for branch_materializer.materialize_segment."""

from __future__ import annotations

import textwrap

import pytest

from nl_planner.branch_materializer import (
    materialize_segment,
    to_brain_tree,
    walk_all_segments,
)
from nl_planner.schemas import Branch, NavPlan, PlanStep
from nl_planner.taxonomy import load_taxonomy


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #

@pytest.fixture
def taxonomy(tmp_path):
    p = tmp_path / "tax.yaml"
    p.write_text(textwrap.dedent("""
        environment: testenv
        modes:
          "Road: On":      [0, 1, 2]
          "Bridge: Enter": [10]
          "Bridge: On":    [11]
          "Bridge: Exit":  [12]
    """))
    return load_taxonomy(p)


@pytest.fixture
def ped_taxonomy():
    """A pedestrian-vocabulary taxonomy (`path` / `junction`).

    The `taxonomy` fixture above uses the CAR vocabulary ("Road: On", "Bridge: On"), and
    `stl_compile.MACRO_MODE` maps every macro onto pedestrian mode names — `\Phi_{Road}`
    resolves to `path`, not to `Road: On`. Without a taxonomy-aware lookup, a formula
    constraint on that car taxonomy lands in `stl_monitors.unapplied`.
    """
    import os
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return load_taxonomy(os.path.join(here, "carla_gt_bridge", "config",
                                      "cluster_map.carla_town01.yaml"))


def _step(idx, desc, start, goal, cue=None, branches=None):
    return PlanStep(
        step=idx, description=desc,
        start_mode=start, goal_mode=goal,
        transition_cue=cue, branches=branches,
    )


@pytest.fixture
def branch_plan():
    """Tree with one decision step."""
    return NavPlan(
        plan_name="bridge or detour", description="d",
        steps=[
            _step(0, "approach", "Road: On", "Bridge: Enter", cue="Detect(Bridge)"),
            _step(1, "decide", "Bridge: Enter", "Bridge: Enter", cue="dp", branches=[
                Branch(vlm_cue="bridge is blocked", sub_plan=[
                    _step(0, "detour", "Bridge: Enter", "Road: On"),
                ]),
                Branch(vlm_cue="default", sub_plan=[
                    _step(0, "cross", "Bridge: Enter", "Bridge: On"),
                    _step(1, "exit", "Bridge: On", "Road: On"),
                ]),
            ]),
        ],
    )


# --------------------------------------------------------------------------- #
# Root segment                                                                 #
# --------------------------------------------------------------------------- #

def test_root_segment_stops_before_decision(taxonomy, branch_plan):
    seg = materialize_segment(branch_plan, taxonomy, path=())

    # Only the leading linear step("approach") makes it into the brain plan.
    assert [s["description"] for s in seg.brain_plan["steps"]] == ["approach"]
    assert seg.brain_plan["steps"][0]["start_cluster"] == 0   # Road: On canonical id
    assert seg.brain_plan["steps"][0]["goal_cluster"] == 10  # Bridge: Enter canonical id
    assert seg.decision_step is not None
    assert seg.decision_step.description == "decide"
    assert not seg.is_terminal
    assert seg.path == ()


def test_root_segment_metadata(taxonomy, branch_plan):
    seg = materialize_segment(branch_plan, taxonomy, path=())
    from nl_planner.branch_materializer import attach_decision_metadata
    attach_decision_metadata(seg)
    meta = seg.brain_plan["nl_planner"]
    assert meta["ends_at_decision"] is True
    assert meta["decision"]["start_mode"] == "Bridge: Enter"
    assert [b["vlm_cue"] for b in meta["decision"]["branches"]] == [
        "bridge is blocked", "default",
    ]


# --------------------------------------------------------------------------- #
# Branch segments                                                              #
# --------------------------------------------------------------------------- #

def test_default_branch_terminates(taxonomy, branch_plan):
    seg = materialize_segment(branch_plan, taxonomy, path=(1,))
    assert [s["description"] for s in seg.brain_plan["steps"]] == ["cross", "exit"]
    assert seg.decision_step is None
    assert seg.is_terminal
    assert seg.path == (1,)


def test_blocked_branch_terminates(taxonomy, branch_plan):
    seg = materialize_segment(branch_plan, taxonomy, path=(0,))
    assert [s["description"] for s in seg.brain_plan["steps"]] == ["detour"]
    assert seg.is_terminal


def test_walk_enumerates_root_plus_each_branch(taxonomy, branch_plan):
    segs = walk_all_segments(branch_plan, taxonomy)
    paths = [list(s.path) for s in segs]
    assert paths == [[], [0], [1]]


# --------------------------------------------------------------------------- #
# Cluster labels                                                               #
# --------------------------------------------------------------------------- #

def test_cluster_labels_match_taxonomy(taxonomy, branch_plan):
    seg = materialize_segment(branch_plan, taxonomy, path=())
    labels = seg.brain_plan["cluster_labels"]
    assert labels["0"] == "Road: On"
    assert labels["10"] == "Bridge: Enter"
    assert labels["11"] == "Bridge: On"


# --------------------------------------------------------------------------- #
# Error paths                                                                  #
# --------------------------------------------------------------------------- #

def test_invalid_path_index_raises(taxonomy, branch_plan):
    with pytest.raises(ValueError):
        materialize_segment(branch_plan, taxonomy, path=(7,))


def test_path_past_terminal_raises(taxonomy, branch_plan):
    # default branch has no further decisions
    with pytest.raises(ValueError):
        materialize_segment(branch_plan, taxonomy, path=(1, 0))


# --------------------------------------------------------------------------- #
# to_brain_tree                                                                #
# --------------------------------------------------------------------------- #

def test_to_brain_tree_linear(taxonomy):
    plan = NavPlan(
        plan_name="linear", description="d",
        steps=[
            _step(0, "approach", "Road: On", "Bridge: Enter"),
            _step(1, "cross",    "Bridge: Enter", "Bridge: On"),
        ],
    )
    tree = to_brain_tree(plan, taxonomy)
    assert tree["plan_name"] == "linear"
    assert tree["nl_planner"]["tree_shaped"] is True
    assert tree["nl_planner"]["version"] == 2
    assert tree["nl_planner"]["environment"] == "testenv"
    assert len(tree["steps"]) == 2
    assert tree["steps"][0]["start_cluster"] == 0  # Road: On
    assert tree["steps"][0]["goal_cluster"] == 10  # Bridge: Enter
    assert tree["steps"][0]["start_mode"] == "Road: On"
    assert tree["steps"][0]["branches"] is None
    assert tree["steps"][1]["start_cluster"] == 10
    assert tree["steps"][1]["goal_cluster"] == 11
    # cluster_labels includes every taxonomy id, keyed as strings
    assert tree["cluster_labels"]["0"] == "Road: On"
    assert tree["cluster_labels"]["11"] == "Bridge: On"


def test_to_brain_tree_preserves_branches(taxonomy, branch_plan):
    tree = to_brain_tree(branch_plan, taxonomy)
    # Root list keeps both linear lead + decision step
    assert [s["description"] for s in tree["steps"]] == ["approach", "decide"]
    decision = tree["steps"][1]
    assert decision["start_cluster"] == 10  # Bridge: Enter
    assert decision["goal_cluster"] == 10
    assert decision["branches"] is not None
    cues = [b["vlm_cue"] for b in decision["branches"]]
    assert cues == ["bridge is blocked", "default"]
    # The 'default' branch's sub_plan must have two linear steps with
    # resolved cluster ids.
    default_branch = decision["branches"][1]
    assert default_branch["vlm_cue"] == "default"
    sub = default_branch["sub_plan"]
    assert [s["description"] for s in sub] == ["cross", "exit"]
    assert sub[0]["start_cluster"] == 10  # Bridge: Enter
    assert sub[0]["goal_cluster"]  == 11  # Bridge: On
    assert sub[0]["branches"]      is None


def test_to_brain_tree_round_trips_through_navigator(taxonomy, branch_plan):
    """The output is consumable by brain.plan_navigator.PlanNavigator."""
    # PlanNavigator lives in the sibling `brain` package. Resolve it relative
    # to this file (…/src/nl_planner/test/ -> …/src/brain/brain/) rather than
    # from an absolute path, and load it without polluting the broader test
    # session. Skip cleanly if brain isn't checked out alongside nl_planner.
    import importlib.util
    from pathlib import Path

    nav_path = (
        Path(__file__).resolve().parents[2] / "brain" / "brain" / "plan_navigator.py"
    )
    if not nav_path.exists():
        pytest.skip(f"brain/plan_navigator.py not found at {nav_path}")

    spec = importlib.util.spec_from_file_location("brain_plan_navigator", nav_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    tree = to_brain_tree(branch_plan, taxonomy)
    nav = mod.PlanNavigator(tree["steps"])
    # Linear lead "approach"
    assert nav.current_step["description"] == "approach"
    nav.advance()
    # Now at the decision step
    assert nav.current_has_branches
    assert nav.default_branch_idx() == 1
    nav.descend(0)  # 'bridge is blocked'
    assert nav.current_step["description"] == "detour"
    nav.advance()
    assert nav.is_complete


# --------------------------------------------------------------------------- #
# `Phi_X U cue` and `[]~X` reach the brain tree as CLUSTER IDS                  #
# --------------------------------------------------------------------------- #

def test_hold_mode_is_resolved_to_cluster_ids_for_brain(taxonomy):
    """Pins that brain gets ids, not a mode name it cannot act on.

    brain never sees the taxonomy — that is the whole reason `accept_clusters` is
    resolved here. A `hold_mode` shipped as the string "Road: On" is a field the
    executor can log and nothing more, and the invariant would silently never
    fire: `InvariantMonitor.observe` treats an EMPTY accept set as "always
    inside", so an unresolved hold reads in every summary as a perfectly held
    invariant.
    """
    plan = NavPlan(
        plan_name="hold", description="d",
        steps=[PlanStep(step=0, description="hold the road",
                        start_mode="Road: On", goal_mode="Road: On",
                        hold_mode="Road: On", until="Detect(Bridge)")])
    step = to_brain_tree(plan, taxonomy)["steps"][0]
    assert step["hold_mode"] == "Road: On"
    assert step["until"] == "Detect(Bridge)"
    assert step["hold_accept_clusters"] == [0, 1, 2]


def test_hold_accept_clusters_is_the_strict_set_not_the_degraded_one(tmp_path):
    """Pins the direction the degraded fallback must NOT be applied in.

    `accept_degraded` exists so a MISSED fine detection cannot strand a positive
    step. An invariant is the opposite question: widening its accept set makes
    leaving the mode harder to notice, which defeats the only thing the monitor
    does. On the CARLA taxonomy `junction`'s degraded set is every path id, so a
    degraded hold on `junction` would accept the entire map.
    """
    p = tmp_path / "tax.yaml"
    p.write_text(textwrap.dedent("""
        environment: e
        modes:
          "path":     [0, 1]
          "junction": [9]
        mode_meta:
          "junction":
            accept_degraded: [0, 1]
    """))
    tax = load_taxonomy(p)
    plan = NavPlan(
        plan_name="hold", description="d",
        steps=[PlanStep(step=0, description="hold the junction",
                        start_mode="junction", goal_mode="junction",
                        hold_mode="junction", until="Detect(Cone)")])
    step = to_brain_tree(plan, tax)["steps"][0]
    assert step["hold_accept_clusters"] == [9]
    # the GOAL-mode set on the same step keeps its degraded fallback — the two
    # are different questions and must not be resolved the same way
    assert step["accept_clusters_degraded"] == [9, 0, 1]


def test_a_step_that_holds_nothing_gains_no_keys(taxonomy):
    """Pins byte-identity of every checked-in `mission.*.json`.

    `--write` regenerates them. Emitting `hold_mode: null` on every step would
    rewrite all of them for a field none of them use.
    """
    plan = NavPlan(plan_name="plain", description="d",
                   steps=[_step(0, "drive", "Road: On", "Bridge: Enter")])
    step = to_brain_tree(plan, taxonomy)["steps"][0]
    assert "hold_mode" not in step
    assert "until" not in step
    assert "hold_accept_clusters" not in step


def test_forbid_modes_resolve_to_the_UNION_of_their_accept_sets(taxonomy):
    """Pins union, not intersection.

    "never enter the bridge or the road" forbids a region that is EITHER.
    Intersection would forbid only what is both — usually nothing — and a
    constraint that forbids nothing reports csr = 1.0, i.e. it manufactures
    evidence that the robot obeyed a rule that was never applied.
    """
    plan = NavPlan(plan_name="x", description="d",
                   steps=[_step(0, "drive", "Road: On", "Bridge: Enter")],
                   forbid_modes=["Bridge: On", "Bridge: Exit"])
    tree = to_brain_tree(plan, taxonomy)
    assert tree["forbid_clusters"] == [11, 12]
    assert tree["forbid_modes"] == ["Bridge: On", "Bridge: Exit"]


def test_forbid_clusters_are_absent_when_no_constraint_is_declared(taxonomy):
    """Pins that "not declared" and "declared as empty" stay distinguishable.

    A run scored against an empty forbidden set is vacuously satisfied; a run
    scored against a declared set it stayed out of has earned it. `run_scoring.C`
    reports which one happened, and it can only do that if the tree does not
    invent a `forbid_clusters: []` for every plan.
    """
    plan = NavPlan(plan_name="x", description="d",
                   steps=[_step(0, "drive", "Road: On", "Bridge: Enter")])
    tree = to_brain_tree(plan, taxonomy)
    assert "forbid_clusters" not in tree and "forbid_modes" not in tree


def test_forbid_modes_are_carried_onto_every_materialized_segment(taxonomy, branch_plan):
    """Pins that a negative constraint survives a branch descent.

    `materialize_segment` slices the tree into per-branch linear plans, and each
    slice is a separate document shipped to brain. A plan-level constraint that
    was written only into the root segment would switch itself off at the first
    decision — precisely where the robot's options widen and the constraint
    starts to matter.
    """
    branch_plan.forbid_modes = ["Bridge: On"]
    for seg in walk_all_segments(branch_plan, taxonomy):
        assert seg.brain_plan["forbid_clusters"] == [11], seg.path


def test_a_mission_wide_positive_invariant_reaches_the_tree(ped_taxonomy):
    """`require_modes` is the positive mirror of `forbid_modes`, and PLAN-level.

    "Stay on the walkway the entire way" binds every branch. A step-scoped version
    (`hold_mode`) would switch itself off at the first fork, which is the same failure
    `forbid_modes` was made plan-level to avoid.
    """
    from nl_planner.schemas import NavPlan, PlanStep
    plan = NavPlan(plan_name="p", description="d", require_modes=["path"],
                   steps=[PlanStep(step=0, description="go", start_mode="path",
                                   goal_mode="junction", trigger="traverse")])
    tree = to_brain_tree(plan, ped_taxonomy)
    assert tree["require_modes"] == ["path"]
    assert tree["require_clusters"], "modes must resolve to cluster ids"


def test_G_phi_in_the_formula_compiles_to_a_mission_invariant(ped_taxonomy):
    """`G(Phi_X)` compiles to a positive invariant.

    A bare `Phi_{Path}` conjunct formally claims Path at t=0 only; `G` is needed to
    claim it throughout.
    """
    from nl_planner.schemas import NavPlan, PlanStep
    plan = NavPlan(plan_name="p", description="d",
                   steps=[PlanStep(step=0, description="go", start_mode="path",
                                   goal_mode="junction", trigger="traverse")])
    tree = to_brain_tree(plan, ped_taxonomy, stl=r"\mathbf{G}\Phi_{Path}")
    assert tree["require_modes"] == ["path"]
    assert {"require_modes": ["path"]} in tree["stl_monitors"]["applied"]


def test_a_bare_conjunct_does_NOT_become_an_invariant(ped_taxonomy):
    """`Phi_{Path} \\land F(...)` asserts Path at the START, not throughout.

    Reading it as an invariant would have the compiler assert something the formula does
    not say.
    """
    from nl_planner.schemas import NavPlan, PlanStep
    plan = NavPlan(plan_name="p", description="d",
                   steps=[PlanStep(step=0, description="go", start_mode="path",
                                   goal_mode="junction", trigger="traverse")])
    tree = to_brain_tree(plan, ped_taxonomy,
                         stl=r"\Phi_{Path} \land \mathbf{F}\Phi_{Junc\_Pass}")
    assert not tree.get("require_modes")


def test_stl_formula_crosses_the_dispatch_boundary_and_reaches_the_monitors():
    """The formula must survive NavPlan serialisation and still compile to a monitor.

    WHY THIS FIELD. The plan's own constraint fields are sometimes wrong or absent while
    the formula compiles to exactly the right monitor. Without a NavPlan field to carry
    the formula, such constraints would be silently dropped at /nl_planner/dispatch.

    Three things are asserted, each independently necessary: the field survives a JSON
    round trip, `stl=` actually produces the monitor, and a plan with no formula still
    validates so existing plans need no migration.
    """
    # the `taxonomy` fixture is the CAR vocabulary and has no Passage, so this needs the
    # combined pedestrian map
    import os
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    tax = load_taxonomy(os.path.join(here, "bev_pipeline", "config",
                                     "cluster_map.combined.yaml"))
    steps = [{"step": 1, "description": "cross to the plaza",
              "start_mode": "Path", "goal_mode": "Space: Open",
              "transition_cue": "Detect(Plaza)"}]
    plan = NavPlan.model_validate({
        "plan_name": "t", "description": "d", "steps": steps,
        "stl_formula": r"\mathbf{G}\lnot\Phi_{Passage}",
    })

    # survives the wire: /nl_planner/dispatch is a single String of NavPlan JSON
    round_tripped = NavPlan.model_validate_json(plan.model_dump_json())
    assert round_tripped.stl_formula == plan.stl_formula

    # and compiling it yields the constraint the plan itself never stated
    assert plan.forbid_modes is None
    assert to_brain_tree(plan, tax).get("forbid_clusters") is None
    tree = to_brain_tree(round_tripped, tax, stl=round_tripped.stl_formula)
    assert tree["forbid_clusters"] == list(tax.resolve("Passage"))

    # backward compatible: every plan recorded before the field still loads
    old = NavPlan.model_validate({"plan_name": "t", "description": "d", "steps": steps})
    assert old.stl_formula is None


def test_constraints_union_within_an_axis_and_intersect_across():
    """A sidewalk is a path; what distinguishes it is the SURFACE.

    Modelling sidewalk as a third topology mode would make a plan requiring `sidewalk`
    merge with a formula requiring `path` by UNION -- "sidewalk OR road", which a route
    driven entirely on the carriageway satisfies. That is the exact opposite of "stay on
    the sidewalk at all times".

    On separate axes the merge is an intersection -- the sidewalk set the mission meant.
    Within one axis union is still right: "a path or a junction" is a genuine
    alternative, not a contradiction.
    """
    import os
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    tax = load_taxonomy(os.path.join(here, "carla_gt_bridge", "config",
                                     "cluster_map.carla_town05_sidewalks.yaml"))
    steps = [{"step": 1, "description": "g", "start_mode": "path",
              "goal_mode": "junction", "transition_cue": "Detect(Junction)"}]
    plan = NavPlan.model_validate({"plan_name": "t", "description": "d", "steps": steps,
                                   "require_modes": ["sidewalk"]})

    n_side = len(tax.resolve("sidewalk"))
    assert to_brain_tree(plan, tax)["require_clusters"] == sorted(tax.resolve("sidewalk"))

    # the formula names a TOPOLOGY mode; both must hold, so the surface set survives
    merged = to_brain_tree(plan, tax, stl=r"\mathbf{G}\Phi_{Path}")
    assert len(merged["require_clusters"]) == n_side
    assert set(merged["require_clusters"]) <= set(tax.resolve("path"))

    # same axis stays a union
    alt = NavPlan.model_validate({"plan_name": "t", "description": "d", "steps": steps,
                                  "require_modes": ["path", "junction"]})
    assert set(to_brain_tree(alt, tax)["require_clusters"]) == (
        set(tax.resolve("path")) | set(tax.resolve("junction")))


# --------------------------------------------------------------------------- #
# Constraint fields must SURVIVE materialization                              #
# --------------------------------------------------------------------------- #
#
# `brain_controller` builds every constraint monitor from the tree this module emits:
#     ConstraintMonitor(plan_data.get("forbid_clusters"))
#     InstanceForbidTracker(plan_data.get("forbid_instances"))
#     InvariantMonitor(... require_clusters ...)
# A field the NavPlan carries and the tree drops is a constraint the robot never receives,
# and nothing anywhere reports it: the plan validates, the run completes, the monitor is
# empty.

def _tax():
    import os
    from nl_planner.taxonomy import load_taxonomy
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return load_taxonomy(os.path.join(here, "carla_gt_bridge", "config",
                                      "cluster_map.carla_town05.yaml"))


def _plan(**extra):
    from nl_planner.schemas import NavPlan
    return NavPlan.model_validate(dict(
        plan_name="t", description="d",
        steps=[{"step": 0, "description": "go", "start_mode": "path",
                "goal_mode": "junction", "transition_cue": "Detect(Junction)",
                "branches": None}], **extra))


def test_forbid_instances_reaches_the_brain_tree():
    """The one brain reads to build InstanceForbidTracker."""
    from nl_planner.branch_materializer import to_brain_tree
    t = to_brain_tree(_plan(forbid_instances=[{"mode": "junction", "ordinal": 2}]), _tax())
    assert t.get("forbid_instances") == [{"mode": "junction", "ordinal": 2}]


def test_forbid_instances_is_NOT_resolved_to_cluster_ids():
    """It must cross as MODES. InstanceForbidTracker counts arrivals at run time because
    which junction is 'second' depends on the branch taken, which is not known until it is
    taken -- and because naming ids would bake in one map."""
    from nl_planner.branch_materializer import to_brain_tree
    t = to_brain_tree(_plan(forbid_instances=[{"mode": "junction", "ordinal": 2}]), _tax())
    assert t["forbid_instances"][0]["mode"] == "junction"
    assert "clusters" not in str(t["forbid_instances"])


def test_mode_level_constraints_still_reach_the_tree():
    """Regression guard: forbid_modes and require_modes must keep surviving
    materialization."""
    from nl_planner.branch_materializer import to_brain_tree
    t = to_brain_tree(_plan(forbid_modes=["junction"]), _tax())
    assert t.get("forbid_modes") == ["junction"] and t.get("forbid_clusters")
    t2 = to_brain_tree(_plan(require_modes=["path"]), _tax())
    assert t2.get("require_modes") == ["path"] and t2.get("require_clusters")


def test_a_plan_with_no_constraint_emits_no_constraint_keys():
    """Absence must stay absent -- an empty list would make the monitor construct itself."""
    from nl_planner.branch_materializer import to_brain_tree
    t = to_brain_tree(_plan(), _tax())
    for k in ("forbid_instances", "forbid_clusters", "require_clusters"):
        assert not t.get(k), f"{k} appeared on a plan that declares no constraint"
