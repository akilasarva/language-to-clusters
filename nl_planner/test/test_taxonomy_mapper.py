"""Tests for taxonomy loading + plan-mode validation + bootstrap inverter."""

from __future__ import annotations

import textwrap

import pytest

from nl_planner.bootstrap.seed_taxonomy import invert_label_map, render_yaml
from nl_planner.schemas import Branch, NavPlan, PlanStep
from nl_planner.taxonomy import TaxonomyError, load_taxonomy, validate_plan_modes


# --------------------------------------------------------------------------- #
# Loader                                                                       #
# --------------------------------------------------------------------------- #

def _write(tmp_path, body: str):
    p = tmp_path / "tax.yaml"
    p.write_text(textwrap.dedent(body))
    return p


def test_load_taxonomy_happy_path(tmp_path):
    p = _write(tmp_path, """
        environment: testenv
        source: somefile.json
        modes:
          "Road: On":     [5, 6, 7]
          "Intersection": [0, 1, 2]
    """)
    tax = load_taxonomy(p)
    assert tax.environment == "testenv"
    assert tax.resolve("Road: On") == (5, 6, 7)
    assert tax.canonical_id("Intersection") == 0
    assert "Road: On" in tax
    assert tax.label_for_id(7) == "Road: On"


def test_load_taxonomy_rejects_missing_modes(tmp_path):
    p = _write(tmp_path, """
        environment: testenv
        modes: {}
    """)
    with pytest.raises(TaxonomyError):
        load_taxonomy(p)


def test_load_taxonomy_rejects_non_int_ids(tmp_path):
    p = _write(tmp_path, """
        environment: testenv
        modes:
          "Road: On": ["a"]
    """)
    with pytest.raises(TaxonomyError):
        load_taxonomy(p)


def test_modes_for_prompt_sorted(tmp_path):
    p = _write(tmp_path, """
        environment: e
        modes:
          "Bridge: On": [1]
          "Road: On":   [2]
          "Along Wall": [3]
    """)
    assert load_taxonomy(p).modes_for_prompt() == ["Along Wall", "Bridge: On", "Road: On"]


# --------------------------------------------------------------------------- #
# Plan-mode validation                                                         #
# --------------------------------------------------------------------------- #

def test_validate_plan_modes_catches_missing(tmp_path):
    p = _write(tmp_path, """
        environment: e
        modes:
          "Road: On":     [0]
          "Bridge: Enter": [1]
    """)
    tax = load_taxonomy(p)

    plan = NavPlan(
        plan_name="x", description="x",
        steps=[
            PlanStep(step=0, description="ok",
                     start_mode="Road: On", goal_mode="Bridge: Enter",
                     transition_cue=None),
            PlanStep(step=1, description="bad",
                     start_mode="Bridge: Enter", goal_mode="Made Up Mode",
                     transition_cue=None),
        ],
    )
    missing = validate_plan_modes(plan, tax)
    assert missing == ["Made Up Mode"]


def test_validate_plan_modes_walks_branches(tmp_path):
    p = _write(tmp_path, """
        environment: e
        modes:
          "Road: On":      [0]
          "Bridge: Enter": [1]
          "Bridge: On":    [2]
    """)
    tax = load_taxonomy(p)

    branch_default = Branch(
        vlm_cue="default",
        sub_plan=[
            PlanStep(step=0, description="bad",
                     start_mode="Bridge: Enter", goal_mode="Mystery Cluster",
                     transition_cue=None),
        ],
    )
    branch_blocked = Branch(
        vlm_cue="blocked",
        sub_plan=[
            PlanStep(step=0, description="ok",
                     start_mode="Bridge: Enter", goal_mode="Road: On",
                     transition_cue=None),
        ],
    )
    plan = NavPlan(
        plan_name="x", description="x",
        steps=[
            PlanStep(step=0, description="approach",
                     start_mode="Road: On", goal_mode="Bridge: Enter"),
            PlanStep(step=1, description="decide",
                     start_mode="Bridge: Enter", goal_mode="Bridge: Enter",
                     transition_cue="dp",
                     branches=[branch_blocked, branch_default]),
        ],
    )
    assert validate_plan_modes(plan, tax) == ["Mystery Cluster"]


# --------------------------------------------------------------------------- #
# seed_taxonomy bootstrap                                                      #
# --------------------------------------------------------------------------- #

def test_invert_label_map_groups_and_sorts():
    raw = {"-1": "Along Wall", "0": "In Int", "1": "In Int", "5": "Road", "3": "Road"}
    inv = invert_label_map(raw)
    assert list(inv.keys()) == ["Along Wall", "In Int", "Road"]
    assert inv["In Int"] == [0, 1]
    assert inv["Road"] == [3, 5]


def test_render_yaml_is_parseable():
    raw = {"0": "A", "1": "A", "2": "B"}
    body = render_yaml(environment="testenv", source="f.json", grouped=invert_label_map(raw))
    import yaml
    parsed = yaml.safe_load(body)
    assert parsed["environment"] == "testenv"
    assert parsed["modes"]["A"] == [0, 1]
    assert parsed["modes"]["B"] == [2]


# --------------------------------------------------------------------------- #
# mode_meta: acceptance sets + per-environment perception binding              #
# --------------------------------------------------------------------------- #

_META_YAML = """
    environment: livox1
    modes:
      "Road: On":         [1, 2, 3, 4]
      "Along Wall":       [2]
      "Intersection: In": [4]
    mode_meta:
      "Along Wall":
        perception_backed: true
        accept_degraded: [1]
      "Intersection: In":
        perception_backed: false
        accept_degraded: [1]
        enter_on: "Detect(StopSign)"
        exit_on: "Bearing(Right)"
"""


def test_accept_clusters_strict_vs_degraded(tmp_path):
    """The blue-building case: an Along Wall step must be satisfiable on `path`."""
    tax = load_taxonomy(_write(tmp_path, _META_YAML))

    # A traverse step, where the cluster IS the evidence, stays strict.
    assert tax.accept_clusters("Along Wall") == (2,)
    assert 1 not in tax.accept_clusters("Along Wall")

    # A landmark step, where Detect(...) carries the evidence, degrades to path.
    assert tax.accept_clusters("Along Wall", degraded=True) == (2, 1)

    # Preferred-first ordering is preserved: the mode's own id stays at index 0.
    assert tax.accept_clusters("Along Wall", degraded=True)[0] == 2


def test_degradation_absent_means_no_widening(tmp_path):
    tax = load_taxonomy(_write(tmp_path, _META_YAML))
    # "Road: On" has no mode_meta entry at all -> degraded == strict.
    assert tax.accept_clusters("Road: On", degraded=True) == tax.accept_clusters("Road: On")


def test_perception_backed_flag(tmp_path):
    tax = load_taxonomy(_write(tmp_path, _META_YAML))
    assert tax.is_perception_backed("Along Wall") is True
    # On the real campus the intersection is a plan construct, not a percept.
    assert tax.is_perception_backed("Intersection: In") is False
    # Unknown modes still raise rather than silently defaulting.
    with pytest.raises(TaxonomyError):
        tax.is_perception_backed("Nope")


def test_mode_meta_defaults_preserve_legacy_behaviour(tmp_path):
    """A taxonomy written before mode_meta existed must behave exactly as before."""
    tax = load_taxonomy(_write(tmp_path, """
        environment: legacy
        modes:
          "Road: On": [5, 6]
    """))
    assert tax.mode_meta is None
    assert tax.accept_clusters("Road: On") == (5, 6)
    assert tax.accept_clusters("Road: On", degraded=True) == (5, 6)
    assert tax.is_perception_backed("Road: On") is True


def test_mode_meta_rejects_unknown_mode(tmp_path):
    with pytest.raises(TaxonomyError, match="unknown mode"):
        load_taxonomy(_write(tmp_path, """
            environment: e
            modes:
              "Road: On": [1]
            mode_meta:
              "Ghost Mode":
                perception_backed: false
        """))


def test_mode_meta_rejects_non_int_degraded_ids(tmp_path):
    with pytest.raises(TaxonomyError, match="accept_degraded"):
        load_taxonomy(_write(tmp_path, """
            environment: e
            modes:
              "Road: On": [1]
            mode_meta:
              "Road: On":
                accept_degraded: ["path"]
        """))


def test_cluster_labels_prefers_the_most_specific_mode(tmp_path):
    """A junction must be labelled `junction`, not `path`.

    The subsumption lattice deliberately lists a junction's id under BOTH `junction` and
    `path` (a junction really is on a path). Resolving to whichever mode comes first in
    the YAML would, where `path` is written first, label every id in the plan `path`.

    Nothing crashes. But the plan hands `cluster_labels` to the controller, which grounds
    "reach a junction" by looking for an adjacent region labelled `junction`; with none,
    the vehicle has nowhere to go. Offline tests do not otherwise catch it.
    """
    from nl_planner.taxonomy import load_taxonomy

    p = tmp_path / "cluster_map.lattice.yaml"
    p.write_text(
        "environment: lattice\n"
        "modes:\n"
        "  path: [4, 5, 45, 53, 66]\n"     # own ids PLUS the junctions, by subsumption
        "  junction: [53, 66]\n"           # own ids only — the finer, smaller set
    )
    labels = load_taxonomy(str(p)).cluster_labels()
    assert labels[53] == "junction"
    assert labels[66] == "junction"
    assert labels[4] == "path"
    assert labels[45] == "path"


def test_cluster_labels_is_order_independent(tmp_path):
    """Writing `junction` first must not change the answer."""
    from nl_planner.taxonomy import load_taxonomy

    p = tmp_path / "cluster_map.reordered.yaml"
    p.write_text(
        "environment: reordered\n"
        "modes:\n"
        "  junction: [53, 66]\n"
        "  path: [4, 5, 45, 53, 66]\n"
    )
    labels = load_taxonomy(str(p)).cluster_labels()
    assert labels[53] == "junction" and labels[4] == "path"


# --------------------------------------------------------------------------- #
# validate_plan_transitions — plans that name legal modes but cannot be walked   #
# --------------------------------------------------------------------------- #

def _tstep(i, start, goal, branches=None):
    return {"step": i, "description": f"s{i}", "start_mode": start,
            "goal_mode": goal, "transition_cue": None, "branches": branches}


def test_transitions_reject_self_loop():
    """`junction -> junction` never leaves the junction it is already in.

    generator.md forbids it in bold and the generator can emit it anyway; routing
    then fails with "no 'junction' adjacent to ...".
    """
    from nl_planner.taxonomy import validate_plan_transitions
    errs = validate_plan_transitions(
        {"steps": [_tstep(0, "path", "junction"), _tstep(1, "junction", "junction")]})
    assert len(errs) == 1
    assert "start_mode == goal_mode" in errs[0]


def test_transitions_allow_self_loop_on_decision_step():
    """A decision step stays put on purpose while the VLM decides."""
    from nl_planner.taxonomy import validate_plan_transitions
    plan = {"steps": [_tstep(0, "path", "junction"),
                      _tstep(1, "junction", "junction", branches=[
                          {"vlm_cue": "cone", "sub_plan": [_tstep(0, "junction", "path")]},
                          {"vlm_cue": "default", "sub_plan": [_tstep(0, "junction", "path")]},
                      ])]}
    assert validate_plan_transitions(plan) == []


def test_transitions_reject_non_chaining_steps():
    """Step N+1 must start where step N ended."""
    from nl_planner.taxonomy import validate_plan_transitions
    errs = validate_plan_transitions(
        {"steps": [_tstep(0, "path", "junction"), _tstep(1, "path", "junction")]})
    assert len(errs) == 1
    assert "does not continue from" in errs[0]


def test_transitions_accept_the_unroll():
    """The documented way to say "the second junction"."""
    from nl_planner.taxonomy import validate_plan_transitions
    plan = {"steps": [_tstep(0, "path", "junction"), _tstep(1, "junction", "path"),
                      _tstep(2, "path", "junction"), _tstep(3, "junction", "path")]}
    assert validate_plan_transitions(plan) == []


def test_transitions_check_inside_branches():
    from nl_planner.taxonomy import validate_plan_transitions
    plan = {"steps": [_tstep(0, "path", "junction", branches=[
        {"vlm_cue": "cone", "sub_plan": [_tstep(0, "junction", "junction")]},
        {"vlm_cue": "default", "sub_plan": [_tstep(0, "junction", "path")]}])]}
    errs = validate_plan_transitions(plan)
    assert len(errs) == 1 and "branches[0]" in errs[0]


def test_transitions_allow_self_loop_on_extent_modes():
    """`path -> path` with an ordinal is generator.md Example 4, not a bug.

    The robot is on the road before the 2nd bench and on the road after it; there
    is no mode change, and emitting one would stall the plan waiting for a
    transition that never happens. Rejecting it would force `path -> junction` and
    hide StepTargeter routing gaps behind a rejected-but-correct plan.
    """
    from nl_planner.taxonomy import validate_plan_transitions
    for mode in ("path", "along_edge", "passage", "open_space", "Road: On"):
        plan = {"steps": [{"step": 0, "description": "stop at the 2nd bench",
                           "start_mode": mode, "goal_mode": mode,
                           "cue_ordinal": 2, "branches": None}]}
        assert validate_plan_transitions(plan) == [], mode


def test_transitions_reject_self_loop_only_on_point_modes():
    from nl_planner.taxonomy import validate_plan_transitions
    errs = validate_plan_transitions(
        {"steps": [{"step": 0, "description": "x", "start_mode": "junction",
                    "goal_mode": "junction", "branches": None}]})
    assert len(errs) == 1 and "pass THROUGH" in errs[0]


# --------------------------------------------------------------------------- #
# validate_decision_cues — a branching step that also asks its own question    #
#                                                                              #
# E.g. `transition_cue='decision point'` on a branching step: the branch       #
# resolves correctly and drives to the right exit, then the step never         #
# completes and the vehicle wanders until the cue budget ends the run.         #
# --------------------------------------------------------------------------- #

def _cstep(i, start, goal, cue=None, branches=None):
    return {"step": i, "description": f"s{i}", "start_mode": start,
            "goal_mode": goal, "transition_cue": cue, "branches": branches}


def _two_branches():
    return [{"vlm_cue": "a cone is in the junction",
             "sub_plan": [_cstep(0, "junction", "path", "Bearing(Straight) completed")]},
            {"vlm_cue": "default",
             "sub_plan": [_cstep(0, "junction", "path", "Bearing(Right) completed")]}]


def test_decision_cue_rejects_prose_on_branching_step():
    from nl_planner.taxonomy import validate_decision_cues
    errs = validate_decision_cues(
        {"steps": [_cstep(0, "path", "junction", "decision point", _two_branches())]})
    assert len(errs) == 1
    assert "decision point" in errs[0]
    # The feedback must name both escapes, not just say "wrong".
    assert "null" in errs[0] and "Detect(" in errs[0]


def test_decision_cue_allows_a_real_predicate_on_a_branching_step():
    """The hand-written mission's shape: `Detect(Intersection)` + branches."""
    from nl_planner.taxonomy import validate_decision_cues
    plan = {"steps": [_cstep(0, "path", "junction",
                             "Detect(Intersection)", _two_branches())]}
    assert validate_decision_cues(plan) == []


def test_decision_cue_allows_no_cue_at_all():
    from nl_planner.taxonomy import validate_decision_cues
    plan = {"steps": [_cstep(0, "path", "junction", None, _two_branches())]}
    assert validate_decision_cues(plan) == []


def test_decision_cue_leaves_prose_on_a_plain_step_alone():
    """`transition_cue` is free text BY DESIGN — brain's VLM polls it.

    "a light brown bench" is the documented pedestrian shape and there is no
    decidable test that separates it from "decision point". The rule is
    structural: only a step whose BRANCHES already ask the question is rejected
    for asking it again.
    """
    from nl_planner.taxonomy import validate_decision_cues
    plan = {"steps": [_cstep(0, "path", "path", "a light brown bench")]}
    assert validate_decision_cues(plan) == []


def test_decision_cue_accepts_bearing_with_trailing_prose():
    """`Bearing(Right) completed` works — missions.py reads the kind by substring."""
    from nl_planner.taxonomy import validate_decision_cues
    plan = {"steps": [_cstep(0, "path", "junction",
                             "Bearing(Right) completed", _two_branches())]}
    assert validate_decision_cues(plan) == []


def test_decision_cue_walks_into_sub_plans():
    from nl_planner.taxonomy import validate_decision_cues
    inner = [_cstep(0, "path", "junction", "the second decision", _two_branches())]
    plan = {"steps": [_cstep(0, "path", "junction", None, [
        {"vlm_cue": "cone", "sub_plan": inner},
        {"vlm_cue": "default", "sub_plan": [_cstep(0, "junction", "path")]}])]}
    errs = validate_decision_cues(plan)
    assert len(errs) == 1
    assert "branches[0]" in errs[0]


def test_is_extent_mode_is_case_and_punctuation_insensitive():
    """`EXTENT_MODES` holds `path`; `cluster_map.combined.yaml` declares `Path`.

    With an exact-string compare, Rule 3b would silently no-op on every taxonomy
    that capitalises its modes — `Detect(Path)` caught on CARLA's lowercase
    vocabulary and waved through on the campus one, which is worse than not
    having the rule, because it reads as covered.
    """
    from nl_planner.taxonomy import is_extent_mode
    for mode in ("path", "Path", "Open Space", "open_space", "Space: Open",
                 "Edge: Along", "along_edge", "Corridor", "Covered", "Road: On"):
        assert is_extent_mode(mode), mode
    for mode in ("junction", "Junction", "Intersection: In", "Intersection: Exit"):
        assert not is_extent_mode(mode), mode


# --------------------------------------------------------------------------- #
# `Phi_X U cue` and `[]~X` — the modes those two fields name                    #
# --------------------------------------------------------------------------- #

def test_transitions_allow_self_loop_on_a_dwell_step():
    """Pins: "hold the road until the intersection" must not be rejected as a
    step that never moved.

    Rule 1 says a step that claims to travel has to move, and a dwell step never
    claimed to — it has no destination. Without the exemption, `Phi_X U cue` is
    unwritable for every POINT mode and the validator's own error message tells
    the generator to insert a traversal, which would delete the invariant it was
    asked for. Same exemption the pure-decision step already gets, for the same
    reason.
    """
    from nl_planner.taxonomy import validate_plan_transitions
    step = _tstep(0, "junction", "junction")
    step.update(hold_mode="junction", until="Detect(Cone)")
    assert validate_plan_transitions({"steps": [step]}) == []


def test_a_self_loop_without_hold_mode_is_still_rejected():
    """Pins that the exemption is NARROW: it keys on `hold_mode`, not on the
    modes being equal.

    The bug rule 1 exists to catch — `junction -> junction` — is byte-identical
    to a dwell step apart
    from that one field. If the exemption widened to "equal modes are fine", the
    rule would be gone and the failure it caught would come straight back.
    """
    from nl_planner.taxonomy import validate_plan_transitions
    errs = validate_plan_transitions({"steps": [_tstep(0, "junction", "junction")]})
    assert len(errs) == 1 and "start_mode == goal_mode" in errs[0]


def test_validate_plan_modes_catches_a_misspelled_hold_mode(tmp_path):
    """Pins: a hold on a mode the taxonomy does not have is a RETRYABLE error.

    `branch_materializer` resolves `hold_mode` through `accept_clusters()`, which
    raises `TaxonomyError` on an unknown mode. That aborts the run instead of
    feeding the generator a correction, and the generator is the thing that can
    fix a spelling. Catching it here is what makes it a retry rather than a crash.
    """
    p = _write(tmp_path, """
        environment: e
        modes:
          "Road: On": [0]
          "Plaza":    [1]
    """)
    tax = load_taxonomy(p)
    plan = {"steps": [{"step": 0, "description": "hold", "start_mode": "Road: On",
                       "goal_mode": "Road: On", "hold_mode": "Roadd: On",
                       "until": "Detect(Plaza)", "branches": None}]}
    assert validate_plan_modes(plan, tax) == ["Roadd: On"]


def test_validate_plan_modes_catches_a_misspelled_forbid_mode(tmp_path):
    """Pins the worst-behaved of the four mode fields.

    An unresolvable `forbid_modes` entry produces an EMPTY forbidden cluster set,
    and an empty set is exactly what "no constraint declared" produces. So a typo
    does not fail loudly — it reports csr = 1.0 on a constraint that was never
    checked, which is the strongest possible evidence for a claim nothing tested.
    """
    p = _write(tmp_path, """
        environment: e
        modes:
          "Road: On": [0]
          "Plaza":    [1]
    """)
    tax = load_taxonomy(p)
    plan = {"steps": [{"step": 0, "description": "go", "start_mode": "Road: On",
                       "goal_mode": "Plaza", "branches": None}],
            "forbid_modes": ["Plazza"]}
    assert validate_plan_modes(plan, tax) == ["Plazza"]


def test_validate_plan_modes_accepts_a_well_spelled_constraint(tmp_path):
    """Pins the no-false-positive half: a legal `forbid_modes` / `hold_mode` must
    not be reported missing, or the retry loop would spin on a correct plan."""
    p = _write(tmp_path, """
        environment: e
        modes:
          "Road: On": [0]
          "Plaza":    [1]
    """)
    tax = load_taxonomy(p)
    plan = {"steps": [{"step": 0, "description": "hold", "start_mode": "Road: On",
                       "goal_mode": "Road: On", "hold_mode": "Road: On",
                       "until": "Detect(Plaza)", "branches": None}],
            "forbid_modes": ["Plaza"]}
    assert validate_plan_modes(plan, tax) == []
