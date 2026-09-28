"""The enforcement instrument: can we tell a violating run from a clean one?

Plan-generation harnesses score whether a PLAN WAS PRODUCED, not whether a RULE WAS
OBEYED, and plan validity is structurally blind to the difference --
a plan that silently drops "never cross the grass" is a perfectly valid plan.
"""
import os
import sys

import pytest

WS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(WS, "nl_planner", "scripts"))

from replay_monitors import Obs, replay, obs_from_jsonl, self_test   # noqa: E402


def _tree(**kw):
    from replay_monitors import _tree as t
    return t(**kw)


def test_self_test_passes():
    """The script's own discriminator. If this fails the harness measures nothing."""
    assert self_test() == 0


def test_a_violating_run_still_completes_its_route():
    """Why region-sequence scoring cannot substitute for this.

    The robot reaches every expected region in order AND breaks the constraint on the
    way. Region-sequence scoring reports a clean pass; only the constraint monitor sees
    it.
    """
    tree = _tree(forbid=[9])
    r = replay(tree, [Obs(1), Obs(9), Obs(9), Obs(2), Obs(2), Obs(2)])
    assert r.ok, "the route completed"
    assert r.flagged, "and the constraint was violated -- B/D would call this a pass"


def test_an_undeclared_constraint_is_vacuous_not_satisfied():
    r = replay(_tree(), [Obs(1), Obs(9), Obs(2), Obs(2), Obs(2)])
    assert not r.forbid_declared
    assert not r.flagged
    assert r.csr == 1.0            # vacuously; `forbid_declared` is what distinguishes it


def test_a_formula_derived_constraint_executes_on_a_car_taxonomy():
    """A formula-only constraint, executed offline.

    The formula never touches NavPlan, planner_node or brain -- compile_monitors turns
    it into cluster ids offline.
    """
    from nl_planner.stl_compile import compile_monitors, parse
    from nl_planner.taxonomy import load_taxonomy
    tax = load_taxonomy(os.path.join(WS, "carla_gt_bridge", "config",
                                     "cluster_map.carla_town01.yaml"))
    spec = compile_monitors(parse(r"\mathbf{G} \lnot \Phi_{Junc\_Pass}")[0], tax)
    assert spec.forbid_modes == ["junction"]
    ids = sorted(set(tax.resolve("junction")))
    tree = _tree(forbid=ids)
    assert not replay(tree, [Obs(1)] * 3 + [Obs(2)] * 3).flagged
    assert replay(tree, [Obs(1), Obs(ids[0]), Obs(ids[0])] + [Obs(2)] * 3).flagged


def test_it_reads_a_real_recorded_carla_run():
    """Replay must work on recorded runs, without re-running the simulator."""
    d = os.path.join(WS, "dgppo_ros_node_pkg", "dgppo_ros_node_pkg", "debug_logs")
    files = sorted((os.path.getsize(os.path.join(d, f)), f)
                   for f in (os.listdir(d) if os.path.isdir(d) else []) if f.endswith(".jsonl"))
    if not files or files[-1][0] < 1000:
        pytest.skip("no recorded runs on disk")
    obs = obs_from_jsonl(os.path.join(d, files[-1][1]))
    assert len(obs) > 100
    assert all(isinstance(o.cluster, int) for o in obs)
    r = replay(_tree(), obs)
    assert r.regions_visited, "a real drive visits regions"


def test_discrimination_requires_a_materialized_plan():
    """A raw NavPlan carries MODES; the monitors compare CLUSTER IDS.

    `goal_cluster` and `accept_clusters` are what `to_brain_tree` resolves modes into.
    Replaying an un-materialized plan finds no clusters and quietly reports "does not
    discriminate" for every arm.
    """
    import stl_ablation as A
    from nl_planner.branch_materializer import to_brain_tree
    from nl_planner.schemas import NavPlan
    from nl_planner.taxonomy import load_taxonomy
    tax = load_taxonomy(os.path.join(WS, "bev_pipeline", "config",
                                     "cluster_map.combined.yaml"))
    plan = NavPlan.model_validate({
        "plan_name": "t", "description": "d",
        "steps": [{"step": 0, "description": "decide", "start_mode": "Path",
                   "goal_mode": "Path", "transition_cue": "decision point",
                   "branches": [
                       {"vlm_cue": "gate shut", "sub_plan": [
                           {"step": 0, "description": "round the back",
                            "start_mode": "Path", "goal_mode": "Passage",
                            "transition_cue": "Detect(Passage)"}]},
                       {"vlm_cue": "default", "sub_plan": [
                           {"step": 0, "description": "straight on",
                            "start_mode": "Path", "goal_mode": "Junction",
                            "transition_cue": "Detect(Junction)"}]}]}]})
    assert not A._cluster_hints(plan.model_dump().get("steps")), "raw plan has no clusters"
    assert A._cluster_hints(to_brain_tree(plan, tax).get("steps")), "materialized plan does"
    assert A._discriminates(plan, tax)
