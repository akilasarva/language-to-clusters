"""The offline harness must be able to FAIL a mission, not just drive good ones.

`goal_reached()` returns True for a landmark step with the reason "in strict accept set,
AWAITING CUE"; a harness that treats that as the step being finished never runs the
ordinal counter, the sighting de-bounce or the cue timeout offline, and a mission that
deadlocks in CARLA would pass here.

These tests pin the harness's ability to catch that class of bug. They are cheap — a
whole mission simulates in well under a second.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#: Brain trees used as executor test inputs (not missions shipped with the project).
PLANS = os.path.join(PKG, "test", "fixtures", "executor_plans")
#: Regions holding a cone for the ordinal fixture: two junctions apart, so the cue
#: necessarily goes false in between -- the gap that makes the second sighting distinct.
ORDINAL_CONES = (63, 60)
sys.path.insert(0, os.path.join(PKG, "scripts"))
sys.path.insert(0, PKG)


def _sim():
    for need in (os.path.join(PLANS, "mission.hard.json"),
                 os.path.join(PKG, "config", "regions.town05.npz")):
        if not os.path.exists(need):
            pytest.skip(f"{need} not present")
    try:
        import missions
    except ImportError as exc:                       # pydantic / nl_planner missing
        pytest.skip(f"missions harness unavailable: {exc}")
    missions.PLAN_DIR = PLANS
    return missions


@pytest.mark.parametrize("mission,branch,expect", [
    # The derived Town05 corridor: pick_corridor() derives
    # 0 -> J63 -> 1 -> J60 -> right 49 | straight 2 over the full 74-region map. (A
    # hand-picked corridor can hide a wrong turn, e.g. `hard` turning right at both
    # junctions, so the corridor is derived rather than chosen.)
    ("easy",   "right",    [0, 63]),
    ("medium", "right",    [0, 63, 17]),
    ("hard",   "right",    [0, 63, 1, 60, 49]),
    ("hard",   "straight", [0, 63, 1, 60, 2]),
])
def test_missions_drive_their_intended_route(mission, branch, expect):
    r = _sim().simulate(mission, branch=branch,
                        cone_regions=ORDINAL_CONES if mission == "ordinal" else None)
    assert r["ok"], r["reason"]
    assert r["regions_visited"] == expect


def test_branch_follows_the_cue_not_the_caller():
    """`branch` describes the WORLD; the decision is derived from it, not asserted.

    Naming the outcome ("take the straight branch") instead of the cause ("a cone is
    present") would let the harness skip the decision entirely.
    """
    m = _sim()
    with_cone = m.simulate("hard", branch="straight")
    without = m.simulate("hard", branch="right")
    # Derived, not typed: if the corridor is re-cut, stale exit ids here would read as
    # the branch going wrong rather than as a stale constant.
    straight_exit = with_cone["regions_visited"][-1]
    right_exit = without["regions_visited"][-1]
    assert straight_exit != right_exit, (
        f"the cue changed nothing — both branches ended at {right_exit}")
    assert without["ok"] and with_cone["ok"]


def test_the_cue_oracle_is_scoped_to_the_intersection():
    """A cone elsewhere in town must not answer yes — it would fire the wrong branch."""
    o = _sim().cue_oracle
    assert o("a traffic cone is in the intersection", "junction", True) is True
    assert o("a traffic cone is in the intersection", "path", True) is False
    assert o("Detect(Intersection)", "junction", False) is True
    assert o("Detect(Intersection)", "path", False) is False


def test_an_unreachable_ordinal_is_caught_offline(tmp_path):
    """An unsatisfiable ordinal must fail offline (in CARLA it produces a runaway).

    cue_ordinal=2 together with an unrolled first intersection is unsatisfiable: sightings
    are counted per STEP and only become distinct after the cue goes false, but once the
    vehicle is standing in the junction the cue is permanently true. The count sticks at 1.

    It must fail, and the reason must say why rather than merely timing out.

    USES 3, NOT 2. On the derived corridor J60 has an exit the vehicle can leave by and
    come back through, which MANUFACTURES a second distinct sighting: cue_ordinal=2
    reports ok=True while driving [0, 63, 1, 60, 2, 60, 2, 59, 44] — re-entering the
    junction to earn the count. The B+D region/heading checks catch that; `ok` alone does
    not, which is one more reason nothing should be scored on `ok`.

    3 is unsatisfiable on this corridor, so the guard still guards.
    """
    m = _sim()
    tree = json.loads(open(os.path.join(PLANS, "mission.hard.json")).read())
    tree["steps"][2]["cue_ordinal"] = 3
    bad = tmp_path / "mission.bad.json"
    bad.write_text(json.dumps(tree))

    r = m.simulate("hard", branch="right", plan_path=str(bad))
    assert not r["ok"], "an unsatisfiable ordinal must not pass"
    assert "never reached sighting 3" in r["reason"]
    assert "stuck at" in r["reason"]


# --------------------------------------------------------------------------- #
# ordinal cues — "stop at the second cone"                                     #
# --------------------------------------------------------------------------- #

def test_the_ordinal_mission_stops_at_the_SECOND_cone():
    """Passes the cone at the first junction, stops at the one at the second.

    This is the only mission that exercises cue_ordinal > 1. Ordinals exist for landmarks
    that are NOT clusters — an intersection is a cluster, so "the second intersection" is
    said by unrolling, and combining both deadlocks. A cone is not a cluster, so counting
    sightings is the only way to say it, exactly as "the 2nd bench" works in the bags.
    """
    m = _sim()
    r = m.simulate("ordinal", cone_regions=ORDINAL_CONES)
    assert r["ok"], r["reason"]
    assert r["regions_visited"] == [0, 63, 1, 60]
    assert "sighting 2/2" in r["advances"][-1]["why"]
    assert r["advances"][-1]["region"] == ORDINAL_CONES[-1], "stopped at the wrong cone"


def test_one_cone_is_not_enough():
    """With a single cone the mission must NOT complete.

    The failure this guards is a counter that fires on the first sighting — which passes
    the happy path and is invisible until an ordinal actually matters.
    """
    m = _sim()
    r = m.simulate("ordinal", cone_regions=ORDINAL_CONES[:1])   # only the first exists
    assert not r["ok"], "completed with only one cone — the ordinal is not being counted"
    assert "never reached sighting 2" in r["reason"]


def test_the_debounce_makes_the_second_sighting_distinct():
    """Two cones in the SAME region must count once, not twice.

    A sighting only becomes distinct after the cue goes false for cue_lost_polls. Without
    that, standing next to one cone would satisfy "the second cone" — which is precisely
    the bug the campus-bag idiom ("the 2nd bench") is written to avoid.
    """
    m = _sim()
    first = ORDINAL_CONES[0]
    r = m.simulate("ordinal", cone_regions=(first, first))   # "two" cones, one place
    assert not r["ok"], "counted the same cone twice"


# --------------------------------------------------------------------------- #
# staying on the road                                                          #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("mission,branch", [
    ("easy", "right"), ("medium", "right"),
    ("hard", "right"), ("hard", "straight"), ("ordinal", "right"),
])
def test_the_vehicle_stays_on_the_drivable_surface(mission, branch):
    """No excursion far beyond the carriageway.

    NOT lane-keeping: the vehicle steers at region centroids and wanders across the whole
    road by design. The weaker guarantee is the one that matters — do not leave the road
    and drive through a building.

    Without sensor input the MPC's EDT and CBF are inert and nothing else prevents it;
    without the drivable-surface penalty staying on the paved surface is luck rather than
    construction.

    The bound is half-width (7 m) plus room to overshoot and recover, since the penalty is
    soft — a hard constraint would reject every rollout where reference-line sampling is
    sparse and hand control to the recovery heuristic.
    """
    r = _sim().simulate(mission, branch=branch,
                        cone_regions=ORDINAL_CONES if mission == "ordinal" else None)
    assert r["ok"], r["reason"]
    assert r["max_offroad_m"] < 9.0, (
        f"strayed {r['max_offroad_m']:.1f} m from the nearest reference line")


def test_the_road_term_actually_changes_the_command():
    """Otherwise the test above passes for free and proves nothing."""
    import numpy as np

    from carla_gt_bridge.region_lookup import load_region_table
    from dgppo_ros_node_pkg.sampling_mpc import MpcConfig, RoadSurface, plan_step

    table = load_region_table(os.path.join(PKG, "config", "regions.town05.npz"))
    road = RoadSurface(table.waypoints[:, :2])
    ids = table.region_ids
    cents = np.array([table.centroid_of(r) for r in ids])
    pos = np.array(list(table.centroid_of(45)))

    # Aimed from region 45 at junction 53 — NOT adjacent, so the straight line leaves the
    # road. This is exactly the situation the penalty exists for.
    free = plan_step(MpcConfig(), pos, -1.2, cents, ids, start_id=45, target_id=53,
                     rng=np.random.default_rng(3))
    kept = plan_step(MpcConfig(), pos, -1.2, cents, ids, start_id=45, target_id=53,
                     rng=np.random.default_rng(3), road=road)
    assert free.offroad_frac == 0.0, "no road given, so nothing should be measured"
    assert (kept.v, kept.omega) != (free.v, free.omega), (
        "the drivable-surface term changed nothing — it is not wired in")


# --------------------------------------------------------------------------- #
# M — did it DRIVE the route, or did its position jump along it?               #
#                                                                              #
# B reads the region sequence and D reads the headings along it. NEITHER can   #
# tell driving from teleporting: a pose that sits still and then jumps 26 m in #
# 0.4 s (63 m/s against a v_max of 5.0) visits the right regions and would     #
# score a PASS. A jump that passes is worse than a failure, so M checks that   #
# the motion is physically plausible.                                          #
# --------------------------------------------------------------------------- #

def _tr(pts):
    return [(x, y, 0, 0, 0) for x, y in pts]


def test_motion_guard_rejects_the_measured_teleport():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "scripts"))
    from run_scoring import motion_is_plausible
    m = motion_is_plausible(
        _tr([(-173.3, 131.8), (-173.3, 131.8), (-147.5, 135.4)]),
        [62.070, 66.997, 67.433])
    assert not m["ok"]
    assert m["max_speed"] > 50
    assert m["at_sample"] == 2          # names the offending interval, not just "bad"


def test_motion_guard_passes_a_real_drive():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "scripts"))
    from run_scoring import motion_is_plausible
    # 5 m/s, which is v_max — must not be flagged.
    m = motion_is_plausible(_tr([(0, 0), (5, 0), (10, 0), (15, 0)]), [0, 1, 2, 3])
    assert m["ok"], m
    assert m["path_len"] == 15.0


def test_motion_guard_is_a_noop_without_timestamps():
    """The offline unicycle integrator cannot jump, and passes no times.

    It must not start failing runs it has always passed — and it must SAY that the
    check did not run, rather than reporting a silent ok.
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "scripts"))
    from run_scoring import motion_is_plausible
    m = motion_is_plausible(_tr([(0, 0), (99, 0)]))
    assert m["ok"]
    assert "unverifiable" in m["checked"]


# --------------------------------------------------------------------------- #
# SEPARATION OF GUIDANCE FROM SCORING                                          #
#                                                                              #
# Ground truth may be used to COMPARE a run afterwards. It must never reach    #
# the executor, or the harness measures its own answer key. This is not a      #
# style rule: a single `expect_regions=` forwarded into simulate() would make  #
# every recorded number meaningless, and it would be invisible in a diff.      #
# --------------------------------------------------------------------------- #

#: Bare "expect" is deliberately NOT here — it occurs in ordinary prose in comments
#: ("expected for the first tick or two"), and a substring match on it flags those.
#: These are identifier names, so they are matched on word boundaries.
GROUND_TRUTH_ONLY = ("expect_regions", "expect_maneuvers", "expect_end", "want_end")


def test_simulate_cannot_see_the_expected_route():
    import inspect, sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "scripts"))
    import missions
    params = set(inspect.signature(missions.simulate).parameters)
    leaked = params & set(GROUND_TRUTH_ONLY)
    assert not leaked, (
        f"simulate() accepts {leaked} — the executor can see the answer key. "
        f"Expected routes belong to run_scoring only.")


def test_the_executor_modules_never_reference_the_answer_key():  # noqa: D401
    """Textual, on purpose: it catches a leak added anywhere in the call chain.

    Scoped to the modules that DRIVE. `run_scoring.py` legitimately names these: it does
    the comparing.
    """
    import os, re
    pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ws = os.path.dirname(pkg)
    targets = [
        os.path.join(pkg, "carla_gt_bridge", "routing.py"),
        os.path.join(ws, "brain", "brain", "trigger_policy.py"),
        os.path.join(ws, "brain", "brain", "plan_navigator.py"),
        os.path.join(ws, "brain", "brain", "brain_controller.py"),
        os.path.join(ws, "dgppo_ros_node_pkg", "dgppo_ros_node_pkg",
                     "carla_mpc_ros_node.py"),
    ]
    for path in targets:
        if not os.path.exists(path):
            continue
        src = open(path).read()
        for token in GROUND_TRUTH_ONLY:
            assert not re.search(rf"\b{token}\b", src), (
                f"{os.path.basename(path)} references {token!r} — the executor can see "
                f"the answer key")


def test_simulate_only_receives_world_state_and_initial_conditions():
    """Pin what simulate() IS allowed — so a new parameter has to be justified.

    `cone_regions` is world state (where the props are), not guidance: the cue oracle
    answers questions ABOUT the world, which is what a perception system would do.
    `start_region` / `start_toward` are initial conditions — a vehicle has to start
    somewhere facing something, and in CARLA that is the spawn pose. They are derived
    from the corridor, so they are declared here rather than left implicit.

    The rest are POLICY and ABLATION knobs. None of them can carry a route: they say
    how strictly to judge what the executor does (`invariant_policy`,
    `invariant_dwell`, `forbid_policy`), how much to corrupt the perception it is
    given (`cluster_noise`, `noise_seed`, `noise_mode`), or which grounding rule to
    run it under (`grounding`). The distinction that matters is not "was a parameter
    added" but "can it tell the executor where to go", and the answer for every one
    of these is no.
    """
    import inspect, sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "scripts"))
    import missions
    allowed = {
        "mission", "branch", "max_ticks", "verbose", "plan_path",
        "centroid_offset", "guidance", "scale", "rot_deg", "regions_npz",
        "start_region", "start_toward", "cone_regions", "cue_timeout_ticks",
        "invariant_policy", "invariant_dwell",
        # ablation knobs: perception noise and grounding rule
        "cluster_noise", "noise_seed", "noise_mode", "grounding",
        # how landmark sightings are COUNTED ("legacy" | "v2": scoped to the goal mode,
        # windowed from the last cue/manoeuvre step). A counting rule, never a route.
        "cue_semantics",
        # what a plan-level `[]~X` violation COSTS: "log" (default, record only)
        # or "fail". It is the response to a measurement, never an input to one.
        "forbid_policy",
    }
    params = set(inspect.signature(missions.simulate).parameters)
    assert params <= allowed, (
        f"simulate() grew {params - allowed}. Every parameter is a channel by which "
        f"ground truth could reach the executor — justify it and add it here.")


# --------------------------------------------------------------------------- #
# `Phi_X U cue` — the DWELL step, both halves                                  #
# --------------------------------------------------------------------------- #

def _tree(name):
    """The written brain tree for a mission, as a mutable dict."""
    with open(os.path.join(PLANS, f"mission.{name}.json")) as f:
        return json.load(f)


def _write_tree(tmp_path, tree, name="edited"):
    p = tmp_path / f"mission.{name}.json"
    p.write_text(json.dumps(tree))
    return str(p)


def _needs(name):
    if not os.path.exists(os.path.join(PLANS, f"mission.{name}.json")):
        pytest.skip(f"mission.{name}.json not present in {PLANS}")


def test_the_until_mission_terminates_on_its_cue_and_not_on_arrival():
    """Pins the RIGHT half of `Phi_X U cue` in the harness that has to run it.

    The `easy` mission drives the same 29 m and stops on a CLUSTER: three
    consecutive readings inside the junction accept set. This one has no
    destination at all — nothing is being travelled to — so if the harness were
    still deciding by `goal_reached`, `goal_mode == start_mode == path` would be
    satisfied on tick one, in the region it started in, and the mission would
    "complete" without moving.
    """
    _needs("until")
    r = _sim().simulate("until")
    assert r["ok"], r["reason"]
    assert r["regions_visited"] == [0, 63], r["regions_visited"]
    assert len(r["advances"]) == 1
    assert "until:" in r["advances"][0]["why"]
    assert r["path_len"] > 20.0, "a dwell step that never moved is not a drive"


def test_a_dwell_step_never_asks_the_router_to_ground_a_target():
    """Pins the one thing about a dwell step every consumer has to know.

    A dwell step has NO destination. `StepTargeter.target_for("path", ...)`
    nevertheless returns a confident answer for it — the nearest region of that
    label — and the vehicle would drive to a place the plan never named, on a
    step whose only content is "keep doing what you are doing". The failure is
    invisible in the result: the run completes, the route looks reasonable, and
    the trajectory is not the one the mission asked for.

    Enforced by making the call itself an error, because asserting on the
    resulting path would pass for the wrong reason whenever the invented target
    happened to lie ahead.
    """
    _needs("until")
    m = _sim()
    from carla_gt_bridge import routing

    original = routing.StepTargeter.target_for

    def _boom(self, *a, **kw):
        raise AssertionError("StepTargeter.target_for called on a dwell step")

    routing.StepTargeter.target_for = _boom
    try:
        r = m.simulate("until")
    finally:
        routing.StepTargeter.target_for = original
    assert r["ok"], r["reason"]


def test_leaving_the_held_mode_fails_the_run_and_names_the_mode(tmp_path):
    """Pins the LEFT half — the half brain has never executed.

    "stay on the walkway until the plaza" and "wander anywhere until the plaza"
    must be distinguishable. This is that difference, made to cost something: the plan holds `junction`, the vehicle is
    on `path` from the first tick, and after `invariant_dwell` consecutive
    readings outside the accept set the run must FAIL.

    The reason must NAME the mode. "invariant breached" alone sends the reader to
    the wrong step — a plan can hold different modes on different steps, and the
    monitor is reset at every advance.
    """
    _needs("until")
    tree = _tree("until")
    step = tree["steps"][0]
    step["hold_mode"] = "junction"
    step["hold_accept_clusters"] = sorted(
        int(cid) for cid, lab in tree["cluster_labels"].items() if lab == "junction")
    r = _sim().simulate("until", plan_path=_write_tree(tmp_path, tree),
                        invariant_dwell=3)
    assert not r["ok"]
    assert "invariant breached" in r["reason"]
    assert "junction" in r["reason"], r["reason"]
    assert r["invariant"]["breached"] is True


def test_the_hold_invariant_is_binding_without_being_switched_on(tmp_path):
    """Pins that `hold_mode` does NOT read `invariant_policy`.

    `invariant_policy` defaults to "off" so the reporting monitor stays a no-op
    by default. If the hold monitor shared that switch, a plan could write
    `hold_mode`, get no invariant at all, and report a clean pass — the field
    would be decoration. Writing
    `hold_mode` IS the request; there is nothing further to enable.
    """
    _needs("until")
    tree = _tree("until")
    tree["steps"][0]["hold_mode"] = "junction"
    tree["steps"][0]["hold_accept_clusters"] = [53]
    r = _sim().simulate("until", plan_path=_write_tree(tmp_path, tree),
                        invariant_policy="off", invariant_dwell=2)
    assert not r["ok"] and "invariant breached" in r["reason"]


def test_an_until_cue_that_never_fires_is_reported_as_such(tmp_path):
    """Pins that a non-terminating dwell reads as a CUE failure, not a timeout.

    Same rule the landmark budget already follows. Without it the vehicle holds
    heading until it runs off the end of the corridor or the tick budget expires,
    and both of those log lines send the reader to the router — which is working
    perfectly. The unanswerable cue is the fault and the message has to say so.
    """
    _needs("until")
    tree = _tree("until")
    tree["steps"][0]["until"] = "Detect(Cone)"
    r = _sim().simulate("until", plan_path=_write_tree(tmp_path, tree),
                        cone_regions=(), cue_timeout_ticks=5)
    assert not r["ok"]
    assert "never fired" in r["reason"] and "Detect(Cone)" in r["reason"]


# --------------------------------------------------------------------------- #
# `[]~X` — the plan-level negative constraint                                  #
# --------------------------------------------------------------------------- #

def test_the_avoid_mission_measures_its_violation_instead_of_hiding_it():
    """Pins the end-to-end number, on the mission that exists to produce it.

    `until` and `avoid` are the same plan; the only difference is the one
    plan-level `forbid_modes` line. So the csr `avoid` reports is attributable to
    that line and nothing else. On this corridor it cannot be 1.0 — the dwell step
    ends BY entering an intersection — and a version of this feature that reported
    a clean 1.0 here would be reporting that the check never ran.
    """
    _needs("avoid")
    m = _sim()
    r = m.simulate("avoid")
    assert r["ok"], r["reason"]                 # log policy must not abort the run
    assert r["forbid_declared"] is True
    assert r["violation_ticks"] >= 1
    assert 0.0 < r["csr"] < 1.0
    assert r["forbid_regions"] == [63]
    # same drive, same length — the constraint is measured, it does not steer
    assert m.simulate("until")["path_len"] == pytest.approx(r["path_len"])


def test_csr_is_one_and_vacuous_when_no_constraint_is_declared():
    """Pins that missions without a constraint score csr = 1.0.

    A plan that forbids nothing cannot violate anything, so csr is 1.0 — but that
    is VACUOUS satisfaction, and `forbid_declared` is what tells it apart from a
    constraint that was declared and honoured. Collapsing the two is how "the
    check never ran" comes to look like "the check passed".
    """
    for name in ("easy", "medium", "hard", "ordinal"):
        _needs(name)
        r = _sim().simulate(name, cone_regions=ORDINAL_CONES if name == "ordinal" else None)
        assert r["csr"] == 1.0
        assert r["violation_ticks"] == 0
        assert r["forbid_declared"] is False


def test_forbid_policy_fail_aborts_and_log_does_not():
    """Pins that the response is ONE knob, and that the default is the harmless one.

    Detection is identical either way — the same set-membership test on the same
    tick — so `log` and `fail` must differ only in what happens next. The default
    is `log` (measure before enforce): with noisy perception labels, a hard abort
    aborts on perception noise rather than on the robot entering the region.
    """
    _needs("avoid")
    m = _sim()
    logged = m.simulate("avoid", forbid_policy="log")
    failed = m.simulate("avoid", forbid_policy="fail")
    assert logged["ok"] and not failed["ok"]
    assert "forbidden region" in failed["reason"] and "63" in failed["reason"]
    assert failed["violation_ticks"] == 1
    assert m.simulate("avoid")["ok"], "the DEFAULT must be the non-aborting one"


def test_an_unknown_forbid_policy_is_rejected_rather_than_ignored():
    """Pins a typo as an error. `forbid_policy="warn"` silently behaving like
    `log` would report a run as constrained when nothing enforced it."""
    _needs("avoid")
    with pytest.raises(ValueError):
        _sim().simulate("avoid", forbid_policy="warn")


# --------------------------------------------------------------------------- #
# C — reported, not enforced                                                   #
# --------------------------------------------------------------------------- #

def test_C_is_reported_but_kept_out_of_the_ok_conjunction():
    """Pins the measure-before-enforce discipline at the scoring boundary.

    Same reasoning as `InvariantMonitor`: with noisy perception labels, a
    criterion promoted into `ok` before its violation rate is known fails runs on
    classifier noise, and every number downstream then describes the classifier.
    C must be visible in the dict and absent from `ok`, so the rate can be
    collected on runs whose pass/fail stays comparable.
    """
    _needs("avoid")
    sys.path.insert(0, os.path.join(PKG, "scripts"))
    from run_scoring import score

    m = _sim()
    r = m.simulate("avoid")
    labels = {}
    s = score(r, labels)
    assert s["C_constraint"] is False           # it really did enter a junction
    assert s["constraint"]["declared"] is True
    assert s["ok"] is True, "C must not be able to fail a run yet"


def test_C_distinguishes_vacuous_satisfaction_from_earned():
    """Pins the distinction the dict exists to carry.

    "no constraint declared" and "constraint declared and never violated" both
    score C=True, and they are not the same claim. `run_scoring.motion_is_plausible`
    already has this rule written down for M ("absence of a check is not a pass");
    C obeys it by reporting `declared` next to the rate rather than leaving it to
    be inferred from `csr == 1.0`.
    """
    sys.path.insert(0, os.path.join(PKG, "scripts"))
    from run_scoring import constraint_satisfied

    vacuous = constraint_satisfied({"csr": 1.0, "forbid_declared": False})
    earned = constraint_satisfied({"csr": 1.0, "forbid_declared": True,
                                   "violation_ticks": 0, "constraint_ticks": 40})
    assert vacuous["ok"] and earned["ok"]
    assert vacuous["declared"] is False and earned["declared"] is True
    assert "vacuously" in vacuous["checked"]


def test_C_defaults_to_satisfied_for_results_that_predate_the_field():
    """Pins backward compatibility of the SCORER, not just the harness.

    `score()` is called on stored result dicts
    that may not carry `csr`. A KeyError there would break re-scoring; a False
    would fail every such run.
    """
    sys.path.insert(0, os.path.join(PKG, "scripts"))
    from run_scoring import score

    s = score({"ok": True, "trace": [], "regions_visited": [1, 2]}, {})
    assert s["C_constraint"] is True
    assert s["constraint"]["declared"] is False
    assert s["ok"] is True


def test_a_hold_with_no_resolved_clusters_fails_instead_of_passing_vacuously(tmp_path):
    """Pins that an UNRESOLVED invariant cannot report itself as held.

    `InvariantMonitor.observe` reads an empty accept set as "always inside", so a
    `hold_mode` whose ids never got resolved runs the whole mission and reports
    zero violations. That is worse than no feature: the summary asserts the robot
    stayed in a mode nothing ever checked. Same rule `motion_is_plausible` states
    for M — absence of a check is not a pass.
    """
    _needs("until")
    tree = _tree("until")
    tree["steps"][0].pop("hold_accept_clusters")
    r = _sim().simulate("until", plan_path=_write_tree(tmp_path, tree))
    assert not r["ok"]
    assert "no hold_accept_clusters" in r["reason"]


def test_the_simulator_emits_timestamps_so_the_motion_guard_is_not_vacuous():
    """M sat inside the pass conjunction as an unconditional True.

    `motion_is_plausible` returns ok=True when timestamps are missing, so if the
    simulator emitted no `trace_times`, every `ok` would include an M that was never
    evaluated -- the "absence of a check is not a pass" failure inside the pass
    criterion itself.
    """
    import missions as M
    import run_scoring as S
    r = M.simulate("easy")
    times = r.get("trace_times")
    assert times and len(times) == len(r["trace"]), "one timestamp per trace row"
    m = S.motion_is_plausible(r["trace"], times)
    assert "unverifiable" not in m["checked"], f"M still vacuous: {m['checked']}"
    assert m["ok"] and m["max_speed"] > 0.0

    # and it must be able to FAIL, or emitting the times bought nothing
    tr = list(r["trace"])
    i = len(tr) // 2
    tr[i] = (tr[i][0] + 500.0,) + tuple(tr[i][1:])
    assert not S.motion_is_plausible(tr, times)["ok"]
