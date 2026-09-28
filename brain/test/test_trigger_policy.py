"""Tests for the trigger typology — the rule deciding when a step is satisfied.

The behaviours pinned here are the ones the old ``current == goal_cluster`` test
could not express, so a regression would silently reintroduce a stalled plan.
"""

from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain.trigger_policy import (  # noqa: E402
    StepProgress,
    accept_set,
    bearing_complete,
    bearing_direction,
    goal_reached,
    trigger_of,
    wrap_pi,
)

# Cluster ids as emitted by the geometric vocabulary.
OPEN_SPACE, PATH, ALONG_EDGE, PASSAGE, JUNCTION = 0, 1, 2, 3, 4


def _step(**kw):
    """A plan step with sane defaults, overridable per test."""
    base = {
        "step": 0,
        "description": "test step",
        "start_cluster": PATH,
        "goal_cluster": PATH,
        "goal_mode": "Road: On",
        "transition_cue": None,
        "trigger": None,
        "cue_ordinal": None,
        "accept_clusters": [PATH],
        "accept_clusters_degraded": [PATH],
        "perception_backed": True,
    }
    base.update(kw)
    return base


# --------------------------------------------------------------------------- #
# trigger inference                                                           #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("cue,expected", [
    (None,                          "traverse"),
    ("",                            "traverse"),
    ("Detect(StopSign)",            "landmark"),
    ("Detect(BlueBuilding)",        "landmark"),
    ("\\text{Detect}(\\text{Bench})", "landmark"),
    ("Bearing(Right) completed",    "topology"),
    ("Bearing(Left)",               "topology"),
    ("a light brown bench",         "landmark"),   # free-text, still a look-for
])
def test_trigger_inferred_from_cue(cue, expected):
    assert trigger_of(_step(transition_cue=cue)) == expected


def test_explicit_trigger_wins_over_inference():
    step = _step(transition_cue="Detect(StopSign)", trigger="topology")
    assert trigger_of(step) == "topology"


def test_unknown_explicit_trigger_falls_back_to_inference():
    step = _step(transition_cue="Detect(StopSign)", trigger="nonsense")
    assert trigger_of(step) == "landmark"


def test_trigger_inference_matches_nl_planner():
    """brain duplicates the rule; the two must not drift apart."""
    nl = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "nl_planner",
    )
    if not os.path.isdir(nl):
        pytest.skip("nl_planner not checked out alongside brain")
    sys.path.insert(0, nl)
    try:
        from nl_planner.schemas import infer_trigger
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"nl_planner.schemas not importable: {exc}")

    for cue in [None, "", "Detect(StopSign)", "Bearing(Right) completed",
                "a light brown bench", "decision point"]:
        assert infer_trigger(cue) == trigger_of(_step(transition_cue=cue)), cue


# --------------------------------------------------------------------------- #
# acceptance sets                                                             #
# --------------------------------------------------------------------------- #

def test_accept_set_falls_back_to_equality_for_legacy_plans():
    """A hand-written plan.json with no acceptance sets behaves as it always did."""
    legacy = {"goal_cluster": 3}
    assert accept_set(legacy, degraded=False) == {3}
    assert accept_set(legacy, degraded=True) == {3}


def test_accept_set_reads_both_sets():
    step = _step(accept_clusters=[ALONG_EDGE], accept_clusters_degraded=[ALONG_EDGE, PATH])
    assert accept_set(step, degraded=False) == {ALONG_EDGE}
    assert accept_set(step, degraded=True) == {ALONG_EDGE, PATH}


# --------------------------------------------------------------------------- #
# THE case: landmark step degrades, traverse step does not                    #
# --------------------------------------------------------------------------- #

def _blue_building_step():
    """"...go down the road till you pass the blue building"."""
    return _step(
        description="pass the blue building",
        goal_mode="Along Wall",
        goal_cluster=ALONG_EDGE,
        transition_cue="Detect(BlueBuilding)",
        accept_clusters=[ALONG_EDGE],
        accept_clusters_degraded=[ALONG_EDGE, PATH],
    )


def test_landmark_step_advances_on_degraded_cluster():
    """The blue-building case: along_edge never fires, we are on path, proceed."""
    step = _blue_building_step()
    ok, why = goal_reached(PATH, step, StepProgress())
    assert ok
    assert "DEGRADED" in why


def test_landmark_step_still_prefers_the_fine_cluster():
    ok, why = goal_reached(ALONG_EDGE, _blue_building_step(), StepProgress())
    assert ok
    assert "strict" in why


def test_landmark_step_vetoes_outside_the_degraded_set():
    """Being somewhere the step cannot be must NOT open the cue check."""
    ok, why = goal_reached(PASSAGE, _blue_building_step(), StepProgress())
    assert not ok
    assert "outside" in why


def test_traverse_step_never_degrades():
    """"go down the road" — the cluster IS the evidence, so no widening."""
    step = _step(
        goal_mode="Along Wall",
        trigger="traverse",
        accept_clusters=[ALONG_EDGE],
        accept_clusters_degraded=[ALONG_EDGE, PATH],
    )
    prog = StepProgress(dwell_frames=1)
    assert goal_reached(PATH, step, prog)[0] is False
    assert goal_reached(ALONG_EDGE, step, prog)[0] is True


def test_traverse_requires_sustained_membership():
    """One flickered frame must not advance the plan."""
    step = _step(trigger="traverse", accept_clusters=[PATH], accept_clusters_degraded=[PATH])
    prog = StepProgress(dwell_frames=3)
    assert goal_reached(PATH, step, prog)[0] is False   # 1/3
    assert goal_reached(PATH, step, prog)[0] is False   # 2/3
    assert goal_reached(PATH, step, prog)[0] is True    # 3/3


def test_traverse_dwell_resets_on_leaving_the_set():
    step = _step(trigger="traverse", accept_clusters=[PATH], accept_clusters_degraded=[PATH])
    prog = StepProgress(dwell_frames=3)
    goal_reached(PATH, step, prog)
    goal_reached(PATH, step, prog)
    goal_reached(JUNCTION, step, prog)                  # left the set -> reset
    assert prog.dwell == 0
    assert goal_reached(PATH, step, prog)[0] is False   # back to 1/3


def test_subsumption_lets_a_road_step_be_satisfied_at_a_junction():
    """Upward relation: junction IS on a road, so a Road: On step is satisfied."""
    step = _step(
        goal_mode="Road: On",
        trigger="traverse",
        accept_clusters=[PATH, ALONG_EDGE, PASSAGE, JUNCTION],
        accept_clusters_degraded=[PATH, ALONG_EDGE, PASSAGE, JUNCTION],
    )
    assert goal_reached(JUNCTION, step, StepProgress(dwell_frames=1))[0] is True


# --------------------------------------------------------------------------- #
# topology: intersections where the cluster is not perception-backed          #
# --------------------------------------------------------------------------- #

def test_topology_step_does_not_wait_for_an_unbacked_cluster():
    """On the real campus junction is chance-level; don't wait for it."""
    step = _step(
        goal_mode="Intersection: In",
        goal_cluster=JUNCTION,
        transition_cue="Bearing(Right) completed",
        accept_clusters=[JUNCTION],
        accept_clusters_degraded=[JUNCTION, PATH],
        perception_backed=False,
    )
    # Even sitting on plain `path`, the cluster does not block the step.
    ok, why = goal_reached(PATH, step, StepProgress())
    assert ok
    assert "plan construct" in why


def test_topology_step_uses_the_cluster_when_it_is_backed():
    step = _step(
        goal_mode="Intersection: In",
        goal_cluster=JUNCTION,
        transition_cue="Bearing(Right) completed",
        accept_clusters=[JUNCTION],
        accept_clusters_degraded=[JUNCTION, PATH],
        perception_backed=True,
    )
    assert goal_reached(JUNCTION, step, StepProgress())[0] is True
    assert goal_reached(PASSAGE, step, StepProgress())[0] is False


# --------------------------------------------------------------------------- #
# bearing completion                                                          #
# --------------------------------------------------------------------------- #

def test_bearing_needs_the_named_direction():
    right = _step(transition_cue="Bearing(Right) completed")
    left = _step(transition_cue="Bearing(Left) completed")
    # -90 deg = clockwise = right (REP-103: +z is counter-clockwise).
    turned_right = math.radians(-90)
    turned_left = math.radians(90)

    assert bearing_complete(right, turned_right, 0.0, 60.0) is True
    assert bearing_complete(right, turned_left, 0.0, 60.0) is False
    assert bearing_complete(left, turned_left, 0.0, 60.0) is True
    assert bearing_complete(left, turned_right, 0.0, 60.0) is False


def test_bearing_needs_to_exceed_the_threshold():
    right = _step(transition_cue="Bearing(Right) completed")
    assert bearing_complete(right, math.radians(-30), 0.0, 60.0) is False
    assert bearing_complete(right, math.radians(-75), 0.0, 60.0) is True


def test_bearing_is_false_without_odometry():
    """hockfield/akila_data cached no odom topic at all — must not crash."""
    right = _step(transition_cue="Bearing(Right) completed")
    assert bearing_complete(right, None, 0.0, 60.0) is False
    assert bearing_complete(right, 0.0, None, 60.0) is False


def test_bearing_handles_the_pi_seam():
    """A turn straddling +/-pi must wrap, not read as a ~270 deg turn the other way."""
    right = _step(transition_cue="Bearing(Right) completed")
    left = _step(transition_cue="Bearing(Left) completed")

    # Turning RIGHT (clockwise) from -170 deg wraps through -180 to +100 deg.
    # Raw difference is +270; wrapped it is -90, i.e. a right turn.
    entry, now = math.radians(-170), math.radians(100)
    assert abs(math.degrees(wrap_pi(now - entry)) + 90) < 1e-9
    assert bearing_complete(right, now, entry, 60.0) is True
    assert bearing_complete(left, now, entry, 60.0) is False

    # And the mirror image: turning LEFT from +170 deg wraps to -100 deg = +90.
    entry, now = math.radians(170), math.radians(-100)
    assert abs(math.degrees(wrap_pi(now - entry)) - 90) < 1e-9
    assert bearing_complete(left, now, entry, 60.0) is True
    assert bearing_complete(right, now, entry, 60.0) is False


def test_bearing_direction_parsing():
    assert bearing_direction("Bearing(Right) completed") == "right"
    assert bearing_direction("Bearing(Left)") == "left"
    assert bearing_direction("Bearing(Straight)") == "straight"
    assert bearing_direction("Detect(StopSign)") is None
    assert bearing_direction(None) is None


# --------------------------------------------------------------------------- #
# ordinals: "stop at the 2nd bench"                                           #
# --------------------------------------------------------------------------- #

def test_first_sighting_satisfies_ordinal_one():
    prog = StepProgress()
    assert prog.note_cue(True, 1) is True


def test_same_landmark_held_in_view_counts_once():
    """The bug this guards: a bench stays in frame for many polls."""
    prog = StepProgress(cue_lost_polls=2)
    assert prog.note_cue(True, 2) is False    # sighting 1
    assert prog.note_cue(True, 2) is False    # still the SAME bench
    assert prog.note_cue(True, 2) is False
    assert prog.sightings == 1


def test_second_distinct_sighting_satisfies_ordinal_two():
    prog = StepProgress(cue_lost_polls=2)
    prog.note_cue(True, 2)                    # bench 1 enters view
    prog.note_cue(True, 2)                    # still bench 1
    prog.note_cue(False, 2)                   # 1 absent poll — not yet "lost"
    assert prog.in_view is True
    prog.note_cue(False, 2)                   # 2 absent polls — now lost
    assert prog.in_view is False
    assert prog.note_cue(True, 2) is True     # bench 2
    assert prog.sightings == 2


def test_brief_dropout_does_not_double_count():
    """A single missed poll mid-approach must not manufacture a second bench."""
    prog = StepProgress(cue_lost_polls=3)
    prog.note_cue(True, 2)
    prog.note_cue(False, 2)                   # flicker
    assert prog.note_cue(True, 2) is False
    assert prog.sightings == 1


def test_reset_clears_everything():
    prog = StepProgress(dwell_frames=2)
    prog.note_cluster(PATH, {PATH})
    prog.note_cue(True, 3)
    prog.reset(entry_yaw=1.23)
    assert prog.dwell == 0
    assert prog.sightings == 0
    assert prog.in_view is False
    assert prog.entry_yaw == 1.23


# --------------------------------------------------------------------------- #
# Regression: unbacked landmark modes must not wedge the plan                 #
# --------------------------------------------------------------------------- #

def test_landmark_step_on_unbacked_mode_without_a_degrade_entry():
    """A landmark step whose mode no classifier can emit must still open the cue.

    The degraded set only rescues modes that happen to have a DEGRADE_TO entry.
    A mode that is not perception-backed AND has no coarser fallback (so
    accept == degraded == the unreachable id) would otherwise never open the cue
    check, and the plan would wedge with no diagnostic. The topology branch
    always had this guard; the landmark branch did not.
    """
    step = _step(
        description="stop at the gate",
        goal_mode="Gate: At",
        goal_cluster=99,                       # an id the classifier never emits
        transition_cue="Detect(Gate)",
        accept_clusters=[99],
        accept_clusters_degraded=[99],         # no coarser fallback at all
        perception_backed=False,
    )
    ok, why = goal_reached(PATH, step, StepProgress())
    assert ok, f"unbacked landmark step wedged: {why}"
    assert "plan construct" in why


def test_landmark_step_on_BACKED_mode_still_vetoes():
    """The guard must not turn into 'always advance' for perceivable modes."""
    step = _step(
        goal_mode="Passage",
        goal_cluster=PASSAGE,
        transition_cue="Detect(Bridge)",
        accept_clusters=[PASSAGE],
        accept_clusters_degraded=[PASSAGE],
        perception_backed=True,
    )
    ok, why = goal_reached(OPEN_SPACE, step, StepProgress())
    assert not ok
    assert "outside" in why


def test_sightings_survive_a_cue_timeout():
    """Deliberate: a timed-out '2nd bench' step must not demand two MORE benches.

    Mirrors what brain_controller._cue_check_timed_out does — it zeroes `dwell`
    but leaves `sightings` alone, because a sighting is a real observation.
    """
    prog = StepProgress(dwell_frames=2, cue_lost_polls=1)
    prog.note_cue(True, 2)                     # bench 1 seen
    prog.note_cluster(PATH, {PATH})            # some dwell accumulated
    assert prog.sightings == 1

    # ---- the timeout path ----
    prog.dwell = 0                             # what the controller clears
    assert prog.sightings == 1, "sightings must survive the timeout"

    prog.note_cue(False, 2)                    # bench 1 leaves view
    assert prog.note_cue(True, 2) is True      # bench 2 completes the step


# --------------------------------------------------------------------------- #
# Regression: the cue-timeout must not oscillate                              #
# --------------------------------------------------------------------------- #

def test_abandoned_cue_latch():
    """A timed-out cue must not re-arm until the robot leaves the accept set.

    The bug: _cue_check_timed_out reverted to NAVIGATING, but the cluster
    condition that opened the cue check was still true (the robot had not moved),
    so the very next /predicted_cluster message re-entered CHECKING_CUE. That is
    an infinite NAVIGATING<->CHECKING_CUE oscillation paying for a VLM call every
    poll interval, not the clean give-up the timeout was meant to provide.
    """
    prog = StepProgress()
    assert prog.cue_abandoned is False

    prog.note_cue(True, 3)                      # saw it once, needed 3
    prog.abandon_cue()                          # timeout fires
    assert prog.cue_abandoned is True
    assert prog.dwell == 0
    # Sightings survive the give-up (we really did see one).
    assert prog.sightings == 1

    assert prog.clear_cue_abandoned() is True   # robot left the accept set
    assert prog.cue_abandoned is False
    assert prog.clear_cue_abandoned() is False  # idempotent


def test_reset_clears_the_abandoned_latch():
    """A new step must start with a live cue, never an inherited give-up."""
    prog = StepProgress()
    prog.abandon_cue()
    prog.reset()
    assert prog.cue_abandoned is False


def test_topology_cue_is_a_maneuver_not_a_visual_cue():
    """Guards the category error: Bearing(...) is not something a camera sees.

    brain_controller._on_arrived_at_goal must skip the visual cue check for
    topology steps. This pins the classification the guard depends on.
    """
    step = _step(transition_cue="Bearing(Right) completed")
    assert trigger_of(step) == "topology"
    # ...and a Detect cue on the same step shape is still a visual check.
    assert trigger_of(_step(transition_cue="Detect(Bench)")) == "landmark"


# --------------------------------------------------------------------------- #
# InvariantMonitor — the left half of `Phi_X U cue`                            #
# --------------------------------------------------------------------------- #

from brain.trigger_policy import InvariantMonitor  # noqa: E402

#: "stay on the path until you see a cone", then the robot clips a corner across
#: open_space for three readings and rejoins. accept = the path ids.
_CLIP_CORNER = [1, 1, 9, 9, 9, 1, 1]
_ACCEPT = {1, 2, 3}


def _run(policy, dwell=5, binding=True, seq=_CLIP_CORNER):
    m = InvariantMonitor(policy=policy, dwell_frames=dwell)
    breaches = [m.observe(c, _ACCEPT, binding=binding) for c in seq]
    return m, breaches


def test_invariant_off_is_a_true_no_op():
    """The default must not measure OR act — behaviour identical to before."""
    m, breaches = _run("off")
    assert not any(breaches)
    assert m.summary()["violations"] == 0
    assert m.summary()["frames_violating"] == 0


def test_invariant_log_measures_but_never_breaches():
    m, breaches = _run("log")
    assert not any(breaches)
    s = m.summary()
    assert s["violations"] == 1 and s["frames_violating"] == 3
    assert s["longest_run"] == 3 and s["breached"] is False


def test_invariant_dwell_breaches_on_the_nth_consecutive_reading():
    m, breaches = _run("dwell", dwell=3)
    assert breaches.index(True) == 4      # 3rd violating reading (indices 2,3,4)
    assert m.summary()["breached"] is True


def test_hard_fail_is_dwell_with_n_equals_one():
    """The three policies are one mechanism: hard-fail is not a separate path."""
    m, breaches = _run("dwell", dwell=1)
    assert [i for i, b in enumerate(breaches) if b] == [2, 3, 4]
    assert m.summary()["breached"] is True


def test_advisory_invariant_is_measured_but_never_breaches():
    """`binding=False` separates "follow the road past the cones" (advisory) from
    "do not cut across the grass" (binding) without a second knob."""
    m, breaches = _run("dwell", dwell=1, binding=False)
    assert not any(breaches)
    assert m.summary()["violations"] == 1     # still counted


def test_separate_episodes_are_counted_separately():
    m, _ = _run("log", seq=[1, 9, 1, 9, 9, 1])
    s = m.summary()
    assert s["violations"] == 2 and s["frames_violating"] == 3 and s["longest_run"] == 2


def test_empty_accept_set_never_violates():
    """A mode with no perception backing must not read as a permanent violation."""
    m = InvariantMonitor(policy="dwell", dwell_frames=1)
    assert not any(m.observe(c, set()) for c in [9, 9, 9])
    assert m.summary()["violations"] == 0


def test_unknown_policy_rejected():
    import pytest as _pytest
    with _pytest.raises(ValueError, match="policy must be one of"):
        InvariantMonitor(policy="fail-fast")


# --------------------------------------------------------------------------- #
# the PURE DECISION STEP — `X -> X` with branches, which does not move         #
#                                                                              #
# `taxonomy.validate_plan_transitions` exempts this shape from the `X -> X`    #
# rule on purpose ("a decision step stays put while the VLM decides") and the  #
# generator emits it. Without explicit handling, require_cluster_change demands #
# a move from a step whose point is not to move, and StepTargeter hunts a      #
# SECOND junction adjacent to the one we stand in ("plan and map disagree").   #
# --------------------------------------------------------------------------- #

def _decision_step(**kw):
    """A junction decision step with the REAL degraded set, not a tidy one.

    `to_brain_tree` writes the subsumption union into `accept_clusters_degraded` —
    on town01 that is every region in the map, because a junction is also a path.
    Hand-writing `[JUNCTION]` here would have hidden the bug this pins: the guard
    must key off the STRICT set, or the decision step fires wherever the robot
    stands and resolves its branches at the wrong junction.
    """
    base = dict(start_mode="junction", goal_mode="junction",
                start_cluster=JUNCTION, goal_cluster=JUNCTION,
                accept_clusters=[JUNCTION],
                accept_clusters_degraded=[JUNCTION, OPEN_SPACE, PATH,
                                          ALONG_EDGE, PASSAGE],
                branches=[{"vlm_cue": "a cone", "sub_plan": []},
                          {"vlm_cue": "default", "sub_plan": []}])
    base.update(kw)
    return _step(**base)


def test_decision_step_is_satisfied_without_moving():
    """require_cluster_change must not apply to a step that by definition stays put.

    This is the exception to that flag's rule that it "cannot make a reachable step
    unreachable".
    """
    progress = StepProgress(require_cluster_change=True, start_cluster=JUNCTION)
    ok, why = goal_reached(JUNCTION, _decision_step(), progress)
    assert ok, why
    assert "decision" in why


def test_decision_step_is_satisfied_on_the_very_first_tick():
    """No dwell either — there is nothing to traverse, so routing never runs.

    That matters beyond tidiness: on any tick where the step is not yet done the
    caller grounds a target, and grounding `junction` from inside a junction is
    the exact failure this fixes.
    """
    progress = StepProgress(dwell_frames=3, require_cluster_change=True,
                            start_cluster=JUNCTION)
    ok, _ = goal_reached(JUNCTION, _decision_step(), progress)
    assert ok


def test_decision_step_is_NOT_satisfied_before_arriving():
    """It still has to be at the place the decision is made."""
    progress = StepProgress(require_cluster_change=True, start_cluster=PATH)
    ok, why = goal_reached(PATH, _decision_step(), progress)
    assert not ok
    assert "not yet at" in why
    # PATH is in the step's DEGRADED set (junctions subsume paths). Keying the guard
    # off `degraded` would make this pass anywhere in the map.
    assert PATH in _decision_step()["accept_clusters_degraded"]


def test_a_branching_step_that_DOES_move_keeps_the_normal_rules():
    """`path -> junction` with branches is a traversal; only `X -> X` is exempt."""
    step = _decision_step(start_mode="path", goal_mode="junction")
    progress = StepProgress(require_cluster_change=True, start_cluster=JUNCTION)
    ok, why = goal_reached(JUNCTION, step, progress)
    assert not ok
    assert "start cluster" in why


def test_a_non_branching_self_loop_keeps_the_normal_rules():
    """`path -> path` with no branches is "stop at the Nth bench" — it moves."""
    step = _step(start_mode="path", goal_mode="path")
    progress = StepProgress(require_cluster_change=True, start_cluster=PATH)
    ok, why = goal_reached(PATH, step, progress)
    assert not ok
    assert "start cluster" in why


# --------------------------------------------------------------------------- #
# ConstraintMonitor — global negative constraint, []~X                         #
#                                                                              #
# Lives here rather than in brain_controller because that module cannot be     #
# imported without rclpy, so anything defined only inside the node is          #
# untestable and drifts from the offline harness (the same advancement policy  #
# written twice diverged in several places).                                   #
# --------------------------------------------------------------------------- #

def test_an_undeclared_constraint_is_a_true_noop():
    """No forbid set means the plan asserted nothing — not that it passed a check.

    `satisfied` is vacuously True and `csr` is 1.0, but `declared` is False so a caller
    can tell "nothing forbidden" from "nothing violated". Collapsing those is how a
    constraint that was never configured comes to look like one that held.
    """
    from brain.trigger_policy import ConstraintMonitor
    c = ConstraintMonitor()
    assert not c.declared
    for cl in (0, 1, 2, 99):
        assert c.observe(cl) is False
    assert c.ticks == 0 and c.csr == 1.0 and c.satisfied


def test_violations_are_counted_as_a_rate_not_a_boolean():
    """With a noisy classifier one forbidden reading is more likely noise than entry.

    A boolean would make the metric a coin toss on the noisiest class, so the monitor
    reports a rate and leaves the policy to the caller.
    """
    from brain.trigger_policy import ConstraintMonitor
    c = ConstraintMonitor({7})
    for cl in (1, 1, 7, 1, 1, 1, 1, 1, 1, 1):
        c.observe(cl)
    assert c.ticks == 10 and c.violations == 1
    assert abs(c.csr - 0.9) < 1e-9
    assert not c.satisfied


def test_longest_violation_run_is_tracked_separately_from_the_count():
    """Ten scattered frames and ten consecutive ones are different events.

    Burst length is what distinguishes a misclassification from actually driving
    through the forbidden region — misclassification runs are typically a few frames
    long, so the count alone cannot tell them apart.
    """
    from brain.trigger_policy import ConstraintMonitor
    scattered = ConstraintMonitor({7})
    for cl in [7, 1, 7, 1, 7, 1]:
        scattered.observe(cl)
    burst = ConstraintMonitor({7})
    for cl in [1, 7, 7, 7, 1, 1]:
        burst.observe(cl)
    assert scattered.violations == burst.violations == 3
    assert scattered.longest_run == 1
    assert burst.longest_run == 3


def test_a_none_reading_is_not_counted():
    """No cluster observed yet is not a satisfied tick, and not a violating one."""
    from brain.trigger_policy import ConstraintMonitor
    c = ConstraintMonitor({7})
    assert c.observe(None) is False
    assert c.ticks == 0


def test_summary_reports_everything_the_scorer_needs():
    from brain.trigger_policy import ConstraintMonitor
    c = ConstraintMonitor({4, 5})
    for cl in (1, 4, 1):
        c.observe(cl)
    s = c.summary()
    assert s["forbid_declared"] and s["forbid_clusters"] == [4, 5]
    assert s["constraint_ticks"] == 3 and s["violation_ticks"] == 1
    assert abs(s["csr"] - 0.6667) < 1e-3
    assert s["constraint_satisfied"] is False


def test_the_state_enum_has_every_member_the_controller_assigns():
    """Both enforcement paths assigned `State.ERROR`, which did not exist.

    A constraint violation under `forbid_policy="fail"` and every InvariantMonitor breach
    raised AttributeError inside the /predicted_cluster callback, so the two paths that
    are supposed to STOP the robot crashed instead. The `log` default never exercises
    those paths, so nothing else catches it.

    brain_controller cannot be imported here (no rclpy), so this reads the source rather
    than the module. Crude, but it is the only check that can run in this environment and
    it pins the exact failure.
    """
    import os
    import re
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "brain", "brain_controller.py")
    src = open(p).read()
    block = src.split("class State(Enum):")[1].split("\n\n")[0]
    defined = set(re.findall(r"^\s+([A-Z_]+)\s*=", block, re.M))
    assigned = set(re.findall(r"State\.([A-Z_]+)", src))
    missing = assigned - defined
    assert not missing, f"brain_controller assigns State members that do not exist: {missing}"


def test_a_cue_naming_an_object_and_a_place_answers_about_the_OBJECT():
    """The branch bug: "traffic cone is present in the intersection" matched BOTH
    `cone` and `intersection`, and taking the first dict hit gave the PLACE answer --
    true at every junction -- so the branch fired unconditionally and the robot took
    option 1 whether or not the cone was there. Whether it bit depended only on how
    the cue happened to be phrased.
    """
    # the publisher's order IS the specificity order; object keys precede place keys
    answers = {
        "a traffic cone is in the intersection": False,
        "Detect(TrafficCone)": False,
        "cone": False,
        "Detect(Intersection)": True,
        "intersection": True,
    }
    cue = "traffic cone is present in the intersection"
    low = cue.lower()
    hits = [(k, v) for k, v in answers.items() if k.lower() in low or low in k.lower()]
    assert hits, "the cue must match something"
    assert hits[0][1] is False, (
        f"first match is {hits[0][0]!r}={hits[0][1]} -- a cone cue must not be answered "
        f"by the intersection key")
    assert len({v for _, v in hits}) > 1, "this cue is genuinely ambiguous; brain warns"


def test_ordinal_counting_fails_when_two_instances_are_visible_at_once():
    """A real limitation of the temporal de-bounce, pinned rather than assumed.

    `note_cue` counts a DISTINCT sighting only after the cue has read NO for
    `cue_lost_polls` consecutive polls. That de-bounce exists for a good reason -- without
    it "stop at the 2nd bench" is satisfied by two consecutive polls of the FIRST bench,
    because a landmark stays in frame for many seconds on approach.

    But it assumes the instances are SEPARATED IN TIME. If the second bench comes into
    view while the first is still visible -- which is ordinary on a straight path with
    two benches thirty metres apart -- the cue never drops, the counter stays at 1, and
    the step waits until its cue budget expires. It does not fail; it TIMES OUT, which is
    the failure mode that reads as a planner bug.

    Recorded as a test because the fix is a design choice: counting requires either
    instance identity (which cue answers do not carry --
    they are booleans) or a spatial gate (count a new sighting once the robot has moved
    N metres since the last one). The second is cheap and does not need perception to
    change; the first is correct and needs a different cue contract.
    """
    from brain.trigger_policy import StepProgress

    def sightings(seq):
        p = StepProgress(dwell_frames=3, cue_lost_polls=2)
        for v in seq:
            p.note_cue(v, needed=2)
        return p.sightings

    # separated in time: works as designed
    assert sightings([1, 1, 1, 0, 0, 0, 1, 1, 1]) == 2
    # both in frame at once: the counter never advances
    assert sightings([1] * 9) == 1
    # and a gap shorter than cue_lost_polls does not count either
    assert sightings([1, 1, 1, 1, 1, 0, 1, 1, 1]) == 1


def test_spatial_gate_separates_instances_the_debounce_cannot_and_what_it_costs():
    """The spatial gate, and the failure mode it trades for.

    WHAT IT FIXES. Two landmarks in frame at once -- ordinary for two benches thirty
    metres apart on a straight path -- never separate under the temporal de-bounce: the
    cue never drops, the counter sticks at 1, and the step times out rather than failing.

    WHAT IT COSTS, and this is why it is OFF by default. The gate counts a new sighting
    every `resight_metres` of travel while the cue stays true. It cannot tell "I have
    driven past a second bench" from "I have driven a long way with one bench still in
    view", because a cue answer is a boolean and carries no instance identity. So a long
    approach to a SINGLE landmark over-counts, which advances the step early -- a wrong
    answer where the old behaviour was a hang.

    Neither is correct. The gate is opt-in (`travelled=None` keeps the temporal rule
    exactly), so the choice of which failure to take is the caller's. The alternative is
    to state the assumption that instances are seen one at a time.
    """
    from brain.trigger_policy import StepProgress

    def sightings(seq, dist=None, resight=12.0):
        p = StepProgress(dwell_frames=3, cue_lost_polls=2, resight_metres=resight)
        for i, v in enumerate(seq):
            p.note_cue(v, 2, travelled=None if dist is None else dist[i])
        return p.sightings

    # unchanged when the caller passes no distance -- every recorded trace scores as before
    assert sightings([1] * 9) == 1
    assert sightings([1, 1, 1, 0, 0, 0, 1, 1, 1]) == 2

    # the fix: both in frame, robot moves past them
    assert sightings([1] * 9, [i * 4.0 for i in range(9)]) >= 2

    # fails safe when the robot has not moved
    assert sightings([1] * 9, [0.0] * 9) == 1

    # THE COST, asserted so it is not a surprise later: a long approach to ONE landmark
    # over-counts, because distance is a proxy for instance identity and not the thing
    assert sightings([1] * 9, [i * 4.0 for i in range(9)]) == 3


def test_a_topology_step_cannot_advance_from_the_cluster_path_without_the_maneuver():
    """The turn must be tested on BOTH advancement paths, not just the odometry one.

    `_on_arrived_at_goal` is reached two ways: from `_odom_cb`, which calls
    `bearing_complete` before it calls, and from `_cluster_cb`, which does not. The
    topology branch skipped the visual cue check on the grounds that heading had already
    confirmed the maneuver — true on the first path, false on the second. So a topology
    step advanced the moment the vehicle entered any cluster in the goal mode's accept
    set, whether or not it turned.

    A straight drive through a step whose cue is `Bearing(Right) completed` was then
    reported as "Navigation plan complete". Region-sequence scoring cannot see it — both
    exits of the junction are in the accept set.

    brain_controller cannot be imported here (no rclpy), so this reads the source, the
    same way test_the_state_enum_has_every_member_the_controller_assigns does. It pins
    the one thing that matters: the topology branch tests the heading before it advances.
    """
    import os
    import re
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "brain", "brain_controller.py")
    src = open(p).read()
    # the topology branch of _on_arrived_at_goal, up to where it hands off
    body = src.split('if cue and trigger == "topology":')[1]
    branch = body.split("self._after_step_satisfied()")[0]
    assert "_bearing_complete(" in branch, (
        "the topology branch of _on_arrived_at_goal advances without testing the "
        "heading; a step whose cue is Bearing(Right) will complete on a straight drive"
    )


def test_the_offline_and_online_advancement_paths_agree_on_topology():
    """StepAdvancer and brain_controller must not disagree about what finishes a step.

    One way they can: the offline path tests the heading while the online cluster path
    does not. Pin that both call the shared predicate rather than re-deriving it.
    """
    import os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for mod in ("brain_controller.py", "step_advancer.py"):
        src = open(os.path.join(here, "brain", mod)).read()
        assert "_bearing_complete(" in src or "bearing_complete(" in src, (
            f"{mod} does not consult trigger_policy.bearing_complete"
        )


def test_the_hold_check_is_not_behind_the_navigating_guard():
    """`Phi_X U cue` must be evaluated while WAITING for the cue, not only while moving.

    The hold covers the run-up to a cue -- that is what the "until" means. But a
    `landmark` step with a broad accept set satisfies `goal_reached` on its first reading,
    so the brain enters CHECKING_CUE straight away and spends the whole run-up there. With
    the hold check behind the NAVIGATING guard it would never be evaluated on the one
    step shape it exists for, and a `hold_mode` would change nothing about a drive.
    `_check_constraint` and `_check_require` sit in front of the guard for the same reason.

    brain_controller cannot be imported here (no rclpy), so this reads the source.
    """
    import os
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "brain", "brain_controller.py")
    src = open(p).read()
    body = src.split("def _cluster_cb")[1]
    hold = body.index("_check_hold")
    guard = body.index("if self.state != State.NAVIGATING")
    assert hold < guard, (
        "_check_hold is evaluated after the NAVIGATING guard, so a hold covering the "
        "run-up to a cue is skipped for the whole time the brain waits in CHECKING_CUE"
    )


def test_brain_feeds_the_odometer_to_the_spatial_debounce():
    """`resight_metres` is only live in the driven path if brain passes the odometer.

    `StepProgress.note_cue` takes a `travelled` argument; called with two arguments, the
    SPATIAL gate never runs in CARLA. Its own docstring describes what that costs: two landmarks thirty metres apart are both in
    frame at once, the cue never goes false, the counter sticks at 1, and the step TIMES
    OUT rather than failing.

    Asserted on the source because exercising it needs a ROS node. Both halves matter:
    the odometer must exist, and it must actually reach note_cue.
    """
    import pathlib
    src = (pathlib.Path(__file__).parents[1] / "brain" / "brain_controller.py").read_text()
    assert "self._travelled" in src, "brain has no odometer"
    assert "travelled=self._travelled if self._cue_spatial_debounce else None" in src, \
        "note_cue is still called without the odometer -- resight_metres cannot fire"
    # Map-free: the odometer is integrated from SPEED, never from position.
    assert "msg.pose.pose.position" not in src, \
        "brain must not use odometry POSITION -- the plan is map-free by design"


def test_the_frame_handed_to_the_vlm_reports_its_age():
    """A stale frame is indistinguishable from a model error without this.

    Answers to the same branch at nearly the same distance from the same cone can differ;
    frame age is the quantity otherwise invisible in the log. Age is reported in METRES as well as
    seconds, because what degrades the answer is how much further away the robot was when
    the shutter opened -- a stationary robot loses nothing to a 10 s old frame.
    """
    import pathlib
    src = (pathlib.Path(__file__).parents[1] / "brain" / "brain_controller.py").read_text()
    assert "_image_rx_travelled" in src, "the frame is not stamped with the odometer"
    # THE AGE MUST BE SNAPSHOTTED AT COPY TIME, not read inside the VLM worker.
    # Reading `_image_rx_t` after the OpenAI round-trip is wrong: newer frames have
    # overwritten it by then, so the number describes a DIFFERENT image than the one
    # that was sent. Both call sites must snapshot beside the `.copy()` and pass it down.
    assert src.count("age_snap = self._snapshot_frame_age()") >= 2, \
        "frame age must be snapshotted where the image is COPIED, on both VLM paths"
    for call in ("args=(cue, image_copy, age_snap)",
                 "args=(branches, image_copy, age_snap)"):
        assert call in src, f"the snapshot is not passed to the worker: {call}"
    assert "self._age_str(age)" in src, \
        "the logged age must come from the snapshot, not from a live re-read"


def test_the_branch_decision_is_not_rate_limited_by_the_spend_knob():
    """Poll rate and OpenAI spend limit are different things and must not be conflated.

    `vlm_check_interval` (2.0 s) exists to bound what re-polling a cue costs. If it is
    also the delay before the FIRST look, brain waits up to 2 s after entering DECIDING
    before copying a frame -- ~10 m at 5 m/s, enough to drive past the cue. Late
    decisions then show an empty road from inside the junction: the model answers
    correctly about a picture taken too late.
    """
    import pathlib
    src = (pathlib.Path(__file__).parents[1] / "brain" / "brain_controller.py").read_text()
    assert "self.create_timer(min(0.2, cue_interval), self._vlm_timer_cb)" in src, \
        "the VLM timer is back on the spend interval -- the first look is late again"
    assert "_cue_repoll_interval" in src, "the spend limit has no home on the cue path"
    # and a new step must not inherit the previous step's repoll timestamp
    # Slice to the NEXT def rather than a character count, so the test fails on
    # behaviour rather than on formatting.
    reset = src.split("def _reset_step_progress")[1].split("\n    def ")[0]
    assert "_last_cue_poll_t = 0.0" in reset, \
        "a new step does not get its first look immediately"


# --------------------------------------------------------------------------- #
# Bearing scope: the check that certified turns which never happened
# --------------------------------------------------------------------------- #

_RIGHT_STEP = {"transition_cue": "Bearing(Right) completed"}


def test_road_curvature_no_longer_certifies_a_turn_that_never_happened():
    """The unbounded check certifies a turn built from road curvature.

    `Bearing(Right) completed` can be satisfied while the junction the turn was asked for
    saw only a few degrees: the 60 deg comes from ~150 m of ordinary route curvature after
    the decision point, so a vehicle that drove straight through is certified as having
    turned.
    """
    turned = math.radians(-65.0)      # enough heading change, wrong place
    # Unbounded (the old behaviour) accepts it...
    assert bearing_complete(_RIGHT_STEP, turned, 0.0, 60.0) is True
    # ...and so does a scope the manoeuvre genuinely fits inside.
    assert bearing_complete(_RIGHT_STEP, turned, 0.0, 60.0,
                            travelled_since=20.0, scope_m=45.0) is True
    # But 150 m later it is the road bending, not the turn.
    assert bearing_complete(_RIGHT_STEP, turned, 0.0, 60.0,
                            travelled_since=150.0, scope_m=45.0) is False


def test_the_scope_is_opt_outable_and_does_not_disturb_recorded_traces():
    """`scope_m=None` or 0 must reproduce the unbounded behaviour exactly.

    Recorded traces were scored without this bound, so it has to be possible to reproduce
    them rather than silently re-baselining.
    """
    turned = math.radians(-65.0)
    for scope in (None, 0.0):
        assert bearing_complete(_RIGHT_STEP, turned, 0.0, 60.0,
                                travelled_since=10_000.0, scope_m=scope) is True


def test_the_scope_does_not_rescue_a_wrong_direction_turn():
    """Scoping bounds WHERE, not WHETHER. A left turn still cannot satisfy Bearing(Right)."""
    left = math.radians(+70.0)
    assert bearing_complete(_RIGHT_STEP, left, 0.0, 60.0,
                            travelled_since=5.0, scope_m=45.0) is False


def test_brain_passes_the_odometer_datum_to_both_bearing_call_sites():
    """Two call sites; one unscoped would leave the bug live on that path.

    Asserted on the source because exercising them needs a ROS node -- the same shape as
    the other wiring tests here, and the reason the cue vocabulary has a drift test.
    """
    import pathlib
    src = (pathlib.Path(__file__).parents[1] / "brain" / "brain_controller.py").read_text()
    assert src.count("scope_m=self._bearing_scope_m") == 2, \
        "a Bearing(...) call site is still unbounded"
    assert src.count("travelled_since=self._travelled_this_step()") == 2
    reset = src.split("def _reset_step_progress")[1].split("\n    def ")[0]
    assert "entry_travelled = self._travelled" in reset, \
        "the odometer datum is not taken when a step begins, so the scope has no origin"


def test_a_manoeuvre_is_anchored_at_the_decision_point_after_reset():
    """Ordering matters: `_progress.reset()` sets entry_yaw and clears the odometer datum.

    Applying the decision-point datum BEFORE reset() lets reset overwrite it, making the
    datum silently inert. The assignment must come after, and `_last_cue_poll_t` must be
    reset there too.
    """
    import pathlib as _p
    src = (_p.Path(__file__).parents[1] / "brain" / "brain_controller.py").read_text()
    body = src.split("def _reset_step_progress")[1].split("\n    def ")[0]
    i_reset = body.index("self._progress.reset(")
    i_datum = body.index("self._progress.entry_travelled = self._travelled")
    assert i_datum > i_reset, \
        "the odometer datum is set BEFORE _progress.reset(), which then clears it"
    assert "_decide_datum = (self._yaw, self._travelled)" in src, \
        "nothing captures yaw+odometer when DECIDING begins"
    # The yaw datum must NOT be back-dated to the decision point: the approach into the
    # junction already contains heading change, so counting it toward the threshold
    # certifies the turn early and truncates it.
    assert "self._progress.entry_yaw = yaw0" not in body, \
        "the manoeuvre's yaw datum is back-dated again -- this truncates the turn"


# --------------------------------------------------------------------------- #
# The accumulation WINDOW: heading change counted only where the manoeuvre belongs
# --------------------------------------------------------------------------- #

def test_heading_accumulates_only_inside_the_window():
    """Curvature outside the junction must not count toward the turn.

    The original check differenced current yaw against the step's entry yaw, so ~150 m of
    ordinary route curvature satisfied `Bearing(Right)` while the junction saw a few degrees.
    Integrating tick-to-tick and gating on the window fixes that at the source.
    """
    pr = StepProgress()
    pr.reset()
    # 40 deg of real turning at the junction
    pr.note_heading(0.0, True)
    for d in range(1, 41):
        pr.note_heading(math.radians(-d), True)
    assert pr.bearing_accum == pytest.approx(-40.0, abs=1.0)
    # then 200 deg of road bending far away -- must not accumulate
    for d in range(1, 201):
        pr.note_heading(math.radians(-40 - d), False)
    assert pr.bearing_accum == pytest.approx(-40.0, abs=1.0), \
        "heading outside the window was counted toward the manoeuvre"


def test_leaving_and_returning_to_the_window_does_not_double_count():
    """Re-datum on every reading, accumulate only in-window -- so a gap is skipped, not
    folded in as one large jump when the window reopens."""
    pr = StepProgress()
    pr.reset()
    pr.note_heading(0.0, True)
    pr.note_heading(math.radians(-10), True)          # -10 in window
    pr.note_heading(math.radians(-90), False)         # big change, OUT of window
    pr.note_heading(math.radians(-100), True)         # back in: only this -10 counts
    assert pr.bearing_accum == pytest.approx(-20.0, abs=1.0)


def test_bearing_complete_judges_the_accumulator_when_given_one():
    right = {"transition_cue": "Bearing(Right) completed"}
    # The windowed value decides, regardless of the raw yaw-vs-entry delta.
    assert bearing_complete(right, 0.0, 0.0, 60.0, accumulated_deg=-70.0) is True
    assert bearing_complete(right, 0.0, 0.0, 60.0, accumulated_deg=-30.0) is False
    # direction is still enforced
    assert bearing_complete(right, 0.0, 0.0, 60.0, accumulated_deg=+70.0) is False
    # and a huge raw delta cannot rescue a small windowed one
    assert bearing_complete(right, math.radians(-170), 0.0, 60.0,
                            accumulated_deg=-5.0) is False


def test_brain_gates_the_window_on_the_decision_junction_with_an_exit_grace():
    """A turn completes as the vehicle EXITS, so the window cannot end at the boundary."""
    import pathlib
    src = (pathlib.Path(__file__).parents[1] / "brain" / "brain_controller.py").read_text()
    assert "self._decide_cluster = self.current_cluster" in src, \
        "the junction the branch was decided at is not recorded"
    assert "_bearing_exit_grace_m" in src, "no exit grace -- the turn is under-counted"
    assert src.count("accumulated_deg=self._windowed_bearing()") == 2, \
        "a Bearing(...) call site still judges the unwindowed delta"


# --------------------------------------------------------------------------- #
# JunctionCueLedger: the scoped conjunction with sequential accumulation
# --------------------------------------------------------------------------- #

from brain.trigger_policy import JunctionCueLedger  # noqa: E402


def test_concern_1_a_cone_with_no_junction_never_counts():
    """The first scenario, end to end.

    World: a cone with no intersection, then an intersection with NO cone, then both.
    The object-only question fires on the lone cone and turns at the decoy. The scoped
    question is asked about the NEXT intersection, so a cone lying on the road answers NO
    and nothing is latched -- there is no stale sighting to "drop", which is the point.
    """
    led = JunctionCueLedger(evidence_m=12.0)
    # 0-12 m: a cone beside the road, no intersection ahead carries it -> scoped NO
    for m in (0.0, 4.0, 8.0, 12.0):
        led.note_approach(False, m)
    # 32 m: the DECOY junction. Nothing pending, so it is sealed as "no cue".
    assert led.note_enter(cluster=63, travelled=32.0) is False
    led.note_exit()
    # 44-56 m: approaching the real one, the cue IS at the next intersection
    for m in (44.0, 48.0, 52.0, 56.0):
        led.note_approach(True, m)
    assert led.note_enter(cluster=60, travelled=60.0) is True
    assert led.count_with_cue() == 1
    assert led.nth_with_cue(1) == 60


def test_concern_2_a_cue_seen_on_approach_survives_leaving_the_frame():
    """The second scenario: visible on approach, out of FOV on arrival.

    A verge prop is ~100 deg off-axis at 5 m, so the last frames before
    arrival show nothing. The sighting was taken while it was still visible and is sealed
    on entry, so the step completes instead of timing out.
    """
    led = JunctionCueLedger(evidence_m=12.0)
    led.note_approach(True, 8.0)          # seen at 8 m
    led.note_approach(None, 14.0)         # unanswerable as it leaves the frame
    led.note_approach(False, 18.0)        # and now plainly not visible
    assert led.note_enter(cluster=63, travelled=20.0) is True, \
        "a sighting taken on approach must survive the cue leaving the frame"


def test_a_no_does_not_overwrite_a_live_yes_but_distance_does():
    """The latch is a sighting, not the last word -- yet it must still expire.

    Expiring is what keeps concern (1) fixed once concern (2)'s memory exists: a sighting
    that has travelled further than the evidence window cannot reach the next junction.
    """
    led = JunctionCueLedger(evidence_m=12.0)
    led.note_approach(True, 0.0)
    led.note_approach(False, 5.0)
    assert led.note_enter(cluster=63, travelled=10.0) is True      # 10 m < 12 m, fresh
    led2 = JunctionCueLedger(evidence_m=12.0)
    led2.note_approach(True, 0.0)
    assert led2.note_enter(cluster=63, travelled=20.0) is False, \
        "a sighting 20 m back must not seal onto this junction"


def test_unanswerable_is_not_a_confident_no():
    led = JunctionCueLedger(evidence_m=12.0)
    led.note_approach(None, 0.0)
    assert led._pending is None, "None was latched as a negative"


def test_the_ordinal_is_built_by_MEMORY_because_the_vlm_cannot_see_it():
    """"stop at the 2nd cone" / "turn before the one with the cone".

    The VLM does not reliably attribute a prop to the SECOND intersection -- ordinal and
    relational phrasings fail for the far junction even for a large prop plainly in frame.
    So the ordinal cannot be read off one image; it is accumulated as junctions are passed.
    """
    led = JunctionCueLedger(evidence_m=12.0)
    led.note_approach(False, 0.0);  led.note_enter(10, 5.0);  led.note_exit()   # no cue
    led.note_approach(True, 20.0);  led.note_enter(20, 26.0); led.note_exit()   # 1st cue
    led.note_approach(False, 40.0); led.note_enter(30, 45.0); led.note_exit()   # no cue
    led.note_approach(True, 60.0);  led.note_enter(40, 66.0); led.note_exit()   # 2nd cue
    assert led.count_with_cue() == 2
    assert led.nth_with_cue(2) == 40
    # and the lookahead the VLM could not answer per-frame
    assert led.junction_before_cue() == 10


def test_junction_before_cue_is_only_known_after_the_fact():
    """The limit: sequential memory cannot act BEFORE it has seen the cue.

    A plan that must turn at the junction preceding the cue needs the route traversed
    once, or the ordinal carried by the plan. The method returning None here is that
    limitation made visible rather than silently returning something usable-looking.
    """
    led = JunctionCueLedger(evidence_m=12.0)
    led.note_approach(True, 0.0)
    led.note_enter(63, 5.0)
    assert led.junction_before_cue() is None, \
        "the FIRST junction carries the cue -- there is no junction before it"


def test_brain_wires_the_scoped_approach_end_to_end():
    """All four links, because any one missing makes the mechanism silently inert.

    Asserted on the source (a ROS node is needed to run it), the same shape as the other
    wiring tests here. The links are: ask on approach while NAVIGATING; latch the answer;
    seal it on junction entry; consume it at the cue check.
    """
    import pathlib
    src = (pathlib.Path(__file__).parents[1] / "brain" / "brain_controller.py").read_text()
    assert "q = scoped_cue_question(approach_cue)" in src, \
        "the approach query is never asked"
    # ...and it must fall back to the upcoming BRANCH cue, because the step being driven
    # into a decision point has no transition_cue of its own -- without this the junction
    # is sealed "cue absent" with the prop standing in it.
    assert "self._navigator.upcoming_branch_cue()" in src, \
        "the approach query does not look ahead to the branch's cue"
    assert "self._cue_ledger.note_approach(answer, travelled)" in src, \
        "the approach answer is never latched"
    assert "self._cue_ledger.note_enter(new_cluster, self._travelled)" in src, \
        "the latched answer is never sealed to a junction"
    # BOTH consumers: a cue check AND a branch decision. Wiring only the first leaves a
    # branching plan ignoring the ledger entirely -- the junction seals correctly and the
    # branch is still taken by a fresh query.
    assert src.count("sealed = self._cue_ledger.current_answer") == 2, \
        "the sealed answer must be consumed by BOTH the cue check and the branch decision"
    # spend must be bounded by DISTANCE, not wall time -- a time limit becomes a latency
    # the vehicle outruns, and the branch decision misses the cue.
    assert "moved >= self._approach_every_m" in src, \
        "approach queries are not rate-limited by distance"
    # and the whole thing must default OFF, like require_cluster_change before it
    assert 'declare_parameter("cue_scoped_approach",     False)' in src, \
        "the scoped approach is on by default -- it must stay opt-in"


def test_a_failed_approach_call_latches_nothing_rather_than_a_negative():
    """An exception is not evidence of absence.

    If a failed query latched False, the junction would read 'no cue' on a question that
    was never answered -- the same collapse of unknown into confident-negative that
    `answer_keys` avoids by leaving a family out rather than publishing False.
    """
    led = JunctionCueLedger(evidence_m=12.0)
    led.note_approach(None, 0.0)
    assert led.note_enter(cluster=63, travelled=2.0) is False
    assert led.sealed == [(63, False)]
    # ...but a real YES before it still wins
    led2 = JunctionCueLedger(evidence_m=12.0)
    led2.note_approach(True, 0.0)
    led2.note_approach(None, 4.0)
    assert led2.note_enter(cluster=63, travelled=6.0) is True


def test_a_late_yes_upgrades_the_junction_you_are_standing_in():
    """Approach queries cannot be dense enough to guarantee a reading in the good window.

    Each VLM call is ~1.5 s of WALL time while the simulator runs ~5x faster, so queries
    land ~30 simulated metres apart whatever spacing is requested. The far query can say
    NO, get sealed, and the correct YES arrive just after entry.
    A positive asked from inside the junction is still about THIS junction, so it corrects
    the seal.
    """
    led = JunctionCueLedger(evidence_m=12.0)
    led.note_approach(False, 0.0)
    assert led.note_enter(cluster=63, travelled=10.0) is False
    led.note_approach(True, 12.0)                 # the good reading, just too late
    assert led.current_answer is True
    assert led.sealed == [(63, True)]


def test_a_late_no_does_not_erase_a_sighting():
    """Upward only. A late negative undoing a sighting is concern (2) all over again."""
    led = JunctionCueLedger(evidence_m=12.0)
    led.note_approach(True, 0.0)
    assert led.note_enter(cluster=63, travelled=4.0) is True
    led.note_approach(False, 6.0)                 # cue has left the frame on the way in
    assert led.current_answer is True
    # and a late NO must not start a pending answer for the NEXT junction either
    led.note_exit()
    assert led.note_enter(cluster=60, travelled=40.0) is False
