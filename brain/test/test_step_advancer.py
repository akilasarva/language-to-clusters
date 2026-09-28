"""Tests for the one place a plan step's completion is decided.

Each test here corresponds to a divergence between brain_controller's copy of this
logic and the offline harness's copy. Consolidating them
is only worth anything if the consolidated version keeps every rule both copies had
between them — so those rules are pinned individually.
"""

from __future__ import annotations

import math
import os
import sys

import pytest

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)

from brain.step_advancer import StepAdvancer          # noqa: E402


def _step(**kw):
    base = {"start_cluster": 1, "goal_cluster": 2, "accept_clusters": [2],
            "accept_clusters_degraded": [2], "trigger": "traverse",
            "transition_cue": None, "cue_ordinal": None, "branches": None,
            "perception_backed": True}
    base.update(kw)
    return base


def test_traverse_needs_the_cluster_to_hold():
    a = StepAdvancer(dwell_frames=3)
    a.begin_step(_step(), cluster=1)
    assert not a.observe(cluster=2).advanced      # dwell 1
    assert not a.observe(cluster=2).advanced      # dwell 2
    assert a.observe(cluster=2).advanced          # dwell 3


def test_a_flicker_resets_the_dwell():
    a = StepAdvancer(dwell_frames=3)
    a.begin_step(_step(), cluster=1)
    a.observe(cluster=2)
    a.observe(cluster=9)                          # not in the accept set
    assert not a.observe(cluster=2).advanced
    assert not a.observe(cluster=2).advanced


def test_topology_needs_the_heading_too():
    """Divergence 3: the harness hand-rolled this and brain used bearing_complete()."""
    a = StepAdvancer(dwell_frames=1, bearing_complete_deg=60.0)
    a.begin_step(_step(trigger="topology", transition_cue="Bearing(Right)"),
                 cluster=1, yaw=0.0)
    d = a.observe(cluster=2, yaw=0.0)
    assert not d.advanced and d.awaiting_bearing
    assert a.observe(cluster=2, yaw=math.radians(-70)).advanced


def test_topology_without_odometry_waits_rather_than_advancing():
    """Bags with no odom must fall back, not silently succeed."""
    a = StepAdvancer(dwell_frames=1)
    a.begin_step(_step(trigger="topology", transition_cue="Bearing(Left)"), cluster=1)
    d = a.observe(cluster=2)
    assert not d.advanced and d.awaiting_bearing


def test_landmark_waits_for_a_cue_rather_than_assuming_one():
    """Divergence 2: the harness treated "awaiting cue" as done."""
    a = StepAdvancer(dwell_frames=1)
    a.begin_step(_step(trigger="landmark", transition_cue="Detect(Bench)"), cluster=1)
    d = a.observe(cluster=2)
    assert not d.advanced and d.awaiting_cue
    assert a.observe(cluster=2, cue_answer=True).advanced


def test_an_ordinal_needs_distinct_sightings():
    a = StepAdvancer(dwell_frames=1, cue_lost_polls=2)
    a.begin_step(_step(trigger="landmark", transition_cue="Detect(Bench)",
                       cue_ordinal=2), cluster=1)
    assert not a.observe(cluster=2, cue_answer=True).advanced     # bench 1
    assert not a.observe(cluster=2, cue_answer=True).advanced     # still bench 1
    a.observe(cluster=2, cue_answer=False)
    a.observe(cluster=2, cue_answer=False)                        # now lost
    assert a.observe(cluster=2, cue_answer=True).advanced         # bench 2


def test_the_cluster_change_guard_applies_to_every_trigger():
    """Divergence 1: brain's odometry path bypassed goal_reached, so it bypassed this.

    Upward subsumption puts a junction's id inside `path`, so `junction -> path` is
    satisfied standing still unless the cluster is required to change.
    """
    for trig, extra in (("traverse", {}),
                        ("topology", {"transition_cue": "Bearing(Right)"}),
                        ("landmark", {"transition_cue": "Detect(X)"})):
        a = StepAdvancer(dwell_frames=1, require_cluster_change=True)
        a.begin_step(_step(trigger=trig, accept_clusters=[2, 7],
                           accept_clusters_degraded=[2, 7], **extra),
                     cluster=7, yaw=0.0)
        d = a.observe(cluster=7, yaw=math.radians(-90), cue_answer=True)
        assert not d.advanced, f"{trig} advanced without leaving the start cluster"
        assert "start cluster" in d.reason


def test_a_finished_step_reports_its_branches():
    br = [{"vlm_cue": "a cone", "sub_plan": []}, {"vlm_cue": "default", "sub_plan": []}]
    a = StepAdvancer(dwell_frames=1)
    a.begin_step(_step(branches=br), cluster=1)
    d = a.observe(cluster=2)
    assert d.advanced and len(d.branches) == 2


def test_beginning_a_step_clears_the_previous_one():
    a = StepAdvancer(dwell_frames=3)
    a.begin_step(_step(), cluster=1)
    a.observe(cluster=2)
    a.observe(cluster=2)
    a.begin_step(_step(), cluster=1)          # branch descend / plan swap
    assert not a.observe(cluster=2).advanced, "dwell carried over from the last step"


def test_no_cluster_yet_is_not_a_failure():
    a = StepAdvancer()
    a.begin_step(_step())
    d = a.observe(yaw=0.0)
    assert not d.advanced and "no cluster" in d.reason


def test_a_step_that_begins_at_its_own_destination_can_still_complete():
    """The other half of the cluster-change guard, and it is a race, not a constant.

    `require_cluster_change` must not block a step that BEGAN standing in a cluster whose
    own label is the step's goal mode -- there is nothing such a step could do to become
    unblocked. Example: if step 0 (a straight-through manoeuvre) is still completing as
    the vehicle enters a junction, step 1, "approach the next junction", begins already
    at the junction it is asking for. Without this exception it never completes and the
    vehicle stalls in that junction; whether it happens depends on a few ticks of timing.

    Keyed on the cluster's OWN label rather than accept-set membership, because upward
    subsumption puts a junction's id inside the `path` accept set -- see
    test_the_cluster_change_guard_applies_to_every_trigger, which must keep passing.
    """
    a = StepAdvancer(dwell_frames=1, require_cluster_change=True)
    step = _step(trigger="traverse", goal_mode="junction",
                 accept_clusters=[7, 2], accept_clusters_degraded=[7, 2])
    a.begin_step(step, cluster=7, yaw=0.0, start_label="junction")
    assert a.observe(cluster=7, yaw=0.0).advanced, \
        "a step that began at its own destination must be allowed to complete"

    # ...and WITHOUT the label the guard is unchanged, so no existing caller regresses.
    b = StepAdvancer(dwell_frames=1, require_cluster_change=True)
    b.begin_step(step, cluster=7, yaw=0.0)
    assert not b.observe(cluster=7, yaw=0.0).advanced


def test_a_decision_step_may_complete_without_leaving_its_cluster():
    """A branching step keeps goal_mode == start_mode BY DESIGN -- the robot stays put
    while the branch is chosen (schemas.py). Requiring a cluster change asks it to leave
    the junction it is deciding at, which nothing in the plan tells it to do, and
    deadlocks whenever the step fires after the vehicle has entered the junction rather
    than before."""
    a = StepAdvancer(dwell_frames=1, require_cluster_change=True)
    step = _step(trigger="traverse", goal_mode="junction",
                 accept_clusters=[7], accept_clusters_degraded=[7],
                 branches=[{"vlm_cue": "a cone is present", "sub_plan": []},
                           {"vlm_cue": "default", "sub_plan": []}])
    a.begin_step(step, cluster=7, yaw=0.0)
    assert a.observe(cluster=7, yaw=0.0).advanced, \
        "a decision step must not be required to leave the cluster it decides at"


def test_the_offline_twin_carries_brains_bearing_scope():
    """A twin that scores the same situation differently is one rule with two implementations.

    brain bounds Bearing(...) by distance, because the unbounded check can certify a
    right turn built from ~150 m of road curvature while the junction itself saw only a
    few degrees. StepAdvancer is not wired into brain yet; carrying the bound here keeps
    the two consistent when it is. Default stays unbounded so this class's existing
    tests and every recorded trace score exactly as before.
    """
    turned = math.radians(-65.0)

    def run(scope, travelled):
        a = StepAdvancer(dwell_frames=1, bearing_complete_deg=60.0,
                         bearing_scope_m=scope)
        a.begin_step(_step(trigger="topology", transition_cue="Bearing(Right)"),
                     cluster=1, yaw=0.0)
        a.progress.entry_travelled = 0.0
        return a.observe(cluster=2, yaw=turned, travelled=travelled)

    assert run(None, 150.0).advanced, "unbounded must keep the old behaviour"
    assert run(45.0, 20.0).advanced, "a turn inside the scope still completes"
    d = run(45.0, 150.0)
    assert not d.advanced and d.awaiting_bearing, \
        "150 m of curvature is not a junction turn"
