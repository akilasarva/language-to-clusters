"""Tests for grounding a topological plan step onto a concrete region.

The turn-direction tests are the important ones. Getting the sign backwards swaps every
branch decision — the vehicle takes the wrong exit at a junction — and for the first few
metres that is indistinguishable from correct behaviour, so it has to be pinned by
construction rather than caught by watching a run.
"""

from __future__ import annotations

import os
import sys

import pytest
import yaml

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)

from carla_gt_bridge.routing import (adjacency_from_bearing_map,  # noqa: E402
                                     candidates, maneuver_toward, next_region,
                                     region_for_maneuver, StepTargeter,
                                     turn_magnitude)


def _corridor():
    """The real Town05 mission corridor, loaded from the generated cluster map."""
    p = os.path.join(PKG, "config", "cluster_map.carla_town05.yaml")
    if not os.path.exists(p):
        pytest.skip("cluster_map.carla_town05.yaml not present")
    doc = yaml.safe_load(open(p))
    labels = {}
    for mode, ids in doc["modes"].items():
        for i in ids:
            labels.setdefault(i, mode)
    # junction ids also appear under `path` by subsumption; the finest label wins, which
    # is what `taxonomy.cluster_labels()` does too.
    for i in doc["modes"]["junction"]:
        labels[i] = "junction"
    return doc["bearing_map"], labels


# --------------------------------------------------------------------------- #
# adjacency                                                                   #
# --------------------------------------------------------------------------- #

def test_adjacency_comes_out_of_the_bearing_map_keys():
    adj = adjacency_from_bearing_map({"1-2": 0.0, "2-1": 180.0, "2-3": 90.0})
    assert adj == {1: {2}, 2: {1, 3}, 3: {2}}


def test_malformed_bearing_keys_are_ignored_not_fatal():
    adj = adjacency_from_bearing_map({"1-2": 0.0, "junk": 1.0, "a-b": 2.0})
    assert adj == {1: {2}, 2: {1}}


def test_town05_adjacency_matches_the_corridor_chain():
    """The corridor chain must be present. It is no longer the WHOLE graph.

    `regions.town05.npz` is the full Town05.xodr (74 regions, 21 junctions), so 45
    also borders junction 56, 66 also borders 46, and 53 also borders 7. Pinning
    equality to the corridor would pin a pruned graph, which can hide wrong
    manoeuvres.
    """
    bm, _ = _corridor()
    adj = adjacency_from_bearing_map(bm)
    assert {66} <= adj[45]
    assert {4, 45} <= adj[66]
    assert adj[4] == {53, 66}
    assert {4, 5, 8} <= adj[53]


# --------------------------------------------------------------------------- #
# grounding a step                                                            #
# --------------------------------------------------------------------------- #

def test_the_first_junction_step_targets_the_FIRST_junction():
    """The bug this module exists to prevent.

    `taxonomy.canonical_id('junction')` returns 53 — the SECOND junction — because it is
    the lowest junction id in the corridor. A plan step saying "reach a junction" must
    ground onto 66, the adjacent one, not onto the one after it.
    """
    bm, labels = _corridor()
    adj = adjacency_from_bearing_map(bm)
    # On the 74-region map 45 borders junctions 56 AND 66, so this is a genuine fork
    # and `previous` is what resolves it — which is the whole reason `previous` is a
    # required argument. The assertion that matters: grounding must NOT reach past 66
    # to 53.
    ambiguous, cands = next_region(adj, labels, current=45, goal_label="junction")
    assert ambiguous is None and set(cands) == {56, 66}
    target, cands = next_region(adj, labels, current=45, goal_label="junction",
                                previous=56)
    assert target == 66
    assert labels[53] == "junction"                          # 53 is a junction too
    assert 53 not in cands, "grounded onto the second junction instead of the first"


def test_previous_region_disambiguates_two_adjacent_junctions():
    """From region 4 BOTH neighbours are junctions, so history is required, not optional."""
    bm, labels = _corridor()
    adj = adjacency_from_bearing_map(bm)
    ambiguous, cands = next_region(adj, labels, current=4, goal_label="junction")
    assert ambiguous is None and cands == [53, 66]
    target, _ = next_region(adj, labels, current=4, goal_label="junction", previous=66)
    assert target == 53


def test_a_genuine_fork_returns_every_candidate_and_picks_none():
    """At the decision junction the router must NOT choose. Brain's maneuver decides."""
    bm, labels = _corridor()
    adj = adjacency_from_bearing_map(bm)
    target, cands = next_region(adj, labels, current=53, goal_label="path", previous=4)
    assert target is None
    # 7 is part of the fork on the full map.
    assert set(cands) == {5, 7, 8}


def test_no_candidate_means_the_plan_and_the_map_disagree():
    bm, labels = _corridor()
    adj = adjacency_from_bearing_map(bm)
    target, cands = next_region(adj, labels, current=45, goal_label="passage")
    assert target is None and cands == []


def test_candidates_excludes_where_we_came_from():
    bm, labels = _corridor()
    adj = adjacency_from_bearing_map(bm)
    # 46 is 66's straight exit.
    assert candidates(adj, labels, 66, "path", previous=45) == [4, 46]
    assert candidates(adj, labels, 66, "path", previous=None) == [4, 45, 46]
    assert 45 not in candidates(adj, labels, 66, "path", previous=45)


# --------------------------------------------------------------------------- #
# turn direction — the sign that must not flip                                #
# --------------------------------------------------------------------------- #

def test_town05_decision_junction_offers_straight_and_right():
    """r5 is straight on, r8 is a RIGHT turn.

    Derivable by hand from the centroids: approaching j53 (-125.3, 89.3) from r4
    (-126.2, 114.5) the heading is -88 deg, i.e. travelling south. The exit to r8
    (-158.4, 89.9) is at 179 deg, i.e. west. Travelling south, west is on the right.
    """
    bm, labels = _corridor()
    assert maneuver_toward(bm, current=53, previous=4, target=5) == "straight"
    assert maneuver_toward(bm, current=53, previous=4, target=8) == "right"
    assert turn_magnitude(bm, 4, 53, 8) == pytest.approx(-93.0, abs=3.0)


def test_a_left_turn_is_a_positive_angle_in_the_planar_frame():
    """Pinned against a hand-built map, independent of any town.

    Heading east then north is a LEFT turn, and in the planar frame (+y north, headings
    counter-clockwise) that is a POSITIVE angle change. In CARLA's own left-handed frame
    the sign is opposite; conflating the two swaps every branch.
    """
    bm = {"1-2": 0.0, "2-3": 90.0, "2-4": -90.0, "2-5": 2.0, "2-6": 178.0}
    assert maneuver_toward(bm, current=2, previous=1, target=3) == "left"
    assert maneuver_toward(bm, current=2, previous=1, target=4) == "right"
    assert maneuver_toward(bm, current=2, previous=1, target=5) == "straight"
    assert maneuver_toward(bm, current=2, previous=1, target=6) == "u_turn"


def test_maneuver_selects_the_branch_region():
    bm, labels = _corridor()
    assert region_for_maneuver(bm, 53, 4, [5, 8], "straight") == 5
    assert region_for_maneuver(bm, 53, 4, [5, 8], "right") == 8
    assert region_for_maneuver(bm, 53, 4, [5, 8], "left") is None


def test_maneuver_is_straight_at_the_start_of_a_run():
    """With no previous region there is no turn to measure."""
    bm, _ = _corridor()
    assert maneuver_toward(bm, current=45, previous=None, target=66) == "straight"


def test_missing_bearing_is_reported_not_guessed():
    assert maneuver_toward({"1-2": 0.0}, current=2, previous=1, target=9) == "unknown"
    assert turn_magnitude({"1-2": 0.0}, 1, 2, 9) is None


# --------------------------------------------------------------------------- #
# target_for falls back to next_along — the initial-grounding gap              #
#                                                                              #
# `next_along` (BFS) and `advance_target` handle a goal two hops away; the     #
# INITIAL grounding must too. That is the documented shape for "stop at the    #
# Nth landmark" (generator.md examples), which adjacency-only grounding fails  #
# with "no 'path' adjacent to 7 (prev 14)".                                    #
# --------------------------------------------------------------------------- #

def _line_town():
    """7 -(path)- 14, 7 - 30 (junction) - 6 (path) - 28 (junction) - 24 (path)."""
    adj = {7: {14, 30}, 14: {7}, 30: {7, 6}, 6: {30, 28}, 28: {6, 24}, 24: {28}}
    labels = {7: "path", 14: "path", 30: "junction", 6: "path",
              28: "junction", 24: "path"}
    return adj, labels


def test_target_for_grounds_a_path_to_path_step_two_hops_away():
    t = StepTargeter(*_line_town(), bearing_map={})
    t.observe(14)          # the region behind us, seeded like simulate() does
    t.observe(7)
    target, note = t.target_for("path", step_key=("s", 0))
    assert target == 6, note
    # `via` must carry the junction between, the way advance_target populates it,
    # or the MPC steers straight at a region it cannot reach in one hop.
    assert t.via == [30]
    assert "further along" in note


def test_target_for_still_prefers_an_adjacent_match():
    """The fallback must never pre-empt adjacency — it only runs when nothing fits."""
    t = StepTargeter(*_line_town(), bearing_map={})
    t.observe(14)
    t.observe(7)
    target, note = t.target_for("junction", step_key=("s", 0))
    assert target == 30, note
    assert t.via == []


def test_target_for_still_fails_loudly_when_nothing_is_reachable():
    """A real plan/map disagreement must not be silently routed somewhere else."""
    adj, labels = _line_town()
    t = StepTargeter(adj, labels, bearing_map={})
    t.observe(7)
    target, note = t.target_for("passage", step_key=("s", 0))
    assert target is None
    assert "TOPOLOGY failure" in note
    assert "positions being off would not cause it" in note
