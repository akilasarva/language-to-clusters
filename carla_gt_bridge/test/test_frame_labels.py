"""Tests for the derived per-frame ground-truth labels.

These are built from hand-constructed geometry rather than from a bag, so that the two
subtle pieces of logic — the junction PHASE and the trajectory relations — are exercised
on the geometry where their specific failure modes show up:

  * phasing a junction traversal on the sign of d(distance)/dt flips mid-junction, not at
    the boundary, so `approach`/`exit` come out shifted by the junction's width;
  * "around a building" and "turning at an intersection" are the same motion, and the
    only thing that separates them is that they are on different axes.

Both are pinned below.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)

from carla_gt_bridge.frame_labels import (AROUND_TURN_DEG,  # noqa: E402
                                          ENCLOSURE_LABELS,
                                          STRUCTURE_TAGS, FrameRow, StructureMap,
                                          assign_cluster_labels, body_to_world,
                                          enclosure_relations, junction_phases,
                                          smooth_enclosure,
                                          structure_sides, to_cluster_map_blocks,
                                          trajectory_breaks)

HZ = 10
DT_NS = int(1e9 / HZ)


def _cloud(points_sensor, tag):
    """(N,3) sensor-frame points all carrying one ObjTag."""
    xyz = np.asarray(points_sensor, dtype=float)
    return xyz, np.full(len(xyz), tag, dtype=int)


# --------------------------------------------------------------------------- #
# structure_sides — the one y negation                                        #
# --------------------------------------------------------------------------- #

def test_carla_y_is_negated_exactly_once():
    """A wall at CARLA +y (which is RIGHT) must come out on the RIGHT.

    `lidar_ranges_node` negates y for exactly this reason and `region_lookup` must NOT,
    because odometry is already planar. Getting this backwards mirrors every enclosure
    label without changing any range, so nothing looks wrong until a `building_left`
    ground truth is compared against a model that saw it on the right.
    """
    xyz, tags = _cloud([[0.0, 5.0, -1.0], [2.0, 5.0, -1.0]], 3)   # CARLA +y
    out = structure_sides(xyz, tags)
    assert "building_right" in out
    assert "building_left" not in out
    rng, lat, fwd = out["building_right"]
    assert rng == pytest.approx(5.0)
    assert lat < 0                       # planar: negative lateral is right
    assert fwd == pytest.approx(0.0)     # abeam


def test_z_band_drops_road_and_overhead():
    """Default band is ROAD_LEVEL_BAND: sensor-frame -1.6..-0.5, i.e. 0.4..1.5 m above
    the road for a 2 m mount."""
    below = [[3.0, 2.0, -2.0]]           # road surface, below the band
    above = [[3.0, 2.0, 1.0]]            # 3 m up: gantry / overhanging deck
    inside = [[3.0, 2.0, -1.0]]          # 1 m above the road: a wall
    for pts, expect in ((below, False), (above, False), (inside, True)):
        xyz, tags = _cloud(pts, 3)
        assert ("building_right" in structure_sides(xyz, tags)) is expect


def test_forward_offset_keeps_its_sign():
    """A wall already passed is BEHIND. Recovering forward as sqrt(rng^2 - lat^2)
    drops that sign and places it ahead, which puts the blob lookup on empty road."""
    xyz, tags = _cloud([[-8.0, 3.0, -1.0]], 3)
    rng, lat, fwd = structure_sides(xyz, tags)["building_right"]
    assert fwd == pytest.approx(-8.0)
    assert rng == pytest.approx(math.hypot(8.0, 3.0))


def test_buildings_and_walls_merge_to_one_concept():
    """CARLA splits a city block across tag 3 and tag 4; the nearer wins."""
    xyz = np.array([[0.0, 9.0, -1.0], [0.0, 4.0, -1.0]])
    tags = np.array([3, 4])
    out = structure_sides(xyz, tags)
    assert out["building_right"][0] == pytest.approx(4.0)
    assert STRUCTURE_TAGS[3] == STRUCTURE_TAGS[4] == "building"


# --------------------------------------------------------------------------- #
# StructureMap — the ObjIdx stand-in                                          #
# --------------------------------------------------------------------------- #

def test_same_wall_from_two_poses_is_one_blob():
    """Semantic LiDAR ObjIdx is 0 for ALL static scenery, so instances have to come
    from the world-frame geometry instead. Two views of one wall must agree."""
    m = StructureMap(cell_m=4.0)
    wall = np.array([[10.0, y] for y in np.arange(0.0, 30.0, 0.5)])
    m.add(wall)
    m.add(wall + np.array([0.0, 0.2]))        # same wall, later frame, slight jitter
    assert m.n_blobs == 1
    assert m.blob_at(10.0, 5.0) == m.blob_at(10.0, 25.0)


def test_two_separated_buildings_are_two_blobs():
    m = StructureMap(cell_m=4.0)
    m.add(np.array([[0.0, 0.0], [1.0, 1.0]]))
    m.add(np.array([[80.0, 80.0], [81.0, 81.0]]))
    assert m.n_blobs == 2
    assert m.blob_at(0.5, 0.5) != m.blob_at(80.5, 80.5)


def test_body_to_world_round_trips_through_yaw():
    for yaw in (0.0, math.pi / 2, -math.pi / 3, math.pi):
        wx, wy = body_to_world(3.0, 1.0, 10.0, -5.0, yaw)
        # invert: rotate back and subtract
        c, s = math.cos(-yaw), math.sin(-yaw)
        dx, dy = wx - 10.0, wy + 5.0
        assert c * dx - s * dy == pytest.approx(3.0, abs=1e-9)
        assert s * dx + c * dy == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# Junction phase — the mid-junction sign flip                                 #
# --------------------------------------------------------------------------- #

def _straight_run(n, x0, dx, label_at):
    rows = []
    for i in range(n):
        x = x0 + i * dx
        rows.append(FrameRow(t_ns=i * DT_NS, x=x, y=0.0, yaw=0.0,
                             region_label=label_at(x), rid=0))
    return rows


def test_phase_does_not_flip_inside_the_junction():
    """Drive west->east through a junction spanning x in [-10, 10], centroid at 0.

    The derivative of distance-to-centroid flips sign at x=0, which is the MIDDLE of the
    junction. If the phase came from that derivative, frames at x in [-10, 0) would be
    `approach` and (0, 10] would be `exit`, and the junction itself would never be
    labelled. The region label has to win.
    """
    def lab(x):
        if abs(x) <= 10:
            return "junction"
        return "approach" if abs(x) <= 30 else "path"

    rows = _straight_run(61, -60.0, 2.0, lab)
    junction_phases(rows, {0: (0.0, 0.0)})
    inside = [r.topology for r in rows if abs(r.x) <= 10]
    assert set(inside) == {"junction"}, inside

    before = [r.topology for r in rows if -30 <= r.x < -10]
    after = [r.topology for r in rows if 10 < r.x <= 30]
    assert set(before) == {"approach"}, before
    assert set(after) == {"exit"}, after


def test_approach_region_is_direction_free_so_both_phases_come_from_motion():
    """The SAME region id yields `approach` driving in and `exit` driving out — that is
    the whole reason the segmenter emits one direction-free label."""
    def lab(x):
        return "approach" if abs(x) <= 30 else "path"

    fwd = _straight_run(20, -50.0, 2.0, lab)
    junction_phases(fwd, {0: (0.0, 0.0)})
    rev = _straight_run(20, 50.0, -2.0, lab)
    junction_phases(rev, {0: (0.0, 0.0)})
    assert "approach" in {r.topology for r in fwd}
    assert "approach" in {r.topology for r in rev}
    # and each run also produces its own exit half
    assert "exit" in {r.topology for r in fwd}


def test_off_network_is_not_forced_into_a_region():
    rows = [FrameRow(t_ns=0, x=0, y=0, region_label="path", off_network=True)]
    junction_phases(rows, {})
    assert rows[0].topology == "off_network"


# --------------------------------------------------------------------------- #
# Trajectory relations                                                        #
# --------------------------------------------------------------------------- #

def _turn(n, deg_total, blob_side, near_m=8.0, t0=0):
    """n frames turning `deg_total` in total, with a blob held at `near_m` on one side."""
    rows = []
    step = math.radians(deg_total) / max(n - 1, 1)
    for i in range(n):
        r = FrameRow(t_ns=t0 + i * DT_NS, x=float(i), y=0.0, yaw=i * step, speed=5.0)
        if blob_side == "left":
            r.building_left_m, r.building_blob_left = near_m, 7
            r.building_fwd_left = 1.0          # abeam/ahead, not yet passed
        elif blob_side == "right":
            r.building_right_m, r.building_blob_right = near_m, 7
            r.building_fwd_right = 1.0
        rows.append(r)
    return rows


def test_around_is_the_relation_and_along_edge_is_still_the_axis():
    """"Around a building" refines the AXIS value, it does not replace it.

    `along_edge` is the settled enclosure concept (one side closed) and `around` is what
    the trajectory did within it. Emitting `around_building` as the label instead would
    produce a mode that `ClusterTaxonomy.resolve` has never heard of -- discovered only
    when the cluster map is loaded, long after the collection.
    """
    rows = _turn(20, 90.0, "left")          # CCW turn, building on the inside
    enclosure_relations(rows)
    assert (rows[-1].enclosure, rows[-1].relation) == ("along_edge", "around")

    bare = _turn(20, 90.0, None)            # same turn, nothing near
    enclosure_relations(bare)
    assert bare[-1].enclosure == "open_space"

    outside = _turn(20, 90.0, "right")      # building on the OUTSIDE of the turn
    enclosure_relations(outside)
    assert outside[-1].relation != "around"


def test_along_edge_is_a_straight_run_beside_one():
    rows = _turn(20, 0.0, "right")
    enclosure_relations(rows)
    assert (rows[-1].enclosure, rows[-1].relation) == ("along_edge", "along")


def test_past_is_the_relation_once_it_falls_behind_the_beam():
    rows = []
    for i in range(20):
        r = FrameRow(t_ns=i * DT_NS, x=float(i), y=0.0, yaw=0.0, speed=5.0)
        r.building_right_m = 5.0 + i * 1.0
        r.building_fwd_right = 6.0 - i          # ahead -> abeam -> behind
        r.building_blob_right = 3
        rows.append(r)
    enclosure_relations(rows)
    assert (rows[0].enclosure, rows[0].relation) == ("along_edge", "along")
    assert (rows[-1].enclosure, rows[-1].relation) == ("along_edge", "past")


def test_a_frame_with_no_structure_return_is_open_space():
    """With a look-back-only rule, frames with no building return at all come out
    `past_building`. Those phantom labels get voted into a cluster and sink its purity,
    and the result reads as a perception failure rather than a labelling bug."""
    rows = []
    for i in range(40):
        r = FrameRow(t_ns=i * DT_NS, x=float(i), y=0.0, yaw=0.0, speed=5.0)
        if i < 4:                                # a wall, briefly, then nothing
            r.building_right_m, r.building_fwd_right = 6.0, 2.0
            r.building_blob_right = 3
        rows.append(r)
    enclosure_relations(rows)
    assert rows[-1].enclosure == "open_space"
    assert all(r.enclosure == "open_space" for r in rows
               if r.building_right_m == float("inf"))


def test_an_off_road_frame_still_gets_an_enclosure_label():
    """The payoff of two axes, and the only way `open_space` is reachable at all.

    A pose 30 m off the carriageway has NO topology — no region owns it, so the
    labeller says `off_network` — but its enclosure is perfectly well defined, because
    the LiDAR can see that nothing is close on either side. The `.xodr` never emits
    `open_space` and a lane-following agent never leaves the road, so without off-road
    frames that concept has zero examples in the corpus.
    """
    rows = [FrameRow(t_ns=i * DT_NS, x=float(i), y=0.0, yaw=0.0, speed=5.0,
                     region_label="off_network", off_network=True, rid=-1)
            for i in range(20)]
    junction_phases(rows, {})
    enclosure_relations(rows)
    assert all(r.topology == "off_network" for r in rows)
    assert all(r.enclosure == "open_space" for r in rows)


def test_one_sided_bridge_at_15_m_is_not_a_passage():
    """Driving PAST a bridge gives returns on ONE side, often near the 14 m threshold;
    a one-sided min-range test would give knife-edge false positives."""
    rows = _turn(12, 0.0, None)
    for r in rows:
        r.bridge_right_m = r.bridge_m = 13.9     # under the threshold, one side only
    enclosure_relations(rows)
    assert "passage" not in {r.enclosure for r in rows}


def test_both_sides_close_is_passage_whatever_it_is_made_of():
    """A deck's parapets and a walled alley are the same concept: the axis is about how
    many sides are closed, not about what closes them."""
    deck = _turn(12, 0.0, "right", near_m=3.0)
    for r in deck:
        r.bridge_left_m = r.bridge_right_m = r.bridge_m = 4.0
    enclosure_relations(deck)
    assert {r.enclosure for r in deck} == {"passage"}

    alley = _turn(12, 0.0, None)
    for r in alley:
        r.building_left_m = r.building_right_m = 4.0
        r.building_blob_left, r.building_blob_right = 1, 2
        r.building_fwd_left = r.building_fwd_right = 1.0
    enclosure_relations(alley)
    assert {r.enclosure for r in alley} == {"passage"}


def test_the_two_axes_are_independent():
    """Turning at an intersection is also turning around the corner building. It is not an ambiguity — the frame is `junction` on the topology
    axis and `along_edge`/`around` on the enclosure axis, and both are true at once."""
    rows = _turn(20, 90.0, "left")
    for r in rows:
        r.region_label = "junction"
        r.rid = 4
    junction_phases(rows, {4: (0.0, 0.0)})
    enclosure_relations(rows)
    assert rows[-1].topology == "junction"
    assert (rows[-1].enclosure, rows[-1].relation) == ("along_edge", "around")


def test_borderline_turn_puts_the_runner_up_in_degraded():
    rows = _turn(20, AROUND_TURN_DEG - 5, "left")
    enclosure_relations(rows)
    assert (rows[-1].enclosure, rows[-1].relation) == ("along_edge", "along")
    assert rows[-1].enclosure_degraded == "passage"


# --------------------------------------------------------------------------- #
# Cluster assignment — the purity floor that replaces the operator's "skip"   #
# --------------------------------------------------------------------------- #

def test_clean_cluster_takes_its_majority_label():
    a = assign_cluster_labels([0] * 10, ["path"] * 10)
    assert a[0].label == "path" and a[0].purity == 1.0 and not a[0].degraded


def test_mixed_cluster_is_unlabeled_not_force_mapped():
    labels = ["path"] * 5 + ["junction"] * 5
    a = assign_cluster_labels([0] * 10, labels)
    assert a[0].label == "unlabeled"
    assert a[0].distribution == {"path": 5, "junction": 5}


def test_noise_cluster_is_always_unlabeled():
    a = assign_cluster_labels([-1] * 50, ["path"] * 50)
    assert a[-1].label == "unlabeled"


def test_small_cluster_is_unlabeled():
    a = assign_cluster_labels([0] * 3, ["path"] * 3, min_size=8)
    assert a[0].label == "unlabeled"


def test_runner_up_above_the_floor_becomes_degraded():
    labels = ["along_edge"] * 8 + ["junction"] * 2
    a = assign_cluster_labels([0] * 10, labels)
    assert a[0].label == "along_edge"
    assert a[0].degraded == ["junction"]


def test_cluster_map_blocks_are_what_the_taxonomy_reads():
    assn = assign_cluster_labels(
        [0] * 10 + [1] * 10 + [-1] * 10,
        ["path"] * 10 + ["along_edge"] * 8 + ["junction"] * 2 + ["path"] * 10)
    blocks = to_cluster_map_blocks(assn)
    assert blocks["modes"] == {"along_edge": [1], "path": [0]}
    # cluster 1 also partly looked like a junction, so a `junction` step accepts it
    # on a landmark-triggered step via accept_degraded.
    assert blocks["mode_meta"]["junction"]["accept_degraded"] == [1]
    assert -1 not in sum(blocks["modes"].values(), [])


# --------------------------------------------------------------------------- #
# Against the REAL shipped Town01 table, not synthetic geometry                #
# --------------------------------------------------------------------------- #

def _real_traverse(town="town01", span=60.0, step=1.0):
    """Straight traverse along the dominant road axis through a real junction."""
    from carla_gt_bridge.region_lookup import load_region_table
    p = os.path.join(PKG, "config", f"regions.{town}.approach.npz")
    if not os.path.exists(p):
        pytest.skip(f"{p} not generated")
    table = load_region_table(p)
    labels = {r: table.label_of(r) for r in table.region_ids}
    juncs = [r for r, lab in labels.items() if lab == "junction"]
    jrid = juncs[len(juncs) // 2]
    jx, jy = table.centroid_of(jrid)
    wp = table.waypoints
    near = wp[np.hypot(wp[:, 0] - jx, wp[:, 1] - jy) < 70.0]
    c = near[:, :2] - np.array([jx, jy])
    u = np.linalg.svd(c - c.mean(0), full_matrices=False)[2][0]
    rows = []
    for i, d0 in enumerate(np.arange(-span, span + step, step)):
        px, py = jx + u[0] * d0, jy + u[1] * d0
        prid, dist = table.nearest(px, py)
        rows.append(FrameRow(t_ns=int(i * 1e8), x=float(px), y=float(py),
                             yaw=math.atan2(u[1], u[0]), speed=5.0, rid=int(prid),
                             region_label=labels[int(prid)], region_dist_m=dist))
    junction_phases(rows, {r: table.centroid_of(r) for r in juncs})
    return rows, table, jrid, (jx, jy)


def test_real_junction_is_fully_covered_by_the_junction_phase():
    rows, _table, jrid, _c = _real_traverse()
    inside = [r for r in rows if r.rid == jrid]
    assert inside, "traverse never entered the junction"
    assert all(r.topology == "junction" for r in inside)


def test_on_the_real_map_closest_approach_falls_INSIDE_the_junction():
    """The measurement the phase logic exists for, taken on the shipped Town01 table.

    Distance to the junction centroid bottoms out well inside the junction region, so a
    phase derived from the sign of d(distance)/dt alone would call the first half of the
    intersection `approach` and the second half `exit`, and would never emit `junction`
    at all. If this ever fails, the junction regions have changed shape and the phase
    rule should be re-derived rather than assumed.
    """
    rows, _table, jrid, (jx, jy) = _real_traverse()
    d = [math.hypot(r.x - jx, r.y - jy) for r in rows]
    flip = int(np.argmin(d))
    assert rows[flip].rid == jrid


def test_real_traverse_produces_the_expected_phase_order():
    """Assert the ORDERING, not the prefix: which junction `_real_traverse` picks
    depends on the region table, so pinning seq[0] would break on a regenerated
    `.npz` for a reason that has nothing to do with the phase rule."""
    rows, _t, _j, _c = _real_traverse()
    import itertools
    seq = [k for k, _ in itertools.groupby(r.topology for r in rows)]
    want = ["approach", "junction", "exit"]
    assert any(seq[i:i + 3] == want for i in range(len(seq) - 2)), seq


def test_every_axis_value_is_a_real_taxonomy_concept():
    """The labels must be the SETTLED spellings or the cluster map grounds nothing.

    Spellings such as `open`, `along_building` or `on_bridge` are not in
    `mode_concepts.CONCEPTS`, so every mode in the generated cluster map would raise
    TaxonomyError at resolve time — after collection, training and scoring.
    """
    import importlib.util
    mc = os.path.join(os.path.dirname(PKG), "nl_planner", "nl_planner",
                      "mode_concepts.py")
    if not os.path.exists(mc):
        pytest.skip("nl_planner not present")
    spec = importlib.util.spec_from_file_location("mode_concepts", mc)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert set(ENCLOSURE_LABELS) <= set(m.CONCEPTS), (
        set(ENCLOSURE_LABELS) - set(m.CONCEPTS))
    assert {"path", "junction"} <= set(m.CONCEPTS)


def test_a_teleport_breaks_the_relation_window():
    """`drive_collection.py` teleports past an obstruction it cannot reverse out of.

    A window spanning that jump would report `past` for a building the vehicle never
    drove past — the blob was near before the teleport and is gone after, which is
    exactly the receding-structure signature. Detected geometrically, because the
    driver's log is in wall time while frames carry bag time.
    """
    rows = []
    for i in range(30):
        r = FrameRow(t_ns=i * DT_NS, x=float(i), y=0.0, yaw=0.0, speed=5.0)
        if i < 5:                                  # a wall, then a teleport away
            r.building_right_m, r.building_fwd_right = 6.0, 2.0
            r.building_blob_right = 3
        if i >= 10:
            r.x = 5000.0 + i                       # jumped far away
        rows.append(r)
    breaks = trajectory_breaks(rows)
    assert 10 in breaks, breaks
    enclosure_relations(rows, breaks=breaks)
    assert all(r.enclosure == "open_space" for r in rows[10:])


def test_no_break_on_ordinary_driving():
    rows = [FrameRow(t_ns=i * DT_NS, x=0.8 * i, y=0.0, yaw=0.0, speed=8.0)
            for i in range(30)]
    assert trajectory_breaks(rows) == set()


def test_bursty_bag_timestamps_do_not_look_like_teleports():
    """Bag timestamps are RECEIVE times, so consecutive frames can arrive ~1 ms apart.
    A speed-based test (d/dt > 40 m/s) would turn an ordinary 0.4 m step into 400 m/s and
    report spurious teleports, which resets the relation window constantly and quietly
    destroys every `past`/`around` label."""
    rows = []
    for i in range(30):
        t = i * DT_NS if i % 3 else i * DT_NS + 1_000_000   # jitter to ~1 ms gaps
        rows.append(FrameRow(t_ns=t, x=0.4 * i, y=0.0, yaw=0.0, speed=8.0))
    assert trajectory_breaks(rows) == set()


def test_mode_filter_removes_single_frame_enclosure_flicker():
    """Raw enclosure labels flicker frame to frame (sparse returns drop out). A cluster
    voting on that inherits the noise as impurity and reads as a perception failure."""
    rows = []
    for i in range(40):
        r = FrameRow(t_ns=i * DT_NS, x=0.4 * i, y=0.0, yaw=0.0, speed=8.0)
        # a wall that drops out every third frame, as a real sparse return does
        if i % 3:
            r.building_right_m, r.building_fwd_right = 6.0, 2.0
            r.building_blob_right = 3
        rows.append(r)
    enclosure_relations(rows)
    raw = sum(1 for i in range(1, len(rows))
              if rows[i].enclosure != rows[i - 1].enclosure)
    smooth_enclosure(rows, window_m=8.0)
    sm = sum(1 for i in range(1, len(rows))
             if rows[i].enclosure != rows[i - 1].enclosure)
    assert raw > 10, raw
    assert sm == 0, [r.enclosure for r in rows]
    assert all(r.enclosure == "along_edge" for r in rows[5:-5])


def test_mode_filter_does_not_smooth_across_a_teleport():
    rows = []
    for i in range(30):
        r = FrameRow(t_ns=i * DT_NS, x=0.4 * i, y=0.0, yaw=0.0, speed=8.0)
        if i < 15:
            r.building_right_m, r.building_fwd_right = 6.0, 2.0
            r.building_blob_right = 3
        else:
            r.x = 9000.0 + i
        rows.append(r)
    breaks = trajectory_breaks(rows)
    enclosure_relations(rows, breaks=breaks)
    smooth_enclosure(rows, window_m=8.0, breaks=breaks)
    assert rows[14].enclosure == "along_edge"
    assert rows[15].enclosure == "open_space"
