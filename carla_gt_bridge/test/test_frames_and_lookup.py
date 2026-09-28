"""Tests for the frame conversion and the ground-truth region lookup.

The frame tests exist because geometry in this stack is stored in both conventions
(raw CARLA y + degrees in some cluster maps, negated y + radians in ``plans/bridge.json``).
Crossing them produces a mirrored target or a 57x heading error and announces itself only
as "the vehicle drove somewhere wrong", so it is pinned here by construction rather than
by a comment.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)

from carla_gt_bridge.frames import (bearing_deg_to_planar_rad,  # noqa: E402
                                   carla_to_planar, planar_to_carla, wrap_180,
                                   wrap_pi)
from carla_gt_bridge.region_lookup import (RegionTable,  # noqa: E402
                                           load_region_table)


# --------------------------------------------------------------------------- #
# frames                                                                      #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("pose", [
    (0.0, 0.0, 0.0),
    (335.49, 273.74, 90.0),
    (-126.15, 114.50, -179.9),
    (12.5, -7.25, 45.0),
    (1e4, -1e4, 180.0),
])
def test_pose_round_trips_exactly(pose):
    """world -> planar -> world must be the identity, to floating point."""
    px, py, pyaw = carla_to_planar(*pose)
    back = planar_to_carla(px, py, pyaw)
    assert back[0] == pytest.approx(pose[0], abs=1e-9)
    assert back[1] == pytest.approx(pose[1], abs=1e-9)
    assert back[2] == pytest.approx(wrap_180(pose[2]), abs=1e-9)


def test_opendrive_geometry_is_already_planar_not_carla():
    """Settles which frame the derived geometry is in.

    Town01's `.xodr` spans y in [-328.6, 0.0]. A running CARLA server reports
    (335.49, 273.74) for spawn point 0 and (334.7, 318.7) for junction 143 in the SAME
    town — positive y, and of the same magnitude. So CARLA's y is the negation of
    OpenDRIVE's, and every centroid / bearing / waypoint derived from the .xodr is already
    in the planar right-handed frame.

    This matters because it is the difference between "no conversion needed" and "the map
    is mirrored about the x axis". Applying `carla_xy_to_planar` to already-planar data
    negates y twice and the vehicle drives confidently at a reflected target.
    """
    p = os.path.join(PKG, "config", "Town01.xodr")
    if not os.path.exists(p):
        pytest.skip("Town01.xodr not present")
    from carla_gt_bridge.opendrive import load
    from carla_gt_bridge.segmenter import segment
    wp = segment(load(p)).waypoints()
    assert wp[:, 1].max() <= 1.0, "OpenDRIVE y for Town01 should be <= 0"
    assert wp[:, 1].min() < -300.0
    # The recorded live-server y values, mirrored, must land inside that range. 5 m of
    # slack because the table samples reference LINES: a spawn point sits in a lane, up to a
    # couple of metres beyond the outermost reference line (330.54 is 1.9 m outside).
    lo, hi = wp[:, 1].min() - 5.0, wp[:, 1].max() + 5.0
    for carla_y in (273.74, 330.54, 318.7):
        assert lo <= -carla_y <= hi, f"CARLA y={carla_y} does not mirror into the map"
        assert not (lo <= carla_y <= hi), (
            f"CARLA y={carla_y} would ALSO fit unmirrored — the test cannot "
            f"discriminate, so the frame claim is unproven")
    # ...and the conversion is exactly that mirror
    assert carla_to_planar(335.49, 273.74, 90.0)[1] == pytest.approx(-273.74)


def test_conversion_is_an_isometry():
    """Distances are preserved, so a single angle can be converted instead of every point."""
    a, b = (10.0, 20.0), (13.0, 24.0)
    d_carla = math.dist(a, b)
    pa = carla_to_planar(*a, 0.0)[:2]
    pb = carla_to_planar(*b, 0.0)[:2]
    assert math.dist(pa, pb) == pytest.approx(d_carla)


def test_yaw_sense_actually_flips():
    """A CARLA right turn must be a planar right turn — the sign, not just the value.

    In CARLA, yaw increases clockwise; in the planar frame it increases
    counter-clockwise. Turning right in CARLA (+yaw) must therefore DECREASE planar
    yaw. If this ever passes with the same sign, the unicycle model steers the wrong
    way and the vehicle turns left at every right turn.
    """
    _, _, y0 = carla_to_planar(0.0, 0.0, 0.0)
    _, _, y1 = carla_to_planar(0.0, 0.0, 30.0)
    assert y0 == pytest.approx(0.0)
    assert y1 < 0.0
    assert y1 == pytest.approx(math.radians(-30.0))


def test_carla_north_is_negative_y():
    """CARLA's +y points SOUTH. Driving north decreases carla_y, increases planar y."""
    assert carla_to_planar(0.0, -50.0, 0.0)[1] == pytest.approx(50.0)


def test_bearing_map_degrees_convert_to_planar_radians():
    """A bearing_map entry is a CARLA-frame heading in DEGREES."""
    assert bearing_deg_to_planar_rad(0.0) == pytest.approx(0.0)
    assert bearing_deg_to_planar_rad(90.0) == pytest.approx(-math.pi / 2)
    # never returns something 57x too large, i.e. never treats degrees as radians
    assert abs(bearing_deg_to_planar_rad(179.0)) <= math.pi


def test_wrappers_agree_with_map_regions_on_which_end_is_closed():
    """Due west is -180, never +180 — the same choice `map_regions._wrap180` makes.

    Physically it does not matter which end is closed; two functions in the same repo
    disagreeing about it does, because a bearing comparison then flips sign at west.
    """
    assert wrap_180(180.0) == pytest.approx(-180.0)
    assert wrap_180(-180.0) == pytest.approx(-180.0)
    assert wrap_180(270.0) == pytest.approx(-90.0)
    assert wrap_pi(math.pi) == pytest.approx(-math.pi)
    assert wrap_pi(3 * math.pi) == pytest.approx(-math.pi)


# --------------------------------------------------------------------------- #
# region lookup                                                               #
# --------------------------------------------------------------------------- #

def _table():
    """Two 20 m regions side by side, 40 m apart, sampled every 2 m."""
    rows = []
    for x in np.arange(0.0, 20.1, 2.0):
        rows.append((x, 0.0, 7.0))
        rows.append((x, 40.0, 9.0))
    wp = np.array(rows, dtype=float)
    return RegionTable(
        waypoints=wp,
        rids=np.array([7, 9]),
        labels=np.array(["path", "junction"]),
        centroids=np.array([[10.0, 0.0], [10.0, 40.0]]),
        town="synth")


def test_nearest_waypoint_is_exact_where_nearest_centroid_would_not_be():
    """The reason this node uses waypoints and the MPC uses centroids.

    At (0, 0) — the far end of region 7 — the nearest waypoint is region 7's own, but
    it sits 40 m from region 9's centroid and 10 m from its OWN centroid, so a
    centroid test is only right by luck. Extend the geometry and it stops being right.
    """
    t = _table()
    assert t.region_at(0.0, 0.0) == 7
    assert t.region_at(20.0, 40.0) == 9
    assert t.region_at(10.0, 39.0) == 9


def test_off_network_returns_none_rather_than_the_least_wrong_region():
    t = _table()
    assert t.region_at(10.0, 500.0) is None
    # ...but a pose in an outer lane, a few metres off the reference line, is fine
    assert t.region_at(10.0, 4.0) == 7


def test_max_distance_is_respected():
    t = _table()
    assert t.region_at(10.0, 11.0, max_distance_m=12.0) == 7
    assert t.region_at(10.0, 11.0, max_distance_m=10.0) is None


def test_brute_force_agrees_with_the_kdtree():
    """The scipy-free fallback must not be a different answer, only a slower one."""
    t = _table()
    tree = t._tree
    assert tree is not None, "scipy present, so the kd-tree path is the one under test"
    t._tree = None
    try:
        brute = [t.nearest(x, y) for x, y in ((0.0, 0.0), (7.0, 3.0), (13.0, 38.0))]
    finally:
        t._tree = tree
    kd = [t.nearest(x, y) for x, y in ((0.0, 0.0), (7.0, 3.0), (13.0, 38.0))]
    assert brute == kd


def test_unknown_region_id_raises():
    t = _table()
    with pytest.raises(KeyError):
        t.label_of(99)
    with pytest.raises(KeyError):
        t.centroid_of(8)          # between 7 and 9 — must not silently resolve


# --------------------------------------------------------------------------- #
# Town05 corridor — the table the missions actually load                      #
# --------------------------------------------------------------------------- #

def _town05_table():
    p = os.path.join(PKG, "config", "regions.town05.npz")
    if not os.path.exists(p):
        pytest.skip("regions.town05.npz not present — run scripts/map_regions.py")
    return load_region_table(p)


#: The six regions the four Phase A missions actually drive. `regions.town05.npz` is the
#: full Town05.xodr (74 regions, 21 junctions); a table pruned to the corridor would hide
#: exits (e.g. junction 66's straight exit) and so hide wrong-turn failures.
CORRIDOR = [4, 5, 8, 45, 53, 66]


def test_town05_table_is_the_whole_town_and_contains_the_corridor():
    t = _town05_table()
    assert len(t.region_ids) == 74
    assert sum(1 for r in t.region_ids if t.label_of(r) == "junction") == 21
    assert set(CORRIDOR) <= set(t.region_ids)
    assert t.label_of(53) == "junction"
    assert t.label_of(66) == "junction"
    assert all(t.label_of(r) == "path" for r in (4, 5, 8, 45))


def test_every_corridor_centroid_recovers_its_own_region():
    """Nearest-WAYPOINT must be exact even where nearest-centroid is not.

    Some of the corridor's waypoints resolve to a neighbour under the MPC's fast
    nearest-centroid test. This node must not inherit that error — mission advancement is
    what it drives.

    Scoped to the CORRIDOR. That is not a weakening: a centroid is the mean of a region's
    waypoints, so an L-shaped or curved region can have its centroid outside itself, and
    some of the full town's regions do. It is a fact about region shape, not a lookup
    defect. What must hold is that the regions the missions
    drive recover exactly, and the spawn/cone poses are derived from centroids — see
    test_full_town_centroid_recovery_rate for the town-wide number.
    """
    t = _town05_table()
    for rid in CORRIDOR:
        cx, cy = t.centroid_of(rid)
        assert t.region_at(cx, cy) == rid, f"region {rid} did not recover its centroid"


def test_full_town_centroid_recovery_rate():
    """Pin the town-wide rate so a REGRESSION is visible even though 100% is not the bar.

    Deriving a pose from a centroid is only safe for a region that
    recovers, which is why `spawn_config` and the cone placement assert it rather than
    trusting it.
    """
    t = _town05_table()
    ok = [r for r in t.region_ids if t.region_at(*t.centroid_of(r)) == r]
    assert len(ok) / len(t.region_ids) > 0.85
    assert set(CORRIDOR) <= set(ok)


def test_town05_waypoints_only_ever_disagree_on_an_exact_boundary_tie():
    """No genuine misassignments — only shared boundary points.

    The few corridor waypoints that resolve to a neighbour are all exact coincidences
    (distance 0.0 to both): a junction's connecting road starts at
    precisely the point the approaching road ends, so both regions sample it. Which id
    wins there is arbitrary and harmless — it is one frame of flicker at region entry,
    which brain's dwell counting already absorbs.

    The assertion that matters is that there is no mismatch where the OTHER region is
    genuinely closer. That would be a real misassignment and would move a plan-advance
    to the wrong place.
    """
    t = _town05_table()
    wp = t.waypoints
    true = wp[:, 2].astype(int)
    for (x, y, r) in wp:
        got, d_got = t.nearest(x, y)
        if got == int(r):
            continue
        own = wp[true == int(r)][:, :2]
        d_own = float(np.linalg.norm(own - np.array([x, y]), axis=1).min())
        assert d_own == pytest.approx(d_got, abs=1e-9), (
            f"({x:.2f}, {y:.2f}) belongs to {int(r)} at {d_own:.3f} m but resolved to "
            f"{got} at {d_got:.3f} m — a genuine misassignment, not a boundary tie")


def test_pose_off_the_road_network_is_rejected_not_mapped():
    """Publishing the nearest id for a vehicle that has left the road would look
    exactly like normal progress.

    (0, 0) is inside region 0 of the real town, so the assertion uses poses genuinely
    off the network; the town spans roughly x -271..202, y -200..199.
    """
    t = _town05_table()
    assert t.region_at(1000.0, 1000.0) is None
    assert t.region_at(-2000.0, 500.0) is None


# --------------------------------------------------------------------------- #
# CARLA spawn / cone poses — the only pre-flight available offline             #
#                                                                              #
# A sign error here costs a CARLA run and presents as "the MPC never moved"    #
# or, worse, as a cone answered correctly about the wrong place.               #
# The conversion is exercised in the OUTBOUND direction only here — everywhere #
# else in the stack it runs inbound — which is the direction `frames.py` was   #
# written to protect.                                                          #
# --------------------------------------------------------------------------- #

def _scripts_on_path():
    sys.path.insert(0, os.path.join(PKG, "scripts"))


def test_town05_spawn_faces_the_corridor_not_the_lowest_neighbour():
    """The spawn heading must come from a NAMED target, not from neighbour ordering.

    Deriving it from `sorted(adj[45])[0]` works only while region 45 has one neighbour.
    In the full Town05.xodr it has two (56 and 66), so the ordering-based expression
    would spawn the ego facing AWAY from the corridor -- observable only in CARLA.
    """
    _scripts_on_path()
    import missions
    from carla_gt_bridge.routing import adjacency_from_bearing_map
    import yaml

    doc = yaml.safe_load(open(os.path.join(PKG, "config",
                                           "cluster_map.carla_town05.yaml")))
    adj = adjacency_from_bearing_map(doc["bearing_map"])
    nbrs = sorted(adj[missions.START_REGION])
    assert missions.START_TOWARD in nbrs
    assert len(nbrs) > 1, "a start with one neighbour cannot exhibit the bug"

    # TEST THE MECHANISM, NOT A COINCIDENCE. Asserting that sorted(nbrs)[0] differs from
    # START_TOWARD depends on id ordering and could only catch the bug by luck.
    #
    # Instead: aim the constant at each neighbour in turn and require the derived spawn
    # heading to FOLLOW it. A spawn_config still keyed on sorted()[0] returns the same
    # heading for every target and fails here whatever the ids happen to be.
    headings = {nb: missions.spawn_seed(start_toward=nb)[5] for nb in nbrs}
    assert len(set(round(v, 1) for v in headings.values())) == len(nbrs), (
        f"spawn heading did not change with start_toward: {headings} — the pose is "
        f"being derived from an ordering, not from the named target")


@pytest.mark.parametrize("town,cmap,npz,start,toward", [
    ("town05", "cluster_map.carla_town05.yaml", "regions.town05.npz", 45, 66),
    ("town01", "cluster_map.carla_town01.yaml", "regions.town01.npz", 7, 30),
])
def test_spawn_pose_round_trips_and_points_the_right_way(tmp_path, town, cmap, npz,
                                                         start, toward):
    _scripts_on_path()
    import json as _json
    import math as _math
    import missions
    from carla_gt_bridge.frames import carla_to_planar
    from carla_gt_bridge.region_lookup import load_region_table

    cfg = os.path.join(PKG, "config")
    for f in (cmap, npz):
        if not os.path.exists(os.path.join(cfg, f)):
            pytest.skip(f"{f} not present")

    p = missions.spawn_config(str(tmp_path), town=town,
                              cluster_map=os.path.join(cfg, cmap),
                              regions_npz=os.path.join(cfg, npz),
                              start_region=start, start_toward=toward)
    sp = _json.load(open(p))["objects"][0]["spawn_point"]

    # 1. CARLA -> planar -> the region we meant. A sign error fails here.
    table = load_region_table(os.path.join(cfg, npz))
    # carla_to_planar returns yaw in RADIANS (it takes degrees). Converting again
    # here is a 174-degree error that looks exactly like a frame sign bug.
    px, py, pyaw_rad = carla_to_planar(sp["x"], sp["y"], sp["yaw"])
    assert table.region_at(px, py) == start

    # 2. The heading points at `toward` and NOT at any other neighbour. An
    #    ordering-derived heading fails here.
    tx, ty = table.centroid_of(toward)
    want = _math.atan2(ty - py, tx - px)
    err = abs((pyaw_rad - want + _math.pi) % (2 * _math.pi) - _math.pi)
    assert err < _math.radians(5), f"spawn yaw is {_math.degrees(err):.1f} deg off"


def test_town01_cone_poses_land_in_their_own_junctions():
    """A cone one region off is answered correctly by the oracle about the wrong place."""
    _scripts_on_path()
    import missions
    from carla_gt_bridge.frames import carla_to_planar
    from carla_gt_bridge.region_lookup import load_region_table

    npz = os.path.join(PKG, "config", "regions.town01.npz")
    if not os.path.exists(npz):
        pytest.skip("regions.town01.npz not present")
    table = load_region_table(npz)
    for rid in (30, 28):
        cx, cy = missions.carla_pose_of(table, rid)
        px, py, _ = carla_to_planar(cx, cy, 0.0)
        assert table.region_at(px, py) == rid, f"cone for {rid} landed elsewhere"
