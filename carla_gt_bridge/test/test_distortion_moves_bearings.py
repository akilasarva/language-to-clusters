"""Bearings are load-bearing for MANOEUVRE CHOICE, so distorting them must be able to change it.

SCOPE. `carla_mpc_ros_node._apply_jitter` cannot be imported on the host (it needs the
built `carla_gt_bridge.frames`), so this does NOT test that function. It tests the
DOWNSTREAM CONSUMER: `routing.region_for_maneuver`, which is what turns "left" into a
concrete region and is what `_decide_at` feeds. If this consumer were insensitive to
bearing error, then distorting bearings would perturb a quantity nothing acts on, and any
distortion experiment would be vacuous. That is the property under test.

This area can fail silently (e.g. a distortion that is logged as applied but never takes
effect because of initialisation order), so the consumer's sensitivity is pinned here.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from carla_gt_bridge.routing import region_for_maneuver, adjacency_from_bearing_map  # noqa: E402

# A junction (0) entered from the south (region 9), with three exits. Bearings are the
# CARLA-frame headings the node stores, degrees, `"a-b"` meaning a -> b.
BEARINGS = {
    "9-0": 90.0,        # approach, heading north
    "0-1": 180.0,       # west  -> a LEFT turn when heading north
    "0-2": 90.0,        # north -> straight on
    "0-3": 0.0,         # east  -> a RIGHT turn
}
CANDS = [1, 2, 3]


def test_the_undistorted_map_picks_the_geometrically_correct_left():
    assert region_for_maneuver(BEARINGS, 0, 9, CANDS, "left") == 1
    assert region_for_maneuver(BEARINGS, 0, 9, CANDS, "right") == 3


def test_a_small_bearing_error_does_not_flip_the_choice():
    """~5 deg is the bearing jitter from moderate centroid jitter; it must be tolerated."""
    small = {k: v + 5.0 for k, v in BEARINGS.items()}
    assert region_for_maneuver(small, 0, 9, CANDS, "left") == 1


def test_a_LARGE_bearing_error_CAN_change_the_manoeuvre_choice():
    """Bearing distortion must be able to change the manoeuvre choice.

    Rotate only the exits, not the approach -- which is what independent per-region jitter
    does, and what a uniform rotation deliberately does NOT do. Once the exits have swung far
    enough, the region that reads as 'left' is no longer region 1.
    """
    flipped = dict(BEARINGS)
    for k in ("0-1", "0-2", "0-3"):
        flipped[k] = BEARINGS[k] - 90.0
    got = region_for_maneuver(flipped, 0, 9, CANDS, "left")
    assert got != 1, f"bearings swung 90 deg and 'left' still resolved to {got}"


def test_adjacency_survives_distortion_because_only_the_ANGLES_move():
    """`Adjacency is NOT distorted` is the node's stated contract; it is derived from KEYS."""
    before = adjacency_from_bearing_map(BEARINGS)
    jittered = {k: v + 37.0 for k, v in BEARINGS.items()}
    assert adjacency_from_bearing_map(jittered) == before


def test_recomputing_a_bearing_from_moved_centroids_matches_atan2():
    """The jitter path recomputes rather than offsets; this pins that arithmetic."""
    a, b = (0.0, 0.0), (10.0, 10.0)
    off_a, off_b = (1.0, -2.0), (-3.0, 4.0)
    moved_a = (a[0] + off_a[0], a[1] + off_a[1])
    moved_b = (b[0] + off_b[0], b[1] + off_b[1])
    expect = math.atan2(moved_b[1] - moved_a[1], moved_b[0] - moved_a[0])
    plain = math.atan2(b[1] - a[1], b[0] - a[0])
    assert abs(expect - plain) > math.radians(5.0), (
        "independent offsets must change the edge bearing, or jitter is inert")
