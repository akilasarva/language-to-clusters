"""Tests for the steering aim point (`sampling_mpc.aim_point`).

THE FAILURE MODE. Town05 region 1 is a **51 m long `path`** whose centroid is at
x = -86.7. A vehicle at x = -104.7, driving along the region away from the centroid, is
~20 m past it. Aiming `bearing_ref` at that centroid puts the target BEHIND the vehicle
(`|bearing_ref| > 90 deg`), so the controller steers backwards down the region it is
already correctly driving along, turns until it is perpendicular to the road, leaves the
carriageway and stops.

The target region is right; aiming at its centroid is wrong. The region controller
handles this in `plan_step`: *"the strictly nearest point is the region's near EDGE, so
the vehicle drives to the boundary and stops there ... preferring a point at least
`lookahead_m` away aims THROUGH the region instead of at it."* `aim_point` applies the
same rule, made directional, for both controllers.

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest \\
         dgppo_ros_node_pkg/test/test_aim_point.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)

from dgppo_ros_node_pkg.sampling_mpc import RoadSurface, aim_point  # noqa: E402

#: The geometry of Town05 region 1, straightened onto the x axis: a 51 m path whose
#: centroid sits in the middle, with the vehicle near one end driving away from the centroid.
REGION = 1
_XS = np.linspace(-111.4, -61.7, 130)
ROAD = RoadSurface(np.stack([_XS, np.zeros_like(_XS),
                             np.full_like(_XS, REGION)], axis=1))
CENTROID = np.array([_XS.mean(), 0.0])          # ~ -86.5, matching the real -86.7

#: The next region of the route, continuing past region 1's far end -- what the plan would
#: step to next, and what supplies an aim once the target region runs out ahead.
NEXT_REGION = 63
_XN = np.linspace(-141.0, -111.6, 80)
ROAD_WITH_NEXT = RoadSurface(np.vstack([
    np.stack([_XS, np.zeros_like(_XS), np.full_like(_XS, REGION)], axis=1),
    np.stack([_XN, np.zeros_like(_XN), np.full_like(_XN, NEXT_REGION)], axis=1)]))


def test_the_centroid_is_behind_a_vehicle_that_has_driven_past_it():
    """The premise. If this fails the scenario is not the one described above."""
    pos = np.array([-104.7, 0.0])
    yaw = math.pi                                # driving toward -x, away from the centroid
    to_centroid = CENTROID - pos
    forward = np.array([math.cos(yaw), math.sin(yaw)])
    assert float(to_centroid @ forward) < 0, "the centroid must be behind for this to bite"
    assert abs(np.linalg.norm(to_centroid) - 18.0) < 4.0


def test_the_aim_is_ahead_when_the_vehicle_is_already_inside_the_target_region():
    """THE FIX. Inside a long region, aim THROUGH it, never back at its centroid.

    With centroid aiming the aim is behind the vehicle here.
    """
    pos = np.array([-104.7, 0.0])
    yaw = math.pi
    aim = aim_point(ROAD, REGION, pos, yaw, lookahead_m=12.0)
    forward = np.array([math.cos(yaw), math.sin(yaw)])
    assert float((aim - pos) @ forward) > 0, "the aim point is behind the vehicle"
    bearing = math.atan2(*((aim - pos)[::-1])) - yaw
    bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
    assert abs(bearing) < math.radians(30), f"bearing_ref {math.degrees(bearing):.0f} deg"


def test_when_the_region_runs_out_ahead_the_aim_comes_from_the_route_AHEAD():
    """The aim is NOT always >= `lookahead_m` away within the target region alone.

    At x = -104.7 heading -x there is only **6.7 m** of region 1 left ahead, so no point 12 m
    ahead exists INSIDE the target region, and a target-only pool degrades to the furthest
    scrap of it: a 6 m lever arm, and a weak steering signal exactly where the vehicle is
    about to need a decision. That is not a bug in the rule, it is the rule running out of
    road, and the answer is the NEXT region -- which is what `also` carries.
    """
    pos = np.array([-104.7, 0.0])
    # Target-only: the best available is the region's far end, under the lookahead.
    near = aim_point(ROAD, REGION, pos, math.pi, lookahead_m=12.0)
    assert np.linalg.norm(near - pos) < 12.0
    # With the next region of the route in the pool, a proper lookahead exists again.
    aim = aim_point(ROAD_WITH_NEXT, REGION, pos, math.pi, lookahead_m=12.0, also=[NEXT_REGION])
    assert np.linalg.norm(aim - pos) >= 12.0 - 1e-6
    forward = np.array([math.cos(math.pi), math.sin(math.pi)])
    assert float((aim - pos) @ forward) > 0


def test_it_does_not_overshoot_to_the_far_end_of_a_long_region():
    """Take the NEAREST point that is far enough, not the farthest. A 51 m region's far end
    is a 50 m lever arm and turns every small lateral error into a large bearing error."""
    pos = np.array([-104.7, 0.0])
    aim = aim_point(ROAD, REGION, pos, math.pi, lookahead_m=12.0)
    assert np.linalg.norm(aim - pos) < 20.0


def test_a_target_region_genuinely_ahead_is_unaffected():
    """The common case must not change: when the target is in front, aim at it as before."""
    pos = np.array([-120.0, 0.0])
    yaw = 0.0                                    # driving toward +x, region ahead
    aim = aim_point(ROAD, REGION, pos, yaw, lookahead_m=12.0)
    forward = np.array([math.cos(yaw), math.sin(yaw)])
    assert float((aim - pos) @ forward) > 0
    assert np.linalg.norm(aim - pos) >= 12.0 - 1e-6


def test_a_region_entirely_behind_still_returns_an_aim_rather_than_nothing():
    """A real reversal must not return None and stop the vehicle: the controller needs a
    finite bearing every tick. Aiming backwards is then CORRECT -- the point is that it
    happens because the region is behind, not because of a stale centroid."""
    pos = np.array([-40.0, 0.0])
    yaw = 0.0                                    # driving away from the whole region
    aim = aim_point(ROAD, REGION, pos, yaw, lookahead_m=12.0)
    assert aim is not None and np.isfinite(aim).all()
    assert float((aim - pos) @ np.array([1.0, 0.0])) < 0


def test_an_unknown_region_returns_none_rather_than_a_wrong_point():
    """Silently substituting some other region's geometry would surface as a steering bug
    far downstream; the caller must be able to tell and fall back explicitly."""
    assert aim_point(ROAD, 999, np.array([0.0, 0.0]), 0.0, lookahead_m=12.0) is None
