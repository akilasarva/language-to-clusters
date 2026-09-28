"""The drivable surface must cover the lanes, not just the road reference line.

WHY. `OpenDriveSegment.sample` returns the reference line, and its docstring justifies
that with "a lane sits ~1.75 m to one side ... so it cannot change which region a point
belongs to". True for REGION CLASSIFICATION. The same table is also the MPC's DRIVABLE
SURFACE (`sampling_mpc.RoadSurface.excess` = distance to nearest waypoint minus a global
`road_half_width = 7.0`), and there the offset is the whole question.

On Town05 there is a ~35 m stretch near y = 7 where the only reference line is ~8.4 m
away, so a vehicle there is beyond the surface with nothing to steer back to.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XODR = os.path.join(PKG, "config", "Town05.xodr")

#: Observed stranded poses in that stretch (vehicle stuck beyond the reference-line surface).
WEDGE_POSES = [(-243.8, 7.1), (-243.4, 7.1), (-244.6, 7.3), (-249.4, 6.9), (-243.7, 7.1),
               (-249.3, 6.9), (-243.8, 7.1), (-248.9, 6.9), (-249.6, 6.9), (-243.8, 7.0),
               (-243.9, 7.1), (-243.7, 7.0), (-243.8, 7.0), (-244.8, 7.3), (-243.8, 7.0),
               (-243.8, 7.0), (-251.9, 6.9), (-251.1, 6.9), (-251.8, 6.9), (-247.6, 6.9),
               (-250.6, 6.9), (-249.2, 6.9), (-249.2, 6.9)]

ROAD_HALF_WIDTH = 7.0          # sampling_mpc.MpcConfig.road_half_width


pytestmark = pytest.mark.skipif(not os.path.exists(XODR), reason="Town05.xodr not present")


def _seg(**kw):
    import sys
    sys.path.insert(0, PKG)
    from carla_gt_bridge import opendrive as OD, segmenter as SG
    return SG.segment(OD.load(XODR), **kw)


#: What the shipped table is generated with. Changing this changes the drivable surface
#: for every driven run, so it is pinned here rather than left to whoever last ran the CLI.
SHIPPED_LANES_PER_SIDE = 2

#: The reference-line-only table size. Kept as a number so a regression that silently
#: drops the lane samples is caught by value, not by absence.
REFERENCE_LINE_ONLY_WAYPOINTS = 7817


def test_the_default_is_still_reference_line_only():
    """`lanes_per_side=0` must reproduce the historical table exactly.

    The flag is opt-in; the reference-line-only table has to stay reproducible.
    """
    assert _seg().waypoints().shape[0] == REFERENCE_LINE_ONLY_WAYPOINTS


def test_the_shipped_table_is_the_lane_sampled_one():
    """Pins what is actually deployed. `config/regions.town05.npz` IS the drivable
    surface every driven run uses, so it must not drift from the generator."""
    shipped = np.load(os.path.join(PKG, "config", "regions.town05.npz"),
                      allow_pickle=True)["waypoints"]
    assert shipped.shape[0] == _seg(lanes_per_side=SHIPPED_LANES_PER_SIDE).waypoints().shape[0]
    assert shipped.shape[0] > REFERENCE_LINE_ONLY_WAYPOINTS


def test_lane_samples_bring_every_wedge_pose_onto_the_surface():
    w = _seg(lanes_per_side=SHIPPED_LANES_PER_SIDE).waypoints()
    far = [p for p in WEDGE_POSES
           if np.hypot(w[:, 0] - p[0], w[:, 1] - p[1]).min() > ROAD_HALF_WIDTH]
    assert not far, f"{len(far)} wedge poses still beyond the drivable surface: {far[:3]}"


def test_the_north_carriageway_hole_is_closed():
    """A 35.3 m stretch with no sample on one side, against 2.0 m on the other."""
    w = _seg(lanes_per_side=SHIPPED_LANES_PER_SIDE).waypoints()
    north = w[(w[:, 1] > 2) & (w[:, 1] < 12) & (w[:, 0] > -275) & (w[:, 0] < -200)]
    assert len(north) > 100
    assert np.diff(np.sort(north[:, 0])).max() < 5.0


def test_lane_offsets_are_perpendicular_to_the_heading():
    """A straight road's lane samples must be offset across it, not along it."""
    import math
    from carla_gt_bridge.segmenter import _lane_samples
    pts = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)]        # heading +x
    out = _lane_samples(pts, driving_width=8.0, lanes_per_side=1)
    ys = sorted({round(y, 3) for _, y, _ in out})
    assert ys == [-2.0, 0.0, 2.0], ys              # centres of two 4 m lanes
    assert all(math.isclose(x, 0.0) or math.isclose(x, 2.0) for x, _, _ in out)
