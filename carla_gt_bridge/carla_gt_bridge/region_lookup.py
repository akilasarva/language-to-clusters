"""Which cluster region is a pose in? Ground-truth answer, no CARLA, no ROS.

This is the *exact* test — nearest sampled waypoint over the region table produced by
`scripts/map_regions.py`. It stands in for the LiDAR classifier so the plan-to-motion
chain can be validated before any perception is involved, and it is what
`nodes/gt_cluster_node.py` publishes on ``/predicted_cluster``.

Deliberately NOT the same test the MPC uses
-------------------------------------------
The MPC scores 500x8 rollout endpoints every tick, so it uses nearest **centroid** —
a Voronoi approximation, chosen for speed. This node advances the mission, where
fidelity matters, so it uses nearest **waypoint**. The two disagree by design:
nearest-centroid misassigns a substantial fraction of waypoints (especially on long
`path` regions, whose far end is nearer a neighbour's centroid), while nearest
waypoint is exact by construction.

Both read the same `regions.<town>.npz`, so they cannot disagree about what a region
*is* — only about how precisely its extent is approximated.

Off-road poses
--------------
The waypoint table samples road *reference lines*, so a pose on a wide road's outer
lane is legitimately ~5 m from the nearest sample. ``max_distance_m`` therefore
defaults generously (12 m) and exists only to catch a pose that is nowhere near the
network at all — a mis-set town, a wrong frame, a vehicle that fell off the map. It
returns ``None`` rather than the least-wrong region, because silently reporting a
plausible id is how a frame bug survives to the end of a run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["RegionTable", "load_region_table"]

#: Beyond this, a pose is treated as off-network rather than assigned a region.
#: Generous on purpose: reference-line sampling puts an outer lane several metres from
#: the nearest sample, and a false "off network" would stall a mission that is fine.
DEFAULT_MAX_DISTANCE_M = 12.0


@dataclass
class RegionTable:
    """Region waypoints + labels, with a nearest-waypoint query.

    ``waypoints`` is the (N, 3) ``x, y, rid`` array from ``regions.<town>.npz``, in
    **planar** metres (+y north) — the same frame the cluster map's centroids and the
    ros-bridge's odometry use, so a pose taken off `/carla/<role>/odometry` needs no
    conversion at all. A pose from the CARLA Python API does; see
    :mod:`carla_gt_bridge.frames`.
    """

    waypoints: np.ndarray                 # (N, 3) x, y, rid
    rids: np.ndarray                      # (R,) sorted region ids
    labels: np.ndarray                    # (R,) label per rid, index-aligned
    centroids: np.ndarray                 # (R, 2) per rid, index-aligned
    town: str = ""

    def __post_init__(self) -> None:
        self._xy = np.ascontiguousarray(self.waypoints[:, :2], dtype=float)
        self._rid = self.waypoints[:, 2].astype(int)
        self._tree = None
        try:                              # scipy is a soft dependency
            from scipy.spatial import cKDTree
            self._tree = cKDTree(self._xy)
        except ImportError:               # pragma: no cover - exercised by fallback test
            pass

    # -- queries ---------------------------------------------------------- #

    def nearest(self, x: float, y: float) -> tuple[int, float]:
        """Return ``(region_id, distance_m)`` of the nearest sampled waypoint."""
        if self._tree is not None:
            d, i = self._tree.query([x, y])
            return int(self._rid[int(i)]), float(d)
        # Brute force keeps this usable without scipy: a town table is ~10k points,
        # which is well under a millisecond, so the fallback is not a degraded mode.
        d2 = np.square(self._xy - np.array([x, y])).sum(axis=1)
        i = int(np.argmin(d2))
        return int(self._rid[i]), float(np.sqrt(d2[i]))

    def region_at(self, x: float, y: float,
                  max_distance_m: float = DEFAULT_MAX_DISTANCE_M) -> int | None:
        """Region id containing ``(x, y)``, or ``None`` if it is off the network."""
        rid, dist = self.nearest(x, y)
        return None if dist > max_distance_m else rid

    def label_of(self, rid: int) -> str:
        idx = int(np.searchsorted(self.rids, rid))
        if idx >= len(self.rids) or int(self.rids[idx]) != int(rid):
            raise KeyError(f"region id {rid} not in table for {self.town or '?'}")
        return str(self.labels[idx])

    def centroid_of(self, rid: int) -> tuple[float, float]:
        idx = int(np.searchsorted(self.rids, rid))
        if idx >= len(self.rids) or int(self.rids[idx]) != int(rid):
            raise KeyError(f"region id {rid} not in table for {self.town or '?'}")
        return (float(self.centroids[idx][0]), float(self.centroids[idx][1]))

    @property
    def region_ids(self) -> list[int]:
        return [int(r) for r in self.rids]


def load_region_table(npz_path: str) -> RegionTable:
    """Load a ``regions.<town>.npz`` written by ``scripts/map_regions.py``."""
    with np.load(npz_path, allow_pickle=False) as z:
        missing = [k for k in ("waypoints", "rids", "labels", "centroids")
                   if k not in z]
        if missing:
            raise KeyError(f"{npz_path} is missing {missing}; regenerate it with "
                           f"scripts/map_regions.py")
        table = RegionTable(
            waypoints=z["waypoints"].astype(float),
            rids=z["rids"].astype(int),
            labels=z["labels"].astype(str),
            centroids=z["centroids"].astype(float),
            town=npz_path.split("/")[-1].replace("regions.", "").replace(".npz", ""),
        )
    return table
