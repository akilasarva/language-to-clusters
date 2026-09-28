"""Multi-channel polar features from one LiDAR scan.

WHY THIS EXISTS. The original CARLA feature is a single channel: min-range per angular
sector over the band |z| in [0.1, 1.4]. On the `bridge1_carla` PCDs that band returns
nothing in most sectors, so the encoder learns a vector that is mostly constant filler.
This module computes several (band, reducer) channels plus a per-sector validity mask.

THE REDUCER MATTERS AS MUCH AS THE BAND:

    min over a STRUCTURE band -> distance to the nearest thing that can hit you
    max over a GROUND band    -> how far the drivable surface extends

Those are different questions. A junction should show long ground-extent lobes on four
sides where a corridor shows two; that is a reducer change, not a new sensor.

STATUS. Coverage and per-sector variance can be checked directly (see
:func:`channel_coverage`). Whether the extra channels SEPARATE the classes has not been
established -- that needs ground truth. Treat the channel design as motivated, not
validated.

Do not rank features on HDBSCAN noise or flip rate: those trade against each other
(a large min_cluster_size reduces flips by collapsing to few clusters) and
nearest-centroid reassignment drives noise to 0 by construction, so any ranking on them
prefers whichever feature degenerates most gracefully.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["ChannelSpec", "PolarFeatureConfig", "polar_channels", "feature_vector",
           "channel_coverage"]


@dataclass(frozen=True)
class ChannelSpec:
    """One (band, reducer) pair -> one row of the feature."""

    name: str
    z_lo: float
    z_hi: float
    #: "min" nearest return (obstacles) | "max" furthest return (free-space extent)
    #: | "count" returns per sector (occupancy density, range-independent)
    #: | "zext" vertical extent within the sector (how tall the structure is)
    reducer: str
    #: Value written where a sector has no return. For `min` this is the range cap
    #: ("nothing out to here"); for `max`/count/zext it is 0.
    fill: float | None = None

    def fill_value(self, max_range: float) -> float:
        """What an unmeasured sector reads.

        THE FILL MUST MATCH THE CHANNEL'S SEMANTICS, and getting it wrong inverts the
        signal. A `min` (obstacle) channel fills with `max_range`, meaning "nothing out
        to here" -- correct and conservative. A `max` (free-space extent) channel must
        fill with ZERO, meaning "no drivable surface observed this way". Filling extent
        with `max_range` makes an unobserved direction read as MAXIMALLY OPEN, so a
        corridor (more empty sectors) would score as more open than a crossroads.
        """
        if self.fill is not None:
            return self.fill
        return max_range if self.reducer == "min" else 0.0


@dataclass
class PolarFeatureConfig:
    """Bands are ROBOT-RELATIVE metres: z = 0 at the sensor, ground at -sensor_height.

    ``sensor_height`` is required and has no default. The bands mean nothing without it,
    and a default would silently produce a plausible-looking feature computed over the
    wrong slice of the world.
    """

    sensor_height: float
    num_sectors: int = 72
    min_range: float = 1.0
    max_range: float = 25.0
    #: Reject isolated returns before reducing. Mirrors `lidar_processor`'s filter so the
    #: two do not diverge; 0 disables.
    density_radius: float = 0.0
    min_neighbors: int = 0
    channels: list[ChannelSpec] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.channels:
            g = -self.sensor_height
            self.channels = [
                # free-space extent: how far the drivable surface reaches
                ChannelSpec("ground_extent", g - 0.1, g + 0.3, "max"),
                # nearest structure: what could hit you
                ChannelSpec("structure_near", g + 0.5, g + 2.5, "min"),
                # occupancy density, independent of distance
                ChannelSpec("ground_count", g - 0.1, g + 0.3, "count"),
                # how tall the structure is in this direction
                ChannelSpec("structure_zext", g + 0.3, g + 4.0, "zext"),
            ]
        if self.num_sectors < 4:
            raise ValueError(f"num_sectors must be >= 4, got {self.num_sectors}")


def _sector_index(xy: np.ndarray, n: int) -> np.ndarray:
    ang = np.arctan2(xy[:, 1], xy[:, 0]) % (2.0 * np.pi)
    return np.minimum((ang / (2.0 * np.pi) * n).astype(int), n - 1)


def polar_channels(points: np.ndarray, cfg: PolarFeatureConfig
                   ) -> tuple[np.ndarray, np.ndarray]:
    """(C, S) channel values and (C, S) bool "this sector had a return".

    The validity mask is returned rather than folded away because a filled sector and a
    measured one at the same value are not the same observation, and every aggregate
    computed without it is confounded -- a saturated sector contributes zero to a
    frame-to-frame difference no matter what the world did.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(f"points must be (N, >=3), got {pts.shape}")
    S, C = cfg.num_sectors, len(cfg.channels)
    out = np.zeros((C, S), dtype=float)
    valid = np.zeros((C, S), dtype=bool)

    d_all = np.linalg.norm(pts[:, :2], axis=1)
    keep = (d_all >= cfg.min_range) & (d_all <= cfg.max_range) & np.isfinite(d_all)
    pts, d_all = pts[keep], d_all[keep]

    if len(pts) and cfg.density_radius > 0 and cfg.min_neighbors > 0:
        try:
            from scipy.spatial import cKDTree
            cnt = cKDTree(pts[:, :2]).query_ball_point(
                pts[:, :2], r=cfg.density_radius, return_length=True)
            m = (cnt - 1) >= cfg.min_neighbors
            pts, d_all = pts[m], d_all[m]
        except ImportError:                                  # pragma: no cover
            pass

    for ci, ch in enumerate(cfg.channels):
        out[ci, :] = ch.fill_value(cfg.max_range)
        if not len(pts):
            continue
        band = (pts[:, 2] >= ch.z_lo) & (pts[:, 2] <= ch.z_hi)
        if not band.any():
            continue
        p, d = pts[band], d_all[band]
        sec = _sector_index(p[:, :2], S)
        for s in np.unique(sec):
            m = sec == s
            if ch.reducer == "min":
                out[ci, s] = d[m].min()
            elif ch.reducer == "max":
                out[ci, s] = d[m].max()
            elif ch.reducer == "count":
                out[ci, s] = m.sum()
            elif ch.reducer == "zext":
                out[ci, s] = p[m, 2].max() - p[m, 2].min()
            else:
                raise ValueError(f"unknown reducer {ch.reducer!r}")
            valid[ci, s] = True
    return out, valid


def feature_vector(points: np.ndarray, cfg: PolarFeatureConfig) -> np.ndarray:
    """(C*S,) normalised, channel-major. Distances by `max_range`; counts by their own
    per-frame sum (density is relative -- an absolute count depends on the sensor's point
    budget, which must not leak into the feature); zext by a 4 m nominal."""
    ch, _ = polar_channels(points, cfg)
    rows = []
    for ci, spec in enumerate(cfg.channels):
        v = ch[ci]
        if spec.reducer in ("min", "max"):
            rows.append(v / cfg.max_range)
        elif spec.reducer == "count":
            # By the SUM, not the max. Dividing by the max leaves a residual dependence
            # on the sensor's point budget through integer rounding -- doubling the
            # returns took a sector from 5/6 = 0.833 to 10/11 = 0.909 for identical
            # geometry. A share of the total is exactly scale-invariant.
            rows.append(v / max(v.sum(), 1.0))
        else:
            rows.append(np.clip(v / 4.0, 0.0, 1.0))
    return np.concatenate(rows)


def channel_coverage(points: np.ndarray, cfg: PolarFeatureConfig) -> dict[str, float]:
    """Fraction of sectors each channel actually measured. Worth checking for any new
    band before trusting it."""
    _, valid = polar_channels(points, cfg)
    return {c.name: float(valid[i].mean()) for i, c in enumerate(cfg.channels)}
