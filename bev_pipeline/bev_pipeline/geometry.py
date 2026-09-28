"""Geometric primitives, features, and structural auto-labeling.

Everything here is computed directly from a gravity-leveled, ground-removed
(ideally submap-accumulated) cloud, so both the FEATURES and the LABELS derived
from it are, by construction, recoverable from LiDAR geometry (unlike
camera-semantic labels, which LiDAR features cannot always reproduce).

Two products:
  * :func:`geometric_features` — an interpretable fixed-length vector (polar
    free-space profile + clearances + overhead occupancy + height histogram)
    for use as a small-data-friendly extractor.
  * :func:`classify_geometry` — a deterministic pedestrian-navigation state from
    that geometry, one of :data:`GEOM_LABELS`.

The "sensor origin" is the robot; +X forward, +Y left; z is up with the ground
near ``z = -sensor_height`` (ground already removed, so points are structure).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

# GEOMETRIC cluster vocabulary — must stay in lockstep with the `name` field of
# config/nav_modes.yaml (the hand-labeling vocab). This is deliberately NOT the
# planner-facing vocabulary; cluster_map.<env>.yaml is the seam that binds these
# ids to planner mode names. See taxonomy_export.HIERARCHY.
#
# Differences from the legacy vocabulary (GEOM_LABELS_LEGACY):
#   - "corridor" renamed  -> "passage"  (align with nav_modes.yaml / labeling)
#   - "covered"  dropped  -> a down-tilted LiDAR cannot observe overhead, and an
#     upright LiDAR sees canopy/ceiling almost everywhere, so an overhead-first
#     rule labels every frame 'covered'. It is now a FLAG on the summary
#     ('overhead_frac'), not a class — same treatment as 'constriction'.
#   - "path"     added    -> the common default (a channeled way); without it
#     every channeled way falls through to open_space.
GEOM_LABELS = ["open_space", "path", "along_edge", "passage", "junction"]

#: Vocabulary used by datasets built before the rename. Datasets record their own
#: ``geom_label_names`` in dataset_meta.json, so stored ``labels_geom.npy``
#: indices remain valid — read the meta, not GEOM_LABELS, when loading old data.
#: Use :func:`canonicalize_geom_label` to translate legacy names forward.
GEOM_LABELS_LEGACY = ["open_space", "along_edge", "corridor", "covered", "junction"]

#: Legacy geometric label -> current name. 'covered' has no current equivalent
#: (it is no longer a class); it maps to None so callers must decide explicitly
#: whether to drop those frames or fold them.
GEOM_LABEL_ALIASES = {
    "corridor": "passage",
    "covered": None,
}


def canonicalize_geom_label(label: str):
    """Translate a legacy geometric label to the current vocabulary.

    Returns None for labels that no longer exist as a class (``covered``), so the
    caller must handle them deliberately rather than silently mis-binding an id.
    """
    if label in GEOM_LABEL_ALIASES:
        return GEOM_LABEL_ALIASES[label]
    return label if label in GEOM_LABELS else None


@dataclass
class GeomParams:
    # Defaults tuned on plane-fit-cleaned full_campus (balanced 5/6-state split).
    n_sectors: int = 36           # 10-degree azimuth sectors
    min_range: float = 2.0        # ignore near-origin self-returns / noise
    max_range: float = 20.0       # cap for free-space range
    profile_pct: float = 10.0     # per-sector percentile for "structure begins"
                                  # (min is too sensitive to sparse near clutter)
    min_sector_pts: int = 5       # sectors with fewer returns count as open
    struct_z_min: float = 0.7     # only returns ABOVE this (robot-relative) count
                                  # as vertical structure (walls), not low clutter
    side_dist: float = 6.0        # lateral clearance threshold (structure "close")
    front_dist: float = 3.0       # forward clearance flag (not a class)
    open_range: float = 12.0      # a sector is "open" if nearest return > this
    overhead_z: float = 2.5       # points above this (robot-relative) = overhead
    overhead_frac_thr: float = 0.03   # 'covered' FLAG threshold, no longer a class
    # Open-direction-group ladder separating the vertically-open states. Typical
    # n_open by class: passage/along_edge ~0, path ~1-2, junction ~2-3,
    # open_space ~6. `junction` is the least reliable of these across
    # environments — the plan, not this rule, is the authority on where an
    # intersection is.
    # PROVISIONAL — calibrate against hand labels; do not treat these as tuned.
    junction_n_open: int = 3      # >= this many distinct open directions
    open_space_n_open: int = 5    # genuinely roam-anywhere; checked BEFORE junction
    # In CARLA, frames inside an intersection typically score n_open == 1 with
    # side clearance at max_range (nothing close), so this rule cannot see the
    # intersection. CARLA's perception-backed `Intersection: In` binding must
    # come from the legacy AE+HDBSCAN model (trained on CARLA), NOT from
    # classify_geometry.
    front_arc_deg: float = 30.0
    side_arc_deg: float = 40.0     # half-width of the left/right cones
    n_height_bins: int = 8
    z_lo: float = -2.0
    z_hi: float = 6.0


def polar_profile(points: np.ndarray, p: GeomParams) -> np.ndarray:
    """Nearest structure range per azimuth sector (a 2D free-space scan).

    Empty sectors are set to ``max_range``. Uses horizontal range of all
    (non-ground) points. Returns ``(n_sectors,)``.
    """
    prof = np.full(p.n_sectors, p.max_range, dtype=np.float64)
    if points is None or points.shape[0] == 0:
        return prof
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.sqrt(x * x + y * y)
    # Only tall returns (vertical structure), within range band — this is the
    # key to measuring distance-to-wall rather than distance-to-ground-clutter.
    m = (r >= p.min_range) & (r <= p.max_range) & np.isfinite(r) & (z > p.struct_z_min)
    if not np.any(m):
        return prof
    x, y, r = x[m], y[m], r[m]
    az = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0
    sec = np.clip((az / (360.0 / p.n_sectors)).astype(np.int64), 0, p.n_sectors - 1)
    # Per-sector robust range: the profile_pct-th percentile of ranges, so a few
    # stray near points don't collapse the whole sector. Sparse sectors (< min
    # points) are treated as open (max_range).
    for k in range(p.n_sectors):
        rk = r[sec == k]
        if rk.size >= p.min_sector_pts:
            prof[k] = np.percentile(rk, p.profile_pct)
    return prof


def _arc_min(prof: np.ndarray, center_deg: float, half_deg: float,
             p: GeomParams) -> float:
    """Min profile range within an angular arc centered at center_deg."""
    step = 360.0 / p.n_sectors
    centers = (np.arange(p.n_sectors) + 0.5) * step
    d = np.abs((centers - center_deg + 180) % 360 - 180)
    sel = d <= half_deg
    return float(prof[sel].min()) if np.any(sel) else p.max_range


def _open_direction_groups(prof: np.ndarray, p: GeomParams) -> int:
    """Count contiguous groups of 'open' sectors (distinct traversable openings)."""
    openmask = prof >= p.open_range
    if not openmask.any():
        return 0
    # wrap-around contiguous groups
    m = np.concatenate([openmask, openmask[:1]])
    transitions = np.sum((~m[:-1]) & m[1:])
    # if fully open, that's one big opening
    if openmask.all():
        return 1
    return int(transitions)


def geometry_summary(points: np.ndarray, p: GeomParams = None) -> Dict[str, float]:
    """Compute the interpretable scalar geometry summary used by labeler + features."""
    p = p or GeomParams()
    prof = polar_profile(points, p)
    front = _arc_min(prof, 0.0, p.front_arc_deg, p)
    left = _arc_min(prof, 90.0, p.side_arc_deg, p)
    right = _arc_min(prof, 270.0, p.side_arc_deg, p)
    n_open = _open_direction_groups(prof, p)
    if points is not None and points.shape[0] > 0:
        z = points[:, 2]
        rr = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
        near = (rr >= p.min_range) & (rr < p.max_range)
        overhead_frac = float(np.mean((z[near] > p.overhead_z))) if near.any() else 0.0
    else:
        overhead_frac = 0.0
    return {"front": front, "left": left, "right": right,
            "n_open": float(n_open), "overhead_frac": overhead_frac,
            "mean_range": float(prof.mean()), "min_range": float(prof.min())}


def classify_from_summary(s: dict, p: GeomParams = None) -> str:
    """Assign a state from a precomputed geometry summary (priority order).

    Split out from :func:`classify_geometry` so label-tuning can vary the
    threshold params over cached summaries without reprocessing point clouds.
    """
    p = p or GeomParams()
    left_close = s["left"] < p.side_dist
    right_close = s["right"] < p.side_dist
    # NOTE: overhead_frac is deliberately NOT consulted here any more. It used to
    # return "covered" as the FIRST branch, which swallowed every frame on the
    # upright Ouster bags (they see canopy/ceiling). Read s["overhead_frac"] as a
    # flag if you need it. See GEOM_LABELS.
    if left_close and right_close:
        return "passage"
    if left_close ^ right_close:
        return "along_edge"
    # Vertically open. Separate roam-anywhere from a channeled way, with
    # `junction` in between. open_space is checked first because its threshold is
    # the stricter one.
    if s["n_open"] >= p.open_space_n_open:
        return "open_space"
    if s["n_open"] >= p.junction_n_open:
        return "junction"
    return "path"


def classify_geometry(points: np.ndarray, p: GeomParams = None) -> str:
    """Deterministic pedestrian-navigation state from geometry.

    Returns one of GEOM_LABELS. 'constriction' (low forward clearance) is NOT a
    class — read ``geometry_summary(...)['front']`` as a flag if needed.
    """
    p = p or GeomParams()
    return classify_from_summary(geometry_summary(points, p), p)


def geometric_features(points: np.ndarray, p: GeomParams = None) -> np.ndarray:
    """Interpretable fixed-length geometric feature vector for classification.

    Concatenates: normalized polar profile (n_sectors) + [front,left,right,
    n_open,overhead_frac,mean_range,min_range] + height histogram (n_height_bins).
    """
    p = p or GeomParams()
    prof = polar_profile(points, p) / p.max_range
    s = geometry_summary(points, p)
    scalars = np.array([s["front"] / p.max_range, s["left"] / p.max_range,
                        s["right"] / p.max_range, s["n_open"] / p.n_sectors,
                        s["overhead_frac"], s["mean_range"] / p.max_range,
                        s["min_range"] / p.max_range], dtype=np.float64)
    if points is not None and points.shape[0] > 0:
        hh, _ = np.histogram(np.clip(points[:, 2], p.z_lo, p.z_hi),
                             bins=p.n_height_bins, range=(p.z_lo, p.z_hi))
        hh = hh / max(1, hh.sum())
    else:
        hh = np.zeros(p.n_height_bins)

    # Extended descriptors NOT used by classify_geometry's decision rule — they
    # add signal (structure continuity vs scatter, side asymmetry, enclosure)
    # and reduce circularity when geom_features is scored against geom labels.
    mr = p.max_range
    asymmetry = abs(s["left"] - s["right"]) / mr
    fs_mean = 0.5 * (s["left"] + s["right"]) + 1e-6
    front_side_ratio = min(3.0, s["front"] / fs_mean) / 3.0
    roughness = float(np.mean(np.abs(np.diff(prof))))          # walls smooth, veg rough
    closed_frac = float(np.mean(prof < (p.open_range / mr)))    # enclosure
    prof_std = float(np.std(prof))
    extended = np.array([asymmetry, front_side_ratio, roughness, closed_frac, prof_std],
                        dtype=np.float64)
    return np.concatenate([prof, scalars, hh, extended]).astype(np.float32)


def feature_dim(p: GeomParams = None) -> int:
    p = p or GeomParams()
    return p.n_sectors + 7 + p.n_height_bins + 5
