"""Ground removal via Patchwork++ (with an Open3D RANSAC fallback).

Patchwork++ (Lee et al., IROS 2022) fits an adaptive, concentric-zone ground
model — far more robust outdoors than a single global plane, which is why the
legacy planar z-slice failed on slopes/curbs. It assumes a **gravity-aligned**
cloud with +Z up, so the pipeline runs :func:`frame_geometry.gravity_align`
first.

:class:`GroundRemover` wraps the C++ binding and, if it is unavailable or
throws, transparently falls back to Open3D's RANSAC ``segment_plane`` so the
rest of the pipeline still runs (the fallback is cruder but keeps things
working on machines without the binding).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def plane_level_and_remove(points: np.ndarray, dist_thresh: float = 0.20,
                           ransac_iters: int = 300, min_points: int = 200,
                           struct_margin: float = 0.0
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """Self-calibrating ground fit: RANSAC the dominant plane, rotate its normal
    to +Z (fine tilt-leveling, no yaw change), and split ground/non-ground.

    This fixes the failure mode where Patchwork++ (expecting a flat plane at a
    fixed sensor height) barely removes a *tilted* ground — a residual mount
    pitch that odometry-only leveling leaves behind. The rotation axis lies in
    the XY plane (n x z), so heading/yaw is preserved (robocentric-safe).

    Returns ``(nonground_leveled, ground_leveled)`` — both rotated into the
    level frame. Falls back to returning the input unchanged (as non-ground) if
    the cloud is too small or the fit degenerates.
    """
    if points is None or points.shape[0] < min_points:
        empty = np.empty((0, points.shape[1] if points is not None else 4),
                         dtype=points.dtype if points is not None else np.float32)
        return (points.copy() if points is not None else empty, empty)
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(points[:, :3]))
        model, inl = pcd.segment_plane(distance_threshold=dist_thresh,
                                       ransac_n=3, num_iterations=ransac_iters)
    except Exception:                              # noqa: BLE001
        return points.copy(), points[:0]
    a, b, c, _d = model
    n = np.array([a, b, c], dtype=np.float64)
    nn = np.linalg.norm(n)
    if nn < 1e-9:
        return points.copy(), points[:0]
    n = n / nn
    if n[2] < 0:
        n = -n
    # minimal rotation aligning n -> +z (Rodrigues), axis = n x z lies in XY
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(n, z)
    s = np.linalg.norm(v)
    cth = float(np.clip(n @ z, -1.0, 1.0))
    if s < 1e-8:
        R = np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - cth) / (s * s))
    leveled_xyz = points[:, :3] @ R.T
    if points.shape[1] > 3:
        leveled = np.column_stack([leveled_xyz, points[:, 3:]]).astype(points.dtype)
    else:
        leveled = leveled_xyz.astype(points.dtype)
    inl = np.asarray(inl, dtype=np.int64)
    mask = np.zeros(len(points), dtype=bool)
    mask[inl] = True
    if struct_margin > 0:
        # also treat anything within struct_margin above the plane level as ground
        zlevel = np.median(leveled[mask, 2]) if mask.any() else 0.0
        mask |= leveled[:, 2] < (zlevel + struct_margin)
    return leveled[~mask], leveled[mask]


def keep_macro_structure(points: np.ndarray, eps: float = 0.7, min_points: int = 8,
                         min_extent: float = 2.0, min_count: int = 120) -> np.ndarray:
    """Keep only large-scale structure (walls/buildings); drop small blobs.

    Clusters the (non-ground) cloud and keeps clusters that are spatially
    EXTENDED (bbox side >= min_extent m) or POPULOUS (>= min_count pts). Small
    compact clusters — pedestrians, a follower walking behind, bikes, poles —
    are dropped, so the geometry reflects the macro navigable structure rather
    than transient obstacles. Catches the constant-distance follower that the
    submap temporal-persistence filter cannot (it looks static).
    """
    if points is None or points.shape[0] < min_points:
        return points if points is not None else np.empty((0, 4), np.float32)
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(points[:, :3]))
        labels = np.asarray(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    except Exception:                              # noqa: BLE001
        return points
    keep = np.zeros(len(points), dtype=bool)
    for c in np.unique(labels[labels >= 0]):
        m = labels == c
        xy = points[m, :2]
        extent = float((xy.max(axis=0) - xy.min(axis=0)).max())
        if extent >= min_extent or int(m.sum()) >= min_count:
            keep[m] = True
    return points[keep] if keep.any() else points[:0]


def polar_structure(points: np.ndarray, quat: np.ndarray, R: float = 14.0,
                    nb: int = 72, dr: float = 0.5, floor: float = 0.35,
                    lowpct: float = 15, minc: int = 3, rmin: float = 0.6) -> np.ndarray:
    """Tilt-robust standing-structure extraction (the render fed to the VLM).

    Global plane-fit leveling silently fails on some frames (fits a ~30 deg tilt,
    so the ground ramps up with range and reads as false 'structure'). Instead we
    bin the gravity-aligned cloud by (bearing, range) and drop the local low-z
    ground surface per bin; whatever stands ``floor`` metres above it is kept as
    structure. Robust to a mis-leveled frame because it never assumes one global
    ground plane. Returns (M,4) xyz(+intensity) in the leveled robocentric frame.

    Also used by ``tools/label_frames.py`` (labeler panel) so the human labels the
    SAME representation the VLM sees.
    """
    from .frame_geometry import gravity_align
    if points is None or quat is None or not np.all(np.isfinite(quat)):
        return np.zeros((0, 4))
    lev = gravity_align(points, quat)
    if lev is None or len(lev) < 50:
        return np.zeros((0, 4))
    if lev.shape[1] == 3:
        lev = np.column_stack([lev, np.zeros(len(lev))])
    x, y, z = lev[:, 0], lev[:, 1], lev[:, 2]
    r = np.hypot(x, y); br = np.arctan2(y, x) + np.pi
    m = (r > rmin) & (r < R)
    lev, z, r, br = lev[m], z[m], r[m], br[m]
    if len(lev) < minc:
        return np.zeros((0, 4))
    bi = np.clip((br / (2 * np.pi) * nb).astype(int), 0, nb - 1)
    ri = np.clip((r / dr).astype(int), 0, int(R / dr) - 1)
    key = bi * 1000 + ri
    keep = np.zeros(len(lev), bool)
    order = np.argsort(key); ks = key[order]
    _, st = np.unique(ks, return_index=True); st = np.append(st, len(ks))
    for j in range(len(st) - 1):
        idx = order[st[j]:st[j + 1]]
        if len(idx) < minc:
            continue
        gz = np.percentile(z[idx], lowpct)
        keep[idx[z[idx] > gz + floor]] = True
    return lev[keep]


@dataclass
class GroundRemovalParams:
    """Subset of Patchwork++ parameters we expose + fallback knobs."""

    sensor_height: float = 0.6      # metres; lidar height above ground
    min_range: float = 0.5
    max_range: float = 60.0
    verbose: bool = False
    # Open3D RANSAC fallback:
    ransac_dist_thresh: float = 0.15
    ransac_n: int = 3
    ransac_iters: int = 200


class GroundRemover:
    """Split a gravity-aligned cloud into (nonground, ground)."""

    def __init__(self, params: Optional[GroundRemovalParams] = None,
                 prefer_patchwork: bool = True):
        self.params = params or GroundRemovalParams()
        self._pw = None
        self._backend = "none"
        if prefer_patchwork:
            self._try_init_patchwork()

    def _try_init_patchwork(self) -> None:
        try:
            import pypatchworkpp
            p = pypatchworkpp.Parameters()
            p.sensor_height = self.params.sensor_height
            p.min_range = self.params.min_range
            p.max_range = self.params.max_range
            p.verbose = self.params.verbose
            self._pw = pypatchworkpp.patchworkpp(p)
            self._backend = "patchworkpp"
        except Exception:                       # noqa: BLE001
            self._pw = None
            self._backend = "none"

    @property
    def backend(self) -> str:
        """Which backend is active: 'patchworkpp', 'ransac', or 'none'."""
        return self._backend if self._backend != "none" else "ransac"

    # ------------------------------------------------------------------ #
    def remove_ground(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(nonground, ground)`` arrays with the same column count.

        ``points`` is ``(N,3)`` or ``(N,4)``. Empty input yields two empties.
        """
        if points is None or points.shape[0] == 0:
            empty = np.empty((0, points.shape[1] if points is not None else 4),
                             dtype=np.float32)
            return empty, empty.copy()

        if self._pw is not None:
            try:
                return self._patchwork_split(points)
            except Exception:                   # noqa: BLE001 - fall back
                pass
        return self._ransac_split(points)

    def _patchwork_split(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Patchwork++ wants float32/64 (N,4) [x,y,z,intensity]; add a zero
        # intensity column if absent.
        pts = points
        if pts.shape[1] == 3:
            pts = np.column_stack([pts, np.zeros(len(pts), dtype=pts.dtype)])
        pts = np.ascontiguousarray(pts.astype(np.float64))
        self._pw.estimateGround(pts)
        gi = np.asarray(self._pw.getGroundIndices(), dtype=np.int64)
        ngi = np.asarray(self._pw.getNongroundIndices(), dtype=np.int64)
        # Index back into the ORIGINAL points (preserve original columns/dtype).
        ground = points[gi] if gi.size else points[:0]
        nonground = points[ngi] if ngi.size else points[:0]
        return nonground, ground

    def _ransac_split(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        try:
            import open3d as o3d
        except Exception:                       # noqa: BLE001
            # Last-ditch: crude height threshold about the sensor height.
            z = points[:, 2]
            gmask = z < (-self.params.sensor_height + self.params.ransac_dist_thresh)
            return points[~gmask], points[gmask]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(points[:, :3]))
        try:
            _plane, inliers = pcd.segment_plane(
                distance_threshold=self.params.ransac_dist_thresh,
                ransac_n=self.params.ransac_n,
                num_iterations=self.params.ransac_iters,
            )
        except Exception:                       # noqa: BLE001
            z = points[:, 2]
            gmask = z < (-self.params.sensor_height + self.params.ransac_dist_thresh)
            return points[~gmask], points[gmask]
        inliers = np.asarray(inliers, dtype=np.int64)
        mask = np.zeros(len(points), dtype=bool)
        mask[inliers] = True
        return points[~mask], points[mask]
