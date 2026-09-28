"""Ground-plane BEV modality: sidewalk-graph structure the main pipeline drops.

In open pedestrian areas the navigable structure is sidewalk-in-grass paths and
junctions defined on the GROUND plane, not by vertical walls. The main pipeline
removes the ground, so it is blind to this. This module builds a robocentric
top-down raster of the accumulated ground with three channels that separate
pavement from grass:

  ch0 intensity  — range-detrended LiDAR return strength (pavement bright)
  ch1 roughness  — std of plane-residual height per cell (grass scatters)
  ch2 density    — log occupancy (coverage / confidence)

Ground is accumulated over a +/-W frame window, reprojected via *relative*
odometry exactly like :class:`submap.SubmapAccumulator` does for non-ground
(gravity-leveled, yaw+translation only). Single frames are too sparse; the
accumulation is what makes the sidewalk graph legible.
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from .frame_geometry import (gravity_align, invert_transform, quat_to_euler,
                             transform_points)
from .submap import _leveled_frame_matrix


def gravity_ground(points: np.ndarray, quat: np.ndarray, dist: float = 0.25
                   ) -> Optional[np.ndarray]:
    """Gravity-align then return the RANSAC ground inliers (xyz + intensity)."""
    if points is None or len(points) < 200 or quat is None \
            or not np.all(np.isfinite(quat)):
        return None
    lev = gravity_align(points, quat)
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(lev[:, :3]))
        _m, inl = pcd.segment_plane(dist, 3, 200)
    except Exception:                              # noqa: BLE001
        return None
    g = np.zeros(len(lev), bool)
    g[np.asarray(inl, np.int64)] = True
    out = lev[g]
    if out.shape[1] == 3:                          # ensure an intensity column
        out = np.column_stack([out, np.zeros(len(out), out.dtype)])
    return out


def _accumulate(ground_by_frame: List[Optional[np.ndarray]], poses: np.ndarray,
                fi: int, W: int, R: float) -> Optional[np.ndarray]:
    """Reproject ground from frames [fi-W, fi+W] into frame fi (robocentric)."""
    n = len(poses)
    pos_ref, q_ref = poses[fi, :3], poses[fi, 3:7]
    if not np.all(np.isfinite(np.r_[pos_ref, q_ref])):
        return None
    T_L_world = invert_transform(_leveled_frame_matrix(pos_ref, quat_to_euler(q_ref)[2]))
    chunks = []
    for i in range(max(0, fi - W), min(n, fi + W + 1)):
        g = ground_by_frame[i]
        if g is None:
            continue
        pos_i, q_i = poses[i, :3], poses[i, 3:7]
        if not np.all(np.isfinite(np.r_[pos_i, q_i])):
            continue
        T_L_i = T_L_world @ _leveled_frame_matrix(pos_i, quat_to_euler(q_i)[2])
        xyz = transform_points(g[:, :3], T_L_i)
        chunks.append(np.column_stack([xyz, g[:, 3]]))
    if not chunks:
        return None
    acc = np.vstack(chunks)
    m = (np.abs(acc[:, 0]) < R) & (np.abs(acc[:, 1]) < R)
    return acc[m] if m.any() else None


def _detrend_plane(acc: np.ndarray) -> np.ndarray:
    """Residual height after subtracting the dominant plane z = ax+by+c."""
    A = np.column_stack([acc[:, 0], acc[:, 1], np.ones(len(acc))])
    coef, *_ = np.linalg.lstsq(A, acc[:, 2], rcond=None)
    return acc[:, 2] - A @ coef


def rasterize_ground_bev(acc: Optional[np.ndarray], R: float = 14.0,
                         cell: float = 0.25) -> np.ndarray:
    """(3, H, W) ground raster: intensity / roughness / density, each in [0,1]."""
    ng = int(round(2 * R / cell))
    out = np.zeros((3, ng, ng), np.float32)
    if acc is None or len(acc) < 10:
        return out
    resid = _detrend_plane(acc)
    ix = np.clip(((acc[:, 1] + R) / cell).astype(int), 0, ng - 1)   # +y -> left col
    iy = np.clip(((acc[:, 0] + R) / cell).astype(int), 0, ng - 1)   # +x -> forward row
    # range-detrend intensity: subtract per-1m radial-ring median
    ring = np.hypot(acc[:, 0], acc[:, 1]).astype(int)
    inten = acc[:, 3].astype(np.float64).copy()
    for k in np.unique(ring):
        mm = ring == k
        inten[mm] -= np.median(inten[mm])
    cellkey = iy * ng + ix
    order = np.argsort(cellkey, kind="stable")
    ck = cellkey[order]
    bounds = np.searchsorted(ck, np.arange(ng * ng + 1))
    im = np.full(ng * ng, np.nan)
    rg = np.full(ng * ng, np.nan)
    dn = np.zeros(ng * ng)
    ino, rso = inten[order], resid[order]
    for c in range(ng * ng):
        a, b = bounds[c], bounds[c + 1]
        if b <= a:
            continue
        dn[c] = b - a
        im[c] = np.median(ino[a:b])
        if b - a >= 3:
            rg[c] = rso[a:b].std()
    im = im.reshape(ng, ng); rg = rg.reshape(ng, ng); dn = dn.reshape(ng, ng)
    # normalize each channel to [0,1]; empty stays 0
    fin = np.isfinite(im)
    if fin.any():
        lo, hi = np.percentile(im[fin], [5, 95])
        out[0][fin] = np.clip((im[fin] - lo) / (hi - lo + 1e-6), 0, 1)
    finr = np.isfinite(rg)
    out[1][finr] = np.clip(rg[finr] / 0.12, 0, 1)
    if dn.max() > 0:
        out[2] = (np.log1p(dn) / np.log1p(dn.max())).astype(np.float32)
    return out


def build_ground_bev(env_dir: str, W: int = 4, R: float = 14.0,
                     cell: float = 0.25, verbose: bool = True) -> np.ndarray:
    """Build + save ``modality_ground_bev.npy`` (N, 3, H, W) for a dataset dir."""
    poses = np.load(os.path.join(env_dir, "poses.npy"))
    n = len(poses)
    # precompute per-frame gravity-aligned ground once (avoid redundant RANSAC)
    ground: List[Optional[np.ndarray]] = []
    for i in range(n):
        fp = os.path.join(env_dir, "points", f"frame_{i:05d}.npy")
        p = np.load(fp) if os.path.exists(fp) else None
        ground.append(gravity_ground(p, poses[i, 3:7]) if p is not None else None)
        if verbose and (i + 1) % 100 == 0:
            print(f"  ground split {i+1}/{n}", flush=True)
    rasters = np.zeros((n, 3, int(round(2 * R / cell)), int(round(2 * R / cell))),
                       np.float32)
    for i in range(n):
        rasters[i] = rasterize_ground_bev(_accumulate(ground, poses, i, W, R), R, cell)
        if verbose and (i + 1) % 100 == 0:
            print(f"  ground raster {i+1}/{n}", flush=True)
    np.save(os.path.join(env_dir, "modality_ground_bev.npy"), rasters)
    if verbose:
        print(f"saved {env_dir}/modality_ground_bev.npy {rasters.shape}", flush=True)
    return rasters
