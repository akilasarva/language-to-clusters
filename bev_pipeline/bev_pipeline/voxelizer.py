"""Coarse 3D voxel grid construction (feeds the vol3d_cnn candidate only).

Unlike the BEV rasterizer, which flattens height into channels, this keeps a
genuine 3D grid so a Conv3d model can learn vertical structure directly. The
resolution is deliberately COARSE (few z bins, modest x/y) to control the
compute/data cost of 3D convolutions — this is the heaviest candidate and the
one most in tension with the "small data, minimize training" constraint.

Output is ``(C, GX, GY, GZ)`` float32 with two channels:
  ch0  log density   — log1p(point count) per voxel
  ch1  mean intensity — reflectivity per voxel
Empty voxels are zero; the output is always finite.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VoxelParams:
    extent: float = 32.0     # metres half-width in x and y
    gx: int = 64
    gy: int = 64
    gz: int = 8              # coarse height bins
    z_min: float = -3.0
    z_max: float = 8.0


def voxelize(points: np.ndarray, params: VoxelParams = None) -> np.ndarray:
    """Build a ``(2, gx, gy, gz)`` float32 voxel grid from ``(N,>=3)`` points."""
    params = params or VoxelParams()
    gx, gy, gz, E = params.gx, params.gy, params.gz, params.extent
    out = np.zeros((2, gx, gy, gz), dtype=np.float32)
    if points is None or points.shape[0] == 0:
        return out

    xyz = points[:, :3].astype(np.float64)
    inten = (points[:, 3].astype(np.float64) if points.shape[1] > 3
             else np.zeros(len(xyz)))
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    m = ((x >= -E) & (x < E) & (y >= -E) & (y < E)
         & (z >= params.z_min) & (z < params.z_max) & np.isfinite(z))
    if not np.any(m):
        return out
    x, y, z, inten = x[m], y[m], z[m], inten[m]

    res = (2 * E) / gx
    res_y = (2 * E) / gy
    zres = (params.z_max - params.z_min) / gz
    ix = np.clip(((E - x) / res).astype(np.int64), 0, gx - 1)     # forward -> low idx
    iy = np.clip(((y + E) / res_y).astype(np.int64), 0, gy - 1)
    iz = np.clip(((z - params.z_min) / zres).astype(np.int64), 0, gz - 1)

    flat = (ix * gy + iy) * gz + iz
    ncells = gx * gy * gz
    counts = np.bincount(flat, minlength=ncells).astype(np.float64)
    sum_i = np.bincount(flat, weights=inten, minlength=ncells)

    dens = np.log1p(counts)
    mean_i = np.zeros(ncells)
    occ = counts > 0
    mean_i[occ] = sum_i[occ] / counts[occ]

    out[0] = dens.reshape(gx, gy, gz).astype(np.float32)
    out[1] = mean_i.reshape(gx, gy, gz).astype(np.float32)
    return out
