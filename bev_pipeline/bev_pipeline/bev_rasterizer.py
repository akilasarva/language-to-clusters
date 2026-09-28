"""4-channel robocentric BEV raster construction.

Rasterizes a gravity-leveled (ideally submap-accumulated) cloud into a fixed
``(4, H, W)`` top-down image centered on the robot, +X = forward, +Y = left:

  ch0  max height        — tallest return per cell (structure silhouette)
  ch1  height spread      — (max - min) z per cell (verticality: walls, curbs)
  ch2  log density        — log1p(point count) per cell (occupancy strength)
  ch3  mean intensity      — reflectivity per cell (lane paint, materials)

This replaces the legacy single planar z-slice: height information (curbs,
railings, overpasses) is preserved across the channels rather than thrown away.
Empty cells are zero; the output is always finite.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BevParams:
    extent: float = 32.0      # metres from robot to raster edge (half-width)
    size: int = 128           # output is size x size cells
    z_min: float = -3.0       # clip range for height channels
    z_max: float = 8.0
    n_height_bands: int = 3   # extra per-height occupancy channels (multiband)


def rasterize(points: np.ndarray, params: BevParams = None) -> np.ndarray:
    """Build a ``(4, size, size)`` float32 BEV raster from ``(N,>=3)`` points.

    Column 4 (intensity) is used for ch3 if present, else ch3 is zero. Points
    outside ``[-extent, extent]`` in x or y are dropped. Always returns a finite
    array (zeros where no points fall).
    """
    params = params or BevParams()
    S, E = params.size, params.extent
    out = np.zeros((4, S, S), dtype=np.float32)
    if points is None or points.shape[0] == 0:
        return out

    xyz = points[:, :3].astype(np.float64)
    inten = (points[:, 3].astype(np.float64) if points.shape[1] > 3
             else np.zeros(len(xyz)))
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    # in-bounds mask
    m = (x >= -E) & (x < E) & (y >= -E) & (y < E) & np.isfinite(z)
    if not np.any(m):
        return out
    x, y, z, inten = x[m], y[m], z[m], inten[m]
    z = np.clip(z, params.z_min, params.z_max)

    # map metric -> cell index. +X forward -> row 0 at far forward (image "up").
    res = (2 * E) / S
    col = ((y + E) / res).astype(np.int64)           # +Y left -> increasing col
    row = ((E - x) / res).astype(np.int64)           # +X forward -> row up
    np.clip(col, 0, S - 1, out=col)
    np.clip(row, 0, S - 1, out=row)
    flat = row * S + col
    ncells = S * S

    # counts
    counts = np.bincount(flat, minlength=ncells).astype(np.float64)
    # max height per cell
    maxz = np.full(ncells, params.z_min, dtype=np.float64)
    np.maximum.at(maxz, flat, z)
    # min height per cell
    minz = np.full(ncells, params.z_max, dtype=np.float64)
    np.minimum.at(minz, flat, z)
    # sum intensity per cell (for mean)
    sum_i = np.bincount(flat, weights=inten, minlength=ncells)

    occupied = counts > 0
    height = np.zeros(ncells)
    spread = np.zeros(ncells)
    mean_i = np.zeros(ncells)
    height[occupied] = maxz[occupied]
    spread[occupied] = maxz[occupied] - minz[occupied]
    mean_i[occupied] = sum_i[occupied] / counts[occupied]
    dens = np.log1p(counts)

    out[0] = height.reshape(S, S).astype(np.float32)
    out[1] = spread.reshape(S, S).astype(np.float32)
    out[2] = dens.reshape(S, S).astype(np.float32)
    out[3] = mean_i.reshape(S, S).astype(np.float32)
    return out


def rasterize_multiband(points: np.ndarray, params: BevParams = None,
                        band_lo: float = 0.0, band_hi: float = 6.0) -> np.ndarray:
    """4 base channels + N per-height-band occupancy channels -> (4+N, S, S).

    The band channels are log1p point-count in each height slice over
    ``[band_lo, band_hi]`` (empty cell -> 0 in every band, so empty space is
    naturally black — fixing the flat-red-background artifact of the raw
    height channel). Used by the bev_dino "height_bands" projection and a
    richer bev_cnn input. Vertical structure is thereby encoded as color.
    """
    params = params or BevParams()
    base = rasterize(points, params)
    S, E, nb = params.size, params.extent, params.n_height_bands
    bands = np.zeros((nb, S, S), dtype=np.float32)
    if points is not None and points.shape[0] > 0 and nb > 0:
        xyz = points[:, :3].astype(np.float64)
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        m = (x >= -E) & (x < E) & (y >= -E) & (y < E) & (z >= band_lo) & (z < band_hi) \
            & np.isfinite(z)
        if np.any(m):
            x, y, z = x[m], y[m], z[m]
            res = (2 * E) / S
            col = np.clip(((y + E) / res).astype(np.int64), 0, S - 1)
            row = np.clip(((E - x) / res).astype(np.int64), 0, S - 1)
            bw = (band_hi - band_lo) / nb
            bi = np.clip(((z - band_lo) / bw).astype(np.int64), 0, nb - 1)
            for k in range(nb):
                sel = bi == k
                if np.any(sel):
                    flat = row[sel] * S + col[sel]
                    counts = np.bincount(flat, minlength=S * S).astype(np.float64)
                    bands[k] = np.log1p(counts).reshape(S, S).astype(np.float32)
    return np.concatenate([base, bands], axis=0)
