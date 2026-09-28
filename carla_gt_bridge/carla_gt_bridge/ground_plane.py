#!/usr/bin/env python3
"""Per-frame ground estimation, so a z-band means a HEIGHT rather than a sensor reading.

WHY THIS EXISTS. The CARLA rig sits 2.00 m up and its modal return is at z = -1.99 m;
the real `ground_lidar` sits ~0.4 m up and its modal return is at z = -0.38 m. A band
written as `z in [-1.6, -0.5]` therefore selects 0.39-1.49 m above the road on one sensor
and nothing at all on the other, so a band that looks sensible in sensor frame does not
transfer across sensors. Expressed as a height above the ground the two bands the repo
uses are almost the same physical slice:

    road-level  z -1.60..-0.50 on CARLA        -> ground + 0.39 .. 1.49 m
    training    z  0.10.. 1.40 on ground_lidar -> ground + 0.51 .. 1.81 m

WHAT IT DOES NOT BUY. The per-frame ground offset is nearly constant WITHIN a corpus
(centimetre-scale std, fitted plane tilt typically under 1 deg), so normalising removes
only a few centimetres of jitter from a band a metre tall. Its value is portability
ACROSS sensors.

MODES
  none   : h = z. The legacy behaviour, kept for comparison.
  offset : h = z - g_frame, a scalar per frame. Removes mount height, suspension and
           slope, keeps any pitch.
  plane  : h = z - (a*x + b*y + c), an iteratively-reweighted least-squares fit to the
           returns near the modal height. Also removes pitch and roll.
"""
from __future__ import annotations

import numpy as np

R_LO, R_HI = 3.0, 20.0     # the annulus the ground is estimated from: past the ego body,
                           # inside the range where a 32-beam sweep still resolves it
BIN_M = 0.05
INLIER_M = 0.35
MODES = ("none", "offset", "plane")


def _annulus(points: np.ndarray) -> np.ndarray:
    r = np.hypot(points[:, 0], points[:, 1])
    return points[(r >= R_LO) & (r <= R_HI)]


def ground_offset(points: np.ndarray) -> float:
    """Modal return height in the annulus. NaN when there is not enough to judge.

    The MODE, not a percentile: on a 32-beam sweep staring downwards the road is the
    single densest height by a wide margin, while a percentile moves with how much
    building happens to be in view.
    """
    q = _annulus(points)
    if len(q) < 200:
        return float("nan")
    z = q[:, 2]
    if not np.isfinite(z).all():
        z = z[np.isfinite(z)]
        if len(z) < 200:
            return float("nan")
    edges = np.arange(z.min(), z.max() + BIN_M, BIN_M)
    if len(edges) < 2:
        return float(z.mean())
    h, e = np.histogram(z, bins=edges)
    k = int(h.argmax())
    return float((e[k] + e[k + 1]) / 2)


def ground_plane(points: np.ndarray, iters: int = 3) -> tuple[float, float, float]:
    """(a, b, c) for z = a*x + b*y + c. Falls back to a flat plane at the modal height."""
    g = ground_offset(points)
    if not np.isfinite(g):
        return (0.0, 0.0, float("nan"))
    q = _annulus(points)
    q = q[np.abs(q[:, 2] - g) < INLIER_M]
    if len(q) < 100:
        return (0.0, 0.0, g)
    A = np.c_[q[:, 0], q[:, 1], np.ones(len(q))]
    c = np.array([0.0, 0.0, g])
    for _ in range(iters):
        c, *_ = np.linalg.lstsq(A, q[:, 2], rcond=None)
        res = q[:, 2] - A @ c
        keep = np.abs(res) < max(0.08, 2 * res.std())
        if keep.sum() < 50:
            break
        q, A = q[keep], A[keep]
    return (float(c[0]), float(c[1]), float(c[2]))


def heights(points: np.ndarray, mode: str = "offset",
            *, g_ref: float = 0.0) -> np.ndarray:
    """Per-point height for band filtering, in the same units the band is written in.

    `g_ref` re-centres the result onto a corpus-nominal ground, so a band expressed in
    sensor-frame z keeps its literal numbers and only the PER-FRAME deviation is removed.
    Pass `g_ref=0.0` to get a true height above ground.
    """
    if mode == "none":
        return points[:, 2]
    if mode == "offset":
        g = ground_offset(points)
        if not np.isfinite(g):
            g = g_ref
        return points[:, 2] - g + g_ref
    if mode == "plane":
        a, b, c = ground_plane(points)
        if not np.isfinite(c):
            return points[:, 2]
        return points[:, 2] - (a * points[:, 0] + b * points[:, 1] + c) + g_ref
    raise ValueError(mode)
