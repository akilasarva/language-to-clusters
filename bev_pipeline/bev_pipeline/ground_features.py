"""Interpretable GROUND-channel features from LiDAR (the sidewalk-in-grass axis).

Complements the VERTICAL channel (wall enclosure / n_open, from non-ground geom).
The key scalars, computed on the accumulated ground with near-field (<3 m)
EXCLUDED (the near-field plane-fit residual is unreliable):

  pavement_frac   fraction of ground that reads as pavement (bright + smooth)
  arm_count       # of distinct pavement directions on the 5-10 m ring
                  (2 = straight path, 3 = Y/T, 4 = cross -> ground junction)
  on_path_front   is there pavement straight ahead (forward cone, 3-8 m)?
  corridor_w_deg  angular width of the forward pavement arc (narrow path vs plaza)
  open_frac       fraction of ground that is traversable-but-not-pavement (grass)

Pavement = range-detrended intensity high AND roughness low. Intensity is
per-frame relative (so an all-pavement lot reads ~uniform — the ground channel
is only meaningful where grass/pavement contrast exists, i.e. campus lawn).
"""
from __future__ import annotations

import numpy as np

from .ground_bev import gravity_ground, _accumulate

FEAT_NAMES = ["pavement_frac", "arm_count", "coverage", "on_path_front"]


def _otsu(x):
    """1-D Otsu split of a bimodal (grass/pavement) intensity histogram."""
    hist, edges = np.histogram(x, bins=64)
    p = hist / max(hist.sum(), 1)
    w = np.cumsum(p)
    mids = (edges[:-1] + edges[1:]) / 2
    mu = np.cumsum(p * mids)
    tot = mu[-1]
    den = w * (1 - w)
    den[den == 0] = 1e-9
    sig = (tot * w - mu) ** 2 / den
    return mids[int(np.nanargmax(sig))]


def _pavement_points(acc, rin=2.5, rout=10.0):
    """(pavement_xy, all_xy) beyond near-field; pavement = above Otsu intensity.

    Intensity is range-detrended per 1 m radial ring, then split by Otsu — an
    ABSOLUTE (not fixed-percentile) threshold, so open lawn yields little
    pavement and a plaza yields a lot. Near-field (<rin) excluded (residual).
    """
    r = np.hypot(acc[:, 0], acc[:, 1])
    band = acc[(r > rin) & (r < rout)]
    if len(band) < 30:
        return band[:0, :2], band[:, :2]
    ring = np.hypot(band[:, 0], band[:, 1]).astype(int)
    inten = band[:, 3].astype(float).copy()
    for k in np.unique(ring):
        m = ring == k
        inten[m] -= np.median(inten[m])
    pav = band[inten >= _otsu(inten)]
    return pav[:, :2], band[:, :2]


def _arms(pav_xy, nbin=36):
    """(arm_count, coverage) via peaks in the pavement bearing histogram.

    arm_count = # of dominant pavement directions (2=straight, 3=Y/T, 4=cross);
    coverage = fraction of bearings with pavement (wide plaza vs narrow path).
    """
    if len(pav_xy) < 6:
        return 0, 0.0
    ang = (np.degrees(np.arctan2(pav_xy[:, 1], pav_xy[:, 0])) % 360)
    h, _ = np.histogram(ang, bins=nbin, range=(0, 360))
    hs = np.convolve(np.r_[h[-2:], h, h[:2]], np.ones(3) / 3, "same")[2:-2]
    thr = max(hs.max() * 0.35, 3)
    peaks = sum(1 for j in range(nbin)
                if hs[j] >= thr and hs[j] >= hs[(j - 1) % nbin] and hs[j] > hs[(j + 1) % nbin])
    coverage = float((hs >= thr).sum()) / nbin
    return peaks, coverage


def ground_scalars(acc):
    """Ground feature vector [pavement_frac, arm_count, coverage, on_path_front]."""
    if acc is None or len(acc) < 50:
        return np.zeros(len(FEAT_NAMES), np.float32)
    pav_xy, all_xy = _pavement_points(acc)
    pav_frac = len(pav_xy) / max(len(all_xy), 1)
    arms, coverage = _arms(pav_xy)
    r = np.hypot(pav_xy[:, 0], pav_xy[:, 1])
    ang = np.degrees(np.arctan2(pav_xy[:, 1], pav_xy[:, 0]))
    front = pav_xy[(r > 2.5) & (r < 7) & (np.abs(ang) < 25)]
    on_path_front = 1.0 if len(front) >= 4 else 0.0
    return np.array([pav_frac, arms, coverage, on_path_front], dtype=np.float32)


def precompute_ground(env_dir, poses=None):
    """Per-frame gravity-aligned ground list (cache for accumulation)."""
    import os
    if poses is None:
        poses = np.load(os.path.join(env_dir, "poses.npy"))
    n = len(poses)
    out = []
    for i in range(n):
        fp = os.path.join(env_dir, "points", f"frame_{i:05d}.npy")
        p = np.load(fp) if os.path.exists(fp) else None
        out.append(gravity_ground(p, poses[i, 3:7]) if p is not None else None)
    return out, poses


def build_ground_features(env_dir, W=4, R=12.0, verbose=True):
    """Compute + save modality_ground_feats.npy (N, D) for a dataset dir."""
    import os
    ground, poses = precompute_ground(env_dir)
    n = len(poses)
    feats = np.zeros((n, len(FEAT_NAMES)), np.float32)
    for i in range(n):
        acc = _accumulate(ground, poses, i, W, R)
        feats[i] = ground_scalars(acc)
        if verbose and (i + 1) % 100 == 0:
            print(f"  ground-feats {i+1}/{n}", flush=True)
    np.save(os.path.join(env_dir, "modality_ground_feats.npy"), feats)
    if verbose:
        print(f"saved {env_dir}/modality_ground_feats.npy {feats.shape}", flush=True)
    return feats
