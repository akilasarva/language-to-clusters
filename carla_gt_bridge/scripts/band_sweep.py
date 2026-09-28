#!/usr/bin/env python3
"""Score every z-band this repo has used, per town, per axis, per ground-normalisation.

    ~/miniconda3/bin/python scripts/band_sweep.py

The band explorer shows which band LOOKS structured. This says which one is separable,
which is a different question.

BANDS ARE DEFINED AS HEIGHTS ABOVE THE GROUND, not as sensor-frame z. The CARLA rig's
modal return is at z = -1.99 m and the real `ground_lidar`'s is at -0.38 m, so the same
z-band is a different physical slice on each. Each band below carries the legacy
sensor-frame numbers it came from and the corpus they were written against.

READ THE CONVERSION CAREFULLY. `road-level`, `ground-only`, `low-wide` and `above-road`
were written against CARLA, so on CARLA they map back to their literal legacy z.
`training-abs` and `live-node` were written against `ground_lidar`, whose ground sits
1.61 m higher in sensor frame -- so on CARLA they select a DIFFERENT z band than the
legacy numbers did. Applying the legacy numbers to CARLA literally would select an
unintended physical slice, the same mistake as serving weights fitted under one band
through a node configured with another.

WHAT NORMALISATION CANNOT DO: the per-frame ground offset is nearly constant within each
corpus (plane tilt under ~1 deg), so `offset` and `plane` are expected to score about the
same as `none` within a town. Their value is portability across sensors.

One pass over the PCDs builds the scan under every (band, ground) pair at once. The
binner is a vectorised rewrite of `lidar_processor.get_ranges_from_points`, checked
against the original on the first 30 frames of every town under `--ground none`; the run
aborts on any disagreement, because a lookalike feature is worse than no number at all.

Protocol is `temporal_window.py`'s: blocked 70/30 split in time, accuracy against the
TEST-set majority, macro-F1 alongside, and the count of distinct predicted classes so a
collapsed classifier cannot read as a win.

Every axis is reported twice, all frames and moving only (>= 0.1 m/s), from one pass of
features. Some labels are dominated by stationary frames (Town07 `exit` in particular); a
parked vehicle has per-bin std ~= 0 under the `meanstd` window, so such a label is
reachable just by detecting that nothing is changing.
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import sys

import numpy as np

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(os.path.dirname(PKG), "clustering"))
sys.path.insert(0, os.path.join(PKG, "scripts"))

from carla_gt_bridge.ground_plane import MODES, heights   # noqa: E402
from scaling_transfer import TOWNS, stamp_of              # noqa: E402
from temporal_window import windowed                      # noqa: E402
from viz_cluster_panel import read_pcd                    # noqa: E402

# Modal ground return per corpus (sensor-frame z). This is what converts a legacy
# sensor-frame band into a height, and what `--ground none` converts back with.
G_REF = {"town01": -1.99, "town10hd": -1.99, "town07": -1.99, "ground_lidar": -0.38}

# name, (h_lo, h_hi) ABOVE GROUND, mirrored, (r_min, r_max), legacy source
BANDS = [
    ("road-level",   (0.39, 1.49),  False, (3.0, 25.0),
     "z -1.6..-0.5 on CARLA; frame_labels.py"),
    ("training-abs", (0.51, 1.81),  True,  (1.0, 25.0),
     "z 0.1..1.4 abs on ground_lidar; cluster_training.py:564"),
    ("live-node",    (-0.09, 0.56), False, (0.5, 8.0),
     "z -0.5..0.15 on ground_lidar; live_cluster_inference_node.py:99"),
    ("ground-only",  (-0.21, 0.29), False, (3.0, 25.0),
     "the modal return itself"),
    ("low-wide",     (-0.21, 1.79), False, (3.0, 25.0),
     "road surface + curb + low wall"),
    ("above-road",   (1.49, 3.49),  False, (3.0, 25.0),
     "everything above the road plane"),
]
NB = 72
STEP = 2 * np.pi / NB


def cfg_of(band, town, ground):
    """Band in the coordinate the filter will actually compare against.

    `none` keeps sensor-frame z, so the height band is shifted back by the corpus ground;
    `offset`/`plane` re-centre each frame onto that same nominal ground, so the numbers
    are identical and only the per-frame deviation differs. The mirror is a sensor-frame
    operation (`abs(z)`), so it is applied in z and converted back.
    """
    _, (hl, hh), mir, (rl, rh), _ = band
    g = G_REF[town]
    zl, zh = hl + g, hh + g
    z2l, z2h = (-zh, -zl) if mir else (0.0, 0.0)
    return dict(num_ranges=NB, max_lidar_range=rh, min_lidar_range=rl,
                z_threshold_lower=zl, z_threshold_upper=zh,
                z_threshold_lower_2=z2l, z_threshold_upper_2=z2h,
                use_intensity=False, density_radius=0.30, min_neighbors=4,
                ground=ground, g_ref=g)


def ranges_fast(pts, cfg):
    """Vectorised twin of get_ranges_from_points, on ground-normalised heights."""
    z = heights(pts, cfg.get("ground", "none"), g_ref=cfg.get("g_ref", 0.0))
    m = ((z >= cfg["z_threshold_lower"]) & (z <= cfg["z_threshold_upper"])
         | (z >= cfg["z_threshold_lower_2"]) & (z <= cfg["z_threshold_upper_2"]))
    p = pts[m]
    if not len(p):
        return np.full(NB, cfg["max_lidar_range"])
    d = np.hypot(p[:, 0], p[:, 1])
    m = (d >= cfg["min_lidar_range"]) & (d <= cfg["max_lidar_range"])
    p, d = p[m], d[m]
    if not len(p):
        return np.full(NB, cfg["max_lidar_range"])
    mn = int(cfg["min_neighbors"])
    if len(p) > mn:
        from scipy.spatial import cKDTree
        c = cKDTree(p[:, :2]).query_ball_point(p[:, :2], r=cfg["density_radius"],
                                               return_length=True)
        keep = (c - 1) >= mn
        p, d = p[keep], d[keep]
    if not len(p):
        return np.full(NB, cfg["max_lidar_range"])
    idx = np.mod(np.round(np.arctan2(p[:, 1], p[:, 0]) / STEP).astype(int), NB)
    out = np.full(NB, float(cfg["max_lidar_range"]))
    np.minimum.at(out, idx, d)
    return out


def verify(files, town, n=30):
    """`--ground none` must equal the shipped implementation exactly."""
    from clustering.lidar_processor import get_ranges_from_points
    for band in BANDS:
        cfg = cfg_of(band, town, "none")
        for f in files[:n]:
            pts = read_pcd(f)
            a, b = ranges_fast(pts, cfg), get_ranges_from_points(pts, cfg)
            if not np.allclose(a, b, atol=1e-6):
                raise SystemExit(f"binner mismatch on {band[0]} / {os.path.basename(f)}: "
                                 f"max |d| = {np.abs(a - b).max():.4f}")
    return len(BANDS) * n


def score(F, y, cut):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(F[:cut])
    m = LogisticRegression(max_iter=3000).fit(sc.transform(F[:cut]), y[:cut])
    pred = m.predict(sc.transform(F[cut:]))
    yte = y[cut:]
    return (100 * (pred == yte).mean(),
            100 * f1_score(yte, pred, average="macro"),
            len(set(pred.tolist())))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--towns", nargs="*", default=None)
    ap.add_argument("--ground", nargs="*", default=list(MODES), choices=list(MODES))
    ap.add_argument("--window", type=int, default=15)
    a = ap.parse_args(argv)

    from clustering.cluster_training import LidarDataset
    towns = [t for t in (a.towns or list(TOWNS)) if os.path.isdir(TOWNS[t][0])]

    for t in towns:
        ds = LidarDataset(TOWNS[t][0], cfg_of(BANDS[0], t, "none"))
        gt = {int(r["stamp_ns"]): r for r in csv.DictReader(open(TOWNS[t][1]))}
        files = [f for f in ds.pcd_files if stamp_of(f) in gt]
        moving = np.array([float(gt[stamp_of(f)]["speed"]) >= 0.1 for f in files])
        print(f"=== {t}: {len(files)} labelled frames, {int(moving.sum())} moving "
              f"(ground reference z = {G_REF[t]} m)", flush=True)
        print(f"    binner agrees with lidar_processor on "
              f"{verify(files, t)} (band, frame) pairs under --ground none", flush=True)

        pairs = [(b, g) for g in a.ground for b in BANDS]
        cfgs = [cfg_of(b, t, g) for b, g in pairs]
        X = np.empty((len(pairs), len(files), NB), dtype=np.float32)
        for i, f in enumerate(files):
            pts = read_pcd(f)
            for k, c in enumerate(cfgs):
                X[k, i] = ranges_fast(pts, c)
            if i % 1000 == 0:
                print(f"    {i}/{len(files)}", flush=True)

        for axis in ("topology", "enclosure"):
          for tag, sel in (("all frames", np.ones(len(files), bool)),
                           ("moving only", moving)):
            y = np.array([gt[stamp_of(f)][axis] for f in files])[sel]
            if len(set(y)) < 2:
                continue
            cut = int(len(y) * 0.7)
            maj = 100 * collections.Counter(y[cut:]).most_common(1)[0][1] / len(y[cut:])
            print(f"\n  -- {axis}, {tag}  (n={len(y)}, test-majority {maj:.1f}%)")
            print(f"     {'band':<13}{'height above ground':<21}{'gate':<11}"
                  + "".join(f"{g:>17}" for g in a.ground))
            for bi, b in enumerate(BANDS):
                hs = f"{b[1][0]:+.2f}..{b[1][1]:+.2f} m" + (" abs" if b[2] else "")
                cells = []
                for gi, g in enumerate(a.ground):
                    F = windowed(X[gi * len(BANDS) + bi][sel], a.window, "meanstd")
                    acc, mf1, npred = score(F, y, cut)
                    cells.append(f"{mf1:11.1f}%{'*' if npred == 1 else ' '}    ")
                print(f"     {b[0]:<13}{hs:<21}{b[3][0]}-{b[3][1]:<7}" + "".join(cells))
            print("     (macro-F1; * = collapsed to one class)")
        print(flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
