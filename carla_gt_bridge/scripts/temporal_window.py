#!/usr/bin/env python3
"""Does a TEMPORAL WINDOW of scans classify better than one scan? Per town, no transfer.

    ~/miniconda3/bin/python scripts/temporal_window.py

The enclosure relations (`past`, `around`) are DEFINED over a trajectory, yet the
classifier has only ever seen a single 72-bin scan. This tests whether a window makes
`along_edge` / `passage` separable without changing the output contract.

METRICS, chosen so a collapsed classifier cannot read as a win: a classifier emitting ONE
CONSTANT CLASS can show a large "gain over majority" when the dummy baseline predicts the
train majority and that class barely exists in the test set. So every row here reports:

  * accuracy against the TEST-set majority (the best constant predictor on that split),
    not the train majority;
  * macro-F1, which a constant predictor cannot inflate;
  * the number of distinct classes actually predicted -- 1 means collapsed, and no
    accuracy number can hide it.

SPLIT. Blocked 70/30 in time within each town. Frames are ~0.4 m apart, so a random split
puts near-identical neighbours on both sides and every score comes out inflated.
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import re
import sys

import numpy as np

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(os.path.dirname(PKG), "clustering"))
sys.path.insert(0, os.path.join(PKG, "scripts"))

from scaling_transfer import TOWNS, stamp_of  # noqa: E402


def windowed(X: np.ndarray, w: int, mode: str) -> np.ndarray:
    """Build a feature per frame from the w frames ending at it.

    `stack`  : the scan now, w//2 ago and w ago, concatenated -- keeps the shape of the
               change, which is what `past` vs `along` actually differ by.
    `meanstd`: per-bin mean and std over the window, plus the current scan. Cheaper and
               order-free; if it wins, the ORDER does not matter and only the spread does.
    """
    n, b = X.shape
    idx = np.arange(n)
    back = lambda k: X[np.maximum(idx - k, 0)]          # noqa: E731
    if w <= 1:
        return X
    if mode == "stack":
        return np.hstack([X, back(w // 2), back(w)])
    if mode == "meanstd":
        acc = np.stack([back(k) for k in range(w)])
        return np.hstack([X, acc.mean(axis=0), acc.std(axis=0)])
    raise ValueError(mode)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--axis", default="enclosure", choices=("enclosure", "topology"))
    ap.add_argument("--towns", nargs="*", default=None)
    ap.add_argument("--windows", type=int, nargs="*", default=[1, 5, 15, 30, 60])
    ap.add_argument("--max-range", type=float, default=25.0)
    ap.add_argument("--min-range", type=float, default=3.0,
                    help="3.0 excludes the ego halo; see frame_labels")
    a = ap.parse_args(argv)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler
    from clustering.cluster_training import LidarDataset

    towns = [t for t in (a.towns or list(TOWNS)) if os.path.isdir(TOWNS[t][0])]
    print(f"axis: {a.axis}   min_range {a.min_range} m (ego halo excluded)\n")

    for t in towns:
        cfg = dict(num_ranges=72, max_lidar_range=a.max_range,
                   min_lidar_range=a.min_range,
                   z_threshold_lower=-1.6, z_threshold_upper=-0.5,
                   z_threshold_lower_2=0.0, z_threshold_upper_2=0.0,
                   use_intensity=False, density_radius=0.30, min_neighbors=4)
        ds = LidarDataset(TOWNS[t][0], cfg)
        X = np.stack([ds[i].numpy() for i in range(len(ds))])
        gt = {int(r["stamp_ns"]): r for r in csv.DictReader(open(TOWNS[t][1]))}
        st = [stamp_of(p) for p in ds.pcd_files]
        keep = [i for i, s in enumerate(st) if s in gt]
        X = X[keep]
        y = np.array([gt[st[i]][a.axis] for i in keep])
        n = len(X)
        cut = int(n * 0.7)
        mix = collections.Counter(y)
        print(f"=== {t}   n={n}   " + "  ".join(
            f"{k} {100 * v / n:.0f}%" for k, v in mix.most_common()))
        print(f"  {'window':>6s} {'mode':>8s} {'test-maj':>9s} {'acc':>7s} "
              f"{'vs maj':>7s} {'macroF1':>8s} {'classes predicted':>18s}")
        for w in a.windows:
            for mode in (["stack"] if w <= 1 else ["stack", "meanstd"]):
                F = windowed(X, w, mode)
                sc = StandardScaler().fit(F[:cut])
                m = LogisticRegression(max_iter=3000).fit(sc.transform(F[:cut]), y[:cut])
                pred = m.predict(sc.transform(F[cut:]))
                yte = y[cut:]
                acc = 100 * (pred == yte).mean()
                temaj = 100 * collections.Counter(yte).most_common(1)[0][1] / len(yte)
                mf1 = 100 * f1_score(yte, pred, average="macro")
                npred = len(set(pred.tolist()))
                flag = "  <- COLLAPSED" if npred == 1 else ""
                print(f"  {w:6d} {mode:>8s} {temaj:8.1f}% {acc:6.1f}% "
                      f"{acc - temaj:+6.1f} {mf1:7.1f}% {npred:12d}{flag}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
