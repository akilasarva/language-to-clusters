#!/usr/bin/env python3
"""Does scaling the label AND the feature together make enclosure transfer between towns?

    ~/miniconda3/bin/python scripts/scaling_transfer.py

Background. A linear probe recovers the `enclosure` label within a town but not across
towns. The towns differ in scale by about 2x: Town01's total road width is ~16.6 m,
Town10HD's ~31.3 m.

Scaling only the LABEL (`NEAR_STRUCTURE_M` -> k x road width) or only the FEATURE
(`max_lidar_range`) is not a fair test, because the two are COUPLED: the label is
"structure within X metres" and the feature is "range / max_range", so changing one
alone guarantees they disagree. This script scales BOTH by the same per-town factor

    f_town = total_road_width(town) / total_road_width(Town01)

so that `enclosed` and `how far things are` are expressed in the same units in every
town, and compares against the uncoupled baseline on every ordered town pair.

Town01 is the reference because its fixed 14 m threshold has been checked against the
live building-bbox labels.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import re
import sys

import numpy as np

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(os.path.dirname(PKG), "clustering"))

#: town -> (pcd dir, fixed-threshold labels, width-scaled labels, xodr stem)
TOWNS = {
    "town01": (os.path.expanduser("~/carla_data/town01_v2/town01_v2_pcds"),
               "reports/frame_labels/town01_v2.csv",
               "reports/frame_labels/town01_v2_ws.csv", "Town01"),
    "town10hd": (os.path.expanduser("~/carla_data/town10hd_v2/town10hd_v2_pcds"),
                 "reports/frame_labels/town10hd_v2.csv",
                 "reports/frame_labels/town10hd_v2_ws.csv", "Town10HD"),
    "town07": (os.path.expanduser("~/carla_data/town07_v2/town07_v2_pcds"),
               "reports/frame_labels/town07_v2.csv",
               "reports/frame_labels/town07_v2_ws.csv", "Town07"),
}
REF = "town01"
BASE_MAX_RANGE = 25.0


def stamp_of(path: str) -> int:
    m = re.match(r"(\d+)-(\d+)\.pcd$", os.path.basename(path))
    return int(m.group(1)) * 10 ** 9 + int(m.group(2))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--towns", nargs="*", default=None)
    ap.add_argument("--axis", default="enclosure", choices=("enclosure", "topology"))
    a = ap.parse_args(argv)

    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from clustering.cluster_training import LidarDataset
    from carla_gt_bridge.opendrive import load as load_xodr

    towns = [t for t in (a.towns or list(TOWNS))
             if os.path.isdir(TOWNS[t][0]) and os.path.exists(TOWNS[t][1])
             and os.path.exists(TOWNS[t][2])]
    missing = [t for t in (a.towns or list(TOWNS)) if t not in towns]
    if missing:
        print(f"skipping (data not ready): {missing}")
    if len(towns) < 2:
        print("need at least two towns")
        return 1

    width = {}
    for t in towns:
        m = load_xodr(os.path.join(PKG, "config", f"{TOWNS[t][3]}.xodr"))
        width[t] = float(np.median([r.total_width for r in m.path_roads]))
    f = {t: width[t] / width[REF] for t in towns}
    print("town        total width   scale f   max_range   label threshold")
    for t in towns:
        print(f"{t:10s} {width[t]:9.1f} m {f[t]:9.2f} {BASE_MAX_RANGE * f[t]:9.1f} m "
              f"{14.0 * f[t]:12.1f} m")

    def feats(t, max_r, labels_csv):
        cfg = dict(num_ranges=72, max_lidar_range=max_r, min_lidar_range=1.0,
                   z_threshold_lower=-1.6, z_threshold_upper=-0.5,
                   z_threshold_lower_2=0.0, z_threshold_upper_2=0.0,
                   use_intensity=False, density_radius=0.30, min_neighbors=4)
        ds = LidarDataset(TOWNS[t][0], cfg)
        X = np.stack([ds[i].numpy() for i in range(len(ds))])
        gt = {int(r["stamp_ns"]): r for r in csv.DictReader(open(labels_csv))}
        st = [stamp_of(p) for p in ds.pcd_files]
        keep = [i for i, s in enumerate(st) if s in gt]
        return X[keep], np.array([gt[st[i]][a.axis] for i in keep])

    def probe(Xtr, ytr, Xte, yte):
        sc = StandardScaler().fit(Xtr)
        d = DummyClassifier(strategy="most_frequent").fit(
            sc.transform(Xtr), ytr).score(sc.transform(Xte), yte)
        m = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), ytr)
        return 100 * d, 100 * m.score(sc.transform(Xte), yte)

    print(f"\naxis: {a.axis}\n")
    conditions = {
        "UNCOUPLED (fixed 14 m label, fixed 25 m feature)":
            {t: (BASE_MAX_RANGE, TOWNS[t][1]) for t in towns},
        "COUPLED (both scaled by f)":
            {t: (BASE_MAX_RANGE * f[t], TOWNS[t][2]) for t in towns},
    }
    for cname, spec in conditions.items():
        data = {t: feats(t, *spec[t]) for t in towns}
        print(f"=== {cname}")
        for t in towns:
            import collections
            mix = collections.Counter(data[t][1])
            n = sum(mix.values())
            print(f"    {t:10s} n={n:6d}  "
                  + "  ".join(f"{k} {100 * v / n:.0f}%" for k, v in sorted(mix.items())))
        rows = []
        for tr, te in itertools.permutations(towns, 2):
            d, l = probe(data[tr][0], data[tr][1], data[te][0], data[te][1])
            rows.append(l - d)
            print(f"    {tr:>9s} -> {te:<9s} majority {d:5.1f}%  linear {l:5.1f}%  "
                  f"gain {l - d:+6.1f}")
        print(f"    mean cross-town gain: {np.mean(rows):+.1f}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
