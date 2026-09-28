#!/usr/bin/env python3
"""Train a SUPERVISED enclosure-mode classifier and save it for the live node.

    ~/miniconda3/bin/python scripts/train_mode_classifier.py --town town01

This replaces HDBSCAN-over-an-autoencoder for the one axis LiDAR can actually carry.
Nothing downstream changes: the live node still publishes an Int16 on
`/predicted_cluster`, still against the same `cluster_map.<env>.yaml`, still matched by
`taxonomy.accept_clusters`. Only how the id is DERIVED changes -- classification against
ground truth instead of unsupervised grouping.

WHY SUPERVISED. On identical features a supervised probe beats the clustering by a wide
margin, because HDBSCAN asks "do these features form natural groups?" and nothing in a
reconstruction objective makes those groups line up with `open_space` / `along_edge` /
`passage`. The ground truth to train against is free (`label_frames.py`), which is what
makes this cheap.

WHAT IT DOES NOT DO. It does not predict TOPOLOGY (`path`/`approach`/`junction`/`exit`).
That axis is not per-frame perceivable -- a per-frame classifier does not beat the
majority baseline, from the raw scan or the embedding -- so a classifier for it would be
a confident random number. Topology stays plan-executed and path-confirmed.

FEATURES: 72-bin scan at `min_range 3.0` (the ego body sits at 1.0-1.5 m and would pin
many bins otherwise), plus the per-bin mean and std over a 15-frame window. The window
improves macro-F1, and `meanstd` beats a stacked window -- the spread carries the signal,
not the ordering.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import pickle
import sys

import numpy as np

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(os.path.dirname(PKG), "clustering"))
sys.path.insert(0, os.path.join(PKG, "scripts"))

from scaling_transfer import TOWNS, stamp_of          # noqa: E402
from temporal_window import windowed                  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--town", default="town01", choices=list(TOWNS))
    ap.add_argument("--window", type=int, default=15)
    ap.add_argument("--min-range", type=float, default=3.0)
    ap.add_argument("--max-range", type=float, default=25.0)
    ap.add_argument("--out-dir", default="")
    a = ap.parse_args(argv)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    from sklearn.preprocessing import StandardScaler
    from clustering.cluster_training import LidarDataset

    cfg = dict(num_ranges=72, max_lidar_range=a.max_range,
               min_lidar_range=a.min_range,
               z_threshold_lower=-1.6, z_threshold_upper=-0.5,
               z_threshold_lower_2=0.0, z_threshold_upper_2=0.0,
               use_intensity=False, density_radius=0.30, min_neighbors=4)
    ds = LidarDataset(TOWNS[a.town][0], cfg)
    X = np.stack([ds[i].numpy() for i in range(len(ds))])
    gt = {int(r["stamp_ns"]): r for r in csv.DictReader(open(TOWNS[a.town][1]))}
    st = [stamp_of(p) for p in ds.pcd_files]
    keep = [i for i, s in enumerate(st) if s in gt]
    X, y = X[keep], np.array([gt[st[i]]["enclosure"] for i in keep])
    F = windowed(X, a.window, "meanstd")
    n = len(F)
    cut = int(n * 0.7)
    print(f"{a.town}: {n} frames, feature dim {F.shape[1]} "
          f"(72-bin scan + mean + std over {a.window} frames)")
    print(f"  labels: {dict(collections.Counter(y))}")

    sc = StandardScaler().fit(F[:cut])
    best = None
    for name, mk in (("logreg", lambda: LogisticRegression(max_iter=3000)),
                     ("logreg-balanced",
                      lambda: LogisticRegression(max_iter=3000,
                                                 class_weight="balanced")),
                     ("rf", lambda: RandomForestClassifier(
                         n_estimators=300, min_samples_leaf=5, n_jobs=-1,
                         random_state=0)),
                     ("rf-balanced", lambda: RandomForestClassifier(
                         n_estimators=300, min_samples_leaf=5, n_jobs=-1,
                         random_state=0, class_weight="balanced"))):
        m = mk().fit(sc.transform(F[:cut]), y[:cut])
        pred = m.predict(sc.transform(F[cut:]))
        yte = y[cut:]
        acc = 100 * (pred == yte).mean()
        maj = 100 * collections.Counter(yte).most_common(1)[0][1] / len(yte)
        mf1 = 100 * f1_score(yte, pred, average="macro")
        print(f"  {name:16s} acc {acc:5.1f}% (test-majority {maj:5.1f}%, "
              f"{acc - maj:+5.1f})  macroF1 {mf1:5.1f}%  "
              f"classes predicted {len(set(pred.tolist()))}")
        if best is None or mf1 > best[0]:
            best = (mf1, name, m, pred, yte)

    mf1, name, model, pred, yte = best
    print(f"\nbest: {name}  macroF1 {mf1:.1f}%")
    labs = sorted(set(yte.tolist()) | set(pred.tolist()))
    print("\nconfusion (rows = truth, cols = predicted): " + "  ".join(labs))
    for row, lab in zip(confusion_matrix(yte, pred, labels=labs), labs):
        print(f"  {lab:12s} " + "  ".join(f"{v:6d}" for v in row))
    print()
    print(classification_report(yte, pred, digits=3, zero_division=0))

    out = a.out_dir or os.path.join(
        os.path.dirname(PKG), "clustering", "clustering", "encoder_weights",
        f"modeclf_{a.town}")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "mode_classifier.pkl"), "wb") as f:
        pickle.dump({"scaler": sc, "model": model, "classes": list(model.classes_)}, f)
    cfg_out = dict(cfg, window=a.window, window_mode="meanstd", axis="enclosure",
                   town=a.town, kind=name, macro_f1_holdout=mf1 / 100.0,
                   written_by="train_mode_classifier.py")
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg_out, f, indent=2)
    print(f"saved -> {out}")
    print("  the live node needs: this config, a 15-frame scan buffer, and a "
          "mode -> cluster-id lookup from the town's cluster_map yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
