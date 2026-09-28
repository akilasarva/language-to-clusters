#!/usr/bin/env python3
"""Train the phase classifier on TOWN05 -- the town the driving actually happens in.

    ~/miniconda3/bin/python scripts/train_phase_town05.py

WHY TOWN05. The driving runs happen in Town05, so the phase classifier has to be evaluated
on the town it will run in, not only on Town01, Town07 or Town10HD.

THE TRANSFER TEST. `town01_dual_newfov` and `town05_dual_newfov` were recorded with the
IDENTICAL rig (-12..+12, 32 ch, same spawn point), so training on one and testing on the
other varies the TOWN and nothing else. Both directions are reported, because transfer is
not symmetric and a single direction can flatter.

Three decoders per arm, so the sequence quality is visible and not just the frame accuracy:
per-frame argmax, the constrained HMM at scale 0.25, and ground truth for the run counts.
"""
from __future__ import annotations

import argparse
import collections
import csv
import itertools
import os
import sys

import numpy as np

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(os.path.dirname(PKG), "clustering"))
sys.path.insert(0, os.path.join(PKG, "scripts"))

from band_sweep import BANDS, cfg_of, ranges_fast                      # noqa: E402
from clean_mask import clean_mask                                     # noqa: E402
from phase_hmm import EPS, fit_transitions, order_score, runs_of, viterbi  # noqa: E402
from scaling_transfer import stamp_of                                  # noqa: E402
from temporal_window import windowed                                   # noqa: E402
from viz_cluster_panel import read_pcd                                 # noqa: E402

PH = ("approach", "junction", "exit")


def load(corpus, labels, band, window=15, clean=True):
    """Features and labels for one corpus, restricted to frames worth training on.

    The mask is computed over the FULL CSV trajectory and then looked up by stamp, not
    over the subset that happens to have a PCD: displacement and path length are
    trajectory quantities, and computing them on a strided subset silently rescales the
    efficiency test.
    """
    from clustering.cluster_training import LidarDataset
    cfg = cfg_of(band, "town01", "none")
    ds = LidarDataset(f"{os.path.expanduser('~')}/carla_data/{corpus}/{corpus}_pcds", cfg)
    all_rows = list(csv.DictReader(open(os.path.join(PKG, labels))))
    if clean:
        km, _ = clean_mask(all_rows)
    else:
        km = np.array([float(r["speed"]) >= 0.1 for r in all_rows])
    ok = {int(r["stamp_ns"]) for r, k in zip(all_rows, km) if k}
    gt = {int(r["stamp_ns"]): r for r in all_rows}
    files = [f for f in ds.pcd_files if stamp_of(f) in gt]
    rows = [gt[stamp_of(f)] for f in files]
    X = np.stack([ranges_fast(read_pcd(f), cfg) for f in files])
    y = np.array([r["topology"] for r in rows])
    sel = np.array([int(r["stamp_ns"]) in ok for r in rows])
    # window BEFORE masking: the temporal features must come from real neighbours
    return windowed(X, window, "meanstd")[sel], y[sel]


def spans_of(y):
    out, i = [], 0
    for lab, n in runs_of(y):
        if lab == "junction":
            out.append((i, n))
        i += n
    return out


def decode(P, states, y_train_for_trans, y_eval, scale=0.25):
    A = fit_transitions(y_train_for_trans, states)
    pr = np.array([max((y_train_for_trans == s).sum(), 1) for s in states], float)
    pr /= pr.sum()
    emis = scale * (np.log(np.clip(P, EPS, None)) - np.log(pr)[None, :])
    return np.array(states)[viterbi(emis, np.log(np.clip(A, EPS, None)),
                                    np.log(np.clip(pr, EPS, None)))]


def report(tag, y, frame, hmm):
    from sklearn.metrics import f1_score
    sp = spans_of(y)
    gtr = len(runs_of(y))
    print(f"\n  {tag}   n={len(y)}   GT {gtr} runs, {len(sp)} traversals")
    print(f"    {'decoder':<18}{'macroF1':>9}{'acc':>8}{'runs':>7}{'order':>9}")
    for nm, lab in (("per-frame", frame), ("HMM 0.25", hmm)):
        print(f"    {nm:<18}{100*f1_score(y,lab,average='macro'):8.1f}%"
              f"{100*(lab==y).mean():7.1f}%{len(runs_of(lab)):7d}"
              f"{order_score(lab,sp):6d}/{len(sp)}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--band", default="above-road")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--no-clean", action="store_true",
                    help="keep thrashing frames (speed filter only), for comparison")
    a = ap.parse_args(argv)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.preprocessing import StandardScaler

    band = dict((b[0], b) for b in BANDS)[a.band]
    C = {"town05": ("town05_dual_newfov", "reports/frame_labels/town05_dual.csv"),
         "town01": ("town01_dual_newfov", "reports/frame_labels/town01_dual.csv")}
    D = {}
    for k, (c, l) in C.items():
        if not os.path.isdir(f"{os.path.expanduser('~')}/carla_data/{c}/{c}_pcds"):
            print(f"missing corpus {c}")
            return 1
        D[k] = load(c, l, band, clean=not a.no_clean)
        print(f"{k}: {len(D[k][1])} moving frames, "
              f"{dict(collections.Counter(D[k][1]))}")

    # ---- within Town05, out-of-fold over contiguous blocks -----------------
    F, y = D["town05"]
    states = sorted(set(y))
    si = {s: k for k, s in enumerate(states)}
    n = len(y)
    P = np.zeros((n, len(states)))
    hmm = np.empty(n, dtype=object)
    edges = [int(n * i / a.folds) for i in range(a.folds + 1)]
    for lo, hi in zip(edges, edges[1:]):
        tr = np.r_[np.arange(0, lo), np.arange(hi, n)]
        sc = StandardScaler().fit(F[tr])
        m = LogisticRegression(max_iter=3000, class_weight="balanced").fit(
            sc.transform(F[tr]), y[tr])
        P[lo:hi][:, [si[c] for c in m.classes_]] = m.predict_proba(sc.transform(F[lo:hi]))
        hmm[lo:hi] = decode(P[lo:hi], states, y[tr], y[lo:hi])
    frame = np.array(states)[P.argmax(axis=1)]
    print("\n=== WITHIN TOWN05 (out-of-fold, 5 contiguous folds) ===")
    report("town05 -> town05", y, frame, hmm.astype(str))
    print()
    print(classification_report(y, frame, digits=3, zero_division=0))

    # ---- cross-town, same rig ---------------------------------------------
    print("\n=== CROSS-TOWN, IDENTICAL RIG (-12..+12) ===")
    for src, dst in (("town01", "town05"), ("town05", "town01")):
        Fs, ys = D[src]
        Ft, yt = D[dst]
        common = sorted(set(ys) & set(yt))
        ks, kt = np.isin(ys, common), np.isin(yt, common)
        sc = StandardScaler().fit(Fs[ks])
        m = LogisticRegression(max_iter=3000, class_weight="balanced").fit(
            sc.transform(Fs[ks]), ys[ks])
        Pt = np.zeros((int(kt.sum()), len(common)))
        idx = {c: i for i, c in enumerate(common)}
        Pt[:, [idx[c] for c in m.classes_]] = m.predict_proba(sc.transform(Ft[kt]))
        fr = np.array(common)[Pt.argmax(axis=1)]
        hm = decode(Pt, common, ys[ks], yt[kt])
        report(f"{src} -> {dst}", yt[kt], fr, hm.astype(str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
