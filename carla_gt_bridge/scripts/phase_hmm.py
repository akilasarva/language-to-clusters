#!/usr/bin/env python3
"""Decode the phase SEQUENCE, not each frame independently. Viterbi over a constrained HMM.

    ~/miniconda3/bin/python scripts/phase_hmm.py --town town01_dual_newfov

WHY A SEQUENCE MODEL. The per-frame classifier is memoryless. Consecutive frames are
~0.4 m apart, so their features are nearly identical, and wherever the decision sits near
a class boundary small noise flips it: the per-frame prediction fragments into far more,
far shorter runs than the ground truth. Nothing in the model pays a cost for changing its
mind, and nothing forbids `exit` from being emitted before `junction`.

WHY NOT JUST SMOOTH. A majority filter punishes CHANGE but knows nothing about ORDER.
Pushed hard enough to reach the right number of runs, it absorbs whole phases into their
neighbours and destroys the approach->junction->exit order. Smoothing cannot distinguish
"this phase is short" from "this phase is noise".

WHAT THE HMM ADDS, exactly two things:
  * a DWELL cost. The self-transition probability makes leaving a state expensive, so the
    emission evidence has to actually overcome it. This is the part that kills flicker.
  * an ORDER constraint. Transitions never seen in the training folds get probability
    zero, so the decode cannot emit exit->junction or approach->exit. This is the part a
    filter can never supply.

The transition matrix is ESTIMATED FROM THE TRAINING FOLDS, not hand-written. A
transition that does not occur in ground truth is forbidden because the data says so,
not because someone listed it.

POSTERIOR -> LIKELIHOOD. The classifier emits P(state | x). Viterbi wants P(x | state),
so each posterior is divided by the training class prior. Skipping that double-counts the
class balance -- once in the classifier, once in the transition matrix -- and quietly
biases the decode toward `path`.
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

from band_sweep import BANDS, cfg_of, ranges_fast          # noqa: E402
from scaling_transfer import stamp_of                      # noqa: E402
from temporal_window import windowed                       # noqa: E402
from viz_cluster_panel import read_pcd                     # noqa: E402

EPS = 1e-12


def runs_of(seq):
    return [(k, sum(1 for _ in g)) for k, g in itertools.groupby(seq)]


def fit_transitions(y, states, *, forbid_unseen=True):
    """Row-stochastic transition matrix from a ground-truth label sequence."""
    i = {s: k for k, s in enumerate(states)}
    C = np.zeros((len(states), len(states)))
    for a, b in zip(y, y[1:]):
        C[i[a], i[b]] += 1
    if forbid_unseen:
        C[C > 0] += 1.0            # Laplace, but only where the data went
    else:
        C += 1.0
    rs = C.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return C / rs


def viterbi(log_emis, log_trans, log_start):
    n, k = log_emis.shape
    dp = np.full((n, k), -np.inf)
    bp = np.zeros((n, k), dtype=int)
    dp[0] = log_start + log_emis[0]
    for t in range(1, n):
        m = dp[t - 1][:, None] + log_trans
        bp[t] = m.argmax(axis=0)
        dp[t] = m.max(axis=0) + log_emis[t]
    path = np.empty(n, dtype=int)
    path[-1] = dp[-1].argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = bp[t, path[t]]
    return path


def order_score(labels, spans, pad=60):
    ok = 0
    for s, n in spans:
        lo, hi = max(0, s - pad), min(len(labels), s + n + pad)
        dd = [k for k, _ in itertools.groupby(
            [k for k, _ in runs_of(labels[lo:hi])
             if k in ("approach", "junction", "exit")])]
        it = iter(dd)
        if all(t in it for t in ("approach", "junction", "exit")):
            ok += 1
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--town", default="town01_dual_newfov")
    ap.add_argument("--labels", default="")
    ap.add_argument("--band", default="above-road")
    ap.add_argument("--window", type=int, default=15)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--scale", type=float, nargs="*", default=[1.0, 0.6, 0.4, 0.25, 0.15],
                    help="emission scale. <1 tempers an overconfident classifier so the "
                         "transition prior carries more of the decode")
    a = ap.parse_args(argv)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler
    from clustering.cluster_training import LidarDataset

    pcds = f"{os.path.expanduser('~')}/carla_data/{a.town}/{a.town}_pcds"
    csvp = a.labels or os.path.join(PKG, "reports", "frame_labels",
                                    f"{a.town.rsplit('_', 1)[0]}.csv")
    band = dict((b[0], b) for b in BANDS)[a.band]
    cfg = cfg_of(band, "town01", "none")

    ds = LidarDataset(pcds, cfg)
    gt = {int(r["stamp_ns"]): r for r in csv.DictReader(open(csvp))}
    files = [f for f in ds.pcd_files if stamp_of(f) in gt]
    rows = [gt[stamp_of(f)] for f in files]
    X = np.stack([ranges_fast(read_pcd(f), cfg) for f in files])
    y = np.array([r["topology"] for r in rows])
    mv = np.array([float(r["speed"]) >= 0.1 for r in rows])
    F = windowed(X, a.window, "meanstd")[mv]
    y = y[mv]
    states = sorted(set(y))
    si = {s: k for k, s in enumerate(states)}
    n = len(y)
    print(f"{a.town}  band {a.band}  n={n}  states={states}")

    # out-of-fold POSTERIORS over contiguous blocks, plus a per-fold transition matrix
    P = np.zeros((n, len(states)))
    frame_pred = np.empty(n, dtype=object)
    decoded = {sc_: np.empty(n, dtype=object) for sc_ in a.scale}
    edges = [int(n * i / a.folds) for i in range(a.folds + 1)]
    for lo, hi in zip(edges, edges[1:]):
        tr = np.r_[np.arange(0, lo), np.arange(hi, n)]
        sc = StandardScaler().fit(F[tr])
        m = LogisticRegression(max_iter=3000, class_weight="balanced")
        m.fit(sc.transform(F[tr]), y[tr])
        cols = [si[c] for c in m.classes_]
        p = m.predict_proba(sc.transform(F[lo:hi]))
        P[lo:hi][:, cols] = p
        frame_pred[lo:hi] = m.classes_[p.argmax(axis=1)]

        # transitions and priors from the TRAINING folds only
        A = fit_transitions(y[tr], states)
        prior = np.array([max((y[tr] == s).sum(), 1) for s in states], float)
        prior /= prior.sum()
        emis = np.log(np.clip(P[lo:hi], EPS, None)) - np.log(prior)[None, :]
        start = np.log(np.clip(prior, EPS, None))
        lA = np.log(np.clip(A, EPS, None))
        for sc_ in a.scale:
            decoded[sc_][lo:hi] = np.array(states)[viterbi(sc_ * emis, lA, start)]
    frame_pred = frame_pred.astype(str)

    spans, idx = [], 0
    for lab, ln in runs_of(y):
        if lab == "junction":
            spans.append((idx, ln))
        idx += ln
    gtr = len(runs_of(y))

    print(f"\n  {'decoder':<22}{'macroF1':>9}{'runs':>8}{'medrun':>8}{'order':>10}")
    cand = [("ground truth", y), ("per-frame argmax", frame_pred)]
    cand += [(f"HMM scale={sc_:g}", decoded[sc_].astype(str)) for sc_ in a.scale]
    for name, lab in cand:
        r = runs_of(lab)
        f1 = 100 * f1_score(y, lab, average="macro")
        print(f"  {name:<22}{f1:8.1f}%{len(r):8d}{int(np.median([q for _, q in r])):8d}"
              f"{order_score(lab, spans):7d}/{len(spans)}")
    print(f"\n  (ground truth has {gtr} runs over {len(spans)} junction traversals)")

    A = fit_transitions(y, states)
    print("\n  transition matrix learned from ground truth (rows sum to 1, "
          "0.000 = forbidden by the data):")
    print("      " + "".join(f"{s:>11s}" for s in states))
    for s, row in zip(states, A):
        print(f"  {s:<10s}" + "".join(f"{v:11.3f}" for v in row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
