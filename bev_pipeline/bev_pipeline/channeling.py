#!/usr/bin/env python3
"""Channeling / boundedness features from the free-space angular profile.

The within-open distinction (open_space vs path vs junction) is fundamentally
"how is the *free space* shaped around you":
  - omnidirectional (open in ~all directions)  -> open_space  (no defined channel)
  - two opposite lobes (open fore/aft, closed sides) -> path   (a channel)
  - three+ lobes                                -> junction    (ways branch)

We extract the free-space lobes from the geom 36-sector clearance profile and
derive: lobe count, angular coverage, largest-lobe width, opposite-pair-ness,
and directional entropy. These target the "no defined way = open" definition via
the *shape* of free space (not just its amount, which overlapped).
"""
import numpy as np

MR = 20.0            # max range the profile is normalized by (GeomParams)
NS = 36              # sectors


def lobe_features(profile_norm, open_thresh=0.35):
    """profile_norm: (36,) free-space clearance in [0,1] (geom[:36]). Returns dict."""
    prof = np.asarray(profile_norm, float)
    openmask = prof > open_thresh                      # "can travel far this way"
    cover = float(openmask.mean())
    # contiguous open lobes with wrap-around
    if openmask.all():
        lobes = [(0, NS)]
    elif not openmask.any():
        lobes = []
    else:
        m2 = np.concatenate([openmask, openmask])
        lobes = []; i = 0
        # find a closed sector to start (so wrap-around lobes aren't split)
        start = int(np.argmax(~openmask))
        seq = np.roll(openmask, -start)
        j = 0
        while j < NS:
            if seq[j]:
                k = j
                while k < NS and seq[k]:
                    k += 1
                lobes.append((j, k))
                j = k
            else:
                j += 1
    widths = [ (b - a) for a, b in lobes ]
    n_lobes = len(lobes)
    maxw = max(widths) / NS if widths else 0.0
    # opposite-pair-ness: for a 2-lobe path, the two lobes' centers are ~180 apart
    opp = 0.0
    if n_lobes == 2:
        c0 = ((lobes[0][0] + lobes[0][1]) / 2) % NS
        c1 = ((lobes[1][0] + lobes[1][1]) / 2) % NS
        d = abs(c0 - c1) % NS; d = min(d, NS - d)
        opp = 1.0 - abs(d - NS / 2) / (NS / 2)         # 1.0 when exactly opposite
    # directional entropy of the (open) clearance distribution
    p = prof / (prof.sum() + 1e-9); ent = float(-(p * np.log(p + 1e-9)).sum() / np.log(NS))
    return dict(n_lobes=n_lobes, cover=cover, max_lobe=maxw, opposite=opp, entropy=ent)


def classify_within_open(profile_norm, **kw):
    """Rule: omnidirectional -> open_space; 2 opposite lobes -> path; 3+ -> junction."""
    f = lobe_features(profile_norm, **kw)
    if f["cover"] > 0.8 or (f["n_lobes"] <= 1 and f["cover"] > 0.55):
        return "open_space"
    if f["n_lobes"] >= 3:
        return "junction"
    if f["n_lobes"] == 2:
        return "path" if f["opposite"] > 0.5 else "junction"
    return "path"                                      # single narrow lobe / channel end


def features_matrix(geom):
    """(N,36)->(N,5) lobe-feature matrix to augment the classifier."""
    out = []
    for i in range(len(geom)):
        f = lobe_features(geom[i, :NS])
        out.append([f["n_lobes"], f["cover"], f["max_lobe"], f["opposite"], f["entropy"]])
    return np.array(out, float)
