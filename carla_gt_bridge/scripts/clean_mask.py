#!/usr/bin/env python3
"""Which frames of a collection drive are worth training on?

    python3 scripts/clean_mask.py --labels <frames.csv>

WHY. A collection drive can wedge: stretches of reverse-and-re-approach with recoveries
climbing, interleaved with clean stretches. A corpus built from wedge-and-reverse cycles
has the ego re-approaching the same geometry from alternating directions, which skews the
approach/exit balance of the labels.

WHY NOT JUST TAKE A PREFIX. Clean stretches also occur late in a drive; a prefix cut throws
them away. The thing to remove is THRASHING, not "late", so the mask is derived from what
the trajectory does.

THREE CRITERIA, all local so a bad patch costs only that patch:
  moving     speed >= 0.1 m/s. Stationary frames are near-duplicates and can dominate
             a label.
  not warped a trajectory jump over `jump_m` is a teleport; frames within `pad` of one are
             dropped because the window features straddle a discontinuity.
  efficient  over a window, net displacement / path length. Driving straight scores ~1,
             a corner ~0.9, reversing out of a wedge and re-approaching ~0.1-0.3. This is
             the criterion that finds thrashing, and it needs no recovery log -- which
             matters, because a killed drive never writes one.

SANITY CHECK BUILT IN: run it on a corpus known to be clean (no recoveries, no teleports)
and it should keep essentially everything. A filter that cuts a clean drive is measuring
itself, not the data.
"""
from __future__ import annotations

import argparse
import collections
import csv
import math
import os

import numpy as np

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def clean_mask(rows, *, min_speed=0.1, jump_m=20.0, pad=15,
               window=100, min_eff=0.5, min_path_m=2.0):
    """Boolean mask over `rows` (dicts with x, y, speed). True = keep."""
    x = np.array([float(r["x"]) for r in rows])
    y = np.array([float(r["y"]) for r in rows])
    sp = np.array([float(r["speed"]) for r in rows])
    n = len(rows)
    step = np.hypot(np.diff(x), np.diff(y))

    keep = sp >= min_speed

    # teleports: a jump is not motion, and a window spanning one is meaningless
    warp = np.zeros(n, bool)
    for i in np.flatnonzero(step > jump_m):
        warp[max(0, i - pad):min(n, i + pad + 2)] = True
    keep &= ~warp

    # progress efficiency over a centred window, with jumps excluded from the path
    s = np.where(step > jump_m, 0.0, step)
    cum = np.concatenate([[0.0], np.cumsum(s)])
    h = window // 2
    lo = np.clip(np.arange(n) - h, 0, n - 1)
    hi = np.clip(np.arange(n) + h, 0, n - 1)
    path = cum[hi] - cum[lo]
    net = np.hypot(x[hi] - x[lo], y[hi] - y[lo])
    eff = np.where(path > min_path_m, net / np.maximum(path, 1e-9), 1.0)
    keep &= eff >= min_eff
    return keep, dict(moving=int((sp >= min_speed).sum()), warped=int(warp.sum()),
                      inefficient=int((eff < min_eff).sum()))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--min-eff", type=float, default=0.5)
    ap.add_argument("--window", type=int, default=100)
    a = ap.parse_args(argv)

    p = a.labels if os.path.isabs(a.labels) else os.path.join(PKG, a.labels)
    rows = list(csv.DictReader(open(p)))
    keep, why = clean_mask(rows, window=a.window, min_eff=a.min_eff)
    top = [r["topology"] for r in rows]
    before = collections.Counter(top)
    after = collections.Counter(t for t, k in zip(top, keep) if k)

    print(f"{os.path.basename(p)}: {len(rows)} frames -> {int(keep.sum())} kept "
          f"({100*keep.mean():.1f}%)")
    print(f"  rejected by: stationary {len(rows)-why['moving']}, "
          f"near a teleport {why['warped']}, thrashing {why['inefficient']}")
    print(f"\n  {'label':<12}{'before':>9}{'after':>9}{'kept':>8}")
    for k in sorted(before, key=lambda z: -before[z]):
        print(f"  {k:<12}{before[k]:9d}{after[k]:9d}{100*after[k]/before[k]:7.0f}%")
    for nm, c in (("before", before), ("after", after)):
        ap_, ex = c.get("approach", 0), c.get("exit", 0)
        print(f"  approach/exit {nm:<7}: {ap_}/{ex} = "
              f"{ap_/max(ex,1):.2f}x  (1.0 is a route that enters and leaves each "
              f"junction once)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
