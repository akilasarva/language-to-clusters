#!/usr/bin/env python3
"""Did the vehicle perform the manoeuvre the MAP says that traversal is?

WHY A SECOND SCORER, and why the first one could not answer this.
`score_maneuvers.classify_run` integrates yaw over the ticks spent INSIDE the junction
region. That is not the quantity `map_regions.maneuvers` defines a manoeuvre to be. The
map compares the APPROACH road's bearing to the EXIT road's bearing and calls anything
under 45 degrees straight -- it says nothing about the path taken between them. The gap
is not subtle: approaching J60 from region 1 and leaving to region 2 is `straight(+0)` in
the map, yet a straight-through run can integrate tens of degrees of in-junction weave
inside it, which yaw integration misreads as a turn.

So this measures what the map defines: entry-road heading vs exit-road heading, discarding
the junction segment's own yaw. `run_scoring.maneuvers` already implements exactly that,
with the boundary and corner-clip cases worked out, and it uses STRAIGHT_DEG=45 -- the same
threshold as the map. This file is the adapter plus the validation, not a third rule.

TWO DIFFERENT QUESTIONS, kept apart on purpose:

  --validate  Does the scorer agree with the MAP on every traversal? This is
              mission-independent and has a known answer, so disagreement means the
              SCORER is wrong, not the run. Run this before trusting the default mode.

  (default)   What manoeuvre did each traversal perform? Compare that against what the
              English asked for -- which this file deliberately does not know, because
              the mission expectation is an input to the experiment, not a property of
              the trace.

POSITIONAL, NOT YAW. `run_scoring` derives heading from successive positions, which is
immune to the body yaw oscillating while the path stays straight. It is NOT the quantity
`score_maneuvers`' 60-degree threshold and its 40-70 insensitivity sweep were calibrated
on, so --sweep re-checks window sensitivity here rather than inheriting it.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_scoring import group_by_region, classify, _wrap180, _bearing  # noqa: E402

#: Heading baseline in METRES, not ticks. `run_scoring.HEADING_WINDOW` is 3 TICKS, and tick
#: density is not a property of the road: it varies by an order of magnitude between a
#: clean run and a stalled one, so 3 ticks is over a metre on one run and centimetres on
#: another. A window that spans too little road simply has no heading to measure.
#:
#: 8 m is chosen on physical grounds and NOT tuned: it is a lane-scale baseline, and it is the
#: MPC's own planning horizon (N=8 x dt=0.2 s at ~5 m/s). Tuning the window to maximise map
#: agreement would be circular -- the same window is then used to detect departures FROM the
#: map, and a window picked to agree would suppress exactly what it is meant to find.
HEADING_BASE_M = 8.0

CMAP = os.path.join(PKG, "config", "cluster_map.carla_town05.yaml")
XODR = os.path.join(PKG, "config", "Town05.xodr")


def labels_for(cmap_path: str) -> dict[int, str]:
    """region id -> 'junction' | 'path'. `junction` is checked FIRST because the taxonomy's
    `path` list is a superset containing every region id, so testing `path` first would
    label every junction a path and silently score zero traversals."""
    import yaml
    modes = yaml.safe_load(open(cmap_path))["modes"]
    junc = set(modes.get("junction", []))
    return {int(r): ("junction" if int(r) in junc else "path")
            for r in set(modes.get("path", [])) | junc}


def map_expectations() -> dict[tuple[int, int, int], str]:
    """(approach_rid, junction_rid, exit_rid) -> 'straight'|'left'|'right', from the map."""
    from map_regions import maneuvers as map_maneuvers
    from carla_gt_bridge.opendrive import load
    from carla_gt_bridge.segmenter import segment
    m = load(XODR)
    rm = segment(m)
    out = {}
    for _jid, j_rid, a_rid, exits in map_maneuvers(m, rm):
        for kind, _turn, b_rid in exits:
            out[(a_rid, j_rid, b_rid)] = kind
    return out


#: Below this, two successive poses carry NO heading information and their bearing is noise.
#: This is not a tuning knob -- it is the difference between a measurement and a coin flip.
MIN_STEP_M = 0.05


def moving(rows, eps: float = MIN_STEP_M):
    """Rows with the stationary ticks dropped.

    WHY THIS IS REQUIRED, not a refinement. A run that completes prints `brain reports plan
    complete - holding stop` and then sits still for the rest of its 150 s, so the LAST ticks
    of every successful run are at zero displacement. `run_scoring.maneuvers` takes the exit
    heading from exactly those ticks when the junction is the closing group, and the bearing
    between two identical points is undefined -- in practice it flips to +-180 on the
    final junction of a completed run. Dropping stationary ticks makes the exit heading
    the last real motion instead of the jitter after it.
    """
    out, last = [], None
    for r in rows:
        x, y = r.get("x"), r.get("y")
        if x is None or y is None or r.get("gt_cluster") is None:
            continue
        if last is not None and (x - last[0]) ** 2 + (y - last[1]) ** 2 < eps * eps:
            continue
        out.append(r)
        last = (x, y)
    return out


def _walk(trace, start, step, base_m):
    """Index `base_m` metres away from `start` along the trace, or None if the trace ends first."""
    d, i = 0.0, start
    while 0 <= i + step < len(trace):
        nxt = i + step
        d += ((trace[nxt][0] - trace[i][0]) ** 2 + (trace[nxt][1] - trace[i][1]) ** 2) ** 0.5
        i = nxt
        if d >= base_m:
            return i
    return i if d >= base_m * 0.5 else None


def _heading_before(trace, k, base_m):
    """Road heading on the APPROACH: bearing over the `base_m` of travel ending at index k."""
    j = _walk(trace, k, -1, base_m)
    return None if j is None or j == k else _bearing(trace[j], trace[k])


def _heading_after(trace, k, base_m):
    """Road heading on the EXIT: bearing over the `base_m` of travel starting at index k."""
    j = _walk(trace, k, +1, base_m)
    return None if j is None or j == k else _bearing(trace[k], trace[j])


def score(trace_rows, labels, window=HEADING_BASE_M):
    """[(approach, junction, exit, observed_kind, delta_deg)] for each traversal."""
    trace = [(r["x"], r["y"], r.get("gt_cluster"), None, None)
             for r in moving(trace_rows)]
    groups = group_by_region(trace)
    gids = [g[0] for g in groups]
    out = []
    for gi, rid in enumerate(gids):
        if labels.get(rid) != "junction":
            continue
        # Entered and left into the SAME region -> the vehicle clipped the junction's corner
        # while driving past it; that is not a traversal. `run_scoring.maneuvers` documents
        # this case (Town06 read `4,73,5,73,5,74,38` for one straight drive) and skips it.
        if 0 < gi < len(gids) - 1 and gids[gi - 1] == gids[gi + 1]:
            continue
        a = gids[gi - 1] if gi > 0 else None
        b = gids[gi + 1] if gi < len(gids) - 1 else None
        ks = groups[gi][1]
        h_in = _heading_before(trace, ks[0], window)
        h_out = _heading_after(trace, ks[-1], window)
        if h_in is None or h_out is None:
            out.append((a, rid, b, None, None))
            continue
        d = _wrap180(h_out - h_in)
        out.append((a, rid, b, classify(d), round(d, 1)))
    return out


def load_runs(roots):
    for root in roots:
        for d in sorted(glob.glob(os.path.join(root, "*/"))):
            f = os.path.join(d, "run.jsonl")
            if not os.path.exists(f):
                continue
            try:
                rows = [json.loads(l) for l in open(f) if l.strip()]
            except Exception:                                        # noqa: BLE001
                continue
            if rows:
                yield os.path.basename(d.rstrip("/")), rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="+")
    ap.add_argument("--validate", action="store_true",
                    help="check every traversal against the map's own answer")
    ap.add_argument("--sweep", action="store_true",
                    help="re-check window sensitivity instead of inheriting it")
    ap.add_argument("--window", type=float, default=HEADING_BASE_M,
                    help="heading baseline in METRES")
    a = ap.parse_args(argv)

    labels = labels_for(CMAP)
    runs = list(load_runs(a.root))
    if not runs:
        print("no runs with run.jsonl found", file=sys.stderr)
        return 1

    if a.sweep:
        exp = map_expectations()
        print(f"{'base_m':>7}  {'agree':>12}  note")
        for w in (2.0, 4.0, 6.0, 8.0, 12.0, 20.0):
            ok = tot = 0
            for _n, rows in runs:
                for t in score(rows, labels, window=w):
                    want = exp.get((t[0], t[1], t[2]))
                    if want is None or t[3] is None:
                        continue
                    tot += 1
                    ok += (want == t[3])
            print(f"{w:7.1f}  {ok:5d}/{tot:<6d}  {ok / tot:.0%}" if tot else f"{w:7.1f}  no data")
        return 0

    exp = map_expectations() if a.validate else {}
    ok = tot = 0
    for name, rows in runs:
        ts = score(rows, labels, window=a.window)
        bits = []
        for app, j, ex, kind, d in ts:
            s = f"{app}->J{j}->{ex}:{kind or '?'}"
            if a.validate:
                want = exp.get((app, j, ex))
                if want is not None and kind is not None:
                    tot += 1
                    ok += (want == kind)
                    s += "" if want == kind else f"(map says {want})"
                else:
                    s += "(no map answer)"
            if d is not None:
                s += f"[{d:+.0f}]"
            bits.append(s)
        print(f"{name:22s} {'  '.join(bits)}")
    if a.validate and tot:
        print(f"\nagreement with the map: {ok}/{tot} = {ok / tot:.0%}")
        print("disagreement here means the SCORER is wrong, not the run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
