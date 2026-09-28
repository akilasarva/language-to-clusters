#!/usr/bin/env python3
"""Did the vehicle actually perform the maneuver the mission named?

WHY THIS EXISTS. Region-sequence scoring cannot see a turn: driving straight through a
junction and turning right through it produce the same region sequence whenever both
exits lie in the goal mode's accept set. A run can therefore drive the wrong maneuver
(or straight through a `Bearing(Right) completed` step) and still report "plan complete".

So this scores the one thing region sequences cannot: heading. It reads no plan and
trusts no monitor. It integrates yaw over the run from the odometry and asks whether the
net change matches what the mission called for -- an independent check on the executor.

ROS REP-103: +z yaw is counter-clockwise, so a LEFT turn is positive and a RIGHT turn is
negative. "Straight" is anything under the threshold in either direction.

SOUND ONLY FOR SINGLE-DECISION MISSIONS, and this is a real bound rather than a caveat.
Net heading is integrated over the whole run, so a mission that turns right and then left
integrates to roughly zero and scores "straight" -- indistinguishable from never turning
at all. Every mission scored here has one decision point. A two-decision mission needs
per-segment integration between region transitions, which this does not do.

The 60-degree threshold is not a fine-tuned value; verdicts are insensitive to it over a
wide range around 60.

PAIRS, NOT RUNS. An arm that always drives straight scores 100% on the straight world
while being wrong about everything. So `--pair` requires both worlds of a mission to be right before either counts.

    python3 carla_gt_bridge/scripts/score_maneuvers.py reports/runs/<campaign> \
        --expect absent=right --expect present=straight --pair
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

#: brain's `bearing_complete_deg`, confirmed in carla_gt_bridge/launch/mission.launch.py
THRESH_DEG = 60.0


def wrap(d: float) -> float:
    """Signed shortest angular difference, in degrees."""
    return (d + 180.0) % 360.0 - 180.0


def segment_turns(rows) -> list[float]:
    """Heading change per REGION segment, so two turns cannot cancel.

    Integrating over the whole run collapses a right-then-left into "straight": a run
    that turns right ~110 deg and then left twice nets under the threshold and would be
    scored as straight.
    """
    out: list[float] = []
    cur = None
    acc = 0.0
    prev = None
    for a in rows:
        c, y = a.get("gt_cluster"), a.get("yaw_deg")
        if y is None:
            continue
        if c != cur:
            if cur is not None:
                out.append(acc)
            cur, acc = c, 0.0
        if prev is not None:
            acc += wrap(y - prev)
        prev = y
    if cur is not None:
        out.append(acc)
    return out


def classify_run(rows, thresh: float = THRESH_DEG) -> tuple[str, list[float]]:
    """The maneuver this run performed, judged per segment.

    A mission asks for ONE maneuver. A run containing two or more large turns in
    OPPOSING directions did not perform it -- it wandered -- and calling that "straight"
    because the turns cancel would credit a run that never followed the instruction.
    `wander` is therefore its own verdict and is never correct for any expectation.
    """
    segs = segment_turns(rows)
    big = [v for v in segs if abs(v) >= thresh]
    if not big:
        return "straight", segs
    if any(v > 0 for v in big) and any(v < 0 for v in big):
        return "wander", segs
    return ("left" if big[0] > 0 else "right"), segs


def net_heading(rows) -> float | None:
    """Total signed heading change over the run, unwrapped tick to tick.

    Unwrapping matters and is not cosmetic: a westbound run sits at yaw = +-180, where
    raw differencing turns sub-degree jitter into 360 deg of phantom rotation.
    """
    ys = [r["yaw_deg"] for r in rows if r.get("yaw_deg") is not None]
    if len(ys) < 3:
        return None
    return sum(wrap(ys[i + 1] - ys[i]) for i in range(len(ys) - 1))


def classify(net: float, thresh: float = THRESH_DEG) -> str:
    if abs(net) < thresh:
        return "straight"
    return "left" if net > 0 else "right"


def matches(name: str, sub: str) -> bool:
    """`sub` is an AND of `+`-joined substrings, so a per-mission world can be named.

    A campaign runs several missions with opposite polarities -- `polarity` runs "right if
    cone" and "right unless cone" side by side -- so "present" alone cannot say what was
    expected. `m1+present` can.
    """
    return all(part in name for part in sub.split("+"))


def group_key(name: str, worlds) -> str:
    """Run name with the world markers stripped, so the two worlds of a pair collide.

    Only the world tokens are stripped, never the mission tag: stripping `m1` too would
    collapse m1 and m2 into one pair and silently score across missions.
    """
    for w in worlds:
        name = name.replace(w, "")
    return name


#: runs without a run.jsonl copy kept only console.txt, but the MPC's own debug log
#: carries the same rows -- run.jsonl IS a copy of the debug log.
DEBUG_LOGS = "dgppo_ros_node_pkg/dgppo_ros_node_pkg/debug_logs"


def _debug_log_index():
    return sorted((os.path.getmtime(f), f)
                  for f in glob.glob(os.path.join(DEBUG_LOGS, "*.jsonl")))


def _nearest_log(index, when: float, tol: float = 600.0):
    """The debug log written closest to `when`, or None outside `tol` seconds."""
    import bisect
    times = [t for t, _ in index]
    i = bisect.bisect_left(times, when)
    best = None
    for j in (i - 1, i, i + 1):
        if 0 <= j < len(index):
            d = abs(index[j][0] - when)
            if d < tol and (best is None or d < best[0]):
                best = (d, index[j][1])
    return None if best is None else best[1]


def rows_near_region(rows, region: int, town: str, radius: float):
    """Only the ticks within `radius` of `region`'s centroid.

    WHY THIS EXISTS. Net heading over the whole run is sound ONLY while the mission has
    one decision AND the run does not drive past it -- the module docstring says so. A run
    that completes the route traverses further junctions, and the whole-run net becomes
    dominated by driving AFTER the decision point: it can read as a large left drift while
    the manoeuvre AT the decision junction is straight and correct. Scoping to the decision
    junction removes that contamination.
    """
    import numpy as _np
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    z = _np.load(os.path.join(here, "config", f"regions.{town}.npz"))
    rids = list(z["rids"])
    if region not in rids:
        return rows
    cx, cy = z["centroids"][rids.index(region)]
    return [r for r in rows
            if r.get("x") is not None
            and math.hypot(r["x"] - cx, r["y"] - cy) <= radius]


def measure(root: str, thresh: float, expect, index=None,
            at_region: int | None = None, town: str = "town05",
            radius: float = 22.0):
    """One row per run directory under `root`.

    Prefers the run's own run.jsonl; falls back to the MPC debug log written at the same
    time, which is how runs without a run.jsonl get scored at all.

    `at_region` scopes the heading integration to the ticks near that region's centroid --
    use it whenever a run drives beyond its decision point. See rows_near_region.
    """
    out = []
    for d in sorted(glob.glob(os.path.join(root, "*/"))):
        f = os.path.join(d, "run.jsonl")
        if not os.path.exists(f):
            console = os.path.join(d, "console.txt")
            f = (_nearest_log(index, os.path.getmtime(console))
                 if index is not None and os.path.exists(console) else None)
        if not f or not os.path.exists(f):
            continue
        name = os.path.basename(d.rstrip("/"))
        try:
            rows = [json.loads(l) for l in open(f) if l.strip()]
            if at_region is not None:
                scoped = rows_near_region(rows, at_region, town, radius)
                # Fall back loudly rather than silently scoring the whole run: a run that
                # never reached the decision junction must not be scored as if it had.
                if len(scoped) > 2:
                    rows = scoped
                else:
                    rows = []
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip] {name}: {exc}", file=sys.stderr)
            continue
        net = net_heading(rows)
        if net is None:
            continue
        got, _segs = classify_run(rows, thresh)
        want = next((m for sub, m in expect if matches(name, sub)), None)
        out.append((name, net, got, want, None if want is None else got == want))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="+",
                    help="directories of run subdirectories, each with run.jsonl")
    ap.add_argument("--expect", action="append", default=[], metavar="SUBSTRING=MANEUVER",
                    help="runs whose name contains SUBSTRING must end in MANEUVER "
                         "(left|right|straight). Repeatable.")
    ap.add_argument("--pair", action="store_true",
                    help="score PAIRS: both worlds of a mission must be right")
    ap.add_argument("--world", action="append", default=[], metavar="TOKEN",
                    help="run-name token that distinguishes worlds (e.g. absent). "
                         "Stripped to form the pair key. Repeatable; required with "
                         "--pair when --expect uses AND-substrings.")
    ap.add_argument("--threshold", type=float, default=THRESH_DEG)
    ap.add_argument("--at-region", type=int, default=None,
                    help="score the manoeuvre only near this region's centroid (the "
                         "decision junction). REQUIRED once runs drive past the decision "
                         "point -- whole-run net heading is then dominated by later "
                         "junctions. See rows_near_region.")
    ap.add_argument("--at-radius", type=float, default=22.0)
    ap.add_argument("--quiet", action="store_true", help="totals only")
    ap.add_argument("--no-fallback", action="store_true",
                    help="score only runs carrying their own run.jsonl")
    a = ap.parse_args(argv)

    expect = []
    for e in a.expect:
        if "=" not in e:
            ap.error(f"--expect needs SUBSTRING=MANEUVER, got {e!r}")
        sub, man = e.split("=", 1)
        if man not in ("left", "right", "straight"):
            ap.error(f"maneuver must be left|right|straight, got {man!r}")
        expect.append((sub, man))
    if a.pair and len(expect) < 2:
        ap.error("--pair needs at least two --expect conditions")

    index = None if a.no_fallback else _debug_log_index()
    rows = []
    for root in a.root:
        for r in measure(root, a.threshold, expect, index,
                         at_region=a.at_region, town=a.town if hasattr(a, "town") else "town05",
                         radius=a.at_radius):
            rows.append((os.path.basename(root.rstrip("/")),) + r)
    if not rows:
        print("no runs found (looked for */run.jsonl)", file=sys.stderr)
        return 2

    w = max(len(f"{r[0]}/{r[1]}") for r in rows) + 2
    if not a.quiet:
        print(f"{'run':<{w}}{'net deg':>9}  {'maneuver':<10}{'expected':<10}")
        for root, name, net, got, want, ok in rows:
            mark = "" if ok is None else ("  ok" if ok else "  MISMATCH")
            print(f"{root + '/' + name:<{w}}{net:>9.1f}  {got:<10}"
                  f"{(want or '-'):<10}{mark}")

    if a.pair:
        subs = a.world or [s for s, _ in expect]
        groups: dict[str, list] = {}
        for root, name, net, got, want, ok in rows:
            if want is None:
                continue
            groups.setdefault(f"{root}/{group_key(name, subs)}", []).append((name, ok))
        # A pair has as many members as there are WORLDS, not as there are conditions:
        # a campaign running two missions of opposite polarity needs four --expect
        # clauses to describe two worlds, and counting clauses called every pair
        # incomplete.
        n_worlds = len(a.world) if a.world else len(expect)
        full = {k: v for k, v in groups.items() if len(v) >= n_worlds}
        partial = {k: v for k, v in groups.items() if len(v) < n_worlds}
        n_ok = sum(1 for v in full.values() if all(o for _, o in v))
        if not a.quiet:
            print()
            for k, v in sorted(full.items()):
                bad = [n for n, o in v if not o]
                print(f"{k:<{w}}" + ("both worlds correct" if not bad
                                     else f"FAILS ({', '.join(bad)})"))
            for k, v in sorted(partial.items()):
                print(f"{k:<{w}}incomplete pair ({len(v)} of {n_worlds} worlds)")
        print(f"\n{n_ok}/{len(full)} PAIRS correct in both worlds")
        return 0 if n_ok == len(full) else 1

    scored = [r for r in rows if r[5] is not None]
    n_ok = sum(1 for r in scored if r[5])
    if scored:
        print(f"\n{n_ok}/{len(scored)} runs performed the maneuver the mission named "
              f"({100 * n_ok / len(scored):.0f}%)")
    else:
        print(f"\n{len(rows)} runs measured; pass --expect to score them")
    return 0 if n_ok == len(scored) else 1


if __name__ == "__main__":
    raise SystemExit(main())
