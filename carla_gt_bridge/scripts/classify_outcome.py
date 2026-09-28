#!/usr/bin/env python3
"""Why did this run not finish? Separate PLAN faults from PERCEPTION faults from CONTROL.

WHY THIS EXISTS. "did not finish" is the same string for several unrelated causes. A
mission whose branch does not exist in the map, a cue no publisher can answer, a vehicle
wedged against a building the planner cannot see, and a run that simply hit its time budget
all report identically, and telling them apart from a console is slow and error-prone.

THE CATEGORIES, and the signal that identifies each. Order matters: the first that matches
wins, because an infra failure makes every later signal meaningless.

  INFRA       no MPC rows at all, or the stack aborted during bring-up. Nothing was
              measured; the run says nothing about the plan.
  PLAN        the plan could not be executed as written -- generation failed, or the
              targeter never grounded a step, or a named manoeuvre is not offered at the
              junction the robot actually reached.
  PERCEPTION  a cue went unanswered. `no branch answer yet` means no published key matched
              a branch's vlm_cue; a CUE TIMEOUT with 0 sightings means the step's own cue
              was never satisfied. The plan may be perfect.
  CONTROL     the vehicle was commanded to move and did not. Full throttle with near-zero
              displacement is the signature of geometry the planner has no map for --
              buildings are Unreal-only and appear in no `.xodr`, and Phase A runs with the
              LiDAR EDT disabled, so nothing rejects a rollout that drives into one.
  BUDGET      it was still making progress when RUN_SECONDS ran out. Not a fault.
  COMPLETED   the plan reported completion.

    python3 carla_gt_bridge/scripts/classify_outcome.py reports/runs/<campaign>/*/
    python3 carla_gt_bridge/scripts/classify_outcome.py --all
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: A window of this many seconds of sim with commanded speed above CMD_MIN and
#: displacement below MOVED_MIN is "commanded but not moving".
STUCK_WINDOW_S = 30.0
CMD_MIN = 0.5          # m/s commanded
MOVED_MIN = 1.0        # m actually travelled in the window


def _rows(run_dir: str) -> list[dict]:
    """This run's MPC telemetry, or [] if it wrote none.

    Many runs do NOT write run.jsonl; they leave the trace in the shared dgppo
    debug_logs. Falling back to it is required to classify them at all -- but the
    fallback MUST be time-guarded, because taking the newest file unconditionally lets
    several runs report the same stale trace. A log is this run's only if its time
    matches the run's console.
    """
    f = os.path.join(run_dir, "run.jsonl")
    if not os.path.exists(f):
        console = os.path.join(run_dir, "console.txt")
        if not os.path.exists(console):
            return []
        t0 = os.path.getmtime(console) - 600          # console is written as the run ends
        cands = [g for g in glob.glob(os.path.join(
                    os.path.dirname(PKG), "dgppo_ros_node_pkg", "dgppo_ros_node_pkg",
                    "debug_logs", "carla_mpc_*.jsonl"))
                 if os.path.getmtime(g) >= t0]
        if not cands:
            return []
        # MATCH BY LOG START TIME, not nearest mtime. The filename encodes when the log
        # started; mtime is when it stopped, and a log stops ~2.5 min before its run's
        # console is written because teardown follows the MPC. Nearest-mtime therefore
        # prefers the NEXT run's log. Any other tool that matches logs to runs must use
        # the same rule, or the tools disagree about the same run.
        import time as _t
        def _started(g):
            m = re.search(r"(\d{8}_\d{6})", os.path.basename(g))
            return (_t.mktime(_t.strptime(m.group(1), "%Y%m%d_%H%M%S")) if m
                    else os.path.getmtime(g))
        t_end = os.path.getmtime(console)
        before = [g for g in cands if _started(g) <= t_end]
        f = (max(before, key=_started) if before
             else min(cands, key=lambda g: abs(os.path.getmtime(g) - t_end)))
    out = []
    for line in open(f):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def _text(run_dir: str, *names: str) -> str:
    buf = []
    for n in names:
        p = os.path.join(run_dir, n)
        if os.path.exists(p):
            buf.append(open(p, errors="ignore").read())
    return "\n".join(buf)


def classify(run_dir: str) -> tuple[str, str]:
    """Return (category, one-line evidence)."""
    txt = _text(run_dir, "out.txt", "console.txt")
    rows = _rows(run_dir)

    # -- INFRA ----------------------------------------------------------------
    if "GENERATE FAILED" in txt:
        g = re.search(r"gates=\[[^\]]*\]", txt)
        return "PLAN", f"generation never produced a plan {g.group(0) if g else ''}"
    if "not adjacent to start_region" in txt:
        return "INFRA", "spawn rejected: start_toward is not adjacent to start_region"
    if "off-by" in txt and re.search(r"off-by\s+1[0-9]{2}", txt):
        m = re.search(r"off-by\s+([0-9.]+) deg", txt)
        return "INFRA", f"spawn heading {m.group(1)}deg from the lane — one-way the other way"
    m_ab = re.search(r"\d+ packages? aborted: ([\w ]+)", txt)
    if m_ab:
        pkgs = m_ab.group(1).split()
        # NAME THE PACKAGES, and do not over-claim; this is a warning, not INFRA.
        # `colcon build --symlink-install` links the install tree at the SOURCE
        # (egg-link -> build/<pkg> -> symlink -> src/<pkg>/<pkg>), so a pure-Python edit is
        # live whether or not the build completed; entry-point scripts are regenerated
        # separately. An abort only invalidates a run when the aborted package is one the
        # run depends on. Treating it as INFRA would misfile valid (mostly CONTROL) runs,
        # so INFRA is reserved for runs that also produced no telemetry.
        if not rows:
            return "INFRA", (f"colcon aborted: {', '.join(pkgs)} AND no telemetry — "
                         f"testing reached the container before discarding the run")
    # A NODE THAT DIED AT STARTUP. E.g. gt_cue_node raising InvalidParameterTypeException
    # on a launch override dies and publishes no cues, so every world drives identically.
    # That reads as a clean non-discrimination result when nothing was measured.
    m = re.search(r"\[([a-z_]+)-\d+\] Traceback", txt)
    if m or "process has died" in txt:
        who = m.group(1) if m else "a node"
        err = re.search(r"(\w*(?:Exception|Error))", txt.split("Traceback")[-1]) if m else None
        return "INFRA", (f"{who} DIED at startup"
                         + (f" ({err.group(1)})" if err else "")
                         + " — nothing it publishes was available to this run")
    if not rows:
        return "INFRA", "no MPC telemetry attributable to this run"

    # -- COMPLETED ------------------------------------------------------------
    # CHECKED BEFORE THE STUCK TEST, and that ordering is the whole point. A vehicle that
    # finished its plan holds a stop for the rest of the budget, so by displacement alone
    # it is indistinguishable from one wedged against a building -- both command a little
    # throttle and move nothing, so a completed, parked run would be misclassified as
    # CONTROL/wedged. Note the completion line is printed by the MPC to the console, not
    # to mission.log.
    if "RESULT  COMPLETED" in txt or "brain reports plan complete" in txt:
        return "COMPLETED", (f"{len(rows)} ticks"
                             + ("; MPC held stop after completion"
                                if "brain reports plan complete" in txt else ""))

    # -- CONTROL: commanded but not moving ------------------------------------
    xs = [(r.get("t_sim"), r.get("x"), r.get("y"), r.get("v_cmd"))
          for r in rows if r.get("x") is not None and r.get("t_sim") is not None]
    if len(xs) > 10:
        t0 = xs[-1][0]
        win = [p for p in xs if p[0] >= t0 - STUCK_WINDOW_S]
        if len(win) > 5:
            moved = math.dist((win[0][1], win[0][2]), (win[-1][1], win[-1][2]))
            cmds = [p[3] for p in win if p[3] is not None]
            mean_cmd = sum(cmds) / len(cmds) if cmds else 0.0
            if mean_cmd > CMD_MIN and moved < MOVED_MIN:
                last = rows[-1]
                return "CONTROL", (f"commanded {mean_cmd:.1f} m/s, moved {moved:.2f} m in the "
                                   f"last {STUCK_WINDOW_S:.0f}s of sim, wedged in cluster "
                                   f"{last.get('gt_cluster')} (edt_blocked="
                                   f"{last.get('edt_blocked')}, n_hits={last.get('n_hits')})")

    # -- PERCEPTION -----------------------------------------------------------
    n_branch = txt.count("no branch answer yet")
    if n_branch:
        return "PERCEPTION", (f"{n_branch}x 'no branch answer yet' — a branch cue matched no "
                              f"published key, so the navigator never descended")
    m = re.search(r"CUE TIMEOUT after \d+s waiting for '([^']+)' \((\d+) sighting", txt)
    if m and m.group(2) == "0":
        return "PERCEPTION", f"cue {m.group(1)!r} never answered (0 sightings)"

    # -- BUDGET ---------------------------------------------------------------
    return "BUDGET", (f"still moving when the run ended: {len(rows)} ticks, "
                      f"last cluster {rows[-1].get('gt_cluster')}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="*")
    ap.add_argument("--all", action="store_true", help="every run dir under reports/runs")
    a = ap.parse_args(argv)

    dirs = a.dirs or []
    if a.all or not dirs:
        dirs = sorted(glob.glob(os.path.join(PKG, "reports", "runs", "*", "*/")))
    dirs = [d for d in dirs if os.path.isdir(d)]

    tally: dict[str, int] = {}
    print(f"{'run':<44}{'cause':<12} evidence")
    print("-" * 118)
    for d in dirs:
        cat, why = classify(d)
        tally[cat] = tally.get(cat, 0) + 1
        name = "/".join(d.rstrip("/").split("/")[-2:])
        print(f"{name[:43]:<44}{cat:<12} {why[:60]}")
    print("-" * 118)
    print("  " + "   ".join(f"{k} {v}" for k, v in sorted(tally.items(), key=lambda x: -x[1])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
