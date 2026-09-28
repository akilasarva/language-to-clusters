#!/usr/bin/env python3
"""Replay a trace through the monitors, offline, with no simulator and no CARLA.

WHY THIS EXISTS. Plan-generation harnesses score whether a *plan was produced*, not
whether a *rule was obeyed*. Plan validity is structurally blind to the latter: a plan
that silently drops "never cross the grass" is a perfectly valid plan, and every
validator passes it.

The monitors are reusable for this directly. `brain.trigger_policy` and `brain.step_advancer` are ROS-free and
per-observation — `StepAdvancer.observe(cluster=, yaw=, cue_answer=)`,
`InvariantMonitor.observe(cluster, accept)`, `ConstraintMonitor.observe(cluster)`. A
trace is therefore just a list of ``(cluster_id, yaw, cue_found)``. Nothing needs
simulating to ask "would this configuration have caught that violation?".

FORMULAS CAN BE EXECUTED OFFLINE. `stl_compile.compile_monitors()` turns a formula into
a `MonitorSpec`, so a formula can be *executed* against a trace here without touching
`NavPlan`, `planner_node`, or brain -- which answers "does the formula do something the
tree cannot" on recorded traces.

WHAT IT CANNOT DO. A replayed trace is the outcome of whatever controller produced it.
Feeding it back answers "would a different monitor configuration have advanced, or
flagged, differently on this drive". It cannot answer "would the vehicle have driven
somewhere else" — for counterfactual trajectories you still want `missions.simulate`,
which closes that loop with a real unicycle integrator.

    python3 replay_monitors.py --self-test
    python3 replay_monitors.py --jsonl <a recorded carla_mpc_*.jsonl>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

WS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (os.path.join(WS, "brain"), os.path.join(WS, "nl_planner"),
           os.path.join(WS, "carla_gt_bridge", "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from brain.plan_navigator import PlanNavigator                     # noqa: E402
from brain.step_advancer import StepAdvancer                       # noqa: E402
from brain.trigger_policy import (ConstraintMonitor,               # noqa: E402
                                  InvariantMonitor, accept_set)


# --------------------------------------------------------------------------- #
# The trace                                                                   #
# --------------------------------------------------------------------------- #

@dataclass
class Obs:
    """One tick. Exactly what the monitors consume — nothing else is needed."""

    cluster: int
    yaw: float | None = None
    cue: bool | None = None


@dataclass
class ReplayResult:
    """What the replay saw. Shaped to feed `run_scoring.score` directly."""

    ok: bool = False
    reason: str = ""
    regions_visited: list[int] = field(default_factory=list)
    trace: list[tuple] = field(default_factory=list)
    trace_times: list[float] = field(default_factory=list)
    advances: list[str] = field(default_factory=list)
    steps_completed: int = 0
    n_steps: int = 0
    #: Which branch was taken at each decision, in order. This is the ONLY way a
    #: trace-driven replay can show that a plan behaved differently: the observed
    #: regions are fixed by the trace, so a plan cannot change where the robot went,
    #: only which of its own paths it walked.
    branch_path: list[int] = field(default_factory=list)

    # the enforcement verdicts — the thing nothing else measures
    forbid_declared: bool = False
    forbid_violations: int = 0
    csr: float = 1.0
    require_declared: bool = False
    require_breached: bool = False
    hold_breached: bool = False

    @property
    def flagged(self) -> bool:
        """True when ANY constraint the plan declared was observed to be broken.

        Independent of `ok`: a run can complete its route and still violate an
        invariant on the way, which route/region-sequence scoring cannot see.
        """
        return bool(self.forbid_violations or self.require_breached or self.hold_breached)

    def summary(self) -> dict:
        return {
            "ok": self.ok, "reason": self.reason, "flagged": self.flagged,
            "regions_visited": self.regions_visited,
            "steps_completed": self.steps_completed, "n_steps": self.n_steps,
            "forbid_declared": self.forbid_declared,
            "forbid_violations": self.forbid_violations, "csr": round(self.csr, 4),
            "require_declared": self.require_declared,
            "require_breached": self.require_breached,
            "hold_breached": self.hold_breached,
            "branch_path": self.branch_path,
        }


# --------------------------------------------------------------------------- #
# The replay                                                                  #
# --------------------------------------------------------------------------- #

def replay(tree: dict, obs: Iterable[Obs], *, dt: float = 0.1,
           dwell_frames: int = 3, hold_dwell_frames: int = 5,
           degraded: bool = True) -> ReplayResult:
    """Walk ``obs`` through the monitor configuration in ``tree``.

    ``tree`` is what `branch_materializer.to_brain_tree` emits — the same dict brain
    loads. That is deliberate: replaying anything else would test a reconstruction
    rather than the artifact the robot actually runs.

    ``degraded`` picks which accept set the step guards use, matching brain's default.
    The constraint monitors use the STRICT sets, because a degraded set on a negative
    constraint would forbid half the map.
    """
    steps = tree.get("steps") or []
    res = ReplayResult(n_steps=len(steps))
    if not steps:
        res.reason = "plan has no steps"
        return res

    nav = PlanNavigator(steps)
    adv = StepAdvancer(dwell_frames=dwell_frames)

    forbid = {int(c) for c in (tree.get("forbid_clusters") or ())}
    require = {int(c) for c in (tree.get("require_clusters") or ())}
    constraint = ConstraintMonitor(forbid)
    # `dwell` rather than `off`: this harness exists to MEASURE, and `off` does not
    # even count. brain's own two live constructions also pass "dwell".
    require_mon = InvariantMonitor(policy="dwell", dwell_frames=hold_dwell_frames)
    hold_mon: InvariantMonitor | None = None
    hold_accept: set[int] = set()

    res.forbid_declared = bool(forbid)
    res.require_declared = bool(require)

    def _begin(step, o: Obs) -> None:
        nonlocal hold_mon, hold_accept
        adv.begin_step(step, cluster=o.cluster, yaw=o.yaw)
        hold_mon, hold_accept = None, set()
        if step and step.get("hold_mode"):
            hold_accept = {int(c) for c in (step.get("hold_accept_clusters") or ())}
            if hold_accept:
                hold_mon = InvariantMonitor(policy="dwell", dwell_frames=hold_dwell_frames)

    obs = list(obs)
    if not obs:
        res.reason = "empty trace"
        return res
    _begin(nav.current_step, obs[0])

    for i, o in enumerate(obs):
        res.trace.append((float(i), 0.0, o.cluster, nav.step_idx, None))
        res.trace_times.append(i * dt)
        if not res.regions_visited or res.regions_visited[-1] != o.cluster:
            res.regions_visited.append(o.cluster)

        # constraints first: a violation counts on the tick it happens, whether or not
        # the step advances on the same tick
        if constraint.observe(o.cluster):
            res.forbid_violations += 1
        if require:
            require_mon.observe(o.cluster, require, binding=True)
        step = nav.current_step
        if hold_mon is not None and step is not None:
            hold_mon.observe(o.cluster, hold_accept, binding=True)

        if nav.is_complete or step is None:
            continue

        d = adv.observe(cluster=o.cluster, yaw=o.yaw, cue_answer=o.cue)
        if not d.advanced:
            continue
        res.advances.append(f"t{i} step{nav.step_idx} in {o.cluster}: {d.reason}")
        res.steps_completed += 1
        if d.branches:
            pick = _pick_branch(d.branches, o)
            res.branch_path.append(pick)
            nav.descend(pick)
        else:
            nav.advance()
        _begin(nav.current_step, o)

    res.csr = constraint.csr
    res.require_breached = bool(require and require_mon.breached)
    res.hold_breached = bool(hold_mon is not None and hold_mon.breached)
    res.ok = nav.is_complete
    res.reason = ("complete" if res.ok else
                  f"ran out of trace at step {nav.step_idx} of {res.n_steps}")
    return res


def _pick_branch(branches: Sequence[dict], o: Obs) -> int:
    """Choose a branch from the observation.

    The cue answer decides when there is one; otherwise the default. Replay cannot ask
    a VLM, so a trace that needs a branch decision must carry the cue.
    """
    for i, b in enumerate(branches):
        if b.get("vlm_cue") and o.cue is True:
            return i
    for i, b in enumerate(branches):
        if b.get("default"):
            return i
    return 0


# --------------------------------------------------------------------------- #
# Recorded CARLA runs                                                         #
# --------------------------------------------------------------------------- #

def obs_from_jsonl(path: str, cue: bool | None = None) -> list[Obs]:
    """Read a recorded `carla_mpc_*.jsonl` into observations.

    The MPC logs `gt_cluster` and `yaw_deg` per tick. It does NOT log cue answers —
    `/cue/confirmations` is published but never recorded — so cue-driven steps cannot be
    replayed faithfully from a bag today. Rather than invent an answer, ``cue`` is passed
    through explicitly and defaults to None (unknown), which leaves landmark steps
    un-advanced instead of falsely advanced.
    """
    out: list[Obs] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            c = row.get("gt_cluster")
            if c is None:
                continue
            yaw = row.get("yaw_deg")
            out.append(Obs(int(c),
                           math.radians(yaw) if yaw is not None else None,
                           cue))
    return out


# --------------------------------------------------------------------------- #
# Self-test — the discriminator                                               #
# --------------------------------------------------------------------------- #

def _tree(forbid=(), require=(), hold=None, steps=None) -> dict:
    steps = steps or [dict(step=1, description="go", trigger="traverse",
                           goal_cluster=2, accept_clusters=[2],
                           accept_clusters_degraded=[2],
                           start_mode="path", goal_mode="path")]
    t = dict(plan_name="t", steps=steps)
    if forbid:
        t["forbid_clusters"] = list(forbid)
    if require:
        t["require_clusters"] = list(require)
    if hold:
        t["steps"][0].update(hold)
    return t


def self_test() -> int:
    """A satisfying trace must PASS and a violating trace must FLAG.

    If this cannot distinguish the two, the harness measures nothing — so this is a
    correctness gate, not a smoke test.
    """
    checks: list[tuple[str, bool]] = []

    def chk(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))

    # -- negative constraint: never enter cluster 9 ------------------------ #
    tree = _tree(forbid=[9])
    clean = replay(tree, [Obs(1), Obs(1), Obs(1), Obs(2), Obs(2), Obs(2)])
    dirty = replay(tree, [Obs(1), Obs(9), Obs(9), Obs(2), Obs(2), Obs(2)])
    chk("forbid: clean trace not flagged", not clean.flagged)
    chk("forbid: violating trace FLAGGED", dirty.flagged)
    chk("forbid: violation counted", dirty.forbid_violations == 2)
    chk("forbid: csr below 1", dirty.csr < 1.0)
    chk("forbid: csr exactly 1 when clean", clean.csr == 1.0)
    # the key case: the route still completes while violating
    chk("forbid: a VIOLATING run still completes its route", dirty.ok)

    # -- positive invariant: never leave {1,2} ----------------------------- #
    tree = _tree(require=[1, 2])
    clean = replay(tree, [Obs(1)] * 4 + [Obs(2)] * 4)
    dirty = replay(tree, [Obs(1)] * 4 + [Obs(7)] * 8 + [Obs(2)] * 4)
    chk("require: clean trace not flagged", not clean.flagged)
    chk("require: leaving the required set FLAGGED", dirty.require_breached)

    # -- undeclared constraints stay vacuous ------------------------------- #
    none_declared = replay(_tree(), [Obs(1), Obs(9), Obs(2), Obs(2), Obs(2)])
    chk("undeclared: not flagged", not none_declared.flagged)
    chk("undeclared: reported as undeclared", not none_declared.forbid_declared)

    # -- a formula, compiled and executed, with no plan field set ---------- #
    # A formula-only constraint, executed without any plan field.
    from nl_planner.stl_compile import compile_monitors, parse
    from nl_planner.taxonomy import load_taxonomy
    tax = load_taxonomy(os.path.join(WS, "carla_gt_bridge", "config",
                                     "cluster_map.carla_town01.yaml"))
    spec = compile_monitors(parse(r"\mathbf{G} \lnot \Phi_{Junc\_Pass}")[0], tax)
    chk("stl: G!Phi compiles to a forbid on a CAR taxonomy",
        spec.forbid_modes == ["junction"])
    forbid_ids = set()
    for m in spec.forbid_modes:
        forbid_ids |= set(tax.resolve(m))
    chk("stl: the mode resolves to cluster ids", bool(forbid_ids))
    victim = sorted(forbid_ids)[0]
    ftree = _tree(forbid=sorted(forbid_ids))
    fclean = replay(ftree, [Obs(1)] * 3 + [Obs(2)] * 3)
    fdirty = replay(ftree, [Obs(1), Obs(victim), Obs(victim)] + [Obs(2)] * 3)
    chk("stl: formula-derived constraint passes a clean trace", not fclean.flagged)
    chk("stl: formula-derived constraint FLAGS a violating trace", fdirty.flagged)

    width = max(len(n) for n, _ in checks)
    print("REPLAY SELF-TEST — can the harness tell a violation from a clean run?\n")
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<{width}}")
    bad = [n for n, ok in checks if not ok]
    print(f"\n  {len(checks) - len(bad)}/{len(checks)} checks pass")
    if bad:
        print("  FAILED: " + "; ".join(bad))
        return 1
    print("\n  A violating trace is detected and a clean one is not.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                    help="prove the harness can distinguish a violation from a clean run")
    ap.add_argument("--jsonl", help="replay a recorded carla_mpc_*.jsonl")
    ap.add_argument("--plan", help="brain-tree JSON to replay it against")
    ap.add_argument("--cue", choices=["true", "false", "unknown"], default="unknown",
                    help="cue answer to assume; bags do not record /cue/confirmations")
    a = ap.parse_args()

    if a.self_test:
        return self_test()

    if not a.jsonl:
        ap.error("give --self-test or --jsonl")
    cue = {"true": True, "false": False, "unknown": None}[a.cue]
    obs = obs_from_jsonl(a.jsonl, cue)
    if not obs:
        print(f"no usable rows in {a.jsonl} (no gt_cluster field?)")
        return 1
    tree = json.load(open(a.plan)) if a.plan else _tree()
    res = replay(tree, obs)
    print(f"{os.path.basename(a.jsonl)}: {len(obs)} ticks")
    print(json.dumps(res.summary(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
