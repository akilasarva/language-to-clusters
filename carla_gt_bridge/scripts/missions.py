#!/usr/bin/env python3
"""Closed-loop offline simulation of a brain tree, and ego-spawn helpers. No CARLA.

``--simulate PLAN``  close the loop offline: ground-truth clusters from the region table,
                plan advancement from ``brain.trigger_policy`` (which is ROS-free by
                design), targets from ``carla_gt_bridge.routing``, control from
                ``dgppo_ros_node_pkg.sampling_mpc``, and a unicycle integrator standing
                in for the vehicle.

``spawn_seed`` / ``spawn_config`` / ``carla_pose_of`` derive the ego spawn for a start
region; run_phase_a.sh and drive_english.py use them.

Why the simulation is worth having rather than just starting CARLA
-----------------------------------------------------------------
It exercises every piece of the chain except the simulator itself, in about a second,
with no GPU. If a plan fails here it fails for a reason visible in the log -- wrong target
region, mirrored frame, a step that can never advance. If it passes here and fails in
CARLA, the fault is in the bridge, the vehicle dynamics or the topic wiring.

It is NOT a substitute for CARLA: a unicycle is not a car, there are no obstacles, and
the ground truth is exact. It proves the plan and the geometry agree, nothing more.

Usage:
  python3 scripts/missions.py --simulate plan.json [--cone-regions 63,60] [--plot]
  python3 scripts/missions.py --spawn
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS = os.path.dirname(PKG)
for p in (PKG, os.path.join(WS, "nl_planner"), os.path.join(WS, "brain"),
          os.path.join(WS, "dgppo_ros_node_pkg")):
    sys.path.insert(0, p)

CLUSTER_MAP = os.path.join(PKG, "config", "cluster_map.carla_town05.yaml")
REGIONS_NPZ = os.path.join(PKG, "config", "regions.town05.npz")

#: Where the ego starts. Must match the bridge's spawn configuration, since this node no
#: longer teleports (a teleport is how a run "completes" without driving).
#:
#: The corridor comes from `pick_corridor()` over the full 74-region map, which scores
#: candidates on geometry and requires every non-junction region on the route to be
#: labelled `path`:
#:
#:     0 -> J63 (straight +2 deg) -> 1 -> J60 -> right 49 (-90) | straight 2 (+0)
#:
#: J60 is a clean four-way. The corridor is derived rather than hand-picked (a
#: hand-picked corridor on a pruned region table can lose a junction's straight exit),
#: and is re-derivable with `--town town05 --list`.
START_REGION = 0

#: WHICH neighbour it faces, named rather than derived. `sorted(adj[START_REGION])[0]`
#: is correct only while the start region has one neighbour; with several it returns the
#: lowest id and can spawn the ego facing AWAY from the corridor. That error only shows
#: up in CARLA, which is why it must not be derived from an ordering.
START_TOWARD = 63


# --------------------------------------------------------------------------- #
# the three missions                                                          #
# --------------------------------------------------------------------------- #
#
# Each is a strict superset of mechanism over the last, following the trigger typology:
# traverse -> topology -> landmark + branch. All three are written in the PEDESTRIAN
# vocabulary (`path`, `junction`), because the vocabulary is geometric and therefore
# agent-agnostic: a driving lane is a `path`, an intersection is a `junction`.
#
# Note what makes the medium and hard missions differ from the easy one at the plan level:
# the extra path->junction->path traversals. "the SECOND intersection" is expressed by
# UNROLLING, not by naming an id, which is what lets the same plan shape run on the campus
# bags where there are no stable intersection ids at all.

#: Directory holding plan trees for `simulate(name)` (``mission.<name>.json``). Unset by
#: default: pass `plan_path`, or point this at a directory of trees (the tests use
#: test/fixtures/executor_plans).
PLAN_DIR: str | None = None


def _misclassify(rid: int, adj: dict, labels: dict, rng, mode: str = "flip") -> int:
    """Report a neighbouring region instead of the true one.

    TWO MODES, because they test opposite things and a curve that mixes them hides the
    actual mechanism.

    ``miss_junction`` corrupts ONLY a true junction, reporting an adjacent path. This is
    a MISSED DETECTION, and it is what the acceptance sets exist to survive: the fine
    classifier does not fire, so a landmark step falls back to the coarser cluster and
    proceeds on the Detect(...) cue. It also mirrors the dominant real-perception error:
    junctions being called path.

    ``flip`` corrupts any region symmetrically. That injects FALSE POSITIVES as well, and
    acceptance sets make those WORSE, not better: a wider accept set is easier to satisfy
    spuriously, so a step can advance somewhere it never arrived. Reporting one blended
    curve would let the mechanism's cost hide inside its benefit.

    Models the confusion that actually occurs — a boundary mistake between adjacent
    regions — rather than uniform random noise over the whole map, which no classifier
    makes and which no grounding rule could survive.

    PREFERS a neighbour whose LABEL DIFFERS, because junction-vs-path is the distinction
    the whole mode vocabulary rests on and the weakest one in real perception. A
    confusion that keeps the label is much less damaging and would understate the
    difficulty.
    """
    if mode == "miss_junction" and labels.get(rid) != "junction":
        return rid
    nbrs = sorted(adj.get(rid, ()))
    if not nbrs:
        return rid
    flipped = [n for n in nbrs if labels.get(n) != labels.get(rid)]
    pool = flipped or nbrs
    return int(pool[rng.integers(len(pool))])


def _strip_accept_sets(steps: list, *, drop_fine: bool) -> list:
    """Build a BASELINE by removing data, never by adding a second code path.

    `brain.trigger_policy.accept_set` reads `accept_clusters_degraded` when degradation
    is allowed and `accept_clusters` otherwise, falling back to `{goal_cluster}` when
    neither is present — its docstring notes that fallback "reproduces the historical
    equality test exactly". So each condition is the SAME executor on different plan
    data, and no baseline can silently drift from the system under test.

    ``drop_fine=False`` -> `no_degrade`: the mode's own cluster set, no coarser fallback.
    ``drop_fine=True``  -> `strict_eq`:  hard `cluster == goal_cluster`, the deployed rule.
    """
    import copy
    out = []
    for st in copy.deepcopy(steps):
        st.pop("accept_clusters_degraded", None)
        if drop_fine:
            st.pop("accept_clusters", None)
        for b in (st.get("branches") or []):
            b["sub_plan"] = _strip_accept_sets(b.get("sub_plan") or [], drop_fine=drop_fine)
        out.append(st)
    return out


def simulate(mission: str, branch: str = "right", max_ticks: int = 900,
             verbose: bool = False, plan_path: str | None = None,
             centroid_offset: tuple[float, float] = (0.0, 0.0),
             guidance: str = "centroid",
             scale: float = 1.0, rot_deg: float = 0.0,
             regions_npz: str | None = None,
             start_region: int | None = None,
             start_toward: int | None = None,
             cone_regions=None,
             cue_timeout_ticks: int = 150,
             invariant_policy: str = "off",
             invariant_dwell: int = 5,
             forbid_policy: str = "log",
             cluster_noise: float = 0.0,
             noise_seed: int = 0,
             noise_mode: str = "flip",
             grounding: str = "ours",
             cue_semantics: str | None = None):
    """Drive the mission with a unicycle. Returns a dict of results.

    ``cue_semantics`` (default: env CUE_SEMANTICS, else "legacy"). "v2" mirrors brain's v2: a landmark
    step's cue is counted whenever the vehicle stands in a region of the step's GOAL MODE -- including
    the region the step started in -- and only there; an ordinal count starts after the last step that
    carried a cue or manoeuvre; and an unmet cue never times the mission out while there is a further
    region of that mode to reach (the vehicle keeps driving to it, as brain + the MPC do).

    ``branch`` describes the WORLD, not the decision: "straight" means a cone is present.
    Which branch gets taken is then derived from the cue, the same way brain derives it,
    rather than asserted here. Naming the outcome instead of the cause would skip the
    decision entirely.

    ``regions_npz`` / ``start_region`` / ``cone_regions`` default to the Town05 corridor
    constants, so every existing call — including the distortion ablation — behaves
    EXACTLY as before. They exist so a second town can be driven
    without forking this loop: the loop is town-agnostic, only the constants were not.

    ``forbid_policy`` is what the plan-level ``[]~X`` constraint COSTS, and it is one
    knob for the same reason ``InvariantMonitor``'s policy is: detection is identical
    either way, only the response differs.

    ``log``   (default) count the ticks spent inside ``forbid_clusters`` and report
              ``csr`` / ``violation_ticks``. Changes no outcome; a run with no
              ``forbid_clusters`` reports ``csr=1.0`` vacuously.
    ``fail``  abort the run the first tick the constraint is violated.

    Default is ``log`` (measure before enforcing): with noisy junction-vs-path labels, a
    hard abort fires on perception noise, not on the robot entering the plaza. Measure
    the rate, then choose.
    """
    regions_npz  = regions_npz or REGIONS_NPZ
    start_region = START_REGION if start_region is None else start_region
    # `cone_regions` names the regions holding a cone; without it, `branch` just says
    # whether one cone is present at the decision junction.
    if cone_regions is not None:
        cone_present = set(cone_regions)
    else:
        cone_present = branch == "straight"
    import numpy as np
    from brain.trigger_policy import (InvariantMonitor, StepProgress, accept_set,
                                      goal_reached, trigger_of)
    from carla_gt_bridge.region_lookup import load_region_table
    from carla_gt_bridge.routing import StepTargeter, adjacency_from_bearing_map
    from dgppo_ros_node_pkg.sampling_mpc import MpcConfig, RoadSurface, plan_step

    if plan_path is None:
        if PLAN_DIR is None:
            raise ValueError("simulate() needs plan_path (or missions.PLAN_DIR set)")
        plan_path = os.path.join(PLAN_DIR, f"mission.{mission}.json")
    tree = json.load(open(plan_path))
    table = load_region_table(regions_npz)
    bearing_map = tree["bearing_map"]
    labels = {int(k): v for k, v in tree["cluster_labels"].items()}
    for rid in table.region_ids:                    # finest label wins, as brain's does
        labels[rid] = table.label_of(rid)
    adj = adjacency_from_bearing_map(bearing_map)
    ids = table.region_ids
    cents = np.array([table.centroid_of(r) for r in ids], dtype=float)
    # Deliberately corrupt the geometry while leaving cluster membership intact. This is
    # the discriminating experiment: centroids are DERIVED and may be miscalibrated, while
    # the cluster a pose is in is PERCEIVED and accurate. A controller that steers only by
    # centroid distance inherits the error; one constrained by region membership should
    # still complete the mission.
    # A map distortion is a SIMILARITY TRANSFORM, not just a shift: a miscalibrated map
    # is typically rotated and rescaled about some origin as well as translated. Applied
    # about the corridor's own centre so a pure rotation does not also translate it.
    pivot = cents.mean(axis=0)
    th = math.radians(rot_deg)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    cents = (cents - pivot) @ R.T * scale + pivot + np.asarray(centroid_offset, float)

    # -- start pose: at region 45's centroid, aimed at its only neighbour ------
    x, y = table.centroid_of(start_region)
    # Aim at the neighbour the route actually goes to. `sorted(...)[0]` is safe only
    # when the start region has exactly ONE neighbour — with two, the lowest id can
    # point the car AWAY from the corridor, and the scenario times out at step 0
    # without moving. The default keeps Town05 unchanged.
    nxt = start_toward if start_toward is not None else sorted(adj[start_region])[0]
    if nxt not in adj[start_region]:
        raise ValueError(
            f"start_toward={nxt} is not adjacent to start_region={start_region} "
            f"(neighbours: {sorted(adj[start_region])})"
        )
    nx, ny = table.centroid_of(nxt)
    yaw = math.atan2(ny - y, nx - x)

    noise_rng = np.random.default_rng(noise_seed)
    steps = list(tree["steps"])
    if grounding == "strict_eq":
        steps = _strip_accept_sets(steps, drop_fine=True)
    elif grounding == "no_degrade":
        steps = _strip_accept_sets(steps, drop_fine=False)
    elif grounding != "ours":
        raise ValueError(
            f"grounding must be 'ours' | 'no_degrade' | 'strict_eq', got {grounding!r}")
    step_i = 0
    # require_cluster_change: a step must LEAVE the region it began in. Without it the
    # taxonomy's upward subsumption (a junction is also a path) lets `junction -> path`
    # be satisfied without moving (several steps advance within a few ticks inside the
    # first junction). Off by default in trigger_policy; on here.
    progress = StepProgress(require_cluster_change=True, start_cluster=start_region)
    # The left half of `Phi_X U cue`: is the step's mode still holding? Default
    # "off" is a true no-op. See
    # brain.trigger_policy.InvariantMonitor for why measuring precedes enforcing.
    invariant = InvariantMonitor(policy=invariant_policy, dwell_frames=invariant_dwell)
    inv_log: list[dict] = []
    # `Phi_X U cue` — the LEFT half, enforced. The monitor above is the
    # reporting one bound to the step's GOAL mode; this one is bound to the mode a
    # dwell step declares it will hold, and it is `dwell`/binding unconditionally
    # because the plan asked for the invariant by writing `hold_mode` at all. A
    # step that says "stay on the road" and is not held to it would make the field
    # decoration.
    hold_monitor: "InvariantMonitor | None" = None
    # `[]~X` — the plan-level negative constraint. Resolved by the materializer, so
    # a tree that declares none gives an empty set and the whole feature costs one
    # set-membership test per tick that can never be true.
    if forbid_policy not in ("log", "fail"):
        raise ValueError(f"forbid_policy must be 'log' | 'fail', got {forbid_policy!r}")
    forbid_clusters = {int(c) for c in (tree.get("forbid_clusters") or ())}
    forbid_ticks = 0            # ticks spent INSIDE the forbidden set
    observed_ticks = 0          # ticks with a region reading at all — csr's denominator
    forbid_regions: list[int] = []

    def _csr() -> dict:
        """CSR, and the two ways it can be 1.0 — which are not the same claim.

        No constraint declared is VACUOUS satisfaction; a declared constraint never
        entered is EARNED. `run_scoring.score` needs to tell them apart, so
        `forbid_declared` is reported next to the rate rather than left to be
        inferred from `csr == 1.0`.
        """
        rate = 1.0 if observed_ticks == 0 else 1.0 - forbid_ticks / observed_ticks
        return dict(csr=rate, violation_ticks=forbid_ticks,
                    constraint_ticks=observed_ticks,
                    forbid_clusters=sorted(forbid_clusters),
                    forbid_declared=bool(forbid_clusters),
                    forbid_regions=forbid_regions,
                    forbid_policy=forbid_policy)
    targeter = StepTargeter(adj, labels, bearing_map)
    # Seed the region BEHIND us before the one we are in, so `next_along`'s
    # "never turn back" rule has something to exclude on the FIRST step.
    #
    # Without this, `previous` is None at step 0 and BFS breaks ties by region id.
    # That is invisible when the goal mode is sparse (only one `junction` is within
    # a hop of the start, so there is no tie) and wrong when it is not: with
    # goal_mode `path` — which is what "follow the road and stop at the 2nd cone"
    # correctly produces — e.g. region 7's neighbours are {14: path, 30: junction}, and
    # the router can pick 14, which is behind the vehicle, sending the route backwards
    # out of the corridor.
    #
    # Same root cause as the start-heading issue above: the router must be told which
    # way the vehicle faces.
    behind = None
    _best = None
    for nb in adj.get(start_region, ()):
        bx, by = table.centroid_of(nb)
        d = abs(_wrap(math.atan2(by - y, bx - x) - yaw))
        if _best is None or d > _best:
            _best, behind = d, nb
    if behind is not None and _best is not None and _best > math.radians(90):
        targeter.observe(behind)
    targeter.observe(start_region)
    entry_yaw = yaw
    cfg = MpcConfig(guidance=guidance)
    # Same drivable surface the ROS node uses, so "stays on the road" is testable here
    # rather than only observable in CARLA.
    road = RoadSurface(table.waypoints)      # carries region ids too
    rng = np.random.default_rng(0)
    trace, path_len, advances = [], 0.0, []
    # Timestamps, so the M (motion-plausibility) guard can actually evaluate.
    # `run_scoring.motion_is_plausible` returns ok=True when times are missing, so without
    # them M would be an unconditional True in the pass conjunction ("absence of a check
    # is not a pass"). Uniform dt, because both integrator branches step by cfg.dt.
    trace_times: list[float] = []
    max_offroad = 0.0
    # ~30 s at 5 Hz by default. This is a HARNESS budget for "the cue should have
    # shown up by now", not a property of the plan — and it scales with how far apart
    # the sightings are. Town05's ordinal corridor fits inside 150 ticks; Town01's is
    # longer. Parameterised so a longer corridor, where the vehicle is still driving
    # toward the second cone, does not read as "a step that can never advance".
    cue_ticks = 0
    _cs = (cue_semantics or os.environ.get("CUE_SEMANTICS", "") or "legacy").strip().lower()
    v2 = _cs == "v2"
    win_start = 0          # index into `visited` where the current ordinal counting window begins
    v2_counted = set()     # regions already counted for THIS step (one sighting per distinct region)
    # Visited regions are tracked SEPARATELY from the trace. A tick that
    # completes a step `continue`s before appending, so the region the mission
    # finished in would never reach the trace — a reporting gap that reads exactly
    # like the controller taking the wrong exit.
    visited: list[int] = []

    for tick in range(max_ticks):
        if step_i >= len(steps):
            break
        step = steps[step_i]

        # -- perception, ground truth or corrupted -----------------------------
        rid = table.region_at(x, y)
        if rid is not None and cluster_noise > 0.0 and noise_rng.random() < cluster_noise:
            rid = _misclassify(rid, adj, labels, noise_rng, noise_mode)
        if rid is None:
            return dict(mission=mission, ok=False, reason="left the corridor",
                        ticks=tick, step=step_i, x=x, y=y, path_len=path_len,
                        trace=trace, trace_times=trace_times, advances=advances, regions_visited=visited,
                        **_csr())
        if targeter.observe(rid):
            progress.entry_yaw = yaw
        cur_region = rid
        if not visited or visited[-1] != rid:
            visited.append(rid)

        # -- []~X: is this tick inside a region the plan forbids? ---------------
        # Counted EVERY tick, not once per region entry, because the quantity that
        # matters is how long the constraint was violated, not how many times. A
        # vehicle that clips a forbidden corner for one tick and one that parks in
        # it for two hundred are the same event under a per-entry count.
        observed_ticks += 1
        if cur_region in forbid_clusters:
            forbid_ticks += 1
            if not forbid_regions or forbid_regions[-1] != cur_region:
                forbid_regions.append(cur_region)
            if forbid_policy == "fail":
                return dict(mission=mission, ok=False, regions_visited=visited,
                            reason=(f"forbidden region: entered {cur_region} which is "
                                    f"in forbid_clusters "
                                    f"{sorted(forbid_clusters)} — the plan says never"),
                            ticks=tick, step=step_i, path_len=path_len, trace=trace, trace_times=trace_times,
                            advances=advances, **_csr())

        # -- invariant: is the mode the step travels IN still holding? ----------
        # Measured against the step's own accept set, which brain already carries;
        # no schema change is needed to start collecting this.
        if invariant.observe(cur_region, accept_set(step, degraded=True)):
            return dict(mission=mission, ok=False, regions_visited=visited,
                        reason=(f"invariant breached: left {step.get('goal_mode')!r} "
                                f"for {invariant.longest_run} consecutive ticks"),
                        ticks=tick, step=step_i, path_len=path_len, trace=trace, trace_times=trace_times,
                        advances=advances, invariant=invariant.summary(),
                        invariant_per_step=inv_log, **_csr())

        # -- `Phi_X U cue`: a DWELL step, held and terminated ------------------
        # Both halves, in the one place that can run them: the mode must hold for
        # every tick of the step (left half), and the `until` cue ends it (right
        # half). Handled BEFORE `goal_reached` and not through it, because
        # `goal_reached` answers "have we arrived", and a dwell step is not going
        # anywhere — asking it produces an answer that is not wrong so much as
        # about a different question.
        hold_mode = step.get("hold_mode")
        if hold_mode:
            if hold_monitor is None:
                hold_monitor = InvariantMonitor(policy="dwell",
                                                dwell_frames=invariant_dwell)
            hold_accept = {int(c) for c in (step.get("hold_accept_clusters") or ())}
            if not hold_accept:
                # ABSENCE OF A CHECK IS NOT A PASS — the same rule
                # `motion_is_plausible` states for M. `InvariantMonitor.observe`
                # reads an empty accept set as "always inside", so an unresolved
                # hold would run to completion and report a perfectly held
                # invariant that was never tested. Only a hand-written tree can get
                # here; the materializer emits the ids alongside `hold_mode`.
                return dict(mission=mission, ok=False, regions_visited=visited,
                            reason=(f"step holds {hold_mode!r} but carries no "
                                    f"hold_accept_clusters — the invariant would "
                                    f"accept every region and never fire"),
                            ticks=tick, step=step_i, path_len=path_len, trace=trace, trace_times=trace_times,
                            advances=advances, **_csr())
            if hold_monitor.observe(cur_region, hold_accept, binding=True):
                return dict(mission=mission, ok=False, regions_visited=visited,
                            reason=(f"invariant breached: left {hold_mode!r} for "
                                    f"{hold_monitor.longest_run} consecutive ticks "
                                    f"(in region {cur_region}, accept "
                                    f"{sorted(hold_accept)})"),
                            ticks=tick, step=step_i, path_len=path_len, trace=trace, trace_times=trace_times,
                            advances=advances, invariant=hold_monitor.summary(),
                            invariant_per_step=inv_log, **_csr())
            until_cue = step.get("until") or ""
            done = cue_oracle(until_cue, labels.get(cur_region, ""),
                              cone_present, cur_region)
            why = (f"until: held {hold_mode!r}, cue {until_cue!r} -> {done}")
            if not done:
                # Same budget the landmark cue gets, and needed for the same reason:
                # `hold_mode` with an `until` nothing ever answers holds heading
                # forever, and "drove off the end of the corridor" is a much worse
                # log line than "this cue never fired".
                cue_ticks += 1
                if cue_ticks > cue_timeout_ticks:
                    return dict(mission=mission, ok=False, regions_visited=visited,
                                reason=(f"until cue {until_cue!r} never fired while "
                                        f"holding {hold_mode!r} — a dwell step that "
                                        f"can never terminate"),
                                ticks=tick, step=step_i, path_len=path_len,
                                trace=trace, trace_times=trace_times, advances=advances, **_csr())
            else:
                cue_ticks = 0
            if done:
                advances.append(dict(tick=tick, step=step_i, region=cur_region,
                                     why=why))
                if verbose:
                    print(f"    t{tick:4d} step {step_i} done in region "
                          f"{cur_region}: {why}")
                inv_log.append({"step": step_i, "hold_mode": hold_mode,
                                **hold_monitor.summary()})
                hold_monitor = None
                invariant.reset()
                step_i += 1
                progress = StepProgress(require_cluster_change=True,
                                        start_cluster=cur_region)
                entry_yaw = yaw
                continue
            # STILL DWELLING. No target is grounded — a dwell step has no
            # destination, and handing `StepTargeter` a `goal_mode` it was never
            # travelling to gets a confident answer to a question the plan did not
            # ask. Hold heading and keep moving, which is what the pure-decision
            # case does: the vehicle carries on doing what it was doing until the
            # cue says stop.
            v, omega = cfg.v_max, 0.0
            nxx = x + v * math.cos(yaw) * cfg.dt
            nyy = y + v * math.sin(yaw) * cfg.dt
            path_len += math.dist((x, y), (nxx, nyy))
            x, y = nxx, nyy
            yaw = _wrap(yaw + omega * cfg.dt)
            trace.append((x, y, cur_region, step_i, None))
            trace_times.append((len(trace) - 1) * cfg.dt)
            max_offroad = max(max_offroad, table.nearest(x, y)[1])
            continue

        # -- brain: has this step's goal been reached? --------------------------
        done, why = goal_reached(cur_region, step, progress)
        trig = trigger_of(step)

        if done and trig == "topology":
            # A topology step also needs the heading change. brain gets this from /odom;
            # here it comes from the integrator. Without it, "turn right" would be
            # satisfied by merely entering the next region, and the vehicle could
            # advance having gone straight.
            turned = abs(_wrap(yaw - entry_yaw))
            done = turned > math.radians(60.0)
            why += f" (turned {math.degrees(turned):.0f} deg)"

        if v2 and trig == "landmark" and step.get("transition_cue"):
            # v2: count in ANY region of the goal mode (the start region included), one sighting per
            # distinct region; done once the ordinal is met. No tick timeout: the targeter keeps driving
            # to the next region of the mode, and running out of such regions is the failure (below).
            needed = int(step.get("cue_ordinal") or 1)
            _scope_ok = step.get("goal_mode") != "junction" or labels.get(cur_region, "") == "junction"
            if _scope_ok and cur_region not in v2_counted:
                if cue_oracle(step["transition_cue"], labels.get(cur_region, ""), cone_present, cur_region):
                    v2_counted.add(cur_region); progress.sightings += 1
            done = progress.sightings >= needed and labels.get(cur_region, "") == step.get("goal_mode")
            why = f"landmark(v2): cue {step['transition_cue']!r} sighting {progress.sightings}/{needed}"
        elif done and trig == "landmark" and step.get("transition_cue"):
            # RUN THE CUE MACHINERY, do not assume it succeeds.
            #
            # goal_reached() returns True here with the reason "in strict accept set,
            # AWAITING CUE" — in brain that only opens CHECKING_CUE, and the step does not
            # advance until enough distinct sightings accumulate. Treating that True as
            # "step complete" would skip the ordinal counter, the de-bounce and the
            # timeout offline.
            #
            # Running them catches, offline, e.g. a mission with cue_ordinal=2 AND an
            # unrolled first intersection, which deadlocks in CARLA because sightings are
            # per-step and the cue never goes false once parked in the junction.
            answer = cue_oracle(step["transition_cue"], labels.get(cur_region, ""),
                                cone_present, cur_region)
            needed = int(step.get("cue_ordinal") or 1)
            done = progress.note_cue(answer, needed)
            why = (f"landmark: cue {step['transition_cue']!r} -> {answer}, "
                   f"sighting {progress.sightings}/{needed}")
            if not done:
                cue_ticks += 1
                if cue_ticks > cue_timeout_ticks:
                    return dict(mission=mission, ok=False, regions_visited=visited,
                                reason=(f"cue {step['transition_cue']!r} never reached "
                                        f"sighting {needed} "
                                        f"(stuck at {progress.sightings}) — a step that "
                                        f"can never advance"),
                                ticks=tick, step=step_i, path_len=path_len, trace=trace, trace_times=trace_times,
                                advances=advances, **_csr())
            else:
                cue_ticks = 0
        if done:
            advances.append(dict(tick=tick, step=step_i, region=cur_region, why=why))
            if verbose:
                print(f"    t{tick:4d} step {step_i} done in region {cur_region}: {why}")
            if step.get("branches"):
                chosen = _pick_branch(step, cone_present, labels.get(cur_region, ""),
                                      cur_region)
                steps = steps[:step_i + 1] + chosen + steps[step_i + 1:]
                if verbose:
                    print(f"           branch -> {[s['description'][:40] for s in chosen]}")
            inv_log.append({"step": step_i, **invariant.summary()})
            invariant.reset()
            if v2 and step.get("transition_cue"):
                win_start = max(len(visited) - 1, 0)      # the next ordinal count starts after this instruction
            step_i += 1
            progress = StepProgress(require_cluster_change=True,
                                    start_cluster=cur_region)
            v2_counted = set()
            if v2 and step_i < len(steps):
                nst = steps[step_i]
                if trigger_of(nst) == "landmark" and nst.get("transition_cue"):
                    # WINDOW: sightings of this cue at regions of the goal mode passed since the last
                    # cue/manoeuvre step (not counting where we stand -- that is counted live above).
                    past = [r for r in dict.fromkeys(visited[win_start:-1])
                            if nst.get("goal_mode") != "junction" or labels.get(r, "") == "junction"]
                    pre = [r for r in past if cue_oracle(nst["transition_cue"], labels.get(r, ""), cone_present, r)]
                    if pre:
                        progress.sightings = len(pre); v2_counted |= set(pre)
                        if verbose:
                            print(f"           v2 window: {len(pre)} earlier sighting(s) of {nst['transition_cue']!r} at {pre}")
            entry_yaw = yaw
            continue

        # -- ground the step's goal MODE onto a concrete region -----------------
        # The plan says "reach a junction"; which junction comes from the map. Note the
        # maneuver: a step whose cue is Bearing(Right) wants the right-hand exit, which is
        # how the branch decision becomes a geometric choice.
        cue = step.get("transition_cue") or ""
        maneuver = "right" if "Right" in cue else ("left" if "Left" in cue else "straight")
        target, note = targeter.target_for(step["goal_mode"], step_key=(step_i, id(step)),
                                           maneuver=maneuver)
        # Arrived, but the cue count is not met: keep going to the NEXT region of this
        # label instead of parking. This is what "the second cone" means, and without it
        # the vehicle sits at the first one forever.
        if (target is not None and cur_region == target
                and trig == "landmark" and step.get("transition_cue")
                and progress.sightings < int(step.get("cue_ordinal") or 1)):
            nxt, adv_note = targeter.advance_target(step["goal_mode"])
            if nxt is not None:
                target, note = nxt, adv_note
                if verbose:
                    print(f"    t{tick:4d} {adv_note}")
        if target is None:
            return dict(mission=mission, ok=False, reason=note,
                        ticks=tick, step=step_i, path_len=path_len, trace=trace, trace_times=trace_times,
                        advances=advances, regions_visited=visited, **_csr())

        # -- control + integrate ------------------------------------------------
        # start_id is the region we are steering FROM, which while dwelling in the target
        # is the previous one — the progress axis needs two distinct regions.
        start_id = (targeter.previous if cur_region == target and targeter.previous
                    is not None else cur_region)
        if start_id == target:
            start_id = cur_region
        res = plan_step(cfg, np.array([x, y]), yaw, cents, ids,
                        start_id=start_id, target_id=target, rng=rng, road=road,
                        via=targeter.via)
        nxx = x + res.v * math.cos(yaw) * cfg.dt
        nyy = y + res.v * math.sin(yaw) * cfg.dt
        path_len += math.dist((x, y), (nxx, nyy))
        x, y = nxx, nyy
        yaw = _wrap(yaw + res.omega * cfg.dt)
        trace.append((x, y, cur_region, step_i, target))
        trace_times.append((len(trace) - 1) * cfg.dt)
        max_offroad = max(max_offroad, table.nearest(x, y)[1])

    ok = step_i >= len(steps)
    straight = math.dist(trace[0][:2], trace[-1][:2]) if trace else 0.0
    return dict(mission=mission, ok=ok,
                reason="complete" if ok else f"timed out at step {step_i}",
                ticks=len(trace), step=step_i, n_steps=len(steps),
                path_len=path_len, straight_line=straight,
                regions_visited=visited, max_offroad_m=max_offroad,
                advances=advances, trace=trace, trace_times=trace_times,
                invariant=invariant.summary(),
                invariant_per_step=inv_log + [{"step": step_i, **invariant.summary()}],
                **_csr())


def cue_oracle(cue: str, label: str, cone_present, region: int | None = None) -> bool:
    """Ground-truth answer for a cue. The offline twin of ``nodes/gt_cue_node.py``.

    The SAME IMPLEMENTATION as the node (via `cue_answers`), not a parallel one that
    agrees by inspection. Two implementations can resolve the same cue differently (e.g.
    one testing `"cone" in cue` first, the other matching a place key first), and then
    the sim certifies plans the robot executes differently.

    What stays here is the only thing that is genuinely offline-specific: turning "where
    are the cones" into "is there a cone HERE". Everything about which cues exist and what
    matches them is in `cue_answers`.
    """
    # Deferred like every other package import in this file: scripts/ is run directly,
    # so sys.path is only right once __main__ has set it up.
    from carla_gt_bridge import cue_answers

    at_junction = label == "junction"
    # cone_present may be a bool (one cone, wherever we are) or a set of regions (several
    # cones, each somewhere specific). The set form is what the ordinal mission needs and
    # what spawning two props corresponds to.
    if isinstance(cone_present, (set, frozenset, tuple, list)):
        cone = region in cone_present
    else:
        cone = bool(cone_present) and at_junction
    # None (nothing answers this cue) collapses to False for the offline caller, which
    # has always treated "unanswered" as "not yet" -- a step waits rather than advancing.
    return bool(cue_answers.resolve(cue, at_junction=at_junction, cone=cone))


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _dedupe(seq):
    out = []
    for s in seq:
        if not out or out[-1] != s:
            out.append(s)
    return out


def _pick_branch(step, cone_present, label: str, region: int | None = None) -> list:
    """Choose a branch from the CUE ANSWERS, the way brain's _topic_choose_branch does.

    A branch wins when its own vlm_cue is answered true; otherwise `default` wins. Asking
    the oracle rather than reading a flag means the harness exercises the same decision
    brain makes, instead of asserting its outcome.
    """
    branches = step["branches"]
    default_idx = next((i for i, b in enumerate(branches)
                        if (b.get("vlm_cue") or "").strip().lower() == "default"), 0)
    for i, b in enumerate(branches):
        cue = (b.get("vlm_cue") or "").strip()
        if not cue or cue.lower() == "default":
            continue
        if cue_oracle(cue, label, cone_present, region):
            return list(b["sub_plan"])
    return list(branches[default_idx]["sub_plan"])


def plot(result, out_path: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from carla_gt_bridge.region_lookup import load_region_table

    table = load_region_table(REGIONS_NPZ)
    fig, ax = plt.subplots(figsize=(9, 8), dpi=130)
    wp = table.waypoints
    for rid in table.region_ids:
        m = wp[:, 2].astype(int) == rid
        ax.scatter(wp[m, 0], wp[m, 1], s=8, alpha=.5,
                   c="#C2410C" if table.label_of(rid) == "junction" else "#2A62D0")
        cx, cy = table.centroid_of(rid)
        ax.annotate(str(rid), (cx, cy), fontsize=8, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=.15", fc="white", ec="#CCC", lw=.5))
    tr = result["trace"]
    ax.plot([t[0] for t in tr], [t[1] for t in tr], lw=2.0, color="#111", zorder=5)
    ax.scatter([tr[0][0]], [tr[0][1]], s=80, marker="o", color="#16A34A", zorder=6)
    ax.scatter([tr[-1][0]], [tr[-1][1]], s=110, marker="*", color="#DC2626", zorder=6)
    for a in result["advances"]:
        i = min(a["tick"], len(tr) - 1)
        ax.scatter([tr[i][0]], [tr[i][1]], s=60, marker="D", color="#F59E0B", zorder=7)
    ax.set_aspect("equal")
    ax.grid(alpha=.15, lw=.5)
    ax.set_xlabel("x (m), OpenDRIVE/ROS frame")
    ax.set_ylabel("y (m), +north")
    ax.set_title(f"{result['mission']} — {result['reason']}\n"
                 f"{result['ticks']} ticks, {result['path_len']:.0f} m driven, "
                 f"regions {result['regions_visited']}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path)
    return out_path


def carla_pose_of(table, region: int) -> tuple[float, float]:
    """A region's centroid in CARLA coordinates, checked for recoverability.

    Used for the ego spawn and for siting cone props. Raises if the centroid does not
    resolve back to its own region: an L-shaped or curved region can have its mean
    outside itself (9 of Town05's 74 do), and placing a cone at such a point puts it in
    the WRONG region — which the cue oracle would then answer correctly about the wrong
    place.
    """
    from carla_gt_bridge.frames import planar_to_carla

    px, py = table.centroid_of(region)
    got = table.region_at(px, py)
    if got != region:
        raise ValueError(
            f"region {region}'s centroid ({px:.1f}, {py:.1f}) resolves to region {got}, "
            f"not {region} — its centroid lies outside it, so it cannot be used to place "
            f"anything. Pick a waypoint inside the region instead."
        )
    cx, cy, _ = planar_to_carla(px, py, 0.0)
    return cx, cy


def spawn_seed(*, cluster_map: str | None = None, regions_npz: str | None = None,
               start_region: int | None = None, start_toward: int | None = None):
    """The ego's start pose, in both frames: ``(px, py, pyaw_rad, cx, cy, cyaw_deg)``.

    Shared by ``spawn_config`` (which writes the seed file), run_phase_a.sh and
    drive_english.py (which derive the lane_spawn arguments), so they cannot disagree
    about where the vehicle goes.
    """
    import yaml
    from carla_gt_bridge.frames import planar_to_carla
    from carla_gt_bridge.region_lookup import load_region_table
    from carla_gt_bridge.routing import adjacency_from_bearing_map

    cluster_map = cluster_map or CLUSTER_MAP
    regions_npz = regions_npz or REGIONS_NPZ
    start_region = START_REGION if start_region is None else start_region
    nxt = START_TOWARD if start_toward is None else start_toward

    doc = yaml.safe_load(open(cluster_map))
    table = load_region_table(regions_npz)
    adj = adjacency_from_bearing_map(doc["bearing_map"])
    if nxt not in adj.get(start_region, ()):
        raise ValueError(
            f"start_toward={nxt} is not adjacent to start_region={start_region} "
            f"(neighbours: {sorted(adj.get(start_region, ()))})"
        )
    px, py = table.centroid_of(start_region)
    if table.region_at(px, py) != start_region:
        raise ValueError(
            f"region {start_region}'s centroid does not lie inside it; the ego would "
            f"spawn in region {table.region_at(px, py)}"
        )
    nx, ny = table.centroid_of(nxt)
    pyaw = math.atan2(ny - py, nx - px)
    cx, cy, cyaw = planar_to_carla(px, py, pyaw)
    return px, py, pyaw, cx, cy, cyaw


def spawn_config(out_dir: str, vehicle: str = "vehicle.tesla.model3",
                 role: str = "ego_vehicle", *, town: str = "town05",
                 cluster_map: str | None = None, regions_npz: str | None = None,
                 start_region: int | None = None,
                 start_toward: int | None = None) -> str:
    """Write the SEED pose that `lane_spawn.py` snaps onto a real driving lane.

    This does NOT write the file the bridge reads. `objects.<town>.json` is produced by
    `lane_spawn.py` against the LIVE map, because a region centroid sits on the road's
    reference line — the lane divider on a two-way road — and spawning there straddles
    both lanes. That script also adds the pseudo-sensors without which
    `/carla/<role>/odometry` never exists, and writes ROS coordinates rather than CARLA
    ones (the bridge's SpawnObject converts inbound, so writing CARLA coords mirrors the
    pose and fails as "collision at spawn position").

    So the output here is `objects.<town>.seed.json`, and its job is to hand
    `lane_spawn.py` a derived starting point instead of a hand-typed one. It must not
    write `objects.<town>.json`, or running it after `lane_spawn.py` would silently
    replace a good spawn with a mirrored, sensorless one.

    Configuring the spawn replaces an initial teleport. Teleporting with physics off is how a run "completes" without driving —
    it is why the metrics harness needs a path-length guard — and it also hides a spawn
    that is in the wrong place. Configuring the spawn instead means the vehicle is where the
    plan expects from tick zero, under physics, with nothing to discount.

    The pose is derived, not guessed: the start region's centroid, headed at the
    neighbour the route actually goes to. It is also the one place the planar -> CARLA
    conversion runs in the outbound direction, so it exercises the sign that `frames.py`
    exists to protect.

    ``start_toward`` is REQUIRED to be named (it defaults to the ``START_TOWARD``
    constant, not to an ordering). See that constant for why. Defaults
    reproduce the Town05 corridor exactly.
    """
    import yaml
    from carla_gt_bridge.frames import planar_to_carla
    from carla_gt_bridge.region_lookup import load_region_table
    from carla_gt_bridge.routing import adjacency_from_bearing_map

    cluster_map = cluster_map or CLUSTER_MAP
    regions_npz = regions_npz or REGIONS_NPZ
    start_region = START_REGION if start_region is None else start_region
    nxt = START_TOWARD if start_toward is None else start_toward

    px, py, pyaw, cx, cy, cyaw = spawn_seed(
        cluster_map=cluster_map, regions_npz=regions_npz,
        start_region=start_region, start_toward=nxt)

    payload = {"objects": [{
        "type": vehicle, "id": role, "role_name": role,
        "spawn_point": {"x": round(cx, 2), "y": round(cy, 2), "z": 0.5,
                        "roll": 0.0, "pitch": 0.0, "yaw": round(cyaw, 1)},
        # Phase A runs with no_rendering_mode on, so no sensors at all. Phase B adds the
        # LiDAR here (remapped to /livox/lidar); Phase C adds rgb_front for the VLM.
        "sensors": []}]}
    p = os.path.join(out_dir, f"objects.{town}.seed.json")
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  region {start_region} centroid, planar : ({px:.2f}, {py:.2f})  "
          f"heading {math.degrees(pyaw):+.1f} deg toward region {nxt}")
    print(f"  same pose in CARLA coordinates  : ({cx:.2f}, {cy:.2f})  yaw {cyaw:+.1f} deg")
    print(f"  -> {os.path.basename(p)}   (SEED — feed it to lane_spawn.py)")
    print(f"  lane_spawn.py --town {town} --region {start_region} "
          f"--x {cx:.2f} --y {cy:.2f} --yaw {cyaw:.1f}")
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spawn", action="store_true",
                    help="write the bridge objects.json that spawns the ego at START_REGION")
    ap.add_argument("--simulate", metavar="PLAN", default=None,
                    help="brain tree JSON to drive offline (e.g. from drive_english --dry)")
    ap.add_argument("--cone-regions", default="",
                    help="comma-separated regions holding a cone; default: none")
    ap.add_argument("--branch", default="right", choices=["right", "straight"])
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out-dir", default=os.path.join(PKG, "config"))
    args = ap.parse_args()

    if args.spawn:
        os.makedirs(args.out_dir, exist_ok=True)
        print("ego spawn configuration:")
        spawn_config(args.out_dir)

    if args.simulate:
        cones = [int(c) for c in args.cone_regions.split(",") if c.strip()]
        name = os.path.splitext(os.path.basename(args.simulate))[0]
        r = simulate(name, branch=args.branch, verbose=args.verbose,
                     plan_path=args.simulate, cone_regions=cones)
        print("closed-loop simulation (no CARLA):")
        print(f"  [{'PASS' if r['ok'] else 'FAIL'}] {name} {r['reason']:28s} "
              f"{r['ticks']:4d} ticks  {r.get('path_len', 0):6.0f} m  "
              f"regions {r.get('regions_visited')}")
        for a in r["advances"]:
            print(f"          advance @t{a['tick']:<4d} step {a['step']} "
                  f"in region {a['region']}: {a['why']}")
        if r.get("forbid_declared"):
            print(f"          []~{r['forbid_clusters']}: csr {r['csr']:.3f} "
                  f"({r['violation_ticks']}/{r['constraint_ticks']} ticks inside, "
                  f"regions {r['forbid_regions']}) policy={r['forbid_policy']}")
        if args.plot and r.get("trace"):
            print("         ", plot(r, os.path.join(args.out_dir, f"dryrun.{name}.png")))
        sys.exit(0 if r["ok"] else 1)


if __name__ == "__main__":
    main()
