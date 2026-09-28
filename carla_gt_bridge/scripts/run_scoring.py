#!/usr/bin/env python3
"""Score a driven run against what the English actually asserted.

WHY THIS EXISTS
Scoring by REGION SEQUENCE alone (did the vehicle visit 45, 66, 4, 53, 8?) says
where it went, never what it did when it got there. A run of the `hard` mission
("go straight through the first intersection, turn right at the second") can turn
RIGHT at the first intersection and still produce the expected region sequence,
because a step with `trigger: traverse` carries no `Bearing(...)` cue and so
nothing measures a heading.

TWO CHECKS, PAIRED — they answer different questions and neither subsumes the
other:

    B (reached)    did the vehicle reach the expected region, in order?
                   Catches "never got there". Says nothing about maneuvers.
    D (maneuvers)  at each junction it actually traversed, did it turn the way
                   the English said? Catches "got there the wrong way".
                   Says nothing about whether it arrived.

A run passes only if both hold. D alone would pass a run that never reached the
second junction (empty maneuver list vacuously matches nothing); B alone misses
the wrong-maneuver case above.

Two more checks complete the set. M (motion) asks whether the vehicle DROVE the route rather than jumping
along it — see ``motion_is_plausible``. C (constraint) asks whether it stayed out
of the modes the plan forbade — see ``constraint_satisfied``. M is in the ``ok``
conjunction; **C is not, yet**, and the comment in ``score`` says what is needed
before it goes in.

NOT A "STOPS THERE" TEST. "Turn right at the second intersection" does not say
stop, so B asks whether the expected region was REACHED in order, not whether the
route ends on it. Overshoot is the plan having extra steps, not a wrong turn.
"""
from __future__ import annotations

import math

#: Same boundary `map_regions.maneuvers()` uses to classify junction exits, so a
#: corridor selected as "straight" is scored as "straight" by identical maths.
STRAIGHT_DEG = 45.0

#: Ticks either side of a junction used to measure heading. One segment is noisy
#: (the integrator wobbles ~0.1 rad/s); a short window is stable without smearing
#: the turn itself. At 5 Hz this is ~0.6 s.
HEADING_WINDOW = 3


def _bearing(p, q) -> float:
    return math.degrees(math.atan2(q[1] - p[1], q[0] - p[0]))


def _wrap180(d: float) -> float:
    return (d + 180.0) % 360.0 - 180.0


def classify(delta_deg: float) -> str:
    if abs(delta_deg) < STRAIGHT_DEG:
        return "straight"
    return "right" if delta_deg < 0 else "left"


def group_by_region(trace):
    """[(region_id, [tick indices]), ...] in visit order.

    ``trace`` rows are ``(x, y, region, step, target)`` as ``missions.simulate``
    writes them.
    """
    groups: list[tuple[int, list[int]]] = []
    for k, row in enumerate(trace):
        rid = row[2]
        if groups and groups[-1][0] == rid:
            groups[-1][1].append(k)
        else:
            groups.append((rid, [k]))
    return groups


def maneuvers(trace, labels, window: int = HEADING_WINDOW):
    """What the vehicle DID at each junction it traversed.

    Returns ``[{region, delta_deg, kind, complete}, ...]``.

    THE BOUNDARY CASES:

    * A junction that is the FIRST region group has no preceding region to
      measure an entry heading from. Measured from ticks inside the group
      instead; flagged ``complete=False`` if there are too few.
    * A junction that is the LAST group has no following region — same handling.
      Skipping these cases would silently drop the only junction in the `medium`
      mission and one of the two in `hard`.
    * A junction crossed within fewer than two ticks has no measurable heading
      change at all. Reported with ``complete=False`` rather than as
      ``straight`` — "could not tell" and "went straight" are different
      claims and must not be conflated.
    * **A junction entered and left back into the SAME region is not a traversal.**
      Vehicles clip a junction's boundary on the way past it, so the region stream
      reads `5 -> 73 -> 5` for what was one continuous drive along region 5. A
      stream like `[4, 73, 5, 73, 5, 74, 38]` would otherwise score
      `['straight','straight','right']` against `['straight','right']` and FAIL a
      correctly driven route purely from that re-entry.

      Skipped rather than reported, because it is not a maneuver in the sense the
      English asserts: "turn right at the second intersection" counts intersections
      you go THROUGH. A genuine U-turn also has same-region entry and exit, so this
      does discard real u-turns — acceptable while no mission asserts one, and
      flagged here because the day one does, this rule has to become
      "same region AND heading reversed".

      NOTE this matters far more with real perception than in the sim: cluster
      flicker at a boundary is expected with a noisy junction-vs-path classifier, and
      every such flicker would otherwise invent a maneuver.
    """
    out = []
    groups = group_by_region(trace)
    n = len(trace)
    for gi, (rid, ks) in enumerate(groups):
        if labels.get(rid) != "junction":
            continue
        # entered and left into the same region -> clipped the corner, did not traverse
        if 0 < gi < len(groups) - 1 and groups[gi - 1][0] == groups[gi + 1][0]:
            continue
        a, b = ks[0], ks[-1]
        # Entry heading: prefer the run-up from the previous region; fall back to
        # the first movement inside the junction when this is the opening group.
        i0 = max(0, a - window) if gi > 0 else a
        i1 = a if gi > 0 else min(b, a + window)
        # Exit heading: prefer the run-out into the next region; fall back to the
        # last movement inside the junction when this is the closing group.
        j0 = b if gi < len(groups) - 1 else max(a, b - window)
        j1 = min(n - 1, b + window) if gi < len(groups) - 1 else b
        complete = (i1 > i0) and (j1 > j0)
        if not complete:
            out.append({"region": rid, "delta_deg": None, "kind": None,
                        "complete": False})
            continue
        d = _wrap180(_bearing(trace[j0], trace[j1]) - _bearing(trace[i0], trace[i1]))
        out.append({"region": rid, "delta_deg": round(d, 1),
                    "kind": classify(d), "complete": True})
    return out


def reached_in_order(regions_visited, expect_regions) -> bool:
    """B: every expected region appears, in order. Extra regions are allowed."""
    if not expect_regions:
        return True
    it = iter(regions_visited)
    return all(any(r == want for r in it) for want in expect_regions)


#: Fastest the vehicle can legitimately travel, m/s. v_max is 5.0 in the MPC; the margin
#: absorbs the odometry sampling jitter that makes a single interval look fast.
PLAUSIBLE_V_MAX = 12.0


def motion_is_plausible(trace, times=None, v_max: float = PLAUSIBLE_V_MAX) -> dict:
    """Did the vehicle DRIVE the route, or did its position jump along it?

    `missions.py` notes that "Teleporting with physics off is how a run 'completes'
    without driving"; this is the CARLA-path equivalent of the offline path-length
    guard. Without it, a run whose position jumps tens of metres in a fraction of a
    second (far above v_max) scores as a PASS whenever the region sequence is right.

    WHAT A FAILURE MEANS — two causes, and they are not the same problem. An interval
    above v_max is either the vehicle genuinely jumping, OR the pose stream being too
    sparse to see the driving in between (many control ticks on only a few distinct
    odometry samples). Either way the run is unusable and must not score
    as a pass — a controller closing the loop on a pose that old is open-loop — but the
    fix is different, so the message must not assert "teleport".

    Returns a dict with ``ok`` False when any interval exceeds ``v_max``, plus the worst
    offender so the log names it. With no timestamps, only a total-displacement check is
    possible and ``checked`` says so.
    """
    pts = [(t[0], t[1]) for t in (trace or [])]
    if len(pts) < 2:
        return {"ok": True, "checked": "too few points", "max_speed": None,
                "path_len": 0.0}
    path_len = sum(math.dist(pts[i - 1], pts[i]) for i in range(1, len(pts)))
    if not times or len(times) != len(pts):
        return {"ok": True, "checked": "no timestamps — speed unverifiable",
                "max_speed": None, "path_len": path_len}
    worst = {"speed": 0.0, "i": None}
    for i in range(1, len(pts)):
        dt = times[i] - times[i - 1]
        if dt <= 0:
            continue
        v = math.dist(pts[i - 1], pts[i]) / dt
        if v > worst["speed"]:
            worst = {"speed": v, "i": i}
    distinct = len({(round(x, 2), round(y, 2)) for x, y in pts})
    return {
        "ok": worst["speed"] <= v_max,
        "checked": f"{len(pts)} samples, {distinct} distinct",
        # A handful of distinct poses across many samples means the POSE STREAM is the
        # problem, not the vehicle. Reported separately so the two causes stay separable.
        "distinct_poses": distinct,
        "pose_stream_sparse": distinct < max(3, len(pts) // 10),
        "max_speed": round(worst["speed"], 1),
        "at_sample": worst["i"],
        "v_max": v_max,
        "path_len": round(path_len, 1),
    }


def constraint_satisfied(result) -> dict:
    """C: did the run stay OUT of the modes the plan forbade (``[]~X``)?

    A fourth question, and like the other three nothing else asks it. B says the
    expected regions appeared, D says the turns were right, M says it drove there
    — and a run can satisfy all three while spending half its length in the plaza
    the mission said never to enter, because no check reads a region the plan did
    not ask for.

    ``csr`` is the fraction of ticks spent OUTSIDE ``forbid_clusters``. True when
    it is 1.0, and also true when no constraint was declared — a plan that forbids
    nothing cannot violate anything, and scoring that as a failure would fail every
    unconstrained run.

    THOSE TWO ARE NOT THE SAME CLAIM and the dict says which one happened.
    ``declared=False`` is vacuous satisfaction; ``declared=True, csr=1.0`` is
    earned. Collapsing them is how "the check never ran" comes to look like "the
    check passed", which is the failure mode `motion_is_plausible` already has a
    rule about ("absence of a check is not a pass").
    """
    declared = bool(result.get("forbid_declared"))
    csr = result.get("csr")
    if not declared:
        return {"ok": True, "declared": False, "csr": 1.0, "violation_ticks": 0,
                "checked": "no forbid_clusters declared — vacuously satisfied"}
    csr = 1.0 if csr is None else float(csr)
    return {
        "ok": csr >= 1.0,
        "declared": True,
        "csr": round(csr, 4),
        "violation_ticks": int(result.get("violation_ticks") or 0),
        "constraint_ticks": int(result.get("constraint_ticks") or 0),
        "forbid_clusters": result.get("forbid_clusters") or [],
        "forbid_regions": result.get("forbid_regions") or [],
        "checked": f"{result.get('constraint_ticks')} ticks scored",
    }


def score(result, labels, expect_regions=None, expect_maneuvers=None) -> dict:
    """Combine B, D, M and C over one ``missions.simulate`` result."""
    trace = result.get("trace") or []
    got_man = maneuvers(trace, labels)
    kinds = [m["kind"] for m in got_man if m["complete"]]
    incomplete = [m["region"] for m in got_man if not m["complete"]]

    b_ok = reached_in_order(result.get("regions_visited", []), expect_regions or [])
    d_ok = (expect_maneuvers is None) or (kinds == list(expect_maneuvers))
    # M — did it MOVE there? B and D both read the region sequence and the headings
    # along it, and neither can tell driving from teleporting. Off by default for the
    # offline sim, whose unicycle integrator cannot jump; the CARLA path passes times.
    m = motion_is_plausible(trace, result.get("trace_times"))
    # C — did it stay out of what the plan forbade? REPORTED, NOT ENFORCED, and
    # deliberately absent from `ok` below.
    #
    # Same policy as `InvariantMonitor`: with a noisy junction-vs-path classifier, a
    # criterion that fails a run on region membership fails it on perception noise as
    # often as on the robot entering the forbidden mode. Enforcing at that label
    # quality would make every downstream number a statement about the classifier.
    #
    # WHAT MOVES IT INTO `ok`: a low false-violation rate on runs that did NOT
    # violate — i.e. once csr == 1.0 is the norm, and the residual is attributable
    # to real entries rather than flicker. Until then it is a report column, and
    # pass/fail stays comparable with runs already recorded.
    c = constraint_satisfied(result)

    return {
        "drove": bool(result.get("ok")),
        "B_reached": b_ok,
        "D_maneuvers": d_ok,
        "M_motion": m["ok"],
        "motion": m,
        "C_constraint": c["ok"],
        "constraint": c,
        # NOTE: C is NOT in this conjunction. See the comment above — it is
        # reported before it is enforced, on purpose.
        "ok": bool(result.get("ok")) and b_ok and d_ok and m["ok"],
        "expect_regions": expect_regions,
        "regions_visited": result.get("regions_visited", []),
        "expect_maneuvers": list(expect_maneuvers) if expect_maneuvers else None,
        "maneuvers": got_man,
        "maneuvers_kinds": kinds,
        "maneuvers_unmeasurable": incomplete,
        "reason": result.get("reason", ""),
    }


def fmt(name: str, s: dict) -> str:
    flag = "OK  " if s["ok"] else "FAIL"
    man = ", ".join(
        f"{m['region']}:{m['kind'] or '?'}"
        + (f"({m['delta_deg']:+.0f})" if m["delta_deg"] is not None else "")
        for m in s["maneuvers"]) or "none"
    lines = [f"  [{flag}] {name:<26} drove={s['drove']} B={s['B_reached']} D={s['D_maneuvers']}",
             f"         route     {s['regions_visited']}"]
    if s["expect_regions"]:
        lines.append(f"         expect    {s['expect_regions']}")
    lines.append(f"         maneuvers {man}")
    if s["expect_maneuvers"]:
        lines.append(f"         expect    {s['expect_maneuvers']}")
    # Printed only when a constraint was DECLARED. A vacuous `C=True` on every
    # unconstrained run would add noise for no information.
    if s.get("constraint", {}).get("declared"):
        c = s["constraint"]
        lines.append(f"         []~clusters C={c['ok']} csr={c['csr']:.3f} "
                     f"({c['violation_ticks']}/{c['constraint_ticks']} ticks inside "
                     f"{c['forbid_regions']})  REPORTED, NOT ENFORCED")
    if s["maneuvers_unmeasurable"]:
        lines.append(f"         UNMEASURABLE at {s['maneuvers_unmeasurable']} "
                     f"(too few ticks to read a heading)")
    if not s["drove"]:
        lines.append(f"         reason    {s['reason'][:96]}")
    return "\n".join(lines)


__all__ = ["classify", "maneuvers", "reached_in_order", "score", "fmt",
           "constraint_satisfied", "motion_is_plausible",
           "group_by_region", "STRAIGHT_DEG", "HEADING_WINDOW"]
