"""Ground a topological plan step onto a concrete region, using the map.

The gap this closes
-------------------
A plan step says ``goal_mode: junction``. It cannot say *which* junction, and it should
not have to: the language layer works in the vocabulary (`path`, `junction`, `passage`),
and "the second intersection" is expressed by *unrolling* the plan into repeated
path -> junction -> path steps, not by naming ids. That is what makes one plan portable
between CARLA and the campus bags.

But the MPC has to steer at something. ``taxonomy.canonical_id('junction')`` returns the
first junction id in the corridor, which on Town05 is **53 — the second junction** — so a
plan whose first step is "reach a junction" would aim the vehicle past junction 66 at the
one after it. The vehicle would cut the corner or leave the corridor, and the log would
say only that the MPC drove the wrong way.

So grounding happens here, at run time, from the map: **the target is the adjacent region
whose label matches the step's goal, excluding the region we came from.** The plan stays
topological; the map supplies the geometry. Adjacency comes from the plan's own
``bearing_map`` keys (``"45-66"``), so no extra artifact is needed and the routing cannot
disagree with the bearings the controller steers by.

Ambiguity is real and is resolved, not ignored
----------------------------------------------
From region 4 both neighbours (66 and 53) are junctions, so "the next junction" is
ambiguous without history — which is exactly why ``previous`` is a required argument
rather than an optional hint. When ambiguity survives even that (a genuine fork, e.g.
leaving a junction with two `path` exits), :func:`next_region` returns every candidate and
the caller must disambiguate — for a branch step, with the brain's chosen maneuver. It
never silently picks one, because picking the wrong exit at a junction is the most
expensive failure in these missions and looks identical to correct behaviour for the
first few metres.
"""

from __future__ import annotations

__all__ = ["StepTargeter", "adjacency_from_bearing_map", "bearing_between",
           "candidates", "maneuver_toward", "next_region", "region_for_maneuver",
           "turn_magnitude"]


def adjacency_from_bearing_map(bearing_map: dict) -> dict[int, set[int]]:
    """``{"45-66": deg}`` -> ``{45: {66}, 66: {45}}``.

    Reuses the bearing map rather than shipping a separate adjacency table, so the graph
    the router walks is by construction the same graph the controller has headings for.
    """
    adj: dict[int, set[int]] = {}
    for key in bearing_map:
        try:
            a_s, b_s = str(key).split("-")
            a, b = int(a_s), int(b_s)
        except ValueError:
            continue
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    return adj


#: Modes you can remain in and still be making progress. For these, a step whose goal
#: label equals the current label means the NEXT such region; for a point mode (junction)
#: it means the one you are in.
EXTENT_LABELS = frozenset({"path", "along_edge", "passage", "open_space"})


def candidates(adj: dict[int, set[int]], labels: dict[int, str],
               current: int, goal_label: str,
               previous: int | None = None) -> list[int]:
    """Adjacent regions labelled ``goal_label``, excluding ``previous``, sorted."""
    out = sorted(r for r in adj.get(current, ())
                 if labels.get(r) == goal_label and r != previous)
    return out


def next_region(adj: dict[int, set[int]], labels: dict[int, str],
                current: int, goal_label: str,
                previous: int | None = None) -> tuple[int | None, list[int]]:
    """Return ``(target, all_candidates)``.

    ``target`` is set only when the choice is unambiguous. With several candidates the
    caller must disambiguate (a branch step does it with the brain's maneuver); with none
    the plan and the map disagree and the caller should stop rather than improvise.
    """
    # NOTE: do not short-circuit to `current` when the region we are in already
    # satisfies `goal_label`. That looks like a fix for a real race (the same plan can
    # ground `junction` differently depending on whether the step fires on the approach
    # or after entry), but it makes the vehicle loop between two regions.
    #
    # The reason is that this function serves TWO purposes:
    #   1. what the step's destination IS  -- "already here" is the right answer
    #   2. what the MPC should STEER AT    -- "here" is not a waypoint; the controller
    #                                          needs something ahead on the route
    # Returning `current` satisfies (1) and breaks (2). A correct fix has to separate
    # them: let arrival be judged by membership, and steer at the exit the step's manoeuvre
    # implies. That is a change to the targeter's contract, not a one-line guard.
    cands = candidates(adj, labels, current, goal_label, previous)
    return (cands[0] if len(cands) == 1 else None), cands


def maneuver_toward(bearing_map: dict, current: int, previous: int | None,
                    target: int, straight_deg: float = 45.0) -> str:
    """Classify ``previous -> current -> target`` as straight / left / right / u_turn.

    Used to pick between several candidates at a branch: brain decides *which maneuver*
    ("turn right"), and this says which region that maneuver corresponds to.

    Angles come from the plan's ``bearing_map``, which is computed on OpenDRIVE geometry
    — the **planar** frame: +y north, headings counter-clockwise. So a **negative** turn
    is a right turn. (In CARLA's own left-handed frame the sign is opposite; see
    :mod:`carla_gt_bridge.frames` for which data is in which frame.) Getting this
    backwards swaps every branch decision.

    Worked example, the Town05 decision junction: approaching j53 from r4 the heading is
    -88 deg (travelling south); the exit to r8 is at 179 deg (west). turn = -93 deg, and
    when travelling south, west is on the right — a RIGHT turn, as the sign says.
    """
    if previous is None:
        return "straight"
    h_in = bearing_map.get(f"{previous}-{current}")
    h_out = bearing_map.get(f"{current}-{target}")
    if h_in is None or h_out is None:
        return "unknown"
    turn = (float(h_out) - float(h_in) + 180.0) % 360.0 - 180.0
    if abs(turn) < straight_deg:
        return "straight"
    if abs(turn) > 180.0 - straight_deg:
        return "u_turn"
    return "right" if turn < 0 else "left"


def region_for_maneuver(bearing_map: dict, current: int, previous: int | None,
                        cands: list[int], maneuver: str) -> int | None:
    """Which candidate corresponds to ``maneuver``, or None if not exactly one does."""
    matches = [c for c in cands
               if maneuver_toward(bearing_map, current, previous, c) == maneuver]
    return matches[0] if len(matches) == 1 else None


def bearing_between(bearing_map: dict, a: int, b: int) -> float | None:
    """Convenience: the stored CARLA-frame heading from ``a`` to ``b``, in degrees."""
    v = bearing_map.get(f"{a}-{b}")
    return None if v is None else float(v)


class StepTargeter:
    """Holds "which region am I steering at right now" across ticks.

    One target per plan step, grounded when the step starts. That is the invariant that
    matters, and it is easy to get wrong in a way that deadlocks: the naive version
    re-grounds every tick, so the moment the vehicle *arrives* in the target region — but
    before brain has counted enough dwell frames to advance the step — the router is asked
    for "a junction adjacent to the junction I am standing in", finds none, and the run
    stops one metre short of success.

    Also tracks ``previous``, which :func:`next_region` needs to disambiguate and which no
    single ROS message carries: brain publishes the current cluster, not the one before it.
    """

    def __init__(self, adjacency: dict[int, set[int]], labels: dict[int, str],
                 bearing_map: dict, confirm_ticks: int = 1) -> None:
        self.adj = adjacency
        self.labels = labels
        self.bearing_map = bearing_map
        #: How many CONSECUTIVE observations a new region needs before it is accepted.
        #: 1 = accept immediately (the default behaviour).
        #:
        #: WHY IT IS A KNOB. Straddling a junction boundary the classifier can oscillate
        #: between two regions, and with confirm_ticks=1 `observe` re-grounds the target on
        #: every flicker -- including back to the region it came FROM, or off the route --
        #: and by the time it settles the vehicle may have driven straight through the
        #: turn. Whether this happens depends on the boundary geometry of the junction.
        self.confirm_ticks = max(1, int(confirm_ticks))
        self._pending: int | None = None
        self._pending_n = 0
        self.previous: int | None = None
        self.current: int | None = None
        self.target: int | None = None
        #: regions the current route passes through, so the caller can stop forbidding
        #: them. Empty for an adjacent target.
        self.via: list[int] = []
        self._step_key = None

    def observe(self, region: int) -> bool:
        """Feed the latest cluster id. Returns True if the accepted region changed.

        A change must persist for `confirm_ticks` consecutive observations before it is
        accepted, so a single-tick flicker at a junction boundary cannot re-ground the
        target. Returning to the current region cancels any pending change.
        """
        if region == self.current:
            self._pending, self._pending_n = None, 0
            return False
        if region != self._pending:
            self._pending, self._pending_n = region, 1
        else:
            self._pending_n += 1
        if self._pending_n < self.confirm_ticks:
            return False
        self.previous, self.current = self.current, region
        self._pending, self._pending_n = None, 0
        return True

    def next_along(self, goal_label: str) -> tuple[int | None, list[int]]:
        """Nearest region labelled ``goal_label`` reachable WITHOUT turning back.

        Breadth-first from the current region, never re-entering ``previous``, so it
        finds the next such region *ahead* rather than the one just left. Adjacency alone
        cannot answer this: from junction 66 the next junction is 53, two hops away
        through path region 4.

        Needed for ordinal cues. "Stop at the second cone" requires the vehicle to keep
        moving while the count accumulates, so when the cue is still unmet at the current
        target the step has to re-ground further along. Without it the vehicle parks at
        the first cone and the ordinal can never advance — which is the same deadlock
        shape as counting the same landmark twice, from the opposite direction.

        Returns ``(target, path)``; ``path`` is the region sequence for logging.
        """
        if self.current is None:
            return None, []
        frontier = [(self.current, [self.current])]
        seen = {self.current}
        if self.previous is not None:
            seen.add(self.previous)
        while frontier:
            node, path = frontier.pop(0)
            for nxt in sorted(self.adj.get(node, ())):
                if nxt in seen:
                    continue
                seen.add(nxt)
                if self.labels.get(nxt) == goal_label:
                    return nxt, path + [nxt]
                frontier.append((nxt, path + [nxt]))
        return None, []

    def advance_target(self, goal_label: str) -> tuple[int | None, str]:
        """Re-ground PAST the current target, for a step whose cue is not yet satisfied.

        Only meaningful once the vehicle has arrived: the cluster half of the step is
        satisfied, the cue half is not, so the plan says keep looking.
        """
        nxt, path = self.next_along(goal_label)
        self.via = [r for r in path[1:-1]] if nxt is not None else []
        if nxt is None:
            return None, (f"cue unmet at {self.current} but no further {goal_label!r} "
                          f"ahead — the landmark count cannot be reached")
        self.target = nxt
        return nxt, (f"cue unmet at {self.current}; advancing to the next "
                     f"{goal_label!r} -> {nxt} via {path}")

    def target_for(self, goal_label: str, step_key,
                   maneuver: str | None = None,
                   from_region: int | None = None,
                   from_previous: int | None = None) -> tuple[int | None, str]:
        """Region to steer at for this step, plus a one-line reason for the log.

        ``from_region``/``from_previous`` GROUND THE STEP FROM WHERE THE DECISION WAS
        MADE rather than from wherever the vehicle is now. A turn is defined relative to
        the junction it happens at, and by the time a branch commits the vehicle may have
        left it: the VLM call is wall-clock bound while the simulator advances on sim
        time, and CARLA can run several times faster than real time, so a 1.5 s call can
        be ~25-30 m driven -- past a junction whose radius is ~18 m. Grounding from the
        current region would then try to find a RIGHT TURN from the region after the
        junction, where no right turn exists, and the vehicle goes straight through.

        Slowing the vehicle cannot fix this -- it is a time-base mismatch, not a speed
        problem -- so the fix is to remember the junction instead.
        """
        if step_key != self._step_key:
            self._step_key = step_key
            self.target = None
            self.via = []
        if self.target is not None:
            return self.target, f"holding target {self.target}"
        cur = self.current if from_region is None else from_region
        prev = self.previous if from_region is None else from_previous
        if cur is None:
            return None, "no cluster observed yet"

        target, cands = next_region(self.adj, self.labels, cur,
                                    goal_label, prev)
        if target is not None:
            self.target = target
            return target, (f"grounded {goal_label!r} -> {target} from {cur} "
                            f"(prev {prev})"
                            + ("" if from_region is None else " [at the decision junction]"))
        if not cands:
            # Nothing ADJACENT matches — but that does not mean the plan disagrees with
            # the map. A step whose goal mode is the mode it is already in needs the
            # NEXT such region, which is two hops away through whatever separates them:
            #
            #   previous=14, current=7, goal='path'
            #     adjacency-only -> None, "no 'path' adjacent to 7"
            #     next_along     -> 6, via [7, 30, 6]
            #
            # `next_along` (BFS, never re-entering `previous`) already finds it and
            # `advance_target` already uses it. That shape — goal_mode equal to
            # start_mode — is the documented encoding for "stop at the Nth landmark"
            # (examples in nl_planner's prompts/generator.md), so it is not an edge case.
            #
            # Reached ONLY where adjacency-only grounding finds nothing, so the
            # unambiguous-adjacent case is unaffected.
            far, path = self.next_along(goal_label)
            if far is not None:
                self.target = far
                self.via = list(path[1:-1])
                return far, (f"no {goal_label!r} adjacent to {self.current}; grounded "
                             f"further along -> {far} via {path} (prev {self.previous})")
            # WHAT THIS DOES AND DOES NOT MEAN. The working assumption is that a plan is
            # TOPOLOGICALLY sound — the sequence of modes and turns it asks for is
            # achievable on the graph — while the ABSOLUTE position and orientation of
            # the landmarks is NOT guaranteed (map distortion, e.g. a similarity
            # transform on the centroids with cluster membership left true).
            #
            # So reaching here is a CONNECTIVITY failure, not a geometry one: no region
            # of this mode is reachable ahead without turning back. Metric error cannot
            # produce it, and no amount of re-registration will fix it. Either the graph
            # is missing an edge or the plan asked for a mode this map does not afford.
            return None, (f"no {goal_label!r} reachable ahead from {self.current} "
                          f"(prev {self.previous}): the plan asks for a mode this map "
                          f"does not connect to. This is a TOPOLOGY failure — landmark "
                          f"positions being off would not cause it.")
        if maneuver:
            pick = region_for_maneuver(self.bearing_map, cur, prev,
                                       cands, maneuver)
            if pick is not None:
                self.target = pick
                return pick, f"branch {maneuver!r} -> {pick} among {cands}"
            return None, f"maneuver {maneuver!r} matches none of {cands}"
        return None, (f"{len(cands)} candidates {cands} for {goal_label!r} — needs a "
                      f"maneuver to disambiguate")


def turn_magnitude(bearing_map: dict, previous: int, current: int,
                   target: int) -> float | None:
    """Signed turn in degrees for ``previous -> current -> target`` (CARLA frame)."""
    h_in = bearing_map.get(f"{previous}-{current}")
    h_out = bearing_map.get(f"{current}-{target}")
    if h_in is None or h_out is None:
        return None
    return (float(h_out) - float(h_in) + 180.0) % 360.0 - 180.0
