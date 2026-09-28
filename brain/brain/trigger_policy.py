"""Trigger typology: the rule deciding when a plan step is satisfied.

This is the part of the brain that answers *"which cluster gets used when"*, kept
in its own ROS-free module so it can be unit-tested without rclpy/cv2/openai
(none of which are importable in a plain venv) and reasoned about on its own.

The problem it solves
---------------------
The original controller advanced a step on ``current_cluster == goal_cluster``.
That single rule cannot express the cases the language plans actually need:

- *"go down the road"* — the cluster genuinely IS the evidence, and a single
  flickered frame must not advance the plan.
- *"pass the blue building"* — the evidence is ``Detect(BlueBuilding)``. The
  cluster is only a guard, and requiring the exact fine cluster would strand the
  plan: ``along_edge`` is poorly recognised from LiDAR alone and does not transfer
  across bags.
- *"turn right at the intersection"* — the evidence is that the turn happened.
  On the real campus ``junction`` is not separable from ``path`` by the
  cluster classifier (junction-vs-rest is at chance across bags), so waiting
  for the cluster would wait forever.

So each step carries a ``trigger`` naming what its evidence is, and the cluster's
role follows from that. See ``nl_planner.schemas.TRIGGERS`` for the planner-side
definition; the two must stay in agreement.

Why the duplication with nl_planner
-----------------------------------
``brain`` deliberately does not depend on ``nl_planner``: they are separate ROS
packages, brain must also run against hand-written ``plan.json`` files, and the
nl_planner half needs pydantic. The inference rule is small and stable enough to
mirror; ``test_trigger_policy.py`` pins the two to the same behaviour.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

TRIGGERS = ("traverse", "landmark", "topology")


# --------------------------------------------------------------------------- #
# Reading a step                                                              #
# --------------------------------------------------------------------------- #

def trigger_of(step: Mapping[str, Any]) -> str:
    """A step's trigger, inferred from its cue when the plan predates the field.

    Mirrors ``nl_planner.schemas.infer_trigger``.
    """
    t = str(step.get("trigger") or "").strip().lower()
    if t in TRIGGERS:
        return t
    cue = str(step.get("transition_cue") or "").strip().lower()
    if not cue:
        return "traverse"
    if "detect(" in cue:
        return "landmark"
    if "bearing(" in cue:
        return "topology"
    # Free-text cue with no predicate ("a light brown bench"). Still a visual
    # landmark check, so treat it as one.
    return "landmark"


def accept_set(step: Mapping[str, Any], *, degraded: bool) -> set[int]:
    """Cluster ids that count as "in this step's goal mode".

    ``degraded=True`` returns the wider set that includes the coarser fallbacks
    (``mode_meta.accept_degraded``, resolved by the planner). Falls back to
    ``{goal_cluster}`` for plans written before the acceptance sets existed,
    which reproduces the historical equality test exactly.
    """
    key = "accept_clusters_degraded" if degraded else "accept_clusters"
    ids = step.get(key) or step.get("accept_clusters")
    if not ids:
        goal = step.get("goal_cluster")
        return {int(goal)} if goal is not None else set()
    return {int(i) for i in ids}


def bearing_direction(cue: str | None) -> str | None:
    """Extract 'left' / 'right' / 'straight' from a Bearing(...) cue, if named."""
    low = (cue or "").lower()
    for name in ("left", "right", "straight"):
        if name in low:
            return name
    return None


def wrap_pi(angle: float) -> float:
    """Wrap radians to (-pi, pi] so heading deltas across the +/-pi seam work."""
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def bearing_complete(
    step: Mapping[str, Any],
    yaw: float | None,
    entry_yaw: float | None,
    threshold_deg: float,
    travelled_since: float | None = None,
    scope_m: float | None = None,
    accumulated_deg: float | None = None,
) -> bool:
    """Has the maneuver named by the step's ``Bearing(...)`` cue happened?

    Uses RELATIVE heading change since the step was entered — no map, no absolute
    frame, no metric goal. Direction is enforced when the cue names one, so a left
    turn cannot satisfy a ``Bearing(Right)`` step.

    ROS REP-103: +z yaw is counter-clockwise, i.e. a LEFT turn is positive.

    ``scope_m`` BOUNDS THE MANEUVER IN DISTANCE, and without it this check certifies
    turns that never happened: 60 deg accumulated over ~150 m of ordinary route
    curvature satisfies ``Bearing(Right)`` even when the junction itself saw only a few
    degrees. The comparison is to the step's entry yaw with no requirement that the change
    occur AT the decision point, so any long enough drive eventually satisfies it, and a
    vehicle that drove straight through is certified as having turned.

    Distance rather than region id on purpose: this module is map-free, and
    ``travelled_since`` is a scalar from the odometer (integrated speed, never position).
    A junction manoeuvre completes within a few tens of metres; past that, heading change
    is the road bending, not the turn. ``scope_m=None`` keeps the old unbounded behaviour
    so existing callers and recorded traces score exactly as before.
    """
    if accumulated_deg is not None:
        # The windowed measure: heading change gathered only where the manoeuvre belongs.
        deg = accumulated_deg
        if abs(deg) < threshold_deg:
            return False
        want = bearing_direction(step.get("transition_cue"))
        if want == "left":
            return deg > 0
        if want == "right":
            return deg < 0
        return True
    if yaw is None or entry_yaw is None:
        return False
    if (scope_m is not None and scope_m > 0
            and travelled_since is not None and travelled_since > scope_m):
        # Out of scope: the manoeuvre did not happen where it was asked for. Returning
        # False lets the step TIME OUT, which is visible, rather than advancing on a turn
        # that never occurred, which is not.
        return False
    deg = math.degrees(wrap_pi(yaw - entry_yaw))
    if abs(deg) < threshold_deg:
        return False
    want = bearing_direction(step.get("transition_cue"))
    if want == "left":
        return deg > 0
    if want == "right":
        return deg < 0
    # "straight" or unnamed: any completed maneuver counts.
    return True


# --------------------------------------------------------------------------- #
# Per-step progress                                                           #
# --------------------------------------------------------------------------- #

class JunctionCueLedger:
    """Per-junction cue answers, gathered on APPROACH and sealed on ARRIVAL.

    WHY THIS EXISTS, and why it is not a boolean. Three separate failures share one cause
    -- asking "can you see a cone" of the here-and-now:

      (1) "turn at the intersection with the cone", where the map has a cone with no
          intersection, then an intersection with no cone. The object-only question fires
          on the lone cone and the robot turns at the wrong junction.
      (2) the query happens on arrival, by which point the cue is out of frame -- a
          verge-placed prop is ~100 deg off-axis at 5 m -- so a cue that was plainly
          visible on approach is never confirmed and the step times out.
      (3) "turn at the intersection BEFORE the one with the cone", which needs to know
          WHICH junction the cue belongs to.

    This decides the design. A SCOPED question -- "is there an X in the nearest
    intersection ahead?" -- discriminates this junction from the next at ~15 m before the
    junction. The VLM does not reliably attribute a prop to the SECOND intersection:
    ordinal and relational phrasings fail for the far junction even for a large prop
    plainly in frame. So per-frame lookahead is not assumed, and the ordinal is built by
    REMEMBERING each junction as it is passed.

    Hence: ask the scoped question on approach, seal the answer to the junction when the
    classifier says we have entered one, and keep the sealed answers in order. (1) is
    solved because the scoped question does not fire on a cone with no junction ahead and
    the pending answer is consumed by the next junction; (2) because the answer was taken
    while the cue was still visible; (3) because sealed answers can be counted.

    MAP-FREE. Cluster ids come from /predicted_cluster, which is perception, and distance
    from the odometer (integrated speed, never position). Nothing here reads a map.
    """

    def __init__(self, evidence_m: float = 12.0) -> None:
        #: How far a pending approach answer stays valid. Too short (under ~4 m) and the
        #: answer is forgotten before arrival; too long (~20 m+) and a stale one reaches
        #: the NEXT junction and fires there. 12 m sits inside that band.
        self.evidence_m = float(evidence_m)
        self._pending: bool | None = None
        self._pending_at: float | None = None
        #: (cluster id, answer) for each junction entered, in order.
        self.sealed: list[tuple[int, bool]] = []
        self._current: int | None = None

    # -- approach -------------------------------------------------------- #

    def note_approach(self, answer: bool | None, travelled: float) -> None:
        """One scoped-question answer taken while approaching a junction.

        None means unanswerable and is ignored rather than latched as False: a confident
        negative and "nobody could tell me" behave differently downstream, and collapsing
        them is how an unanswerable cue starts looking like a confident no.
        """
        if answer is None:
            return
        if answer:
            # A YES WHILE STANDING IN A JUNCTION UPGRADES THAT JUNCTION.
            #
            # The approach query cannot be dense enough to guarantee a reading inside the
            # ~15 m window the scoped question actually works in: each VLM call costs
            # ~1.5 s of WALL time while the simulator runs ~5x faster, so consecutive
            # queries are ~30 simulated metres apart however small the requested spacing.
            # So the far query can answer NO, be sealed, and the correct YES land just
            # after entry -- sealing a junction "cue absent" with the cone standing in it.
            #
            # "Is there an X in the nearest intersection ahead" asked from inside a
            # junction is still about THIS junction, so a positive is allowed to correct
            # the seal. Only upward: a late NO must not erase a sighting, or concern (2)
            # comes straight back.
            if self._current is not None:
                for i, (cid, ans) in enumerate(self.sealed):
                    if cid == self._current and not ans:
                        self.sealed[i] = (cid, True)
                        break
                return
            self._pending = True
            self._pending_at = travelled
        elif self._pending is None:
            # A NO does not overwrite a live YES. The cue can leave the frame on the run-in
            # -- that is concern (2) -- so the last word before arrival is not the most
            # reliable one; the sighting is. An unseen cue simply leaves the latch empty.
            self._pending = False
            self._pending_at = travelled

    def _fresh(self, travelled: float) -> bool:
        return (self._pending_at is not None
                and (travelled - self._pending_at) <= self.evidence_m)

    # -- arrival --------------------------------------------------------- #

    def note_enter(self, cluster: int, travelled: float) -> bool:
        """A junction begins. Seal the pending answer to it and return that answer."""
        answer = bool(self._pending) and self._fresh(travelled)
        self.sealed.append((int(cluster), answer))
        self._current = int(cluster)
        self._pending = None
        self._pending_at = None
        return answer

    def note_exit(self) -> None:
        self._current = None

    # -- reading --------------------------------------------------------- #

    @property
    def current_answer(self) -> bool | None:
        """Does the junction we are standing in carry the cue? None if we are not in one."""
        if self._current is None:
            return None
        for cid, ans in reversed(self.sealed):
            if cid == self._current:
                return ans
        return None

    def count_with_cue(self) -> int:
        """How many junctions so far carried the cue -- the ordinal, built by memory."""
        return sum(1 for _, a in self.sealed if a)

    def nth_with_cue(self, n: int) -> int | None:
        """Cluster id of the n-th (1-based) junction that carried the cue, or None."""
        hits = [cid for cid, a in self.sealed if a]
        return hits[n - 1] if 0 < n <= len(hits) else None

    def junction_before_cue(self) -> int | None:
        """The junction immediately BEFORE the first one carrying the cue.

        This is the lookahead the VLM cannot answer per-frame (it does not reliably
        attribute a prop to the second intersection). Sequential memory can, but only
        AFTER the cue junction has been reached -- so a plan that must ACT there needs the
        route traversed once, or the ordinal carried by the plan. Returning it here makes
        that limitation explicit rather than leaving it implied.
        """
        for i, (cid, ans) in enumerate(self.sealed):
            if ans:
                return self.sealed[i - 1][0] if i > 0 else None
        return None

    def reset(self) -> None:
        self._pending = None
        self._pending_at = None
        self.sealed.clear()
        self._current = None


# --------------------------------------------------------------------------- #
# Constraint severity: how hard is "stay on the sidewalk"?
# --------------------------------------------------------------------------- #

#: How hard a plan's constraint is. "the road is freshly tarred so stay on the
#: sidewalk AT ALL TIMES" and "stay on the sidewalk to avoid getting hit" are
#: different demands; without a severity both would compile to the same
#: `InvariantMonitor.observe(..., binding=True)`.
SEVERITIES = ("advisory", "preferred", "binding")

#: Default for any constraint the plan does not rate. DELIBERATELY the weakest.
#: Many generated `require_modes` constraints are self-defeating -- e.g. `Path` and
#: `Junction` accept disjoint cluster sets, so "stay on this road" + require:path
#: aborts at the first junction. Defaulting to `binding` would turn most
#: constraint-carrying plans into aborts even on ground-truth clusters once
#: policy=dwell is set. Until generation is fixed, an unrated constraint is
#: measured, never enforced.
DEFAULT_SEVERITY = "advisory"


def constraint_keys(plan: Mapping[str, Any]) -> set[str]:
    """Every constraint a plan actually states, in the `kind:mode` spelling.

    Same spelling `nl_planner/scripts/stl_ablation.py` uses (`forbid:`, `require:`,
    `hold:`) rather than a second vocabulary -- two spellings for one concept drift.
    """
    keys: set[str] = set()
    for m in plan.get("forbid_modes") or []:
        keys.add(f"forbid:{m}")
    for m in plan.get("require_modes") or []:
        keys.add(f"require:{m}")

    def _walk(steps) -> None:
        for st in steps or []:
            if st.get("hold_mode"):
                keys.add(f"hold:{st['hold_mode']}")
            for br in (st.get("branches") or []):
                _walk(br.get("sub_plan"))

    _walk(plan.get("steps"))
    return keys


def severity_of(table: Mapping[str, Any] | None, key: str,
                default: str = DEFAULT_SEVERITY) -> str:
    """Severity of one constraint from the TABLE, by its `kind:mode` key.

    Table-based because the consumer (brain_controller) keeps the plan's fields, not
    the plan dict, so handing it a whole plan would mean retaining one just for this.
    An unknown value falls back to the default rather than raising: `validate_severities`
    is where a bad table is reported, at load, once -- not on every cluster reading.
    """
    sev = (table or {}).get(key, default)
    return sev if sev in SEVERITIES else default


def is_binding_of(table: Mapping[str, Any] | None, key: str) -> bool:
    """Whether a breach of `key` should COST anything, from the table."""
    return severity_of(table, key) == "binding"


def severity_for(plan: Mapping[str, Any], key: str,
                 default: str = DEFAULT_SEVERITY) -> str:
    """Severity of one constraint, by its `kind:mode` key."""
    return severity_of(plan.get("constraint_severity"), key, default)


def is_binding(plan: Mapping[str, Any], key: str) -> bool:
    """Whether a breach of `key` should COST anything.

    `advisory` and `preferred` are both measured and neither breaches; they differ
    only in what a scorer weights, which is not this module's business. Only
    `binding` reaches `InvariantMonitor`'s dwell branch -- and even then nothing
    happens unless the policy is `dwell`, which is not the default.
    """
    return severity_for(plan, key) == "binding"


def validate_severities(plan: Mapping[str, Any]) -> list[str]:
    """Problems with a plan's `constraint_severity` table. Empty list == clean.

    TWO CHECKS, AND THE SECOND IS THE POINT. An unknown severity value is loud and
    easy. A severity keyed to a constraint THE PLAN DOES NOT STATE is silent: it
    reads as a hardened plan, governs nothing, and no run will ever tell you -- a
    flag that does nothing.
    """
    table = plan.get("constraint_severity")
    if not table:
        return []
    problems: list[str] = []
    if not isinstance(table, Mapping):
        return [f"constraint_severity must be a mapping, got {type(table).__name__}"]
    stated = constraint_keys(plan)
    for key, sev in table.items():
        if sev not in SEVERITIES:
            problems.append(
                f"constraint_severity[{key!r}] = {sev!r}; must be one of {SEVERITIES}")
        if key not in stated:
            problems.append(
                f"constraint_severity[{key!r}] governs nothing -- the plan states no "
                f"such constraint. Stated: {sorted(stated) or '(none)'}")
    return problems


class InvariantMonitor:
    """Tracks whether a step's mode invariant is holding, and for how long.

    THE THING `U` MEANS. ``\\Phi_{Path} \\mathbf{U} Detect(Cone)`` — "stay on the path
    until you see a cone" — has two halves. The cue terminating the step is the
    right half; this class covers the left one: noticing if the robot leaves
    ``path`` partway through. Without it, "stay on the walkway, do not cut across
    the grass" and "wander anywhere until you reach the plaza" are
    indistinguishable in the logs.

    ONE MECHANISM, THREE POLICIES. Detection is identical whichever policy you
    pick; only the response differs, so this is a knob rather than three code
    paths:

    ``off``    do not even measure (default — behaviour is byte-identical to before)
    ``log``    count and time violations, never act
    ``dwell``  a violation lasting ``dwell_frames`` consecutive readings BREACHES,
               and the caller decides what a breach costs. Hard-fail is this with
               ``dwell_frames=1``.

    WHY THE DEFAULT IS ``off`` AND THE RECOMMENDED FIRST STEP IS ``log``:
    a breach policy is only as good as the cluster labels underneath it, and on
    the campus bags junction-vs-path is at chance across bags. Enforcing an
    invariant on labels that good would abort on perception noise rather than on
    real violations. Measure the violation rate first; pick ``dwell_frames`` from
    that number instead of guessing it.
    """

    POLICIES = ("off", "log", "dwell")

    def __init__(self, policy: str = "off", dwell_frames: int = 5) -> None:
        if policy not in self.POLICIES:
            raise ValueError(f"policy must be one of {self.POLICIES}; got {policy!r}")
        self.policy = policy
        self.dwell_frames = max(1, int(dwell_frames))
        self.reset()

    def reset(self) -> None:
        self.run = 0                 # current consecutive violating readings
        self.violations = 0          # distinct violation episodes this step
        self.frames_violating = 0    # total violating readings this step
        self.longest_run = 0
        self.breached = False

    def observe(self, cluster: int, accept: set[int],
                binding: bool = True) -> bool:
        """Fold one cluster reading in. True when this reading BREACHES.

        ``binding`` lets the plan mark an invariant advisory ("follow the road
        past the cones") rather than binding ("do not cut across the grass"), so
        one policy can serve both without a second knob. An advisory invariant is
        still measured — it just never breaches.
        """
        if self.policy == "off":
            return False
        inside = (not accept) or (cluster in accept)
        if inside:
            self.run = 0
            return False
        if self.run == 0:
            self.violations += 1     # a new episode, not another frame of the old one
        self.run += 1
        self.frames_violating += 1
        self.longest_run = max(self.longest_run, self.run)
        if (self.policy == "dwell" and binding
                and self.run >= self.dwell_frames):
            self.breached = True
            return True
        return False

    def summary(self) -> dict:
        """Per-step record, for the log-only policy and for offline sweeps."""
        return {"policy": self.policy, "dwell_frames": self.dwell_frames,
                "violations": self.violations,
                "frames_violating": self.frames_violating,
                "longest_run": self.longest_run, "breached": self.breached}


class StepProgress:
    """Mutable per-step bookkeeping for the three triggers.

    One instance lives on the controller and is reset on every step change
    (advance, branch descend, plan swap).
    """

    def __init__(self, *, dwell_frames: int = 3, cue_lost_polls: int = 2,
                 require_cluster_change: bool = False,
                 start_cluster: int | None = None, resight_metres: float = 12.0,
                 is_decision_step: bool = False,
                 start_at_goal_mode: bool = False) -> None:
        self.dwell_frames   = max(1, int(dwell_frames))
        self.cue_lost_polls = max(1, int(cue_lost_polls))
        #: How far the robot must travel before the SAME cue counts as a NEW instance.
        #: 12 m: shorter than the spacing of two distinct roadside landmarks, longer than
        #: the jitter of one being re-acquired across a few frames. Only consulted when
        #: the caller passes `travelled`, so the temporal rule is unchanged without it.
        self.resight_metres = float(resight_metres)
        self._last_sight_at: float | None = None
        self.entry_travelled: float | None = None
        #: Heading change accumulated ONLY inside the manoeuvre's window (see
        #: note_heading). This is what `bearing_complete` should judge, not a raw
        #: yaw-minus-entry_yaw delta -- that delta counts every degree the road bends
        #: anywhere in the step and certified turns that never happened.
        self.bearing_accum = 0.0
        self._bearing_last_yaw: float | None = None
        # A step may only complete once the cluster has CHANGED since the step began.
        #
        # Why this is needed at all: the taxonomy's UPWARD subsumption puts a junction's
        # id inside the `path` mode too (a junction really is on a path), so a step
        # `junction -> path` is satisfied by standing still in the junction. A "second
        # intersection" mission can then advance several steps without leaving the FIRST
        # junction and fire its branch there: every step "succeeds", the mission is
        # meaningless.
        #
        # Requiring a change is what "transition" already meant. It cannot make a
        # reachable step unreachable, because the accept set is checked as well: the step
        # still needs the right KIND of region, it now also needs a different one.
        #
        # OFF by default so offline replay behaviour is unchanged — turning it on is a
        # policy change. The CARLA path turns it on.
        self.require_cluster_change = bool(require_cluster_change)
        #: A decision step keeps goal_mode == start_mode by design; see
        #: blocked_by_no_change for why it must be exempt from that guard.
        self.is_decision_step = is_decision_step
        #: True when the step BEGAN in a cluster whose OWN LABEL is the step's goal mode.
        #: Not accept-set membership -- see blocked_by_no_change for why that is different.
        self.start_at_goal_mode = start_at_goal_mode
        self.reset(start_cluster=start_cluster)

    def reset(self, entry_yaw: float | None = None,
              start_cluster: int | None = None,
              is_decision_step: bool | None = None,
              start_at_goal_mode: bool | None = None,
              entry_travelled: float | None = None) -> None:
        self.start_cluster = start_cluster
        self.dwell        = 0
        self.sightings    = 0
        self.absent_polls = 0
        self.in_view      = False
        self.entry_yaw    = entry_yaw
        self.bearing_accum = 0.0
        self._bearing_last_yaw = None
        #: Odometer reading when this step began, so a manoeuvre can be bounded in
        #: DISTANCE (see bearing_complete's scope_m). None keeps the old unbounded
        #: behaviour for callers that have no odometer.
        self.entry_travelled = entry_travelled
        # None means "leave as constructed", so a caller that does not know about these
        # keeps the behaviour it had.
        if is_decision_step is not None:
            self.is_decision_step = is_decision_step
        if start_at_goal_mode is not None:
            self.start_at_goal_mode = start_at_goal_mode
        # Set when a cue check has been given up on for THIS step, so the
        # controller does not immediately re-arm it. Cleared by
        # clear_cue_abandoned() once the robot leaves the step's accept set.
        self.cue_abandoned = False

    def note_heading(self, yaw: float | None, in_window: bool) -> None:
        """Fold one heading reading into the manoeuvre's accumulator.

        ``in_window`` is the CALLER's judgement of whether the robot is still where the
        manoeuvre is supposed to be happening -- in brain, inside the junction the branch
        was decided at, plus a bounded exit grace because a turn finishes as you leave.
        Readings outside the window are used to re-datum but never accumulate, so the
        road bending 150 m later cannot satisfy a junction turn (a raw yaw delta lets 60
        deg of route curvature complete `Bearing(Right)` while the junction itself saw
        only a few degrees).

        Integrating tick-to-tick rather than differencing against a fixed datum also makes
        the value independent of WHERE the datum was taken -- back-dating that origin was
        tried, and truncated every turn, because the approach heading counted toward the
        threshold.
        """
        if yaw is None:
            return
        last = self._bearing_last_yaw
        self._bearing_last_yaw = yaw
        if last is None or not in_window:
            return
        self.bearing_accum += math.degrees(wrap_pi(yaw - last))

    def abandon_cue(self) -> None:
        """Mark this step's cue as given up on (after a timeout)."""
        self.cue_abandoned = True
        self.dwell = 0

    def clear_cue_abandoned(self) -> bool:
        """Re-arm the cue. Returns True if it had been abandoned."""
        was = self.cue_abandoned
        self.cue_abandoned = False
        return was

    # -- traverse ---------------------------------------------------------- #

    def note_cluster(self, cluster: int, strict: set[int]) -> bool:
        """Advance/reset the dwell counter. True once the cluster has held."""
        if cluster in strict and not self.blocked_by_no_change(cluster):
            self.dwell += 1
            return self.dwell >= self.dwell_frames
        self.dwell = 0
        return False

    def blocked_by_no_change(self, cluster: int) -> bool:
        """True when ``require_cluster_change`` is on and we have not left the start.

        A DECISION STEP IS EXEMPT, and it has to be: the schema specifies that a step
        carrying `branches` keeps `goal_mode == start_mode` because *the robot stays put*
        while the branch is chosen. Requiring it to change cluster asks it to leave the
        junction it is deciding at, which nothing in the plan tells it to do.

        Without the exemption it is a race, not a constant failure. With the same plan at
        the same junction:
          - if the decision step begins while still on the approach path, start_cluster
            is the path; entering the junction IS a change -> it completes and decides
          - if it begins a few ticks later, already inside the junction, start_cluster is
            the junction itself -> permanently blocked while the targeter aims at the
            next junction beyond
        The outcome would be decided by control timing alone.
        """
        if self.is_decision_step:
            return False
        # A STEP THAT BEGAN AT ITS OWN DESTINATION IS ALREADY THERE.
        #
        # KEYED ON THE CLUSTER'S OWN LABEL, deliberately, and NOT on accept-set membership.
        # Upward subsumption puts a junction's id inside the `path` accept set, so a
        # `junction -> path` step standing in a junction is "in the accept set" without
        # having moved onto the path at all -- that is precisely what this guard exists to
        # catch, and keying on membership silently disabled it
        # (test_the_cluster_change_guard_applies_to_every_trigger). The guard exists to
        # stop "go to the next junction" being satisfied by the junction you are leaving --
        # and in that shape the step begins on a PATH, which is not in its accept set, so
        # the guard still fires. It must not fire when the step began inside the accept set,
        # because then there is nothing it could ever do to become unblocked.
        #
        # Same race as the decision-step case, one level below: if step 0 (a
        # straight-through manoeuvre) is still completing as the vehicle enters a
        # junction, step 1 -- "approach the next junction" -- begins already at the
        # junction it is asking for. Without this exemption it can never complete and the
        # vehicle stalls there while the targeter aims at the junction beyond.
        #
        # The junction->junction shape that would make this unsafe is already illegal for
        # non-decision steps (validate_plan_transitions: "junction is something you pass
        # THROUGH"), and the ordinal encoding path->path begins on a path region that is in
        # the accept set only once the NEXT one is reached -- checked in the unit test.
        if self.start_at_goal_mode:
            return False
        return (self.require_cluster_change
                and self.start_cluster is not None
                and cluster == self.start_cluster)

    # -- landmark ---------------------------------------------------------- #

    def note_cue(self, cue_found: bool, needed: int,
                 travelled: float | None = None) -> bool:
        """Fold one VLM poll into the ordinal counter.

        Returns True once the required Nth *distinct* sighting is reached.

        TWO WAYS TO SEPARATE INSTANCES, because one is not enough.

        The original rule was temporal: a sighting is distinct only after the cue has read
        NO for ``cue_lost_polls`` consecutive polls. That de-bounce is necessary -- without
        it "stop at the 2nd bench" is satisfied by two consecutive polls of the FIRST
        bench, since a landmark stays in frame for many seconds on approach.

        But it assumes the instances are separated IN TIME, and they often are not. Two
        benches thirty metres apart on a straight path are both in frame at once, the cue
        never drops, the counter sticks at 1, and the step waits until its cue budget
        expires -- it does not fail, it TIMES OUT, which is the mode that reads as a
        planner bug (``[1]*9`` yields one sighting, never satisfied).

        So there is now a SPATIAL gate as well. Pass ``travelled`` -- cumulative distance
        in metres, from odometry the executor already has -- and a new sighting is counted
        once the robot has moved ``resight_metres`` since the last one, even if the cue
        never went false. Either gate alone is sufficient to separate two instances.

        Spatial rather than instance identity: cue answers are booleans and carry no
        identity, so genuinely telling bench A from bench B needs a different cue
        contract. Distance is the cheap approximation that needs no perception change,
        and it fails safe -- if the robot has not moved, nothing is recounted.

        ``travelled=None`` keeps the pure temporal behaviour, so every existing caller
        and every recorded trace scores exactly as before.
        """
        needed = max(1, int(needed))
        if cue_found:
            moved_on = (travelled is not None
                        and self._last_sight_at is not None
                        and travelled - self._last_sight_at >= self.resight_metres)
            if not self.in_view or moved_on:
                self.in_view = True
                self.sightings += 1
                if travelled is not None:
                    self._last_sight_at = travelled
            self.absent_polls = 0
        else:
            self.absent_polls += 1
            if self.in_view and self.absent_polls >= self.cue_lost_polls:
                self.in_view = False
        return self.sightings >= needed


def branch_turns(branch: Mapping[str, Any]) -> bool:
    """Does this branch perform a LEFT or RIGHT turn? Straight does not count.

    Lives here rather than on the ROS node so it is testable without rclpy and shared with
    the offline harness -- a rule with two homes drifts.

    Straight is excluded deliberately. An instance-scoped prohibition on TURNING at the
    Nth junction must still allow passing through it; vetoing a straight branch would
    strand any mission whose route continues past the forbidden instance.
    """
    for st in (branch.get("sub_plan") or []):
        cue = str(st.get("transition_cue") or "")
        if "Bearing(" in cue and ("Right" in cue or "Left" in cue):
            return True
    return False


class InstanceForbidTracker:
    """Resolve "not the Nth X" to a region id, by counting at run time.

    WHY THIS CANNOT BE DONE EARLIER. `forbid_modes` is mode-scoped on purpose -- "do not
    enter the plaza" is a property of the mission, and hanging it off a step would stop
    applying the moment that step advanced. But an INSTANCE-scoped prohibition ("not the
    SECOND intersection") has no mode-level spelling at all, and both representations
    currently over-generalise it: the plan writes `forbid_modes: ['junction']` and the
    formula writes `G(not Phi_Junc)`, each of which forbids every junction on a route the
    same mission says to traverse. That contradiction is real and the containment check
    finds it.

    It cannot be resolved at materialisation either. A plan names modes, never ids -- that
    is what makes one plan portable between CARLA and the campus bags -- so resolving "the
    2nd junction" to id 60 would bake in one map. Worse, on a branching plan WHICH junction
    is second depends on the branch taken, which is not known until it is taken.

    So it resolves here, the same way ordinal CUES already do: count instances as they are
    entered, and when the Nth arrives, hand its id to the constraint monitor. The asymmetry
    this closes is that `cue_ordinal` has always counted instances for ADVANCEMENT while
    nothing counted them for PROHIBITION.

    Counts DISTINCT consecutive entries: re-entering the region you are already in is not a
    second instance, which is the same rule `require_cluster_change` encodes.
    """

    def __init__(self, specs: Iterable[Mapping[str, Any]] | None = None) -> None:
        #: [{"mode": "junction", "ordinal": 2}, ...]
        self.specs = [dict(x) for x in (specs or [])]
        self.counts: dict[str, int] = {}
        self.forbidden: set[int] = set()
        self._last: int | None = None

    def observe(self, cluster: int | None, label: str | None) -> set[int]:
        """Feed the current cluster and its mode label. Returns ids newly forbidden."""
        if cluster is None or label is None or cluster == self._last:
            return set()
        self._last = int(cluster)
        self.counts[label] = self.counts.get(label, 0) + 1
        new = set()
        for spec in self.specs:
            if str(spec.get("mode")) != label:
                continue
            try:
                want = int(spec.get("ordinal"))
            except (TypeError, ValueError):
                continue
            if self.counts[label] == want and int(cluster) not in self.forbidden:
                self.forbidden.add(int(cluster))
                new.add(int(cluster))
        return new

    @property
    def pending(self) -> bool:
        """True while some instance prohibition has not yet been resolved to an id."""
        for spec in self.specs:
            try:
                want = int(spec.get("ordinal"))
            except (TypeError, ValueError):
                continue
            if self.counts.get(str(spec.get("mode")), 0) < want:
                return True
        return False


class ConstraintMonitor:
    """Global negative constraint — `[]~X`. Counts, so the caller can decide.

    ROS-FREE AND SHARED, deliberately. brain_controller cannot be imported without
    rclpy, so anything living only in the node is untestable here and drifts from the
    offline harness. The same advancement policy written twice diverged in several places,
    so the counting lives here and both callers delegate.

    Reports a RATE, not a boolean. With a noisy cluster classifier a single reading inside a
    forbidden region is more likely to be a misclassification than a real incursion,
    and a boolean would make the metric a coin toss on the noisiest class. The caller
    picks the policy; this only measures.
    """

    def __init__(self, forbid: set[int] | None = None) -> None:
        #: Empty when the plan declares no constraint — then this is a true no-op and
        #: `satisfied` is vacuously True, which is what an undeclared constraint means.
        self.forbid = {int(c) for c in (forbid or ())}
        self.ticks = 0
        self.violations = 0
        self.longest_run = 0
        self._run = 0

    @property
    def declared(self) -> bool:
        return bool(self.forbid)

    @property
    def csr(self) -> float:
        """Constraint satisfaction rate. 1.0 when nothing was ever declared."""
        if not self.ticks:
            return 1.0
        return 1.0 - self.violations / self.ticks

    @property
    def satisfied(self) -> bool:
        return self.violations == 0

    def observe(self, cluster: int | None) -> bool:
        """Record one reading. Returns True if THIS reading violates."""
        if not self.forbid or cluster is None:
            return False
        self.ticks += 1
        if int(cluster) not in self.forbid:
            self._run = 0
            return False
        self.violations += 1
        self._run += 1
        self.longest_run = max(self.longest_run, self._run)
        return True

    def summary(self) -> dict:
        return {
            "forbid_declared": self.declared,
            "forbid_clusters": sorted(self.forbid),
            "constraint_ticks": self.ticks,
            "violation_ticks": self.violations,
            "longest_violation_run": self.longest_run,
            "csr": round(self.csr, 4),
            "constraint_satisfied": self.satisfied,
        }


# --------------------------------------------------------------------------- #
# The dispatch itself                                                         #
# --------------------------------------------------------------------------- #

def goal_reached(
    cluster: int,
    step: Mapping[str, Any],
    progress: StepProgress,
) -> tuple[bool, str]:
    """Has ``cluster`` satisfied ``step``'s CLUSTER condition?

    Returns ``(satisfied, reason)``; ``reason`` is a short tag for logging and for
    the ``/brain/state`` diagnostics, so it is always visible WHY a step advanced.

    Note this is only the cluster half. ``landmark`` steps still require the VLM
    cue (``StepProgress.note_cue``) and ``topology`` steps still require the
    heading change (``bearing_complete``) before the step actually advances —
    a True here only means "the cluster does not stand in the way".
    """
    trigger = trigger_of(step)
    strict = accept_set(step, degraded=False)
    degraded = accept_set(step, degraded=True)

    # A PURE DECISION STEP does not move, so nothing below applies to it.
    #
    # `junction -> junction` with branches means "stand in the junction and let the
    # branches decide" — `taxonomy.validate_plan_transitions` exempts exactly this
    # shape from the `X -> X` rule for that reason, and the generator emits it. Without
    # this branch `require_cluster_change` demands a region change from a step whose
    # whole point is not to move, and `StepTargeter` goes hunting for a SECOND junction
    # adjacent to the one we are standing in ("no 'junction' adjacent to ...; plan and
    # map disagree") -- the plan is right and the executor cannot run it.
    #
    # This is the exception to `require_cluster_change` "cannot make a reachable step
    # unreachable" (see StepProgress): it can, for a step that by definition stays put.
    #
    # Returning True here means only "the cluster does not stand in the way", which is
    # this function's documented contract — a landmark decision step still has its cue
    # checked by the caller.
    # STRICT, not degraded, and the difference is the whole guard. On town01 a
    # junction step's degraded set is the subsumption union — every region, because
    # every junction is also a path — so `cluster in degraded` is true EVERYWHERE and
    # the step would fire the moment the plan reached it, resolving its branches
    # wherever the robot happened to be. That is precisely the failure
    # `require_cluster_change` was added to stop ("fired its branch at the FIRST
    # junction"), reintroduced through the exemption. `accept_clusters` also documents
    # degraded as valid only for LANDMARK steps, and these are traverse.
    if step.get("branches") and step.get("start_mode") == step.get("goal_mode"):
        if cluster in strict:
            return True, (f"decision: standing in {step.get('goal_mode')!r} "
                          f"({cluster}) — the branches decide, no traversal")
        return False, (f"decision: {cluster} is not in {sorted(strict)}; not yet at "
                       f"the {step.get('goal_mode')!r} the decision is made at")

    # Applies to every trigger, not just traverse. A landmark step's cue check and a
    # topology step's bearing check would otherwise open while the robot is still in the
    # region the step started from — which is how the CARLA "second intersection" mission
    # fired its branch at the FIRST junction. See StepProgress.require_cluster_change.
    if progress.blocked_by_no_change(cluster):
        return False, (f"{trigger}: still in the start cluster {cluster} — a step must "
                       f"leave the region it began in")

    if trigger == "traverse":
        # The cluster IS the evidence: strict set, and it must hold.
        if progress.note_cluster(cluster, strict):
            return True, f"traverse: held {progress.dwell}/{progress.dwell_frames} in {sorted(strict)}"
        return False, f"traverse: dwell {progress.dwell}/{progress.dwell_frames}"

    if trigger == "landmark":
        # Permissive guard. Reaching here only opens the cue check.
        if cluster in strict:
            return True, "landmark: in strict accept set, awaiting cue"
        if cluster in degraded:
            return True, (
                f"landmark: DEGRADED match ({cluster} not in {sorted(strict)} but in "
                f"{sorted(degraded)}) — Detect(...) carries the evidence"
            )
        if not step.get("perception_backed", True):
            # The mode is a plan construct in this environment, so the classifier
            # will NEVER emit an id in either accept set. Without this guard the
            # cue check would never open and the plan would wedge silently — the
            # degraded set only rescues modes that happen to have a DEGRADE_TO
            # entry, which is not guaranteed. Same reasoning as the topology
            # branch below.
            return True, (
                "landmark: mode is a plan construct here (not perception-backed), "
                "awaiting cue"
            )
        return False, f"landmark: {cluster} outside degraded accept set {sorted(degraded)}"

    if trigger == "topology":
        # Heading change is authoritative. Where the mode is not perception-backed
        # in this environment there is no cluster to wait for at all.
        if not step.get("perception_backed", True):
            return True, "topology: mode is a plan construct here, awaiting bearing"
        if cluster in degraded:
            return True, "topology: plausible cluster, awaiting bearing"
        return False, f"topology: {cluster} outside {sorted(degraded)}"

    # Unknown trigger -> historical equality behaviour.
    ok = cluster == step.get("goal_cluster")
    return ok, f"unknown trigger {trigger!r}: equality fallback"


__all__ = [
    "ConstraintMonitor",
    "InstanceForbidTracker",
    "branch_turns",
    "TRIGGERS",
    "InvariantMonitor",
    "StepProgress",
    "accept_set",
    "bearing_complete",
    "bearing_direction",
    "goal_reached",
    "trigger_of",
    "wrap_pi",
]
