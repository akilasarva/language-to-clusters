"""The single answer to "has this plan step finished?".

Why this module exists
----------------------
That question was answered TWICE: once in ``brain_controller.py`` (the ROS node, spread
across ``_cluster_cb``, ``_odom_cb`` and ``_vlm_timer_cb``) and once inline in
``carla_gt_bridge/scripts/missions.py`` (the offline harness, which replaces CARLA with a
Python loop and so has to make the same decision). Two copies of one rule drift; these
diverged in three places:

===============================  ===========================  =========================
what diverged                    brain_controller             missions.py
===============================  ===========================  =========================
cluster-change guard             missing on the odom path     had it
landmark cue                     ran the ordinal machinery    assumed cues succeed
heading check                    ``bearing_complete()``       hand-rolled ``> 60 deg``
===============================  ===========================  =========================

The landmark-cue divergence let the offline harness pass a mission that ran away in CARLA.

So the policy lives here, in one place, with no ROS and no simulator. Both callers feed
it observations and read back a decision; neither decides anything itself. This is a
*move*, not a redesign — the rules are exactly those already in ``trigger_policy`` plus
the sequencing that was previously implicit in each caller's control flow.

The sequencing, stated once
---------------------------
A step completes when its CLUSTER condition holds and then, depending on the trigger:

``traverse``   nothing further. The cluster is the evidence.
``topology``   the heading must also have changed by ``bearing_complete_deg`` since the
               step began. "Turn right" is a maneuver, not something a camera sees.
``landmark``   a cue must be confirmed ``cue_ordinal`` times, each sighting distinct
               (the cue has to go false in between). A ``Detect(...)`` predicate carries
               the evidence; the cluster is only a permissive guard.

And, when enabled, across all three: the cluster must have CHANGED since the step began.
Upward subsumption puts a junction's id inside `path`, so ``junction -> path`` is
otherwise satisfied without moving.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from brain.trigger_policy import (StepProgress,
                                  bearing_complete as _bearing_complete,
                                  goal_reached as _goal_reached,
                                  trigger_of as _trigger_of)

__all__ = ["Decision", "StepAdvancer"]


@dataclass
class Decision:
    """What the caller should do with this observation."""

    #: the step is finished; advance (or descend into a branch)
    advanced: bool = False
    #: the cluster half holds but a cue is still owed — the caller should poll for it
    awaiting_cue: bool = False
    #: the cluster half holds but the maneuver has not happened yet
    awaiting_bearing: bool = False
    #: human-readable, and published on /brain/state so a log says WHY
    reason: str = ""
    #: set when the finished step has branches; the caller picks one
    branches: list = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return not self.advanced


class StepAdvancer:
    """Owns the "is this step done?" decision for exactly one step at a time.

    Stateful, because two of the three triggers are: dwell counts, cue sightings and the
    heading datum all accumulate over a step and reset when it changes. Call
    :meth:`begin_step` on every step change — advance, branch descend, plan swap.
    """

    def __init__(self, *, dwell_frames: int = 3, cue_lost_polls: int = 2,
                 bearing_complete_deg: float = 60.0,
                 require_cluster_change: bool = False,
                 bearing_scope_m: float | None = None) -> None:
        self._dwell_frames = dwell_frames
        self._cue_lost_polls = cue_lost_polls
        self._bearing_deg = bearing_complete_deg
        #: Distance bound on a Bearing(...) manoeuvre. Default None = unbounded, which is
        #: this class's existing behaviour and what its tests pin.
        #:
        #: CARRIED HERE EVEN THOUGH THIS CLASS IS NOT YET WIRED INTO BRAIN. It is the
        #: "offline twin" of brain's advance policy, and a twin that scores the same
        #: situation differently hides bugs in whichever copy is wrong. brain bounds
        #: Bearing(...) by distance because the unbounded check certified turns that
        #: never happened; the same bound must exist here before this class is used.
        self._bearing_scope_m = bearing_scope_m
        self._require_change = require_cluster_change
        self._step: Mapping[str, Any] | None = None
        self.progress = StepProgress(dwell_frames=dwell_frames,
                                     cue_lost_polls=cue_lost_polls,
                                     require_cluster_change=require_cluster_change)

    # -- lifecycle -------------------------------------------------------- #

    def begin_step(self, step: Mapping[str, Any] | None, *,
                   cluster: int | None = None, yaw: float | None = None,
                   start_label: str | None = None) -> None:
        """Start a new step. ``cluster`` and ``yaw`` are the data the guards compare to.

        ``cluster`` must be captured HERE rather than read later: by the time the goal is
        otherwise satisfied, the current cluster IS the answer and there is nothing left
        to compare against.
        """
        self._step = step
        self.progress = StepProgress(dwell_frames=self._dwell_frames,
                                     cue_lost_polls=self._cue_lost_polls,
                                     require_cluster_change=self._require_change,
                                     start_cluster=cluster,
                                     # `step` is None when the advancer is reset
                                     # between plans, not only on a real step.
                                     is_decision_step=bool(
                                         step.get("branches") if step else False),
                                     # Defaults to None, so a caller that does not supply
                                     # a label keeps the guard exactly as it was.
                                     start_at_goal_mode=bool(
                                         step is not None and start_label is not None
                                         and start_label == step.get("goal_mode")))
        self.progress.entry_yaw = yaw

    @property
    def step(self) -> Mapping[str, Any] | None:
        return self._step

    @property
    def trigger(self) -> str:
        return _trigger_of(self._step) if self._step else "traverse"

    # -- the decision ----------------------------------------------------- #

    def observe(self, *, cluster: int | None = None, yaw: float | None = None,
                cue_answer: bool | None = None,
                travelled: float | None = None) -> Decision:
        """Fold one observation in and say whether the step is finished.

        Every argument is optional because the callers observe at different rates and from
        different sources — brain gets clusters, odometry and cue answers on three
        separate topics; the harness has all three every tick. Passing only what changed
        is correct.
        """
        step = self._step
        if step is None:
            return Decision(reason="no step")

        if yaw is not None and self.progress.entry_yaw is None:
            self.progress.entry_yaw = yaw

        if cluster is None:
            return Decision(reason="no cluster observed yet")

        ok, why = _goal_reached(cluster, step, self.progress)
        if not ok:
            return Decision(reason=why)

        trig = _trigger_of(step)

        if trig == "topology":
            # The maneuver, not a camera: "turn right" is established by the heading
            # delta since the step began. Bags with no odometry never satisfy this and
            # fall back to the cluster/cue path, which is the intended degradation.
            if yaw is None or self.progress.entry_yaw is None:
                return Decision(awaiting_bearing=True,
                                reason=why + " (no heading yet)")
            entry_t = getattr(self.progress, "entry_travelled", None)
            since = None if (travelled is None or entry_t is None) else travelled - entry_t
            if not _bearing_complete(step, yaw, self.progress.entry_yaw,
                                     self._bearing_deg,
                                     travelled_since=since,
                                     scope_m=self._bearing_scope_m):
                return Decision(awaiting_bearing=True,
                                reason=why + " (awaiting the maneuver)")
            return self._finish(why + " (maneuver complete)")

        if trig == "landmark" and step.get("transition_cue"):
            if cue_answer is None:
                return Decision(awaiting_cue=True, reason=why)
            needed = int(step.get("cue_ordinal") or 1)
            if not self.progress.note_cue(cue_answer, needed):
                return Decision(awaiting_cue=True,
                                reason=(f"cue {step['transition_cue']!r} -> {cue_answer}, "
                                        f"sighting {self.progress.sightings}/{needed}"))
            return self._finish(f"cue {step['transition_cue']!r} confirmed, "
                                f"sighting {self.progress.sightings}/{needed}")

        return self._finish(why)

    def _finish(self, reason: str) -> Decision:
        return Decision(advanced=True, reason=reason,
                        branches=list((self._step or {}).get("branches") or []))
