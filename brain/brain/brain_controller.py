#!/usr/bin/env python3
"""
Brain controller node (ROS 2) — tree-aware (v2).

Drives a state machine that walks a TREE-shaped NavPlan. The plan is provided
by ``nl_planner``'s executor on a latched ``/brain/incoming_plan`` topic and
swapped in atomically when ``/brain/load_plan`` is called. The node starts in
``WAITING_FOR_PLAN`` and never reads ``plan.json`` from disk.

Plan navigation:
  - Linear steps: advance when the current_cluster reaches the step's
    goal_cluster AND, if a ``transition_cue`` is present, the VLM confirms it.
  - Decision steps (steps with a ``branches`` list): once arrived (+ cue
    confirmed if any), enter the ``DECIDING`` state and ask the VLM to pick
    among the branches' ``vlm_cue`` descriptions. Retries up to
    ``vlm_decide_max_attempts`` times with exponential backoff; on persistent
    failure, falls back to the "default" branch (the schema guarantees one
    such branch exists).

States: ``WAITING_FOR_PLAN`` -> ``NAVIGATING`` -> (``CHECKING_CUE``) ->
        (``DECIDING``) -> ``COMPLETE``. The two parenthesised states are
        optional per-step.

Subscribed topics:
  /predicted_cluster   (std_msgs/Int16)    — from live_cluster_inference_node
  <image_topic>        (sensor_msgs/Image) — camera feed (reused for cue and
                                              branch decisions)
  /brain/incoming_plan (std_msgs/String, transient_local) — latched tree
                                                            JSON from
                                                            nl_planner. Call
                                                            /brain/load_plan
                                                            to swap it in.

Published topics:
  /brain/state         (std_msgs/String)   — JSON snapshot, including
                                              ``branch_path`` and whether the
                                              current step is a decision step.

Services:
  /brain/load_plan     (std_srvs/Trigger)  — atomically swap the active plan.

ROS 2 parameters:
  image_topic              camera topic                        (default: /hamilton/.../rgb)
  vlm_check_interval       seconds between cue VLM polls       (default: 2.0)
  vlm_model                OpenAI model                        (default: gpt-4o)
  vlm_decide_max_attempts  retry cap for choose_branch         (default: 3)
  vlm_decide_backoff_s     base seconds between retries        (default: 1.0)
  plan_snapshot_path       if non-empty, every successful
                           /brain/load_plan writes the active
                           plan JSON (pretty-printed) here
                           atomically                          (default: disabled)
"""

import os
import json
import time
import base64
import threading
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Int16, String
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from std_srvs.srv import Trigger
import openai

from brain.plan_navigator import PlanNavigator, count_leaf_paths, unreachable_steps
from brain.cue_phrasing import cue_question, scoped_cue_question, is_perceptual
from brain.trigger_policy import (
    ConstraintMonitor,
    InstanceForbidTracker,
    branch_turns,
    InvariantMonitor,
    StepProgress,
    is_binding_of as _is_binding_of,
    validate_severities as _validate_severities,
    accept_set as _accept_set,
    bearing_complete as _bearing_complete,
    JunctionCueLedger,
    goal_reached as _goal_reached,
    trigger_of as _trigger_of,
    wrap_pi as _wrap_pi,
)


# Transient-local QoS so a late subscriber still receives the most recent plan.
LATCHED_QOS = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
)


def imgmsg_to_bgr(msg: Image) -> np.ndarray:
    """Decode a sensor_msgs/Image to a BGR numpy array without cv_bridge."""
    enc = msg.encoding.lower()
    if enc in ("mono8", "8uc1"):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif enc in ("rgb8",):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        return img[:, :, ::-1].copy()  # RGB -> BGR
    elif enc in ("bgr8",):
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3).copy()
    elif enc in ("rgba8",):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 4)
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif enc in ("bgra8",):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 4)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    else:
        raise ValueError(f"Unsupported image encoding: {msg.encoding}")


class State(Enum):
    WAITING_FOR_PLAN = "WAITING_FOR_PLAN"  # idle until /brain/load_plan succeeds
    NAVIGATING       = "NAVIGATING"        # moving toward goal cluster
    CHECKING_CUE     = "CHECKING_CUE"      # at goal, polling VLM for cue
    DECIDING         = "DECIDING"          # cue ok, picking among branches
    COMPLETE         = "COMPLETE"          # leaf path exhausted
    # A binding constraint or invariant was breached and the policy says stop. Terminal
    # and distinct from COMPLETE: a run that ended because it violated its specification
    # did not finish the mission, and collapsing the two would let a breach be scored as
    # a success. Both enforcement paths (`forbid_policy="fail"` and InvariantMonitor
    # breaches) assign this member; test_trigger_policy pins that it exists.
    ERROR            = "ERROR"             # specification breached; run aborted


class BrainController(Node):

    def __init__(self) -> None:
        super().__init__("brain_controller")

        # --- Parameters ---
        self.declare_parameter("image_topic",             "/carla/ego_vehicle/rgb_front/image")
        self.declare_parameter("vlm_check_interval",      2.0)
        self.declare_parameter("vlm_model",               "gpt-4o")
        self.declare_parameter("vlm_decide_max_attempts", 3)
        self.declare_parameter("vlm_decide_backoff_s",    1.0)
        # If non-empty, every time /brain/load_plan succeeds the active plan
        # JSON is pretty-printed to this file (atomically). Useful for
        # debugging — previously brain/plan.json was the static input; now it
        # serves as a "last loaded plan" snapshot you can `cat` at any time.
        self.declare_parameter("plan_snapshot_path",      "")
        # Odometry, used ONLY for relative heading change on `topology` steps
        # ("has the right turn completed yet?"). Never for metric goals — the
        # plan is map-free. Bags without an odom topic simply never satisfy a
        # topology step by heading; they fall back to the cluster/cue path.
        self.declare_parameter("odom_topic",              "/odom")
        # A `topology` step's Bearing(...) is complete once |heading change since
        # entering the step| exceeds this. 60 deg is a deliberately loose read of
        # "the turn happened" — we are detecting a maneuver, not measuring it.
        self.declare_parameter("bearing_complete_deg",    60.0)
        # A `traverse` step needs the cluster held for this many consecutive
        # readings before it counts. The cluster IS the evidence there, so a
        # single flickered frame must not advance the plan. The upstream
        # smoothing (W=5, hyst=0.6) already cuts most flicker; this is belt-and-braces.
        self.declare_parameter("traverse_dwell_frames",   3)
        # Give up polling for a cue after this long and revert to NAVIGATING, so
        # a cue that never appears cannot wedge the plan forever. 0 disables.
        self.declare_parameter("cue_timeout_s",           120.0)
        # CUE SEMANTICS. "legacy" = the original behaviour. "v2" =
        #   (a) SCOPED: a landmark sighting counts only while the current cluster's label is the step's
        #       goal mode ("the junction with the stop sign"), not on the road between junctions;
        #   (b) WINDOWED: an ordinal landmark step starts its count from the end of the last step that
        #       carried a cue or manoeuvre, not from its own start, so "drive straight ... turn at the 2nd
        #       stop sign" counts the sign passed during an unrolled traverse step.
        # Topic (ground-truth) cue source only for (b); the VLM path is unaffected.
        self.declare_parameter("cue_semantics",           "legacy")
        # Ordinal de-bounce: the cue must read NO this many consecutive times
        # before the next YES counts as a NEW sighting. Without this, "the 2nd
        # bench" would be satisfied by two polls of the SAME bench.
        self.declare_parameter("cue_lost_polls",          2)
        # What a violated constraint COSTS. Detection is identical either way; only the
        # response differs, which is why this is one knob and not two code paths.
        #   log   count and report, never act  (default — behaviour unchanged)
        #   fail  abort the plan on breach
        # Default `log` on purpose: a policy that can abort a run should have its
        # violation rate measured before it is armed.
        self.declare_parameter("forbid_policy",           "log")
        # A `hold_mode` step's invariant breaches after this many consecutive readings
        # outside the held mode. 1 is hard-fail; higher tolerates the burst
        # misclassification that real perception produces (typically a few frames),
        # at the cost of noticing a real departure later.
        self.declare_parameter("hold_dwell_frames",       5)
        # Require the cluster to CHANGE before a step may complete. The taxonomy's
        # upward subsumption puts a junction's id inside `path` too (a junction really is
        # on a path), so `junction -> path` is otherwise satisfied without moving, and a
        # "second intersection" mission can fire its branch inside the FIRST junction.
        #
        # Default False so offline replay behaviour is unchanged — this is a policy
        # change. The CARLA launch turns it on. See brain.trigger_policy.StepProgress.
        self.declare_parameter("require_cluster_change",  False)
        # Where cue answers and branch decisions come from:
        #   "vlm"   ask OpenAI about the camera image (default; unchanged behaviour)
        #   "topic" read the latest answer off `cue_topic`
        #
        # One seam, three uses: CARLA ground-truth cues (no camera needed), a real VLM,
        # and offline replay of cached answers.
        # Everything downstream — the ordinal de-bounce, the timeout, the branch pick — is
        # identical either way, because all of it consumes a single boolean.
        self.declare_parameter("cue_source",              "vlm")
        self.declare_parameter("cue_topic",               "/cue/confirmations")
        # How often to sample a TOPIC cue. Much faster than vlm_check_interval, and
        # deliberately a separate knob: that 2 s interval exists to limit what we spend on
        # OpenAI calls, and reading a topic costs nothing.
        #
        # The rate is not cosmetic. A sighting only becomes distinct after the cue reads
        # false for cue_lost_polls, and a vehicle can cross the gap between two cones in
        # well under a second — at 0.5 Hz the gap is never sampled, the second sighting
        # never counts, and "stop at the second cone" times out holding sighting 1. The
        # de-bounce must be sampled faster than the world changes.
        self.declare_parameter("cue_check_interval",      0.1)
        # Feed the odometer to StepProgress, enabling the SPATIAL de-bounce.
        #
        # `StepProgress.resight_metres` (12.0) only applies when `note_cue` receives the
        # odometer. It prevents this failure: two landmarks thirty metres apart are both
        # in frame at once, the cue never goes false, the counter sticks at 1, and the
        # step times out rather than failing.
        #
        # OFF by default because turning it on is a POLICY change that alters when an
        # ordinal step fires. Same treatment as `require_cluster_change`.
        self.declare_parameter("cue_spatial_debounce",    False)
        # How far a Bearing(...) manoeuvre may take before it stops counting.
        #
        # Without a bound, `Bearing(Right) completed` fires on ANY 60 deg of accumulated
        # heading change since the step began -- including ~150 m of ordinary road
        # curvature -- so a vehicle that drove straight through the junction can be
        # reported as having turned.
        #
        # 45 m: a junction manoeuvre completes well inside that (Town05 junction regions
        # are ~37 m median, ~47 m max), while it is short enough that route curvature
        # cannot accumulate the threshold. Set to 0 to disable the bound and reproduce the
        # old behaviour exactly.
        self.declare_parameter("bearing_scope_m",         45.0)
        # How far past the decision junction a manoeuvre may still accumulate heading.
        # A right turn completes as the vehicle EXITS, so the window cannot stop at the
        # cluster boundary or the turn is under-counted; 15 m is comfortably longer than
        # the exit arc and far shorter than the ~150 m of route curvature that can
        # otherwise satisfy the check.
        self.declare_parameter("bearing_exit_grace_m",    15.0)
        # SCOPED CONJUNCTION WITH SEQUENTIAL ACCUMULATION.
        #
        # Ask "is there an X in the NEAREST intersection ahead?" while approaching, seal
        # the answer to the junction on entry, and keep the sealed answers in order.
        # That scoped question discriminates this junction from the next at ~15 m, while
        # the unscoped "can you see an X" cannot tell a cone on the road from one at this
        # junction from one at the next -- and per-frame phrasings do not reliably reach
        # the SECOND intersection, so an ordinal has to be remembered.
        #
        # OFF by default: it changes when a cue is confirmed and therefore when steps
        # fire. Same treatment as `require_cluster_change` and `cue_spatial_debounce`.
        self.declare_parameter("cue_scoped_approach",     False)
        # How far a pending approach answer stays valid (metres). Too short (under ~4 m)
        # and the sighting is forgotten before arrival; too long (~20 m+) and a stale one
        # reaches the NEXT junction and fires there.
        self.declare_parameter("cue_evidence_m",          12.0)
        # Distance between approach queries. Bounds OpenAI spend the right way: by ground
        # covered rather than by wall time, so it cannot become a latency the vehicle
        # outruns (a latency that makes the branch decision miss the cue).
        self.declare_parameter("cue_approach_every_m",    8.0)

        self.image_topic              = self.get_parameter("image_topic").value
        vlm_check_interval            = float(self.get_parameter("vlm_check_interval").value)
        self.vlm_model                = self.get_parameter("vlm_model").value
        self._vlm_decide_max_attempts = int(self.get_parameter("vlm_decide_max_attempts").value)
        self._vlm_decide_backoff_s    = float(self.get_parameter("vlm_decide_backoff_s").value)
        self._plan_snapshot_path      = str(self.get_parameter("plan_snapshot_path").value).strip()
        self._odom_topic              = str(self.get_parameter("odom_topic").value)
        self._bearing_complete_deg    = float(self.get_parameter("bearing_complete_deg").value)
        self._bearing_scope_m         = float(self.get_parameter("bearing_scope_m").value)
        self._bearing_exit_grace_m    = float(self.get_parameter("bearing_exit_grace_m").value)
        self._cue_scoped_approach     = bool(self.get_parameter("cue_scoped_approach").value)
        self._approach_every_m        = float(self.get_parameter("cue_approach_every_m").value)
        self._last_approach_at        = -1e9
        self._cue_ledger = (JunctionCueLedger(
            evidence_m=float(self.get_parameter("cue_evidence_m").value))
            if self._cue_scoped_approach else None)
        self._traverse_dwell_frames   = int(self.get_parameter("traverse_dwell_frames").value)
        self._cue_timeout_s           = float(self.get_parameter("cue_timeout_s").value)
        self._cue_semantics           = str(self.get_parameter("cue_semantics").value or "legacy").strip().lower()
        #: v2 counting window: one entry per cluster entered since the last cue/manoeuvre step ended,
        #: {"cluster", "label", "answers"} with answers OR-merged over the whole stay in that cluster.
        self._cue_window: list[dict] = []
        self._answers_fresh = True            # False from a cluster change until the next cue answer arrives
        self._v2_counted: set = set()         # clusters already counted for THIS step (one sighting per cluster)
        print(f"[BRAIN] cue_semantics={self._cue_semantics}")
        self._cue_lost_polls          = int(self.get_parameter("cue_lost_polls").value)
        self._require_cluster_change  = bool(self.get_parameter("require_cluster_change").value)
        self._cue_spatial_debounce    = bool(self.get_parameter("cue_spatial_debounce").value)
        self._cue_source              = str(self.get_parameter("cue_source").value).strip().lower()
        if self._cue_source not in ("vlm", "topic"):
            raise ValueError(f"cue_source must be 'vlm' or 'topic', got {self._cue_source!r}")
        #: latest {cue_text: bool} from cue_topic, when cue_source == "topic"
        self._cue_answers: dict[str, bool] = {}

        # --- Internal state (guarded by self._lock) ---
        # No plan loaded yet — we idle in WAITING_FOR_PLAN until
        # /brain/load_plan succeeds (driven by nl_planner's executor).
        self._lock              = threading.Lock()
        self._navigator: PlanNavigator | None = None
        self._plan_name         = ""
        self._cluster_labels: dict[int, str] = {}
        self.state              = State.WAITING_FOR_PLAN
        self.current_cluster    = None
        self.latest_image       = None
        # Single in-flight VLM call across both cue checks and branch
        # decisions — they share the camera and the OpenAI client.
        self._vlm_busy          = False
        # Latched plan payload deposited by nl_planner's executor.
        self._pending_plan_json = None

        # --- Per-step trigger bookkeeping (see brain.trigger_policy) ---
        # Dwell counts, cue sightings and the heading datum for the CURRENT step.
        #: `[]~X`. Shared with the offline harness via trigger_policy so the two cannot
        #: drift; a no-op until a plan declares forbid_clusters.
        self._constraint = ConstraintMonitor()
        #: Instance-scoped prohibitions ("not the SECOND intersection"). A no-op
        #: until a plan declares `forbid_instances`.
        self._inst_forbid = InstanceForbidTracker()
        #: `[]X` — the mission-wide POSITIVE invariant, from `require_clusters`.
        #: Separate from `self._hold`, which is scoped to one step and dies with it;
        #: this one binds every branch for the life of the plan.
        self._require: InvariantMonitor | None = None
        self._require_accept: set[int] = set()
        self._plan_require_modes: list[str] = []
        #: `constraint_severity` from the plan: {"forbid:Sidewalk": "binding", ...}.
        #: Empty means every constraint is `advisory`, i.e. measured and never enforced.
        self._plan_severity: dict[str, str] = {}
        #: The left half of `Phi_X U cue` for the CURRENT step, or None when the step
        #: holds nothing. Rebuilt per step by _reset_step_progress.
        self._hold: InvariantMonitor | None = None
        # Reset on every step change via _reset_step_progress().
        self._progress = StepProgress(
            dwell_frames=self._traverse_dwell_frames,
            cue_lost_polls=self._cue_lost_polls,
            require_cluster_change=self._require_cluster_change,
            start_cluster=None,
        )
        # When the current CHECKING_CUE began, for the timeout.
        self._cue_started_at = None
        # Latest heading (radians). None until odom arrives / if there is no odom.
        self._yaw            = None
        #: Metres travelled, integrated from odometry speed. Scalar: no position.
        self._travelled      = 0.0
        #: (yaw, odometer) captured when DECIDING began, so the manoeuvre chosen there is
        #: measured from the decision point rather than from wherever the branch commits.
        self._decide_datum: tuple[float | None, float] | None = None
        #: Cluster the branch was decided at, and the odometer reading when we left it.
        self._decide_cluster: int | None = None
        self._left_decide_at: float | None = None
        self._odom_prev_t: float | None = None
        #: (wall time, odometer reading) when the newest camera frame arrived, so the age
        #: of the frame actually handed to the VLM can be reported in seconds AND metres.
        self._image_rx_t: float | None = None
        self._image_rx_travelled: float | None = None

        # --- OpenAI client ---
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            self.get_logger().warn(
                "OPENAI_API_KEY not set — VLM cue checks AND branch decisions will fail; "
                "decisions will fall back to the 'default' branch"
            )
        # Construct the client ONLY when there is a key. `openai.OpenAI(api_key=None)`
        # raises in the constructor, so the degraded mode promised by the warning above
        # would be unreachable.
        #
        # Running without a key is a first-class case, not an accident: ground-truth CARLA
        # missions need no VLM, and every consumer of `self._openai` is already
        # behind a cue or branch check. Guard on None there rather than at startup.
        self._openai = openai.OpenAI(api_key=api_key) if api_key else None

        # --- Subscribers ---
        self.create_subscription(Int16, "/predicted_cluster",   self._cluster_cb,       10)
        if self._cue_source == "topic":
            cue_topic = str(self.get_parameter("cue_topic").value)
            self.create_subscription(String, cue_topic, self._cue_answer_cb, 10)
            self.get_logger().info(
                f"cue_source=topic — cue answers and branch choices come from "
                f"{cue_topic}, no VLM will be called")
        self.create_subscription(Image, self.image_topic,       self._image_cb,         10)
        self.create_subscription(Odometry, self._odom_topic,    self._odom_cb,          10)
        # nl_planner pipe: latched plan JSON + Trigger to swap it in atomically.
        self.create_subscription(
            String, "/brain/incoming_plan", self._incoming_plan_cb, LATCHED_QOS,
        )

        # --- Services ---
        self._load_plan_srv = self.create_service(
            Trigger, "/brain/load_plan", self._load_plan_cb,
        )

        # --- Publisher ---
        self._state_pub = self.create_publisher(String, "/brain/state", 10)

        # --- VLM poll timer ---
        cue_interval = (float(self.get_parameter("cue_check_interval").value)
                        if self._cue_source == "topic" else vlm_check_interval)
        # THE POLL RATE AND THE SPEND LIMIT ARE TWO DIFFERENT THINGS.
        #
        # `vlm_check_interval` (2.0 s) exists to bound OpenAI spend while RE-POLLING a
        # cue. If it were also the latency before the FIRST look, brain would wait up to
        # 2 s after entering DECIDING before copying a frame -- at ~5 m/s that is ~10 m,
        # enough to drive past the cue, and the model then answers correctly about a
        # picture taken after the cone has gone behind it.
        #
        # So the timer ticks fast and the SPEND limit lives on the cue-repoll path. A
        # branch decision is a single call guarded by `_vlm_busy`, so firing it promptly
        # costs nothing extra.
        self._cue_repoll_interval = cue_interval
        self._last_cue_poll_t = 0.0
        self.create_timer(min(0.2, cue_interval), self._vlm_timer_cb)
        self.get_logger().info(
            f"cue polling every {cue_interval:.2f}s (source={self._cue_source}), "
            f"bearing scope {self._bearing_scope_m:.0f} m "
            f"(0 = unbounded, the old behaviour), "
            f"spatial de-bounce {'ON' if self._cue_spatial_debounce else 'OFF'} "
            f"(resight_metres={self._progress.resight_metres:.0f})")

        self._print_idle_banner()
        # Publish initial WAITING_FOR_PLAN so listeners (executor, UIs) see we're alive.
        self._publish_state()

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def _step(self) -> dict[str, Any] | None:
        if self._navigator is None:
            return None
        return self._navigator.current_step

    def _lbl(self, cluster_id) -> str:
        """Return 'Label (id=N)' if a label exists, otherwise just 'id=N'."""
        if cluster_id is None:
            return "None"
        label = self._cluster_labels.get(int(cluster_id))
        return f"{label} (id={cluster_id})" if label else f"id={cluster_id}"

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _cluster_cb(self, msg: Int16) -> None:
        new_cluster = int(msg.data)

        with self._lock:
            if new_cluster != self.current_cluster:
                print(f"\n[CLUSTER CHANGE] {self._lbl(self.current_cluster)} -> {self._lbl(new_cluster)}")
                self.get_logger().info(
                    f"Cluster transition: {self._lbl(self.current_cluster)} -> {self._lbl(new_cluster)}"
                )
                prev_cluster = self.current_cluster
                self.current_cluster = new_cluster
                if self._cue_semantics == "v2":
                    # EMPTY, not dict(self._cue_answers): the answers held at the instant of the change describe
                    # the cluster just LEFT (gt_cue_node republishes a moment later). Copying them attributes the
                    # previous junction's cue to the road after it, and can fire a branch one junction early.
                    self._cue_window.append({"cluster": new_cluster, "label": self._label_of(new_cluster), "answers": {}})
                    self._answers_fresh = False
                    self._cluster_changed_t = time.time()
                # SEAL THE APPROACH ANSWER ONTO THE JUNCTION WE HAVE JUST ENTERED.
                # The scoped question was asked while the cue was still in frame; this is
                # where that sighting becomes "THIS junction carries the cue". Leaving a
                # junction closes it, so the next approach starts clean.
                if self._cue_ledger is not None:
                    was_j = self._label_of(prev_cluster) == "junction"
                    now_j = self._label_of(new_cluster) == "junction"
                    if now_j and not was_j:
                        got = self._cue_ledger.note_enter(new_cluster, self._travelled)
                        self.get_logger().info(
                            f"junction {new_cluster}: cue {'PRESENT' if got else 'absent'} "
                            f"(sealed from the approach; "
                            f"{self._cue_ledger.count_with_cue()} with the cue so far)")
                    elif was_j and not now_j:
                        self._cue_ledger.note_exit()
                self._publish_state()

            # `[]~X` is checked on EVERY reading, not only on a transition and not only
            # while NAVIGATING: a constraint the robot violates while waiting on a cue is
            # still violated. Counted before the navigation guard below for that reason.
            self._check_constraint(new_cluster)
            if self._check_require(new_cluster):
                return

            # `Phi_X U cue` is checked on EVERY reading too, and for the SAME reason the
            # two above are. Behind the NAVIGATING guard below it would be unobservable
            # on exactly the step shape it exists for: a `landmark` step whose accept set
            # is broad satisfies `goal_reached` on its first reading, so the brain enters
            # CHECKING_CUE immediately and waits there for the cue. The hold covers the
            # run-up to that cue -- that IS the "until" -- and the run-up is spent in
            # CHECKING_CUE, where the guard would skip the check entirely.
            if self._step is not None and self._check_hold(new_cluster):
                return

            # Only respond to cluster transitions while actively navigating.
            if self.state != State.NAVIGATING or self._step is None:
                return

            # Leaving the step's accept set re-arms a cue we had given up on:
            # the robot has actually moved, so the next arrival is a fresh
            # opportunity rather than the same failed one.
            if (self._progress.cue_abandoned
                    and new_cluster not in _accept_set(self._step, degraded=True)):
                self._progress.clear_cue_abandoned()
                self.get_logger().info(
                    f"left the accept set at {self._lbl(new_cluster)} — cue re-armed"
                )

            satisfied, why = _goal_reached(new_cluster, self._step, self._progress)
            if not satisfied:
                self.get_logger().debug(f"cluster condition not met — {why}")
                return

            goal = self._step["goal_cluster"]
            print(
                f"\n[BRAIN] Cluster condition satisfied for {self._lbl(goal)} "
                f"(trigger={_trigger_of(self._step)}, at {self._lbl(new_cluster)}, "
                f"step {self._navigator.step_idx}, "
                f"path={self._navigator.branch_path})\n"
                f"        {why}"
            )
            self.get_logger().info(f"Cluster condition satisfied — {why}")
            self._on_arrived_at_goal()

    def _check_constraint(self, cluster: int) -> None:
        """Global negative constraint. Caller holds self._lock.

        Reports a constraint-satisfaction RATE rather than a boolean, because with a
        noisy cluster classifier a single frame inside a forbidden region is more likely
        to be a misclassification than a genuine incursion, and a boolean would make the
        metric a coin toss on the noisiest class.
        """
        # Count instances BEFORE the constraint check, so the Nth is already forbidden
        # on the tick it is entered rather than one tick late.
        newly = self._inst_forbid.observe(cluster, self._lbl(cluster))
        if newly:
            self.get_logger().info(
                f"[FORBID] instance rule resolved to cluster(s) {sorted(newly)}")
        if not self._constraint.observe(cluster):
            return
        c = self._constraint
        msg = (f"CONSTRAINT VIOLATED: {self._lbl(cluster)} is in the forbidden set "
               f"{sorted(c.forbid)} — {c.violations} tick(s), csr={c.csr:.3f}")
        if str(self.get_parameter("forbid_policy").value) == "fail":
            self.get_logger().error(msg + " — policy=fail, aborting the plan")
            self.state = State.ERROR
            self._publish_state()
        else:
            self.get_logger().warning(msg)

    def _check_require(self, cluster: int) -> bool:
        """Mission-wide `[]X`. True when it breached. Caller holds self._lock.

        Checked on EVERY reading and in every state, like the negative constraint: an
        invariant the robot breaks while waiting on a cue is still broken. Uses the same
        dwell as a step hold, so a burst misclassification does not abort a run — with a
        noisy cluster classifier, dwell=1 would fire on noise rather than on the robot
        actually leaving.
        """
        if self._require is None:
            return False
        # Severity comes from the PLAN. With the policy at its `off` default this
        # changes nothing; it only decides whether a `dwell` breach costs anything.
        # Unrated constraints default to `advisory` (see DEFAULT_SEVERITY -- many
        # generated require_modes are self-defeating).
        _binding = any(_is_binding_of(self._plan_severity, f"require:{m}")
                       for m in (self._plan_require_modes or []))
        if not self._require.observe(cluster, self._require_accept, binding=_binding):
            return False
        modes = ", ".join(self._plan_require_modes) or "?"
        self.get_logger().error(
            f"MISSION INVARIANT BREACHED: left {modes} for "
            f"{self._require.longest_run} consecutive readings (now "
            f"{self._lbl(cluster)}) — the plan required it throughout")
        self.state = State.ERROR
        self._publish_state()
        return True

    def _check_hold(self, cluster: int) -> bool:
        """`Phi_X U cue`, left half. True when the invariant breached. Holds self._lock.

        The right half is a cue terminating the step; this is the left: noticing if the
        robot leaves the mode it was told to stay in. Without it, "stay on the walkway
        until the plaza" and "wander anywhere until the plaza" are indistinguishable in
        the logs.
        """
        if self._hold is None:
            return False
        accept = {int(c) for c in (self._step.get("hold_accept_clusters") or [])}
        _held = self._step.get("hold_mode")
        _binding = (_is_binding_of(self._plan_severity, f"hold:{_held}")
                    if _held else False)
        if not self._hold.observe(cluster, accept, binding=_binding):
            return False
        held = self._step.get("hold_mode")
        self.get_logger().error(
            f"INVARIANT BREACHED: left {held!r} for {self._hold.longest_run} "
            f"consecutive readings (now {self._lbl(cluster)}) — `{held} U "
            f"{self._step.get('until')}` does not hold")
        self.state = State.ERROR
        self._publish_state()
        return True

    # ------------------------------------------------------------------
    # Trigger dispatch — "which cluster gets used when"
    # ------------------------------------------------------------------

    def _reset_step_progress(self) -> None:
        """Clear all per-step trigger bookkeeping. Caller holds self._lock.

        ``start_cluster`` is the cluster we are standing in as the step begins, which is
        what ``require_cluster_change`` compares against. It must be captured here rather
        than read later: by the time the step's goal is otherwise satisfied, the current
        cluster IS the answer and there is nothing left to compare.
        """
        # Per-step flags the no-change guard needs. Both are properties OF THIS STEP, so
        # they must be refreshed here and not left over from the previous one:
        #   is_decision_step   a branching step keeps goal_mode == start_mode by design
        #                      (schemas.py) -- the robot stays put while the branch is
        #                      chosen, so requiring it to change cluster deadlocks it.
        #   start_at_goal_mode the step began standing in a cluster whose OWN label is its
        #                      goal mode, i.e. it is already where it was going.
        # THE DECISION-JUNCTION BEARING WINDOW IS GOOD FOR ONE STEP, like the targeter's
        # `_decide_at`. If `_decide_cluster` were never cleared, `bearing_accum` would only
        # ever grow NEAR THE DECISION JUNCTION -- and because `bearing_complete` returns
        # early whenever `accumulated_deg` is supplied, it would then ignore both `scope_m`
        # and the entry-yaw comparison. A later turn in the same sub-plan happens somewhere
        # else entirely, so its heading change would never be counted and the step could
        # not complete (e.g. a `Bearing(Left)` one junction after the decision junction).
        # Clearing the window restores the unwindowed comparison, which
        # `_windowed_bearing`'s own docstring names as the correct behaviour for a plain
        # topology step -- and that comparison is still bounded by `bearing_scope_m`, so
        # the 150 m-of-road-curvature failure the window was introduced to stop cannot
        # come back.
        if self._navigator is not None and self._navigator.step_idx != 0:
            if self._decide_cluster is not None:
                self.get_logger().info(
                    f"bearing window from the decision junction {self._decide_cluster} "
                    f"released at sub-plan step {self._navigator.step_idx}; this "
                    f"manoeuvre is measured where it happens")
            self._decide_cluster = None
            self._left_decide_at = None

        step = self._step or {}
        here = self.current_cluster
        label = self._cluster_labels.get(int(here)) if here is not None else None
        self._progress.reset(
            entry_yaw=self._yaw,
            start_cluster=here,
            is_decision_step=bool(step.get("branches")),
            start_at_goal_mode=bool(label is not None
                                    and label == step.get("goal_mode")))
        # AFTER reset(), which sets entry_yaw and clears the odometer datum.
        #
        # A NEW STEP GETS ITS FIRST LOOK IMMEDIATELY: without this the repoll limiter
        # carries the previous step's timestamp forward and the cue is asked up to
        # cue_repoll_interval late -- enough to drive past the cone.
        self._last_cue_poll_t = 0.0
        self._progress.entry_travelled = self._travelled
        if self._cue_semantics == "v2":
            self._v2_counted = set()
            self._v2_step_win0 = max(len(self._cue_window) - 1, 0)      # the entry we stand in as the step begins
        if (self._cue_semantics == "v2" and self._cue_source == "topic" and step.get("transition_cue")
                and _trigger_of(step) == "landmark"):
            goal, cue = step.get("goal_mode"), step["transition_cue"]
            pre = self._v2_count(cue, goal)
            if pre:
                self._progress.sightings = pre
                print(f"[BRAIN] v2 window: {pre} earlier {goal} sighting(s) of {cue!r} carried into this step")
        # reset() already zeroed bearing_accum; re-seed the integrator's datum so the
        # first reading after a step change does not fold in a stale delta.
        self._progress._bearing_last_yaw = self._yaw

        # The manoeuvre's yaw datum is deliberately NOT back-dated to the decision point.
        # The approach into the junction already contains heading change; counting that
        # toward the 60 deg threshold makes brain certify the turn EARLY and end the step
        # while the vehicle is still turning (turns are truncated at a consistent ~45 deg).
        # `_decide_datum` is still captured, because the DISTANCE half of it is sound and
        # is what a future scope-from-the-junction would need; only the yaw must not be
        # back-dated.

        self._cue_started_at = None
        self._hold = self._make_hold_monitor()

    def _make_hold_monitor(self) -> "InvariantMonitor | None":
        """The left half of `Phi_X U cue` for the current step, or None.

        A dwell step declares `hold_mode` and the materialiser resolves it to
        `hold_accept_clusters`. Brain never re-derives that set: the taxonomy lives on
        the planner side, and a second derivation here is a second thing to go stale.

        Uses the STRICT set. A held invariant is the one place permissiveness is exactly
        wrong — the whole assertion is "the robot did not leave this mode", and a
        degraded set that admits the coarser parent would make the claim vacuous.
        """
        step = self._step
        if not step or not step.get("hold_mode"):
            return None
        ids = step.get("hold_accept_clusters") or []
        if not ids:
            # Fail loudly. Silently not monitoring is how a plan comes to claim an
            # invariant it is not getting.
            self.get_logger().error(
                f"step declares hold_mode={step['hold_mode']!r} but carries no "
                f"hold_accept_clusters — the invariant is NOT being enforced")
            return None
        return InvariantMonitor(
            policy="dwell",
            dwell_frames=int(self.get_parameter("hold_dwell_frames").value))

    def _odom_cb(self, msg: Odometry) -> None:
        """Track heading and DISTANCE TRAVELLED. No position is ever used.

        The odometer is a scalar integrated from the twist's linear speed, so the
        map-free property is intact: nothing here knows where the robot is, only how far
        it has gone. Two things need it:

        * `StepProgress.note_cue(travelled=...)` -- the SPATIAL de-bounce
          (`resight_metres`). Without the odometer, two landmarks both in frame at once
          count as one sighting and the step times out.
        * IMAGE AGE IN METRES -- how far the robot moved between the frame being captured
          and the VLM being asked about it. Seconds are the wrong unit: what changes the
          cue's apparent size is distance. A 4 s old frame at 5 m/s was taken 20 m further
          back, where the cue is noticeably harder to detect.
        """
        q = msg.pose.pose.orientation
        # Yaw from quaternion (z-axis rotation); roll/pitch are irrelevant here.
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = float(np.arctan2(siny_cosp, cosy_cosp))
        v = msg.twist.twist.linear
        speed = float(np.hypot(v.x, v.y))
        now = time.time()
        with self._lock:
            # Integrate on WALL time between odometry messages. dt is clamped: a gap from
            # a stalled bridge would otherwise add metres the robot never travelled, and
            # over-counting distance is the direction that silently splits one sighting
            # into two.
            if self._odom_prev_t is not None:
                dt = now - self._odom_prev_t
                if 0.0 < dt < 1.0:
                    self._travelled += speed * dt
            self._odom_prev_t = now
            self._yaw = yaw
            # WINDOWED HEADING ACCUMULATION. A manoeuvre counts only while the robot is
            # still at the junction the branch was decided at, plus `bearing_exit_grace_m`
            # metres after leaving it -- a right turn finishes as you exit, so cutting the
            # window at the cluster boundary alone would under-count it.
            #
            # Differencing current yaw against the step's entry yaw instead lets ~150 m of
            # ordinary route curvature satisfy `Bearing(Right)` while the junction itself
            # saw only a few degrees.
            if self._decide_cluster is not None:
                here = self.current_cluster
                if here is not None and here != self._decide_cluster:
                    if self._left_decide_at is None:
                        self._left_decide_at = self._travelled
                elif here == self._decide_cluster:
                    self._left_decide_at = None
            in_window = (
                self._decide_cluster is not None
                and (self._left_decide_at is None
                     or (self._travelled - self._left_decide_at)
                     <= self._bearing_exit_grace_m))
            self._progress.note_heading(yaw, in_window)
            if self._progress.entry_yaw is None:
                self._progress.entry_yaw = yaw
            if self.state != State.NAVIGATING or self._step is None:
                return
            if _trigger_of(self._step) != "topology":
                return
            # The cluster guard applies here too. This path advances a topology step
            # from ODOMETRY, bypassing goal_reached() entirely — so without this check it
            # would also bypass `require_cluster_change`, and a `junction -> path` step
            # would complete on the heading change alone while the vehicle is still inside
            # the junction: the turn is real, but arrival on the `path` is never
            # established.
            if self._progress.blocked_by_no_change(self.current_cluster):
                return
            if not _bearing_complete(
                self._step, self._yaw, self._progress.entry_yaw,
                self._bearing_complete_deg,
                travelled_since=self._travelled_this_step(),
                scope_m=self._bearing_scope_m,
                accumulated_deg=self._windowed_bearing(),
            ):
                return
            print(
                f"\n[BRAIN] Bearing complete for step "
                f"{self._navigator.step_idx} ({self._step.get('transition_cue')!r})"
            )
            self.get_logger().info("Topology step satisfied by heading change")
            self._on_arrived_at_goal()

    def _on_arrived_at_goal(self) -> None:
        """Decide whether to wait for a cue, branch, or simply advance.

        Caller holds self._lock; current step is non-None.
        """
        step = self._step
        cue  = step.get("transition_cue") if step else None
        trigger = _trigger_of(step) if step else "traverse"

        # A `topology` step's cue is a Bearing(...) MANEUVER, not something a
        # camera can see. Its completion is already established by the heading
        # delta in _odom_cb / _bearing_complete, so there is nothing left to
        # confirm visually.
        #
        # Without this guard the controller would ask the VLM "Is 'Bearing(Right)
        # completed' visible in this image?" — a category error that always
        # answers NO, times out after cue_timeout_s, reverts to NAVIGATING, and
        # is then immediately re-triggered by _odom_cb because the robot is still
        # turned: an infinite NAVIGATING<->CHECKING_CUE loop on every turn,
        # burning an OpenAI call every vlm_check_interval.
        # BUT "already established by the heading delta" only holds when we got here
        # from _odom_cb, which tests the heading before it calls. The CLUSTER path
        # (_cluster_cb -> _goal_reached -> here) never tests it, so the heading is
        # re-checked below; otherwise a topology step would advance the moment the
        # vehicle entered any cluster in the goal mode's accept set, turn or no turn.
        # Region-sequence scoring cannot see that failure, because both exits of the
        # junction are in the accept set.
        #
        # StepAdvancer.observe applies the same heading test (step_advancer.py).
        if cue and trigger == "topology":
            if not _bearing_complete(step, self._yaw, self._progress.entry_yaw,
                                     self._bearing_complete_deg,
                                     travelled_since=self._travelled_this_step(),
                                     scope_m=self._bearing_scope_m,
                                     accumulated_deg=self._windowed_bearing()):
                # Not a failure: the cluster half holds and the maneuver is still owed.
                # Stay in NAVIGATING; _odom_cb advances the step once the heading turns.
                self.get_logger().info(
                    f"topology step: cluster condition holds but {cue!r} has not "
                    f"happened yet — staying in NAVIGATING until the heading turns"
                )
                return
            self.get_logger().info(
                f"topology step: {cue!r} is a maneuver, already confirmed by "
                f"heading — skipping the visual cue check"
            )
            self._after_step_satisfied()
            return

        # A cue we already gave up on must not immediately re-arm — see
        # _cue_check_timed_out. It re-arms only after the robot leaves the
        # step's accept set (handled in _cluster_cb).
        if cue and self._progress.cue_abandoned:
            self.get_logger().debug(
                "cue previously timed out for this step; not re-arming until the "
                "robot leaves and re-enters the accept set"
            )
            return

        if cue:
            ordinal = int(step.get("cue_ordinal") or 1)
            suffix = f" (sighting #{ordinal})" if ordinal > 1 else ""
            print(f"[BRAIN] Entering CUE_CHECK — looking for: '{cue}'{suffix}")
            self.get_logger().info(f"Switching to CUE_CHECK for: '{cue}'{suffix}")
            self.state            = State.CHECKING_CUE
            self._cue_started_at  = time.time()
            self._publish_state()
            return

        # No cue: jump straight to deciding/advancing.
        self._after_step_satisfied()

    def _after_step_satisfied(self) -> None:
        """Cue confirmed (or no cue): branch or advance. Caller holds self._lock."""
        step = self._step
        if step is None or self._navigator is None:
            return
        if self._cue_semantics == "v2" and step.get("transition_cue"):
            # The next ordinal count starts after the junction where this instruction was EXECUTED -- its
            # start cluster -- not where it was confirmed: 'Bearing(Straight) completed' at a junction is only
            # confirmed once the heading settles, inside the NEXT junction, and clearing then would drop that junction's cue.
            w0 = min(getattr(self, "_v2_step_win0", 0), len(self._cue_window))
            idx = [i for i in range(w0, len(self._cue_window)) if self._cue_window[i]["label"] == "junction"]
            self._cue_window = self._cue_window[idx[0] + 1:] if idx else []

        if step.get("branches"):
            n_b = len(step["branches"])
            print(f"[BRAIN] Step is a decision point with {n_b} branches; entering DECIDING")
            self.get_logger().info(f"Entering DECIDING ({n_b} branches)")
            self.state = State.DECIDING
            # ANCHOR THE MANOEUVRE AT THE DECISION POINT.
            #
            # A turn is measured from where it was decided, not from wherever the robot
            # has got to by the time the branch commits. The VLM round-trip is wall-clock
            # bound while the simulator advances on sim time (~5x real here), so the
            # post-branch step BEGINS ~20-30 m past the junction. Taking
            # entry_yaw/entry_travelled there spends most of the
            # bearing budget before measurement starts, and the heading datum is the
            # already-turned-onto-the-exit heading rather than the approach heading.
            #
            # The TARGET has the same late-anchoring problem
            # (StepTargeter.target_for(from_region=...)); this is the other half.
            self._decide_datum = (self._yaw, self._travelled)
            #: The junction the branch is being decided AT. The manoeuvre that follows is
            #: measured only while the robot is still here (plus an exit grace), so road
            #: curvature elsewhere cannot satisfy it.
            self._decide_cluster = self.current_cluster
            self._left_decide_at = None
            self._publish_state()
            return  # _vlm_timer_cb will kick off choose_branch

        # Linear: advance one step.
        try:
            self._navigator.advance()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"navigator.advance() failed: {exc}")
            return

        if self._navigator.is_complete:
            self._enter_complete()
            return

        self.state = State.NAVIGATING
        # New step: dwell counts, cue sightings and the heading datum all restart.
        self._reset_step_progress()
        nxt = self._step
        print(
            f"\n[STEP] Advanced to step {self._navigator.step_idx}/"
            f"{self._navigator.n_steps_in_sub_plan - 1} "
            f"(path={self._navigator.branch_path})"
        )
        if nxt is not None:
            print(f"[STEP] Now navigating: -> {self._lbl(nxt['goal_cluster'])}")
            if nxt.get("branches"):
                print(f"[STEP] (this is a DECISION step with {len(nxt['branches'])} branches)")
            if nxt.get("transition_cue"):
                print(f"[STEP] Next cue to find: '{nxt['transition_cue']}'")
        print()
        self._publish_state()

    def _image_cb(self, msg: Image) -> None:
        try:
            img = imgmsg_to_bgr(msg)
            with self._lock:
                self.latest_image = img
                # STAMP THE FRAME, so downstream can say how old the image sent to the
                # VLM was -- a stale frame is otherwise indistinguishable from a model
                # error in the result.
                self._image_rx_t = time.time()
                self._image_rx_travelled = self._travelled
            return
        except Exception as exc:
            self.get_logger().warn(f"Image conversion failed: {exc}", throttle_duration_sec=5.0)

    # ------------------------------------------------------------------
    # nl_planner hot-swap (latched topic + Trigger service)
    # ------------------------------------------------------------------

    def _incoming_plan_cb(self, msg: String) -> None:
        """Cache the most recent plan JSON. /brain/load_plan reads it."""
        with self._lock:
            self._pending_plan_json = msg.data
        self.get_logger().info(
            f"[INCOMING PLAN] received {len(msg.data)} bytes on /brain/incoming_plan "
            f"(call /brain/load_plan to swap it in)"
        )

    def _load_plan_cb(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        """Atomically replace the active plan with the latched payload."""
        with self._lock:
            payload = self._pending_plan_json
            if not payload:
                response.success = False
                response.message = "no plan payload on /brain/incoming_plan yet"
                return response
            try:
                plan_data = json.loads(payload)
                new_steps = plan_data.get("steps") or []
                if not new_steps:
                    raise ValueError("plan has no steps")
                navigator = PlanNavigator(new_steps)
                new_labels = {
                    int(k): v for k, v in plan_data.get("cluster_labels", {}).items()
                }
            except Exception as exc:  # noqa: BLE001
                response.success = False
                response.message = f"failed to parse incoming plan: {exc}"
                self.get_logger().error(response.message)
                return response

            self._navigator      = navigator
            self._cluster_labels = new_labels
            self._plan_name      = plan_data.get("plan_name", "Unnamed")
            # PLAN-LEVEL, not per-step: `[]~X` is a property of the whole mission, so it
            # must survive a branch descent. The materialiser resolves forbid_modes to
            # cluster ids; brain never re-derives them, so there is one definition.
            self._constraint = ConstraintMonitor(plan_data.get("forbid_clusters"))
            # INSTANCE-SCOPED, resolved at RUN TIME and not at materialisation: a plan
            # names modes rather than ids (which is what makes it portable), and on a
            # branching plan WHICH junction is second depends on the branch taken.
            self._inst_forbid = InstanceForbidTracker(
                plan_data.get("forbid_instances"))
            req = {int(c) for c in (plan_data.get("require_clusters") or [])}
            self._require_accept = req
            self._plan_require_modes = list(plan_data.get("require_modes") or ())
            self._plan_severity = dict(plan_data.get("constraint_severity") or {})
            # Loud at load, once. A severity keyed to a constraint the plan does not
            # state governs nothing and no run would ever report it.
            for _problem in _validate_severities(plan_data):
                self.get_logger().error(f"PLAN SEVERITY: {_problem}")
            self._require = InvariantMonitor(
                policy="dwell",
                dwell_frames=int(self.get_parameter("hold_dwell_frames").value)
            ) if req else None
            self.state           = State.NAVIGATING
            self._vlm_busy       = False
            self._reset_step_progress()
            self._publish_state()

            n_root  = len(new_steps)
            n_paths = count_leaf_paths(new_steps)
            dead    = unreachable_steps(new_steps)

        banner = (
            f"\n{'=' * 60}\n"
            f"[BRAIN] HOT-SWAP plan: {self._plan_name} "
            f"({n_root} root steps; {n_paths} leaf paths through the tree)\n"
            f"{'=' * 60}\n"
        )
        print(banner)
        self.get_logger().info(
            f"[LOAD PLAN] swapped in {self._plan_name!r}: "
            f"{n_root} root step(s), {n_paths} leaf path(s); state=NAVIGATING"
        )
        # Steps written after a decision step can never run — descend() replaces
        # the sub_plan and never returns to it. nl_planner's schema rejects that
        # shape, but hand-written plans and anything published straight onto
        # /brain/incoming_plan bypass pydantic entirely. Loudly, at load: a plan
        # that silently skips its tail otherwise looks like a clean success.
        if dead:
            warning = (
                f"[LOAD PLAN] {len(dead)} step(s) in this plan are UNREACHABLE and "
                f"will never run — they follow a decision step in the same list, "
                f"and a branch never returns to the list it forked from: {dead}. "
                f"Move each one into the branch it belongs to."
            )
            print(f"\n*** {warning} ***\n")
            self.get_logger().error(warning)
        # Best-effort snapshot to disk (outside the lock — file I/O is slow).
        self._maybe_snapshot_plan(plan_data)
        response.success = True
        response.message = f"loaded {self._plan_name!r} ({n_root} root steps, {n_paths} leaf paths)"
        return response

    def _maybe_snapshot_plan(self, plan_data: dict[str, Any]) -> None:
        """Pretty-print the active plan to ``plan_snapshot_path`` (if configured).

        Writes through a temp file + ``os.replace`` so a reader (editor, tail,
        another process) never sees a half-written JSON document. Failures are
        logged at WARN level and never propagated — snapshotting is a debug
        convenience, not a correctness requirement.
        """
        if not self._plan_snapshot_path:
            return
        target = Path(self._plan_snapshot_path).expanduser()
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(target.suffix + ".tmp")
            tmp.write_text(json.dumps(plan_data, indent=2) + "\n", encoding="utf-8")
            os.replace(tmp, target)
            self.get_logger().info(f"[LOAD PLAN] snapshotted plan JSON to {target}")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(
                f"[LOAD PLAN] failed to snapshot plan to {target}: {exc}"
            )

    # ------------------------------------------------------------------
    # VLM poll timer — dispatches cue checks AND branch decisions
    # ------------------------------------------------------------------

    def _vlm_timer_cb(self) -> None:
        with self._lock:
            if self._vlm_busy:
                return
            step = self._step
            if step is None:
                return

            # APPROACH QUERY. The cue must be asked BEFORE arrival or a sighting cannot
            # exist to seal: a verge prop is ~100 deg off-axis at 5 m, so by the time
            # the junction is entered there may be nothing in frame. Rate-limited by
            # DISTANCE, not time -- a time-based spend limit doubling as a poll rate
            # lets the vehicle drive past the cue.
            # WHAT TO ASK ABOUT ON APPROACH. The step being driven usually has no cue --
            # a branch's question lives on the BRANCH, one step later -- so asking only
            # about `transition_cue` asks nothing at all on the way into a decision point,
            # and the junction gets sealed "cue absent" with the cone standing in it.
            # Fall back to the upcoming branch's cue.
            approach_cue = step.get("transition_cue")
            if not approach_cue and self._navigator is not None:
                approach_cue = self._navigator.upcoming_branch_cue()
            if (self._cue_ledger is not None and self.state == State.NAVIGATING
                    and self.latest_image is not None and approach_cue):
                q = scoped_cue_question(approach_cue)
                moved = self._travelled - self._last_approach_at
                if q and moved >= self._approach_every_m:
                    self._last_approach_at = self._travelled
                    self._vlm_busy = True
                    threading.Thread(
                        target=self._run_scoped_approach,
                        args=(q, self.latest_image.copy(), self._travelled),
                        daemon=True).start()
                    return

            if self.state == State.CHECKING_CUE:
                if self._cue_check_timed_out():
                    return
                # PREFER THE SEALED ANSWER. It was taken on approach while the cue was
                # still in frame and is scoped to THIS junction, so it answers both
                # "the cue is not here, it is at the next one" and "the cue has left the
                # frame since". Falls through to the live query when there is none.
                if self._cue_ledger is not None:
                    sealed = self._cue_ledger.current_answer
                    if sealed is not None:
                        if self._register_cue_observation(sealed):
                            self.get_logger().info(
                                f"cue confirmed from the approach sighting sealed to "
                                f"junction {self.current_cluster}")
                            self._after_step_satisfied()
                        return
                if self._cue_source == "topic":
                    answer = self._topic_cue_answer(step["transition_cue"] or "")
                    if answer is None:
                        self.get_logger().warn(
                            f"no answer yet for cue {step['transition_cue']!r} on "
                            f"{self.get_parameter('cue_topic').value}",
                            throttle_duration_sec=5.0)
                        return
                    if self._register_cue_observation(answer):
                        self.get_logger().info(
                            f"cue confirmed from topic: {step['transition_cue']!r}")
                        self._after_step_satisfied()
                    return
                if self.latest_image is None:
                    self.get_logger().warn(
                        "No image available for VLM cue check",
                        throttle_duration_sec=5.0,
                    )
                    return
                # Spend limit lives HERE now: a cue is re-asked at most every
                # cue_repoll_interval. The first look after entering CHECKING_CUE is
                # immediate, because _last_cue_poll_t is reset when the step does.
                if (time.time() - self._last_cue_poll_t) < self._cue_repoll_interval:
                    return
                self._last_cue_poll_t = time.time()
                self._vlm_busy = True
                cue        = step["transition_cue"]
                image_copy = self.latest_image.copy()
                age_snap = self._snapshot_frame_age()
                threading.Thread(
                    target=self._run_vlm_cue,
                    args=(cue, image_copy, age_snap),
                    daemon=True,
                ).start()
                return

            if self.state == State.DECIDING:
                # A DECISION IS A CUE TOO. The sealed answer was taken on approach, while
                # the cue was still in frame and scoped to THIS junction, so it is the
                # better evidence -- asking again here means deciding on a frame taken
                # after the vehicle has driven past the cone. Without this the ledger
                # would only serve CHECKING_CUE steps and a branching plan would ignore it
                # entirely (junction sealed correctly, branch taken by a fresh query).
                if self._cue_ledger is not None:
                    sealed = self._cue_ledger.current_answer
                    if sealed is not None:
                        branches = list(step["branches"])
                        d = self._navigator.default_branch_idx()
                        idx = d
                        if sealed:
                            idx = next((i for i, b in enumerate(branches)
                                        if (b.get("vlm_cue") or "").strip().lower()
                                        != "default"), d)
                        idx = self._suppress_forbidden_turn(branches, idx)
                        if idx is not None:
                            self.get_logger().info(
                                f"branch from the approach sighting sealed to junction "
                                f"{self.current_cluster}: cue "
                                f"{'PRESENT' if sealed else 'absent'} -> branch {idx}")
                            self._commit_branch(branches, idx)
                            return
                if self._cue_source == "topic":
                    branches = list(step["branches"])
                    idx = self._suppress_forbidden_turn(
                        branches, self._topic_choose_branch(branches))
                    if idx is None:
                        self.get_logger().warn(
                            "no branch answer yet on "
                            f"{self.get_parameter('cue_topic').value}",
                            throttle_duration_sec=5.0)
                        return
                    self._commit_branch(branches, idx)
                    return
                # We allow branch decisions to proceed even without an image —
                # the worker logs and falls back to the default branch.
                self._vlm_busy = True
                branches   = list(step["branches"])
                image_copy = self.latest_image.copy() if self.latest_image is not None else None
                age_snap = self._snapshot_frame_age()
                threading.Thread(
                    target=self._run_vlm_decide,
                    args=(branches, image_copy, age_snap),
                    daemon=True,
                ).start()
                return

    # ------------------------------------------------------------------
    # VLM workers
    # ------------------------------------------------------------------

    def _cue_check_timed_out(self) -> bool:
        """Abandon a cue that never appears. Caller holds self._lock.

        Without this a step whose cue is absent (wrong environment, occluded
        landmark, VLM consistently saying NO) polls forever and the plan is
        wedged with no diagnostic. Reverting to NAVIGATING lets the cluster
        condition re-arm, so the step gets another chance if the robot moves.

        DELIBERATE: ``_progress.sightings`` is PRESERVED across a timeout; only
        ``dwell`` is cleared. A sighting is a real observation of a real landmark
        — the robot did pass that first bench — so a timed-out "2nd bench" step
        must not demand two *more* benches on re-entry. Only the dwell counter is
        about the current approach and is meaningless once we re-arm.
        """
        if self._cue_timeout_s <= 0 or self._cue_started_at is None:
            return False
        waited = time.time() - self._cue_started_at
        if waited < self._cue_timeout_s:
            return False
        step = self._step
        cue = step.get("transition_cue") if step else None
        print(
            f"\n[BRAIN] CUE TIMEOUT after {waited:.0f}s waiting for {cue!r} "
            f"({self._progress.sightings} sighting(s) seen) — reverting to NAVIGATING"
        )
        self.get_logger().warn(
            f"Cue {cue!r} not confirmed within {self._cue_timeout_s:.0f}s; "
            f"reverting to NAVIGATING (sightings={self._progress.sightings})"
        )
        self.state           = State.NAVIGATING
        self._cue_started_at = None
        # Latch it off so _on_arrived_at_goal does not re-arm on the very next
        # /predicted_cluster message. The cluster condition that opened this cue
        # check is still true (the robot has not moved), so without the latch we
        # oscillate NAVIGATING<->CHECKING_CUE forever and keep paying for VLM
        # calls. The latch clears when the robot actually leaves the accept set.
        self._progress.abandon_cue()
        self._publish_state()
        return True

    def _cue_answer_cb(self, msg: String) -> None:
        """Latest ground-truth / replayed cue answers: {"<cue text>": true, ...}.

        Keyed by cue text rather than a bare boolean so one publisher can answer several
        cues at once, and so an answer that arrives for a DIFFERENT step is ignored rather
        than silently satisfying this one — the failure that would look like a plan
        advancing for no reason.
        """
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn(f"cue answer is not JSON: {msg.data[:80]!r}")
            return
        if isinstance(payload, dict):
            with self._lock:
                self._cue_answers = {str(k): bool(v) for k, v in payload.items()}
                # gt_cue_node ticks on its own timer, so the first answer after a change can still describe the
                # region just left: only answers >= 0.3 s after the change count as this region's (llm_brain: 0.4 s).
                if time.time() - getattr(self, "_cluster_changed_t", 0.0) < 0.3:
                    return
                self._answers_fresh = True
                if (self._cue_semantics == "v2" and self._cue_window
                        and self._cue_window[-1]["cluster"] == self.current_cluster):
                    a = self._cue_window[-1]["answers"]
                    for k, v in self._cue_answers.items():
                        a[k] = bool(a.get(k, False) or v)

    def _topic_cue_answer(self, cue: str) -> bool | None:
        """The answer for ``cue``, or None if nobody has answered it yet."""
        if cue in self._cue_answers:
            return self._cue_answers[cue]
        # Tolerate light rewording between plan and publisher — the plan says
        # "Detect(Intersection)", a ground-truth publisher may key on "intersection".
        #
        # ORDER MATTERS. A branch cue naming both an object and a place -- "traffic cone
        # is present in the intersection" -- matches BOTH `cone` and `intersection`.
        # Taking an arbitrary match can give the PLACE answer, which is true whenever the
        # vehicle is at a junction, so the branch fires unconditionally regardless of the
        # cone.
        #
        # Publisher order is therefore respected as a SPECIFICITY order (the object key comes
        # before the place key), and a disagreement among matches is logged rather than
        # silently resolved -- an ambiguous cue is a plan defect and should be visible.
        low = cue.lower().replace("_", " ").replace("-", " ")
        # SEPARATOR-NORMALISE THE CUE SIDE ONLY. Without it `Detect(stop_sign)` is
        # rejected even though `stop_sign` is answerable, because the vocabulary
        # spellings are 'stop sign' and 'stopsign' and an underscore matches neither
        # (`traffic_cone` would pass only because it contains 'cone'). Only the cue is normalised
        # -- no vocabulary key contains a separator -- so nothing that already matched
        # can change family. The same three characters must be applied in all THREE
        # implementations of this rule or the validator and the runtime drift apart.
        hits = [(k, v) for k, v in self._cue_answers.items()
                if k.lower() in low or low in k.lower()]
        if not hits:
            return None
        if len({v for _, v in hits}) > 1:
            self.get_logger().warn(
                f"cue {cue!r} matches {len(hits)} published answers that DISAGREE "
                f"({', '.join(f'{k}={v}' for k, v in hits)}); taking {hits[0][0]!r}. "
                f"A branch cue should name ONE predicate.",
                throttle_duration_sec=10.0)
        return hits[0][1]

    def _suppress_forbidden_turn(self, branches: list[dict[str, Any]],
                                 idx: int | None) -> int | None:
        """Veto a turning branch at an instance-forbidden junction; take the default.

        THE ASYMMETRY THIS CLOSES. `cue_ordinal` counts instances for ADVANCEMENT --
        "the 2nd bench" -- and without this nothing counts them for PROHIBITION, so
        "turn at the cone-marked intersection but not the second one" had no
        representation: the plan writes `forbid_modes: ['junction']` and the formula
        `G(not Phi_Junc)`, each forbidding every junction on a route the same mission says
        to traverse. That contradiction is real and the containment check finds it.

        Detection alone is not enough either. `forbid_policy` defaults to "log", so a
        prohibition would be *observed* and never *acted on*: the branch fires, the vehicle
        turns, and the violation appears in the log after the fact. Here the prohibition
        changes the decision instead, which is the only form in which a constraint can do
        work the route could not have done by itself.

        Only TURNS are vetoed. Going straight through a forbidden-to-turn-at junction is
        not a violation, and vetoing it would strand a mission that has to pass through.
        """
        if idx is None or not self._inst_forbid.forbidden:
            return idx
        cur = self._cluster
        if cur is None or int(cur) not in self._inst_forbid.forbidden:
            return idx
        if not branch_turns(branches[idx]):
            return idx
        default_idx = next((i for i, b in enumerate(branches)
                            if (b.get("vlm_cue") or "").strip().lower() == "default"), None)
        if default_idx is None or default_idx == idx:
            self.get_logger().warn(
                f"[FORBID] branch {idx} turns at forbidden instance {cur} but there is no "
                f"default to fall back to; taking it anyway and reporting the violation")
            return idx
        self.get_logger().info(
            f"[FORBID] vetoed branch {idx} ({branches[idx].get('vlm_cue', '?')!r}): it "
            f"turns at cluster {cur}, which an instance-scoped prohibition forbids. "
            f"Taking default branch {default_idx} instead.")
        return default_idx

    def _commit_branch(self, branches: list[dict[str, Any]], idx: int) -> None:
        """Descend into branch ``idx`` and resume navigating. Caller holds self._lock.

        Shares the descend + reset with the VLM path deliberately: two copies of "which
        branch did we take and what resets" would drift apart.
        """
        try:
            chosen = self._navigator.descend(idx)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"navigator.descend({idx}) failed: {exc}")
            return
        self.state = State.NAVIGATING
        self._reset_step_progress()
        self.get_logger().info(
            f"[BRANCH] took branch {idx} ({chosen.get('vlm_cue', '?')!r}); "
            f"path={self._navigator.branch_path}")
        self._publish_state()

    def _topic_choose_branch(self, branches: list[dict[str, Any]]) -> int | None:
        """Pick a branch from the topic answers. Caller holds self._lock.

        A branch wins when its own ``vlm_cue`` is answered TRUE. If every
        non-default branch is answered false, the `default` branch wins — which is what
        `default` means, and why the schema insists one exists. None means nobody has
        answered yet, so the caller waits rather than guessing.
        """
        default_idx = next((i for i, b in enumerate(branches)
                            if (b.get("vlm_cue") or "").strip().lower() == "default"), 0)
        answered = 0
        for i, b in enumerate(branches):
            cue = (b.get("vlm_cue") or "").strip()
            if not cue or cue.lower() == "default":
                continue
            ans = self._topic_cue_answer(cue)
            if ans is None:
                continue
            answered += 1
            if ans:
                return i
        return default_idx if answered else None

    def _dump_frame(self, image, tag: str,
                    age: tuple[float | None, float | None] | None = None) -> str:
        """Save the EXACT image sent to the VLM, when VLM_FRAME_DUMP names a directory.

        The one thing that cannot be inferred from a log: a wrong answer may be model
        variance on a good image or an image that does not show what the geometry says it
        should, and those are indistinguishable without the pixels.

        Off unless the env var is set, so ordinary runs are unchanged and do not write
        hundreds of PNGs. Named with the tag and the frame age so a dumped
        image can be matched to the [VLM] line that used it.
        """
        d = os.getenv("VLM_FRAME_DUMP", "").strip()
        if not d or image is None:
            return ""
        try:
            os.makedirs(d, exist_ok=True)
            secs, mets = age if age else self._frame_age()
            name = (f"{tag}_{time.strftime('%H%M%S')}"
                    f"_{(secs or 0):.2f}s_{(mets or 0):.1f}m.png")
            fp = os.path.join(d, name)
            cv2.imwrite(fp, image)
            return fp
        except Exception as exc:                                   # noqa: BLE001
            self.get_logger().warn(f"frame dump failed: {exc}")
            return ""

    def _snapshot_frame_age(self) -> tuple[float | None, float | None]:
        """Age of the frame being copied RIGHT NOW. Caller holds self._lock.

        THIS MUST BE TAKEN AT COPY TIME. Reading `_image_rx_t` inside the VLM worker,
        i.e. AFTER the OpenAI round-trip, is wrong: newer frames have overwritten it by
        then, so the number describes a different (newer) image than the one that was
        sent, and under-reports the age of the frame actually used.
        """
        if self._image_rx_t is None:
            return None, None
        secs = time.time() - self._image_rx_t
        mets = (self._travelled - self._image_rx_travelled
                if self._image_rx_travelled is not None else None)
        return secs, mets

    @staticmethod
    def _age_str(age: tuple[float | None, float | None] | None) -> str:
        if not age or age[0] is None:
            return "frame age unknown"
        s, m = age
        return f"frame {s:.2f}s / {m:.1f}m old" if m is not None else f"frame {s:.2f}s old"

    def _label_of(self, cluster: int | None) -> str:
        return self._cluster_labels.get(int(cluster), "") if cluster is not None else ""

    def _windowed_bearing(self) -> float | None:
        """Heading accumulated inside the manoeuvre's window, or None if there is none.

        None means "no decision junction is known", which happens for a plain topology
        step that never went through DECIDING -- those keep the original unwindowed
        comparison so nothing that used to work changes shape.
        """
        if self._decide_cluster is None:
            return None
        return self._progress.bearing_accum

    def _travelled_this_step(self) -> float | None:
        """Metres since the current step began, or None if the odometer has no datum."""
        et = getattr(self._progress, "entry_travelled", None)
        return None if et is None else (self._travelled - et)

    def _frame_age(self) -> tuple[float | None, float | None]:
        """(seconds, metres) since the newest frame arrived. Caller need not hold the lock.

        METRES IS THE NUMBER THAT MATTERS. What degrades a cue answer is how much further
        away the robot was when the shutter opened, not how long ago in wall time -- a
        stationary robot can hold a 10 s old frame and lose nothing. Reported together so
        a slow pipeline (seconds high, metres low) is distinguishable from a fast one on a
        moving vehicle (both high).
        """
        if self._image_rx_t is None:
            return None, None
        secs = time.time() - self._image_rx_t
        mets = (self._travelled - self._image_rx_travelled
                if self._image_rx_travelled is not None else None)
        return secs, mets

    def _frame_age_str(self) -> str:
        s, m = self._frame_age()
        if s is None:
            return "frame age unknown"
        return f"frame {s:.2f}s / {m:.1f}m old" if m is not None else f"frame {s:.2f}s old"

    def _v2_count(self, cue: str, goal) -> int:
        """Regions in the v2 window whose merged ground-truth answers satisfy `cue`; a junction-goal step
        ("the 2nd junction with a stop sign") counts junction regions only. Caller holds self._lock."""
        keep, n = self._cue_answers, 0
        try:
            for e in self._cue_window:
                if goal == "junction" and e["label"] != "junction":
                    continue
                self._cue_answers = e["answers"]
                if e["answers"] and self._topic_cue_answer(cue):
                    n += 1
        finally:
            self._cue_answers = keep
        return n

    def _register_cue_observation(self, cue_found: bool) -> bool:
        """Fold one VLM poll into the ordinal counter. Caller holds self._lock.

        Returns True when the required Nth *distinct* sighting has been reached.

        A sighting is distinct only after the cue has gone out of view for
        ``cue_lost_polls`` consecutive polls. Without that de-bounce, "stop at the
        2nd bench" would be satisfied by two consecutive polls of the FIRST bench,
        since the cue stays in frame for many seconds as the robot approaches.
        """
        step = self._step
        needed = int((step.get("cue_ordinal") if step else None) or 1)
        if self._cue_semantics == "v2" and self._cue_source == "topic" and step and step.get("transition_cue"):
            # DERIVED, not event-driven: the count IS the number of window regions (since the junction where the
            # last manoeuvre was executed) whose merged answers show the cue. Recomputed every poll, so an answer
            # that lands after the step changed is still counted, and nothing counts twice.
            n = self._v2_count(step["transition_cue"], step.get("goal_mode"))
            if n != self._progress.sightings:
                print(f"[VLM] cue sighting {n}/{needed} ({'satisfies' if n >= needed else 'not yet'})")
            self._progress.sightings = n
            return n >= needed
        if self._cue_semantics == "v2":
            if not self._answers_fresh or self.current_cluster in self._v2_counted:
                cue_found = False      # stale answer, or this cluster already counted (also acts as "out of view")
        if self._cue_semantics == "v2" and step and step.get("goal_mode") == "junction":
            # SCOPED, for junction-goal steps only: "the 2nd junction with a stop sign" -- a sign on the road
            # in between is not one. A road-goal step ("after the 2nd light, ...") counts the cue wherever it
            # is seen: scoping it to roads would refuse every traffic light (they live in junctions).
            if self._label_of(self.current_cluster) != "junction":
                cue_found = False
        before = self._progress.sightings
        # `travelled=None` keeps the pure temporal de-bounce (the default behaviour);
        # passing the odometer additionally separates two
        # sightings by distance. See cue_spatial_debounce.
        satisfied = self._progress.note_cue(
            cue_found, needed,
            travelled=self._travelled if self._cue_spatial_debounce else None)
        if self._cue_semantics == "v2" and self._progress.sightings != before and self.current_cluster is not None:
            self._v2_counted.add(self.current_cluster)
        if self._progress.sightings != before:
            print(
                f"[VLM] cue sighting {self._progress.sightings}/{needed} "
                f"({'satisfies' if satisfied else 'not yet'})"
            )
        return satisfied

    def _run_scoped_approach(self, question: str, image, travelled: float) -> None:
        """Ask the scoped question about the NEXT intersection and latch the answer.

        Answers YES/NO/unknown. An exception latches NOTHING rather than a False: a failed
        call is not evidence of absence, and recording it as one would make the junction
        read "no cue" on a question that was never answered.
        """
        answer: bool | None = None
        try:
            if self._openai is not None:
                _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
                b64 = base64.b64encode(buf).decode("utf-8")
                prompt = ("You are assisting a mobile robot navigating a road. "
                          "Answer ONLY with YES or NO. " + question)
                resp = self._openai.chat.completions.create(
                    model=self.vlm_model, max_tokens=5,
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}", "detail": "low"}}]}])
                raw = (resp.choices[0].message.content or "").strip().upper()
                answer = True if "YES" in raw else (False if "NO" in raw else None)
                print(f"[VLM] approach: {question[:48]}... -> {raw!r} "
                      f"[{self._age_str(self._snapshot_frame_age())}]")
        except Exception as exc:                                   # noqa: BLE001
            self.get_logger().warn(f"scoped approach query failed: {exc}")
        with self._lock:
            self._vlm_busy = False
            if self._cue_ledger is not None:
                self._cue_ledger.note_approach(answer, travelled)

    def _run_vlm_cue(self, cue: str, image: np.ndarray,
                     age: tuple[float | None, float | None] | None = None) -> None:
        """Single-shot YES/NO check that the cue is visible."""
        try:
            _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            img_b64 = base64.b64encode(buf).decode("utf-8")

            # Normalise the plan's token into a noun phrase before asking. Without this
            # the model is asked `Is 'Detect(BusStandLeft)' visible in this image?` and
            # has to parse our notation first -- and the spatial half, which is what the
            # mission turns on, stays buried in CamelCase where it is easiest to ignore.
            phrase = cue_question(cue)
            prompt = (
                "You are assisting a mobile robot navigating its environment. "
                "Directions like left and right are from the ROBOT's point of view, "
                "looking forward. Answer ONLY with YES or NO. "
                f"Question: Can you see {phrase} in this image?"
            )

            if self._openai is None:
                # No key. A cue can never be confirmed, so say NO rather than pretend:
                # the cue timeout then does its job and the step is not silently skipped.
                self.get_logger().warn(
                    f"no OpenAI client — cannot check cue {cue!r}; answering NO",
                    throttle_duration_sec=30.0)
                return False

            resp = self._openai.chat.completions.create(
                model=self.vlm_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {
                            "url":    f"data:image/jpeg;base64,{img_b64}",
                            "detail": "low",
                        }},
                    ],
                }],
                max_tokens=5,
            )

            answer    = (resp.choices[0].message.content or "").strip().upper()
            cue_found = "YES" in answer

            dumped = self._dump_frame(image, f"cue_{'yes' if cue_found else 'no'}", age)
            print(f"[VLM] Cue query: {cue!r} | Response: {answer} "
                  f"[{self._age_str(age)}]" + (f" -> {dumped}" if dumped else ""))
            self.get_logger().info(f"VLM cue check for {cue!r}: {answer}")

            with self._lock:
                self._vlm_busy = False
                if self.state != State.CHECKING_CUE:
                    return
                # Fold into the ordinal counter — a single sighting is enough for
                # the common cue_ordinal=1 case, but "the 2nd bench" needs two
                # distinct ones.
                if self._register_cue_observation(cue_found):
                    print(f"\n*** [CUE DETECTED] {cue!r} confirmed by VLM! ***\n")
                    self.get_logger().info(f"Perception cue confirmed: {cue!r}")
                    self._after_step_satisfied()

        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"VLM cue check raised: {exc}")
            with self._lock:
                self._vlm_busy = False

    def _run_vlm_decide(
        self,
        branches: list[dict[str, Any]],
        image: np.ndarray | None,
        age: tuple[float | None, float | None] | None = None,
    ) -> None:
        """Multi-choice VLM call over the decision step's branches.

        Retries up to ``vlm_decide_max_attempts`` times with exponential
        backoff; on persistent failure, falls back to the 'default' branch.
        """
        chosen_idx: int | None = None
        last_exc: BaseException | None = None

        if image is None:
            self.get_logger().warn(
                "No image available for VLM branch decision — falling back to default"
            )
        else:
            for attempt in range(self._vlm_decide_max_attempts):
                try:
                    chosen_idx = self._suppress_forbidden_turn(
                        branches, self._vlm_choose_branch(branches, image, age))
                    break
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    self.get_logger().warn(
                        f"choose_branch attempt {attempt + 1}/"
                        f"{self._vlm_decide_max_attempts} failed: {exc!r}"
                    )
                    if attempt + 1 < self._vlm_decide_max_attempts:
                        time.sleep(self._vlm_decide_backoff_s * (attempt + 1))

        with self._lock:
            self._vlm_busy = False
            if self.state != State.DECIDING or self._navigator is None:
                # Plan was swapped or completed mid-flight; abort silently.
                return

            if chosen_idx is None:
                chosen_idx = self._navigator.default_branch_idx()
                if last_exc is not None:
                    self.get_logger().error(
                        f"choose_branch exhausted {self._vlm_decide_max_attempts} "
                        f"attempts; falling back to default (idx {chosen_idx}). "
                        f"Last error: {last_exc!r}"
                    )

            try:
                chosen = self._navigator.descend(chosen_idx)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().error(f"navigator.descend({chosen_idx}) failed: {exc}")
                return

            self.state = State.NAVIGATING
            # Descending into a branch starts a new step: reset dwell/sighting/yaw.
            self._reset_step_progress()
            cue = chosen.get("vlm_cue", "<unknown>")
            print(
                f"\n[BRANCH] Took branch {chosen_idx} ({cue!r}); "
                f"path={self._navigator.branch_path}"
            )
            nxt = self._step
            if nxt is not None:
                print(
                    f"[BRANCH] Now navigating: -> {self._lbl(nxt['goal_cluster'])}"
                )
                if nxt.get("transition_cue"):
                    print(f"[BRANCH] Next cue: {nxt['transition_cue']!r}")
            self.get_logger().info(
                f"Branch chosen: idx={chosen_idx} cue={cue!r} "
                f"path={self._navigator.branch_path}"
            )
            self._publish_state()

    def _vlm_choose_branch(
        self,
        branches: list[dict[str, Any]],
        image: np.ndarray,
        age: tuple[float | None, float | None] | None = None,
    ) -> int:
        """One round-trip to the OpenAI vision model returning the branch index."""
        # Build a numbered multiple-choice prompt. Always include the 'default'
        # option so the VLM has an explicit "none of the above" answer.
        default_idx = next(
            (i for i, b in enumerate(branches)
             if (b.get("vlm_cue") or "").strip().lower() == "default"),
            0,
        )
        lines = [
            "You are assisting a mobile robot that has arrived at a decision",
            "point. Look at the image and pick the option whose description",
            "best matches what you see right now.",
            f"If none of the descriptive options clearly match, choose option {default_idx + 1}.",
            "Answer with ONLY a single integer (the option number), with no other text.",
            "",
            "Options:",
        ]
        for i, b in enumerate(branches):
            cue = b.get("vlm_cue") or ""
            label = "default (none of the above clearly match)" if i == default_idx else cue
            lines.append(f"  {i + 1}) {label}")
        prompt = "\n".join(lines)

        if self._openai is None:
            # No key: take the `default` branch, which is exactly the documented
            # fallback and the reason the schema requires one.
            self.get_logger().warn(
                "no OpenAI client — taking the 'default' branch without looking")
            return default_idx

        _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_b64 = base64.b64encode(buf).decode("utf-8")

        resp = self._openai.chat.completions.create(
            model=self.vlm_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url":    f"data:image/jpeg;base64,{img_b64}",
                        "detail": "low",
                    }},
                ],
            }],
            max_tokens=8,
        )

        answer = (resp.choices[0].message.content or "").strip()
        dumped = self._dump_frame(image, f"branch_ans{answer.strip() or 'x'}", age)
        print(f"[VLM] Branch query (default={default_idx + 1}): {answer!r} "
              f"[{self._age_str(age)}]" + (f" -> {dumped}" if dumped else ""))

        digits = "".join(c for c in answer if c.isdigit())
        if not digits:
            raise ValueError(f"unparseable VLM answer {answer!r}")
        idx_1based = int(digits[:2]) if len(digits) > 1 else int(digits)
        if not (1 <= idx_1based <= len(branches)):
            raise ValueError(
                f"VLM answer {idx_1based} out of range (have {len(branches)} branches)"
            )
        return idx_1based - 1

    # ------------------------------------------------------------------
    # Completion / state
    # ------------------------------------------------------------------

    def _enter_complete(self) -> None:
        """Mark the plan complete. Caller holds self._lock."""
        self.state = State.COMPLETE
        path = self._navigator.branch_path if self._navigator else []
        print(f"\n{'=' * 60}")
        print(f"[BRAIN] PLAN COMPLETE — branch path: {path}")
        print(f"{'=' * 60}\n")
        self.get_logger().info(f"Navigation plan complete (branch_path={path})")
        self._publish_state()

    def _publish_state(self) -> None:
        nav = self._navigator
        payload: dict[str, Any] = {
            "state":           self.state.value,
            "current_cluster": self.current_cluster,
            "current_label":   self._cluster_labels.get(self.current_cluster, "unknown")
                                 if self.current_cluster is not None else "unknown",
            "plan_name":       self._plan_name,
        }
        if self._constraint.declared:
            payload.update(self._constraint.summary())
        if self._require is not None:
            payload["require_modes"] = self._plan_require_modes
            payload["require_breached"] = self._require.breached
        if nav is not None:
            payload["step"]        = nav.step_idx
            payload["branch_path"] = nav.branch_path
            payload["sub_plan_len"] = nav.n_steps_in_sub_plan
            payload["complete"]    = nav.is_complete
            step = nav.current_step
            if step is not None:
                payload.update({
                    "start_cluster":  step["start_cluster"],
                    "start_label":    self._cluster_labels.get(step["start_cluster"], "unknown"),
                    "goal_cluster":   step["goal_cluster"],
                    "goal_label":     self._cluster_labels.get(step["goal_cluster"], "unknown"),
                    "transition_cue": step.get("transition_cue"),
                    "has_branches":   bool(step.get("branches")),
                    "n_branches":     len(step["branches"]) if step.get("branches") else 0,
                    # Trigger typology + acceptance sets, so a consumer can tell
                    # WHY a step advanced (cluster vs cue vs heading) and whether
                    # it advanced on a degraded cluster match.
                    "trigger":           _trigger_of(step),
                    "accept_clusters":   sorted(_accept_set(step, degraded=False)),
                    "accept_degraded":   sorted(_accept_set(step, degraded=True)),
                    "perception_backed": bool(step.get("perception_backed", True)),
                    "on_degraded_cluster": (
                        self.current_cluster is not None
                        and self.current_cluster in _accept_set(step, degraded=True)
                        and self.current_cluster not in _accept_set(step, degraded=False)
                    ),
                    "cue_ordinal":     int(step.get("cue_ordinal") or 1),
                    "cue_sightings":   self._progress.sightings,
                    "dwell":           self._progress.dwell,
                    "heading_delta_deg": (
                        round(float(np.degrees(_wrap_pi(self._yaw - self._progress.entry_yaw))), 1)
                        if self._yaw is not None and self._progress.entry_yaw is not None
                        else None
                    ),
                })
        msg = String()
        msg.data = json.dumps(payload)
        self._state_pub.publish(msg)

    def _print_idle_banner(self) -> None:
        sep = "=" * 60
        print(f"\n{sep}")
        print("[BRAIN] WAITING_FOR_PLAN (tree-aware controller v2)")
        print(f"[BRAIN] VLM: {self.vlm_model} | "
              f"check interval: {self.get_parameter('vlm_check_interval').value}s | "
              f"branch retries: {self._vlm_decide_max_attempts}")
        print("[BRAIN] No plan loaded. Listening on /brain/incoming_plan and")
        print("[BRAIN] waiting for the /brain/load_plan Trigger to swap one in.")
        print(f"[BRAIN] Cluster source: /predicted_cluster (Int16)")
        print(f"[BRAIN] Camera: {self.image_topic}")
        print(f"{sep}\n")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BrainController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
