"""Sampling-MPC controller driven by the BRAIN plan, over pure ROS 2.

    /brain/incoming_plan  (String)   -> region centroids + bearings (the geometry)
    /brain/state          (String)   -> which step, hence start/goal region
    /carla/<role>/odometry(Odometry) -> ego pose
    /processed_ranges     (F32Array) -> optional LiDAR EDT
                                     -> /carla/<role>/twist  (Twist)

Why this file exists next to ``carla_sampling_mpc_ros_node.py``
--------------------------------------------------------------
The two halves of this stack lived in different files and had never been connected.
``dgppo_ros_node.py`` had the ``/brain/*`` intake but only teleported the vehicle with
physics off; ``carla_sampling_mpc_ros_node.py`` drove with physics but read a static
``plans/*.json`` and ran its OWN plan-advance. This is the merge — brain's plan and the
MPC's driving — as a new file, so the old node keeps working untouched and nothing that
already ran is at risk.

Three deliberate differences from the old node:

**Brain owns plan advancement.** The old node advanced when
``mapped_cluster == expected_next``, i.e. strict equality on a single id. That directly
contradicts the acceptance-set logic brain implements (a `junction` also satisfies a
`path` step; a landmark step may advance on a degraded cluster match while a
``Detect(...)`` predicate carries the real evidence). Two components deciding
independently when a step is done is a race with no correct outcome, so this node has no
plan-advance at all: it reads ``start_cluster`` / ``goal_cluster`` off ``/brain/state``
and steers. It also drops the old ``_map_cluster_id`` remap table, which existed to
squeeze arbitrary classifier ids into four DGPPO slots — the ground-truth region ids
already ARE the plan's ids.

**No ``carla`` import.** Pose comes from the bridge's odometry topic and control goes out
as ``Twist`` for ``carla_twist_to_control``, which was already in the launch recipe with
nothing publishing to it. The unicycle's ``(v, omega)`` maps onto ``linear.x`` /
``angular.z`` exactly. Trade-off: commands now go through throttle and steering rather
than overriding velocity, so tracking is looser — which is more physically honest for a
car, and it is the reason ``max_curvature`` exists in :class:`MpcConfig`.

**No initial teleport.** The old node teleported to the first centroid with physics off.
That is how a run "completes" without driving, and it is why the metrics harness needs a
path-length guard. Spawn the ego in the corridor via the bridge's spawn config instead;
this node reports an error if the first pose is not in the plan's regions.

Frames: everything here is **planar** — metres, +y north, yaw counter-clockwise radians —
and nothing is converted, because nothing arrives in CARLA's frame. The ros-bridge
publishes odometry and consumes twist in the ROS right-handed convention, and the plan's
centroids and bearings come from OpenDRIVE geometry, which is that same frame. See
``carla_gt_bridge.frames`` for the table of which data is in which frame — an
unnecessary conversion here mirrors the entire map.
"""

from __future__ import annotations

import datetime
import json
import math
import os
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, QoSProfile, ReliabilityPolicy,
                       qos_profile_sensor_data)
from std_msgs.msg import Float32MultiArray, Int16, String

from carla_gt_bridge.frames import wrap_pi
from carla_gt_bridge.routing import StepTargeter, adjacency_from_bearing_map

from .sampling_mpc import (LocalOccGrid, MpcConfig, RoadSurface, aim_point,
                           plan_step)
from .terrain_mpc import (TerrainMpcConfig, arc_rollout_k, arc_weights,
                          curvature_fan, plan_step_terrain,
                          waypoint_terrain_source)
from carla_gt_bridge.camera_terrain import CameraModel, MaskTerrainSource
from carla_gt_bridge.nodes.terrain_mask_node import parse_frame_id
from carla_gt_bridge.terrain_classes import DEFAULT_COSTS, UNOBSERVED, classes_from_names
from carla_gt_bridge.terrain_overlay import render_fan


def _yaw_from_quaternion(q) -> float:
    """Yaw (radians) from a geometry_msgs Quaternion."""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class CarlaMpcNode(Node):
    """Steers toward the region brain says is next. Never decides when to advance."""

    def __init__(self) -> None:
        super().__init__("carla_mpc_node")
        self._declare_params()

        cfg = MpcConfig(
            K=self._int("mpc_K"), N=self._int("mpc_N"), dt=self._float("mpc_dt"),
            # Graded obstacle penalty alongside the -inf veto, ported from terrain_mpc.
            # 0.0 = veto only. The veto alone gives no gradient toward turning away
            # from an obstacle dead ahead; the graded term supplies one.
            w_obstacle=self._float("w_obstacle"),
            obstacle_influence_m=self._float("obstacle_influence_m"),
            v_max=self._float("v_max"), omega_max=self._float("omega_max"),
            max_curvature=self._float("max_curvature"),
            safety_radius=self._float("safety_radius"),
            d_safe=self._float("d_safe"))
        self._cfg = cfg

        self._occ = LocalOccGrid()
        cfg.road_half_width = self._float("road_half_width")
        cfg.reject_offroad = self._bool("reject_offroad")
        cfg.guidance = self._str("guidance")
        cfg.aim_source = self._str("aim_source") or "region"
        cfg.pure_pursuit = self._bool("pure_pursuit")
        cfg.collision_horizon = self._int("collision_horizon")
        cfg.debug_rollouts = self._int("debug_rollouts")
        self._road = self._load_road()

        # WHICH CONTROLLER IS DRIVING, logged at startup and validated. A mode that
        # silently falls back to another controller is indistinguishable from a working
        # one in the logs, so an unknown value raises instead.
        self._controller = self._str("controller")
        if self._controller not in ("region", "terrain"):
            raise ValueError(f"controller must be 'region' or 'terrain', "
                             f"got {self._controller!r}")
        self._terrain_mode = self._controller == "terrain"
        self._kappa_prev = 0.0
        self._terrain_source = self._str("terrain_source")
        if self._terrain_source not in ("waypoint", "camera"):
            raise ValueError(f"terrain_source must be 'waypoint' or 'camera', "
                             f"got {self._terrain_source!r}")
        self._forbid_terrain = classes_from_names(
            [t for t in self._str("forbid_terrain").split(",") if t.strip()])
        self._mask_src = None
        self._mask_seen = 0
        self._bearing_ref = float('nan')
        self._steer_target = -1
        self._overlay_every = self._int("overlay_every_n")
        # Dump snapshots instead of rendering inline. False restores the in-process
        # render, kept only for comparison.
        self._overlay_dump = True
        self._overlay_busy = threading.Event()
        self._overlay_skipped = 0
        self._overlay_dir = ""          # set once _log_path exists; see _open_log below
        if self._terrain_mode and self._terrain_source == "camera":
            cam = CameraModel(width=self._int("cam_width"), height=self._int("cam_height"),
                              fov_deg=self._float("cam_fov"), mount_x=self._float("cam_x"),
                              mount_z=self._float("cam_z"),
                              pitch_down_deg=self._float("cam_pitch_down_deg"))
            self._mask_src = MaskTerrainSource(
                cam, max_frame_age_s=self._float("max_mask_age_s"))
            self.get_logger().info(
                f"terrain_source=camera: fov={cam.fov_deg} pitch_down={cam.pitch_down_deg} "
                f"x={cam.mount_x} z={cam.mount_z} min_range={cam.effective_min_range_m():.2f}m "
                f"forbid={sorted(self._forbid_terrain)} "
                # LOGGED WITH THE VALUE IN FORCE: a run whose scoring knobs are
                # invisible cannot be compared with a run under different values.
                f"w_unknown={TerrainMpcConfig.w_unknown} "
                f"min_adjudicable={TerrainMpcConfig.min_adjudicable_frac} "
                f"decay={self._float('terrain_decay_m') or None}")
        if self._terrain_mode and self._road is None:
            raise ValueError(
                "controller='terrain' needs the waypoint table (regions_npz) -- it is "
                "the ground-truth stand-in for the camera terrain source. Without it "
                "every rollout would classify as unknown and the terrain term would be "
                "a constant, i.e. the controller would silently be bearing-only.")
        self.get_logger().info(
            f"controller={self._controller}"
            + (f" fan={self._int('fan_size')} segments={self._int('fan_segments')} "
               f"kappa_max={self._float('kappa_max')} "
               f"arc_len={self._float('arc_len_m')}m "
               f"w_obstacle={TerrainMpcConfig.w_obstacle} (from TerrainMpcConfig; "
               f"the w_obstacle parameter is not read in terrain mode)"
               if self._terrain_mode
               else f" guidance={self._str('guidance')} aim={self._str('aim_source')}"
                    f" w_obstacle={self._float('w_obstacle')}"))
        # Guard against a silent no-op flag: `guidance` only takes effect when a road
        # surface with real region ids exists; without one, `plan_step` falls back to
        # centroid aiming and logs nothing. Say so.
        if cfg.guidance == "region":
            n_rid = (0 if self._road is None
                     else len(set(self._road._rid.tolist())))     # noqa: SLF001
            if n_rid < 2:
                self.get_logger().error(
                    f"guidance='region' but the road surface carries {n_rid} distinct "
                    f"region id(s) — pure-pursuit aiming is INACTIVE and the MPC will "
                    f"steer at centroids. Check regions_npz has 3 columns.")
            else:
                self.get_logger().info(
                    f"guidance='region': pure-pursuit aiming over {n_rid} regions, "
                    f"lookahead {cfg.lookahead_m:.1f} m, bearing weight "
                    f"x{cfg.bearing_weight_region}")
        # seed=0 (the default) is a real seed, not "fall through to OS entropy": the
        # sampler draws K=500 rollouts per tick and some junctions are knife-edge (a few
        # metres of entry offset decide whether the vehicle crosses or pins against a
        # wall), so unseeded runs are not reproducible.
        #
        # Negative means "use OS entropy" and must be asked for explicitly. Repeated runs
        # should set seed=<repeat index> so repeats differ from each other but each one
        # can be replayed.
        _seed = self._int("seed")
        self._rng = np.random.default_rng(None if _seed < 0 else _seed)
        self.get_logger().info(
            f"MPC sampler seed = {'OS entropy (NOT reproducible)' if _seed < 0 else _seed}")

        # plan geometry, from /brain/incoming_plan
        self._region_ids: list[int] = []
        self._centroids = np.empty((0, 2))       # planar metres
        self._bearings: dict[str, float] = {}    # planar radians
        self._labels: dict[int, str] = {}
        self._plan_name = ""

        # plan progress, from /brain/state
        self._start_id: int | None = None
        self._goal_id: int | None = None
        self._goal_label: str | None = None
        self._maneuver = "straight"
        self._branch_path = None
        self._brain_state = ""
        #: (region, previous) captured when brain ENTERS DECIDING, so a turn chosen there
        #: is grounded at that junction even though the vehicle has driven on by the time
        #: the branch commits. See StepTargeter.target_for(from_region=...).
        self._decide_at: tuple[int | None, int | None] | None = None
        #: THE ANCHOR IS GOOD FOR EXACTLY ONE STEP. Overshoot is a property of the
        #: decision itself -- the VLM round-trip happens once, so only the step
        #: IMMEDIATELY after the branch can have its junction behind the vehicle. If
        #: `_decide_at` were kept until plan end, every later turn in a branch sub-plan
        #: would be grounded at the DECISION junction too, resolving to a neighbour of
        #: the wrong junction (and `target_for` would cache that wrong target). This only
        #: shows up in branch sub-plans with more than one turn after the decision.
        self._step_idx: int | None = None
        self._complete = False
        self._targeter: StepTargeter | None = None

        self._pose: tuple[float, float, float] | None = None   # planar x, y, yaw
        self._ranges: np.ndarray | None = None
        self._cluster: int | None = None
        self._start_pose_checked = False
        # Starvation counters. The single-threaded executor let the control
        # loop block its own odometry callback; these make that visible in
        # the log instead of inferable from position history.
        self._n_odom = 0
        self._speed = 0.0
        self._last_note = ""
        self._n_ticks = 0

        role = self._str("role_name")
        self._twist_pub = self.create_publisher(Twist, f"/carla/{role}/twist", 10)
        self._guidance_pub = self.create_publisher(
            Float32MultiArray, "/sampling_mpc_guidance_debug", 10)
        self._rollout_pub = self.create_publisher(
            Float32MultiArray, "/sampling_mpc_best_rollout", 10)

        # LATCHED, matching brain's own subscription. The plan is published once by a
        # loader node that stays alive holding the latch; a plain volatile subscriber only receives it if
        # discovery happens to have completed first, otherwise this node waits for
        # /brain/incoming_plan forever while brain publishes steps. Transient-local means
        # a subscriber that joins late still gets the last sample.
        self.create_subscription(
            String, "/brain/incoming_plan", self._plan_cb,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                       reliability=ReliabilityPolicy.RELIABLE))
        # ONE mutually-exclusive group for everything, deliberately.
        #
        # Odometry publishes at ~300 Hz on the wire, and with `use_sim_time` the bridge
        # runs the world far faster than real time, so a dt=0.2 SIM-second timer would
        # fire every ~14 ms of wall clock while one iteration samples K=500 rollouts over
        # an N=8 horizon. Under a single thread a long control loop starves the odometry
        # callback; with a ReentrantCallbackGroup and a thread pool the 300 Hz odometry
        # saturates the pool and starves the control timer instead. At this publish rate
        # whichever callback may preempt the other will starve it.
        #
        # What actually fixes pose staleness is the QoS below, not concurrency: depth=1
        # BEST_EFFORT means the callback always receives the NEWEST pose. With depth=10
        # RELIABLE each serviced callback works through a backlog of poses that are
        # already obsolete, so the controller effectively steers open-loop on a stale
        # pose. Rebinding `self._pose` to a new tuple is atomic under the GIL, so the
        # control loop reads either the old pose or the new one, never a torn one.
        self._sensor_cbg = MutuallyExclusiveCallbackGroup()
        self._control_cbg = self._sensor_cbg

        self.create_subscription(String, "/brain/state", self._state_cb, 10,
                                 callback_group=self._sensor_cbg)
        # depth 1: at ~300 Hz the only pose worth having is the latest one. A depth-10
        # queue just hands the controller a backlog of stale poses to work through.
        self.create_subscription(Odometry, self._str("odom_topic"), self._odom_cb,
                                 QoSProfile(depth=1,
                                            reliability=ReliabilityPolicy.BEST_EFFORT),
                                 callback_group=self._sensor_cbg)
        self.create_subscription(Int16, "/predicted_cluster", self._cluster_cb, 10,
                                 callback_group=self._sensor_cbg)
        self.create_subscription(Float32MultiArray, "/processed_ranges",
                                 self._ranges_cb, qos_profile=qos_profile_sensor_data,
                                 callback_group=self._sensor_cbg)
        if self._terrain_mode and self._terrain_source != "waypoint":
            # Depth 1, BEST_EFFORT, same as odometry: a queue would hand the controller a
            # backlog of stale masks to work through, which is the one thing a mask must not
            # be. The callback does nothing but decode and rebind.
            self.create_subscription(Image, self._str("mask_topic"), self._mask_cb,
                                     QoSProfile(depth=1,
                                                reliability=ReliabilityPolicy.BEST_EFFORT),
                                     callback_group=self._sensor_cbg)
            self.get_logger().info(f"terrain masks from {self._str('mask_topic')}")

        self._log_path = self._open_log()
        # ABSOLUTE, and derived from the log path rather than from a relative default -- and
        # set HERE because `_log_path` does not exist earlier in __init__. A relative LOG_DIR
        # silently writes logs into a shadow tree relative to the working directory while
        # saving still reports success.
        self._overlay_dir = os.path.join(
            os.path.dirname(os.path.abspath(self._log_path or ".")), "terrain_overlays")
        if self._overlay_every > 0:
            os.makedirs(self._overlay_dir, exist_ok=True)
            self.get_logger().info(
                f"terrain overlays every {self._overlay_every} ticks -> {self._overlay_dir}")
        #: one control step per N odometry messages == per N world ticks. See _odom_cb
        #: for why this is a count and not a clock.
        self._control_every_n = max(1, self._int("control_every_n_odom"))
        # NO control timer — see _odom_cb. A timer here cannot win against a ~300 Hz
        # sensor topic on one thread, and giving it its own thread just moves the
        # starvation to the sensor.
        self.get_logger().info(
            f"carla_mpc_node ready: K={cfg.K} N={cfg.N} dt={cfg.dt} "
            f"v_max={cfg.v_max} decide_v_max={self._float('decide_v_max')} "
            f"om_max={cfg.omega_max} kappa_max={cfg.max_curvature} "
            f"guidance={cfg.guidance} aim={cfg.aim_source} "
            f"pure_pursuit={cfg.pure_pursuit} safety_radius={cfg.safety_radius} "
            f"collision_horizon={cfg.collision_horizon or cfg.N}; "
            f"twist -> /carla/{role}/twist; waiting for /brain/incoming_plan")

    # -- params ----------------------------------------------------------- #

    def _declare_params(self) -> None:
        self.declare_parameter("role_name", "ego_vehicle")
        self.declare_parameter("odom_topic", "/carla/ego_vehicle/odometry")
        self.declare_parameter("mpc_K", 500)
        self.declare_parameter("mpc_N", 8)
        # Control steps per odometry message. One odometry message is one world tick in
        # synchronous mode, so this is the only monotonic clock available — see _odom_cb.
        #
        # Default 1, NOT dt / fixed_delta_seconds (0.2 / 0.05 = 4). 4 plans at the rate
        # the horizon assumes, which is right for the PLANNER and catastrophic for the
        # CLOCK: under `synchronous_mode_wait_for_vehicle_control_command` the CARLA
        # server holds each world tick until it receives a vehicle control command, and
        # this node publishes a Twist only inside `_control_loop`. At 4, three of every
        # four world ticks have nothing to unblock them and each stalls on the server's
        # ~1 s timeout (a serviced tick costs ~14 ms, an unserviced one ~1 s), making the
        # sim far slower than real time.
        #
        # Replanning every tick is cheap (`plan_step` is a few ms at K=500) and yields
        # smoother control, since the same manoeuvre is spread over more decisions.
        #
        # Each Twist is held for control_every_n_odom world ticks (0.05 s at 1) while the
        # rollouts assume dt; at 1 the plan is simply refreshed before that matters.
        # If a config ever runs WITHOUT wait-for-control, 4 is fine again and cheaper.
        self.declare_parameter("control_every_n_odom", 1)
        # Tesla Model 3 front-to-rear axle. Only used to turn the planned angular
        # velocity into a steering angle; see _steer_angle_for.
        self.declare_parameter("wheelbase", 2.875)
        # Below this the bicycle model divides by ~0 and demands full lock. Clamping
        # here means a stationary vehicle is commanded a plausible angle instead of a
        # saturated one it would carry into its first metre of travel.
        self.declare_parameter("min_steer_speed", 1.0)
        self.declare_parameter("mpc_dt", 0.2)
        self.declare_parameter("v_max", 5.0)
        # Speed cap while brain is in DECIDING, so the vehicle does not leave the
        # decision junction before the branch commits and the turn target is grounded.
        # 1.0 m/s: slow enough that a ~1.5 s VLM call costs ~1.5 m rather than ~8 m,
        # fast enough that the loop never reports zero motion. Set it to v_max to
        # disable the clamp and reproduce the old behaviour.
        self.declare_parameter("decide_v_max", 1.0)
        self.declare_parameter("omega_max", 1.0)
        self.declare_parameter("max_curvature", 0.25)
        self.declare_parameter("safety_radius", 1.5)
        self.declare_parameter("collision_horizon", 0)
        self.declare_parameter("d_safe", 1.5)
        #: REGION CONTROLLER ONLY. The terrain branch builds TerrainMpcConfig without
        #: passing this, so it uses that class's own w_obstacle (3.0) and setting this
        #: parameter under controller="terrain" does NOTHING. Logged either way so the
        #: no-op is visible.
        self.declare_parameter("w_obstacle", 0.0)
        self.declare_parameter("obstacle_influence_m", 4.0)
        #: >=0 is a real seed and the run replays exactly; <0 asks for OS entropy.
        self.declare_parameter("seed", 0)
        # Phase A has NO sensors at all (no_rendering_mode is on), so the EDT must be
        # optional. Requiring ranges would make the ground-truth phase impossible to
        # run, which is the phase everything else is validated against.
        self.declare_parameter("require_ranges", False)
        self.declare_parameter("dry_run", False)
        self.declare_parameter("log_dir", "")
        # Region waypoints, used to keep the vehicle on the road. Empty falls back to the
        # packaged regions.town05.npz.
        self.declare_parameter("regions_npz", "")
        self.declare_parameter("road_half_width", 7.0)
        #: "region" | "centroid". Which signal the MPC steers by.
        #:
        #:   centroid -- aim at the target region's CENTROID, bearing at full weight.
        #:   region   -- aim at the nearest point of the target REGION at least
        #:               `lookahead_m` ahead (pure pursuit), membership from the waypoint
        #:               table, bearing demoted to `bearing_weight_region`.
        #:
        #: Region guidance exists to stop the vehicle cutting corners at junctions, and
        #: offline it keeps junction exits much closer to the drivable surface than
        #: centroid aiming does.
        #:
        #: DEFAULT IS "centroid", deliberately. In driven runs region guidance did not
        #: improve the outcome at a stalling junction, because that failure is not an
        #: off-road path. Worse, region guidance demotes the bearing term to x0.25, and
        #: at a junction the cluster term is often FLAT (the target region's nearest
        #: waypoint can lie far beyond the ~8 m horizon), so bearing is the only gradient
        #: left and demoting it removes the last signal. Do not default to it until the
        #: demotion is made conditional on the target actually being reachable.
        self.declare_parameter("guidance", "centroid")
        #: HARD-reject rollouts that leave the road corridor. Separate from `w_offroad`,
        #: which merely penalises it. The corridor exists only as a stand-in for obstacle
        #: sensing -- see the note above regions_npz: "the correct setting once Phase B's
        #: LiDAR provides a real EDT" is to disable it. That EDT is now real.
        #:
        #: Known failure mode: inside a junction the vehicle can crawl, and the crawl
        #: tracks the fraction of rollouts rejected as off-road, not obstacle clearance
        #: (d_min stays well above safety_radius). A junction is wider than the corridor
        #: derived from its adjacent roads' waypoints, so swinging wide -- which is what
        #: turning through a junction IS -- reads as leaving the road and gets hard-rejected.
        #: `recovery` never fired, so the MPC always found *a* rollout; just a crawling one.
        self.declare_parameter("reject_offroad", True)
        # How many CANDIDATE rollouts to log per tick for the viewer, alongside
        # the chosen one. 0 (default) logs none and costs nothing — this is a
        # 500-rollout sampler at 5 Hz and the control loop never reads them.
        # ~24 is enough to show what the sampler rejected without bloating the
        # JSONL; the sample is stratified so rejected candidates survive it.
        self.declare_parameter("debug_rollouts", 0)
        #: Similarity transform applied to the map's DERIVED geometry only (centroids and
        #: bearings), never to cluster membership. This is the map-distortion ablation:
        #: how wrong may the map be before the plan stops executing?
        #: "region" | "centroid" -- see MpcConfig.aim_source. The distortion
        #: ablation needs membership correct and the GOAL wrong, which is
        #: guidance="region" with aim_source="centroid".
        self.declare_parameter("aim_source", "region")
        #: Baseline: steer at the goal point with no region-occupancy term.
        self.declare_parameter("pure_pursuit", False)
        #: Which controller drives. "region" is the default region-occupancy sampler;
        #: "terrain" is the curvature-fan controller that never asks which region a
        #: FUTURE position belongs to. The terrain controller is additive and opt-in.
        self.declare_parameter("controller", "region")
        #: Where the terrain labels come from. "waypoint" is the CARLA ground-truth stand-in
        #: and is the DEFAULT; "camera" consumes `/terrain/class_mask` from `terrain_mask_node`, which is the
        #: only one of the two a robot can have.
        self.declare_parameter("terrain_source", "waypoint")
        self.declare_parameter("mask_topic", "/terrain/class_mask")
        #: Write a fan overlay every N control ticks. 0 = off, which is the default: a run
        #: that renders 65 arcs x 200 samples x 2 panels every tick would spend more time
        #: drawing than steering. At 25 it is one picture every ~5 s of sim.
        self.declare_parameter("overlay_every_n", 0)
        #: Comma-separated class NAMES the mission forbids outright, e.g. "grass" or
        #: "grass,sidewalk". Resolved through `terrain_classes`, which refuses an unknown
        #: name rather than dropping it -- a prohibition that silently loses a term completes
        #: the run and prohibits nothing.
        self.declare_parameter("forbid_terrain", "")
        #: Distance weighting on the graded terrain cost. <= 0 keeps the unweighted mean
        #: (the default).
        self.declare_parameter("terrain_decay_m", 0.0)
        #: Seconds after which a mask is refused. At 5 m/s a 1 s old frame was taken 5 m back.
        self.declare_parameter("max_mask_age_s", 1.0)
        #: Camera geometry -- `camera_terrain.TERRAIN_CAMERA`.
        #: These MUST match what `terrain_mask_node` spawns; preflight checks that they do.
        self.declare_parameter("cam_x", 2.425)
        self.declare_parameter("cam_z", 1.573)
        self.declare_parameter("cam_fov", 130.0)
        self.declare_parameter("cam_pitch_down_deg", 30.0)
        self.declare_parameter("cam_width", 800)
        self.declare_parameter("cam_height", 600)
        self.declare_parameter("kappa_max", 0.25)
        #: 6.28 * kappa_max 0.25 = exactly 90 deg of turn per arc, which is
        #: TerrainMpcConfig.max_turn_rad. A longer arc (e.g. 10.0 m -> 143 deg) raises
        #: at config construction.
        self.declare_parameter("arc_len_m", 6.28)
        self.declare_parameter("fan_size", 65)
        #: 1 = one constant-curvature arc per candidate (default).
        #: 2 = a (kappa_1, kappa_2) grid whose arcs can swerve AND RETURN. In the 2D rig,
        #: single arcs cannot thread an offset gap that some two-segment arcs can, and a
        #: denser single-segment fan does not help -- the shape is absent at any density.
        #: Untested in CARLA.
        self.declare_parameter("fan_segments", 1)
        #: "map" (mean of all centroids -- a global reframing) or "start" (the
        #: first region, so error grows with distance travelled). See
        #: `_apply_distortion` -- the two are very different experiments.
        #: "similarity" (rotate/scale/translate the whole map -- a FRAME error) or
        #: "jitter" (displace each region independently -- a METRIC error). See
        #: `_apply_jitter` for why they test different things.
        self.declare_parameter("distortion_model", "similarity")
        #: jitter sigma, as a fraction of the median edge length.
        self.declare_parameter("distortion_jitter_frac", 0.0)
        #: Per-region displacement in METRES, which overrides the fraction when > 0. A
        #: fraction of the median edge is scale-free but hard to reason about: 0.30 works
        #: out at 11.1 m per axis, i.e. a typical displacement of 13.9 m against a 7.0 m
        #: road half-width -- the map puts the road twice as far away as the road is wide,
        #: which is not a stale map so much as a different one. 5-10 m is the regime worth
        #: measuring, because it straddles that tolerance instead of dwarfing it.
        self.declare_parameter("distortion_jitter_m", 0.0)
        #: Per-region ROTATION about that region's own centroid, sigma in degrees. This is
        #: the part translation cannot model: a displaced region keeps its internal layout,
        #: so the surface stays parallel to the true road and only shifts sideways. A
        #: rotated one does not, and a junction whose arms point the wrong way is a
        #: qualitatively different error from one that sits in the wrong place.
        self.declare_parameter("distortion_region_rot_deg", 0.0)
        #: Also displace the DRIVABLE SURFACE by the same per-region offsets. Off by
        #: default because it is a strictly harsher experiment. With it off, the waypoint
        #: table is pristine -- which means both the goal point AND cluster membership
        #: read unperturbed data, so a controller that aims at the surface is unaffected
        #: BY CONSTRUCTION rather than by robustness. That shows which quantity sat in the
        #: metric loop; it is not evidence of tolerance to a wrong map.
        self.declare_parameter("distortion_surface", False)
        self.declare_parameter("distortion_pivot", "map")
        self.declare_parameter("distortion_angle_deg", 0.0)
        self.declare_parameter("distortion_scale", 1.0)
        self.declare_parameter("distortion_tx", 0.0)
        self.declare_parameter("distortion_ty", 0.0)

    def _bool_param(self, name: str) -> bool:
        try:
            return bool(self.get_parameter(name).value)
        except Exception:
            return False

    def _apply_distortion(self) -> None:
        """Rotate / scale / translate the centroids about their own centre, in place.

        The pivot is the centroid cloud's mean rather than the origin: rotating about the
        origin would translate the whole town by hundreds of metres, which is a spawn
        failure rather than a map error. Bearings are rotated by the same angle; scale and
        translation do not change an angle between two points that both moved with them.
        """
        model = self._str("distortion_model") or "similarity"
        if model == "jitter":
            self._apply_jitter()
            return
        ang = math.radians(self._float("distortion_angle_deg"))
        sc = self._float("distortion_scale") or 1.0
        tx, ty = self._float("distortion_tx"), self._float("distortion_ty")
        if not ang and sc == 1.0 and not tx and not ty:
            return
        if len(self._centroids):
            # PIVOT CHOICE IS THE WHOLE EXPERIMENT, not a detail. Rotating the town about
            # its own mean is a GLOBAL REFRAMING: on Town05 the median centroid sits
            # ~144 m from that mean, so 30 deg moves the average goal ~75 m on a map
            # whose regions are 20-40 m apart. That is not a stale map, it is a different
            # map, and it is not comparable to pivoting on the middle centroid of the
            # plan's own corridor.
            #
            # "start" pivots on the region the robot begins in, so error grows with
            # distance travelled -- the map is right where you are and wrong far away,
            # which is what a stale overhead map actually looks like. At 30 deg that is a
            # few metres near the start rather than 75 m everywhere.
            if self._str("distortion_pivot") == "start" and len(self._centroids):
                pivot = self._centroids[0].astype(float)
            else:
                pivot = self._centroids.mean(axis=0)
            c, s_ = math.cos(ang), math.sin(ang)
            rel = self._centroids - pivot
            rot = np.stack([rel[:, 0] * c - rel[:, 1] * s_,
                            rel[:, 0] * s_ + rel[:, 1] * c], axis=1)
            self._centroids = pivot + sc * rot + np.array([tx, ty])
        self._bearings = {k: v + ang for k, v in self._bearings.items()}
        self.get_logger().warn(
            f"MAP DISTORTED (pivot={self._str('distortion_pivot') or 'map'}): "
            f"rot {math.degrees(ang):.1f} deg, scale {sc:g}, "
            f"translate ({tx:g}, {ty:g}) applied to {len(self._centroids)} centroids and "
            f"{len(self._bearings)} bearings. Cluster membership is NOT distorted.")

    def _apply_jitter(self) -> None:
        """Displace each region independently: topology right, metric wrong.

        This models a map whose topology is right and whose metric is wrong; a
        similarity transform does not. Rotating or scaling the whole map preserves ALL relative geometry -- every
        edge scales by the same factor, every bearing rotates by the same angle -- so it is
        a FRAME error, internally perfect and merely misaligned, which any localiser could
        estimate away. It leaves relative edge lengths intact.

        Independent per-region displacement destroys relative geometry while leaving
        adjacency untouched: every edge length is individually wrong, every bearing is
        individually wrong, and no transform undoes it. That is "coarse topology right,
        metric unreliable".

        SIGMA IS A FRACTION OF THE MEDIAN EDGE LENGTH (37.1 m on Town05), because a metre
        value means nothing without the map's scale. Resulting bearing error at the vehicle:
            sigma  5% ->  0.6 deg     20% -> 2.7 deg     50% -> 6.9 deg
        which brackets the 1.3-4.0 deg an odometry drift of 1-3 deg/100 m accumulates over
        these 132 m routes.

        Seeded, so a run replays the same map. Adjacency is NOT recomputed: the graph is
        the part that is asserted correct.
        """
        frac = self._float("distortion_jitter_frac")
        jm = self._float("distortion_jitter_m")
        rot = self._float("distortion_region_rot_deg")
        if (frac <= 0 and jm <= 0 and rot <= 0) or not len(self._centroids):
            return
        edges = []
        for key in self._bearings:
            try:
                a, b = (int(x) for x in str(key).split("-"))
            except ValueError:
                continue
            if a in self._region_ids and b in self._region_ids:
                ia, ib = self._region_ids.index(a), self._region_ids.index(b)
                edges.append(float(np.linalg.norm(self._centroids[ia] - self._centroids[ib])))
        med = float(np.median(edges)) if edges else 30.0
        # Metres win when given: a reader can compare 7 m against the 7.0 m road half-width
        # without first knowing the map's median edge length.
        sigma = jm if jm > 0 else frac * med
        rng = np.random.default_rng(abs(self._int("seed")) or 0)
        # KEEP THE OFFSETS. The surface arm displaces each region's waypoints by the SAME
        # vector as its centroid, so the stored map stays internally consistent and is
        # wrong only relative to the true vehicle pose -- i.e. a localisation error rather
        # than an incoherent map. Applying independent noise to the points instead would
        # destroy each region's own shape, which no real map error does.
        offs = rng.normal(0.0, sigma, self._centroids.shape)
        self._jitter_offsets = {int(r): offs[i] for i, r in enumerate(self._region_ids)}
        rots = (rng.normal(0.0, math.radians(rot), len(self._region_ids))
                if rot > 0 else np.zeros(len(self._region_ids)))
        self._jitter_rots = {int(r): float(rots[i]) for i, r in enumerate(self._region_ids)}
        self._centroids = self._centroids + offs
        # Bearings must be RECOMPUTED from the moved centroids, not offset: unlike a
        # rotation, this perturbation is different for every edge.
        for key in list(self._bearings):
            try:
                a, b = (int(x) for x in str(key).split("-"))
            except ValueError:
                continue
            if a in self._region_ids and b in self._region_ids:
                ia, ib = self._region_ids.index(a), self._region_ids.index(b)
                d = self._centroids[ib] - self._centroids[ia]
                self._bearings[key] = math.atan2(d[1], d[0])
        # ORDER MATTERS. `_load_road()` runs in __init__, before the offsets exist, so
        # the surface must be reloaded here, now that `_jitter_offsets` is populated. The
        # log line reports the OUTCOME (`_surface_moved`), not the flag.
        surf = ""
        if self._bool_param("distortion_surface"):
            self._road = self._load_road()
            surf = (" the drivable surface was ALSO DISPLACED"
                    if getattr(self, "_surface_moved", 0) else
                    " the drivable surface was NOT displaced (reload produced no table)")
        else:
            surf = " the drivable surface is pristine"
        self.get_logger().warn(
            f"MAP JITTERED: sigma {sigma:.1f} m translation"
            f"{f', {rot:.0f} deg per-region rotation' if rot > 0 else ''} applied "
            f"independently to {len(self._centroids)} centroids; {len(self._bearings)} "
            f"bearings recomputed. Adjacency is NOT distorted;{surf}.")

    def _load_road(self) -> "RoadSurface | None":
        """The drivable surface, from the same region table gt_cluster_node reads."""
        import os as _os
        npz = self._str("regions_npz")
        if not npz:
            try:
                from ament_index_python.packages import get_package_share_directory
                npz = _os.path.join(get_package_share_directory("carla_gt_bridge"),
                                    "config", "regions.town05.npz")
            except Exception:
                return None
        if not _os.path.exists(npz):
            self.get_logger().warn(
                f"no region table at {npz!r} — the drivable-surface penalty is OFF, so "
                f"nothing stops the vehicle leaving the road")
            return None
        import numpy as _np
        with _np.load(npz, allow_pickle=False) as z:
            # ALL THREE COLUMNS. Column 2 is the region id, and `RoadSurface` falls
            # back to all-zeros when handed a 2-column array -- silently, because an
            # array of the wrong width is still a valid array. With rids missing,
            # `region_of` returns 0 everywhere and `nearest_point_of` returns None for
            # every target, so `guidance="region"` degrades to centroid aiming with no
            # error anywhere.
            _wp = z["waypoints"]
            if self._bool_param("distortion_surface") and getattr(self, "_jitter_offsets", None):
                _wp = _wp.copy()
                _moved = 0
                for _rid, _d in self._jitter_offsets.items():
                    _m = _wp[:, 2].astype(int) == _rid
                    if _m.any():
                        _th = getattr(self, "_jitter_rots", {}).get(_rid, 0.0)
                        if _th:
                            # About the region's OWN centroid, so rotation changes the
                            # region's orientation without also translating it -- the two
                            # error modes stay separable and can be swept independently.
                            _c = _wp[_m, :2].mean(axis=0)
                            _ct, _st = math.cos(_th), math.sin(_th)
                            _rel = _wp[_m, :2] - _c
                            _wp[_m, 0] = _c[0] + _rel[:, 0] * _ct - _rel[:, 1] * _st
                            _wp[_m, 1] = _c[1] + _rel[:, 0] * _st + _rel[:, 1] * _ct
                        _wp[_m, 0] += _d[0]; _wp[_m, 1] += _d[1]; _moved += int(_m.sum())
                self._surface_moved = _moved
                self.get_logger().warn(
                    f"SURFACE DISPLACED: {_moved} of {len(_wp)} waypoints moved with their "
                    f"region. Cluster membership and the goal point now BOTH read a wrong "
                    f"map -- this is the arm the design is not expected to survive.")
            surface = RoadSurface(_wp)
        self.get_logger().info(
            f"drivable surface from {_os.path.basename(npz)}: penalty beyond "
            f"{self._float('road_half_width'):.1f} m from a reference line")
        return surface

    def _int(self, n: str) -> int:
        return self.get_parameter(n).get_parameter_value().integer_value

    def _float(self, n: str) -> float:
        return self.get_parameter(n).get_parameter_value().double_value

    def _str(self, n: str) -> str:
        return self.get_parameter(n).get_parameter_value().string_value

    def _bool(self, n: str) -> bool:
        return self.get_parameter(n).get_parameter_value().bool_value

    # -- callbacks -------------------------------------------------------- #

    def _plan_cb(self, msg: String) -> None:
        """Take the plan's GEOMETRY. Its structure is brain's business, not ours."""
        try:
            plan = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().error(f"/brain/incoming_plan is not JSON: {exc}")
            return
        cents = plan.get("centroids") or {}  # planar metres; see frames.py's frame table
        if not cents:
            self.get_logger().error(
                "plan carries no `centroids` — the taxonomy it was materialised from "
                "had no geometry, so there is nothing to steer toward. Regenerate the "
                "cluster map with scripts/map_regions.py.")
            return
        ids = sorted(int(k) for k in cents)
        self._region_ids = ids
        # NO frame conversion. Centroids and bearings are derived from OpenDRIVE
        # geometry, which is already the planar right-handed frame — the same frame the
        # ros-bridge publishes odometry in. E.g. Town01's .xodr spans y in [-328.6, 0.0]
        # while the CARLA API reports y ~ +273..+330 for the same town, so CARLA-API y is
        # the NEGATION of both. Putting either of these through
        # carla_xy_to_planar would mirror the map about the x axis, and the vehicle would
        # drive confidently toward a target reflected across the town.
        self._centroids = np.array(
            [[float(cents[str(i)][0]), float(cents[str(i)][1])] for i in ids], dtype=float)
        self._bearings = {k: math.radians(float(v))
                          for k, v in (plan.get("bearing_map") or {}).items()}
        # MAP DISTORTION. A similarity transform applied to the map's DERIVED geometry --
        # centroids and the bearings between them -- and to nothing else. Cluster
        # membership still comes from the waypoint table via /predicted_cluster, i.e. from
        # what the robot MEASURES, so this tests whether the plan still executes on a
        # stale, metrically wrong map when progress is judged by which region the vehicle
        # is in rather than by where the map says it is.
        #
        # Distorting membership too would be a different (and much less interesting)
        # experiment -- it would corrupt the perception the method relies on rather than
        # the prior it deliberately does not.
        self._apply_distortion()
        self._labels = {int(k): v for k, v in (plan.get("cluster_labels") or {}).items()}
        self._plan_name = plan.get("plan_name", "")

        # PROHIBITION BECOMES PREVENTION. `MpcConfig.hard_forbid` removes any rollout
        # ending in a forbidden region. Without populating it from the plan's
        # `forbid_modes`, ConstraintMonitor only logs a violation after the fact and
        # "never enter X" is detection, not enforcement.
        #
        # Resolved HERE rather than in brain because the sampler needs region IDs and the
        # tree already carries `cluster_labels` (id -> mode), so the mapping is local and
        # needs no taxonomy import.
        forbid_modes = [str(m).strip().lower()
                        for m in (plan.get("forbid_modes") or []) if m]
        if forbid_modes:
            banned = frozenset(
                i for i, lab in self._labels.items()
                if str(lab).strip().lower() in forbid_modes)
            self._cfg.hard_forbid = banned
            if banned:
                self.get_logger().info(
                    f"hard_forbid: {sorted(forbid_modes)} -> {len(banned)} region(s) "
                    f"{sorted(banned)[:12]} — rollouts ending there are rejected")
            else:
                # Loud on purpose: a prohibition naming a mode this map does not have is
                # a silently vacuous constraint.
                self.get_logger().error(
                    f"forbid_modes {sorted(forbid_modes)} matched NO region in this plan's "
                    f"cluster_labels {sorted(set(map(str, self._labels.values())))} — the "
                    f"prohibition is vacuous and nothing will be prevented.")
        else:
            self._cfg.hard_forbid = frozenset()
        # Grounding: brain names a MODE, we have to steer at a REGION. See
        # carla_gt_bridge.routing — `canonical_id('junction')` is the lowest junction id
        # in the corridor, which on Town05 is 53, the SECOND junction. Steering at that
        # from region 45 means aiming past junction 66 at the one after it, and the
        # vehicle just turns hard toward it. StepTargeter resolves the adjacent one.
        self._targeter = StepTargeter(
            adjacency_from_bearing_map(plan.get("bearing_map") or {}),
            self._labels, plan.get("bearing_map") or {})
        named = ", ".join("{}={}".format(i, self._labels.get(i, "?")) for i in ids)
        self.get_logger().info(
            f"plan '{self._plan_name}': regions {ids} ({named}), "
            f"{len(self._bearings)} bearings")

    def _state_cb(self, msg: String) -> None:
        try:
            st = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        prev_state = self._brain_state
        self._brain_state = st.get("state", "")
        # CAPTURE THE JUNCTION AT THE MOMENT THE DECISION STARTS, not when it finishes.
        # The VLM round-trip is wall-clock bound and the simulator runs ~5x real time, so
        # the vehicle is typically 25-32 m past this point by the time a branch commits.
        if self._brain_state == "DECIDING" and prev_state != "DECIDING":
            self._decide_at = (self._targeter.current, self._targeter.previous)
            self.get_logger().info(
                f"decision point: grounding the NEXT turn from region "
                f"{self._decide_at[0]} (prev {self._decide_at[1]})")
        elif self._brain_state in ("", "WAITING_FOR_PLAN"):
            self._decide_at = None
        self._step_idx = st.get("step")
        self._branch_path = st.get("branch_path")
        self._complete = bool(st.get("complete"))
        new_start, new_goal = st.get("start_cluster"), st.get("goal_cluster")
        # The LABEL is what we act on; the ids are brain's canonical stand-ins and are
        # only logged. `transition_cue` carries the maneuver for a branch fork.
        self._goal_label = st.get("goal_label")
        cue = st.get("transition_cue") or ""
        self._maneuver = ("right" if "Right" in cue
                          else "left" if "Left" in cue else "straight")
        if (new_start, new_goal) != (self._start_id, self._goal_id):
            self.get_logger().info(
                f"brain step {self._step_idx}: {st.get('start_label')} -> "
                f"{self._goal_label} (canonical {new_start} -> {new_goal}, "
                f"not steering targets), trigger={st.get('trigger')}, "
                f"maneuver={self._maneuver}, state={self._brain_state}")
        self._start_id, self._goal_id = new_start, new_goal

    def _odom_cb(self, msg: Odometry) -> None:
        """Pose in, and — every ``dt`` of stamp time — one control step out.

        THE CONTROL LOOP RUNS HERE, not on a ROS timer. The bridge publishes odometry at
        ~300 Hz, and on a single-threaded executor that starves a 5 Hz timer completely.
        Giving the sensors their own reentrant group and more threads only inverts it. At
        this publish rate one of the two callbacks always wins, so the fix cannot be a
        scheduling tweak.

        Running control ON THE POSE removes the competition and two bugs with it: the
        controller can no longer act on a stale pose (there is exactly one, the one that
        just arrived), and the rate is bounded by dt regardless of how fast the simulator
        chooses to run.

        Throttled on the MESSAGE STAMP rather than wall or node time, so the cadence is
        dt of SIMULATED seconds — which is what dt meant when the world ran near real
        time, and what the MPC's horizon assumes.
        """
        self._n_odom += 1
        # carla_ros_bridge already publishes in the ROS right-handed convention, which IS
        # the planar frame — so position and yaw both pass through untouched. Converting
        # here would double-negate y against the (also planar) OpenDRIVE centroids.
        p = msg.pose.pose.position
        self._pose = (float(p.x), float(p.y),
                      _yaw_from_quaternion(msg.pose.pose.orientation))
        # Measured forward speed. Needed to convert an angular VELOCITY into a steering
        # ANGLE — see _steer_angle_for.
        self._speed = abs(float(msg.twist.twist.linear.x))

        # COUNT MESSAGES, DO NOT READ A CLOCK. No available clock is usable:
        #   /clock                races far faster than real time
        #   odom header stamp     effectively frozen across hundreds of messages, so a
        #                         stamp-throttled loop runs only once
        #   wall clock            meaningless when the sim does not track it
        # In synchronous mode one odometry message IS one world tick, so counting them
        # is a real clock: the only one in this system that advances monotonically with
        # simulated time. See `control_every_n_odom` for why the default is 1 rather
        # than dt / fixed_delta_seconds.
        if self._n_odom % self._control_every_n == 0:
            self._control_loop()

    def _ranges_cb(self, msg: Float32MultiArray) -> None:
        self._ranges = np.asarray(msg.data, dtype=np.float32)

    def _cluster_cb(self, msg: Int16) -> None:
        self._cluster = int(msg.data)
        if self._targeter is not None and self._targeter.observe(self._cluster):
            self.get_logger().info(
                f"cluster -> {self._cluster} "
                f"({self._labels.get(self._cluster, '?')}), "
                f"previous {self._targeter.previous}")

    # -- control ---------------------------------------------------------- #

    def _hits_body(self) -> np.ndarray | None:
        """LiDAR ranges -> body-frame hit points, or None when there is no scan."""
        if self._ranges is None or len(self._ranges) == 0:
            return None
        r = self._ranges
        ang = np.linspace(0.0, 2 * math.pi, len(r), endpoint=False)
        # The publisher's bin 0 faces +x (forward) and bins advance CCW, matching the
        # planar frame the MPC rolls out in.
        hits = np.stack([r * np.cos(ang), r * np.sin(ang)], axis=1)
        keep = (r < r.max() * 0.99) & (np.linalg.norm(hits, axis=1) > 0.1)
        hits = hits[keep]
        return hits if len(hits) else None

    def _stop(self) -> None:
        self._twist_pub.publish(Twist())

    def _control_loop(self) -> None:
        if not self._region_ids:
            self.get_logger().warning("waiting for /brain/incoming_plan",
                                      throttle_duration_sec=5.0)
            return
        if self._pose is None:
            self.get_logger().warning(f"waiting for {self._str('odom_topic')}",
                                      throttle_duration_sec=5.0)
            return
        if self._complete:
            self._stop()
            self.get_logger().info("brain reports plan complete — holding stop",
                                   throttle_duration_sec=10.0)
            return
        if self._goal_label is None or self._targeter is None:
            self.get_logger().warning(
                f"waiting for /brain/state to name a step (state={self._brain_state!r})",
                throttle_duration_sec=5.0)
            self._stop()
            return
        if (self._terrain_mode and self._terrain_source == "camera"
                and self._mask_seen == 0):
            # One level past the `terrain_class is None` guard: a source that
            # returns all-UNOBSERVED runs happily and reports terrain-scored numbers while
            # the controller is really bearing-only. Stop instead of driving on it.
            self.get_logger().error(
                f"terrain_source=camera but no mask has arrived on "
                f"{self._str('mask_topic')} -- is terrain_mask_node up, and is rendering on? "
                f"NOT driving terrain-scored on an absent camera.",
                throttle_duration_sec=5.0)
            self._stop()
            return
        if self._bool("require_ranges") and self._ranges is None:
            self.get_logger().warning("waiting for /processed_ranges",
                                      throttle_duration_sec=5.0)
            self._stop()
            return

        x, y, yaw = self._pose
        self._check_start_pose(x, y)

        # A TURN IS DEFINED RELATIVE TO ITS JUNCTION. Ground left/right from where the
        # decision was taken; `straight` needs no such anchor and keeps the live region so
        # ordinary driving is untouched.
        from_region = from_prev = None
        if self._maneuver in ("left", "right") and self._decide_at is not None:
            # VALID FOR THE FIRST STEP OF THE SUB-PLAN ONLY, and that test is POSITIONAL
            # rather than use-based. Binding it to the first step that USES it is wrong:
            # a sub-plan whose step 0 is `Bearing(Straight)` never consults the anchor, so
            # the first use can be a later turn at a different junction -- precisely the
            # step it must not apply to. `descend()` restarts numbering inside the
            # sub-plan, so the one step that can have its junction behind the vehicle is
            # step 0.
            if self._step_idx == 0:
                from_region, from_prev = self._decide_at
            else:
                self.get_logger().info(
                    f"decision anchor {self._decide_at} does not apply at sub-plan step "
                    f"{self._step_idx}; grounding this turn from the live region "
                    f"{self._targeter.current}")
                self._decide_at = None
        target, note = self._targeter.target_for(
            self._goal_label, step_key=(self._step_idx, self._branch_path),
            maneuver=self._maneuver,
            from_region=from_region, from_previous=from_prev)
        # Arrived, but brain is still counting sightings: keep going to the NEXT region
        # of this label rather than parking. That is what an ordinal cue means — "the
        # second cone" cannot be satisfied standing at the first one. brain announces this
        # by staying in CHECKING_CUE after the cluster half of the step is satisfied.
        if (target is not None and self._targeter.current == target
                and self._brain_state == "CHECKING_CUE"):
            nxt, adv = self._targeter.advance_target(self._goal_label)
            if nxt is not None:
                target = nxt
                if adv != self._last_note:
                    self.get_logger().info(adv)
                    self._last_note = adv
        if target is None:
            # "no cluster observed yet" is expected for the first tick or two — odometry
            # arrives before /predicted_cluster does — so it is logged at INFO. Anything
            # else here IS a real disagreement between the plan and the map.
            starting = "no cluster observed yet" in note
            log = self.get_logger().info if starting else self.get_logger().error
            log(f"cannot ground this step yet: {note}"
                + ("" if starting else " — stopping"),
                throttle_duration_sec=5.0)
            self._stop()
            return
        if note != self._last_note:
            self.get_logger().info(note)
            self._last_note = note
        # Steer FROM the region we came from once we are already standing in the target:
        # the progress term needs two distinct regions to define an axis.
        cur = self._targeter.current
        start = (self._targeter.previous if cur == target and
                 self._targeter.previous is not None else cur)
        if start is None or start == target:
            start = cur if cur != target else self._region_ids[0]

        if self._terrain_mode:
            # The terrain source closes over the CURRENT pose, so it is rebuilt per tick
            # rather than cached -- a stale R/pos would classify the rollouts against
            # where the vehicle was -- a silent error.
            cy, sy = math.cos(yaw), math.sin(yaw)
            R_l2g = np.array([[cy, -sy], [sy, cy]])
            pos = np.array([x, y])
            if self._terrain_source == "waypoint":
                klass = {int(r): (1 if int(r) in self._cfg.hard_forbid else 0)
                         for r in self._region_ids}
                terrain_class = waypoint_terrain_source(
                    self._road, R_l2g, pos, klass,
                    half_width=float(self._float("road_half_width")))
                terrain_costs, forbid, unobserved = {0: 0.0, 1: 5.0, -1: 5.0}, \
                    frozenset({1, -1}), None
            else:
                # REBUILT PER TICK, like the waypoint source and for the same reason: it
                # closes over the CURRENT pose, and a cached one would carry rollout points
                # back into the wrong frame -- classifying them against where the vehicle
                # used to be.
                terrain_class = self._mask_src.source_for((x, y, yaw), self._now_s())
                terrain_costs = dict(DEFAULT_COSTS)
                forbid, unobserved = self._forbid_terrain, UNOBSERVED
            decay = self._float("terrain_decay_m")
            s_min = (self._mask_src.cam.effective_min_range_m()
                     if self._mask_src is not None else 0.0)
            tcfg = TerrainMpcConfig(
                K=self._int("fan_size"), n=16,
                segments=self._int("fan_segments"),
                arc_len_m=float(self._float("arc_len_m")),
                kappa_max=float(self._float("kappa_max")),
                v_max=float(self._float("v_max")),
                safety_radius=float(self._float("safety_radius")),
                terrain_class=terrain_class,
                terrain_costs=terrain_costs,
                unobserved_class=unobserved,
                terrain_decay_m=(decay if decay > 0 else None),
                terrain_s_min_m=s_min,
                forbid_classes=forbid)
            # AIM THROUGH THE REGION, NOT AT ITS CENTROID.
            #
            # Aiming at the centroid fails on long regions: once the vehicle is past the
            # centroid of a long `path` it is correctly driving along, the aim is BEHIND it
            # and the controller steers backwards down the road and off the carriageway.
            #
            # `aim_point` is the same rule the REGION controller uses: the nearest point
            # of the route that is far enough ahead to steer toward. `via` supplies the
            # next regions, because near a region's end there is no lookahead left inside
            # it.
            aim = aim_point(self._road, int(target), pos, yaw,
                            # `lookahead_m` is an MpcConfig FIELD, not a declared ROS
                            # parameter, so read the config rather than the params.
                            lookahead_m=float(self._cfg.lookahead_m),
                            also=list(self._targeter.via or []))
            if aim is None:
                # Explicit, logged fallback: a silent centroid substitution would hide
                # the aim-behind failure described above.
                aim = self._centroids[self._region_ids.index(int(target))]
                self.get_logger().warning(
                    f"no drivable points for region {target}; aiming at its centroid",
                    throttle_duration_sec=10.0)
            d = np.asarray(aim, dtype=float) - pos
            bearing_ref = math.atan2(*(R_l2g.T @ d)[::-1])
            self._bearing_ref = float(bearing_ref)
            self._steer_target = int(target)
            res = plan_step_terrain(tcfg, bearing_ref, kappa_prev=self._kappa_prev,
                                    hits_body=self._hits_body(), occ=self._occ)
            self._kappa_prev = (res.omega / res.v) if res.v > 1e-6 else 0.0
        else:
            try:
                res = plan_step(
                    self._cfg, np.array([x, y]), yaw,
                    self._centroids, self._region_ids,
                    int(start), int(target),
                    hits_body=self._hits_body(), occ=self._occ, rng=self._rng,
                    road=self._road, via=self._targeter.via)
            except KeyError as exc:
                # brain named a region this plan has no centroid for. Stop rather than
                # steering at a guess: the two are out of sync and driving on would put the
                # vehicle somewhere no later log could explain.
                self.get_logger().error(f"{exc} — stopping", throttle_duration_sec=5.0)
                self._stop()
                return

        if not self._bool("dry_run"):
            tw = Twist()
            v_out = float(res.v)
            # CREEP WHILE THE BRANCH IS BEING DECIDED.
            #
            # At full speed the vehicle drives straight OUT of the decision junction while
            # the VLM is being consulted, so by the time the turn step is grounded the
            # targeter is grounding it from the region AFTER the junction, where the turn
            # does not exist, and the vehicle goes straight through.
            #
            # A CLAMP, NOT A STOP. `_stop()` inside a junction under
            # synchronous_mode_wait_for_vehicle_control_command tends to produce the
            # "commanded, did not move" failure. Creeping keeps the control loop alive and
            # avoids turning from rest, which is a different control problem than turning
            # at speed.
            if self._brain_state == "DECIDING":
                v_out = min(v_out, float(self._float("decide_v_max")))
            tw.linear.x = v_out
            # NOT res.omega. See _steer_angle_for — carla_twist_to_control reads this
            # field as a steering ANGLE, and we plan in angular VELOCITY.
            tw.angular.z = float(self._steer_angle_for(res.omega, res.v))
            self._twist_pub.publish(tw)

        if (self._overlay_every > 0 and self._terrain_mode
                and self._terrain_source == "camera"
                and self._n_ticks % self._overlay_every == 0):
            self._write_overlay(res, tcfg, x, y, yaw)
        self._publish_debug(res)
        self._log(x, y, yaw, res, start, target)
        self._n_ticks += 1
        self.get_logger().info(
            f"[mpc] ({x:.1f},{y:.1f}) yaw={math.degrees(yaw):+.0f} "
            f"{start}->{target} ({self._goal_label}) cluster={self._cluster} "
            f"v={res.v:.2f} om={res.omega:+.2f} edt={res.edt_blocked_frac:.2f} "
            + (f"cov={res.coverage_frac:.2f} blind={res.blind_arcs} "
               f"soft={res.terrain_soft:.2f} vetoed={res.forbidden_rejected} "
               f"masks={self._mask_seen} "
               if self._terrain_mode and self._terrain_source == "camera" else "")
            + f"ratio={res.guidance_ratio:.2f} odom/tick={self._n_odom}/{self._n_ticks}"
            + ("  RECOVERY" if res.recovery else ""),
            throttle_duration_sec=1.0)

    def _mask_cb(self, msg: Image) -> None:
        """Store the mask together with the pose it was TAKEN from.

        Not the pose now. The camera is throttled to 2 Hz against a ~5 Hz control loop, so at
        5 m/s a frame can describe ground 2.5 m behind where the rollouts start -- a quarter
        of the arc. `MaskTerrainSource` carries every rollout point back into this pose before
        projecting, which is only possible because the pose travels WITH the mask.
        """
        if msg.encoding != "mono8":
            self.get_logger().error(
                f"mask encoding {msg.encoding!r}, want mono8 -- refusing to guess",
                throttle_duration_sec=10.0)
            return
        pose = parse_frame_id(msg.header.frame_id)
        if pose is None:
            # NOT a fallback to the current pose. A mask placed at the wrong pose classifies
            # every rollout against the wrong patch of world and reads as a segmentation
            # failure, which is the hardest kind of bug to find from a log.
            self.get_logger().error(
                f"mask header.frame_id carries no pose: {msg.header.frame_id!r}",
                throttle_duration_sec=10.0)
            return
        x, y, yaw, _n = pose
        klass = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width))
        try:
            self._mask_src.update(klass.astype(np.int16), (x, y, yaw),
                                  stamp=self._now_s())
        except ValueError as exc:
            self.get_logger().error(str(exc), throttle_duration_sec=10.0)
            return
        self._mask_seen += 1

    def _write_overlay(self, res, cfg, x: float, y: float, yaw: float) -> None:
        """Draw what the controller SAW and CHOSE this tick, OFF THE CONTROL LOOP.

        One render takes seconds of wall time -- 65 arcs x 200 samples x 2 panels. Run
        inline it blocks this node for that long, and because the world keeps ticking the
        stored mask then age-checks as far too old on the next tick, so the tick after
        each overlay is driven with no usable terrain.

        So the render happens on a daemon thread and a tick that arrives while one is still
        going is SKIPPED rather than queued. Skipping is right: a backlog of renders would
        reproduce the stall with extra steps, and the overlays are a sample, not a record.
        Arrays are copied under the caller's stack before the thread starts, because the
        mask and the result are rebound by later ticks.

        The offline gate draws the same picture from captured frames; this draws it from the
        mask the vehicle is actually steering on, through `terrain_overlay.render_fan`, so
        there is one renderer rather than an offline one and a live one that quietly disagree.

        Best-effort: a failed render must never take the control loop down with it. An
        overlay is a debugging aid, and a vehicle that stops driving because a PNG could not
        be written is a worse failure than no PNG.
        """
        mask = getattr(self._mask_src, "_mask", None)
        if mask is None or res.arc_soft is None:
            return
        # DUMP, DO NOT DRAW. A full render can take ~10 s in-container, so at any usable
        # interval nearly every overlay would be skipped. The inputs, though, are tiny --
        # one uint8 mask plus a few 65-element arrays -- and `np.savez_compressed` of them
        # is milliseconds. So the tick writes a snapshot that can be drawn afterwards
        # through the SAME `render_fan`, at full quality and for every tick asked for. No
        # thread, no skip logic, no control-loop cost, and the picture can be re-rendered
        # with different weights without re-driving.
        if self._overlay_dump:
            try:
                np.savez_compressed(
                    os.path.join(self._overlay_dir, f"tick_{self._n_ticks:05d}.npz"),
                    mask=np.asarray(mask, dtype=np.uint8),
                    soft=np.asarray(res.arc_soft, dtype=np.float32),
                    forbidden=np.asarray(res.arc_forbidden, dtype=bool),
                    best=int(res.best_index), tick=int(self._n_ticks),
                    kappa_max=float(cfg.kappa_max), fan_k=int(cfg.K),
                    arc_len_m=float(cfg.arc_len_m), n_samples=int(cfg.n),
                    decay_m=float(cfg.terrain_decay_m or 0.0),
                    s_min_m=float(cfg.terrain_s_min_m),
                    cam_w=int(self._mask_src.cam.width), cam_h=int(self._mask_src.cam.height),
                    cam_fov=float(self._mask_src.cam.fov_deg),
                    cam_x=float(self._mask_src.cam.mount_x),
                    cam_z=float(self._mask_src.cam.mount_z),
                    cam_pitch_down=float(self._mask_src.cam.pitch_down_deg),
                    x=float(x), y=float(y), yaw=float(yaw),
                    coverage=float(res.coverage_frac), blind=int(res.blind_arcs),
                    vetoed=int(res.forbidden_rejected),
                    mask_age_s=float(self._mask_src.last_age_s))
            except Exception as exc:                   # noqa: BLE001
                self.get_logger().warning(f"overlay dump failed: {exc}",
                                          throttle_duration_sec=10.0)
            return
        if self._overlay_busy.is_set():
            # A render still in flight. Skipping is deliberate (a backlog would reproduce the
            # control-loop stall), but it must be VISIBLE: if a render is slower than the
            # overlay interval, every overlay after the first is skipped and the run looks
            # like the feature is off rather than saturated.
            self._overlay_skipped += 1
            self.get_logger().info(
                f"overlay skipped, render still in flight ({self._overlay_skipped} so far) "
                f"-- raise overlay_every_n if this keeps happening",
                throttle_duration_sec=15.0)
            return
        self._overlay_busy.set()
        snap = (np.array(mask, copy=True), np.array(res.arc_soft, copy=True),
                np.array(res.arc_forbidden, copy=True), int(res.best_index),
                int(self._n_ticks), float(res.coverage_frac), int(res.blind_arcs),
                int(res.forbidden_rejected), float(self._mask_src.last_age_s))
        threading.Thread(target=self._render_overlay, args=(snap, cfg, x, y, yaw),
                         daemon=True).start()

    def _render_overlay(self, snap, cfg, x: float, y: float, yaw: float) -> None:
        try:
            # UNPACKED INSIDE THE TRY. Otherwise a snapshot of the wrong shape would raise
            # outside the `finally` that clears `_overlay_busy` -- the flag would latch set,
            # every later tick would take the skip branch, and the run would quietly produce
            # exactly ONE overlay. Thread exceptions do not reach the ROS logger, so there
            # would be no trace at all.
            mask, soft, forb, best, tick, cov, blind, vetoed, age = snap
            # The overlay must draw the fan that was actually SCORED; drawing single
            # arcs while a two-segment fan drives would make the picture a lie about the
            # decision it is captioned with.
            if getattr(cfg, "segments", 1) == 2:
                _pairs = two_segment_fan(cfg.kappa_max, int(round(math.sqrt(cfg.K))))
                kappa = _pairs[:, 0]
                dense = arc_rollout_2seg(_pairs, cfg.arc_len_m, 200)[:, 1:, :2]
            else:
                kappa = curvature_fan(cfg.kappa_max, cfg.K)
                dense = arc_rollout_k(np.zeros(3), kappa, cfg.arc_len_m, 200)[:, 1:, :2]
            w_arc = arc_weights(cfg.arc_len_m, cfg.n, cfg.terrain_decay_m,
                                cfg.terrain_s_min_m)
            out = os.path.join(self._overlay_dir, f"tick_{tick:05d}.png")
            render_fan(mask, self._mask_src.cam, dense, kappa, w_arc, best, forb, soft, out,
                       # ~22 of 65 arcs: a full render is slow enough in-container that
                       # later overlays get skipped. The chart still shows all K.
                       max_arcs=22,
                       title=(f"tick {tick}  ({x:.1f},{y:.1f}) "
                              f"yaw={math.degrees(yaw):+.0f}  cov={cov:.2f} "
                              f"blind={blind}  vetoed={vetoed}  mask {age:.2f}s old"))
        except Exception as exc:                       # noqa: BLE001 -- see the docstring
            self.get_logger().warning(f"overlay render failed: {exc}",
                                      throttle_duration_sec=10.0)
        finally:
            self._overlay_busy.clear()

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _steer_angle_for(self, omega: float, v_planned: float) -> float:
        """Angular velocity (rad/s) -> front-wheel steering angle (rad).

        UNIT CONVERSION. `carla_twist_to_control` converts this node's Twist with

            control.steer = -clamp(twist.angular.z, +-max_steer) / max_steer

        i.e. it reads `angular.z` as a steering ANGLE in radians. The MPC plans in
        angular VELOCITY. The two coincide at exactly one speed and nowhere else, so
        publishing omega directly is a gain error that VARIES WITH SPEED. For a Tesla
        Model 3 (L = 2.875 m, max steer 1.2217 rad):

            v = 5.00 m/s, omega = -0.489  ->  applied 0.400, wanted 0.224   (1.78x too much)
            v = 1.73 m/s, omega = -0.489  ->  applied 0.400, wanted 0.559   (0.72x too little)

        A vehicle accelerating from rest passes through both regimes, so the heading
        error grows, the controller corrects, and it diverges.

        The offline harness never sees this because a unicycle applies omega directly.

        USE THE PLANNED SPEED, NOT THE MEASURED ONE. With the measured speed, at low
        speed the model demands a SHARPER angle for the same yaw rate, so a vehicle
        still accelerating gets hard lock and carves a tight arc. The MPC's omega and v
        are a matched pair describing one path CURVATURE,
        kappa = omega / v; the steering angle should implement that curvature, so the
        car follows the planned geometry while it accelerates onto it.

        Bicycle model: delta = atan(L * kappa). The sign works out — a planar clockwise
        turn (omega < 0) gives delta < 0, which the converter's negation turns into
        positive steer, and positive steer is right in CARLA.
        """
        v = max(abs(v_planned), self._float("min_steer_speed"))
        return math.atan(self._float("wheelbase") * omega / v)

    def _check_start_pose(self, x: float, y: float) -> None:
        """Once, at the first tick: is the ego actually inside the plan's corridor?

        Catches the two failures that otherwise present identically as "the MPC drives
        the wrong way": a vehicle spawned outside the corridor, and a frame sign error.
        """
        if self._start_pose_checked:
            return
        self._start_pose_checked = True
        d = np.linalg.norm(self._centroids - np.array([x, y]), axis=1)
        near_i = int(np.argmin(d))
        near_id = self._region_ids[near_i]
        if d[near_i] > 60.0:
            self.get_logger().error(
                f"initial pose ({x:.1f}, {y:.1f}) is {d[near_i]:.0f} m from the nearest "
                f"plan region ({near_id}) — wrong town, wrong spawn point, or a frame "
                f"sign error. The run will not be meaningful.")
        else:
            self.get_logger().info(
                f"initial pose ({x:.1f}, {y:.1f}) nearest region {near_id} "
                f"({self._labels.get(near_id, '?')}) at {d[near_i]:.1f} m")
        self._seed_previous_from_heading(x, y)

    def _seed_previous_from_heading(self, x: float, y: float) -> None:
        """Tell the router which way we FACE, by feeding it the region behind us first.

        `next_along` and `region_for_maneuver` both refuse to turn back into `previous`.
        At step 0 `previous` is None, so nothing is excluded and every candidate ties.
        `maneuver_toward` returns "straight" for ALL of them when previous is None, so a
        start region with two adjacent junctions is unresolvable and the run stops with
        `maneuver 'straight' matches none of [63, 64]` — which is really "matches BOTH".

        `simulate()` does the same. A start region with a single junction neighbour
        (e.g. Town05 region 45) has no tie to break; one with two (region 0, neighbours
        63 AND 64) needs this.

        Seeds the neighbour more than 90 degrees off the current heading: the one we
        would have come FROM if we had driven here.
        """
        if self._targeter is None or self._pose is None:
            return
        yaw = self._pose[2]
        best, best_off = None, None
        for nb in self._targeter.adj.get(self._cluster, ()):
            if nb not in self._region_ids:
                continue
            cx, cy = self._centroids[self._region_ids.index(nb)]
            off = abs(wrap_pi(math.atan2(cy - y, cx - x) - yaw))
            if best_off is None or off > best_off:
                best, best_off = nb, off
        if best is not None and best_off is not None and best_off > math.pi / 2:
            # ORDER MATTERS, and getting it wrong is silent. The cluster callback has
            # ALREADY observed the start region by now, so observing `best` alone leaves
            # previous=start, current=behind — the router then plans from the region
            # behind us. Observe the behind-region and then the real one, so the pair
            # ends up (previous=behind, current=start). `simulate()` avoids this only
            # because it seeds before observing anything.
            self._targeter.observe(best)
            self._targeter.observe(self._cluster)
            self.get_logger().info(
                f"seeded previous={best} ({math.degrees(best_off):.0f} deg behind the "
                f"ego), current={self._cluster} — the router now knows which way we face")

    # -- debug + logging -------------------------------------------------- #

    def _publish_debug(self, res) -> None:
        g = Float32MultiArray()
        g.data = [float(res.guidance_ratio), float(res.cluster_score),
                  float(res.bearing_score), float(res.edt_blocked_frac),
                  float(res.v), float(res.omega)]
        self._guidance_pub.publish(g)
        r = Float32MultiArray()
        r.data = res.best_rollout.flatten().tolist()
        self._rollout_pub.publish(r)

    def _open_log(self) -> str:
        d = self._str("log_dir") or os.path.join(os.path.dirname(__file__), "debug_logs")
        os.makedirs(d, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        p = os.path.join(d, f"carla_mpc_{ts}.jsonl")
        self._log_file = open(p, "w")
        self.get_logger().info(f"per-tick log: {p}")
        return p

    def _log(self, x, y, yaw, res, start=None, target=None) -> None:
        # Body -> world rotation for the candidate rollouts logged below. Built
        # here rather than passed in: `_log` already has the pose, and doing it
        # once per tick keeps the caller's signature unchanged.
        _c, _s = math.cos(yaw), math.sin(yaw)
        _R_l2g = np.array([[_c, -_s], [_s, _c]], dtype=float)
        _pos = np.array([x, y], dtype=float)

        self._log_file.write(json.dumps({
            # BOTH clocks. `t` is wall time; `t_sim` is the simulator's. They differ by
            # a large factor because the bridge free-runs, so any speed computed from wall
            # time is meaningless (e.g. 50 m/s for a 5 m/s command).
            "t": time.time(),
            "t_sim": self.get_clock().now().nanoseconds * 1e-9,
            "tick": self._n_ticks,
            # Which mission this is. Without it a log is just a list of coordinates and
            # the run has to be identified by eye from its region sequence.
            "plan_name": self._plan_name,
            "x": x, "y": y, "yaw_deg": math.degrees(wrap_pi(yaw)),
            "brain_state": self._brain_state, "step": self._step_idx,
            "start_cluster": self._start_id, "goal_cluster": self._goal_id,
            "steer_from": start, "steer_to": target, "goal_label": self._goal_label,
            "gt_cluster": self._cluster,
            "v_cmd": res.v, "om_cmd": res.omega,
            "cluster_score": res.cluster_score, "bearing_score": res.bearing_score,
            "guidance_ratio": res.guidance_ratio,
            "edt_blocked": res.edt_blocked_frac,
            "offroad_frac": res.offroad_frac,
            "forbidden_rejected": int(getattr(res, "forbidden_rejected", 0)),
            "recovery": res.recovery,
            # LOGGED separately from `recovery`: a missing key reads as False in analysis,
            # which is a check that cannot fail.
            "obstacle_relaxed": bool(getattr(res, "obstacle_relaxed", False)),
            "terrain_relaxed": bool(getattr(res, "terrain_relaxed", False)),
            # The terrain arm's own evidence. Without these a run that drove on an absent or
            # stale camera is indistinguishable in the log from one that saw clean road --
            # `terrain_soft` is the graded cost of the arc actually chosen, `coverage_frac`
            # and `blind_arcs` say how much of the fan the source could describe at all, and
            # `mask_age_s` says how old the frame behind that was.
            "terrain_source": self._terrain_source if self._terrain_mode else "",
            # The terrain branch's steering target. `start`, `target` and `nearest_region`
            # are None in terrain mode, so without these the log can show a symptom (e.g.
            # the chosen arc pointing away from the reference) but nothing about its cause.
            "bearing_ref": getattr(self, "_bearing_ref", float("nan")),
            "steer_target": getattr(self, "_steer_target", -1),
            "kappa_cmd": (float(res.omega / res.v) if res.v > 1e-6 else 0.0),
            "terrain_soft": res.terrain_soft,
            "coverage_frac": res.coverage_frac,
            "blind_arcs": int(res.blind_arcs),
            "mask_age_s": (self._mask_src.last_age_s
                           if self._mask_src is not None else float("nan")),
            "masks_seen": self._mask_seen,
            "d_min": res.d_min if math.isfinite(res.d_min) else -1.0,
            "n_hits": res.n_hits, "rollout_region": res.nearest_region,
            # Candidate rollouts for the viewer. Present only when
            # MpcConfig.debug_rollouts > 0 (see the `debug_rollouts` ROS param);
            # otherwise the key is absent and the viewer falls back to
            # reconstructing the committed arc from (v, omega).
            #
            # WORLD frame, decimated to 2 dp: the sampler works in the BODY frame,
            # but the viewer draws on a map, and converting once here beats
            # shipping the pose and redoing it per frame in JS.
            **({"rollouts": [
                    [[round(float(px), 2), round(float(py), 2)] for px, py in
                     (r_[:, :2] @ _R_l2g.T + _pos)]
                    for r_ in res.sampled_rollouts],
                "rollouts_rejected": [bool(b) for b in res.sampled_rejected]}
               if res.sampled_rollouts is not None else {}),
        }) + "\n")
        self._log_file.flush()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CarlaMpcNode()
    # Single-threaded on purpose — see the callback-group comment in __init__.
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
