"""Sampling MPC core: pure numpy, no ROS, no CARLA, no JAX.

Extracted from ``carla_sampling_mpc_ros_node.py`` so the part that decides where the
vehicle goes can be tested directly, without CARLA. A sign error in the frame
conversion otherwise looks exactly like "the MPC doesn't work".

The method
----------
Sample ``K`` control sequences of length ``N``, roll each out through a unicycle model,
score the endpoints, take the first control of the best sequence. No gradients, no
learned model. Scoring has three terms:

* **cluster** — which region does the endpoint land in? Nearest-CENTROID Voronoi, which
  is an approximation chosen for speed (K*N distance computations per tick). It is
  reliable within a mission corridor but degrades town-wide, because a long street's
  far end is nearer a neighbour's centroid. Scope plans to a corridor.
* **bearing** — align the final heading with the direction to the target centroid. This
  is the PRIMARY signal: the cluster term is piecewise-constant and gives no gradient
  until an endpoint actually crosses a Voronoi boundary, so on its own the vehicle
  wanders. Bearing is what makes it drive.
* **progress** — distance advanced along the start->target axis, to break the tie
  between "heading the right way" and "getting there".

Frames
------
Everything here is in the **planar** frame: metres, +y north, yaw radians
counter-clockwise. Convert CARLA poses with :mod:`carla_gt_bridge.frames` before calling
in. The unicycle integration assumes counter-clockwise-positive yaw, so feeding raw
CARLA yaw makes every turn go the wrong way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["MpcConfig", "MpcResult", "LocalOccGrid", "RoadSurface", "unicycle_rollout",
           "sample_controls", "plan_step"]


@dataclass
class MpcConfig:
    """Tunables. Defaults are sized for a CAR on a road, not the original 1 m/s."""

    K: int = 500                    # rollouts per tick
    N: int = 8                      # horizon steps
    dt: float = 0.2                 # seconds per horizon step
    v_max: float = 5.0              # m/s
    omega_max: float = 1.0          # rad/s
    #: Cap on |omega| / v, i.e. 1 / minimum turning radius. 0.25 => R_min = 4 m.
    #: Without this the sampler proposes rollouts that spin in place at speed, which a
    #: car cannot do; the MPC then picks one, the drivetrain refuses, and the tracking
    #: error looks like a controller bug.
    max_curvature: float = 0.25
    safety_radius: float = 1.5      # m, EDT rejection distance
    #: How many of the N horizon steps the collision test examines. 0 = all N, the
    #: original behaviour. The test is `.any()` over every step, so a rollout is killed
    #: by its LAST step -- 1.6 s out, the least trustworthy prediction it makes -- even
    #: though the controller replans every tick and would never execute that step.
    #:
    #: At a large junction with traffic-light poles a few metres from the driving line
    #: (e.g. Town05 junction 57), checking all N steps rejects most rollouts, leaving
    #: only near-stationary ones, so the vehicle crawls and never crosses.
    collision_horizon: int = 0
    w_target: float = 10.0
    w_start: float = 1.0
    w_forbidden: float = -15.0
    #: Regions the PLAN forbids (`forbid_clusters`), as opposed to regions merely off the
    #: current step's route. The distinction matters: `w_forbidden` is a soft -15 that a
    #: strong bearing term can outvote, which is correct for "not on the route" and wrong
    #: for "never enter the parking lot". A plan-level prohibition is not a preference.
    hard_forbid: frozenset = frozenset()
    #: When True a rollout ending in `hard_forbid` is removed from consideration entirely,
    #: the same way `reject_offroad` removes one that leaves the road. Skipped if it would
    #: reject every sample -- that is a statement about sparse sampling, not about the
    #: world, and silently freezing is worse than entering the region and reporting it.
    reject_forbidden: bool = True
    w_bearing_cos: float = 3.0
    w_bearing_dot: float = 1.0
    w_progress: float = 0.8
    #: GRADED OBSTACLE PENALTY, ported from terrain_mpc. OFF by default.
    #:
    #: The -inf veto alone is only the right tool where a legal alternative exists; when
    #: every candidate is blocked it destroys the ordering and hands over to the
    #: creep-and-turn recovery, which does not know where the obstacle is. A penalty on
    #: closest approach keeps a gradient: with an obstacle dead ahead, the penalty turns
    #: the vehicle away far more reliably than the veto alone.
    #:
    #: ARC TRUNCATION WAS ALSO PORTED AND THEN REMOVED, because it is inert here. In
    #: terrain_mpc the veto RELAXES when it would kill every candidate and scoring
    #: continues, which is the window where endpoint-scored progress credits a rollout
    #: with distance it drives THROUGH an obstacle. `sampling_mpc` has no relax branch: a
    #: colliding rollout is already -inf before progress is scored, so there is nothing
    #: for truncation to correct: behaviour is identical with it on and off, both for a
    #: single obstacle and for a full wall.
    w_obstacle: float = 0.0
    #: Distance at which the graded penalty reaches zero. Should exceed the turning radius
    #: (1 / max_curvature = 4 m at 0.25) or it carries no gradient where steering is the
    #: only option left.
    obstacle_influence_m: float = 4.0
    #: Penalty on the magnitude of the steering actually commanded. Breaks ties between
    #: arcs that end up in much the same place, in favour of the straighter one.
    w_smooth: float = 0.5
    #: Hold one (v, omega) for the whole horizon instead of sampling a fresh pair per
    #: step. This is not a tuning knob — it is a correctness fix. With per-step controls
    #: the score depends only on where the rollout ENDS, so a wiggle that arrives at the
    #: same endpoint scores the same as driving straight; the command returned is that
    #: wiggle's first control, which is then essentially random: even aimed dead at the
    #: target, omega varies across seeds up to full lock in both directions, which in
    #: CARLA is continuous weaving. Constant controls make each rollout a single arc, so the
    #: argmax IS the chosen arc (the Dynamic-Window formulation). Set False to recover
    #: the original per-step sampling.
    constant_controls: bool = True
    #: CBF post-filter: keep this clear of the nearest obstacle.
    d_safe: float = 1.5
    cbf_alpha: float = 2.0
    #: REJECT rollouts that leave the road outright, rather than merely penalising them.
    #:
    #: The map knows where road is, so a rollout landing off it is not a bad option to be
    #: weighed against others — it is not an option. Rejection also makes the controller
    #: depend on region MEMBERSHIP rather than on centroid geometry, which matters because
    #: the two have very different reliability: cluster membership comes from the undistorted ground-truth waypoint table and
    #: accurate, while centroids are derived and may be offset. A controller that steers
    #: only by centroid distance inherits every calibration error in the geometry; one
    #: that is constrained by membership does not.
    #:
    #: Falls back to the soft penalty if EVERY rollout would be rejected, which happens
    #: where reference-line sampling is sparse inside junctions. Without that fallback the
    #: recovery heuristic takes over exactly where the geometry is hardest.
    reject_offroad: bool = True

    #: How many CANDIDATE rollouts to hand back for visualisation, alongside the
    #: chosen one. 0 (the default) keeps the hot path allocation-free — this is a
    #: 500-rollout sampler running at 5 Hz and nothing in the control loop reads
    #: these. Set it non-zero only for a logging/replay run.
    #:
    #: Without these the log records only the DECISION, not the options it was
    #: chosen from. Integrating the issued (v, omega) recovers the winner but says
    #: nothing about what was rejected — which is the interesting half when a run
    #: hugs the kerb or refuses to turn.
    debug_rollouts: int = 0
    #: How a rollout is judged to be "in the target region", and what it steers at.
    #:
    #:   "centroid"  nearest-centroid Voronoi, and aim at the target centroid.
    #:   "region"    nearest-WAYPOINT membership — the same test the classifier
    #:               publishes — and aim at the nearest point OF the target region.
    #:
    #: The difference only shows up when the map is distorted. Centroids are derived
    #: geometry and can be offset; which cluster you are in comes from the undistorted waypoint table and is not. A
    #: controller guided by membership is therefore expected to tolerate a distortion
    #: that a centroid-guided one cannot.
    guidance: str = "centroid"
    #: Steer at a point of the target region at least this far away, so the vehicle drives
    #: THROUGH the region rather than stopping at its near edge.
    lookahead_m: float = 12.0
    #: Where the steering AIM comes from, independent of where MEMBERSHIP comes from.
    #: The two are separable and the map-distortion ablation is exactly why:
    #:
    #:   membership   waypoint table (`region_of`) -- ground truth, immune to a wrong map
    #:   aim          "region": nearest point of the target region -- also ground truth
    #:                "centroid": the plan's centroid for the target -- DERIVED, and the
    #:                            thing a stale map gets wrong
    #:
    #: Distorting centroids under guidance="centroid" corrupts BOTH, because that mode
    #: computes membership as a Voronoi cell over the same centroids -- so the robot no
    #: longer knows which cluster it is in, which is not what the ablation tests. Distorting them under
    #: guidance="region" with aim_source="centroid" corrupts only the goal, which is: the
    #: map says the target is somewhere wrong, perception still says which cluster you are
    #: standing in, and the question is whether that is enough to stay contained.
    #: "region" | "centroid" | "carrot" -- see the aim selection in `plan_step`.
    aim_source: str = "region"
    #: PURE-PURSUIT BASELINE. Zeroes the region-occupancy weights so the controller steers
    #: at the goal point and nothing else -- no notion of which cluster it is in. This is
    #: the metric-free comparison: under a distorted map it has nothing to fall back on,
    #: while the region term does.
    pure_pursuit: bool = False
    #: Multiplier on both bearing weights when guidance == "region".
    #:
    #: Bearing is computed from the MAP, so under distortion it is exactly the signal you
    #: cannot trust — while membership ("which cluster am I in") is ground truth and stays
    #: true. So bearing is demoted to a weak prior that breaks ties and keeps the vehicle
    #: pointing sensibly, and the cluster term does the driving. At full weight it drags
    #: the vehicle toward a rotated or rescaled target and the perception advantage is
    #: spent fighting it.
    bearing_weight_region: float = 0.25
    #: Penalty per metre a rollout strays beyond the drivable surface. Used for the
    #: fallback above, and to break ties among rollouts that are all legal.
    #:
    #: This is not lane-keeping — the vehicle is free to wander across the whole road,
    #: and steering at region centroids means it will. It is the weaker guarantee that
    #: matters: do not leave the road and drive through a building. Phase A has no
    #: sensors, so the EDT and CBF are inert and NOTHING else prevents that.
    #:
    #: Soft rather than a hard reject: reference-line sampling is sparse at junctions, and
    #: a hard constraint there would reject every rollout and hand control to the recovery
    #: heuristic. A penalty that grows with the excess also pushes back toward the road
    #: instead of merely forbidding a step.
    w_offroad: float = 12.0
    #: Metres from a road's reference line that still count as drivable. The corridor's
    #: roads are 14 m of carriageway (7 m each side of the centre line); beyond the paved
    #: edge at 11-14.6 m is kerb, verge and then buildings.
    road_half_width: float = 7.0


@dataclass
class MpcResult:
    v: float
    omega: float
    best_rollout: np.ndarray        # (N+1, 3) in the BODY frame
    cluster_score: float
    bearing_score: float
    edt_blocked_frac: float
    #: fraction of rollouts that left the drivable surface at any point
    offroad_frac: float
    #: how many were REJECTED for it (0 when the fallback engaged)
    offroad_rejected: int
    #: Rollouts discarded for crossing a `hard_forbid` region, per tick. -1 means EVERY
    #: sample crossed one and the rejection was skipped. Surfaced because it is the direct
    #: evidence that a prohibition is doing something: the trajectory may barely change
    #: while hundreds of options are being removed each tick.
    forbidden_rejected: int
    recovery: bool                  # all rollouts blocked -> creep-and-turn fallback
    d_min: float                    # m to nearest obstacle (inf if none)
    n_hits: int
    nearest_region: int             # region the best rollout ends in
    top_scores: np.ndarray

    #: GRADED terrain cost of the chosen rollout, and the fraction of queried points the
    #: terrain source could describe. Both NaN for this controller, which has no terrain term
    #: at all -- NaN rather than 0.0 so "absent" cannot be read as "scored zero", the same
    #: distinction `cluster_score` carries in the other direction.
    terrain_soft: float = float("nan")
    coverage_frac: float = float("nan")
    #: Rollouts whose terrain the source could not describe at any sample. -1 when not asked.
    blind_arcs: int = -1

    #: PER-ARC verdicts, carried so a live overlay can draw what the scorer actually decided
    #: rather than recomputing it. Recomputation would be a second implementation of the one
    #: thing in this system that must not have two. 65 floats and 65 bools per tick.
    #:
    #: WHICH CONSTRAINT WAS RELAXED, reported separately from `recovery`.
    #:
    #: `recovery` used to be `recovery or obstacle_relaxed`, which conflated two different
    #: events: the creep-and-turn FALLBACK (the vehicle slows to 0.5 m/s because every score
    #: is -inf) and a VETO being skipped because it would have removed every candidate. The
    #: merged flag could read True on nearly every tick while the vehicle never crept, which
    #: made it useless for asking whether a run went wrong.
    obstacle_relaxed: bool = False
    terrain_relaxed: bool = False

    arc_soft: np.ndarray | None = None          # (K,) graded terrain cost per candidate
    arc_forbidden: np.ndarray | None = None     # (K,) bool, crossed a forbidden class
    best_index: int = -1                        # index into the fan of the chosen arc

    #: (M, N+1, 3) BODY-frame sample of the candidates considered this tick, and
    #: (M,) True where that candidate was REJECTED (off-road or in collision).
    #: Empty unless ``MpcConfig.debug_rollouts`` is set. Deliberately a subsample:
    #: all 500 would dominate the log and are visually indistinguishable.
    sampled_rollouts: np.ndarray | None = None
    sampled_rejected: np.ndarray | None = None

    @property
    def guidance_ratio(self) -> float:
        """How much of the chosen rollout's score came from the cluster term.

        Near 0 means bearing is doing all the work (normal); near 1 means the cluster
        term dominates, which happens next to a Voronoi boundary and is where the
        controller is most likely to dither.
        """
        a, b = abs(self.cluster_score), abs(self.bearing_score)
        return a / (a + b + 1e-6)


class RoadSurface:
    """Where the road is, as a nearest-point query over the region waypoint table.

    Ground truth standing in for perception, exactly as `gt_cluster_node` does for the
    classifier: in Phase A there are no sensors, so "is there a wall there" cannot be
    answered by looking. Phase B replaces this with the LiDAR EDT, and the two are
    deliberately separate terms so turning sensors on does not silently remove this.
    """

    def __init__(self, xy: np.ndarray, rid: np.ndarray | None = None) -> None:
        arr = np.asarray(xy, dtype=float)
        self._xy = np.ascontiguousarray(arr[:, :2])
        self._rid = (np.asarray(rid, dtype=int) if rid is not None
                     else (arr[:, 2].astype(int) if arr.shape[1] > 2
                           else np.zeros(len(arr), dtype=int)))
        self._tree = None
        try:
            from scipy.spatial import cKDTree
            self._tree = cKDTree(self._xy)
        except ImportError:                       # pragma: no cover
            pass

    def region_of(self, pts: np.ndarray) -> np.ndarray:
        """Which region each point falls in — nearest WAYPOINT, not nearest centroid.

        This is the same test `gt_cluster_node` publishes, i.e. the PERCEIVED cluster. It
        is independent of the centroids, which is the whole point: centroids are derived
        geometry and can be miscalibrated, while which cluster you are standing in is
        measured.
        """
        flat = pts.reshape(-1, 2)
        if self._tree is not None:
            _, i = self._tree.query(flat)
        else:
            i = np.argmin(((flat[:, None, :] - self._xy[None, :, :]) ** 2).sum(-1), axis=1)
        return self._rid[np.asarray(i)].reshape(pts.shape[:-1])

    def nearest_point_of(self, rid: int) -> np.ndarray | None:
        """Any point known to be inside region ``rid`` — used as a steering target."""
        m = self._rid == rid
        return self._xy[m] if m.any() else None

    def excess(self, pts: np.ndarray, half_width: float) -> np.ndarray:
        """Metres each point lies BEYOND the drivable surface. Zero when on it."""
        flat = pts.reshape(-1, 2)
        if self._tree is not None:
            d, _ = self._tree.query(flat)
        else:
            d = np.sqrt(((flat[:, None, :] - self._xy[None, :, :]) ** 2)
                        .sum(-1)).min(axis=1)
        return np.clip(d.reshape(pts.shape[:-1]) - half_width, 0.0, None)


def aim_point(road: "RoadSurface", target_id: int, pos: np.ndarray, yaw: float,
              lookahead_m: float = 12.0, also: "list[int] | None" = None
              ) -> np.ndarray | None:
    """A steering aim point ON the target region, AHEAD of the vehicle. None if unknown.

    WHAT THIS REPLACES. The terrain controller previously aimed `bearing_ref` at the target
    region's CENTROID. For a long region the vehicle is already driving along (e.g. Town05
    region 1, a 51 m path), the vehicle can be well past the centroid, so the aim is
    behind (`|bearing_ref| > 90 deg`) and the controller turns until it sits across the
    road axis, stuck.

    A centroid is the wrong object to steer at for a region you are ALREADY INSIDE, which is
    the normal case for a `path` step. The region controller already knows this and says so
    in `plan_step`: preferring a point at least `lookahead_m` away "aims THROUGH the region
    instead of at it". This is that rule, made directional and available to both controllers.

    Three choices worth stating:

    * FORWARD FIRST. Among the region's points, only those ahead of the vehicle are
      considered. Without that, a long region straddling the vehicle offers "far enough"
      points in both directions and the nearest one can be the one behind -- the same
      failure as above.
    * NEAREST OF THE FAR-ENOUGH ones, not the farthest. A 51 m region's far end is a 50 m
      lever arm, and every small lateral error there becomes a large bearing error here.
    * A REGION GENUINELY BEHIND STILL GETS AN AIM. If nothing is ahead the vehicle really
      does need to turn around, and the controller needs a finite bearing every tick; the
      point of the fix is that this now happens because the geometry says so, not because a
      centroid was stale.
    """
    # THE ROUTE AHEAD, not just the target region. Near the end of a region there may be
    # no point `lookahead_m` ahead INSIDE it, so a
    # target-only pool degrades to "the furthest scrap of region left", which is a 6 m lever
    # arm and a weak steering signal exactly where the vehicle is about to need a decision.
    # `also` carries the next regions of the plan, mirroring the region controller's carrot
    # pool (`[i_start, i_target] + via`).
    pools = [road.nearest_point_of(int(target_id))]
    for rid in (also or []):
        q = road.nearest_point_of(int(rid))
        if q is not None and len(q):
            pools.append(q)
    pools = [q for q in pools if q is not None and len(q)]
    pts = np.vstack(pools) if pools else None
    if pts is None or not len(pts):
        # No silent substitution. A wrong region's geometry looks like a steering bug
        # hundreds of metres later; the caller decides what to do instead.
        return None
    pos = np.asarray(pos, dtype=float).reshape(2)
    forward = np.array([math.cos(yaw), math.sin(yaw)])
    delta = pts - pos
    dist = np.linalg.norm(delta, axis=1)
    ahead = (delta @ forward) > 0.0

    far_ahead = ahead & (dist >= lookahead_m)
    if far_ahead.any():
        return pts[far_ahead][int(np.argmin(dist[far_ahead]))]
    if ahead.any():                      # inside the lookahead: take the furthest we have
        return pts[ahead][int(np.argmax(dist[ahead]))]
    return pts[int(np.argmin(dist))]     # genuinely behind: nearest, and turn around


class LocalOccGrid:
    """Minimal EDT over LiDAR hits in the body frame: min distance to any hit."""

    def __init__(self) -> None:
        self._hits = np.empty((0, 2), dtype=np.float32)

    def update(self, hits) -> None:
        self._hits = (np.empty((0, 2), dtype=np.float32) if hits is None or len(hits) == 0
                      else np.asarray(hits, dtype=np.float32))

    def check_collisions(self, traj: np.ndarray) -> np.ndarray:
        """(K, N) min distance from each rollout point to the nearest hit."""
        K, N = traj.shape[:2]
        if len(self._hits) == 0:
            return np.full((K, N), np.inf, dtype=np.float32)
        diff = traj[:, :, None, :2] - self._hits[None, None, :, :]
        return np.linalg.norm(diff, axis=-1).min(axis=-1).astype(np.float32)


def unicycle_rollout(state0: np.ndarray, ctrl: np.ndarray, dt: float) -> np.ndarray:
    """Batch unicycle rollout. ``ctrl`` is (K, N, 2) of [v, omega]; returns (K, N+1, 3)."""
    K, N = ctrl.shape[:2]
    traj = np.zeros((K, N + 1, 3), dtype=np.float32)
    traj[:, 0, :] = state0
    for n in range(N):
        x, y, th = traj[:, n, 0], traj[:, n, 1], traj[:, n, 2]
        v, om = ctrl[:, n, 0], ctrl[:, n, 1]
        traj[:, n + 1, 0] = x + v * np.cos(th) * dt
        traj[:, n + 1, 1] = y + v * np.sin(th) * dt
        traj[:, n + 1, 2] = th + om * dt
    return traj


def sample_controls(cfg: MpcConfig, rng: np.random.Generator) -> np.ndarray:
    """(K, N, 2) control samples, curvature-limited so a car could execute them."""
    shape = (cfg.K, 1) if cfg.constant_controls else (cfg.K, cfg.N)
    v = rng.uniform(0.0, cfg.v_max, shape)
    om = rng.uniform(-cfg.omega_max, cfg.omega_max, shape)
    # |omega| <= v * max_curvature: at a standstill a car cannot rotate, and at speed
    # its turn radius is bounded. Clip rather than reject, to keep all K usable.
    om = np.clip(om, -v * cfg.max_curvature, v * cfg.max_curvature)
    ctrl = np.stack([v, om], axis=2).astype(np.float32)
    if cfg.constant_controls:
        ctrl = np.repeat(ctrl, cfg.N, axis=1)
    return ctrl


def plan_step(
    cfg: MpcConfig,
    pos: np.ndarray,                # (2,) planar metres
    yaw: float,                     # planar radians
    centroids: np.ndarray,          # (R, 2) planar metres, index-aligned with region_ids
    region_ids: list[int],
    start_id: int,
    target_id: int,
    hits_body: np.ndarray | None = None,
    occ: LocalOccGrid | None = None,
    rng: np.random.Generator | None = None,
    road: "RoadSurface | None" = None,
    via: "list[int] | None" = None,
) -> MpcResult:
    """One control decision.

    ``via`` lists regions the route legitimately passes THROUGH, which must therefore not
    be forbidden. It matters when the target is more than one hop away — "keep going to
    the next junction" crosses a path region on the way, and forbidding it (-15) makes the
    crossing itself the most expensive option available. Under strong bearing guidance the
    vehicle bulldozes through anyway; once bearing is demoted to a weak prior it does not,
    and the step deadlocks, even with an undistorted map.

    ``start_id`` / ``target_id`` are region ids (as published by the GT clusterer and
    named in the brain plan), NOT indices — the caller should not have to know the
    ordering of ``centroids``. Everything not start or target is forbidden, which is
    what keeps the vehicle from cutting through a region the plan did not authorise.
    """
    rng = rng or np.random.default_rng()
    occ = occ if occ is not None else LocalOccGrid()
    idx = {rid: i for i, rid in enumerate(region_ids)}
    if start_id not in idx or target_id not in idx:
        raise KeyError(f"start {start_id} / target {target_id} not among the plan's "
                       f"regions {region_ids}")
    i_start, i_target = idx[start_id], idx[target_id]

    ctrl = sample_controls(cfg, rng)
    rollouts = unicycle_rollout(np.zeros(3), ctrl, cfg.dt)     # body frame
    traj = rollouts[:, 1:, :]

    occ.update(hits_body)
    _cl = occ.check_collisions(traj)
    _h = cfg.collision_horizon
    if _h and 0 < _h < _cl.shape[1]:
        _cl = _cl[:, :_h]
    collision = (_cl < cfg.safety_radius).any(axis=1)

    cy, sy = math.cos(yaw), math.sin(yaw)
    R_l2g = np.array([[cy, -sy], [sy, cy]])
    end_global = traj[:, -1, :2] @ R_l2g.T + pos

    # -- cluster term ---------------------------------------------------------
    use_region = cfg.guidance == "region" and road is not None
    if use_region:
        # Membership from the ground-truth waypoint table. Immune to any
        # error in the centroids.
        rid_at = road.region_of(end_global)
        lookup = {r: i for i, r in enumerate(region_ids)}
        nearest = np.array([lookup.get(int(r), i_start) for r in rid_at])
    else:
        d_c = np.linalg.norm(end_global[:, None, :] - centroids[None, :, :], axis=2)
        nearest = np.argmin(d_c, axis=1)
    forbidden_mask = np.ones(len(region_ids), dtype=bool)
    allowed = [i_start, i_target] + [idx[r] for r in (via or []) if r in idx]
    forbidden_mask[allowed] = False
    cluster_parts = (cfg.w_target * (nearest == i_target)
                     + cfg.w_start * (nearest == i_start)
                     + cfg.w_forbidden * forbidden_mask[nearest]).astype(float)
    if cfg.pure_pursuit:
        # The baseline has no cluster term at all: it steers at the goal point and has no
        # way to notice it is in the wrong region. Everything else -- horizon, dynamics,
        # obstacle rejection -- is held identical, so the comparison isolates exactly one
        # thing: whether membership is used.
        cluster_parts = np.zeros_like(cluster_parts)

    # -- bearing term: the primary steering signal ---------------------------
    aim = centroids[i_target]
    if use_region and cfg.aim_source == "carrot":
        # A REACHABLE AIM POINT ON THE ROUTE, which is what turning out of a junction needs.
        #
        # The failure this fixes: inside a junction the target region's nearest
        # waypoint is ~31 m away against an 8 m horizon, so NO rollout can end in it and
        # `cluster_score` sits pinned at 1.01 -- the value for ending where you started.
        # The occupancy term contributes no gradient at all, and only the bearing prior
        # (x0.25) and the progress term are left. Going straight that is enough; turning it
        # is not.
        #
        # So aim at the nearest drivable point that is (a) far enough to steer toward and
        # (b) actually closer to the goal than we are -- a carrot inside the horizon, taken
        # from the regions the route passes through. Junction regions are densely sampled
        # (region 57 alone has 370 waypoints), so this puts the aim ON the drivable surface
        # THROUGH the junction rather than across whatever sits between here and the exit.
        cand = [i_start, i_target] + [idx[r] for r in (via or []) if r in idx]
        pool = [road.nearest_point_of(region_ids[i]) for i in cand]
        pool = [q for q in pool if q is not None and len(q)]
        goal_pts = road.nearest_point_of(target_id)
        if pool and goal_pts is not None and len(goal_pts):
            allpts = np.vstack(pool)
            d_self = np.linalg.norm(allpts - pos, axis=1)
            # progress measured against the nearest point of the TARGET REGION, which is
            # ground-truth geometry -- not the centroid, which a distorted map moves.
            g = goal_pts[int(np.argmin(((goal_pts - pos) ** 2).sum(axis=1)))]
            d_goal = np.linalg.norm(allpts - g, axis=1)
            here = float(np.linalg.norm(pos - g))
            ok = (d_self >= cfg.lookahead_m) & (d_goal < here)
            if not ok.any():                       # nothing far enough: take any progress
                ok = d_goal < here
            if ok.any():
                aim = allpts[ok][int(np.argmin(d_self[ok]))]
    elif use_region and cfg.aim_source == "region":
        pts = road.nearest_point_of(target_id)
        if pts is not None and len(pts):
            # Aim at the nearest bit of the target REGION rather than its centroid, so a
            # displaced centroid does not displace the direction of travel either.
            #
            # With a LOOK-AHEAD, though: the strictly nearest point is the region's near
            # EDGE, so the vehicle drives to the boundary and stops there. An ordinal
            # mission has to traverse a region for its cue to go false and make the next
            # sighting distinct. Preferring a point at least `lookahead_m`
            # away aims through the region instead of at it — the pure-pursuit idea.
            d2 = ((pts - pos) ** 2).sum(axis=1)
            far = d2 >= cfg.lookahead_m ** 2
            aim = (pts[far][int(np.argmin(d2[far]))] if far.any()
                   else pts[int(np.argmax(d2))])
    # Global -> body is a rotation by MINUS yaw. For row vectors that is `delta @ R_l2g`
    # (equivalently R_l2g.T @ delta), NOT `delta @ R_l2g.T`.
    #
    # Using `delta @ [[cy, sy], [-sy, cy]]` instead rotates by *plus* yaw, reflecting the
    # bearing about the heading axis: a target dead ahead comes out dead BEHIND
    # (`bear_local` = pi instead of 0) and the winning rollout is full-lock steering.
    tgt_local = (aim - pos) @ R_l2g
    bear_local = math.atan2(tgt_local[1], tgt_local[0])
    cdir = tgt_local / (np.linalg.norm(tgt_local) + 1e-6)
    end_th = traj[:, -1, 2]
    bw = cfg.bearing_weight_region if use_region else 1.0
    bearing_parts = bw * (cfg.w_bearing_cos * np.cos(end_th - bear_local)
                          + cfg.w_bearing_dot * (np.cos(end_th) * cdir[0]
                                                 + np.sin(end_th) * cdir[1]))

    scores = cluster_parts + bearing_parts

    # -- progress along the start->target axis -------------------------------
    axis = aim - pos if use_region else centroids[i_target] - centroids[i_start]
    axis = axis / (np.linalg.norm(axis) + 1e-6)
    scores += cfg.w_progress * np.clip((end_global - pos) @ axis, 0, None)

    # -- steering-effort penalty ---------------------------------------------
    scores -= cfg.w_smooth * np.abs(ctrl[:, 0, 1])

    # -- stay on the road ----------------------------------------------------
    offroad_frac = 0.0
    offroad_rejected = 0
    if road is not None:
        excess = road.excess(traj[:, :, :2] @ R_l2g.T + pos, cfg.road_half_width)
        leaves = (excess > 0).any(axis=1)
        offroad_frac = float(leaves.mean())
        scores -= cfg.w_offroad * excess.mean(axis=1)
        if cfg.reject_offroad and not leaves.all():
            # Not merely worse — not an option. Skipped when it would reject everything,
            # which is a statement about sparse sampling rather than about the world.
            scores[leaves] = -np.inf
            offroad_rejected = int(leaves.sum())

    # -- plan-level prohibition: `forbid_clusters` ---------------------------
    # Distinct from `w_forbidden` above, which is a soft penalty on leaving the current
    # step's route. This is the mission saying "never go there", so it removes the option
    # rather than pricing it -- otherwise the bearing term outvotes it whenever the
    # forbidden region happens to lie toward the goal, which is exactly when it matters.
    forbidden_rejected = 0
    if cfg.hard_forbid:
        # WHOLE ROLLOUT, not just the endpoint. "Never enter X" was implemented as "never
        # END in X" would let a rollout cross the region and stop outside it -- at N=8,
        # dt=0.2 that is 1.6 s of driving through somewhere the mission forbids. This
        # matches `reject_offroad` above, which tests every point with `.any(axis=1)`.
        if road is not None and use_region:
            pts = traj[:, :, :2] @ R_l2g.T + pos              # (K, N, 2) global
            rid_all = road.region_of(pts.reshape(-1, 2)).reshape(pts.shape[:2])
            hard = np.isin(rid_all, np.fromiter(cfg.hard_forbid, dtype=int)).any(axis=1)
        else:
            # No waypoint table: fall back to the endpoint test, which is all the
            # centroid-Voronoi path can answer.
            hard = np.array([region_ids[i] in cfg.hard_forbid for i in nearest], dtype=bool)
        if hard.any() and not hard.all():
            scores[hard] = -np.inf
            forbidden_rejected = int(hard.sum())
        elif hard.all():
            # Every sample enters it. Do NOT freeze: the vehicle would sit still while the
            # step times out, which reads as a planner bug. Take the least-bad option and
            # let ConstraintMonitor report the violation -- visible beats silent.
            scores -= cfg.w_offroad
            forbidden_rejected = -1

    # -- graded obstacle penalty, alongside the veto -------------------------
    # Orders the survivors by how close they get, so there is still a gradient when the
    # veto would otherwise leave nothing to choose between.
    if cfg.w_obstacle != 0.0 and _cl.size:
        _near = _cl.min(axis=1)
        scores = scores - cfg.w_obstacle * np.clip(
            (cfg.obstacle_influence_m - _near) / max(cfg.obstacle_influence_m, 1e-9),
            0.0, 1.0)

    scores[collision] = -np.inf
    edt_blocked = float(collision.sum()) / cfg.K

    recovery = False
    if not np.isfinite(scores).any():
        # Everything is blocked. Creep forward and turn toward the target rather than
        # freezing: a stopped vehicle never clears the obstacle that stopped it, and in
        # a synchronous sim it will sit there until the run times out.
        recovery = True
        v = min(0.5, cfg.v_max * 0.1)
        omega = float(np.clip(bear_local, -cfg.omega_max, cfg.omega_max)) * 0.8
        best_k, best_cp, best_bp = 0, 0.0, 0.0
    else:
        best_k = int(np.argmax(scores))
        v = float(ctrl[best_k, 0, 0])
        omega = float(ctrl[best_k, 0, 1])
        best_cp = float(cluster_parts[best_k])
        best_bp = float(bearing_parts[best_k])

    # -- CBF post-filter on the chosen control -------------------------------
    d_min = math.inf
    if hits_body is not None and len(hits_body) > 0:
        hb = np.asarray(hits_body, dtype=float)
        dists = np.linalg.norm(hb, axis=1)
        d_min = float(dists.min())
        near = hb[int(np.argmin(dists))]
        ang = math.atan2(near[1], near[0])
        cos_a, sin_a = math.cos(ang), math.sin(ang)
        h = d_min - cfg.d_safe
        if cos_a > 1e-3 and v * cos_a > cfg.cbf_alpha * h:
            v = max(0.0, cfg.cbf_alpha * h / cos_a)
        if d_min < 2.0 * cfg.d_safe and cos_a > 0.3:
            omega = float(np.clip(
                omega - 2.0 * cfg.cbf_alpha * sin_a * (1.0 - h / cfg.d_safe),
                -cfg.omega_max, cfg.omega_max))

    # -- candidate sample for the viewer ------------------------------------
    # Stratified, not uniform: rejected rollouts are usually the minority, and a
    # uniform sample of 24 out of 500 would often contain none of them — which is
    # exactly the case the picture needs to show.
    sampled = sampled_rej = None
    if cfg.debug_rollouts > 0:
        blocked = ~np.isfinite(scores)
        idx_bad = np.flatnonzero(blocked)
        idx_ok = np.flatnonzero(~blocked)
        m = int(cfg.debug_rollouts)
        take_bad = min(len(idx_bad), m // 2)
        take_ok = min(len(idx_ok), m - take_bad)
        pick = np.concatenate([
            idx_bad[np.linspace(0, len(idx_bad) - 1, take_bad).astype(int)]
            if take_bad else np.empty(0, dtype=int),
            idx_ok[np.linspace(0, len(idx_ok) - 1, take_ok).astype(int)]
            if take_ok else np.empty(0, dtype=int),
        ])
        if best_k not in pick:            # the winner must always be in the sample
            pick = np.append(pick, best_k)
        sampled = rollouts[pick].astype(np.float32).copy()
        sampled_rej = blocked[pick].copy()

    finite = scores[np.isfinite(scores)]
    top = (np.sort(finite)[::-1][:10] if finite.size
           else np.array([], dtype=np.float32))

    return MpcResult(
        v=v, omega=omega,
        best_rollout=rollouts[best_k].copy(),
        cluster_score=best_cp, bearing_score=best_bp,
        edt_blocked_frac=edt_blocked, offroad_frac=offroad_frac,
        offroad_rejected=offroad_rejected, forbidden_rejected=forbidden_rejected,
        recovery=recovery,
        d_min=d_min, n_hits=0 if hits_body is None else len(hits_body),
        nearest_region=int(region_ids[int(nearest[best_k])]),
        top_scores=top,
        sampled_rollouts=sampled, sampled_rejected=sampled_rej)
