"""Terrain-scored sampling MPC: the controller the hardware path needs.

SEPARATE FILE ON PURPOSE. `sampling_mpc.py` is the node's default controller
(run_phase_a.sh selects this one with CONTROLLER=terrain). Nothing here
touches it, so a mistake in this file cannot break it, and the two can be run side by
side on the same missions for comparison.

WHAT IS DIFFERENT, AND WHY
--------------------------
1. NO REGION OCCUPANCY IN THE SCORE. `sampling_mpc.plan_step` ranks rollouts by
   `region_of(endpoint)` -- which region each candidate would END in. That asks for a
   label on a HYPOTHETICAL future position, i.e. region labels over SPACE. CARLA answers
   it from the waypoint table; on hardware nothing does. `/predicted_cluster` publishes
   one Int16 for the ego frame, and junction is not reliably perceivable per frame from
   any of the available sensors (ZED metric-depth BEV, monocular BEV, forward SegFormer).

   So membership is left to the EXECUTOR, which only ever tests the robot's own position,
   for step completion. The controller here needs traversable surface in its own frame,
   obstacle returns, and a direction -- none of which requires attributing a remote patch
   of the world to a named region.

2. ARC-LENGTH ROLLOUTS. Under the time parameterisation a slow candidate covers less
   ground than a fast one, so the K rollouts span different distances and the score
   compares trajectories of unequal extent -- a slow arc scores "safe" largely by going
   nowhere. At fixed arc length every candidate reaches the same distance and differs
   only in shape.

3. CLOSED FORM, NOT A SPLINE. With one constant `(v, omega)` the exact solution IS a
   circular arc of curvature `omega / v`. `unicycle_rollout` forward-Eulers that arc and
   carries an O(dt) error; `arc_rollout` returns the arc. Nothing is fitted. A spline
   would only be needed if the control varied across the horizon, which it does not.

4. ORDERED FALLBACK. When a hard test would eliminate every candidate it is relaxed
   rather than freezing -- that case is a statement about sampling sparsity, not about
   the world. `sampling_mpc` relaxes its two tests independently; here terrain relaxes
   BEFORE obstacles, because crossing rough ground is recoverable and a collision is not.

5. CURVATURE IS THE SEARCH VARIABLE, AND THE SEARCH IS DETERMINISTIC. Once arcs are
   fixed-length, reach no longer depends on speed: what the score ranks is SHAPE, i.e.
   curvature. `sampling_mpc.sample_controls` draws v and omega independently and then
   clips |omega| <= v * max_curvature, so k = omega / v comes out with a speed-dependent
   distribution piled at zero -- every low-v draw returns straight. That samples densely
   where the score cannot tell candidates apart and sparsely among the turns.

   So curvature is swept directly, on an evenly spaced deterministic fan. Beyond better
   coverage this removes the RNG from the control path. Seeding does NOT make a driven run
   reproducible, because the rng is one stream consumed once per control tick and the
   number and timing of ticks is set by ROS/CARLA scheduling, so the same plan, seed and
   spawn pose can diverge. A deterministic fan removes that variance source instead of
   seeding it.

6. SPEED IS A RULE, NOT A SEARCH DIMENSION. With curvature sampled directly and the arc
   length fixed, v does not change the scored geometry at all. Scoring it would be
   searching a dimension the objective cannot see. It is set by `speed_for` instead.

7. SMOOTHNESS PENALISES CHANGE, NOT MAGNITUDE. `- w_s * |omega|` taxes a steady turn every
   tick although nothing about it is jerky, and nothing couples consecutive ticks at all.
   The term is now |k - k_prev| against the previously commanded curvature, which is what
   actually resists chatter between ticks.

WHAT IS NOT IMPLEMENTED
-----------------------
The terrain source. `terrain_class` is injected and there is deliberately no default.
A default of "all zeros" would be a silently-inert setting: a terrain term with no source
contributes nothing while the run still reports itself as terrain-scored.
`TerrainMpcConfig` raises instead.

In CARLA `no_rendering_mode`, camera sensors do not produce frames (LiDAR raycast does --
it is a physics query, not a render). `waypoint_terrain_source` below is the CARLA
ground-truth stand-in, exactly as the cluster and cue oracles are; the camera-based
source lives in `carla_gt_bridge.camera_terrain`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np

from .sampling_mpc import LocalOccGrid, MpcResult

__all__ = ["TerrainMpcConfig", "arc_rollout_k", "arc_rollout", "curvature_fan",
           "arc_weights", "terrain_soft_costs", "speed_for", "plan_step_terrain",
           "waypoint_terrain_source"]


# --------------------------------------------------------------------------- #
# Rollouts                                                                     #
# --------------------------------------------------------------------------- #

def arc_rollout_k(state0: np.ndarray, kappa: np.ndarray, arc_len: float,
                  n: int) -> np.ndarray:
    """Closed-form constant-curvature arcs, sampled by ARC LENGTH. (K,) -> (K, n+1, 3).

    Curvature is the whole parameterisation: an arc's geometry depends on ``kappa`` and
    ``arc_len`` and on nothing else. Speed does not appear, which is the point -- it does
    not change the shape being scored.

    The ``k -> 0`` branch is not an optimisation. The closed form divides by ``k``, and a
    straight candidate is not a rare edge case: an evenly spaced fan crosses zero exactly.
    """
    k = np.asarray(kappa, dtype=float).ravel()
    x0, y0, th0 = float(state0[0]), float(state0[1]), float(state0[2])
    s_grid = np.linspace(0.0, float(arc_len), int(n) + 1)[None, :]       # (1, n+1)

    th = th0 + k[:, None] * s_grid                                       # (K, n+1)
    x = x0 + s_grid * math.cos(th0) + np.zeros_like(th)
    y = y0 + s_grid * math.sin(th0) + np.zeros_like(th)

    curved = np.abs(k) > 1e-9
    if curved.any():
        kc = k[curved][:, None]
        x[curved] = x0 + (np.sin(th[curved]) - math.sin(th0)) / kc
        y[curved] = y0 - (np.cos(th[curved]) - math.cos(th0)) / kc
    return np.stack([x, y, th], axis=2)


def arc_rollout(state0: np.ndarray, ctrl: np.ndarray, arc_len: float,
                n: int) -> np.ndarray:
    """``arc_rollout_k`` from (K, N, 2) [v, omega] controls, for comparison against the
    time-parameterised integrator. Only ``ctrl[:, 0, :]`` is read.

    A standstill candidate (v == 0) has no defined curvature and travels nowhere however
    long the arc; it must not inherit the straight branch, which would teleport it
    ``arc_len`` metres forward.
    """
    v = np.asarray(ctrl[:, 0, 0], dtype=float)
    om = np.asarray(ctrl[:, 0, 1], dtype=float)
    moving = v > 1e-6
    k = np.zeros_like(v)
    k[moving] = om[moving] / v[moving]
    out = arc_rollout_k(state0, k, arc_len, n)
    if (~moving).any():
        out[~moving, :, 0] = state0[0]
        out[~moving, :, 1] = state0[1]
        out[~moving, :, 2] = state0[2]
    return out


def arc_weights(arc_len: float, n: int, decay_m: float | None = None,
                s_min: float = 0.0) -> np.ndarray:
    """Per-sample weights along an arc, for the terrain cost. ``(n,)``, summing to 1.

    ONE VECTOR FOR ALL K, AND THAT IS WHY THIS IS SAFE. Every arc is sampled at the same
    ``s_grid`` (`arc_rollout_k`), so a distance weighting is a constant vector and the
    weighted cost is a dot product. Nothing is normalised per arc, so no candidate can
    improve its score by being shorter or by leaving the sensor's view early -- which a
    per-arc renormalisation would reward.

    ``decay_m is None`` reproduces the unweighted mean EXACTLY. That is the default, so the
    weighting is strictly opt-in and existing configurations behave unchanged.

    THE DECAY STARTS AT ``s_min``, NOT AT THE VEHICLE, and that is the whole subtlety. The
    point of weighting by distance is that a hazard about to be driven over matters more than
    one 10 m away. But the camera this feeds cannot see the ground nearer than about 3.5 m
    (`camera_terrain.CameraModel.min_visible_range_m`), so samples below that are UNOBSERVED
    whatever the world contains. Decaying from s = 0 would therefore put most of the weight on
    samples that are a constant `unknown_cost` for every candidate -- the term would look
    tuned and be inert. Passing the camera's own minimum range puts the weight where there is
    actually a road-vs-grass distinction to grade.
    """
    # Drop the origin, exactly as the scorer does.
    s = np.linspace(0.0, float(arc_len), int(n) + 1)[1:]
    if decay_m is None:
        return np.full(s.shape, 1.0 / s.size)
    if decay_m <= 0:
        raise ValueError(f"decay_m must be positive or None, got {decay_m}")
    w = np.where(s >= s_min, np.exp(-(s - s_min) / float(decay_m)), 0.0)
    total = w.sum()
    if total <= 0.0:
        # Every sample fell below s_min: the arc ends before the sensor's view begins, so the
        # terrain term would be identically zero while still being reported. That is the
        # failure class this file's docstring is about -- refuse it rather than emit it.
        raise ValueError(
            f"arc_weights: no sample of a {arc_len} m arc reaches s_min={s_min} m, so the "
            f"terrain term would be identically zero. Lengthen the arc or lower s_min.")
    return w / total


# --------------------------------------------------------------------------- #
# The search: a deterministic fan, and a speed rule                            #
# --------------------------------------------------------------------------- #

def reachable_index(blocked: np.ndarray) -> np.ndarray:
    """(K, n) per-point blocked mask -> (K,) index of the last REACHABLE point in `traj`.

    WHY SCORING NEEDS THIS. The progress term reads the arc's ENDPOINT, so an arc that
    drives through a wall at 2 m is still credited with where it would have been at 10 m.
    Example, obstacle dead ahead at 2 m, without truncation:

        straight arc         progress = +10.00   closest approach = 0.12 m   <- wins
        best-clearance arc   progress =  -1.00   closest approach = 0.65 m

    Straight wins by being credited with distance it could never travel, so the controller
    drives into walls at <= 3 m and into dead ends. It is NOT fixable by weighting: the
    obstacle penalty is bounded by `w_obstacle` while progress runs to
    `w_progress * arc_len_m`, so even a very large w_obstacle still drives straight into
    a U-shaped dead end. The quantity itself is wrong, not its coefficient.

    Indexing: `blocked[k, j]` describes arc point j, which is `traj[k, j + 1]` -- index 0
    is the shared origin and is never scored. So a first block at point j leaves the arc
    reachable up to traj index j, and no block at all leaves it reachable to traj index n.
    """
    n = int(blocked.shape[1])
    return np.where(blocked.any(axis=1), blocked.argmax(axis=1), n)


def _arc_from(state0: np.ndarray, k: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Closed-form constant-curvature arc from a PER-CANDIDATE start state.

    ``arc_rollout_k`` takes one shared origin, which is all a single-segment fan needs.
    A second segment starts wherever the first one ended, so each candidate has its own
    start pose -- hence this vectorised form. (K, 3) x (K,) x (m,) -> (K, m, 3).
    """
    x0, y0, th0 = state0[:, 0:1], state0[:, 1:2], state0[:, 2:3]
    th = th0 + k[:, None] * s[None, :]
    straight_x = x0 + s[None, :] * np.cos(th0)
    straight_y = y0 + s[None, :] * np.sin(th0)
    curved = np.abs(k) > 1e-9
    kc = np.where(curved, k, 1.0)[:, None]          # 1.0 is a placeholder, masked below
    x = np.where(curved[:, None], x0 + (np.sin(th) - np.sin(th0)) / kc, straight_x)
    y = np.where(curved[:, None], y0 - (np.cos(th) - np.cos(th0)) / kc, straight_y)
    return np.stack([x, y, th], axis=2)


def two_segment_fan(kappa_max: float, per_seg: int) -> np.ndarray:
    """(K, 2) grid of (kappa_1, kappa_2). Deterministic, enumerated, NOT sampled.

    WHY TWO SEGMENTS. A single constant curvature has to commit: turn hard enough to miss
    an obstacle and you leave the corridor; stay inside it and you hit the obstacle.
    Example geometry from the 2D rig -- a 0.04 m obstacle 0.25 m ahead between walls at
    +-0.175 m: no one-segment arc clears it AND stays inside, at any fan resolution,
    because the shape needed is swerve-and-return and no circular arc is one. A
    two-segment grid contains such arcs. No obstacle weighting can fix this -- the answer
    is not in the one-segment candidate set.

    NOT A RANDOM WALK, which is the other way to get these shapes. This is an exhaustive
    grid over a 2D parameter space: the same 225 candidates every tick, no RNG in the
    control path. Per-step random sampling is what `sampling_mpc` REMOVED (its score
    depends only on the endpoint, so a wiggle arriving in the same place scores the same
    as a clean arc and the command returned is near-independent of the score), and it is
    an irreducible source of run-to-run divergence.

    THE COST: at 225 candidates each segment gets `per_seg` values where one segment had
    K, so curvature resolution per segment drops roughly 4x.
    """
    per = int(per_seg)
    if per < 3:
        raise ValueError(f"each segment needs at least 3 curvatures, got {per}")
    if per % 2 == 0:
        per += 1                                     # keep a straight candidate per segment
    g = np.linspace(-float(kappa_max), float(kappa_max), per)
    k1, k2 = np.meshgrid(g, g, indexing="ij")
    return np.stack([k1.ravel(), k2.ravel()], axis=1)


def arc_rollout_2seg(kappas: np.ndarray, arc_len: float, n: int) -> np.ndarray:
    """(K, 2) curvature pairs -> (K, n+1, 3). First half at kappa_1, second at kappa_2.

    The join is C0 and C1 continuous in position and heading (the second segment starts at
    the first's exact end pose); curvature steps at the midpoint, exactly as the
    single-segment fan steps it between ticks.
    """
    k = np.asarray(kappas, dtype=float)
    if k.ndim != 2 or k.shape[1] != 2:
        raise ValueError(f"kappas must be (K, 2), got {k.shape}")
    K = k.shape[0]
    n1 = int(n) // 2
    n2 = int(n) - n1
    half = float(arc_len) / 2.0
    seg1 = _arc_from(np.zeros((K, 3)), k[:, 0], np.linspace(0.0, half, n1 + 1))
    seg2 = _arc_from(seg1[:, -1, :], k[:, 1], np.linspace(0.0, half, n2 + 1)[1:])
    return np.concatenate([seg1, seg2], axis=1)


def corridor_from_hits(hits_body: np.ndarray | None, min_hits: int = 4
                       ) -> tuple[np.ndarray, float] | None:
    """Corridor direction and the robot's offset from its centreline, from LiDAR alone.

    WHY THIS CONTROLLER NEEDS IT. Without a corridor term this controller fails to pass a
    walled corridor on the 2D rig once the arc reach exceeds the slot geometry. The
    mechanism is arc termination -- in a slot narrower than the arc, every candidate ends
    on a wall whichever way it curves, truncation removes them all, and no gradient is
    left. A centring term supplies one that obstacle distance alone cannot: "stay in the
    middle" is defined even when every arc is blocked.

    NO MAP, AND NO HISTORY. Everything is in the BODY frame, where the robot is at the
    origin facing +x, so the 180-degree ambiguity in a PCA eigenvector resolves by a fixed
    convention -- take the perpendicular pointing LEFT (+y) -- rather than by consulting a
    map. (The 2D implementation this is ported from resolved the sign and axis against the
    true map and fell back to the true bridge centre when hits were sparse, which leaks
    undistorted ground truth into a term meant to be map-independent.)

    All returns are used; see the comment in the body for why no angular filter is
    applied. Returns None when there is nothing to see -- the caller then DROPS the term
    for this tick rather than inventing a centreline, which would make sparse stretches
    look like successful perception.

    Returns ``(perp_body, offset_m)``: the unit vector across the corridor, and how far
    the robot is from the centreline along it (signed).
    """
    if hits_body is None or len(hits_body) < min_hits:
        return None
    h = np.asarray(hits_body, dtype=float)
    # NO ANGULAR FILTER, and both obvious ones are wrong. The 2D implementation kept only
    # |x| < |y| "lateral" returns, which keeps points near x = 0 and collapses two long
    # walls into two blobs whose spread runs ACROSS the corridor -- PCA then returns the
    # along-corridor axis, off by a right angle (perp = [1, 0] for a corridor along +x).
    # Excluding a forward CONE instead fails the same way, because a wall 8 m ahead
    # subtends only 10 degrees and is dropped with the obstacle (about 45 deg off). Walls dominate by sheer count, so the raw PCA is already robust: with walls at
    # y = +1.5 / -2.5 it gives [0, 1] exactly, and stays at [-0.03, 1.00] even against a
    # 60-point obstacle dead ahead.
    pts = h
    _, evecs = np.linalg.eigh(np.cov(pts.T))
    perp = evecs[:, 0]                       # min-variance = across the corridor
    if perp[1] < 0:                          # fixed body-frame convention: point LEFT
        perp = -perp
    proj = pts @ perp
    med = float(np.median(proj))
    hi, lo = proj[proj > med], proj[proj < med]
    if not len(hi) or not len(lo):
        return None
    centre_proj = 0.5 * (float(hi.mean()) + float(lo.mean()))
    # The robot sits at the origin, so its offset from the centreline is -centre_proj.
    return perp, -centre_proj


def curvature_fan(kappa_max: float, k: int) -> np.ndarray:
    """``k`` evenly spaced curvatures in [-kappa_max, +kappa_max], symmetric about zero.

    ODD ``k`` ON PURPOSE, forced below. With an even count the fan straddles zero and
    there is no straight candidate at all -- the controller would be unable to propose
    driving straight ahead, which is the commonest correct answer.

    Deterministic: same fan every tick, so two runs of the same mission issue the same
    candidate set. See module docstring item 5.
    """
    k = int(k)
    if k < 3:
        raise ValueError(f"a fan needs at least 3 candidates, got {k}")
    if k % 2 == 0:
        k += 1
    return np.linspace(-float(kappa_max), float(kappa_max), k)


def speed_for(kappa: np.ndarray, cfg: "TerrainMpcConfig",
              clearance: float | None = None) -> np.ndarray:
    """Speed per candidate, by rule rather than by search.

    Two effects, both deliberately simple and neither tuned. Curvature taper: a tight arc
    is driven slower, with ``kappa_taper`` the curvature at which speed halves. Clearance
    cap: with the nearest obstacle at ``clearance`` metres, speed is capped so the vehicle
    can stop within what it can see, using ``a_max``.

    This is a placeholder rule. It is NOT inherited from `sampling_mpc`, whose speed comes
    out of the sampler.
    """
    k = np.abs(np.asarray(kappa, dtype=float))
    v = cfg.v_max / (1.0 + k / max(cfg.kappa_taper, 1e-9))
    if clearance is not None and np.isfinite(clearance):
        head = max(clearance - cfg.safety_radius, 0.0)
        v = np.minimum(v, math.sqrt(2.0 * cfg.a_max * head))
    return np.clip(v, 0.0, cfg.v_max)


def terrain_soft_costs(klass: np.ndarray, cost: np.ndarray, w_arc: np.ndarray,
                       unobserved_class: int | None, unknown_cost: float,
                       w_unknown: float = 0.5
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-arc graded terrain cost, and which arcs saw nothing. ``(K, n) -> (K,), (K,)``.

    EXTRACTED SO THERE IS ONE OF IT. Visualisation tools that draw a per-arc cost chart
    must call this rather than recompute the cost, or the chart can disagree with the cost
    the controller actually used to choose.
    """
    if unobserved_class is None:
        k0 = klass.shape[0]
        return cost @ w_arc, np.zeros(k0, dtype=bool), np.ones(k0)
    # UNKNOWN IS NOT THE SAME FACT AS BAD. A point is unobserved for a GEOMETRIC reason --
    # it left the lens's wedge -- and the arcs that leave it are the ones that turn hardest.
    # Charging them `unknown_cost` as terrain would tax turning and attribute that tax to
    # terrain.
    observed = klass != int(unobserved_class)
    wm = observed * w_arc[None, :]
    total = wm.sum(axis=1)
    blind = total <= 0.0
    # OBSERVED EVIDENCE, PLUS AN HONEST PRICE FOR WHAT WAS NOT SEEN.
    #
    # Dividing by the observed weight alone would renormalise an arc onto whatever fraction
    # of itself the camera happened to describe: a hard-turning arc seen over a third of its
    # length would score as confidently as a well-seen straight one, and the controller
    # would drift toward the fan extremes.
    #
    # Not charging unobserved samples at all lets an arc buy a clean score by leaving the
    # field of view. The unseen fraction therefore carries its own charge, so an arc is only
    # as good as the evidence for it, and a well-seen clean arc beats a half-seen one.
    seen = np.clip(total, 0.0, 1.0)
    # TWO QUANTITIES, KEPT APART. Adding "what the ground is" to "how much of the arc was
    # observed" in one mean lets the unseen part dominate, because `unknown_cost = 1.0` is
    # the cost of a WALL while road is 0.0 and sidewalk 0.25 -- candidates end up ranked
    # almost entirely by how much of each fell inside the lens.
    #
    # The terrain part is now an EXPECTATION over the evidence: the weighted mean across the
    # samples actually observed, so seeing half an arc says the same about that arc as seeing
    # all of it. Ignorance is priced by its own weight, `w_unknown`.
    soft = (cost * wm).sum(axis=1) / np.where(blind, 1.0, seen)
    soft = soft + float(w_unknown) * (1.0 - seen)
    # THE ONE CASE THAT CANNOT BE DROPPED. An arc with no observed sample has no evidence at
    # all, and scoring it 0.0 would rank it EQUAL to a verified-clean road -- which makes
    # "never cross the grass" satisfiable by turning into ground the camera never saw, a
    # prohibition that silently does not bind. It takes `unknown_cost` instead.
    #
    # That is also what keeps a DEAD camera harmless rather than catastrophic: every arc is
    # blind at once, so every arc takes the same charge, the difference shifts all K equally,
    # the argmax is unchanged, and the controller degrades to bearing-only. With
    # `unobserved_class` refused entry to `forbid_classes`, nothing is vetoed either.
    return np.where(blind, float(unknown_cost), soft), blind, seen


# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #

@dataclass
class TerrainMpcConfig:
    """Knobs for the terrain-scored controller.

    Weights carry NO inherited values. The `sampling_mpc` numbers (`w_offroad=12.0`,
    `w_forbidden=-15.0`, `bearing_weight_region=0.25`) were tuned against a score that
    included a region-occupancy term; with that term gone their balance means nothing.
    They are placeholders until this controller is tuned on its own.
    """

    K: int = 65                     # fan size; forced odd so a straight candidate exists
    n: int = 16                     # samples along each arc
    #: How far each candidate reaches, in metres. 6.28 = (pi/2) / kappa_max, so the
    #: tightest arc turns exactly 90 deg. A 10.0 m arc at kappa_max 0.25 wraps 143 deg and
    #: costs the tightest candidates 4x their forward progress -- see `max_turn_rad`. The
    #: shorter arc also makes driven runs markedly less variable.
    arc_len_m: float = 6.28
    #: 1/m, the tightest arc the platform will propose. 0.25 (R_min 4.0 m) matches the
    #: launch configuration and `sampling_mpc.max_curvature`.
    kappa_max: float = 0.25

    v_max: float = 5.0              # m/s, speed cap
    kappa_taper: float = 0.15       # 1/m at which the speed rule halves v_max
    a_max: float = 2.0              # m/s^2, used only by the clearance cap
    safety_radius: float = 1.5      # m, used by `speed_for`'s clearance cap

    #: HARD obstacle veto radius. An arc passing closer than this to a return is removed.
    #:
    #: DELIBERATELY MUCH SMALLER THAN `safety_radius`. Vetoing at 1.5 m deadlocks the
    #: controller inside junctions: traffic-light poles sit 1.7-2.4 m from the driving
    #: line, and at `kappa_max` 0.25 the tightest arc has a 4 m radius, so NO candidate can
    #: steer around something that close. Every arc is vetoed, the relax-everything
    #: fallback engages, and the vehicle circles instead of progressing.
    #:
    #: A veto is the right tool only where a legal alternative exists. At 1.0 m it removes
    #: arcs that are genuinely about to hit something while leaving the near-but-passable
    #: case to the graded penalty below, which can still express a preference.
    obstacle_hard_m: float = 1.0
    #: Distance at which the SOFT obstacle penalty falls to zero. Must exceed the turning
    #: radius (1 / kappa_max = 4 m at 0.25), or the penalty carries no gradient in exactly
    #: the band where steering is the only remaining option.
    obstacle_influence_m: float = 4.0
    #: Weight on the graded penalty. Closest approach along the arc, ramped linearly from
    #: 0 at `obstacle_influence_m` to 1 at contact.
    #:
    #: SOFT RATHER THAN A SECOND VETO, which is the whole point: a penalty orders the
    #: candidates even when every one of them is bad, so there is always a least-bad arc
    #: to drive and a gradient pointing away from the obstacle. A veto that fires on all K
    #: destroys that ordering and hands over to a fallback that knows nothing about where
    #: the obstacle is. `sampling_mpc` reaches the same conclusion by a different route --
    #: `w_offroad` is a graded penalty for exactly this reason, and its hard twin
    #: `reject_offroad` is off in `mission.launch.py` because it makes junction turns
    #: crawl.
    w_obstacle: float = 3.0

    #: Per-point terrain CLASS: a callable taking (K, n, 2) body-frame points and
    #: returning (K, n) integer class ids. Required; see the module docstring.
    terrain_class: Callable[[np.ndarray], np.ndarray] | None = None
    #: class id -> cost. A class absent from the table costs `unknown_cost`.
    terrain_costs: Mapping[int, float] = field(default_factory=dict)
    #: Cost for a class the table does not name. Non-zero on purpose: an unrecognised
    #: surface is not evidence of a good one.
    unknown_cost: float = 1.0
    #: Class id meaning "the source could not describe this point" -- out of frame, occluded,
    #: behind the camera, or the frame was too old. ``None`` means the source always has an
    #: answer, which is true of `waypoint_terrain_source` and is the DEFAULT so that arm is
    #: untouched. When set, those samples are DROPPED from the graded cost rather than priced;
    #: see `plan_step_terrain`.
    unobserved_class: int | None = None
    #: Weight on NOT HAVING LOOKED, applied to the unobserved weight fraction. Separate
    #: from `unknown_cost` (which prices an arc with no evidence at all) so ignorance and
    #: terrain can be read and tuned apart.
    #:
    #: Chosen by a sweep over captured Town05 poses: 0.25 is the smallest value at which
    #: the chosen arc is fully observed both with and without a ban; larger values buy no
    #: coverage and halve the terrain share of the cost with each doubling. It cannot be 0,
    #: because `min_adjudicable_frac` applies only when a prohibition is active -- with no
    #: ban this is the only thing discouraging arcs the camera cannot see.
    w_unknown: float = 0.25
    #: With a forbid set active, the fraction of an arc that must be OBSERVED before "no
    #: forbidden sample was seen" counts as evidence of legality.
    #:
    #: THE ESCAPE THIS CLOSES. With a ban active, the veto can form a BAND: moderately
    #: turning arcs are vetoed while arcs turning HARDER toward the forbidden surface are
    #: not, because they leave the lens before reaching it and UNOBSERVED can never be
    #: forbidden. Without this threshold the prohibition is escapable by steering where
    #: the camera cannot adjudicate, with only a soft cost objecting.
    #:
    #: Applied ONLY when `forbid_classes` is non-empty, and the ordered fallback still
    #: prevents a deadlock when nothing qualifies.
    #:
    #: 0.50 is the knee of a sweep over captured Town05 poses: it removes thinly certified
    #: arcs entirely, while higher values only veto more arcs. On a narrower lens, watch
    #: for the ban going inert (no well-observed legal arc remaining).
    min_adjudicable_frac: float = 0.5
    #: Classes the mission forbids, resolved from `phi_avoid`. HARD.
    forbid_classes: frozenset = frozenset()

    #: Length scale of the distance weighting on the terrain cost, in metres. ``None`` keeps
    #: the unweighted mean (the default).
    terrain_decay_m: float | None = None
    #: Arc length at which that weighting starts. Pass the camera's
    #: `min_visible_range_m()`; see `arc_weights` for why zero is the wrong answer.
    terrain_s_min_m: float = 0.0

    #: 1 = the single constant-curvature fan (default). 2 = a (kappa_1, kappa_2) grid
    #: whose arcs can swerve and RETURN, needed for centred-obstacle geometries where a
    #: circular arc cannot thread the gap at any fan density or obstacle weighting.
    #: `K` is then the TOTAL candidate count; each segment gets round(sqrt(K)).
    segments: int = 1

    #: Weight on staying mid-corridor, from LiDAR only (`corridor_from_hits`).
    #: DEFAULT 0.0 = OFF, so this is additive. Without it this controller cannot pass a
    #: narrow walled corridor (see `corridor_from_hits`).
    #: Cap on the heading change a single arc may make, radians. Default pi/2; a config
    #: whose arcs exceed it raises. None disables the check.
    #:
    #: WHY. `arc_len_m` and `kappa_max` are independent knobs whose PRODUCT is the turn
    #: angle, and nothing else relates them. At arc_len 10 m, kappa_max 0.25 the tightest
    #: arc turns 143 deg: 24 of 65 candidates
    #: exceed 90 deg, and the tightest reaches only +2.39 m of forward progress against
    #: +10.00 m for straight. That 4x penalty on turning is an artifact of the arc
    #: WRAPPING, not a property of the manoeuvre, and it biases every scored comparison
    #: between turning and going straight.
    #:
    #: FIXED BY SHORTENING THE ARC, NOT BY CAPPING IT PER CANDIDATE. Capping each arc's
    #: length individually would break the property item 2 of the module docstring exists
    #: for -- every candidate reaching the same distance, so the score ranks SHAPE rather
    #: than extent. Shortening `arc_len_m` to `max_turn_rad / kappa_max` keeps equal
    #: extent, caps the turn, and preserves turning authority: at kappa_max 0.25 that is
    #: 6.28 m with R_min unchanged at 4.0 m. It costs lookahead, 10 m -> 6.28 m, which is
    #: still well beyond the ~3 m at which avoidance has to begin and the ~1 m executed
    #: per tick.
    #:
    #: Raises rather than clamping: a silent clamp is how a config field ends up meaning
    #: something other than what it says.
    max_turn_rad: float | None = math.pi / 2

    w_corridor: float = 0.0
    #: Width of the Gaussian on lateral offset, in metres. Should be a fraction of the
    #: corridor half-width: the 2D rig uses 0.10 m in a 0.40 m slot (25%), so a 14 m road
    #: wants roughly 2 m. Not tuned.
    corridor_sigma_m: float = 2.0
    #: Lateral returns needed before the estimate is trusted at all.
    corridor_min_hits: int = 4

    w_bearing: float = 1.0
    w_progress: float = 1.0
    w_terrain: float = 1.0
    w_smooth: float = 0.1           # on |k - k_prev|, not on |k|

    def __post_init__(self) -> None:
        """Reject a configuration that would run and measure nothing.

        A config field with more than one legal value must be
        asserted FUNCTIONAL, not merely set.
        """
        if self.terrain_class is None:
            raise ValueError(
                "TerrainMpcConfig needs `terrain_class`. Without it the terrain term is "
                "identically zero and the controller silently degrades to bearing-only "
                "while still reporting terrain-scored results.")
        if (self.unobserved_class is not None
                and self.unobserved_class in self.forbid_classes):
            # A veto that can fire on EVERY candidate is not a constraint, it is a deadlock
            # (see `obstacle_hard_m` for the obstacle version of this). "Unobserved" is
            # strictly worse, because a dead camera makes every point unobserved at once, so
            # the deadlock would be TRIGGERED by the sensor dropping out.
            raise ValueError(
                f"unobserved_class {self.unobserved_class} is in forbid_classes. A stale or "
                f"dead camera would then veto all K candidates and hand over to the "
                f"relax-everything fallback. Price it with `unknown_cost` instead.")
        if self.max_turn_rad is not None:
            turn = self.arc_len_m * self.kappa_max
            if turn > self.max_turn_rad + 1e-9:
                raise ValueError(
                    f"arc_len_m {self.arc_len_m} x kappa_max {self.kappa_max} = "
                    f"{math.degrees(turn):.0f} deg of turn, over the "
                    f"{math.degrees(self.max_turn_rad):.0f} deg cap. Set arc_len_m to "
                    f"{self.max_turn_rad / self.kappa_max:.2f} (keeps kappa_max and so "
                    f"the turning radius), or kappa_max to "
                    f"{self.max_turn_rad / self.arc_len_m:.3f} (keeps the lookahead).")
        if self.arc_len_m <= 0:
            raise ValueError(f"arc_len_m must be positive, got {self.arc_len_m}")
        if self.kappa_max <= 0:
            raise ValueError(f"kappa_max must be positive, got {self.kappa_max}")


# --------------------------------------------------------------------------- #
# The CARLA stand-in for a terrain source                                      #
# --------------------------------------------------------------------------- #

def waypoint_terrain_source(road, R_l2g: np.ndarray, pos: np.ndarray,
                            class_of_region: Mapping[int, int],
                            half_width: float = 7.0,
                            off_surface_class: int = -1) -> Callable:
    """Terrain classes from the waypoint table: a ground-truth check of where a point lands.

    The CARLA instantiation of an interface a camera fills on hardware, exactly as
    `gt_cluster_node` stands in for the LiDAR classifier and `gt_cue_node` for the VLM. It
    is not a perception result and must not be reported as one.

    BOTH QUERIES ARE NEEDED. ``RoadSurface.region_of`` is a NEAREST-WAYPOINT lookup: it
    returns the closest region for any point in the plane, including one inside a
    building, so on its own it can never answer "off the surface" -- every rollout would
    read as on-road and the terrain term would be uniformly zero. ``excess`` supplies the
    missing half: metres a point lies BEYOND the drivable surface, zero when on it.
    """
    def _classify(pts_body: np.ndarray) -> np.ndarray:
        flat = pts_body.reshape(-1, 2) @ R_l2g.T + pos
        out = np.full(flat.shape[0], off_surface_class, dtype=int)
        on = road.excess(flat, half_width) <= 0.0
        if on.any():
            rid = road.region_of(flat[on])
            sub = np.full(rid.shape, off_surface_class, dtype=int)
            for r, c in class_of_region.items():
                sub[rid == r] = c
            out[on] = sub
        return out.reshape(pts_body.shape[:2])
    return _classify


# --------------------------------------------------------------------------- #
# The control decision                                                         #
# --------------------------------------------------------------------------- #

def plan_step_terrain(
    cfg: TerrainMpcConfig,
    bearing_ref: float,                     # radians in the BODY frame; 0 = straight ahead
    kappa_prev: float = 0.0,                # last commanded curvature, for the smooth term
    hits_body: np.ndarray | None = None,
    occ: LocalOccGrid | None = None,
) -> MpcResult:
    """One control decision, scored without any region label over space.

    ``bearing_ref`` is the only map-derived quantity, and it is a DIRECTION, not a
    position: the topological map's bearing toward the next region, in the body frame. A
    rotated or rescaled map moves it; nothing else here consults the map.

    There is no ``rng`` parameter. That is the point of item 5 -- the candidate set is the
    same every tick, so the control path contributes no run-to-run variance.
    """
    occ = occ if occ is not None else LocalOccGrid()
    if hits_body is not None:
        occ.update(hits_body)

    if cfg.segments == 2:
        # Each segment gets sqrt(K) values so the TOTAL candidate count still honours K.
        pairs = two_segment_fan(cfg.kappa_max, int(round(math.sqrt(cfg.K))))
        traj = arc_rollout_2seg(pairs, cfg.arc_len_m, cfg.n)
        # kappa_1 is what gets COMMANDED this tick -- the second segment is a plan for the
        # next metre, which the controller will re-decide before reaching. So speed, the
        # smoothness term and the returned omega all key off kappa_1, exactly as the
        # single-segment fan keys off its one curvature.
        kappa = pairs[:, 0]
    else:
        kappa = curvature_fan(cfg.kappa_max, cfg.K)
        traj = arc_rollout_k(np.zeros(3), kappa, cfg.arc_len_m, cfg.n)   # (K, n+1, 3)
    K = kappa.size
    pts = traj[:, 1:, :2]                                                # drop the origin

    d_min = (float(np.min(np.linalg.norm(hits_body, axis=1)))
             if hits_body is not None and len(hits_body) else float("inf"))
    v = speed_for(kappa, cfg, clearance=d_min)

    # -- terrain: soft cost, and a hard class ------------------------------
    # Classified BEFORE the soft terms because truncation needs to know where each arc
    # stops being drivable, and that depends on both terrain and obstacles.
    klass = np.asarray(cfg.terrain_class(pts))
    if klass.shape != pts.shape[:2]:
        raise ValueError(f"terrain_class returned {klass.shape}, want {pts.shape[:2]}")
    cost = np.full(klass.shape, float(cfg.unknown_cost))
    for c, w in cfg.terrain_costs.items():
        cost[klass == c] = float(w)
    # WEIGHTED, not averaged: a hazard about to be driven over outranks one at the end of the
    # arc. `arc_weights` returns the uniform vector when `terrain_decay_m is None`, so this
    # reduces to the mean it replaces until the knob is set.
    w_arc = arc_weights(cfg.arc_len_m, cfg.n, cfg.terrain_decay_m, cfg.terrain_s_min_m)
    soft, blind, seen = terrain_soft_costs(
        klass, cost, w_arc, cfg.unobserved_class, cfg.unknown_cost, cfg.w_unknown)
    scores = -cfg.w_terrain * soft

    # EVERY POINT, not just the endpoint. "Never enter X" implemented as "never END in X"
    # lets a rollout cross the region, stop outside it, and still be accepted.
    bad_terrain = (np.isin(klass, np.fromiter(cfg.forbid_classes, dtype=int)).any(axis=1)
                   if cfg.forbid_classes else np.zeros(K, dtype=bool))
    if cfg.forbid_classes and cfg.min_adjudicable_frac > 0.0:
        # NOT CERTIFIED LEGAL. "No forbidden sample observed" is evidence of legality only
        # when enough of the arc was observed; otherwise an arc escapes the ban by leaving
        # the lens. See `min_adjudicable_frac`.
        bad_terrain = bad_terrain | (seen < float(cfg.min_adjudicable_frac))
    # THRESHOLDED, and WITHOUT the origin. `check_collisions` returns DISTANCES, not a
    # verdict, so they must be compared against a radius to get a boolean mask. Using the
    # float array as a mask directly would make `collision.all()` True for any non-zero
    # distance, latch `obstacle_relaxed`, and silently disable the obstacle constraint.
    #
    # Column 0 is dropped for the same reason `sampling_mpc` passes `rollouts[:, 1:, :]`:
    # every arc starts at the ego origin, so that column is the SAME point for all K and
    # equals `d_min`. Keeping it makes the test read "all K collide" whenever an obstacle
    # is already inside `safety_radius` -- which trips `.all()` and relaxes the constraint
    # exactly when it matters most. It is not enough on its own (10 m arcs sample densely
    # enough near the origin that the first arc point is often inside the radius too), but
    # leaving it in guarantees the degenerate reading.
    #
    # TWO RESPONSES, NOT ONE. The hard veto at `obstacle_hard_m` removes arcs that are
    # about to hit something; the graded penalty orders everything else by how close it
    # gets. See `obstacle_hard_m` for why a single 1.5 m veto deadlocks.
    if hits_body is not None:
        d_arc = occ.check_collisions(traj[:, 1:, :])          # (K, n) distances
        d_near = d_arc.min(axis=1)                            # closest approach per arc
        collision = d_near < cfg.obstacle_hard_m
        # Linear ramp, bounded in [0, 1]: 0 beyond the influence radius, 1 at contact.
        # Bounded on purpose -- a 1/d form is unbounded near zero and would swamp every
        # other term the moment a single return lands close.
        obstacle_cost = np.clip(
            (cfg.obstacle_influence_m - d_near) / max(cfg.obstacle_influence_m, 1e-9),
            0.0, 1.0)
    else:
        collision = np.zeros(K, dtype=bool)
        obstacle_cost = np.zeros(K)
    # -- ARC TRUNCATION, then the soft terms -------------------------------
    # An arc earns progress only for the part it could actually drive. Blocking is
    # obstacle OR forbidden terrain, per point; `reachable_index` turns that into the last
    # reachable index and the bearing/progress terms read THAT point rather than the
    # endpoint. An arc blocked at 2 m now earns 2 m, not 10.
    #
    # Deliberately applied to progress and bearing only. The terrain soft cost above is
    # still averaged over the whole arc, which over-charges an arc for ground it never
    # reaches -- a smaller version of the same error, handled separately by the distance
    # weighting (`terrain_decay_m`).
    blocked = np.zeros((K, cfg.n), dtype=bool)
    if hits_body is not None:
        blocked |= d_arc < cfg.obstacle_hard_m
    if cfg.forbid_classes:
        blocked |= np.isin(klass, np.fromiter(cfg.forbid_classes, dtype=int))
    reach = reachable_index(blocked)
    rows = np.arange(K)
    end_xy = traj[rows, reach, :2]
    th_end = traj[rows, reach, 2]

    u_ref = np.array([math.cos(bearing_ref), math.sin(bearing_ref)])
    scores = (scores
              + cfg.w_bearing * np.cos(th_end - bearing_ref)
              + cfg.w_progress * (end_xy @ u_ref)
              - cfg.w_smooth * np.abs(kappa - float(kappa_prev))
              - cfg.w_obstacle * obstacle_cost)

    # -- stay mid-corridor, from LiDAR only ---------------------------------
    # Scored at the REACHABLE endpoint, like bearing and progress, so a blocked arc is not
    # rewarded for being nicely centred somewhere it cannot get to. Silently skipped when
    # `w_corridor` is 0 (the default) or the scan shows no corridor -- and skipped is the
    # honest outcome, not a substituted centreline.
    corridor_offset = float("nan")
    if cfg.w_corridor != 0.0:
        est = corridor_from_hits(hits_body, cfg.corridor_min_hits)
        if est is not None:
            perp, corridor_offset = est
            # Offset of each candidate endpoint from the CENTRELINE, along `perp`.
            # `corridor_offset` is the robot's own offset from that line, i.e.
            # `-centre_proj`, so the candidate's offset is its projection MINUS
            # `centre_proj`, which is `+ corridor_offset`. The sign matters: subtracting it
            # puts the target line at +offset and the term steers AWAY from centre.
            across = (end_xy @ perp) + corridor_offset
            sig = max(float(cfg.corridor_sigma_m), 1e-6)
            scores = scores + cfg.w_corridor * np.exp(-(across ** 2) / (sig ** 2))

    # -- ordered fallback ---------------------------------------------------
    # Terrain relaxes before obstacles: crossing rough ground is recoverable, a collision
    # is not. Freezing is worse than either -- a stopped vehicle never clears what stopped
    # it, and the violation is visible to the monitor where a stall is not.
    terrain_relaxed = bool(bad_terrain.all())
    obstacle_relaxed = bool(collision.all())
    if not terrain_relaxed:
        scores[bad_terrain] = -np.inf
    if not obstacle_relaxed:
        scores[collision] = -np.inf

    if not np.isfinite(scores).any():
        best = int(np.argmin(soft))
        v_cmd, recovery = min(0.5, cfg.v_max * 0.1), True
    else:
        best = int(np.argmax(scores))
        v_cmd, recovery = float(v[best]), False

    return MpcResult(
        v=v_cmd, omega=float(v_cmd * kappa[best]), best_rollout=traj[best],
        # NaN, not 0.0: there IS no occupancy term, and a zero would read as one that
        # scored zero. Anything consuming this field must notice the difference.
        cluster_score=float("nan"),
        bearing_score=float(np.cos(th_end[best] - bearing_ref)),
        edt_blocked_frac=float(collision.mean()),
        offroad_frac=float(bad_terrain.mean()),
        offroad_rejected=(0 if terrain_relaxed else int(bad_terrain.sum())),
        forbidden_rejected=(-1 if terrain_relaxed else int(bad_terrain.sum())),
        # The GRADED terrain cost of the arc actually chosen. Reported separately because
        # `offroad_*` and `forbidden_rejected` all derive from the HARD mask, so without this
        # the log cannot tell "the chosen arc crosses some rough ground" from "arcs were
        # vetoed" -- a field that reads plausibly while answering another question.
        terrain_soft=float(soft[best]),
        # How many arcs the source could not describe AT ALL. The number that says whether a
        # terrain-scored run was really terrain-scored: 65 of 65 means the camera contributed
        # nothing and the controller drove on bearing, which completes and reports numbers.
        blind_arcs=int(blind.sum()),
        arc_soft=soft, arc_forbidden=bad_terrain, best_index=int(best),
        # Fraction of the K x n queried points the source could actually describe. NaN when
        # the source does not report one (the waypoint stand-in always knows). Without it a
        # dead camera returning all-unknown is indistinguishable from a clean road, which is
        # `terrain_class is None` (refused at construction) one level deeper.
        coverage_frac=float(getattr(cfg.terrain_class, "coverage_frac", float("nan"))),
        # RECOVERY MEANS THE CREEP FALLBACK, AND NOTHING ELSE. `obstacle_relaxed` is
        # reported separately; folding it in would make `recovery` read True on almost
        # every tick near obstacles even at full speed.
        recovery=bool(recovery),
        obstacle_relaxed=bool(obstacle_relaxed),
        terrain_relaxed=bool(terrain_relaxed),
        d_min=d_min,
        n_hits=int(len(hits_body)) if hits_body is not None else 0,
        nearest_region=-1,          # no region is consulted; -1 means "not asked"
        top_scores=np.sort(scores[np.isfinite(scores)])[-5:].tolist(),
    )
