"""Tests for the terrain-scored controller.

Separate from `test_sampling_mpc.py` for the same reason the module is separate: these
can fail without implicating the default sampling controller.

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest dgppo_ros_node_pkg/test/test_terrain_mpc.py
(the autoload guard is needed because ROS's launch_testing plugin and the conda pytest
disagree about a hook signature; it has nothing to do with these tests).
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)

from dgppo_ros_node_pkg.sampling_mpc import (MpcConfig,  # noqa: E402
                                             sample_controls, unicycle_rollout)
from dgppo_ros_node_pkg.terrain_mpc import (TerrainMpcConfig,  # noqa: E402
                                            arc_rollout, plan_step_terrain)

FLAT = dict(terrain_class=lambda p: np.zeros(p.shape[:2], dtype=int),
            terrain_costs={0: 0.0})


# --------------------------------------------------------------------------- #
# The closed form must BE the thing Euler approximates                         #
# --------------------------------------------------------------------------- #

def test_the_arc_agrees_with_euler_and_the_gap_is_the_euler_error():
    """Asserting a magnitude at one ``dt`` would not establish agreement: a sign error and
    a legitimate integration error look identical at a single step size. What separates
    them is the RATE. Forward Euler is first order, so quartering ``dt`` must quarter the
    error; a wrong closed form would plateau at a constant offset and the ratio would fall
    toward 1.

    Expected errors: 0.3635 -> 0.0900 -> 0.0225 -> 0.0056 for dt 0.2 -> 0.05 -> 0.0125 ->
    0.003125, i.e. ratios 4.04 / 4.01 / 4.00.
    """
    cfg = MpcConfig()
    ctrl = sample_controls(cfg, np.random.default_rng(0))       # constant controls
    v = ctrl[:, 0, 0]

    errs = []
    for n in (8, 32, 128, 512):
        dt = (cfg.N * cfg.dt) / n                               # same horizon, finer steps
        euler = unicycle_rollout(np.zeros(3), np.repeat(ctrl[:, :1, :], n, axis=1), dt)
        worst = 0.0
        for kk in range(0, cfg.K, 37):                          # sample, not all 500
            # Same total distance: Euler covers v * n * dt, so the arc must too.
            arc = arc_rollout(np.zeros(3), ctrl[kk:kk + 1], float(v[kk]) * n * dt, n)
            worst = max(worst, float(np.abs(arc[0, :, :2] - euler[kk, :, :2]).max()))
        errs.append(worst)

    for coarse, fine in zip(errs, errs[1:]):
        ratio = coarse / fine
        assert 3.5 < ratio < 4.5, (
            f"error fell {ratio:.2f}x when dt fell 4x; first-order convergence wants ~4. "
            f"A ratio near 1 means a constant offset, i.e. the closed form is wrong rather "
            f"than merely coarse. sequence={errs}")


def test_every_arc_spans_the_same_distance():
    """The substantive change, and the thing a time parameterisation cannot give: under
    time a slow candidate covers less ground, so the score compares trajectories of
    unequal extent and a slow arc reads as "safe" largely by going nowhere."""
    cfg = MpcConfig()
    ctrl = sample_controls(cfg, np.random.default_rng(1))
    L = 10.0
    arcs = arc_rollout(np.zeros(3), ctrl, L, 64)

    seg = np.linalg.norm(np.diff(arcs[:, :, :2], axis=1), axis=2).sum(axis=1)
    moving = ctrl[:, 0, 0] > 1e-6
    assert np.allclose(seg[moving], L, atol=1e-2), \
        f"arc lengths spread {seg[moving].min():.3f}..{seg[moving].max():.3f}, want {L}"

    # A standstill candidate must NOT be teleported L metres down the straight branch.
    if (~moving).any():
        assert np.allclose(arcs[~moving, :, :2], 0.0)


def test_a_straight_candidate_is_not_a_divide_by_zero():
    """omega == 0 is common, not an edge case: `max_curvature` clips omega to zero for
    every near-standstill draw."""
    ctrl = np.zeros((1, 8, 2), dtype=np.float32)
    ctrl[:, :, 0] = 2.0                                          # v = 2, omega = 0
    arc = arc_rollout(np.array([0.0, 0.0, math.pi / 2]), ctrl, 6.0, 8)
    assert np.all(np.isfinite(arc))
    assert abs(arc[0, -1, 0]) < 1e-9                             # heading +y: x stays 0
    assert abs(arc[0, -1, 1] - 6.0) < 1e-9                       # y advances by the arc


# --------------------------------------------------------------------------- #
# The config must refuse to run and measure nothing                            #
# --------------------------------------------------------------------------- #

def test_a_terrain_scorer_with_no_terrain_source_is_refused():
    """An unsupported or unwired option must be rejected at config time rather than
    silently accepted. A terrain term with no source would degrade to bearing-only while
    still reporting terrain-scored numbers."""
    with pytest.raises(ValueError, match="terrain_class"):
        TerrainMpcConfig()


def test_a_fan_always_contains_a_straight_candidate():
    """An even fan straddles zero and offers no straight arc at all -- the controller
    could not propose driving straight ahead, which is the commonest correct answer."""
    from dgppo_ros_node_pkg.terrain_mpc import curvature_fan
    for asked in (3, 4, 32, 64, 65):
        fan = curvature_fan(0.35, asked)
        assert fan.size % 2 == 1, f"asked {asked}, got an even fan of {fan.size}"
        assert np.isclose(np.abs(fan).min(), 0.0), "no straight candidate in the fan"
        assert np.allclose(fan, -fan[::-1]), "fan is not symmetric about zero"
    with pytest.raises(ValueError, match="at least 3"):
        curvature_fan(0.35, 2)


def test_the_search_is_deterministic_so_two_ticks_agree():
    """Seeding does NOT make a driven run reproducible, because the rng
    stream desynchronises from the physical state when the tick count varies with ROS
    scheduling, so two runs with the same seed diverge. Removing the rng from
    the control path removes that variance source rather than seeding it."""
    cfg = TerrainMpcConfig(**FLAT)
    a = plan_step_terrain(cfg, 0.4)
    b = plan_step_terrain(cfg, 0.4)
    assert (a.v, a.omega) == (b.v, b.omega)


def test_speed_falls_with_curvature_and_with_clearance():
    """Speed is a rule, not a search dimension: with arcs of fixed length it does not
    change the geometry being scored, so scoring it would search a dimension the objective
    cannot see."""
    from dgppo_ros_node_pkg.terrain_mpc import curvature_fan, speed_for
    cfg = TerrainMpcConfig(**FLAT)
    fan = curvature_fan(cfg.kappa_max, cfg.K)
    v = speed_for(fan, cfg)
    straight = int(np.argmin(np.abs(fan)))
    assert v[straight] == v.max(), "the straight candidate should be the fastest"
    assert v[0] < v[straight] and v[-1] < v[straight], "tight arcs should be slower"
    near = speed_for(fan, cfg, clearance=cfg.safety_radius + 0.5)
    assert near.max() < v.max(), "a close obstacle must cap speed"


def test_smoothness_penalises_change_not_magnitude():
    """`-w|omega|` taxes a steady turn every tick although nothing about it is jerky, and
    couples no two ticks. |k - k_prev| is what actually resists chatter."""
    cfg = TerrainMpcConfig(w_smooth=5.0, w_bearing=0.0, w_progress=0.0, **FLAT)
    # With bearing and progress zeroed, only smoothness is left: the choice must be
    # whatever curvature was commanded last, however tight that was.
    # Derived from the fan, not hardcoded: a hardcoded curvature can fall outside the fan
    # when kappa_max changes, and the nearest offered arc would then read as chatter. These
    # three are the fan's endpoints and its centre by construction.
    for k_prev in (-cfg.kappa_max, 0.0, cfg.kappa_max):
        r = plan_step_terrain(cfg, 0.0, kappa_prev=k_prev)
        k_cmd = r.omega / r.v if r.v > 1e-6 else 0.0
        assert abs(k_cmd - k_prev) < 0.02, \
            f"k_prev={k_prev} but commanded {k_cmd:.3f}"


# --------------------------------------------------------------------------- #
# Behaviour                                                                    #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("bearing,sign", [(0.0, 0), (0.6, +1), (-0.6, -1)])
def test_it_steers_toward_the_reference_bearing(bearing, sign):
    """The one map-derived input is a DIRECTION. On flat ground nothing else should move
    the choice, so the sign of omega must follow the sign of the bearing."""
    cfg = TerrainMpcConfig(**FLAT)
    r = plan_step_terrain(cfg, bearing)
    if sign == 0:
        assert abs(r.omega) < 0.05
    else:
        assert math.copysign(1, r.omega) == sign, \
            f"bearing {bearing:+.1f} produced omega {r.omega:+.3f}"


def test_the_selected_arc_never_enters_a_forbidden_class():
    """The contract, tested on the OUTPUT rather than on the rejection count.

    A rejection count says nothing: with a stripe to the left and a left-leaning bearing,
    many rollouts are rejected and the chosen one still turns left -- correctly, via
    a tighter arc that curves back before reaching the stripe. What must hold is that the
    arc actually chosen never enters the class, at any point along it and at any seed.
    """
    def klass(p):
        return np.where(p[:, :, 1] > 2.0, 9, 0)                  # forbidden stripe, left

    cfg = TerrainMpcConfig(terrain_class=klass, terrain_costs={0: 0.0, 9: 5.0},
                           forbid_classes=frozenset({9}))
    for seed in range(40):
        r = plan_step_terrain(cfg, 0.6, kappa_prev=0.0)
        assert float(r.best_rollout[:, 1].max()) <= 2.0, \
            f"seed {seed}: chosen arc reached y={r.best_rollout[:, 1].max():.2f}"


def test_the_constraint_actually_changes_the_choice():
    """Guards against the constraint being inert -- the failure mode where a test passes
    because the unconstrained answer already satisfied it."""
    def klass(p):
        return np.where(p[:, :, 1] > 2.0, 9, 0)

    free = plan_step_terrain(TerrainMpcConfig(**FLAT), 0.6)
    held = plan_step_terrain(
        TerrainMpcConfig(terrain_class=klass, terrain_costs={0: 0.0, 9: 5.0},
                         forbid_classes=frozenset({9})),
        0.6)
    assert float(free.best_rollout[:, 1].max()) > 2.0, \
        "the unconstrained choice already avoided the stripe; the test proves nothing"
    assert float(held.best_rollout[:, 1].max()) <= 2.0


def test_no_region_is_ever_consulted():
    """The architectural contract, asserted rather than trusted: this controller must not
    report an occupancy score, because it does not compute one. NaN and not 0.0, so a
    consumer cannot read "no occupancy term" as "occupancy scored zero"."""
    r = plan_step_terrain(TerrainMpcConfig(**FLAT), 0.0)
    assert math.isnan(r.cluster_score)
    assert r.nearest_region == -1


# --------------------------------------------------------------------------- #
# Obstacles                                                                    #
# --------------------------------------------------------------------------- #
#
# `LocalOccGrid.check_collisions` returns DISTANCES. Using the return value directly as
# a boolean mask disables the obstacle constraint: `.all()` is True for any non-zero
# distance, the veto is skipped as "would reject everything", and `edt_blocked_frac`
# reports a mean distance where a fraction is promised. Nothing fails visibly, so these
# tests guard it explicitly.


def _obstacle_at(pt) -> np.ndarray:
    return np.asarray(pt, dtype=float).reshape(1, 2)


def test_an_obstacle_removes_some_arcs_but_not_all():
    """A fraction strictly between 0 and 1 is the contract: 0 means the constraint is
    inert, 1 means it relaxed and is equally inert. The distance-as-mask bug produces a
    value in metres (e.g. 3.52), which is not a fraction of anything."""
    r = plan_step_terrain(TerrainMpcConfig(**FLAT), 0.6,
                          hits_body=_obstacle_at([5.0, 3.0]))
    assert 0.0 < r.edt_blocked_frac < 1.0, r.edt_blocked_frac
    assert not r.recovery


def test_edt_blocked_frac_is_a_fraction_and_not_a_mean_distance():
    """The regression, named. Under the defect a single hit at 2 m reports 3.52. Any
    obstacle far enough away to veto nothing must report exactly 0.0, and no placement
    may ever report more than 1.0."""
    for pt in ([2.0, 0.0], [5.0, 3.0], [-1.0, 0.0], [40.0, 40.0]):
        r = plan_step_terrain(TerrainMpcConfig(**FLAT), 0.0, hits_body=_obstacle_at(pt))
        assert 0.0 <= r.edt_blocked_frac <= 1.0, f"{pt}: {r.edt_blocked_frac}"
    far = plan_step_terrain(TerrainMpcConfig(**FLAT), 0.0,
                            hits_body=_obstacle_at([40.0, 40.0]))
    assert far.edt_blocked_frac == 0.0


@pytest.mark.parametrize("bearing", [-0.6, -0.2, 0.0, 0.2, 0.6])
def test_the_chosen_arc_clears_the_hard_veto_radius(bearing):
    """The contract, tested on the OUTPUT rather than the rejection count.

    Asserts `obstacle_hard_m`, NOT `safety_radius`. A single 1.5 m veto deadlocks inside
    junctions with nearby poles -- see `obstacle_hard_m`. `safety_radius`
    now only feeds `speed_for`'s clearance cap, and an arc passing at 1.2 m is a legal
    choice that the graded penalty merely prices.

    The obstacle is placed ON the arc chosen without one, at each bearing, so the test can
    never pass because the free answer happened to clear it.
    """
    cfg = TerrainMpcConfig(**FLAT)
    free = plan_step_terrain(cfg, bearing)
    ob = _obstacle_at(free.best_rollout[10, :2])
    r = plan_step_terrain(cfg, bearing, hits_body=ob)
    d = float(np.linalg.norm(r.best_rollout[1:, :2] - ob[0], axis=1).min())
    assert d >= cfg.obstacle_hard_m, f"chosen arc passed {d:.2f} m from the obstacle"


def test_a_close_obstacle_field_does_not_deadlock_the_whole_fan():
    """The regression this design exists for, taken from CARLA geometry.

    Inside Town05 junction 60 the LiDAR returns traffic-light poles at 1.7-2.4 m. Under a
    single 1.5 m veto every one of the 65 arcs is rejected, the relax-everything fallback
    takes over, and the vehicle circles instead of crossing the junction.

    What must hold is that SOME candidate survives and the result is a real choice, not a
    fallback: a fallback has no idea where the obstacle is, so it cannot steer away.
    """
    poles = np.array([[1.9, 1.2], [2.1, -1.4], [3.0, 2.2]])
    r = plan_step_terrain(TerrainMpcConfig(**FLAT), 0.3, hits_body=poles)
    assert r.edt_blocked_frac < 1.0, "every arc vetoed -- the deadlock is back"
    assert not r.recovery, "fell back instead of choosing"
    assert r.v > 0.0


def test_the_graded_penalty_buys_clearance_beyond_the_veto():
    """The soft term must ORDER the survivors, not just exist. Without it the argmax is
    free to hug the hard boundary, which is the behaviour a bare veto already gave."""
    ob = _obstacle_at([4.0, 1.2])

    def clearance(w):
        cfg = TerrainMpcConfig(w_obstacle=w, **FLAT)
        r = plan_step_terrain(cfg, 0.3, hits_body=ob)
        return float(np.linalg.norm(r.best_rollout[1:, :2] - ob[0], axis=1).min())

    assert clearance(20.0) > clearance(0.0), \
        "weighting the obstacle term changed nothing -- it is inert"


def test_the_obstacle_veto_actually_changes_the_choice():
    """Guards against passing because the unconstrained answer already cleared it -- the
    failure mode that let the dead constraint sit here unnoticed. The obstacle is placed
    ON the arc chosen without one, so the free choice MUST be illegal."""
    cfg = TerrainMpcConfig(**FLAT)
    free = plan_step_terrain(cfg, 0.6)
    ob = _obstacle_at(free.best_rollout[12, :2])          # midway along the free choice

    def clearance(res):
        return float(np.linalg.norm(res.best_rollout[1:, :2] - ob[0], axis=1).min())

    # The veto radius is `obstacle_hard_m`, NOT `safety_radius` (a single 1.5 m veto
    # deadlocks junctions with nearby poles). A safety_radius assertion passed only
    # incidentally with a 10 m arc; at the 6.28 m default the choice clears 1.03 m, which
    # meets obstacle_hard_m but would fail a safety_radius assertion.
    assert clearance(free) < cfg.obstacle_hard_m, \
        "the unconstrained choice already cleared the obstacle; the test proves nothing"
    held = plan_step_terrain(cfg, 0.6, hits_body=ob)
    assert clearance(held) >= cfg.obstacle_hard_m


def test_the_shared_origin_does_not_veto_every_arc():
    """Every arc starts at the ego origin, so column 0 is the same point for all K and
    equals `d_min`. Scoring it makes an obstacle ALREADY inside the safety radius read as
    "all K collide", which trips the relax branch and switches the constraint off exactly
    when it matters. An obstacle BEHIND the vehicle separates the two readings: the origin
    is within the radius, every arc travels away from it."""
    r = plan_step_terrain(TerrainMpcConfig(**FLAT), 0.0,
                          hits_body=_obstacle_at([-1.0, 0.0]))
    assert r.edt_blocked_frac == 0.0
    assert not r.recovery


def test_an_unavoidable_obstacle_relaxes_rather_than_freezing():
    """The ordered fallback's obstacle half: with
    `kappa_max` 0.35 the tightest arc has R = 2.9 m, so a return 2 m dead ahead is inside
    every candidate's swept corridor and no steering clears it. The controller must still
    return a finite command and SAY so, rather than emit nothing."""
    r = plan_step_terrain(TerrainMpcConfig(**FLAT), 0.0,
                          hits_body=_obstacle_at([2.0, 0.0]))
    assert r.edt_blocked_frac == 1.0
    # `obstacle_relaxed`, NOT `recovery`. They are separate flags because on a street
    # nearly every arc passes within the veto radius of some roadside return, so a merged
    # flag would read True almost constantly. Relaxing a veto is not creeping; the vehicle
    # here is still driving at the clearance-capped speed asserted below.
    assert r.obstacle_relaxed, "every candidate was blocked and the result did not say so"
    assert not r.recovery, "the creep fallback did not engage and must not be reported"
    assert math.isfinite(r.v) and math.isfinite(r.omega)
    # `speed_for`'s clearance cap is the only thing left holding the vehicle off it.
    assert r.v <= math.sqrt(2.0 * TerrainMpcConfig(**FLAT).a_max * (2.0 - 1.5)) + 1e-6


# --------------------------------------------------------------------------- #
# Arc truncation: an arc earns progress only for the part it can drive         #
# --------------------------------------------------------------------------- #


def test_reachable_index_maps_blocked_points_to_traj_indices():
    """Indexing is off-by-one by construction: `blocked[k, j]` is arc point j, which is
    `traj[k, j+1]`, because traj index 0 is the shared origin and is never scored."""
    from dgppo_ros_node_pkg.terrain_mpc import reachable_index
    b = np.array([[False, False, False],      # nothing blocks -> full arc, index n
                  [True, False, False],       # blocked at the first point -> index 0
                  [False, True, False],       # blocked at the second     -> index 1
                  [False, False, True]])      # blocked at the last       -> index 2
    assert reachable_index(b).tolist() == [3, 0, 1, 2]


def test_truncation_removes_the_false_gradient_when_everything_is_blocked():
    """What truncation fixes, as the quantity rather than the outcome.

    Scoring the ENDPOINT gives the same 11.00 m spread of progress across the fan whether
    the path is clear or every single arc is walled off at 2 m -- a confident gradient
    computed over distance no candidate can travel. Truncation collapses it to the real
    one: ~0 when nothing is reachable, preserved when something is.
    """
    from dgppo_ros_node_pkg.terrain_mpc import (arc_rollout_k, curvature_fan,
                                                reachable_index)
    from dgppo_ros_node_pkg.sampling_mpc import LocalOccGrid
    cfg = TerrainMpcConfig(**FLAT)
    kap = curvature_fan(cfg.kappa_max, cfg.K)
    traj = arc_rollout_k(np.zeros(3), kap, cfg.arc_len_m, cfg.n)
    u = np.array([1.0, 0.0])
    rows = np.arange(len(kap))

    def spread(hits):
        occ = LocalOccGrid()
        occ.update(hits)
        blocked = occ.check_collisions(traj[:, 1:, :]) < cfg.obstacle_hard_m
        reach = reachable_index(blocked)
        endpoint = traj[:, -1, :2] @ u
        truncated = traj[rows, reach, :2] @ u
        return np.ptp(endpoint), np.ptp(truncated), blocked.any(1).mean()

    e_near, t_near, frac_near = spread(np.array([[2.0, 0.0]]))
    e_far, t_far, _ = spread(np.array([[5.0, 3.0]]))
    assert frac_near == 1.0, "expected every arc blocked by an obstacle 2 m dead ahead"
    assert e_near == pytest.approx(e_far, abs=1e-6), \
        "endpoint scoring is blind to blocking -- that is the defect"
    # Thresholds as FRACTIONS of the arc, so they do not encode a particular arc length.
    assert t_near < 0.08 * cfg.arc_len_m, \
        f"truncated spread {t_near:.2f} -- false gradient survives"
    assert t_far > 0.30 * cfg.arc_len_m, \
        f"truncated spread {t_far:.2f} -- real gradient was destroyed"


def test_an_unavoidable_obstacle_now_steers_instead_of_driving_in():
    """With an obstacle 2 m dead ahead, endpoint scoring commands kappa = 0 -- straight
    into it -- because straight is credited with full progress THROUGH the obstacle.
    Raising `w_obstacle` does not fix that; the progress quantity is wrong, not its
    weight."""
    r = plan_step_terrain(TerrainMpcConfig(**FLAT), 0.0,
                          hits_body=_obstacle_at([2.0, 0.0]))
    kappa = r.omega / r.v if r.v > 1e-6 else 0.0
    assert abs(kappa) > 0.2, f"kappa {kappa:+.3f}: still driving into it"


def test_a_clear_road_is_still_driven_straight():
    """Truncation must not make the controller swerve when nothing blocks it."""
    r = plan_step_terrain(TerrainMpcConfig(**FLAT), 0.0, hits_body=None)
    kappa = r.omega / r.v if r.v > 1e-6 else 0.0
    assert abs(kappa) < 1e-6, f"kappa {kappa:+.3f} on an empty scene"
    assert r.v == pytest.approx(TerrainMpcConfig(**FLAT).v_max)


# --------------------------------------------------------------------------- #
# Two-segment arcs: the shape a single constant curvature cannot make          #
# --------------------------------------------------------------------------- #


def test_the_fan_is_a_grid_and_keeps_a_straight_candidate_per_segment():
    """Enumerated, not sampled -- the distinction that separates this from a random walk.
    `per_seg` is forced odd for the same reason `curvature_fan` is: an even grid straddles
    zero and cannot propose going straight, which is the commonest correct answer."""
    from dgppo_ros_node_pkg.terrain_mpc import two_segment_fan
    f = two_segment_fan(0.35, 15)
    assert f.shape == (225, 2)
    assert (np.abs(f).min(axis=0) == 0).all(), "no straight candidate in one of the segments"
    even = two_segment_fan(0.35, 14)
    assert even.shape == (225, 2), "an even per_seg must be bumped to odd"
    with pytest.raises(ValueError):
        two_segment_fan(0.35, 2)


def test_the_fan_is_deterministic_across_calls():
    """No RNG in the control path. Seeding alone does NOT make a
    driven run reproducible, so the candidate set must not be a source of variance."""
    from dgppo_ros_node_pkg.terrain_mpc import two_segment_fan
    assert np.array_equal(two_segment_fan(0.35, 15), two_segment_fan(0.35, 15))


def test_a_two_segment_arc_joins_continuously_at_the_midpoint():
    """Position and heading are continuous at the join; only curvature steps -- exactly as
    the single-segment fan steps curvature between ticks. A discontinuity in position
    would mean the second segment started somewhere the first never reached."""
    from dgppo_ros_node_pkg.terrain_mpc import arc_rollout_2seg
    t = arc_rollout_2seg(np.array([[0.30, -0.30]]), arc_len=10.0, n=16)
    mid = t.shape[1] // 2
    step_before = np.linalg.norm(t[0, mid] - t[0, mid - 1])
    step_after = np.linalg.norm(t[0, mid + 1] - t[0, mid])
    assert step_after == pytest.approx(step_before, rel=0.25), "position jumps at the join"


def test_equal_curvatures_reproduce_the_single_segment_arc():
    """kappa_1 == kappa_2 IS a constant-curvature arc, so the two primitives must agree
    there. Guards the closed form in `_arc_from`, which re-derives the arc from a
    per-candidate start pose rather than the shared origin."""
    from dgppo_ros_node_pkg.terrain_mpc import arc_rollout_2seg, arc_rollout_k
    for k in (0.0, 0.12, -0.31):
        two = arc_rollout_2seg(np.array([[k, k]]), arc_len=10.0, n=16)
        one = arc_rollout_k(np.zeros(3), np.array([k]), 10.0, 16)
        assert np.allclose(two[0], one[0], atol=1e-9), f"kappa={k}: the two forms disagree"


def test_two_segments_can_swerve_and_return_where_one_cannot():
    """Why two-segment arcs exist, as a geometric property.

    On the 2D rig geometry -- a 0.04 m obstacle 0.25 m ahead between walls at +-0.175 m --
    none of the 65 single arcs clear the obstacle AND stay inside, while some two-segment
    arcs do. A denser single-segment fan does not help: a circular arc committed enough to
    miss the obstacle has already left the corridor.
    """
    from dgppo_ros_node_pkg.terrain_mpc import (arc_rollout_2seg, arc_rollout_k,
                                                curvature_fan, two_segment_fan)
    L, kmax, need, wall = 0.5, 1 / 0.143, 0.05, 0.175
    ob = np.array([0.25, 0.0])

    def clears(pts):
        return (np.linalg.norm(pts - ob, axis=1).min() >= need
                and np.abs(pts[:, 1]).max() <= wall)

    one = arc_rollout_k(np.zeros(3), curvature_fan(kmax, 65), L, 60)
    two = arc_rollout_2seg(two_segment_fan(kmax, 15), L, 60)
    n_one = sum(clears(one[i, :, :2]) for i in range(len(one)))
    n_two = sum(clears(two[i, :, :2]) for i in range(len(two)))
    assert n_one == 0, f"{n_one} single arcs thread it -- the premise no longer holds"
    assert n_two > 0, "two segments cannot thread it either; the primitive is not the fix"


def test_selecting_two_segments_commands_the_FIRST_curvature():
    """The second segment is a plan for a metre the controller will re-decide before
    reaching, so speed, smoothness and the issued omega all key off kappa_1."""
    cfg = TerrainMpcConfig(segments=2, K=225, **FLAT)
    r = plan_step_terrain(cfg, 0.0, hits_body=None)
    kappa = r.omega / r.v if r.v > 1e-6 else 0.0
    assert abs(kappa) <= cfg.kappa_max + 1e-9
    assert abs(kappa) < 1e-6, "an empty scene should still command straight"


def test_one_segment_remains_the_default():
    """The default is segments=1, the single constant-curvature fan."""
    assert TerrainMpcConfig(**FLAT).segments == 1


# --------------------------------------------------------------------------- #
# Corridor centring, from LiDAR only                                           #
# --------------------------------------------------------------------------- #


def _corridor(y_hi, y_lo):
    """Two walls along +x. The centreline is their midpoint; the robot sits at y = 0."""
    xs = np.arange(-3, 8.1, 0.4)
    return np.array([[x, y_hi] for x in xs] + [[x, y_lo] for x in xs])


def test_the_perpendicular_is_across_the_corridor_not_along_it():
    """PCA's min-variance eigenvector must come out ACROSS the walls. Two plausible
    filters both invert it: keeping only |x| < |y| "lateral" hits collapses two long
    walls into two blobs whose spread runs across, giving perp = [1, 0]; excluding a
    forward cone drops distant wall points (a wall 8 m ahead subtends 10 deg) and gives
    45 deg off. Unfiltered PCA is dominated by the walls and is exact."""
    from dgppo_ros_node_pkg.terrain_mpc import corridor_from_hits
    perp, _ = corridor_from_hits(_corridor(1.5, -2.5))
    assert abs(perp[0]) < 0.05 and perp[1] > 0.99, perp


def test_the_offset_measures_the_robot_against_the_centreline():
    from dgppo_ros_node_pkg.terrain_mpc import corridor_from_hits
    _, off = corridor_from_hits(_corridor(1.5, -2.5))     # centre at y = -0.5
    assert off == pytest.approx(0.5, abs=0.02), off
    _, off2 = corridor_from_hits(_corridor(2.0, -2.0))    # centred
    assert abs(off2) < 0.02, off2


def test_a_forward_obstacle_does_not_rotate_the_estimate():
    """Unfiltered PCA already resists this without any hit filtering."""
    from dgppo_ros_node_pkg.terrain_mpc import corridor_from_hits
    ob = np.repeat(np.array([[3.0, 0.0], [3.2, 0.1], [3.1, -0.1]]), 20, axis=0)
    perp, _ = corridor_from_hits(np.vstack([_corridor(2.0, -2.0), ob]))
    assert abs(perp[0]) < 0.10, perp


@pytest.mark.parametrize("hi,lo,sign", [(1.5, -2.5, -1), (2.5, -1.5, +1)])
def test_it_steers_toward_the_centreline(hi, lo, sign):
    """THE SIGN. `corridor_from_hits` returns the ROBOT's offset from the centreline
    (-centre_proj), so a candidate's offset is its projection PLUS that, not minus.
    Subtracting puts the target line on the wrong side and the term steers AWAY from
    centre."""
    cfg = TerrainMpcConfig(w_corridor=4.0, corridor_sigma_m=1.0, **FLAT)
    r = plan_step_terrain(cfg, 0.0, hits_body=_corridor(hi, lo))
    kappa = r.omega / r.v if r.v > 1e-6 else 0.0
    assert np.sign(kappa) == sign, f"kappa {kappa:+.4f}, expected sign {sign:+d}"


def test_a_centred_robot_is_not_pushed_off_line():
    cfg = TerrainMpcConfig(w_corridor=4.0, corridor_sigma_m=1.0, **FLAT)
    r = plan_step_terrain(cfg, 0.0, hits_body=_corridor(2.0, -2.0))
    assert abs(r.omega / r.v if r.v > 1e-6 else 0.0) < 1e-6


def test_no_corridor_visible_drops_the_term_rather_than_inventing_one():
    """With nothing to see the honest answer is to skip the term. Substituting a
    centreline would make sparse stretches read as successful perception."""
    from dgppo_ros_node_pkg.terrain_mpc import corridor_from_hits
    assert corridor_from_hits(None) is None
    assert corridor_from_hits(np.array([[3.0, 0.0]])) is None


def test_the_corridor_term_is_off_by_default():
    """Additive: default behaviour is unchanged until w_corridor is set."""
    assert TerrainMpcConfig(**FLAT).w_corridor == 0.0
    a = plan_step_terrain(TerrainMpcConfig(**FLAT), 0.0, hits_body=_corridor(1.5, -2.5))
    b = plan_step_terrain(TerrainMpcConfig(w_corridor=0.0, **FLAT), 0.0,
                          hits_body=_corridor(1.5, -2.5))
    assert a.omega == pytest.approx(b.omega)


# --------------------------------------------------------------------------- #
# arc_len_m x kappa_max: two knobs whose PRODUCT is the turn angle             #
# --------------------------------------------------------------------------- #


def test_the_defaults_satisfy_their_own_turn_cap():
    """The cap is now ON by default and the defaults sit exactly at it: arc_len 6.28 x
    kappa_max 0.25 = 90 deg, R_min 4.0 m. `arc_len_m` and `kappa_max` are otherwise
    independent knobs whose product is the turn angle; unchecked they can wrap past 90 deg."""
    cfg = TerrainMpcConfig(**FLAT)
    assert cfg.max_turn_rad == pytest.approx(math.pi / 2)
    assert cfg.arc_len_m * cfg.kappa_max <= cfg.max_turn_rad + 1e-9
    assert 1.0 / cfg.kappa_max == pytest.approx(4.0)


def test_an_over_long_arc_config_violates_the_90_degree_cap_and_says_how_to_fix_it():
    """arc_len 10 m x kappa_max 0.25 = 143 deg. 24 of 65 candidates exceed 90 deg and the
    tightest reaches +2.39 m of forward progress against +10.00 for straight -- a 4x
    penalty on turning that comes from the arc WRAPPING, not from the manoeuvre."""
    with pytest.raises(ValueError) as e:
        TerrainMpcConfig(arc_len_m=10.0, kappa_max=0.25,
                         max_turn_rad=math.pi / 2, **FLAT)
    msg = str(e.value)
    assert "143 deg" in msg and "6.28" in msg, msg


def test_shortening_the_arc_satisfies_the_cap_and_keeps_the_turning_radius():
    """The fix must not cost turning authority: kappa_max is untouched, so R_min stays
    4.0 m. Reducing kappa_max instead would satisfy the cap by making the vehicle unable
    to turn, which is the opposite of what a junction needs."""
    cfg = TerrainMpcConfig(arc_len_m=math.pi / 2 / 0.25, kappa_max=0.25,
                           max_turn_rad=math.pi / 2, **FLAT)
    assert cfg.kappa_max == 0.25
    assert 1.0 / cfg.kappa_max == pytest.approx(4.0)


def test_capping_per_arc_is_NOT_what_this_does():
    """Every candidate must still span the same distance. Item 2 of the module docstring
    is that fixed arc length makes the score rank SHAPE rather than extent; capping each
    arc's length individually would quietly reintroduce the unequal-extent comparison the
    time parameterisation was abandoned for."""
    from dgppo_ros_node_pkg.terrain_mpc import arc_rollout_k, curvature_fan
    cfg = TerrainMpcConfig(arc_len_m=6.28, kappa_max=0.25, max_turn_rad=math.pi / 2,
                           **FLAT)
    kap = curvature_fan(cfg.kappa_max, cfg.K)
    traj = arc_rollout_k(np.zeros(3), kap, cfg.arc_len_m, cfg.n)
    seg = np.linalg.norm(np.diff(traj[:, :, :2], axis=1), axis=2).sum(axis=1)
    # Tolerance is for the CHORD SUM, not the arcs: an n-point polyline under-measures a
    # curved arc, and more so the tighter it curves, so the chord-sum spread is 0.04% at
    # n=16 while the true arc lengths are identical by construction. A real per-arc cap
    # would show up here as a spread of tens of percent.
    assert np.ptp(seg) / seg.mean() < 0.005, (
        f"spread {np.ptp(seg) / seg.mean():.3%} -- candidates no longer span equal distance")


def test_the_cap_removes_the_wrap_penalty_on_turning():
    """The point of the fix, as the quantity it changes. Forward progress of the tightest
    arc, relative to straight: 0.24 uncapped (10 m), 0.64 capped (6.28 m)."""
    from dgppo_ros_node_pkg.terrain_mpc import arc_rollout_k, curvature_fan
    u = np.array([1.0, 0.0])

    def ratio(arc_len):
        kap = curvature_fan(0.25, 65)
        t = arc_rollout_k(np.zeros(3), kap, arc_len, 16)
        p = t[:, -1, :2] @ u
        return float(p.min() / p.max())

    assert ratio(10.0) < 0.30
    assert ratio(math.pi / 2 / 0.25) > 0.55
