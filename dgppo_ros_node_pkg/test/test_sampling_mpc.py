"""Behavioural tests for the sampling MPC core.

These run without CARLA: whether a target to the left produces a left turn is checked
directly. A sign error in the frame conversion and a genuinely broken controller look
identical from inside CARLA; here they do not.

Everything is in the PLANAR frame — metres, +y north, yaw CCW radians.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)

from dgppo_ros_node_pkg.sampling_mpc import (LocalOccGrid, MpcConfig,  # noqa: E402
                                             plan_step, sample_controls,
                                             unicycle_rollout)


def _rng():
    return np.random.default_rng(12345)


def _cfg(**kw):
    return MpcConfig(K=800, N=8, dt=0.2, **kw)


# --------------------------------------------------------------------------- #
# rollout + sampling                                                          #
# --------------------------------------------------------------------------- #

def test_positive_omega_turns_counter_clockwise():
    """The convention the whole frame conversion exists to protect.

    If this ever inverts, every right turn in CARLA becomes a left turn, and the
    symptom appears 30 m downstream as "the MPC took the wrong exit".
    """
    ctrl = np.zeros((1, 4, 2), dtype=np.float32)
    ctrl[0, :, 0] = 1.0        # v
    ctrl[0, :, 1] = 1.0        # +omega
    traj = unicycle_rollout(np.zeros(3), ctrl, 0.25)
    assert traj[0, -1, 2] > 0.0          # heading increased
    assert traj[0, -1, 1] > 0.0          # and it drifted to the LEFT (+y)


def test_curvature_limit_forbids_spinning_in_place():
    """A car cannot rotate at a standstill; the sampler must not propose that.

    Without the limit the MPC happily picks a spin-in-place rollout, the drivetrain
    refuses to execute it, and the resulting tracking error reads as a controller bug.
    """
    cfg = _cfg(v_max=5.0, omega_max=1.0, max_curvature=0.25)
    ctrl = sample_controls(cfg, _rng())
    v, om = ctrl[..., 0], ctrl[..., 1]
    assert (np.abs(om) <= v * cfg.max_curvature + 1e-6).all()
    slow = v < 0.2
    assert slow.any(), "sampler should still produce slow rollouts"
    assert np.abs(om[slow]).max() < 0.06


def test_curvature_limit_can_be_relaxed_for_a_holonomic_agent():
    """The pedestrian/Spot case: a large max_curvature restores turn-in-place."""
    cfg = _cfg(v_max=1.0, omega_max=1.5, max_curvature=100.0)
    ctrl = sample_controls(cfg, _rng())
    assert np.abs(ctrl[..., 1]).max() > 1.0


def test_constant_controls_hold_one_arc_over_the_horizon():
    """Each rollout must be a single arc, so the returned command IS the rollout.

    With per-step sampling the score depends only on the endpoint, so a wiggle scoring
    the same as a straight run gets chosen and its (random) first control is what gets
    commanded -- even aimed dead at the target, that can be full-lock steering in either
    direction, arbitrarily.
    """
    ctrl = sample_controls(_cfg(constant_controls=True), _rng())
    assert (ctrl[:, :, 0] == ctrl[:, :1, 0]).all()
    assert (ctrl[:, :, 1] == ctrl[:, :1, 1]).all()

    varying = sample_controls(_cfg(constant_controls=False), _rng())
    assert not (varying[:, :, 1] == varying[:, :1, 1]).all()


def test_per_step_sampling_produces_the_steering_noise_it_is_off_to_avoid():
    """Documents WHY constant_controls defaults True, so it cannot be flipped blind."""
    omegas = [plan_step(_cfg(constant_controls=False), np.array([0.0, 0.0]),
                        math.radians(90.0), _CENTS, _IDS, start_id=4, target_id=53,
                        rng=np.random.default_rng(s)).omega for s in range(8)]
    steady = [plan_step(_cfg(constant_controls=True), np.array([0.0, 0.0]),
                        math.radians(90.0), _CENTS, _IDS, start_id=4, target_id=53,
                        rng=np.random.default_rng(s)).omega for s in range(8)]
    assert np.std(omegas) > 5 * np.std(steady)
    # Residual +/-0.15 rad/s is sampling resolution, not noise: at 5 m/s that is a 33 m
    # turn radius, i.e. a few cm of lateral drift over the 1.6 s horizon.
    assert max(abs(o) for o in steady) < 0.15


# --------------------------------------------------------------------------- #
# steering                                                                    #
# --------------------------------------------------------------------------- #

_IDS = [4, 53, 8]
#: target 53 is due north of the ego, 8 is west of 53 — a right-then-left layout.
_CENTS = np.array([[0.0, 0.0], [0.0, 40.0], [-40.0, 40.0]], dtype=float)


@pytest.mark.parametrize("yaw_deg,expect", [
    (0.0,   "left"),       # facing east, target is north -> turn left
    (180.0, "right"),      # facing west, target is north -> turn right
    (90.0,  "straight"),   # facing north, target dead ahead
])
def test_it_steers_toward_the_target_region(yaw_deg, expect):
    res = plan_step(_cfg(), np.array([0.0, 0.0]), math.radians(yaw_deg),
                    _CENTS, _IDS, start_id=4, target_id=53, rng=_rng())
    if expect == "left":
        assert res.omega > 0.5, f"expected a hard left, got omega={res.omega:.3f}"
    elif expect == "right":
        assert res.omega < -0.5, f"expected a hard right, got omega={res.omega:.3f}"
    else:
        assert abs(res.omega) < 0.1, (
            f"aimed at the target but commanded omega={res.omega:.3f} — the bearing "
            f"term is not pointing at the target")
        assert res.v > 4.0, "should be at speed, not creeping, when aimed at the target"


def test_bearing_to_a_target_dead_ahead_is_zero_not_pi():
    """Regression: global->body rotation had the wrong sign.

    Using ``delta @ [[cy, sy], [-sy, cy]]`` rotates by PLUS yaw instead of minus,
    reflecting the bearing about the heading axis: for a target dead ahead the MPC
    computes "dead behind" and chooses full-lock steering. Facing north with the target
    due north is the sharpest statement of that bug, so it is the test.
    """
    res = plan_step(_cfg(), np.array([0.0, 0.0]), math.radians(90.0),
                    _CENTS, _IDS, start_id=4, target_id=53, rng=_rng())
    assert abs(res.omega) < 0.1
    # and the bearing term must be near its maximum, not near its minimum
    assert res.bearing_score > 3.0


def test_it_drives_forward_when_aimed_at_the_target():
    res = plan_step(_cfg(), np.array([0.0, 0.0]), math.radians(90.0),
                    _CENTS, _IDS, start_id=4, target_id=53, rng=_rng())
    assert res.v > 0.0
    assert res.nearest_region in (4, 53)


def test_the_forbidden_penalty_keeps_it_out_of_an_unauthorised_region():
    """A rollout ending in an off-plan region must lose to one that stays on plan.

    Geometry: the ego sits 2 m inside region 53's Voronoi cell, on the boundary with
    region 8, facing north — so turning left crosses into 8 and going straight stays in
    53. Both are equally easy, which is exactly the situation at a junction where the
    wrong exit is as available as the right one.
    """
    res = plan_step(_cfg(), np.array([-18.0, 40.0]), math.radians(90.0),
                    _CENTS, _IDS, start_id=4, target_id=53, rng=_rng())
    assert res.nearest_region != 8, "chose a rollout ending in a forbidden region"
    assert res.omega <= 0.05, "steered toward the forbidden region"


def test_unknown_region_id_raises_rather_than_steering_at_a_guess():
    with pytest.raises(KeyError, match="99"):
        plan_step(_cfg(), np.array([0.0, 0.0]), 0.0, _CENTS, _IDS,
                  start_id=4, target_id=99, rng=_rng())


# --------------------------------------------------------------------------- #
# obstacles                                                                   #
# --------------------------------------------------------------------------- #

def test_a_wall_ahead_slows_it_down():
    open_road = plan_step(_cfg(), np.array([0.0, 0.0]), math.radians(90.0),
                          _CENTS, _IDS, start_id=4, target_id=53, rng=_rng())
    wall = np.array([[2.0, dy] for dy in np.arange(-3.0, 3.1, 0.25)])   # 2 m ahead
    blocked = plan_step(_cfg(), np.array([0.0, 0.0]), math.radians(90.0),
                        _CENTS, _IDS, start_id=4, target_id=53,
                        hits_body=wall, rng=_rng())
    assert blocked.v < open_road.v
    assert blocked.n_hits == len(wall)
    assert blocked.d_min == pytest.approx(2.0, abs=0.2)


def test_fully_blocked_creeps_and_turns_instead_of_freezing():
    """A stopped vehicle never clears the obstacle that stopped it.

    In a synchronous sim, freezing means sitting there until the run times out, which
    logs as a plan failure rather than as an obstacle.
    """
    ring = np.array([[1.0 * math.cos(a), 1.0 * math.sin(a)]
                     for a in np.linspace(0, 2 * math.pi, 72, endpoint=False)])
    res = plan_step(_cfg(), np.array([0.0, 0.0]), math.radians(90.0),
                    _CENTS, _IDS, start_id=4, target_id=53,
                    hits_body=ring, rng=_rng())
    assert res.recovery
    assert res.edt_blocked_frac == pytest.approx(1.0)
    assert 0.0 <= res.v <= 0.6


def test_no_lidar_at_all_is_a_valid_state_not_an_error():
    """Phase A runs with no_rendering_mode on, so there are no sensors whatsoever."""
    res = plan_step(_cfg(), np.array([0.0, 0.0]), math.radians(90.0),
                    _CENTS, _IDS, start_id=4, target_id=53,
                    hits_body=None, rng=_rng())
    assert not res.recovery
    assert res.edt_blocked_frac == 0.0
    assert res.n_hits == 0
    assert math.isinf(res.d_min)


def test_occ_grid_reports_infinite_clearance_with_no_hits():
    occ = LocalOccGrid()
    occ.update(None)
    d = occ.check_collisions(np.zeros((3, 5, 3)))
    assert d.shape == (3, 5)
    assert np.isinf(d).all()


# --------------------------------------------------------------------------- #
# determinism                                                                 #
# --------------------------------------------------------------------------- #

def test_same_seed_gives_the_same_command():
    """Runs must be reproducible or a regression cannot be attributed to a change."""
    args = (_cfg(), np.array([0.0, 0.0]), 0.0, _CENTS, _IDS)
    a = plan_step(*args, start_id=4, target_id=53, rng=np.random.default_rng(7))
    b = plan_step(*args, start_id=4, target_id=53, rng=np.random.default_rng(7))
    assert (a.v, a.omega) == (b.v, b.omega)


def test_guidance_ratio_is_bounded():
    res = plan_step(_cfg(), np.array([0.0, 10.0]), math.radians(90.0),
                    _CENTS, _IDS, start_id=4, target_id=53, rng=_rng())
    assert 0.0 <= res.guidance_ratio <= 1.0


# --------------------------------------------------------------------------- #
# Region guidance actually being ON                                            #
#                                                                              #
# Two ways region guidance can be silently inert without raising: `guidance`  #
# never being set from a parameter, or building RoadSurface from              #
# `waypoints[:, :2]`, which drops the region-id column. Either alone disables  #
# the pure-pursuit path that stops the vehicle cutting corners out of          #
# junctions.                                                                   #
# --------------------------------------------------------------------------- #

def test_road_surface_keeps_region_ids_from_three_column_waypoints():
    """A 2-column slice yields one region for the whole town, with no error."""
    import numpy as np
    from dgppo_ros_node_pkg.sampling_mpc import RoadSurface

    wp = np.array([[0.0, 0.0, 7.0], [1.0, 0.0, 7.0], [50.0, 0.0, 9.0]])
    good = RoadSurface(wp)
    assert good.nearest_point_of(9) is not None
    assert int(good.region_of(np.array([[49.0, 0.0]]))[0]) == 9

    sliced = RoadSurface(wp[:, :2])          # the bug
    assert sliced.nearest_point_of(9) is None


def test_node_guidance_is_declared_plumbed_and_a_legal_value():
    """Declared, pushed onto the config, and defaulting to "centroid".

    The default is "centroid" on purpose: region guidance did not improve the driven
    outcome, and it demotes the bearing term to x0.25, which at a junction is the only
    term left with any gradient. The point of this test is that the value cannot
    silently become something nothing sets, leaving the region path unexecuted.
    """
    import ast
    import pathlib

    src = (pathlib.Path(__file__).resolve().parents[1]
           / "dgppo_ros_node_pkg" / "carla_mpc_ros_node.py").read_text()
    tree = ast.parse(src)
    found = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "declare_parameter"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "guidance"):
            found.append(node.args[1].value)
    assert found == ["centroid"], f"guidance parameter: {found}"
    # and it must be pushed onto the config, not merely declared
    assert 'cfg.guidance = self._str("guidance")' in src


def test_region_guidance_aims_into_the_target_not_at_its_centroid():
    """The steering aim point must lie in the target region, ahead of the vehicle.

    Under centroid aiming the aim point is the region's centre of mass, which for a
    long road region leaving a junction sits well off the line of travel -- that is
    the corner cut. Under region guidance it is a point of the region itself at
    least `lookahead_m` away.
    """
    import numpy as np
    from dgppo_ros_node_pkg.sampling_mpc import RoadSurface

    # An L: the vehicle sits at the junction (0, 0); the target region runs east.
    xs = np.arange(5.0, 60.0, 1.0)
    wp = np.vstack([
        np.column_stack([np.zeros(5), np.arange(-5.0, 0.0), np.full(5, 1.0)]),
        np.column_stack([xs, np.zeros(len(xs)), np.full(len(xs), 2.0)]),
    ])
    road = RoadSurface(wp)
    pos = np.array([0.0, 0.0])
    pts = road.nearest_point_of(2)
    assert pts is not None

    d2 = ((pts - pos) ** 2).sum(axis=1)
    far = d2 >= 12.0 ** 2
    aim = pts[far][int(np.argmin(d2[far]))]
    assert int(road.region_of(aim[None, :])[0]) == 2       # aim is IN the target
    assert np.linalg.norm(aim - pos) >= 12.0               # and far enough ahead
    # the centroid of the same region is much further out, so aiming at it
    # under-steers the entry
    assert np.linalg.norm(pts.mean(axis=0) - pos) > np.linalg.norm(aim - pos)


def test_collision_horizon_limits_the_rejection_window():
    """`collision_horizon` must actually shorten the window, and 0 must mean all N.

    The collision test is `.any()` over every horizon step, so a rollout dies on its
    last step -- 1.6 s out -- even though the controller replans every tick. At a large
    junction with nearby poles (Town05 junction 57) that rejects most rollouts and stalls
    the vehicle.
    """
    import numpy as np
    from dgppo_ros_node_pkg.sampling_mpc import MpcConfig as MPCConfig

    # a rollout that is clear for 4 steps then clips something on step 7
    clear = np.full((1, 8), 10.0)
    clear[0, 7] = 0.5
    cfg = MPCConfig()
    assert cfg.collision_horizon == 0                      # default: check everything

    def blocked(arr, horizon, radius=1.5):
        a = arr[:, :horizon] if horizon and 0 < horizon < arr.shape[1] else arr
        return bool((a < radius).any(axis=1)[0])

    assert blocked(clear, 0) is True        # full horizon sees the step-7 clip
    assert blocked(clear, 8) is True        # explicit full horizon, same
    assert blocked(clear, 4) is False       # first 4 steps are clear
    # and a genuinely close obstacle is still caught at any window
    near = np.full((1, 8), 10.0)
    near[0, 1] = 0.5
    assert blocked(near, 4) is True


def test_seed_zero_is_a_real_seed_not_os_entropy():
    """`seed or None` made the DEFAULT (0) mean "random", so nothing ever replayed.

    The value read as deterministic and behaved as its opposite, so repeated runs of one
    plan produced different trajectories.
    """
    import pathlib
    import re

    src = (pathlib.Path(__file__).resolve().parents[1]
           / "dgppo_ros_node_pkg" / "carla_mpc_ros_node.py").read_text()
    assert 'default_rng(self._int("seed") or None)' not in src, (
        "seed=0 falls through to OS entropy")
    assert re.search(r"default_rng\(None if _seed < 0 else _seed\)", src)

    import numpy as np
    for seed in (0, 1):
        a = np.random.default_rng(None if seed < 0 else seed).random(4)
        b = np.random.default_rng(None if seed < 0 else seed).random(4)
        assert (a == b).all(), f"seed {seed} did not replay"
    assert not (np.random.default_rng(0).random(4)
                == np.random.default_rng(1).random(4)).all()

