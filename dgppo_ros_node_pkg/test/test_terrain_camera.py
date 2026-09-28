"""Tests for the camera terrain source: the projection, the staleness transform, and the
two rules that decide what an unobserved sample means.

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest \
         dgppo_ros_node_pkg/test/test_terrain_camera.py
(the autoload guard is needed because ROS's launch_testing plugin and the conda pytest
disagree about a hook signature; it has nothing to do with these tests).

Every number asserted here is computed from the real fan and the real mount, not chosen to
make a test pass, so a mount or lens change announces itself here first.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(PKG)), "src",
                                "carla_gt_bridge"))
sys.path.insert(0, os.path.join(os.path.dirname(PKG), "carla_gt_bridge"))

from carla_gt_bridge.camera_terrain import (CUE_CAMERA, TERRAIN_CAMERA,  # noqa: E402
                                            CameraModel, MaskTerrainSource, body_to_body)
from carla_gt_bridge.terrain_classes import (GRASS, OTHER, ROAD,  # noqa: E402
                                             SIDEWALK, UNOBSERVED)
from dgppo_ros_node_pkg.terrain_mpc import (TerrainMpcConfig,  # noqa: E402
                                            arc_rollout_k, arc_weights, curvature_fan,
                                            plan_step_terrain, terrain_soft_costs)

FAN = dict(kappa_max=0.35, K=65, arc_len=10.0, n=16)


def fan_points(cam_free: bool = False) -> np.ndarray:
    """The exact (65, 16, 2) body-frame sample set the controller scores."""
    k = curvature_fan(FAN["kappa_max"], FAN["K"])
    return arc_rollout_k(np.zeros(3), k, FAN["arc_len"], FAN["n"])[:, 1:, :2]


# --------------------------------------------------------------------------- #
# The projection                                                               #
# --------------------------------------------------------------------------- #

def test_a_point_on_the_left_lands_left_of_centre():
    """The y-sign is an easy silent bug (see `carla_gt_bridge.frames` for the convention);
    getting it backwards here mirrors
    every rollout about the heading while still producing plausible pixels.

    Body +y is LEFT; image +u is RIGHT. So a point to the left must give u < cx.
    """
    u, v, vis = CUE_CAMERA.project_ground(np.array([[10.0, 2.0]]))
    assert bool(vis[0])
    assert u[0] < CUE_CAMERA.cx
    u_r, _, _ = CUE_CAMERA.project_ground(np.array([[10.0, -2.0]]))
    assert u_r[0] > CUE_CAMERA.cx
    # Symmetric about the principal point, or the sign is right and the scale is not.
    assert CUE_CAMERA.cx - u[0] == pytest.approx(u_r[0] - CUE_CAMERA.cx)


def test_the_horizon_is_the_principal_row_and_the_near_cut_is_where_the_maths_says():
    """Two falsifiable predictions of the pinhole model, both checkable on one real frame.

    With pitch 0 the ground plane approaches row `cy` = 300 at infinite range, and the
    bottom row looks at ground 3.5 m ahead of the EGO (2.0 m ahead of the lens). The second
    number is what `arc_weights` is given as `s_min`, so if it is wrong the distance
    weighting is spent on ground that does not exist.
    """
    _, v_far, _ = CUE_CAMERA.project_ground(np.array([[1e6, 0.0]]))
    assert v_far[0] == pytest.approx(CUE_CAMERA.horizon_row(), abs=1e-3)
    assert CUE_CAMERA.horizon_row() == pytest.approx(300.0)
    assert CUE_CAMERA.min_visible_range_m() == pytest.approx(3.5, abs=1e-6)
    _, v_edge, vis_edge = CUE_CAMERA.project_ground(np.array([[3.51, 0.0]]))
    assert bool(vis_edge[0]) and v_edge[0] < CUE_CAMERA.height
    _, _, vis_in = CUE_CAMERA.project_ground(np.array([[3.49, 0.0]]))
    assert not bool(vis_in[0])


def test_a_point_behind_the_camera_is_never_a_pixel():
    """Rejected BEFORE the divide. A point just behind the lens has a small negative depth,
    and dividing by it produces a large-magnitude but perfectly in-frame-looking pixel with
    the sign flipped -- so it would read as a confident label for ground that is behind the
    vehicle. No inf or nan may escape either.
    """
    pts = np.array([[1.0, 0.0], [1.5, 0.0], [1.49, 0.3], [-5.0, 0.0]])
    u, v, vis = CUE_CAMERA.project_ground(pts)
    assert not vis.any()
    assert np.isfinite(u).all() and np.isfinite(v).all()


def test_the_blind_zone_of_the_cue_camera_matches_its_geometry():
    """The cue camera's blind zone, pinned so it cannot drift silently.

    Against the cue camera's geometry (fov 90, pitch 0, 1.5 m up, 1.5 m forward), 20 of the
    65 fan arcs have NO visible sample at all and mean coverage is 0.345. The binding limit
    is lateral (`|y| < x - 1.5`), not the near cut, so it falls on exactly the arcs that
    turn hardest. That is why the terrain arm does not reuse this camera.
    """
    _, _, vis = CUE_CAMERA.project_ground(fan_points())
    cov = vis.mean(axis=1)
    assert cov.mean() == pytest.approx(0.345, abs=0.005)
    assert int((cov == 0).sum()) == 20
    k = curvature_fan(FAN["kappa_max"], FAN["K"])
    assert np.abs(k[cov == 0]).min() == pytest.approx(0.252, abs=0.005)
    straight = int(np.argmin(np.abs(k)))
    assert int(vis[straight].sum()) == 11


def test_the_terrain_camera_has_no_blind_arc():
    """The whole justification for a second, wider, tilted camera instead of a cleverer
    score. With no fully-blind arc, "unknown" can be dropped from the cost without letting a
    prohibition be satisfied by turning where nothing was seen.

    If this ever fails, the scoring rule is no longer safe and the mount must be fixed --
    not the rule.
    """
    _, _, vis = TERRAIN_CAMERA.project_ground(fan_points())
    cov = vis.mean(axis=1)
    assert int((cov == 0).sum()) == 0
    # 0.576 at the front-most mount (fov 130 / pitch 30). The naive front mount -- same
    # position, the old fov 110 / pitch 15 -- gives 0.388 with 20 blind arcs, so this
    # assertion is what stops the lens silently reverting and taking the drop rule's
    # justification with it.
    assert cov.mean() > 0.55
    assert TERRAIN_CAMERA.mount_x > 2.0          # at the front, not mid-bonnet
    assert TERRAIN_CAMERA.occluded_below_m == 0.0  # nothing is in front of it


def test_tilting_down_sees_nearer_ground_and_raises_the_horizon():
    """A sign test, and not an optional one: with the pitch sign inverted the camera looks
    UP, coverage falls, and every number downstream still looks reasonable.

    Tilting DOWN moves the scene UP the image, so the horizon ROW falls toward 0.
    """
    level = CameraModel()
    tilted = CameraModel(pitch_down_deg=15.0)
    assert tilted.min_visible_range_m() < level.min_visible_range_m()
    assert tilted.horizon_row() < level.horizon_row()


@pytest.mark.parametrize("pitch", [0.0, 5.0, 15.0, 30.0])
def test_the_horizon_matches_the_projection_when_the_camera_is_tilted(pitch):
    """`horizon_row()` must agree with `project_ground` at every pitch.

    `horizon_row()` is a closed form; `project_ground` is the thing that actually places
    pixels. They are two statements of one geometry and nothing else forces them to agree.
    Testing only pitch 0.0 is not enough: there `tan(0) == 0` and an inverted pitch sign
    still gives `cy`. This test checks every pitch, and asserts the direction too: no ground
    point in front of the camera may ever project ABOVE the horizon.
    """
    cam = CameraModel(pitch_down_deg=pitch)
    _, v_far, _ = cam.project_ground(np.array([[1e7, 0.0]]))
    assert v_far[0] == pytest.approx(cam.horizon_row(), abs=1e-2)
    ranges = np.stack([np.geomspace(4.0, 5e4, 60), np.zeros(60)], axis=1)
    _, v, _ = cam.project_ground(ranges)
    assert (v >= cam.horizon_row() - 1e-6).all()
    assert (np.diff(v) <= 1e-9).all()        # farther is always higher up the image


# --------------------------------------------------------------------------- #
# Staleness                                                                    #
# --------------------------------------------------------------------------- #

def test_a_world_fixed_point_projects_identically_from_two_poses():
    """The staleness transform, tested against the only thing that is actually invariant:
    a point fixed in the world must land on the same pixel whichever body frame it is
    expressed in, once carried back to the pose the frame was taken from.
    """
    pose_cap = (12.0, -3.0, 0.4)
    pose_now = (14.1, -2.2, 0.55)                      # 2.2 m and 8.6 deg later
    world = np.array([[20.0, 1.0], [25.0, -4.0], [18.0, -1.0]])

    def to_body(p, pose):
        x, y, th = pose
        R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        return (p - np.array([x, y])) @ R

    in_now = to_body(world, pose_now)
    carried = body_to_body(in_now, pose_now, pose_cap)
    np.testing.assert_allclose(carried, to_body(world, pose_cap), atol=1e-9)


def test_skipping_the_transform_moves_the_pixels_a_lot():
    """Without this, the test above passes for a transform that does nothing. 2.2 m of
    travel at this range is tens of pixels, which is the whole reason the mask carries a pose.
    """
    pose_cap = (12.0, -3.0, 0.4)
    pose_now = (14.1, -2.2, 0.55)
    pts = np.array([[8.0, 0.0], [10.0, 2.0]])
    u_c, v_c, _ = TERRAIN_CAMERA.project_ground(body_to_body(pts, pose_now, pose_cap))
    u_n, v_n, _ = TERRAIN_CAMERA.project_ground(pts)
    assert np.max(np.hypot(u_c - u_n, v_c - v_n)) > 20.0


@pytest.mark.parametrize("age,reason", [(5.0, "old"), (None, "no frame")])
def test_a_stale_or_absent_mask_yields_unobserved_not_the_last_good_labels(age, reason):
    """Reusing a 3 s old frame at 5 m/s describes ground 15 m behind the vehicle while still
    reporting terrain-scored numbers. All-UNOBSERVED and a zero coverage is the honest answer.
    """
    src = MaskTerrainSource(TERRAIN_CAMERA, max_frame_age_s=1.0)
    if age is not None:
        src.update(np.full((TERRAIN_CAMERA.height, TERRAIN_CAMERA.width), ROAD, np.int16),
                   (0.0, 0.0, 0.0), stamp=0.0)
        classify = src.source_for((0.0, 0.0, 0.0), now=age)
    else:
        classify = src.source_for((0.0, 0.0, 0.0), now=0.0)
    out = classify(fan_points())
    assert (out == UNOBSERVED).all()
    assert src.coverage_frac == 0.0
    assert reason in src.last_reason


def test_a_mask_of_the_wrong_size_is_refused():
    """Projecting into the wrong image size samples the wrong pixel EVERYWHERE and reads as
    a segmentation error rather than a configuration one.
    """
    src = MaskTerrainSource(TERRAIN_CAMERA)
    with pytest.raises(ValueError, match="does not match the camera model"):
        src.update(np.zeros((480, 640), np.int16), (0.0, 0.0, 0.0), stamp=0.0)


# --------------------------------------------------------------------------- #
# What an unobserved sample MEANS                                               #
# --------------------------------------------------------------------------- #

def _cfg(**kw):
    base = dict(K=FAN["K"], n=FAN["n"], arc_len_m=FAN["arc_len"],
                kappa_max=FAN["kappa_max"], terrain_costs={ROAD: 0.0, GRASS: 1.0,
                                                           SIDEWALK: 0.25, OTHER: 1.0},
                unobserved_class=UNOBSERVED,
                # OPTS OUT OF THE TURN CAP, which became a default elsewhere. This FAN is
                # (10.0, 0.35) = 201 deg of turn, which `max_turn_rad` now rejects. These
                # tests are about the CAMERA terrain source, not arc geometry, and the
                # numbers asserted in this file (coverage 0.69 straight / 0.00 at
                # kappa_max, 20 of 65 arcs blind) are for this fan -- so the geometry is
                # held exactly and only the turn-cap check is waived. Adopting the capped
                # pair here would silently re-baseline every number in this file.
                max_turn_rad=None)
    base.update(kw)
    return TerrainMpcConfig(**base)


def test_an_arc_is_not_penalised_merely_for_leaving_the_frame():
    """THE FOV-BIAS TEST. Coverage is a function of curvature -- 0.69 straight, 0.00 at
    |kappa| >= 0.25 on the cue camera -- so charging `unknown_cost` to out-of-frame samples
    is arithmetically an anti-turn tax, applied to ground the sensor never observed and
    attributed to terrain.

    With a uniformly-road mask, every arc that sees ANYTHING must score identically.
    """
    mask = np.full((CUE_CAMERA.height, CUE_CAMERA.width), ROAD, np.int16)
    src = MaskTerrainSource(CUE_CAMERA)
    src.update(mask, (0.0, 0.0, 0.0), stamp=0.0)
    cfg = _cfg(terrain_class=src.source_for((0.0, 0.0, 0.0), now=0.0),
               terrain_decay_m=4.0, terrain_s_min_m=CUE_CAMERA.min_visible_range_m())
    res = plan_step_terrain(cfg, bearing_ref=0.0)
    # 20 arcs see nothing at all on this camera and take `unknown_cost`; the rest see only
    # road and must all score exactly 0.0, whatever fraction of them was visible.
    assert res.blind_arcs == 20
    assert res.terrain_soft == pytest.approx(0.0)


def test_an_arc_that_sees_nothing_at_all_is_not_scored_as_clean_road():
    """The hazard in the opposite direction, and the reason the drop rule is not unconditional.

    If a zero-coverage arc scored 0.0 it would rank EQUAL to a verified-clean road arc, and
    since UNOBSERVED can never be forbidden, "never cross the grass" would be satisfiable by
    turning into ground the camera never saw -- a prohibition that silently does not bind.
    """
    mask = np.full((CUE_CAMERA.height, CUE_CAMERA.width), ROAD, np.int16)
    src = MaskTerrainSource(CUE_CAMERA)
    src.update(mask, (0.0, 0.0, 0.0), stamp=0.0)
    cfg = _cfg(terrain_class=src.source_for((0.0, 0.0, 0.0), now=0.0), unknown_cost=1.0)
    klass = cfg.terrain_class(fan_points())
    blind = (klass == UNOBSERVED).all(axis=1)
    assert blind.sum() == 20
    res = plan_step_terrain(cfg, bearing_ref=0.0)
    assert res.blind_arcs == 20
    # PARTIAL blindness now costs too, in proportion to the unseen weight. On an all-road
    # mask a fully-seen arc scores 0.0 and a half-seen one scores about 0.5 -- so the chosen
    # arc's cost is its unseen fraction, not zero. Renormalising onto whatever the camera
    # happened to see would score a barely-seen fan extreme identically to a well-seen
    # straight arc, biasing the choice toward the extremes.
    assert 0.0 < res.terrain_soft < 1.0
    per_arc = res.arc_soft
    assert per_arc[blind].min() == pytest.approx(1.0), "a fully blind arc pays in full"
    sighted = ~blind
    assert per_arc[sighted].max() < 1.0, "a sighted arc on clean road must beat a blind one"


def test_a_dead_camera_leaves_the_choice_exactly_where_bearing_puts_it():
    """THE REGRESSION THAT MATTERS MOST.

    Every arc blind at once must not veto the fan. Vetoing all K hands over to the
    relax-everything fallback, which knows nothing about where the hazard is and leaves the
    vehicle circling (the same failure as the all-arcs obstacle veto). Here every arc takes the SAME `unknown_cost`, so the terrain term is a constant
    shift, the argmax is untouched, and the controller is bearing-only -- loudly, via
    `blind_arcs == 65` and `coverage_frac == 0`, not silently.
    """
    src = MaskTerrainSource(TERRAIN_CAMERA, max_frame_age_s=1.0)      # never fed a frame
    cfg = _cfg(terrain_class=src.source_for((0.0, 0.0, 0.0), now=0.0),
               forbid_classes=frozenset({GRASS}))
    dead = plan_step_terrain(cfg, bearing_ref=0.3)

    flat = _cfg(terrain_class=lambda p: np.full(p.shape[:2], ROAD, dtype=int),
                w_terrain=0.0, forbid_classes=frozenset({GRASS}))
    bearing_only = plan_step_terrain(flat, bearing_ref=0.3)

    assert dead.blind_arcs == FAN["K"]
    assert dead.coverage_frac == 0.0
    # -1, not 0: with `min_adjudicable_frac` active a dead camera makes every arc
    # UNADJUDICABLE, so the veto set is everything and the ordered fallback relaxes it. The
    # sentinel means "nothing could be certified"; 0 would wrongly mean "nothing was
    # forbidden". What must NOT change is the outcome: relaxing
    # means `scores[bad_terrain] = -inf` is skipped, so the choice is still bearing-only.
    assert dead.terrain_relaxed
    assert dead.forbidden_rejected == -1
    assert not dead.recovery
    assert dead.v == pytest.approx(bearing_only.v)
    assert dead.omega == pytest.approx(bearing_only.omega)


def test_unobserved_can_never_be_forbidden():
    """Refused at construction, in the same spirit as `terrain_class=None`: a config that
    would run and measure the wrong thing does not get to exist.
    """
    with pytest.raises(ValueError, match="unobserved_class"):
        _cfg(terrain_class=lambda p: np.zeros(p.shape[:2], int),
             forbid_classes=frozenset({UNOBSERVED}))


# --------------------------------------------------------------------------- #
# The distance weighting                                                       #
# --------------------------------------------------------------------------- #

def test_the_default_weighting_is_exactly_the_mean_it_replaces():
    """Additive: the default is the unweighted mean, so existing behaviour stays
    reproducible.
    """
    w = arc_weights(10.0, 16, None)
    np.testing.assert_allclose(w, np.full(16, 1.0 / 16))
    cost = np.random.default_rng(0).random((65, 16))
    np.testing.assert_allclose(cost @ w, cost.mean(axis=1))


def test_the_weighting_starts_where_the_camera_starts_seeing():
    """Decaying from s = 0 would put most of the weight on the near band that the camera
    structurally cannot see (the first 5 of 16 samples on EVERY arc), so the term would look
    tuned and be inert. Weight below `s_min` must be exactly zero.
    """
    s_min = CUE_CAMERA.min_visible_range_m()
    w = arc_weights(10.0, 16, decay_m=4.0, s_min=s_min)
    s = np.linspace(0.0, 10.0, 17)[1:]
    assert (w[s < s_min] == 0.0).all()
    assert (w[s >= s_min] > 0.0).all()
    assert w.sum() == pytest.approx(1.0)
    covered = w[s >= s_min]
    assert (np.diff(covered) < 0).all()          # nearer is heavier


def test_near_terrain_outweighs_far_terrain():
    """Decision 2, tested rather than configured: the same patch of grass must cost more
    when it is about to be driven over than when it is at the end of the arc.
    """
    s_min = CUE_CAMERA.min_visible_range_m()
    w = arc_weights(10.0, 16, decay_m=4.0, s_min=s_min)
    s = np.linspace(0.0, 10.0, 17)[1:]
    near = np.argmin(np.abs(s - 4.5))
    far = np.argmin(np.abs(s - 9.5))
    cost_near = np.zeros(16)
    cost_near[near] = 1.0
    cost_far = np.zeros(16)
    cost_far[far] = 1.0
    assert cost_near @ w > cost_far @ w


def test_an_arc_that_ends_before_the_camera_sees_is_refused_not_scored_zero():
    """All weight below `s_min` means the terrain term is identically zero while still being
    reported -- a silent no-op. Refuse it instead.
    """
    with pytest.raises(ValueError, match="no sample"):
        arc_weights(2.0, 16, decay_m=4.0, s_min=3.5)


# --------------------------------------------------------------------------- #
# The arm is not inert                                                         #
# --------------------------------------------------------------------------- #

def test_grass_across_the_straight_ahead_moves_the_chosen_arc_away_from_it():
    """Guards against the whole arm being wired and doing nothing -- a code
    path that has never executed.

    Two setups look like they should turn the vehicle but correctly do not:

    (1) Grass over the image's left HALF, expecting a right turn. It does not turn, and it
    should not: `u = cx - fx*y/(x-1.5)`, so column `cx` IS the vehicle's centreline and the
    straight arc reads clean road the whole way. The grass has to cover the straight-ahead
    before avoiding it means anything -- the same trap as siting a prohibition on a route
    where every controller drives identically.

    (2) `w_terrain=5` with the grass edge 120 px past centre, still no turn. Also correct.
    `w_progress` multiplies a displacement in METRES, so over a 10 m arc that term spans ~11
    while the terrain cost is bounded by `w_terrain * 1.0`. Chosen curvature against grass
    edge in px right of centre:

        edge  +0 px :  straight at every w_terrain      (the straight arc is genuinely clean)
        edge +60 px :  0.000 at wT=1,  -0.098 from wT=3 up
        edge +120 px :  0.000 up to wT=5,  -0.186 from wT=10 up

    So the terrain term only outranks progress above `w_terrain` ~ 3, and the further the
    grass reaches the more weight it takes to buy the bigger detour. That is a TUNING fact
    about the pair of weights, not a defect -- and it is why the hard veto below exists: a
    prohibition must bind without anyone having chosen a number.
    """
    h, w = TERRAIN_CAMERA.height, TERRAIN_CAMERA.width
    road = np.full((h, w), ROAD, np.int16)
    past_centre = road.copy()
    past_centre[:, : w // 2 + 60] = GRASS         # covers straight ahead and everything left

    def choose(mask, w_terrain):
        src = MaskTerrainSource(TERRAIN_CAMERA)
        src.update(mask, (0.0, 0.0, 0.0), stamp=0.0)
        cfg = _cfg(terrain_class=src.source_for((0.0, 0.0, 0.0), now=0.0),
                   terrain_decay_m=4.0,
                   terrain_s_min_m=TERRAIN_CAMERA.min_visible_range_m(),
                   w_terrain=w_terrain)
        r = plan_step_terrain(cfg, bearing_ref=0.0)
        return r.omega / r.v if r.v > 1e-6 else 0.0

    assert choose(road, 3.0) == pytest.approx(0.0)        # nothing to avoid -> straight
    assert choose(past_centre, 3.0) < -1e-3               # steers RIGHT, away from the grass
    # And it is the grass doing it, not the weight: at w_terrain=0 the same mask goes straight.
    assert choose(past_centre, 0.0) == pytest.approx(0.0)


def test_forbidding_grass_removes_the_arcs_that_cross_it():
    """Prevention rather than detection: `hard_forbid` must change the trajectory when the
    forbidden class lies on the preferred arc. In map-based terrain grass is not part of
    the drivable surface at all; from pixels it is.
    """
    h, w = TERRAIN_CAMERA.height, TERRAIN_CAMERA.width
    half = np.full((h, w), ROAD, np.int16)
    half[:, : w // 2] = GRASS
    src = MaskTerrainSource(TERRAIN_CAMERA)
    src.update(half, (0.0, 0.0, 0.0), stamp=0.0)
    cfg = _cfg(terrain_class=src.source_for((0.0, 0.0, 0.0), now=0.0),
               forbid_classes=frozenset({GRASS}))
    res = plan_step_terrain(cfg, bearing_ref=0.0)
    assert res.forbidden_rejected > 0
    assert not res.recovery                      # a legal alternative existed


# --------------------------------------------------------------------------- #
# The LIVE overlay path, exercised without a CARLA server                      #
# --------------------------------------------------------------------------- #

def test_the_live_overlay_renders_what_the_scorer_decided():
    """The MPC's per-tick overlay, end to end, on a real captured mask.

    WHY THIS TEST EXISTS. The node's `_write_overlay` is the one piece of the live path that
    a smoke run would exercise LAST and that fails softly by design -- it catches its own
    exceptions so a bad render can never stop the vehicle. That is the right behaviour, but
    it also means a broken renderer goes unnoticed while every run "succeeds".

    So the path is driven here instead: a real mask off disk, a real `plan_step_terrain`, and
    the per-arc verdicts it returns handed to the same `render_fan` the node calls. If
    `MpcResult` stops carrying `arc_soft` / `arc_forbidden` / `best_index`, or the renderer's
    signature drifts, this fails offline in a second rather than silently at runtime.
    """
    import json

    from carla_gt_bridge.terrain_classes import DEFAULT_COSTS, from_carla_tags
    from carla_gt_bridge.terrain_overlay import render_fan
    from dgppo_ros_node_pkg.terrain_mpc import arc_weights, curvature_fan

    # PKG is .../src/dgppo_ros_node_pkg, so the sibling package is one level up, not two.
    # Getting this wrong makes the test find nothing and skip, which is the very failure
    # it exists to prevent.
    frames = os.path.join(os.path.dirname(PKG), "carla_gt_bridge",
                          "reports", "terrain_frames", "Town05_front")
    meta_p = os.path.join(frames, "poses.json")
    if not os.path.exists(meta_p):
        pytest.skip(f"no captured frames at {frames}")
    g = json.load(open(meta_p))["cameras"]["sem_f130_p30"]
    cam = CameraModel(width=g["width"], height=g["height"], fov_deg=g["fov"],
                      mount_x=g["x"], mount_z=1.573, pitch_down_deg=g["pitch_down_deg"])
    mask = from_carla_tags(np.load(os.path.join(frames, "0012_sem_f130_p30_tags.npy")))

    src = MaskTerrainSource(cam)
    src.update(mask, (0.0, 0.0, 0.0), stamp=0.0)
    cfg = TerrainMpcConfig(
        K=65, n=16, arc_len_m=10.0, kappa_max=0.35, max_turn_rad=None,
        terrain_class=src.source_for((0.0, 0.0, 0.0), now=0.0),
        terrain_costs=dict(DEFAULT_COSTS), unobserved_class=UNOBSERVED,
        terrain_decay_m=4.0, terrain_s_min_m=cam.effective_min_range_m(),
        w_terrain=5.0, forbid_classes=frozenset({SIDEWALK}))
    res = plan_step_terrain(cfg, bearing_ref=-0.5)

    # The fields the node hands to the renderer must actually be there.
    assert res.arc_soft is not None and res.arc_soft.shape == (65,)
    assert res.arc_forbidden is not None and res.arc_forbidden.shape == (65,)
    assert 0 <= res.best_index < 65
    assert res.arc_forbidden.any(), "this frame should veto some arcs, or the test is vacuous"

    kappa = curvature_fan(cfg.kappa_max, cfg.K)
    dense = arc_rollout_k(np.zeros(3), kappa, cfg.arc_len_m, 200)[:, 1:, :2]
    w_arc = arc_weights(cfg.arc_len_m, cfg.n, cfg.terrain_decay_m, cfg.terrain_s_min_m)
    out = os.path.join(os.environ.get("PYTEST_TMP", "/tmp"), "live_overlay_test.png")
    render_fan(mask, cam, dense, kappa, w_arc, int(res.best_index),
               res.arc_forbidden, res.arc_soft, out, title="live path test")
    assert os.path.exists(out) and os.path.getsize(out) > 10_000


# --------------------------------------------------------------------------- #
# Separating "what is there" from "how much is known", and closing the escape  #
# --------------------------------------------------------------------------- #

def test_terrain_cost_is_an_expectation_over_what_was_actually_seen():
    """The terrain signal must not be drowned by the unknown charge.

    The form `sum(cost*w over observed) + (1-seen)*unknown_cost` mixes two different
    quantities into one number; on real frames the unseen share can dominate, so a
    terrain-scored controller would rank candidates almost entirely by how much of each
    falls inside the lens. The terrain part is therefore a proper
    expectation -- the weighted mean over the samples that WERE observed -- and ignorance is
    priced by its own weight, so the two can be read and tuned separately.
    """
    w = np.full(4, 0.25)
    cost = np.array([[0.0, 1.0, 0.0, 1.0]])              # half the arc costly
    klass = np.array([[ROAD, GRASS, ROAD, GRASS]])
    soft, blind, seen = terrain_soft_costs(klass, cost, w, UNOBSERVED, 1.0, w_unknown=0.0)
    assert not blind[0]
    assert seen[0] == pytest.approx(1.0)
    assert soft[0] == pytest.approx(0.5), "fully seen: the mean of what is there"

    # Same terrain, but only half the arc observed. The EXPECTATION is unchanged.
    klass2 = np.array([[ROAD, GRASS, UNOBSERVED, UNOBSERVED]])
    cost2 = np.array([[0.0, 1.0, 1.0, 1.0]])
    soft2, _, seen2 = terrain_soft_costs(klass2, cost2, w, UNOBSERVED, 1.0, w_unknown=0.0)
    assert seen2[0] == pytest.approx(0.5)
    assert soft2[0] == pytest.approx(0.5), "the evidence says the same thing either way"


def test_ignorance_is_priced_separately_and_does_not_swamp_terrain():
    """`w_unknown` is its own knob. At the default it must be comparable to a sidewalk
    crossing, not to a wall -- `unknown_cost=1.0` would make "not observed" exactly as
    bad as "structure", letting the unseen term dominate.
    """
    w = np.full(4, 0.25)
    clean_half_seen = terrain_soft_costs(
        np.array([[ROAD, ROAD, UNOBSERVED, UNOBSERVED]]), np.zeros((1, 4)), w,
        UNOBSERVED, 1.0, w_unknown=0.5)[0][0]
    sidewalk_all_seen = terrain_soft_costs(
        np.array([[SIDEWALK] * 4]), np.full((1, 4), 0.25), w, UNOBSERVED, 1.0,
        w_unknown=0.5)[0][0]
    assert clean_half_seen == pytest.approx(0.25)
    assert sidewalk_all_seen == pytest.approx(0.25)
    # and a fully blind arc still pays in full -- that case has no evidence at all
    blind_soft, blind_flag, _ = terrain_soft_costs(
        np.array([[UNOBSERVED] * 4]), np.ones((1, 4)), w, UNOBSERVED, 1.0, w_unknown=0.5)
    assert blind_flag[0] and blind_soft[0] == pytest.approx(1.0)


def test_an_arc_cannot_become_legal_by_leaving_the_field_of_view():
    """THE ESCAPE, closed here.

    With sidewalk banned on a real Town05 frame, the veto without an adjudicability rule
    forms a BAND:

        right turn                  straight                  left turn
        ..........VVVVVVVVVVVVVVV........................................
        ##########++++++                                 ++++++##########   (# = cov<0.4)

    The middle-right arcs are vetoed. Arcs that turn HARDER right -- further into the kerb --
    are not vetoed at all, because they leave the lens before reaching it and UNOBSERVED can
    never be forbidden. The prohibition is escapable by steering somewhere unseen,
    with only a soft cost objecting.

    So when a forbid set is active an arc must also be ADJUDICABLE: enough of it observed for
    "no forbidden sample" to mean anything. Below `min_adjudicable_frac` it is not certified
    legal. The ordered fallback still protects against vetoing everything.
    """
    cfg = _cfg(terrain_class=lambda p: np.full(p.shape[:2], UNOBSERVED, dtype=int),
               forbid_classes=frozenset({SIDEWALK}), min_adjudicable_frac=0.5)
    res = plan_step_terrain(cfg, bearing_ref=0.0)
    # Everything unadjudicable -> the fallback engages rather than freezing.
    assert res.terrain_relaxed
    assert not res.recovery

    # A frame where half the fan is well seen: the poorly-seen arcs must NOT be certified.
    mask = np.full((TERRAIN_CAMERA.height, TERRAIN_CAMERA.width), ROAD, np.int16)
    src = MaskTerrainSource(CUE_CAMERA)          # the narrow lens: 20 of 65 arcs are blind
    src.update(np.full((CUE_CAMERA.height, CUE_CAMERA.width), ROAD, np.int16),
               (0.0, 0.0, 0.0), stamp=0.0)
    cfg2 = _cfg(terrain_class=src.source_for((0.0, 0.0, 0.0), now=0.0),
                forbid_classes=frozenset({SIDEWALK}), min_adjudicable_frac=0.5)
    r2 = plan_step_terrain(cfg2, bearing_ref=0.0)
    assert r2.forbidden_rejected > 0, "unadjudicable arcs must not pass as legal"
    assert not r2.terrain_relaxed, "well-seen arcs remain, so nothing should relax"
    assert mask.size > 0
