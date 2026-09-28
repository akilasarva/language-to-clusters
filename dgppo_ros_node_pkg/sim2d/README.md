# 2D comparison rig

Drives several controllers through the same randomised bridge geometries, with the same
LiDAR and the same map distortion, and draws them on one figure. A run is ~5 s against
~3 min for a CARLA mission.

    python3 compare_controllers.py --n 5        # figures -> results/
    streamlit run compare_viz_app.py            # interactive, localhost:8501

## The arms

| arm | what it is |
|---|---|
| Carson NMPC | CasADi baseline tracking metric waypoints |
| Ours-metric | ablation: distance to the DISTORTED centroid |
| Ours-cluster | ablation: Voronoi cluster of the rollout endpoint |
| Ours (graded obstacle) | as above, with a graded obstacle penalty instead of a veto |
| MPPI | weighted average over samples rather than argmax |
| **Terrain-MPC 1-seg** | **the SHIPPED CARLA controller**, imported from `terrain_mpc` |
| **Terrain-MPC 2-seg** | the same, with the opt-in two-segment fan |

The last two call `dgppo_ros_node_pkg.terrain_mpc.plan_step_terrain` directly rather than
reimplementing it, so the rig cannot drift from the shipped controller. Note the two
primitives differ: the rig's own arms sample 500 random per-step (v, omega) pairs over
~1.2 m, while the shipped controller sweeps a deterministic 65-arc curvature fan in closed
form over 10 m of ARC LENGTH.

## Caveats

**Scale.** Every length in `TerrainMpcConfig` is metres in a world with a 14 m road; this
world has 0.26-0.55 m gaps. `scale` maps them, and `kappa_max` scales INVERSELY, being
1/metres. Copying the metre values unchanged puts a 10 m arc in a 0.5 m corridor.

**Reach is a confound, not a constant.** The arms do not look equally far unless you make
them: `run_ours` reaches `v_max * N * dt` (1.20 m at N=8, longer than the whole bridge)
and the terrain arms reach `arc_len_m`, so an unequalised comparison can differ in reach
by several times. `horizon_n` and `scale` are how you equalise them.

**This world is a narrow slot; a CARLA road is not.** Arc-to-free-width is 0.33-0.71 in
CARLA and passes 1.0 here at about 0.40 m of reach -- which is exactly where the terrain
arms collapse, because a long arc in a slot ends on a wall whichever way it curves. So the
rig is good for comparing PRIMITIVES at matched reach, and cannot tell you the right arc
length for CARLA.

**`map_free_corridor`.** The default corridor term resolves its PCA sign against the TRUE
map perpendicular, its axis against the true map axis, and falls back to the true bridge
centre when hits are sparse -- while being labelled map-free. Under distortion those are
undistorted truth. Set `map_free_corridor` to use the map-free estimator; the legacy branch
is kept for reproducibility. Two further copies of the map-reading
block remain in `run_ours_dead_reckoning` and `run_mppi`.

The 2-D sampling MPC and region associator the rig uses are in `dgppo_ros_node_pkg/vendor/`.
