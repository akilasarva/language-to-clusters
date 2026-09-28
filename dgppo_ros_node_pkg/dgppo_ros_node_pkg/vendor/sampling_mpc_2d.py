"""
sampling_mpc.py
===============
Sampling-Based MPC controller for autonomous navigation in DGPPO environments.

Replaces the RNN DGPPO policy with a memoryless, shooting-method planner.
Operates entirely in a rolling ego-centric local frame.

Robot model: first-order unicycle (v, omega commands), matching Unitree Go2 /
Spot cmd_vel interface. The sim adapter (→ double-integrator ax, ay) lives only
in run_mpc_episode, not in plan().

Identical plan() interface in sim and on hardware; only the source of inputs
(env graph vs. external oracle) differs.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt


# ─── Layer 1: Vectorized First-Order Unicycle Kinematics ──────────────────────

def unicycle_rollout(
    x0: np.ndarray,            # (3,)   [x=0, y=0, yaw=0] — always ego-origin
    control_seqs: np.ndarray,  # (K, N, 2)  [v, omega] per sample and step
    dt: float = 0.2,
) -> np.ndarray:               # (K, N+1, 3)  trajectory including initial state
    """
    Vectorized first-order unicycle rollout.

    The K=500 samples are fully parallel (independent); the N=8 horizon steps
    are sequential (each state depends on the previous). The inner loop runs N
    times on (K,) NumPy arrays — effectively 500x parallelism at each step.

    Dynamics (as in the Carson NMPC baseline's waypoint follower, simplified to
    first-order since Unitree/Spot accept direct velocity commands):
        x   += v * cos(yaw) * dt
        y   += v * sin(yaw) * dt
        yaw += omega * dt
    """
    K, N, _ = control_seqs.shape
    buf = np.empty((K, N + 1, 3), dtype=np.float64)
    buf[:, 0, :] = x0  # broadcast x0 to all K samples

    for n in range(N):
        v   = control_seqs[:, n, 0]    # (K,)
        om  = control_seqs[:, n, 1]    # (K,)
        yaw = buf[:, n, 2]             # (K,)
        buf[:, n + 1, 0] = buf[:, n, 0] + v * np.cos(yaw) * dt
        buf[:, n + 1, 1] = buf[:, n, 1] + v * np.sin(yaw) * dt
        buf[:, n + 1, 2] = yaw + om * dt

    return buf   # (K, N+1, 3)


# ─── Layer 2: Local Occupancy Grid + Euclidean Distance Field ─────────────────

class LocalOccupancyGrid:
    """
    Converts robot-frame LiDAR hit points to a 2D Euclidean Distance Field.

    All coordinates are in the robot-local frame: robot is always at (0, 0).
    The grid covers ±grid_size meters in both x and y.

    Default: grid_size=0.6 m, resolution=0.01 m → 120×120 cell grid.
    (comm_radius in the DGPPO env is 0.5 m, so 0.6 m gives a small margin.)
    """

    def __init__(self, grid_size: float = 0.6, resolution: float = 0.01):
        self.grid_size  = grid_size
        self.resolution = resolution
        self.n_cells    = int(2 * grid_size / resolution)
        # Distance field in meters; inf = no obstacle within grid window
        self.dist_map   = np.full((self.n_cells, self.n_cells), np.inf)

    # ------------------------------------------------------------------
    def update(self, lidar_hits_local: Optional[np.ndarray]) -> None:
        """
        Rebuild the distance field from new LiDAR hits.

        lidar_hits_local: (M, 2) robot-frame hit positions, or None / empty.
        """
        occ = np.zeros((self.n_cells, self.n_cells), dtype=bool)

        if lidar_hits_local is not None and len(lidar_hits_local) > 0:
            col = np.floor(
                (lidar_hits_local[:, 0] + self.grid_size) / self.resolution
            ).astype(int)
            row = np.floor(
                (lidar_hits_local[:, 1] + self.grid_size) / self.resolution
            ).astype(int)
            valid = (
                (col >= 0) & (col < self.n_cells) &
                (row >= 0) & (row < self.n_cells)
            )
            occ[row[valid], col[valid]] = True

        if occ.any():
            # EDT: distance in meters from each free cell to nearest obstacle
            self.dist_map = distance_transform_edt(~occ) * self.resolution
        else:
            self.dist_map = np.full((self.n_cells, self.n_cells), np.inf)

    # ------------------------------------------------------------------
    def check_collisions(self, trajectories: np.ndarray) -> np.ndarray:
        """
        Vectorized obstacle-distance lookup for all trajectory waypoints.

        trajectories: (K, N, 3) [x, y, yaw] in robot local frame.
        Returns:      (K, N) distance in meters to nearest obstacle.
                      Points outside the grid window → 0.0 (treated as obstacle).
        """
        xy  = trajectories[:, :, :2]   # (K, N, 2)
        col = np.floor(
            (xy[:, :, 0] + self.grid_size) / self.resolution
        ).astype(int)                  # (K, N)
        row = np.floor(
            (xy[:, :, 1] + self.grid_size) / self.resolution
        ).astype(int)

        in_bounds = (
            (col >= 0) & (col < self.n_cells) &
            (row >= 0) & (row < self.n_cells)
        )
        col_c = np.clip(col, 0, self.n_cells - 1)
        row_c = np.clip(row, 0, self.n_cells - 1)

        dist = self.dist_map[row_c, col_c]          # (K, N)
        # Out-of-bounds = unknown obstacle (conservative)
        return np.where(in_bounds, dist, 0.0)


# ─── Layer 3 + 4: SamplingMPC Orchestrator ────────────────────────────────────

class SamplingMPC:
    """
    Shooting-method MPC controller.

    Topological cluster identity is the primary reward; obstacle avoidance is a
    hard constraint (collision mask). The planner runs entirely in robot-local
    coordinates — no global references inside plan().

    Sim usage:
        mpc = SamplingMPC()
        mpc.reset(associator)           # once per episode
        v, w = mpc.plan(...)            # each control step

    Real-robot usage (no BehaviorAssociator):
        mpc = SamplingMPC()             # no reset() needed
        v, w = mpc.plan(...)            # pass centroids from external oracle
    """

    # Velocity / omega limits from the Carson NMPC baseline's waypoint follower
    V_MAX:     float = 0.75   # m/s  (forward only; no reverse)
    OMEGA_MAX: float = 1.5    # rad/s

    def __init__(
        self,
        K: int   = 500,    # number of random rollouts per planning step
        N: int   = 8,      # horizon steps
        dt: float = 0.2,   # seconds per step (matches the baseline's default)
        safety_radius: float = 0.10,     # meters — half robot body radius
        lidar_grid_size: float = 0.6,    # meters — half-width of local EDT window
        lidar_resolution: float = 0.01,  # meters per cell
        # Reward weights
        w_target:    float =  10.0,   # bonus for landing in goal cluster
        w_start:     float =   1.0,   # bonus for staying in start cluster
        w_forbidden: float = -15.0,   # penalty for straying into wrong cluster
        w_bearing:   float =   3.0,   # bearing alignment
        w_direction: float =   1.0,   # soft directional pull toward goal centroid
    ):
        self.K             = K
        self.N             = N
        self.dt            = dt
        self.safety_radius = safety_radius
        self.w_target      = w_target
        self.w_start       = w_start
        self.w_forbidden   = w_forbidden
        self.w_bearing     = w_bearing
        self.w_direction   = w_direction

        self._occ_grid = LocalOccupancyGrid(lidar_grid_size, lidar_resolution)

        # Populated by reset(); None until then (real-robot mode: stays None,
        # caller passes all_centroids_local directly to plan())
        self._centroids_global: Optional[np.ndarray] = None

        # Debug / visualization state — populated after each plan() call
        self._last_rollouts: Optional[np.ndarray] = None  # (K, N+1, 3)
        self._last_best_k:   Optional[int]         = None

    # ------------------------------------------------------------------
    def reset(self, associator) -> None:
        """
        Call once at the start of each episode when bridge geometry changes.

        Caches the global cluster centroids from the BehaviorAssociator so the
        runner can transform them to local frame each planning step.
        """
        self._centroids_global = np.array(
            associator.all_region_centroids_jax_array, dtype=np.float64
        )  # (n_clusters, 2)

    # ------------------------------------------------------------------
    def plan(
        self,
        robot_v: float,
        lidar_hits_local: Optional[np.ndarray],   # (M, 2) robot-frame hits
        target_cluster_id: int,                    # goal of current pair
        current_cluster_id: int,                   # robot's current cluster
        start_cluster_id: int,                     # fixed at start of this pair
        all_centroids_local: np.ndarray,           # (n_clusters, 2) robot frame
        forbidden_cluster_ids: List[int],
        target_bearing_local: float,               # target_bearing − robot_yaw
    ) -> Tuple[float, float]:
        """
        Sample K velocity trajectories and return the best first-step command.

        Returns (v_cmd, omega_cmd) ready to send to hardware cmd_vel.
        Operates entirely in robot-local coordinates; zero global references.
        """
        K, N, dt = self.K, self.N, self.dt

        # 1. Sample K control sequences — uniform (v, omega) per step
        v_seqs  = np.random.uniform(0.0, self.V_MAX,    (K, N))
        om_seqs = np.random.uniform(-self.OMEGA_MAX, self.OMEGA_MAX, (K, N))
        control_seqs = np.stack([v_seqs, om_seqs], axis=2)   # (K, N, 2)

        # 2. Roll out — robot starts at ego-centric origin facing +x
        x0       = np.zeros(3)
        rollouts = unicycle_rollout(x0, control_seqs, dt)    # (K, N+1, 3)
        traj     = rollouts[:, 1:, :]                         # (K, N, 3)

        # 3. Obstacle distance field
        self._occ_grid.update(lidar_hits_local)
        dist_values = self._occ_grid.check_collisions(traj)   # (K, N)

        # 4. Score all trajectories
        scores = self._score(
            traj, dist_values,
            all_centroids_local,
            target_cluster_id, current_cluster_id, start_cluster_id,
            forbidden_cluster_ids, target_bearing_local,
        )   # (K,)

        # 5. Degenerate fallback: stop if every path is blocked
        if np.all(~np.isfinite(scores)):
            return 0.0, 0.0

        best_k = int(np.argmax(scores))

        # 6. Cache for visualization
        self._last_rollouts = rollouts
        self._last_best_k   = best_k

        # 7. First-step velocity command from the winning trajectory
        v_cmd  = float(np.clip(control_seqs[best_k, 0, 0], 0.0,           self.V_MAX))
        om_cmd = float(np.clip(control_seqs[best_k, 0, 1], -self.OMEGA_MAX, self.OMEGA_MAX))
        return v_cmd, om_cmd

    # ------------------------------------------------------------------
    def _score(
        self,
        traj: np.ndarray,                # (K, N, 3) local frame
        dist_values: np.ndarray,         # (K, N) obstacle distances in meters
        all_centroids_local: np.ndarray, # (n_clusters, 2) robot frame
        target_cluster_id: int,
        current_cluster_id: int,         # noqa: used for forbidden filter context
        start_cluster_id: int,
        forbidden_cluster_ids: List[int],
        target_bearing_local: float,
    ) -> np.ndarray:                     # (K,) float scores
        """
        Score K trajectories.  Reward priority (highest → lowest):

        1. Cluster identity  — nearest-centroid Voronoi classification of each
                               trajectory's endpoint.  Topological ordering is
                               reliable even when metric centroids drift.
        2. Bearing alignment — cos(final yaw − target bearing), always active.
        3. Directional pull  — cos(final heading angle to target centroid),
                               soft guidance without demanding precise waypoint.
        4. Collision mask    — hard -inf veto for any step < safety_radius.
        """
        K = self.K

        # ── 1. Cluster identity: nearest-centroid classification ──────────
        end_xy = traj[:, -1, :2]                              # (K, 2)
        # (K, n_clusters) — squared distance to each centroid
        dists = np.linalg.norm(
            end_xy[:, None, :] - all_centroids_local[None, :, :],
            axis=2,
        )
        nearest_cluster = np.argmin(dists, axis=1)            # (K,)

        in_target    = (nearest_cluster == target_cluster_id).astype(float)
        in_start     = (nearest_cluster == start_cluster_id).astype(float)
        in_forbidden = (
            np.isin(nearest_cluster, forbidden_cluster_ids).astype(float)
            if forbidden_cluster_ids else np.zeros(K)
        )

        scores  = self.w_target    * in_target
        scores += self.w_start     * in_start
        scores += self.w_forbidden * in_forbidden   # w_forbidden is negative

        # ── 2. Bearing alignment (final heading vs. target direction) ─────
        # cos() ∈ [-1, +1]: +1 = perfectly aligned, -1 = heading backward.
        # Uses only the final step — avoids penalising mid-path turning.
        scores += self.w_bearing * np.cos(traj[:, -1, 2] - target_bearing_local)

        # ── 3. Soft directional pull toward goal centroid ─────────────────
        # Rewards heading alignment to the centroid direction rather than raw
        # distance. Robust to metric inaccuracies in centroid placement.
        target_centroid = all_centroids_local[target_cluster_id]          # (2,)
        cdir = target_centroid / (np.linalg.norm(target_centroid) + 1e-6) # unit vec
        end_dir = np.stack(
            [np.cos(traj[:, -1, 2]), np.sin(traj[:, -1, 2])], axis=1
        )                                                                  # (K, 2)
        scores += self.w_direction * (end_dir * cdir).sum(axis=1)

        # ── 4. Hard collision mask ─────────────────────────────────────────
        # Trajectories with ANY step closer than safety_radius are invalid.
        collision = (dist_values < self.safety_radius).any(axis=1)
        scores[collision] = -np.inf

        return scores


# ─── Visualization Helper ──────────────────────────────────────────────────────

def visualize_rollouts(
    ax,
    rollouts_local: np.ndarray,   # (K, N+1, 3) from mpc._last_rollouts
    robot_pos: np.ndarray,        # (2,) global position, for world-frame transform
    robot_yaw: float,             # global heading in radians
    best_k: int,                  # index of the chosen trajectory
    n_show: int = 30,             # thin out candidates to avoid clutter
) -> None:
    """
    Overlay MPC candidate trajectories on an existing DGPPO matplotlib axes.

    Call this inside the render loop after plot.py has drawn the env frame.
    mpc._last_rollouts and mpc._last_best_k are available after every plan().
    """
    cy, sy = np.cos(robot_yaw), np.sin(robot_yaw)
    R = np.array([[cy, -sy], [sy, cy]])   # local → global

    show_ids = np.random.choice(
        len(rollouts_local), size=min(n_show, len(rollouts_local)), replace=False
    )
    for k in show_ids:
        xy_global = rollouts_local[k, :, :2] @ R.T + robot_pos   # (N+1, 2)
        ax.plot(xy_global[:, 0], xy_global[:, 1],
                color='skyblue', alpha=0.2, linewidth=0.5, zorder=3)

    best_xy = rollouts_local[best_k, :, :2] @ R.T + robot_pos
    ax.plot(best_xy[:, 0], best_xy[:, 1],
            color='lime', linewidth=2.0, zorder=4, label='MPC best')
    ax.scatter(best_xy[-1, 0], best_xy[-1, 1],
               color='lime', s=30, zorder=5)
