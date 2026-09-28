#!/usr/bin/env python3
"""
compare_controllers.py
======================
Side-by-side demonstration: Carson NMPC vs SamplingMPC (topological) vs
SamplingMPC-metric (ablation) under map distortion.

Per-run bridge geometry:
  - Bridge center fixed at (BC_X, BC_Y) = (0.75, 0.75)
  - bridge_theta, bridge_len, bridge_gap, wall_thick sampled each run
  - 0–2 random obstacles placed flush against inner walls
  - Map distortion = rotation + scale applied to overhead-map centroids

Controller comparison:
  Carson (red):         NMPC tracks distorted metric waypoints → fails
  Ours-metric (blue):  SamplingMPC + metric distance to distorted centroid → fails
  Ours-topological (green): SamplingMPC + TRUE Voronoi membership + distorted bearing → succeeds

Run:
    python compare_controllers.py [--n N] [--seed S] [--debug]
"""

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ── The SHIPPED CARLA controller, imported rather than reimplemented ──────────
# `terrain_mpc` is pure numpy with no ROS or CARLA dependency, so the 2D rig can drive the
# ACTUAL controller instead of an inline copy of its scoring. This matters because the two
# had drifted apart completely: `run_ours` samples 500 random per-step (v, omega) pairs and
# integrates them forward-Euler over ~1.2 m, while the shipped controller sweeps a
# deterministic 65-arc curvature fan in closed form over 10 m of ARC LENGTH, truncates each
# arc at its first blockage, and has a terrain term. Tuning one says nothing about the
# other, so the rig drives the real module to keep a single implementation of the rule.
_TERRAIN_MPC = None
try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from dgppo_ros_node_pkg import terrain_mpc as _TERRAIN_MPC   # noqa: E402
except Exception as _e:                                          # pragma: no cover
    print(f"  [warn] terrain_mpc unavailable ({_e}); its arm will be skipped")

# ── 2-D MPC modules (vendor/) ─────────────────────────────────────────────────────────────
import importlib.util

REPO_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "dgppo_ros_node_pkg", "vendor")

def _load_vendor_module(name: str, rel: str):
    path = os.path.join(REPO_ROOT, rel)
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_ba  = _load_vendor_module("behavior_associator", "behavior_associator.py")
_mpc = _load_vendor_module("sampling_mpc_2d",     "sampling_mpc_2d.py")

BehaviorAssociator = _ba.BehaviorAssociator
SamplingMPC        = _mpc.SamplingMPC
unicycle_rollout   = _mpc.unicycle_rollout

try:
    import casadi as ca
    CASADI_OK = True
except ImportError:
    CASADI_OK = False
    print("WARNING: CasADi not installed – Carson panels will be skipped.")


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

BC_X, BC_Y = 0.75, 0.75   # Fixed bridge center (world frame)

T_SIM  = 80
MPC_DT = 0.2
WAYPOINT_REACH_DIST = 0.15  # m — Carson waypoint advancement threshold


# ══════════════════════════════════════════════════════════════════════════════
# Per-run geometry
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RunGeom:
    cx: float
    cy: float
    theta: float           # physical bridge orientation (rad)
    length: float
    gap: float
    thick: float
    wall_obbs: list        # [(cx,cy,len,thick,theta), (cx,cy,len,thick,theta)]
    robot_start: np.ndarray  # (5,)  [x,y,yaw,0,0]
    rotation_ctr: np.ndarray # (2,)  bridge entrance — distortion pivot
    axis_dir: np.ndarray     # (2,)  unit vector along bridge (approach → exit)
    perp_dir: np.ndarray     # (2,)  unit perpendicular (pointing toward upper wall)


def build_bridge_walls(cx: float, cy: float, length: float, gap: float,
                       thick: float, theta: float) -> list:
    """Return two OBB wall tuples [(cx,cy,len,thick,theta), ...] for BehaviorAssociator."""
    half   = gap / 2 + thick / 2
    perp   = np.array([-math.sin(theta), math.cos(theta)])
    ctr    = np.array([cx, cy])
    w1 = tuple(float(v) for v in (ctr + half * perp)) + (length, thick, theta)
    w2 = tuple(float(v) for v in (ctr - half * perp)) + (length, thick, theta)
    return [w1, w2]


def make_run_geom(theta: float, length: float, gap: float, thick: float,
                  cx: float = BC_X, cy: float = BC_Y,
                  approach_offset: float = 0.70) -> RunGeom:
    """Build per-run geometry from bridge parameters."""
    axis_dir = np.array([math.cos(theta), math.sin(theta)])
    perp_dir = np.array([-math.sin(theta), math.cos(theta)])
    start_xy = np.array([cx, cy]) - approach_offset * axis_dir
    robot_start = np.array([float(start_xy[0]), float(start_xy[1]), theta, 0.0, 0.0])
    entrance = np.array([cx, cy]) - (length / 2) * axis_dir
    wall_obbs = build_bridge_walls(cx, cy, length, gap, thick, theta)
    return RunGeom(cx=cx, cy=cy, theta=theta, length=length, gap=gap, thick=thick,
                   wall_obbs=wall_obbs, robot_start=robot_start,
                   rotation_ctr=entrance, axis_dir=axis_dir, perp_dir=perp_dir)


# ══════════════════════════════════════════════════════════════════════════════
# Geometry helpers
# ══════════════════════════════════════════════════════════════════════════════

def _rotate_pts(pts: np.ndarray, deg: float, ctr: np.ndarray) -> np.ndarray:
    theta = math.radians(deg)
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return (pts - ctr) @ R.T + ctr


def _distort_pts(pts: np.ndarray, deg: float, scale: float,
                 ctr: np.ndarray,
                 tx: float = 0.0, ty: float = 0.0) -> np.ndarray:
    theta = math.radians(deg)
    c, s = math.cos(theta), math.sin(theta)
    R = scale * np.array([[c, -s], [s, c]])
    return (pts - ctr) @ R.T + ctr + np.array([tx, ty])


# ══════════════════════════════════════════════════════════════════════════════
# Carson NMPC
# ══════════════════════════════════════════════════════════════════════════════

def _wrap_angle(a: float, center: float) -> float:
    return (a + math.pi - center) % (2 * math.pi) - math.pi + center


if CASADI_OK:
    class CarsonNMPC:
        MAX_OBSTACLES = 8

        def __init__(self, N=8, dt=0.2, max_v=0.75, max_omega=1.5,
                     max_acc=1.0, max_yaw_acc=2.5, safe_dist=0.06,
                     ipopt_max_iter=50):
            self.N = N; self.dt = dt
            self.max_v = max_v; self.max_omega = max_omega
            self.max_acc = max_acc; self.max_yaw_acc = max_yaw_acc
            self.safe_dist = safe_dist
            self.ipopt_max_iter = ipopt_max_iter
            self._prev_X = None; self._prev_U = None
            self._obstacles: List[Tuple] = []
            self._build_mpc_problem()

        def set_obstacles(self, obs: List[Tuple]) -> None:
            self._obstacles = obs[:self.MAX_OBSTACLES]

        def _build_mpc_problem(self) -> None:
            N, dt = self.N, self.dt
            n_state, n_ctrl = 5, 2
            n_obs = self.MAX_OBSTACLES
            x_sym = ca.SX.sym("x", n_state)
            u_sym = ca.SX.sym("u", n_ctrl)
            f_dyn = ca.Function("f", [x_sym, u_sym], [ca.vertcat(
                x_sym[3] * ca.cos(x_sym[2]),
                x_sym[3] * ca.sin(x_sym[2]),
                x_sym[4],
                u_sym[0],
                u_sym[1],
            )])
            X = ca.SX.sym("X", n_state, N + 1)
            U = ca.SX.sym("U", n_ctrl, N)
            P = ca.SX.sym("P", n_state + 2 + n_obs * 3)
            x0   = P[:n_state]; goal = P[n_state:n_state + 2]; obs_p = P[n_state + 2:]
            Q = ca.diag([500, 500, 30, 0, 0]); R = ca.diag([0.1, 10]); slack_w = 1e5
            slacks = ca.SX.sym("slacks", n_obs, N)
            obj = 0; g = []; lb_g, ub_g = [], []
            g.append(X[:, 0] - x0); lb_g.extend([0]*n_state); ub_g.extend([0]*n_state)
            for k in range(N):
                x_err = X[:, k]
                x_err2 = ca.vertcat(x_err[0]-goal[0], x_err[1]-goal[1],
                                    _wrap_angle(float(0), float(0)), x_err[3], x_err[4])
                obj += ca.mtimes([x_err2.T, Q, x_err2])
                obj += ca.mtimes([U[:, k].T, R, U[:, k]])
                obj += slack_w * ca.sum1(slacks[:, k] ** 2)
                g.append(X[:, k+1] - f_dyn(X[:, k], U[:, k]) * dt - X[:, k])
                lb_g.extend([0]*n_state); ub_g.extend([0]*n_state)
                for i in range(n_obs):
                    ox = obs_p[3*i]; oy = obs_p[3*i+1]; orad = obs_p[3*i+2]
                    dist2 = (X[0, k]-ox)**2 + (X[1, k]-oy)**2
                    g.append(dist2 - (self.safe_dist+orad)**2 + slacks[i, k])
                    lb_g.append(0); ub_g.append(ca.inf)
            x_err_f = ca.vertcat(X[0,N]-goal[0], X[1,N]-goal[1], 0, X[3,N], X[4,N])
            obj += 10 * ca.mtimes([x_err_f.T, Q, x_err_f])
            opt_vars = ca.vertcat(ca.reshape(X,-1,1), ca.reshape(U,-1,1), ca.reshape(slacks,-1,1))
            nlp = {"x": opt_vars, "f": obj, "g": ca.vertcat(*g), "p": P}
            opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.max_iter": self.ipopt_max_iter}
            self._solver = ca.nlpsol("solver", "ipopt", nlp, opts)
            self._n_state = n_state; self._n_ctrl = n_ctrl; self._n_obs = n_obs
            self._lb_g = lb_g; self._ub_g = ub_g
            self._n_slack = n_obs * N; self._N_nlp = N

        def _solve_mpc(self, x0, goal, obstacles):
            N = self._N_nlp; n_state, n_ctrl = self._n_state, self._n_ctrl; n_obs = self._n_obs
            obs_pad = list(obstacles) + [(0,0,0)]*(n_obs-len(obstacles))
            obs_flat = [v for o in obs_pad for v in o]
            p_val = list(x0) + list(goal) + obs_flat
            n_x = n_state*(N+1); n_u = n_ctrl*N; n_s = n_obs*N
            if self._prev_X is None:
                x_init = np.zeros(n_x + n_u + n_s)
            else:
                x_init = np.concatenate([self._prev_X.flatten(order="F"),
                                         self._prev_U.flatten(order="F"),
                                         np.zeros(n_s)])
            lbx = np.array([-np.inf]*n_x + [-self.max_acc, -self.max_yaw_acc]*N + [0.0]*n_s, dtype=float)
            ubx = np.array([ np.inf]*n_x + [ self.max_acc,  self.max_yaw_acc]*N + [np.inf]*n_s, dtype=float)
            vel_idx = np.arange(3, n_x, n_state)
            lbx[vel_idx] = 0.0; ubx[vel_idx] = self.max_v
            ub_g = [np.inf if v == ca.inf else v for v in self._ub_g]
            sol = self._solver(x0=x_init, p=p_val, lbx=lbx, ubx=ubx, lbg=self._lb_g, ubg=ub_g)
            sol_x = np.array(sol["x"]).flatten()
            X_val = sol_x[:n_x].reshape(n_state, N+1, order="F")
            U_val = sol_x[n_x:n_x+n_u].reshape(n_ctrl, N, order="F")
            self._prev_X = X_val; self._prev_U = U_val
            return X_val.T, U_val

        def step(self, state, goal_xy, goal_yaw=0.0):
            try:
                X_pred, U_val = self._solve_mpc(state, goal_xy, self._obstacles)
            except Exception as e:
                print(f"    [NMPC fallback] {e}")
                return None, 0.0, 0.0
            v_next     = float(np.clip(state[3] + U_val[0, 0]*self.dt, 0.0, self.max_v))
            omega_next = float(np.clip(state[4] + U_val[1, 0]*self.dt, -self.max_omega, self.max_omega))
            return X_pred, v_next, omega_next

        def reset_warm_start(self):
            self._prev_X = None; self._prev_U = None


# ══════════════════════════════════════════════════════════════════════════════
# MPPI controller
# ══════════════════════════════════════════════════════════════════════════════

class MPPIController:
    """
    Model Predictive Path Integral controller.

    Maintains a mean control sequence U_mean (N, 2) = [v, ω] per step.
    Each tick:
      1. Sample K OU-correlated perturbations around U_mean
      2. Forward-simulate + score (externally)
      3. Importance-weighted update: U_mean ← Σ w_k * ctrl_k
      4. Shift U_mean forward one step (receding horizon warm start)
      5. Execute U_mean[0] as the command

    Differences from uniform argmax:
    - Smooth rollouts by construction (OU noise, not iid)
    - Cross-tick continuity (U_mean persists and shifts)
    - Uses information from ALL K rollouts via soft weighting, not just winner
    """

    def __init__(self, K=500, N=8, dt=0.2,
                 v_mean=0.40, om_mean=0.0,
                 v_sigma=0.20, om_sigma=0.40,
                 ou_theta_v=0.10, ou_theta_om=0.15,
                 temperature=1.0,
                 v_lo=0.0, v_hi=0.75,
                 om_lo=-1.5, om_hi=1.5):
        self.K, self.N, self.dt   = K, N, dt
        self.v_mean, self.om_mean = v_mean, om_mean
        self.v_sigma, self.om_sigma       = v_sigma, om_sigma
        self.ou_theta_v, self.ou_theta_om = ou_theta_v, ou_theta_om
        self.temperature          = temperature
        self.v_lo, self.v_hi      = v_lo, v_hi
        self.om_lo, self.om_hi    = om_lo, om_hi
        self.U_mean = np.stack([
            np.full(N, v_mean, dtype=np.float32),
            np.zeros(N, dtype=np.float32),
        ], axis=1)   # (N, 2)

    def _ou_perturb(self, K, N, theta, sigma):
        """Zero-mean OU noise: temporally correlated, x[t+1] = (1-θ)*x[t] + σ*ε."""
        x   = np.zeros(K, dtype=np.float32)
        out = np.empty((K, N), dtype=np.float32)
        for t in range(N):
            x += theta * (-x) + sigma * np.random.randn(K).astype(np.float32)
            out[:, t] = x
        return out

    def sample(self, explore_frac: float = 0.20):
        """Return v_seqs (K,N), om_seqs (K,N).

        80% OU-correlated around U_mean for exploitation; 20% pure uniform
        for exploration so the distribution can always escape obstacle regions.
        """
        K_exploit = int((1.0 - explore_frac) * self.K)
        K_explore  = self.K - K_exploit

        # Exploitation: OU perturbations around current mean
        eps_v  = self._ou_perturb(K_exploit, self.N, self.ou_theta_v,  self.v_sigma)
        eps_om = self._ou_perturb(K_exploit, self.N, self.ou_theta_om, self.om_sigma)
        v_exp  = np.clip(self.U_mean[:, 0] + eps_v,  self.v_lo, self.v_hi)
        om_exp = np.clip(self.U_mean[:, 1] + eps_om, self.om_lo, self.om_hi)

        # Exploration: pure uniform — always samples full control box
        v_rand  = np.random.uniform(self.v_lo,  self.v_hi,  (K_explore, self.N)).astype(np.float32)
        om_rand = np.random.uniform(self.om_lo, self.om_hi, (K_explore, self.N)).astype(np.float32)

        return (np.vstack([v_exp,  v_rand]),
                np.vstack([om_exp, om_rand]))

    def update(self, scores, v_seqs, om_seqs):
        """Importance-weighted mean update from all scored rollouts."""
        finite = np.isfinite(scores)
        if not finite.any():
            # All blocked — decay toward nominal to recover
            self.U_mean[:, 0] = 0.7 * self.U_mean[:, 0] + 0.3 * self.v_mean
            self.U_mean[:, 1] *= 0.7
            return
        s = scores.copy()
        s[~finite] = s[finite].min() - 1.0   # soft-penalise collision rollouts
        s -= s.max()                           # numerical stability before exp
        w  = np.exp(s / self.temperature)
        w /= w.sum()
        self.U_mean[:, 0] = np.clip(w @ v_seqs,  self.v_lo,  self.v_hi)
        self.U_mean[:, 1] = np.clip(w @ om_seqs, self.om_lo, self.om_hi)

    def shift(self):
        """Advance mean by one step; fill last step with nominal (v_mean, 0)."""
        self.U_mean[:-1] = self.U_mean[1:]
        self.U_mean[-1]  = [self.v_mean, 0.0]

    def get_cmd(self):
        """Return current first-step command from the updated mean."""
        return float(self.U_mean[0, 0]), float(self.U_mean[0, 1])

    def reset(self):
        self.U_mean[:, 0] = self.v_mean
        self.U_mean[:, 1] = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Ray-casting helpers
# ══════════════════════════════════════════════════════════════════════════════

def _ray_aabb(origin: np.ndarray, direction: np.ndarray,
              bmin: np.ndarray, bmax: np.ndarray) -> float:
    inv  = np.where(np.abs(direction) > 1e-12, 1.0 / direction, np.inf)
    t1   = (bmin - origin) * inv; t2 = (bmax - origin) * inv
    tmin = np.minimum(t1, t2); tmax = np.maximum(t1, t2)
    entry = float(np.max(tmin)); exit_ = float(np.min(tmax))
    return entry if (exit_ >= entry >= 0) else np.inf


def _ray_obb(origin: np.ndarray, direction: np.ndarray,
             cx: float, cy: float, olen: float, othick: float, otheta: float) -> float:
    """Ray vs OBB: transform to OBB local frame, delegate to AABB test."""
    c, s  = math.cos(otheta), math.sin(otheta)
    delta = origin - np.array([cx, cy])
    o_loc = np.array([ c*delta[0]    + s*delta[1],
                      -s*delta[0]    + c*delta[1]])
    d_loc = np.array([ c*direction[0] + s*direction[1],
                      -s*direction[0] + c*direction[1]])
    return _ray_aabb(o_loc, d_loc,
                     np.array([-olen/2, -othick/2]),
                     np.array([ olen/2,  othick/2]))


def _ray_circle(origin: np.ndarray, direction: np.ndarray,
                center: np.ndarray, radius: float) -> float:
    d    = origin - center
    b    = 2.0 * float(np.dot(d, direction))
    c    = float(np.dot(d, d)) - radius**2
    disc = b*b - 4.0*c
    if disc < 0:
        return np.inf
    sq = math.sqrt(disc)
    for t in sorted([(-b - sq) / 2.0, (-b + sq) / 2.0]):
        if t > 1e-5:
            return t
    return np.inf


def _inside_obb(x: float, y: float,
                cx: float, cy: float, olen: float, othick: float, otheta: float) -> bool:
    c, s = math.cos(otheta), math.sin(otheta)
    dx, dy = x - cx, y - cy
    lx =  c*dx + s*dy
    ly = -s*dx + c*dy
    return abs(lx) <= olen/2 + 1e-6 and abs(ly) <= othick/2 + 1e-6


def _segment_intersects_obb(x0: float, y0: float, x1: float, y1: float,
                             cx: float, cy: float, olen: float, othick: float,
                             otheta: float) -> bool:
    """Slab test: returns True if segment (x0,y0)→(x1,y1) passes through the OBB.
    Catches step-through when an endpoint lies outside but the path crosses the wall."""
    c, s = math.cos(otheta), math.sin(otheta)
    # Transform both endpoints to OBB local frame
    dx0, dy0 = x0 - cx, y0 - cy
    lx0 =  c*dx0 + s*dy0;  ly0 = -s*dx0 + c*dy0
    dx1, dy1 = x1 - cx, y1 - cy
    lx1 =  c*dx1 + s*dy1;  ly1 = -s*dx1 + c*dy1

    hl, ht = olen / 2, othick / 2
    # Per-axis slab intersection on the parametric segment [0,1]
    t_min, t_max = 0.0, 1.0
    for p0, p1, half in ((lx0, lx1, hl), (ly0, ly1, ht)):
        d = p1 - p0
        if abs(d) < 1e-12:
            if abs(p0) > half:
                return False   # parallel and outside this slab
        else:
            t1, t2 = (-half - p0) / d, (half - p0) / d
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
            if t_min > t_max:
                return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
# LiDAR simulation
# ══════════════════════════════════════════════════════════════════════════════

def simulate_lidar(pos: np.ndarray, yaw: float,
                   wall_obbs: list,
                   n_rays: int = 16, max_range: float = 0.8,
                   circles: Optional[List[Tuple]] = None) -> np.ndarray:
    hits = []
    cy_r, sy_r = math.cos(yaw), math.sin(yaw)
    for angle in np.linspace(-math.pi, math.pi, n_rays, endpoint=False):
        wa = yaw + angle
        rd = np.array([math.cos(wa), math.sin(wa)])
        d  = min((_ray_obb(pos, rd, *obb) for obb in wall_obbs), default=np.inf)
        if circles:
            for (cx, cy_c, cr) in circles:
                d = min(d, _ray_circle(pos, rd, np.array([cx, cy_c]), cr))
        if d < max_range:
            dg = d * rd
            hits.append([cy_r*dg[0] + sy_r*dg[1], -sy_r*dg[0] + cy_r*dg[1]])
    return np.array(hits) if hits else np.zeros((0, 2))


# ══════════════════════════════════════════════════════════════════════════════
# Unicycle step
# ══════════════════════════════════════════════════════════════════════════════

def _step(state: np.ndarray, v: float, omega: float, dt: float,
          wall_obbs: Optional[list] = None,
          obs_circles: Optional[List[Tuple]] = None,
          hard_physics: bool = True,
          return_hit: bool = False):
    s = state.copy()
    new_x = state[0] + v * math.cos(state[2]) * dt
    new_y = state[1] + v * math.sin(state[2]) * dt
    hit = False

    if hard_physics:
        # OBB wall backstop: revert if segment (start→end) crosses any wall.
        # Segment test catches step-through of thin walls that endpoint-only check misses.
        if wall_obbs:
            for obb in wall_obbs:
                if _segment_intersects_obb(state[0], state[1], new_x, new_y, *obb):
                    new_x, new_y = state[0], state[1]
                    hit = True
                    break

        # Segment-circle obstacle backstop (numerical safety for fast steps)
        if obs_circles:
            dx_seg, dy_seg = new_x - state[0], new_y - state[1]
            seg_sq = dx_seg**2 + dy_seg**2
            for (cx, cy_c, cr) in obs_circles:
                if seg_sq > 1e-12:
                    t = max(0.0, min(1.0,
                        ((cx - state[0])*dx_seg + (cy_c - state[1])*dy_seg) / seg_sq))
                    min_d = math.hypot(state[0] + t*dx_seg - cx,
                                       state[1] + t*dy_seg - cy_c)
                else:
                    min_d = math.hypot(state[0] - cx, state[1] - cy_c)
                if min_d < cr:
                    new_x, new_y = state[0], state[1]
                    hit = True
                    break

    s[0] = new_x; s[1] = new_y
    s[2] = state[2] + omega * dt
    s[3] = v; s[4] = omega
    return (s, hit) if return_hit else s


def _cbf_filter(v_cmd: float, om_cmd: float,
                hits_local: Optional[np.ndarray],
                d_safe: float = 0.04,
                alpha: float = 2.0) -> Tuple[float, float]:
    """CBF safety filter: reduce forward speed when approaching nearest LiDAR hit."""
    if hits_local is None or len(hits_local) == 0:
        return v_cmd, om_cmd

    dists   = np.linalg.norm(hits_local, axis=1)
    nearest = hits_local[int(np.argmin(dists))]
    d_min   = float(np.min(dists))
    h       = d_min - d_safe

    angle_to_obs = math.atan2(nearest[1], nearest[0])
    cos_a        = math.cos(angle_to_obs)

    if cos_a > 1e-3 and v_cmd * cos_a > alpha * h:
        v_cmd = max(0.0, alpha * h / cos_a)

    if d_min < 2.0 * d_safe and cos_a > 0.3:
        sin_a = math.sin(angle_to_obs)
        om_correction = -2.0 * alpha * sin_a * (1.0 - h / d_safe)
        om_cmd = float(np.clip(om_cmd + om_correction, -1.5, 1.5))

    return v_cmd, om_cmd


# ══════════════════════════════════════════════════════════════════════════════
# Simulation runners
# ══════════════════════════════════════════════════════════════════════════════

def run_carson(nmpc: "CarsonNMPC", waypoints: List[np.ndarray],
               geom: RunGeom,
               obs_circles: Optional[List[Tuple]] = None,
               hard_physics: bool = False) -> dict:
    nmpc.reset_warm_start()
    nmpc.set_obstacles(obs_circles or [])
    state = geom.robot_start.copy()
    traj  = [state[:3].copy()]
    planned = []

    wp_idx = 0
    current_wp = np.array(waypoints[wp_idx])

    print(f"  Carson {len(waypoints)}-WP ...", end="", flush=True)
    for t in range(T_SIM):
        if (np.linalg.norm(state[:2] - current_wp) < WAYPOINT_REACH_DIST
                and wp_idx + 1 < len(waypoints)):
            wp_idx += 1
            current_wp = np.array(waypoints[wp_idx])

        X_pred, v_next, omega_next = nmpc.step(state, current_wp, goal_yaw=geom.theta)
        planned.append(X_pred[:, :3].copy() if X_pred is not None else None)
        state = _step(state, v_next, omega_next, MPC_DT,
                      wall_obbs=geom.wall_obbs, obs_circles=obs_circles,
                      hard_physics=hard_physics)
        traj.append(state[:3].copy())
        if t % 10 == 9:
            print(".", end="", flush=True)

    print(f" done. final=({state[0]:.2f},{state[1]:.2f})")
    return {"traj": np.array(traj), "planned": planned, "waypoints": waypoints}


def corridor_from_lidar(hits_local, pos, yaw, R_l2g, prev_perp=None):
    """Corridor axis and centreline from LiDAR alone. No map, no ground truth.

    WHY. A PCA eigenvector is defined only up to SIGN -- it gives the wall-to-wall line,
    not which side is left. The legacy inline estimator resolved that 180 deg ambiguity
    with `geom.perp_dir` / `geom.axis_dir` (the TRUE map geometry), fell back to
    `bridge_center` when hits were sparse, and gated itself on a map-derived
    `inside_bridge`. Under the distortion ablation those are undistorted truth (the
    transform is applied to the centroids, not to `geom`), so that estimator is not
    map-free. This one uses no map quantities.

    THE SIGN, RESOLVED WITHOUT A MAP. A corridor cannot flip between consecutive frames,
    so the previous tick's perpendicular disambiguates this one. The first tick seeds from
    the robot's OWN heading -- +90 deg of travel is "left" by convention -- which is
    proprioceptive, not cartographic.

    NO GROUND-TRUTH FALLBACK. With too few lateral returns there is genuinely nothing to
    see, so this reports that and the caller DROPS the term for this tick. Substituting
    the true centreline there would make sparse-hit stretches look like successful
    perception.

    Returns ``(perp, centre, axis, ok)``; when ``ok`` is False the first three are None.
    """
    heading = np.array([math.cos(yaw), math.sin(yaw)])
    if hits_local is None or len(hits_local) < 4:
        return None, None, None, False
    lateral = np.abs(hits_local[:, 0]) < np.abs(hits_local[:, 1])
    if lateral.sum() < 4:
        return None, None, None, False
    hits_g = hits_local @ R_l2g.T + pos
    pca_hits = hits_g[lateral]
    _, evecs = np.linalg.eigh(np.cov(pca_hits.T))
    perp = evecs[:, 0]                      # min-variance = across the corridor
    ref = prev_perp if prev_perp is not None else np.array([-heading[1], heading[0]])
    if np.dot(perp, ref) < 0:
        perp = -perp
    proj = hits_g @ perp
    upper, lower = hits_g[proj > np.median(proj)], hits_g[proj < np.median(proj)]
    if not len(upper) or not len(lower):
        return None, None, None, False
    centre = (upper.mean(axis=0) + lower.mean(axis=0)) / 2
    axis = np.array([-perp[1], perp[0]])
    if np.dot(axis, heading) < 0:           # orient along travel, from ODOMETRY
        axis = -axis
    return perp, centre, axis, True


def run_ours(assoc: BehaviorAssociator, rid: dict,
             geom: RunGeom,
             true_cents: np.ndarray,
             dist_cents: np.ndarray,
             obs_circles: Optional[List[Tuple]] = None,
             hard_physics: bool = True,
             debug: bool = False,
             weights: Optional[dict] = None,
             obstacle_mode: str = "hard") -> dict:
    """
    SamplingMPC with topological guidance.

    ``obstacle_mode`` selects how LiDAR returns act on the score:

      "hard"    every rollout passing within ``safety_radius`` is set to -inf. The
                original behaviour, and an ablation rather than the shipped design.
      "graded"  a much smaller hard veto, plus a penalty linear in CLOSEST APPROACH.

    The graded form exists because the hard one deadlocks. In CARLA junction 60,
    traffic-light poles sit 1.7-2.4 m from the driving line and the tightest arc the
    platform can propose has a 4 m radius, so a 1.5 m veto rejects all 65 candidates,
    the relax-everything fallback engages, and the vehicle circles instead of
    progressing. A veto is only the right tool where a legal alternative exists; a
    penalty still ORDERS the candidates when every one of them is bad, which is what
    leaves a way out.

    Cluster membership (in_target / in_forbidden) uses TRUE physical centroids
    (robot knows its true location via SLAM).
    Bearing direction hint uses DISTORTED centroid direction (overhead map belief).
    """
    w = weights or {}
    w_target    = w.get("weight_target",    10.0)
    w_forbidden = w.get("weight_forbidden", -15.0)
    w_bearing   = w.get("weight_bearing",    3.0)
    w_corridor  = w.get("weight_corridor",   2.0)
    w_progress  = w.get("weight_progress",   0.8)
    # Graded-obstacle knobs. Scaled to THIS sim (safety_radius 0.06 m, bridge gap
    # 0.25-0.55 m) and NOT carried over from CARLA -- the two differ by ~25x in length
    # and the weights do not transfer. Only the shape of the answer does.
    w_obstacle  = w.get("weight_obstacle",      3.0)
    obs_hard_m  = w.get("obstacle_hard_m",      0.04)
    obs_infl_m  = w.get("obstacle_influence_m", 0.20)

    mpc = SamplingMPC(K=500, N=8, dt=MPC_DT, safety_radius=0.06, lidar_grid_size=1.5)
    mpc.reset(assoc)
    mpc._centroids_global = dist_cents.copy()

    n_c = true_cents.shape[0]
    state  = geom.robot_start.copy()
    traj   = [state[:3].copy()]

    pairs = [
        (rid["approach_bridge_0"], rid["on_bridge_0"]),
        (rid["on_bridge_0"],       rid["exit_bridge_0"]),
    ]
    pair_idx  = 0
    start_id  = pairs[0][0]
    target_id = pairs[0][1]
    forbidden = [c for c in range(n_c) if c not in (start_id, target_id)]

    record_step       = 5   # capture mid-approach, before bridge crossing
    last_rollouts     = None
    last_best_k       = 0
    recorded_pos      = geom.robot_start[:2].copy()
    recorded_yaw      = geom.robot_start[2]
    record_tgt_cent   = None
    last_cluster_parts = None
    last_bearing_parts = None
    stuck_count       = 0   # consecutive recovery steps where CBF clamped v→0
    prev_perp         = None   # carries the corridor sign between ticks, map-free
    reached_exit      = False
    max_along         = -1e9
    wall_hit_count    = 0

    # REACH IS THE COMPARISON VARIABLE. This arm's lookahead is v_max * N * dt, so N is
    # how it gets equalised against the terrain arm's `arc_len_m`. Default 8 = 1.20 m,
    # which is LONGER THAN THE WHOLE BRIDGE (0.5-1.0 m) -- an advantage the terrain arm
    # at scale 0.05 (0.50 m) does not have, so reach must be equalised for a fair comparison.
    K = 500
    N = int(w.get('horizon_n', 8))
    bridge_center = np.array([geom.cx, geom.cy])

    print("  Ours ...", end="", flush=True)
    for t in range(T_SIM):
        pos = state[:2];  yaw = state[2]
        cy_, sy_ = math.cos(yaw), math.sin(yaw)
        R_g2l = np.array([[ cy_,  sy_], [-sy_,  cy_]])
        R_l2g = np.array([[ cy_, -sy_], [ sy_,  cy_]])

        # ── Tier A: pair advancement via nearest true centroid (Voronoi) ─────
        # get_current_behavior() has an open_space region that covers the entire
        # world and is alphabetically last, so it always overwrites specific
        # region matches.  Nearest-centroid Voronoi is consistent with Tier B.
        curr_id = int(np.argmin(np.linalg.norm(true_cents - pos, axis=1)))

        if curr_id == target_id and pair_idx + 1 < len(pairs):
            pair_idx  += 1
            start_id   = pairs[pair_idx][0]
            target_id  = pairs[pair_idx][1]
            forbidden  = [c for c in range(n_c) if c not in (start_id, target_id)]

        # ── LiDAR ─────────────────────────────────────────────────────────────
        hits = simulate_lidar(pos, yaw, geom.wall_obbs, circles=obs_circles)
        if len(hits) > 0:
            hit_dists = np.linalg.norm(hits, axis=1)
            hits = hits[hit_dists > 0.005]
        hits_in = hits if len(hits) > 0 else None

        # ── Generate rollouts ─────────────────────────────────────────────────
        v_seqs  = np.random.uniform(0.0, 0.75, (K, N))
        om_seqs = np.random.uniform(-1.5, 1.5,  (K, N))
        ctrl    = np.stack([v_seqs, om_seqs], axis=2)
        rollouts    = unicycle_rollout(np.zeros(3), ctrl, MPC_DT)
        traj_local  = rollouts[:, 1:, :]

        # ── EDT collision check ───────────────────────────────────────────────
        mpc._occ_grid.update(hits_in)
        dist_values = mpc._occ_grid.check_collisions(traj_local)

        # ── Rollout endpoints in global frame ─────────────────────────────────
        end_local  = traj_local[:, -1, :2]
        end_global = end_local @ R_l2g.T + pos

        # ── Tier B: TRUE centroid Voronoi for cluster membership ──────────────
        dists_true      = np.linalg.norm(end_global[:, None, :] - true_cents[None, :, :], axis=2)
        nearest_cluster = np.argmin(dists_true, axis=1)

        in_target    = (nearest_cluster == target_id).astype(float)
        in_start     = (nearest_cluster == start_id).astype(float)
        in_forbidden = np.isin(nearest_cluster, forbidden).astype(float)

        cluster_parts = w_target * in_target + 1.0 * in_start + w_forbidden * in_forbidden
        scores = cluster_parts.copy()

        # ── Bearing toward DISTORTED target centroid (map direction hint) ─────
        tgt_global    = dist_cents[target_id]
        tgt_local     = (tgt_global - pos) @ R_g2l.T
        bearing_local = math.atan2(tgt_local[1], tgt_local[0])
        bearing_cos   = w_bearing * np.cos(traj_local[:, -1, 2] - bearing_local)

        cdir   = tgt_local / (np.linalg.norm(tgt_local) + 1e-6)
        end_fx = np.cos(traj_local[:, -1, 2])
        end_fy = np.sin(traj_local[:, -1, 2])
        bearing_dot   = 1.0 * (end_fx * cdir[0] + end_fy * cdir[1])
        bearing_parts = bearing_cos + bearing_dot
        scores += bearing_parts

        # ── Corridor centering, from LiDAR ONLY ──────────────────────────────
        # `map_free_corridor` in the weights selects the map-free estimator. The legacy
        # branch is kept for reproducibility of the original behaviour: it resolves the PCA
        # sign with the TRUE map perpendicular and falls back to the TRUE bridge centre,
        # which under distortion is undistorted ground truth.
        if w.get("map_free_corridor", False):
            lp, lc, la, ok_corr = corridor_from_lidar(hits_in, pos, yaw, R_l2g, prev_perp)
            if ok_corr:
                lidar_perp, lidar_center, lidar_axis = lp, lc, la
                prev_perp = lp.copy()
            else:
                # Nothing to see: DROP the term this tick rather than substitute truth.
                lidar_perp = lidar_center = None
                lidar_axis = cdir @ R_l2g.T          # fall back to the bearing direction
        else:
            lidar_perp   = geom.perp_dir
            lidar_center = bridge_center
            along_from_center = float((pos - bridge_center) @ geom.axis_dir)
            inside_bridge = along_from_center > -(geom.length / 2)
            if inside_bridge and hits_in is not None and len(hits_in) >= 4:
                hits_g = hits_in @ R_l2g.T + pos
                lateral_mask = np.abs(hits_in[:, 0]) < np.abs(hits_in[:, 1])
                pca_hits = hits_g[lateral_mask] if lateral_mask.sum() >= 4 else hits_g
                _, evecs = np.linalg.eigh(np.cov(pca_hits.T))
                lidar_perp = evecs[:, 0]
                if np.dot(lidar_perp, geom.perp_dir) < 0:
                    lidar_perp = -lidar_perp
                proj = hits_g @ lidar_perp
                upper_hits = hits_g[proj > np.median(proj)]
                lower_hits = hits_g[proj < np.median(proj)]
                if len(upper_hits) > 0 and len(lower_hits) > 0:
                    lidar_center = (upper_hits.mean(axis=0) + lower_hits.mean(axis=0)) / 2
            lidar_axis = np.array([-lidar_perp[1], lidar_perp[0]])
            if np.dot(lidar_axis, geom.axis_dir) < 0:
                lidar_axis = -lidar_axis

        if lidar_perp is not None:
            end_across = (end_global - lidar_center) @ lidar_perp
            scores += w_corridor * np.exp(-(end_across**2) / (0.10**2))

        # ── Progress along LiDAR-derived corridor axis ────────────────────────
        end_along = (end_global - pos) @ lidar_axis
        scores += w_progress * np.clip(end_along, 0, None)

        # ── EDT hard collision mask ───────────────────────────────────────────
        if obstacle_mode == "graded":
            d_near = dist_values.min(axis=1)              # closest approach per rollout
            collision = d_near < obs_hard_m
            # Bounded linear ramp: 0 beyond the influence radius, 1 at contact. Bounded
            # on purpose -- a 1/d form is unbounded near zero, so one close return would
            # swamp every other term in the score.
            scores = scores - w_obstacle * np.clip(
                (obs_infl_m - d_near) / max(obs_infl_m, 1e-9), 0.0, 1.0)
        else:
            collision = (dist_values < mpc.safety_radius).any(axis=1)
        scores[collision] = -np.inf

        if np.all(~np.isfinite(scores)):
            # Use true geometry for recovery — lidar_perp/lidar_center are unreliable
            # outside the corridor (PCA sees oblique wall edges, not parallel walls).
            across_to_center = float((bridge_center - pos) @ geom.perp_dir)
            om_cmd = 1.5 * np.sign(across_to_center)
            v_cmd  = 0.15
            best_k = 0
            recovery = True
        else:
            best_k = int(np.argmax(scores))
            v_cmd  = float(ctrl[best_k, 0, 0])
            om_cmd = float(ctrl[best_k, 0, 1])
            recovery = False

        v_cmd, om_cmd = _cbf_filter(v_cmd, om_cmd, hits_in)

        # Stuck detection: recovery fired but CBF clamped v→0 (obstacle too close).
        # After 5 consecutive blocked steps, back up to gain clearance.
        if recovery and v_cmd < 0.04:
            stuck_count += 1
            if stuck_count >= 5:
                v_cmd = -0.20
                om_cmd = 0.0
                stuck_count = 0
        else:
            stuck_count = 0

        mpc._last_rollouts = rollouts
        mpc._last_best_k   = best_k

        if t == record_step:
            last_rollouts      = rollouts.copy()
            last_best_k        = best_k
            recorded_pos       = pos.copy()
            recorded_yaw       = yaw
            record_tgt_cent    = dist_cents[target_id].copy()
            last_cluster_parts = cluster_parts.copy()
            last_bearing_parts = bearing_parts.copy()

        if debug:
            d_min = float(np.min(np.linalg.norm(hits_in, axis=1))) if hits_in is not None and len(hits_in) > 0 else float('inf')
            ratio = float(np.mean(np.abs(cluster_parts[~collision])) /
                          (np.mean(np.abs(cluster_parts[~collision])) + np.mean(np.abs(bearing_parts[~collision])) + 1e-6)) if (~collision).any() else 0.0
            print(f"\n  t={t:3d}  pos=({pos[0]:.3f},{pos[1]:.3f})  yaw={math.degrees(yaw):.1f}°"
                  f"  cluster={curr_id}→{target_id}  v={v_cmd:.3f}  ω={om_cmd:.3f}"
                  f"  d_min={d_min:.3f}  recovery={recovery}  guidance_ratio={ratio:.2f}")

        state, hit = _step(state, v_cmd, om_cmd, MPC_DT,
                           wall_obbs=geom.wall_obbs, obs_circles=obs_circles,
                           hard_physics=hard_physics, return_hit=True)
        wall_hit_count += int(hit)
        traj.append(state[:3].copy())
        if t % 10 == 9 and not debug:
            print(".", end="", flush=True)

        along  = float((state[:2] - bridge_center) @ geom.axis_dir)
        across = float((state[:2] - bridge_center) @ geom.perp_dir)
        # A large axis-projection alone doesn't mean it crossed the bridge —
        # a path that veers off to the side can still have a large `along`
        # component by coincidence. Require it to also be roughly within the
        # corridor laterally, not just "far enough in the right direction."
        lateral_ok = abs(across) < (geom.gap / 2 + 0.20)
        if lateral_ok:
            max_along = max(max_along, along)
        if along > geom.length / 2 + 0.10 and lateral_ok:
            reached_exit = True
            for _ in range(t + 1, T_SIM):
                traj.append(traj[-1].copy())
            break

    print(f" done. final=({state[0]:.2f},{state[1]:.2f})")
    exit_thresh = geom.length / 2 + 0.10
    return {
        "traj":               np.array(traj),
        "last_rollouts":      last_rollouts,
        "last_best_k":        last_best_k,
        "recorded_pos":       recorded_pos,
        "recorded_yaw":       recorded_yaw,
        "record_tgt_cent":    record_tgt_cent,
        "last_cluster_parts": last_cluster_parts,
        "last_bearing_parts": last_bearing_parts,
        "cents_used":         dist_cents.copy(),
        "reached_exit":       reached_exit,
        "final_progress_frac": float(np.clip(max_along / exit_thresh, 0.0, 1.0)),
        "wall_hit_count":     wall_hit_count,
    }


def _dist_to_obb(pts, obb):
    """Euclidean distance from each point to an oriented box (0 inside). Vectorised."""
    cx, cy, blen, bthick, th = obb
    c_, s_ = math.cos(th), math.sin(th)
    # World -> box frame (rotate by -th), then the standard box distance.
    d = pts - np.array([cx, cy])
    local = np.stack([d[..., 0] * c_ + d[..., 1] * s_,
                      -d[..., 0] * s_ + d[..., 1] * c_], axis=-1)
    q = np.abs(local) - np.array([blen / 2.0, bthick / 2.0])
    return np.linalg.norm(np.maximum(q, 0.0), axis=-1)


def sidewalk_traversal(traj, wall_obbs, width):
    """Fraction of DRIVEN PATH LENGTH spent on the sidewalk strip.

    The per-arc cost landscape shows the term SCORES; it does not show the robot ends up
    anywhere different. This is the outcome the term exists to move, evaluated on the path
    actually driven, so it tests the term's effect rather than only illustrating it.

    Length-weighted, not sample-weighted: the simulator holds the pose after an early exit
    (`traj` is padded with repeats), and counting samples would let a run that stops ON the
    kerb accumulate an unbounded score while standing still.
    """
    xy = np.asarray(traj)[:, :2]
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    total = float(seg.sum())
    if total <= 1e-9:
        return float("nan")
    mid = 0.5 * (xy[:-1] + xy[1:])
    on = np.zeros(len(mid), dtype=bool)
    for obb in wall_obbs:
        on |= _dist_to_obb(mid, obb) <= width
    return float(seg[on].sum() / total)


def sidewalk_terrain_source(wall_obbs, pose_ref, *, width=0.05, soft_class=1,
                            unobserved_class=None, blind_radius_m=0.0):
    """Terrain = a THIN BORDER hugging the buildings, which is what sidewalk actually is.

    Note: a disc of soft ground in the middle of the corridor is not terrain, it is an
    OBSTACLE, and the obstacle term already handles obstacles -- a figure built on one
    would show the obstacle rule wearing a terrain label.

    Real sidewalk is a strip a few tens of centimetres wide running ALONG the buildings,
    parallel to the direction of travel. With that geometry the ban BINDS ONLY OFF-AXIS,
    at turn time (as also observed in CARLA). Driving straight down a
    corridor, every candidate arc is parallel to the strip and none of them touch it, so
    the term is silent. It starts costing the moment an arc swings wide -- cutting a
    corner, or drifting toward a building -- which is exactly when you want it.

    A demo built on a blob cannot show that, because a blob binds hardest going straight.

    `width` is the strip width in this world's metres (bridge gaps here are 0.25-0.55 m,
    so 0.05 is a realistically thin kerb, not a wall).
    """
    def _classify(pts):
        pos, yaw = pose_ref["pos"], pose_ref["yaw"]
        c_, s_ = math.cos(yaw), math.sin(yaw)
        world = pts @ np.array([[c_, -s_], [s_, c_]]).T + pos
        klass = np.zeros(pts.shape[:2], dtype=int)
        for obb in wall_obbs:
            klass[_dist_to_obb(world, obb) <= width] = soft_class
        if unobserved_class is not None and blind_radius_m > 0.0:
            klass[np.linalg.norm(pts, axis=-1) < blind_radius_m] = unobserved_class
        return klass
    return _classify


def run_terrain_mpc(assoc: "BehaviorAssociator", rid: dict,
                    geom: RunGeom,
                    true_cents: np.ndarray,
                    dist_cents: np.ndarray,
                    obs_circles: Optional[List[Tuple]] = None,
                    hard_physics: bool = True,
                    weights: Optional[dict] = None,
                    scale: float = 0.05) -> dict:
    """THE SHIPPED CARLA CONTROLLER, driven in the 2D world. Not a reimplementation.

    Calls `terrain_mpc.plan_step_terrain` directly, so whatever is measured here is the
    code that runs on the vehicle: the deterministic curvature fan, closed-form arcs
    parameterised by ARC LENGTH, arc truncation at the first blockage, the hard-veto +
    graded-penalty obstacle response.

    WHY `scale`. Every length in `TerrainMpcConfig` is metres in a world where the road is
    14 m wide and the car does 5 m/s. This world has 0.25-0.55 m bridge gaps and a 0.75 m/s
    robot -- about 20x smaller. The constants are SCALED, not copied, and `kappa_max` scales
    INVERSELY because it is 1/metres: at scale 0.05 the 10 m arc becomes 0.5 m (a bridge
    length) and the 2.86 m minimum turning radius becomes 0.14 m (turnable inside the gap).
    Copying the metre values unchanged would put a 10 m arc in a 0.5 m corridor, which is
    not a comparison of controllers but of unit errors.

    NO TERRAIN HERE, stated rather than faked: the bridge world has no surface variation,
    so `terrain_class` returns one class at zero cost. That makes this arm bearing +
    progress + obstacles, which is exactly the part that is comparable to the others.
    """
    if _TERRAIN_MPC is None:
        return None                                   # caller treats None as "not run"
    w = weights or {}
    # The terrain source needs the LIVE pose (see sidewalk_terrain_source); this dict is
    # rewritten every tick below. class 0 = firm ground, 1 = soft, 2 = unobserved.
    _pose_ref = {"pos": geom.robot_start[:2].copy(), "yaw": float(geom.robot_start[2])}
    if w.get("tm_terrain"):
        _sidewalk_w = float(w.get("tm_sidewalk_m", 0.05))
        _terrain_src = sidewalk_terrain_source(
            geom.wall_obbs, _pose_ref, width=_sidewalk_w, unobserved_class=2,
            blind_radius_m=float(w.get("tm_blind_m", 0.0)))
        # WHICH SURFACE IS BANNED IS AN INSTRUCTION, NOT A CONSTANT. "stay off the
        # sidewalk" and "the road is freshly tarred so stay on the sidewalk" are the same
        # machinery with the classes swapped -- which is the point of making the ban
        # instructable rather than hardcoding a drivable surface. class 0 = road (the open
        # middle), class 1 = the kerb strip along the buildings.
        if w.get("tm_ban_road"):
            _terrain_costs = {0: float(w.get("tm_soft_cost", 1.0)), 1: 0.0}
            _banned_class = 0
        else:
            _terrain_costs = {0: 0.0, 1: float(w.get("tm_soft_cost", 1.0))}
            _banned_class = 1
    else:
        _sidewalk_w = 0.0
        _terrain_src = lambda pts: np.zeros(pts.shape[:2], dtype=int)
        _terrain_costs = {0: 0.0}
        _banned_class = 1
    cfg = _TERRAIN_MPC.TerrainMpcConfig(
        K=int(w.get("tm_fan", 65)),
        segments=int(w.get("tm_segments", 1)),
        n=16,
        arc_len_m=10.0 * scale,
        # kappa_max is 1/metres so it scales inversely -- BUT scaling it by 1/scale keeps
        # `arc_len * kappa_max` pinned at 3.5 rad (200 deg), which is the CARLA pathology
        # rather than a property of this world, and it made every reach comparison measure
        # that instead of reach. `tm_turn_rad` sets the maximum heading change per arc and
        # derives kappa_max from the arc length, so reach can be swept without dragging a
        # constant 200-degree turn along with it. None = the old inverse scaling.
        kappa_max=(float(w["tm_turn_rad"]) / (10.0 * scale)
                   if w.get("tm_turn_rad") else 0.35 / scale),
        # This rig OWNS its turn budget via `tm_turn_rad` (see the comment above), so the
        # controller's own `max_turn_rad` guard must not also police it. The default guard
        # is pi/2, and this arm's default geometry is 10.0 * 0.35 = 3.5 rad = 201 deg --
        # deliberate here, a pathology in CARLA -- so with the guard on the terrain arm
        # would raise ValueError.
        max_turn_rad=None,
        v_max=0.75,                                   # the 2D robot's own limit
        kappa_taper=0.15 / scale,
        a_max=2.0 * scale,
        safety_radius=1.5 * scale,
        obstacle_hard_m=float(w.get("tm_hard_m", 1.0 * scale)),
        obstacle_influence_m=float(w.get("tm_infl_m", 4.0 * scale)),
        w_obstacle=float(w.get("tm_w_obstacle", 3.0)),
        w_bearing=float(w.get("tm_w_bearing", 1.0)),
        w_progress=float(w.get("tm_w_progress", 1.0)),
        # Terrain is OFF by default.
        # `tm_w_terrain` > 0 with `tm_terrain` on turns the term on for the explainer.
        w_terrain=(float(w.get("tm_w_terrain", 1.0)) if w.get("tm_terrain") else 0.0),
        # Corridor centring, off unless asked for. sigma scales with the world: the 2D rig
        # uses 0.10 m in a 0.40 m slot (25% of the width), and 2.0 * scale reproduces that
        # at scale 0.05.
        w_corridor=float(w.get("tm_w_corridor", 0.0)),
        corridor_sigma_m=float(w.get("tm_corridor_sigma", 2.0 * scale)),
        terrain_class=_terrain_src,
        terrain_costs=_terrain_costs,
        unobserved_class=(2 if w.get("tm_blind_m") else None),
        forbid_classes=(frozenset({_banned_class}) if w.get("tm_terrain_forbid")
                        else frozenset()))

    state = geom.robot_start.copy()
    traj = [state[:3].copy()]
    pairs = [(rid["approach_bridge_0"], rid["on_bridge_0"]),
             (rid["on_bridge_0"], rid["exit_bridge_0"])]
    pair_idx, kappa_prev = 0, 0.0
    fan_rollouts, rec_pos, rec_yaw = None, geom.robot_start[:2].copy(), geom.robot_start[2]
    # Per-arc scoring state at the display tick. The fan GEOMETRY is deterministic and is
    # regenerated below, but these are functions of the pose and the world at that tick and
    # CANNOT be regenerated -- capturing them anywhere else silently draws one tick's
    # colours over another tick's arcs, which nothing would flag.
    fan_best, fan_soft, fan_forbid, fan_bearing = None, None, None, None
    # True while the captured tick is only the tick-5 fallback, so a later tick where
    # terrain genuinely overrides bearing may still replace it.
    fan_best_is_fallback = True
    fan_tick, fan_overrode = None, False
    # Body-frame fan geometry is pose-independent, so the bearing-only argmax can be
    # evaluated every tick without rebuilding anything.
    if cfg.segments == 2:
        _fan_geom = _TERRAIN_MPC.arc_rollout_2seg(
            _TERRAIN_MPC.two_segment_fan(cfg.kappa_max, int(round(math.sqrt(cfg.K)))),
            cfg.arc_len_m, cfg.n)
    else:
        _fan_geom = _TERRAIN_MPC.arc_rollout_k(
            np.zeros(3), _TERRAIN_MPC.curvature_fan(cfg.kappa_max, cfg.K),
            cfg.arc_len_m, cfg.n)
    target_id = pairs[0][1]
    bridge_center = np.array([geom.cx, geom.cy])
    reached_exit, max_along, wall_hits = False, -1e9, 0

    print("  Terrain-MPC ...", end="", flush=True)
    for t in range(T_SIM):
        pos, yaw = state[:2], state[2]
        _pose_ref["pos"], _pose_ref["yaw"] = np.asarray(pos, float).copy(), float(yaw)
        cy_, sy_ = math.cos(yaw), math.sin(yaw)
        R_g2l = np.array([[cy_, sy_], [-sy_, cy_]])

        curr = int(np.argmin(np.linalg.norm(true_cents - pos, axis=1)))
        if curr == target_id and pair_idx + 1 < len(pairs):
            pair_idx += 1
            target_id = pairs[pair_idx][1]

        # The only map-derived quantity this controller consumes is a DIRECTION, and it is
        # taken from the DISTORTED centroids like every other arm's bearing, so the
        # distortion ablation means the same thing here.
        tgt_local = (dist_cents[target_id] - pos) @ R_g2l.T
        bearing_ref = math.atan2(tgt_local[1], tgt_local[0])

        hits = simulate_lidar(pos, yaw, geom.wall_obbs, circles=obs_circles)
        if len(hits):
            hits = hits[np.linalg.norm(hits, axis=1) > 0.005]
        res = _TERRAIN_MPC.plan_step_terrain(
            cfg, bearing_ref, kappa_prev=kappa_prev,
            hits_body=hits if len(hits) else None)
        v_cmd, om_cmd = float(res.v), float(res.omega)
        kappa_prev = (om_cmd / v_cmd) if v_cmd > 1e-6 else 0.0

        # WHICH TICK TO DISPLAY. Tick 5 is arbitrary and, with a sidewalk, usually shows
        # nothing: driving straight down a corridor the strip is parallel to every
        # candidate, so terrain is silent and the chosen arc IS the bearing arc. The
        # interesting tick is the one where terrain actually overrides bearing, which is
        # the off-axis/turn-time case. So: capture the FIRST tick where the two disagree,
        # and fall back to tick 5 if they never do (meaning the term never changed a
        # decision on this run).
        _th_end = _fan_geom[:, -1, 2]        # body frame, identical every tick
        _b_only = int(np.argmax(np.cos(_th_end - bearing_ref)))
        _disagree = int(res.best_index) != _b_only
        if (_disagree and fan_best_is_fallback) or (t == 5 and fan_rollouts is None):
            fan_rollouts = _fan_geom          # same construction the controller used
            rec_pos, rec_yaw = pos.copy(), yaw
            fan_best_is_fallback = not _disagree
            # Same tick as the geometry above -- see the note at the declaration.
            fan_best = int(res.best_index)
            fan_soft = np.asarray(res.arc_soft, dtype=float).copy()
            fan_forbid = np.asarray(res.arc_forbidden, dtype=bool).copy()
            fan_bearing = float(bearing_ref)
            fan_tick, fan_overrode = t, _disagree

        state, hit = _step(state, v_cmd, om_cmd, MPC_DT, wall_obbs=geom.wall_obbs,
                           obs_circles=obs_circles, hard_physics=hard_physics,
                           return_hit=True)
        wall_hits += int(hit)
        traj.append(state[:3].copy())
        along = float((state[:2] - bridge_center) @ geom.axis_dir)
        across = float((state[:2] - bridge_center) @ geom.perp_dir)
        lateral_ok = abs(across) < (geom.gap / 2 + 0.20)
        if lateral_ok:
            max_along = max(max_along, along)
        if along > geom.length / 2 + 0.10 and lateral_ok:
            reached_exit = True
            for _ in range(t + 1, T_SIM):
                traj.append(traj[-1].copy())
            break
    print(f" done. final=({state[0]:.2f},{state[1]:.2f})")
    exit_thresh = geom.length / 2 + 0.10
    return {"traj": np.array(traj), "reached_exit": reached_exit,
            "final_progress_frac": float(np.clip(max_along / exit_thresh, 0.0, 1.0)),
            "wall_hit_count": wall_hits, "cents_used": dist_cents.copy(),
            # Record the chosen arc so the plot can distinguish it from the rejected ones.
            "last_rollouts": fan_rollouts, "last_best_k": fan_best,
            "last_arc_soft": fan_soft, "last_arc_forbidden": fan_forbid,
            "last_bearing_ref": fan_bearing, "sidewalk_width": _sidewalk_w,
            "fan_tick": fan_tick, "terrain_overrode_bearing": fan_overrode,
            "recorded_pos": rec_pos,
            "recorded_yaw": rec_yaw, "record_tgt_cent": None,
            "last_cluster_parts": None, "last_bearing_parts": None}


def run_ours_metric(assoc: BehaviorAssociator, rid: dict,
                    geom: RunGeom,
                    true_cents: np.ndarray,
                    dist_cents: np.ndarray,
                    obs_circles: Optional[List[Tuple]] = None,
                    hard_physics: bool = True,
                    weights: Optional[dict] = None) -> dict:
    """
    Ablation: SamplingMPC with metric distance scoring instead of cluster identity.

    Replaces the binary cluster reward (+10/-15) with continuous proximity to
    distorted centroid: 10/(1+dist). Everything else identical (EDT, CBF, rollouts).
    This isolates whether cluster identity (not the MPC framework) is what enables bridge
    crossing under map distortion.
    """
    w = weights or {}
    w_progress = w.get("weight_progress", 0.8)

    mpc = SamplingMPC(K=500, N=8, dt=MPC_DT, safety_radius=0.06, lidar_grid_size=1.5)
    mpc.reset(assoc)
    mpc._centroids_global = dist_cents.copy()

    n_c = true_cents.shape[0]
    state = geom.robot_start.copy()
    traj  = [state[:3].copy()]

    pairs = [
        (rid["approach_bridge_0"], rid["on_bridge_0"]),
        (rid["on_bridge_0"],       rid["exit_bridge_0"]),
    ]
    pair_idx  = 0
    start_id  = pairs[0][0]
    target_id = pairs[0][1]
    reached_exit   = False
    max_along      = -1e9
    wall_hit_count = 0

    K, N = 500, 8
    bridge_center = np.array([geom.cx, geom.cy])

    print("  Ours-metric ...", end="", flush=True)
    for t in range(T_SIM):
        pos = state[:2]; yaw = state[2]
        cy_, sy_ = math.cos(yaw), math.sin(yaw)
        R_l2g = np.array([[cy_, -sy_], [sy_, cy_]])

        # Tier A: nearest true centroid Voronoi (mirrors run_ours fix)
        curr_id = int(np.argmin(np.linalg.norm(true_cents - pos, axis=1)))
        if curr_id == target_id and pair_idx + 1 < len(pairs):
            pair_idx  += 1
            start_id   = pairs[pair_idx][0]
            target_id  = pairs[pair_idx][1]

        # LiDAR
        hits = simulate_lidar(pos, yaw, geom.wall_obbs, circles=obs_circles)
        if len(hits) > 0:
            dsts = np.linalg.norm(hits, axis=1)
            hits = hits[dsts > 0.005]
        hits_in = hits if len(hits) > 0 else None

        # Rollouts
        v_seqs  = np.random.uniform(0.0, 0.75, (K, N))
        om_seqs = np.random.uniform(-1.5, 1.5,  (K, N))
        ctrl    = np.stack([v_seqs, om_seqs], axis=2)
        rollouts    = unicycle_rollout(np.zeros(3), ctrl, MPC_DT)
        traj_local  = rollouts[:, 1:, :]

        mpc._occ_grid.update(hits_in)
        dist_values = mpc._occ_grid.check_collisions(traj_local)

        end_local  = traj_local[:, -1, :2]
        end_global = end_local @ R_l2g.T + pos

        # Metric scoring: proximity to DISTORTED target centroid (single change vs run_ours)
        tgt_global  = dist_cents[target_id]
        dist_to_tgt = np.linalg.norm(end_global - tgt_global, axis=1)
        scores      = 10.0 / (1.0 + dist_to_tgt)

        # Weak axis-progress to avoid stall
        end_along = (end_global - pos) @ geom.axis_dir
        scores += 0.3 * w_progress * np.clip(end_along, 0, None)

        # EDT mask
        collision = (dist_values < mpc.safety_radius).any(axis=1)
        scores[collision] = -np.inf

        if np.all(~np.isfinite(scores)):
            om_cmd = 0.0; v_cmd = 0.15; best_k = 0
        else:
            best_k = int(np.argmax(scores))
            v_cmd  = float(ctrl[best_k, 0, 0])
            om_cmd = float(ctrl[best_k, 0, 1])

        v_cmd, om_cmd = _cbf_filter(v_cmd, om_cmd, hits_in)

        state, hit = _step(state, v_cmd, om_cmd, MPC_DT,
                           wall_obbs=geom.wall_obbs, obs_circles=obs_circles,
                           hard_physics=hard_physics, return_hit=True)
        wall_hit_count += int(hit)
        traj.append(state[:3].copy())
        if t % 10 == 9:
            print(".", end="", flush=True)

        along  = float((state[:2] - bridge_center) @ geom.axis_dir)
        across = float((state[:2] - bridge_center) @ geom.perp_dir)
        # A large axis-projection alone doesn't mean it crossed the bridge —
        # a path that veers off to the side can still have a large `along`
        # component by coincidence. Require it to also be roughly within the
        # corridor laterally, not just "far enough in the right direction."
        lateral_ok = abs(across) < (geom.gap / 2 + 0.20)
        if lateral_ok:
            max_along = max(max_along, along)
        if along > geom.length / 2 + 0.10 and lateral_ok:
            reached_exit = True
            for _ in range(t + 1, T_SIM):
                traj.append(traj[-1].copy())
            break

    print(f" done. final=({state[0]:.2f},{state[1]:.2f})")
    exit_thresh = geom.length / 2 + 0.10
    return {
        "traj": np.array(traj),
        "reached_exit": reached_exit,
        "final_progress_frac": float(np.clip(max_along / exit_thresh, 0.0, 1.0)),
        "wall_hit_count": wall_hit_count,
    }


def run_ours_dead_reckoning(assoc: BehaviorAssociator, rid: dict,
                            geom: RunGeom,
                            true_cents: np.ndarray,
                            dist_cents: np.ndarray,
                            obs_circles: Optional[List[Tuple]] = None,
                            hard_physics: bool = True,
                            debug: bool = False,
                            weights: Optional[dict] = None,
                            drift_params: Optional[dict] = None) -> dict:
    """
    SamplingMPC with topological guidance, PLUS a real dead-reckoning model.

    Identical to run_ours() except every scoring/decision use of the robot's
    own pose (Tier A pair-advancement, Tier B Voronoi cluster membership,
    bearing-to-target, LiDAR-PCA corridor centering) reads a drifting pose
    ESTIMATE rather than ground truth. Only LiDAR simulation, the actual
    physics integration, and the exit/termination bookkeeping stay on true
    state — those represent real sensing, real motion, and the simulator's
    own record of what actually happened, none of which the robot's odometry
    belief can affect.

    This models the real deployment gap run_ours() doesn't: on a real robot
    there is no SLAM ground truth, only a dead-reckoned (visual-inertial
    odometry) estimate that drifts from truth over distance/time. Combined
    with dist_cents (the existing distorted-map condition), this measures
    tolerance to the two real, independent error sources at once instead of
    only the map.
    """
    w = weights or {}
    w_target    = w.get("weight_target",    10.0)
    w_forbidden = w.get("weight_forbidden", -15.0)
    w_bearing   = w.get("weight_bearing",    3.0)
    w_corridor  = w.get("weight_corridor",   2.0)
    w_progress  = w.get("weight_progress",   0.8)

    d = drift_params or {}
    trans_bias      = d.get("odom_trans_bias",      1.0)   # persistent per-run multiplicative bias
    yaw_drift_rate  = d.get("odom_yaw_drift_rate",  0.0)   # persistent per-run rad/s bias
    trans_noise_std = d.get("odom_trans_noise_std", 0.0)   # per-step noise, metres
    yaw_noise_std   = d.get("odom_yaw_noise_std",   0.0)   # per-step noise, radians

    mpc = SamplingMPC(K=500, N=8, dt=MPC_DT, safety_radius=0.06, lidar_grid_size=1.5)
    mpc.reset(assoc)
    mpc._centroids_global = dist_cents.copy()

    n_c = true_cents.shape[0]
    state  = geom.robot_start.copy()
    traj   = [state[:3].copy()]

    # Dead-reckoned pose estimate — starts at the true start pose (localization
    # initializes correctly), diverges from there as the robot moves.
    est_pos = geom.robot_start[:2].copy()
    est_yaw = float(geom.robot_start[2])

    pairs = [
        (rid["approach_bridge_0"], rid["on_bridge_0"]),
        (rid["on_bridge_0"],       rid["exit_bridge_0"]),
    ]
    pair_idx  = 0
    start_id  = pairs[0][0]
    target_id = pairs[0][1]
    forbidden = [c for c in range(n_c) if c not in (start_id, target_id)]

    record_step       = 5
    last_rollouts     = None
    last_best_k       = 0
    recorded_pos      = geom.robot_start[:2].copy()
    recorded_yaw      = geom.robot_start[2]
    record_tgt_cent   = None
    last_cluster_parts = None
    last_bearing_parts = None
    stuck_count       = 0
    reached_exit      = False
    max_along         = -1e9
    wall_hit_count    = 0

    K, N = 500, 8
    bridge_center = np.array([geom.cx, geom.cy])

    print("  Ours-dead-reckoning ...", end="", flush=True)
    for t in range(T_SIM):
        true_pos = state[:2]; true_yaw = float(state[2])
        # Every scoring/decision use below reads the ESTIMATE, not truth —
        # this is exactly what run_ours() calls `pos`/`yaw`, just re-pointed.
        pos, yaw = est_pos, est_yaw
        cy_, sy_ = math.cos(yaw), math.sin(yaw)
        R_g2l = np.array([[ cy_,  sy_], [-sy_,  cy_]])
        R_l2g = np.array([[ cy_, -sy_], [ sy_,  cy_]])

        # ── Tier A: pair advancement via nearest true centroid (Voronoi) ─────
        curr_id = int(np.argmin(np.linalg.norm(true_cents - pos, axis=1)))

        if curr_id == target_id and pair_idx + 1 < len(pairs):
            pair_idx  += 1
            start_id   = pairs[pair_idx][0]
            target_id  = pairs[pair_idx][1]
            forbidden  = [c for c in range(n_c) if c not in (start_id, target_id)]

        # ── LiDAR — real sensing, must use TRUE pose ───────────────────────────
        hits = simulate_lidar(true_pos, true_yaw, geom.wall_obbs, circles=obs_circles)
        if len(hits) > 0:
            hit_dists = np.linalg.norm(hits, axis=1)
            hits = hits[hit_dists > 0.005]
        hits_in = hits if len(hits) > 0 else None

        # ── Generate rollouts ─────────────────────────────────────────────────
        v_seqs  = np.random.uniform(0.0, 0.75, (K, N))
        om_seqs = np.random.uniform(-1.5, 1.5,  (K, N))
        ctrl    = np.stack([v_seqs, om_seqs], axis=2)
        rollouts    = unicycle_rollout(np.zeros(3), ctrl, MPC_DT)
        traj_local  = rollouts[:, 1:, :]

        # ── EDT collision check ───────────────────────────────────────────────
        mpc._occ_grid.update(hits_in)
        dist_values = mpc._occ_grid.check_collisions(traj_local)

        # ── Rollout endpoints, scored against the pose ESTIMATE ────────────────
        end_local  = traj_local[:, -1, :2]
        end_global = end_local @ R_l2g.T + pos

        # ── Tier B: TRUE-centroid Voronoi, but against the ESTIMATED endpoint ──
        dists_true      = np.linalg.norm(end_global[:, None, :] - true_cents[None, :, :], axis=2)
        nearest_cluster = np.argmin(dists_true, axis=1)

        in_target    = (nearest_cluster == target_id).astype(float)
        in_start     = (nearest_cluster == start_id).astype(float)
        in_forbidden = np.isin(nearest_cluster, forbidden).astype(float)

        cluster_parts = w_target * in_target + 1.0 * in_start + w_forbidden * in_forbidden
        scores = cluster_parts.copy()

        # ── Bearing toward DISTORTED target centroid (map direction hint) ─────
        tgt_global    = dist_cents[target_id]
        tgt_local     = (tgt_global - pos) @ R_g2l.T
        bearing_local = math.atan2(tgt_local[1], tgt_local[0])
        bearing_cos   = w_bearing * np.cos(traj_local[:, -1, 2] - bearing_local)

        cdir   = tgt_local / (np.linalg.norm(tgt_local) + 1e-6)
        end_fx = np.cos(traj_local[:, -1, 2])
        end_fy = np.sin(traj_local[:, -1, 2])
        bearing_dot   = 1.0 * (end_fx * cdir[0] + end_fy * cdir[1])
        bearing_parts = bearing_cos + bearing_dot
        scores += bearing_parts

        # ── Corridor centering — LiDAR-derived axis (map-free) ───────────────
        # Real hits reprojected to "world" via the pose ESTIMATE — on a real
        # robot this reprojection is exactly what dead reckoning corrupts.
        lidar_perp   = geom.perp_dir
        lidar_center = bridge_center
        along_from_center = float((pos - bridge_center) @ geom.axis_dir)
        inside_bridge = along_from_center > -(geom.length / 2)
        if inside_bridge and hits_in is not None and len(hits_in) >= 4:
            hits_g = hits_in @ R_l2g.T + pos
            lateral_mask = np.abs(hits_in[:, 0]) < np.abs(hits_in[:, 1])
            pca_hits = hits_g[lateral_mask] if lateral_mask.sum() >= 4 else hits_g
            _, evecs = np.linalg.eigh(np.cov(pca_hits.T))
            lidar_perp = evecs[:, 0]
            if np.dot(lidar_perp, geom.perp_dir) < 0:
                lidar_perp = -lidar_perp
            proj       = hits_g @ lidar_perp
            upper_hits = hits_g[proj > np.median(proj)]
            lower_hits = hits_g[proj < np.median(proj)]
            if len(upper_hits) > 0 and len(lower_hits) > 0:
                lidar_center = (upper_hits.mean(axis=0) + lower_hits.mean(axis=0)) / 2

        lidar_axis = np.array([-lidar_perp[1], lidar_perp[0]])
        if np.dot(lidar_axis, geom.axis_dir) < 0:
            lidar_axis = -lidar_axis

        end_across = (end_global - lidar_center) @ lidar_perp
        scores += w_corridor * np.exp(-(end_across**2) / (0.10**2))

        end_along = (end_global - pos) @ lidar_axis
        scores += w_progress * np.clip(end_along, 0, None)

        # ── EDT hard collision mask ───────────────────────────────────────────
        collision = (dist_values < mpc.safety_radius).any(axis=1)
        scores[collision] = -np.inf

        if np.all(~np.isfinite(scores)):
            across_to_center = float((bridge_center - pos) @ geom.perp_dir)
            om_cmd = 1.5 * np.sign(across_to_center)
            v_cmd  = 0.15
            best_k = 0
            recovery = True
        else:
            best_k = int(np.argmax(scores))
            v_cmd  = float(ctrl[best_k, 0, 0])
            om_cmd = float(ctrl[best_k, 0, 1])
            recovery = False

        v_cmd, om_cmd = _cbf_filter(v_cmd, om_cmd, hits_in)

        if recovery and v_cmd < 0.04:
            stuck_count += 1
            if stuck_count >= 5:
                v_cmd = -0.20
                om_cmd = 0.0
                stuck_count = 0
        else:
            stuck_count = 0

        mpc._last_rollouts = rollouts
        mpc._last_best_k   = best_k

        if t == record_step:
            last_rollouts      = rollouts.copy()
            last_best_k        = best_k
            recorded_pos       = pos.copy()
            recorded_yaw       = yaw
            record_tgt_cent    = dist_cents[target_id].copy()
            last_cluster_parts = cluster_parts.copy()
            last_bearing_parts = bearing_parts.copy()

        if debug:
            d_min = float(np.min(np.linalg.norm(hits_in, axis=1))) if hits_in is not None and len(hits_in) > 0 else float('inf')
            drift_err = float(np.linalg.norm(est_pos - true_pos))
            print(f"\n  t={t:3d}  true=({true_pos[0]:.3f},{true_pos[1]:.3f})  "
                  f"est=({pos[0]:.3f},{pos[1]:.3f})  drift_err={drift_err:.3f}m"
                  f"  cluster={curr_id}→{target_id}  v={v_cmd:.3f}  ω={om_cmd:.3f}"
                  f"  d_min={d_min:.3f}  recovery={recovery}")

        # ── Real physics: TRUE state only ──────────────────────────────────────
        state_before = state.copy()
        state, hit = _step(state, v_cmd, om_cmd, MPC_DT,
                           wall_obbs=geom.wall_obbs, obs_circles=obs_circles,
                           hard_physics=hard_physics, return_hit=True)
        wall_hit_count += int(hit)
        traj.append(state[:3].copy())
        if t % 10 == 9 and not debug:
            print(".", end="", flush=True)

        # ── Dead-reckoning update: corrupt the TRUE incremental motion, then
        # integrate into the ESTIMATE using the estimate's own heading — this
        # is what makes heading bias compound into position error over
        # distance, the dominant real-world odometry failure mode.
        dp_world_true = state[:2] - state_before[:2]
        dyaw_true     = float(state[2] - state_before[2])
        c0, s0 = math.cos(state_before[2]), math.sin(state_before[2])
        dp_body_true = np.array([
            c0 * dp_world_true[0] + s0 * dp_world_true[1],
            -s0 * dp_world_true[0] + c0 * dp_world_true[1],
        ])
        dp_body_meas = dp_body_true * trans_bias
        if trans_noise_std > 0:
            dp_body_meas = dp_body_meas + np.random.normal(0, trans_noise_std, 2)
        dyaw_meas = dyaw_true + yaw_drift_rate * MPC_DT
        if yaw_noise_std > 0:
            dyaw_meas += np.random.normal(0, yaw_noise_std)

        ce, se = math.cos(est_yaw), math.sin(est_yaw)
        est_pos = est_pos + np.array([
            ce * dp_body_meas[0] - se * dp_body_meas[1],
            se * dp_body_meas[0] + ce * dp_body_meas[1],
        ])
        est_yaw = est_yaw + dyaw_meas

        along  = float((state[:2] - bridge_center) @ geom.axis_dir)
        across = float((state[:2] - bridge_center) @ geom.perp_dir)
        # A large axis-projection alone doesn't mean it crossed the bridge —
        # a path that veers off to the side can still have a large `along`
        # component by coincidence. Require it to also be roughly within the
        # corridor laterally, not just "far enough in the right direction."
        lateral_ok = abs(across) < (geom.gap / 2 + 0.20)
        if lateral_ok:
            max_along = max(max_along, along)
        if along > geom.length / 2 + 0.10 and lateral_ok:
            reached_exit = True
            for _ in range(t + 1, T_SIM):
                traj.append(traj[-1].copy())
            break

    print(f" done. final=({state[0]:.2f},{state[1]:.2f})")
    exit_thresh = geom.length / 2 + 0.10
    return {
        "traj":               np.array(traj),
        "last_rollouts":      last_rollouts,
        "last_best_k":        last_best_k,
        "recorded_pos":       recorded_pos,
        "recorded_yaw":       recorded_yaw,
        "record_tgt_cent":    record_tgt_cent,
        "last_cluster_parts": last_cluster_parts,
        "last_bearing_parts": last_bearing_parts,
        "cents_used":         dist_cents.copy(),
        "reached_exit":       reached_exit,
        "final_progress_frac": float(np.clip(max_along / exit_thresh, 0.0, 1.0)),
        "wall_hit_count":     wall_hit_count,
        "final_est_pos":      est_pos.copy(),
        "final_true_pos":     state[:2].copy(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MPPI simulation runner
# ══════════════════════════════════════════════════════════════════════════════

def run_mppi(assoc: "BehaviorAssociator", rid: dict,
             geom: RunGeom,
             true_cents: np.ndarray,
             dist_cents: np.ndarray,
             obs_circles: Optional[List[Tuple]] = None,
             hard_physics: bool = True,
             debug: bool = False,
             weights: Optional[dict] = None,
             mppi_params: Optional[dict] = None) -> dict:
    """
    MPPI controller with the same topological scoring as run_ours.

    Identical environment, identical scoring function — the only difference is
    the sampling distribution and update rule:
      - run_ours:  K uniform samples → argmax → discard all context
      - run_mppi:  K OU-correlated samples around U_mean → importance-weighted
                   update of U_mean → execute U_mean[0] → shift forward

    This makes the comparison apples-to-apples: any performance difference is
    attributable solely to the sampling distribution and update rule.
    """
    w = weights or {}
    w_target    = w.get("weight_target",    10.0)
    w_forbidden = w.get("weight_forbidden", -15.0)
    w_bearing   = w.get("weight_bearing",    3.0)
    w_corridor  = w.get("weight_corridor",   2.0)
    w_progress  = w.get("weight_progress",   0.8)

    mp = mppi_params or {}
    controller = MPPIController(
        K=500, N=8, dt=MPC_DT,
        v_mean=0.40,
        v_sigma=mp.get("mppi_v_sigma",     0.20),
        om_sigma=mp.get("mppi_om_sigma",   0.40),
        temperature=mp.get("mppi_temperature", 1.0),
    )

    mpc = SamplingMPC(K=500, N=8, dt=MPC_DT, safety_radius=0.06, lidar_grid_size=1.5)
    mpc.reset(assoc)
    mpc._centroids_global = dist_cents.copy()

    n_c   = true_cents.shape[0]
    state = geom.robot_start.copy()
    traj  = [state[:3].copy()]

    pairs = [
        (rid["approach_bridge_0"], rid["on_bridge_0"]),
        (rid["on_bridge_0"],       rid["exit_bridge_0"]),
    ]
    pair_idx  = 0
    start_id  = pairs[0][0]
    target_id = pairs[0][1]
    forbidden = [c for c in range(n_c) if c not in (start_id, target_id)]

    record_step        = 5
    last_rollouts      = None
    last_best_k        = 0
    recorded_pos       = geom.robot_start[:2].copy()
    recorded_yaw       = geom.robot_start[2]
    record_tgt_cent    = None
    last_cluster_parts = None
    last_bearing_parts = None
    stuck_count        = 0
    reached_exit       = False
    max_along          = -1e9
    wall_hit_count     = 0

    K, N          = 500, 8
    bridge_center = np.array([geom.cx, geom.cy])

    print("  MPPI ...", end="", flush=True)
    for t in range(T_SIM):
        pos = state[:2]; yaw = state[2]
        cy_, sy_ = math.cos(yaw), math.sin(yaw)
        R_g2l = np.array([[ cy_,  sy_], [-sy_,  cy_]])
        R_l2g = np.array([[ cy_, -sy_], [ sy_,  cy_]])

        # ── Tier A: pair advancement ──────────────────────────────────────────
        curr_id = int(np.argmin(np.linalg.norm(true_cents - pos, axis=1)))
        if curr_id == target_id and pair_idx + 1 < len(pairs):
            pair_idx  += 1
            start_id   = pairs[pair_idx][0]
            target_id  = pairs[pair_idx][1]
            forbidden  = [c for c in range(n_c) if c not in (start_id, target_id)]
            controller.reset()   # reset mean when plan step changes

        # ── LiDAR ────────────────────────────────────────────────────────────
        hits = simulate_lidar(pos, yaw, geom.wall_obbs, circles=obs_circles)
        if len(hits) > 0:
            hits = hits[np.linalg.norm(hits, axis=1) > 0.005]
        hits_in = hits if len(hits) > 0 else None

        # ── Sample rollouts around U_mean (OU-correlated) ────────────────────
        v_seqs, om_seqs = controller.sample()
        ctrl     = np.stack([v_seqs, om_seqs], axis=2)
        rollouts = unicycle_rollout(np.zeros(3), ctrl, MPC_DT)
        traj_local = rollouts[:, 1:, :]

        # ── EDT collision check ───────────────────────────────────────────────
        mpc._occ_grid.update(hits_in)
        dist_values = mpc._occ_grid.check_collisions(traj_local)

        # ── Rollout endpoints in global frame ─────────────────────────────────
        end_local  = traj_local[:, -1, :2]
        end_global = end_local @ R_l2g.T + pos

        # ── Scoring (identical to run_ours) ───────────────────────────────────
        dists_true      = np.linalg.norm(
            end_global[:, None, :] - true_cents[None, :, :], axis=2)
        nearest_cluster = np.argmin(dists_true, axis=1)
        in_target    = (nearest_cluster == target_id).astype(float)
        in_start     = (nearest_cluster == start_id).astype(float)
        in_forbidden = np.isin(nearest_cluster, forbidden).astype(float)
        cluster_parts = w_target * in_target + 1.0 * in_start + w_forbidden * in_forbidden
        scores = cluster_parts.copy()

        tgt_global    = dist_cents[target_id]
        tgt_local_v   = (tgt_global - pos) @ R_g2l.T
        bearing_local = math.atan2(tgt_local_v[1], tgt_local_v[0])
        bearing_cos   = w_bearing * np.cos(traj_local[:, -1, 2] - bearing_local)
        cdir          = tgt_local_v / (np.linalg.norm(tgt_local_v) + 1e-6)
        bearing_dot   = 1.0 * (
            np.cos(traj_local[:, -1, 2]) * cdir[0] +
            np.sin(traj_local[:, -1, 2]) * cdir[1])
        bearing_parts = bearing_cos + bearing_dot
        scores += bearing_parts

        lidar_perp    = geom.perp_dir
        lidar_center  = bridge_center
        inside_bridge = float((pos - bridge_center) @ geom.axis_dir) > -(geom.length / 2)
        if inside_bridge and hits_in is not None and len(hits_in) >= 4:
            hits_g       = hits_in @ R_l2g.T + pos
            lat_mask     = np.abs(hits_in[:, 0]) < np.abs(hits_in[:, 1])
            pca_hits     = hits_g[lat_mask] if lat_mask.sum() >= 4 else hits_g
            _, evecs     = np.linalg.eigh(np.cov(pca_hits.T))
            lidar_perp   = evecs[:, 0]
            if np.dot(lidar_perp, geom.perp_dir) < 0:
                lidar_perp = -lidar_perp
            proj         = hits_g @ lidar_perp
            upper_hits   = hits_g[proj > np.median(proj)]
            lower_hits   = hits_g[proj < np.median(proj)]
            if len(upper_hits) > 0 and len(lower_hits) > 0:
                lidar_center = (upper_hits.mean(axis=0) + lower_hits.mean(axis=0)) / 2

        lidar_axis = np.array([-lidar_perp[1], lidar_perp[0]])
        if np.dot(lidar_axis, geom.axis_dir) < 0:
            lidar_axis = -lidar_axis

        scores += w_corridor * np.exp(
            -((end_global - lidar_center) @ lidar_perp) ** 2 / (0.10 ** 2))
        scores += w_progress * np.clip((end_global - pos) @ lidar_axis, 0, None)

        collision = (dist_values < mpc.safety_radius).any(axis=1)
        scores[collision] = -np.inf

        # ── MPPI update then execute mean ─────────────────────────────────────
        controller.update(scores, v_seqs, om_seqs)

        if np.all(~np.isfinite(scores)):
            across_to_center = float((bridge_center - pos) @ geom.perp_dir)
            om_cmd   = 1.5 * np.sign(across_to_center)
            v_cmd    = 0.15
            best_k   = 0
            recovery = True
        else:
            v_cmd, om_cmd = controller.get_cmd()
            best_k   = int(np.argmax(scores))   # argmax for fan visualization only
            recovery = False

        controller.shift()
        v_cmd, om_cmd = _cbf_filter(v_cmd, om_cmd, hits_in)

        if recovery and v_cmd < 0.04:
            stuck_count += 1
            if stuck_count >= 5:
                v_cmd = -0.20; om_cmd = 0.0; stuck_count = 0
        else:
            stuck_count = 0

        if t == record_step:
            last_rollouts      = rollouts.copy()
            last_best_k        = best_k
            recorded_pos       = pos.copy()
            recorded_yaw       = yaw
            record_tgt_cent    = dist_cents[target_id].copy()
            last_cluster_parts = cluster_parts.copy()
            last_bearing_parts = bearing_parts.copy()

        if debug:
            d_min = (float(np.min(np.linalg.norm(hits_in, axis=1)))
                     if hits_in is not None and len(hits_in) > 0 else float('inf'))
            print(f"\n  t={t:3d}  pos=({pos[0]:.3f},{pos[1]:.3f})  "
                  f"yaw={math.degrees(yaw):.1f}°  {curr_id}→{target_id}  "
                  f"v={v_cmd:.3f}  ω={om_cmd:.3f}  d_min={d_min:.3f}  "
                  f"recovery={recovery}")

        state, hit = _step(state, v_cmd, om_cmd, MPC_DT,
                           wall_obbs=geom.wall_obbs, obs_circles=obs_circles,
                           hard_physics=hard_physics, return_hit=True)
        wall_hit_count += int(hit)
        traj.append(state[:3].copy())
        if t % 10 == 9 and not debug:
            print(".", end="", flush=True)

        along  = float((state[:2] - bridge_center) @ geom.axis_dir)
        across = float((state[:2] - bridge_center) @ geom.perp_dir)
        lateral_ok = abs(across) < (geom.gap / 2 + 0.20)
        if lateral_ok:
            max_along = max(max_along, along)
        if along > geom.length / 2 + 0.10 and lateral_ok:
            reached_exit = True
            for _ in range(t + 1, T_SIM):
                traj.append(traj[-1].copy())
            break

    print(f" done. final=({state[0]:.2f},{state[1]:.2f})")
    exit_thresh = geom.length / 2 + 0.10
    return {
        "traj":                np.array(traj),
        "last_rollouts":       last_rollouts,
        "last_best_k":         last_best_k,
        "recorded_pos":        recorded_pos,
        "recorded_yaw":        recorded_yaw,
        "record_tgt_cent":     record_tgt_cent,
        "last_cluster_parts":  last_cluster_parts,
        "last_bearing_parts":  last_bearing_parts,
        "cents_used":          dist_cents.copy(),
        "reached_exit":        reached_exit,
        "final_progress_frac": float(np.clip(max_along / exit_thresh, 0.0, 1.0)),
        "wall_hit_count":      wall_hit_count,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ══════════════════════════════════════════════════════════════════════════════

def _draw_walls(ax, wall_obbs: list) -> None:
    for (wcx, wcy, wlen, wthick, wtheta) in wall_obbs:
        c, s  = math.cos(wtheta), math.sin(wtheta)
        hw, hh = wlen / 2, wthick / 2
        corners_local = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
        R = np.array([[c, -s], [s, c]])
        corners = corners_local @ R.T + np.array([wcx, wcy])
        ax.add_patch(mpatches.Polygon(corners, closed=True,
            lw=1.5, edgecolor="#222", facecolor="#666", zorder=6))


def _compute_ax_limits(geom: RunGeom, padding: float = 0.40):
    """Dynamic axis limits centred on bridge, encompassing robot start and exit."""
    bridge_center = np.array([geom.cx, geom.cy])
    exit_pt  = bridge_center + (geom.length / 2 + 0.70) * geom.axis_dir
    start_pt = geom.robot_start[:2]
    points   = np.vstack([start_pt, bridge_center, exit_pt])
    xmin, ymin = points.min(axis=0) - padding
    xmax, ymax = points.max(axis=0) + padding
    size = max(xmax - xmin, ymax - ymin)
    xctr = (xmin + xmax) / 2
    yctr = (ymin + ymax) / 2
    return xctr - size / 2, xctr + size / 2, yctr - size / 2, yctr + size / 2


def _setup_ax(ax, assoc: BehaviorAssociator, geom: RunGeom) -> None:
    assoc.visualize_behavior_regions(ax)
    _draw_walls(ax, geom.wall_obbs)
    xl0, xl1, yl0, yl1 = _compute_ax_limits(geom)
    ax.set_xlim(xl0, xl1); ax.set_ylim(yl0, yl1)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", fontsize=9); ax.set_ylabel("y (m)", fontsize=9)
    ax.tick_params(labelsize=8)


def _draw_arc_fan(ax, result, base_colour, *, thin_to=60):
    """Draw a terrain-MPC candidate fan so the DECISION is legible.

    A fan drawn at one alpha in one colour does not show which arc was chosen. Four
    things carry the meaning instead:

      * rejected arcs, faint, in the arm's colour;
      * arcs a HARD terrain class vetoed, dashed red -- absent when there is no terrain;
      * the CHOSEN arc, thick and opaque;
      * the arc BEARING ALONE would have taken, dashed gold, drawn only when it differs
        from the chosen one. That gap IS the terrain/obstacle term doing
        something, and without it the viewer has to take the thick arc on trust.

    Everything here is drawn from state captured at the SAME tick (see `fan_best` in
    run_terrain_mpc); nothing is recomputed from the config.
    """
    fr = result.get("last_rollouts")
    if fr is None:
        return
    yaw = result["recorded_yaw"]
    c_, s_ = math.cos(yaw), math.sin(yaw)
    R = np.array([[c_, -s_], [s_, c_]])
    xy_of = lambda i: fr[i, :, :2] @ R.T + result["recorded_pos"]

    best = result.get("last_best_k")
    forbid = result.get("last_arc_forbidden")
    step = max(1, len(fr) // thin_to)
    drawn = set(range(0, len(fr), step))
    if best is not None:
        drawn.discard(best)                      # drawn separately, on top

    # Colour the fan by TERRAIN COST when there is any variation to show -- the cost
    # landscape is the thing being explained, and a fan at one alpha cannot show it. With
    # terrain off (the default) `soft` is all zeros and this falls back
    # to the faint single colour, so the comparison figure is unchanged.
    soft = result.get("last_arc_soft")
    graded = soft is not None and len(soft) == len(fr) and float(np.ptp(soft)) > 1e-9
    if graded:
        import matplotlib.cm as _cm
        from matplotlib.colors import Normalize as _Norm
        norm, cmap = _Norm(vmin=0.0, vmax=float(np.max(soft))), _cm.get_cmap("YlOrRd")

    for i in sorted(drawn):
        vetoed = forbid is not None and i < len(forbid) and bool(forbid[i])
        if vetoed:
            ax.plot(*xy_of(i).T, color="red", alpha=0.5, lw=0.9,
                    linestyle=":", zorder=3)
        elif graded:
            ax.plot(*xy_of(i).T, color=cmap(norm(float(soft[i]))), alpha=0.75,
                    lw=1.0, zorder=2)
        else:
            ax.plot(*xy_of(i).T, color=base_colour, alpha=0.16, lw=0.6, zorder=2)

    # What bearing alone would have picked: max cos(th_end - bearing_ref), the same
    # quantity the controller's bearing term maximises, over the same arcs.
    bref = result.get("last_bearing_ref")
    if bref is not None and best is not None:
        th_end = fr[:, -1, 2]
        b_only = int(np.argmax(np.cos(th_end - bref)))
        if b_only != best:
            ax.plot(*xy_of(b_only).T, color="deepskyblue", lw=2.0,
                    linestyle="--", zorder=8)

    if best is not None:
        xy = xy_of(best)
        ax.plot(*xy.T, color="white", lw=4.6, alpha=0.95, zorder=9)   # halo, so the
        ax.plot(*xy.T, color=base_colour, lw=2.6, alpha=1.0, zorder=10)  # arc reads over
        ax.scatter(*xy[-1], marker="o", s=42, facecolor=base_colour,    # a busy fan
                   edgecolor="white", lw=1.4, zorder=11)


def _draw_trajectory_gradient(ax, xy: np.ndarray, color: str,
                               label: str, lw: float = 2.0, zorder: int = 7) -> None:
    """Draw trajectory with alpha gradient: early=faint, late=opaque."""
    n = len(xy) - 1
    for i in range(n):
        alpha = 0.20 + 0.80 * (i / max(n - 1, 1))
        ax.plot(xy[i:i+2, 0], xy[i:i+2, 1], color=color, alpha=alpha,
                lw=lw, zorder=zorder, solid_capstyle="round")
    # Invisible entry for legend
    ax.plot([], [], color=color, lw=lw, label=label, alpha=1.0)


def _draw_mpc_fan(ax, rollouts: np.ndarray, best_k: int,
                  pos: np.ndarray, yaw: float,
                  cluster_parts: Optional[np.ndarray] = None,
                  bearing_parts: Optional[np.ndarray] = None,
                  n_show: int = 60) -> None:
    cy_, sy_ = math.cos(yaw), math.sin(yaw)
    R = np.array([[cy_, -sy_], [sy_, cy_]])
    K_total = len(rollouts)
    sampled = np.random.choice(K_total, size=min(n_show, K_total), replace=False)
    for k in sampled:
        xy = rollouts[k, :, :2] @ R.T + pos
        if cluster_parts is not None and bearing_parts is not None:
            color = "cyan" if cluster_parts[k] > abs(bearing_parts[k]) else "magenta"
            alpha = 0.22
        else:
            color = "deepskyblue"; alpha = 0.18
        ax.plot(xy[:, 0], xy[:, 1], color=color, alpha=0.12, lw=0.6, zorder=5)
    # Best rollout — gold dashed
    bxy = rollouts[best_k, :, :2] @ R.T + pos
    ax.plot(bxy[:, 0], bxy[:, 1], color="gold", lw=1.8, linestyle="--", zorder=6)


def _draw_map_corridor(ax, rot_cents_3: np.ndarray, geom: RunGeom) -> None:
    approach_r, _, exit_r = rot_cents_3[0], rot_cents_3[1], rot_cents_3[2]
    vec    = exit_r - approach_r
    length = np.linalg.norm(vec)
    if length < 1e-4:
        return
    perp    = np.array([-vec[1], vec[0]]) / length * (geom.gap / 2)
    corners = np.array([approach_r - perp, approach_r + perp,
                        exit_r + perp,     exit_r - perp])
    ax.add_patch(mpatches.Polygon(corners, closed=True, facecolor="orange",
                                  edgecolor="darkorange", alpha=0.18, lw=1.5,
                                  linestyle="--", zorder=3,
                                  label="Map's corridor belief"))
    ax.annotate("", xy=exit_r, xytext=approach_r,
                arrowprops=dict(arrowstyle="->", color="darkorange",
                                lw=2.0, connectionstyle="arc3,rad=0.0"))


def _draw_obstacle(ax, obs: Tuple[float, float, float]) -> None:
    cx, cy_o, cr = obs
    ax.add_patch(mpatches.Circle((cx, cy_o), cr, color="saddlebrown",
                                 alpha=0.90, zorder=8))
    ax.add_patch(mpatches.Circle((cx, cy_o), cr + 0.06, fill=False,
                                 edgecolor="saddlebrown", alpha=0.45,
                                 lw=1.0, linestyle="--", zorder=7))


# ══════════════════════════════════════════════════════════════════════════════
# Figure builder
# ══════════════════════════════════════════════════════════════════════════════

def _build_figure(assoc, rid, true_cents,
                  ours_acc, ours_dist, ours_metric_acc, ours_metric_dist,
                  carson_acc, carson_dist,
                  wp_order, rot_cents, obs_list, rot_deg, scale, tx, ty,
                  geom, run_idx, n_runs,
                  mppi_acc=None, mppi_dist=None,
                  graded_acc=None, graded_dist=None,
                  terrain_acc=None, terrain_dist=None,
                  seg2_acc=None, seg2_dist=None):

    nav_ids      = [rid[n] for n in wp_order]
    true_cents_3 = true_cents[nav_ids]
    rot_cents_3  = rot_cents[nav_ids]

    fig, (ax_acc, ax_dist) = plt.subplots(
        1, 2, figsize=(14.0, 6.5),
        gridspec_kw={"bottom": 0.22, "top": 0.85})

    theta_deg = math.degrees(geom.theta)
    trans_str = f"  trans ({tx:+.2f},{ty:+.2f})m" if (abs(tx) > 1e-4 or abs(ty) > 1e-4) else ""
    fig.suptitle(
        f"Metric vs. Topological MPC — Bridge Navigation Under Map Distortion  "
        f"[run {run_idx+1}/{n_runs}]\n"
        f"Physical θ={theta_deg:.0f}°  |  Map error +{rot_deg:.0f}°  scale {scale:.2f}×"
        f"{trans_str}  |  Bridge {geom.length:.2f}m × gap {geom.gap:.2f}m",
        fontsize=11, weight="bold",
    )

    def _panel(ax, c_result, o_result, om_result, mppi_result, is_distorted,
               g_result=None, t_result=None, s2_result=None):
        _setup_ax(ax, assoc, geom)

        for obs in obs_list:
            _draw_obstacle(ax, obs)

        # True centroid markers (no connecting line — too cluttered)
        tc = true_cents_3
        ax.scatter(tc[:, 0], tc[:, 1], marker="*", color="gold", s=160,
                   edgecolors="goldenrod", linewidths=0.5, zorder=10)

        if is_distorted:
            _draw_map_corridor(ax, rot_cents_3, geom)
            rc = rot_cents_3
            # Distorted centroid markers only — no arrows to true centroids
            ax.scatter(rc[:, 0], rc[:, 1],
                       marker="P", color="darkorange", s=120,
                       edgecolors="saddlebrown", linewidths=0.5, zorder=10)

        # Carson (red)
        if c_result is not None:
            _draw_trajectory_gradient(ax, c_result["traj"][:, :2], "red",
                                      "Carson NMPC")
            ax.scatter(c_result["traj"][-1, 0], c_result["traj"][-1, 1],
                       marker="^", color="red", s=120, zorder=9)
            ph = c_result["planned"][T_SIM // 2]
            if ph is not None:
                ax.plot(ph[:, 0], ph[:, 1], "--", color="red",
                        lw=0.8, alpha=0.4, zorder=6)

        # Ours-metric ablation (dodgerblue)
        if om_result is not None:
            _draw_trajectory_gradient(ax, om_result["traj"][:, :2], "dodgerblue",
                                      "Ours-metric (endpoint ablation)")
            ax.scatter(om_result["traj"][-1, 0], om_result["traj"][-1, 1],
                       marker="s", color="dodgerblue", s=100, zorder=9)

        # Ours-topological (limegreen)
        _draw_trajectory_gradient(ax, o_result["traj"][:, :2], "limegreen",
                                  "Ours-cluster (endpoint ablation)")
        ax.scatter(o_result["traj"][-1, 0], o_result["traj"][-1, 1],
                   marker="D", color="limegreen", edgecolors="black",
                   s=110, zorder=10)

        # Ours, graded obstacle (teal) — THE SHIPPED DESIGN, not an ablation
        if g_result is not None:
            _draw_trajectory_gradient(ax, g_result["traj"][:, :2], "teal",
                                      "Ours (graded obstacle)")
            ax.scatter(g_result["traj"][-1, 0], g_result["traj"][-1, 1],
                       marker="*", color="teal", edgecolors="black",
                       s=170, zorder=12)

        # Terrain-MPC (black) — the SHIPPED CARLA controller, not a reimplementation
        if t_result is not None:
            _draw_arc_fan(ax, t_result, "dimgray")
        if t_result is not None:
            _draw_trajectory_gradient(ax, t_result["traj"][:, :2], "black",
                                      "Terrain-MPC (shipped CARLA)")
            ax.scatter(t_result["traj"][-1, 0], t_result["traj"][-1, 1],
                       marker="X", color="black", s=130, zorder=13)

        # Two-segment terrain MPC (orangered) — THE PROPOSED CHANGE
        if s2_result is not None:
            _draw_arc_fan(ax, s2_result, "orangered")
        if s2_result is not None:
            _draw_trajectory_gradient(ax, s2_result["traj"][:, :2], "orangered",
                                      "Terrain-MPC 2-segment (proposed)")
            ax.scatter(s2_result["traj"][-1, 0], s2_result["traj"][-1, 1],
                       marker="P", color="orangered", edgecolors="black", s=140, zorder=14)

        # MPPI (mediumorchid) — drawn on top
        if mppi_result is not None:
            _draw_trajectory_gradient(ax, mppi_result["traj"][:, :2], "mediumorchid",
                                      "MPPI (ours)")
            ax.scatter(mppi_result["traj"][-1, 0], mppi_result["traj"][-1, 1],
                       marker="h", color="mediumorchid", edgecolors="black",
                       s=120, zorder=11)

        # Start position
        ax.scatter(geom.robot_start[0], geom.robot_start[1],
                   marker="o", color="white", edgecolors="black",
                   s=90, zorder=10)

        # Rollout fan + guidance coloring
        if o_result["last_rollouts"] is not None:
            _draw_mpc_fan(ax, o_result["last_rollouts"], o_result["last_best_k"],
                          o_result["recorded_pos"],    o_result["recorded_yaw"],
                          o_result["last_cluster_parts"],
                          o_result["last_bearing_parts"])


    _panel(ax_acc,  carson_acc,  ours_acc,  ours_metric_acc,  mppi_acc,  is_distorted=False, g_result=graded_acc, t_result=terrain_acc, s2_result=seg2_acc)
    ax_acc.set_title("Accurate map  (true centroids)",
                     fontsize=10, weight="bold")

    _panel(ax_dist, carson_dist, ours_dist, ours_metric_dist, mppi_dist, is_distorted=True, g_result=graded_dist, t_result=terrain_dist, s2_result=seg2_dist)
    trans_lbl = f", ({tx:+.2f},{ty:+.2f})m trans" if (abs(tx) > 1e-4 or abs(ty) > 1e-4) else ""
    ax_dist.set_title(
        f"Distorted map  (+{rot_deg:.0f}° rotation, {scale:.2f}× scale{trans_lbl})\n"
        f"Orange overlay = overhead map's corridor belief",
        fontsize=9, weight="bold")

    # ── Shared legend ─────────────────────────────────────────────────────────
    legend_elements = [
        Line2D([0], [0], color="red",           lw=2, label="Carson NMPC"),
        Line2D([0], [0], color="dodgerblue",   lw=2, label="Ours-metric (endpoint ablation)"),
        Line2D([0], [0], color="limegreen",    lw=2, label="Ours-cluster (endpoint ablation)"),
        Line2D([0], [0], color="teal",         lw=2, label="Ours (graded obstacle)"),
        Line2D([0], [0], color="black",        lw=2, label="Terrain-MPC 1-seg (shipped)"),
        Line2D([0], [0], color="orangered",    lw=2, label="Terrain-MPC 2-seg (proposed)"),
        Line2D([0], [0], color="mediumorchid", lw=2, label="MPPI (ours)"),
        Line2D([0], [0], color="cyan",        lw=1.2, alpha=0.7,
               label="Rollout: cluster-driven (cyan)"),
        Line2D([0], [0], color="magenta",     lw=1.2, alpha=0.7,
               label="Rollout: bearing-driven (magenta)"),
        Line2D([0], [0], color="gold",        lw=1.6, linestyle="--",
               label="Best rollout at step 20"),
        Line2D([0], [0], color="dimgray",     lw=2.6,
               label="Terrain-MPC: arc CHOSEN"),
        Line2D([0], [0], color="deepskyblue", lw=2.0, linestyle="--",
               label="Terrain-MPC: arc bearing alone would pick"),
        Line2D([0], [0], color="red",         lw=1.2, linestyle=":",
               label="Terrain-MPC: arc vetoed by terrain class"),
        Line2D([0], [0], color="none", marker="^", markersize=9,
               markerfacecolor="red",        label="Carson endpoint"),
        Line2D([0], [0], color="none", marker="s", markersize=8,
               markerfacecolor="dodgerblue", label="Ours-metric endpoint"),
        Line2D([0], [0], color="none", marker="D", markersize=7,
               markerfacecolor="limegreen",    label="Ours-cluster endpoint"),
        Line2D([0], [0], color="none", marker="*", markersize=12,
               markerfacecolor="teal", markeredgecolor="black",
               label="Ours (graded) endpoint"),
        Line2D([0], [0], color="none", marker="h", markersize=8,
               markerfacecolor="mediumorchid", label="MPPI endpoint"),
        Line2D([0], [0], color="none", marker="o", markersize=8,
               markerfacecolor="white", markeredgecolor="black",
               label="Start position"),
        Line2D([0], [0], color="none", marker="*", markersize=11,
               markerfacecolor="gold",       label="True region centroids"),
        Line2D([0], [0], color="none", marker="P", markersize=8,
               markerfacecolor="darkorange", label=f"Distorted centroids (+{rot_deg:.0f}°)"),
        Patch(facecolor="orange", edgecolor="darkorange", alpha=0.25, linestyle="--",
              label="Map's believed corridor"),
        Patch(facecolor="saddlebrown", alpha=0.85, label="Physical obstacle"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=5, fontsize=7.5, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.01))

    return fig


def run_drift_path_comparison(params: dict = None,
                              levels: Optional[List[float]] = None,
                              n_trials_per_level: int = 3,
                              seed: Optional[int] = None):
    """
    Qualitative companion to run_drift_sweep(): where the sweep answers "how
    often does it fail," this answers "what does it actually look like when
    it degrades." Holds ONE bridge geometry fixed across all levels (so paths
    are visually comparable on the same scenario, unlike the sweep which
    re-randomizes geometry every trial for statistical coverage), runs
    n_trials_per_level trajectories per level, and draws them all on one plot
    over the bridge — color = drift severity, marker = reached exit or not.

    Returns a matplotlib Figure (caller saves/displays it — same convention
    as _build_figure).
    """
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    levels = levels if levels is not None else DRIFT_SWEEP_LEVELS

    global T_SIM, MPC_DT
    T_SIM  = p["t_sim"]
    MPC_DT = p["mpc_dt"]

    rng = np.random.default_rng(seed if seed is not None else p["seed"])

    bridge_theta = float(rng.uniform(0.0, 2 * math.pi)) if p["random_theta"] else 0.0
    bridge_len   = float(rng.uniform(p["bridge_len_min"], p["bridge_len_max"]))
    bridge_gap   = float(rng.uniform(p["bridge_gap_min"], p["bridge_gap_max"]))
    wall_thick   = float(rng.uniform(p["wall_thick_min"], p["wall_thick_max"]))
    rot_deg      = float(rng.uniform(p["rot_deg_min"], p["rot_deg_max"]))
    scale        = float(rng.uniform(p["scale_min"],   p["scale_max"]))
    trans_max = p.get("trans_max", 0.0)
    tx = float(rng.uniform(-trans_max, trans_max)) if trans_max > 0 else 0.0
    ty = float(rng.uniform(-trans_max, trans_max)) if trans_max > 0 else 0.0

    geom = make_run_geom(bridge_theta, bridge_len, bridge_gap, wall_thick)
    bridges_deg = [(cx, cy, l, t, math.degrees(th)) for cx, cy, l, t, th in geom.wall_obbs]
    assoc = BehaviorAssociator(bridges=bridges_deg, buildings=[], obstacles=[])
    rid   = assoc.region_name_to_id
    true_cents = np.array(assoc.all_region_centroids_jax_array)
    rot_cents  = _distort_pts(true_cents.copy(), rot_deg, scale, geom.rotation_ctr, tx, ty)

    weights_dict = {k: p[k] for k in (
        "weight_target", "weight_forbidden", "weight_bearing",
        "weight_corridor", "weight_progress",
        "weight_obstacle", "obstacle_hard_m", "obstacle_influence_m",
        "tm_segments", "tm_fan",
        # Terrain knobs. A key absent from THIS tuple never reaches the controller no
        # matter what the sidebar sets -- a silent no-op.
        "tm_terrain", "tm_w_terrain", "tm_soft_cost",
        "tm_terrain_forbid", "tm_blind_m", "tm_sidewalk_m",
        "tm_ban_road") if k in p}

    obs_r = p["obs_radius"]
    min_gap = obs_r + 2 * p["safety_radius"] + 0.06
    n_obs_possible = p["n_obs_max"] if bridge_gap >= min_gap else 0
    n_obs = int(rng.integers(0, min(n_obs_possible + 1, 4)))
    obs_list = []
    for _ in range(n_obs):
        along  = float(rng.uniform(-bridge_len / 2 + 0.15, bridge_len / 2 - 0.15))
        side   = float(rng.choice([-1.0, 1.0]))
        across = side * (bridge_gap * 0.30)
        world_pt = (np.array([geom.cx, geom.cy])
                    + along  * geom.axis_dir
                    + across * geom.perp_dir)
        obs_list.append((float(world_pt[0]), float(world_pt[1]), obs_r))

    max_level = max(levels) or 1e-6
    runs = []   # (level, traj_xy, reached_exit)
    for level in levels:
        trans_bias_range = (1.0 - level, 1.0 + level)
        yaw_drift_range  = (-level * YAW_DRIFT_RATE_SCALE, level * YAW_DRIFT_RATE_SCALE)
        trans_noise_std  = level * TRANS_NOISE_SCALE
        yaw_noise_std    = level * YAW_NOISE_SCALE

        for trial in range(n_trials_per_level):
            # Pick one endpoint of the range, not a continuous draw across it:
            # sampling uniformly across (1-level, 1+level) means most trials land
            # near the mild center of the range by chance, diluting "level" into
            # "somewhere between none and this much drift" instead of a
            # consistent severity.
            odom_trans_bias     = float(rng.choice(trans_bias_range))
            odom_yaw_drift_rate = float(rng.choice(yaw_drift_range))
            drift_params = dict(
                odom_trans_bias=odom_trans_bias,
                odom_yaw_drift_rate=odom_yaw_drift_rate,
                odom_trans_noise_std=trans_noise_std,
                odom_yaw_noise_std=yaw_noise_std,
            )
            np.random.seed(int((p["seed"] * 1000 + level * 100 + trial) * 1000) % (2**31))
            result = run_ours_dead_reckoning(
                assoc, rid, geom, true_cents, rot_cents.copy(),
                obs_circles=obs_list, hard_physics=p["hard_physics"],
                debug=False, weights=weights_dict, drift_params=drift_params)
            runs.append((level, result["traj"][:, :2], result["reached_exit"]))

    # ── Build the figure ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    _setup_ax(ax, assoc, geom)
    _draw_walls(ax, geom.wall_obbs)
    for obs in obs_list:
        _draw_obstacle(ax, obs)

    tc = true_cents[[rid[n] for n in ("approach_bridge_0", "on_bridge_0", "exit_bridge_0")]]
    ax.scatter(tc[:, 0], tc[:, 1], marker="*", color="gold", s=160,
               edgecolors="goldenrod", linewidths=0.5, zorder=10,
               label="True region centroids")

    cmap = plt.cm.RdYlGn_r
    for level, xy, reached in runs:
        color = cmap(level / max_level)
        ax.plot(xy[:, 0], xy[:, 1], color=color, alpha=0.75, lw=1.8, zorder=7)
        marker = "D" if reached else "X"
        ax.scatter(xy[-1, 0], xy[-1, 1], color=color, marker=marker,
                   s=90, edgecolors="black", linewidths=0.6, zorder=9)

    ax.scatter(geom.robot_start[0], geom.robot_start[1],
               marker="o", color="white", edgecolors="black", s=90, zorder=10,
               label="Start")

    legend_elements = [
        Line2D([0], [0], color=cmap(l / max_level), lw=2.5,
               label=f"Drift level {l:g}") for l in levels
    ] + [
        Line2D([0], [0], color="none", marker="D", markersize=8,
               markerfacecolor="gray", markeredgecolor="black", label="Reached exit"),
        Line2D([0], [0], color="none", marker="X", markersize=9,
               markerfacecolor="gray", markeredgecolor="black", label="Did not reach exit"),
        Line2D([0], [0], color="none", marker="*", markersize=11,
               markerfacecolor="gold", label="True region centroids"),
        Line2D([0], [0], color="none", marker="o", markersize=8,
               markerfacecolor="white", markeredgecolor="black", label="Start"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_title(
        f"Ours (topological) paths vs. dead-reckoning drift severity\n"
        f"(fixed bridge: len={geom.length:.2f}m, gap={geom.gap:.2f}m, "
        f"θ={math.degrees(geom.theta):.0f}° — {n_trials_per_level} trials/level)",
        fontsize=10, weight="bold")

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Callable entry point for Streamlit and other callers
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_PARAMS = dict(
    n_runs         = 5,
    seed           = 0,
    rot_deg_min    = 10.0,
    rot_deg_max    = 50.0,
    scale_min      = 0.80,
    scale_max      = 1.20,
    trans_max      = 0.0,   # max translation error per axis (m); 0 = off
    odom_trans_bias_min     = 1.00,  # persistent per-run multiplicative bias on sensed
    odom_trans_bias_max     = 1.00,  # body-frame translation; 1.0 = perfect odometry
    odom_yaw_drift_rate_min = 0.0,   # rad/s, persistent per-run constant heading-drift bias
    odom_yaw_drift_rate_max = 0.0,   # (gyro/calibration error)
    odom_trans_noise_std    = 0.0,   # m, per-step zero-mean Gaussian noise stdev per axis
    odom_yaw_noise_std      = 0.0,   # rad, per-step zero-mean Gaussian noise stdev
    use_dead_reckoning      = False, # if True, run_all also runs run_ours_dead_reckoning
    bridge_len_min = 0.50,
    bridge_len_max = 1.00,
    bridge_gap_min = 0.25,
    bridge_gap_max = 0.55,
    wall_thick_min = 0.05,
    wall_thick_max = 0.10,
    n_obs_max      = 2,
    obs_radius     = 0.04,
    n_rays         = 16,
    max_range      = 0.8,
    K_rollouts     = 500,
    N_horizon      = 8,
    safety_radius  = 0.06,
    d_safe         = 0.04,
    cbf_alpha      = 2.0,
    weight_target  = 10.0,
    weight_forbidden = -15.0,
    weight_bearing = 0.5,
    weight_corridor = 2.0,
    weight_progress = 0.8,
    # Graded obstacle response. UNTUNED placeholders scaled to this sim's dimensions
    # (safety_radius 0.06 m, gap 0.25-0.55 m); the CARLA values do not transfer.
    weight_obstacle      = 3.0,
    # Shipped-CARLA arm. segments=1 and scale=0.05 are the default configuration.
    tm_segments          = 1,
    tm_scale             = 0.05,
    tm_fan               = 65,
    tm_seg2_scale        = 0.02,
    obstacle_hard_m      = 0.04,
    obstacle_influence_m = 0.20,
    t_sim          = 80,
    mpc_dt         = 0.2,
    random_theta   = True,
    hard_physics   = True,
    show_metric_ablation = True,
    show_rollout_fan     = True,
    colour_rollouts      = True,
    debug                = False,
    show_mppi            = True,
    mppi_temperature     = 1.0,
    mppi_v_sigma         = 0.20,
    mppi_om_sigma        = 0.40,
)


def run_all(params: dict = None,
            progress_cb=None,
            nmpc_instance=None) -> list:
    """
    Run n_runs simulations with given params.  Returns list of matplotlib Figures.

    progress_cb: optional callable(run_idx, n_runs) called after each run.
    nmpc_instance: optional pre-built CarsonNMPC to reuse (avoids re-JIT).
    """
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)

    global T_SIM, MPC_DT
    T_SIM  = p["t_sim"]
    MPC_DT = p["mpc_dt"]

    if nmpc_instance is not None:
        nmpc = nmpc_instance
    elif CASADI_OK:
        nmpc = CarsonNMPC()
    else:
        nmpc = None

    wp_order = ["approach_bridge_0", "on_bridge_0", "exit_bridge_0"]
    rng = np.random.default_rng(p["seed"])
    figures = []

    for run_idx in range(p["n_runs"]):
        bridge_theta = float(rng.uniform(0.0, 2 * math.pi)) if p["random_theta"] else 0.0
        bridge_len   = float(rng.uniform(p["bridge_len_min"], p["bridge_len_max"]))
        bridge_gap   = float(rng.uniform(p["bridge_gap_min"], p["bridge_gap_max"]))
        wall_thick   = float(rng.uniform(p["wall_thick_min"], p["wall_thick_max"]))
        rot_deg      = float(rng.uniform(p["rot_deg_min"],    p["rot_deg_max"]))
        scale        = float(rng.uniform(p["scale_min"],      p["scale_max"]))

        trans_max = p.get("trans_max", 0.0)
        tx = float(rng.uniform(-trans_max, trans_max)) if trans_max > 0 else 0.0
        ty = float(rng.uniform(-trans_max, trans_max)) if trans_max > 0 else 0.0

        # Dead-reckoning drift — persistent per-run bias, sampled alongside the
        # existing map-distortion params above. Inert (perfect odometry) unless
        # odom_*_min/max are widened from their DEFAULT_PARAMS 1.0/0.0 defaults.
        odom_trans_bias = float(rng.uniform(p["odom_trans_bias_min"], p["odom_trans_bias_max"]))
        odom_yaw_drift_rate = float(rng.uniform(p["odom_yaw_drift_rate_min"], p["odom_yaw_drift_rate_max"]))
        drift_params = dict(
            odom_trans_bias=odom_trans_bias,
            odom_yaw_drift_rate=odom_yaw_drift_rate,
            odom_trans_noise_std=p["odom_trans_noise_std"],
            odom_yaw_noise_std=p["odom_yaw_noise_std"],
        )

        geom = make_run_geom(bridge_theta, bridge_len, bridge_gap, wall_thick)
        # BehaviorAssociator expects angles in degrees; wall_obbs stores radians for ray/physics
        bridges_deg = [(cx, cy, l, t, math.degrees(th)) for cx, cy, l, t, th in geom.wall_obbs]
        assoc = BehaviorAssociator(bridges=bridges_deg, buildings=[], obstacles=[])
        rid   = assoc.region_name_to_id
        true_cents = np.array(assoc.all_region_centroids_jax_array)
        rot_cents  = _distort_pts(true_cents.copy(), rot_deg, scale, geom.rotation_ctr, tx, ty)

        weights_dict = {k: p[k] for k in (
            "weight_target", "weight_forbidden", "weight_bearing",
            "weight_corridor", "weight_progress",
            "weight_obstacle", "obstacle_hard_m", "obstacle_influence_m",
            "tm_segments", "tm_fan",
        # Terrain knobs. A key absent from THIS tuple never reaches the controller no
        # matter what the sidebar sets -- a silent no-op.
        "tm_terrain", "tm_w_terrain", "tm_soft_cost",
        "tm_terrain_forbid", "tm_blind_m", "tm_sidewalk_m",
        "tm_ban_road") if k in p}

        # Only place obstacles when the bridge is wide enough to navigate around them.
        # Required navigable clearance: obs_r + 2*safety_r on each side of the gap.
        obs_r = p["obs_radius"]
        # Robot needs obs_r + safety_r clearance from obstacle on the side it passes,
        # plus safety_r from the wall on the other side.
        min_gap = obs_r + 2 * p["safety_radius"] + 0.06
        n_obs_possible = p["n_obs_max"] if bridge_gap >= min_gap else 0
        n_obs = int(rng.integers(0, min(n_obs_possible + 1, 4)))
        obs_list = []
        for _ in range(n_obs):
            along  = float(rng.uniform(-bridge_len / 2 + 0.15, bridge_len / 2 - 0.15))
            side   = float(rng.choice([-1.0, 1.0]))
            # Place at 60% from axis (not flush against wall): guarantees clear path on both sides
            across = side * (bridge_gap * 0.30)
            world_pt = (np.array([geom.cx, geom.cy])
                        + along  * geom.axis_dir
                        + across * geom.perp_dir)
            obs_list.append((float(world_pt[0]), float(world_pt[1]), obs_r))

        wp_acc  = [true_cents[rid[n]] for n in wp_order]
        wp_dist = [rot_cents[rid[n]]  for n in wp_order]

        np.random.seed(p["seed"] * 10000 + run_idx * 100 + 42)

        ours_acc  = run_ours(assoc, rid, geom, true_cents, true_cents.copy(),
                             obs_circles=obs_list, hard_physics=p["hard_physics"],
                             debug=p["debug"], weights=weights_dict)
        ours_dist = run_ours(assoc, rid, geom, true_cents, rot_cents.copy(),
                             obs_circles=obs_list, hard_physics=p["hard_physics"],
                             debug=p["debug"], weights=weights_dict)

        # THE SHIPPED DESIGN: same scorer, graded obstacle response. Runs as its own
        # line item so the two ablations it is being compared against -- metric endpoint
        # and cluster endpoint -- stay visible beside it rather than being replaced.
        ours_graded_acc  = run_ours(assoc, rid, geom, true_cents, true_cents.copy(),
                                    obs_circles=obs_list, hard_physics=p["hard_physics"],
                                    debug=p["debug"], weights=weights_dict,
                                    obstacle_mode="graded")
        ours_graded_dist = run_ours(assoc, rid, geom, true_cents, rot_cents.copy(),
                                    obs_circles=obs_list, hard_physics=p["hard_physics"],
                                    debug=p["debug"], weights=weights_dict,
                                    obstacle_mode="graded")

        # THE SHIPPED CARLA CONTROLLER, as its own line item.
        terrain_acc  = run_terrain_mpc(assoc, rid, geom, true_cents, true_cents.copy(),
                                       obs_circles=obs_list, hard_physics=p["hard_physics"],
                                       weights=weights_dict,
                                       scale=p.get("tm_scale", 0.05))
        terrain_dist = run_terrain_mpc(assoc, rid, geom, true_cents, rot_cents.copy(),
                                       obs_circles=obs_list, hard_physics=p["hard_physics"],
                                       weights=weights_dict, scale=p.get("tm_scale", 0.05))
        # THE PROPOSED CHANGE, as its own line rather than a toggle on the one above, so
        # the shipped controller and the two-segment variant are visible side by side.
        _w2 = dict(weights_dict); _w2["tm_segments"] = 2
        _w2["tm_fan"] = max(int(weights_dict.get("tm_fan", 65)), 225)
        seg2_acc  = run_terrain_mpc(assoc, rid, geom, true_cents, true_cents.copy(),
                                    obs_circles=obs_list, hard_physics=p["hard_physics"],
                                    weights=_w2, scale=p.get("tm_seg2_scale", 0.02))
        seg2_dist = run_terrain_mpc(assoc, rid, geom, true_cents, rot_cents.copy(),
                                    obs_circles=obs_list, hard_physics=p["hard_physics"],
                                    weights=_w2, scale=p.get("tm_seg2_scale", 0.02))

        if p["show_metric_ablation"]:
            ours_metric_acc  = run_ours_metric(assoc, rid, geom, true_cents,
                                               true_cents.copy(), obs_circles=obs_list,
                                               hard_physics=p["hard_physics"],
                                               weights=weights_dict)
            ours_metric_dist = run_ours_metric(assoc, rid, geom, true_cents,
                                               rot_cents.copy(),  obs_circles=obs_list,
                                               hard_physics=p["hard_physics"],
                                               weights=weights_dict)
        else:
            ours_metric_acc = ours_metric_dist = None

        if nmpc is not None:
            carson_acc  = run_carson(nmpc, wp_acc,  geom, obs_circles=obs_list,
                                     hard_physics=p["hard_physics"])
            carson_dist = run_carson(nmpc, wp_dist, geom, obs_circles=obs_list,
                                     hard_physics=p["hard_physics"])
        else:
            carson_acc = carson_dist = None

        mppi_params_dict = {k: p[k] for k in
                            ("mppi_temperature", "mppi_v_sigma", "mppi_om_sigma") if k in p}
        if p.get("show_mppi", True):
            mppi_acc  = run_mppi(assoc, rid, geom, true_cents, true_cents.copy(),
                                 obs_circles=obs_list, hard_physics=p["hard_physics"],
                                 debug=p["debug"], weights=weights_dict,
                                 mppi_params=mppi_params_dict)
            mppi_dist = run_mppi(assoc, rid, geom, true_cents, rot_cents.copy(),
                                 obs_circles=obs_list, hard_physics=p["hard_physics"],
                                 debug=p["debug"], weights=weights_dict,
                                 mppi_params=mppi_params_dict)
        else:
            mppi_acc = mppi_dist = None

        if p["use_dead_reckoning"]:
            ours_dr = run_ours_dead_reckoning(
                assoc, rid, geom, true_cents, rot_cents.copy(),
                obs_circles=obs_list, hard_physics=p["hard_physics"],
                debug=p["debug"], weights=weights_dict, drift_params=drift_params)
            print(f"  [dead-reckoning] reached_exit={ours_dr['reached_exit']}  "
                  f"final_progress_frac={ours_dr['final_progress_frac']:.2f}  "
                  f"wall_hit_count={ours_dr['wall_hit_count']}  "
                  f"trans_bias={odom_trans_bias:.3f}  yaw_drift_rate={odom_yaw_drift_rate:.3f}rad/s")

        fig = _build_figure(
            assoc, rid, true_cents,
            ours_acc, ours_dist, ours_metric_acc, ours_metric_dist,
            carson_acc, carson_dist,
            wp_order, rot_cents, obs_list,
            rot_deg, scale, tx, ty, geom, run_idx, p["n_runs"],
            mppi_acc=mppi_acc, mppi_dist=mppi_dist,
            graded_acc=ours_graded_acc, graded_dist=ours_graded_dist,
            terrain_acc=terrain_acc, terrain_dist=terrain_dist,
            seg2_acc=seg2_acc, seg2_dist=seg2_dist)

        figures.append(fig)
        if progress_cb is not None:
            progress_cb(run_idx + 1, p["n_runs"])

    return figures


# ══════════════════════════════════════════════════════════════════════════════
# Dead-reckoning tolerance sweep
# ══════════════════════════════════════════════════════════════════════════════

DRIFT_SWEEP_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]   # fractional severity, 0 = perfect odometry

# Tuned empirically against this bridge-crossing scenario (short T_SIM, short
# bridge — limited distance/time for drift to accumulate). At n=15 trials/level,
# per-trial bridge-geometry randomness (length/gap/theta/obstacles) is a bigger
# source of run-to-run variance than mild drift, so DRIFT_SWEEP_LEVELS spans up
# to 1.0 (not just 0.2) to get a signal above that noise floor. Raise
# sweep-trials for a cleaner curve if needed.
YAW_DRIFT_RATE_SCALE = 2.5    # rad/s, yaw_drift_rate sampled up to ±(level * this)
TRANS_NOISE_SCALE    = 0.10   # m,   per-step trans noise stdev at level=1.0
YAW_NOISE_SCALE      = 0.25   # rad, per-step yaw noise stdev at level=1.0


def run_drift_sweep(params: dict = None,
                    levels: Optional[List[float]] = None,
                    n_trials: int = 20,
                    results_dir: str = "results") -> List[dict]:
    """
    Headless dead-reckoning-drift tolerance sweep: for each severity level,
    run n_trials of run_ours_dead_reckoning (map distortion sampled exactly
    as run_all already does, PLUS drift derived from the level), and report
    success_rate / mean_final_progress_frac / mean_wall_hit_count per level.

    Skips _build_figure/PNG generation entirely — this is metrics-only, so
    it's cheap enough to run many more trials than the default PNG mode.
    """
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    levels = levels if levels is not None else DRIFT_SWEEP_LEVELS

    global T_SIM, MPC_DT
    T_SIM  = p["t_sim"]
    MPC_DT = p["mpc_dt"]

    os.makedirs(results_dir, exist_ok=True)
    weights_dict = {k: p[k] for k in (
        "weight_target", "weight_forbidden", "weight_bearing",
        "weight_corridor", "weight_progress",
        "weight_obstacle", "obstacle_hard_m", "obstacle_influence_m",
        "tm_segments", "tm_fan",
        # Terrain knobs. A key absent from THIS tuple never reaches the controller no
        # matter what the sidebar sets -- a silent no-op.
        "tm_terrain", "tm_w_terrain", "tm_soft_cost",
        "tm_terrain_forbid", "tm_blind_m", "tm_sidewalk_m",
        "tm_ban_road") if k in p}

    summaries = []
    for level in levels:
        trans_bias_range = (1.0 - level, 1.0 + level)
        yaw_drift_range  = (-level * YAW_DRIFT_RATE_SCALE, level * YAW_DRIFT_RATE_SCALE)
        trans_noise_std  = level * TRANS_NOISE_SCALE
        yaw_noise_std    = level * YAW_NOISE_SCALE

        rng = np.random.default_rng(p["seed"] * 1000 + int(level * 1000))
        n_success = 0
        progress_fracs = []
        wall_hit_counts = []

        print(f"Sweep level {level:.2f}  "
              f"(trans_bias={trans_bias_range}, yaw_drift={yaw_drift_range} rad/s, "
              f"trans_noise={trans_noise_std:.4f}m, yaw_noise={yaw_noise_std:.4f}rad)")

        for trial in range(n_trials):
            bridge_theta = float(rng.uniform(0.0, 2 * math.pi)) if p["random_theta"] else 0.0
            bridge_len   = float(rng.uniform(p["bridge_len_min"], p["bridge_len_max"]))
            bridge_gap   = float(rng.uniform(p["bridge_gap_min"], p["bridge_gap_max"]))
            wall_thick   = float(rng.uniform(p["wall_thick_min"], p["wall_thick_max"]))
            rot_deg      = float(rng.uniform(p["rot_deg_min"], p["rot_deg_max"]))
            scale        = float(rng.uniform(p["scale_min"],   p["scale_max"]))
            trans_max = p.get("trans_max", 0.0)
            tx = float(rng.uniform(-trans_max, trans_max)) if trans_max > 0 else 0.0
            ty = float(rng.uniform(-trans_max, trans_max)) if trans_max > 0 else 0.0

            # Pick one endpoint of the range, not a continuous draw across it:
            # sampling uniformly across (1-level, 1+level) means most trials land
            # near the mild center of the range by chance, diluting "level" into
            # "somewhere between none and this much drift" instead of a
            # consistent severity.
            odom_trans_bias     = float(rng.choice(trans_bias_range))
            odom_yaw_drift_rate = float(rng.choice(yaw_drift_range))
            drift_params = dict(
                odom_trans_bias=odom_trans_bias,
                odom_yaw_drift_rate=odom_yaw_drift_rate,
                odom_trans_noise_std=trans_noise_std,
                odom_yaw_noise_std=yaw_noise_std,
            )

            geom = make_run_geom(bridge_theta, bridge_len, bridge_gap, wall_thick)
            bridges_deg = [(cx, cy, l, t, math.degrees(th)) for cx, cy, l, t, th in geom.wall_obbs]
            assoc = BehaviorAssociator(bridges=bridges_deg, buildings=[], obstacles=[])
            rid   = assoc.region_name_to_id
            true_cents = np.array(assoc.all_region_centroids_jax_array)
            rot_cents  = _distort_pts(true_cents.copy(), rot_deg, scale, geom.rotation_ctr, tx, ty)

            obs_r = p["obs_radius"]
            min_gap = obs_r + 2 * p["safety_radius"] + 0.06
            n_obs_possible = p["n_obs_max"] if bridge_gap >= min_gap else 0
            n_obs = int(rng.integers(0, min(n_obs_possible + 1, 4)))
            obs_list = []
            for _ in range(n_obs):
                along  = float(rng.uniform(-bridge_len / 2 + 0.15, bridge_len / 2 - 0.15))
                side   = float(rng.choice([-1.0, 1.0]))
                across = side * (bridge_gap * 0.30)
                world_pt = (np.array([geom.cx, geom.cy])
                            + along  * geom.axis_dir
                            + across * geom.perp_dir)
                obs_list.append((float(world_pt[0]), float(world_pt[1]), obs_r))

            np.random.seed(p["seed"] * 10000 + int(level * 1000) * 100 + trial)

            result = run_ours_dead_reckoning(
                assoc, rid, geom, true_cents, rot_cents.copy(),
                obs_circles=obs_list, hard_physics=p["hard_physics"],
                debug=False, weights=weights_dict, drift_params=drift_params)

            n_success += int(result["reached_exit"])
            progress_fracs.append(result["final_progress_frac"])
            wall_hit_counts.append(result["wall_hit_count"])
            print(f"    trial {trial+1}/{n_trials}: "
                  f"reached_exit={result['reached_exit']}  "
                  f"progress={result['final_progress_frac']:.2f}")

        summary = dict(
            level=level,
            odom_trans_bias_range=trans_bias_range,
            odom_yaw_drift_rate_range=yaw_drift_range,
            odom_trans_noise_std=trans_noise_std,
            odom_yaw_noise_std=yaw_noise_std,
            n_trials=n_trials,
            n_success=n_success,
            success_rate=n_success / n_trials,
            mean_final_progress_frac=float(np.mean(progress_fracs)),
            mean_wall_hit_count=float(np.mean(wall_hit_counts)),
        )
        summaries.append(summary)
        print(f"  → level {level:.2f}: success_rate={summary['success_rate']:.0%}  "
              f"mean_progress={summary['mean_final_progress_frac']:.2f}  "
              f"mean_wall_hits={summary['mean_wall_hit_count']:.1f}\n")

    # Console table
    print("\n" + "=" * 78)
    print(f"{'level':>6} {'success_rate':>13} {'mean_progress':>14} {'mean_wall_hits':>15}")
    for s in summaries:
        print(f"{s['level']:>6.2f} {s['success_rate']:>12.0%} "
              f"{s['mean_final_progress_frac']:>14.2f} {s['mean_wall_hit_count']:>15.1f}")
    print("=" * 78)

    # CSV
    csv_path = os.path.join(results_dir, "drift_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["level", "odom_trans_bias_min", "odom_trans_bias_max",
                      "odom_yaw_drift_rate_min", "odom_yaw_drift_rate_max",
                      "odom_trans_noise_std", "odom_yaw_noise_std",
                      "n_trials", "n_success", "success_rate",
                      "mean_final_progress_frac", "mean_wall_hit_count"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summaries:
            writer.writerow(dict(
                level=s["level"],
                odom_trans_bias_min=s["odom_trans_bias_range"][0],
                odom_trans_bias_max=s["odom_trans_bias_range"][1],
                odom_yaw_drift_rate_min=s["odom_yaw_drift_rate_range"][0],
                odom_yaw_drift_rate_max=s["odom_yaw_drift_rate_range"][1],
                odom_trans_noise_std=s["odom_trans_noise_std"],
                odom_yaw_noise_std=s["odom_yaw_noise_std"],
                n_trials=s["n_trials"],
                n_success=s["n_success"],
                success_rate=s["success_rate"],
                mean_final_progress_frac=s["mean_final_progress_frac"],
                mean_wall_hit_count=s["mean_wall_hit_count"],
            ))
    print(f"CSV saved → {csv_path}")

    # Plot: success rate + mean progress vs. drift level, twin y-axes
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    xs = [s["level"] * 100 for s in summaries]
    ax1.plot(xs, [s["success_rate"] for s in summaries],
             "o-", color="limegreen", label="Success rate")
    ax1.set_xlabel("Drift severity level (%)")
    ax1.set_ylabel("Success rate", color="limegreen")
    ax1.set_ylim(-0.05, 1.05)
    ax1.tick_params(axis="y", labelcolor="limegreen")
    ax2 = ax1.twinx()
    ax2.plot(xs, [s["mean_final_progress_frac"] for s in summaries],
             "s--", color="dodgerblue", label="Mean final progress")
    ax2.set_ylabel("Mean final progress fraction", color="dodgerblue")
    ax2.set_ylim(-0.05, 1.05)
    ax2.tick_params(axis="y", labelcolor="dodgerblue")
    fig.suptitle("Dead-reckoning drift tolerance (Ours-topological)", weight="bold")
    fig.tight_layout()
    png_path = os.path.join(results_dir, "drift_sweep.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved → {png_path}")

    return summaries


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Carson NMPC vs SamplingMPC under map distortion.")
    parser.add_argument("--n",     type=int,   default=5,
                        help="Number of comparison runs (default: 5)")
    parser.add_argument("--seed",  type=int,   default=0,
                        help="Master RNG seed (default: 0)")
    parser.add_argument("--debug", action="store_true",
                        help="Print per-step diagnostic info for Ours")
    parser.add_argument("--sweep-drift", action="store_true",
                        help="Run the odometry-drift tolerance sweep instead of "
                             "the per-run PNG comparison")
    parser.add_argument("--sweep-trials", type=int, default=20,
                        help="Trials per drift level in sweep mode (default: 20)")
    parser.add_argument("--sweep-levels", type=str, default=None,
                        help="Comma-separated drift levels, e.g. '0,0.02,0.05,0.1,0.2' "
                             "(default: DRIFT_SWEEP_LEVELS)")
    parser.add_argument("--drift-paths", action="store_true",
                        help="Plot trajectories vs. drift level on one fixed bridge "
                             "(qualitative companion to --sweep-drift) instead of "
                             "the per-run PNG comparison")
    parser.add_argument("--paths-per-level", type=int, default=3,
                        help="Trajectories to draw per drift level in --drift-paths mode")
    args = parser.parse_args()

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)

    params = dict(DEFAULT_PARAMS)
    params["n_runs"] = args.n
    params["seed"]   = args.seed
    params["debug"]  = args.debug

    if args.sweep_drift:
        levels = ([float(x) for x in args.sweep_levels.split(",")]
                  if args.sweep_levels else None)
        run_drift_sweep(params, levels=levels, n_trials=args.sweep_trials,
                        results_dir=results_dir)
        return

    if args.drift_paths:
        levels = ([float(x) for x in args.sweep_levels.split(",")]
                  if args.sweep_levels else None)
        fig = run_drift_path_comparison(params, levels=levels,
                                        n_trials_per_level=args.paths_per_level)
        out = os.path.join(results_dir, "drift_paths.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved → {out}")
        return

    nmpc = CarsonNMPC() if CASADI_OK else None

    def progress(i, n):
        print(f"  Run {i}/{n} complete.")

    figures = run_all(params, progress_cb=progress, nmpc_instance=nmpc)
    for run_idx, fig in enumerate(figures):
        out = os.path.join(results_dir, f"comparison_{run_idx:03d}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {out}")

    print(f"\nDone. {args.n} figures saved to {results_dir}/")


if __name__ == "__main__":
    main()



def _explainer_world(seed=0, *, theta=None, length=0.75, gap=0.34, thick=0.06):
    """A fixed, deliberately EASY bridge world for the explainer.

    The comparison figure randomises geometry because it is measuring robustness. This one
    is teaching a rule, so the world is pinned: a wide gap and no obstacles, leaving the
    terrain patch as the ONLY thing that can move the chosen arc. If the arc still bends
    with `tm_terrain` off, the figure is showing an obstacle or bearing effect mislabelled
    as terrain -- which is why `terrain_explainer_figure` prints the chosen arc's cost.
    """
    rng = np.random.default_rng(seed)
    th = float(rng.uniform(0, 2 * math.pi)) if theta is None else float(theta)
    geom = make_run_geom(th, length, gap, thick)
    bridges_deg = [(cx, cy, l, t, math.degrees(a)) for cx, cy, l, t, a in geom.wall_obbs]
    assoc = BehaviorAssociator(bridges=bridges_deg, buildings=[], obstacles=[])
    true_cents = np.array(assoc.all_region_centroids_jax_array)
    return geom, assoc.region_name_to_id, assoc, true_cents, true_cents.copy(), []

def terrain_explainer_figure(seed=0, weights=None, ax=None, rot_deg=30.0):
    """One panel: how the arc MPC picks, and what the terrain term changes.

    Deliberately NOT the 7-arm comparison figure. That one answers "which controller wins";
    this one answers "what is the controller doing", which is the question a fan of 65
    identical grey arcs was failing to answer.

    EXPLAINER, NOT EVIDENCE -- the caption says so on the figure itself, because a figure
    travels away from the sentence that qualified it.
    """
    import matplotlib.pyplot as _plt
    from matplotlib.patches import Circle as _Circle
    # w_terrain=2.0, NOT a larger value: on this world a much larger weight (e.g. 6.0)
    # avoids the patch so hard that the robot never crosses the bridge, while 2.0 detours
    # and still completes. An explainer whose default setting
    # quietly fails the task teaches the wrong lesson; raise it in the app to see that
    # failure deliberately.
    w = dict(tm_terrain=1, tm_w_terrain=4.0, tm_soft_cost=1.0, tm_sidewalk_m=0.05)
    w.update(weights or {})
    geom, rid, assoc, true_cents, dist_cents, obs = _explainer_world(seed)
    # WHY THE MAP IS DISTORTED HERE. With an accurate map this figure shows nothing: the
    # corridor is straight, the sidewalk runs parallel to every candidate arc, and bearing
    # already points down clean road -- so terrain is silent and the chosen arc IS the
    # bearing arc. That is expected (the ban binds OFF-AXIS, at turn time), not a broken
    # term, but it makes for a figure with no lesson in it.
    # Distorting the map puts the bearing off-axis, which is the regime where terrain has
    # something to say. Around 15-30 deg terrain overrides bearing and the robot still
    # crosses; at larger rotations it fails to cross. 30 sits in the useful band.
    if rot_deg:
        dist_cents = _distort_pts(true_cents.copy(), rot_deg, 1.0, geom.rotation_ctr,
                                  0.0, 0.0)
    res = run_terrain_mpc(assoc, rid, geom, true_cents, dist_cents,
                          obs_circles=obs, weights=w, scale=0.05)
    if ax is None:
        _fig, ax = _plt.subplots(figsize=(8.2, 8.2))
    _draw_walls(ax, geom.wall_obbs)
    # The sidewalk strip: each wall grown by `width`, drawn UNDER the wall so only the
    # border shows. This is the terrain the controller is avoiding.
    sw = res.get("sidewalk_width", 0.0)
    if sw > 0:
        for (wcx, wcy, wlen, wthick, wth) in geom.wall_obbs:
            c_, s_ = math.cos(wth), math.sin(wth)
            hw, hh = wlen / 2 + sw, wthick / 2 + sw
            corners = (np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
                       @ np.array([[c_, -s_], [s_, c_]]).T + np.array([wcx, wcy]))
            ax.add_patch(_plt.Polygon(corners, closed=True, facecolor="mediumpurple",
                                      alpha=0.55, edgecolor="rebeccapurple", lw=1.2,
                                      zorder=5))
    _draw_arc_fan(ax, res, "dimgray")
    ax.plot(res["traj"][:, 0], res["traj"][:, 1], color="black", lw=1.3,
            alpha=0.35, linestyle="-", zorder=7, label="driven path (whole run)")
    ax.scatter(*geom.robot_start[:2], marker="o", s=110, facecolor="white",
               edgecolor="black", zorder=12, label="start")
    n_forbid = int(np.sum(res["last_arc_forbidden"])) if res.get(
        "last_arc_forbidden") is not None else 0
    soft = res.get("last_arc_soft")
    chosen_cost = (float(soft[res["last_best_k"]])
                   if soft is not None and res.get("last_best_k") is not None else float("nan"))
    if soft is not None and float(np.ptp(soft)) > 1e-9:
        import matplotlib.cm as _cm
        from matplotlib.colors import Normalize as _Norm
        sm = _cm.ScalarMappable(_Norm(0.0, float(np.max(soft))), _cm.get_cmap("YlOrRd"))
        cb = ax.figure.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label("terrain cost of the candidate arc", fontsize=9)
    from matplotlib.lines import Line2D as _L2D
    ax.legend(handles=[
        _L2D([0], [0], color="dimgray", lw=2.6, label="arc CHOSEN"),
        _L2D([0], [0], color="deepskyblue", lw=2.0, ls="--",
             label="arc BEARING ALONE would pick"),
        _L2D([0], [0], color="black", lw=1.3, alpha=0.35, label="driven path"),
        _L2D([0], [0], color="none", marker="s", markersize=10,
             markerfacecolor="mediumpurple", markeredgecolor="rebeccapurple",
             label=f"sidewalk ({sw*100:.0f} cm border)"),
        _L2D([0], [0], color="none", marker="o", markersize=9,
             markerfacecolor="white", markeredgecolor="black", label="start"),
    ], fontsize=9, loc="upper left", framealpha=0.93)
    ax.set_aspect("equal"); ax.grid(alpha=0.15)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    crossed = "crossed ✓" if res["reached_exit"] else "FAILED to cross ✗"
    overrode = ("terrain OVERRODE bearing" if res.get("terrain_overrode_bearing")
                else "terrain agreed with bearing")
    ax.set_title("Wrong map points at the building; terrain refuses",
                 fontsize=12.5, weight="bold", pad=38)
    # A second set_title(loc="left") would render ON TOP of the centred one -- same row.
    ax.text(0.5, 1.012,
            f"map error +{rot_deg:.0f}°  ·  w_terrain={w['tm_w_terrain']}"
            f"  ·  chosen arc cost {chosen_cost:.3f}\n{overrode}  ·  {crossed}",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9.5, color="0.3")
    ax.figure.text(0.5, 0.012,
                   "EXPLAINER, NOT EVIDENCE — synthetic terrain in a 2D rig. The real "
                   "terrain source is a camera + segmentation verified in CARLA; this rig "
                   "has mispredicted CARLA before.",
                   ha="center", fontsize=8.5, style="italic", color="0.35", wrap=True)
    return ax.figure, res
