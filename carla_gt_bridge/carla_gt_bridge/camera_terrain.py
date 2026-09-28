"""Camera terrain source for `terrain_mpc`: rollout points -> pixels -> terrain class.

`TerrainMpcConfig.terrain_class` is a required callable, ``(K, n, 2)`` body-frame points ->
``(K, n)`` int class ids. `waypoint_terrain_source` implements it from CARLA's waypoint
table, which is ground truth and unavailable on hardware. This module implements it from a
segmented camera frame.

`/current_terrain` and `/predicted_cluster` describe the current position, while the scorer
needs labels at K x n hypothetical future positions. A segmented frame carries terrain over
space in pixels, so each rollout point is projected onto the ground plane in the image and
the class at that pixel is read. Nothing here consults the waypoint table, so terrain that
the table does not hold (e.g. grass) can still be scored.

FAILURE MODES THIS MODULE GUARDS AGAINST
----------------------------------------
1. **A narrow camera cannot see where most of the fan goes.** The near cut is constant
   across K (for the cue camera, the bottom image row meets the ground 3.5 m ahead of the
   ego, so the first samples of every arc are below the frame). The lateral limit is not
   constant: a point is in frame only while roughly `|y| < x - 1.5`, so the hardest-turning
   arcs leave sideways and may never be seen at all. That is why `TERRAIN_CAMERA` below is
   wider and tilted rather than a copy of the cue camera.
2. **Every mask is stale.** The camera is throttled (`sensor_tick` 0.5 s) because an
   unthrottled camera starves the MPC loop. At 5 m/s a frame can be 2.5 m behind the pose
   the rollouts are drawn from, so the mask is stored WITH the pose it was taken from and
   rollout points are carried back into that pose before projecting.
3. **A dead camera must not read as clean road, and must not veto everything either.** No
   mask, a stale mask, an occluded point or a point out of frame all return `UNOBSERVED`,
   which is refused entry to any forbid set (`terrain_classes.check_forbid`) and is DROPPED
   from the graded cost rather than priced -- unknown is not the same fact as bad. An arc with
   no observed sample at all takes `unknown_cost`; see `terrain_mpc.plan_step_terrain` for
   why that keeps a dead camera harmless (the charge is identical for all K, so the
   controller degrades to bearing-only instead of vetoing itself into a corner).
   `coverage_frac` is reported per tick so "saw nothing" is distinguishable from "saw clean
   road" in the log.

FLAT GROUND IS AN ASSUMPTION
----------------------------
Every point is projected onto z = 0 in the body frame. Town05 is near enough flat for that.
The model makes two checkable predictions: with pitch 0 the horizon sits at row ``cy``
exactly, and the LiDAR's ground returns must project onto pixels the mask calls ground.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .terrain_classes import UNOBSERVED

__all__ = ["CameraModel", "CUE_CAMERA", "TERRAIN_CAMERA", "MaskTerrainSource",
           "body_to_body"]


# --------------------------------------------------------------------------- #
# The camera                                                                   #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CameraModel:
    """Pinhole camera on the vehicle, and the ground plane it sees.

    Defaults are the deployment's own camera, so a caller that passes nothing gets the one
    that is actually spawned rather than a plausible-looking stand-in.

    No intrinsics are read from the sensor: nothing subscribes to
    `/carla/ego_vehicle/rgb_front/camera_info`, and the camera is configured only by fov and
    resolution, so `fx` is DERIVED from them here. The horizon check (`horizon_row`) is the
    simplest test of that derivation against a real frame.
    """

    width: int = 800
    height: int = 600
    fov_deg: float = 90.0
    #: Mount in the ROS BODY frame: x forward, y left, z up, origin at the vehicle.
    mount_x: float = 1.5
    mount_y: float = 0.0
    mount_z: float = 1.5
    #: Downward tilt in degrees. POSITIVE LOOKS DOWN, which is the opposite sign convention
    #: to CARLA's `spawn_point.pitch`; named `pitch_down` so the two cannot be confused by
    #: reading. The cue camera is level (0.0).
    pitch_down_deg: float = 0.0
    #: Points nearer than this along the optical axis are treated as behind the camera. Not
    #: zero: the projection divides by it.
    z_near: float = 0.3
    #: Occlusion floor, in metres from the ego origin: ground nearer than this is hidden by
    #: the vehicle's own bodywork. Zero means "not set", not "none".
    #:
    #: This is not geometry and cannot be derived from the lens. `min_visible_range_m` says
    #: where the bottom image row meets the ground, but a mid-bonnet camera sees the ego's
    #: hood in front of that (for the default mount the hood edge back-projects to ~3.1 m).
    #: Without this floor, fan samples landing on the car's paintwork are tagged
    #: Car -> OTHER -> full cost, a near-constant offset on every arc that no projection or
    #: label check reveals.
    occluded_below_m: float = 0.0

    @property
    def fx(self) -> float:
        return self.width / (2.0 * math.tan(math.radians(self.fov_deg) / 2.0))

    @property
    def fy(self) -> float:
        # Square pixels. CARLA's camera has one `fov`, which is HORIZONTAL, so the vertical
        # focal length is the same number and the vertical FOV follows from the aspect ratio.
        return self.fx

    @property
    def cx(self) -> float:
        return self.width / 2.0

    @property
    def cy(self) -> float:
        return self.height / 2.0

    def horizon_row(self) -> float:
        """Image row the ground plane approaches at infinite range.

        With `pitch_down_deg == 0` this is exactly `cy` -- 300 for this camera. It is the
        cheapest falsifiable prediction the model makes, and one frame tests it.

        The sign is MINUS: tilting the camera DOWN moves the scene UP the image, so the
        horizon rises toward row 0. A level-camera test cannot tell the signs apart
        (`tan(0) == 0`), so `test_the_horizon_matches_the_projection_when_the_camera_is_tilted`
        pins this closed form against `project_ground` itself.
        """
        return self.cy - self.fy * math.tan(math.radians(self.pitch_down_deg))

    def effective_min_range_m(self) -> float:
        """The nearer of what the lens can reach and what the bodywork allows.

        This is the number `terrain_mpc.arc_weights` should be given as `s_min`: weighting
        toward ground that the hood hides is the same mistake as weighting toward ground
        below the frame, one cause further along.
        """
        return max(self.min_visible_range_m(), self.occluded_below_m)

    def min_visible_range_m(self) -> float:
        """Ground range, from the EGO origin, of the bottom image row.

        Nearer than this the ground is below the frame and no pixel describes it. For the
        deployment camera: 2.0 m from the lens, 3.5 m from the ego. `terrain_mpc.arc_weights`
        takes this so the distance weighting starts where the evidence starts.

        ``inf`` when the camera is level enough that the bottom row never meets the ground --
        which cannot happen for a downward-looking mount but can for a raised horizon.
        """
        # Depression angle of the bottom row below the optical axis, plus the mount tilt.
        depress = math.atan2(self.height - self.cy, self.fy) + math.radians(self.pitch_down_deg)
        if depress <= 1e-6:
            return float("inf")
        return self.mount_x + self.mount_z / math.tan(depress)

    def project_ground(self, pts_body_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Body-frame ground points -> pixels. ``(..., 2)`` -> ``(u, v, visible)``.

        Points are assumed to lie on z = 0 in the body frame (see the module docstring).
        ``visible`` is False for anything behind the camera or outside the image, and ``u``
        and ``v`` are meaningless there -- they are clipped rather than left as NaN so a
        caller that forgets the mask gets a wrong pixel rather than an exception, and
        `MaskTerrainSource` always applies it.
        """
        p = np.asarray(pts_body_xy, dtype=float)
        return self.project_3d(np.stack([p[..., 0], p[..., 1],
                                         np.zeros(p.shape[:-1])], axis=-1))

    def project_3d(self, pts_body_xyz: np.ndarray,
                   pitch_sign: float = 1.0, y_sign: float = 1.0
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Body-frame 3-D points -> pixels. ``(..., 3)`` -> ``(u, v, visible)``.

        `project_ground` is this with z = 0, so there is ONE projection in this file rather
        than a ground copy and a general copy that can drift apart.

        ``pitch_sign`` and ``y_sign`` exist for mutation testing: flipping either must make a
        LiDAR-to-mask agreement check COLLAPSE. A validation that still passes with the
        y-axis mirrored is not testing the axis convention, and the y-sign (planar vs CARLA
        frame) is an easy error to make silently.
        """
        p = np.asarray(pts_body_xyz, dtype=float)
        dx = p[..., 0] - self.mount_x
        dy = (p[..., 1] * float(y_sign)) - self.mount_y
        dz = p[..., 2] - self.mount_z

        # Rotation about the body +y (left) axis. phi > 0 tilts the optical axis DOWN.
        phi = math.radians(self.pitch_down_deg) * float(pitch_sign)
        cp, sp = math.cos(phi), math.sin(phi)
        z_c = dx * cp - dz * sp                 # forward
        x_c = -dy                               # right
        y_c = -dx * sp - dz * cp                # down

        ahead = z_c > self.z_near
        safe = np.where(ahead, z_c, 1.0)        # avoid a divide-by-zero warning behind us
        u = self.cx + self.fx * x_c / safe
        v = self.cy + self.fy * y_c / safe

        visible = (ahead & (u >= 0.0) & (u < self.width)
                   & (v >= 0.0) & (v < self.height))
        if self.occluded_below_m > 0.0:
            # Hidden by our own bodywork: UNOBSERVED, not "whatever the hood is labelled".
            visible &= np.hypot(p[..., 0], p[..., 1]) >= self.occluded_below_m
        u = np.clip(u, 0.0, self.width - 1.0)
        v = np.clip(v, 0.0, self.height - 1.0)
        return u, v, visible


#: The camera the VLM cue path already spawns (`run_phase_a.sh`). Present so the SegFormer
#: source can run against the existing sensor, and so its geometry is written down once
#: rather than re-derived.
CUE_CAMERA = CameraModel()

#: The camera the terrain source spawns: at the vehicle's front-most point, wide and tilted
#: down.
#:
#: **Position.** x = 2.425 m is the front-most point of the body, derived from the actor's
#: own bounding box (`extent.x` 2.396 + `location.x` 0.029 for the Tesla model3). Mounting
#: here removes the bodywork occlusion a mid-bonnet camera suffers (see `occluded_below_m`),
#: so no occlusion floor is needed.
#:
#: **Lens.** Moving the mount forward moves the near limit with it -- `min_visible_range_m`
#: is `mount_x + mount_z/tan(depress)` -- so a narrower/shallower lens (e.g. fov 110 /
#: pitch 15) leaves the hardest-turning arcs fully blind, and a blind arc lets a prohibition
#: be satisfied by turning where the camera never looked (the drop rule in
#: `terrain_mpc.terrain_soft_costs` depends on this). fov 130 / pitch 30 leaves no blind arcs
#: in the default fan and no bodywork in view; a wider lens (140) starts catching the front
#: fenders again.
#:
#: **A real lens is not a pinhole at 130 degrees.** CARLA's is, so nothing here models
#: distortion; on hardware that has to be calibrated and `CameraModel` extended.
#:
#: `rgb_front` is not changed by this: the VLM cue path depends on its fov 90, 800x600
#: geometry.
TERRAIN_CAMERA = CameraModel(fov_deg=130.0, pitch_down_deg=30.0, mount_x=2.425,
                             mount_z=1.573)


# --------------------------------------------------------------------------- #
# Carrying rollout points back to the pose the frame was taken from            #
# --------------------------------------------------------------------------- #

def body_to_body(pts_xy: np.ndarray, pose_from: tuple[float, float, float],
                 pose_to: tuple[float, float, float]) -> np.ndarray:
    """Re-express body-frame points from one vehicle pose in another's body frame.

    Two applications of the `R_l2g` idiom used in `terrain_mpc.py` and
    `carla_mpc_ros_node.py`: out to the planar world, then back in.

    Poses are ``(x, y, yaw)`` in the PLANAR ROS frame -- x east, y north, yaw CCW from +x.
    Raw CARLA yaw is clockwise and must be converted first; `carla_gt_bridge.frames` owns that
    and the bridge has already applied it to the odometry this node reads.
    """
    x0, y0, th0 = pose_from
    x1, y1, th1 = pose_to
    c0, s0 = math.cos(th0), math.sin(th0)
    c1, s1 = math.cos(th1), math.sin(th1)
    R0 = np.array([[c0, -s0], [s0, c0]])        # body(from) -> world
    R1 = np.array([[c1, -s1], [s1, c1]])        # body(to)   -> world
    flat = np.asarray(pts_xy, dtype=float).reshape(-1, 2)
    world = flat @ R0.T + np.array([x0, y0])
    # world -> body(to). R1 is orthonormal, so the inverse is the transpose, and for ROW
    # vectors `v @ R1` IS `R1.T @ v`. Writing it as `@ R1` rather than `@ R1.T.T` is not a
    # simplification to undo later -- getting this backwards mirrors the trajectory about the
    # heading and is silent.
    out = (world - np.array([x1, y1])) @ R1
    return out.reshape(np.asarray(pts_xy).shape)


# --------------------------------------------------------------------------- #
# The source                                                                   #
# --------------------------------------------------------------------------- #

class MaskTerrainSource:
    """Holds the latest class mask and hands out `terrain_class` callables.

    One instance lives on the node; `update` is called from the image callback and
    `source_for` from the control loop. They run at different rates on purpose -- the camera
    is throttled to 2 Hz and the control loop runs per odometry message.
    """

    def __init__(self, cam: CameraModel | None = None, *,
                 max_frame_age_s: float = 1.0) -> None:
        self.cam = cam if cam is not None else CameraModel()
        self.max_frame_age_s = float(max_frame_age_s)
        self._mask: np.ndarray | None = None
        self._pose: tuple[float, float, float] | None = None
        self._stamp: float = -math.inf
        #: Fraction of the last batch of queried points that a pixel actually described.
        #: 0.0 until the first successful classification. NOT initialised to 1.0: an
        #: un-run controller must not report full coverage.
        self.coverage_frac: float = 0.0
        #: Age in seconds of the frame used by the last `source_for` call, for the log.
        self.last_age_s: float = float("inf")
        #: Why the last call saw nothing, or "" when it saw something. Surfaced so a log line
        #: can say WHICH silent failure happened rather than only that coverage was zero.
        self.last_reason: str = "no frame yet"

    # -- input ------------------------------------------------------------- #

    def update(self, class_mask: np.ndarray, pose_xyyaw: tuple[float, float, float],
               stamp: float) -> None:
        """Store a mask together with the pose it was taken from.

        The pose is the caller's business: the node snapshots its latest odometry at image
        RECEIPT, which in synchronous mode is about one tick after capture. That residual is
        logged (`last_age_s`) rather than assumed away.
        """
        m = np.asarray(class_mask)
        if m.ndim != 2:
            raise ValueError(f"class mask must be (H, W), got {m.shape}")
        if m.shape != (self.cam.height, self.cam.width):
            raise ValueError(
                f"class mask {m.shape} does not match the camera model "
                f"{(self.cam.height, self.cam.width)} -- projecting into the wrong image "
                f"size samples the wrong pixel everywhere and reads as a segmentation error")
        self._mask = m
        self._pose = (float(pose_xyyaw[0]), float(pose_xyyaw[1]), float(pose_xyyaw[2]))
        self._stamp = float(stamp)

    # -- output ------------------------------------------------------------ #

    def source_for(self, pose_now: tuple[float, float, float],
                   now: float) -> Callable[[np.ndarray], np.ndarray]:
        """Return the ``(K, n, 2) -> (K, n)`` callable `TerrainMpcConfig` requires.

        Bound to ONE tick: it closes over the current pose, so a cached instance would
        classify rollouts against where the vehicle used to be. `waypoint_terrain_source` is
        rebuilt per tick for the same reason (in `carla_mpc_ros_node.py`).

        A missing or stale frame yields all-`UNOBSERVED` rather than the last good mask.
        Reusing a 3-second-old frame at 5 m/s would describe ground 15 m behind the vehicle
        while reporting terrain-scored numbers.
        """
        mask, pose_cap, stamp = self._mask, self._pose, self._stamp
        age = float(now) - stamp
        stale = mask is None or age > self.max_frame_age_s

        def _classify(pts_body: np.ndarray) -> np.ndarray:
            pts = np.asarray(pts_body, dtype=float)
            shape = pts.shape[:-1]
            if stale:
                self.coverage_frac = 0.0
                _classify.coverage_frac = 0.0
                self.last_age_s = age
                self.last_reason = ("no frame yet" if mask is None
                                    else f"frame {age:.2f}s old > {self.max_frame_age_s:.2f}s")
                return np.full(shape, UNOBSERVED, dtype=int)

            at_capture = body_to_body(pts, pose_now, pose_cap)
            u, v, vis = self.cam.project_ground(at_capture)
            out = np.full(shape, UNOBSERVED, dtype=int)
            if vis.any():
                rows = np.rint(v[vis]).astype(int)
                cols = np.rint(u[vis]).astype(int)
                out[vis] = mask[rows, cols]
            self.coverage_frac = float(vis.mean()) if vis.size else 0.0
            self.last_age_s = age
            self.last_reason = "" if vis.any() else "every point outside the frame"
            _classify.coverage_frac = self.coverage_frac
            return out

        # `plan_step_terrain` reads `cfg.terrain_class.coverage_frac` after calling it, so the
        # number has to live on the CALLABLE, not only on this object -- the scorer is handed
        # the closure and never sees the source. NaN until the first call, which is what the
        # waypoint stand-in also reports, so "not asked yet" stays distinct from "saw nothing".
        _classify.coverage_frac = float("nan")
        return _classify
