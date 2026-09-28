"""Gravity alignment and relative-pose transform math.

Two jobs:

  * :func:`gravity_align` — cancel the sensor's recorded roll/pitch so the
    ground is level in the working frame, while KEEPING yaw (the pipeline is
    robocentric: +X stays "current vehicle forward"). This is what makes the
    BEV raster tilt-invariant.
  * :func:`relative_transform` / :func:`transform_points` — express one pose's
    points in another pose's frame, using *relative* odometry only (never
    grounded to absolute/GPS coordinates). Used by the submap accumulator.

All quaternions are ``[x, y, z, w]`` (ROS convention). Points may be ``(N,3)``
or ``(N,4)`` (a 4th intensity column is passed through untouched).
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def quat_to_matrix(q_xyzw: np.ndarray) -> np.ndarray:
    """Unit-normalized xyzw quaternion -> 3x3 rotation matrix (world<-body)."""
    x, y, z, w = [float(v) for v in q_xyzw]
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def quat_to_euler(q_xyzw: np.ndarray) -> Tuple[float, float, float]:
    """xyzw quaternion -> (roll, pitch, yaw) in radians (ZYX / aerospace)."""
    x, y, z, w = [float(v) for v in q_xyzw]
    # roll (x-axis)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    # pitch (y-axis)
    sinp = 2 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    # yaw (z-axis)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def _rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def leveling_rotation(q_xyzw: np.ndarray) -> np.ndarray:
    """Rotation taking body-frame points into a gravity-leveled, heading-
    following frame G.

    G shares the body's yaw (heading) but has zero roll/pitch, so applying this
    to a LiDAR cloud levels the ground while keeping +X = current forward
    (robocentric). Derivation: ``R_WG = Rz(yaw)`` and points transform as
    ``p_G = R_WG^T @ R_WB @ p_B = Rz(-yaw) @ R_WB @ p_B``. This uses the full
    rotation matrix (not an order-dependent euler triple), so it is exact for
    any orientation.
    """
    R = quat_to_matrix(q_xyzw)                 # world <- body
    yaw = math.atan2(R[1, 0], R[0, 0])
    return _rot_z(-yaw) @ R


def gravity_align(points: np.ndarray, q_xyzw: np.ndarray) -> np.ndarray:
    """Level a point cloud by cancelling the sensor's roll/pitch.

    ``points`` is ``(N,3)`` or ``(N,4)`` (intensity column preserved). Returns a
    new array of the same shape. A near-identity / None-equivalent orientation
    is a no-op.
    """
    if points.size == 0:
        return points.copy()
    R = leveling_rotation(q_xyzw)
    xyz = points[:, :3]
    leveled = xyz @ R.T
    if points.shape[1] > 3:
        return np.column_stack([leveled, points[:, 3:]]).astype(points.dtype)
    return leveled.astype(points.dtype)


def pose_matrix(position: np.ndarray, q_xyzw: np.ndarray) -> np.ndarray:
    """4x4 homogeneous world<-body transform."""
    m = np.eye(4)
    m[:3, :3] = quat_to_matrix(q_xyzw)
    m[:3, 3] = np.asarray(position, dtype=np.float64)
    return m


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Invert a 4x4 rigid transform."""
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def relative_transform(pos_ref, quat_ref, pos_src, quat_src) -> np.ndarray:
    """Transform mapping points in the *src* body frame into the *ref* body frame.

    ``T_ref_src = inv(T_world_ref) @ T_world_src``. Uses only the two poses'
    relative geometry, never absolute coordinates.
    """
    T_world_ref = pose_matrix(pos_ref, quat_ref)
    T_world_src = pose_matrix(pos_src, quat_src)
    return invert_transform(T_world_ref) @ T_world_src


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to ``(N,3)`` or ``(N,4)`` points (intensity kept)."""
    if points.size == 0:
        return points.copy()
    xyz = points[:, :3]
    out = xyz @ T[:3, :3].T + T[:3, 3]
    if points.shape[1] > 3:
        return np.column_stack([out, points[:, 3:]]).astype(points.dtype)
    return out.astype(points.dtype)
