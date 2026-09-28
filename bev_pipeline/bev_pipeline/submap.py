"""Sliding-window submap accumulation in the robocentric leveled frame.

Accumulating the last few gravity-leveled frames into the *current* frame does
two things:

  * densifies sparse outdoor structure (fences, poles, thin railings) that a
    single sweep barely captures, and
  * encodes approach-vs-recede motion: because reprojection uses *relative*
    odometry across the window, nearby structure visibly converges or recedes
    — a purely local signal, never grounded to absolute/GPS coordinates.

Frames are reprojected using the **leveled** frame G (yaw-only rotation +
odometry translation), matching :func:`frame_geometry.gravity_align`, which
already removed roll/pitch. If a frame has no pose, only the current sweep is
returned (no reprojection possible).
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np

from .frame_geometry import invert_transform, quat_to_euler, transform_points


def persistence_filter(chunks, voxel: float = 0.5, min_frames: int = 2) -> np.ndarray:
    """Keep only points in voxels occupied across >= min_frames of the window.

    ``chunks`` is a list of per-frame point arrays already reprojected into the
    common (latest) frame. Static structure (walls, poles) occupies the same
    voxel every frame -> high persistence; moving objects (pedestrians) smear
    across voxels -> low persistence and get dropped. This leverages the submap
    volume to separate dynamic obstacles from persistent structure.
    """
    chunks = [c for c in chunks if c is not None and len(c)]
    if len(chunks) <= 1:
        return np.ascontiguousarray(chunks[0]) if chunks else np.empty((0, 4), np.float32)
    # per-voxel set of frame indices that occupy it
    from collections import defaultdict
    occ = defaultdict(set)
    keyed = []
    for fi, c in enumerate(chunks):
        vk = np.floor(c[:, :3] / voxel).astype(np.int64)
        keys = [tuple(k) for k in vk]
        keyed.append(keys)
        for k in keys:
            occ[k].add(fi)
    out = []
    for fi, (c, keys) in enumerate(zip(chunks, keyed)):
        mask = np.array([len(occ[k]) >= min_frames for k in keys], dtype=bool)
        if mask.any():
            out.append(c[mask])
    return np.ascontiguousarray(np.vstack(out)) if out else np.empty((0, chunks[0].shape[1]), np.float32)


def _leveled_frame_matrix(position: np.ndarray, yaw: float) -> np.ndarray:
    """World<-G transform: yaw-only rotation about Z + odometry translation."""
    c, s = math.cos(yaw), math.sin(yaw)
    m = np.eye(4)
    m[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    m[:3, 3] = np.asarray(position, dtype=np.float64)
    return m


class SubmapAccumulator:
    """Fixed-length deque of leveled frames, reprojected to the latest one."""

    def __init__(self, window_size: int = 5, filter_dynamic: bool = False,
                 dyn_voxel: float = 0.5, dyn_min_frames: int = 2):
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        self.window_size = window_size
        self.filter_dynamic = filter_dynamic
        self.dyn_voxel = dyn_voxel
        self.dyn_min_frames = dyn_min_frames
        self._buf: Deque[Tuple[np.ndarray, Optional[np.ndarray], Optional[float]]] = \
            deque(maxlen=window_size)

    def reset(self) -> None:
        self._buf.clear()

    def __len__(self) -> int:
        return len(self._buf)

    def push(self, points: np.ndarray, position=None, quat=None) -> np.ndarray:
        """Add a leveled frame and return the accumulated submap in its frame.

        ``position``/``quat`` are the odometry pose for this frame (``quat`` is
        xyzw). Either may be ``None`` (pose unavailable).
        """
        yaw = quat_to_euler(quat)[2] if quat is not None else None
        pos = np.asarray(position, dtype=np.float64) if position is not None else None
        self._buf.append((points, pos, yaw))
        return self.accumulate()

    def accumulate(self) -> np.ndarray:
        """Reproject every buffered frame into the most-recent frame."""
        if not self._buf:
            return np.empty((0, 4), dtype=np.float32)
        latest_pts, latest_pos, latest_yaw = self._buf[-1]

        # If the latest frame lacks a pose, we can't reproject anything else.
        if latest_pos is None or latest_yaw is None:
            return np.ascontiguousarray(latest_pts)

        T_world_L = _leveled_frame_matrix(latest_pos, latest_yaw)
        T_L_world = invert_transform(T_world_L)

        chunks = []
        for pts, pos, yaw in self._buf:
            if pts.shape[0] == 0:
                continue
            if pos is None or yaw is None:
                # unposed older frame: only trust it if it's the latest
                if pts is latest_pts:
                    chunks.append(pts)
                continue
            T_world_i = _leveled_frame_matrix(pos, yaw)
            T_L_i = T_L_world @ T_world_i
            chunks.append(transform_points(pts, T_L_i))
        if not chunks:
            return np.empty((0, latest_pts.shape[1] if latest_pts.ndim == 2 else 4),
                            dtype=np.float32)
        if self.filter_dynamic and len(chunks) > 1:
            return persistence_filter(chunks, self.dyn_voxel, self.dyn_min_frames)
        return np.ascontiguousarray(np.vstack(chunks))
