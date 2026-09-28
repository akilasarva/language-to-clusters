"""Tests for sliding-window submap accumulation."""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bev_pipeline.submap import SubmapAccumulator, _leveled_frame_matrix   # noqa: E402


def _yaw_quat(yaw):
    return np.array([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])


def _world_point_in_leveled_frame(world_pt, position, yaw):
    """Express a world point in the leveled frame G(position, yaw)."""
    from bev_pipeline.frame_geometry import invert_transform
    T_world_G = _leveled_frame_matrix(position, yaw)
    T_G_world = invert_transform(T_world_G)
    hp = np.array([world_pt[0], world_pt[1], world_pt[2], 1.0])
    return (T_G_world @ hp)[:3]


def test_window_length_and_reset():
    acc = SubmapAccumulator(window_size=3)
    for i in range(5):
        acc.push(np.zeros((2, 4), np.float32), position=[float(i), 0, 0], quat=_yaw_quat(0))
    assert len(acc) == 3
    acc.reset()
    assert len(acc) == 0


def test_fixed_world_landmark_coincides_after_reprojection():
    # A single landmark fixed in the world. The robot moves/turns; each frame
    # observes it in that frame's leveled coordinates. After accumulation into
    # the latest frame, all reprojected copies must land on the SAME point.
    world_lm = np.array([10.0, 4.0, 1.5])
    poses = [
        (np.array([0.0, 0.0, 0.0]), math.radians(0)),
        (np.array([2.0, 0.5, 0.0]), math.radians(15)),
        (np.array([4.0, 1.0, 0.0]), math.radians(30)),
    ]
    acc = SubmapAccumulator(window_size=3)
    out = None
    for pos, yaw in poses:
        local = _world_point_in_leveled_frame(world_lm, pos, yaw)
        pts = np.array([[local[0], local[1], local[2], 7.0]], dtype=np.float32)
        out = acc.push(pts, position=pos, quat=_yaw_quat(yaw))

    assert out.shape[0] == 3            # three reprojected copies
    # All copies should coincide (same world landmark) within tolerance.
    xyz = out[:, :3]
    spread = xyz.max(axis=0) - xyz.min(axis=0)
    assert np.all(spread < 1e-4), f"reprojected landmark not coincident: {spread}"
    # And they should match the landmark's position in the latest frame.
    expected = _world_point_in_leveled_frame(world_lm, poses[-1][0], poses[-1][1])
    assert np.allclose(xyz[0], expected, atol=1e-4)
    # intensity column preserved
    assert np.allclose(out[:, 3], 7.0)


def test_no_pose_returns_latest_only():
    acc = SubmapAccumulator(window_size=3)
    acc.push(np.ones((5, 4), np.float32), position=None, quat=None)
    out = acc.push(np.ones((3, 4), np.float32) * 2, position=None, quat=None)
    # latest frame had no pose -> only its own points come back
    assert out.shape[0] == 3
    assert np.allclose(out, 2.0)


def test_accumulation_densifies():
    acc = SubmapAccumulator(window_size=4)
    out = None
    for i in range(4):
        out = acc.push(np.random.RandomState(i).randn(10, 4).astype(np.float32),
                       position=[float(i), 0, 0], quat=_yaw_quat(0))
    assert out.shape[0] == 40          # 4 frames x 10 points, densified
