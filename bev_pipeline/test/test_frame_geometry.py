"""Tests for gravity alignment + relative-transform math."""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bev_pipeline import frame_geometry as fg   # noqa: E402


def _quat_from_euler(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([x, y, z, w])


def test_euler_roundtrip():
    for roll, pitch, yaw in [(0.1, -0.2, 0.3), (0.0, 0.0, 1.0), (-0.05, 0.05, -2.0)]:
        q = _quat_from_euler(roll, pitch, yaw)
        r2, p2, y2 = fg.quat_to_euler(q)
        assert abs(r2 - roll) < 1e-6
        assert abs(p2 - pitch) < 1e-6
        assert abs(y2 - yaw) < 1e-6


def test_gravity_align_flattens_tilted_ground():
    # A flat ground patch at z=0.
    xs, ys = np.meshgrid(np.linspace(-5, 5, 40), np.linspace(-5, 5, 40))
    ground = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)])

    # Tilt it by a known 8-degree pitch (rotate about y). Real sensor-frame
    # points are p_B = R_WB^T @ p_W, i.e. flat world ground appears as
    # ground @ R in row-vector form.
    pitch = math.radians(8.0)
    q = _quat_from_euler(0.0, pitch, 0.0)
    R = fg.quat_to_matrix(q)
    tilted = ground @ R

    # Before alignment the z-spread is large; after, it should collapse.
    spread_before = tilted[:, 2].max() - tilted[:, 2].min()
    leveled = fg.gravity_align(tilted, q)
    spread_after = leveled[:, 2].max() - leveled[:, 2].min()

    assert spread_before > 1.0
    assert spread_after < 1e-6, f"ground not leveled: spread={spread_after}"


def test_leveling_is_identity_for_pure_yaw():
    # With zero roll/pitch, leveling must not touch the cloud regardless of yaw
    # (robocentric: heading is preserved, not cancelled).
    for yaw in (0.0, 0.5, math.radians(90), -2.0):
        q = _quat_from_euler(0.0, 0.0, yaw)
        R_lev = fg.leveling_rotation(q)
        assert np.allclose(R_lev, np.eye(3), atol=1e-9), f"yaw={yaw}"


def test_gravity_align_preserves_intensity_and_shape():
    # Leveling is a rigid rotation on xyz: intensity column and pairwise
    # distances are preserved, output shape matches input.
    rng = np.random.RandomState(1)
    pts = np.column_stack([rng.randn(30, 3), rng.rand(30)])   # xyzi
    q = _quat_from_euler(math.radians(5), math.radians(10), math.radians(90))
    out = fg.gravity_align(pts, q)
    assert out.shape == pts.shape
    assert np.allclose(out[:, 3], pts[:, 3])                  # intensity kept
    # rigid: distance from origin preserved per point
    assert np.allclose(np.linalg.norm(out[:, :3], axis=1),
                       np.linalg.norm(pts[:, :3], axis=1), atol=1e-9)


def test_relative_transform_roundtrip():
    pos_a = np.array([1.0, 2.0, 0.0])
    q_a = _quat_from_euler(0, 0, 0.5)
    pos_b = np.array([4.0, 1.0, 0.0])
    q_b = _quat_from_euler(0, 0, -0.3)

    pts_b = np.random.RandomState(0).randn(50, 3)
    T_ab = fg.relative_transform(pos_a, q_a, pos_b, q_b)
    pts_in_a = fg.transform_points(pts_b, T_ab)
    # Going back should recover original.
    T_ba = fg.relative_transform(pos_b, q_b, pos_a, q_a)
    back = fg.transform_points(pts_in_a, T_ba)
    assert np.allclose(back, pts_b, atol=1e-9)


def test_relative_transform_identity_for_same_pose():
    pos = np.array([3.0, -2.0, 1.0])
    q = _quat_from_euler(0.1, 0.2, 0.3)
    T = fg.relative_transform(pos, q, pos, q)
    assert np.allclose(T, np.eye(4), atol=1e-9)
