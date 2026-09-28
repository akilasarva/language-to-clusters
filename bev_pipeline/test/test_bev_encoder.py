"""Tests for the BEV rasterizer: shape, finiteness, tilt-invariance."""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bev_pipeline.bev_rasterizer import BevParams, rasterize   # noqa: E402
from bev_pipeline import frame_geometry as fg                   # noqa: E402


def _structured_scene(rng):
    """A scene with walls + scattered structure, gravity-leveled (z up)."""
    # two walls (vertical structure) + ground-ish scatter
    wall1 = np.column_stack([
        np.full(500, 8.0), rng.uniform(-10, 10, 500),
        rng.uniform(-0.5, 3.0, 500), rng.uniform(0, 100, 500)])
    wall2 = np.column_stack([
        rng.uniform(-10, 10, 500), np.full(500, -6.0),
        rng.uniform(-0.5, 2.0, 500), rng.uniform(0, 100, 500)])
    scatter = np.column_stack([
        rng.uniform(-20, 20, 1000), rng.uniform(-20, 20, 1000),
        rng.uniform(-1.0, 0.2, 1000), rng.uniform(0, 100, 1000)])
    return np.vstack([wall1, wall2, scatter]).astype(np.float32)


def test_shape_and_dtype():
    rng = np.random.RandomState(0)
    bev = rasterize(_structured_scene(rng))
    assert bev.shape == (4, 128, 128)
    assert bev.dtype == np.float32


def test_empty_and_finite():
    bev = rasterize(np.empty((0, 4), dtype=np.float32))
    assert bev.shape == (4, 128, 128)
    assert np.all(bev == 0)
    # NaN/Inf inputs are filtered, output stays finite
    pts = np.array([[1.0, 1.0, np.nan, 5.0],
                    [np.inf, 0.0, 1.0, 5.0],
                    [2.0, 2.0, 1.5, 10.0]], dtype=np.float32)
    bev = rasterize(pts)
    assert np.all(np.isfinite(bev))


def test_custom_params_shape():
    rng = np.random.RandomState(1)
    bev = rasterize(_structured_scene(rng), BevParams(extent=20.0, size=64))
    assert bev.shape == (4, 64, 64)


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("skimage") is None,
    reason="scikit-image not installed",
)
def test_tilt_invariance_via_gravity_align():
    from skimage.metrics import structural_similarity as ssim

    rng = np.random.RandomState(2)
    scene = _structured_scene(rng)                 # already level (sensor frame)

    bev_flat = rasterize(scene)

    # Simulate a 5-degree pitch of the SENSOR: real sensor points would be
    # p_B = R^T p_W, i.e. scene @ R. Then gravity_align should recover level.
    pitch = math.radians(5.0)
    q = _pitch_quat(pitch)
    R = fg.quat_to_matrix(q)
    tilted = np.column_stack([scene[:, :3] @ R, scene[:, 3]]).astype(np.float32)
    releveled = fg.gravity_align(tilted, q)
    bev_relevel = rasterize(releveled)

    # Compare the max-height channel (most tilt-sensitive).
    a, b = bev_flat[0], bev_relevel[0]
    rng_val = float(max(a.max(), b.max()) - min(a.min(), b.min())) or 1.0
    score = ssim(a, b, data_range=rng_val)
    assert score > 0.9, f"tilt-invariance SSIM too low: {score:.3f}"


def _pitch_quat(pitch):
    return np.array([0.0, math.sin(pitch / 2), 0.0, math.cos(pitch / 2)])
