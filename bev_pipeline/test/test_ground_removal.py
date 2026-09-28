"""Tests for ground removal (Patchwork++ with RANSAC fallback).

The Patchwork++ path is skipped if the binding is unavailable; the RANSAC
fallback is always exercised.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bev_pipeline.ground_removal import GroundRemover, GroundRemovalParams  # noqa: E402


GROUND_Z = -1.0            # ground plane 1 m below the sensor (realistic lidar)
SENSOR_H = 1.0


def _scene():
    """A dense flat ground plane at z=GROUND_Z plus a raised wall/blob.

    Ground sits below the sensor (as a real lidar sees it), which is what
    Patchwork++'s concentric-zone model expects.
    """
    rng = np.random.RandomState(0)
    gx, gy = np.meshgrid(np.linspace(-15, 15, 80), np.linspace(-15, 15, 80))
    ground = np.column_stack([
        gx.ravel(), gy.ravel(),
        GROUND_Z + rng.normal(0, 0.01, gx.size),   # ~flat at GROUND_Z
        rng.uniform(0, 20, gx.size),
    ]).astype(np.float32)
    # a wall/blob rising well above the ground at x~6
    n = 800
    blob = np.column_stack([
        rng.uniform(5.5, 6.5, n),
        rng.uniform(-2, 2, n),
        rng.uniform(GROUND_Z + 0.8, GROUND_Z + 4.0, n),
        rng.uniform(0, 20, n),
    ]).astype(np.float32)
    return ground, blob


def test_empty_input():
    gr = GroundRemover(GroundRemovalParams(sensor_height=0.0), prefer_patchwork=False)
    ng, g = gr.remove_ground(np.empty((0, 4), dtype=np.float32))
    assert ng.shape[0] == 0 and g.shape[0] == 0


def test_ransac_fallback_separates_blob_from_ground():
    ground, blob = _scene()
    pts = np.vstack([ground, blob])
    gr = GroundRemover(GroundRemovalParams(sensor_height=SENSOR_H, ransac_dist_thresh=0.1),
                       prefer_patchwork=False)
    nonground, gnd = gr.remove_ground(pts)
    # Most blob points (well above ground) survive as nonground.
    assert nonground.shape[0] > 0
    high_frac = np.mean(nonground[:, 2] > GROUND_Z + 0.4)
    assert high_frac > 0.8, f"nonground not dominated by blob: {high_frac:.2f}"
    # ground plane captured most of the low points
    assert gnd.shape[0] > 0.5 * ground.shape[0]
    # columns preserved (x,y,z,intensity)
    assert nonground.shape[1] == 4


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("pypatchworkpp") is None,
    reason="pypatchworkpp not installed",
)
def test_patchworkpp_separates_blob_from_ground():
    ground, blob = _scene()
    pts = np.vstack([ground, blob])
    gr = GroundRemover(GroundRemovalParams(sensor_height=SENSOR_H), prefer_patchwork=True)
    assert gr.backend == "patchworkpp"
    nonground, gnd = gr.remove_ground(pts)
    assert nonground.shape[0] > 0 and gnd.shape[0] > 0
    # Blob points should be overwhelmingly in nonground.
    high_frac = np.mean(nonground[:, 2] > GROUND_Z + 0.4)
    assert high_frac > 0.7, f"patchwork nonground not blob-dominated: {high_frac:.2f}"
