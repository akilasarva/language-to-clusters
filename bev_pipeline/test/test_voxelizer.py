"""Tests for the coarse 3D voxel grid."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bev_pipeline.voxelizer import VoxelParams, voxelize   # noqa: E402


def test_shape_and_finite():
    rng = np.random.RandomState(0)
    pts = np.column_stack([
        rng.uniform(-30, 30, 2000), rng.uniform(-30, 30, 2000),
        rng.uniform(-2, 5, 2000), rng.uniform(0, 100, 2000)]).astype(np.float32)
    vg = voxelize(pts)
    assert vg.shape == (2, 64, 64, 8)
    assert vg.dtype == np.float32
    assert np.all(np.isfinite(vg))


def test_empty():
    vg = voxelize(np.empty((0, 4), dtype=np.float32))
    assert vg.shape == (2, 64, 64, 8)
    assert np.all(vg == 0)


def test_multiband_bev_shape_and_black_empty():
    # rasterize_multiband: 4 base + N height-band occupancy channels; empty
    # cells are 0 in every band (fixing the flat-red-background artifact).
    from bev_pipeline.bev_rasterizer import rasterize_multiband, BevParams
    rng = np.random.RandomState(0)
    pts = np.column_stack([rng.uniform(-20, 20, 3000), rng.uniform(-20, 20, 3000),
                           rng.uniform(0, 5, 3000), rng.uniform(0, 100, 3000)]).astype(np.float32)
    mb = rasterize_multiband(pts, BevParams(n_height_bands=3))
    assert mb.shape == (7, 128, 128)
    assert np.all(np.isfinite(mb))
    # band channels (4,5,6) are non-negative occupancy with many zero (empty) cells
    assert mb[4:].min() >= 0 and (mb[4:] == 0).mean() > 0.3


def test_nan_inf_filtered():
    pts = np.array([[1.0, 1.0, np.nan, 1.0],
                    [np.inf, 0.0, 1.0, 1.0],
                    [2.0, 2.0, 1.0, 9.0]], dtype=np.float32)
    vg = voxelize(pts)
    assert np.all(np.isfinite(vg))
    assert vg[0].sum() > 0        # the one valid point registered


def test_custom_coarse_shape():
    rng = np.random.RandomState(3)
    pts = np.column_stack([rng.uniform(-10, 10, 500), rng.uniform(-10, 10, 500),
                           rng.uniform(-1, 3, 500), rng.uniform(0, 50, 500)]).astype(np.float32)
    vg = voxelize(pts, VoxelParams(extent=16.0, gx=32, gy=32, gz=4))
    assert vg.shape == (2, 32, 32, 4)


def test_density_and_intensity_channels():
    # Two points in the same voxel -> density log1p(2); mean intensity = avg.
    pts = np.array([[0.1, 0.1, 0.1, 10.0],
                    [0.11, 0.11, 0.11, 20.0]], dtype=np.float32)
    vg = voxelize(pts, VoxelParams(extent=32.0, gx=64, gy=64, gz=8))
    occ = vg[0] > 0
    assert occ.sum() == 1
    assert np.isclose(vg[0][occ][0], np.log1p(2), atol=1e-5)
    assert np.isclose(vg[1][occ][0], 15.0, atol=1e-4)
