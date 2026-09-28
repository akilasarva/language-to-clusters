"""Tests for the four feature-extractor candidates.

CNN candidates: assert a fit()->extract_batch() cycle produces fixed-size
features and a forward/backward pass runs. DINOv2 candidates: assert params are
frozen and output shape is (N, 384). DINOv2 tests are skipped if the weights /
network are unavailable so CI without them still passes.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bev_pipeline.feature_extractors import (make_extractor, input_kind_of,   # noqa: E402
                                             requires_training_of, CANDIDATES, _VARIANTS)


def _bev_batch(n=12):
    rng = np.random.RandomState(0)
    return rng.randn(n, 4, 128, 128).astype(np.float32)


def _voxel_batch(n=12):
    rng = np.random.RandomState(1)
    return rng.rand(n, 2, 64, 64, 8).astype(np.float32)


def _labels(n=12, k=3):
    return (np.arange(n) % k).astype(np.int64)


def test_registry_and_input_kinds():
    assert set(CANDIDATES) == {"vol3d_cnn", "bev_cnn", "bev_dinov2",
                               "camera_dinov2", "geom_features"}
    assert input_kind_of("vol3d_cnn") == "voxel"
    assert input_kind_of("bev_cnn") == "bev"
    assert input_kind_of("camera_dinov2") == "camera"
    assert input_kind_of("geom_features") == "geom"


def test_variants_route_correctly():
    # variants map to a base modality + frozen/trainable of their base
    assert input_kind_of("bev_dino_mh") == "bev"
    assert input_kind_of("camera_dinov3") == "camera"
    assert requires_training_of("bev_dino_mh") is False
    assert requires_training_of("bev_cnn") is True
    # a variant instance carries its own name
    ex = make_extractor("bev_dino_mh")
    assert ex.name == "bev_dino_mh" and ex.projection == "height_bands"


def test_bev_cnn_infers_channels():
    # multiband (7-channel) BEV: bev_cnn should infer in_ch at fit
    X = np.random.RandomState(0).rand(10, 7, 128, 128).astype(np.float32)
    y = _labels(10)
    ex = make_extractor("bev_cnn", epochs=2)
    ex.fit(X, y, n_classes=3)
    assert ex.in_ch == 7
    assert ex.extract_batch(X).shape == (10, ex.dim)


def test_point_encoder_fit_extract():
    rng = np.random.RandomState(0)
    X = (rng.randn(12, 512, 4) * 10).astype(np.float32)   # 512 pts/frame
    y = _labels(12)
    ex = make_extractor("point_encoder", epochs=2)
    assert ex.requires_training and input_kind_of("point_encoder") == "points"
    ex.fit(X, y, n_classes=3)
    feats = ex.extract_batch(X)
    assert feats.shape == (12, ex.dim) and np.all(np.isfinite(feats))


def test_point_encoder_archs():
    rng = np.random.RandomState(1)
    X = (rng.randn(10, 512, 4) * 10).astype(np.float32)
    y = _labels(10)
    for name in ("point_msp", "point_ssg"):
        ex = make_extractor(name, epochs=2)
        assert input_kind_of(name) == "points" and requires_training_of(name)
        ex.fit(X, y, n_classes=3)
        f = ex.extract_batch(X)
        assert f.shape == (10, ex.dim) and np.all(np.isfinite(f))


def test_point_multitask_aux():
    rng = np.random.RandomState(2)
    X = (rng.randn(10, 512, 4) * 10).astype(np.float32)
    y = _labels(10)
    aux = rng.rand(10, 56).astype(np.float32)
    ex = make_extractor("point_mt", epochs=2)
    assert ex.aux_weight > 0
    ex.fit(X, y, n_classes=3, aux_features=aux)          # multi-task path
    assert ex.extract_batch(X).shape == (10, ex.dim)


def test_cnn_save_load_roundtrip(tmp_path):
    X = np.random.RandomState(1).rand(8, 4, 128, 128).astype(np.float32)
    y = _labels(8)
    ex = make_extractor("bev_cnn", epochs=2)
    ex.fit(X, y, n_classes=3)
    f1 = ex.extract_batch(X)
    p = str(tmp_path / "bev_cnn.pt")
    ex.save(p)
    ex2 = make_extractor("bev_cnn")
    ex2.load(p)
    assert np.allclose(f1, ex2.extract_batch(X), atol=1e-5)
    assert ex2.in_ch == 4


def test_bev_cnn_fit_extract():
    X, y = _bev_batch(), _labels()
    ex = make_extractor("bev_cnn", epochs=2)
    assert ex.requires_training
    ex.fit(X, y, n_classes=3)
    feats = ex.extract_batch(X)
    assert feats.shape == (len(X), ex.dim)
    assert np.all(np.isfinite(feats))


def test_vol3d_cnn_fit_extract():
    X, y = _voxel_batch(), _labels()
    ex = make_extractor("vol3d_cnn", epochs=2)
    assert ex.requires_training
    ex.fit(X, y, n_classes=3)
    feats = ex.extract_batch(X)
    assert feats.shape == (len(X), ex.dim)
    assert np.all(np.isfinite(feats))


def _dino_available():
    # Weights cached locally?
    cache = os.path.expanduser("~/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth")
    return os.path.exists(cache) or os.getenv("DINOV2_LOCAL_REPO")


@pytest.mark.skipif(not _dino_available(), reason="DINOv2 weights not available")
def test_camera_dinov2_frozen_and_shape():
    from bev_pipeline.feature_extractors import _dino
    ex = make_extractor("camera_dinov2")
    assert not ex.requires_training
    imgs = [np.random.RandomState(i).randint(0, 255, (360, 640, 3), dtype=np.uint8)
            for i in range(4)]
    feats = ex.extract_batch(imgs)
    assert feats.shape == (4, 384)
    assert _dino.all_frozen(ex._lazy())          # strictly frozen


@pytest.mark.skipif(not _dino_available(), reason="DINOv2 weights not available")
def test_bev_dinov2_frozen_and_shape():
    from bev_pipeline.feature_extractors import _dino
    ex = make_extractor("bev_dinov2")
    assert not ex.requires_training
    feats = ex.extract_batch(_bev_batch(4))
    assert feats.shape == (4, 384)
    assert _dino.all_frozen(ex._lazy())
