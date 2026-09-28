"""Tests for the supervised state classifier (both backends)."""

import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bev_pipeline.classifier import StateClassifier   # noqa: E402


def _blobs(n_per=40, k=3, d=8, seed=0):
    rng = np.random.RandomState(seed)
    centers = rng.randn(k, d) * 5
    X = np.vstack([centers[c] + rng.randn(n_per, d) for c in range(k)]).astype(np.float32)
    y = np.repeat(np.arange(k), n_per).astype(np.int64)
    return X, y


LABELS3 = ["open_road", "approach_building", "along_building"]


@pytest.mark.parametrize("backend", ["rf", "mlp"])
def test_fit_predict_roundtrip(backend):
    X, y = _blobs()
    clf = StateClassifier(backend=backend, mlp_epochs=60).fit(X, y, labels=LABELS3)
    pred = clf.predict(X)
    assert pred.shape == (len(X),)
    # separable blobs -> high train accuracy
    assert (pred == y).mean() > 0.9


@pytest.mark.parametrize("backend", ["rf", "mlp"])
def test_predictions_within_label_set(backend):
    # The closed-set guarantee: no prediction ever falls outside trained labels.
    X, y = _blobs()
    clf = StateClassifier(backend=backend, mlp_epochs=40).fit(X, y, labels=LABELS3)
    Xnew = np.random.RandomState(9).randn(50, X.shape[1]).astype(np.float32) * 20
    pred = clf.predict(Xnew)
    assert pred.min() >= 0 and pred.max() < len(LABELS3)


def test_proba_shape_and_sums_to_one():
    X, y = _blobs()
    clf = StateClassifier(backend="rf").fit(X, y, labels=LABELS3)
    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), len(LABELS3))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


@pytest.mark.parametrize("backend", ["rf", "mlp"])
def test_save_load_roundtrip(backend):
    X, y = _blobs()
    clf = StateClassifier(backend=backend, mlp_epochs=40).fit(X, y, labels=LABELS3)
    p1 = clf.predict(X)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "clf.pkl")
        clf.save(path)
        clf2 = StateClassifier.load(path)
    p2 = clf2.predict(X)
    assert np.array_equal(p1, p2)
    assert clf2.labels == LABELS3
