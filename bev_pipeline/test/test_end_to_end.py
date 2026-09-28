"""End-to-end pipeline test (TDD spec item).

Exercises the full offline path on a CACHED dataset (drive-free): modality ->
feature extractor -> classifier -> predictions, checking the mandated criteria:
  * 0% unclassified (closed-set guarantee),
  * < 50 ms/frame inference latency,
  * accuracy meaningfully above the majority-class baseline.

Uses the geom_features extractor (frozen, fast, no model download) on the
geometry-grounded labels. Skips gracefully if no cached dataset is present
(set BEV_PIPELINE_TEST_ENV to point at a datasets/<env> dir; defaults to
datasets/full_campus_1hz).
"""

import os
import sys
import time

import numpy as np
import pytest

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)

ENV_DIR = os.environ.get("BEV_PIPELINE_TEST_ENV",
                         os.path.join(PKG, "datasets", "full_campus_1hz"))


def _has_dataset():
    return (os.path.exists(os.path.join(ENV_DIR, "modality_geom.npy"))
            and os.path.exists(os.path.join(ENV_DIR, "labels_geom.npy")))


@pytest.mark.skipif(not _has_dataset(),
                    reason=f"no cached dataset at {ENV_DIR}")
def test_end_to_end_geom():
    from bev_pipeline.feature_extractors import make_extractor
    from bev_pipeline.classifier import StateClassifier
    from sklearn.model_selection import train_test_split

    X = np.load(os.path.join(ENV_DIR, "modality_geom.npy"))
    y = np.load(os.path.join(ENV_DIR, "labels_geom.npy"))
    n_labels = int(y.max()) + 1
    names = [str(i) for i in range(n_labels)]

    ex = make_extractor("geom_features")
    feats = ex.extract_batch(X)
    tr, te = train_test_split(np.arange(len(y)), test_size=0.25, random_state=0,
                              stratify=y if np.min(np.bincount(y)) >= 2 else None)
    clf = StateClassifier(backend="rf").fit(feats[tr], y[tr], labels=names)

    # latency: time single-frame extract + predict
    t0 = time.time()
    for i in te[:50]:
        f = ex.extract_batch(X[i:i + 1])
        _ = clf.predict(f)
    per_frame_ms = (time.time() - t0) / max(1, len(te[:50])) * 1000

    pred = clf.predict(feats[te])
    acc = float((pred == y[te]).mean())
    # majority baseline
    from collections import Counter
    maj = Counter(y[tr].tolist()).most_common(1)[0][0]
    base = float((y[te] == maj).mean())

    # 0% unclassified: every prediction is a valid trained label id
    assert pred.min() >= 0 and pred.max() < n_labels
    # latency well under 50 ms/frame
    assert per_frame_ms < 50.0, f"latency {per_frame_ms:.1f} ms/frame too high"
    # accuracy above baseline (geom_features is strong on geom labels)
    assert acc > base, f"accuracy {acc:.3f} not above baseline {base:.3f}"
