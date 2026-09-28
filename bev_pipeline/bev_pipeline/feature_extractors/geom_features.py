"""geom_features — interpretable hand-engineered geometric features.

Small-data-friendly, no training: the fixed-length geometric vector from
``geometry.geometric_features`` (polar free-space profile + clearances +
overhead + height histogram). Complements the frozen DINO features and, unlike
the from-scratch CNNs, works well with limited data. The dataset builder
precomputes these into ``modality_geom.npy`` so this extractor is an identity
pass-through over the precomputed vectors.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .base import FeatureExtractor


class GeomFeaturesExtractor(FeatureExtractor):
    name = "geom_features"
    input_kind = "geom"
    requires_training = False

    def __init__(self, **_kwargs):
        self._dim = None

    @property
    def dim(self) -> int:
        return self._dim if self._dim is not None else 0

    def extract_batch(self, X: Sequence) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float32)
        self._dim = arr.shape[1] if arr.ndim == 2 else 0
        return arr
