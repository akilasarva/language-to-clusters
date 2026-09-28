"""bev_cnn — small Conv2d CNN trained from scratch on the 4-channel BEV raster.

Extends the legacy ``LidarEncoder``'s conv-encoder style (Conv2d + global pool +
linear) from a 1-D range scan to the full 2-D BEV tensor. Trained end-to-end as
a classifier on the TRAIN split; the penultimate embedding is exposed as the
feature vector so it plugs into the same downstream classifier as the frozen
candidates.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .base import FeatureExtractor, train_classifier_head, backbone_features


def _make_backbone(in_ch: int, feat_dim: int):
    import torch.nn as nn

    class BevBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, feat_dim, 3, stride=2, padding=1), nn.BatchNorm2d(feat_dim),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )

        def forward(self, x):
            return self.net(x).flatten(1)

    return BevBackbone()


class BevCNNExtractor(FeatureExtractor):
    name = "bev_cnn"
    input_kind = "bev"
    requires_training = True

    def __init__(self, in_ch: int = 4, feat_dim: int = 64, device: str = "cpu",
                 epochs: int = 25):
        self.in_ch = in_ch
        self.feat_dim = feat_dim
        self.device = device
        self.epochs = epochs
        self._backbone = None

    @property
    def dim(self) -> int:
        return self.feat_dim

    def _new_backbone(self):
        return _make_backbone(self.in_ch, self.feat_dim)

    def fit(self, X: Sequence, y: Sequence[int], n_classes: int) -> None:
        Xarr = np.asarray(X, dtype=np.float32)
        self.in_ch = Xarr.shape[1]                 # infer channel count (4 or 4+bands)
        self._backbone = _make_backbone(self.in_ch, self.feat_dim)
        train_classifier_head(self._backbone, self.feat_dim, n_classes, Xarr, y,
                              epochs=self.epochs, device=self.device)

    def extract_batch(self, X: Sequence) -> np.ndarray:
        if self._backbone is None:
            raise RuntimeError("bev_cnn must be fit() before extract_batch()")
        return backbone_features(self._backbone, np.asarray(X, dtype=np.float32),
                                 device=self.device)
