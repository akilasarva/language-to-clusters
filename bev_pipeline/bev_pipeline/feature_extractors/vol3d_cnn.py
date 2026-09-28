"""vol3d_cnn — small Conv3d CNN trained from scratch on the coarse voxel grid.

The legacy ``LidarEncoder`` philosophy (conv encoder + global pool + linear
head) taken genuinely 3-D: Conv3d over the ``(C, GX, GY, GZ)`` voxel grid, so
vertical structure is learned directly rather than flattened into BEV channels.
This is the heaviest candidate; the voxel grid is kept coarse (few z bins) and
the net small to stay tractable on CPU with limited data. Trained end-to-end as
a classifier; penultimate embedding exposed as features.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .base import FeatureExtractor, train_classifier_head, backbone_features


def _make_backbone(in_ch: int, feat_dim: int):
    import torch.nn as nn

    class Vol3dBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv3d(in_ch, 8, 3, stride=(2, 2, 1), padding=1),
                nn.BatchNorm3d(8), nn.ReLU(),
                nn.Conv3d(8, 16, 3, stride=(2, 2, 1), padding=1),
                nn.BatchNorm3d(16), nn.ReLU(),
                nn.Conv3d(16, feat_dim, 3, stride=(2, 2, 2), padding=1),
                nn.BatchNorm3d(feat_dim), nn.ReLU(),
                nn.AdaptiveAvgPool3d((1, 1, 1)),
            )

        def forward(self, x):
            return self.net(x).flatten(1)

    return Vol3dBackbone()


class Vol3dCNNExtractor(FeatureExtractor):
    name = "vol3d_cnn"
    input_kind = "voxel"
    requires_training = True

    def __init__(self, in_ch: int = 2, feat_dim: int = 32, device: str = "cpu",
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
        self._backbone = _make_backbone(self.in_ch, self.feat_dim)
        Xarr = np.asarray(X, dtype=np.float32)
        train_classifier_head(self._backbone, self.feat_dim, n_classes, Xarr, y,
                              epochs=self.epochs, device=self.device)

    def extract_batch(self, X: Sequence) -> np.ndarray:
        if self._backbone is None:
            raise RuntimeError("vol3d_cnn must be fit() before extract_batch()")
        return backbone_features(self._backbone, np.asarray(X, dtype=np.float32),
                                 device=self.device)
