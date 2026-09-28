"""point_encoder — point-native encoders over the raw accumulated cloud.

Operates directly on points (no lossy BEV/voxel projection). Three architectures
(``arch``), progressively capturing more local structure than a global max-pool:

  * "vanilla" — shared per-point MLP + global MAX pool (original PointNet-lite).
    Fast, but max-pool discards relative structure.
  * "msp"     — same per-point MLP, but MULTI-SCALE pool (max+mean+std) so the
    global descriptor keeps distributional info a single max throws away.
  * "ssg"     — PointNet++-lite single-scale grouping: sample centroids, group
    k nearest neighbors, encode LOCAL relative geometry, pool. Captures "wall on
    my left AND opening ahead" as spatial relationships.

Supports self-supervised pretrained weights (``pretrained_weights``) and a
multi-task auxiliary geometry-regression head at fit time (``aux_features``),
so the LiDAR-native path can leverage unlabeled data + geometry supervision.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .base import FeatureExtractor, train_classifier_head, backbone_features


def _mlp1d(dims):
    import torch.nn as nn
    layers = []
    for i in range(len(dims) - 1):
        layers += [nn.Conv1d(dims[i], dims[i + 1], 1), nn.BatchNorm1d(dims[i + 1]), nn.ReLU()]
    return nn.Sequential(*layers)


def make_point_backbone(arch: str, in_ch: int, feat_dim: int):
    """Build a point-cloud backbone: (B, N, C) -> (B, feat_dim)."""
    import torch
    import torch.nn as nn

    class Vanilla(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = _mlp1d([in_ch, 64, 128, feat_dim])

        def forward(self, x):                       # x: (B, N, C)
            f = self.mlp(x.transpose(1, 2))         # (B, feat, N)
            return f.max(dim=2).values

    class MSP(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = _mlp1d([in_ch, 64, 128, feat_dim])
            self.head = nn.Sequential(nn.Linear(3 * feat_dim, feat_dim), nn.ReLU())

        def forward(self, x):
            f = self.mlp(x.transpose(1, 2))         # (B, feat, N)
            pooled = torch.cat([f.max(2).values, f.mean(2), f.std(2)], dim=1)
            return self.head(pooled)

    class SSG(nn.Module):
        """Single-scale-grouping: local set abstraction + global pool."""
        def __init__(self, n_centroids=256, k=16):
            super().__init__()
            self.n_centroids, self.k = n_centroids, k
            self.local = _mlp1d([in_ch + 3, 64, 128])   # +3 relative xyz
            self.glob = _mlp1d([128, feat_dim])

        def forward(self, x):                        # x: (B, N, C)
            B, N, C = x.shape
            m = min(self.n_centroids, N)
            # random centroid sampling (seed-free; SSL/augmentation handles variety)
            cidx = torch.randint(0, N, (B, m), device=x.device)
            centroids = torch.gather(x, 1, cidx[:, :, None].expand(-1, -1, C))  # (B,m,C)
            d = torch.cdist(centroids[..., :3], x[..., :3])                     # (B,m,N)
            knn = d.topk(min(self.k, N), dim=2, largest=False).indices          # (B,m,k)
            grp = torch.gather(x[:, None].expand(-1, m, -1, -1), 2,
                               knn[..., None].expand(-1, -1, -1, C))            # (B,m,k,C)
            rel = grp[..., :3] - centroids[:, :, None, :3]
            feat_in = torch.cat([rel, grp], dim=-1)                            # (B,m,k,C+3)
            bm = feat_in.reshape(B * m, feat_in.shape[2], C + 3).transpose(1, 2)
            local = self.local(bm).max(dim=2).values.reshape(B, m, 128)        # (B,m,128)
            g = self.glob(local.transpose(1, 2))                               # (B,feat,m)
            return g.max(dim=2).values

    return {"vanilla": Vanilla, "msp": MSP, "ssg": SSG}[arch]()


def _normalize(X: np.ndarray, extent: float = 32.0, zlo: float = -3.0,
               zhi: float = 8.0) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32).copy()
    X[..., 0] = np.clip(X[..., 0] / extent, -1, 1)
    X[..., 1] = np.clip(X[..., 1] / extent, -1, 1)
    X[..., 2] = np.clip((X[..., 2] - zlo) / (zhi - zlo) * 2 - 1, -1, 1)
    if X.shape[-1] > 3:
        X[..., 3] = np.clip(X[..., 3] / 255.0, 0, 1)
    return X


class PointEncoderExtractor(FeatureExtractor):
    name = "point_encoder"
    input_kind = "points"
    requires_training = True

    def __init__(self, in_ch: int = 4, feat_dim: int = 128, device: str = "cpu",
                 epochs: int = 25, arch: str = "vanilla",
                 pretrained_weights: Optional[str] = None, aux_weight: float = 0.0, **_kw):
        self.in_ch = in_ch
        self.feat_dim = feat_dim
        self.device = device
        self.epochs = epochs
        self.arch = arch
        self.pretrained_weights = pretrained_weights
        self.aux_weight = aux_weight
        self._backbone = None

    @property
    def dim(self) -> int:
        return self.feat_dim

    def _new_backbone(self):
        return make_point_backbone(self.arch, self.in_ch, self.feat_dim)

    def _resolve_weights(self):
        """Find the weights file: as given, or relative to the package root."""
        import os
        w = self.pretrained_weights
        if not w:
            return None
        if os.path.exists(w):
            return w
        pkg = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cand = os.path.join(pkg, w)
        return cand if os.path.exists(cand) else None

    def _load_pretrained(self, bb):
        path = self._resolve_weights()
        if self.pretrained_weights and path is None:
            import warnings
            warnings.warn(f"point_encoder: pretrained weights {self.pretrained_weights!r} "
                          "not found; training from scratch.")
        if path:
            import torch
            blob = torch.load(path, map_location=self.device, weights_only=False)
            bb.load_state_dict(blob.get("backbone", blob), strict=False)
        return bb

    def fit(self, X: Sequence, y: Sequence[int], n_classes: int,
            aux_features: Optional[np.ndarray] = None) -> None:
        Xn = _normalize(np.asarray(X, dtype=np.float32))
        self.in_ch = Xn.shape[-1]
        self._backbone = self._load_pretrained(make_point_backbone(
            self.arch, self.in_ch, self.feat_dim))
        aux = (np.asarray(aux_features, dtype=np.float32)
               if (aux_features is not None and self.aux_weight > 0) else None)
        train_classifier_head(self._backbone, self.feat_dim, n_classes, Xn, y,
                              epochs=self.epochs, device=self.device,
                              aux_targets=aux, aux_weight=self.aux_weight)

    def extract_batch(self, X: Sequence) -> np.ndarray:
        if self._backbone is None:
            raise RuntimeError("point_encoder must be fit() before extract_batch()")
        return backbone_features(self._backbone, _normalize(np.asarray(X, dtype=np.float32)),
                                 device=self.device)
