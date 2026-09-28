"""Common interface for the four feature-extractor candidates.

Every candidate maps a per-frame input to a fixed-length feature vector, so the
same downstream classifier can be trained on any of them and compared fairly:

    extractor.fit(X_train, y_train, n_classes)   # trains CNNs; no-op if frozen
    feats = extractor.extract_batch(X)            # (N, dim) float32

``input_kind`` tells the dataset which modality to feed:
  "bev"    -> (4,H,W) BEV raster        (bev_cnn, bev_dinov2)
  "voxel"  -> (C,GX,GY,GZ) voxel grid   (vol3d_cnn)
  "camera" -> (H,W,3) uint8 RGB image   (camera_dinov2)

Frozen extractors (DINOv2) set ``requires_training = False`` and ignore fit().
The CNNs train from scratch end-to-end as classifiers on the TRAIN split only,
then expose their penultimate embedding as features (a learned representation,
kept apples-to-apples with the frozen ones by feeding all of them to the same
downstream classifier).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np


class FeatureExtractor:
    """Abstract base. Subclasses set name/input_kind/requires_training."""

    name: str = "base"
    input_kind: str = "bev"
    requires_training: bool = False

    def fit(self, X: Sequence, y: Sequence[int], n_classes: int) -> None:
        """Train the extractor (no-op for frozen candidates)."""
        return None

    def extract_batch(self, X: Sequence) -> np.ndarray:
        raise NotImplementedError

    @property
    def dim(self) -> int:
        raise NotImplementedError

    # -- weight persistence (trained CNN/point backbones) ------------------ #
    def _new_backbone(self):
        """Rebuild an untrained backbone matching current config (trainable only)."""
        raise NotImplementedError

    def save(self, path: str) -> None:
        """Persist a trained backbone (+ config) so it can run in the node."""
        import torch
        bb = getattr(self, "_backbone", None)
        if bb is None:
            raise RuntimeError(f"{self.name}: nothing to save (frozen or not fit)")
        torch.save({"state_dict": bb.state_dict(), "in_ch": getattr(self, "in_ch", None),
                    "feat_dim": getattr(self, "feat_dim", None), "name": self.name}, path)

    def load(self, path: str) -> "FeatureExtractor":
        import torch
        blob = torch.load(path, map_location="cpu", weights_only=False)
        self.in_ch = blob["in_ch"]
        self.feat_dim = blob["feat_dim"]
        self._backbone = self._new_backbone()
        self._backbone.load_state_dict(blob["state_dict"])
        self._backbone.eval()
        return self


# --------------------------------------------------------------------------- #
# Shared CNN training utilities (CPU-friendly, small nets, modest epochs)      #
# --------------------------------------------------------------------------- #

def _seed_torch(seed: int = 0):
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_classifier_head(backbone, feat_dim, n_classes, X, y,
                          *, epochs=25, batch=16, lr=1e-3, device="cpu",
                          weight_decay=1e-4, verbose=False,
                          aux_targets=None, aux_weight=0.0):
    """Train ``backbone`` (input->features) + a linear head as a classifier.

    Optional MULTI-TASK auxiliary: if ``aux_targets`` (N, aux_dim) is given with
    ``aux_weight``>0, a second head regresses those targets (e.g. the geometric
    feature vector) and total loss = CE + aux_weight * MSE. This regularizes the
    shared embedding to encode geometry (which transfers across environments).

    ``backbone`` is an nn.Module mapping a batched input tensor to ``(N, feat_dim)``
    features. ``X`` is a stacked float32 array of inputs; ``y`` int labels.
    Returns the trained head (backbone is trained in place). Uses class-balanced
    sampling weights to cope with the imbalanced label distribution.
    """
    import torch
    import torch.nn as nn

    _seed_torch()
    dev = torch.device(device)
    backbone = backbone.to(dev).train()
    head = nn.Linear(feat_dim, n_classes).to(dev)

    Xt = torch.as_tensor(np.asarray(X, dtype=np.float32), device=dev)
    yt = torch.as_tensor(np.asarray(y, dtype=np.int64), device=dev)
    n = len(yt)

    # class-balanced loss weights (guard divide-by-zero for absent classes)
    counts = np.bincount(np.asarray(y), minlength=n_classes).astype(np.float64)
    cw = np.divide(1.0, counts, out=np.zeros_like(counts), where=counts > 0)
    cw = cw / cw.sum() * (counts > 0).sum() if cw.sum() > 0 else cw
    class_w = torch.as_tensor(cw, dtype=torch.float32, device=dev)
    crit = nn.CrossEntropyLoss(weight=class_w)

    params = list(backbone.parameters()) + list(head.parameters())
    aux_head = None
    if aux_targets is not None and aux_weight > 0:
        at = torch.as_tensor(np.asarray(aux_targets, dtype=np.float32), device=dev)
        aux_head = nn.Linear(feat_dim, at.shape[1]).to(dev)
        params += list(aux_head.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            feats = backbone(Xt[idx])
            loss = crit(head(feats), yt[idx])
            if aux_head is not None:
                loss = loss + aux_weight * nn.functional.mse_loss(aux_head(feats), at[idx])
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f"    epoch {ep}: loss={tot/n:.4f}")
    backbone.eval()
    return head


def backbone_features(backbone, X, *, batch=32, device="cpu") -> np.ndarray:
    """Run a trained backbone over ``X`` returning ``(N, feat_dim)`` features."""
    import torch
    dev = torch.device(device)
    backbone = backbone.to(dev).eval()
    Xt = torch.as_tensor(np.asarray(X, dtype=np.float32))
    outs: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(Xt), batch):
            f = backbone(Xt[i:i + batch].to(dev))
            outs.append(f.cpu().numpy())
    return np.concatenate(outs, axis=0).astype(np.float32) if outs \
        else np.zeros((0, 0), dtype=np.float32)
