"""Shared frozen-DINO backbone loading for the DINO candidates (v2 and v3).

DINO stays STRICTLY frozen everywhere (``requires_grad = False``) — no LoRA, no
partial unfreeze. Supports:
  * DINOv2 (default, weights auto-download + cached, ungated), and
  * DINOv3 (stronger, but weights are LICENSE-GATED by Meta — pass a local
    weights file via ``weights_path`` / ``$DINO_WEIGHTS`` after accepting the
    license; the architecture loads from the (ungated) hub repo with
    pretrained=False and we load_state_dict the local file).
  * CLS-token or mean-patch-token pooling.
Embedding dim is auto-detected, so any variant/size works.
"""

from __future__ import annotations

import os

import numpy as np

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_MODEL_CACHE = {}


def _repo_for(model_name: str) -> str:
    return "facebookresearch/dinov3" if model_name.startswith("dinov3") \
        else "facebookresearch/dinov2"


def load_dino(model_name: str = "dinov2_vits14", device: str = "cpu",
              weights_path: str = None):
    """Load a frozen DINO backbone (cached per process).

    ``weights_path`` (or $DINO_WEIGHTS) loads local weights and bypasses any
    gated download — required for DINOv3.
    """
    import torch

    weights_path = weights_path or os.getenv("DINO_WEIGHTS")
    key = (model_name, device, weights_path)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    repo = _repo_for(model_name)
    cached_repo = os.path.join(torch.hub.get_dir(),
                               repo.replace("/", "_") + "_main")
    src = cached_repo if os.path.isdir(cached_repo) else repo
    source = "local" if src == cached_repo else "github"

    if weights_path:
        model = torch.hub.load(src, model_name, source=source,
                               pretrained=False, trust_repo=True)
        state = torch.load(weights_path, map_location=device)
        state = state.get("model", state) if isinstance(state, dict) else state
        model.load_state_dict(state, strict=False)
    else:
        model = torch.hub.load(src, model_name, source=source,
                               pretrained=True, trust_repo=True)

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    _MODEL_CACHE[key] = model
    return model


def embed_dim(model, device: str = "cpu", pooling: str = "cls") -> int:
    """Auto-detect the output feature dim with a dummy forward."""
    import torch
    with torch.no_grad():
        out = _forward(model, torch.zeros(1, 3, 224, 224, device=device), pooling)
    return int(out.shape[1])


def all_frozen(model) -> bool:
    return all(not p.requires_grad for p in model.parameters())


def normalize_imagenet(rgb01: np.ndarray) -> np.ndarray:
    x = (rgb01 - _IMAGENET_MEAN) / _IMAGENET_STD
    return np.transpose(x, (2, 0, 1)).astype(np.float32)


def _forward(model, xb, pooling: str):
    """Return (B, D) features, CLS token or mean of patch tokens."""
    if pooling == "mean" and hasattr(model, "forward_features"):
        feats = model.forward_features(xb)
        if isinstance(feats, dict):
            pt = feats.get("x_norm_patchtokens")
            if pt is not None:
                return pt.mean(dim=1)
            return feats.get("x_norm_clstoken", next(iter(feats.values())))
    out = model(xb)
    if isinstance(out, dict):
        out = out.get("x_norm_clstoken", next(iter(out.values())))
    return out


def dino_embed(model, batch_chw: np.ndarray, device: str = "cpu",
               batch_size: int = 16, pooling: str = "cls") -> np.ndarray:
    """Run frozen DINO over a normalized ``(N,3,H,W)`` batch -> ``(N,D)``."""
    import torch

    outs = []
    with torch.no_grad():
        for i in range(0, len(batch_chw), batch_size):
            xb = torch.as_tensor(batch_chw[i:i + batch_size]).to(device)
            outs.append(_forward(model, xb, pooling).cpu().numpy())
    if not outs:
        return np.zeros((0, 384), dtype=np.float32)
    return np.concatenate(outs, axis=0).astype(np.float32)
