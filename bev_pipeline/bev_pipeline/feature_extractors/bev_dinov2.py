"""bev_dino — frozen DINO (v2 or v3) on a 3-channel BEV projection.

Projects the BEV raster to a 3-channel pseudo-image and runs a frozen DINO
forward pass. Zero training. Two projection modes:
  * "vico3d"        : max-height / log-density / mean-intensity (the original)
  * "height_bands"  : low / mid / high occupancy bands as R/G/B — encodes
                      vertical structure into the 3 channels DINO sees (the
                      "recover 3D through the 2D pipeline" variant)

Configurable backbone (model_name incl. dinov3), local weights (weights_path),
and pooling (cls / mean).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .base import FeatureExtractor
from . import _dino


def _norm01(ch: np.ndarray) -> np.ndarray:
    lo, hi = float(ch.min()), float(ch.max())
    if hi - lo < 1e-9:
        return np.zeros_like(ch)
    return (ch - lo) / (hi - lo)


def _fixed01(ch: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Fixed physical-range normalization -> empty (0) stays black and the same
    structure looks the same across frames (per-image min-max does neither)."""
    return np.clip((ch - lo) / (hi - lo), 0.0, 1.0)


# fixed per-channel display ranges (metres / log-count / intensity)
_FIXED = {"height": (0.0, 6.0), "density": (0.0, 4.0), "intensity": (0.0, 255.0),
          "band": (0.0, 4.0)}


def _bev_to_rgb(bev: np.ndarray, size: int, projection: str,
                norm: str = "fixed") -> np.ndarray:
    from PIL import Image

    def n(ch, key):
        return _fixed01(ch, *_FIXED[key]) if norm == "fixed" else _norm01(ch)

    if projection == "passthrough":
        # ground BEV is already 3 semantic channels in [0,1] (intensity/rough/density)
        r, g, b = (np.clip(bev[0], 0, 1), np.clip(bev[1], 0, 1), np.clip(bev[2], 0, 1))
    elif projection == "height_bands" and bev.shape[0] >= 7:
        # low/mid/high occupancy bands (rasterize_multiband); empty=0=black
        r, g, b = n(bev[4], "band"), n(bev[5], "band"), n(bev[6], "band")
    else:
        # vico3d: height / log-density / intensity (fixed-range => empty black)
        r, g, b = n(bev[0], "height"), n(bev[2], "density"), n(bev[3], "intensity")
    rgb = np.stack([r, g, b], axis=-1)
    img = Image.fromarray((rgb * 255).astype(np.uint8)).resize((size, size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return _dino.normalize_imagenet(arr)


class BevDinoV2Extractor(FeatureExtractor):
    name = "bev_dinov2"
    input_kind = "bev"
    requires_training = False

    def __init__(self, model_name: str = "dinov2_vits14", device: str = "cpu",
                 size: int = 224, pooling: str = "cls", projection: str = "vico3d",
                 norm: str = "fixed", weights_path: Optional[str] = None, **_kw):
        self.model_name = model_name
        self.device = device
        self.size = size                            # 224 = patch-aligned
        self.pooling = pooling
        self.projection = projection
        self.norm = norm                            # "fixed" (empty=black) or "minmax"
        self.weights_path = weights_path
        self._model = None
        self._dim = None

    def _lazy(self):
        if self._model is None:
            self._model = _dino.load_dino(self.model_name, self.device, self.weights_path)
            self._dim = _dino.embed_dim(self._model, self.device, self.pooling)
        return self._model

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._lazy()
        return self._dim

    def extract_batch(self, X: Sequence) -> np.ndarray:
        model = self._lazy()
        batch = np.stack([_bev_to_rgb(b, self.size, self.projection, self.norm)
                          for b in X], axis=0)
        return _dino.dino_embed(model, batch, device=self.device, pooling=self.pooling)
