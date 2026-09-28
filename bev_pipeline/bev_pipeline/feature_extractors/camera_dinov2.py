"""camera_dino — frozen DINO (v2 or v3) on the nearest camera frame.

DINO's in-domain use: real RGB images. Zero training. Configurable backbone
(model_name), local weights (weights_path, for gated DINOv3), and pooling
(CLS token or mean of patch tokens).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .base import FeatureExtractor
from . import _dino


def _prep_image(rgb: np.ndarray, size: int = 224) -> np.ndarray:
    from PIL import Image
    if rgb is None:
        return np.zeros((3, size, size), dtype=np.float32)
    img = Image.fromarray(np.ascontiguousarray(rgb.astype(np.uint8))).convert("RGB")
    img = img.resize((size, size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return _dino.normalize_imagenet(arr)


class CameraDinoV2Extractor(FeatureExtractor):
    name = "camera_dinov2"
    input_kind = "camera"
    requires_training = False

    def __init__(self, model_name: str = "dinov2_vits14", device: str = "cpu",
                 size: int = 224, pooling: str = "cls",
                 weights_path: Optional[str] = None, **_kw):
        self.model_name = model_name
        self.device = device
        self.size = size
        self.pooling = pooling
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
        batch = np.stack([_prep_image(img, self.size) for img in X], axis=0)
        return _dino.dino_embed(model, batch, device=self.device, pooling=self.pooling)
