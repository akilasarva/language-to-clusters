"""Feature-extractor candidates behind a common interface.

Use :func:`make_extractor` to build one by name. ``input_kind`` on each tells
the dataset which modality to feed ("bev", "voxel", or "camera").
"""

from .base import FeatureExtractor

# Default comparison set. Extra variants (below) are constructible by name but
# kept out of the default set to keep routine runs lean.
CANDIDATES = ("vol3d_cnn", "bev_cnn", "bev_dinov2", "camera_dinov2", "geom_features")

# name -> (module class, preset kwargs). Variants let us A/B encodings/backbones.
_VARIANTS = {
    "bev_dino_mh": ("bev_dinov2", {"projection": "height_bands"}),
    "bev_dino_mean": ("bev_dinov2", {"pooling": "mean"}),
    "bev_dinov3": ("bev_dinov2", {"model_name": "dinov3_vits16"}),      # needs weights
    "camera_dinov3": ("camera_dinov2", {"model_name": "dinov3_vits16"}),  # needs weights
    # LiDAR point-encoder variants (all on modality_points)
    "point_msp": ("point_encoder", {"arch": "msp"}),
    "point_ssg": ("point_encoder", {"arch": "ssg"}),
    "point_ssl": ("point_encoder", {"arch": "ssg",
                                    "pretrained_weights": "models/point_ssl.pt"}),
    "point_mt":  ("point_encoder", {"arch": "ssg", "aux_weight": 0.5}),  # multi-task (needs aux)
    # ground-plane BEV (sidewalk graph) on modality_ground_bev
    "ground_cnn": ("bev_cnn", {}),                            # from-scratch CNN
    "ground_dino": ("bev_dinov2", {"projection": "passthrough"}),  # frozen DINO
}

# variants whose input modality differs from their base extractor's default.
_VARIANT_KIND = {"ground_cnn": "ground_bev", "ground_dino": "ground_bev"}


def make_extractor(name: str, **kwargs) -> FeatureExtractor:
    """Factory: build a candidate (or variant) extractor by name."""
    if name in _VARIANTS:
        base, preset = _VARIANTS[name]
        merged = {**preset, **kwargs}
        ex = make_extractor(base, **merged)
        ex.name = name
        return ex
    if name == "point_encoder":
        from .point_encoder import PointEncoderExtractor
        return PointEncoderExtractor(**kwargs)
    if name == "geom_features":
        from .geom_features import GeomFeaturesExtractor
        return GeomFeaturesExtractor(**kwargs)
    if name == "vol3d_cnn":
        from .vol3d_cnn import Vol3dCNNExtractor
        return Vol3dCNNExtractor(**kwargs)
    if name == "bev_cnn":
        from .bev_cnn import BevCNNExtractor
        return BevCNNExtractor(**kwargs)
    if name == "bev_dinov2":
        from .bev_dinov2 import BevDinoV2Extractor
        return BevDinoV2Extractor(**kwargs)
    if name == "camera_dinov2":
        from .camera_dinov2 import CameraDinoV2Extractor
        return CameraDinoV2Extractor(**kwargs)
    raise ValueError(f"unknown extractor {name!r}; choices: {CANDIDATES} + {tuple(_VARIANTS)}")


def input_kind_of(name: str) -> str:
    """Return the modality a candidate consumes without instantiating models."""
    if name in _VARIANT_KIND:
        return _VARIANT_KIND[name]
    if name in _VARIANTS:
        name = _VARIANTS[name][0]
    return {
        "vol3d_cnn": "voxel",
        "bev_cnn": "bev",
        "bev_dinov2": "bev",
        "camera_dinov2": "camera",
        "geom_features": "geom",
        "point_encoder": "points",
    }[name]


_TRAINABLE = {"vol3d_cnn", "bev_cnn", "point_encoder"}


def requires_training_of(name: str) -> bool:
    """Whether a candidate/variant needs fit() (True) or is frozen (False)."""
    if name in _VARIANTS:
        name = _VARIANTS[name][0]
    return name in _TRAINABLE


__all__ = ["FeatureExtractor", "make_extractor", "input_kind_of",
           "requires_training_of", "CANDIDATES"]
