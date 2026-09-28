"""Runtime configuration for the BEV inference node (YAML + dataclasses).

Replaces the legacy node's hardcoded ``training_data_name = "livox1"`` and
inline constants with a validated config file, so switching environments,
topics, or the active extractor is a config change — not a code edit.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import yaml


class ConfigError(ValueError):
    """Raised for malformed / missing runtime configuration."""


@dataclass
class RuntimeConfig:
    environment: str
    lidar_topic: str
    odom_topic: Optional[str] = None
    image_topic: Optional[str] = None
    imu_topic: Optional[str] = None
    gps_topic: Optional[str] = None
    extractor: str = "camera_dinov2"
    classifier_path: str = ""
    classifier_meta_path: str = ""
    extractor_weights: Optional[str] = None   # trained CNN/point backbone (if trainable)
    # geometry / raster params
    submap_window: int = 5
    sensor_height: float = 0.6
    bev_extent: float = 32.0
    bev_size: int = 128
    smoothing_window: int = 10
    predicted_cluster_topic: str = "/predicted_cluster"
    predicted_state_topic: str = "/predicted_state"
    override_topic: str = "/cluster_override"
    device: str = "cpu"

    _VALID_EXTRACTORS = ("vol3d_cnn", "bev_cnn", "bev_dinov2", "camera_dinov2",
                         "geom_features", "point_encoder", "bev_dino_mh",
                         "bev_dino_mean", "bev_dinov3", "camera_dinov3")

    def validate(self) -> "RuntimeConfig":
        if not self.environment or not str(self.environment).strip():
            raise ConfigError("config: 'environment' must be a non-empty string")
        if not self.lidar_topic or not str(self.lidar_topic).strip():
            raise ConfigError("config: 'lidar_topic' is required")
        if self.extractor not in self._VALID_EXTRACTORS:
            raise ConfigError(
                f"config: 'extractor' must be one of {self._VALID_EXTRACTORS}, "
                f"got {self.extractor!r}")
        if self.extractor == "camera_dinov2" and not self.image_topic:
            raise ConfigError(
                "config: extractor 'camera_dinov2' requires 'image_topic'")
        if self.submap_window < 1:
            raise ConfigError("config: 'submap_window' must be >= 1")
        if self.bev_size < 8:
            raise ConfigError("config: 'bev_size' must be >= 8")
        return self

    @classmethod
    def from_dict(cls, d: dict) -> "RuntimeConfig":
        if not isinstance(d, dict):
            raise ConfigError("config must be a mapping at the top level")
        known = cls.__dataclass_fields__.keys()
        unknown = set(d) - set(known)
        if unknown:
            raise ConfigError(f"config: unknown keys {sorted(unknown)}")
        try:
            obj = cls(**d)
        except TypeError as e:
            raise ConfigError(f"config: {e}") from e
        return obj.validate()

    @classmethod
    def from_yaml(cls, path: str) -> "RuntimeConfig":
        if not os.path.exists(path):
            raise ConfigError(f"config file not found: {path}")
        try:
            raw = yaml.safe_load(open(path).read()) or {}
        except yaml.YAMLError as e:
            raise ConfigError(f"could not parse config YAML {path}: {e}") from e
        # allow an optional top-level 'bev_pipeline:' namespace
        if "bev_pipeline" in raw and isinstance(raw["bev_pipeline"], dict):
            raw = raw["bev_pipeline"]
        return cls.from_dict(raw)
