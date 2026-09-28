"""Tests for RuntimeConfig loading/validation."""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bev_pipeline.config import RuntimeConfig, ConfigError   # noqa: E402

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _base(**over):
    d = {"environment": "full_campus", "lidar_topic": "/l", "extractor": "bev_cnn"}
    d.update(over)
    return d


def test_valid_minimal():
    c = RuntimeConfig.from_dict(_base())
    assert c.environment == "full_campus"
    assert c.extractor == "bev_cnn"


def test_missing_environment():
    with pytest.raises(ConfigError):
        RuntimeConfig.from_dict(_base(environment=""))


def test_missing_lidar():
    d = _base()
    del d["lidar_topic"]
    with pytest.raises(ConfigError):
        RuntimeConfig.from_dict(d)


def test_bad_extractor():
    with pytest.raises(ConfigError):
        RuntimeConfig.from_dict(_base(extractor="nope"))


def test_camera_requires_image_topic():
    with pytest.raises(ConfigError):
        RuntimeConfig.from_dict(_base(extractor="camera_dinov2"))
    # ok when image topic provided
    c = RuntimeConfig.from_dict(_base(extractor="camera_dinov2", image_topic="/img"))
    assert c.image_topic == "/img"


def test_unknown_key_rejected():
    with pytest.raises(ConfigError):
        RuntimeConfig.from_dict(_base(bogus=1))


def test_shipped_yaml_loads():
    path = os.path.join(PKG, "config", "bev_pipeline.yaml")
    c = RuntimeConfig.from_yaml(path)
    assert c.environment
    assert c.lidar_topic
    assert c.extractor in RuntimeConfig._VALID_EXTRACTORS


def test_from_yaml_missing_file():
    with pytest.raises(ConfigError):
        RuntimeConfig.from_yaml("/no/such/config.yaml")
