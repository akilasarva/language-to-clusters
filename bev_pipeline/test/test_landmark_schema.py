"""Tests for the per-bag landmark annotation schema."""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bev_pipeline.landmark_schema import (   # noqa: E402
    load_landmarks, parse_landmarks, save_landmarks, LandmarkSet, Landmark,
    LandmarkSchemaError, SCHEMA_VERSION)


def _valid():
    return {
        "schema_version": SCHEMA_VERSION,
        "bag_name": "full_campus",
        "frame_id": "hamilton/odom",
        "landmarks": [
            {"id": 0, "type": "building", "center": [1.0, 2.0, 0.0],
             "heading_deg": 10.0, "length": 18.0, "width": 12.0},
            {"id": 1, "type": "bridge", "center": [40.0, -3.0, 0.0],
             "heading_deg": 87.5, "length": 22.0, "width": 4.5, "notes": "overpass"},
        ],
    }


def test_valid_parse():
    ls = parse_landmarks(_valid())
    assert ls.bag_name == "full_campus"
    assert len(ls.landmarks) == 2
    assert ls.types() == {"building", "bridge"}


def test_version_mismatch():
    d = _valid(); d["schema_version"] = 999
    with pytest.raises(LandmarkSchemaError):
        parse_landmarks(d)


def test_duplicate_id():
    d = _valid(); d["landmarks"][1]["id"] = 0
    with pytest.raises(LandmarkSchemaError):
        parse_landmarks(d)


def test_unknown_type_rejected_when_types_given():
    d = _valid()
    with pytest.raises(LandmarkSchemaError):
        parse_landmarks(d, valid_types={"building"})   # bridge not allowed
    # ok when allowed
    parse_landmarks(d, valid_types={"building", "bridge"})


def test_bad_center():
    d = _valid(); d["landmarks"][0]["center"] = [1.0, 2.0]
    with pytest.raises(LandmarkSchemaError):
        parse_landmarks(d)


def test_nonpositive_dims():
    d = _valid(); d["landmarks"][0]["length"] = 0
    with pytest.raises(LandmarkSchemaError):
        parse_landmarks(d)


def test_empty_landmarks():
    d = _valid(); d["landmarks"] = []
    with pytest.raises(LandmarkSchemaError):
        parse_landmarks(d)


def test_save_load_roundtrip():
    ls = LandmarkSet(bag_name="b", frame_id="f", landmarks=[
        Landmark(id=3, type="gate", center=[0.0, 0.0, 0.0], heading_deg=0.0,
                 length=5.0, width=3.0, notes="x")])
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "b.landmarks.yaml")
        save_landmarks(ls, p)
        ls2 = load_landmarks(p)
    assert ls2.landmarks[0].id == 3
    assert ls2.landmarks[0].type == "gate"
    assert ls2.landmarks[0].notes == "x"
