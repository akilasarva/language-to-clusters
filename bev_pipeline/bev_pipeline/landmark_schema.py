"""Per-bag landmark annotation schema (training-time only).

Each training bag gets its own ``<bag>.landmarks.yaml`` of independently-drawn
landmark bounding boxes, tagged only with a *type* (bridge/building/... ) — no
cross-bag instance identity. These drive the geometry-derived trajectory labels
(:mod:`bev_pipeline.trajectory_labeler`). They are NOT needed at inference time.

Schema::

    schema_version: 1
    bag_name: <str>
    frame_id: <str>          # frame the bboxes are expressed in (this bag only)
    landmarks:
      - id: 0                # unique WITHIN this file only
        type: bridge         # must be a key in landmark_types.yaml
        center: [x, y, z]
        heading_deg: 87.5    # bbox long-axis heading
        length: 22.0
        width: 4.5
        notes: "..."         # optional

Validation raises :class:`LandmarkSchemaError` (same convention as nl_planner's
TaxonomyError).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import yaml

SCHEMA_VERSION = 1


class LandmarkSchemaError(ValueError):
    """Raised for malformed landmark annotation files."""


@dataclass
class Landmark:
    id: int
    type: str
    center: List[float]           # [x, y, z]
    heading_deg: float
    length: float
    width: float
    notes: str = ""

    def to_dict(self) -> dict:
        d = {"id": self.id, "type": self.type, "center": list(self.center),
             "heading_deg": self.heading_deg, "length": self.length,
             "width": self.width}
        if self.notes:
            d["notes"] = self.notes
        return d


@dataclass
class LandmarkSet:
    bag_name: str
    frame_id: str
    landmarks: List[Landmark] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    def types(self) -> set:
        return {lm.type for lm in self.landmarks}


def _require(cond, msg):
    if not cond:
        raise LandmarkSchemaError(msg)


def load_landmarks(path: str, valid_types: Optional[set] = None) -> LandmarkSet:
    """Load + validate a ``<bag>.landmarks.yaml``.

    ``valid_types`` (e.g. from landmark_types.yaml) restricts allowed ``type``
    values; if None, any non-empty string is accepted.
    """
    if not os.path.exists(path):
        raise LandmarkSchemaError(f"landmarks file not found: {path}")
    try:
        raw = yaml.safe_load(open(path).read()) or {}
    except yaml.YAMLError as e:
        raise LandmarkSchemaError(f"could not parse YAML {path}: {e}") from e
    return parse_landmarks(raw, valid_types)


def parse_landmarks(raw: dict, valid_types: Optional[set] = None) -> LandmarkSet:
    _require(isinstance(raw, dict), "landmarks doc must be a mapping")
    ver = raw.get("schema_version")
    _require(ver == SCHEMA_VERSION,
             f"schema_version must be {SCHEMA_VERSION}, got {ver!r}")
    bag_name = raw.get("bag_name")
    _require(isinstance(bag_name, str) and bag_name.strip(),
             "missing non-empty 'bag_name'")
    frame_id = raw.get("frame_id")
    _require(isinstance(frame_id, str) and frame_id.strip(),
             "missing non-empty 'frame_id'")
    lm_raw = raw.get("landmarks")
    _require(isinstance(lm_raw, list) and len(lm_raw) > 0,
             "'landmarks' must be a non-empty list")

    seen_ids = set()
    landmarks = []
    for i, d in enumerate(lm_raw):
        _require(isinstance(d, dict), f"landmark {i} must be a mapping")
        lid = d.get("id")
        _require(isinstance(lid, int), f"landmark {i}: 'id' must be an int")
        _require(lid not in seen_ids, f"duplicate landmark id {lid}")
        seen_ids.add(lid)
        ltype = d.get("type")
        _require(isinstance(ltype, str) and ltype.strip(),
                 f"landmark {lid}: 'type' must be a non-empty string")
        if valid_types is not None:
            _require(ltype in valid_types,
                     f"landmark {lid}: type {ltype!r} not in {sorted(valid_types)}")
        center = d.get("center")
        _require(isinstance(center, (list, tuple)) and len(center) == 3
                 and all(isinstance(v, (int, float)) for v in center),
                 f"landmark {lid}: 'center' must be 3 numbers")
        length = d.get("length")
        width = d.get("width")
        _require(isinstance(length, (int, float)) and length > 0,
                 f"landmark {lid}: 'length' must be positive")
        _require(isinstance(width, (int, float)) and width > 0,
                 f"landmark {lid}: 'width' must be positive")
        heading = d.get("heading_deg", 0.0)
        _require(isinstance(heading, (int, float)),
                 f"landmark {lid}: 'heading_deg' must be a number")
        landmarks.append(Landmark(
            id=lid, type=ltype, center=[float(c) for c in center],
            heading_deg=float(heading), length=float(length),
            width=float(width), notes=str(d.get("notes", ""))))
    return LandmarkSet(bag_name=bag_name, frame_id=frame_id, landmarks=landmarks)


def save_landmarks(ls: LandmarkSet, path: str) -> None:
    """Write a LandmarkSet to YAML (used by the bbox editor tool)."""
    doc = {"schema_version": ls.schema_version, "bag_name": ls.bag_name,
           "frame_id": ls.frame_id,
           "landmarks": [lm.to_dict() for lm in ls.landmarks]}
    with open(path, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False)


def load_landmark_types(path: str) -> dict:
    """Load landmark_types.yaml -> {type: {kind, priority}} + open_road_priority."""
    if not os.path.exists(path):
        raise LandmarkSchemaError(f"landmark_types.yaml not found: {path}")
    raw = yaml.safe_load(open(path).read()) or {}
    types = raw.get("types") or {}
    _require(isinstance(types, dict) and types, "landmark_types 'types' empty")
    return raw
