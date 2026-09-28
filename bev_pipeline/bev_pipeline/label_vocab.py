"""Closed label vocabulary derived from ``landmark_types.yaml``.

A label is a ``(phase, landmark_type)`` pair flattened to a string like
``approach_bridge`` / ``along_building``, plus the special ``open_road`` label
for "no active landmark". The vocabulary is closed: both the VLM auto-labeler
and the supervised classifier choose only from this set, so nothing downstream
can emit a state nl_planner has never seen.

This module is intentionally dependency-light (PyYAML only) so it can be used
by the labeler, the classifier metadata, and taxonomy export alike.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List

import yaml


@dataclass(frozen=True)
class LandmarkVocab:
    """Closed label set + the type/kind/priority tables behind it."""

    labels: List[str]                 # ordered; open_label first
    open_label: str
    kinds: Dict[str, str]             # type -> "pass_through" | "perimeter"
    priorities: Dict[str, int]        # type -> priority
    phases_by_kind: Dict[str, List[str]]
    open_road_priority: int

    def index_of(self, label: str) -> int:
        return self.labels.index(label)

    def phases_for(self, landmark_type: str) -> List[str]:
        return self.phases_by_kind[self.kinds[landmark_type]]

    def label(self, phase: str, landmark_type: str) -> str:
        return f"{phase}_{landmark_type}"

    def is_valid(self, label: str) -> bool:
        return label in self.labels


def build_vocab(types_cfg: dict) -> LandmarkVocab:
    """Build a :class:`LandmarkVocab` from a parsed landmark_types.yaml dict."""
    types = types_cfg.get("types") or {}
    phases_by_kind = types_cfg.get("phases") or {
        "pass_through": ["approach", "on", "exit"],
        "perimeter": ["approach", "along", "exit"],
    }

    # Guard against the YAML 1.1 `on`->True boolean trap (and any other
    # non-string token) so a mis-quoted config fails loudly instead of
    # producing a "True_bridge" label.
    _bool_map = {True: "on", False: "off"}
    for kind, plist in phases_by_kind.items():
        fixed = []
        for p in plist:
            if isinstance(p, bool):
                raise ValueError(
                    f"phase token in kind {kind!r} parsed as boolean {p!r} — "
                    f"quote it in landmark_types.yaml (e.g. \"on\"). "
                    f"Interpreting as {_bool_map[p]!r} would be ambiguous.")
            fixed.append(str(p))
        phases_by_kind[kind] = fixed
    open_label = types_cfg.get("open_label", "open_road")
    open_road_priority = int(types_cfg.get("open_road_priority", 0))

    kinds: Dict[str, str] = {}
    priorities: Dict[str, int] = {}
    for tname, tinfo in types.items():
        kind = tinfo.get("kind", "pass_through")
        if kind not in phases_by_kind:
            raise ValueError(f"type {tname!r} has unknown kind {kind!r}")
        kinds[tname] = kind
        priorities[tname] = int(tinfo.get("priority", 1))

    labels = [open_label]
    for tname in types:                      # deterministic: config order
        for phase in phases_by_kind[kinds[tname]]:
            labels.append(f"{phase}_{tname}")

    return LandmarkVocab(
        labels=labels,
        open_label=open_label,
        kinds=kinds,
        priorities=priorities,
        phases_by_kind=phases_by_kind,
        open_road_priority=open_road_priority,
    )


def load_vocab(types_yaml_path: str) -> LandmarkVocab:
    """Load and build the vocabulary from a landmark_types.yaml file path."""
    if not os.path.exists(types_yaml_path):
        raise FileNotFoundError(f"landmark_types.yaml not found: {types_yaml_path}")
    with open(types_yaml_path) as f:
        cfg = yaml.safe_load(f) or {}
    return build_vocab(cfg)
