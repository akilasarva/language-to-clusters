"""Geometry-derived behavioral-state labels from trajectory vs. landmark bboxes.

Given the vehicle's trajectory (odometry positions) and a per-bag
:class:`~bev_pipeline.landmark_schema.LandmarkSet`, assign each frame a
``(phase, landmark_type)`` label — e.g. ``approach_bridge`` / ``on_bridge`` /
``exit_bridge`` — purely from geometry, no camera or VLM. This is the
label-modality-fair alternative to the VLM auto-labeler: the target is defined
by the robot's spatial relationship to structure, which the LiDAR can recover.

Two phase vocabularies by landmark kind (from landmark_types.yaml):
  * pass_through (bridge/ramp/gate/intersection): approach / on / exit
  * perimeter    (building):                      approach / along / exit  (no "on")

A per-landmark monotonic state machine (NONE -> APPROACH -> ON/ALONG -> EXIT,
reset when the landmark leaves influence) guarantees the TDD invariants:
approach is only emitted while heading toward a not-yet-passed landmark, and
exit is never emitted without having passed through / alongside it first.

Overlap: when several landmarks are active at once, the highest-`priority` one
wins (ties broken by nearest centroid); the rest are kept as metadata only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .landmark_schema import Landmark, LandmarkSet

# monotonic states
_NONE, _APPROACH, _MID, _EXIT = 0, 1, 2, 3


@dataclass
class LabelerParams:
    lateral_margin: float = 2.0       # extra half-width tolerance (pass_through)
    approach_range: float = 15.0      # how far ahead "approach" starts (m)
    perimeter_influence: float = 15.0  # perimeter: approach/exit band outer (m)
    perimeter_along_band: float = 6.0  # perimeter: "along" when within this (m)
    open_label: str = "open_road"


@dataclass
class _Tracker:
    """Per-landmark monotonic state across the trajectory."""
    kind: str
    state: int = _NONE

    def reset(self):
        self.state = _NONE


def _axes(heading_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    th = math.radians(heading_deg)
    u = np.array([math.cos(th), math.sin(th)])       # along heading
    w = np.array([-math.sin(th), math.cos(th)])      # lateral
    return u, w


def _dist_to_rect(p: np.ndarray, lm: Landmark) -> float:
    """Distance from point p (2D) to the landmark's oriented rectangle (0 inside)."""
    c = np.array(lm.center[:2])
    u, w = _axes(lm.heading_deg)
    v = p - c
    s = abs(v @ u) - lm.length / 2.0
    d = abs(v @ w) - lm.width / 2.0
    s = max(s, 0.0)
    d = max(d, 0.0)
    return math.hypot(s, d)


def _pass_through_phase(tr: _Tracker, p: np.ndarray, lm: Landmark,
                        prm: LabelerParams) -> Optional[str]:
    c = np.array(lm.center[:2])
    u, w = _axes(lm.heading_deg)
    v = p - c
    s = v @ u                      # signed along-axis
    d = v @ w                      # signed lateral
    half_L, half_W = lm.length / 2.0, lm.width / 2.0
    laterally_in = abs(d) <= (half_W + prm.lateral_margin)
    within_influence = laterally_in and (abs(s) <= half_L + prm.approach_range)

    if not within_influence:
        # completed a pass? reset for a possible future pass
        if tr.state in (_MID, _EXIT):
            tr.reset()
        else:
            tr.state = _NONE
        return None

    if abs(s) <= half_L:                     # inside footprint
        tr.state = _MID
    elif s < -half_L:                        # before entry
        if tr.state < _MID:
            tr.state = _APPROACH
        # if already passed (>=MID) and back in BEFORE: keep MID (turned around)
    else:                                    # s > half_L, after exit
        if tr.state >= _MID:
            tr.state = _EXIT
        # if never was inside, do NOT emit exit (skirted the far end)
    return {_NONE: None, _APPROACH: "approach", _MID: "on", _EXIT: "exit"}[tr.state]


def _perimeter_phase(tr: _Tracker, p: np.ndarray, lm: Landmark,
                     prm: LabelerParams) -> Optional[str]:
    dist = _dist_to_rect(p, lm)
    if dist > prm.perimeter_influence:
        if tr.state in (_MID, _EXIT):
            tr.reset()
        else:
            tr.state = _NONE
        return None
    if dist <= prm.perimeter_along_band:     # alongside
        tr.state = _MID
    else:                                    # in the outer band
        if tr.state < _MID:
            tr.state = _APPROACH
        elif tr.state >= _MID:
            tr.state = _EXIT
    return {_NONE: None, _APPROACH: "approach", _MID: "along", _EXIT: "exit"}[tr.state]


def label_trajectory(positions, landmark_set: LandmarkSet, types_cfg: dict,
                     params: Optional[LabelerParams] = None
                     ) -> Tuple[List[str], List[dict]]:
    """Label a trajectory. Returns (labels, overlap_meta) per frame.

    ``positions`` is (N,2) or (N,3) in the landmark frame. ``types_cfg`` is the
    parsed landmark_types.yaml (for kind + priority).
    """
    prm = params or LabelerParams()
    types = types_cfg.get("types") or {}
    open_prio = int(types_cfg.get("open_road_priority", 0))
    pos = np.asarray(positions, dtype=float)[:, :2]

    trackers = {lm.id: _Tracker(kind=types.get(lm.type, {}).get("kind", "pass_through"))
                for lm in landmark_set.landmarks}

    labels: List[str] = []
    metas: List[dict] = []
    for p in pos:
        candidates = []      # (priority, dist_to_center, label, lm)
        for lm in landmark_set.landmarks:
            tr = trackers[lm.id]
            if tr.kind == "perimeter":
                phase = _perimeter_phase(tr, p, lm, prm)
            else:
                phase = _pass_through_phase(tr, p, lm, prm)
            if phase is not None:
                prio = int(types.get(lm.type, {}).get("priority", 1))
                dc = float(np.linalg.norm(p - np.array(lm.center[:2])))
                candidates.append((prio, dc, f"{phase}_{lm.type}", lm))
        if not candidates:
            labels.append(prm.open_label)
            metas.append({})
            continue
        # highest priority wins; ties -> nearest centroid
        candidates.sort(key=lambda x: (-x[0], x[1]))
        best = candidates[0]
        labels.append(best[2])
        metas.append({"active": [{"id": c[3].id, "type": c[3].type, "label": c[2]}
                                 for c in candidates],
                      "chosen_id": best[3].id})
    return labels, metas
