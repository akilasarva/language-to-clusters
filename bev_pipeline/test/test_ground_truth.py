"""Tests for geometry-derived trajectory labels (TDD spec item).

Synthetic trajectories toward/through/around pass-through and perimeter
landmarks; assert phase sequencing, the monotonicity guard (no exit without
having passed through), type-dependent vocabulary, and overlap priority.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bev_pipeline.landmark_schema import LandmarkSet, Landmark          # noqa: E402
from bev_pipeline.trajectory_labeler import label_trajectory, LabelerParams  # noqa: E402

TYPES = {
    "types": {
        "bridge": {"kind": "pass_through", "priority": 10},
        "intersection": {"kind": "pass_through", "priority": 9},
        "building": {"kind": "perimeter", "priority": 8},
    },
    "open_road_priority": 0,
}


def _seq(labels):
    """Collapse consecutive duplicates into an ordered phase sequence."""
    out = []
    for x in labels:
        if not out or out[-1] != x:
            out.append(x)
    return out


def test_pass_through_approach_on_exit():
    bridge = Landmark(id=0, type="bridge", center=[10.0, 0.0, 0.0],
                      heading_deg=0.0, length=10.0, width=4.0)
    ls = LandmarkSet("b", "f", [bridge])
    xs = np.linspace(-20, 40, 121)
    pos = np.column_stack([xs, np.zeros_like(xs)])
    labels, _ = label_trajectory(pos, ls, TYPES)
    seq = _seq(labels)
    # should pass open_road -> approach -> on -> exit -> open_road, in order
    assert "approach_bridge" in seq and "on_bridge" in seq and "exit_bridge" in seq
    assert seq.index("approach_bridge") < seq.index("on_bridge") < seq.index("exit_bridge")


def test_monotonicity_no_exit_without_on():
    # approach then turn around BEFORE entering the footprint -> never on/exit
    bridge = Landmark(id=0, type="bridge", center=[10.0, 0.0, 0.0],
                      heading_deg=0.0, length=10.0, width=4.0)
    ls = LandmarkSet("b", "f", [bridge])
    fwd = np.linspace(-20, 2, 45)      # stops at x=2 (s=-8, still 'approach')
    back = np.linspace(2, -20, 45)
    xs = np.concatenate([fwd, back])
    pos = np.column_stack([xs, np.zeros_like(xs)])
    labels, _ = label_trajectory(pos, ls, TYPES)
    assert "approach_bridge" in labels
    assert "on_bridge" not in labels        # never entered
    assert "exit_bridge" not in labels      # and so never exits


def test_perimeter_uses_along_not_on():
    building = Landmark(id=0, type="building", center=[0.0, 10.0, 0.0],
                        heading_deg=0.0, length=10.0, width=6.0)
    ls = LandmarkSet("b", "f", [building])
    xs = np.linspace(-25, 25, 101)
    pos = np.column_stack([xs, np.full_like(xs, 3.0)])   # skirts 4 m south of it
    labels, _ = label_trajectory(pos, ls, TYPES)
    seq = _seq(labels)
    assert "along_building" in seq
    assert "on_building" not in labels       # perimeter has no 'on'
    assert "approach_building" in seq and "exit_building" in seq
    assert seq.index("approach_building") < seq.index("along_building") < seq.index("exit_building")


def test_overlap_priority_tie_break():
    # bridge (prio 10) and intersection (prio 9) both active at the origin pass
    bridge = Landmark(id=0, type="bridge", center=[0.0, 0.0, 0.0],
                      heading_deg=0.0, length=8.0, width=6.0)
    inter = Landmark(id=1, type="intersection", center=[0.0, 0.0, 0.0],
                     heading_deg=0.0, length=8.0, width=6.0)
    ls = LandmarkSet("b", "f", [bridge, inter])
    pos = np.column_stack([np.linspace(-10, 10, 41), np.zeros(41)])
    labels, metas = label_trajectory(pos, ls, TYPES)
    # at the frame nearest origin, both active -> bridge (higher priority) wins
    mid = labels[20]
    assert mid.endswith("_bridge"), mid
    # metadata records both candidates
    assert any(len(m.get("active", [])) >= 2 for m in metas)


def test_open_road_when_far():
    bridge = Landmark(id=0, type="bridge", center=[100.0, 100.0, 0.0],
                      heading_deg=0.0, length=5.0, width=3.0)
    ls = LandmarkSet("b", "f", [bridge])
    pos = np.column_stack([np.linspace(-10, 10, 21), np.zeros(21)])
    labels, _ = label_trajectory(pos, ls, TYPES)
    assert all(x == "open_road" for x in labels)
