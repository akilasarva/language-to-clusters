"""Deterministic tests for the map-free phase-deriver + hierarchical taxonomy."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bev_pipeline.phase_deriver import (segment_runs, derive_phases, match_plan,  # noqa: E402
                                        progress_summary)
from bev_pipeline.taxonomy_export import build_hierarchical_cluster_map           # noqa: E402


def test_segment_and_phases():
    # road ... junction(3) ... road ... junction(2) ... road
    stream = (["path"] * 4 + ["junction"] * 3 + ["path"] * 4 +
              ["junction"] * 2 + ["path"] * 3)
    _, runs = derive_phases(stream, approach_win=2, exit_win=2)
    assert len(runs) == 2
    assert runs[0].rtype == "junction" and (runs[0].start, runs[0].end) == (4, 6)
    phases, _ = derive_phases(stream, approach_win=2, exit_win=2)
    # frames 2,3 = approach; 4-6 = in; 7,8 = exit
    assert phases[3][1] == "approach" and phases[5][1] == "in" and phases[7][1] == "exit"


def test_debounce_singleton():
    # a lone junction frame with min_len=2 is dropped (noise)
    stream = ["path"] * 3 + ["junction"] + ["path"] * 3
    runs = segment_runs(stream, min_len=2)
    assert runs == []


def test_plan_defaulting():
    # two junctions; plan wants the one WITH a stop sign (2nd). 1st must default.
    stream = ["path"] * 3 + ["junction"] * 2 + ["path"] * 3 + ["junction"] * 2 + ["path"] * 2
    _, runs = derive_phases(stream)
    assert len(runs) == 2
    runs[1].cues.add("StopSign")                 # cue only on 2nd
    plan = [{"type": "junction", "cue": "StopSign"}]
    match_plan(runs, plan)
    s = progress_summary(runs, plan)
    assert runs[0].matched_step is None          # 1st defaulted (passed as road)
    assert runs[1].matched_step == 0             # plan advances at the cued 2nd
    assert s["complete"] and s["regions_defaulted"] == 1


def test_hierarchy_overlap():
    """Subsumption in the PEDESTRIAN binding, where planner name == cluster name."""
    labels = ["open_space", "path", "along_edge", "passage", "junction"]
    doc = build_hierarchical_cluster_map(labels, "env")
    jid = labels.index("junction")
    # subsumption: junction id in BOTH its own mode and the parent `path` mode
    assert jid in doc["modes"]["junction"]
    assert jid in doc["modes"]["path"]
    # along_edge and passage also default down to `path`
    assert labels.index("along_edge") in doc["modes"]["path"]
    assert labels.index("passage") in doc["modes"]["path"]
    # open_space does NOT default to path
    assert labels.index("open_space") not in doc["modes"].get("path", [])
