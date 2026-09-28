"""Region geometry for a brain tree, built from a town's cluster map.

Every tree the MPC drives carries the town's `cluster_labels`, `centroids` and
`bearing_map` (it steers by them). This builds those three from
`config/cluster_map.carla_<town>.yaml` through the planner's own taxonomy loader, so a
hand-assembled plan and a generated one carry identical geometry.
"""
from __future__ import annotations

import os
import sys

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS = os.path.dirname(PKG)
for _p in (os.path.join(WS, "nl_planner"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def region_geometry(town: str) -> dict:
    """`cluster_labels`, `centroids`, `bearing_map` for `town` (e.g. "town05")."""
    from nl_planner.taxonomy import load_taxonomy
    tax = load_taxonomy(os.path.join(PKG, "config", f"cluster_map.carla_{town}.yaml"))
    return {"cluster_labels": {str(k): v for k, v in tax.cluster_labels().items()},
            "centroids": tax.centroid_map(),
            "bearing_map": tax.bearing_map_dict()}


def neutral_plan(town: str, start: int) -> dict:
    """Geometry plus one traverse step (path -> the next junction), and nothing else.

    For BRAIN_MODE=llm: the LLM brain ignores the tree, but the MPC takes its region
    geometry from the latched plan. No forbids, requirements or monitors, so none of a
    mission's own constraints leak into the baseline.
    """
    geo = region_geometry(town)
    labels = {int(k): v for k, v in geo["cluster_labels"].items()}
    junctions = sorted(k for k, v in labels.items() if v == "junction")
    return {"plan_name": f"neutral_{town}", "description": "geometry only (BRAIN_MODE=llm)",
            **geo,
            "steps": [{"step": 0, "description": "Continue along the path to the next junction",
                       "start_cluster": start, "goal_cluster": junctions[0],
                       "start_mode": "path", "goal_mode": "junction",
                       "transition_cue": None, "notes": None, "trigger": "traverse",
                       "cue_ordinal": None, "accept_clusters": junctions,
                       "accept_clusters_degraded": junctions, "branches": None}]}
