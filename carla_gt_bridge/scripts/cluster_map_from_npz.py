#!/usr/bin/env python3
"""Emit a hierarchical cluster_map from a region table, keeping the geometry.

WHY THIS EXISTS. `map_regions.py` builds a table AND its cluster map together, straight
from the XODR. `add_sidewalks.py` extends a table afterwards and writes only the npz, so
the matching cluster map has only `environment`, `axes` and `modes`: no centroids, no
bearing_map. A plan materialized against it has nothing for the controller to steer by.

This closes that gap for any derived table. Centroids come from the npz. The bearing map
comes from the ROAD table the derived one was built on, because adjacency is a property of
the road network: you never route TO a sidewalk, you forbid it, and StepTargeter only
needs adjacency for modes a step can target.

    python3 cluster_map_from_npz.py \\
        --npz config/regions.town05.approach_sidewalks.npz \\
        --bearings-from config/cluster_map.carla_town05.approach.yaml \\
        --env carla_town05_approach_sidewalks
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import yaml

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS = os.path.dirname(PKG)
for _p in (os.path.join(WS, "bev_pipeline"), PKG):
    if _p not in sys.path:
        sys.path.insert(0, _p)

#: Label -> (mode name, ancestors). Extends taxonomy_export.ROAD_HIERARCHY with the two
#: derived labels. `sidewalk` is deliberately NOT under `Road: On`: the drivable surface
#: and the pedestrian one are different axes, and making a sidewalk satisfy a road step
#: would let a route be planned across it.
HIERARCHY = {
    "path":      ("Road: On", []),
    "approach":  ("Intersection: Approach/Enter", ["Road: On"]),
    "junction":  ("Intersection: In", ["Road: On"]),
    "sidewalk":  ("Sidewalk", []),
    "open_space": ("Open Space", []),
    "along_edge": ("Along Wall", ["Road: On"]),
    "passage":   ("Passage", ["Road: On"]),
    "other":     ("Other", []),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--bearings-from", default=None,
                    help="cluster_map yaml whose bearing_map covers the road regions")
    ap.add_argument("--env", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    z = np.load(a.npz, allow_pickle=True)
    rids = [int(r) for r in z["rids"]]
    labels = [str(x) for x in z["labels"]]
    cents = {int(r): [float(c[0]), float(c[1])]
             for r, c in zip(rids, z["centroids"])}

    bearings = {}
    if a.bearings_from:
        src = yaml.safe_load(open(a.bearings_from))
        bearings = src.get("bearing_map") or {}

    from bev_pipeline.taxonomy_export import write_hierarchical_cluster_map

    out = a.out or os.path.join(PKG, "config", f"cluster_map.{a.env}.yaml")
    write_hierarchical_cluster_map(
        [], a.env, out,
        source=f"cluster_map_from_npz.py from {os.path.basename(a.npz)}",
        hierarchy=HIERARCHY,
        centroids=cents,
        bearing_map=bearings,
        id_labels=dict(zip(rids, labels)),
    )
    doc = yaml.safe_load(open(out))
    modes = {k: len(v) for k, v in (doc.get("modes") or {}).items()}
    print(f"  {out}")
    print(f"  modes {modes}")
    print(f"  centroids {len(doc.get('centroids') or {})}  "
          f"bearing_map {len(doc.get('bearing_map') or {})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
