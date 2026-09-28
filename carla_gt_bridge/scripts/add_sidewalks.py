#!/usr/bin/env python3
"""Add sidewalk regions to a town's region table, as a mode of their own.

WHY A MODE AND NOT A RELABEL. "Make sure to stay on the sidewalk at all times" currently
produces `G Phi_Path`, and on a two-mode CARLA map `Path` resolves to ALL 74 regions -- so
the invariant accepts the whole world and can never be violated. It reports satisfied on
every trace, including one that never obeys it.

Giving sidewalks their own mode fixes that at the root: `stay on the sidewalk` resolves to
the sidewalk regions and nothing else, so the invariant is strict and can actually fail.

HOW THE REGIONS ARE CUT. By (road_id, lane_id), which is how OpenDRIVE already segments a
sidewalk -- one continuous run alongside one road. That is the same granularity the
driving regions use, so a sidewalk region is comparable in size to a path region rather
than being one giant blob or 3000 singletons. Runs longer than `--max-len` are split so a
single long boulevard does not become one region spanning the map.

WHAT THIS DOES NOT DO. It does not decide `along_edge` -- that needs building bounding
boxes from a live server. Sidewalks land as `sidewalk`; a later pass can promote the
enclosed ones.

    python3 add_sidewalks.py Town05 --out config/regions.town05.sidewalks.npz
"""
from __future__ import annotations

import argparse
import collections
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = os.path.join(os.path.dirname(HERE), "config")


def sidewalk_runs(town: str, step: float = 2.0, max_len: int = 60):
    """(road_id, lane_id) -> ordered waypoint xy, from the shipped OpenDRIVE. No server."""
    import carla
    m = carla.Map(town, open(os.path.join(CFG, f"{town}.xodr")).read())
    by_lane: dict[tuple, list] = collections.defaultdict(list)
    seen = set()
    for wp in m.generate_waypoints(step):
        cur, hops = wp, 0
        while cur is not None and hops < 6:
            cur = cur.get_right_lane()
            hops += 1
            if cur is None:
                break
            if str(cur.lane_type).split(".")[-1] == "Sidewalk":
                t = cur.transform.location
                key = (round(t.x, 1), round(t.y, 1))
                if key not in seen:
                    seen.add(key)
                    by_lane[(cur.road_id, cur.lane_id)].append((t.x, t.y, cur.s))
                break
    runs = []
    for (rid, lid), pts in sorted(by_lane.items()):
        pts.sort(key=lambda p: p[2])            # along the road's own s coordinate
        arr = np.array([[p[0], p[1]] for p in pts])
        for i in range(0, len(arr), max_len):   # split so one boulevard is not one region
            chunk = arr[i:i + max_len]
            if len(chunk) >= 4:                 # too short to be a meaningful region
                runs.append(chunk)
    return runs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("town")
    ap.add_argument("--base", default=None, help="region table to extend")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-len", type=int, default=60)
    a = ap.parse_args()

    slug = a.town.lower()
    base = a.base or os.path.join(CFG, f"regions.{slug}.npz")
    out = a.out or os.path.join(CFG, f"regions.{slug}.sidewalks.npz")
    d = np.load(base, allow_pickle=True)
    wp, rids, labels, cents = d["waypoints"], d["rids"], d["labels"], d["centroids"]

    runs = sidewalk_runs(a.town, max_len=a.max_len)
    if not runs:
        print(f"  {a.town}: no sidewalk lanes found")
        return 1

    next_id = int(rids.max()) + 1
    new_wp, new_rids, new_lab, new_cent = [], [], [], []
    for chunk in runs:
        rid = next_id
        next_id += 1
        new_wp.append(np.column_stack([chunk, np.full(len(chunk), rid, dtype=float)]))
        new_rids.append(rid)
        new_lab.append("sidewalk")
        new_cent.append(chunk.mean(axis=0))

    np.savez(
        out,
        waypoints=np.vstack([wp, np.vstack(new_wp)]),
        rids=np.concatenate([rids, np.array(new_rids, dtype=rids.dtype)]),
        labels=np.concatenate([labels, np.array(new_lab, dtype="<U10")]),
        centroids=np.vstack([cents, np.array(new_cent)]),
    )
    print(f"  {a.town}: {len(rids)} road regions + {len(new_rids)} sidewalk regions "
          f"= {len(rids) + len(new_rids)}")
    print(f"    sidewalk waypoints {sum(len(c) for c in runs)}  ids {new_rids[0]}..{new_rids[-1]}")
    print(f"    wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
