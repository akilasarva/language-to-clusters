#!/usr/bin/env python3
"""Building footprints from CARLA, and the road-region BLOCK each one is enclosed by.

WHY. A building footprint IS the block: the junctions ringing it are the decision points,
and how many there are is the turn count. Deriving the block from geometry replaces mining
driven traces for a closed region cycle and guessing the turn count.

Runs INSIDE the bridge container (host carla is 0.10.0, server 0.9.14).
"""
import argparse, json, math, os, sys
import numpy as np
import carla

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from carla_gt_bridge.region_lookup import load_region_table   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--town", default="town05")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--min-extent-m", type=float, default=6.0,
                    help="skip clutter; a block-forming building is not 3 m across")
    ap.add_argument("--ring-m", type=float, default=90.0,
                    help="a junction this far from the footprint centre may ring it")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    town = a.town.lower()
    client = carla.Client(a.host, a.port); client.set_timeout(60.0)
    world = client.get_world()
    if town not in world.get_map().name.lower():
        world = client.load_world(town.capitalize().replace("town", "Town"))
    print(f"map: {world.get_map().name}")

    bbs = world.get_level_bbs(carla.CityObjectLabel.Buildings)
    print(f"{len(bbs)} building bounding boxes")

    table = load_region_table(os.path.join(HERE, "config", f"regions.{town}.npz"))
    W = np.load(os.path.join(HERE, "config", f"regions.{town}.npz"))["waypoints"]
    rid = W[:, 2].astype(int)
    juncs = sorted({int(r) for r in set(rid) if table.label_of(int(r)) == "junction"})
    jc = {r: table.centroid_of(r) for r in juncs}

    out = []
    for i, bb in enumerate(bbs):
        ex, ey = float(bb.extent.x), float(bb.extent.y)
        if max(ex, ey) < a.min_extent_m:
            continue
        # THE ONLY FRAME CONVERSION: ROS y is -CARLA y (see lane_spawn.py).
        cx, cy = float(bb.location.x), float(-bb.location.y)
        ring = [(r, math.hypot(jc[r][0] - cx, jc[r][1] - cy)) for r in juncs]
        ring = [(r, d) for r, d in ring if d <= a.ring_m]
        if len(ring) < 3:
            continue
        # order the ring by bearing about the footprint centre -- that IS the traversal order
        ring.sort(key=lambda rd: math.atan2(jc[rd[0]][1] - cy, jc[rd[0]][0] - cx))
        out.append(dict(id=f"bldg_{i}", x=round(cx, 2), y=round(cy, 2),
                        extent=[round(ex, 2), round(ey, 2), round(float(bb.extent.z), 2)],
                        ring=[r for r, _ in ring],
                        ring_dist_m=[round(d, 1) for _, d in ring]))

    # one entry per distinct ring: many footprints share a block
    by_ring: dict[tuple, dict] = {}
    for b in out:
        k = tuple(sorted(b["ring"]))
        by_ring.setdefault(k, dict(ring=b["ring"], members=[], n_junctions=len(b["ring"])))
        by_ring[k]["members"].append(b["id"])
    blocks = sorted(by_ring.values(), key=lambda d: -len(d["members"]))

    print(f"{len(out)} block-forming buildings -> {len(blocks)} distinct rings")
    for bl in blocks[:12]:
        print(f"   {bl['n_junctions']} junctions {bl['ring']}  ({len(bl['members'])} buildings)")

    path = a.out or os.path.join(HERE, "config", f"buildings.{town}.json")
    json.dump({"town": town,
               "_note": ("Building footprints (ROS frame) and the junction ring enclosing "
                         "each. `ring` is ordered by bearing about the footprint centre, so "
                         "len(ring) is the TURN COUNT for an around-the-block mission. "
                         "Rings are CANDIDATES from proximity, not verified circuits -- "
                         "check a ring is driveable before writing English against it."),
               "buildings": out, "blocks": blocks}, open(path, "w"), indent=1)
    print(f"wrote {path}")
    return 0 if blocks else 3


if __name__ == "__main__":
    sys.exit(main())
