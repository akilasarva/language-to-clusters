#!/usr/bin/env python3
"""Label the regions that pass UNDER a building deck, so `Detect(Overpass)` has something to fire on.

WHY THIS AND NOT THE `Bridge` LABEL. Town05's overpasses between J58 and J64 are commercial
buildings spanning the carriageway, and CARLA labels them `Buildings`. Three different checks
missed them for three different reasons, which is why this is extracted rather than inferred:

  `CityObjectLabel.Bridge`  0 boxes on that corridor -- the deck is not a Bridge.
  stacked lane waypoints    nothing DRIVABLE runs on top, so there is no second lane level.
  buildings.town05.json     stores x, y, extent and NO Z, so a deck at +7.8 m and a painted
                            median are the same record.

The signature that does work: a `Buildings` box sitting ABOVE the road with road waypoints
underneath it. On Town05 the decks sit at z = +7.82, half-height 1.49, in pairs
straddling the carriageway at y = +-9.3, over road_id 39 whose surface is z = 0.00.

Writes an `overpass` key into `landmarks.<town>.json`, MERGING rather than replacing, because
`extract_traffic_lights.py` owns the other keys in that file and `gt_cue_node` reads every
list-valued key as a region-scoped family.
"""
import argparse, json, os, sys
import carla

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from carla_gt_bridge.region_lookup import load_region_table   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--town", default="town05")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--min-deck-z", type=float, default=2.0,
                    help="a deck must clear the road; 2 m excludes kerbs, medians and signage")
    ap.add_argument("--max-road-z", type=float, default=1.5,
                    help="the lane beneath must be at ground level, or it is not an underpass")
    ap.add_argument("--max-dist-m", type=float, default=25.0)
    # A DRIVABLE CLEARANCE BAND IS THE WHOLE FILTER. Without it, a tall tower block scores
    # a large "clearance" simply because a road runs past its foot, and buildings whose
    # bottom face sits ON the verge score ~+0.1..+0.2. Both are "a Buildings box above z=2
    # with tarmac underneath" and neither is something you drive under. The real deck on
    # the J58-J64 corridor clears 6.33 m.
    ap.add_argument("--min-clearance-m", type=float, default=2.5,
                    help="a vehicle must fit; also excludes boxes resting on the verge")
    ap.add_argument("--max-clearance-m", type=float, default=10.0,
                    help="above this it is a tall building beside the road, not a deck over it")
    a = ap.parse_args()

    town = a.town.lower()
    client = carla.Client(a.host, a.port); client.set_timeout(120.0)
    world = client.get_world()
    if town not in world.get_map().name.lower():
        world = client.load_world(town.capitalize().replace("town", "Town"))
    print(f"map: {world.get_map().name}", flush=True)

    table = load_region_table(os.path.join(HERE, "config", f"regions.{town}.npz"))
    cmap = world.get_map()

    decks = [bb for bb in world.get_level_bbs(carla.CityObjectLabel.Buildings)
             if bb.location.z >= a.min_deck_z]
    print(f"{len(decks)} Buildings boxes with z >= {a.min_deck_z} m", flush=True)

    rows, seen = [], {}
    for bb in decks:
        L, E = bb.location, bb.extent
        # Sample the footprint and ask the map for the lane under each sample. A deck over a
        # road returns a waypoint at ground level; a deck over a roof or a plaza returns
        # nothing near, or something already elevated.
        hits = []
        n = 3
        for i in range(-n, n + 1):
            for j in range(-n, n + 1):
                px = L.x + (E.x * i / n if E.x else 0.0)
                py = L.y + (E.y * j / n if E.y else 0.0)
                wp = cmap.get_waypoint(carla.Location(x=px, y=py, z=0.0),
                                       project_to_road=True)
                if wp is None:
                    continue
                wl = wp.transform.location
                if abs(wl.z) > a.max_road_z:
                    continue
                if abs(wl.x - px) > 6.0 or abs(wl.y - py) > 6.0:
                    continue                      # the lane is not actually under the box
                hits.append((wl.x, wl.y, wl.z))
        if not hits:
            continue
        # ROS/planar frame: y = -carla_y (lane_spawn.py owns this conversion).
        for hx, hy, hz in hits:
            rid, dist = table.nearest(float(hx), float(-hy))
            if dist > a.max_dist_m:
                continue
            clear = float(L.z - E.z - hz)
            if not (a.min_clearance_m <= clear <= a.max_clearance_m):
                continue
            # KEEP THE LARGEST QUALIFYING CLEARANCE, not the smallest: keeping the minimum lets
            # any low box near the same region hide the actual deck (region 39 sits directly
            # under the x=46 deck pair, 6.33 m).
            prev = seen.get(int(rid))
            if prev is None or clear > prev["clearance_m"]:
                seen[int(rid)] = dict(region=int(rid), x=round(float(hx), 2),
                                      y=round(float(-hy), 2), deck_z=round(float(L.z), 2),
                                      clearance_m=round(clear, 2),
                                      label=table.label_of(int(rid)))
    rows = sorted(seen.values(), key=lambda r: r["region"])
    print(f"\n{len(rows)} region(s) pass under a deck:")
    for r in rows:
        print(f"   region {r['region']:3} ({r['label']:8})  deck z {r['deck_z']:+5.2f}"
              f"  clearance {r['clearance_m']:+5.2f} m   at ({r['x']:+7.1f},{r['y']:+7.1f})")
    if not rows:
        print("   none -- nothing to write"); return 3

    path = os.path.join(HERE, "config", f"landmarks.{town}.json")
    blob = json.load(open(path)) if os.path.exists(path) else {"town": town}
    blob["overpass"] = rows
    blob["_note_overpass"] = (
        "Regions that pass UNDER a building deck. NOT from CityObjectLabel.Bridge -- Town05's "
        "overpasses on the J58-J64 corridor are Buildings spanning the carriageway, and the "
        "Bridge label returns 0 boxes there. Region-scoped like traffic_light/stop_sign: "
        "gt_cue_node answers it by membership, so it needs no pose topic.")
    with open(path, "w") as f:
        json.dump(blob, f, indent=1)
    print(f"\nwrote {path}  (overpass: {len(rows)} regions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
