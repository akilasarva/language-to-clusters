#!/usr/bin/env python3
"""Build the map-native landmark region table (traffic lights) from a live CARLA server.

WHY THIS TABLE EXISTS. `_landmark_near` answers a landmark cue by proximity, using poses
from `/carla/objects`. That topic does not publish these actors, so without this table
every traffic-light cue resolves False. Cones survive the same gap only because they have
a region-scoped path. This gives map-native landmarks the same path.

Traffic lights are NOT spawned, so this table must NOT be read into `_region_family`:
that dict names the families this deployment can PLACE, and the absent-family
short-circuit keys off it. Hence a separate file, `landmarks.<town>.json`, whose name
matches neither the `cones.*.json` nor the `props.*.json` glob.

Runs INSIDE the bridge container: the host `carla` module is 0.10.0 and the server 0.9.14.
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
    ap.add_argument("--families", default="traffic_light,stop_sign",
                    help="map-native families to extract. These are NOT spawned props -- they "
                         "exist in the map, /carla/objects publishes no pose for them, and so "
                         "they need the region-scoped path that cones already have.")
    ap.add_argument("--max-dist-m", type=float, default=25.0,
                    help="a light sits at a junction CORNER, metres off the lane samples")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    town = a.town.lower()
    client = carla.Client(a.host, a.port)
    client.set_timeout(60.0)
    world = client.get_world()
    if town not in world.get_map().name.lower():
        print(f"loading {town} (was {world.get_map().name})")
        world = client.load_world(town.capitalize().replace("town", "Town"))
    print(f"map: {world.get_map().name}")

    npz = os.path.join(HERE, "config", f"regions.{town}.npz")
    table = load_region_table(npz)

    # blueprint filter per family. Keep this next to LANDMARK_ACTOR_TYPES in gt_cue_node:
    # that maps an ACTOR TYPE STRING to a family for the runtime lookup, this maps a family
    # to the BLUEPRINT FILTER for extraction. Two directions of the same relation, so a new
    # family needs both.
    FILTERS = {"traffic_light": "traffic.traffic_light*", "stop_sign": "traffic.stop*"}
    fams = [f.strip() for f in a.families.split(",") if f.strip()]
    bad = [f for f in fams if f not in FILTERS]
    if bad:
        sys.exit(f"no blueprint filter for {bad}; known: {sorted(FILTERS)}")
    actors = {f: list(world.get_actors().filter(FILTERS[f])) for f in fams}
    for f, xs in actors.items():
        print(f"{len(xs)} {f} actors")
    per_family, total = {}, 0
    for fam in fams:
        rows, unassigned = [], 0
        for act in actors[fam]:
            loc = act.get_location()
            # THE ONLY FRAME CONVERSION: ROS y is -CARLA y (see lane_spawn.py).
            x, y = float(loc.x), float(-loc.y)
            rid, dist = table.nearest(x, y)
            if dist > a.max_dist_m:
                unassigned += 1
                continue
            rows.append(dict(id=int(act.id), region=int(rid), x=round(x, 3), y=round(y, 3),
                             z=round(float(loc.z), 3), dist_m=round(float(dist), 2),
                             label=table.label_of(int(rid))))
        by_region = {}
        for r in rows:
            by_region[r["region"]] = by_region.get(r["region"], 0) + 1
        njunc = sum(1 for rid in by_region
                    if any(x["region"] == rid and x["label"] == "junction" for x in rows))
        print(f"  {fam}: assigned {len(rows)} to {len(by_region)} regions "
              f"({njunc} labelled junction); {unassigned} beyond {a.max_dist_m} m")
        for rid in sorted(by_region):
            lab = next(x["label"] for x in rows if x["region"] == rid)
            print(f"     region {rid:3} ({lab:8}) {by_region[rid]}")
        per_family[fam] = rows
        total += len(rows)

    out = a.out or os.path.join(HERE, "config", f"landmarks.{town}.json")
    blob = {
        "town": town,
        "_note": ("MAP-NATIVE landmarks: present in every world, never spawned. Read by "
                  "gt_cue_node as a REGION-SCOPED answer, because /carla/objects does not "
                  "publish poses for these actors (the log says '0 located' while the actor "
                  "list knows their ids), so proximity can never be computed from it. "
                  "Deliberately NOT named cones.*/props.*: those globs feed _region_family, "
                  "which names the families that can be SPAWNED, and the absent-family "
                  "short-circuit keys off it -- adding lights there would make a spawned "
                  "cone answer traffic_light=False all over again. One key per family; "
                  "gt_cue_node._load_native_landmarks reads every list-valued key, so a new "
                  "family needs no code change on the reading side."),
    }
    blob.update(per_family)
    with open(out, "w") as f:
        json.dump(blob, f, indent=1)
    print(f"wrote {out}  ({total} actors across {len(fams)} famil{'y' if len(fams)==1 else 'ies'})")
    return 0 if total else 3


if __name__ == "__main__":
    sys.exit(main())
