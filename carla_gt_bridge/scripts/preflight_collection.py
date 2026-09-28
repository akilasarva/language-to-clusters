#!/usr/bin/env python3
"""The three questions a live CARLA server answers that no offline analysis can.

    ~/miniconda3/bin/python scripts/preflight_collection.py \
        --towns Town01 Town05 Town07 Town10HD

Run this BEFORE committing to collection drives. Everything else in the
collection plan was derived offline from the `.xodr`; these three cannot be,
and two of them can still change which towns get driven.

1. BUILDING DENSITY AND THE CARRIAGEWAY ENCLOSURE SPLIT.
   Buildings appear in NO town's `.xodr` — every `<object>` record across all eight
   maps is a crosswalk, a parking bay or `type="-1"`. They are Unreal-only, so
   `get_environment_objects(CityObjectLabel.Buildings)` against a live server is the
   only way to know a town's enclosure content before driving it. This measures it
   for every candidate town.

2. IS A RECORDED BAG'S TOWN01 STOCK?
   A bag such as `carla_data/straight_line_terrain` can match stock Town01 by road-id
   set, but road ids survive Unreal-side edits. Stock Town01 has ZERO `<bridge>`
   records, yet semantic LiDAR on such a bag can report Bridge- and Water-tagged points
   on a straight road. Either semantic LiDAR is simply richer than the `.xodr`, or the
   bag came off an edited map. Asking the live server for Bridge and Water objects near
   the driven pose settles it, and it decides whether "collect Town01 for buildings"
   means anything.

3. TICK COST WITH TWO LIDARS.
   Tick cost with rendering OFF, two ray-cast LiDARs and no camera decides whether a
   collection drive costs minutes or hours of wall clock.

Writes `reports/collection/preflight.json` and prints a summary. Read-only against the
world except for step 3, which spawns one vehicle and two sensors and destroys them.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)

import carla  # noqa: E402

#: Same threshold used for the Town05 split, and the same
#: `NEAR_STRUCTURE_M` the labeller uses, so the prediction and the label agree.
NEAR_M = 14.0


def collect_lidar_specs(path: str = "") -> dict:
    """{sensor type: spec} for every LiDAR in objects.collect.json.

    The collection config is the single source of truth for the rig. Anything that needs
    to know the channel count or the FOV reads it from here, so the number cannot drift
    between the spawner, the preflight and the pre-drive gate.
    """
    import json as _json
    path = path or os.path.join(PKG, "config", "objects.collect.json")
    with open(path) as f:
        d = _json.load(f)
    out = {}
    for obj in d["objects"]:
        for sen in obj.get("sensors", []):
            if "lidar" in sen["type"]:
                out[sen["type"]] = sen
    if not out:
        raise SystemExit(f"no LiDAR declared in {path}")
    span = {(s["lower_fov"], s["upper_fov"], s["channels"], s["range"],
             s["rotation_frequency"]) for s in out.values()}
    if len(span) != 1:
        raise SystemExit(f"the LiDARs in {path} do not match: {span} -- the semantic "
                         f"GT would describe different geometry than the scan")
    return out


def _box_distance(bb: "carla.BoundingBox", px: float, py: float) -> float:
    """Planar distance from a point to a ROTATED bounding box, in metres.

    Uses the box's own yaw rather than treating it as axis-aligned: CARLA's building
    boxes are rotated with the street grid, and an AABB approximation inflates a long
    thin block's footprint by its whole diagonal.
    """
    dx, dy = px - bb.location.x, py - bb.location.y
    yaw = math.radians(bb.rotation.yaw)
    c, s = math.cos(-yaw), math.sin(-yaw)
    lx, ly = c * dx - s * dy, s * dx + c * dy          # into the box's frame
    ox = max(abs(lx) - bb.extent.x, 0.0)
    oy = max(abs(ly) - bb.extent.y, 0.0)
    return math.hypot(ox, oy)


def enclosure_split(world, carla_map, buildings, *, step: float = 4.0,
                    near_m: float = NEAR_M, radius: float = 60.0) -> dict:
    """Fraction of the carriageway enclosed on both / one / neither side.

    For every driving waypoint, the gap to the nearest building on the LEFT and on the
    RIGHT of the direction of travel. That is the enclosure axis exactly as the
    labeller defines it, computed from ground truth instead of from LiDAR returns.
    """
    wps = carla_map.generate_waypoints(step)
    if not buildings:
        return {"waypoints": len(wps), "both": 0.0, "one": 0.0, "neither": 1.0}
    bpos = np.array([[b.bounding_box.location.x, b.bounding_box.location.y]
                     for b in buildings])
    both = one = neither = 0
    for w in wps:
        t = w.transform
        px, py = t.location.x, t.location.y
        yaw = math.radians(t.rotation.yaw)
        near = np.where(np.hypot(bpos[:, 0] - px, bpos[:, 1] - py) < radius)[0]
        gl = gr = float("inf")
        for i in near:
            b = buildings[int(i)].bounding_box
            d = _box_distance(b, px, py)
            if d > near_m:
                continue
            # which side of the direction of travel: CARLA is left-handed, so a
            # POSITIVE cross product here is the vehicle's RIGHT.
            vx, vy = b.location.x - px, b.location.y - py
            cross = math.cos(yaw) * vy - math.sin(yaw) * vx
            if cross >= 0:
                gr = min(gr, d)
            else:
                gl = min(gl, d)
        if gl <= near_m and gr <= near_m:
            both += 1
        elif gl <= near_m or gr <= near_m:
            one += 1
        else:
            neither += 1
    n = max(len(wps), 1)
    return {"waypoints": len(wps), "both": both / n, "one": one / n,
            "neither": neither / n}


def survey_town(client, town: str, *, step: float, timeout: float) -> dict:
    t0 = time.time()
    world = client.load_world(town)
    world.wait_for_tick()
    cmap = world.get_map()
    out: dict = {"town": town, "load_s": round(time.time() - t0, 1),
                 "map_name": cmap.name}

    for label in ("Buildings", "Bridge", "Water", "Walls", "Fences",
                  "GuardRail", "Vegetation"):
        try:
            objs = world.get_environment_objects(
                getattr(carla.CityObjectLabel, label))
        except Exception:
            objs = []
        out[f"n_{label.lower()}"] = len(objs)
        if label == "Buildings":
            buildings = objs

    t1 = time.time()
    out["enclosure"] = enclosure_split(world, cmap, buildings, step=step)
    out["enclosure_s"] = round(time.time() - t1, 1)
    return out


def check_edited_town01(client, xy=(393.7, -127.2), radius=250.0) -> dict:
    """Does STOCK Town01 have the bridge and water a recorded bag's LiDAR reported?

    `xy` is the bag's first odometry pose, in the PLANAR frame; the CARLA API is
    left-handed, so y is negated on the way in.
    """
    world = client.load_world("Town01")
    world.wait_for_tick()
    px, py = xy[0], -xy[1]
    out = {"probe_planar": list(xy), "radius_m": radius}
    for label in ("Bridge", "Water", "Buildings"):
        try:
            objs = world.get_environment_objects(
                getattr(carla.CityObjectLabel, label))
        except Exception:
            objs = []
        near = [o for o in objs
                if math.hypot(o.bounding_box.location.x - px,
                              o.bounding_box.location.y - py) < radius]
        out[f"{label.lower()}_total"] = len(objs)
        out[f"{label.lower()}_near_probe"] = len(near)
    return out


def tick_cost(client, town: str, n_ticks: int = 120) -> dict:
    """ms per tick with the collection sensor rig attached, rendering off."""
    world = client.load_world(town)
    settings = world.get_settings()
    old = (settings.synchronous_mode, settings.fixed_delta_seconds,
           settings.no_rendering_mode)
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.1
    settings.no_rendering_mode = True
    world.apply_settings(settings)

    bp = world.get_blueprint_library()
    sp = world.get_map().get_spawn_points()[0]
    veh = world.try_spawn_actor(bp.filter("vehicle.tesla.model3")[0], sp)
    sensors = []
    try:
        if veh is None:
            return {"error": "could not spawn ego"}
        # READ THE RIG FROM objects.collect.json, never restate it: a local copy of
        # channels and fov would let the tick-cost measurement describe a sensor that
        # is not the one spawned.
        specs = collect_lidar_specs()
        for kind, spec in specs.items():
            b = bp.find(kind)
            for attr in ("range", "channels", "points_per_second",
                         "rotation_frequency", "upper_fov", "lower_fov"):
                b.set_attribute(attr, str(spec[attr]))
            sp_ = spec["spawn_point"]
            tf = carla.Transform(carla.Location(x=sp_["x"], y=sp_["y"], z=sp_["z"]))
            s = world.spawn_actor(b, tf, attach_to=veh)
            counts = {"n": 0, "pts": 0}
            s.listen(lambda d, c=counts: (c.__setitem__("n", c["n"] + 1),
                                          c.__setitem__("pts", len(d))))
            sensors.append((kind, s, counts))

        for _ in range(10):                      # warm up
            world.tick()
        t0 = time.time()
        for _ in range(n_ticks):
            world.tick()
        dt = time.time() - t0
        ms = 1000.0 * dt / n_ticks
        return {"ms_per_tick": round(ms, 2),
                "realtime_factor": round(0.1 / (dt / n_ticks), 2),
                "frames": {k: c["n"] for k, _s, c in sensors},
                "points_last": {k: c["pts"] for k, _s, c in sensors}}
    finally:
        for _k, s, _c in sensors:
            try:
                s.stop()
                s.destroy()
            except Exception:
                pass
        if veh is not None:
            veh.destroy()
        settings.synchronous_mode, settings.fixed_delta_seconds, \
            settings.no_rendering_mode = old
        world.apply_settings(settings)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--towns", nargs="*",
                    default=["Town01", "Town05", "Town07", "Town10HD"])
    ap.add_argument("--step", type=float, default=4.0,
                    help="carriageway sampling for the enclosure split")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--out", default=os.path.join(PKG, "reports", "collection",
                                                  "preflight.json"))
    ap.add_argument("--skip-tick-cost", action="store_true")
    ap.add_argument("--skip-probe", action="store_true")
    ap.add_argument("--tick-town", default="",
                    help="town to measure tick cost on; default the first surveyed")
    ap.add_argument("--merge", nargs="*", default=None,
                    help="merge these per-town json files into --out and stop")
    a = ap.parse_args(argv)

    if a.merge is not None:
        merged: dict = {"towns": []}
        for f in a.merge:
            if not os.path.exists(f):
                continue
            d = json.load(open(f))
            merged["towns"].extend(d.get("towns", []))
            for k in ("server", "town01_probe", "tick_cost"):
                if k in d and k not in merged:
                    merged[k] = d[k]
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump(merged, fh, indent=2)
        print(f"merged {len(merged['towns'])} towns -> {a.out}")
        return 0

    client = carla.Client(a.host, a.port)
    client.set_timeout(a.timeout)
    print(f"CARLA {client.get_server_version()} at {a.host}:{a.port}\n")

    report: dict = {"server": client.get_server_version(), "towns": []}

    print(f"{'town':10s} {'build':>6s} {'bridge':>6s} {'water':>6s} {'walls':>6s} "
          f"{'fences':>6s} | {'both':>6s} {'one':>6s} {'neither':>7s}  (carriageway)")
    for town in a.towns:
        try:
            r = survey_town(client, town, step=a.step, timeout=a.timeout)
        except Exception as e:
            print(f"{town:10s} ERROR {e}")
            continue
        e = r["enclosure"]
        report["towns"].append(r)
        print(f"{town:10s} {r['n_buildings']:6d} {r['n_bridge']:6d} "
              f"{r['n_water']:6d} {r['n_walls']:6d} {r['n_fences']:6d} | "
              f"{100 * e['both']:5.1f}% {100 * e['one']:5.1f}% "
              f"{100 * e['neither']:6.1f}%   ({e['waypoints']} wp)")

    if not a.skip_probe:
      print("\nIs stock Town01 the map the 2026 bag came off?")
      try:
        report["town01_probe"] = check_edited_town01(client)
        p = report["town01_probe"]
        print(f"  bridge objects: {p['bridge_total']} total, "
              f"{p['bridge_near_probe']} within {p['radius_m']:.0f} m of the bag's "
              f"first pose")
        print(f"  water  objects: {p['water_total']} total, "
              f"{p['water_near_probe']} near")
        if p["bridge_near_probe"] == 0 and p["water_near_probe"] == 0:
            print("  -> stock Town01 has NEITHER near that pose. The bag's 28,742 "
                  "Bridge and\n     10,563 Water points cannot have come from this "
                  "map: it is an EDITED Town01.")
        else:
            print("  -> stock Town01 does carry them; semantic LiDAR is simply "
                  "richer than the .xodr.")
      except Exception as e:
        print(f"  ERROR {e}")

    if not a.skip_tick_cost:
        print("\nTick cost with the collection rig (2 LiDARs, no rendering):")
        try:
            tt = a.tick_town or (a.towns[0] if a.towns else "Town01")
            report["tick_cost"] = tick_cost(client, tt)
            t = report["tick_cost"]
            if "error" in t:
                print(f"  {t['error']}")
            else:
                print(f"  {t['ms_per_tick']} ms/tick -> {t['realtime_factor']}x real "
                      f"time at 10 Hz")
                print(f"  sensor frames: {t['frames']}")
        except Exception as e:
            print(f"  ERROR {e}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
