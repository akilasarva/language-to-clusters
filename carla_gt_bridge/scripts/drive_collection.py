#!/usr/bin/env python3
"""Drive the planned collection route with a BasicAgent, so a bag can record it.

    python3 scripts/drive_collection.py --route reports/collection/drivable.town01.json

Runs INSIDE the bridge container (it needs the 0.9.14 client and `/carla/agents`); the
host's conda client is 0.10.0 and `load_world` aborts the process on a version mismatch.

IT ATTACHES TO THE EGO, IT DOES NOT SPAWN ONE
---------------------------------------------
`carla_spawn_objects` has already created the vehicle and both LiDARs from
`objects.collect.json`, and those sensors are what publish the topics being recorded.
Spawning a second vehicle here would drive a car that no topic describes, and the bag
would contain a stationary ego's scans while a ghost toured the town. The ego is found
by `role_name`, which is the same key the bridge names its topics after.

WHY A PER-LEG TIMEOUT AND A STUCK DETECTOR
-------------------------------------------
A collection drive visits dozens of targets and any one of them can wedge the car against
a building that appears in no `.xodr`. Without a per-leg cap, one wedge silently eats the
whole run and the bag ends up being ten minutes of a stationary vehicle -- which still
records thousands of frames and still labels cleanly, so nothing downstream would report
a problem. A skipped leg costs one region's coverage and says so.

WHAT IT DOES NOT DO
-------------------
No traffic, no other actors, no rendering. The corpus is about static geometry; a
pedestrian wandering into a scan is noise in the enclosure label, not signal.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, "/carla")
sys.path.insert(0, os.path.expanduser("~/carla"))


def _imports():
    """Imported lazily so --help and argument errors work outside the container."""
    import carla
    try:
        from agents.navigation.basic_agent import BasicAgent
        from agents.navigation.global_route_planner import GlobalRoutePlanner
    except ImportError as e:                               # pragma: no cover
        raise SystemExit(
            f"cannot import the CARLA agents ({e}). This script runs inside the bridge "
            f"container with the CARLA PythonAPI mounted at /carla; BasicAgent also "
            f"needs shapely and networkx, which run_collection.sh installs.")
    return carla, BasicAgent, GlobalRoutePlanner


def find_ego(world, role: str, timeout_s: float = 30.0):
    """The vehicle `carla_spawn_objects` created, by role_name."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        for a in world.get_actors().filter("vehicle.*"):
            if a.attributes.get("role_name") == role:
                return a
        time.sleep(1.0)
    return None


def drive(args) -> int:
    carla, BasicAgent, GlobalRoutePlanner = _imports()
    route = json.load(open(args.route))
    targets = route["targets"]
    if args.max_targets:
        targets = targets[:args.max_targets]
    print(f"route: {args.route}")
    print(f"  {len(targets)} targets, "
          f"{route['stats'].get('length_m', 0) / 1000:.2f} km planned, "
          f"town {route['stats'].get('town', '?')}")

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    print(f"  server {client.get_server_version()}, map {world.get_map().name}")

    ego = find_ego(world, args.role)
    if ego is None:
        print(f"no vehicle with role_name={args.role!r}. carla_spawn_objects has not "
              f"run, or it failed — check /tmp/spawn.log in the container.")
        return 1
    print(f"  ego {ego.type_id} id={ego.id} at "
          f"({ego.get_location().x:.1f}, {ego.get_location().y:.1f})")

    sensors = [a for a in world.get_actors().filter("sensor.*")
               if a.parent is not None and a.parent.id == ego.id]
    kinds = sorted(s.type_id for s in sensors)
    print(f"  sensors attached: {kinds}")
    # Guard: no sensor means no topic and no bag content.
    # Fail here rather than after a ten-minute drive.
    if not any("ray_cast_semantic" in k for k in kinds):
        print("  REFUSING: no semantic LiDAR attached, so the enclosure axis would have "
              "no ground truth and the bag would be unlabelable for it. Check that "
              "objects.collect.json is the spawn file in use.")
        if not args.allow_missing_sensors:
            return 2
    if not any(k.endswith("lidar.ray_cast") for k in kinds):
        print("  REFUSING: no ray_cast LiDAR — nothing to label.")
        if not args.allow_missing_sensors:
            return 2

    cmap = world.get_map()

    # ONE CONTINUOUS PLAN, NOT SEPARATE DESTINATIONS.
    # Driving target-to-target re-plans from wherever the vehicle actually IS each time,
    # and that is not where the offline route assumed it would be: overshooting a
    # centroid onto the far side of a junction can leave the next leg needing a U-turn
    # the car cannot make, so it sits and burns every remaining target on the stuck
    # timer. `plan_drivable_route.py` already resolved this target list to a
    # continuous, drivable polyline through GlobalRoutePlanner, so hand the agent that
    # whole polyline and let its local planner follow it.
    plan = []
    if route.get("path"):
        # The planner's OWN polyline, in order. `plan_drivable_route.py` produced it by
        # tracing from the real end pose of each leg; re-deriving it here from the
        # target centroids can come out several times longer, because a centroid snaps
        # to whichever lane is nearest and the plan loops the block to reach it.
        from agents.navigation.local_planner import RoadOption
        opts = {o.name: o for o in RoadOption}
        pathpts = route["path"]
        if args.max_targets:
            # keep the prefix of the path that covers the first N targets
            keep = targets[-1]
            end = carla.Location(x=keep["carla"][0], y=keep["carla"][1], z=0.0)
            best = min(range(len(pathpts)),
                       key=lambda i: (pathpts[i][0] - end.x) ** 2
                       + (pathpts[i][1] - end.y) ** 2)
            pathpts = pathpts[:best + 1]
        for x, y, o in pathpts:
            wp = cmap.get_waypoint(carla.Location(x=x, y=y, z=0.0),
                                   project_to_road=True)
            if wp is not None:
                plan.append((wp, opts.get(o, RoadOption.LANEFOLLOW)))
        print(f"  following the planner's own path: {len(plan)} waypoints")
    else:
        print("  route has no 'path' — falling back to re-tracing the targets, which "
              "inflates the route; regenerate with plan_drivable_route.py")
        grp = GlobalRoutePlanner(cmap, 2.0)
        for a_t, b_t in zip(targets, targets[1:]):
            la = carla.Location(x=a_t["carla"][0], y=a_t["carla"][1], z=0.0)
            lb = carla.Location(x=b_t["carla"][0], y=b_t["carla"][1], z=0.0)
            try:
                leg = grp.trace_route(la, lb)
            except Exception:
                leg = []
            plan.extend(leg if not plan else leg[1:])

    if not plan:
        print("  could not trace a route through the targets")
        return 3
    # Length of the plan ACTUALLY traced, not route["stats"]["length_m"]: with
    # --max-targets the two differ by design, and comparing against the full-route
    # figure would warn on a healthy truncated run.
    planned_km = sum(
        plan[i][0].transform.location.distance(plan[i + 1][0].transform.location)
        for i in range(len(plan) - 1)) / 1000.0
    print(f"  global plan: {len(plan)} waypoints over {len(targets)} targets, "
          f"{planned_km:.2f} km")

    # Put the ego ON the route's first waypoint. The spawn pose comes from
    # objects.collect.json (deliberately unset, so the bridge picks one) and is not
    # where the route starts; driving there first would add an unplanned, unlabelled
    # leg. A single teleport before recording matters is the same thing dgppo does.
    w0 = plan[0][0]
    start = carla.Transform(
        carla.Location(x=w0.transform.location.x, y=w0.transform.location.y,
                       z=w0.transform.location.z + 0.3),
        w0.transform.rotation)
    ego.set_transform(start)
    ego.set_target_velocity(carla.Vector3D(0, 0, 0))
    time.sleep(1.0)
    print(f"  teleported to route start ({start.location.x:.1f}, "
          f"{start.location.y:.1f})")

    agent = BasicAgent(ego, target_speed=args.speed_kmh)
    try:
        agent.ignore_traffic_lights(True)
        agent.ignore_stop_signs(True)
        agent.ignore_vehicles(True)
    except AttributeError:
        pass                                   # older agent API; defaults are fine
    agent.set_global_plan(plan, stop_waypoint_creation=True, clean_queue=True)

    t_run = time.time()
    dist_total = 0.0
    last = ego.get_location()
    stuck_since, stuck_ref = time.time(), ego.get_location()
    recoveries, teleports = 0, 0
    tele_log = []
    next_report = 0.0

    while not agent.done():
        if args.run_seconds and time.time() - t_run > args.run_seconds:
            print("  run-seconds cap reached")
            break
        ego.apply_control(agent.run_step())
        time.sleep(1.0 / args.control_hz)

        loc = ego.get_location()
        step = math.hypot(loc.x - last.x, loc.y - last.y)
        if step < 50.0:
            dist_total += step
        last = loc

        if math.hypot(loc.x - stuck_ref.x, loc.y - stuck_ref.y) > args.stuck_m:
            stuck_since, stuck_ref = time.time(), loc
        elif time.time() - stuck_since > args.stuck_seconds:
            # First try reversing: a nose against a kerb frees with 1.5 s of reverse,
            # and that keeps the trajectory continuous, which the `past_`/`around_`
            # relations depend on.
            recoveries += 1
            print(f"  stuck at ({loc.x:.1f}, {loc.y:.1f}) — reversing "
                  f"[recovery {recoveries}]", flush=True)
            for _ in range(int(1.5 * args.control_hz)):
                ego.apply_control(carla.VehicleControl(throttle=0.5, reverse=True,
                                                       steer=0.0))
                time.sleep(1.0 / args.control_hz)
            ego.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
            stuck_since, stuck_ref = time.time(), ego.get_location()

            if recoveries % args.teleport_after == 0:
                # Last resort: jump to the nearest plan waypoint still AHEAD of us and
                # carry on. A teleport breaks the trajectory, so it is timestamped into
                # a sidecar and `label_frames.py` resets its relation window there --
                # otherwise `past_building` would fire on a building the vehicle never
                # drove past.
                idx = min(range(len(plan)),
                          key=lambda i: plan[i][0].transform.location.distance(loc))
                jump = plan[min(idx + 10, len(plan) - 1)][0].transform
                ego.set_transform(carla.Transform(
                    carla.Location(jump.location.x, jump.location.y,
                                   jump.location.z + 0.3), jump.rotation))
                ego.set_target_velocity(carla.Vector3D(0, 0, 0))
                # RE-SYNC THE PLAN. Without this the agent still holds the waypoint
                # queue from before the jump and immediately tries to drive BACK to it,
                # gets stuck again, teleports forward again, and oscillates between
                # two positions.
                rest = plan[min(idx + 10, len(plan) - 1):]
                if len(rest) > 1:
                    agent.set_global_plan(rest, stop_waypoint_creation=True,
                                          clean_queue=True)
                teleports += 1
                tele_log.append({"wall_time": time.time(),
                                 "carla_xy": [jump.location.x, jump.location.y]})
                time.sleep(0.5)
                stuck_since, stuck_ref = time.time(), ego.get_location()
                print(f"  teleported past the obstruction [teleport {teleports}]",
                      flush=True)

        el = time.time() - t_run
        if el > next_report:
            next_report = el + 15.0
            print(f"  t={el:5.0f}s  driven={dist_total / 1000:5.2f} km  "
                  f"recoveries={recoveries} teleports={teleports}", flush=True)

    reached = 1 if agent.done() else 0
    ego.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
    el = time.time() - t_run
    print(f"\ndone: plan {'COMPLETED' if reached else 'NOT completed'}")
    print(f"      {dist_total / 1000:.2f} km driven of {planned_km:.2f} planned, "
          f"in {el / 60:.1f} min ({dist_total / max(el, 1):.1f} m/s average)")
    print(f"      {recoveries} reverse recoveries, {teleports} teleports")
    if args.teleport_log and tele_log:
        with open(args.teleport_log, "w") as f:
            json.dump({"teleports": tele_log}, f, indent=2)
        print(f"      teleport times -> {args.teleport_log}")
    if planned_km and dist_total / 1000.0 < 0.75 * planned_km:
        print(f"      WARNING: drove only "
              f"{100 * dist_total / 1000 / planned_km:.0f}% of the planned distance, so "
              f"region coverage is well under the planned 100%. Check the corpus "
              f"composition before training on this bag.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--route", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--role", default="ego_vehicle")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--speed-kmh", type=float, default=28.8,
                    help="8 m/s, the speed the route's frame counts assume")
    ap.add_argument("--control-hz", type=float, default=20.0)
    ap.add_argument("--leg-timeout", type=float, default=90.0)
    ap.add_argument("--stuck-seconds", type=float, default=15.0)
    ap.add_argument("--stuck-m", type=float, default=2.0)
    ap.add_argument("--run-seconds", type=float, default=0.0,
                    help="hard cap on the whole drive; 0 = no cap")
    ap.add_argument("--max-targets", type=int, default=0)
    ap.add_argument("--teleport-after", type=int, default=2,
                    help="teleport past an obstruction after this many failed reverses")
    ap.add_argument("--teleport-log", default="",
                    help="write teleport timestamps here; label_frames resets its "
                         "relation window across them")
    ap.add_argument("--allow-missing-sensors", action="store_true",
                    help="drive anyway with an incomplete sensor rig (for smoke tests)")
    return drive(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
