#!/usr/bin/env python3
"""Snap the derived spawn onto a real driving LANE. Runs inside the bridge container.

``spawn_config`` derives the pose from the start region's centroid, which is a point on
the road's *reference line*. On a two-way road that line is the centre — the lane divider
— so spawning there straddles both lanes. This asks the live map for the lane centre at
that point, and picks the lane whose heading actually goes toward the first junction.

This is the only script in the package that imports ``carla``, and it is a one-off: its
output is a static ``objects.<town>.json`` that the bridge reads. Nothing at run time
needs a CARLA client.

The seed is passed as arguments, with the Town05 values as defaults, so a second town does
not mean a second edited copy of this file — and so the seed cannot silently go stale
when the region map is regenerated.
The seed normally comes from `missions.spawn_seed` (run_phase_a.sh derives it):

  docker exec humble_dev_with_code bash -c \
    'source /carla_ws/install/setup.bash && python3 /ros_ws/src/carla_gt_bridge/scripts/lane_spawn.py \
       --town town01 --region 7 --x 366.65 --y 328.61 --yaw -174.2'
"""
from __future__ import annotations

import argparse
import json
import os

import carla

#: Town05 defaults — region 45's centroid in CARLA coordinates and the heading toward
#: junction 66, as `missions.spawn_config` derives them. See `carla_gt_bridge.frames`.
SPAWN_XY = (-172.55, -130.66)
WANT_YAW = -5.9
ROAD_ID = 45
OUT_DIR = "/ros_ws/src/carla_gt_bridge/config"


def lanes_at(base, limit: int = 8):
    """Every distinct lane across the road at this s.

    Bounded and de-duplicated: ``get_left_lane()`` can return a lane whose own left lane
    is the one you started from (opposing lanes reference each other), so the obvious
    while-loop never terminates.
    """
    seen, out = set(), []
    for step in ("left", "right"):
        w, n = base, 0
        while w is not None and n < limit:
            key = (w.road_id, w.lane_id)
            if key in seen:
                break
            seen.add(key)
            out.append(w)
            n += 1
            w = w.get_left_lane() if step == "left" else w.get_right_lane()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--town", default="town05")
    # A region id IS the OpenDRIVE road id here — map_regions segments by road — which is
    # what makes filtering candidate lanes by road_id meaningful.
    ap.add_argument("--region", type=int, default=ROAD_ID)
    ap.add_argument("--x", type=float, default=SPAWN_XY[0], help="seed x, CARLA frame")
    ap.add_argument("--y", type=float, default=SPAWN_XY[1], help="seed y, CARLA frame")
    ap.add_argument("--yaw", type=float, default=WANT_YAW, help="seed yaw, CARLA degrees")
    ap.add_argument("--role", default="ego_vehicle")
    ap.add_argument("--out-dir", default=OUT_DIR)
    a = ap.parse_args()
    road_id, want_yaw = a.region, a.yaw
    out = f"{a.out_dir}/objects.{a.town}.json"
    print(f"seed: {a.town} region {road_id} CARLA ({a.x:.2f}, {a.y:.2f}) yaw {want_yaw:.1f}")

    client = carla.Client("localhost", 2000)
    client.set_timeout(30.0)
    m = client.get_world().get_map()
    got = m.name.split("/")[-1].lower()
    if got != a.town.lower():
        raise SystemExit(
            f"the live map is {m.name!r} but --town is {a.town!r}. Snapping a Town01 seed "
            f"onto Town05 geometry would produce a plausible pose in the wrong town."
        )

    base = m.get_waypoint(carla.Location(x=a.x, y=a.y, z=0.5), project_to_road=True)
    print(f"reference-line projection: road {base.road_id} lane {base.lane_id} "
          f"junction={base.is_junction}")

    best = None
    for w in lanes_at(base):
        if w.road_id != road_id or str(w.lane_type) != "Driving":
            continue
        t = w.transform
        off = abs((t.rotation.yaw - want_yaw + 180.0) % 360.0 - 180.0)
        print(f"   lane {w.lane_id:+d}  ({t.location.x:8.2f}, {t.location.y:8.2f})  "
              f"yaw={t.rotation.yaw:7.1f}  off-by {off:5.1f} deg")
        if best is None or off < best[0]:
            best = (off, w)

    if best is None:
        raise SystemExit(f"no driving lane found on road {road_id}")
    off, w = best
    t = w.transform
    if off > 90.0:
        # Every lane points the wrong way: the corridor's direction of travel is not
        # drivable from here. Better to fail than to spawn a car facing backwards.
        raise SystemExit(f"best lane is {off:.0f} deg from the intended heading — "
                         f"the corridor may be one-way in the other direction")
    print(f"CHOSEN lane {w.lane_id:+d} at ({t.location.x:.2f}, {t.location.y:.2f}) "
          f"yaw={t.rotation.yaw:.1f}  ({off:.1f} deg off the region-to-region bearing; "
          f"the lane's own heading is the truth here, the bearing is a chord average)")

    # objects.json `spawn_point` is in ROS COORDINATES, not CARLA's.
    #
    # `carla_spawn_objects.create_spawn_point()` builds a geometry_msgs Pose and hands it
    # to the bridge's SpawnObject service, which converts ROS -> CARLA on the way in —
    # negating y and the yaw. Writing CARLA coordinates here negates them a SECOND time,
    # putting the vehicle at the mirror image of the intended spot. It fails as
    # "Spawn failed because of collision at spawn position", which reads like something is
    # parked there rather than like a frame error (`world.try_spawn_actor`, which takes
    # CARLA coordinates directly, spawns the same pose fine).
    #
    # ROS coordinates are the planar frame, so this is where the outbound conversion
    # belongs — the only one in the package.
    px, py = t.location.x, -t.location.y
    pyaw = (-t.rotation.yaw + 180.0) % 360.0 - 180.0
    print(f"   road surface z here is {t.location.z:.2f}; spawning 0.3 m above it")
    print(f"   CARLA  ({t.location.x:.2f}, {t.location.y:.2f}) yaw {t.rotation.yaw:.1f}")
    print(f"   ROS    ({px:.2f}, {py:.2f}) yaw {pyaw:.1f}   <- what objects.json needs")
    # Odometry is a PSEUDO-SENSOR: `/carla/<role>/odometry` only exists if
    # `sensor.pseudo.odom` is listed here. With `"sensors": []` the bridge happily creates
    # the EgoVehicle and its control topics, ticks the world, and publishes no pose at all
    # — which looks like a bridge fault rather than a missing declaration.
    #
    # Pseudo-sensors cost nothing: they read state the server already has. Phase A adds no
    # REAL sensor, so `no_rendering_mode` stays available. Phase B adds
    # `sensor.lidar.ray_cast` here; Phase C adds `sensor.camera.rgb`.
    payload = {"objects": [
        # actor_list is world-level, and is how gt_cue_node will answer "is a cone in the
        # junction" without any camera.
        {"type": "sensor.pseudo.actor_list", "id": "actor_list"},
        {"type": "sensor.pseudo.objects", "id": "objects"},
        {"type": "vehicle.tesla.model3", "id": a.role, "role_name": a.role,
         "spawn_point": {"x": round(px, 2), "y": round(py, 2),
                         "z": round(t.location.z + 0.3, 2),
                         "roll": 0.0, "pitch": 0.0, "yaw": round(pyaw, 1)},
         "sensors": [
             {"type": "sensor.pseudo.odom", "id": "odometry"},
             {"type": "sensor.pseudo.speedometer", "id": "speedometer"},
             {"type": "sensor.pseudo.tf", "id": "tf"},
             {"type": "actor.pseudo.control", "id": "control"},
             # REAL LIDAR, and it does NOT force rendering. `ray_cast` is a physics
             # raycast; only camera sensors need the renderer, so it still returns
             # points with no_rendering_mode True. (The "Phase A adds no REAL sensor"
             # note above refers to rendering sensors.) Without it the MPC has no
             # obstacle EDT, and "full throttle, no displacement" CONTROL failures follow.
             #
             # lower_fov is shallow to favour obstacles over ground returns; the
             # converter drops the ground anyway, and
             # a steeper scan just spends points on tarmac.
             # MOUNT HEIGHT AND FOV ARE OVERRIDABLE, and this is the ONLY place either
             # is defined. `objects.<town>.json` looks like the config but is REGENERATED
             # here every run, so editing it there changes nothing.
             #
             # LIDAR_Z exists because the deployment bags are recorded at 0.6 m
             # (`bev_pipeline/config.py: sensor_height`), and a 2.0 m roof mount is not
             # the robot's sensor -- CARLA LiDAR at 2.0 m describes different geometry
             # from the robot's.
             #
             # The "shallow to favour obstacles over ground returns" rationale is what
             # a low mount reverses: near the ground a kerb
             # OCCLUDES rather than protruding, so it reads as a range discontinuity
             # instead of a 0.15 m height difference lost in road camber.
             {"type": "sensor.lidar.ray_cast", "id": "lidar",
              "spawn_point": {"x": 0.0, "y": 0.0,
                              "z": float(os.environ.get("LIDAR_Z", 2.0)),
                              "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
              "range": 30.0, "channels": 32, "points_per_second": 100000,
              "rotation_frequency": 10, "upper_fov": 2.0,
              "lower_fov": float(os.environ.get("LIDAR_LOWER_FOV", -15.0))},
             # -- GT-COLLECTION SENSORS, off unless LIDAR_HEIGHTS is set ---------------
             #
             # MULTIPLE HEIGHTS IN ONE RUN, not one run per height. Same tick, same pose,
             # same world, so mount height is the ONLY variable -- a paired comparison
             # frame by frame. Separate runs would confound it with the trajectory:
             # runs with the same plan and seed diverge within tens of ticks, with
             # junction traversal stochastic on top.
             *[{"type": "sensor.lidar.ray_cast", "id": f"lidar_h{int(round(float(h)*100)):03d}",
                "spawn_point": {"x": 0.0, "y": 0.0, "z": float(h),
                                "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
                "range": 30.0, "channels": 32, "points_per_second": 100000,
                "rotation_frequency": 10, "upper_fov": 2.0,
                "lower_fov": float(os.environ.get("LIDAR_LOWER_FOV", -15.0))}
               for h in os.environ.get("LIDAR_HEIGHTS", "").split(",") if h.strip()],
             # THE ORACLE, and deliberately NOT height-matched to the candidates.
             # `frame_labels.structure_sides` answers "how many sides have structure",
             # which is a property of the PLACE. Matching its height to each candidate
             # would give every arm its own ground truth and the comparison would be each
             # arm grading its own homework. So it is mounted for maximum coverage --
             # high, wide FOV -- just as `gt_cluster_node` and `gt_cue_node` are
             # oracles rather than robot sensors.
             *([{"type": "sensor.lidar.ray_cast_semantic", "id": "semantic_lidar",
                 "spawn_point": {"x": 0.0, "y": 0.0, "z": 2.4,
                                 "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
                 "range": 50.0, "channels": 64, "points_per_second": 200000,
                 "rotation_frequency": 10, "upper_fov": 10.0, "lower_fov": -30.0}]
               if os.environ.get("SEMANTIC_LIDAR", "").lower() in ("1", "true", "yes")
               else []),
         ]},
    ]}
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
