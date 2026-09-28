#!/usr/bin/env python3
"""Publish synthetic LiDAR ranges from CARLA's own geometry, with no sensor and no rendering.

WHY THIS EXISTS. The MPC has obstacle avoidance -- `_hits_body()` turns
`/processed_ranges` into body-frame hits, the EDT rejects colliding rollouts, and
`recovery` creeps and turns when every rollout is blocked -- but it needs an input. Phase A
runs `no_rendering_mode`, so without this node there are no ranges, `require_ranges` is
False and the EDT sees zero hits. Nothing rejects a rollout that drives into a wall.

The typical symptom is commanded full throttle with ~zero displacement: buildings are
Unreal level geometry. They appear nowhere in the `.xodr` (its only objects are road
furniture) and they are not actors, so neither the planner's map nor `/carla/actor_list`
knows they exist. Without ranges the vehicle is blind by configuration.

WHY THIS AND NOT A REAL LIDAR. `world.get_environment_objects()` is a MAP query, not a
render, so it works with `no_rendering_mode` on and costs nothing per tick after startup.
That preserves the property that makes Phase A viable -- runs are fast because nothing is
rendered. A real `sensor.lidar.ray_cast` is the Phase B answer and is strictly better
(occlusion, dynamic actors, noise); this exists so the PLANNING side can be validated
before the perception side, exactly as `gt_cue_node` stands in for a camera.

THE CONTRACT, read from `carla_mpc_ros_node._hits_body()` and matched exactly:
  * ``Float32MultiArray`` of N ranges, uniform over a full 2*pi
  * bin 0 faces +x (FORWARD) and bins advance COUNTER-CLOCKWISE, body frame, metres
  * a bin at >= 99% of the array max is treated as a miss, so "nothing here" is max_range
  * hits closer than 10 cm are dropped as artefacts

Getting that wrong is silent: the EDT would place obstacles at the wrong bearing and the
vehicle would swerve away from clear road.
"""
from __future__ import annotations

import math

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float32MultiArray

#: Environment-object classes that are solid to a car. Roads/sidewalks/vegetation are
#: deliberately absent: the road corridor already keeps the vehicle on the carriageway
#: (`regions_npz` + `road_half_width`), and treating a kerb as an obstacle would make every
#: junction impassable.
SOLID_LABELS = ("Buildings", "Walls", "Fences", "Poles", "GuardRail",
                "TrafficSigns", "TrafficLight", "Static")


class GtObstacleNode(Node):
    def __init__(self) -> None:
        super().__init__("gt_obstacle_node")
        self.declare_parameter("host", "localhost")
        self.declare_parameter("port", 2000)
        self.declare_parameter("odom_topic", "/carla/ego_vehicle/odometry")
        self.declare_parameter("n_bins", 360)
        self.declare_parameter("max_range_m", 30.0)
        #: Boxes are inflated by this before ray casting. The MPC's own `safety_radius`
        #: handles the vehicle's extent; this covers the gap between a box's footprint and
        #: the mesh that actually collides.
        self.declare_parameter("inflate_m", 0.5)
        self.declare_parameter("rate_hz", 10.0)

        self._n = int(self.get_parameter("n_bins").value)
        self._max = float(self.get_parameter("max_range_m").value)
        self._pose: tuple[float, float, float] | None = None
        #: (cx, cy, half_x, half_y, yaw) per solid object, in the PLANAR frame
        self._boxes: np.ndarray = np.empty((0, 5), dtype=np.float32)

        self._pub = self.create_publisher(
            Float32MultiArray, "/processed_ranges", qos_profile_sensor_data)
        self.create_subscription(Odometry, str(self.get_parameter("odom_topic").value),
                                 self._odom_cb, qos_profile_sensor_data)
        self._load_geometry()
        self.create_timer(1.0 / float(self.get_parameter("rate_hz").value), self._tick)

    # ------------------------------------------------------------------ #
    def _load_geometry(self) -> None:
        """Query CARLA ONCE for static solids. A map query, not a render."""
        try:
            import carla
        except Exception as exc:                                   # noqa: BLE001
            self.get_logger().error(
                f"no carla module ({exc}); publishing an empty scan, which is exactly the "
                f"blind configuration this node exists to remove")
            return
        try:
            client = carla.Client(str(self.get_parameter("host").value),
                                  int(self.get_parameter("port").value))
            client.set_timeout(20.0)
            world = client.get_world()
        except Exception as exc:                                   # noqa: BLE001
            self.get_logger().error(f"cannot reach CARLA: {exc}")
            return

        rows: list[tuple[float, float, float, float, float]] = []
        for name in SOLID_LABELS:
            label = getattr(carla.CityObjectLabel, name, None)
            if label is None:
                continue
            try:
                objs = world.get_environment_objects(label)
            except Exception:                                      # noqa: BLE001
                continue
            for o in objs:
                bb = o.bounding_box
                # CARLA is left-handed with +y SOUTH; the planar frame negates y. See
                # carla_gt_bridge/frames.py -- getting this wrong mirrors the whole town.
                rows.append((float(bb.location.x), -float(bb.location.y),
                             float(bb.extent.x), float(bb.extent.y),
                             -math.radians(float(bb.rotation.yaw))))
        self._boxes = np.asarray(rows, dtype=np.float32) if rows else np.empty((0, 5), np.float32)
        self.get_logger().info(
            f"loaded {len(self._boxes)} static solids from CARLA "
            f"({', '.join(SOLID_LABELS)}) — no sensor, no rendering")

    def _odom_cb(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self._pose = (float(p.x), float(p.y), yaw)

    # ------------------------------------------------------------------ #
    def _tick(self) -> None:
        ranges = np.full(self._n, self._max, dtype=np.float32)
        if self._pose is not None and len(self._boxes):
            ex, ey, eyaw = self._pose
            infl = float(self.get_parameter("inflate_m").value)
            # Only boxes whose centre is within max_range + their own diagonal can matter.
            d = np.hypot(self._boxes[:, 0] - ex, self._boxes[:, 1] - ey)
            reach = self._max + np.hypot(self._boxes[:, 2], self._boxes[:, 3]) + infl
            near = self._boxes[d <= reach]
            for cx, cy, hx, hy, byaw in near:
                # corners in world, then into body frame (bin 0 = +x forward, CCW)
                c, s = math.cos(byaw), math.sin(byaw)
                corners = np.array([[-hx - infl, -hy - infl], [hx + infl, -hy - infl],
                                    [hx + infl, hy + infl], [-hx - infl, hy + infl]],
                                   dtype=np.float32)
                wx = cx + corners[:, 0] * c - corners[:, 1] * s
                wy = cy + corners[:, 0] * s + corners[:, 1] * c
                dx, dy = wx - ex, wy - ey
                bx = dx * math.cos(-eyaw) - dy * math.sin(-eyaw)
                by = dx * math.sin(-eyaw) + dy * math.cos(-eyaw)
                ang = np.arctan2(by, bx) % (2 * math.pi)
                rad = np.hypot(bx, by)
                # A box spans the angular interval between its extreme corners. Filling
                # that whole interval (rather than 4 bins) is what makes a wall a WALL --
                # four isolated hits let a rollout thread between the corners.
                lo, hi = ang.min(), ang.max()
                if hi - lo > math.pi:      # wraps through 0; split into two arcs
                    idx = np.where((np.arange(self._n) / self._n * 2 * math.pi >= hi) |
                                   (np.arange(self._n) / self._n * 2 * math.pi <= lo))[0]
                else:
                    a = np.arange(self._n) / self._n * 2 * math.pi
                    idx = np.where((a >= lo) & (a <= hi))[0]
                if len(idx):
                    np.minimum.at(ranges, idx, float(rad.min()))
        ranges = np.clip(ranges, 0.0, self._max)
        m = Float32MultiArray()
        m.data = [float(v) for v in ranges]
        self._pub.publish(m)


def main(argv=None) -> None:
    rclpy.init(args=argv)
    node = GtObstacleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
