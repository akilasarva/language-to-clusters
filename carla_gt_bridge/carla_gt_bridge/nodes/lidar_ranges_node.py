#!/usr/bin/env python3
"""CARLA LiDAR point cloud -> the 360-bin scan the MPC's obstacle EDT consumes.

WHY THIS EXISTS. The MPC has obstacle avoidance -- `_hits_body()` turns
`/processed_ranges` into body-frame hits, the EDT rejects colliding rollouts, and
`recovery` creeps and turns when every rollout is blocked -- but it needs an input. With
`no_rendering_mode` on there are no CAMERAS, but `sensor.lidar.ray_cast` is a physics
raycast and still works with rendering off.

Without ranges the vehicle drives into buildings (full throttle, near-zero displacement),
because buildings appear in neither the `.xodr` nor `/carla/actor_list`.

THE OUTPUT CONTRACT, read from `carla_mpc_ros_node._hits_body()` and matched exactly:
  * ``Float32MultiArray`` of N ranges, uniform over a full 2*pi
  * bin 0 faces +x (FORWARD), bins advance COUNTER-CLOCKWISE, body frame, metres
  * an EMPTY bin must be max_range, because a bin at >= 99% of the array max is discarded
    as a miss -- writing 0.0 for "nothing here" would put an obstacle on the bumper
  * hits closer than 10 cm are dropped by the consumer as artefacts

Getting the bearing convention wrong is silent: the EDT would place obstacles at the wrong
angle and the vehicle would swerve away from clear road.
"""
from __future__ import annotations

import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float32MultiArray


class LidarRangesNode(Node):
    def __init__(self) -> None:
        super().__init__("lidar_ranges_node")
        self.declare_parameter("cloud_topic", "/carla/ego_vehicle/lidar")
        self.declare_parameter("n_bins", 360)
        self.declare_parameter("max_range_m", 30.0)
        #: Points below this (sensor frame, sensor sits ~2 m up) are the road. Dropping
        #: them matters more than it sounds: the ground is the overwhelming majority of
        #: returns, and a scan of tarmac would fill every bin at close range and stop the
        #: vehicle dead while reporting a healthy EDT.
        # The sensor sits 2 m up, so a lower cut (e.g. -1.4, keeping everything above
        # 0.6 m) lets a junction's kerbs and road camber fill the scan, blocking the
        # EDT and preventing a correctly decided turn. -1.0 keeps only obstacles at
        # least 1 m tall, which is what actually stops a car.
        self.declare_parameter("ground_z", -1.0)
        #: Points above this are gantries, wires and overhanging foliage the car drives
        #: under. Treating them as obstacles makes bridges impassable, and bridges are
        #: the typical `passage` region.
        self.declare_parameter("ceiling_z", 1.5)
        self.declare_parameter("min_range_m", 0.3)
        #: Inflation is the OTHER lever on the same problem; the MPC already applies
        #: its own `safety_radius` (1.5 m) and `d_safe` (1.5 m), so inflating here too
        #: double-counts clearance. Left at 0 and tuned only if a real collision shows
        #: up -- an obstacle you cannot pass is worse than one you clip.
        self.declare_parameter("inflate_m", 0.0)

        self._n = int(self.get_parameter("n_bins").value)
        self._max = float(self.get_parameter("max_range_m").value)
        self._pub = self.create_publisher(
            Float32MultiArray, "/processed_ranges", qos_profile_sensor_data)
        self.create_subscription(
            PointCloud2, str(self.get_parameter("cloud_topic").value),
            self._cloud_cb, qos_profile_sensor_data)
        self._logged = False

    def _cloud_cb(self, msg: PointCloud2) -> None:
        pts = np.array(
            [(x, y, z) for x, y, z in point_cloud2.read_points(
                msg, field_names=("x", "y", "z"), skip_nans=True)],
            dtype=np.float32)
        ranges = np.full(self._n, self._max, dtype=np.float32)
        if len(pts):
            gz = float(self.get_parameter("ground_z").value)
            cz = float(self.get_parameter("ceiling_z").value)
            mn = float(self.get_parameter("min_range_m").value)
            keep = (pts[:, 2] > gz) & (pts[:, 2] < cz)
            p = pts[keep]
            if len(p):
                # CARLA's lidar frame is left-handed (+y right); the planar frame the MPC
                # rolls out in has +y LEFT. Negating y here is the same correction
                # frames.py applies to every other CARLA quantity -- without it the scan
                # is mirrored and the car steers into what it is trying to avoid.
                r = np.hypot(p[:, 0], p[:, 1])
                ok = (r >= mn) & (r <= self._max)
                r, px, py = r[ok], p[ok, 0], -p[ok, 1]
                if len(r):
                    ang = np.arctan2(py, px) % (2.0 * math.pi)
                    idx = (ang / (2.0 * math.pi) * self._n).astype(np.int32) % self._n
                    np.minimum.at(ranges, idx, r)
        if not self._logged:
            self.get_logger().info(
                f"lidar -> ranges: {len(pts)} points, "
                f"{int(np.sum(ranges < self._max * 0.99))} occupied bins of {self._n}")
            self._logged = True
        m = Float32MultiArray()
        m.data = [float(v) for v in ranges]
        self._pub.publish(m)


def main(argv=None) -> None:
    rclpy.init(args=argv)
    node = LidarRangesNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
