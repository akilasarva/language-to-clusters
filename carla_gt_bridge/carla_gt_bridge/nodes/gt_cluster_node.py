"""Publish the GROUND-TRUTH cluster id on the topic the real classifier uses.

    /carla/<role>/odometry  (nav_msgs/Odometry)  ->  /predicted_cluster  (std_msgs/Int16)

Substituting perception here is what makes the rest of the chain testable. Everything
downstream — brain's acceptance sets, the MPC's target region, plan advancement — is
identical whether the id came from this node or from the LiDAR classifier, because the
contract is one integer on one topic. So a mission that fails with ground-truth
clusters has a bug in the plan or the controller, not in perception, and that
distinction is otherwise very expensive to make inside CARLA.

Also publishes ``/predicted_state`` (Float32MultiArray, one-hot over the table's region
ids) so consumers written against the real classifier's confidence vector work
unchanged — the ground truth is simply confident.

Zero CARLA imports: the pose arrives over ROS from the bridge's odometry topic.

Frame
-----
The bridge publishes odometry in the same frame ``regions.<town>.npz`` uses, so the
lookup needs no conversion. Orientation is not used at all. See :mod:`carla_gt_bridge.frames` for why
frame handling is centralised rather than done inline.
"""

from __future__ import annotations

import os

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Float32MultiArray, Int16, String

from ..cluster_space import REGION, SPACE_TOPIC, space
from ..region_lookup import DEFAULT_MAX_DISTANCE_M, load_region_table


class GtClusterNode(Node):
    """Nearest-waypoint region lookup, published as if it were a classifier."""

    def __init__(self) -> None:
        super().__init__("gt_cluster_node")

        self.declare_parameter("regions_npz", "")
        self.declare_parameter("odom_topic", "/carla/ego_vehicle/odometry")
        self.declare_parameter("cluster_topic", "/predicted_cluster")
        self.declare_parameter("state_topic", "/predicted_state")
        self.declare_parameter("max_distance_m", DEFAULT_MAX_DISTANCE_M)
        # Republish at a fixed rate rather than only on change: brain and the MPC both
        # poll the latest value, and a node that starts after the vehicle stops moving
        # would otherwise never see a cluster at all.
        self.declare_parameter("publish_hz", 10.0)

        npz = self.get_parameter("regions_npz").get_parameter_value().string_value
        if not npz:
            npz = self._default_npz()
        if not npz or not os.path.exists(npz):
            raise SystemExit(
                f"gt_cluster_node: regions table not found ({npz!r}). Generate one "
                f"with:\n  python3 scripts/map_regions.py --xodr config/Town05.xodr "
                f"--corridor 45 66 4 53 5 8\nthen pass -p regions_npz:=<path>.")
        self._table = load_region_table(npz)
        self._max_d = self.get_parameter("max_distance_m") \
                          .get_parameter_value().double_value

        self._last_rid: int | None = None
        self._pose: tuple[float, float] | None = None
        self._off_network_warned = False

        self._cluster_pub = self.create_publisher(
            Int16, self.get_parameter("cluster_topic")
                       .get_parameter_value().string_value, 10)
        self._state_pub = self.create_publisher(
            Float32MultiArray, self.get_parameter("state_topic")
                                   .get_parameter_value().string_value, 10)

        # ANNOUNCE THE ID SPACE, latched. Five nodes publish on /predicted_cluster and the
        # integer means a region id here, an HDBSCAN cluster id in `clustering/`, and a
        # label index in `bev_pipeline/`. They are indistinguishable on the wire, so a
        # consumer holding the wrong cluster_map resolves every id to the wrong mode and
        # drives somewhere plausible for the wrong reason. See carla_gt_bridge.cluster_space.
        self._space = space(REGION, self._table.town, self._table.region_ids)
        self._space_pub = self.create_publisher(
            String, SPACE_TOPIC,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self._space_pub.publish(String(data=self._space))
        self.create_subscription(
            Odometry, self.get_parameter("odom_topic")
                          .get_parameter_value().string_value, self._odom_cb, 10)

        hz = max(1.0, self.get_parameter("publish_hz")
                          .get_parameter_value().double_value)
        self.create_timer(1.0 / hz, self._tick)

        self.get_logger().info(
            f"gt_cluster_node: {len(self._table.region_ids)} regions from "
            f"{os.path.basename(npz)} ({self._table.town}), ids="
            f"{self._table.region_ids}, publishing at {hz:.0f} Hz, "
            f"id space {self._space}")

    # -- helpers ---------------------------------------------------------- #

    def _default_npz(self) -> str:
        """Fall back to the installed share/ copy, then the source tree."""
        candidates = []
        try:
            from ament_index_python.packages import get_package_share_directory
            share = get_package_share_directory("carla_gt_bridge")
            candidates.append(os.path.join(share, "config", "regions.town05.npz"))
        except Exception:                          # not built / not sourced
            pass
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates.append(os.path.join(os.path.dirname(here), "config",
                                       "regions.town05.npz"))
        for c in candidates:
            if os.path.exists(c):
                return c
        return ""

    # -- callbacks -------------------------------------------------------- #

    def _odom_cb(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        self._pose = (float(p.x), float(p.y))

    def _tick(self) -> None:
        if self._pose is None:
            self.get_logger().warning("waiting for odometry",
                                      throttle_duration_sec=5.0)
            return
        x, y = self._pose
        rid, dist = self._table.nearest(x, y)

        if dist > self._max_d:
            # Off-network. Publish nothing: an id here would be a guess, and a wrong
            # id is worse than no id because brain would act on it.
            if not self._off_network_warned:
                self.get_logger().error(
                    f"pose ({x:.1f}, {y:.1f}) is {dist:.1f} m from the nearest "
                    f"waypoint (limit {self._max_d:.1f} m) — wrong town, wrong frame, "
                    f"or the corridor does not cover this pose. Publishing nothing.")
                self._off_network_warned = True
            return
        self._off_network_warned = False

        if rid != self._last_rid:
            self.get_logger().info(
                f"cluster {self._last_rid} -> {rid} ({self._table.label_of(rid)}) "
                f"at ({x:.1f}, {y:.1f}), {dist:.2f} m from reference line")
            self._last_rid = rid

        self._cluster_pub.publish(Int16(data=int(rid)))

        one_hot = [0.0] * len(self._table.region_ids)
        one_hot[self._table.region_ids.index(rid)] = 1.0
        self._state_pub.publish(Float32MultiArray(data=one_hot))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GtClusterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
