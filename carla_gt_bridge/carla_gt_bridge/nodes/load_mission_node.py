"""Publish a mission brain tree, tell brain to load it, and keep holding the latch.

    plan_file (a brain tree JSON)  ->  /brain/incoming_plan (latched String)
                         ->  /brain/load_plan (std_srvs/Trigger)

This is what ``nl_planner``'s ``executor_node`` does after generating a plan. Having it
as a separate node decouples driving from the ``pydantic_ai`` install: a tree
materialised by ``nl_planner.branch_materializer`` (e.g. by drive_english.py) is
byte-identical input for brain and the MPC either way.

Latched (TRANSIENT_LOCAL) so it does not matter whether this runs before or after the
MPC node — a late subscriber still gets the plan. Without that, the ordering of the
launch file silently decides whether the run works.

**And the node stays alive to hold that latch.** Transient-local durability is
publisher-side: the retained sample lives in the publisher, so a loader that publishes
and exits leaves nothing for anyone who subscribes afterwards: the MPC would wait on
/brain/incoming_plan indefinitely while brain (already subscribed) publishes plan steps.
Spinning costs nothing and removes the race.

Usage:
  ros2 run carla_gt_bridge load_mission_node --ros-args -p plan_file:=/path/to/tree.json
"""

from __future__ import annotations

import json
import os

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import Trigger

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     reliability=ReliabilityPolicy.RELIABLE)


class LoadMissionNode(Node):
    def __init__(self) -> None:
        super().__init__("load_mission_node")
        self.declare_parameter("plan_file", "")
        self.declare_parameter("plan_topic", "/brain/incoming_plan")
        self.declare_parameter("load_service", "/brain/load_plan")
        self.declare_parameter("service_timeout_s", 20.0)
        # 0 = stay alive forever, holding the latch. Any positive value exits
        # after that many seconds, which is only useful for a scripted one-shot
        # where something else republishes the plan.
        self.declare_parameter("linger_s", 0.0)

        path = self._resolve()
        with open(path) as f:
            tree = json.load(f)
        n_steps = len(tree.get("steps", []))
        regions = sorted(int(k) for k in (tree.get("centroids") or {}))
        self.get_logger().info(
            f"loaded {os.path.basename(path)}: '{tree.get('plan_name')}' — {n_steps} "
            f"steps over regions {regions}")
        if not tree.get("centroids"):
            self.get_logger().error(
                "this plan carries no `centroids`, so the MPC will have nothing to steer "
                "toward. Regenerate it with scripts/drive_english.py --dry.")

        # Held as an attribute, not a local: a publisher that goes out of scope is
        # garbage-collected, and with it the retained transient-local sample.
        self._pub = self.create_publisher(
            String, self.get_parameter("plan_topic").get_parameter_value().string_value,
            LATCHED)
        self._pub.publish(String(data=json.dumps(tree)))
        self.get_logger().info("published to /brain/incoming_plan (latched)")

        self._call_load()

    def _resolve(self) -> str:
        explicit = self.get_parameter("plan_file").get_parameter_value().string_value
        if explicit:
            if not os.path.exists(explicit):
                raise SystemExit(f"plan_file not found: {explicit}")
            return explicit
        raise SystemExit("plan_file is required: a brain tree JSON (see drive_english.py)")

    def _call_load(self) -> None:
        srv = self.get_parameter("load_service").get_parameter_value().string_value
        timeout = self.get_parameter("service_timeout_s").get_parameter_value().double_value
        cli = self.create_client(Trigger, srv)
        if not cli.wait_for_service(timeout_sec=timeout):
            # Not fatal: the plan is latched, so a brain that starts later will still
            # receive it and can be told to load by hand. Say so rather than dying.
            self.get_logger().error(
                f"{srv} did not appear within {timeout:.0f}s. The plan IS published and "
                f"latched, so once brain_controller is up run:\n"
                f"  ros2 service call {srv} std_srvs/srv/Trigger")
            return
        fut = cli.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, fut, timeout_sec=timeout)
        res = fut.result()
        if res is None:
            self.get_logger().error(f"{srv} call timed out")
        elif res.success:
            self.get_logger().info(f"brain loaded the plan: {res.message}")
        else:
            self.get_logger().error(f"brain refused the plan: {res.message}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LoadMissionNode()
    linger = node.get_parameter("linger_s").get_parameter_value().double_value
    try:
        if linger > 0.0:
            node.get_logger().info(f"holding the latch for {linger:.0f}s")
            rclpy.spin_once(node, timeout_sec=linger)
        else:
            node.get_logger().info(
                "holding the latched plan (Ctrl-C to release). A subscriber that starts "
                "later still receives it only while this node is alive.")
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
