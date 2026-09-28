"""Refuse to let a run proceed when `/predicted_cluster` carries the wrong kind of integer.

    ros2 run carla_gt_bridge cluster_space_guard -p expect:=region:carla_town05

WHY A GUARD RATHER THAN A RENAME. Five live nodes publish on `/predicted_cluster` and
about twelve subscribe; the integer is a region id, an HDBSCAN cluster id or a classifier
label index depending on the producer, and the three are indistinguishable on the wire.
Renaming the topic would have to touch robot-side nodes, and it would not stop the NEXT
producer from guessing wrong. An assertion does.

WHAT IT CATCHES, all three of which are otherwise silent:
  * the wrong KIND of producer (a label index feeding a region cluster_map);
  * TWO producers at once, which is how a spoof node or a leftover classifier ends up
    racing the oracle and neither log says so;
  * the same kind and env but a DIFFERENT id set (e.g. a region corpus and a
    cluster_map with different region counts that both call themselves
    `carla_town01`); the space fingerprint is what separates them.

It also fails when ids arrive and NOBODY announced a space, because a producer that has
not been taught the contract is exactly the one most likely to be wrong. Set
`require_space:=false` while migrating a producer, never in a scored run.
"""
from __future__ import annotations

import os
import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Int16, String

from ..cluster_space import SPACE_TOPIC, TOPIC, compatible


class ClusterSpaceGuard(Node):
    def __init__(self) -> None:
        super().__init__("cluster_space_guard")
        self.declare_parameter("expect", "")
        self.declare_parameter("require_space", True)
        self.declare_parameter("grace_s", 10.0)
        self.declare_parameter("fatal", True)

        self._expect = self.get_parameter("expect").get_parameter_value().string_value
        self._require = self.get_parameter("require_space").get_parameter_value().bool_value
        self._fatal = self.get_parameter("fatal").get_parameter_value().bool_value
        self._grace = self.get_parameter("grace_s").get_parameter_value().double_value
        if not self._expect:
            self.get_logger().error(
                "no `expect` given -- the guard cannot check anything. Pass the id space "
                "the consumers' cluster_map defines, e.g. -p expect:=region:carla_town05")
            raise SystemExit(2)

        self._seen: set[str] = set()
        self._ids = 0
        self._ok = False
        self.create_subscription(
            String, SPACE_TOPIC, self._space_cb,
            QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(Int16, TOPIC, self._id_cb, 10)
        self.create_timer(self._grace, self._deadline)
        self.get_logger().info(
            f"cluster_space_guard: expecting {self._expect!r} on {TOPIC}; "
            f"{'ids without an announced space are FATAL' if self._require else 'unannounced ids allowed'}")

    def _die(self, msg: str) -> None:
        self.get_logger().error(msg)
        if not self._fatal:
            return
        # os._exit, NOT rclpy.shutdown() + sys.exit. Calling shutdown from inside a
        # callback while the executor is spinning does not reliably return control, and
        # can leave the guard alive and silent after detecting a violation, which is the
        # one state a guard must never be in.
        # A guard's contract is that the process dies; flush first so the reason is
        # not lost with the buffer.
        sys.stderr.flush()
        sys.stdout.flush()
        os._exit(3)

    def _space_cb(self, msg: String) -> None:
        d = (msg.data or "").strip()
        if d in self._seen:
            return
        self._seen.add(d)
        if len(self._seen) > 1:
            self._die(f"TWO id spaces on {TOPIC}: {sorted(self._seen)} -- two producers "
                      f"are publishing and the consumers cannot tell them apart")
            return
        if not compatible(d, self._expect):
            self._die(f"id space mismatch: producer says {d!r}, consumers expect "
                      f"{self._expect!r}. The integers would resolve to the WRONG modes.")
            return
        self._ok = True
        self.get_logger().info(f"id space OK: {d}")

    def _id_cb(self, _msg: Int16) -> None:
        self._ids += 1

    def _deadline(self) -> None:
        if self._ok or not self._ids:
            return
        if self._require and not self._seen:
            self._die(f"{self._ids} ids on {TOPIC} and no producer announced a space "
                      f"within {self._grace:.0f}s. An unannounced producer is the one "
                      f"most likely to be publishing the wrong kind of integer.")


def main() -> None:
    rclpy.init()
    n = ClusterSpaceGuard()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
