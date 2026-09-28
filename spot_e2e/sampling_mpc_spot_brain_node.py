#!/usr/bin/env python3
"""Spot MPC driven by the plan tree instead of a flat cluster-pair list.

WHAT THIS CHANGES, AND WHY IT IS ONE METHOD.

`sampling_mpc_spot_ros_node.SamplingMPCSpotNode` walks `self.plan_sequence`, a list of
`{start, next}` cluster pairs, and calls `_advance_plan()` itself the moment the observed
cluster equals `expected_next`. That is the prior system's linear representation: the node
contains zero occurrences of `branch` and zero of `brain`, so it cannot execute a plan
tree, and it cannot wait on a cue, count an ordinal, or confirm a manoeuvre.

None of that needs to move into this node. `brain_controller` already owns all of it and
is deliberately portable -- four topics in (`/predicted_cluster`, a cue topic, an image,
odometry) and `/brain/state` out, carrying `start_cluster`, `goal_cluster`, `trigger` and
`branch_path`. So the node stops deciding WHEN to advance and starts asking:

    plan_sequence[i] -> {start, next}        becomes   /brain/state -> start, goal
    _advance_plan() when cluster == next     becomes   brain advances; we just steer
    _forbidden = everything but the pair     becomes   forbid_clusters from the plan

Everything below the seam is inherited untouched: `_run_sampling_mpc_step`,
`_apply_action`, `_update_spot_state`, `_publish_debug` and the whole Spot SDK path.

STATUS: NEVER RUN ON HARDWARE. Kept in spot_e2e/, separate from the deployed packages.
See spot_e2e/README.md for the wiring checklist.

TWO THINGS THAT NEED NO NEW CODE:
  * `brain_controller` defaults to `cue_source="vlm"` (gpt-4o), so the visual cue path
    needs no new code -- point `image_topic` at the ZED.
  * `cluster_map.livox1.yaml` splits the junction into `Intersection: Approach/Enter`,
    `Intersection: In` and `Intersection: Exit`. CARLA's two-mode Town05 map does not, so
    the sub-junction progression is only observable on the robot.
"""
from __future__ import annotations

import json
import os
import sys

import rclpy
from std_msgs.msg import String

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
for _p in (os.path.join(_SRC, "dgppo_ros_node_pkg"), os.path.join(_SRC, "carla_gt_bridge")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dgppo_ros_node_pkg.sampling_mpc_spot_ros_node import SamplingMPCSpotNode  # noqa: E402
from dgppo_ros_node_pkg.plan import map_cluster_id  # noqa: E402
from carla_gt_bridge.routing import StepTargeter, adjacency_from_bearing_map  # noqa: E402


class SpotBrainMPCNode(SamplingMPCSpotNode):
    """Steers at whatever cluster the brain currently says is the goal."""

    def __init__(self) -> None:
        super().__init__()

        # `/brain/state` is a latched-ish JSON snapshot; the brain republishes it on every
        # state change, so a missed message self-corrects on the next transition.
        self._brain_state: dict = {}
        self.create_subscription(String, "/brain/state", self._brain_state_cb, 10)

        # StepTargeter grounds a MODE ("Intersection: In") to the next region carrying
        # that label, by adjacency and then by BFS that never re-enters the previous
        # region. Without it we would be steering at a cluster id the plan never named --
        # the plan names modes, and which instance satisfies one is a runtime question.
        plan = getattr(self, "plan", None) or {}
        bearing_map = plan.get("bearing_map") or {}
        labels = {int(k): v for k, v in (plan.get("cluster_labels") or {}).items()}
        self._targeter = (
            StepTargeter(adjacency_from_bearing_map(bearing_map), labels)
            if bearing_map and labels else None
        )
        if self._targeter is None:
            self.get_logger().warning(
                "plan carries no bearing_map/cluster_labels, so mode-based targeting is "
                "off; falling back to the brain's goal_cluster verbatim")

        # A plan-level prohibition, if the formula or the plan declared one. Unlike the
        # pair-based `_forbidden` this replaces, it is not "everything except the current
        # pair" -- it is only what the mission actually forbids.
        self._forbid = [int(c) for c in (plan.get("forbid_clusters") or [])]
        if self._forbid:
            self.get_logger().info(f"plan forbids clusters {self._forbid}")

        self.get_logger().info(
            "SpotBrainMPCNode up: advancement is owned by /brain/state, not by this node")

    # ------------------------------------------------------------------ #

    def _brain_state_cb(self, msg: String) -> None:
        try:
            self._brain_state = json.loads(msg.data)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warning(f"unparseable /brain/state: {exc}")

    def _goal_from_brain(self) -> tuple[int | None, int | None, str]:
        """(start_cluster, target_cluster, why). Target is grounded through the taxonomy."""
        st = self._brain_state
        if not st:
            return None, None, "no /brain/state yet"
        if str(st.get("state", "")).upper() in ("COMPLETE", "ERROR"):
            return None, None, f"brain is {st.get('state')}"

        start = st.get("start_cluster")
        goal = st.get("goal_cluster")
        label = st.get("goal_label") or st.get("goal_mode")

        if self._targeter is None or not label:
            return start, goal, "brain goal_cluster used verbatim"

        self._targeter.observe(map_cluster_id(self.latest_predicted_cluster_id))
        target, note = self._targeter.target_for(
            label, (st.get("step"), tuple(st.get("branch_path") or ())))
        return start, (target if target is not None else goal), note

    # ------------------------------------------------------------------ #

    def control_loop(self) -> None:
        """The seam. Identical to the parent below the target selection."""
        debug_mode = bool(
            self.get_parameter("debug_mode").get_parameter_value().bool_value)
        if not self._check_topics(debug_mode):
            return

        start, target, why = self._goal_from_brain()
        if target is None:
            # Not an error: the brain owns advancement, so "nothing to steer at" means it
            # is waiting on a cue, deciding a branch, finished, or has aborted. Standing
            # still is the correct response to all four.
            self.get_logger().info(f"holding: {why}", throttle_duration_sec=3.0)
            if str(self._brain_state.get("state", "")).upper() == "COMPLETE":
                self._handle_plan_complete()
            return

        self._start_id = start if start is not None else target
        self._target_id = target
        self._forbidden = self._forbid          # only what the mission forbids

        state = self._update_spot_state()
        result = self._run_sampling_mpc_step(state, self._start_id, self._target_id)
        self._apply_action(state, result)
        self._publish_debug(state, result, self._start_id, self._target_id)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SpotBrainMPCNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
