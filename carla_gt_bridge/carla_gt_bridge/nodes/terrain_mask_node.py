#!/usr/bin/env python3
"""Publish a terrain CLASS MASK for the MPC, from whichever labeller is selected.

WHAT IT IS FOR. `terrain_mpc.TerrainMpcConfig.terrain_class` needs the surface at K x n
HYPOTHETICAL future positions. A segmented camera frame carries terrain over space -- in
pixels -- so the MPC projects its rollouts into this mask and reads the class at each one.
`camera_terrain.MaskTerrainSource` does the projection; this node produces the mask.

WHY IT IS A SEPARATE NODE AND NOT A CALLBACK ON THE MPC. SegFormer inference is 50-200 ms.
The MPC's control loop runs INSIDE `_odom_cb` on a single-threaded executor, where
high-rate odometry can already starve the loop. A torch callback in that executor is a
non-starter. Here the labeller runs at
its own rate and the MPC only ever does a `np.frombuffer`.

THE POSE TRAVELS WITH THE MASK, IN `header.frame_id`. Ugly, and chosen against two worse
options. A new `.msg` means touching an `ament_cmake` interface package, and a stale
interface build fails silently. Matching a separate pose topic by `header.stamp` is
unreliable because odometry stamps can freeze (many messages sharing nearly the same
stamp). One message is atomically consistent and the parse is unit-testable offline.

`carla_semantic` READS THE SENSOR DIRECTLY, and it has to. `carla_ros_bridge`'s
`SemanticSegmentationCamera.get_carla_image_data_array` calls
`carla_image.convert(carla.ColorConverter.CityScapesPalette)` before publishing, so the wire
carries palette COLOURS and the tag in the red channel is gone. This node attaches its own
camera and reads `raw_data[:, :, 2]`, the pattern `gt_obstacle_node` uses for the same reason.
"""
from __future__ import annotations

import math

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from ..frames import wrap_pi
from ..terrain_classes import NAMES
from ..terrain_labellers import make_labeller

#: The mask's pose is packed here. Parsed by `parse_frame_id`, which is what the MPC uses.
FRAME_ID_FMT = "terrain_mask x={x:.4f} y={y:.4f} yaw={yaw:.6f} n_odom={n}"


def parse_frame_id(frame_id: str):
    """``header.frame_id`` -> ``(x, y, yaw, n_odom)``, or None if it is not one of ours.

    Returns None rather than raising, and rather than defaulting to the origin: a mask
    silently placed at (0, 0, 0) would classify every rollout against the map's origin and
    look like a segmentation failure hundreds of metres away.
    """
    if not frame_id.startswith("terrain_mask "):
        return None
    try:
        kv = dict(p.split("=", 1) for p in frame_id.split()[1:])
        return (float(kv["x"]), float(kv["y"]), float(kv["yaw"]), int(kv["n_odom"]))
    except (KeyError, ValueError):
        return None


class TerrainMaskNode(Node):
    """One mask topic, any labeller."""

    def __init__(self) -> None:
        super().__init__("terrain_mask_node")
        self.declare_parameter("labeller", "carla_semantic")
        self.declare_parameter("mask_topic", "/terrain/class_mask")
        self.declare_parameter("image_topic", "/carla/ego_vehicle/rgb_front/image")
        self.declare_parameter("odom_topic", "/carla/ego_vehicle/odometry")
        self.declare_parameter("role_name", "ego_vehicle")
        self.declare_parameter("carla_host", "localhost")
        self.declare_parameter("carla_port", 2000)
        # Camera geometry. Defaults follow `camera_terrain.TERRAIN_CAMERA` -- front-most
        # mount, and the fov/pitch that leave zero blind arcs.
        self.declare_parameter("cam_x", 2.425)
        self.declare_parameter("cam_z", 1.5)
        self.declare_parameter("fov", 130.0)
        self.declare_parameter("pitch_down_deg", 30.0)
        self.declare_parameter("width", 800)
        self.declare_parameter("height", 600)
        self.declare_parameter("sensor_tick", 0.5)

        p = self.get_parameter
        self._labeller_name = str(p("labeller").value).strip().lower()
        self._role = str(p("role_name").value)
        self._w = int(p("width").value)
        self._h = int(p("height").value)
        self._pose = (0.0, 0.0, 0.0)
        self._n_odom = 0
        self._have_pose = False
        self._published = 0

        self._label = make_labeller(self._labeller_name)
        self._pub = self.create_publisher(Image, str(p("mask_topic").value), 1)
        self.create_subscription(Odometry, str(p("odom_topic").value), self._odom_cb,
                                 qos_profile_sensor_data)

        self.get_logger().info(
            f"labeller={self._labeller_name} -> {p('mask_topic').value} "
            f"({self._w}x{self._h} mono8, class ids {dict(NAMES)})")

        self._carla_actors = []
        if self._labeller_name == "carla_semantic":
            self._start_carla_camera()
        else:
            self.create_subscription(Image, str(p("image_topic").value), self._image_cb,
                                     qos_profile_sensor_data)
            self.get_logger().info(f"subscribed to {p('image_topic').value}")

    # -- pose ---------------------------------------------------------------- #

    def _odom_cb(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self._pose = (msg.pose.pose.position.x, msg.pose.pose.position.y, wrap_pi(yaw))
        self._n_odom += 1
        self._have_pose = True

    # -- the two input paths ------------------------------------------------- #

    def _start_carla_camera(self) -> None:
        """Attach our own semantic camera and read the RAW TAG plane."""
        import carla
        p = self.get_parameter
        client = carla.Client(str(p("carla_host").value), int(p("carla_port").value))
        client.set_timeout(20.0)
        world = client.get_world()
        ego = None
        for a in world.get_actors().filter("vehicle.*"):
            if a.attributes.get("role_name") == self._role:
                ego = a
                break
        if ego is None:
            raise RuntimeError(
                f"no vehicle with role_name={self._role!r}; the mask node must start AFTER "
                f"carla_spawn_objects, or it will publish nothing while looking healthy")
        bp = world.get_blueprint_library().find("sensor.camera.semantic_segmentation")
        bp.set_attribute("image_size_x", str(self._w))
        bp.set_attribute("image_size_y", str(self._h))
        bp.set_attribute("fov", str(float(p("fov").value)))
        # THROTTLED, and it is load-bearing: with rendering on, an unthrottled camera
        # stalls the MPC entirely -- in synchronous mode the bridge services sensor data
        # every tick and starves the control loop the server is waiting on. The MPC's
        # staleness handling exists because of this throttle.
        bp.set_attribute("sensor_tick", str(float(p("sensor_tick").value)))
        # CARLA pitch is NEGATIVE for down; our `pitch_down_deg` is positive (confirmed by
        # checking the horizon row on a rendered frame).
        tf = carla.Transform(
            carla.Location(x=float(p("cam_x").value), z=float(p("cam_z").value)),
            carla.Rotation(pitch=-float(p("pitch_down_deg").value)))
        cam = world.spawn_actor(bp, tf, attach_to=ego)
        self._carla_actors.append(cam)
        cam.listen(self._carla_image_cb)
        self.get_logger().info(
            f"semantic camera attached to {self._role}: x={p('cam_x').value} "
            f"z={p('cam_z').value} fov={p('fov').value} pitch_down={p('pitch_down_deg').value} "
            f"tick={p('sensor_tick').value}s")

    def _carla_image_cb(self, image) -> None:
        arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
            (image.height, image.width, 4))
        self._publish(self._label(arr[:, :, 2]))          # RED channel is the tag

    def _image_cb(self, msg: Image) -> None:
        if msg.encoding not in ("bgr8", "rgb8"):
            self.get_logger().error(
                f"image encoding {msg.encoding!r} is not bgr8/rgb8 -- refusing to guess",
                throttle_duration_sec=10.0)
            return
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        if msg.encoding == "rgb8":
            img = img[:, :, ::-1]
        if self._labeller_name == "segformer":
            img = np.ascontiguousarray(img[:, :, ::-1])   # segformer wants RGB
        self._publish(self._label(np.ascontiguousarray(img)))

    # -- output -------------------------------------------------------------- #

    def _publish(self, klass: np.ndarray) -> None:
        if not self._have_pose:
            self.get_logger().warning(
                "no odometry yet -- a mask with no pose cannot be projected, dropping",
                throttle_duration_sec=5.0)
            return
        k = np.ascontiguousarray(np.clip(klass, 0, 255).astype(np.uint8))
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        x, y, yaw = self._pose
        msg.header.frame_id = FRAME_ID_FMT.format(x=x, y=y, yaw=yaw, n=self._n_odom)
        msg.height, msg.width = k.shape
        msg.encoding = "mono8"
        msg.is_bigendian = 0
        msg.step = k.shape[1]
        msg.data = k.tobytes()
        self._pub.publish(msg)
        self._published += 1
        if self._published % 20 == 1:
            frac = {NAMES[c]: round(float((k == c).mean()), 3)
                    for c in NAMES if (k == c).any()}
            self.get_logger().info(
                f"mask #{self._published} at ({x:.1f},{y:.1f}) n_odom={self._n_odom} {frac}")

    def destroy_node(self) -> bool:
        for a in self._carla_actors:
            try:
                a.stop()
                a.destroy()
            except Exception:
                pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TerrainMaskNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
