"""ROS2 online inference node for the BEV pipeline.

Runs the same geometry pipeline as offline (gravity-align -> plane-fit ground
removal -> submap -> raster/voxel/geom) feeding the configured feature
extractor and the trained classifier, then DUAL-PUBLISHES:

  * ``/predicted_cluster`` (std_msgs/Int16)   — canonical id, unchanged contract
    for nl_planner / brain_controller.
  * ``/predicted_state``   (std_msgs/String)   — JSON with phase, landmark_type,
    cluster_id, confidence, raw/smoothed label.

Config comes from ``config/bev_pipeline.yaml`` (see RuntimeConfig). Retains the
legacy ``/cluster_override`` Int16 UX (-99 clears). The pointcloud callback uses
the most-recently-cached odometry (and image, if the camera extractor is used).

NOTE: frozen extractors (geom_features / bev_dinov2 / camera_dinov2) run live
with no extra state. The from-scratch CNN extractors (bev_cnn / vol3d_cnn) would
need their trained backbone weights persisted/loaded — not yet implemented — so
the node raises a clear error if configured with one.
"""

from __future__ import annotations

import json
import os

import numpy as np


class BevInferenceNode:
    """Constructed lazily inside main() so importing this module needs no ROS."""

    def __init__(self, node, config):
        import collections
        from std_msgs.msg import Int16, String
        from sensor_msgs.msg import PointCloud2, Image
        from nav_msgs.msg import Odometry
        from rclpy.qos import qos_profile_sensor_data

        from bev_pipeline.pipeline import FramePipeline, PipelineParams
        from bev_pipeline.bev_rasterizer import rasterize_multiband, BevParams
        from bev_pipeline.voxelizer import voxelize, VoxelParams
        from bev_pipeline import geometry as geom
        from bev_pipeline.feature_extractors import (make_extractor, input_kind_of,
                                                     requires_training_of)
        from bev_pipeline.classifier import StateClassifier
        from bev_pipeline.smoothing import ModeSmoother

        self.node = node
        self.cfg = config
        self._geom = geom
        self._rasterize, self._BevParams = rasterize_multiband, BevParams
        self._voxelize, self._VoxelParams = voxelize, VoxelParams

        self.pipeline = FramePipeline(PipelineParams(
            submap_window=config.submap_window, use_plane_fit=True))
        self.input_kind = input_kind_of(config.extractor)
        self.extractor = make_extractor(config.extractor, device=config.device)
        if requires_training_of(config.extractor):
            w = getattr(config, "extractor_weights", None)
            if not w or not os.path.exists(w):
                raise FileNotFoundError(
                    f"extractor {config.extractor!r} is trainable and needs "
                    f"'extractor_weights' (a saved backbone). Got {w!r}. Train + "
                    "save one, or use a frozen extractor (geom_features / bev_dino*).")
            self.extractor.load(w)

        self.classifier = StateClassifier.load(config.classifier_path)
        with open(config.classifier_meta_path) as f:
            self.meta = json.load(f)
        self.smoother = ModeSmoother(window=config.smoothing_window, default=0)
        self.override = None
        self._last_odom = None
        self._last_image = None

        # publishers
        self.pub_cluster = node.create_publisher(Int16, config.predicted_cluster_topic, 10)
        self.pub_state = node.create_publisher(String, config.predicted_state_topic, 10)

        # subscriptions
        node.create_subscription(PointCloud2, config.lidar_topic,
                                 self._on_cloud, qos_profile=qos_profile_sensor_data)
        if config.odom_topic:
            node.create_subscription(Odometry, config.odom_topic, self._on_odom, 20)
        if self.input_kind == "camera" and config.image_topic:
            node.create_subscription(Image, config.image_topic, self._on_image,
                                     qos_profile=qos_profile_sensor_data)
        node.create_subscription(Int16, config.override_topic, self._on_override, 10)
        node.get_logger().info(
            f"bev_inference_node ready: extractor={config.extractor}, "
            f"env={config.environment}, publishing {config.predicted_cluster_topic}")

    # -- callbacks --------------------------------------------------------- #
    def _on_odom(self, msg):
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        self._last_odom = (np.array([p.x, p.y, p.z]),
                           np.array([o.x, o.y, o.z, o.w]))

    def _on_image(self, msg):
        from bev_pipeline.bag_io import _image_to_rgb
        self._last_image = _image_to_rgb(msg)

    def _on_override(self, msg):
        self.override = None if msg.data == -99 else int(msg.data)

    def _points_from_msg(self, msg):
        from sensor_msgs_py.point_cloud2 import read_points
        has_i = any(f.name == "intensity" for f in msg.fields)
        names = ("x", "y", "z", "intensity") if has_i else ("x", "y", "z")
        raw = read_points(msg, field_names=names, skip_nans=True)
        if len(raw) == 0:
            return np.empty((0, 4), np.float32)
        if has_i:
            pts = np.column_stack([raw["x"], raw["y"], raw["z"], raw["intensity"]])
        else:
            xyz = np.column_stack([raw["x"], raw["y"], raw["z"]])
            pts = np.column_stack([xyz, np.zeros(len(xyz), np.float32)])
        return pts.astype(np.float32)

    def _on_cloud(self, msg):
        from std_msgs.msg import Int16, String
        pts = self._points_from_msg(msg)
        pos, quat = (self._last_odom if self._last_odom is not None else (None, None))
        acc = self.pipeline.process(pts, position=pos, quat=quat)

        if self.input_kind == "geom":
            feat_in = self._geom.geometric_features(acc)[None, :]
        elif self.input_kind == "points":
            n = 2048
            if acc.shape[0] == 0:
                sub = np.zeros((n, 4), np.float32)
            else:
                idx = np.random.RandomState(0).choice(acc.shape[0], n,
                                                      replace=acc.shape[0] < n)
                sub = acc[idx].astype(np.float32)
            feat_in = sub[None, ...]
        elif self.input_kind == "bev":
            feat_in = self._rasterize(acc, self._BevParams())[None, ...]
        elif self.input_kind == "voxel":
            feat_in = self._voxelize(acc, self._VoxelParams())[None, ...]
        elif self.input_kind == "camera":
            if self._last_image is None:
                return
            feat_in = [self._last_image]
        else:
            return

        feats = self.extractor.extract_batch(feat_in)
        proba = self.classifier.predict_proba(feats)[0]
        raw_id = int(np.argmax(proba))
        conf = float(proba[raw_id])
        smoothed = self.smoother.push(raw_id)
        out_id = self.override if self.override is not None else smoothed

        cm = Int16(); cm.data = int(out_id)
        self.pub_cluster.publish(cm)

        label = self.classifier.label_of(out_id)
        phase, _, ltype = (label.partition("_") if label != "open_road"
                           else ("open_road", "", ""))
        sm = String()
        sm.data = json.dumps({
            "phase": phase, "landmark_type": ltype or None, "cluster_id": int(out_id),
            "confidence": round(conf, 3),
            "raw_label": self.classifier.label_of(raw_id),
            "smoothed_label": label})
        self.pub_state.publish(sm)


def main(args=None):
    import rclpy
    from rclpy.node import Node
    from bev_pipeline.config import RuntimeConfig

    rclpy.init(args=args)
    node = Node("bev_inference_node")
    # config path: ROS param 'config' or the packaged default
    node.declare_parameter("config", "")
    cfg_path = node.get_parameter("config").get_parameter_value().string_value
    if not cfg_path:
        from ament_index_python.packages import get_package_share_directory
        cfg_path = os.path.join(get_package_share_directory("bev_pipeline"),
                                "config", "bev_pipeline.yaml")
    config = RuntimeConfig.from_yaml(cfg_path)
    BevInferenceNode(node, config)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
