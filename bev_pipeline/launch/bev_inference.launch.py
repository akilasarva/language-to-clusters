"""Launch the BEV inference node with a config file.

Usage:
    ros2 launch bev_pipeline bev_inference.launch.py
    ros2 launch bev_pipeline bev_inference.launch.py config:=/abs/path/to/bev_pipeline.yaml
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    default_cfg = os.path.join(get_package_share_directory("bev_pipeline"),
                               "config", "bev_pipeline.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=default_cfg,
                              description="Path to bev_pipeline.yaml"),
        Node(
            package="bev_pipeline",
            executable="bev_inference_node",
            name="bev_inference_node",
            output="screen",
            parameters=[{"config": LaunchConfiguration("config")}],
        ),
    ])
