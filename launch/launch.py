from launch_ros.descriptions import ComposableNode
from launch_ros.actions import ComposableNodeContainer
import subprocess
from launch_ros.actions import Node
import os
import glob
from launch import LaunchDescription
import yaml


# ros2 run opencv_cam opencv_cam_main --ros-args --remap /image_raw:=/my_camera/image_raw --params-file config/cameras.yml


def get_transformer():
    config_file = os.environ.get("TRANSFORMER_CONFIG", "config/pose_transformer.yaml")
    transformer = Node(
        package="odom_transform",
        executable="transformer",
        parameters=[config_file],
        arguments=[
            "--ros-args",
            "--log-level",
            "warn",
        ],
    )

    return [transformer]


def get_estimator():
    estimator_config = os.environ.get("MINS_CONFIG", "config/mins/config.yaml")
    return [
        Node(
            package="mins",
            namespace="",
            executable="subscribe",
            arguments=[estimator_config],
        )
    ]


def aruco_nodes():
    params_file_aruco_detect = os.environ.get(
        "ARUCO_PARAMS", "config/stereo_aruco_detector.yaml"
    )

    tracker_params = os.environ.get("TRACKER_PARAMS", "config/optimizer_transform.yml")

    return [
        Node(
            package="fhnw_localization",
            executable="aruco_detect",
            name="stereo_aruco_detector",
            parameters=[params_file_aruco_detect],
            output="screen",
            arguments=[
                "--ros-args",
                "--log-level",
                "info",
                "--params-file",
                params_file_aruco_detect,
            ],
        ),
        Node(
            package="fhnw_localization",
            executable="optimizer_transform",
            name="transform_optimizer",
            parameters=[tracker_params],
        ),
    ]


def generate_launch_description():
    return LaunchDescription(get_estimator() + get_transformer() + aruco_nodes())
