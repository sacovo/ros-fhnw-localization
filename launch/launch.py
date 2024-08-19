from ament_index_python import get_package_share_directory
import subprocess
from launch_ros.actions import Node
import os
import glob
from launch import LaunchDescription
import yaml


# ros2 run opencv_cam opencv_cam_main --ros-args --remap /image_raw:=/my_camera/image_raw --params-file config/cameras.yml


camera_urls = [
    ("a", os.environ.get("CAMERA_A", "http://172.16.10.35:8080/stream")),
    ("b", os.environ.get("CAMERA_B", "http://172.16.10.35:8081/stream")),
    ("c", os.environ.get("CAMERA_C", "http://172.16.10.36:8080/stream")),
    ("d", os.environ.get("CAMERA_D", "http://172.16.10.36:8081/stream")),
]

ZED_HEIGHT = int(os.environ.get("ZED_HEIGHT", 720))
ZED_WIDTH = int(os.environ.get("ZED_WIDTH", 1280))
ZED_FPS = int(os.environ.get("ZED_FPS", 15))
ZED_INDEX = int(os.environ.get("ZED_INDEX", -1))


def get_camera_nodes():

    # Make sure we have access to all /dev/video* devices
    devices = glob.glob("/dev/video*")
    for device in devices:
        subprocess.run(["sudo", "chmod", "777", device])
    cameras = []
    
    if not os.environ.get("DISABLE_PI", False):
        cameras = [
            Node(
                package="opencv_cam",
                namespace=f"camera_{dir}",
                executable="opencv_cam_main",
                name=f"cam_{dir}",
                remappings=[
                    ("/image_raw", f"/cam_{dir}/raw"),
                ],
                parameters=[
                    {
                        "file": True,
                        "filename": url,
                    }
                ],
            )
            for dir, url in camera_urls
        ]

    return cameras + [
        Node(
            package="opencv_cam",
            namespace=f"zed",
            executable="opencv_cam_main",
            name=f"zed",
            parameters=[
                {
                    "file": False,
                    "index": ZED_INDEX,
                    "filename": "data/video.avi",
                    "split_frame": True,
                    "height": ZED_HEIGHT,
                    "width": ZED_WIDTH * 2,
                    "fps": ZED_FPS,
                    "flip": -1,
                }
            ],
        )
    ]


def get_imu_node():

    config = os.path.join(
        "config",
        "hipnuc_config.yml",
    )

    with open(config, "r") as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)

    serial_port = config_dict["IMU_publisher"]["ros__parameters"]["serial_port"]

    subprocess.run(["sudo", "chmod", "777", serial_port])

    return [
        Node(
            package="hipnuc_imu",
            executable="talker",
            name="IMU_publisher",
            parameters=[config],
            output="screen",
        ),
    ]


def get_wheel_odom():
    return [
        Node(
            package="wheel_odometry",
            namespace="wheel",
            executable="odom_node",
            parameters=[
                {
                    "speed_addr": os.environ.get("SPEED_ADDR", "172.16.10.78:3002"),
                    "angle_addr": os.environ.get("ANGLE_ADDR", "172.16.10.79:3002"),
                }
            ],
        )
    ]


def get_transformer():
    transformer = Node(
        package="odom_transform",
        executable="transformer",
        parameters=[
            {"yaw": 0.0},
            {"initial_position": [0.0, 0.0, 0.0]},
            {"initial_orientation": [0.0, 0.0, 0.0]},
            {"odom_in": os.environ.get("ODOM_IN", "/odomimu")},
            {"odom_out": os.environ.get("ODOM_OUT", "/odomrover")},
            {"path_in": os.environ.get("PATH_IN", "/pathimu")},
            {"path_out": os.environ.get("PATH_OUT", "/pathrover")},
            {"pose_in": os.environ.get("POSE_IN", "/poseimu")},
            {"pose_out": os.environ.get("POSE_OUT", "/poserover")},
        ],
    )

    return [transformer]


def get_estimator():
    estimator = os.environ.get("ESTIMATOR", "openvins")
    if estimator == "openvins":
        estimator_config = os.environ.get("ESTIMATOR_CONFIG", "config/openvins/estimator_config.yaml")
        return [
            Node(
                package="ov_msckf",
                namespace="",
                executable="run_subscribe_msckf",
                arguments=[estimator_config],
            )
        ]
    elif estimator == "mins":
        estimator_config = os.environ.get("ESTIMATOR_CONFIG", "config/mins/config.yaml")
        return [
            Node(
                package="mins",
                namespace="",
                executable="subscribe",
                arguments=[estimator_config],
            )
        ]
    return []


def generate_launch_description():
    return LaunchDescription(
        get_camera_nodes()
        + get_imu_node()
        + get_wheel_odom()
        + get_estimator()
        + get_transformer()
    )
