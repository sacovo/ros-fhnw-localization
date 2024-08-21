import yaml
import numpy as np

from scipy.spatial.transform import Rotation as R
import cv2
from scipy.optimize import least_squares
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from message_filters import ApproximateTimeSynchronizer, Subscriber
import time

import tf2_ros

from rclpy.parameter import Parameter
from rcl_interfaces.srv import GetParameters


def query_odom_transform(self):
    client = self.create_client(GetParameters, "/odom_transform/get_parameters")
    while not client.wait_for_service(timeout_sec=1.0):
        self.get_logger().warn("Waiting for /odom_transform service...")

    request = GetParameters.Request()
    request.names = ["position", "yaw"]

    future = client.call_async(request)
    rclpy.spin_until_future_complete(self, future)

    try:
        response = future.result()
        position = response.values[0].double_array_value
        yaw = response.values[1].double_value
        return position, yaw
    except Exception as e:
        self.get_logger().error(f"Failed to get parameters from /odom_transform: {e}")
        return None, None


def inverse_transform_position(self, position, offset_position, yaw):
    # Create the rotation matrix for the inverse yaw
    rotation_matrix = tf2_ros.transformations.rotation_matrix(-yaw, (0, 0, 1))[:3, :3]

    # Convert the position to a vector
    pos_vector = np.array([position.x, position.y, position.z])

    # Apply the inverse rotation and then the inverse translation
    rotated_position = np.dot(rotation_matrix, pos_vector - offset_position)

    # Update the position object
    position.x = rotated_position[0]
    position.y = rotated_position[1]
    position.z = rotated_position[2]


class ArucoPoseEstimator(Node):
    def __init__(self):
        super().__init__("aruco_pose_estimator")

        # Declare parameters for topics, configuration, and camera calibration file
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("pose_pub_topic", "/pose_with_covariance")
        self.declare_parameter("aruco_dict", "DICT_6X6_250")
        self.declare_parameter("marker_length", 0.1)
        self.declare_parameter("marker_positions_file", "")
        self.declare_parameter("camera_params_file", "")
        self.declare_parameter("height_over_ground", 0.417)
        self.declare_parameter("publish_detections", False)

        # Get parameters
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
        pose_pub_topic = (
            self.get_parameter("pose_pub_topic").get_parameter_value().string_value
        )
        aruco_dict_name = (
            self.get_parameter("aruco_dict").get_parameter_value().string_value
        )
        self.marker_length = (
            self.get_parameter("marker_length").get_parameter_value().double_value
        )
        marker_positions_file = (
            self.get_parameter("marker_positions_file")
            .get_parameter_value()
            .string_value
        )
        camera_params_file = (
            self.get_parameter("camera_params_file").get_parameter_value().string_value
        )
        self.height_over_ground = (
            self.get_parameter("height_over_ground").get_parameter_value().double_value
        )
        self.publish_detections = (
            self.get_parameter("publish_detections").get_parameter_value().bool_value
        )

        # Set up the ArUco dictionary
        aruco_dict_map = {
            "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
            "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
            "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
            # Add other dictionaries as needed
        }
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
            aruco_dict_map[aruco_dict_name]
        )
        self.aruco_params = cv2.aruco.DetectorParameters()

        self.get_logger().info(f"Loaded ArucoDict from {aruco_dict_name}\n")

        # Load marker positions from YAML file
        self.marker_positions = self.load_marker_positions(marker_positions_file)
        self.get_logger().info(
            f"Loaded {len(self.marker_positions)} marker positions from '{marker_positions_file}'"
        )

        for i, marker_postion in self.marker_positions.items():
            self.get_logger().info(f"\t{i}: {marker_postion}")

        # Set up subscriptions for odometry
        self.subscription_odom = self.create_subscription(
            Odometry, odom_topic, self.odom_callback, 10
        )
        self.get_logger().info(f"Subscribed to {odom_topic}")

        # Set up message filters for synchronizing multiple image topics
        self.bridge = CvBridge()
        image_subs = []

        # Load camera parameters from YAML file
        self.cameras = self.load_camera_parameters(camera_params_file)
        self.image_pubs = []
        self.get_logger().info(
            f"Loaded {len(self.cameras)} camera configurations from '{camera_params_file}'"
        )

        for i, cam in enumerate(self.cameras):
            image_subs.append(Subscriber(self, Image, cam["topic"]))
            self.get_logger().info(f"Subscribing to {cam['topic']}")
            if self.publish_detections:
                self.image_pubs.append(
                    self.create_publisher(Image, f"/aruco/image{i:02}", 10)
                )

        # Use ApproximateTimeSynchronizer to synchronize image messages
        self.ts = ApproximateTimeSynchronizer(image_subs, queue_size=10, slop=0.1)
        self.ts.registerCallback(self.image_callback)

        # Publisher
        self.pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, pose_pub_topic, 10
        )
        self.get_logger().info(f"Publishing to {pose_pub_topic}")
        
        self.count_pub = self.create_publisher(
            Int32, '/aruco/count', 10
        )

        self.current_odom = None
        self.last_callback_time = time.time()

        # Set up a timer to check callback frequency
        self.create_timer(0.1, self.check_callback_frequency)
        self.get_logger().info(f"Started watchdog")

    def load_marker_positions(self, filepath):
        try:
            with open(filepath, "r") as file:
                positions = yaml.safe_load(file)["markers"]
                # Convert to numpy arrays
                for key in positions:
                    positions[key] = np.array(positions[key])
                return positions
        except Exception as e:
            self.get_logger().error(f"Failed to load marker positions: {e}")
            return {}

    def load_camera_parameters(self, filepath):
        try:
            with open(filepath, "r") as file:
                params = yaml.safe_load(file)

            cameras = []
            for cam_key, cam_params in params.items():
                camera = {
                    "T_imu_cam": np.array(cam_params.get("T_imu_cam", [])),
                    "intrinsics": np.array(cam_params.get("intrinsics", [])),
                    "dist_coeffs": np.array(cam_params.get("distortion_coeffs", [])),
                    "topic": cam_params.get("topic", ""),
                    "camera_matrix": None,
                    "dist_coeffs": None,
                    "detected_markers": [],  # To store detected markers
                }
                if len(camera["intrinsics"]) == 4:
                    fx, fy, cx, cy = camera["intrinsics"]
                    camera["camera_matrix"] = np.array(
                        [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
                    )
                cameras.append(camera)
            return cameras

        except Exception as e:
            self.get_logger().error(f"Failed to load camera parameters: {e}")
            return []

    def odom_callback(self, msg: Odometry):
        self.current_odom = msg
        self.get_logger().info(
            f"Received odometry: {msg.pose.pose.position}, {msg.pose.pose.orientation}"
        )

    def image_callback(self, *msgs):
        if self.current_odom is None:
            self.get_logger().warn("Received image but have no odometry!")
            return

        self.get_logger().info(f"Received image")
        # Update the last callback time
        self.last_callback_time = time.time()

        # Initial guess based on odometry
        initial_position = np.array(
            [
                self.current_odom.pose.pose.position.x,
                self.current_odom.pose.pose.position.y,
                self.current_odom.pose.pose.position.z,
            ]
        )
        self.get_logger().info(f"Current position: {initial_position}")

        # Convert quaternion to Euler angles (roll, pitch, yaw) for initial orientation guess
        orientation_quat = self.current_odom.pose.pose.orientation
        initial_orientation = self.quaternion_to_euler(orientation_quat)
        self.get_logger().info(f"Current orientation: {initial_orientation}")

        # Combine position and orientation into a single initial guess
        initial_guess = np.hstack((initial_position, initial_orientation))

        # Collect all marker observations from all cameras
        observations = []
        tags = set()

        for j, (msg, cam) in enumerate(zip(msgs, self.cameras)):
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            # Detect ArUco markers
            corners, ids, _ = cv2.aruco.detectMarkers(
                cv_image, self.aruco_dict, parameters=self.aruco_params
            )

            if ids is None:
                continue
            self.get_logger().info(f"Found {len(ids)} aruco markers")

            if self.publish_detections:
                cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)

            for i, marker_id in enumerate(ids.flatten()):
                if not marker_id in self.marker_positions:
                    self.get_logger().warn(f"Found {marker_id}, not in known markers.")
                    continue
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners[i],
                    self.marker_length,
                    cam["camera_matrix"],
                    cam["dist_coeffs"],
                )
                tags.add(marker_id)
                self.get_logger().info(f"ID: {marker_id}, T: {tvec}, R: {rvec}")
                observations.append(
                    {
                        "marker_id": marker_id,
                        "rvec": rvec,
                        "tvec": tvec,
                        "corners": corners[i],
                        "T_imu_cam": cam["T_imu_cam"],
                        "camera_matrix": cam["camera_matrix"],
                        "dist_coeffs": cam["dist_coeffs"],
                    }
                )
                cv2.drawFrameAxes(cv_image, cam['camera_matrix'], cam['dist_coeffs'], rvec, tvec, 0.1)

            if self.publish_detections:
                aruco_msg = self.bridge.cv2_to_imgmsg(cv_image, header=msg.header)
                self.image_pubs[i].publish(aruco_msg)
            
        self.count_pub.publish(Int32(data=len(tags)))

        if len(observations) == 0:
            self.get_logger().warn(f"No aruco markers detected.")
            return

        # Bundle adjustment optimization to minimize reprojection error
        def reprojection_error(params):
            position = params[:3]
            orientation = params[3:]
            rotation_matrix = cv2.Rodrigues(orientation)[0]
            total_error = 0

            for obs in observations:
                marker_id = obs["marker_id"]
                rvec = obs["rvec"]
                tvec = obs["tvec"]
                corners = obs["corners"]
                T_imu_cam = obs["T_imu_cam"]
                camera_matrix = obs["camera_matrix"]
                dist_coeffs = obs["dist_coeffs"]

                marker_global_position = self.marker_positions[marker_id]

                marker_position_homogeneous = np.hstack(
                    (marker_global_position, [1])
                ).reshape(4, 1)
                marker_cam_frame_homogeneous = np.dot(
                    T_imu_cam, marker_position_homogeneous
                )
                marker_cam_frame = marker_cam_frame_homogeneous[:3].reshape(3)

                # Transform the marker position to the camera frame using the rotation and translation
                camera_to_marker = np.dot(rotation_matrix, marker_cam_frame) + position

                # Project the transformed 3D marker position onto the 2D image plane
                projected_points, _ = cv2.projectPoints(
                    camera_to_marker.reshape(-1, 3),
                    rvec,
                    tvec,
                    camera_matrix,
                    dist_coeffs,
                )
                error = np.sum(
                    np.linalg.norm(corners - projected_points.squeeze(), axis=1)
                )
                total_error += error

            return total_error

        # Perform the optimization
        result = least_squares(reprojection_error, initial_guess, x_scale="jac")
        refined_position_orientation = result.x
        self.get_logger().info(f"Optimized pose: {refined_position_orientation}")

        refined_position = refined_position_orientation[:3]
        refined_orientation = refined_position_orientation[3:]

        self.update_pose(refined_position, refined_orientation)

    def check_callback_frequency(self):
        current_time = time.time()
        time_since_last_callback = current_time - self.last_callback_time
        if time_since_last_callback > 0.5:
            self.get_logger().warn(
                f"Image callback hasn't been called for {time_since_last_callback:.2f} seconds!"
            )

    def update_pose(self, position, orientation):
        # Query the latest odom transform parameters
        offset_position, yaw = self.query_odom_transform()
        if offset_position is None or yaw is None:
            self.get_logger().error("Failed to retrieve odom transform parameters.")
            return

        # Apply the inverse transformation to the calculated position
        transformed_position = self.inverse_transform_position(
            position, np.array(offset_position), yaw
        )

        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "map"

        pose_msg.pose.pose.position.x = transformed_position[0]
        pose_msg.pose.pose.position.y = transformed_position[1]
        pose_msg.pose.pose.position.z = transformed_position[2]

        # Convert Euler angles back to quaternion for publishing
        pose_msg.pose.pose.orientation = self.euler_to_quaternion(orientation)

        self.pose_pub.publish(pose_msg)

    def quaternion_to_euler(self, quat):
        r = R.from_quat([quat.x, quat.y, quat.z, quat.w])
        return r.as_euler("xyz", degrees=False)

    def euler_to_quaternion(self, euler):
        r = R.from_euler("xyz", euler, degrees=False)
        quat = r.as_quat()
        return quat  # This returns [x, y, z, w]


def main(args=None):
    rclpy.init(args=args)
    node = ArucoPoseEstimator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
