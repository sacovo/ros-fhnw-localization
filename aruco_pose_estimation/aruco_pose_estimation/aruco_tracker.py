from typing import List, Set, Tuple
import time
import yaml
from rclpy.executors import MultiThreadedExecutor
import numpy as np

from scipy.spatial.transform import Rotation as R
import cv2
from scipy.optimize import least_squares
from std_msgs.msg import Int32, Header
from geometry_msgs.msg import (
    PoseWithCovarianceStamped,
    Pose,
    PoseStamped,
    Point,
    Quaternion,
)
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from message_filters import ApproximateTimeSynchronizer, Subscriber
import time

import tf2_ros
import tf_transformations

from rclpy.parameter import Parameter
from rcl_interfaces.srv import GetParameters

from dataclasses import dataclass


@dataclass
class Camera:
    name: str
    T_imu_cam: np.ndarray
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    topic: str

@dataclass
class TagObservation:
    marker_id: int
    rvec: np.ndarray
    tvec: np.ndarray
    corners: np.ndarray
    T_imu_cam: np.ndarray
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    cam: Camera

class ArucoProcessor:
    def __init__(self, aruco_dict_name, marker_length, marker_positions, camera_params):
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
        self.marker_length = marker_length
        self.marker_positions = marker_positions
        self.camera_params = camera_params

    def detect_markers(self, cv_image):
        corners, ids, _ = cv2.aruco.detectMarkers(
            cv_image, self.aruco_dict, parameters=self.aruco_params
        )
        return corners, ids

    def estimate_pose(self, corners, ids, cam: Camera, image) -> Tuple[List[TagObservation], Set]:
        observations = []
        tags = set()

        for i, marker_id in enumerate(ids.flatten()):
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners[i],
                self.marker_length,
                cam.camera_matrix,
                cam.dist_coeffs,
            )

            if marker_id not in self.marker_positions:
                continue

            cv2.drawFrameAxes(image, cam.camera_matrix, cam.dist_coeffs, rvec, tvec, 0.1)
            tags.add(marker_id)
            observations.append(
                TagObservation(
                    marker_id=marker_id,
                    rvec=rvec,
                    tvec=tvec,
                    corners=corners[i],
                    cam=cam,
                )
            )

        return observations, tags

    @staticmethod
    def reprojection_error(params, observations: List[TagObservation], marker_positions):
        position = params[:3]
        orientation = params[3:]
        rotation_matrix = cv2.Rodrigues(orientation)[0]
        total_error = 0

        for obs in observations:
            marker_id = obs.marker_id
            corners = obs.corners
            tvec = obs.tvec
            rvec = obs.rvec
            T_imu_cam = obs.cam.T_imu_cam
            camera_matrix = obs.cam.camera_matrix
            dist_coeffs = obs.cam.dist_coeffs

            marker_global_position = marker_positions[marker_id]
            rot = R.from_matrix(T_imu_cam[:3, :3]) * R.from_euler("xyz", orientation)
            
            # Project the transformed 3D marker position onto the 2D image plane
            projected_points, _ = cv2.projectPoints(
                marker_global_position,
                rot.as_matrix(),
                position + (T_imu_cam[:3, 3]),
                camera_matrix,
                dist_coeffs,
            )
            
            center = np.mean(corners, axis=1)

            error = np.sum(
                np.linalg.norm(center - projected_points.squeeze(), axis=1)
            )
            total_error += error

        return total_error

    def optimize_pose(self, initial_guess, observations: List[TagObservation]):
        # Wrap reprojection_error in a lambda to pass observations
        result = least_squares(
            ArucoProcessor.reprojection_error, 
            initial_guess, 
            x_scale="jac",
            kwargs={"observations": observations, "marker_positions": self.marker_positions},
        )
        refined_position_orientation = result.x
        refined_position = refined_position_orientation[:3]
        refined_orientation = refined_position_orientation[3:]
        
        return refined_position, refined_orientation

    @staticmethod
    def quaternion_to_euler(quat):
        r = R.from_quat([quat.x, quat.y, quat.z, quat.w])
        return r.as_euler("xyz", degrees=False)

    @staticmethod
    def euler_to_quaternion(euler):
        r = R.from_euler("xyz", euler, degrees=False)
        quat = r.as_quat()
        return quat  # This returns [x, y, z, w]

    @staticmethod
    def inverse_transform_position(position, offset_position, yaw):
        # Create the rotation matrix for the inverse yaw manually
        cos_yaw = np.cos(-yaw)
        sin_yaw = np.sin(-yaw)
        rotation_matrix = np.array(
            [[cos_yaw, -sin_yaw, 0], [sin_yaw, cos_yaw, 0], [0, 0, 1]]
        )

        # Convert the position to a vector
        pos_vector = np.array(position)

        # Apply the inverse rotation and then the inverse translation
        rotated_position = np.dot(rotation_matrix, pos_vector - offset_position)

        return rotated_position

    @staticmethod
    def get_global_position(tvec, rvec, T_imu_cam):
        """
        This method returns the global position of a detected tag using the camera's extrinsics
        and intrinsics.
        """
        # Convert the rotation vector to a rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rvec)

        # Compute the position of the marker in the camera frame
        marker_cam_frame = np.dot(rotation_matrix, tvec.T).T

        # Convert to homogeneous coordinates
        marker_cam_frame_homogeneous = np.hstack((marker_cam_frame[0], [1])).reshape(4, 1)

        # Transform the marker position to the global frame using the extrinsics
        marker_global_homogeneous = np.dot(np.linalg.inv(T_imu_cam), marker_cam_frame_homogeneous)
        
        # Return the global position as a 3D vector
        marker_global_position = marker_global_homogeneous[:3].reshape(3)
        return marker_global_position

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

        self.get_logger().info(f"Loaded ArucoDict from {aruco_dict_name}\n")

        # Load marker positions from YAML file
        self.marker_positions = self.load_marker_positions(marker_positions_file)

        # Create a publisher for each marker pose
        self.marker_pubs = {}
        for i, marker_position in self.marker_positions.items():
            self.marker_pubs[i] = self.create_publisher(
                PoseStamped, f"/aruco/pose{i:02}", 10
            )

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
            image_subs.append(Subscriber(self, Image, cam.topic))
            self.get_logger().info(f"Subscribing to {cam.topic}")
            if self.publish_detections:
                self.image_pubs.append(
                    self.create_publisher(Image, f"/aruco/image{i:02}", 10)
                )

        self.processor = ArucoProcessor(
            aruco_dict_name=aruco_dict_name,
            marker_length=self.marker_length,
            marker_positions=self.marker_positions,
            camera_params=self.cameras,
        )
        # Use ApproximateTimeSynchronizer to synchronize image messages
        self.ts = ApproximateTimeSynchronizer(image_subs, queue_size=10, slop=0.1)
        self.ts.registerCallback(self.image_callback)

        # Publisher
        self.pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, pose_pub_topic, 10
        )
        self.get_logger().info(f"Publishing to {pose_pub_topic}")

        self.count_pub = self.create_publisher(Int32, "/aruco/count", 10)

        self.current_odom = Odometry()
        self.last_callback_time = time.time()

        # Set up a timer to check callback frequency
        self.create_timer(0.1, self.check_callback_frequency)
        self.create_timer(1.0, self.query_odom_transform)
        self.get_logger().info(f"Started watchdog")

        self.yaw = 0.0
        self.offset_position = np.array([0.0, 0.0, 0.0])

        self.client = self.create_client(
            GetParameters, "/pose_transformer/get_parameters"
        )
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Waiting for /pose_transformer service...")

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

    @staticmethod
    def load_camera_parameters(filepath):
        with open(filepath, "r") as file:
            params = yaml.safe_load(file)

        cameras = []
        for cam_key, cam_params in params.items():
            camera = Camera(
               name=cam_key,
                T_imu_cam=np.array(cam_params.get("T_imu_cam", [])),
                camera_matrix=np.array(cam_params.get("intrinsics", [])),
                dist_coeffs=np.array(cam_params.get("distortion_coeffs", [])),
                topic=cam_params.get("topic", ""),
            )
            }
            if len(camera["intrinsics"]) == 4:
                fx, fy, cx, cy = camera.intrinsics
                camera.camera_matrix = np.array(
                    [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
                )
            cameras.append(camera)
        return cameras


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
        initial_orientation = ArucoProcessor.quaternion_to_euler(orientation_quat)
        self.get_logger().info(f"Current orientation: {initial_orientation}")

        # Combine position and orientation into a single initial guess
        initial_guess = np.hstack((initial_position, initial_orientation))

        # Collect all marker observations from all cameras
        observations = []
        tags = set()

        for j, (msg, cam) in enumerate(zip(msgs, self.cameras)):
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            corners, ids = self.processor.detect_markers(cv_image)
            if ids is None:
                self.get_logger().warn("No markers detected!")
                continue

            cam_observations, cam_tags = self.processor.estimate_pose(corners, ids, cam, cv_image)
            for obs in cam_observations:
                tvec = ArucoProcessor.get_global_position(obs.tvec[0], obs.rvec[0][0], obs.T_imu_cam)
                tvec = obs.tvec[0][0]

                quat = R.from_rotvec(obs.rvec[0][0]).as_quat()
                ypr = R.from_rotvec(obs.rvec[0][0]).as_euler("xyz", degrees=False)

                self.get_logger().info(
                    f"Detected marker {obs.marker_id} at {tvec}, {ypr}"
                )
                self.marker_pubs[obs.marker_id].publish(
                    PoseStamped(
                        pose=Pose(
                            position=Point(x=tvec[0], y=tvec[1], z=tvec[2]),
                            orientation=Quaternion(
                                x=quat[0], y=quat[1], z=quat[2], w=quat[3]
                            ),
                        ),
                        header=Header(frame_id="global", stamp=msg.header.stamp),
                    )
                )

            observations.extend(cam_observations)
            tags.update(cam_tags)

            if self.publish_detections:
                cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
                aruco_msg = self.bridge.cv2_to_imgmsg(cv_image, header=msg.header)
                self.image_pubs[j].publish(aruco_msg)

        self.count_pub.publish(Int32(data=len(tags)))
        if len(observations) == 0:
            self.get_logger().warn("No markers detected!")
            return
        t0 = time.time()
        refined_position, refined_orientation = self.processor.optimize_pose(
            initial_guess, observations
        )
        self.get_logger().info(f"Optimization took {time.time() - t0:.2f} seconds")
        
        self.get_logger().info(f"Refined position: {refined_position}, orientation: {refined_orientation}")

        self.update_pose(refined_position, refined_orientation)

    def check_callback_frequency(self):
        current_time = time.time()
        time_since_last_callback = current_time - self.last_callback_time
        if time_since_last_callback > 0.5:
            self.get_logger().warn(
                f"Image callback hasn't been called for {time_since_last_callback:.2f} seconds!"
            )

    def update_pose(self, position, orientation):
        # Apply the inverse transformation to the calculated position
        transformed_position = self.processor.inverse_transform_position(
            position, self.offset_position, self.yaw
        )

        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "global"

        pose_msg.pose.pose.position.x = transformed_position[0]
        pose_msg.pose.pose.position.y = transformed_position[1]
        pose_msg.pose.pose.position.z = transformed_position[2]

        # Convert Euler angles back to quaternion for publishing
        quat_array = self.processor.euler_to_quaternion(orientation)
        quat = Quaternion(
            x=quat_array[0], y=quat_array[1], z=quat_array[2], w=quat_array[3]
        )
        pose_msg.pose.pose.orientation = quat

        self.pose_pub.publish(pose_msg)

    def query_odom_transform(self):
        if not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("The /pose_transformer service is not available.")
            return None, None
        else:
            self.get_logger().info("The /pose_transformer service is available.")

        request = GetParameters.Request()
        request.names = ["initial_position", "yaw"]

        self.get_logger().info(f"Sending request to /pose_transformer {request}")

        future = self.client.call_async(request)

        future.add_done_callback(self.callback_odom_transform)

    def callback_odom_transform(self, future):
        response = future.result()
        self.get_logger().info(f"Received response from /pose_transformer {response}")
        self.position = response.values[0].double_array_value
        self.yaw = response.values[1].double_value


def main(args=None):
    rclpy.init(args=args)
    node = ArucoPoseEstimator()

    # Use MultiThreadedExecutor to handle multiple callbacks in parallel
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
