from typing import Optional
import threading
import gtsam
import numpy as np
from scipy.spatial.transform import Rotation as R
from collections import deque

from gtsam.symbol_shorthand import X, L

from rclpy.node import Node
from geometry_msgs.msg import PointStamped, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry


class ArucoPoseTracker:

    @staticmethod
    def get_isam(node: Node) -> gtsam.ISAM2:
        """Get an instance of ISAM2 with parameters from the ROS node."""
        parameters = gtsam.ISAM2Params()
        parameters.setRelinearizeThreshold(
            node.declare_parameter("relinearize_threshold", 0.05)
            .get_parameter_value()
            .double_value
        )

        parameters.cacheLinearizedFactors = True
        parameters.relinearizeSkip = (
            node.declare_parameter("relinearize_skip", 1)
            .get_parameter_value()
            .integer_value
        )

        return gtsam.ISAM2(parameters)

    @staticmethod
    def msg_to_gtsam_pose(msg: PoseWithCovarianceStamped) -> gtsam.Pose3:
        """Convert a PoseStamped message to a GTSAM Pose3."""
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation

        return gtsam.Pose3(
            gtsam.Rot3.Quaternion(
                orientation.w, orientation.x, orientation.y, orientation.z
            ),
            gtsam.Point3(position.x, position.y, position.z),
        )

    @staticmethod
    def get_doubles(node: Node, name: str, default: Optional[list] = None) -> list:
        """Get a list of doubles from the ROS node parameters."""
        if default is None:
            default = []
        return (
            node.declare_parameter(name, default)
            .get_parameter_value()
            .double_array_value
        )

    @staticmethod
    def msg_to_gtsam_point(msg: PointStamped) -> gtsam.Point3:
        """Convert a PointStamped message to a GTSAM Point3."""
        position = msg.point
        return gtsam.Point3(position.x, position.y, position.z)

    def __init__(self, node: Node, marker_positions: dict):
        self.node = node
        self.marker_positions = marker_positions

        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimate = gtsam.Values()

        self.max_poses = (
            node.declare_parameter("max_poses", 50).get_parameter_value().integer_value
        )
        self.pose_keys = deque(maxlen=self.max_poses * 2)  # Some buffer
        self.current_pose = self._setup_initial_pose()
        self._setup_markers()

        self.isam = ArucoPoseTracker.get_isam(node)
        self.isam.update(self.graph, self.initial_estimate)
        self.initial_estimate.clear()

        self.last_pose = None

        self.pose_noise = gtsam.noiseModel.Diagonal.Sigmas(
            ArucoPoseTracker.get_doubles(
                node, "pose_noise", [0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
            )
        )
        self.observation_noise = gtsam.noiseModel.Diagonal.Sigmas(
            ArucoPoseTracker.get_doubles(node, "observation_noise", [0.1, 0.1, 0.5])
        )

        self.pose_idx = 0

        # Sliding window parameters
        self.marginalization_frequency = (
            node.declare_parameter("marginalization_frequency", 30)
            .get_parameter_value()
            .integer_value
        )
        self.poses_to_marginalize = (
            node.declare_parameter("poses_to_marginalize", 10)
            .get_parameter_value()
            .integer_value
        )

        # Keep track of pose keys for marginalization
        self.marginalization_counter = 0

        # Batch updates for efficiency
        self.batch_size = (
            node.declare_parameter("batch_size", 2).get_parameter_value().integer_value
        )
        self.update_counter = 0

        self.graph_lock = threading.Lock()

        # Setup publishers and subscribers
        self.pose_subscriber = node.create_subscription(
            PoseWithCovarianceStamped,
            self.node.declare_parameter("pose_topic", "/mins/imu/pose")
            .get_parameter_value()
            .string_value,
            self.pose_callback,
            10,
        )

        for marker_id in self.marker_positions:
            topic = f"/aruco/marker_{marker_id:02}"
            node.create_subscription(
                PointStamped,
                topic,
                lambda msg, id=marker_id: self.marker_callback(msg, id),
                10,
            )

        self.publisher = node.create_publisher(
            PoseStamped,
            self.node.declare_parameter("pose_publish_topic", "/current_pose")
            .get_parameter_value()
            .string_value,
            10,
        )

        self.publisher_timer = node.create_timer(
            self.node.declare_parameter("publish_rate", 0.1)
            .get_parameter_value()
            .double_value,
            self.publisher_callback,
        )

    def pose_callback(self, msg: PoseStamped):
        """Callback for receiving pose messages."""
        pose = ArucoPoseTracker.msg_to_gtsam_pose(msg)

        if self.last_pose is None:
            self.last_pose = pose
            return

        relative_transform = self.last_pose.between(pose)
        self.last_pose = pose
        self.current_pose = self.current_pose.compose(relative_transform)

        with self.graph_lock:
            self.pose_idx += 1
            current_key = X(self.pose_idx)

            odometry_factor = gtsam.BetweenFactorPose3(
                X(self.pose_idx - 1),
                current_key,
                relative_transform,
                self.pose_noise,
            )

            self.graph.add(odometry_factor)
            self.initial_estimate.insert(current_key, self.current_pose)
            self.pose_keys.append(current_key)

            self.update_counter += 1

            # Check if we need to marginalize old poses
            if len(self.pose_keys) > self.max_poses:
                self._marginalize_old_poses()

    def marker_callback(self, msg: PointStamped, marker_id: int):
        """Callback for receiving marker position messages."""
        position = ArucoPoseTracker.msg_to_gtsam_point(msg)
        bearing = gtsam.Unit3(position)

        with self.graph_lock:
            pose_key = X(self.pose_idx)
            marker_key = L(marker_id)

            marker_factor = gtsam.BearingRangeFactor3D(
                pose_key,
                marker_key,
                bearing,
                np.linalg.norm(position),
                self.observation_noise,
            )
            self.graph.add(marker_factor)
            self.update_counter += 1

    def _marginalize_old_poses(self):
        """Marginalize old poses to keep the optimization efficient."""
        if len(self.pose_keys) <= self.max_poses:
            return

        # Marginalize oldest poses
        keys_to_marginalize = []
        for _ in range(
            min(
                self.poses_to_marginalize,
                len(self.pose_keys) - self.max_poses + self.poses_to_marginalize,
            )
        ):
            if self.pose_keys:
                old_key = self.pose_keys.popleft()
                keys_to_marginalize.append(old_key)

        if keys_to_marginalize:
            try:
                # Update first to ensure all factors are in the system
                if not self.graph.empty():
                    self.isam.update(self.graph, self.initial_estimate)
                    self.graph.resize(0)
                    self.initial_estimate.clear()

                # Marginalize old variables
                marginals = gtsam.Marginals(
                    self.isam.getFactorsUnsafe(), self.isam.calculateEstimate()
                )
                for key in keys_to_marginalize:
                    if self.isam.calculateEstimate().exists(key):
                        # Create a prior factor for the marginalized pose to maintain connectivity
                        pose = self.isam.calculateEstimate().atPose3(key)
                        covariance = marginals.marginalCovariance(key)
                        noise_model = gtsam.noiseModel.Gaussian.Covariance(covariance)

                        # Don't actually remove, just add a strong prior to "freeze" the pose
                        # This is a simpler approach than full marginalization
                        prior_factor = gtsam.PriorFactorPose3(key, pose, noise_model)
                        self.graph.add(prior_factor)

            except Exception as e:
                self.node.get_logger().warning(f"Marginalization failed: {e}")
                # Re-add keys back if marginalization failed
                for key in reversed(keys_to_marginalize):
                    self.pose_keys.appendleft(key)

    def publisher_callback(self):
        """Publish the current pose and marker positions."""
        t0 = self.node.get_clock().now()

        with self.graph_lock:
            # Only update if we have accumulated enough factors or it's been a while
            should_update = (
                self.update_counter >= self.batch_size or not self.graph.empty()
            )

            if should_update:
                self.isam.update(self.graph, self.initial_estimate)
                self.graph.resize(0)
                self.initial_estimate.clear()
                self.update_counter = 0

            current_estimate = self.isam.calculateEstimate()

        if not current_estimate.exists(X(self.pose_idx)):
            return

        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.node.get_clock().now().to_msg()
        pose_msg.header.frame_id = "global"

        pose = current_estimate.atPose3(X(self.pose_idx))
        position = pose.translation()
        orientation = R.from_matrix(pose.rotation().matrix()).as_quat()

        x, y, z = position
        qx, qy, qz, qw = orientation

        pose_msg.pose.position.x = x
        pose_msg.pose.position.y = y
        pose_msg.pose.position.z = z
        pose_msg.pose.orientation.x = qx
        pose_msg.pose.orientation.y = qy
        pose_msg.pose.orientation.z = qz
        pose_msg.pose.orientation.w = qw

        self.publisher.publish(pose_msg)

        optimization_time = (self.node.get_clock().now() - t0).nanoseconds / 1e6
        if optimization_time > 10:  # Log if it takes more than 10ms
            self.node.get_logger().info(f"Optimization took {optimization_time:.2f} ms")

    def _setup_initial_pose(self):
        position_init = ArucoPoseTracker.get_doubles(
            self.node, "initial_position", [0.0, 0.0, 0.0]
        )
        orientation_init = ArucoPoseTracker.get_doubles(
            self.node,
            "initial_orientation",
            [6.08969028e-17, 1.04528463e-01, -6.40052240e-18, 9.94521895e-01],
        )

        position = gtsam.Point3(*position_init)
        orientation = gtsam.Rot3.Quaternion(*orientation_init)
        pose = gtsam.Pose3(orientation, position)

        key = X(0)
        self.pose_keys.append(key)

        pose_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.01, 0.01, 0.01, 10, 10, 10])
        )
        prior_pose_factor = gtsam.PriorFactorPose3(key, pose, pose_prior_noise)

        self.graph.add(prior_pose_factor)
        self.initial_estimate.insert(key, pose)

        return pose

    def _setup_markers(self):
        """Setup markers in the graph."""
        for marker_id, position in self.marker_positions.items():
            key = L(marker_id)
            position = gtsam.Point3(position["x"], position["y"], position["z"])
            prior_factor = gtsam.PriorFactorPoint3(
                key,
                position,
                gtsam.noiseModel.Diagonal.Sigmas([0.0001, 0.0001, 0.0001]),
            )

            self.graph.add(prior_factor)
            self.initial_estimate.insert(key, position)


def main():
    import rclpy
    from rclpy.node import Node

    rclpy.init()
    node = Node("aruco_pose_tracker")

    marker_positions = {
        51: {"x": -25.48, "y": 1.22, "z": -0.01},
        52: {"x": -4.341, "y": 8.83, "z": 0.02},
        53: {"x": -18.79, "y": 7.10, "z": -0.15},
        55: {"x": -9.338, "y": 1.88, "z": -0.65},
    }

    tracker = ArucoPoseTracker(node, marker_positions)

    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
