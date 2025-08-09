from typing import Dict, List
import os
from scipy.spatial.transform import Rotation

import numpy as np
from rclpy.node import Node, Publisher, Subscription
import yaml

from std_msgs.msg import Header
from builtin_interfaces.msg import Time
from tf2_ros import transform_listener
from geometry_msgs.msg import (
    PointStamped,
    Point,
    PoseWithCovarianceStamped,
    PoseStamped,
    Transform,
)

from nav_msgs.msg import Odometry, Path


from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


def stamp_to_seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def point_to_numpy(point):
    return [point.x, point.y, point.z]


def apply_transform(transform: Transform, point: Point):
    tvec = np.array(
        [transform.translation.x, transform.translation.y, transform.translation.z]
    )
    rot = Rotation.from_quat(
        [
            transform.rotation.x,
            transform.rotation.y,
            transform.rotation.z,
            transform.rotation.w,
        ]
    )
    x, y, z = rot.apply([point.x, point.y, point.z]) + tvec
    point.x = x
    point.y = y
    point.z = z

    return point


def problem(transform, measurements, variances, goal_positions):
    # [x, y, z, qx, qy, qz, qw]
    tvec = transform[:3]
    rot = Rotation.from_rotvec(transform[3:6])  # Convert quaternion to rotation object

    transformed = rot.apply(measurements) + tvec

    error = transformed - goal_positions
    weighted_error = error / variances[:, np.newaxis]
    return np.sum(weighted_error**2)


class TransformOptimizer:

    def __init__(self, node: Node):
        path = os.environ.get("MARKER_CONFIG", "config/markers_fh.yml")

        initial_transform = (
            node.declare_parameter("initial_transform", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            .get_parameter_value()
            .double_array_value
        )

        self.rot = Rotation.from_rotvec(initial_transform[3:6])
        self.tvec = np.array(initial_transform[:3])

        offset = (
            node.declare_parameter("offset", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            .get_parameter_value()
            .double_array_value
        )

        self.offset_rot = Rotation.from_euler("XYZ", offset[3:6], degrees=True)
        self.offest_tvec = np.array(offset[:3])

        node.get_logger().info(f"Initial transform {initial_transform}")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, node)

        self.new_measurements = False

        self.transform_publisher = node.create_publisher(
            PoseStamped, "/optimizer/transform", 10
        )

        with open(path, "r") as file:
            self.marker_config = yaml.safe_load(file)

        self.node = node

        self.marker_positions = {}
        self.marker_variances = {}

        self.marker_subscriptions: Dict[int, Subscription] = {}
        self.marker_measurements: Dict[int, List[PointStamped]] = {
            marker_id: [] for marker_id in self.marker_config["markers"].keys()
        }

        self.transformed_pose_publisher = node.create_publisher(
            PoseWithCovarianceStamped, "/transformed/pose", 10
        )
        self.transformed_path_publisher = node.create_publisher(
            Path, "/transformed/path", 10
        )
        self.transformed_odom_publisher = node.create_publisher(
            Odometry, "/transformed/odom", 10
        )

        self.pose_sub = node.create_subscription(
            PoseWithCovarianceStamped,
            node.declare_parameter("pose_topic", "/mins/imu/pose")
            .get_parameter_value()
            .string_value,
            self.pose_callback,
            10,
        )

        self.odom_sub = node.create_subscription(
            Odometry,
            node.declare_parameter("odom_topic", "/mins/imu/odom")
            .get_parameter_value()
            .string_value,
            self.odom_callback,
            10,
        )
        self.path_sub = node.create_subscription(
            Path,
            node.declare_parameter("path_topic", "/mins/imu/path")
            .get_parameter_value()
            .string_value,
            self.path_callback,
            10,
        )

        for marker_id in self.marker_config["markers"].keys():
            topic = f"/aruco/marker_{marker_id:02}"
            self.marker_subscriptions[marker_id] = node.create_subscription(
                PointStamped,
                topic,
                lambda msg, id=marker_id: self.marker_callback(msg, id),
                10,
            )

        self.pose_publishers: Dict[int, Publisher] = {}
        self.corrected_marker_pub: Dict[int, Publisher] = {}
        self.original_marker_pub: Dict[int, Publisher] = {}

        for i in self.marker_config["markers"].keys():
            self.pose_publishers[i] = node.create_publisher(
                PoseWithCovarianceStamped, f"/aruco/marker_{i:02}_filtered", 10
            )
            self.corrected_marker_pub[i] = node.create_publisher(
                PointStamped, f"/aruco/marker_{i:02}_corrected", 10
            )
            self.original_marker_pub[i] = node.create_publisher(
                PointStamped, f"/aruco/marker_{i:02}_original", 10
            )

        self.timer = node.create_timer(0.1, self.optimizer_callback)

        self.null_time = Time()

    def transform(self, point: Point):
        """
        Apply the current transform to a point.
        """
        transformed_point = self.rot.apply([point.x, point.y, point.z]) + self.tvec

        point.x = transformed_point[0]
        point.y = transformed_point[1]
        point.z = transformed_point[2]
        return point

    def pose_callback(self, msg: PoseWithCovarianceStamped):
        # Apply the transform to the pose
        msg.pose.pose.position = self.transform(msg.pose.pose.position)

        # rotate the orientation with the offset rotation
        self.transformed_pose_publisher.publish(msg)

    def odom_callback(self, msg: Odometry):
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header = msg.header
        pose_msg.pose.pose = msg.pose.pose

        # Apply the transform to the pose
        pose_msg.pose.pose.position = self.transform(pose_msg.pose.pose.position)

        self.transformed_pose_publisher.publish(pose_msg)

    def path_callback(self, msg: Path):
        transformed_path = Path()
        transformed_path.header = msg.header

        for pose in msg.poses:
            transformed_pose = PoseStamped()
            transformed_pose.header = pose.header
            transformed_pose.pose = pose.pose
            transformed_pose.pose.position = self.transform(
                transformed_pose.pose.position,
            )
            transformed_path.poses.append(transformed_pose)
        self.transformed_path_publisher.publish(transformed_path)

    def optimizer_callback(self):
        self.optimize_transform()
        self.publish_transform()

    def publish_transform(self):
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.node.get_clock().now().to_msg()
        pose_msg.header.frame_id = "global"

        pose_msg.pose.position.x = self.tvec[0]
        pose_msg.pose.position.y = self.tvec[1]
        pose_msg.pose.position.z = self.tvec[2]

        quat = self.rot.as_quat()
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]

        self.transform_publisher.publish(pose_msg)

    def optimize_transform(self):
        if not self.new_measurements:
            return

        if len(self.marker_positions) < 2:
            return
        measurements = np.array(
            [self.marker_positions[i] for i in self.marker_positions.keys()]
        )
        variances = np.array(
            [self.marker_variances[i] for i in self.marker_variances.keys()]
        )
        goal_positions = np.array(
            [self.marker_config["markers"][i] for i in self.marker_positions.keys()]
        )

        self.new_measurements = False

        self.node.get_logger().info(
            "Starting optimization with " f"{len(measurements)} measurements."
        )
        self.node.get_logger().info(f"Goal positions: {goal_positions}")
        self.node.get_logger().info(f"Measurements: {measurements}")

        initial_transform = np.concatenate([self.tvec, self.rot.as_rotvec()])
        from scipy.optimize import minimize

        result = minimize(
            problem,
            initial_transform,
            args=(measurements, variances, goal_positions),
            method="L-BFGS-B",
        )
        if not result.success:
            self.node.get_logger().error("Optimization failed: " + result.message)
            return
        self.node.get_logger().info("Optimization successful.")
        self.node.get_logger().info(f"Optimized transform: {result.x}")

        # Update the transform with the optimized values
        optimized_transform = result.x
        self.tvec = optimized_transform[:3]
        self.rot = Rotation.from_rotvec(optimized_transform[3:7])

        for marker_id in self.marker_positions.keys():
            self.publish_corrected_marker(marker_id)

    def marker_callback(self, msg: PointStamped, marker_id: int):
        self.node.get_logger().info(
            f"Received marker {marker_id} measurement: {msg.point}"
        )
        try:
            transform = self.tf_buffer.lookup_transform("global", "imu", self.null_time)

            self.node.get_logger().info(f"Applying transform: {transform}")
            msg.point = apply_transform(transform.transform, msg.point)
            self.node.get_logger().info(
                f"Transformed marker {marker_id} position: {msg.point}"
            )
            self.marker_measurements[marker_id].append(msg)
            self.new_measurements = True
        except Exception as ex:
            self.node.get_logger().error(f"Error: {ex}")
            raise ex

        self.estimate_marker_position(marker_id)
        try:
            self.publish_marker_pose(marker_id)
        except Exception as ex:
            self.node.get_logger().error(
                f"Error publishing pose for marker {marker_id}: {ex}"
            )

    def estimate_marker_position(self, id):
        measurements = self.marker_measurements[id]
        if not measurements:
            self.node.get_logger().warn(f"No measurements for marker {id}.")
            return None

        # calculate weighted average based on timestamps
        timestamps = np.array([stamp_to_seconds(m.header.stamp) for m in measurements])
        points = np.array([point_to_numpy(m.point) for m in measurements])

        # larger timestamp means more recent measurement, heigher weight
        weights = 1 / (timestamps - timestamps.min() + 1e-6)
        weights /= np.sum(weights)
        weighted_average = np.average(points, axis=0, weights=weights)
        latest_measurement = points[-1]
        self.node.get_logger().info(
            f"Estimated position for marker {id}: {weighted_average} ({latest_measurement})"
        )
        self.marker_positions[id] = weighted_average

        # calculate variance as the weighted average of squared distances from the mean
        squared_distances = np.sum((points - weighted_average) ** 2, axis=1)
        weighted_variance = np.average(squared_distances, weights=weights)
        self.marker_variances[id] = weighted_variance

    def publish_corrected_marker(self, marker_id: int):
        if marker_id not in self.marker_positions:
            self.node.get_logger().warn(
                f"No position estimated for marker {marker_id}."
            )
            return

        corrected_msg = PointStamped()
        corrected_msg.header.stamp = self.node.get_clock().now().to_msg()
        corrected_msg.header.frame_id = "global"

        position = self.marker_positions[marker_id]
        position = self.rot.apply(position) + self.tvec
        corrected_msg.point.x = position[0]
        corrected_msg.point.y = position[1]
        corrected_msg.point.z = position[2]

        self.corrected_marker_pub[marker_id].publish(corrected_msg)

        original_msg = PointStamped()
        original_msg.header.stamp = self.node.get_clock().now().to_msg()
        original_msg.header.frame_id = "global"
        original_msg.point.x = self.marker_config["markers"][marker_id][0]
        original_msg.point.y = self.marker_config["markers"][marker_id][1]
        original_msg.point.z = self.marker_config["markers"][marker_id][2]

        self.original_marker_pub[marker_id].publish(original_msg)

        self.node.get_logger().info(
            f"Published corrected marker {marker_id}: Position: {position}"
        )

    def publish_marker_pose(self, marker_id: int):
        if marker_id not in self.marker_positions:
            self.node.get_logger().warn(
                f"No position estimated for marker {marker_id}."
            )
            return

        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = self.node.get_clock().now().to_msg()
        pose_msg.header.frame_id = "global"

        position = self.marker_positions[marker_id]
        pose_msg.pose.pose.position.x = position[0]
        pose_msg.pose.pose.position.y = position[1]
        pose_msg.pose.pose.position.z = position[2]

        # Set orientation to identity (no rotation)
        pose_msg.pose.pose.orientation.w = 1.0
        pose_msg.pose.pose.orientation.x = 0.0
        pose_msg.pose.pose.orientation.y = 0.0
        pose_msg.pose.pose.orientation.z = 0.0

        # Set covariance based on variance
        variance = self.marker_variances.get(marker_id, 0.1)
        pose_msg.pose.covariance = [variance] * 36
        pose_msg.pose.covariance[0] = variance  # x position variance
        pose_msg.pose.covariance[7] = variance  # y position variance
        pose_msg.pose.covariance[14] = variance  # z position variance

        self.pose_publishers[marker_id].publish(pose_msg)

        self.node.get_logger().info(
            f"Published pose for marker {marker_id}: "
            f"Position: {position}, Variance: {variance}"
        )


def main(args=None):
    import rclpy
    from rclpy.executors import SingleThreadedExecutor

    rclpy.init(args=args)
    node = Node("optimizer_transform_node")

    optimizer = TransformOptimizer(node)

    executor = SingleThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
