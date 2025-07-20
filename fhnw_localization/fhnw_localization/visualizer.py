#!/usr/bin/env python3
from rclpy.qos import QoSProfile, DurabilityPolicy

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import TransformStamped, PoseStamped, PoseWithCovarianceStamped
from tf2_ros import StaticTransformBroadcaster, Buffer, TransformListener
import yaml
import math
from typing import Dict, List, Any


class MeshMarkerPublisher(Node):
    def __init__(self):
        super().__init__("mesh_marker_publisher")

        # Declare parameters
        self.declare_parameter("config_file", "config/meshes.yaml")

        # Get config file path
        config_file = (
            self.get_parameter("config_file").get_parameter_value().string_value
        )

        # Load configuration
        self.config = self.load_config(config_file)

        # Initialize TF components
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Storage for publishers and subscribers
        self.marker_publishers = {}
        self.pose_subscribers = {}
        self.static_markers = {}
        self.dynamic_markers = {}

        # Publish static transforms
        self.publish_static_transforms()

        # Setup mesh markers
        self.setup_mesh_markers()

        # Timer for publishing static markers
        self.timer = self.create_timer(10, self.publish_static_markers)
        self.publish_static_markers()

        self.get_logger().info("Mesh marker publisher initialized")

    def euler_to_quaternion(self, roll: float, pitch: float, yaw: float) -> tuple:
        """Convert Euler angles (roll, pitch, yaw) to quaternion (x, y, z, w)"""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return (x, y, z, w)

    def load_config(self, config_file: str) -> Dict[str, Any]:
        """Load YAML configuration file"""
        try:
            with open(config_file, "r") as file:
                config = yaml.safe_load(file)
                self.get_logger().info(f"Loaded config from {config_file}")
                return config
        except Exception as e:
            self.get_logger().error(f"Failed to load config file {config_file}: {e}")
            return {"resources": [], "transforms": []}

    def publish_static_transforms(self):
        """Publish static transforms from config"""
        if "transforms" not in self.config:
            return

        transforms = []
        for transform_config in self.config["transforms"]:
            transform = TransformStamped()

            transform.header.stamp = self.get_clock().now().to_msg()
            transform.header.frame_id = transform_config.get("parent_frame", "map")
            transform.child_frame_id = transform_config["frame"]

            # Translation
            translation = transform_config.get("translation", {"x": 0, "y": 0, "z": 0})
            transform.transform.translation.x = float(translation["x"])
            transform.transform.translation.y = float(translation["y"])
            transform.transform.translation.z = float(translation["z"])

            # Rotation (quaternion or RPY)
            rotation = transform_config.get(
                "rotation", {"x": 0, "y": 0, "z": 0, "w": 1}
            )
            if "roll" in rotation or "pitch" in rotation or "yaw" in rotation:
                # Convert RPY to quaternion using built-in function
                roll = rotation.get("roll", 0)
                pitch = rotation.get("pitch", 0)
                yaw = rotation.get("yaw", 0)
                q = self.euler_to_quaternion(roll, pitch, yaw)
                transform.transform.rotation.x = q[0]
                transform.transform.rotation.y = q[1]
                transform.transform.rotation.z = q[2]
                transform.transform.rotation.w = q[3]
            else:
                # Use quaternion directly
                transform.transform.rotation.x = float(rotation["x"])
                transform.transform.rotation.y = float(rotation["y"])
                transform.transform.rotation.z = float(rotation["z"])
                transform.transform.rotation.w = float(rotation["w"])

            transforms.append(transform)

        if transforms:
            self.static_tf_broadcaster.sendTransform(transforms)

    @staticmethod
    def convert_pose_with_cov_to_pose(msg: PoseWithCovarianceStamped) -> PoseStamped:
        """Convert PoseWithCovarianceStamped to PoseStamped"""
        pose_stamped = PoseStamped()
        pose_stamped.header = msg.header
        pose_stamped.pose = msg.pose.pose
        return pose_stamped

    def setup_mesh_markers(self):
        """Setup mesh markers from config"""
        if "resources" not in self.config:
            return
        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,  # This makes it latched
        )
        for i, resource in enumerate(self.config["resources"]):
            marker_id = i
            topic = resource.get("topic", f"/mesh_marker_{i}")

            # Create publisher for this marker
            self.marker_publishers[marker_id] = self.create_publisher(
                Marker, topic, latched_qos
            )

            # Create base marker
            marker = self.create_mesh_marker(resource, marker_id)

            # Check if this marker needs dynamic pose updates
            pose_topic = resource.get("pose_topic")
            covariance = resource.get("has_covariance", False)
            if pose_topic:
                # Dynamic marker - subscribe to pose updates
                self.dynamic_markers[marker_id] = marker
                self.pose_subscribers[marker_id] = self.create_subscription(
                    PoseWithCovarianceStamped if covariance else PoseStamped,
                    pose_topic,
                    lambda msg, mid=marker_id: self.pose_callback(
                        self.convert_pose_with_cov_to_pose(msg) if covariance else msg,
                        mid,
                    ),
                    10,
                )
                self.get_logger().info(
                    f"Created dynamic marker {marker_id} subscribing to {pose_topic}"
                )
            else:
                # Static marker
                self.static_markers[marker_id] = marker
                self.get_logger().info(f"Created static marker {marker_id}")

    def create_mesh_marker(self, resource: Dict[str, Any], marker_id: int) -> Marker:
        """Create a mesh marker from resource config"""
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = resource["frame"]
        marker.ns = resource.get("namespace", "meshes")
        marker.id = marker_id
        marker.type = Marker.MESH_RESOURCE
        marker.action = Marker.ADD

        # Mesh resource URL
        marker.mesh_resource = resource["path"]
        marker.mesh_use_embedded_materials = resource.get("use_materials", True)

        # Scale
        scale = resource.get("scale", {"x": 1.0, "y": 1.0, "z": 1.0})
        marker.scale.x = float(scale.get("x", 1.0))
        marker.scale.y = float(scale.get("y", 1.0))
        marker.scale.z = float(scale.get("z", 1.0))

        # Color (only used if mesh doesn't have embedded materials)
        color = resource.get("color", {"r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0})
        marker.color.r = float(color.get("r", 1.0))
        marker.color.g = float(color.get("g", 1.0))
        marker.color.b = float(color.get("b", 1.0))
        marker.color.a = float(color.get("a", 1.0))

        # Initial pose (for static markers or default for dynamic ones)
        pose = resource.get("pose", {})
        position = pose.get("position", {"x": 0, "y": 0, "z": 0})
        orientation = pose.get("orientation", {"x": 0, "y": 0, "z": 0, "w": 1})

        # Handle orientation as either quaternion or RPY
        if "roll" in orientation or "pitch" in orientation or "yaw" in orientation:
            roll = orientation.get("roll", 0)
            pitch = orientation.get("pitch", 0)
            yaw = orientation.get("yaw", 0)
            q = self.euler_to_quaternion(roll, pitch, yaw)
            marker.pose.orientation.x = q[0]
            marker.pose.orientation.y = q[1]
            marker.pose.orientation.z = q[2]
            marker.pose.orientation.w = q[3]
        else:
            marker.pose.orientation.x = float(orientation["x"])
            marker.pose.orientation.y = float(orientation["y"])
            marker.pose.orientation.z = float(orientation["z"])
            marker.pose.orientation.w = float(orientation["w"])

        marker.pose.position.x = float(position["x"])
        marker.pose.position.y = float(position["y"])
        marker.pose.position.z = float(position["z"])

        return marker

    def pose_callback(self, msg: PoseStamped, marker_id: int):
        """Handle pose updates for dynamic markers"""
        self.get_logger().info(
            f"Received pose update for marker {marker_id}: {msg.pose}"
        )
        if marker_id in self.dynamic_markers:
            marker = self.dynamic_markers[marker_id]

            # Update marker pose
            marker.header.stamp = msg.header.stamp
            marker.pose = msg.pose

            # Publish updated marker
            self.marker_publishers[marker_id].publish(marker)

    def publish_static_markers(self):
        """Publish static markers periodically"""
        current_time = self.get_clock().now().to_msg()

        for marker_id, marker in self.static_markers.items():
            marker.header.stamp = current_time
            self.marker_publishers[marker_id].publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = MeshMarkerPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
