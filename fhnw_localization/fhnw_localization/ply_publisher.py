import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import numpy as np
import struct
from plyfile import PlyData


class PointCloudPublisher(Node):
    def __init__(self):
        super().__init__("pointcloud_publisher")

        # Create publisher
        self.publisher = self.create_publisher(PointCloud2, "pointcloud", 10)

        # Timer to publish at regular intervals (1 Hz)
        self.timer = self.create_timer(30.0, self.publish_pointcloud)

        # Load point cloud data from PLY file
        self.pointcloud_data = self.load_ply_file("data/fhnw_my.ply")

        self.get_logger().info("PointCloud Publisher Node started")

    def load_ply_file(self, filename):
        """Load point cloud data from PLY file"""
        try:
            plydata = PlyData.read(filename)
            vertex = plydata["vertex"]

            # Extract x, y, z coordinates
            points = np.array(
                [[x, y, z] for x, y, z in zip(vertex["x"], vertex["y"], vertex["z"])],
                dtype=np.float32,
            )

            self.get_logger().info(f"Loaded {len(points)} points from {filename}")
            return points

        except Exception as e:
            self.get_logger().error(f"Failed to load PLY file: {e}")
            # Return dummy data if file loading fails
            return np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

    def create_pointcloud2_msg(self, points):
        """Convert numpy array to PointCloud2 message"""

        # Create header
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "world"  # Change this to your desired frame

        # Define point fields
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        # Create PointCloud2 message
        cloud_msg = PointCloud2()
        cloud_msg.header = header
        cloud_msg.height = 1
        cloud_msg.width = len(points)
        cloud_msg.fields = fields
        cloud_msg.is_bigendian = False
        cloud_msg.point_step = 12  # 3 floats * 4 bytes each
        cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
        cloud_msg.is_dense = True

        # Pack point data
        cloud_data = []
        for point in points:
            cloud_data.extend(struct.pack("fff", point[0], point[1], point[2]))

        cloud_msg.data = bytes(cloud_data)

        return cloud_msg

    def publish_pointcloud(self):
        """Publish the point cloud"""
        if self.pointcloud_data is not None:
            msg = self.create_pointcloud2_msg(self.pointcloud_data)
            self.publisher.publish(msg)
            self.get_logger().info("Published point cloud")


def main(args=None):
    rclpy.init(args=args)

    node = PointCloudPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
