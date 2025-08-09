from collections import defaultdict
import cv2
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import PoseStamped, PointStamped
from sensor_msgs.msg import Image

from cv_bridge import CvBridge
from message_filters import TimeSynchronizer, Subscriber
from rclpy.node import Node
import numpy as np


MARKER_DEPTH = 0.12

MARKER_SIZE = 0.14  # Size of the ArUco marker in meters


def mono_aruco_detect(img, cm, dist, detector, marker_size=MARKER_SIZE):
    """
    Detect ArUco markers in a single camera image.

    Args:
        img: Input image (BGR format)
        cm: Camera matrix (3x3)
        dist: Distortion coefficients (1D array of length 5)
        detector: Initialized ArUco detector
    Returns:
        List of dictionaries containing marker info
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        return []

    ids = ids.flatten()
    markers = []

    t = np.array([[0, 0, marker_size]])  # Dummy translation vector

    for i, id in enumerate(ids):
        marker_info = {
            "id": int(id),
            "corners": corners[i][0],  # Shape: (4, 2)
        }

        # Estimate pose
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners[i], marker_size, cm, dist
        )
        # The marker pose is behinde the marker plane, so we need to add the rotated depth to tvec
        # print(tvec)

        marker_info["rvec"] = rvec[0]
        marker_info["tvec"] = tvec[0]

        markers.append(marker_info)

    return markers


def stereo_aruco_callback(
    img_left,
    img_right,
    cm_left,
    cm_right,
    dist_left,
    dist_right,
    R,
    t,
    detector,
    marker_size=MARKER_SIZE,
):
    """
    True stereo ArUco detection that leverages both cameras for improved accuracy.
    Uses stereo triangulation to get 3D corner positions and then estimates pose.

    Args:
        img_left: Left camera image
        img_right: Right camera image
        cm_left: Left camera matrix (3x3)
        cm_right: Right camera matrix (3x3)
        dist_left: Left camera distortion coefficients
        dist_right: Right camera distortion coefficients
        R: Rotation matrix from right to left camera (3x3)
        t: Translation vector from right to left camera (3x1)

    Returns:
        List of dictionaries containing marker info with stereo-derived poses
    """

    # Convert to grayscale
    gray_left = (
        cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        if len(img_left.shape) == 3
        else img_left
    )
    gray_right = (
        cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        if len(img_right.shape) == 3
        else img_right
    )

    # Detect markers in both images
    corners_left, ids_left, _ = detector.detectMarkers(gray_left)
    corners_right, ids_right, _ = detector.detectMarkers(gray_right)

    if ids_left is None or ids_right is None:
        return []

    ids_left = ids_left.flatten()
    ids_right = ids_right.flatten()

    # Create stereo rectification maps for better triangulation
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        cm_left, dist_left, cm_right, dist_right, gray_left.shape[::-1], R, t
    )

    # Create rectification maps
    map1_left, map2_left = cv2.initUndistortRectifyMap(
        cm_left, dist_left, R1, P1, gray_left.shape[::-1], cv2.CV_32FC1
    )
    map1_right, map2_right = cv2.initUndistortRectifyMap(
        cm_right, dist_right, R2, P2, gray_right.shape[::-1], cv2.CV_32FC1
    )

    matched_markers = []

    for i, id_left in enumerate(ids_left):
        if id_left in ids_right:
            # Find corresponding marker in right image
            right_idx = np.where(ids_right == id_left)[0][0]

            # Get corners for this marker
            corners_l = corners_left[i][0]  # Shape: (4, 2)
            corners_r = corners_right[right_idx][0]  # Shape: (4, 2)

            # Stereo triangulate the 3D position and orientation
            marker_pose = triangulate_stereo_marker_pose(
                corners_l,
                corners_r,
                cm_left,
                cm_right,
                dist_left,
                dist_right,
                R,
                t,
                map1_left,
                map2_left,
                map1_right,
                map2_right,
                Q,
                marker_size=marker_size,
            )

            if marker_pose is not None:
                tvec, rvec, corners_3d = marker_pose

                matched_markers.append(
                    {
                        "id": int(id_left),
                        "tvec": tvec,
                        "rvec": rvec,
                        "corners_left": corners_l,
                        "corners_right": corners_r,
                        "corners_3d": corners_3d,  # 3D corner positions
                        "stereo_confidence": calculate_stereo_confidence(
                            corners_l, corners_r, corners_3d, cm_left, cm_right
                        ),
                    }
                )

    return matched_markers


def triangulate_stereo_marker_pose(
    corners_left,
    corners_right,
    cm_left,
    cm_right,
    dist_left,
    dist_right,
    R,
    t,
    map1_left,
    map2_left,
    map1_right,
    map2_right,
    Q,
    marker_size=MARKER_SIZE,
):
    """
    Triangulate 3D marker pose using true stereo geometry.
    """

    try:
        # Method 1: Direct triangulation with projection matrices
        # Create projection matrices
        P_left = np.hstack([cm_left, np.zeros((3, 1))])
        P_right = np.hstack([cm_right @ R, cm_right @ t.reshape(-1, 1)])

        # Undistort points
        corners_left_undist = cv2.undistortPoints(
            corners_left.reshape(-1, 1, 2), cm_left, dist_left
        ).reshape(-1, 2)
        corners_right_undist = cv2.undistortPoints(
            corners_right.reshape(-1, 1, 2), cm_right, dist_right
        ).reshape(-1, 2)

        # Triangulate each corner
        corners_3d = []
        for i in range(4):
            # Triangulate this corner
            point_4d = cv2.triangulatePoints(
                P_left,
                P_right,
                corners_left_undist[i].reshape(2, 1),
                corners_right_undist[i].reshape(2, 1),
            )
            # Convert from homogeneous coordinates
            point_3d = point_4d[:3] / point_4d[3]
            corners_3d.append(point_3d.flatten())

        corners_3d = np.array(corners_3d)

        # Method 2: Use rectified stereo for validation
        # Rectify the corner points
        corners_left_rect = cv2.remap(
            corners_left.reshape(-1, 1, 2).astype(np.float32),
            map1_left,
            map2_left,
            cv2.INTER_LINEAR,
        ).reshape(-1, 2)
        corners_right_rect = cv2.remap(
            corners_right.reshape(-1, 1, 2).astype(np.float32),
            map1_right,
            map2_right,
            cv2.INTER_LINEAR,
        ).reshape(-1, 2)

        # Calculate disparities
        disparities = corners_left_rect[:, 0] - corners_right_rect[:, 0]

        # Validate disparities (should be positive and reasonable)
        if np.any(disparities <= 0) or np.any(disparities > 200):
            print("Warning: Invalid disparities detected")

        # Alternative 3D reconstruction using disparity
        corners_3d_alt = []
        for i in range(4):
            # Use reprojectImageTo3D equivalent
            disparity = disparities[i]
            if disparity > 0:
                # Manual 3D reconstruction
                x_rect = corners_left_rect[i, 0]
                y_rect = corners_left_rect[i, 1]

                # From stereo rectification parameters
                cx = Q[0, 3]
                cy = Q[1, 3]
                f = Q[2, 3]

                X = (x_rect - cx) / disparity
                Y = (y_rect - cy) / disparity
                Z = f / disparity

                corners_3d_alt.append([X, Y, Z])

        # Use the method that gives more consistent results
        if len(corners_3d_alt) == 4:
            corners_3d_alt = np.array(corners_3d_alt)
            # Choose the method with smaller variance in Z (more consistent depth)
            if np.var(corners_3d_alt[:, 2]) < np.var(corners_3d[:, 2]):
                corners_3d = corners_3d_alt

        # Estimate marker pose from 3D corners
        tvec, rvec = estimate_pose_from_3d_corners(corners_3d)

        return tvec, rvec, corners_3d

    except Exception as e:
        print(f"Error in stereo triangulation: {e}")
        return None


def estimate_pose_from_3d_corners(corners_3d):
    """
    Estimate 6DOF pose from 3D corner positions.
    """
    # Calculate centroid
    centroid = np.mean(corners_3d, axis=0)

    # Create coordinate system from corners
    # Vector from corner 0 to corner 1 (x-axis direction)
    x_axis = corners_3d[1] - corners_3d[0]
    x_axis = x_axis / np.linalg.norm(x_axis)

    # Vector from corner 0 to corner 3 (y-axis direction)
    y_axis = corners_3d[3] - corners_3d[0]
    y_axis = y_axis / np.linalg.norm(y_axis)

    # Z-axis (normal to marker plane)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)

    # Ensure right-handed coordinate system
    if np.dot(z_axis, np.cross(x_axis, y_axis)) < 0:
        z_axis = -z_axis

    # Create rotation matrix
    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])

    # Convert to rotation vector
    rvec, _ = cv2.Rodrigues(rotation_matrix)

    return centroid, rvec.flatten()


def calculate_stereo_confidence(
    corners_left, corners_right, corners_3d, cm_left, cm_right
):
    """
    Calculate confidence metric for stereo triangulation.
    """
    try:
        # Reproject 3D points back to both cameras
        tvec_dummy = np.zeros(3)
        rvec_dummy = np.zeros(3)

        # Project to left camera
        proj_left, _ = cv2.projectPoints(
            corners_3d.reshape(-1, 1, 3), rvec_dummy, tvec_dummy, cm_left, np.zeros(5)
        )
        proj_left = proj_left.reshape(-1, 2)

        # Project to right camera (need to transform to right camera frame first)
        # This is simplified - in practice you'd apply the stereo transform
        proj_right, _ = cv2.projectPoints(
            corners_3d.reshape(-1, 1, 3), rvec_dummy, tvec_dummy, cm_right, np.zeros(5)
        )
        proj_right = proj_right.reshape(-1, 2)

        # Calculate reprojection errors
        error_left = np.mean(np.linalg.norm(proj_left - corners_left, axis=1))
        error_right = np.mean(np.linalg.norm(proj_right - corners_right, axis=1))

        # Simple confidence metric (lower error = higher confidence)
        confidence = 1.0 / (1.0 + error_left + error_right)

        return confidence

    except:
        return 0.0


def advanced_stereo_aruco_callback(
    img_left, img_right, cm_left, cm_right, dist_left, dist_right, R, t, detector
):
    """
    Advanced stereo ArUco detection with multiple validation methods.
    """

    # Get basic stereo detections
    markers = stereo_aruco_callback(
        img_left, img_right, cm_left, cm_right, dist_left, dist_right, R, t, detector
    )

    # Additional validation and filtering
    validated_markers = []

    for marker in markers:
        # Filter by confidence
        if marker["stereo_confidence"] > 0.5:

            # Additional geometric validation
            corners_3d = marker["corners_3d"]

            # Check if corners form a reasonable square
            # Calculate side lengths
            side_lengths = [
                np.linalg.norm(corners_3d[1] - corners_3d[0]),
                np.linalg.norm(corners_3d[2] - corners_3d[1]),
                np.linalg.norm(corners_3d[3] - corners_3d[2]),
                np.linalg.norm(corners_3d[0] - corners_3d[3]),
            ]

            # Check if sides are roughly equal (within 20%)
            avg_side = np.mean(side_lengths)
            if all(abs(side - avg_side) / avg_side < 0.2 for side in side_lengths):

                # Check if marker is at reasonable distance
                distance = np.linalg.norm(marker["tvec"])
                if 0.1 < distance < 10.0:  # Between 10cm and 10m
                    validated_markers.append(marker)

    return validated_markers


class StereoArucoDetector:

    def __init__(self, node=Node):
        self.node = node

        # Setup aruco dictionary and parameters
        aruco_dict = (
            node.declare_parameter("aruco_dict", "DICT_5X5_250")
            .get_parameter_value()
            .string_value
        )

        self.visualize = (
            node.declare_parameter("visualize", True).get_parameter_value().bool_value
        )

        self.bridge = CvBridge()
        if self.visualize:
            self.image_pub = node.create_publisher(Image, "/aruco/visualization", 10)
            self.node.get_logger().info("Visualization enabled for ArUco markers.")

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, aruco_dict)
        )

        self.aruco_ids = node.declare_parameter("aruco_ids", [51, 52, 53, 54, 55])
        self.pose_publishers = {}

        for i in self.aruco_ids.get_parameter_value().integer_array_value:
            self.pose_publishers[i] = node.create_publisher(
                PointStamped, f"/aruco/marker_{i:02}", 10
            )
        self.node.get_logger().info(
            f"Publishing poses for ArUco IDs: {self.aruco_ids.get_parameter_value().integer_array_value}"
        )

        self.params = cv2.aruco.DetectorParameters()

        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.params)

        topic_left = (
            node.declare_parameter("stereo_left_topic", "/camera/left/image_raw")
            .get_parameter_value()
            .string_value
        )
        topic_right = (
            node.declare_parameter("stereo_right_topic", "/camera/right/image_raw")
            .get_parameter_value()
            .string_value
        )

        self.last_marker_positions = defaultdict(lambda: np.zeros(3))

        # Create subscribers for stereo images with time synchronization
        self.sub_left = Subscriber(node, Image, topic_left)
        self.sub_right = Subscriber(node, Image, topic_right)
        self.sync = TimeSynchronizer([self.sub_left, self.sub_right], 10)
        self.sync.registerCallback(self.stereo_aruco_callback)
        self.node.get_logger().info(
            f"Subscribed to stereo topics: {topic_left} and {topic_right}"
        )

        fx, fy, cx, cy = (
            node.declare_parameter("intrinsics_left", [522.0, 522.0, 320.0, 240.0])
            .get_parameter_value()
            .double_array_value
        )
        cm_left = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        fx, fy, cx, cy = (
            node.declare_parameter("intrinsics_right", [522.0, 522.0, 320.0, 240.0])
            .get_parameter_value()
            .double_array_value
        )
        cm_right = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        R = (
            node.declare_parameter("R", list(np.eye(3).flatten()))
            .get_parameter_value()
            .double_array_value
        )
        self.marker_depth = (
            node.declare_parameter("marker_depth", MARKER_DEPTH)
            .get_parameter_value()
            .double_value
        )
        R = np.array(R).reshape(3, 3)
        t = (
            node.declare_parameter("t", list(np.zeros(3)))
            .get_parameter_value()
            .double_array_value
        )

        t = np.array(t).reshape(3, 1)

        T_global = (
            node.declare_parameter("T_global", list(np.eye(4).flatten()))
            .get_parameter_value()
            .double_array_value
        )
        self.T_global = np.array(T_global).reshape(4, 4)

        dist_left = np.array(
            node.declare_parameter("dist_left", list(np.zeros(5)))
            .get_parameter_value()
            .double_array_value
        )
        dist_right = np.array(
            node.declare_parameter("dist_right", list(np.zeros(5)))
            .get_parameter_value()
            .double_array_value
        )

        self.camera_params = {
            "cm_left": cm_left,
            "cm_right": cm_right,
            "dist_left": dist_left,
            "dist_right": dist_right,
            "R": R,
            "t": t,
        }

        self.node.get_logger().info(
            "Camera parameters loaded for stereo detection:"
            f"\nLeft Camera Matrix: {self.camera_params['cm_left']}"
            f"\nRight Camera Matrix: {self.camera_params['cm_right']}"
            f"\nLeft Distortion Coefficients: {self.camera_params['dist_left']}"
            f"\nRight Distortion Coefficients: {self.camera_params['dist_right']}"
            f"\nRotation Matrix R: {self.camera_params['R']}"
            f"\nTranslation Vector t: {self.camera_params['t']}"
            f"\nGlobal Transformation Matrix T_global: {self.T_global}"
        )

    def get_marker_positions(self, img):
        """
        Process a single image to detect ArUco markers and return their positions.
        """
        markers = mono_aruco_detect(
            img,
            self.camera_params["cm_left"],
            self.camera_params["dist_left"],
            self.detector,
        )

        marker_positions = defaultdict(list)
        for marker in markers:
            self.node.get_logger().info(f"Detected marker ID {marker['id']} in image")

            tvec = marker["tvec"]
            rvec = marker["rvec"]

            t = np.array([[0, 0, self.marker_depth]])
            tvec = tvec - R.from_rotvec(rvec[0]).apply(t)
            marker_positions[marker["id"]].append(tvec)

        marker_avg_positions = {}
        for marker_id, positions in marker_positions.items():
            # Average the positions for stability
            avg_position = np.mean(positions, axis=0).flatten()
            marker_avg_positions[marker_id] = avg_position

        return marker_avg_positions

    def stereo_aruco_callback(self, img_left, img_right):
        """
        Callback for stereo image processing.
        """
        img_left = self.bridge.imgmsg_to_cv2(img_left, desired_encoding="bgr8")
        img_right = self.bridge.imgmsg_to_cv2(img_right, desired_encoding="bgr8")

        positions_left = self.get_marker_positions(img_left)
        positions_right = self.get_marker_positions(img_right)

        # Align right to left
        for marker_id, pos_right in positions_right.items():
            if marker_id in positions_left:
                pos_left = positions_left[marker_id]
                pos_right = pos_right - self.camera_params["t"].flatten()
                # Average the positions from both cameras
                avg_position = (pos_left + pos_right) / 2.0
                positions_left[marker_id] = avg_position
            else:
                self.node.get_logger().warn(
                    f"Marker ID {marker_id} detected in left image but not in right."
                )
                positions_left[marker_id] = (
                    pos_right - self.camera_params["t"].flatten()
                )

        for marker_id, position in positions_left.items():
            if marker_id in self.pose_publishers:
                position = self.T_global @ np.append(position, 1.0)
                position = position[:3] / position[3]  # Convert to 3D position

                # only publish if close to last position
                if (
                    np.linalg.norm(position - self.last_marker_positions[marker_id])
                    > 0.1
                ):
                    self.node.get_logger().info(
                        f"New position for marker {marker_id}: {position}, delta: {np.linalg.norm(position - self.last_marker_positions[marker_id])}"
                    )
                    self.last_marker_positions[marker_id] = position
                    continue
                if np.linalg.norm(position) > 10.0:
                    self.node.get_logger().warn(
                        f"Marker {marker_id} position {position} is too far from origin, skipping publish."
                    )
                    continue

                self.last_marker_positions[marker_id] = position

                pose_msg = PointStamped()
                pose_msg.header.stamp = self.node.get_clock().now().to_msg()
                pose_msg.header.frame_id = "imu"

                # Set position
                pose_msg.point.x = float(position[0])
                pose_msg.point.y = float(position[1])
                pose_msg.point.z = float(position[2])

                self.pose_publishers[marker_id].publish(pose_msg)
            else:
                self.node.get_logger().warn(
                    f"Marker ID {marker_id} not configured for publishing."
                )


def main():
    import rclpy
    from rclpy.executors import SingleThreadedExecutor

    rclpy.init()

    node = Node("stereo_aruco_detector")
    detector = StereoArucoDetector(node)

    executor = SingleThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
