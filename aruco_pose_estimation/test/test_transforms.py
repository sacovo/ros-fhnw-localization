from aruco_pose_estimation.aruco_tracker import ArucoProcessor, ArucoPoseEstimator
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

#   - [ 0., 0., 1., 0.0]
#   - [-1., 0., 0., 0.0]
#   - [ 0.,-1., 0., 0.0]
#   - [ 0., 0., 0., 1.0]


#   - [ 0., 1., 0., 0.0]
#   - [ 0., 0., 1., 0.0]
#   - [ 1., 0., 0., 0.0]
#   - [ 0., 0., 0., 1.0]


def test_projection():
    coordinate_transform = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    camera_matrix = np.array(
        [
            [500, 0, 500.0],
            [0, 500, 300.0],
            [0, 0, 1.0],
        ]
    )

    projected_points, _ = cv2.projectPoints(
        np.array([10.0, 10.0, 0.0]),
        coordinate_transform[:3, :3],
        coordinate_transform[:3, 3],
        camera_matrix,
        np.array([0.0, 0.0, 0.0, 0.0]),
    )

    assert projected_points[0][0][1] == 300.0
    assert projected_points[0][0][0] == 1000.0

    projected_points, _ = cv2.projectPoints(
        np.array([10.0, 0.0, 0.0]),
        coordinate_transform[:3, :3],
        coordinate_transform[:3, 3],
        camera_matrix,
        np.array([0.0, 0.0, 0.0, 0.0]),
    )

    assert projected_points[0][0][1] == 300.0
    assert projected_points[0][0][0] == 500.0

    projected_points, _ = cv2.projectPoints(
        np.array([10.0, -10.0, 0.0]),
        coordinate_transform[:3, :3],
        coordinate_transform[:3, 3],
        camera_matrix,
        np.array([0.0, 0.0, 0.0, 0.0]),
    )

    assert projected_points[0][0][1] == 300.0
    assert projected_points[0][0][0] == 0.0

    projected_points, _ = cv2.projectPoints(
        np.array([10.0, -10.0, 10.0]),
        coordinate_transform[:3, :3],
        coordinate_transform[:3, 3],
        camera_matrix,
        np.array([0.0, 0.0, 0.0, 0.0]),
    )

    assert projected_points[0][0][1] == 800.0
    assert projected_points[0][0][0] == 0.0

    projected_points, _ = cv2.projectPoints(
        np.array([5.0, -10.0, 0.00]),
        coordinate_transform[:3, :3],
        coordinate_transform[:3, 3],
        camera_matrix,
        np.array([0.0, 0.0, 0.0, 0.0]),
    )

    assert projected_points[0][0][1] == 300.0
    assert projected_points[0][0][0] == -500.0

    rotation = R.from_euler("xyz", [0.0, 0.0, -90.0], degrees=True).as_matrix()

    projected_points, _ = cv2.projectPoints(
        np.array([0.0, 10.0, 0.00]),
        coordinate_transform[:3, :3] @ np.linalg.inv(rotation),
        coordinate_transform[:3, 3],
        camera_matrix,
        np.array([0.0, 0.0, 0.0, 0.0]),
    )

    assert projected_points[0][0][1] == 300.0
    assert np.isclose(projected_points[0][0][0], 500.0)

    rotation = R.from_euler("xyz", [0.0, 0.0, 0.0], degrees=True).as_matrix()
    position = np.array([0.0, 5.0, 0])

    projected_points, _ = cv2.projectPoints(
        np.array([10.0, 5.0, 0.00]),
        coordinate_transform[:3, :3] @ np.linalg.inv(rotation),
        -(
            coordinate_transform[:3, :3]
            @ ((rotation @ coordinate_transform[:3, 3]) + position)
        ),
        camera_matrix,
        np.array([0.0, 0.0, 0.0, 0.0]),
    )

    assert projected_points[0][0][1] == 300.0
    assert projected_points[0][0][0] == 500.0

    rotation = R.from_euler("xyz", [0.0, 0.0, -90.0], degrees=True).as_matrix()
    position = np.array([0.0, 10.0, 0])

    projected_points, _ = cv2.projectPoints(
        np.array([10.0, 10.0, 0.00]),
        coordinate_transform[:3, :3] @ np.linalg.inv(rotation),
        -(
            coordinate_transform[:3, :3]
            @ ((rotation @ coordinate_transform[:3, 3]) + position)
        ),
        camera_matrix,
        np.array([0.0, 0.0, 0.0, 0.0]),
    )

    assert projected_points[0][0][1] == 300.0
    assert np.isclose(projected_points[0][0][0], 500.0)

def test_coordinate_transform():
    cam_to_base = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    target = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    x = np.linalg.inv(cam_to_base) @ target
    print(x)

    assert np.allclose(cam_to_base @ x, target)

    assert False