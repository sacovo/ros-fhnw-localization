from aruco_pose_estimation.aruco_tracker import ArucoProcessor, ArucoPoseEstimator
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


def test_marker_to_camera_transform():
    M = R.identity().as_matrix()

    cam = ArucoPoseEstimator.load_camera_parameters("cameras.yml")[0]
    print(cam["T_imu_cam"])
    T_imu_cam = cam["T_imu_cam"]

    print(R.from_matrix(T_imu_cam[:3, :3]).as_euler("xyz", degrees=True))

    ps = [
        [10.0, 0, 0],
        [0.0, 10000, 0],
        [10.0, -10, 0],
        [10.0, 10, 0],
        [0.0, 1.0, 1],
        [0.0, 1.0, 10.0],
        [0.0, 1.0, -10.0],
    ]
    for p in ps:

        x, _ = cv2.projectPoints(
            np.array([p]),
            T_imu_cam[:3, :3],
            T_imu_cam[:3, 3],
            cam["camera_matrix"],
            cam["dist_coeffs"],
        )
        print("P", p)
        print("x", x)
    assert False


if __name__ == "__main__":
    cam = ArucoPoseEstimator.load_camera_parameters("cameras.yml")[0]
    print(cam["T_imu_cam"])
    T_imu_cam = cam["T_imu_cam"]
    
    R_imu_cam = R.from_matrix(T_imu_cam[:3, :3])
    p_imu_cam = T_imu_cam[:3, 3]

    p = np.array([0, 0, 0])
    rot = R.identity()

    while True:
        p = np.array(
            [float(x) for x in input("Enter point coordinates: ").split(" ") if x]
        )
        x, _ = cv2.projectPoints(
            p,
            (R_imu_cam * rot).as_matrix(),
            (R_imu_cam * rot).apply(p_imu_cam) + p,
            cam["camera_matrix"],
            cam["dist_coeffs"],
        )
        print(x)
