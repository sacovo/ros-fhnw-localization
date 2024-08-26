import numpy as np
from aruco_pose_estimation.aruco_tracker import ArucoProcessor, Camera, TagObservation

I = np.eye(3)


def test_marker_calculations():
    result = ArucoProcessor.get_marker_position(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        np.array([0.0, 0.0, 5.0]),
        I,
    )

    assert np.allclose(result, [5.0, 0.0, 0.0])


def test_marker_calculations_2():
    result = ArucoProcessor.get_marker_position(
        [5.0, 3.0, 0.0],
        [0.0, 0.0, 0.0],
        np.array([-1.0, -0.0, 2.0]),
        I,
    )

    assert np.allclose(result, [7.0, 2.0, 0.0])


def test_marker_calculations_3():
    result = ArucoProcessor.get_marker_position(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, np.pi / 2],
        np.array([0.0, 0.0, 5.0]),
        I,
    )

    assert np.allclose(result, [0.0, 5.0, 0.0])

    result = ArucoProcessor.get_marker_position(
        [0.0, 0.0, 0.0],
        [0.0, 0.0,- np.pi / 2],
        np.array([0.0, 0.0, 5.0]),
        I,
    )
    assert np.allclose(result, [0.0, -5.0, 0.0])

    result = ArucoProcessor.get_marker_position(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, np.pi / 4],
        np.array([0.0, 0.0, 1.0]),
        I,
    )
    assert np.allclose(result, [np.cos(np.pi/4), np.sin(np.pi/4), 0.0])

    result = ArucoProcessor.get_marker_position(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.3],
        np.array([0.0, 0.0, 1.0]),
        I,
    )
    assert np.allclose(result, [np.cos(1.3), np.sin(1.3), 0.0])

    result = ArucoProcessor.get_marker_position(
        [5.0, 2.0, 0.0],
        [0.0, 0.0, 1.3],
        np.array([0.0, 0.0, 1.0]),
        I,
    )
    assert np.allclose(result, [5.0 + np.cos(1.3), 2.0 + np.sin(1.3), 0.0])
    
    
def test_position_error():
    params = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    orientation = params[3:]
    
    fx, fy, cx, cy = 500, 500, 320, 240
    
    cam = Camera(
        name="cam01",
        T_imu_cam=np.eye(4),
        camera_matrix=np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]),
        dist_coeffs=np.zeros(5),
        compressed=False,
        topic=""
    )
    
    observations = [
        TagObservation(
            marker_id=1,
            rvec=np.array([0.0, 0.0, 0.0]),
            tvec=np.array([0.0, 0.0, 5.0]),
            corners=np.array([[0, 0], [0, 1], [1, 1], [1, 0]]),
            cam=cam,
        )
    ]
    
    e = ArucoProcessor.position_error(
        params,
        orientation=orientation,
        observations=observations,
        marker_positions={1: np.array([5.0, 0.0, 0.0])},
    )
    
    assert e == 0
    
    e = ArucoProcessor.position_error(
        params,
        orientation=orientation,
        observations=observations,
        marker_positions={1: np.array([4.0, 0.0, 0.0])},
    )
    assert e == 1

    e = ArucoProcessor.position_error(
        params,
        orientation=orientation,
        observations=observations,
        marker_positions={1: np.array([4.0, -1.0, 0.0])},
    )
    assert e == np.sqrt(2)
    
def test_optimize_position():
    pass