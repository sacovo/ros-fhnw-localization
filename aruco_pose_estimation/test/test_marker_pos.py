import numpy as np
import cProfile
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
        [0.0, 0.0, -np.pi / 2],
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
    assert np.allclose(result, [np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0])

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


def _get_camera():
    fx, fy, cx, cy = 500, 500, 320, 240

    cam = Camera(
        name="cam01",
        T_imu_cam=np.eye(4),
        camera_matrix=np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]),
        dist_coeffs=np.zeros(5),
        compressed=False,
        topic="",
    )

    return cam


def test_position_error():
    params = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    orientation = params[3:]

    cam = _get_camera()

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
    global_positions = {
        1: np.array([1.0, 0.0, 0.0]),
        2: np.array([1.0, -1.0, 0.0]),
        3: np.array([1.0, 1.0, 0.0]),
    }

    position = np.array([0.0, 0.0, 0.0])
    orientation = np.array([0.0, 0.0, 0.0])
    cam = _get_camera()
    corners = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    observations = [
        TagObservation(
            marker_id=1,
            rvec=np.array([0.0, 0.0, 0.0]),
            tvec=np.array([0.0, 0.0, 1.0]),
            corners=corners,
            cam=cam,
        ),
        TagObservation(
            marker_id=2,
            rvec=np.array([0.0, 0.0, 0.0]),
            tvec=np.array([-1.0, 0.0, 1.0]),
            corners=corners,
            cam=cam,
        ),
    ]

    pos, rot = ArucoProcessor.optimize_pose(
        np.concatenate([position, orientation]),
        observations,
        error_function=ArucoProcessor.position_error,
        marker_positions=global_positions,
    )

    error = ArucoProcessor.position_error(
        np.concatenate([pos, rot]),
        orientation=rot,
        observations=observations,
        marker_positions=global_positions,
    )

    assert error < 1e-6
 

    assert np.allclose(pos, [0.0, 0.0, 0.0])
    assert np.allclose(rot, [0.0, 0.0, 0.0])

    position = np.array([0.5, 0.3, 0.3])
    orientation = np.array([0.0, 0.0, 0.0])
    
    pos, rot = ArucoProcessor.optimize_pose(
        np.concatenate([position, orientation]),
        observations,
        error_function=ArucoProcessor.position_error,
        marker_positions=global_positions,
    )
    
    error = ArucoProcessor.position_error(
        np.concatenate([pos, rot]),
        orientation=rot,
        observations=observations[:1],
        marker_positions=global_positions,
    )

    assert error < 1e-2
    assert np.allclose(pos[:2], [0., 0.], atol=1e-2)
    
    position = np.array([1.5, 1.123, 0.3])
    orientation = np.array([0.0, 0.0, 0.0])
    
    pos, rot = ArucoProcessor.optimize_pose(
        np.concatenate([position, orientation]),
        observations,
        error_function=ArucoProcessor.position_error,
        marker_positions=global_positions,
    )
    
    error = ArucoProcessor.position_error(
        np.concatenate([pos, rot]),
        orientation=rot,
        observations=observations,
        marker_positions=global_positions,
    )

    assert error < 1e-2
    assert np.allclose(pos[:2], [0., 0.], atol=1e-2)
    
    observations = [
        TagObservation(
            marker_id=1,
            rvec=np.array([0.0, 0.0, 0.0]),
            tvec=np.array([-1.0, 0.0, 1.5]),
            corners=corners,
            cam=cam,
        ),
        TagObservation(
            marker_id=2,
            rvec=np.array([0.0, 0.0, 0.0]),
            tvec=np.array([-2.0, 0.0, 1.5]),
            corners=corners,
            cam=cam,
        ),
    ]   

    pos, rot = ArucoProcessor.optimize_pose(
        np.concatenate([position, orientation]),
        observations,
        error_function=ArucoProcessor.position_error,
        marker_positions=global_positions,
    )

    error = ArucoProcessor.position_error(
        np.concatenate([pos, rot]),
        orientation=rot,
        observations=observations,
        marker_positions=global_positions,
    )

    assert error < 1e-3
    assert np.allclose(pos[:2], [-.5, 1.], atol=1e-2)

    position = np.array([1.5, 1.123, 0.3])
    orientation = np.array([0.0, 0.0, np.pi / 4])

    observations = [
        TagObservation(
            marker_id=1,
            rvec=np.array([0.0, 0.0, 0.0]),
            tvec=np.array([0, 0, 2.]),
            corners=corners,
            cam=cam,
        ),
        TagObservation(
            marker_id=2,
            rvec=np.array([0.0, 0.0, 0.0]),
            tvec=np.array([0.0, 0.0, 3.0]),
            corners=corners,
            cam=cam,
        ),
    ]   
 

    pos, rot = ArucoProcessor.optimize_pose(
        np.concatenate([position, orientation]),
        observations,
        error_function=ArucoProcessor.position_error,
        marker_positions=global_positions,
    )
    
    assert np.allclose(pos[:2], [1, 2.], atol=1e-2)

    position = np.array([1.5, 1.123, 0.3])
    orientation = np.array([0.0, 0.0, np.pi])

    observations = [
        TagObservation(
            marker_id=1,
            rvec=np.array([0.0, 0.0, 0.0]),
            tvec=np.array([0, 0, 1.]),
            corners=corners,
            cam=cam,
        ),
        TagObservation(
            marker_id=2,
            rvec=np.array([0.0, 0.0, 0.0]),
            tvec=np.array([1.0, 0.0, 1.0]),
            corners=corners,
            cam=cam,
        ),
    ]   
 
    pos, rot = ArucoProcessor.optimize_pose(
        np.concatenate([position, orientation]),
        observations,
        error_function=ArucoProcessor.position_error,
        marker_positions=global_positions,
    )
    
    assert np.allclose(pos[:2], [2., 0.], atol=1e-2)
    assert np.allclose(rot[2:], [np.pi ], atol=1e-2)
    
def test_performance():
    cp = cProfile