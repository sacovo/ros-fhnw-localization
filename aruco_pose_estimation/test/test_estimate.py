import numpy as np
import cv2

def refine_position_single_marker(current_position, orientation, camera_matrix, dist_coeffs, detected_markers, marker_positions):
    """
    Refines the current position estimate using one or two detected ArUco markers.

    Parameters:
    - current_position: np.array of shape (3,) representing the current (x, y, z) position estimate.
    - orientation: np.array of shape (3, 3) representing the current rotation matrix.
    - camera_matrix: np.array of shape (3, 3) representing the camera intrinsics matrix.
    - dist_coeffs: np.array of shape (5,) representing the camera distortion coefficients.
    - detected_markers: dict with marker IDs as keys and pixel positions (np.array of shape (2,)) as values.
    - marker_positions: dict with marker IDs as keys and global positions (np.array of shape (3,)) as values.

    Returns:
    - refined_position: np.array of shape (3,) representing the refined (x, y, z) position estimate.
    """
    # Convert current position to a column vector
    
    # Prepare optimization data
    image_points = []
    object_points = []

    for marker_id, pixel_position in detected_markers.items():
        if marker_id in marker_positions:
            global_position = marker_positions[marker_id]
            image_points.append(pixel_position)
            object_points.append(global_position)
    
    # Convert to numpy arrays
    image_points = np.array(image_points, dtype=np.float32)
    object_points = np.array(object_points, dtype=np.float32)
    
    # Define a function to minimize reprojection error
    def reprojection_error(position):
        position = np.reshape(position, (3, 1))
        projected_points, _ = cv2.projectPoints(object_points, cv2.Rodrigues(orientation)[0], position, camera_matrix, dist_coeffs)
        projected_points = projected_points[:, 0, :]  # Convert to Nx2 array
        error = np.sum((projected_points - image_points) ** 2)
        return error
    
    # Optimize the current position using a simple optimizer like scipy's minimize
    from scipy.optimize import minimize
    result = minimize(reprojection_error, current_position, method='BFGS')
    
    refined_position = result.x
    
    return refined_position

# Example usage:
if __name__ == "__main__":
    current_position = np.array([0.5, 1.0, 1.5])
    orientation = np.eye(3)  # Assuming a neutral rotation matrix for simplicity
    camera_matrix = np.array([[800, 0, 320],
                              [0, 800, 240],
                              [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros(5)  # Assuming no distortion for simplicity
    detected_markers = {1: np.array([150, 200]), 2: np.array([300, 400])}
    marker_positions = {1: np.array([2.0, 3.0, 1.0]), 2: np.array([1.0, 2.0, 1.0])}

    refined_position = refine_position_single_marker(current_position, orientation, camera_matrix, dist_coeffs, detected_markers, marker_positions)
    print("Refined Position:", refined_position)
