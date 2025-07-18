import numpy as np
import cv2

# Stereo parameters for HD case
baseline = 119.782 * 0.001  # mm
ty = -0.207959 * 0.001  # mm
tz = 0.447725 * 0.001  # mm

# Rodrigues rotation vectors (in radians)
rx_rod = 0.0017898  # RX_HD
ry_rod = 0.00557495  # CV_HD (optical convergence)
rz_rod = 0.00130407  # RZ_HD

# Create rotation vector
rotation_vector = np.array([rx_rod, ry_rod, rz_rod])

# Convert Rodrigues vector to rotation matrix
R, _ = cv2.Rodrigues(rotation_vector)

# Translation vector (baseline is the primary x-axis separation)
translation = np.array([baseline, ty, tz])

# Create 4x4 transformation matrix
T = np.eye(4)
T[:3, :3] = R
T[:3, 3] = translation

print("Transform Matrix (HD):")
print(T.tolist())
