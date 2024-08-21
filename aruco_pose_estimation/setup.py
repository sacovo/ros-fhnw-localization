from setuptools import find_packages, setup

package_name = 'aruco_pose_estimation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Sandro Covo',
    maintainer_email='sandro.covo@fhnw.ch',
    description='Estimate rover pose using ArUco markers',
    license='FHNW',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'aruco_tracker = aruco_pose_estimation.aruco_tracker:main'
        ],
    },
)
