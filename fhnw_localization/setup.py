from setuptools import find_packages, setup

package_name = "fhnw_localization"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ros",
    maintainer_email="sandro@sandrocovo.ch",
    description="TODO: Package description",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "aruco_detect = fhnw_localization.aruco_detect:main",
            "pose_tracker = fhnw_localization.pose_tracker:main",
            "visualizer = fhnw_localization.visualizer:main",
            "ply_publisher = fhnw_localization.ply_publisher:main",
        ],
    },
)
