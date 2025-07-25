cmake_minimum_required(VERSION 3.8)
project(wheel_odometry)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

add_executable(odom_node src/OdomClientNode.cpp)
ament_target_dependencies(odom_node rclcpp sensor_msgs)

add_executable(calc_node src/WheelCalc.cpp)
ament_target_dependencies(calc_node rclcpp sensor_msgs nav_msgs geometry_msgs)

add_executable(twist_integrator src/TwistIntegrator.cpp)
ament_target_dependencies(twist_integrator tf2_ros rclcpp sensor_msgs nav_msgs geometry_msgs)

install(TARGETS
  odom_node
  calc_node
  twist_integrator
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
