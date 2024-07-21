#!/bin/sh

. /opt/ros/humble/setup.sh && \
    colcon build --paths MINS/thirdparty/* ros2_shared opencv_cam && \
    . install/setup.sh && \
    colcon build --paths MINS/thirdparty/open_vins/* && \
    . install/setup.sh && \
    colcon build --paths MINS/mins MINS/mins_data && \
    . install/setup.sh && \
    colcon build --paths MINS/mins_eval && \
    . install/setup.sh && \
    colcon build