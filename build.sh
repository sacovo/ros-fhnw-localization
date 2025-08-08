#!/bin/bash

source /opt/ros/humble/setup.bash

colcon build --path MINS/thirdparty/libnabo

source install/setup.bash

colcon build --path MINS/thirdparty/libpointmatcher

source install/setup.bash

colcon build --path MINS/thirdparty/

source install/setup.bash

colcon build --path MINS/thirdparty/open_vins/ov_core/

source install/setup.bash

colcon build --path MINS/mins/


source install/setup.bash

colcon build --path fhnw_localization/ fhnw-interfaces/ odom_transform/

source install/setup.bash