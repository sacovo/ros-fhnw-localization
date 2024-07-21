#!/bin/bash

INPUT_BAG_FILE=$1
OUTPUT_BAG_FILE=$2
CONFIG_PATH=$3

source install/setup.bash

ros2 run mins subscribe $CONFIG_PATH &
PID_MINS=$!

rm -r $OUTPUT_BAG_FILE
ros2 bag record -o $OUTPUT_BAG_FILE /mins/imu/pose &
PID_RECORDER=$!


ros2 bag play $INPUT_BAG_FILE 

pkill --full "run mins subscribe config/old/mins_baseline/config.yaml"
pkill --full "ros2 bag record"
pkill --full -SIGKILL "mins/subscribe"