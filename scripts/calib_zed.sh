#!/usr/bin/env bash

set -x

NAME=$1
#rosbags-convert $NAME --include-topic /zed/image_raw2 --include-topic /zed/image_raw --dst $NAME.bag

xhost +
docker compose -f docker-compose.calib.yml run calib rosrun kalibr kalibr_calibrate_cameras \
	--bag $NAME \
	--target config/calibration_pattern_a3.yaml \
	--models pinhole-radtan pinhole-radtan \
	--topics /zed/image_raw /zed/image_raw2 --show-extraction
