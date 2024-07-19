#!/usr/bin/env bash
set -x

xhost +

NAME=$1

CAMERA_CONFIG=results/hipnuc/zed-camchain.yaml

xhost +
docker compose -f docker-compose.calib.yml run --user=$UID calib rosrun kalibr kalibr_calibrate_imu_camera \
	--bag $NAME \
	--target config/calibration_pattern_a3.yaml \
	--imu results/hipnuc/imu.yaml \
	--cams $CAMERA_CONFIG \
	--timeoffset-padding 0.5
