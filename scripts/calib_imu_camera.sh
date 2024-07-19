#!/usr/bin/env bash
set -x

xhost +

NAME=$1
shift 1
CAMERA_NAME=$1
TOPIC=/camera_$CAMERA_NAME/image_raw

CAMERA_CONFIG=results/calibs/camera_$CAMERA_NAME-camchain.yaml

xhost +
docker compose -f docker-compose.calib.yml run --user=$UID calib rosrun kalibr kalibr_calibrate_imu_camera \
	--bag $NAME \
	--target config/calibration_pattern_a3.yaml \
	--imu results/calibs/hipnuc_imu.yml \
	--cams $CAMERA_CONFIG \
	--timeoffset-padding 0.4
