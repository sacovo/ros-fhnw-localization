#!/bin/sh

NAME=$1
TOPIC=/yocto/imu

rosbags-convert $NAME --include-topic $TOPIC --dst $NAME.bag
mkdir ${NAME}_cooked/

docker compose -f docker-compose.calib.yml up -d calib-imu
docker compose -f docker-compose.calib.yml run calib-imu rosrun allan_variance_ros cookbag.py \
	--input $NAME.bag --output ${NAME}_cooked/imu.bag
docker compose -f docker-compose.calib.yml run calib-imu ros rosrun allan_variance_ros allan_variance \
	/${NAME}_cooked/ /config/imu.yaml

docker compose -f docker-compose.calib.yml run calib-imu rosrun allan_variance_ros analysis.py \
	--data /${NAME}_cooked/allan_variance.csv
docker compose -f docker-compose.calib.yml down calib-imu
