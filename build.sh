#!/bin/sh

mkdir -p install && cd install 

rm -rf mins mins_data mins_eval

wget https://github.com/sacovo/MINS/releases/download/latest-build/ros_install.tar.gz
tar -xvf ros_install.tar.gz
rm ros_install.tar.gz

cd ..

. install/setup.sh
colcon build