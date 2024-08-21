#!/bin/sh

mkdir install && cd install 

wget https://github.com/sacovo/MINS/releases/download/latest-build/ros_install.tar.gz
tar -xvf ros_install.tar.gz
rm ros_install.tar.gz

colcon build