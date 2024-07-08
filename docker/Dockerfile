# Dockerfile for building ROS 2 packages
FROM ros:humble-ros-core AS base

ENV DEBIAN_FRONTEND=noninteractive

COPY ./docker/apt.txt ./docker/requirements.txt /tmp/

# Install required packages and Python dependencies
RUN --mount=type=cache,target=/var/cache/apt \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update --fix-missing &&  \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        sudo python3-pip \
        $(cat /tmp/apt.txt | grep -v "^#") && \
    pip3 install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /work

FROM base AS builder

# Install packages for building ROS 2 packages

COPY ./docker/apt-dev.txt /tmp/

RUN  --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    git \
    python3-colcon-common-extensions \
    python3-colcon-mixin \
    python3-rosdep \
    python3-vcstool \
    libboost-all-dev \
    $(cat /tmp/apt-dev.txt | grep -v "^#")
    
FROM builder AS build

COPY . .

RUN --mount=type=cache,target=/work/build \
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

# Switch to the base image and copy the built packages
FROM base

# Entrypoint sources setup.bash with built packages
COPY docker/entrypoint.sh /entrypoint.sh

# Add user ros with sudo privileges
RUN useradd -ms /bin/bash ros && \
    chown ros:ros /work && \
    usermod -aG sudo ros && \
    echo "ros ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# Finnaly, copy the built packages
COPY --chown=ros:ros --from=build /work/install /work/install
COPY --chown=ros:ros --from=build /work/config /work/config

USER ros

ENTRYPOINT [ "/entrypoint.sh" ]