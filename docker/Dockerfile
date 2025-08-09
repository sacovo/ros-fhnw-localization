# Dockerfile for building ROS 2 packages
FROM ros:humble-ros-core AS base

ENV DEBIAN_FRONTEND=noninteractive

COPY ./docker/apt.txt ./docker/requirements.txt /tmp/

ARG USE_PROXY=false

# Install required packages and Python dependencies
RUN --mount=type=cache,target=/var/cache/apt \
    if [ "$USE_PROXY" = "true" ]; then \
    echo 'Acquire::http { Proxy "http://172.17.0.1:3142"; }' | tee -a /etc/apt/apt.conf.d/proxy; \
    fi && \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update --fix-missing &&  \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    sudo python3-pip bash \
    $(cat /tmp/apt.txt | grep -v "^#") && \
    pip3 install --no-cache-dir -r /tmp/requirements.txt && \
    pip3 install "numpy<2"

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
    wget \
    g++ \
    make \
    dos2unix \
    python3-catkin-pkg-modules \
    ros-humble-ament-cmake \
    $(cat /tmp/apt-dev.txt | grep -v "^#")



FROM builder AS build

COPY . .

RUN --mount=type=cache,target=/work/build bash build.sh

RUN dos2unix docker/entrypoint.sh

# Switch to the base image and copy the built packages
FROM base

# Entrypoint sources setup.bash with built packages
COPY --from=build /work/docker/entrypoint.sh /entrypoint.sh

# Add user ros with sudo privileges
RUN useradd -ms /bin/bash ros && \
    chown ros:ros /work && \
    usermod -aG sudo ros && \
    echo "ros ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers && \
    echo "source /opt/ros/humble/setup.bash" >> /home/ros/.bashrc && \
    echo "source /work/install/setup.bash" >> /home/ros/.bashrc

# Finnaly, copy the built packages
COPY --chown=ros:ros --from=build /work/install /work/install
COPY --chown=ros:ros --from=build /work/config /work/config
COPY --chown=ros:ros --from=build /work/launch /work/launch

USER ros

ENTRYPOINT [ "/entrypoint.sh" ]
