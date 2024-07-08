# Dockerfile for building ROS 2 packages
FROM ros:humble-ros-core as base

ENV DEBIAN_FRONTEND=noninteractive

COPY ./docker/apt.txt ./docker/apt-dev.txt ./docker/requirements.txt /tmp/

# Install required packages and Python dependencies
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update --fix-missing &&  \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        sudo python3-pip \
        $(cat /tmp/apt.txt | grep -v "^#") && \
    pip3 install --no-cache-dir -r /tmp/requirements.txt && \
    apt-get clean && rm /tmp/* 

WORKDIR /work

FROM base as builder

# Install packages for building ROS 2 packages

RUN  --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    git \
    python3-colcon-common-extensions \
    python3-colcon-mixin \
    python3-rosdep \
    python3-vcstool \
    $(cat /tmp/apt-dev.txt | grep -v "^#") && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY . .

RUN colcon build

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
COPY --chown=ros:ros --from=builder /work/install /work/install

USER ros

ENTRYPOINT [ "/entrypoint.sh" ]