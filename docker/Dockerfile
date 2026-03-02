

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# Install ROS Noetic
RUN apt-get update && apt-get install -y lsb-release curl gnupg2 && \
    sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
    apt-get update && apt-get install -y ros-noetic-desktop-full && \
    rm -rf /var/lib/apt/lists/*

# Install general dependencies
RUN apt-get update && apt-get install -y \
    apt-utils \
    git \
    openssh-client \
    software-properties-common \
    python3-rosdep \
    python3-catkin-tools \
    python3-vcstool \
    libboost-all-dev \
    libgoogle-glog-dev \
    libzmqpp-dev \
    nlohmann-json3-dev \
    ros-noetic-vision-msgs \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Pre-populate GitHub's host key system-wide (since ~/.ssh is mounted read-only at runtime)
# Enable SSH connection multiplexing so vcs import reuses one connection (avoids GitHub rate limiting)
RUN ssh-keyscan github.com >> /etc/ssh/ssh_known_hosts && \
    printf "Host github.com\n  ControlMaster auto\n  ControlPath /tmp/ssh_mux_%%h_%%p_%%r\n  ControlPersist 600\n" >> /etc/ssh/ssh_config

# Install GTSAM via borglab PPA (not available in rosdep)
RUN add-apt-repository ppa:borglab/gtsam-release-4.1 && \
    apt-get update && apt-get install -y \
    libgtsam-dev \
    libgtsam-unstable-dev \
    && rm -rf /var/lib/apt/lists/*

# Initialize rosdep
RUN rosdep init || true && rosdep update

# Set user arguments (being username, echo $UID, and id -g)
ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=1000

# Create a non-root user with matching UID/GID
RUN groupadd --gid ${USER_GID} ${USERNAME} && \
    useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} && \
    usermod -aG sudo ${USERNAME} && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Add a custom .bashrc with warm colors and fixed hostname
RUN echo 'export PS1="\\[\\e[1;31m\\]\\u@reason_hydra_container\\[\\e[0m\\]:\\[\\e[1;33m\\]\\w\\[\\e[0m\\]\\$ "' > /home/${USERNAME}/.bashrc && \
    chown ${USERNAME}:${USERNAME} /home/${USERNAME}/.bashrc

# Set default user & shell
USER ${USERNAME}
WORKDIR /home/${USERNAME}
SHELL ["/bin/bash", "-c"]
RUN echo "export USER=${USERNAME}" >> ~/.bashrc && \
    echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

# Refresh packages index
RUN sudo apt update

# Install venv
RUN sudo apt install python3.8-venv