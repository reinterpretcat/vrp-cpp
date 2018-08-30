FROM nvidia/cuda:9.0-devel-ubuntu16.04

# install packages
RUN apt-get update && apt-get install -y wget libthrust-dev libomp-dev

WORKDIR /tmp

# install cmake 3.12.1
RUN wget https://cmake.org/files/v3.12/cmake-3.12.1-Linux-x86_64.sh && \
    mkdir /opt/cmake && \
    sh cmake-3.12.1-Linux-x86_64.sh --skip-license --prefix=/opt/cmake && \
    ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake

# create build dir
RUN mkdir -p /repo/build
WORKDIR /repo/build
