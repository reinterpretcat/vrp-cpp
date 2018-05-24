FROM nvidia/cuda:9.0-devel-ubuntu16.04

# install packages
RUN apt-get update && apt-get install -y wget libthrust-dev

WORKDIR /tmp

# install cmake 3.8
RUN wget https://cmake.org/files/v3.8/cmake-3.8.1-Linux-x86_64.sh && \
    mkdir /opt/cmake && \
    sh cmake-3.8.1-Linux-x86_64.sh --skip-license --prefix=/opt/cmake && \
    ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake

# create build dir
RUN mkdir -p /repo/build
WORKDIR /repo/build
