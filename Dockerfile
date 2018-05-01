FROM nvidia/cuda:8.0-devel-ubuntu16.04

RUN apt-get update && apt-get install -y git wget nvidia-cuda-toolkit

WORKDIR /tmp

# install cmake 3.8
RUN wget https://cmake.org/files/v3.8/cmake-3.8.1-Linux-x86_64.sh && \
    mkdir /opt/cmake && \
    sh cmake-3.8.1-Linux-x86_64.sh --skip-license --prefix=/opt/cmake && \
    ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake

# checkout thrust 1.9.0
RUN git clone https://github.com/thrust/thrust.git

WORKDIR /tmp/thrust
RUN git reset --hard 8551c97870cd722486ba7834ae9d867f13e299ad

# replace thrust
RUN rm -rf /usr/include/thrust
RUN cp -r thrust /usr/include
