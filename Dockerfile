FROM ubuntu:18.04

# how to use:
#   docker rm vrp_solver
#   docker build --tag vrp_solver .
#   docker run -it -v $(pwd):/repo --rm vrp_solver
#   mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release ..

ARG CLANG_VERSION=7.0.1
ARG CMAKE_VERSION=3.12.1

# install clang and deps
RUN apt-get update && apt-get install -y \
  xz-utils \
  build-essential \
  cmake \
  curl \
  && rm -rf /var/lib/apt/lists/* \
  && curl -SL http://releases.llvm.org/${CLANG_VERSION}/clang+llvm-${CLANG_VERSION}-x86_64-linux-gnu-ubuntu-18.04.tar.xz \
  | tar -xJC . && \
  mv clang+llvm-${CLANG_VERSION}-x86_64-linux-gnu-ubuntu-18.04 clang_${CLANG_VERSION} && \
  echo 'export PATH=/clang_${CLANG_VERSION}/bin:$PATH' >> ~/.bashrc && \
  echo 'export LD_LIBRARY_PATH=/clang_${CLANG_VERSION}/lib:LD_LIBRARY_PATH' >> ~/.bashrc

# install cmake
RUN curl https://cmake.org/files/v3.12/cmake-${CMAKE_VERSION}-Linux-x86_64.sh --output cmake-${CMAKE_VERSION}.sh && \
  mkdir /opt/cmake && \
  sh cmake-${CMAKE_VERSION}.sh --skip-license --prefix=/opt/cmake && \
  ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake

# TODO make sure that TBB dependency is satisfied

ENV CC=/clang_${CLANG_VERSION}/bin/clang
ENV CXX=/clang_${CLANG_VERSION}/bin/clang++

WORKDIR /repo/build
CMD [ "/bin/bash" ]