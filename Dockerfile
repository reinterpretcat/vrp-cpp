FROM ubuntu:18.04

# docker rm vrp_solver
# docker build --tag vrp_solver .
# docker run -it -v $(pwd):/repo --rm vrp_solver
# cmake -DCMAKE_BUILD_TYPE=Release ..

# install clang and deps
RUN apt-get update && apt-get install -y \
  xz-utils \
  build-essential \
  cmake \
  curl \
  && rm -rf /var/lib/apt/lists/* \
  && curl -SL http://releases.llvm.org/6.0.1/clang+llvm-6.0.1-x86_64-linux-gnu-ubuntu-16.04.tar.xz \
  | tar -xJC . && \
  mv clang+llvm-6.0.1-x86_64-linux-gnu-ubuntu-16.04 clang_6.0.1 && \
  echo 'export PATH=/clang_6.0.1/bin:$PATH' >> ~/.bashrc && \
  echo 'export LD_LIBRARY_PATH=/clang_6.0.1/lib:LD_LIBRARY_PATH' >> ~/.bashrc

# install cmake 3.12.1
RUN curl https://cmake.org/files/v3.12/cmake-3.12.1-Linux-x86_64.sh --output cmake-3.12.1.sh && \
  mkdir /opt/cmake && \
  sh cmake-3.12.1.sh --skip-license --prefix=/opt/cmake && \
  ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake

# TODO make sure that TBB dependency is satisfied

ENV CC=/clang_6.0.1/bin/clang
ENV CXX=/clang_6.0.1/bin/clang++

WORKDIR /repo/build
CMD [ "/bin/bash" ]