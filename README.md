# Description

A prototype of rich Vehicle Routing Problem solver.


# Build in docker

* Make sure that you have all git submodules fetched:

        git submodule update --recursive --remote

* Compile TBB

        # build tbb from https://github.com/philipp-classen/tbb-static-linking-tutorial
        make extra_inc=big_iron.inc

* Modify root CakeLists.txt to use proper TBB library path, e.g.:

        set(TBB_LIBRARY ${PROJECT_SOURCE_DIR}/external/tbb/build/linux_intel64_gcc_cc7.4.0_libc2.27_kernel5.0.0_release)

* Build docker image and run container:

        docker build -t solverex .
        docker run -it -v $(pwd):/repo --rm solverex

* Compile

        mkdir build
        cd build
        cmake ..
        make