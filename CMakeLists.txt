cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(vrp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(MAIN_SOURCE ${PROJECT_SOURCE_DIR}/src)
set(TEST_SOURCE ${PROJECT_SOURCE_DIR}/test)
set(EXTERNAL_SOURCE ${PROJECT_SOURCE_DIR}/external)

set(TBB_ROOT_DIR ${PROJECT_SOURCE_DIR}/external/tbb)
set(TBB_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/external/tbb/include)
# TODO find nicer way to include lib path
set(TBB_LIBRARY ${PROJECT_SOURCE_DIR}/external/tbb/build/linux_intel64_gcc_cc7_libc2.27_kernel4.15.0_debug)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${EXTERNAL_SOURCE}/cmake/")


find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    message("-- Found ccache")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)

find_package(TBB REQUIRED)

add_subdirectory(src)
add_subdirectory(test)
