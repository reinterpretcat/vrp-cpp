cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(vrp)

set(MAIN_SOURCE ${PROJECT_SOURCE_DIR}/src)
set(TEST_SOURCE ${PROJECT_SOURCE_DIR}/test)
set(EXTERNAL_SOURCE ${PROJECT_SOURCE_DIR}/external)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${EXTERNAL_SOURCE}/cmake/")

add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(tools)
