cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(vrp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(MAIN_SOURCE ${PROJECT_SOURCE_DIR}/src)
set(TEST_SOURCE ${PROJECT_SOURCE_DIR}/test)
set(EXTERNAL_SOURCE ${PROJECT_SOURCE_DIR}/external)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${EXTERNAL_SOURCE}/cmake/")

add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(tools)
