cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(vrp)

# configure thrust backend
set(SOLVER_BACKEND "HOST" CACHE STRING "Target thrust backend")
set(SOLVER_BACKENDS HOST OMP TBB CUDA)
set_property(CACHE SOLVER_BACKEND PROPERTY STRINGS ${SOLVER_BACKENDS})
if(NOT SOLVER_BACKEND IN_LIST SOLVER_BACKENDS)
    message(FATAL_ERROR "${SOLVER_BACKEND} must be one of ${SOLVER_BACKENDS}")
endif()
if(${SOLVER_BACKEND} STREQUAL HOST)
    add_definitions(-DUSE_HOST_BACKEND)
elseif(${SOLVER_BACKEND} STREQUAL OMP)
    add_definitions(-DUSE_OMP_BACKEND)
else()
    add_definitions(-DUSE_CUDA_BACKEND)
endif()
message("-- Configured to run on ${SOLVER_BACKEND} backend")


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(MAIN_SOURCE ${PROJECT_SOURCE_DIR}/src)
set(TEST_SOURCE ${PROJECT_SOURCE_DIR}/test)
set(EXTERNAL_SOURCE ${PROJECT_SOURCE_DIR}/external)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${EXTERNAL_SOURCE}/cmake/")
include(UseOpenMP)

add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(tools)


