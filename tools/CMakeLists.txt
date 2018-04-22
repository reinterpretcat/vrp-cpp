project(Solver LANGUAGES CXX CUDA)

find_package(Thrust REQUIRED)

set(LIBRARY_NAME Solver)

include_directories(${MAIN_SOURCE}
        ${EXTERNAL_SOURCE}
        ${CMAKE_CURRENT_BINARY_DIR}
        )

set(EXE_NAME SolverRunner)
add_executable(${EXE_NAME} runner/main.cu)
target_link_libraries(${EXE_NAME}
        ${LIBRARY_NAME}
        ${CUDA_NVTX_LIBRARY}
        )