project(SolverRunner LANGUAGES CXX CUDA)

find_package(Thrust REQUIRED)

set(EXE_NAME SolverRunner)
set(LIBRARY_NAME Solver)

add_executable(${EXE_NAME}
        runner/Resolvers.hpp
        runner/main.cu)

target_link_libraries(${EXE_NAME}
        ${LIBRARY_NAME}
        ${CUDA_NVTX_LIBRARY}
        )

include_directories(${MAIN_SOURCE}
        ${EXTERNAL_SOURCE}
        ${CMAKE_CURRENT_BINARY_DIR}
        )