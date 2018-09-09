project(SolverRunner LANGUAGES CXX CUDA)

find_package(Thrust REQUIRED)

set(EXE_NAME SolverRunner)
set(LIBRARY_NAME Solver)

set(SOURCE_FILES
        runner/main.cu
    )

set(EXTRA_LINK_LIBS "")
if (SOLVER_BACKEND STREQUAL OMP)
    CompileWithOpenMP()
    LinkWithOpenMP()
endif ()

add_executable(${EXE_NAME}
        runner/Resolvers.hpp
        ${SOURCE_FILES})

target_link_libraries(${EXE_NAME}
        ${LIBRARY_NAME}
        ${EXTRA_LINK_LIBS}
        )

include_directories(${MAIN_SOURCE}
        ${EXTERNAL_SOURCE}
        ${CMAKE_CURRENT_BINARY_DIR}
        )