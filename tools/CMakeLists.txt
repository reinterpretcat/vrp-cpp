project(SolverRunner LANGUAGES CXX)

set(EXE_NAME SolverRunner)
set(LIBRARY_NAME Solver)

set(SOURCE_FILES
        runner/main.cc
    )

add_executable(${EXE_NAME}
        ${SOURCE_FILES})

target_link_libraries(${EXE_NAME}
        ${LIBRARY_NAME}
        )

include_directories(${MAIN_SOURCE}
        ${EXTERNAL_SOURCE}
        ${CMAKE_CURRENT_BINARY_DIR}
        )
