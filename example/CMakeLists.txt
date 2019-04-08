project(SolverExample LANGUAGES CXX)

set(SOLVER_EXE SolverExe)
set(SOLVER_LIBRARY SolverLib)

include_directories(${MAIN_SOURCE}
        ${EXTERNAL_SOURCE}/json/include
        ${EXTERNAL_SOURCE}/range-v3/include
        ${EXTERNAL_SOURCE}/parallelstl/include
        ${EXTERNAL_SOURCE}/tbb/include
        ${EXAMPLE_SOURCE}
        ${CMAKE_CURRENT_BINARY_DIR})


#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2")
#set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -O2")

add_executable(${SOLVER_EXE}
        SolverExe.cc
        )

add_library(
        ${SOLVER_LIBRARY} SHARED
        SolverLib.cc
)

foreach(TARGET ${SOLVER_EXE} ${SOLVER_LIBRARY})
    set_target_properties(${TARGET} PROPERTIES
            LINKER_LANGUAGE CXX
            )

    target_link_libraries(${TARGET}
            ${CMAKE_THREAD_LIBS_INIT}
            ${TBB_LIBRARIES}
            )
endforeach()

