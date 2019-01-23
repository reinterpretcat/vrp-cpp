project(SolverExample LANGUAGES CXX)

set(LIBRARY_EXAMPLE SolverExample)

include_directories(${MAIN_SOURCE}
        ${EXTERNAL_SOURCE}/catch/src
        ${EXTERNAL_SOURCE}/range-v3/include
        ${EXTERNAL_SOURCE}/parallelstl/include
        ${EXTERNAL_SOURCE}/tbb/include
        ${EXAMPLE_SOURCE}
        ${CMAKE_CURRENT_BINARY_DIR})

set(SOURCE_FILES
       SolverExample.cc

        )

#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

add_executable(${LIBRARY_EXAMPLE}
        ${SOURCE_FILES}
        )

set_target_properties(${LIBRARY_EXAMPLE} PROPERTIES
        LINKER_LANGUAGE CXX
        )

target_link_libraries(${LIBRARY_EXAMPLE} ${TBB_LIBRARIES})