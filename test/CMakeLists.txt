project(SolverTest LANGUAGES CXX)

set(LIBRARY_NAME Solver)
set(LIBRARY_TEST SolverTest)

include_directories(${MAIN_SOURCE}
        ${EXTERNAL_SOURCE}
        ${TEST_SOURCE}
        ${CMAKE_CURRENT_BINARY_DIR})

set(SOURCE_FILES
        models/solution/TourTest.cc testUtils/models/Factories.hpp)

add_executable(${LIBRARY_TEST}
        main.cpp
        ${SOURCE_FILES}
        )


set_target_properties(${LIBRARY_TEST} PROPERTIES
        LINKER_LANGUAGE CXX
        CUDA_SEPARABLE_COMPILATION ON
        )

target_link_libraries(${LIBRARY_TEST} PRIVATE
        ${LIBRARY_NAME}
        )
