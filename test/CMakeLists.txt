project(SolverTest LANGUAGES CXX CUDA)

set(LIBRARY_NAME Solver)
set(LIBRARY_TEST SolverTest)

include_directories(${MAIN_SOURCE}
        ${EXTERNAL_SOURCE}
        ${TEST_SOURCE}
        ${CMAKE_CURRENT_BINARY_DIR})

add_executable(${LIBRARY_TEST}
        config.hpp
        main.cpp
        EnvironmentTest.cu

        heuristics/CheapestInsertionTest.cu
        streams/input/SolomonReaderTest.cu
        solver/genetic/PopulationFactoryTest.cu

        test_utils/VectorUtils.hpp
        )

set_target_properties(${LIBRARY_TEST} PROPERTIES
        LINKER_LANGUAGE CXX
        CUDA_SEPARABLE_COMPILATION ON
        )

target_link_libraries(${LIBRARY_TEST} PRIVATE
        ${LIBRARY_NAME}
        ${CUDA_LIBRARY}
        )

# Define test assets path
file(READ ${TEST_SOURCE}/config.hpp FILE_CONTENT)
string(REPLACE "_RESOURCES_PATH_" "${CMAKE_SOURCE_DIR}/resources"
        MODIFIED_FILE_CONTENT "${FILE_CONTENT}")
file(WRITE ${TEST_SOURCE}/config.hpp "${MODIFIED_FILE_CONTENT}")