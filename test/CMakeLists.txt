project(SolverTest LANGUAGES CXX CUDA)

set(LIBRARY_NAME Solver)
set(LIBRARY_TEST SolverTest)

include_directories(${MAIN_SOURCE}
        ${EXTERNAL_SOURCE}
        ${TEST_SOURCE}
        ${CMAKE_CURRENT_BINARY_DIR})

set(SOURCE_FILES
        EnvironmentTest.cu

        algorithms/convolutions/BestConvolutionsTest.cu
        algorithms/convolutions/JointConvolutionsTest.cu
        algorithms/convolutions/SlicedConvolutionsTest.cu

        algorithms/transitions/ExecutorsTest.cu
        algorithms/transitions/FactoriesTest.cu

        algorithms/costs/CostsTest.cu

        algorithms/genetic/CrossoverTest.cu
        algorithms/genetic/EvolutionTest.cu
        algorithms/genetic/MutationTest.cu
        algorithms/genetic/PopulationTest.cu
        algorithms/genetic/SelectionTest.cu
        algorithms/genetic/StrategiesTest.cu

        algorithms/heuristics/NearestNeighborTest.cu
        algorithms/heuristics/RandomInsertionTest.cu

        runtime/ConfigTest.cu
        runtime/Memory.cu
        runtime/Types.cu

        streams/input/SolomonReaderTest.cu
        streams/output/GeoJsonWriterTest.cu

        utils/Validations.cu
        )

set(EXTRA_LINK_LIBS "")
if (SOLVER_BACKEND STREQUAL OMP)
    CompileWithOpenMP()
    LinkWithOpenMP()
endif ()

add_executable(${LIBRARY_TEST}
        config.hpp
        main.cpp
        test_utils/ConvolutionUtils.hpp
        test_utils/PopulationFactory.hpp
        test_utils/ProblemStreams.hpp
        test_utils/SolomonBuilder.hpp
        test_utils/TaskUtils.hpp
        test_utils/VectorUtils.hpp

        ${SOURCE_FILES}
        )


set_target_properties(${LIBRARY_TEST} PROPERTIES
        LINKER_LANGUAGE CXX
        CUDA_SEPARABLE_COMPILATION ON
        )

target_link_libraries(${LIBRARY_TEST} PRIVATE
        ${LIBRARY_NAME}
        ${EXTRA_LINK_LIBS}
        )

# Define test assets path
file(READ ${TEST_SOURCE}/config.hpp FILE_CONTENT)
string(REPLACE "_RESOURCES_PATH_" "${CMAKE_SOURCE_DIR}/resources"
        MODIFIED_FILE_CONTENT "${FILE_CONTENT}")
file(WRITE ${TEST_SOURCE}/config.hpp "${MODIFIED_FILE_CONTENT}")