project(SolverTest LANGUAGES CXX)

set(LIBRARY_TEST SolverTest)

include_directories(${MAIN_SOURCE}
        ${EXTERNAL_SOURCE}/catch/src
        ${EXTERNAL_SOURCE}/range-v3/include
        ${EXTERNAL_SOURCE}/parallelstl/include
        ${EXTERNAL_SOURCE}/tbb/include
        ${TEST_SOURCE}
        ${CMAKE_CURRENT_BINARY_DIR})

set(SOURCE_FILES

        algorithms/construction/constraints/VehicleActivitySizeTest.cc
        algorithms/construction/constraints/VehicleActivityTimingTest.cc
        algorithms/construction/constraints/VehicleFixedCostTest.cc

        algorithms/construction/evaluators/ServiceInsertionEvaluatorTest.cc
        algorithms/construction/evaluators/JobInsertionEvaluatorTest.cc
        algorithms/construction/heuristics/CheapestInsertionTest.cc
        algorithms/construction/InsertionConstraintTest.cc
        algorithms/construction/InsertionEvaluatorTest.cc

        algorithms/ruin/JobNeighbourhoodTest.cc

        models/problem/FleetTest.cc
        models/solution/TourTest.cc
        models/solution/RegistryTest.cc

        test_utils/algorithms/construction/Insertions.hpp
        test_utils/algorithms/construction/Contexts.hpp
        test_utils/algorithms/construction/Results.hpp
        test_utils/algorithms/constraints/Helpers.hpp
        test_utils/fakes/TestTransportCosts.hpp
        test_utils/models/Comparators.hpp
        test_utils/models/Extensions.hpp
        test_utils/models/Factories.hpp
        test_utils/models/Matchers.hpp
        test_utils/streams/SolomonBuilder.hpp
        test_utils/streams/SolomonStreams.hpp

        streams/in/SolomonTest.cc

        )

add_executable(${LIBRARY_TEST}
        main.cpp
        ${SOURCE_FILES}
        )

set_target_properties(${LIBRARY_TEST} PROPERTIES
        LINKER_LANGUAGE CXX
        )
