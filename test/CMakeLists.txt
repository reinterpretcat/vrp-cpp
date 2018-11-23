project(SolverTest LANGUAGES CXX)

set(LIBRARY_TEST SolverTest)

include_directories(${MAIN_SOURCE}
        ${EXTERNAL_SOURCE}/catch/src
        ${EXTERNAL_SOURCE}/range-v3/include
        ${TEST_SOURCE}
        ${CMAKE_CURRENT_BINARY_DIR})

set(SOURCE_FILES

        algorithms/construction/insertion/constraints/VehicleActivityTimingTest.cc
        algorithms/construction/insertion/evaluators/ServiceInsertionEvaluatorTest.cc
        algorithms/construction/insertion/evaluators/JobInsertionEvaluatorTest.cc
        algorithms/construction/insertion/InsertionConstraintTest.cc

        models/solution/TourTest.cc
        models/solution/RegistryTest.cc

        test_utils/algorithms/construction/Insertions.hpp
        test_utils/algorithms/construction/Contexts.hpp
        test_utils/fakes/TestTransportCosts.hpp
        test_utils/models/Factories.hpp
        test_utils/models/Matchers.hpp
        )

add_executable(${LIBRARY_TEST}
        main.cpp
        ${SOURCE_FILES}
        )

set_target_properties(${LIBRARY_TEST} PROPERTIES
        LINKER_LANGUAGE CXX
        )
