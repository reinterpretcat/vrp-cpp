project(SolverTest LANGUAGES CXX)

set(LIBRARY_TEST SolverTest)

include_directories(${MAIN_SOURCE}
        ${EXTERNAL_SOURCE}/catch/src
        ${EXTERNAL_SOURCE}/json/include
        ${EXTERNAL_SOURCE}/range-v3/include
        ${EXTERNAL_SOURCE}/parallelstl/include
        ${EXTERNAL_SOURCE}/tbb/include
        ${TEST_SOURCE}
        ${CMAKE_CURRENT_BINARY_DIR})

set(SOURCE_FILES

        algorithms/construction/constraints/ActorActivityTimingTest.cc
        algorithms/construction/constraints/ActorJobLockTest.cc
        algorithms/construction/constraints/ConditionalJobTest.cc
        algorithms/construction/constraints/VehicleActivitySizeTest.cc

        algorithms/construction/extensions/SortersTest.cc

        algorithms/construction/heuristics/BlinkInsertionTest.cc
        algorithms/construction/heuristics/CheapestInsertionTest.cc
        algorithms/construction/InsertionConstraintTest.cc
        algorithms/construction/InsertionEvaluatorTest.cc

        algorithms/objectives/PenalizeUnassignedJobsTest.cc

        algorithms/refinement/acceptance/GreedyAcceptanceTest.cc
        algorithms/refinement/acceptance/ThresholdAcceptanceTest.cc
        algorithms/refinement/extensions/RemoveEmptyToursTest.cc
        algorithms/refinement/extensions/RuinAndRecreateSolutionTest.cc
        algorithms/refinement/extensions/RestoreInsertionContextTest.cc
        algorithms/refinement/recreate/RecreateWithBlinksTest.cc
        algorithms/refinement/ruin/RemoveAdjustedStringTest.cc
        algorithms/refinement/ruin/RemoveRandomRoutesTest.cc
        algorithms/refinement/ruin/RuinWithProbabilitiesTest.cc
        algorithms/refinement/termination/MaxIterationCriteriaTest.cc

        models/extensions/problem/DistancesTest.cc
        models/problem/FleetTest.cc
        models/problem/JobsTest.cc
        models/solution/TourTest.cc
        models/solution/RegistryTest.cc

        test_utils/StreamSolver.hpp
        test_utils/algorithms/acceptance/Factories.hpp
        test_utils/algorithms/construction/Factories.hpp
        test_utils/algorithms/construction/Contexts.hpp
        test_utils/algorithms/construction/Results.hpp
        test_utils/algorithms/construction/constraints/Helpers.hpp
        test_utils/algorithms/refinement/MatrixRoutes.hpp
        test_utils/algorithms/refinement/LogAndValidate.hpp
        test_utils/fakes/FakeDistribution.hpp
        test_utils/fakes/TestTransportCosts.hpp
        test_utils/models/Comparators.hpp
        test_utils/models/Helpers.hpp
        test_utils/models/Factories.hpp
        test_utils/models/Matchers.hpp
        test_utils/models/Validators.hpp
        test_utils/streams/LiLimBuilder.hpp
        test_utils/streams/LiLimStreams.hpp
        test_utils/streams/SolomonBuilder.hpp
        test_utils/streams/SolomonStreams.hpp

        streams/in/json/RichJsonTest.cc
        streams/in/scientific/LiLimTest.cc
        streams/in/scientific/SolomonTest.cc

        SolverTest.cc

        )

#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

add_executable(${LIBRARY_TEST}
        main.cpp
        ${SOURCE_FILES}
        )

set_target_properties(${LIBRARY_TEST} PROPERTIES
        LINKER_LANGUAGE CXX
        )

target_link_libraries(${LIBRARY_TEST}
        ${CMAKE_THREAD_LIBS_INIT}
        ${TBB_LIBRARIES}
        )