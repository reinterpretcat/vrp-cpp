project(SolverTest LANGUAGES CXX)

set(LIBRARY_TEST SolverTest)

include_directories(${MAIN_SOURCE}
        ${EXTERNAL_SOURCE}/catch/src
        ${EXTERNAL_SOURCE}/gsl/include
        ${EXTERNAL_SOURCE}/json/include
        ${EXTERNAL_SOURCE}/range-v3/include
        ${EXTERNAL_SOURCE}/parallelstl/include
        ${EXTERNAL_SOURCE}/tbb/include
        ${TEST_SOURCE}
        ${CMAKE_CURRENT_BINARY_DIR})

set(SOURCE_FILES

        algorithms/construction/constraints/ActorActivityTimingTest.cc
        algorithms/construction/constraints/ActorJobLockTest.cc
        algorithms/construction/constraints/ActorTravelLimitTest.cc
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
        algorithms/refinement/extensions/CreateRefinementContextTest.cc
        algorithms/refinement/extensions/RemoveEmptyToursTest.cc
        algorithms/refinement/extensions/RuinAndRecreateSolutionTest.cc
        algorithms/refinement/extensions/RestoreInsertionContextTest.cc
        algorithms/refinement/recreate/RecreateWithBlinksTest.cc
        algorithms/refinement/ruin/RemoveAdjustedStringTest.cc
        algorithms/refinement/ruin/RemoveRandomRoutesTest.cc
        algorithms/refinement/ruin/RuinWithProbabilitiesTest.cc
        algorithms/refinement/termination/MaxIterationCriteriaTest.cc
        algorithms/refinement/termination/VariationCoefficientCriteriaTest.cc

        models/extensions/problem/DistancesTest.cc
        models/problem/FleetTest.cc
        models/problem/JobsTest.cc
        models/solution/TourTest.cc
        models/solution/RegistryTest.cc

        scenarios/here/breaks/BreakBasicTest.cc
        scenarios/here/breaks/BreakSkipTest.cc
        scenarios/here/breaks/BreakRelationTest.cc
        scenarios/here/fleet/OpenEndBasicTest.cc
        scenarios/here/fleet/ReachableConstraintTest.cc
        scenarios/here/limits/MaxDistanceTest.cc
        scenarios/here/limits/ShiftTimeTest.cc
        scenarios/here/pickdev/PickDevMixedTest.cc
        scenarios/here/pickdev/PickDevBasicTest.cc
        scenarios/here/pickdev/PickDevRelationTest.cc
        scenarios/here/skills/SkillsBasicTest.cc
        scenarios/here/skills/SkillsUnassignedTest.cc
        scenarios/here/relations/BasicSplitStrictNewJobsTest.cc
        scenarios/here/relations/BasicSplitStrictNoNewJobsTest.cc
        scenarios/here/relations/BasicStrictTourNewJobs.cc
        scenarios/here/relations/BasicTourTest.cc
        scenarios/here/relations/MixedSequenceStrictOneVehicleOneNewJob.cc
        scenarios/here/relations/MixedSequenceStrictTwoVehiclesNoNewJobs.cc
        scenarios/here/times/AvoidInitialWaitTest.cc
        scenarios/here/times/TimeStrictUnassignedTest.cc
        scenarios/here/times/TimeStrictTwoToursTest.cc
        scenarios/here/times/TimeMultipleBasicTest.cc
        scenarios/here/times/WaitingBasicTest.cc

        test_utils/Solvers.hpp
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
        test_utils/scenarios/here/Assertions.hpp
        test_utils/scenarios/here/Helpers.hpp
        test_utils/scenarios/here/Variables.hpp
        test_utils/streams/HereModelBuilders.hpp
        test_utils/streams/LiLimBuilder.hpp
        test_utils/streams/LiLimStreams.hpp
        test_utils/streams/SolomonBuilder.hpp
        test_utils/streams/SolomonStreams.hpp

        streams/in/json/HereProblemJsonTest.cc
        streams/in/json/RichProblemJsonTest.cc
        streams/in/scientific/LiLimTest.cc
        streams/in/scientific/SolomonTest.cc
        streams/out/json/HereSolutionJsonTest.cc

        utils/DateTest.cc
        utils/PermutationsTest.cc

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