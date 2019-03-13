project(Solver LANGUAGES CXX)

find_package(ClangTidy)

set(SOURCE_FILES

        algorithms/construction/constraints/ActorActivityTiming.hpp
        algorithms/construction/constraints/ActorJobLock.hpp
        algorithms/construction/constraints/ConditionalJob.hpp
        algorithms/construction/constraints/VehicleActivitySize.hpp

        algorithms/construction/extensions/Comparators.hpp
        algorithms/construction/extensions/Constraints.hpp
        algorithms/construction/extensions/Factories.hpp
        algorithms/construction/extensions/Sorters.hpp

        algorithms/construction/heuristics/BlinkInsertion.hpp
        algorithms/construction/heuristics/CheapestInsertion.hpp

        algorithms/construction/InsertionActivityContext.hpp
        algorithms/construction/InsertionConstraint.hpp
        algorithms/construction/InsertionContext.hpp
        algorithms/construction/InsertionEvaluator.hpp
        algorithms/construction/InsertionHeuristic.hpp
        algorithms/construction/InsertionResult.hpp
        algorithms/construction/InsertionRouteContext.hpp
        algorithms/construction/InsertionRouteState.hpp
        algorithms/construction/InsertionProgress.hpp
        algorithms/construction/InsertionSolutionContext.hpp


        algorithms/objectives/ObjectiveFunction.hpp
        algorithms/objectives/PenalizeUnassignedJobs.hpp


        algorithms/refinement/acceptance/GreedyAcceptance.hpp
        algorithms/refinement/acceptance/ThresholdAcceptance.hpp

        algorithms/refinement/extensions/CreateRefinementContext.hpp
        algorithms/refinement/extensions/RemoveEmptyTours.hpp
        algorithms/refinement/extensions/RuinAndRecreateSolution.hpp
        algorithms/refinement/extensions/RestoreInsertionContext.hpp
        algorithms/refinement/extensions/SelectBestSolution.hpp
        algorithms/refinement/extensions/SelectRandomJob.hpp

        algorithms/refinement/logging/LogToConsole.hpp
        algorithms/refinement/logging/LogToNothing.hpp


        algorithms/refinement/recreate/RecreateWithBlinks.hpp

        algorithms/refinement/ruin/RemoveAdjustedString.hpp
        algorithms/refinement/ruin/RemoveRandomRoutes.hpp
        algorithms/refinement/ruin/RuinWithProbabilities.hpp

        algorithms/refinement/termination/MaxIterationCriteria.hpp


        models/common/Cost.hpp
        models/common/Dimension.hpp
        models/common/Distance.hpp
        models/common/Duration.hpp
        models/common/Location.hpp
        models/common/Schedule.hpp
        models/common/Timestamp.hpp
        models/common/TimeWindow.hpp

        models/costs/ActivityCosts.hpp
        models/costs/MatrixTransportCosts.hpp
        models/costs/TransportCosts.hpp

        models/extensions/problem/Comparators.hpp
        models/extensions/problem/Distances.hpp
        models/extensions/problem/Factories.hpp
        models/extensions/problem/Helpers.hpp

        models/extensions/solution/Comparators.hpp
        models/extensions/solution/DeepCopies.hpp
        models/extensions/solution/Factories.hpp
        models/extensions/solution/Helpers.hpp

        models/problem/Costs.hpp
        models/problem/Driver.hpp
        models/problem/Fleet.hpp
        models/problem/Job.hpp
        models/problem/Jobs.hpp
        models/problem/Service.hpp
        models/problem/Sequence.hpp
        models/problem/Vehicle.hpp

        models/solution/Actor.hpp
        models/solution/Activity.hpp
        models/solution/Registry.hpp
        models/solution/Route.hpp
        models/solution/Solution.hpp
        models/solution/Statistic.hpp
        models/solution/Tour.hpp

        models/Problem.hpp
        models/Solution.hpp

        streams/in/Solomon.hpp
        streams/in/RichJson.hpp
        streams/in/LiLim.hpp
        streams/in/extensions/SolomonCosts.hpp
        streams/out/DumpSolutionAsText.hpp

        utils/Collections.hpp
        utils/Random.hpp
        utils/Measure.hpp
        utils/extensions/Hash.hpp
        utils/extensions/Ranges.hpp

        Solver.hpp

        )

include_directories(${MAIN_SOURCE}
        ${EXTERNAL_SOURCE}/json/include
        ${EXTERNAL_SOURCE}/range-v3/include
        ${EXTERNAL_SOURCE}/parallelstl/include
        ${EXTERNAL_SOURCE}/tbb/include
        ${CMAKE_CURRENT_BINARY_DIR}
        )

if (CLANG_TIDY)
    add_custom_target(${LIBRARY_NAME}Tidy
        COMMAND ${CLANG_TIDY} -p ${CMAKE_CACHEFILE_DIR}/compile_commands.json
        ${SOURCE_FILES}
        WORKING_DIRECTORY ${MAIN_SOURCE}/
        )
endif()
