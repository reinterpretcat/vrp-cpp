project(Solver LANGUAGES CXX)

find_package(ClangTidy)

set(SOURCE_FILES
        algorithms/construction/constraints/VehicleActivitySize.hpp
        algorithms/construction/constraints/VehicleActivityTiming.hpp

        algorithms/construction/extensions/Constraints.hpp
        algorithms/construction/extensions/Fleets.hpp
        algorithms/construction/extensions/Insertions.hpp
        algorithms/construction/extensions/Routes.hpp
        algorithms/construction/extensions/States.hpp

        algorithms/construction/evaluators/JobInsertionEvaluator.hpp
        algorithms/construction/evaluators/ServiceInsertionEvaluator.hpp
        algorithms/construction/evaluators/ShipmentInsertionEvaluator.hpp

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

        models/common/Cost.hpp
        models/common/Dimension.hpp
        models/common/Distance.hpp
        models/common/Duration.hpp
        models/common/Location.hpp
        models/common/Schedule.hpp
        models/common/Timestamp.hpp
        models/common/TimeWindow.hpp

        models/costs/ActivityCosts.hpp
        models/costs/TransportCosts.hpp

        models/extensions/problem/Comparators.hpp
        models/extensions/problem/Factories.hpp
        models/extensions/solution/Comparators.hpp
        models/extensions/solution/Factories.hpp

        models/problem/Costs.hpp
        models/problem/Driver.hpp
        models/problem/Fleet.hpp
        models/problem/Job.hpp
        models/problem/JobDetail.hpp
        models/problem/Service.hpp
        models/problem/Shipment.hpp
        models/problem/Vehicle.hpp
        models/problem/VehicleDetail.hpp

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

        utils/extensions/Hash.hpp
        utils/extensions/Variant.hpp

        )

include_directories(${MAIN_SOURCE}
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
