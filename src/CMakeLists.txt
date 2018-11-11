project(Solver LANGUAGES CXX)

find_package(ClangTidy)

set(LIBRARY_NAME Solver)

set(SOURCE_FILES
        Stub.hpp
        Stub.cc

        algorithms/construction/extensions/Insertions.hpp
        algorithms/construction/insertion/InsertionConstraint.hpp
        algorithms/construction/insertion/InsertionContext.hpp
        algorithms/construction/insertion/InsertionEvaluator.hpp
        algorithms/construction/insertion/InsertionResult.hpp
        algorithms/construction/insertion/evaluators/ServiceInsertionEvaluator.hpp
        algorithms/construction/insertion/evaluators/ShipmentInsertionEvaluator.hpp

        models/common/Dimension.hpp
        models/common/Duration.hpp
        models/common/Location.hpp
        models/common/Schedule.hpp
        models/common/Timestamp.hpp
        models/common/TimeWindow.hpp

        models/extensions/problem/Comparators.hpp
        models/extensions/problem/Factories.hpp
        models/extensions/solution/Comparators.hpp
        models/extensions/solution/Factories.hpp

        models/problem/Actor.hpp
        models/problem/Costs.hpp
        models/problem/Driver.hpp
        models/problem/Vehicle.hpp
        models/problem/Routing.hpp
        models/problem/Job.hpp
        models/problem/Service.hpp
        models/problem/Shipment.hpp

        models/solution/Activity.hpp
        models/solution/Route.hpp
        models/solution/Solution.hpp
        models/solution/Statistic.hpp
        models/solution/Tour.hpp

        utils/Extensions.hpp)

add_library(${LIBRARY_NAME} STATIC
        ${SOURCE_FILES}
        )

target_compile_features(${LIBRARY_NAME} PRIVATE
        cxx_std_17
        )

set_target_properties(${LIBRARY_NAME} PROPERTIES
        LINKER_LANGUAGE CXX
        )

include_directories(${MAIN_SOURCE}
        ${EXTERNAL_SOURCE}/range-v3/include
        ${CMAKE_CURRENT_BINARY_DIR}
        )

if (CLANG_TIDY)
    add_custom_target(${LIBRARY_NAME}Tidy
        COMMAND ${CLANG_TIDY} -p ${CMAKE_CACHEFILE_DIR}/compile_commands.json
        ${SOURCE_FILES}
        WORKING_DIRECTORY ${MAIN_SOURCE}/
        )
endif()
