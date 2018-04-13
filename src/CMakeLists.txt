project(Solver LANGUAGES CXX CUDA)

find_package(Thrust REQUIRED)

set(LIBRARY_NAME Solver)
set(EXE_NAME SolverRunner)

set(HEADER_FILES
        iterators/CartesianProduct.cu
        algorithms/Distances.cu
        algorithms/Costs.cu
        algorithms/Transitions.cu

        heuristics/NearestNeighbor.hpp
        heuristics/NoTransition.cu

        models/Customers.hpp
        models/Problem.hpp
        models/Resources.hpp
        models/RoutingMatrix.hpp
        models/Tasks.hpp
        models/Transition.hpp

        solver/genetic/Populations.hpp
        solver/genetic/Settings.hpp

        streams/input/SolomonReader.cu
        streams/output/GeoJsonWriter.hpp

        utils/Locations.hpp
        )

set(SOURCE_FILES
        ../external/json/json11.cpp

        heuristics/NearestNeighbor.cu
        solver/genetic/Populations.cu
        streams/output/GeoJsonWriter.cu)

add_library(${LIBRARY_NAME} STATIC
        ${HEADER_FILES}
        ${SOURCE_FILES}
        )

target_compile_features(${LIBRARY_NAME} PUBLIC
        cxx_std_11
        )

set_target_properties(${LIBRARY_NAME} PROPERTIES
        LINKER_LANGUAGE CXX
        CUDA_SEPARABLE_COMPILATION ON
        )

include_directories(${MAIN_SOURCE}
        ${EXTERNAL_SOURCE}
        ${CMAKE_CURRENT_BINARY_DIR}
        )

#add_executable(${EXE_NAME} Runner.cu)
#target_link_libraries(${EXE_NAME} ${LIBRARY_NAME})