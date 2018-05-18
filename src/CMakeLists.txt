project(Solver LANGUAGES CXX CUDA)

find_package(Thrust REQUIRED)

set(LIBRARY_NAME Solver)

set(HEADER_FILES
        algorithms/convolutions/BestConvolutions.hpp
        algorithms/convolutions/JointConvolutions.hpp
        algorithms/convolutions/Models.hpp

        algorithms/costs/Models.hpp
        algorithms/costs/SolutionCosts.hpp
        algorithms/costs/TransitionCosts.hpp

        algorithms/distances/Cartesian.hpp
        algorithms/distances/Geographic.hpp

        algorithms/heuristics/Dummy.hpp
        algorithms/heuristics/NearestNeighbor.hpp

        algorithms/transitions/Factories.hpp
        algorithms/transitions/Executors.hpp

        iterators/CartesianProduct.cu

        models/Convolution.hpp
        models/Customers.hpp
        models/Problem.hpp
        models/Resources.hpp
        models/RoutingMatrix.hpp
        models/Tasks.hpp
        models/Transition.hpp
        models/Locations.hpp

        solver/genetic/Populations.hpp
        solver/genetic/Settings.hpp
        solver/genetic/crossovers/AdjustedCostDifference.hpp

        streams/input/SolomonReader.hpp
        streams/output/GeoJsonWriter.hpp
        streams/output/MatrixTextWriter.hpp

        utils/Memory.hpp
        utils/Pool.hpp
        utils/Resolvers.hpp
        )

set(SOURCE_FILES
        ../external/json/json11.cpp

        algorithms/convolutions/BestConvolutions.cu
        algorithms/convolutions/JointConvolutions.cu

        algorithms/costs/SolutionCosts.cu
        algorithms/costs/TransitionCosts.cu

        algorithms/heuristics/Dummy.cu
        algorithms/heuristics/NearestNeighbor.cu

        algorithms/transitions/Factories.cu
        algorithms/transitions/Executors.cu

        solver/genetic/Populations.cu
        solver/genetic/crossovers/AdjustedCostDifference.cu

        streams/input/SolomonReader.cu
        streams/output/GeoJsonWriter.cu
        streams/output/MatrixTextWriter.cu
        )

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
