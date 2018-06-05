project(Solver LANGUAGES CXX CUDA)

find_package(Thrust REQUIRED)

set(LIBRARY_NAME Solver)

set(HEADER_FILES
        algorithms/convolutions/BestConvolutions.hpp
        algorithms/convolutions/JointConvolutions.hpp
        algorithms/convolutions/SlicedConvolutions.hpp
        algorithms/convolutions/Models.hpp

        algorithms/costs/Models.hpp
        algorithms/costs/SolutionCosts.hpp
        algorithms/costs/TransitionCosts.hpp

        algorithms/distances/Cartesian.hpp
        algorithms/distances/Geographic.hpp

        algorithms/genetic/Crossovers.hpp
        algorithms/genetic/Models.hpp
        algorithms/genetic/Populations.hpp

        algorithms/heuristics/Dummy.hpp
        algorithms/heuristics/NearestNeighbor.hpp

        algorithms/transitions/Factories.hpp
        algorithms/transitions/Executors.hpp

        iterators/Aggregates.hpp
        iterators/CartesianProduct.hpp

        models/Convolution.hpp
        models/Customers.hpp
        models/Problem.hpp
        models/Resources.hpp
        models/RoutingMatrix.hpp
        models/Solution.hpp
        models/Tasks.hpp
        models/Transition.hpp
        models/Locations.hpp

        streams/input/SolomonReader.hpp
        streams/output/GeoJsonWriter.hpp
        streams/output/MatrixTextWriter.hpp

        utils/geo/Resolvers.hpp
        utils/memory/Allocations.hpp
        utils/memory/DevicePool.hpp
        utils/memory/DeviceUnique.hpp
        )

set(SOURCE_FILES
        ../external/json/json11.cpp

        algorithms/convolutions/BestConvolutions.cu
        algorithms/convolutions/JointConvolutions.cu
        algorithms/convolutions/SlicedConvolutions.cu

        algorithms/costs/SolutionCosts.cu
        algorithms/costs/TransitionCosts.cu

        algorithms/genetic/Populations.cu
        algorithms/genetic/Crossovers.cu

        algorithms/heuristics/Dummy.cu
        algorithms/heuristics/NearestNeighbor.cu

        algorithms/transitions/Factories.cu
        algorithms/transitions/Executors.cu

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
