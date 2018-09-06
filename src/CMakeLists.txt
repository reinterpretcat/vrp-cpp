project(Solver LANGUAGES CXX CUDA)

find_package(Thrust REQUIRED)
find_package(ClangTidy)

set(LIBRARY_NAME Solver)

set(HEADER_FILES
        algorithms/common/Convolutions.hpp
        algorithms/common/Tours.hpp

        algorithms/convolutions/BestConvolutions.hpp
        algorithms/convolutions/JointConvolutions.hpp
        algorithms/convolutions/SlicedConvolutions.hpp
        algorithms/convolutions/Models.hpp

        algorithms/costs/SolutionCosts.hpp
        algorithms/costs/TransitionCosts.hpp

        algorithms/distances/Cartesian.hpp
        algorithms/distances/Geographic.hpp

        algorithms/genetic/Crossovers.hpp
        algorithms/genetic/Evolution.hpp
        algorithms/genetic/Models.hpp
        algorithms/genetic/Mutations.hpp
        algorithms/genetic/Populations.hpp
        algorithms/genetic/Selection.hpp
        algorithms/genetic/Strategies.hpp

        algorithms/heuristics/CheapestInsertion.hpp
        algorithms/heuristics/ConvolutionInsertion.hpp
        algorithms/heuristics/Dummy.hpp
        algorithms/heuristics/Models.hpp
        algorithms/heuristics/NearestNeighbor.hpp
        algorithms/heuristics/RandomInsertion.hpp

        algorithms/transitions/Factories.hpp
        algorithms/transitions/Executors.hpp

        iterators/Aggregates.hpp
        iterators/CartesianProduct.hpp
        iterators/Sequential.hpp

        models/Convolution.hpp
        models/Customers.hpp
        models/Plan.hpp
        models/Problem.hpp
        models/Resources.hpp
        models/RoutingMatrix.hpp
        models/Solution.hpp
        models/Tasks.hpp
        models/Transition.hpp

        runtime/Config.hpp
        runtime/Variant.hpp
        runtime/UniquePointer.hpp

        streams/input/SolomonReader.hpp
        streams/output/GeoJsonWriter.hpp
        streams/output/MatrixTextWriter.hpp

        utils/random/FilteredDistribution.hpp
        utils/random/TransformedDistribution.hpp
        utils/random/Visualizer.hpp

        utils/templates/IndexOf.hpp
        utils/templates/OneOf.hpp

        utils/validation/SolutionChecker.hpp
        )

set(SOURCE_FILES
        ../external/json/json11.cpp

        algorithms/convolutions/BestConvolutions.cu
        algorithms/convolutions/JointConvolutions.cu
        algorithms/convolutions/SlicedConvolutions.cu

        algorithms/costs/SolutionCosts.cu
        algorithms/costs/TransitionCosts.cu

        algorithms/genetic/Crossovers.cu
        algorithms/genetic/Evolution.cu
        algorithms/genetic/Mutations.cu
        algorithms/genetic/Populations.cu
        algorithms/genetic/Selection.cu
        algorithms/genetic/Strategies.cu

        algorithms/heuristics/CheapestInsertion.cu
        algorithms/heuristics/ConvolutionInsertion.cu
        algorithms/heuristics/Dummy.cu
        algorithms/heuristics/NearestNeighbor.cu
        algorithms/heuristics/RandomInsertion.cu

        algorithms/transitions/Factories.cu
        algorithms/transitions/Executors.cu

        streams/input/SolomonReader.cu
        streams/output/GeoJsonWriter.cu
        streams/output/MatrixTextWriter.cu

        utils/validation/SolutionChecker.cu
        )

if (SOLVER_BACKEND STREQUAL OMP)
    CompileWithOpenMP()
endif()

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

if (CLANG_TIDY)
    add_custom_target(${LIBRARY_NAME}Tidy
        COMMAND ${CLANG_TIDY} -p ${CMAKE_CACHEFILE_DIR}/compile_commands.json
        ${SOURCE_FILES}
        WORKING_DIRECTORY ${MAIN_SOURCE}/
        )
endif()
