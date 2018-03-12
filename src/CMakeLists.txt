project(Solver LANGUAGES CXX CUDA)

find_package(Thrust REQUIRED)

set(LIBRARY_NAME Solver)

set(HEADER_FILES
        algorithms/CartesianProduct.cu
        algorithms/Costs.cu
        algorithms/Transitions.cu

        models/Customers.hpp
        models/Problem.hpp
        models/Resources.hpp
        models/RoutingMatrix.hpp
        models/Tasks.hpp
        models/Transition.hpp

        solver/genetic/PopulationFactory.hpp
        solver/genetic/Settings.hpp

        streams/input/SolomonReader.cu
        streams/output/VectorWriter.hpp
        )

set(SOURCE_FILES
        solver/genetic/PopulationFactory.cu
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
