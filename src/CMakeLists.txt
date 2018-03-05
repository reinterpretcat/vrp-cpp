project(Solver LANGUAGES CXX CUDA)

find_package(Thrust REQUIRED)

set(LIBRARY_NAME Solver)

set(HEADER_FILES
        models/Customers.hpp
        models/Problem.hpp
        models/Resources.hpp
        models/Solutions.hpp
        models/Tasks.hpp

        input/SolomonReader.cu

        utils/CartesianProduct.cu
        utils/StreamUtils.hpp
        )

set(SOURCE_FILES
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
