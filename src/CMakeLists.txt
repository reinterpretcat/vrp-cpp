project(Solver LANGUAGES CXX)

find_package(ClangTidy)

set(LIBRARY_NAME Solver)

set(HEADER_FILES
        Stub.hpp
        )

set(SOURCE_FILES
        Stub.cc
        )

add_library(${LIBRARY_NAME} STATIC
        ${HEADER_FILES}
        ${SOURCE_FILES}
        )

target_compile_features(${LIBRARY_NAME} PRIVATE
        cxx_std_17
        )

set_target_properties(${LIBRARY_NAME} PROPERTIES
        LINKER_LANGUAGE CXX
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
