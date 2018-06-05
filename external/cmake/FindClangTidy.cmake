find_program(CLANG_TIDY clang-tidy)
if(NOT CLANG_TIDY)
    message(STATUS "Could not find clang-tidy.")
else()
    message(STATUS "Found clang-tidy.")

    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
    set(CLANG_TIDY_CHECKS "*")
endif()
