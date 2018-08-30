# Provides the easy way to add openMP

# Configures compiler
macro(CompileWithOpenMP)
    #TODO did not work, why?
    #list(APPEND CMAKE_CUDA_FLAGS "-Xcompiler -fopenmp")

    add_definitions("-x c++")
    add_compile_options("-fopenmp")
    set_source_files_properties(${SOURCE_FILES} PROPERTIES LANGUAGE CXX)
    add_compile_definitions(THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP)
endmacro()

# Configures linker
macro(LinkWithOpenMP)
    find_library(GOMP_LIBRARY gomp)

    if (NOT GOMP_LIBRARY)
        message(SEND_ERROR "GOMP library not found!")
    endif()
    set(EXTRA_LINK_LIBS ${GOMP_LIBRARY})
endmacro()
