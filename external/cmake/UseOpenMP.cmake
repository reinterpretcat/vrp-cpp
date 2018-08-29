# Provides the easy way to add openMP

# Configures compiler
function(CompileOpenMP BINARY_NAME)
    list(APPEND CMAKE_CUDA_FLAGS "-Xcompiler -fopenmp")
    set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} PARENT_SCOPE)
    add_compile_definitions(THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP)
endfunction()

# Configures linker
function(LinkOpenMP BINARY_NAME EXTRA_LINK_LIBS)
    find_library(GOMP_LIBRARY gomp)

    if (NOT GOMP_LIBRARY)
        message(SEND_ERROR "GOMP library not found!")
    endif()
    set(${EXTRA_LINK_LIBS} ${GOMP_LIBRARY} PARENT_SCOPE)

    # TODO avoid duplication
    list(APPEND CMAKE_CUDA_FLAGS "-Xcompiler -fopenmp")
    set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} PARENT_SCOPE)
    add_compile_definitions(THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP)
endfunction()
