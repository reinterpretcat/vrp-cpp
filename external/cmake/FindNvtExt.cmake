find_library(CUDA_NVTX_LIBRARY nvToolsExt
             PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

if(${CUDA_NVTX_LIBRARY})
    message(STATUS "Found nvToolsExt: ${CUDA_NVTX_LIBRARY}")
elseif()
    message(FATAL_ERROR "Cannot find nvToolsExt library")
endif()
