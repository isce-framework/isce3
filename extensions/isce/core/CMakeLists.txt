add_subdirectory(cython)

if (CUDA_VERSION)
    add_subdirectory(cuda)
endif()
