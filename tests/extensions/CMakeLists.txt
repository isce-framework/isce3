add_subdirectory(cython)
if (WITH_CUDA)
    add_subdirectory(cuda)
endif(WITH_CUDA)
