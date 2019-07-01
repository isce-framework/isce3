if (WITH_CUDA)
    message(STATUS "We will be building CUDA lib")
    add_subdirectory(cuda)
endif(WITH_CUDA)
