add_subdirectory(cython)

if (CMAKE_CUDA_COMPILER)
    message(STATUS "We will be building CUDA lib")
    add_subdirectory(cuda)
endif(CMAKE_CUDA_COMPILER)
