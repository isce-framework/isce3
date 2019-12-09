# libisce
add_subdirectory(isce)

# libiscecuda
if (WITH_CUDA)
    message(STATUS "We will be building CUDA lib")
    add_subdirectory(iscecuda)
endif(WITH_CUDA)

# end of file
