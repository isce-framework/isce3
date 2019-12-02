# Provide this location as a preprocessor macro
configure_file(testdata.h testdata.h)
add_compile_options(-include ${CMAKE_CURRENT_BINARY_DIR}/testdata.h)

# the libisce tests
add_subdirectory(isce)

# the libiscecuda tests
if (WITH_CUDA)
    add_subdirectory(iscecuda)
endif(WITH_CUDA)

# end of file
