# Get name of ISCE CUDA library
include(IsceCuda)
include(IsceHelper)

# Create library target
add_library(${LISCECUDA} SHARED)

# Pull in the source files
add_subdirectory(core)
add_subdirectory(geometry)
add_subdirectory(image)

# Force nvcc to compile with c++14
get_target_property(MYPROPS ${LISCECUDA} COMPILE_OPTIONS)
string(REPLACE "c++17" "c++14" MYPROPS ${MYPROPS})
set_target_properties(${LISCECUDA} PROPERTIES COMPILE_OPTIONS ${MYPROPS})

# Allow for separable compilation and device code linking (-dc flag to nvcc)
set_target_properties(${LISCECUDA} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${LISCECUDA} PROPERTIES CUDA_HOST_COMPILATION_CPP ON)

# Add library dependencies
target_link_libraries(${LISCECUDA} ${LISCE})
include_directories(${CEREAL_SRC_INCLUDEDIR}
                    ${ISCE_BUILDINCLUDEDIR}
                    ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/cuda/${PACKAGENAME}
                    ${PYRE_INCLUDE_DIR}
                    ${GDAL_INCLUDE_DIR})

# Install CUDA helper header files
add_isce_libdir("." "" "helper_cuda.h;helper_string.h")

# Install shared library
install(TARGETS ${LISCECUDA}
        DESTINATION ${ISCE_LIBDIR}
        COMPONENT ISCE_CUDA)
