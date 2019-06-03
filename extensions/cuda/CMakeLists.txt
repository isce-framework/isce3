# Get name of ISCE CUDA library
include(IsceCuda)
include(IsceHelper)

# Create library target
add_library(${LISCECUDA} SHARED)

# Pull in the source files
add_subdirectory(core)
add_subdirectory(except)
add_subdirectory(geometry)
add_subdirectory(image)
add_subdirectory(io)
add_subdirectory(signal)
add_subdirectory(matchtemplate)
add_subdirectory(orbit_wip)

# Allow for separable compilation and device code linking (-dc flag to nvcc)
set_target_properties(${LISCECUDA} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${LISCECUDA} PROPERTIES CUDA_HOST_COMPILATION_CPP ON)

# Add library dependencies
find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
target_link_libraries(${LISCECUDA} ${CUDART_LIBRARY})
find_library(CUDAFFT_LIBRARY cufft ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
target_link_libraries(${LISCECUDA} ${CUDAFFT_LIBRARY})
target_link_libraries(${LISCECUDA} ${LISCE})
include_directories(${CEREAL_SRC_INCLUDEDIR}
                    ${ISCE_BUILDINCLUDEDIR}
                    ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/cuda/${PACKAGENAME}
                    ${PYRE_INCLUDE_DIR}
                    ${GDAL_INCLUDE_DIR}
                    ${HDF5_INCLUDE_DIR}
                    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Install shared library
install(TARGETS ${LISCECUDA}
        DESTINATION ${ISCE_LIBDIR}
        COMPONENT ISCE_CUDA)
