#
# Copyright 2017-2018
#

# Find Numpy
find_package(Numpy REQUIRED)

# Add subdirectories for installing pxd files
add_subdirectory(image)
add_subdirectory(geometry)

# Set file properties
set_source_files_properties(
    iscecudaextension.pyx
    PROPERTIES CYTHON_IS_CXX TRUE
)

#Include for pxd files
include_directories(${ISCE_BUILDCYINCLUDEDIR})

# Add Cython module
cython_add_module(iscecudaextension iscecudaextension.pyx)
if(WITH_CODE_COVERAGE)
    SetCoverageOptions(iscecudaextension)
endif()

# Override output shared object name to place in desired location
set_target_properties(
    iscecudaextension
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${ISCE_BUILDPACKAGESDIR}/isce3/extensions
)

# Add necessary include directories
target_include_directories(
        iscecudaextension PUBLIC
        ${ISCE_BUILDINCLUDEDIR}
        ${ISCE_BUILDINCLUDEDIR_CUDA}
        ${ISCE_BUILDCYINCLUDEDIR}
        ${PYTHON_NUMPY_INCLUDE_DIR}
        ${GDAL_INCLUDE_DIR}
        ${HDF5_INCLUDE_DIR}
        ${CMAKE_SOURCE_DIR}/contrib/cereal/include
        ${PYRE_INCLUDE_DIR}
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Add necessary library links
include(IsceCuda)
target_link_libraries(iscecudaextension ${LISCECUDA} ${LISCE} m)

# Install
install(
    TARGETS iscecudaextension
    DESTINATION ${ISCE_PACKAGESDIR}/isce3/extensions
    COMPONENT extensions
)

# end of file
