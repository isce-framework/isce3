#
# Copyright 2017-2018
#

# Find Numpy
find_package(Numpy REQUIRED)

# Set the name for the ISCE C++ libraries
set(LISCE isce.${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR})
set(LISCECUDA iscecuda.${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR})

# Set file properties
set_source_files_properties(
    iscecudaextension.pyx
    PROPERTIES CYTHON_IS_CXX TRUE
)

#Include for pxd files
include_directories(${ISCE_BUILDCYINCLUDEDIR})

# Tweak the binary director to place extensions alongside isceextension
set(CMAKE_CURRENT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/..)

# Initialize build directory for extension module to mimic install directory
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/isce3/extensions)

# Add Cython module
cython_add_module(iscecudaextension iscecudaextension.pyx)

# Override output shared object name to place in desired location
set_target_properties(
    iscecudaextension
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/isce3/extensions
)

# Add necessary include directories
target_include_directories(
        iscecudaextension PUBLIC
        ${ISCE_BUILDINCLUDEDIR}
        ${ISCE_BUILDCYINCLUDEDIR}
        ${PYTHON_NUMPY_INCLUDE_DIR}
        ${GDAL_INCLUDE_DIR}
        ${CMAKE_SOURCE_DIR}/contrib/cereal/include
        ${PYRE_INCLUDE_DIR}
)

# Add necessary library links
target_link_libraries(iscecudaextension ${LISCECUDA} ${LISCE} m)

# Install
install(
    TARGETS iscecudaextension
    DESTINATION ${ISCE_PACKAGESDIR}/isce3/extensions
    COMPONENT extensions
)

# end of file
