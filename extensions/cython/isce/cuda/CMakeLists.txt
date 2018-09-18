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

# Add Cython module
cython_add_module(iscecudaextension iscecudaextension.pyx)

# Add necessary include directories
target_include_directories(
        iscecudaextension PUBLIC
        ${ISCE_BUILDINCLUDEDIR}
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
    DESTINATION ${ISCE_PACKAGESDIR}
    COMPONENT extensions
)

# end of file
