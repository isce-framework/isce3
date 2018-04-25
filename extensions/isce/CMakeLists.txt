#
# Copyright 2017-2018
#

# Find Numpy
find_package(Numpy REQUIRED)

# Set the name for the ISCE C++ library
set(LISCE isce.${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR})

# Set file properties
set_source_files_properties(
    isceextension.pyx
    PROPERTIES CYTHON_IS_CXX TRUE
)

# Add Cython module
cython_add_module(isceextension isceextension.pyx)

# Add necessary include directories
target_include_directories(
        isceextension PUBLIC
        ${ISCE_BUILDINCLUDEDIR}
        ${PYTHON_NUMPY_INCLUDE_DIR}
        ${GDAL_INCLUDE_DIR}
)

# Add necessary library links
target_link_libraries(isceextension ${LISCE} m)

# Install
install(
    TARGETS isceextension
    DESTINATION ${ISCE_PACKAGESDIR}
    COMPONENT extensions
)

# end of file
