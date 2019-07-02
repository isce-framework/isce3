#
# Copyright 2017-2018
#

# Find Numpy
find_package(Numpy REQUIRED)

# Add subdirectories for installing pxd files
add_subdirectory(io)
add_subdirectory(core)
add_subdirectory(product)
add_subdirectory(image)
add_subdirectory(signal)
add_subdirectory(geometry)
add_subdirectory(serialization)

# Set the name for the ISCE C++ library
set(LISCE isce.${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR})

# Set file properties
set_source_files_properties(
    isceextension.pyx
    PROPERTIES CYTHON_IS_CXX TRUE
)

# Put all the pxd files and headers into the build include directory
file(GLOB PXD_LIST ${CMAKE_CURRENT_LIST_DIR}/*/*.pxd)
foreach(HFILE ${PXD_LIST} except/CyError.h)
    get_filename_component(HFILEBASE ${HFILE} NAME)
    configure_file(${HFILE} ${ISCE_BUILDCYINCLUDEDIR}/${HFILEBASE} COPYONLY)
endforeach()

# Add Cython module
cython_add_module(isceextension isceextension.pyx)
SetCoverageOptions(isceextension)

FILE(GLOB PYX_LIST ${CMAKE_CURRENT_LIST_DIR}/*/*.pyx)
target_sources(isceextension PRIVATE ${PYX_LIST})

# Override output shared object name to place in desired location
set_target_properties(
    isceextension
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${ISCE_BUILDPACKAGESDIR}/isce3/extensions
)

# Add necessary include directories
target_include_directories(
        isceextension PUBLIC
        ${ISCE_BUILDINCLUDEDIR}
        ${PYTHON_NUMPY_INCLUDE_DIR}
        ${GDAL_INCLUDE_DIR}
        ${HDF5_INCLUDE_DIR}
        ${CMAKE_SOURCE_DIR}/contrib/cereal/include
        ${PYRE_INCLUDE_DIR}
)

add_subdirectory(except)
target_link_libraries(isceextension ${LISCE} m cyerror)

# Install
install(
    TARGETS isceextension
    DESTINATION ${ISCE_PACKAGESDIR}/isce3/extensions
    COMPONENT extensions
)

# Install cuda cython extensions if cuda available
if (WITH_CUDA)
    add_subdirectory(cuda)
    target_include_directories(iscecudaextension PRIVATE ${CMAKE_CURRENT_LIST_DIR}/except)
endif(WITH_CUDA)
