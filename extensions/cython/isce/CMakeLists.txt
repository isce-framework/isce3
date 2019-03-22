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

###Create a function for concatenating pxd files
function(catpxd IN_FILE OUT_FILE)
    FILE(READ ${IN_FILE} CONTENTS)
    FILE(APPEND ${OUT_FILE} "${CONTENTS}")
endfunction()

# Set the name for the ISCE C++ library
set(LISCE isce.${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR})

# Set file properties
set_source_files_properties(
    isceextension.pyx
    PROPERTIES CYTHON_IS_CXX TRUE
)

# Initialize build directory for extension module to mimic install directory
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/isce3/extensions)

###Put together all the pxd files into one giant pxd
FILE(GLOB PXD_LIST ${CMAKE_CURRENT_LIST_DIR}/*/*.pxd) 
foreach(HFILE ${PXD_LIST})
    GET_FILENAME_COMPONENT(HFILEBASE  ${HFILE} NAME)
    configure_file(${HFILE} ${ISCE_BUILDCYINCLUDEDIR}/${HFILEBASE} COPYONLY)
endforeach()

# Add Cython module
cython_add_module(isceextension isceextension.pyx)

# Override output shared object name to place in desired location
set_target_properties(
    isceextension
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/isce3/extensions
)

# Add necessary include directories
target_include_directories(
        isceextension PUBLIC
        ${ISCE_BUILDINCLUDEDIR}
        ${PYTHON_NUMPY_INCLUDE_DIR}
        ${GDAL_INCLUDE_DIR}
        ${CMAKE_SOURCE_DIR}/contrib/cereal/include
        ${PYRE_INCLUDE_DIR}
)

# Add necessary library links
target_link_libraries(isceextension ${LISCE} m)

# Install
install(
    TARGETS isceextension
    DESTINATION ${ISCE_PACKAGESDIR}/isce3/extensions
    COMPONENT extensions
)

# Install cuda cython extensions if cuda available
if (WITH_CUDA)
    add_subdirectory(cuda)
endif(WITH_CUDA)

# end of file
