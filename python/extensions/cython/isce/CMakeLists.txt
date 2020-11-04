set_source_files_properties(
    isceextension.pyx
    PROPERTIES CYTHON_IS_CXX TRUE
)

cython_add_module(isceextension isceextension.pyx)
SetCoverageOptions(isceextension)

# These need to be exposed to the derived cuda extension
include_directories(
    ${ISCE_BUILDCYINCLUDEDIR}
    ${CMAKE_CURRENT_LIST_DIR}
    )

# Put all the pxd files and headers into the build include directory
file(GLOB PXD_LIST ${CMAKE_CURRENT_LIST_DIR}/*/*.pxd)
foreach(HFILE ${PXD_LIST} except/CyError.h)
    get_filename_component(HFILEBASE ${HFILE} NAME)
    configure_file(${HFILE} ${ISCE_BUILDCYINCLUDEDIR}/${HFILEBASE} COPYONLY)
endforeach()

FILE(GLOB PYX_LIST ${CMAKE_CURRENT_LIST_DIR}/*/*.pyx)
target_sources(isceextension PRIVATE ${PYX_LIST}
    except/CyError.cpp
    io/H5Error.cpp
    )

# Override output shared object name to place in desired location
set_target_properties(
    isceextension
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${ISCE_BUILDPACKAGESDIR}/isce3/extensions
)

target_include_directories(
        isceextension PUBLIC
        ${PYTHON_NUMPY_INCLUDE_DIR}
)

target_link_libraries(isceextension PRIVATE ${LISCE})

install(TARGETS isceextension
        DESTINATION ${ISCE_PACKAGESDIR}/isce3/extensions
        COMPONENT extensions
        )

# Install cuda cython extensions if cuda available
if (WITH_CUDA)
    add_subdirectory(cuda)
endif(WITH_CUDA)
