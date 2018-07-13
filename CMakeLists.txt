###CMake builder for ISCE 3.0

###Minimum version of cmake and boiler plate
cmake_minimum_required(VERSION 3.1)

if (POLICY CMP0048)
    cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)

cmake_policy(SET CMP0060 NEW)

###Project
project(ISCE
        VERSION 3.0)

###Options for documentation
option(WITH_DOC
    "Set ON|OFF (default=OFF) to build documentation" OFF)

###Add .cmake to CMAKE MODULE PATH
list(APPEND CMAKE_MODULE_PATH ${ISCE_SOURCE_DIR}/.cmake)


###Ensure tracking is own for testing
enable_testing()

###Include custom installation paths and checks for the project
include(ConfigISCE)

###Explicit check to prevent in-source builds
AssureOutOfSourceBuilds()

####Check CXX Version and Standard to C++14
CheckCXX()

###Check if an appropriate python is installed
CheckISCEPython()

##Check for Pyre
CheckPyre()

##Check for Cython
include(useCython)

##Check for CUDA
CheckCUDA()

##Check for OpenMP
CheckOpenMP()

##Check for GDAL
CheckGDAL()

##Check for HDF5
CheckHDF5()

##Check for Armadillo
CheckArmadillo()

##Check for Nosetests
CheckNosetests()

###Layout same install directory structure as pyre
InitInstallDirLayout()

###Setup for gtest framework
add_subdirectory(contrib)

###Recurs directories
add_subdirectory(lib)

###Cython lib
add_subdirectory(extensions)

###Setup tests
add_subdirectory(tests)

###Setup doc
if (WITH_DOC)

    ##If Doxygen is available, build C++ documentation
    FIND_PACKAGE(DOXYGEN)
    if (DOXYGEN_FOUND)
        set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/doc/doxygen/Doxyfile.in)
        set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

        CONFIGURE_FILE(${DOXYGEN_IN} ${DOXYGEN_OUT})
        add_custom_target( doc_doxygen ALL
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating documentation using Doxygen"
            VERBATIM )
        install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doc/html 
            DESTINATION ${ISCE_DOCDIR}) 
    else()
        message("Doxygen not found. Skipping C++ doc ....")
    endif(DOXYGEN_FOUND)

    ##If Sphinx is available, build python documentation
    FIND_PACKAGE(SPHINX)
    if (SPHINX_FOUND)
        set(SPHX_HTML_DIR ${CMAKE_CURRENT_BINARY_DIR}/doc/html/sphinx/html)
        set(SPHX_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/doc/html/sphinx/_build)
        set(SPHX_CACHE_DIR ${CMAKE_CURRENT_BINARY_DIR}/doc/html/sphinx/_doctrees)

        set(SPHX_IN ${CMAKE_CURRENT_SOURCE_DIR}/doc/sphinx/conf.py.in)
        set(SPHX_OUT ${SPHX_BUILD_DIR}/conf.py)

        CONFIGURE_FILE(${SPHX_IN} ${SPHX_OUT})

        add_custom_target( doc_sphinx ALL
            COMMAND ${SPHINX_EXECUTABLE} -q -b html -c ${SPHX_BUILD_DIR} -d ${SPHX_CACHE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/doc/sphinx ${SPHX_HTML_DIR}
            DEPENDS isceextension)
        install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doc/html/sphinx
            DESTINATION ${ISCE_DOCDIR}/html)
    else()
        message("Sphinx not found. Skipping python doc ....")
    endif(SPHINX_FOUND)
endif(WITH_DOC)
