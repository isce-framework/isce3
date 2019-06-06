###CMake builder for ISCE 3.0

###Minimum version of cmake and boiler plate
cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

if (POLICY CMP0048)
    cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)

cmake_policy(SET CMP0060 NEW)

if (POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

project(ISCE VERSION 3.0
        LANGUAGES CXX)

option(WITH_CUDA "Set ON|OFF (default=ON) to build CUDA library" OFF)
if (WITH_CUDA)
    enable_language(CUDA)
    if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 9)
        message(FATAL_ERROR "CUDA version must be atleast 9. Detected ${CMAKE_CUDA_COMPILER_VERSION}")
    endif()
endif()

# Set the name for the ISCE C++ library
set(LISCE isce.${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR})

###Add .cmake to CMAKE MODULE PATH
list(APPEND CMAKE_MODULE_PATH ${ISCE_SOURCE_DIR}/.cmake)

###Options for documentation
option(WITH_DOC
    "Set ON|OFF (default=OFF) to build documentation" OFF)

option(WITH_CODE_COVERAGE "Enable code coverage reporting" OFF)
if(WITH_CODE_COVERAGE)
    message("Compiling with code coverage reporting.")
    include(CodeCoverage)
endif()

###Ensure tracking is own for testing
enable_testing()

###Include custom installation paths and checks for the project
include(ConfigISCE)

###Explicit check to prevent in-source builds
AssureOutOfSourceBuilds()

####Check CXX Version and Standard to C++17
CheckCXX()

###Check if an appropriate python is installed
CheckISCEPython()

##Check for Pyre
CheckPyre()

##Check for Cython
include(useCython)

##Check for OpenMP
CheckOpenMP()

##Check for GDAL
CheckGDAL()

##Check for HDF5
CheckHDF5()

#Check FFTW3
CheckFFTW3()
##Check for Armadillo
##CheckArmadillo()

##Check for pytest
CheckPytest()

###Layout same install directory structure as pyre
InitInstallDirLayout()

###Setup for gtest framework
add_subdirectory(contrib)

###Recurs directories
add_subdirectory(lib)

###Package directories
add_subdirectory(packages)

###Cython lib
add_subdirectory(extensions)

###Setup tests
add_subdirectory(tests)

###Setup doc
if (WITH_DOC)

    ##If Doxygen is available, build C++ documentation
    FIND_PACKAGE(Doxygen)
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
        message("Doxygen found. Generating C++ doc ....")
    else()
        message("Doxygen not found. Skipping C++ doc ....")
    endif(DOXYGEN_FOUND)

    ##If Sphinx is available, build python documentation
    FIND_PACKAGE(Sphinx)
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
        message("Sphinx found. Generating python doc ....")
    else()
        message("Sphinx not found. Skipping python doc ....")
    endif(SPHINX_FOUND)
endif(WITH_DOC)
