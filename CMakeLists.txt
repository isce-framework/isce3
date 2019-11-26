cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

if (POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()
if (POLICY CMP0076)
  cmake_policy(SET CMP0076 NEW)
endif()

project(ISCE VERSION 3.0
        LANGUAGES CXX)

option(WITH_CUDA "Set ON|OFF (default=ON) to build CUDA library" OFF)
set(ISCE_CUDA_ARCHS "Auto" CACHE STRING
    "Select target CUDA device architecture, options are:
    - comma-separated compute capabilities (e.g. 3.5,5.0,5.2)
    - \"Auto\" to detect installed CUDA devices and target those architectures
    - \"\" (empty) to use default compilation options")

# add local .cmake directory to CMAKE_MODULE_PATH
list(APPEND CMAKE_MODULE_PATH ${ISCE_SOURCE_DIR}/.cmake)

# import some helper functions
include(IsceCudaHelper)

if (WITH_CUDA)
    enable_language(CUDA)

    # check CUDA version
    set(CUDA_VERSION ${CMAKE_CUDA_COMPILER_VERSION})
    if (CUDA_VERSION VERSION_LESS 9)
        message(FATAL_ERROR "CUDA version must be at least 9. Detected ${CUDA_VERSION}")
    endif()

    # specify target CUDA device architecture(s)
    set_cuda_arch_flags("${ISCE_CUDA_ARCHS}")
endif()

# Set the name for the ISCE C++ library
set(LISCE isce.${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR})

###Options for documentation
option(WITH_DOC
    "Set ON|OFF (default=OFF) to build documentation" OFF)

set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo;Coverage" CACHE STRING "" FORCE)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()
if(NOT CMAKE_BUILD_TYPE IN_LIST CMAKE_CONFIGURATION_TYPES)
    message(FATAL_ERROR "Unsupported build type '${CMAKE_BUILD_TYPE}' "
                        "(must be one of ${CMAKE_CONFIGURATION_TYPES})")
endif()

if(CMAKE_BUILD_TYPE STREQUAL Coverage)
    message("Compiling with code coverage reporting.")
    include(CodeCoverage)
else()
    # Define a dummy function so we don't have to keep checking BUILD_TYPE
    function(SetCoverageOptions target)
    endfunction()
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
option(INSTALL_GTEST "Enable installation of googletest" OFF)

# Setup for Eigen
include_directories(SYSTEM extern/eigen3)

add_subdirectory(extern) # Contributed software (not maintained here)
add_subdirectory(cxx)    # Core C++ library
add_subdirectory(python) # Python bindings
add_subdirectory(tests)  # Unit tests

# Set up documentation
if (WITH_DOC)

    # If Doxygen is available, build C++ documentation
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
