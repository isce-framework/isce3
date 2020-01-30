cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

# CMP0074 (CMake 3.12+)
# find_package() uses <PackageName>_ROOT variables.
# This is helpful for us when using packages non-standard install locations.
if (POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()
# CMP0076 (CMake 3.13+)
# The target_sources() command converts relative paths to absolute.
# We're indifferent, so this quiets the warnings.
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

# Check for OpenMP (optional dependency).
# If not found, default to an empty placeholder target.
find_package(OpenMP OPTIONAL_COMPONENTS CXX)
add_library(OpenMP::OpenMP_CXX_Optional INTERFACE IMPORTED)
if(TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(OpenMP::OpenMP_CXX_Optional INTERFACE OpenMP::OpenMP_CXX)
endif()

##Check for GDAL
CheckGDAL()

##Check for HDF5
CheckHDF5()

#Check FFTW3
find_package(FFTW REQUIRED COMPONENTS
    Float Double FloatThreads DoubleThreads)
##Check for Armadillo
##CheckArmadillo()

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
add_subdirectory(share)  # Examples
add_subdirectory(doc)    # Documentation

configure_file(
    doc/doxygen/Doxyfile.in
    doc/doxygen/Doxyfile
    )
configure_file(
    doc/sphinx/conf.py.in
    doc/sphinx/conf.py
    )
