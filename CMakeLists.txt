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

find_program(CCACHE ccache)
if(CCACHE)
    message(STATUS "Using ccache: ${CCACHE}")
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE})
    set(CMAKE_CUDA_COMPILER_LAUNCHER ${CCACHE})
endif()

set(ISCE_CUDA_ARCHS "Auto" CACHE STRING
    "Select target CUDA device architecture, options are:
    - comma-separated compute capabilities (e.g. 3.5,5.0,5.2)
    - \"Auto\" to detect installed CUDA devices and target those architectures
    - \"\" (empty) to use default compilation options")

# add local .cmake directory to CMAKE_MODULE_PATH
list(APPEND CMAKE_MODULE_PATH ${ISCE_SOURCE_DIR}/.cmake)

# import some helper functions
include(IsceCudaHelper)

# Auto-detect cuda support by default
if(NOT DEFINED WITH_CUDA)
    set(WITH_CUDA Auto CACHE STRING "")
endif()
if(WITH_CUDA STREQUAL Auto)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        set(WITH_CUDA ON CACHE STRING "" FORCE)
    else()
        set(WITH_CUDA OFF CACHE STRING "" FORCE)
    endif()
endif()
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
include(GNUInstallDirs)
InitInstallDirLayout()

# Dependencies

include(.cmake/FetchExternRepo.cmake)

find_package(Cereal QUIET)
if(Cereal_FOUND)
    # Needed to create target alias
    set_target_properties(cereal PROPERTIES IMPORTED_GLOBAL TRUE)
endif()
find_package(Eigen3 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
    find_package(Git REQUIRED)
    set(BUILD_TESTING OFF)
    fetch_extern_repo(eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.3.7
        GIT_SHALLOW TRUE
        PATCH_COMMAND ${GIT_EXECUTABLE} apply
            ${CMAKE_CURRENT_LIST_DIR}/extern/eigen-disable-testing.patch)
    add_library(Eigen3::Eigen ALIAS eigen)
endif()
find_package(GTest QUIET CONFIG)
if(NOT GTEST_FOUND)
    set(BUILD_GMOCK OFF CACHE INTERNAL "")
    set(INSTALL_GTEST OFF CACHE INTERNAL "")
    fetch_extern_repo(gtest
        GIT_REPOSITORY https://github.com/google/googletest
        GIT_TAG release-1.10.0
        GIT_SHALLOW TRUE)
    set_target_properties(gtest gtest_main PROPERTIES EXCLUDE_FROM_ALL TRUE)
    add_library(GTest::gtest ALIAS gtest)
    add_library(GTest::gtest_main ALIAS gtest_main)
endif()
find_package(pyre QUIET)
if(NOT HAVE_PYRE)
    # disable pyre's test suite
    set(BUILD_TESTING OFF)
    fetch_extern_repo(pyre
        GIT_REPOSITORY https://github.com/rtburns-jpl/pyre
        GIT_TAG isce3-stable
        GIT_SHALLOW TRUE)
endif()

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

set(ISCE3_CMAKE_DIR "share/cmake/isce3" CACHE STRING
    "Install directory for cmake files, relative to install prefix"
    )
install(EXPORT isce3-targets
        NAMESPACE ISCE3::
        DESTINATION ${ISCE3_CMAKE_DIR}
        )
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/.cmake/isce3-config.cmake.in
    ${PROJECT_BINARY_DIR}/isce3-config.cmake
    INSTALL_DESTINATION ${ISCE3_CMAKE_DIR})
install(FILES ${PROJECT_BINARY_DIR}/isce3-config.cmake
        DESTINATION ${ISCE3_CMAKE_DIR})
