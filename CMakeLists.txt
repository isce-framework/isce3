cmake_minimum_required(VERSION 3.18)

include(${CMAKE_CURRENT_LIST_DIR}/.cmake/Isce3Version.cmake)
isce3_get_version(ISCE3_VERSION_COMPONENTS ISCE3_VERSION_FULL)

project(isce3
    VERSION   ${ISCE3_VERSION_COMPONENTS}
    LANGUAGES CXX
    )

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
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/.cmake)

# Set the name for the isce3 C++ library.
set(LISCE ${PROJECT_NAME})

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
    if (CUDA_VERSION VERSION_LESS 11)
        message(FATAL_ERROR "CUDA version must be at least 11. Detected ${CUDA_VERSION}")
    endif()

    # specify target CUDA device architecture(s)
    set_cuda_arch_flags("${ISCE_CUDA_ARCHS}")

    # Set the name for the isce3 CUDA library.
    set(LISCECUDA ${PROJECT_NAME}-cuda)

    # For access from code
    set(ISCE3_WITH_CUDA_BOOL 1)
else()
    set(ISCE3_WITH_CUDA_BOOL 0)
endif()

set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo" CACHE STRING "" FORCE)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()
if(NOT CMAKE_BUILD_TYPE IN_LIST CMAKE_CONFIGURATION_TYPES)
    message(FATAL_ERROR "Unsupported build type '${CMAKE_BUILD_TYPE}' "
                        "(must be one of ${CMAKE_CONFIGURATION_TYPES})")
endif()

###Ensure tracking is own for testing
enable_testing()

###Include custom installation paths and checks for the project
include(ConfigISCE)

###Explicit check to prevent in-source builds
AssureOutOfSourceBuilds()

####Check CXX Version and Standard to C++17
CheckCXX()

option(ISCE3_WITH_CYTHON "Enable isce3 cython extension" OFF)
if(ISCE3_WITH_CYTHON)
    message(ERROR "isce3's cython extension has been removed!")
endif()

###Layout same install directory structure as pyre
include(GNUInstallDirs)
InitInstallDirLayout()

# Dependencies

if(DEFINED SKBUILD)
    option(ISCE3_FETCH_DEPS "Fetch external dependencies at build time" OFF)
else()
    option(ISCE3_FETCH_DEPS "Fetch external dependencies at build time" ON)
endif()
include(CMakeDependentOption)
cmake_dependent_option(ISCE3_FETCH_EIGEN "Fetch Eigen at build time" ON
                       "ISCE3_FETCH_DEPS" OFF)
cmake_dependent_option(ISCE3_FETCH_GTEST "Fetch googletest at build time" ON
                       "ISCE3_FETCH_DEPS" OFF)
cmake_dependent_option(ISCE3_FETCH_PYBIND11 "Fetch pybind11 at build time" ON
                       "ISCE3_FETCH_DEPS" OFF)
cmake_dependent_option(ISCE3_FETCH_PYRE "Fetch pyre at build time" ON
                       "ISCE3_FETCH_DEPS" OFF)

include(.cmake/FetchExternRepo.cmake)

add_subdirectory(extern)
getpackage_eigen()
getpackage_fftw()
getpackage_gdal()
getpackage_googletest()
getpackage_hdf5()
getpackage_openmp_optional()
getpackage_pyre()

# These packages required only for the python API. getpackage_python() should
# be executed first in order to ensure a sufficient version of Python is used.
getpackage_python()
getpackage_pybind11()

add_subdirectory(bin)
add_subdirectory(cxx)    # Core C++ library
add_subdirectory(python) # Python bindings
add_subdirectory(tests)  # Unit tests
add_subdirectory(share)  # Examples
add_subdirectory(doc)    # Documentation

configure_file(
    doc/doxygen/Doxyfile.in
    doc/doxygen/Doxyfile
    )

if(NOT DEFINED SKBUILD)
    # If we're using scikit-build, the exported _IMPORT_PREFIX will be wrong:
    # https://github.com/scikit-build/scikit-build-core/issues/894
    # So we cannot export the target config module when using skbuild.
    set(ISCE3_CMAKE_DIR "${ISCE_SHAREDIR}/cmake/isce3" CACHE STRING
        "Install directory for cmake files, relative to install prefix"
        )
    install(EXPORT isce3-targets
            NAMESPACE ISCE3::
            DESTINATION "${ISCE3_CMAKE_DIR}"
            )
    include(CMakePackageConfigHelpers)
    configure_package_config_file(
        "${PROJECT_SOURCE_DIR}/.cmake/isce3-config.cmake.in"
        "${PROJECT_BINARY_DIR}/isce3-config.cmake"
        INSTALL_DESTINATION "${ISCE3_CMAKE_DIR}")
    install(FILES "${PROJECT_BINARY_DIR}/isce3-config.cmake"
            DESTINATION "${ISCE3_CMAKE_DIR}")
endif()

# Enables native packaging using CMake/CPack
include(CPack)
