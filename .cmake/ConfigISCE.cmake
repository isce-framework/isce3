###Custom function to prevent in-source builds
function(AssureOutOfSourceBuilds)
    get_filename_component(srcdir "${CMAKE_SOURCE_DIR}" REALPATH)
    get_filename_component(bindir "${CMAKE_BINARY_DIR}" REALPATH)

    if("${srcdir}" STREQUAL "${bindir}")
        message("################################")
        message(" ISCE should not be configured and built in the soutce directory")
        message(" You must run cmake in a build directory. ")
        message(" When directory structure is finalized .. can add full example here")
        message(FATAL_ERROR "Quitting. In-source builds not allowed....")
    endif()
endfunction()

# Backport modules from CMake 3.19
if(CMAKE_VERSION VERSION_LESS 3.19)
    list(APPEND CMAKE_MODULE_PATH
        "${PROJECT_SOURCE_DIR}/.cmake/kitware-cmake/3.19")
endif()

# Check that compiler supports C++17
# (Only checks GCC and AppleClang currently)
function(CheckCXX)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0)
            message(FATAL_ERROR
                "Insufficient GCC version - requires 7.0 or greater")
        endif()
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5)
            message(FATAL_ERROR
                "Insufficient AppleClang version - requires 5.0 or greater")
        endif()
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    else()
        message(WARNING
            "Unsupported compiler detected - courageously continuing")
    endif()

    if(ISCE3_WITH_CUDA AND CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
        if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11)
            message(FATAL_ERROR
                "NVCC 11+ required - detected ${CMAKE_CUDA_COMPILER_VERSION}")
        endif()
        # TODO: Can we check the CUDA host compiler here? We require GCC 7+
    endif()

    # Require C++17 (no extensions) for all C++ and CUDA code
    set(CMAKE_CXX_STANDARD            17 PARENT_SCOPE)
    set(CMAKE_CXX_STANDARD_REQUIRED   ON PARENT_SCOPE)
    set(CMAKE_CXX_EXTENSIONS         OFF PARENT_SCOPE)
    set(CMAKE_CUDA_STANDARD           17 PARENT_SCOPE)
    set(CMAKE_CUDA_STANDARD_REQUIRED  ON PARENT_SCOPE)
    set(CMAKE_CUDA_EXTENSIONS        OFF PARENT_SCOPE)

    add_library(project_warnings INTERFACE)
    include(Warnings)
    set_warnings(project_warnings)
endfunction()


function(InitInstallDirLayout)
    ###install/packages
    if (NOT ISCE_PACKAGESDIR)
        if(DEFINED SKBUILD)
            set(ISCE_PACKAGESDIR "${SKBUILD_PLATLIB_DIR}" CACHE STRING "isce/packages")
        else()
            set(ISCE_PACKAGESDIR packages CACHE STRING "isce/packages")
        endif()
    endif()
    # Convert to unconditional absolute path
    if(IS_ABSOLUTE ${ISCE_PACKAGESDIR})
        set(ISCE_PACKAGESDIR_FULL ${ISCE_PACKAGESDIR} PARENT_SCOPE)
    else()
        set(ISCE_PACKAGESDIR_FULL ${CMAKE_INSTALL_PREFIX}/${ISCE_PACKAGESDIR} PARENT_SCOPE)
    endif()

    ###build/packages
    if (NOT ISCE_BUILDPACKAGESDIR)
        set (ISCE_BUILDPACKAGESDIR ${CMAKE_BINARY_DIR}/packages CACHE STRING "build/isce/packages")
    endif(NOT ISCE_BUILDPACKAGESDIR)

    if(DEFINED SKBUILD)
        set(ISCE_BINDIR "${SKBUILD_SCRIPTS_DIR}" CACHE STRING "isce/bin")
    else()
        set(ISCE_BINDIR "${CMAKE_INSTALL_BINDIR}" CACHE STRING "isce/bin")
    endif()

    ###install/lib
    if (NOT ISCE_LIBDIR)
        if(DEFINED SKBUILD)
            set(ISCE_LIBDIR "${SKBUILD_DATA_DIR}/lib" CACHE STRING "isce/lib")
        else()
            set(ISCE_LIBDIR "${CMAKE_INSTALL_LIBDIR}" CACHE STRING "isce/lib")
        endif()
    endif(NOT ISCE_LIBDIR)

    ###install/share
    if (NOT ISCE_SHAREDIR)
        if(DEFINED SKBUILD)
            set(ISCE_SHAREDIR "${SKBUILD_DATA_DIR}/share" CACHE STRING "isce/lib")
        else()
            set(ISCE_SHAREDIR "${CMAKE_INSTALL_DATADIR}" CACHE STRING "isce/lib")
        endif()
    endif(NOT ISCE_SHAREDIR)

    if(NOT DEFINED ISCE_INCLUDEDIR)
        if(DEFINED SKBUILD)
            set(ISCE_INCLUDEDIR "${SKBUILD_DATA_DIR}/include" CACHE STRING "isce/include")
        else()
            set(ISCE_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}" CACHE STRING "isce/include")
        endif()
    endif()
endfunction()
