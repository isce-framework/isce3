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


##Check that C++14 is available and CXX 5 or greated is installed
function(CheckCXX)
  set(CMAKE_CXX_STANDARD 14)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  add_compile_options(-std=c++14)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.0)
    message(FATAL_ERROR "Insufficient GCC version. Version 5.0 or greater is required.")
  endif()
endfunction()


##Make sure that a reasonable version of Python is installed
function(CheckISCEPython)
    FIND_PACKAGE(PythonInterp 3.5)
    FIND_PACKAGE(PythonInterp 3.5)
endfunction()


##Check for Pyre installation
function(CheckPyre)
    FIND_PACKAGE(Pyre REQUIRED)
endfunction()


##Check for GDAL installation
function(CheckGDAL)
    FIND_PACKAGE(GDAL REQUIRED)
    execute_process( COMMAND gdal-config --version
                    OUTPUT_VARIABLE GDAL_VERSION)
    string(REGEX MATCHALL "[0-9]+" GDAL_VERSION_PARTS ${GDAL_VERSION})
    list(GET GDAL_VERSION_PARTS 0 GDAL_MAJOR)
    list(GET GDAL_VERSION_PARTS 1 GDAL_MINOR)

    if ((GDAL_VERSION VERSION_GREATER 2.0.0) OR (GDAL_VERSION VERSION_EQUAL 2.0.0))
        message (STATUS "Found gdal:  ${GDAL_VERSION}")
    else()
        message (FATAL_ERROR "Did not find GDAL version >= 2.1")
    endif()
endfunction()

##Check for Armadillo installation
function(CheckArmadillo)
    FIND_PACKAGE(Armadillo REQUIRED)
    message (STATUS "Found Armadillo:  ${ARMADILLO_VERSION_STRING}")
endfunction()


##Check for CUDA installation
set(USE_CUDA TRUE CACHE BOOL "Build CUDA")
function(CheckCUDA)
    if (USE_CUDA)
        FIND_PACKAGE(CUDA)
        if (CUDA_FOUND)
            if ((CUDA_VERSION VERSION_GREATER 8.0) OR (CUDA_VERSION VERSION_EQUAL 8.0))
                message (STATUS "Found CUDA: ${CUDA_VERSION}")
                #set (CUDA_PROPAGATE_HOST_FLAGS ON)
            else()
                message (STATUS "Did not find a suitable CUDA version >= 8.0")
                set(CUDA_FOUND FALSE)
            endif()
        else()
            message (STATUS "CUDA not found. Continuing ... ")
        endif()
    endif()
endfunction()



function(InitInstallDirLayout)
    ###install/bin
    if (NOT ISCE_BINDIR)
        set (ISCE_BINDIR bin CACHE STRING "isce/bin")
    endif(NOT ISCE_BINDIR)

    ###install/packages
    if (NOT ISCE_PACKAGESDIR)
        set (ISCE_PACKAGESDIR packages CACHE STRING "isce/packages")
    endif(NOT ISCE_PACKAGESDIR)

    ###install/lib
    if (NOT ISCE_LIBDIR)
        set (ISCE_LIBDIR lib CACHE STRING "isce/lib") 
    endif(NOT ISCE_LIBDIR)

    ###install/include
    if (NOT ISCE_INCLUDEDIR)
        set (ISCE_INCLUDEDIR include/isce-${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR} CACHE STRING "isce/include")
    endif(NOT ISCE_INCLUDEDIR)

    ###build/include
    if (NOT ISCE_BUILDINCLUDEDIR)
        set (ISCE_BUILDINCLUDEDIR ${CMAKE_BINARY_DIR}/include/isce-${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR} CACHE STRING "build/isce/include")
    endif(NOT ISCE_BUILDINCLUDEDIR)

    ###install/defaults
    if (NOT ISCE_DEFAULTSDIR)
        set (ISCE_DEFAULTSDIR defaults CACHE STRING "isce/defaults")
    endif(NOT ISCE_DEFAULTSDIR)

    ###install/var
    if (NOT ISCE_VARDIR)
        set (ISCE_VARDIR var CACHE STRING "isce/var")
    endif(NOT ISCE_VARDIR)

    ###install/etc
    if (NOT ISCE_ETCDIR)
        set (ISCE_ETCDIR etc CACHE STRING "isce/etc")
    endif(NOT ISCE_ETCDIR)

    ###install/templates
    if (NOT ISCE_TEMPLATESDIR)
        set (ISCE_TEMPLATESDIR templates CACHE STRING "isce/templates")
    endif(NOT ISCE_TEMPLATESDIR)

endfunction()

