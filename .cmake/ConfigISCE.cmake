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


##Check that C++17 is available and CXX 5 or greated is installed
function(CheckCXX)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  add_compile_options(-std=c++17)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.0)
    message(FATAL_ERROR "Insufficient GCC version. Version 5.0 or greater is required.")
  endif()

endfunction()


##Make sure that a reasonable version of Python is installed
function(CheckISCEPython)
    FIND_PACKAGE(PythonInterp 3.6)
    FIND_PACKAGE(PythonLibs 3.6)
endfunction()


##Check for Pyre installation
function(CheckPyre)
    FIND_PACKAGE(Pyre REQUIRED)
endfunction()

function(CheckFFTW3)
    FIND_PACKAGE(FFTW3 REQUIRED)
    message (STATUS "FFTW3 includes: ${FFTW_INCLUDES}")
    message (STATUS "FFTW3 libraries: ${FFTW_LIBRARIES}")
    message (STATUS "FFTW3 double: ${FFTW_LIB} ")
    message (STATUS "FFTW3 single: ${FFTWF_LIB}")
    message (STATUS "FFTW3 quad: ${FFTWL_LIB}")
    message (STATUS "FFTW3 double with threads: ${FFTW_THREADS_LIB}")
    message (STATUS "FFTW3 single with threads: ${FFTWF_THREADS_LIB}")
    message (STATUS "FFTW3 quad with threads: ${FFTWL_THREADS_LIB}")
endfunction()

##Check for GDAL installation
function(CheckGDAL)
    find_package(GDAL 2.3 REQUIRED)
    message(STATUS "Found GDAL: ${GDAL_VERSION}")
endfunction()

##Check for HDF5 installation
function(CheckHDF5)
    FIND_PACKAGE(HDF5 REQUIRED COMPONENTS CXX)
    message(STATUS "Found HDF5: ${HDF5_VERSION} ${HDF5_CXX_LIBRARIES}")
    if (HDF5_VERSION VERSION_LESS "1.10.2")
        message(FATAL_ERROR "Did not find HDF5 version >= 1.10.2")
    endif()

    # Use old glibc++ ABI when necessary for compatibility with HDF5 libs
    if (CMAKE_COMPILER_IS_GNUCXX AND
            CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "5.1.0")
        # get the file path of the hdf5_cpp shared object
        set(hdf5_cpp ${HDF5_CXX_LIBRARIES})
        list(FILTER hdf5_cpp INCLUDE REGEX "hdf5_cpp")

        # look for GCC version comment
        file(STRINGS "${hdf5_cpp}" gcc_comment REGEX "GCC:")
        if (gcc_comment)
            string(REGEX REPLACE "\\(([^\\(]*)\\)" "" gcc_comment ${gcc_comment})
            string(REGEX MATCH "([0-9]|\\.)+" hdf5_gcc_version ${gcc_comment})
            if (hdf5_gcc_version AND hdf5_gcc_version VERSION_LESS "5.1.0")
                message(WARNING "Using old glibc++ ABI "
                    "(found HDF5 compiled with GCC ${hdf5_gcc_version})")
                add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
            endif()
        endif()
    endif()

    # Use more standard names to propagate variables
    set(HDF5_INCLUDE_DIR ${HDF5_INCLUDE_DIRS} CACHE PATH "HDF5 include directory")
    set(HDF5_LIBRARY "${HDF5_CXX_LIBRARIES}" CACHE STRING "HDF5 libraries")
endfunction()

##Check for Armadillo installation
function(CheckArmadillo)
    FIND_PACKAGE(Armadillo REQUIRED)
    message (STATUS "Found Armadillo:  ${ARMADILLO_VERSION_STRING}")
endfunction()

#Check for OpenMP
function(CheckOpenMP)
    FIND_PACKAGE(OpenMP)
endfunction()

#Check for pytest
function(CheckPytest)
    FIND_PACKAGE(Pytest)
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

    ###build/lib

    ###install/include
    if (NOT ISCE_INCLUDEDIR)
        set (ISCE_INCLUDEDIR include/isce-${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR} CACHE STRING "isce/include")
    endif(NOT ISCE_INCLUDEDIR)

    ###build/include
    if (NOT ISCE_BUILDINCLUDEDIR)
        set (ISCE_BUILDINCLUDEDIR ${CMAKE_BINARY_DIR}/include/isce-${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR} CACHE STRING "build/isce/include")
    endif(NOT ISCE_BUILDINCLUDEDIR)

    ###install/cyinclude
    if (NOT ISCE_CYINCLUDEDIR)
        set (ISCE_CYINCLUDEDIR "cyinclude" CACHE STRING "isce/cyinclude")
    endif(NOT ISCE_CYINCLUDEDIR)

    ###build/cyinclude
    if (NOT ISCE_BUILDCYINCLUDEDIR)
        set (ISCE_BUILDCYINCLUDEDIR ${CMAKE_BINARY_DIR}/include/cyinclude CACHE STRING "build/isce/cyinclude")
    endif(NOT ISCE_BUILDCYINCLUDEDIR)

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

    ###install/doc
    if (NOT ISCE_DOCDIR)
        set (ISCE_DOCDIR "doc" CACHE STRING "isce/doc")
    endif(NOT ISCE_DOCDIR)
endfunction()

