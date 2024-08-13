macro(getpackage_eigen)
    if(ISCE3_FETCH_EIGEN)
        find_package(Eigen3 3.3.7 CONFIG)
    else()
        find_package(Eigen3 3.3.7 REQUIRED CONFIG)
    endif()

    if(NOT Eigen3_FOUND)
        include(FetchContent)
        FetchContent_Declare(eigen
            GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
            GIT_TAG 3.3.7
            GIT_SHALLOW TRUE)
        FetchContent_GetProperties(eigen)
        if(NOT eigen_POPULATED)
            FetchContent_Populate(eigen)
        endif()
        add_library(eigen INTERFACE)
        add_library(Eigen3::Eigen ALIAS eigen)
        target_include_directories(eigen SYSTEM INTERFACE
            $<BUILD_INTERFACE:${eigen_SOURCE_DIR}>
            $<INSTALL_INTERFACE:include>)
        install(TARGETS eigen EXPORT isce3-targets)
    endif()
endmacro()

macro(getpackage_fftw)
    find_package(FFTW REQUIRED COMPONENTS
        Float Double FloatThreads DoubleThreads)
endmacro()

macro(getpackage_gdal)
    find_package(GDAL 3.0 REQUIRED)
endmacro()

macro(getpackage_googletest)
    # CONFIG mode only - FindGTest.cmake module doesn't provide version checking
    if(ISCE3_FETCH_GTEST)
        find_package(GTest 1.10.0 CONFIG)
    else()
        find_package(GTest 1.10.0 REQUIRED CONFIG)
    endif()

    if(NOT GTest_FOUND)
        set(BUILD_GMOCK   OFF CACHE INTERNAL "")
        set(INSTALL_GTEST OFF CACHE INTERNAL "")
        fetch_extern_repo(gtest
            GIT_REPOSITORY  https://github.com/google/googletest
            GIT_TAG         release-1.10.0
            GIT_SHALLOW     TRUE
            )
        set_target_properties(gtest gtest_main PROPERTIES EXCLUDE_FROM_ALL TRUE)

        add_library(GTest::gtest      ALIAS gtest)
        add_library(GTest::gtest_main ALIAS gtest_main)
    endif()
endmacro()

macro(getpackage_hdf5)
    find_package(HDF5 1.10.2 REQUIRED COMPONENTS CXX)

    # check whether the hdf5 library includes parallel support
    if(HDF5_IS_PARALLEL)
        # look for MPI
        find_package(MPI REQUIRED COMPONENTS CXX)
        list(APPEND HDF5_INCLUDE_DIRS  ${MPI_CXX_INCLUDE_DIRS})
        list(APPEND HDF5_CXX_LIBRARIES ${MPI_CXX_LIBRARIES})
    endif()

    # XXX This target is defined as INTERFACE rather than IMPORTED so it can be
    # exported in order to save downstream projects from having to redefine it.
    if(NOT TARGET HDF5::HDF5)
        add_library(hdf5 INTERFACE)
        add_library(HDF5::HDF5 ALIAS hdf5)
        set_target_properties(hdf5 PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${HDF5_INCLUDE_DIRS}"
            INTERFACE_LINK_LIBRARIES      "${HDF5_CXX_LIBRARIES}"
            )
        install(TARGETS hdf5 EXPORT isce3-targets)
    else()
        # FindHDF5's HDF5::HDF5 does not reflect the transitive MPI dependency,
        # so we need to explicitly add it to the HDF5 interface here.
        # Otherwise including hdf5.h may fail with a missing mpi.h error.
        if(HDF5_IS_PARALLEL)
            target_link_libraries(HDF5::HDF5 INTERFACE MPI::MPI_CXX)
        endif()
    endif()
endmacro()

macro(getpackage_openmp_optional)
    # Check for OpenMP (optional dependency).
    # If not found, default to an empty placeholder target.
    find_package(OpenMP OPTIONAL_COMPONENTS CXX)
    add_library(OpenMP::OpenMP_CXX_Optional INTERFACE IMPORTED)
    if(TARGET OpenMP::OpenMP_CXX)
        target_link_libraries(OpenMP::OpenMP_CXX_Optional
            INTERFACE OpenMP::OpenMP_CXX)
    endif()
endmacro()

macro(getpackage_pybind11)
    # Force legacy FindPythonInterp module used by pybind11 < 2.6 to 
    # find same installation as modern FindPython
    set(PYTHON_EXECUTABLE "${Python_EXECUTABLE}")
    set(PYBIND11_PYTHON_VERSION "${Python_VERSION}")
    if(ISCE3_FETCH_PYBIND11)
        find_package(pybind11 2.5 CONFIG)
        if(NOT pybind11_FOUND)
            fetch_extern_repo(pybind11
                GIT_REPOSITORY  https://github.com/pybind/pybind11
                GIT_TAG         v2.5.0
                GIT_SHALLOW     TRUE
                )
        endif()
    else()
        find_package(pybind11 2.5 CONFIG REQUIRED)
    endif()
endmacro()

macro(getpackage_pyre)
    if(ISCE3_FETCH_PYRE)
        find_package(pyre 1.12.5 CONFIG)
        if(NOT pyre_FOUND)
            # disable pyre's test suite
            set(BUILD_TESTING OFF)
            fetch_extern_repo(pyre
                GIT_REPOSITORY  https://github.com/pyre/pyre
                GIT_TAG         v1.12.5
                GIT_SHALLOW     TRUE
                )
        endif()
    else()
        find_package(pyre 1.12.5 REQUIRED CONFIG)
    endif()
endmacro()

macro(getpackage_python)
    find_package(Python 3.7 REQUIRED COMPONENTS Interpreter Development)
endmacro()
