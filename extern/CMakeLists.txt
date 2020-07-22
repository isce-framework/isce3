function(getpackage_cereal)
    find_package(Cereal CONFIG)
    if(Cereal_FOUND)
        add_library(Cereal::Cereal INTERFACE IMPORTED)
        set_target_properties(Cereal::Cereal PROPERTIES
            INTERFACE_LINK_LIBRARIES cereal)
    else()
        set(CEREAL_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/extern/cereal/include)
        install(
            DIRECTORY       ${CEREAL_INCLUDE_DIR}/cereal
            DESTINATION     include
            FILES_MATCHING  PATTERN *.hpp PATTERN *.h)

        add_library(cereal INTERFACE)
        add_library(Cereal::Cereal ALIAS cereal)
        target_include_directories(cereal SYSTEM INTERFACE
            $<BUILD_INTERFACE:${CEREAL_INCLUDE_DIR}>
            $<INSTALL_INTERFACE:include>)

        install(TARGETS cereal EXPORT isce3-targets)
    endif()
endfunction()

function(getpackage_cython)
    find_package(Cython 0.28.1 REQUIRED)
endfunction()

function(getpackage_eigen)
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
endfunction()

function(getpackage_fftw)
    find_package(FFTW REQUIRED COMPONENTS
        Float Double FloatThreads DoubleThreads)
endfunction()

function(getpackage_gdal)
    find_package(GDAL 2.3 REQUIRED)
endfunction()

function(getpackage_googletest)
    # CONFIG mode only - FindGTest.cmake module doesn't provide version checking
    if(ISCE3_FETCH_GTEST)
        find_package(GTest 1.10.0 CONFIG)
    else()
        find_package(GTest 1.10.0 REQUIRED CONFIG)
    endif()

    if(NOT GTEST_FOUND)
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
endfunction()

function(getpackage_hdf5)
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
    add_library(hdf5 INTERFACE)
    add_library(HDF5::HDF5 ALIAS hdf5)
    set_target_properties(hdf5 PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${HDF5_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES      "${HDF5_CXX_LIBRARIES}"
        )

    install(TARGETS hdf5 EXPORT isce3-targets)
endfunction()

# TODO This can be removed in favor of adding "NumPy" to the list of required
# components in find_package(Python) if/when we upgrade to CMake >= 3.14
function(getpackage_numpy)
    find_package(Numpy REQUIRED)
endfunction()

function(getpackage_openmp_optional)
    # Check for OpenMP (optional dependency).
    # If not found, default to an empty placeholder target.
    find_package(OpenMP OPTIONAL_COMPONENTS CXX)
    add_library(OpenMP::OpenMP_CXX_Optional INTERFACE IMPORTED)
    if(TARGET OpenMP::OpenMP_CXX)
        target_link_libraries(OpenMP::OpenMP_CXX_Optional
            INTERFACE OpenMP::OpenMP_CXX)
    endif()
endfunction()

function(getpackage_pybind11)
    add_subdirectory(${PROJECT_SOURCE_DIR}/extern/pybind11)
endfunction()

function(getpackage_pyre)
    find_package(pyre 1.9.8 CONFIG)
    if(NOT HAVE_PYRE)
        # disable pyre's test suite
        set(BUILD_TESTING OFF)
        fetch_extern_repo(pyre
            GIT_REPOSITORY  https://github.com/pyre/pyre
            GIT_TAG         v1.9.8
            GIT_SHALLOW     TRUE
            )
    endif()
endfunction()

macro(getpackage_python)
    find_package(Python 3.6 COMPONENTS Interpreter Development)
endmacro()
