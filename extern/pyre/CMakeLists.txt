# -*- cmake -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# cmake setup
cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
# policies
if (POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif(POLICY CMP0048)
cmake_policy(SET CMP0060 NEW)
if (POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif(POLICY CMP0074)
if (POLICY CMP0076)
  cmake_policy(SET CMP0076 NEW)
endif(POLICY CMP0076)

# options
option(WITH_CUDA "enable support for CUDA" OFF)

# adjust the include path
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/.cmake)
# get support
include(pyre_init)
include(pyre_journal)
include(pyre_pyre)
include(pyre_merlin)
include(pyre_cuda)
include(pyre_gsl)
include(pyre_mpi)

# ask git for the pyre version
pyre_getVersion()

# set up the project
project(PYRE VERSION ${REPO_MAJOR}.${REPO_MINOR}.${REPO_MICRO} LANGUAGES CXX)

# packages
# gsl
find_package(GSL)
# mpi
find_package(MPI)
# postgres
find_package(PostgreSQL)
# python
set(PYTHON_COMPONENTS Interpreter Development)
if(GSL_FOUND)
    list(APPEND PYTHON_COMPONENTS NumPy)
endif()
find_package(Python3 REQUIRED COMPONENTS ${PYTHON_COMPONENTS})

# set up cmake
pyre_cmakeInit()
# set up c++
pyre_cxxInit()
# set up python
pyre_pythonInit()

# initialize the variables that describe the staging directory layout
pyre_stagingInit()
# initialize the variables that describe the install directory layout
pyre_destinationInit()

# visit subdirectories
add_subdirectory(packages)
add_subdirectory(lib)
add_subdirectory(extensions)
add_subdirectory(defaults)
add_subdirectory(bin)

if(BUILD_TESTING)
    include(CTest)
    add_subdirectory(tests)
endif()

# end of file
