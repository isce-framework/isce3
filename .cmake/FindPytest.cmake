# Find the Pytest executable.
#
# This code sets the following variables:
#
#  PYTEST_EXECUTABLE
#
# Use the pytest executable that lives next to the Python executable
# if it is a local installation.
cmake_minimum_required(VERSION 3.12)

if(NOT Python_EXECUTABLE)
    set(args 3.6)
    if(Pytest_FIND_QUIETLY)
        list(PREPEND args QUIET)
    endif()
    find_package(Python ${args})
endif()

if(Python_EXECUTABLE)
    ###Import pytest and get version
    execute_process(COMMAND "${Python_EXECUTABLE}" -c
                            "from __future__ import print_function\ntry: import pytest; print(pytest.__version__, end='')\nexcept:pass\n"
                    OUTPUT_VARIABLE __pytest_version)
elseif(__pytest_out)
    message(STATUS "Python executable not found.")
endif()

##First try to find py.test installed right next to python executable
get_filename_component( _python_path ${Python_EXECUTABLE} PATH )
find_program(PYTEST_EXECUTABLE
    NAMES "py.test${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}"
          "py.test-${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}"
          "py.test${Python_VERSION_MAJOR}"
          "py.test-${Python_VERSION_MAJOR}"
          "pytest${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}"
          "pytest-${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}"
          "pytest${Python_VERSION_MAJOR}"
          "pytest-${Python_VERSION_MAJOR}"
          "py.test"
          "pytest"
    HINTS ${_python_path})


if(PYTEST_EXECUTABLE)
    set(PYTEST_FOUND 1 CACHE INTERNAL "Python pytest found")
else()
    message(STATUS "pytest not found")
endif(PYTEST_EXECUTABLE)


include( FindPackageHandleStandardArgs )
FIND_PACKAGE_HANDLE_STANDARD_ARGS( Pytest REQUIRED_VARS PYTEST_EXECUTABLE
                                             VERSION_VAR __pytest_version)

mark_as_advanced(PYTEST_EXECUTABLE PYTEST_FOUND)
