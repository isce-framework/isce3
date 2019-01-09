# Find the Pytest executable.
#
# This code sets the following variables:
#
#  PYTEST_EXECUTABLE
#
# Use the pytest executable that lives next to the Python executable
# if it is a local installation.
cmake_minimum_required(VERSION 2.8)

if(NOT PYTHON_EXECUTABLE)
    if(Pytest_FIND_QUIETLY)
        find_package(PythonInterp QUIET)
    else()
        find_package(PythonInterp)
        set(__pytest_out 1)
    endif()
endif()

if(PYTHON_EXECUTABLE)
    ###Import pytest and get version
    execute_process(COMMAND "${PYTHON_EXECUTABLE}" -c
                            "from __future__ import print_function\ntry: import pytest; print(pytest.__version__, end='')\nexcept:pass\n"
                    OUTPUT_VARIABLE __pytest_version)
elseif(__pytest_out)
    message(STATUS "Python executable not found.")
endif(PYTHON_EXECUTABLE)

##First try to find py.test installed right next to python executable 
get_filename_component( _python_path ${PYTHON_EXECUTABLE} PATH )
find_program(PYTEST_EXECUTABLE
    NAMES "py.test${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}"
        "py.test-${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}"
        "py.test${PYTHON_VERSION_MAJOR}"
        "py.test-${PYTHON_VERSION_MAJOR}"
        "pytest${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}"
        "pytest-${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}"
        "pytest${PYTHON_VERSION_MAJOR}"
        "pytest-${PYTHON_VERSION_MAJOR}"
        "py.test"
        "pytest"
    HINTS ${_python_path})


if(PYTEST_EXECUTABLE)
    set(PYTEST_FOUND 1 CACHE INTERNAL "Python pytest found")
else()
    message(STATUS "pytest not found")
endif(PYTEST_EXECUTABLE)


###Add things like timeouts etc to this function
function(add_pytest path)
    if(NOT PYTEST_EXECUTABLE)
        message(STATUS "skipping pytest(${path}) in project '${PROJECT_NAME}'")
        return()
    endif()

    add_test(${path} ${PYTEST_EXECUTABLE} ${path})
endfunction()

include( FindPackageHandleStandardArgs )
FIND_PACKAGE_HANDLE_STANDARD_ARGS( Pytest REQUIRED_VARS PYTEST_EXECUTABLE
                                             VERSION_VAR __pytest_version)

mark_as_advanced(PYTEST_EXECUTABLE PYTEST_FOUND)
