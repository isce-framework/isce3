# Find the Nosetests executable.
#
# This code sets the following variables:
#
#  NOSETESTS_EXECUTABLE
#
# Use the nosetests executable that lives next to the Python executable
# if it is a local installation.
cmake_minimum_required(VERSION 2.8)

if(NOT PYTHON_EXECUTABLE)
    if(Nosetests_FIND_QUIETLY)
        find_package(PythonInterp QUIET)
    else()
        find_package(PythonInterp)
        set(__nosetests_out 1)
    endif()
endif()

if(PYTHON_EXECUTABLE)
    ###Import nose and get version
    execute_process(COMMAND "${PYTHON_EXECUTABLE}" -c
                            "from __future__ import print_function\ntry: import nose; print(nose.__version__, end='')\nexcept:pass\n"
                    OUTPUT_VARIABLE __nosetests_version)
elseif(__nosetests_out)
    message(STATUS "Python executable not found.")
endif(PYTHON_EXECUTABLE)

##First try to find nosetests installed right next to python executable 
get_filename_component( _python_path ${PYTHON_EXECUTABLE} PATH )
find_program(NOSETESTS_EXECUTABLE
    NAMES "nosetests${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}"
        "nosetests-${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}"
        "nosetests${PYTHON_VERSION_MAJOR}"
        "nosetests-${PYTHON_VERSION_MAJOR}"
        "nosetests"
    HINTS ${_python_path})


if(NOSETESTS_EXECUTABLE)
    set(NOSETESTS_FOUND 1 CACHE INTERNAL "Python nosetests found")
else()
    message(STATUS "Nosetests not found")
endif(NOSETESTS_EXECUTABLE)


###Add things like timeouts etc to this function
function(add_nosetests path)
    if(NOT NOSETESTS_EXECUTABLE)
        message(STATUS "skipping nosetests(${path}) in project '${PROJECT_NAME}'")
        return()
    endif()

    add_test(${path} ${NOSETESTS_EXECUTABLE} ${path})
endfunction()

include( FindPackageHandleStandardArgs )
FIND_PACKAGE_HANDLE_STANDARD_ARGS( Nosetests REQUIRED_VARS NOSETESTS_EXECUTABLE
                                             VERSION_VAR __nosetests_version)

mark_as_advanced(NOSETESTS_EXECUTABLE NOSETESTS_FOUND)
