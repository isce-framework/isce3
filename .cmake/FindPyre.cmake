# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

#.rst:
# FindPyre
# --------
#
#
#
# Locate pyre
#
# This module accepts the following environment variables:
#
# ::
#
#     PYRE_EXPORT_DIR or PYRE_ROOT - Specify the location of Pyre
#
#
#
# This module defines the following CMake variables:
#
# ::
#
#     PYRE_FOUND - True if pyre is found
#     PYRE_LIB_DIR - A variable pointing Pyre libraries
#     PYRE_INCLUDE_DIR - Where to find the headers
#     PYRE_PACKAGE_DIR - Where to find python packages
#     PYRE_VERSION - Version number

#Start with checking if python is installed
if (NOT PYTHON_EXECUTABLE)
    find_package(PythonInterp 3.5)
endif()

if (PYTHON_EXECUTABLE)
    #Find out path
    execute_process(
        COMMAND "${PYTHON_EXECUTABLE}" -c 
                "import pyre, os; print(os.path.dirname(pyre.__file__))"
                OUTPUT_VARIABLE __pyre_path)

    execute_process(
        COMMAND "${PYTHON_EXECUTABLE}" -c
                "import pyre; print('{0}.{1}.{2}'.format(*pyre.version()));"
                OUTPUT_VARIABLE __pyre_ver)
else()
    message(STATUS "Python executable not found.")
endif(PYTHON_EXECUTABLE)

###For now just using journal.h to track pyre include path
find_path(PYRE_JOURNAL_DIR  journal.h
    HINTS 
        "${__pyre_path}/../.."
        ENV PYRE_EXPORT_DIR
        ENV PYRE_ROOT
    PATH_SUFFIXES
        include/pyre
)
   

if (PYRE_JOURNAL_DIR)
    set(PYRE_FOUND 1 CACHE INTERNAL "Pyre found")
    set(PYRE_INCLUDE_DIR "${PYRE_JOURNAL_DIR}/../" CACHE INTERNAL "Pyre inc dir")
    set(PYRE_LIB_DIR "${PYRE_JOURNAL_DIR}/../../lib" CACHE INTERNAL "Pyre lib dir")
    set(PYRE_PACKAGE_DIR "${PYRE_JOURNAL_DIR}/../../packages" CACHE INTERNAL "Pyre packages dir")
endif(PYRE_JOURNAL_DIR)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Pyre REQUIRED_VARS PYRE_INCLUDE_DIR PYRE_LIB_DIR PYRE_PACKAGE_DIR
                                       VERSION_VAR  __pyre_ver)
