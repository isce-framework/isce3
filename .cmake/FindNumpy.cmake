# Find the Python NumPy package
# PYTHON_NUMPY_INCLUDE_DIR
# PYTHON_NUMPY_FOUND
# will be set by this script

cmake_minimum_required(VERSION 3.12)

if(NOT Python_EXECUTABLE)
    set(args 3.6)
  if(NumPy_FIND_QUIETLY)
    list(PREPEND args QUIET)
  endif()
  find_package(Python ${args})
endif()

if(Python_EXECUTABLE)
  # Find out the include path
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
            "from __future__ import print_function\ntry: import numpy; print(numpy.get_include(), end='')\nexcept:pass\n"
            OUTPUT_VARIABLE __numpy_path)
  # And the version
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
            "from __future__ import print_function\ntry: import numpy; print(numpy.__version__, end='')\nexcept:pass\n"
    OUTPUT_VARIABLE __numpy_version)
elseif(__numpy_out)
  message(STATUS "Python executable not found.")
endif()

find_path(PYTHON_NUMPY_INCLUDE_DIR numpy/arrayobject.h
  HINTS "${__numpy_path}" "${Python_INCLUDE_DIRS}" NO_DEFAULT_PATH)

if(PYTHON_NUMPY_INCLUDE_DIR)
  set(PYTHON_NUMPY_FOUND 1 CACHE INTERNAL "Python numpy found")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NumPy REQUIRED_VARS PYTHON_NUMPY_INCLUDE_DIR
                                        VERSION_VAR __numpy_version)

mark_as_advanced(PYTHON_NUMPY_INCLUDE_DIR PYTHON_NUMPY_FOUND)
