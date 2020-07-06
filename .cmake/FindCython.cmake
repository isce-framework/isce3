# Find the Cython compiler.
#
# This code sets the following variables:
#
#  CYTHON_EXECUTABLE
#  CYTHON_FOUND
#  CYTHON_VERSION
#
# See also UseCython.cmake

#=============================================================================
# Copyright 2011 Kitware, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

# Require that the project first defines the python interpreter path in order to
# find a compatible version of cython. Previous versions of this searched for a
# cython executable in the filesystem but this may be unreliable.
if(NOT DEFINED Python_EXECUTABLE)
    message(FATAL_ERROR "find_package(Cython) requires that the "
                        "Python_EXECUTABLE path must first be defined, e.g. "
                        "by using find_package(Python)")
endif()

set(CYTHON_EXECUTABLE ${Python_EXECUTABLE} -m cython CACHE INTERNAL "")

# Check Cython version
execute_process(COMMAND ${CYTHON_EXECUTABLE} --version
                ERROR_VARIABLE CYTHON_VERSION
                RESULT_VARIABLE _EXIT_STATUS
                )
if(NOT _EXIT_STATUS EQUAL 0)
    message(SEND_ERROR "Cython command failed")
endif()
string(REGEX MATCH "([0-9]|\\.)+" CYTHON_VERSION ${CYTHON_VERSION})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Cython
    VERSION_VAR   CYTHON_VERSION
    REQUIRED_VARS CYTHON_EXECUTABLE CYTHON_VERSION)

mark_as_advanced(CYTHON_EXECUTABLE CYTHON_VERSION)
