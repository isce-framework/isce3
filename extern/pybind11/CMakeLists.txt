# CMakeLists.txt -- Build system for the pybind11 modules
#
# Copyright (c) 2015 Wenzel Jakob <wenzel@inf.ethz.ch>
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

cmake_minimum_required(VERSION 3.12)

project(pybind11 CXX)

# Check if pybind11 is being used directly or via add_subdirectory
set(PYBIND11_MASTER_PROJECT OFF)
if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  set(PYBIND11_MASTER_PROJECT ON)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/tools")

include(pybind11Tools)

# Cache variables so pybind11_add_module can be used in parent projects
set(PYBIND11_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/include" CACHE INTERNAL "")
set(Python_INCLUDE_DIRS ${Python_INCLUDE_DIRS} CACHE INTERNAL "")
set(Python_LIBRARIES ${Python_LIBRARIES} CACHE INTERNAL "")
set(Python_VERSION_MAJOR ${Python_VERSION_MAJOR} CACHE INTERNAL "")
set(Python_VERSION_MINOR ${Python_VERSION_MINOR} CACHE INTERNAL "")

# NB: when adding a header don't forget to also add it to setup.py
set(PYBIND11_HEADERS
  include/pybind11/detail/class.h
  include/pybind11/detail/common.h
  include/pybind11/detail/descr.h
  include/pybind11/detail/init.h
  include/pybind11/detail/internals.h
  include/pybind11/detail/typeid.h
  include/pybind11/attr.h
  include/pybind11/buffer_info.h
  include/pybind11/cast.h
  include/pybind11/chrono.h
  include/pybind11/common.h
  include/pybind11/complex.h
  include/pybind11/options.h
  include/pybind11/eigen.h
  include/pybind11/embed.h
  include/pybind11/eval.h
  include/pybind11/functional.h
  include/pybind11/numpy.h
  include/pybind11/operators.h
  include/pybind11/pybind11.h
  include/pybind11/pytypes.h
  include/pybind11/stl.h
  include/pybind11/stl_bind.h
)
string(REPLACE "include/" "${CMAKE_CURRENT_SOURCE_DIR}/include/"
       PYBIND11_HEADERS "${PYBIND11_HEADERS}")

include(GNUInstallDirs)

# extract project version from source
file(STRINGS "${PYBIND11_INCLUDE_DIR}/pybind11/detail/common.h" pybind11_version_defines
     REGEX "#define PYBIND11_VERSION_(MAJOR|MINOR|PATCH) ")
foreach(ver ${pybind11_version_defines})
  if (ver MATCHES "#define PYBIND11_VERSION_(MAJOR|MINOR|PATCH) +([^ ]+)$")
    set(PYBIND11_VERSION_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}" CACHE INTERNAL "")
  endif()
endforeach()
set(${PROJECT_NAME}_VERSION ${PYBIND11_VERSION_MAJOR}.${PYBIND11_VERSION_MINOR}.${PYBIND11_VERSION_PATCH})
message(STATUS "pybind11 v${${PROJECT_NAME}_VERSION}")

# Build an interface library target:
add_library(pybind11 INTERFACE)
add_library(pybind11::pybind11 ALIAS pybind11)  # to match exported target
target_include_directories(pybind11 INTERFACE $<BUILD_INTERFACE:${PYBIND11_INCLUDE_DIR}>
                                              $<BUILD_INTERFACE:${Python_INCLUDE_DIRS}>
                                              $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_options(pybind11 INTERFACE $<BUILD_INTERFACE:${PYBIND11_CPP_STANDARD}>)

add_library(module INTERFACE)
add_library(pybind11::module ALIAS module)
if(NOT MSVC)
  target_compile_options(module INTERFACE -fvisibility=hidden)
endif()
target_link_libraries(module INTERFACE pybind11::pybind11)
if(WIN32 OR CYGWIN)
  target_link_libraries(module INTERFACE $<BUILD_INTERFACE:${Python_LIBRARIES}>)
elseif(APPLE)
  target_link_libraries(module INTERFACE "-undefined dynamic_lookup")
endif()

add_library(embed INTERFACE)
add_library(pybind11::embed ALIAS embed)
target_link_libraries(embed INTERFACE pybind11::pybind11 $<BUILD_INTERFACE:${Python_LIBRARIES}>)
