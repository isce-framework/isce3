# -*- cmake -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# setup cmake
function(pyre_cmakeInit)
  # get the source directory
  get_filename_component(srcdir "${CMAKE_SOURCE_DIR}" REALPATH)
  # get the staging directory
  get_filename_component(stgdir "${CMAKE_BINARY_DIR}" REALPATH)
  # if we are building within the source directory
  if ("${srcdir}" STREQUAL "${stgdir}")
    # complain and bail
    message(FATAL_ERROR "in-source build detected; please run cmake in a build directory")
  endif()

  # host info
  # get
  string(TOLOWER ${CMAKE_HOST_SYSTEM_NAME} HOST_OS)
  string(TOLOWER ${CMAKE_HOST_SYSTEM_PROCESSOR} HOST_ARCH)
  # export
  set(HOST_OS ${HOST_OS} PARENT_SCOPE)
  set(HOST_ARCH ${HOST_ARCH} PARENT_SCOPE)
  set(HOST_PLATFORM ${HOST_OS}_${HOST_ARCH} PARENT_SCOPE)

  # quiet install
  set(CMAKE_INSTALL_MESSAGE LAZY PARENT_SCOPE)

  # if the user asked for CUDA support
  if (WITH_CUDA)
    # turn it on
    enable_language(CUDA)
  endif()

  # all done
endfunction(pyre_cmakeInit)


# setup the c++ compiler
function(pyre_cxxInit)
  # require c++17
  set(CMAKE_CXX_STANDARD 17 PARENT_SCOPE)
  set(CMAKE_CXX_STANDARD_REQUIRED ON PARENT_SCOPE)
  set(CMAKE_CXX_EXTENSIONS OFF PARENT_SCOPE)
  # all done
endfunction(pyre_cxxInit)


# setup python
function(pyre_pythonInit)
  # ask the executable for the module suffix
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
        "from distutils.sysconfig import *; print(get_config_var('EXT_SUFFIX'))"
    RESULT_VARIABLE PYTHON3_SUFFIX_STATUS
    OUTPUT_VARIABLE PYTHON3_SUFFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  # export
  set(PYTHON3_SUFFIX ${PYTHON3_SUFFIX} PARENT_SCOPE)
  # all done
endfunction(pyre_pythonInit)


# describe the layout of the staging area
function(pyre_stagingInit)
  # the layout
  set(PYRE_STAGING_PACKAGES ${CMAKE_BINARY_DIR}/packages PARENT_SCOPE)
  # all done
endfunction(pyre_stagingInit)


# describe the installation layout
function(pyre_destinationInit)
  # create variables to hold the roots in the install directory
  set(PYRE_DEST_INCLUDE include PARENT_SCOPE)
  set(PYRE_DEST_PACKAGES packages PARENT_SCOPE)
  # all done
endfunction(pyre_destinationInit)


# ask git for the most recent tag and use it to build the version
function(pyre_getVersion)
  # git
  find_package(Git REQUIRED)
  # get tag info
  execute_process(
    COMMAND ${GIT_EXECUTABLE} describe --tags --long --always
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    RESULT_VARIABLE GIT_DESCRIBE_STATUS
    OUTPUT_VARIABLE GIT_DESCRIBE_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )

  # the parser of the "git describe" result
  set(GIT_DESCRIBE "v([0-9]+)\\.([0-9]+)\\.([0-9]+)-([0-9]+)-g(.+)" )

  if(GIT_DESCRIBE_VERSION MATCHES ${GIT_DESCRIBE})
      # parse the bits
      string(REGEX REPLACE ${GIT_DESCRIBE} "\\1" REPO_MAJOR ${GIT_DESCRIBE_VERSION} )
      string(REGEX REPLACE ${GIT_DESCRIBE} "\\2" REPO_MINOR ${GIT_DESCRIBE_VERSION})
      string(REGEX REPLACE ${GIT_DESCRIBE} "\\3" REPO_MICRO ${GIT_DESCRIBE_VERSION})
      string(REGEX REPLACE ${GIT_DESCRIBE} "\\5" REPO_COMMIT ${GIT_DESCRIBE_VERSION})
  else()
      set(REPO_MAJOR 1)
      set(REPO_MINOR 0)
      set(REPO_MICRO 0)
      set(REPO_COMMIT ${GIT_DESCRIBE_VERSION})
  endif()

  # export our local variables
  set(REPO_MAJOR ${REPO_MAJOR} PARENT_SCOPE)
  set(REPO_MINOR ${REPO_MINOR} PARENT_SCOPE)
  set(REPO_MICRO ${REPO_MICRO} PARENT_SCOPE)
  set(REPO_COMMIT ${REPO_COMMIT} PARENT_SCOPE)

  # set the variables used in the package meta-data files
  set(MAJOR ${REPO_MAJOR} PARENT_SCOPE)
  set(MINOR ${REPO_MINOR} PARENT_SCOPE)
  set(MICRO ${REPO_MICRO} PARENT_SCOPE)
  set(REVISION ${REPO_COMMIT} PARENT_SCOPE)
  string(TIMESTAMP TODAY PARENT_SCOPE)

  # all done
endfunction(pyre_getVersion)


# end of file
