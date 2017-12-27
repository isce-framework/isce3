###CMake builder for ISCE 3.0

###Minimum version of cmake and boiler plate
cmake_minimum_required(VERSION 3.1)

if (POLICY CMP0048)
    cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)

###Project
project(ISCE
        VERSION 3.0)

###Add .cmake to CMAKE MODULE PATH
list(APPEND CMAKE_MODULE_PATH ${ISCE_SOURCE_DIR}/.cmake)

###Include custom installation paths and checks for the project
include(ConfigISCE)

###Explicit check to prevent in-source builds
AssureOutOfSourceBuilds()

####Check CXX Standard to C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

###Check if an appropriate python is installed
CheckISCEPython()

##Check for Pyre
CheckPyre()

###Layout same install directory structure as pyre
InitInstallDirLayout()

###Setup for gtest framework
add_subdirectory(contrib)

###Recurs directories
add_subdirectory(lib)

###Run tests
#enable_testing()
#add_subdirectory(tests)
