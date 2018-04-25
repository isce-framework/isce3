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


###Ensure tracking is own for testing
enable_testing()

###Include custom installation paths and checks for the project
include(ConfigISCE)

###Explicit check to prevent in-source builds
AssureOutOfSourceBuilds()

####Check CXX Version and Standard to C++14
CheckCXX()

###Check if an appropriate python is installed
CheckISCEPython()

##Check for Pyre
CheckPyre()

##Check for Cython
include(useCython)

##Check for CUDA
CheckCUDA()

##Check for OpenMP
CheckOpenMP()

##Check for GDAL
CheckGDAL()

##Check for Armadillo
CheckArmadillo()

##Check for Nosetests
checkNosetests()


###Layout same install directory structure as pyre
InitInstallDirLayout()

###Setup for gtest framework
add_subdirectory(contrib)

###Recurs directories
add_subdirectory(lib)

###Cython lib
add_subdirectory(extensions)

###Setup tests
add_subdirectory(tests)
