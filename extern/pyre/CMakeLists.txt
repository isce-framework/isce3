cmake_minimum_required(VERSION 3.13)

# project settings
set(CMAKE_CXX_STANDARD          14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS       OFF)

project(pyre LANGUAGES CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# version info
set(PYRE_MAJOR 1)
set(PYRE_MINOR 9)
execute_process(COMMAND "git rev-parse --short HEAD"
                OUTPUT_VARIABLE PYRE_REVISION)

add_library(pyre OBJECT)

add_subdirectory(lib)
add_subdirectory(packages)
