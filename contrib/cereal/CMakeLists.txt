# CMake builder for cereal

# Set cereal source include directory to allow rest of project to find header files
set(CEREAL_SRC_INCLUDEDIR
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    CACHE STRING
    "cereal/include")

# Set output cereal install directory
set(CEREAL_INCLUDEDIR
    include/cereal
    CACHE STRING
    "cereal/include")

add_subdirectory(include)
