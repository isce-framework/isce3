##CMake builder for gtest

set (GTEST_INCLUDEDIR include/gtest CACHE STRING "gtest/include")
set (GTEST_LIBDIR lib CACHE STRING "gtest/lib")

add_subdirectory(include)
add_subdirectory(src)
