message(STATUS "gtest paths: ${GTEST_INCLUDEDIR} ${GTEST_LIBDIR}")

add_executable(square_root_test2    square_root_test2.cc)
target_include_directories(square_root_test2 PUBLIC ${GTEST_INCLUDEDIR}/..)
find_library(GTESTLIBS gtest HINTS ${GTEST_LIBDIR})
target_link_libraries(square_root_test2 PUBLIC ${GTESTLIBS})
add_test(NAME square_root_test COMMAND square_root_test2)
