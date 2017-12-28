add_executable(square_root_test2    square_root_test2.cc)
target_link_libraries(square_root_test2 PUBLIC gtest)
add_test(NAME square_root_test2 COMMAND square_root_test2)

include(CTest)
