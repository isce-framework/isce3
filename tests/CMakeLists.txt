# Create a variable specifying the absolute path to our test data
# it is important that the value includes the terminating '/' since use of this macro relies on
# automatic concatenation of consecutive string literals
set(TESTDATA_DIR ${CMAKE_CURRENT_LIST_DIR}/data/)

add_subdirectory(data)
add_subdirectory(contrib)
add_subdirectory(lib)
add_subdirectory(python)
include(CTest)
