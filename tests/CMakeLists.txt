# Create a variable specifying the absolute path to our test data
# it is important that the value includes the terminating '/' since use of this macro relies on
# automatic concatenation of consecutive string literals
set(TESTDATA_DIR ${CMAKE_CURRENT_LIST_DIR}/data/)

# adjust the PYTHONPATH to point to the freshly built packages
if(DEFINED ENV{DESTDIR})
    set(PYTEST_PYTHONPATH "$ENV{DESTDIR}/${ISCE_PACKAGESDIR_FULL}:$ENV{PYTHONPATH}")
else()
    set(PYTEST_PYTHONPATH "${ISCE_PACKAGESDIR_FULL}:$ENV{PYTHONPATH}")
endif()

add_subdirectory(extern)
add_subdirectory(cxx)
add_subdirectory(python)
include(CTest)
