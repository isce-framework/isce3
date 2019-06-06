
#Get libisce path to add to LD_LIBRARY_PATH
set (LISCELOCATION ${CMAKE_BINARY_DIR}/lib/isce/core)
set (PYTEST_LD_LIBRARY_PATH "${LISCELOCATION}:$ENV{LD_LIBRARY_PATH}" CACHE INTERNAL "library paths for pytest")

#Get iscecore module path for PYTHONPATH
set (MISCELOCATION ${ISCE_BUILDPACKAGESDIR})
set(PYTEST_PYTHONPATH "${MISCELOCATION}:$ENV{PYTHONPATH}" CACHE INTERNAL "python paths for pytest")

add_subdirectory(io)
add_subdirectory(core)
add_subdirectory(geometry)
