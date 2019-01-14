#We need to modify LD_LIBRARY_PATH and PYTHONPATH here to point to
#libraries that just got built to ensure we are testing updated code
#This needs to be handled in a smarter way by automatically getting 
#location of targets - still need to figure this out

#Get libisce path to add to LD_LIBRARY_PATH
set (LISCELOCATION ${CMAKE_BINARY_DIR}/lib/isce/core)
set (PYTEST_LD_LIBRARY_PATH "${LISCELOCATION}:$ENV{LD_LIBRARY_PATH}" CACHE INTERNAL "library paths for pytest")

#Get iscecore module path for PYTHONPATH
set (MISCELOCATION ${CMAKE_BINARY_DIR}/extensions/cython/isce)
set(PYTEST_PYTHONPATH "${MISCELOCATION}:$ENV{PYTHONPATH}" CACHE INTERNAL "python paths for pytest")

add_subdirectory(io)
add_subdirectory(core)
add_subdirectory(geometry)
if (WITH_CUDA)
        add_subdirectory(cuda)
endif(WITH_CUDA)
