#We need to modify LD_LIBRARY_PATH and CUDAPATH here to point to
#libraries that just got built to ensure we are testing updated code
#This needs to be handled in a smarter way by automatically getting 
#location of targets - still need to figure this out

#Get libisce path to add to LD_LIBRARY_PATH
set (LISCELOCATION ${CMAKE_BINARY_DIR}/lib/isce/core)
set (NOSE_LD_LIBRARY_PATH "${LISCELOCATION}:$ENV{LD_LIBRARY_PATH}" CACHE INTERNAL "library paths for nosetests")

#Get iscecore module path for CUDAPATH
set (MISCELOCATION ${CMAKE_BINARY_DIR}/extensions/cuda)
set(NOSE_CUDAPATH "${MISCELOCATION}:$ENV{CUDAPATH}" CACHE INTERNAL "cuda paths for nosetests")

add_subdirectory(core)
