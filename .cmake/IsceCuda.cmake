# Set the name for the ISCE C++/CUDA library
set(LISCECUDA iscecuda.${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR})

# Use correct host compiler with NVCC
set(CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
