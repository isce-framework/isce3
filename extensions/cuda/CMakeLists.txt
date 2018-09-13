
set (LISCE isce.${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR})
set (LCUDAISCE iscecuda.${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR})

# Pull in the source files
include(core/CMakeLists.txt)
include(product/CMakeLists.txt)
include(geometry/CMakeLists.txt)

# Set the output library
add_library(${LCUDAISCE} SHARED
            ${ISCE_CUDA_core_SRCS}
            ${ISCE_CUDA_geometry_SRCS})

# Force nvcc to compile with c++11
get_target_property(MYPROPS ${LCUDAISCE} COMPILE_OPTIONS)
string(REPLACE "c++17" "c++11" MYPROPS ${MYPROPS})
set_target_properties(${LCUDAISCE} PROPERTIES COMPILE_OPTIONS ${MYPROPS})

# Allow for separable compilation and device code linking (-dc flag to nvcc)
set_target_properties(${LCUDAISCE} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${LCUDAISCE} PROPERTIES CUDA_HOST_COMPILATION_CPP ON)

# Add library dependencies
target_link_libraries(${LCUDAISCE} ${LISCE})

# Add any extra CUDA dependencies (helper functions)
set(CUDA_COMMON ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../../../samples/common/inc)
include_directories(${LCUDAISCE} PUBLIC ${CUDA_COMMON})

# Install shared library
install(TARGETS ${LCUDAISCE}
        DESTINATION ${ISCE_LIBDIR}
        COMPONENT ISCE_CUDA)
