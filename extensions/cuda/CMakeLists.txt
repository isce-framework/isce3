
set (LISCE isce.${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR})
set (LCUDAISCE iscecuda.${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR})

# Pull in the source files
include(core/CMakeLists.txt)
#include(geometry/CMakeLists.txt)

# Set the output library
add_library(${LCUDAISCE} SHARED
            ${ISCE_CUDA_core_SRCS})
            #${ISCE_CUDA_GEOMETRY_SRCS})

# Force nvcc to compile with c++11
get_target_property(MYPROPS  ${LCUDAISCE} COMPILE_OPTIONS)
string(REPLACE "c++17" "c++11" MYPROPS ${MYPROPS})
set_target_properties(${LCUDAISCE} PROPERTIES COMPILE_OPTIONS ${MYPROPS})

# Add library dependencies
target_link_libraries(${LCUDAISCE} ${LISCE})

# Install shared library
install(TARGETS ${LCUDAISCE}
        DESTINATION ${ISCE_LIBDIR}
        COMPONENT ISCE_CUDA)
