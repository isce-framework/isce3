
set (LISCE isce.${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR})
set (LCUDAISCE iscecuda.${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR})

# Pull in the source files
include(core/CMakeLists.txt)
include(product/CMakeLists.txt)
include(geometry/CMakeLists.txt)
include(image/CMakeLists.txt)

# Set the output library
add_library(${LCUDAISCE} SHARED
            ${ISCE_CUDA_core_SRCS}
            ${ISCE_CUDA_geometry_SRCS}
            ${ISCE_CUDA_image_SRCS})

# Force nvcc to compile with c++14
get_target_property(MYPROPS ${LCUDAISCE} COMPILE_OPTIONS)
string(REPLACE "c++17" "c++14" MYPROPS ${MYPROPS})
set_target_properties(${LCUDAISCE} PROPERTIES COMPILE_OPTIONS ${MYPROPS})

# Allow for separable compilation and device code linking (-dc flag to nvcc)
set_target_properties(${LCUDAISCE} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${LCUDAISCE} PROPERTIES CUDA_HOST_COMPILATION_CPP ON)

# Add library dependencies
target_link_libraries(${LCUDAISCE} ${LISCE})

# Install CUDA helper header files
set(HELPER_HEADERS "")
foreach(HFILE helper_cuda.h helper_string.h)
    set(DESTFILE ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/cuda/${HFILE})
    configure_file(${CMAKE_CURRENT_LIST_DIR}/${HFILE} ${DESTFILE} COPYONLY)
    list(APPEND ${HELPER_HEADERS} ${DESTFILE})
endforeach()

install(FILES ${HELPER_HEADERS}
        DESTINATION ${ISCE_INCLUDEDIR}/${LOCALPROJ}/cuda
        COMPONENT ISCE_CUDA)

# Install shared library
install(TARGETS ${LCUDAISCE}
        DESTINATION ${ISCE_LIBDIR}
        COMPONENT ISCE_CUDA)
