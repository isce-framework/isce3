include(Sources.cmake)
include(Headers.cmake)

# Create library target
add_library(${LISCECUDA} SHARED ${SOURCES})

# Allow for separable compilation and device code linking (-dc flag to nvcc)
set_target_properties(${LISCECUDA} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${LISCECUDA} PROPERTIES CUDA_HOST_COMPILATION_CPP ON)

# Allow calling host constexpr functions from device code
target_compile_options(${LISCECUDA} PUBLIC
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-Werror=cross-execution-space-call>
    )

# CUDA 13.0 made __global__ kernels hidden and __global__ template stubs
# internal-linkage by default, which breaks cross-shared-library kernel launches
# (template code in the pybind module launches kernels defined in this library).
# Restore the pre-13 behavior. These flags exist since CUDA 12.8.
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
    target_compile_options(${LISCECUDA} PUBLIC
        $<$<COMPILE_LANGUAGE:CUDA>:--device-entity-has-hidden-visibility=false>
        $<$<COMPILE_LANGUAGE:CUDA>:-static-global-template-stub=false>
        )
endif()

# Disable thrust's architecture-specific inline namespace that was introduced in
# https://github.com/NVIDIA/cccl/pull/443. Without this, thrust symbols in header files
# will have different names in C++ TUs vs CUDA TUs, leading to numerous ODR violations
# and linker errors. This can cause ABI issues if we link against other libraries that
# do the same, but in practice should be reasonably safe.
target_compile_definitions(${LISCECUDA} PUBLIC
    THRUST_DISABLE_ABI_NAMESPACE
    THRUST_IGNORE_ABI_NAMESPACE_ERROR
    )

# Add library dependencies
find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(CUDAFFT_LIBRARY cufft ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
target_include_directories(${LISCECUDA} PUBLIC
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(${LISCECUDA} PUBLIC
                      ${LISCE}
                      ${CUDART_LIBRARY}
                      ${CUDAFFT_LIBRARY}
                      OpenMP::OpenMP_CUDA_Optional
                      )

# Specify API version and build version (used to generate name and soname
# symlinks on Unix systems).
set_target_properties(${LISCECUDA} PROPERTIES
    SOVERSION   ${PROJECT_VERSION_MAJOR}
    VERSION     ${PROJECT_VERSION}
    )

# Install public headers
foreach(HEADER ${HEADERS})
    get_filename_component(DIR ${HEADER} DIRECTORY)
    install(FILES       ${HEADER}
            DESTINATION include/isce3/cuda/${DIR}
            COMPONENT   ISCE_CUDA)
endforeach()

# Install shared library
install(TARGETS ${LISCECUDA}
        EXPORT isce3-targets
        DESTINATION ${ISCE_LIBDIR}
        COMPONENT ISCE_CUDA)
