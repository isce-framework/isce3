# Get name of ISCE CUDA library
include(IsceCuda)

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

# Add library dependencies
find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(CUDAFFT_LIBRARY cufft ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
target_include_directories(${LISCECUDA} PUBLIC
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(${LISCECUDA} PUBLIC
                      ${LISCE}
                      ${CUDART_LIBRARY}
                      ${CUDAFFT_LIBRARY}
                      )

# Install public headers
foreach(HEADER ${HEADERS})
    get_filename_component(DIR ${HEADER} DIRECTORY)
    install(FILES       ${HEADER}
            DESTINATION include/iscecuda/${DIR}
            COMPONENT   ISCE_CUDA)
endforeach()

# Install shared library
install(TARGETS ${LISCECUDA}
        EXPORT isce3-targets
        DESTINATION ${ISCE_LIBDIR}
        COMPONENT ISCE_CUDA)
