# get source/header files
include(Sources.cmake)

# add build target
# library name MUST match the name of the pybind11 module
set(ISCEEXTENSION pybind_isce3)

pybind11_add_module(${ISCEEXTENSION} ${SRCS})

# the name of the output
set_target_properties(${ISCEEXTENSION} PROPERTIES LIBRARY_OUTPUT_NAME isce3)

# include path
target_include_directories(${ISCEEXTENSION} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/..)

# dependencies
target_link_libraries(${ISCEEXTENSION} PUBLIC ${LISCE})
target_link_libraries(${ISCEEXTENSION} PRIVATE OpenMP::OpenMP_CXX_Optional)

if(WITH_CUDA)
    target_link_libraries(${ISCEEXTENSION} PUBLIC ${LISCECUDA})
endif()

# ---------------------------------------------------------------------------
# macOS Debug workaround for pybind11 import failures
#
# Conda/micromamba Python interpreters are built in Release mode. When the
# pybind11 extension is compiled with libc++ debug STL enabled (as can happen
# in Debug builds on macOS), container ABI differences can cause pybind11
# signature parsing to fail at module import time:
#
#   ImportError: Internal error while parsing type signature (2)
#
# To avoid this mismatch we explicitly disable libc++ debug mode for the
# Python extension module when building Debug on macOS.
# ---------------------------------------------------------------------------
if(APPLE AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(${ISCEEXTENSION} PRIVATE -U_LIBCPP_DEBUG)
endif()

# install
install(TARGETS ${ISCEEXTENSION}
        LIBRARY DESTINATION ${ISCE_PACKAGESDIR}/isce3/ext
        COMPONENT Extensions
        )
