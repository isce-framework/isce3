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
# Conda/micromamba Python interpreters are built in Release mode (with NDEBUG
# defined). Without NDEBUG, libc++ may use different container layouts or
# hardening modes, causing an ABI mismatch between the extension and the
# interpreter. This manifests as:
#
#   ImportError: Internal error while parsing type signature (2)
#
# Force NDEBUG for the pybind11 extension so its libc++ ABI matches the
# Release-built Python interpreter. The rest of isce3 still builds in full
# Debug mode.
# ---------------------------------------------------------------------------
if(APPLE AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(${ISCEEXTENSION} PRIVATE NDEBUG)
endif()

# install
install(TARGETS ${ISCEEXTENSION}
        LIBRARY DESTINATION ${ISCE_PACKAGESDIR}/isce3/ext
        COMPONENT Extensions
        )
