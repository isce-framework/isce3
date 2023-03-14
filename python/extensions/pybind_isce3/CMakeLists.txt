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

# install
install(TARGETS ${ISCEEXTENSION}
        LIBRARY DESTINATION ${ISCE_PACKAGESDIR}/isce3/ext
        COMPONENT Extensions
        )
