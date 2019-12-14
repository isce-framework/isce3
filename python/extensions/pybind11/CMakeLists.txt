# get source/header files
# XXX no need to install headers, should we still track them?
include(Sources.cmake)
include(Headers.cmake)

# add build target
# library name MUST match the name of the pybind11 module
set(ISCEEXTENSION isce)
pybind11_add_module(${ISCEEXTENSION} ${SRCS})

# required c++ standard
# XXX nominally, c++17 is experimentally supported but it produces a long list
# XXX of awful compilation errors for me (w/ g++ v6.1)
set_target_properties(${ISCEEXTENSION}
    PROPERTIES
        CXX_STANDARD 14
        CXX_STANDARD_REQUIRED ON
    )

# include path
target_include_directories(${ISCEEXTENSION} PUBLIC ${CMAKE_CURRENT_LIST_DIR})

# dependencies
target_link_libraries(${ISCEEXTENSION}
    PUBLIC
        ${LISCE}
        $<$<BOOL:${WITH_CUDA}>:${LISCECUDA}>
    )

# install
install(TARGETS ${ISCEEXTENSION}
        LIBRARY DESTINATION ${ISCE_PACKAGESDIR}
        COMPONENT Extensions
        )
