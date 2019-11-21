FILE(GLOB PXDS *.pxd)
add_isce_cythondir(io "${PXDS}")

target_sources(isceextension PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/H5Error.cpp)
target_include_directories(isceextension PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if(WITH_CUDA)
    target_include_directories(iscecudaextension PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()
