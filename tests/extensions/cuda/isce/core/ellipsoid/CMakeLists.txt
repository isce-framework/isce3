set(LISCE isce.${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR})
set (LCUDAISCE iscecuda.${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR})

add_executable(gpuEllipsoid gpuEllipsoid.cpp)
target_link_libraries(gpuEllipsoid PUBLIC ${LISCE} gtest)
target_include_directories(gpuEllipsoid PUBLIC ${ISCE_BUILDINCLUDEDIR} ${GDAL_INCLUDE_DIR})
add_test(NAME gpuEllipsoid COMMAND gpuEllipsoid)

include(CTest)
