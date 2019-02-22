add_executable(ellipsoid ellipsoid.cpp)
target_link_libraries(ellipsoid PUBLIC ${LISCE} gtest)
target_include_directories(ellipsoid PUBLIC ${ISCE_BUILDINCLUDEDIR} ${GDAL_INCLUDE_DIR})
add_test(NAME ellipsoid COMMAND ellipsoid)

include(CTest)
