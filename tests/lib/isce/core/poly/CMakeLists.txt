set(LISCE isce.${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR})

add_executable(poly1d poly1d.cpp)
target_link_libraries(poly1d PUBLIC ${LISCE} gtest)
target_include_directories(poly1d PUBLIC ${ISCE_BUILDINCLUDEDIR} ${GDAL_INCLUDE_DIR})
add_test(NAME poly1d COMMAND poly1d)

include(CTest)
