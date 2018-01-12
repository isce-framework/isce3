set(LISCE isce.${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR})

add_executable(lut2d lut2d.cpp)
target_link_libraries(lut2d PUBLIC ${LISCE} gtest)
target_include_directories(lut2d PUBLIC ${ISCE_BUILDINCLUDEDIR})
add_test(NAME lut2d COMMAND lut2d)

include(CTest)
