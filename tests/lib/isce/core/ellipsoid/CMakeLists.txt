##Find pyre and journal libraries
set(LISCE isce.${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR})

add_executable(ellipsoid ellipsoid.cpp)
target_link_libraries(ellipsoid PUBLIC ${LISCE} gtest)
target_include_directories(ellipsoid PUBLIC ${ISCE_BUILDINCLUDEDIR})
add_test(NAME ellipsoid COMMAND ellipsoid)

include(CTest)
