##Find pyre and journal libraries
set(LISCE isce.${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR})

add_executable(poly1d poly1d.cpp)
target_link_libraries(poly1d PUBLIC ${LISCE} gtest)
target_include_directories(poly1d PUBLIC ${ISCE_BUILDINCLUDEDIR})
add_test(NAME poly1d COMMAND poly1d)

include(CTest)
