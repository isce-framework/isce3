set(LISCE isce.${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR})

# Copy XML file used for deserialization
file(COPY archive.xml DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

foreach (TESTNAME serializeEllipsoid)
    add_executable(${TESTNAME} ${TESTNAME}.cpp)
    target_link_libraries(${TESTNAME} PUBLIC ${LISCE} gtest)
    target_include_directories(${TESTNAME} PUBLIC ${ISCE_BUILDINCLUDEDIR})
    add_test(NAME ${TESTNAME} COMMAND ${TESTNAME})
endforeach ()

include(CTest)
