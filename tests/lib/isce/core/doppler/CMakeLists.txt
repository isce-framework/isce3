
set(LISCE isce.${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR})

foreach (TESTNAME doppler)
    add_executable(${TESTNAME} ${TESTNAME}.cpp)
    target_link_libraries(${TESTNAME} PUBLIC ${LISCE} gtest)
    target_include_directories(${TESTNAME} PUBLIC ${PYRE_INCLUDE_DIR} ${ISCE_BUILDINCLUDEDIR} ${GDAL_INCLUDE_DIR})
    add_test(NAME ${TESTNAME} COMMAND ${TESTNAME})
    file(COPY orbit_data.txt DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endforeach ()

include(CTest)
