#Find pyre and journal libraries
find_library(LPYRE pyre HINTS ${PYRE_LIB_DIR})
find_library(LJOURNAL journal HINTS ${PYRE_LIB_DIR})
set(LISCE isce.${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR})

foreach (TESTNAME raster)
    add_executable(${TESTNAME} ${TESTNAME}.cpp)
    target_link_libraries(${TESTNAME} PUBLIC ${LISCE} ${LPYRE} ${LJOURNAL} ${GDAL_LIBRARY} gtest)
    target_include_directories(${TESTNAME} PUBLIC ${ISCE_BUILDINCLUDEDIR} ${GDAL_INCLUDE_DIR})
    add_test(NAME ${TESTNAME} COMMAND ${TESTNAME})
    file(COPY test_data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endforeach ()

include(CTest)
