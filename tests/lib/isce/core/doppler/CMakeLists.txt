#Find pyre and journal libraries
find_library(LPYRE pyre HINTS ${PYRE_LIB_DIR})
find_library(LJOURNAL journal HINTS ${PYRE_LIB_DIR})
set(LISCE isce.${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR})

foreach (TESTNAME doppler)
    add_executable(${TESTNAME} ${TESTNAME}.cpp)
    target_link_libraries(${TESTNAME} PUBLIC ${LISCE} ${LPYRE} ${LJOURNAL} gtest)
    target_include_directories(${TESTNAME} PUBLIC ${ISCE_BUILDINCLUDEDIR})
    add_test(NAME ${TESTNAME} COMMAND ${TESTNAME})
    install(FILES orbit_data.txt DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endforeach ()

include(CTest)
