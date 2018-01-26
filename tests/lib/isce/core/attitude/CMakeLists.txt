#Find pyre and journal libraries
#find_library(LPYRE pyre HINTS ${PYRE_LIB_DIR})
#find_library(LJOURNAL journal HINTS ${PYRE_LIB_DIR})
set(LISCE isce.${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR})

foreach (TESTNAME euler)
    add_executable(${TESTNAME} ${TESTNAME}.cpp)
    target_link_libraries(${TESTNAME} PUBLIC ${LISCE} gtest)
    target_include_directories(${TESTNAME} PUBLIC ${ISCE_BUILDINCLUDEDIR})
    add_test(NAME ${TESTNAME} COMMAND ${TESTNAME})
endforeach ()

include(CTest)
