##Find pyre and journal libraries
find_library(LPYRE pyre HINTS ${PYRE_LIB_DIR})
find_library(LJOURNAL journal HINTS ${PYRE_LIB_DIR})
set(LISCE isce.${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR})

add_executable(poly1d poly1d.cpp)
target_link_libraries(poly1d PUBLIC ${LISCE} ${LPYRE} ${LJOURNAL} gtest)
target_include_directories(poly1d PUBLIC ../../../../../lib)
add_test(NAME poly1d COMMAND poly1d)

include(CTest)
