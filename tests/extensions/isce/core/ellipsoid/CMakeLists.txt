##Find pyre and journal libraries
find_library(LPYRE pyre HINTS ${PYRE_LIB_DIR})
find_library(LJOURNAL journal HINTS ${PYRE_LIB_DIR})
set(LISCE isce.${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR})

add_executable(ellipsoid ellipsoid.cpp)
target_link_libraries(ellipsoid PUBLIC ${LISCE} ${LPYRE} ${LJOURNAL} gtest)
target_include_directories(ellipsoid PUBLIC ../../../../../lib)
add_test(NAME ellipsoid COMMAND ellipsoid)

include(CTest)
