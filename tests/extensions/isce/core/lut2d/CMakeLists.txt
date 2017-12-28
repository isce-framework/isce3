##Find pyre and journal libraries
find_library(LPYRE pyre HINTS ${PYRE_LIB_DIR})
find_library(LJOURNAL journal HINTS ${PYRE_LIB_DIR})
set(LISCE isce.${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR})

add_executable(lut2d lut2d.cpp)
target_link_libraries(lut2d PUBLIC ${LISCE} ${LPYRE} ${LJOURNAL} gtest)
target_include_directories(lut2d PUBLIC ../../../../../lib)
add_test(NAME lut2d COMMAND lut2d)

include(CTest)
