set(PACKAGENAME core)
string(TOLOWER ${PROJECT_NAME} LOCALPROJ)

add_executable(gpuLUT1d gpuLUT1d.cpp)
target_link_libraries(gpuLUT1d PUBLIC ${LISCE} ${LCUDAISCE} gtest)
target_include_directories(gpuLUT1d PUBLIC ${ISCE_BUILDINCLUDEDIR} ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/${PACKAGENAME}/cuda ${PYRE_INCLUDE_DIR} ${GDAL_INCLUDE_DIR})
message(STATUS ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/${PACKAGENAME}/cuda)
add_test(NAME gpuLUT1d COMMAND gpuLUT1d)

include(CTest)
