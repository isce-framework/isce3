set(PACKAGENAME core)
string(TOLOWER ${PROJECT_NAME} LOCALPROJ)

add_executable(gpuPoly2d gpuPoly2d.cpp)
target_link_libraries(gpuPoly2d PUBLIC ${LISCE} ${LCUDAISCE} gtest)
target_include_directories(gpuPoly2d PUBLIC ${ISCE_BUILDINCLUDEDIR} ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/${PACKAGENAME}/cuda ${GDAL_INCLUDE_DIR})
message(STATUS ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/${PACKAGENAME}/cuda)
add_test(NAME gpuPoly2d COMMAND gpuPoly2d)

include(CTest)
