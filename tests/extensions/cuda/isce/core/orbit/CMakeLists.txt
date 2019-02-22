set(PACKAGENAME core)
string(TOLOWER ${PROJECT_NAME} LOCALPROJ)

add_executable(gpuOrbit gpuOrbit.cpp)
target_link_libraries(gpuOrbit PUBLIC ${LISCE} ${LCUDAISCE} gtest)
target_include_directories(gpuOrbit PUBLIC ${ISCE_BUILDINCLUDEDIR} ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/${PACKAGENAME}/cuda ${GDAL_INCLUDE_DIR})
message(STATUS ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/${PACKAGENAME}/cuda)
add_test(NAME gpuOrbit COMMAND gpuOrbit)

include(CTest)
