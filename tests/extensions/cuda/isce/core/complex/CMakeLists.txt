set(PACKAGENAME core)
string(TOLOWER ${PROJECT_NAME} LOCALPROJ)

add_executable(gpuComplex gpuComplex.cpp)
target_link_libraries(gpuComplex PUBLIC ${LISCE} ${LCUDAISCE} gtest)
target_include_directories(gpuComplex PUBLIC ${ISCE_BUILDINCLUDEDIR} ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/${PACKAGENAME}/cuda ${GDAL_INCLUDE_DIR})
message(STATUS ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/${PACKAGENAME}/cuda)
add_test(NAME gpuComplex COMMAND gpuComplex)

include(CTest)
