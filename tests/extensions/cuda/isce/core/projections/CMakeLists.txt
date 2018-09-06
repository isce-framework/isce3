set(PACKAGENAME core)
string(TOLOWER ${PROJECT_NAME} LOCALPROJ)

set(LISCE isce.${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR})
set(LCUDAISCE iscecuda.${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR})

add_executable(gpuPolar gpuPolar.cpp)
target_link_libraries(gpuPolar PUBLIC ${LISCE} ${LCUDAISCE} gtest)
target_include_directories(gpuPolar PUBLIC ${ISCE_BUILDINCLUDEDIR} ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/${PACKAGENAME}/cuda ${GDAL_INCLUDE_DIR})
message(STATUS ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/${PACKAGENAME}/cuda)
add_test(NAME gpuPolar COMMAND gpuPolar)

add_executable(gpuPolarBenchmark gpuPolarBenchmark.cpp)
target_link_libraries(gpuPolarBenchmark PUBLIC ${LISCE} ${LCUDAISCE} gtest)
target_include_directories(gpuPolarBenchmark PUBLIC ${ISCE_BUILDINCLUDEDIR} ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/${PACKAGENAME}/cuda ${GDAL_INCLUDE_DIR})
message(STATUS ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/${PACKAGENAME}/cuda)
add_test(NAME gpuPolarBenchmark COMMAND gpuPolarBenchhmark)


include(CTest)
