set(PACKAGENAME geometry)
string(TOLOWER ${PROJECT_NAME} LOCALPROJ)

add_executable(gpuGeometry gpuGeometry.cpp)

target_link_libraries(gpuGeometry PUBLIC ${LISCE} ${LCUDAISCE} gtest)

target_include_directories(gpuGeometry PUBLIC
                           ${ISCE_BUILDINCLUDEDIR}
                           ${CEREAL_SRC_INCLUDEDIR}
                           ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/cuda/${PACKAGENAME}
                           ${GDAL_INCLUDE_DIR}
                           ${PYRE_INCLUDE_DIR})

message(STATUS ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/cuda/${PACKAGENAME})
add_test(NAME gpuGeometry COMMAND gpuGeometry)

include(CTest)
