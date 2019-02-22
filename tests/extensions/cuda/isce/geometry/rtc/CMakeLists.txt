set(PACKAGENAME rtc)
string(TOLOWER ${PROJECT_NAME} LOCALPROJ)

add_executable(gpuRTC gpuRTC.cpp)

target_link_libraries(gpuRTC PUBLIC ${LISCE} ${LCUDAISCE} gtest)

target_include_directories(gpuRTC PUBLIC
                           ${ISCE_BUILDINCLUDEDIR}
                           ${CEREAL_SRC_INCLUDEDIR}
                           ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/cuda/${PACKAGENAME}
                           ${GDAL_INCLUDE_DIR}
                           ${PYRE_INCLUDE_DIR})

message(STATUS ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/cuda/${PACKAGENAME})
add_test(NAME gpuRTC COMMAND gpuRTC)

include(CTest)
