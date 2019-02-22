set(PACKAGENAME image)
string(TOLOWER ${PROJECT_NAME} LOCALPROJ)

add_executable(gpuResampSlc gpuResampSlc.cpp)

target_link_libraries(gpuResampSlc PUBLIC ${LISCE} ${LCUDAISCE} gtest)

target_include_directories(gpuResampSlc PUBLIC
                           ${ISCE_BUILDINCLUDEDIR}
                           ${CEREAL_SRC_INCLUDEDIR}
                           ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/cuda/${PACKAGENAME}
                           ${GDAL_INCLUDE_DIR}
                           ${PYRE_INCLUDE_DIR})

message(STATUS ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/cuda/${PACKAGENAME})
add_test(NAME gpuResampSlc COMMAND gpuResampSlc)

include(CTest)
