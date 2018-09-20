set(PACKAGENAME geometry)
string(TOLOWER ${PROJECT_NAME} LOCALPROJ)

set(LISCE isce.${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR})
set(LCUDAISCE iscecuda.${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR})

add_executable(gpuTopo gpuTopo.cpp)

target_link_libraries(gpuTopo PUBLIC ${LISCE} ${LCUDAISCE} gtest)

target_include_directories(gpuTopo PUBLIC
                           ${ISCE_BUILDINCLUDEDIR}
                           ${CEREAL_SRC_INCLUDEDIR}
                           ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/cuda/${PACKAGENAME}
                           ${GDAL_INCLUDE_DIR}
                           ${PYRE_INCLUDE_DIR})

message(STATUS ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/cuda/${PACKAGENAME})
add_test(NAME gpuTopo COMMAND gpuTopo)

include(CTest)
