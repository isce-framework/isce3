set(PACKAGENAME geometry)
string(TOLOWER ${PROJECT_NAME} LOCALPROJ)

set(LISCE isce.${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR})
set(LCUDAISCE iscecuda.${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR})

add_executable(gpuGeo2rdr gpuGeo2rdr.cpp)

target_link_libraries(gpuGeo2rdr PUBLIC ${LISCE} ${LCUDAISCE} gtest)

target_include_directories(gpuGeo2rdr PUBLIC
                           ${ISCE_BUILDINCLUDEDIR}
                           ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/cuda/${PACKAGENAME}
                           ${GDAL_INCLUDE_DIR}
                           ${PYRE_INCLUDE_DIR})

message(STATUS ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/cuda/${PACKAGENAME})
add_test(NAME gpuGeo2rdr COMMAND gpuGeo2rdr)

include(CTest)
