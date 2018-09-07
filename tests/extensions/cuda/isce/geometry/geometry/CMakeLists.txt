set(PACKAGENAME geometry)
string(TOLOWER ${PROJECT_NAME} LOCALPROJ)

set(LISCE isce.${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR})
set(LCUDAISCE iscecuda.${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR})

add_executable(gpuGeometry gpuGeometry.cpp)

target_link_libraries(gpuGeometry PUBLIC ${LISCE} ${LCUDAISCE} gtest)

target_include_directories(gpuGeometry PUBLIC
                           ${ISCE_BUILDINCLUDEDIR}
                           ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/cuda/${PACKAGENAME}
                           ${GDAL_INCLUDE_DIR}
                           ${PYRE_INCLUDE_DIR})

file(COPY input_data.txt DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY output_data.txt DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY output_data_zerodop.txt DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

message(STATUS ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/cuda/${PACKAGENAME})
add_test(NAME gpuGeometry COMMAND gpuGeometry)

include(CTest)
