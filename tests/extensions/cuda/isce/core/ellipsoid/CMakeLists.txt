set(PACKAGENAME core)
string(TOLOWER ${PROJECT_NAME} LOCALPROJ)

set(LISCE isce.${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR})
set(LCUDAISCE iscecuda.${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR})

add_executable(gpuEllipsoid gpuEllipsoid.cpp)
target_link_libraries(gpuEllipsoid PUBLIC ${LISCE} ${LCUDAISCE} gtest)
target_include_directories(gpuEllipsoid PUBLIC ${ISCE_BUILDINCLUDEDIR} ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/${PACKAGENAME}/cuda ${GDAL_INCLUDE_DIR})
message(STATUS ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/${PACKAGENAME}/cuda)
add_test(NAME gpuEllipsoid COMMAND gpuEllipsoid)

include(CTest)
