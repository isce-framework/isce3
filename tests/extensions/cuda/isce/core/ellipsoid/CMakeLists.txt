set(PACKAGENAME core)
string(TOLOWER ${PROJECT_NAME} LOCALPROJ)

add_executable(gpuEllipsoid gpuEllipsoid.cpp)
target_link_libraries(gpuEllipsoid PUBLIC ${LISCE} ${LCUDAISCE} gtest)
target_include_directories(gpuEllipsoid PUBLIC ${ISCE_BUILDINCLUDEDIR} ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/${PACKAGENAME}/cuda ${GDAL_INCLUDE_DIR})

get_target_property(MYPROPS  gpuEllipsoid COMPILE_OPTIONS)
string(REPLACE "c++17" "c++11" MYPROPS ${MYPROPS})
set_target_properties(gpuEllipsoid PROPERTIES COMPILE_OPTIONS ${MYPROPS})

message(STATUS ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/${PACKAGENAME}/cuda)
add_test(NAME gpuEllipsoid COMMAND gpuEllipsoid)

include(CTest)
