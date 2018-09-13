set(PACKAGENAME core)
string(TOLOWER ${PROJECT_NAME} LOCALPROJ)

set(LISCE isce.${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR})
set(LCUDAISCE iscecuda.${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR})

foreach (TESTNAME gpuPolar gpuUTM gpuCEA gpuGeocent)
    add_executable(${TESTNAME} ${TESTNAME}.cpp)
    target_link_libraries(${TESTNAME} PUBLIC ${LISCE} ${LCUDAISCE} gtest)
    target_include_directories(${TESTNAME} PUBLIC ${ISCE_BUILDINCLUDEDIR} ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/${PACKAGENAME}/cuda ${GDAL_INCLUDE_DIR})
    get_target_property(MYPROPS  ${TESTNAME} COMPILE_OPTIONS)
    string(REPLACE "c++17" "c++11" MYPROPS ${MYPROPS})
    set_target_properties(${TESTNAME} PROPERTIES COMPILE_OPTIONS ${MYPROPS})
    add_test(NAME ${TESTNAME} COMMAND ${TESTNAME})
endforeach()

#add_executable(gpuPolarBenchmark gpuPolarBenchmark.cpp)
#target_link_libraries(gpuPolarBenchmark PUBLIC ${LISCE} ${LCUDAISCE} gtest)
#target_include_directories(gpuPolarBenchmark PUBLIC ${ISCE_BUILDINCLUDEDIR} ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/${PACKAGENAME}/cuda ${GDAL_INCLUDE_DIR})
#message(STATUS ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/${PACKAGENAME}/cuda)
#add_test(NAME gpuPolarBenchmark COMMAND gpuPolarBenchhmark)

include(CTest)
