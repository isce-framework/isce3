
set(LISCE isce.${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR})

foreach (TESTNAME resampslc)
    add_executable(${TESTNAME} ${TESTNAME}.cpp)
    target_link_libraries(${TESTNAME} PUBLIC ${LISCE} gtest)
    target_include_directories(${TESTNAME} PUBLIC ${ISCE_BUILDINCLUDEDIR})
    target_include_directories(${TESTNAME} PUBLIC 
        ${GDAL_INCLUDE_DIR}
        ${PYRE_INCLUDE_DIR}
        ${ISCE_SOURCE_DIR}/contrib/cereal/include)
    add_test(NAME ${TESTNAME} COMMAND ${TESTNAME})
endforeach ()

include(CTest)

# end of file
