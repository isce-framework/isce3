set(LISCE isce.${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR})

# Copy files used for resampling
file(COPY original.slc DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY original.slc.vrt DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY reference.slc DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY reference.slc.vrt DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY offsets/range.off DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/offsets)
file(COPY offsets/range.off.xml DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/offsets)
file(COPY offsets/azimuth.off DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/offsets)
file(COPY offsets/azimuth.off.xml DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/offsets)

foreach (TESTNAME resampslc)
    add_executable(${TESTNAME} ${TESTNAME}.cpp)
    target_link_libraries(${TESTNAME} PUBLIC ${LISCE} gtest)
    target_include_directories(${TESTNAME} PUBLIC ${ISCE_BUILDINCLUDEDIR})
    target_include_directories(${TESTNAME} PUBLIC 
        ${ISCE_SOURCE_DIR}/contrib/cereal/include)
    add_test(NAME ${TESTNAME} COMMAND ${TESTNAME})
endforeach ()

include(CTest)
