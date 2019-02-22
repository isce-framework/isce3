foreach (TESTNAME geometry geometry_equator)
    add_executable(${TESTNAME} ${TESTNAME}.cpp)
    add_dependencies(${TESTNAME} geom_test_data)
    target_link_libraries(${TESTNAME} PUBLIC ${LISCE} gtest)
    target_include_directories(${TESTNAME} PUBLIC ${ISCE_BUILDINCLUDEDIR})
    target_include_directories(${TESTNAME} PUBLIC 
        ${GDAL_INCLUDE_DIR}
        ${PYRE_INCLUDE_DIR}
        ${PYTHON_INCLUDE_DIR}
        ${ISCE_SOURCE_DIR}/contrib/cereal/include)
    add_test(NAME ${TESTNAME} COMMAND ${TESTNAME})
    file(COPY input_data.txt DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
    file(COPY output_data.txt DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
    file(COPY output_data_zerodop.txt DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endforeach ()

include(CTest)

# end of file
